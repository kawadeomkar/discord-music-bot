import asyncio
import random
import time
from collections import deque
from typing import Any, List, Optional, Union

import async_timeout
import discord
import orjson
from discord.ext import commands

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.redis_client import GuildRedisStore
from src.sources import YTSource
from src.telemetry import get_tracer
from src.util import queue_message, get_logger

log = get_logger(__name__)
_tracer = get_tracer(__name__)
from src.youtube import YTDL, QueueObject


def _queue_display_str(title: str, url: str) -> str:
    return f"{title} - {url}"


def _serialize_queue_item(item: Union[QueueObject, YTSource]) -> bytes:
    if isinstance(item, QueueObject):
        return orjson.dumps(
            {
                "type": "qobj",
                "webpage_url": item.webpage_url,
                "title": item.title,
                "requester_id": item.requester.id,
                "ts": item.ts,
            }
        )
    return orjson.dumps(
        {
            "type": "ytsource",
            "ytsearch": item.ytsearch,
            "url": item.url,
            "process": item.process,
            "ts": item.ts,
        }
    )


def _deserialize_queue_item(
    data: bytes, guild: discord.Guild
) -> Optional[Union[QueueObject, YTSource]]:
    try:
        d = orjson.loads(data)
        if d.get("type") == "ytsource":
            return YTSource(
                ytsearch=d.get("ytsearch"),
                url=d.get("url"),
                process=d.get("process"),
                ts=d.get("ts"),
            )
        # "qobj" type or legacy entries written before the type field was added
        member: Union[discord.Member, discord.User, None] = (
            guild.get_member(d["requester_id"]) or guild.owner
        )
        if member is None:
            return None
        return QueueObject(d["webpage_url"], d["title"], member, ts=d.get("ts"))
    except Exception as e:
        log.warning(f"Failed to deserialize queue item: {e}")
        return None


class MusicPlayer:
    __slots__ = (
        "bot",
        "_guild",
        "_channel",
        "_last_author",
        "_cog",
        "current_song",
        "play_next",
        "queue",
        "mutex",
        "play_message",
        "history",
        "song_queue",
        "volume",
        "_player",
        "_prefetch_task",
        "_store",
        "_restore_task",
        "_queue_cleared",
        "_background_tasks",
    )

    bot: commands.Bot
    _guild: discord.Guild
    _channel: discord.TextChannel
    _last_author: Union[discord.User, discord.Member]
    _cog: Any
    current_song: Optional[YTDL]
    play_next: asyncio.Event
    queue: asyncio.Queue
    mutex: asyncio.Lock
    play_message: Optional[discord.Embed]
    history: deque
    song_queue: deque
    volume: float
    _player: Optional[asyncio.Task]
    _prefetch_task: Optional[asyncio.Task]
    _store: Optional[GuildRedisStore]
    _restore_task: Optional[asyncio.Task]
    _queue_cleared: bool
    _background_tasks: set

    def __init__(
        self,
        bot: commands.Bot,
        guild: discord.Guild,
        channel: discord.TextChannel,
        cog: Any,
        redis=None,
    ):
        self.bot = bot
        self._guild = guild
        self._channel = channel
        _fallback: Union[discord.Member, discord.User, None] = guild.me or guild.owner
        self._last_author = _fallback  # type: ignore[assignment]
        self._cog = cog

        self.current_song = None
        self.play_next = asyncio.Event()
        self.queue = asyncio.Queue()
        self.mutex = asyncio.Lock()

        self.play_message = None
        self.history = deque(maxlen=50)
        self.song_queue = deque()
        self.volume = 1.0

        self._store = (
            GuildRedisStore(redis, self._guild.id) if redis is not None else None
        )
        self._player: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        self._restore_task: Optional[asyncio.Task] = None
        self._queue_cleared: bool = False
        self._background_tasks: set = set()

    @classmethod
    def from_context(
        cls,
        bot: commands.Bot,
        ctx: commands.Context,
        redis=None,
    ) -> "MusicPlayer":
        assert ctx.guild is not None
        assert isinstance(ctx.channel, discord.TextChannel)
        assert ctx.cog is not None
        mp = cls(bot, ctx.guild, ctx.channel, ctx.cog, redis=redis)
        mp._last_author = ctx.author
        return mp

    def start(self) -> None:
        """Start the playback loop and (if Redis is configured) the state restore task."""
        self._player = self.bot.loop.create_task(self.loop())
        if self._store is not None:
            self._restore_task = self.bot.loop.create_task(self._restore_state())

    def set_context(self, ctx: commands.Context) -> None:
        assert isinstance(ctx.channel, discord.TextChannel)
        self._channel = ctx.channel
        self._last_author = ctx.author

    def get_queue(self) -> str:
        return queue_message(list(self.song_queue)[:10])

    async def stop(self):
        await self._cog.cleanup(self._guild)

    async def redis_set_state(self, field: str, value: str) -> None:
        """Update a field in the guild state hash."""
        if self._store is not None:
            await self._store.set_state(field, value)

    # ── State restore ─────────────────────────────────────────────────────────

    async def _restore_state(self) -> None:
        """
        Restore queue, history, and volume from Redis after a bot restart.
        Runs as a background task; waits for bot ready so guild members are cached.
        """
        if self._store is None:
            return
        await self.bot.wait_until_ready()
        with _tracer.start_as_current_span(
            "player.state_restore",
            attributes={"discord.guild_id": str(self._guild.id)},
        ) as span:
            try:
                # Restore volume
                state = await self._store.get_state()
                if state and b"volume" in state:
                    self.volume = float(state[b"volume"])

                # Re-queue song that was playing when the bot crashed (at-most-once delivery).
                # current_song_url is written to state when playback starts and cleared when
                # it finishes normally. A non-empty value means the bot died mid-song.
                crashed_url_raw = state.get(b"current_song_url", b"")
                if crashed_url_raw:
                    crashed_url = crashed_url_raw.decode()
                    crashed_title = state.get(b"current_song_title", b"").decode()
                    requester: Union[discord.Member, discord.User, None] = (
                        self._guild.me or self._guild.owner
                    )
                    if requester is not None:
                        crashed = QueueObject(crashed_url, crashed_title, requester)
                        await self.queue.put(crashed)
                        self.song_queue.append(
                            _queue_display_str(crashed_title, crashed_url)
                        )
                        await self._store.set_state("current_song_url", "")
                        await self._store.set_state("current_song_title", "")
                        log.info(
                            f"Re-queued crashed song '{crashed_title}' for guild {self._guild.id}"
                        )

                # Restore queue (Redis list → asyncio.Queue + song_queue deque)
                items = await self._store.get_queue()
                count = 0
                for item in items:
                    restored = _deserialize_queue_item(item, self._guild)
                    if restored is not None:
                        await self.queue.put(restored)
                        if isinstance(restored, QueueObject):
                            self.song_queue.append(
                                _queue_display_str(restored.title, restored.webpage_url)
                            )
                        else:
                            self.song_queue.append(
                                _queue_display_str(
                                    restored.ytsearch or restored.url or "?", ""
                                )
                            )
                        count += 1
                if count:
                    log.info(
                        f"Restored {count} queued songs for guild {self._guild.id}"
                    )

                # Restore history (Redis list is newest-first; deque appends oldest-first)
                hist_items = await self._store.get_history()
                for item in reversed(hist_items):
                    try:
                        self.history.append(orjson.loads(item))
                    except Exception:
                        pass

                span.set_attribute("restore.queue_count", count)
                span.set_attribute("restore.crashed_song", bool(crashed_url_raw))

            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                log.error(
                    f"State restore failed for guild {self._guild.id}: {e}",
                    exc_info=True,
                )
                return

            # Refresh TTL on all guild keys after successful restore.
            await self._store.refresh_ttl()

    # ── Queue operations ──────────────────────────────────────────────────────

    async def queue_put(
        self,
        obj: Union[QueueObject, YTSource, List[QueueObject], List[YTSource]],
        *,
        prefetch: bool = True,
    ):
        items: list[Union[QueueObject, YTSource]]
        if isinstance(obj, list):
            items = list(obj)  # type: ignore[arg-type]
        else:
            items = [obj]
        for item in items:
            await self.queue.put(item)
            if isinstance(item, QueueObject):
                self.song_queue.append(_queue_display_str(item.title, item.webpage_url))
            else:
                self.song_queue.append(
                    _queue_display_str(item.ytsearch or item.url or "?", "")
                )

        # Mirror to Redis and (optionally) kick off stream pre-fetch.
        # prefetch=False for bulk playlist enqueues — spawning N concurrent
        # prefetch tasks saturates the thread pool and produces stream URLs that
        # expire before the song reaches playback position. _prefetch_next_song
        # handles one-ahead prefetch naturally as songs play.
        if self._store is None:
            return
        serializable = [i for i in items if isinstance(i, (QueueObject, YTSource))]
        if not serializable:
            return
        if prefetch:
            for item in serializable:
                await self._store.push_queue(_serialize_queue_item(item))
                if isinstance(item, QueueObject):
                    task = asyncio.create_task(
                        YTDL.prefetch_stream(item, redis=self._store.redis)
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
        else:
            await self._store.push_queue_batch(
                [_serialize_queue_item(item) for item in serializable]
            )

    async def queue_get(self) -> Union[QueueObject, YTSource]:
        return await self.queue.get()

    async def _cancel_prefetch(self) -> None:
        """Cancel any in-flight prefetch task and wait for it to finish.

        Must be called before any bulk queue mutation (clear, shuffle) so that
        the item the prefetch already dequeued via get_nowait() is accounted for
        via its CancelledError handler before we start modifying the queue.
        """
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass

    async def queue_clear(self) -> List[str]:
        await self._cancel_prefetch()
        async with self.mutex:
            self._queue_cleared = True
            for _ in range(self.queue.qsize()):
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            cleared = list(self.song_queue)
            self.song_queue.clear()
        if self._store is not None:
            await self._store.delete_queue()
        return cleared

    async def queue_shuffle(self) -> str:
        await self._cancel_prefetch()

        shuffled: List[Union[QueueObject, YTSource]] = []

        if self.queue.qsize() < 4:
            return "There must be at least 3 songs to shuffle the queue"

        async with self.mutex:
            for _ in range(self.queue.qsize()):
                try:
                    song = self.queue.get_nowait()
                    self.queue.task_done()
                    shuffled.append(song)
                except asyncio.QueueEmpty:
                    break
            random.shuffle(shuffled)
            squeue = []
            for song in shuffled:
                try:
                    self.queue.put_nowait(song)
                    if isinstance(song, QueueObject):
                        squeue.append(_queue_display_str(song.title, song.webpage_url))
                    else:
                        squeue.append(
                            _queue_display_str(song.ytsearch or song.url or "?", "")
                        )
                except asyncio.QueueFull:
                    break
            self.song_queue = deque(squeue)

        # Rebuild Redis mirror atomically: DELETE + RPUSH must be MULTI/EXEC,
        # not plain pipeline — a plain pipeline() leaves a window where the key
        # is empty and a concurrent LPOP sees an empty queue.
        if self._store is not None and shuffled:
            serialized = [
                _serialize_queue_item(s) for s in shuffled if isinstance(s, QueueObject)
            ]
            if serialized:
                await self._store.rebuild_queue(serialized)

        return "Shuffled!"

    # ── Embed building ────────────────────────────────────────────────────────

    def _build_now_playing_embed(self, song: YTDL) -> discord.Embed:
        requester_mention = song.requester.mention if song.requester else "Unknown"
        return (
            discord.Embed(
                title=f"**Now playing:** {song.title}",
                description=f"Requester: [{requester_mention}]",
                color=discord.Color.green(),
            )
            .add_field(name="Youtube link", value=song.webpage_url, inline=False)
            .add_field(name="Duration", value=song.duration)
            .add_field(name="Channel", value=song.uploader)
            .add_field(name="Views", value=str(song.views))
            .add_field(name="Likes", value=str(song.likes))
            .add_field(name="Dislikes", value=str(song.dislikes))
            .set_thumbnail(url=song.thumbnail)
            .set_footer(
                text=f"Avg Bitrate: {song.abr} | Avg Sampling: {song.asr} | Acodec: {song.acodec}"
            )
        )

    async def update_activity(self, song: Optional[YTDL] = None) -> None:
        if song is not None:
            now_ms = int(time.time() * 1000)
            timestamps: dict = {"start": now_ms}
            if song.duration_secs > 0:
                timestamps["end"] = now_ms + song.duration_secs * 1000

            # Bot opcode-3 activities only render `name` reliably in Discord's
            # client. Rich Presence (details, assets) requires the Discord RPC/SDK
            # which connects to a local desktop client — incompatible with server
            # bots. Pack the uploader into `name` as a suffix so it's visible.
            # `details` is kept as a forward-compat fallback; `timestamps` works
            # in the hover tooltip regardless.
            title = song.title or "a song"
            uploader = song.uploader
            raw_name = f"{title} · {uploader}" if uploader else title
            name = raw_name if len(raw_name) <= 128 else raw_name[:127] + "…"

            # state renders in both hover and click card for bot activities.
            # state_url kept for forward-compat (state renders, URL may become
            # clickable). details/details_url confirmed non-rendering for bots.
            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=name,
                state=song.duration,
                state_url=song.webpage_url,  # discord.py >= 2.6; silent no-op if downgraded
                timestamps=timestamps,
            )
        else:
            # Only reset when no other guild is still playing.
            active = any(
                vc.is_playing()
                for vc in self.bot.voice_clients
                if isinstance(vc, discord.VoiceClient)
            )
            if active:
                return
            activity = discord.Game(name="music")
        try:
            await self.bot.change_presence(activity=activity)
        except Exception as e:
            log.warning(f"Failed to update bot activity: {e}", exc_info=True)

    # ── Playback pipeline helpers ─────────────────────────────────────────────

    async def _resolve_source(
        self, source: Union[QueueObject, YTSource]
    ) -> QueueObject:
        if isinstance(source, YTSource):
            return await YTDL.yt_source(
                self._last_author,
                source.ytsearch or "",
                source.process or False,
                redis=self._store.redis if self._store is not None else None,
            )
        return source

    async def _stream_source(self, source: QueueObject) -> Optional[YTDL]:
        try:
            return await YTDL.yt_stream(
                source,
                self._channel,
                volume=self.volume,
                redis=self._store.redis if self._store is not None else None,
            )
        except Exception as e:
            log.error(f"Error processing song: {type(e).__name__}: {e}", exc_info=True)
            return None

    async def _send_now_playing(self, song: YTDL) -> None:
        try:
            embed = self._build_now_playing_embed(song)
            self.play_message = embed
            await self._channel.send(embed=embed)
        except Exception as e:
            log.error(f"embed error: {e}")

    @_tracer.start_as_current_span("player.prefetch")
    async def _prefetch_next_song(self) -> Optional[YTDL]:
        """Pre-resolve and stream the next queued song while the current one plays.

        Only runs if there is already an item in the queue (non-blocking).
        Calls queue.task_done() itself if dequeue succeeds but streaming fails,
        so the main loop's task_done() always accounts for exactly one get().
        """
        if self.queue.empty():
            return None
        try:
            source = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        trace.get_current_span().set_attribute("discord.guild_id", str(self._guild.id))
        try:
            source = await self._resolve_source(source)
            return await self._stream_source(source)
        except asyncio.CancelledError:
            self.queue.task_done()
            raise
        except Exception as e:
            trace.get_current_span().record_exception(e)
            trace.get_current_span().set_status(
                StatusCode.ERROR, f"{type(e).__name__}: {e}"
            )
            log.error(f"Prefetch error: {type(e).__name__}: {e}", exc_info=True)
            self.queue.task_done()
            return None

    # ── Main playback loop ────────────────────────────────────────────────────

    async def loop(self):
        await self.bot.wait_until_ready()
        prefetched_song: Optional[YTDL] = None

        while not self.bot.is_closed():
            self.play_next.clear()
            # Each iteration spans the full song duration (3–5 min typically).
            # This is expected — the span stays open across play_next.wait().
            with _tracer.start_as_current_span(
                "player.loop.iteration",
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    queue_was_cleared = self._queue_cleared
                    self._queue_cleared = False
                    prefetch_used = prefetched_song is not None
                    span.set_attribute("prefetch.used", prefetch_used)
                    if prefetched_song is not None and queue_was_cleared:
                        # The queue was cleared while _prefetch_next_song was running.
                        # The prefetch task completed and consumed a get_nowait() — balance
                        # it with task_done() and release the FFmpeg subprocess via cleanup()
                        # so it doesn't leak when we discard the result.
                        self.queue.task_done()
                        prefetched_song.cleanup()
                        prefetched_song = None
                    if prefetched_song is not None:
                        self.current_song = prefetched_song
                        prefetched_song = None
                    else:
                        try:
                            async with async_timeout.timeout(300):
                                source = await self.queue_get()
                                source = await self._resolve_source(source)
                        except asyncio.TimeoutError:
                            log.warning("Queue timed out, disconnecting")
                            asyncio.create_task(self.stop())
                            return
                        self.current_song = await self._stream_source(source)

                    if self.current_song is None:
                        try:
                            self.song_queue.popleft()
                        except IndexError:
                            pass
                        if self._store is not None:
                            await self._store.pop_queue()
                        self.queue.task_done()
                        try:
                            await self._channel.send(
                                "Failed to load the next song, skipping."
                            )
                        except Exception:
                            pass
                        continue

                    span.set_attribute("song.title", self.current_song.title or "")

                    discard = False
                    async with self.mutex:
                        try:
                            self.song_queue.popleft()
                        except IndexError:
                            # song_queue was cleared while this song was being resolved
                            # (e.g. during the async yt_stream call). Discard without
                            # playing; task_done() balances the queue.get() above.
                            # cleanup() terminates the FFmpeg subprocess that yt_stream
                            # already spawned — omitting it would leak the process.
                            self.queue.task_done()
                            self.current_song.cleanup()
                            self.current_song = None
                            discard = True
                    if discard:
                        continue
                    if self._store is not None:
                        await self._store.pop_queue()

                    vc = self._guild.voice_client
                    assert isinstance(vc, discord.VoiceClient)
                    assert self.current_song is not None
                    vc.play(
                        self.current_song,
                        after=lambda _: self.bot.loop.call_soon_threadsafe(
                            self.play_next.set
                        ),
                    )
                    await self.update_activity(self.current_song)
                    await self._send_now_playing(self.current_song)

                    # Mirror now-playing song to Redis state
                    if self._store is not None and self.current_song is not None:
                        await self._store.set_state(
                            "current_song_title", self.current_song.title or ""
                        )
                        await self._store.set_state(
                            "current_song_url", self.current_song.webpage_url or ""
                        )

                    self._prefetch_task = asyncio.create_task(
                        self._prefetch_next_song()
                    )

                    await self.play_next.wait()

                    try:
                        prefetched_song = await self._prefetch_task
                    except asyncio.CancelledError:
                        prefetched_song = None
                    self._prefetch_task = None

                    if self.current_song is not None:
                        history_entry = f"{self.current_song.title} - {self.current_song.webpage_url}"
                        self.history.append(history_entry)
                        if self._store is not None:
                            await self._store.push_history(orjson.dumps(history_entry))

                    # Clear now-playing state
                    if self._store is not None:
                        await self._store.set_state("current_song_title", "")
                        await self._store.set_state("current_song_url", "")

                    self.queue.task_done()
                    self.current_song = None
                    await self.update_activity(None)
                except asyncio.CancelledError:
                    span.set_attribute("loop.cancelled", True)
                    await self.update_activity(None)
                    raise
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                    log.error(
                        f"Unhandled error in playback loop: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    if self._prefetch_task and not self._prefetch_task.done():
                        self._prefetch_task.cancel()
                    self._prefetch_task = None
                    prefetched_song = None
                    self.current_song = None
                    try:
                        span_ctx = span.get_span_context()
                        embed = discord.Embed(
                            title="Playback error — skipping song",
                            description=f"**{type(e).__name__}:** {e}",
                            color=discord.Color.red(),
                        )
                        if span_ctx.is_valid:
                            embed.set_footer(
                                text=f"trace: {format(span_ctx.trace_id, '032x')}"
                            )
                        await self._channel.send(embed=embed)
                    except Exception:
                        pass
