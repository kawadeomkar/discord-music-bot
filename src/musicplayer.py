import asyncio
import datetime
import random
import time
from collections import deque
from typing import Any, List, Optional, Union
from zoneinfo import ZoneInfo

import async_timeout
import discord
import orjson
from discord.ext import commands

from opentelemetry import trace

from src import config
from src.redis_client import GuildRedisStore, cache_get
from src.sources import YTSource
from src.telemetry import get_tracer
from src.util import (
    cancel_task,
    record_span_error,
    send_embed,
    trace_footer,
    get_logger,
)
from src.youtube import YTDL, QueueObject

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# ETAs in get_queue() are rendered in Pacific time. This is intentional for a
# single-operator bot — update to a per-guild config if multi-tenant support is added.
_PST = ZoneInfo("America/Los_Angeles")


def _fmt_duration(secs: int) -> str:
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_total_duration(secs: int) -> str:
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    return " ".join(parts) or "0s"


def _fmt_clock_time(dt: datetime.datetime) -> str:
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.strftime('%M')} {ampm} PST"


def _fmt_eta(est_dt: datetime.datetime, uncertain: bool) -> str:
    prefix = "~" if uncertain else ""
    return f"{prefix}**{_fmt_clock_time(est_dt)}**"


def _requester_mention(
    requester: Optional[Union[discord.User, discord.Member]],
) -> str:
    return requester.mention if requester else "Unknown"


# Square emoji blocks read as noticeably thicker/higher-contrast than a thin
# dash character, and — unlike a single-color fill — let the played portion
# render in a visibly different color from the remaining portion. Width is
# lower than a typical thin-dash bar since each block glyph is much wider.
_BAR_WIDTH = 12
_BAR_FILL_DONE = "🟦"
_BAR_FILL_REMAINING = "⬜"
_BAR_HEAD = "🔘"

# How long mark_paused()/mark_resumed() wait before firing the embed edit +
# Activity refresh — collapses rapid -pause/-resume toggling into one trailing
# update instead of one Discord API call pair per toggle (Design §5).
_PAUSE_DEBOUNCE_SECS = 0.5


def _build_progress_bar(
    elapsed_secs: float, duration_secs: int, width: int = _BAR_WIDTH
) -> str:
    if duration_secs <= 0:
        return ""
    ratio = max(0.0, min(1.0, elapsed_secs / duration_secs))
    head_pos = min(width - 1, int(ratio * width))
    bar = (
        _BAR_FILL_DONE * head_pos
        + _BAR_HEAD
        + _BAR_FILL_REMAINING * (width - head_pos - 1)
    )
    return (
        f"`{_fmt_duration(int(elapsed_secs))}` {bar} `{_fmt_duration(duration_secs)}`"
    )


def _fmt_finish_time(duration_secs: int) -> str:
    """Clock time `duration_secs` from now — no uncertainty prefix, since a
    song's own remaining duration (unlike a queued song's ETA) is never
    uncertain once it's playing."""
    finish_dt = datetime.datetime.now(tz=_PST) + datetime.timedelta(
        seconds=duration_secs
    )
    return _fmt_clock_time(finish_dt)


def _serialize_queue_item(item: Union[QueueObject, YTSource]) -> bytes:
    if isinstance(item, QueueObject):
        return orjson.dumps(
            {
                "type": "qobj",
                "webpage_url": item.webpage_url,
                "title": item.title,
                "requester_id": item.requester.id,
                "ts": item.ts,
                "user_input": item.user_input,
                "duration": item.duration,
                "uploader": item.uploader,
                "thumbnail": item.thumbnail,
                "persisted": item.persisted,
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
        return QueueObject(
            d["webpage_url"],
            d["title"],
            member,
            ts=d.get("ts"),
            user_input=d.get("user_input"),
            duration=d.get("duration"),
            uploader=d.get("uploader"),
            thumbnail=d.get("thumbnail"),
            persisted=d.get("persisted", True),
        )
    except Exception as e:
        log.warning(f"Failed to deserialize queue item: {e}")
        return None


def _song_now_playing_fields(song: YTDL) -> dict[str, str]:
    """Canonical string-field extraction from a live song — the single source
    of truth for both the live embed and the Redis now_playing snapshot, so
    the two can't drift out of sync."""
    return {
        "title": song.title or "",
        "webpage_url": song.webpage_url or "",
        "uploader": song.uploader or "",
        "duration": song.duration or "",
        "thumbnail": song.thumbnail or "",
        "view_count": str(song.views) if song.views is not None else "",
        "like_count": str(song.likes) if song.likes is not None else "",
        "abr": str(song.abr) if song.abr is not None else "",
        "asr": str(song.asr) if song.asr is not None else "",
        "acodec": song.acodec or "",
        "requester_id": str(song.requester.id) if song.requester else "",
        "requester_mention": _requester_mention(song.requester),
    }


def _build_now_playing_base_embed(
    *,
    title: str,
    description: str,
    webpage_url: str,
    duration: str,
    uploader: str,
    views: str,
    likes: str,
    abr: str,
    asr: str,
    acodec: str,
    thumbnail: str,
) -> discord.Embed:
    """Shared field layout — used by both the live (YTDL-backed) and
    Redis-recovery (bytes-backed) now-playing embed builders."""
    embed = (
        discord.Embed(title=title, description=description, color=discord.Color.green())
        .add_field(name="Youtube link", value=webpage_url, inline=False)
        .add_field(name="Duration", value=duration)
        .add_field(name="Channel", value=uploader)
        .add_field(name="Views", value=views)
        .add_field(name="Likes", value=likes)
        .set_footer(text=f"Avg Bitrate: {abr} | Avg Sampling: {asr} | Acodec: {acodec}")
    )
    if thumbnail:
        embed.set_thumbnail(url=thumbnail)
    return embed


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
        "_restore_complete",
        "_queue_cleared",
        "_background_tasks",
        "_progress_task",
        "_now_playing_message",
        "_pause_debounce_task",
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
    _restore_complete: asyncio.Event
    _queue_cleared: bool
    _background_tasks: set
    _progress_task: Optional[asyncio.Task]
    _now_playing_message: Optional[discord.Message]
    _pause_debounce_task: Optional[asyncio.Task]

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
        self._restore_complete = asyncio.Event()
        self._queue_cleared: bool = False
        self._background_tasks: set = set()
        self._progress_task: Optional[asyncio.Task] = None
        self._now_playing_message: Optional[discord.Message] = None
        self._pause_debounce_task: Optional[asyncio.Task] = None

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
        """Start the playback loop and (if Redis is configured) the state restore task.

        loop() blocks on self._restore_complete before consuming from self.queue —
        see _restore_state() for why. When there's no store, restore is a no-op, so
        the event is set immediately rather than left for _restore_state() to set.
        """
        if self._store is not None:
            self._restore_task = self.bot.loop.create_task(self._restore_state())
        else:
            # No Redis — no restore will run; signal immediately so the prefetch
            # gate in loop() never waits.
            self._restore_complete.set()
        self._player = self.bot.loop.create_task(self.loop())

    def set_context(self, ctx: commands.Context) -> None:
        assert isinstance(ctx.channel, discord.TextChannel)
        self._channel = ctx.channel
        self._last_author = ctx.author

    def _queue_eta_seed(self) -> tuple[datetime.datetime, int, bool]:
        """Seed state for walking ETAs across queued songs.

        Returns (now_pst, cumulative_secs, uncertain), where cumulative_secs
        is seeded with the current song's total duration as a proxy for its
        remaining time (we don't track elapsed; this overestimates but keeps
        the math simple and avoids showing "now" for everything), and
        uncertain flags whether any preceding song had an unknown duration.
        """
        uncertain = False
        cumulative_secs = 0
        if self.current_song is not None:
            secs = getattr(self.current_song, "duration_secs", 0)
            if secs:
                cumulative_secs = secs
            else:
                uncertain = True
        return datetime.datetime.now(tz=_PST), cumulative_secs, uncertain

    def _format_queue_line(
        self,
        item: Union[QueueObject, YTSource],
        index: int,
        now_pst: datetime.datetime,
        cumulative_secs: int,
        uncertain: bool,
    ) -> tuple[str, int, bool]:
        """Format a single queue line with its "Est. playing at" ETA.

        Returns (line, updated cumulative_secs, updated uncertain) so callers
        can chain this across consecutive queue items.
        """
        est_dt = now_pst + datetime.timedelta(seconds=cumulative_secs)
        est_str = _fmt_eta(est_dt, uncertain)

        if isinstance(item, QueueObject):
            title = item.title or "Unknown"
            requester = _requester_mention(item.requester)
            dur = _fmt_duration(item.duration) if item.duration is not None else "?:??"
            channel = item.uploader or "Unknown channel"
            ts_note = f"  ·  starts at `{item.ts}s`" if item.ts else ""
            line = (
                f"`{index}` [**{title}**]({item.webpage_url}) · `{dur}`{ts_note} · Est. playing at {est_str}\n"
                f"{channel} · {requester}"
            )
            if item.duration is not None:
                cumulative_secs += item.duration
            else:
                uncertain = True
        else:
            search = (item.ytsearch or item.url or "?").removeprefix("ytsearch:")
            line = f"`{index}` {search} · *resolving...*"
            uncertain = True

        return line, cumulative_secs, uncertain

    def estimated_playing_at(self) -> str:
        """ETA text for a song appended to the queue right now — i.e. after the
        current song and everything already queued. Reuses the same ETA-walking
        seed as get_queue()/_build_next_up_embed() so all three stay consistent.
        """
        now_pst, cumulative_secs, uncertain = self._queue_eta_seed()
        for item in self.song_queue:
            if isinstance(item, QueueObject) and item.duration is not None:
                cumulative_secs += item.duration
            else:
                uncertain = True
        est_dt = now_pst + datetime.timedelta(seconds=cumulative_secs)
        return _fmt_eta(est_dt, uncertain)

    def get_queue(self) -> discord.Embed:
        items = list(self.song_queue)
        total = len(items)

        total_secs = 0
        duration_partial = False
        for item in items:
            if isinstance(item, QueueObject) and item.duration is not None:
                total_secs += item.duration
            else:
                duration_partial = True

        now_pst, cumulative_secs, uncertain = self._queue_eta_seed()

        lines = []
        for i, item in enumerate(items[:10], start=1):
            line, cumulative_secs, uncertain = self._format_queue_line(
                item, i, now_pst, cumulative_secs, uncertain
            )
            lines.append(line)

        header = f"Songs: **{total}**"
        if total_secs > 0:
            dur_prefix = "~" if duration_partial else ""
            header += (
                f"\nTotal Duration: **{dur_prefix}{_fmt_total_duration(total_secs)}**"
            )

        songs_text = "\n\n".join(lines) if lines else "*The queue is empty.*"
        if total > 10:
            songs_text += f"\n\n*... and {total - 10} more*"

        return discord.Embed(
            title="Queue",
            description=header + "\n\n" + songs_text,
            color=discord.Color.blue(),
        )

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

        loop() waits on self._restore_complete before its first queue_get(). Without
        this, loop() can race ahead and dequeue the crash-recovered "current song"
        this method injects below, then call pop_queue() (Redis LPOP) as part of its
        normal transition bookkeeping — the crashed song was never itself on the
        Redis queue list (it's tracked separately via current_song_url state), so
        that LPOP silently deletes an unrelated, still-queued song from Redis
        before this method ever gets to read the queue itself.
        """
        if self._store is None:
            self._restore_complete.set()
            return
        try:
            await self.bot.wait_until_ready()
            with _tracer.start_as_current_span(
                "player.state_restore",
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    state = await self._store.get_state()

                    # Restore volume
                    if state and b"volume" in state:
                        self.volume = float(state[b"volume"])

                    # Restore now-playing embed so -now works if a song is playing on recovery.
                    np_data = await self._store.get_now_playing()
                    if np_data:
                        self.play_message = self._build_now_playing_embed_from_data(
                            np_data
                        )

                    # Re-queue song that was playing when the bot crashed (at-most-once delivery).
                    # current_song_url is set atomically with the LPOP; a non-empty value means
                    # the bot died after the transaction committed but before the song finished.
                    crashed_url_raw = state.get(b"current_song_url", b"")
                    count = 0
                    if crashed_url_raw:
                        crashed_url = crashed_url_raw.decode()
                        crashed_title = state.get(b"current_song_title", b"").decode()
                        raw_dur = state.get(b"current_song_duration", b"").decode()
                        crashed_duration = int(raw_dur) if raw_dur.isdigit() else None
                        raw_uploader = state.get(b"current_song_uploader", b"").decode()
                        crashed_uploader = raw_uploader or None

                        # Resolve the requester from the ID persisted with the song
                        # itself at start-transaction time.
                        requester_id_raw = state.get(b"current_song_requester_id", b"")
                        requester: Union[discord.Member, discord.User, None] = None
                        if requester_id_raw:
                            try:
                                requester = self._guild.get_member(
                                    int(requester_id_raw)
                                )
                            except ValueError:
                                pass
                        if requester is None:
                            requester = self._guild.me or self._guild.owner

                        # Compute approximate playback position at time of crash.
                        position: Optional[int] = None
                        play_start_raw = state.get(b"play_start_epoch", b"")
                        if play_start_raw:
                            try:
                                now = time.time()
                                elapsed = now - float(play_start_raw)
                                total_pause = int(
                                    float(state.get(b"total_pause_seconds", b"0"))
                                )
                                pause_start_raw_pos = state.get(
                                    b"pause_start_epoch", b""
                                )
                                if pause_start_raw_pos:
                                    total_pause += int(now - float(pause_start_raw_pos))
                                position = max(0, int(elapsed - total_pause))

                                # Cap at duration − 10s to prevent FFmpeg seeking past EOF.
                                stream_data = await cache_get(
                                    self._store.redis, f"ytdl:stream:{crashed_url}"
                                )
                                if stream_data is not None:
                                    raw_duration = stream_data.get("duration")
                                    if raw_duration is not None:
                                        position = min(
                                            position, max(0, int(raw_duration) - 10)
                                        )

                                log.info(
                                    f"Computed recovery position {position}s for '{crashed_title}' "
                                    f"(elapsed={int(elapsed)}s, paused={total_pause}s)"
                                )
                            except Exception as pos_err:
                                log.warning(
                                    f"Failed to compute recovery position: {pos_err}"
                                )
                                position = None

                        if requester is not None:
                            # persisted=False: this item was never RPUSHed to Redis's
                            # queue list (it's tracked separately via current_song_url
                            # state), so the playback loop must skip the matching
                            # Redis pop_queue() for it — see _pop_queue_for_dequeue().
                            crashed = QueueObject(
                                crashed_url,
                                crashed_title,
                                requester,
                                ts=position,
                                duration=crashed_duration,
                                uploader=crashed_uploader,
                                persisted=False,
                            )
                            await self.queue.put(crashed)
                            self.song_queue.append(crashed)
                            log.info(
                                f"Re-queued crashed song '{crashed_title}' for guild {self._guild.id}"
                            )
                        # Always clear regardless of whether requester was resolvable;
                        # leaving current_song_url set would cause every subsequent
                        # restart to re-enter this block and never escape until the
                        # TTL expires.
                        await self._store.clear_song_end_state()

                    # Restore queue + history (Redis list → asyncio.Queue +
                    # song_queue deque). Independent reads — no data dependency
                    # between them — fetched concurrently rather than as two
                    # sequential round-trips, since loop() now blocks on this
                    # method finishing (see self._restore_complete above).
                    items, hist_items = await asyncio.gather(
                        self._store.get_queue(),
                        self._store.get_history(),
                    )
                    for item in items:
                        restored = _deserialize_queue_item(item, self._guild)
                        if restored is not None:
                            await self.queue.put(restored)
                            self.song_queue.append(restored)
                            count += 1
                    if count:
                        log.info(
                            f"Restored {count} queued songs for guild {self._guild.id}"
                        )

                    # Restore history (Redis list is newest-first; deque appends oldest-first)
                    for item in reversed(hist_items):
                        try:
                            self.history.append(orjson.loads(item))
                        except Exception as e:
                            log.warning(
                                f"Failed to deserialize history item in guild {self._guild.id}: {e}"
                            )

                    span.set_attribute("restore.queue_count", count)
                    span.set_attribute("restore.crashed_song", bool(crashed_url_raw))

                except Exception as e:
                    record_span_error(span, e)
                    log.error(
                        f"State restore failed for guild {self._guild.id}: {e}",
                        exc_info=True,
                    )
                    return

                # Refresh TTL on all guild keys after successful restore.
                await self._store.refresh_ttl()
        finally:
            # Always signal loop() that restore has finished or failed so the
            # prefetch gate never blocks indefinitely.
            self._restore_complete.set()

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
            self.song_queue.append(item)

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
                    self._spawn_background(
                        YTDL.prefetch_stream(item, redis=self._store.redis)
                    )
        else:
            await self._store.push_queue_batch(
                [_serialize_queue_item(item) for item in serializable]
            )

    async def queue_get(self) -> Union[QueueObject, YTSource]:
        return await self.queue.get()

    async def _pop_queue_for_dequeue(self, should_pop: bool) -> None:
        """Mirror one self.queue dequeue to Redis via LPOP — unless should_pop
        is False, meaning the item was injected directly into self.queue/
        song_queue rather than via queue_put() (e.g. the crash-recovered
        "current song" _restore_state() re-queues with persisted=False), so it
        was never pushed to Redis's queue list and there's nothing to pop.
        See _restore_state()'s docstring for the bug this prevents.
        """
        if self._store is not None and should_pop:
            await self._store.pop_queue()

    async def _cancel_prefetch(self) -> None:
        """Cancel any in-flight prefetch task and wait for it to finish.

        Must be called before any bulk queue mutation (clear, shuffle, remove) so that
        the item the prefetch already dequeued via get_nowait() is accounted for
        via its CancelledError handler before we start modifying the queue.

        Note: if the prefetch task is blocked inside run_in_executor (a yt-dlp
        extraction), cancellation cannot interrupt the running thread. The await
        blocks until the thread exits, which may take up to socket_timeout seconds.
        This is acceptable at single-guild scale.
        """
        await cancel_task(self._prefetch_task)

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
            cleared_items = list(self.song_queue)
            self.song_queue.clear()
        if self._store is not None:
            await self._store.delete_queue()
        return [
            (
                item.title
                if isinstance(item, QueueObject)
                else (item.ytsearch or item.url or "?").removeprefix("ytsearch:")
            )
            for item in cleared_items
        ]

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
            kept_after_shuffle = []
            for song in shuffled:
                try:
                    self.queue.put_nowait(song)
                    kept_after_shuffle.append(song)
                except asyncio.QueueFull:
                    break
            self.song_queue = deque(kept_after_shuffle)

        # Rebuild Redis mirror atomically: DELETE + RPUSH must be MULTI/EXEC,
        # not plain pipeline — a plain pipeline() leaves a window where the key
        # is empty and a concurrent LPOP sees an empty queue.
        if self._store is not None and kept_after_shuffle:
            # persisted=False items (the crash-recovered "current song") were
            # never RPUSHed to Redis's queue list — never write them back in.
            serialized = [
                _serialize_queue_item(s)
                for s in kept_after_shuffle
                if isinstance(s, (QueueObject, YTSource))
                and getattr(s, "persisted", True)
            ]
            if serialized:
                await self._store.rebuild_queue(serialized)

        return "Shuffled!"

    async def queue_remove(self, url: str) -> list[int]:
        """Remove all queued items whose webpage_url (QueueObject) or url (YTSource) matches url.

        Returns a list of 1-indexed queue positions that were removed.
        """
        await self._cancel_prefetch()
        removed_positions: list[int] = []
        kept: List[Union[QueueObject, YTSource]] = []

        async with self.mutex:
            # Drain everything first so positions are numbered before partitioning.
            drained: List[Union[QueueObject, YTSource]] = []
            for _ in range(self.queue.qsize()):
                try:
                    item = self.queue.get_nowait()
                    self.queue.task_done()
                    drained.append(item)
                except asyncio.QueueEmpty:
                    break

            for pos, item in enumerate(drained, start=1):
                if isinstance(item, QueueObject):
                    match = item.webpage_url == url
                else:
                    match = (item.url or "") == url
                if match:
                    removed_positions.append(pos)
                else:
                    kept.append(item)

            for item in kept:
                self.queue.put_nowait(item)
            self.song_queue = deque(kept)

        if removed_positions and self._store is not None:
            # persisted=False items (the crash-recovered "current song") were
            # never RPUSHed to Redis's queue list — never write them back in.
            serialized = [
                _serialize_queue_item(s)
                for s in kept
                if isinstance(s, (QueueObject, YTSource))
                and getattr(s, "persisted", True)
            ]
            if serialized:
                await self._store.rebuild_queue(serialized)
            else:
                await self._store.delete_queue()

        return removed_positions

    # ── Embed building ────────────────────────────────────────────────────────

    def _build_now_playing_embed(
        self, song: YTDL, *, elapsed_override: Optional[float] = None
    ) -> discord.Embed:
        """elapsed_override lets a caller render the bar at a specific position
        rather than song.elapsed_secs's live value — used by _finalize_now_playing()
        to show the bar fully completed once the song has actually ended."""
        lines = []
        if song.duration_secs > 0:
            elapsed = (
                elapsed_override if elapsed_override is not None else song.elapsed_secs
            )
            bar = _build_progress_bar(elapsed, song.duration_secs)
            if bar:
                # Bar sits directly under the title, above the requester line,
                # with a blank line between them for visual separation.
                lines.append(bar)
                lines.append("")
        requester_line = f"Requester: [{_requester_mention(song.requester)}]"
        if song.duration_secs > 0:
            requester_line += (
                f"  ·  Estimated finish: {_fmt_finish_time(song.duration_secs)}"
            )
        lines.append(requester_line)
        description = "\n".join(lines)
        fields = _song_now_playing_fields(song)
        return _build_now_playing_base_embed(
            title=f"**Now playing:** {song.title}",
            description=description,
            webpage_url=fields["webpage_url"],
            duration=fields["duration"],
            uploader=fields["uploader"],
            views=fields["view_count"],
            likes=fields["like_count"],
            abr=fields["abr"],
            asr=fields["asr"],
            acodec=fields["acodec"],
            thumbnail=fields["thumbnail"],
        )

    @staticmethod
    def _build_now_playing_embed_from_data(data: dict[bytes, bytes]) -> discord.Embed:
        """Reconstruct a now-playing embed from the Redis HASH data (bytes keys/values)."""

        def field(key: str) -> str:
            return data.get(key.encode(), b"").decode()

        return _build_now_playing_base_embed(
            title=f"**Now playing:** {field('title')}",
            description=f"Requester: [{field('requester_mention')}]",
            webpage_url=field("webpage_url"),
            duration=field("duration"),
            uploader=field("uploader"),
            views=field("view_count"),
            likes=field("like_count"),
            abr=field("abr"),
            asr=field("asr"),
            acodec=field("acodec"),
            thumbnail=field("thumbnail"),
        )

    def _build_next_up_embed(self) -> Optional[discord.Embed]:
        if not self.song_queue:
            return None
        item = self.song_queue[0]
        now_pst, cumulative_secs, uncertain = self._queue_eta_seed()
        description, _, _ = self._format_queue_line(
            item, 1, now_pst, cumulative_secs, uncertain
        )
        return discord.Embed(
            title="Up next",
            description=description,
            color=discord.Color.blue(),
        )

    async def update_activity(self, song: Optional[YTDL] = None) -> None:
        if song is not None:
            timestamps: dict = {}
            vc = self._guild.voice_client
            is_paused = isinstance(vc, discord.VoiceClient) and vc.is_paused()
            if not is_paused:
                # Backdated by elapsed time (not always "now") so that resuming
                # mid-song still lands `end` the correct remaining duration in
                # the future, not a full duration_secs from the resume moment.
                now_ms = int(time.time() * 1000)
                elapsed_ms = int(song.elapsed_secs * 1000)
                timestamps["start"] = now_ms - elapsed_ms
                if song.duration_secs > 0:
                    timestamps["end"] = timestamps["start"] + song.duration_secs * 1000
            # else: paused — timestamps stays {} so Discord shows static text with
            # no ticking bar, instead of a bar that keeps animating through the
            # pause (Discord's Activity schema has no "frozen" representation
            # other than omitting timestamps entirely).

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

    async def pause(self, vc: discord.VoiceClient) -> None:
        """Pause playback and synchronize all pause-tracking state in one
        place: the Redis crash-recovery epoch accounting and the
        progress-bar/Activity debounced refresh. Single entry point so a
        future pause call site can't forget one of the two side effects."""
        vc.pause()
        if self._store is not None:
            await self._store.on_pause(time.time())
        self.mark_paused()

    async def resume(self, vc: discord.VoiceClient) -> None:
        vc.resume()
        if self._store is not None:
            await self._store.on_resume(time.time())
        self.mark_resumed()

    def mark_paused(self) -> None:
        self._fire_pause_state_updates()

    def mark_resumed(self) -> None:
        self._fire_pause_state_updates()

    def _fire_pause_state_updates(self) -> None:
        """Debounced trigger for pause()/resume() commands: refreshes the
        now-playing embed and the Activity presence to reflect the new pause
        state. Debounced (not immediate) because nothing rate-limits how fast
        -pause/-resume can be invoked — see Design §5 of the progress-bar plan
        for why an undebounced version could spam two rate-limited Discord
        endpoints under rapid toggling.
        """
        if self.current_song is None:
            return
        if (
            self._pause_debounce_task is not None
            and not self._pause_debounce_task.done()
        ):
            self._pause_debounce_task.cancel()
        self._pause_debounce_task = self._spawn_background(
            self._debounced_pause_update()
        )

    async def _debounced_pause_update(self) -> None:
        try:
            await asyncio.sleep(_PAUSE_DEBOUNCE_SECS)
        except asyncio.CancelledError:
            return
        if self._progress_task is not None and self._now_playing_message is not None:
            self._spawn_background(self._edit_now_playing_once())
        self._spawn_background(self.update_activity(self.current_song))

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
        # Reset before attempting the send (not after failure) so a failed/partial
        # send never leaves this pointing at the *previous* song's message — a
        # stale reference here would let a later mark_paused()/mark_resumed() on
        # the new song silently overwrite the old song's already-sent embed.
        self._now_playing_message = None
        try:
            embed = self._build_now_playing_embed(song)
            self.play_message = embed
            embeds = [embed]
            next_up_embed = self._build_next_up_embed()
            if next_up_embed is not None:
                embeds.append(next_up_embed)
            message = await self._channel.send(embeds=embeds)
            self._now_playing_message = message
            if song.duration_secs >= 5:
                self._progress_task = asyncio.create_task(
                    self._progress_updater(song, message)
                )
        except Exception as e:
            log.error(f"embed error: {e}")

    async def _push_now_playing_edit(
        self,
        song: YTDL,
        message: discord.Message,
        *,
        elapsed_override: Optional[float] = None,
    ) -> bool:
        """Rebuild the now-playing + up-next embeds and push a single edit.

        Shared by the periodic tick, the debounced pause/resume refresh, and the
        song-end finalize edit — all three previously duplicated this exact
        build-embeds/edit/except block. Returns False if the message no longer
        exists (deleted) so callers that loop (_progress_updater) know to stop;
        the one-shot callers just ignore the return value.
        """
        try:
            embed = self._build_now_playing_embed(
                song, elapsed_override=elapsed_override
            )
            next_up = self._build_next_up_embed()
            embeds = [embed] + ([next_up] if next_up else [])
            await message.edit(embeds=embeds)
            return True
        except discord.NotFound:
            return False
        except discord.HTTPException as e:
            log.warning(f"Now-playing edit failed for guild {self._guild.id}: {e}")
            return True

    async def _edit_now_playing_once(self) -> None:
        """Rebuild and push a single embed edit outside the periodic tick — used
        for the debounced pause/resume refresh (mark_paused()/mark_resumed())."""
        song = self.current_song
        message = self._now_playing_message
        if song is None or message is None:
            return
        await self._push_now_playing_edit(song, message)

    async def _finalize_now_playing(self, song: YTDL, message: discord.Message) -> None:
        """One last embed edit once a song has actually ended, showing the bar
        fully completed (elapsed == duration) rather than left frozen wherever
        the last periodic tick happened to land. song/message are captured by
        the caller rather than read off self.current_song/self._now_playing_message
        at run time, since both may already point at the next song by the time
        this (fire-and-forget) task actually runs.
        """
        if song.duration_secs <= 0:
            return  # no bar was ever shown for this song — nothing to finalize
        await self._push_now_playing_edit(
            song, message, elapsed_override=song.duration_secs
        )

    def _spawn_background(self, coro: Any) -> asyncio.Task:
        """Create a fire-and-forget task tracked in _background_tasks."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _fire_finalize_now_playing(self, song: YTDL, message: discord.Message) -> None:
        self._spawn_background(self._finalize_now_playing(song, message))

    async def _progress_updater(self, song: YTDL, message: discord.Message) -> None:
        interval = config.NOW_PLAYING_UPDATE_INTERVAL_SECS
        try:
            while True:
                await asyncio.sleep(interval)
                vc = self._guild.voice_client
                if not isinstance(vc, discord.VoiceClient) or vc.source is not song:
                    return  # song changed under us; loop() owns cancellation but guard defensively
                if vc.is_paused():
                    continue  # frozen — mark_resumed() below fires a debounced edit instead
                if not await self._push_now_playing_edit(song, message):
                    return  # user deleted the message — stop editing a message that no longer exists
        except asyncio.CancelledError:
            raise

    async def _cancel_progress_task(self) -> None:
        """Must be awaited before the next song's _send_now_playing() to prevent a
        concurrent message.edit() for the old song from racing the new message send."""
        await cancel_task(self._progress_task)
        self._progress_task = None

    async def _cancel_pause_debounce(self) -> None:
        await cancel_task(self._pause_debounce_task)
        self._pause_debounce_task = None

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
            record_span_error(trace.get_current_span(), e)
            log.error(f"Prefetch error: {type(e).__name__}: {e}", exc_info=True)
            self.queue.task_done()
            return None

    # ── Main playback loop ────────────────────────────────────────────────────

    async def loop(self):
        await self.bot.wait_until_ready()
        # Wait for _restore_state() to finish populating self.queue before
        # dequeuing anything — see _restore_state()'s docstring for the race
        # this prevents (an erroneous Redis pop_queue() for a crash-recovered
        # song that was never on the Redis queue list in the first place).
        await self._restore_complete.wait()
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
                        # Prefetched items always came through queue_get(), so
                        # they were always real, Redis-mirrored queue entries.
                        should_pop_queue = True
                    else:
                        source = None
                        try:
                            async with async_timeout.timeout(300):
                                source = await self.queue_get()
                                source = await self._resolve_source(source)
                        except asyncio.TimeoutError:
                            log.warning("Queue timed out, disconnecting")
                            asyncio.create_task(self.stop())
                            return
                        except Exception:
                            # _resolve_source() raised (e.g. yt-dlp lookup failure)
                            # after queue_get() already dequeued `source` — balance
                            # that dequeue the same way the "current_song is None"
                            # branch below does, then let the outer handler's
                            # logging/error-embed path run via the re-raise.
                            if source is not None:
                                try:
                                    self.song_queue.popleft()
                                except IndexError:
                                    log.warning(
                                        f"song_queue was empty on resolve failure in guild {self._guild.id}"
                                    )
                                await self._pop_queue_for_dequeue(
                                    getattr(source, "persisted", True)
                                )
                                self.queue.task_done()
                            raise
                        self.current_song = await self._stream_source(source)
                        should_pop_queue = getattr(source, "persisted", True)

                    if self.current_song is None:
                        try:
                            self.song_queue.popleft()
                        except IndexError:
                            log.warning(
                                f"song_queue was empty on failed-song pop in guild {self._guild.id}"
                            )
                        await self._pop_queue_for_dequeue(should_pop_queue)
                        self.queue.task_done()
                        try:
                            await self._channel.send(
                                "Failed to load the next song, skipping."
                            )
                        except Exception as e:
                            log.warning(
                                f"Failed to send skip-notification in guild {self._guild.id}: {e}"
                            )
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

                    vc = self._guild.voice_client
                    assert isinstance(vc, discord.VoiceClient)
                    assert self.current_song is not None
                    # Local binding: pyright's attribute narrowing doesn't survive
                    # the awaits below, and it keeps every write in this iteration
                    # referring to the same song even if current_song is reassigned.
                    song = self.current_song
                    vc.play(
                        song,
                        after=lambda _: self.bot.loop.call_soon_threadsafe(
                            self.play_next.set
                        ),
                    )
                    play_start = time.time()  # capture immediately before any awaits

                    # Mirror now-playing song to Redis state. For a real queue item
                    # (should_pop_queue=True), atomically LPOP the Redis queue and
                    # write all now-playing state fields (including duration/uploader/
                    # requester) plus the now_playing display snapshot in a single
                    # MULTI/EXEC, eliminating the at-most-once window. A crash-
                    # recovered "current song" (should_pop_queue=False) was never on
                    # the Redis queue list, so only the state fields are written —
                    # LPOPing here would erroneously drop an unrelated, still-queued
                    # song.
                    if self._store is not None:
                        # play_start_epoch is backdated by the FFmpeg -ss start
                        # offset so the recovery position math
                        # (now - play_start_epoch - pauses) yields the true audio
                        # position, not merely time-since-vc.play(). Without this,
                        # ?t= songs and double-crash recoveries resume
                        # start_offset seconds early.
                        backdated_start = play_start - song.start_offset
                        dur = song.duration_secs or None
                        requester = song.requester
                        np_fields = _song_now_playing_fields(song)
                        if should_pop_queue:
                            await self._store.pop_queue_and_start_song(
                                url=song.webpage_url or "",
                                title=song.title or "",
                                play_start_epoch=backdated_start,
                                duration=dur,
                                uploader=song.uploader,
                                requester_id=requester.id if requester else None,
                                now_playing_fields=np_fields,
                            )
                        else:
                            await self._store.set_current_song_state(
                                url=song.webpage_url or "",
                                title=song.title or "",
                                play_start_epoch=backdated_start,
                                duration=dur,
                                uploader=song.uploader,
                                requester_id=requester.id if requester else None,
                                now_playing_fields=np_fields,
                            )

                    await self.update_activity(song)
                    await self._send_now_playing(song)

                    self._prefetch_task = asyncio.create_task(
                        self._prefetch_next_song()
                    )

                    await self.play_next.wait()

                    # Must fully retire before the next iteration's _send_now_playing()
                    # sends a new message — otherwise an in-flight message.edit() for
                    # this song could still be resolving concurrently with the new
                    # message being sent (see Design §4 of the progress-bar plan).
                    await self._cancel_progress_task()
                    await self._cancel_pause_debounce()

                    # Song has actually ended (naturally or via -skip) — one last,
                    # fire-and-forget edit so the bar always ends up showing fully
                    # completed instead of frozen at the last periodic tick's
                    # position. Captures current_song/_now_playing_message now,
                    # since both are about to be overwritten for the next song.
                    if (
                        self.current_song is not None
                        and self._now_playing_message is not None
                    ):
                        self._fire_finalize_now_playing(
                            self.current_song, self._now_playing_message
                        )

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

                    if self._store is not None:
                        await self._store.clear_song_end_state()

                    self.queue.task_done()
                    self.current_song = None
                    self.play_message = None  # -now must not serve the finished song
                    await self.update_activity(None)
                except asyncio.CancelledError:
                    span.set_attribute("loop.cancelled", True)
                    await self._cancel_progress_task()
                    await self._cancel_pause_debounce()
                    await self.update_activity(None)
                    raise
                except Exception as e:
                    record_span_error(span, e)
                    log.error(
                        f"Unhandled error in playback loop: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    if self._prefetch_task and not self._prefetch_task.done():
                        self._prefetch_task.cancel()
                    self._prefetch_task = None
                    await self._cancel_progress_task()
                    await self._cancel_pause_debounce()
                    prefetched_song = None
                    self.current_song = None
                    self.play_message = None
                    if self._store is not None:
                        await self._store.clear_song_end_state()
                    try:
                        await send_embed(
                            self._channel,
                            "Playback error — skipping song",
                            f"**{type(e).__name__}:** {e}",
                            discord.Color.red(),
                            footer=trace_footer(span),
                        )
                    except Exception as e:
                        log.warning(
                            f"Failed to send playback-error embed in guild {self._guild.id}: {e}"
                        )
