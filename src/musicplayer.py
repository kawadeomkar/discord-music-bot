import asyncio
import datetime
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Union
from zoneinfo import ZoneInfo

import async_timeout
import discord
from discord.ext import commands

from opentelemetry import trace

from src import config
from src.guild_history import GuildHistory
from src.guild_queue import GuildQueue, ShuffleOutcome, is_persisted
from src.guild_state import HistoryEntry, NowPlayingData, SongQueueEntry
from src.redis_client import GuildRedisStore, cache_get
from src.sources import YTSource
from src.telemetry import get_tracer
from src.util import (
    cancel_task,
    notice_embed,
    record_span_error,
    send_embed,
    trace_footer,
    get_logger,
)
from src.youtube import YTDL, QueueObject

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# ETAs in queue_embed() are rendered in Pacific time. This is intentional for a
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

# ── -playnow interjection (docs/PLAYNOW_PROPOSAL.md) ──────────────────────────
# Songs with less than this many seconds remaining get no resume entry when
# interjected — there is nothing meaningful to return to.
_MIN_RESUME_REMAINING_SECS = 5
# EOF guard for the resume seek (duration metadata is imprecise), matching the
# crash-recovery position cap in _restore_state().
_RESUME_EOF_MARGIN_SECS = 10


@dataclass(frozen=True)
class InterjectOutcome:
    """What MusicPlayer.interject() did — everything the -playnow command
    needs for its confirmation wording."""

    interrupted_title: str
    # None → no resume entry was created (the interrupted song was itself an
    # interjection, was nearly finished, or had no webpage_url to rebuild from).
    resume_position: Optional[int]
    was_paused: bool
    replaced: bool  # the interrupted song was itself a -playnow interjection

    @property
    def resume_position_str(self) -> str:
        return _fmt_duration(self.resume_position or 0)


def _remaining_secs(item: QueueObject) -> Optional[int]:
    """A queued item's expected playtime: full duration, minus the resume
    offset for a -playnow resume entry — it only plays its tail, so ETA math
    counting its full duration would overestimate everything behind it."""
    if item.duration is None:
        return None
    if item.is_resume and item.ts:
        return max(0, item.duration - item.ts)
    return item.duration


def _build_progress_bar(
    elapsed_secs: float, duration_secs: int, width: int = _BAR_WIDTH
) -> str:
    if duration_secs <= 0:
        return ""
    # Clamp before formatting so the elapsed label can never overshoot the
    # duration label (imprecise duration metadata plus an FFmpeg -ss start
    # offset can push the raw position past the reported duration).
    elapsed_secs = max(0.0, min(elapsed_secs, float(duration_secs)))
    ratio = elapsed_secs / duration_secs
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
    Redis-recovery (NowPlayingData-backed) now-playing embed builders."""
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
        "play_message",
        "history",
        "volume",
        "_player",
        "_prefetch_task",
        "store",
        "_restore_task",
        "_restore_complete",
        "_background_tasks",
        "_progress_task",
        "_np_host_message",
        "_np_host_own_embeds",
        "_np_host_dedicated",
        "_np_edit_lock",
        "_pause_debounce_task",
        "_skip_history_for",
    )

    bot: commands.Bot
    _guild: discord.Guild
    _channel: discord.TextChannel
    _last_author: Union[discord.User, discord.Member]
    _cog: Any
    current_song: Optional[YTDL]
    play_next: asyncio.Event
    queue: GuildQueue
    play_message: Optional[discord.Embed]
    history: GuildHistory
    volume: float
    _player: Optional[asyncio.Task]
    _prefetch_task: Optional[asyncio.Task]
    store: Optional[GuildRedisStore]
    _restore_task: Optional[asyncio.Task]
    _restore_complete: asyncio.Event
    _background_tasks: set
    _progress_task: Optional[asyncio.Task]
    _np_host_message: Optional[discord.Message]
    _np_host_own_embeds: list[discord.Embed]
    _np_host_dedicated: bool
    _np_edit_lock: asyncio.Lock
    _pause_debounce_task: Optional[asyncio.Task]
    _skip_history_for: Optional[YTDL]

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

        self.play_message = None
        self.volume = 1.0

        self.store = (
            GuildRedisStore(redis, self._guild.id) if redis is not None else None
        )
        # All queue state (asyncio queue, display order, Redis mirror, bulk
        # mutex, cleared-flag) lives behind this one object — see guild_queue.py.
        self.queue = GuildQueue(guild, self.store)
        # Played-song history (in-memory ring + Redis mirror) — see guild_history.py.
        self.history = GuildHistory(self.store)
        self._player: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        self._restore_task: Optional[asyncio.Task] = None
        self._restore_complete = asyncio.Event()
        self._background_tasks: set = set()
        self._progress_task: Optional[asyncio.Task] = None
        # Now-playing host state: the one message currently carrying the NP
        # embed block, its own (cached, static) embeds that follow the block,
        # and whether it's a dedicated NP message (deleted on retire) or a
        # command response (strip-edited on retire). See
        # docs/NOW_PLAYING_EMBED_ATTACH_PLAN.md.
        self._np_host_message: Optional[discord.Message] = None
        self._np_host_own_embeds: list[discord.Embed] = []
        self._np_host_dedicated: bool = False
        self._np_edit_lock = asyncio.Lock()
        self._pause_debounce_task: Optional[asyncio.Task] = None
        # Set by interject() to the song it stopped WITH a resume entry
        # pending: the stop-transition's history step skips that song's add,
        # so it is recorded once — when its tail finishes — instead of twice.
        # Holds the song's identity (not a bare flag) because the song can end
        # naturally during interject()'s awaits, after its own history step
        # already ran; a stale boolean would then eat the NEXT song's entry.
        self._skip_history_for: Optional[YTDL] = None

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
        if self.store is not None:
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
            if item.is_resume and item.ts:
                ts_note = f"  ·  ⏮ resumes at `{_fmt_duration(item.ts)}`"
            elif item.ts:
                ts_note = f"  ·  starts at `{item.ts}s`"
            else:
                ts_note = ""
            line = (
                f"`{index}` [**{title}**]({item.webpage_url}) · `{dur}`{ts_note} · Est. playing at {est_str}\n"
                f"{channel} · {requester}"
            )
            remaining = _remaining_secs(item)
            if remaining is not None:
                cumulative_secs += remaining
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
        seed as queue_embed()/_build_next_up_embed() so all three stay consistent.
        """
        now_pst, cumulative_secs, uncertain = self._queue_eta_seed()
        for item in self.queue.display_items():
            remaining = _remaining_secs(item) if isinstance(item, QueueObject) else None
            if remaining is not None:
                cumulative_secs += remaining
            else:
                uncertain = True
        est_dt = now_pst + datetime.timedelta(seconds=cumulative_secs)
        return _fmt_eta(est_dt, uncertain)

    def queue_embed(self) -> discord.Embed:
        items = self.queue.display_items()
        total = len(items)

        total_secs = 0
        duration_partial = False
        for item in items:
            remaining = _remaining_secs(item) if isinstance(item, QueueObject) else None
            if remaining is not None:
                total_secs += remaining
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
        if self.store is None:
            self._restore_complete.set()
            return
        try:
            await self.bot.wait_until_ready()
            with _tracer.start_as_current_span(
                "player.state_restore",
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    # One pipelined read covers the whole playback aggregate:
                    # state hash, pending queue, now-playing snapshot, history.
                    snapshot = await self.store.get_playback_snapshot()
                    if snapshot is None:
                        # Redis read failed — abort restore rather than proceeding
                        # with fabricated defaults. _restore_complete is still set
                        # in the finally block, so loop() is never blocked.
                        log.warning(
                            f"State restore aborted for guild {self._guild.id}: "
                            f"Redis unavailable"
                        )
                        return
                    guild_state = snapshot.state

                    # Restore volume — only when a value was actually stored.
                    # An unconditional assign would clobber a concurrently
                    # issued -volume command with the default.
                    if guild_state.volume is not None:
                        self.volume = guild_state.volume

                    # Now-playing display snapshot — so -now works if a song
                    # was playing on recovery.
                    if snapshot.now_playing is not None:
                        self.play_message = self._build_now_playing_embed_from_data(
                            snapshot.now_playing
                        )

                    # Re-queue song that was playing when the bot crashed (at-most-once delivery).
                    # current_song_url is set atomically with the LPOP; a non-empty value means
                    # the bot died after the transaction committed but before the song finished.
                    if guild_state.has_crashed_song:
                        # Approximate playback position at crash time — pure math
                        # on the snapshot.
                        position = guild_state.crashed_position_at(time.time())
                        if position is not None:
                            # Cap at duration − 10s to prevent FFmpeg seeking past
                            # EOF. Kept inside a narrow try/except: a malformed
                            # cached duration must degrade to "no cap", not abort
                            # the whole restore.
                            try:
                                stream_data = await cache_get(
                                    self.store.redis,
                                    f"ytdl:stream:{guild_state.current_song_url}",
                                )
                                if stream_data is not None:
                                    raw_duration = stream_data.get("duration")
                                    if raw_duration is not None:
                                        position = min(
                                            position, max(0, int(raw_duration) - 10)
                                        )
                            except Exception as pos_err:
                                log.warning(
                                    f"Failed to cap recovery position: {pos_err}"
                                )
                            log.info(
                                f"Computed recovery position {position}s for "
                                f"'{guild_state.current_song_title}'"
                            )

                        # The crashed current_song_* state fields ARE a queue
                        # entry — the one the start transaction LPOPed. Rebuild
                        # it and re-queue through the same rehydration path as
                        # every other entry. The crashed requester chain falls
                        # back to guild.me then guild.owner.
                        crashed_entry = SongQueueEntry.from_crashed_state(
                            guild_state, position=position
                        )
                        if (
                            crashed_entry is not None
                            and await self.queue.restore_crashed(
                                crashed_entry,
                                requester_fallback=self._guild.me or self._guild.owner,
                            )
                        ):
                            log.info(
                                f"Re-queued crashed song "
                                f"'{guild_state.current_song_title}' for guild {self._guild.id}"
                            )
                        # Always clear regardless of whether the song could be
                        # re-queued; leaving current_song_url set would cause every
                        # subsequent restart to re-enter this block and never
                        # escape until the TTL expires.
                        await self.store.clear_song_end_state()

                    # Restore the pending queue (after the crashed head, so the
                    # interrupted song plays first).
                    count = await self.queue.restore_entries(snapshot.queue)
                    if count:
                        log.info(
                            f"Restored {count} queued songs for guild {self._guild.id}"
                        )

                    # Corrupt entries were already dropped at parse time.
                    self.history.restore(snapshot.history)

                    span.set_attribute("restore.queue_count", count)
                    span.set_attribute(
                        "restore.crashed_song", guild_state.has_crashed_song
                    )

                except Exception as e:
                    record_span_error(span, e)
                    log.error(
                        f"State restore failed for guild {self._guild.id}: {e}",
                        exc_info=True,
                    )
                    return

                # Refresh TTL on all guild keys after successful restore.
                await self.store.refresh_ttl()
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
        """Enqueue and (optionally) kick off stream pre-fetch.

        prefetch=False for bulk playlist enqueues — the Redis mirror is
        written in one batch round-trip, and no per-item prefetch tasks are
        spawned (N concurrent prefetches saturate the thread pool and produce
        stream URLs that expire before the song reaches playback position;
        _prefetch_next_song handles one-ahead prefetch naturally as songs play).
        """
        items: list[Union[QueueObject, YTSource]]
        if isinstance(obj, list):
            items = list(obj)  # type: ignore[arg-type]
        else:
            items = [obj]
        await self.queue.put(items, batch=not prefetch)
        if prefetch and self.store is not None:
            for item in items:
                if isinstance(item, QueueObject):
                    self._spawn_background(
                        YTDL.prefetch_stream(item, redis=self.store.redis)
                    )

    async def queue_get(self) -> Union[QueueObject, YTSource]:
        return await self.queue.get()

    async def _cancel_prefetch(self) -> None:
        """Cancel any in-flight prefetch task and wait for it to finish.

        Must be called before any bulk queue mutation (clear, shuffle, remove) so that
        the item the prefetch already dequeued via get_nowait() is returned to the
        front of the pending queue (requeue_front, in its CancelledError handler)
        before the mutation drains — the mutation then clears/shuffles/removes it
        together with everything else instead of stranding it.

        Note: if the prefetch task is blocked inside run_in_executor (a yt-dlp
        extraction), cancellation cannot interrupt the running thread. The await
        blocks until the thread exits, which may take up to socket_timeout seconds.
        This is acceptable at single-guild scale.
        """
        await cancel_task(self._prefetch_task)

    async def queue_clear(self) -> List[str]:
        await self._cancel_prefetch()  # before the drain — see _cancel_prefetch
        cleared_items = await self.queue.clear()
        return [
            (
                item.title
                if isinstance(item, QueueObject)
                else (item.ytsearch or item.url or "?").removeprefix("ytsearch:")
            )
            for item in cleared_items
        ]

    async def queue_shuffle(self) -> str:
        # Cancel BEFORE the too-few guard (inside shuffle()), matching the
        # original ordering: a prefetch holding a dequeued item must be
        # accounted for even when the shuffle ends up a no-op.
        await self._cancel_prefetch()
        outcome = await self.queue.shuffle()
        if outcome is ShuffleOutcome.TOO_FEW_SONGS:
            return "There must be at least 3 songs to shuffle the queue"
        return "Shuffled!"

    async def queue_remove(self, url: str) -> list[int]:
        """Remove all queued items whose webpage_url (QueueObject) or url (YTSource) matches url.

        Returns a list of 1-indexed queue positions that were removed.
        """
        await self._cancel_prefetch()
        return await self.queue.remove(url)

    # ── Embed building ────────────────────────────────────────────────────────

    def _build_now_playing_embed(
        self, song: YTDL, *, position_override: Optional[float] = None
    ) -> discord.Embed:
        """position_override lets a caller render the bar at a specific position
        rather than song.position_secs's live value — used by _finalize_now_playing()
        to show the bar fully completed once the song has actually ended."""
        lines = []
        position = 0.0
        if song.duration_secs > 0:
            position = (
                position_override
                if position_override is not None
                else song.position_secs
            )
            bar = _build_progress_bar(position, song.duration_secs)
            if bar:
                # Bar sits directly under the title, above the requester line,
                # with a blank line between them for visual separation.
                lines.append(bar)
                lines.append("")
        requester_line = f"Requester: [{_requester_mention(song.requester)}]"
        if song.duration_secs > 0:
            # Remaining time, not total duration: a song started mid-stream
            # (?t= link, crash recovery, -playnow resume) finishes sooner than
            # its full length from now.
            remaining = max(0, song.duration_secs - int(position))
            requester_line += f"  ·  Estimated finish: {_fmt_finish_time(remaining)}"
        lines.append(requester_line)
        description = "\n".join(lines)
        fields = NowPlayingData.from_song(song)
        return _build_now_playing_base_embed(
            title=f"**Now playing:** {song.title}",
            description=description,
            webpage_url=fields.webpage_url,
            duration=fields.duration,
            uploader=fields.uploader,
            views=fields.view_count,
            likes=fields.like_count,
            abr=fields.abr,
            asr=fields.asr,
            acodec=fields.acodec,
            thumbnail=fields.thumbnail,
        )

    def build_pause_confirmation_embed(self) -> Optional[discord.Embed]:
        """Slim confirmation embed for the -pause command: just the pause
        position. The -pause response message hosts the live NP block directly
        below this embed (MusicContext attach), so the bar, requester, link
        fields, and thumbnail would all render twice if repeated here — the
        one thing the NP block does NOT show is the paused state itself.
        position_secs is frozen while paused, so it captures the exact pause
        point (including any FFmpeg -ss start offset). Returns None when
        there's no live song to describe."""
        song = self.current_song
        if song is None:
            return None
        position = int(song.position_secs)
        duration_secs = song.duration_secs
        if duration_secs > 0:
            paused_at = f"{_fmt_duration(position)} / {_fmt_duration(duration_secs)}"
        else:
            paused_at = _fmt_duration(position)
        return discord.Embed(
            title=f"⏸️ Paused: {song.title}",
            description=f"Paused at: `{paused_at}`",
            color=discord.Color.orange(),
        )

    @staticmethod
    def _build_now_playing_embed_from_data(data: NowPlayingData) -> discord.Embed:
        """Reconstruct a now-playing embed from the recovered Redis snapshot."""
        return _build_now_playing_base_embed(
            title=f"**Now playing:** {data.title}",
            description=f"Requester: [{data.requester_mention}]",
            webpage_url=data.webpage_url,
            duration=data.duration,
            uploader=data.uploader,
            views=data.view_count,
            likes=data.like_count,
            abr=data.abr,
            asr=data.asr,
            acodec=data.acodec,
            thumbnail=data.thumbnail,
        )

    def _build_next_up_embed(self) -> Optional[discord.Embed]:
        item = self.queue.peek_next()
        if item is None:
            return None
        now_pst, cumulative_secs, uncertain = self._queue_eta_seed()
        description, _, _ = self._format_queue_line(
            item, 1, now_pst, cumulative_secs, uncertain
        )
        return discord.Embed(
            title="Up next",
            description=description,
            color=discord.Color.blue(),
        )

    # ── Now-playing host management ───────────────────────────────────────────
    # The NP embed block lives in exactly one "host" message at a time — always
    # the newest bot message in the channel, so the progress bar never gets
    # buried. Command responses adopt the block by prepending it to their own
    # embeds at send time (MusicContext.send) — the block leads the message,
    # the response's own embeds follow it. The previous host is retired:
    # deleted if it was a dedicated NP message, strip-edited back to its own
    # embeds otherwise. Full design: docs/NOW_PLAYING_EMBED_ATTACH_PLAN.md.

    def np_embed_block(
        self, *, now_playing: Optional[discord.Embed] = None
    ) -> list[discord.Embed]:
        """The [now_playing, next_up?] embed block, or [] when no song is live.
        The single place encoding the block's internal order. `now_playing`
        lets a caller that already built this song's NP embed supply it instead
        of building an identical one (_send_now_playing stores it as
        play_message first)."""
        song = self.current_song
        if song is None:
            return []
        block = [
            (
                now_playing
                if now_playing is not None
                else self._build_now_playing_embed(song)
            )
        ]
        next_up = self._build_next_up_embed()
        if next_up is not None:
            block.append(next_up)
        return block

    def _adopt_np_host(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        dedicated: bool = False,
    ) -> None:
        """Pointer-first host swap. The pointer/own-embeds/dedicated update is
        synchronous (atomic on the event loop), so any progress tick that
        starts after this call targets the new host. Retiring the old host is
        fire-and-forget; the lock inside _retire_np_host orders it after any
        in-flight tick edit against the old message."""
        old_msg = self._np_host_message
        old_own = self._np_host_own_embeds
        old_dedicated = self._np_host_dedicated
        if old_msg is not None and message.id < old_msg.id:
            # Two overlapping sends can complete out of order: channel position
            # is send-START order, but adopts run in send-RETURN order. Adopting
            # the older message would pull the block up from the true bottom —
            # keep the newer host and shed the older message's block instead.
            self._spawn_background(self._retire_np_host(message, own_embeds, dedicated))
            return
        self._np_host_message = message
        self._np_host_own_embeds = own_embeds
        self._np_host_dedicated = dedicated
        if old_msg is not None and old_msg.id != message.id:
            self._spawn_background(
                self._retire_np_host(old_msg, old_own, old_dedicated)
            )

    def _adopt_np_host_if_current(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        song: Optional[YTDL],
        *,
        dedicated: bool = False,
    ) -> bool:
        """Adopt gate for every attach site: the NP block inside `message` was
        built for `song` BEFORE the send's await, and the song may have ended
        (or been replaced) while the HTTP call was in flight. Adopting then
        would install a stale block as host — and delete-retire the next
        song's freshly sent NP message, or (with an empty queue) leave a bogus
        frozen block that nothing ever cleans up. Instead the just-sent
        message sheds the stale block it is carrying (strip-edit back to its
        own embeds, or delete when it is a dedicated NP message). Returns True
        when the message was adopted."""
        if song is not None and self.current_song is song:
            self._adopt_np_host(message, own_embeds, dedicated=dedicated)
            return True
        self._spawn_background(self._retire_np_host(message, own_embeds, dedicated))
        return False

    async def _retire_np_host(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        dedicated: bool,
    ) -> None:
        """Remove the NP block from a message that is no longer the host. Holds
        the edit lock so a tick edit already in flight against this message
        finishes first — two concurrent PATCHes resolve last-write-wins server
        side, and a tick landing after the strip would resurrect the NP block
        on the retired host with nothing left to clean it up."""
        async with self._np_edit_lock:
            try:
                if dedicated:
                    await message.delete()  # pure NP message → remove entirely
                else:
                    # response → strip NP block, keep its own embeds
                    await message.edit(embeds=own_embeds)
            except discord.NotFound:
                pass  # user already deleted it — nothing to retire
            except discord.HTTPException as e:
                log.warning(f"NP host retire failed for guild {self._guild.id}: {e}")

    def _release_np_host(self) -> None:
        """Clear host state WITHOUT retiring the message. Used at song end: the
        finished song's completed bar stays in the channel as a historical
        record, and the next song's adopt sees no old host to retire."""
        self._np_host_message = None
        self._np_host_own_embeds = []
        self._np_host_dedicated = False

    async def retire_np_host_on_stop(self) -> None:
        """-stop / alone-disconnect teardown: dispose of the host so no message
        keeps a live-looking bar for a player that no longer exists. Song end
        RELEASES instead — a completed bar is a truthful historical record — but
        a bar frozen mid-song on a stopped player is not, so here the dedicated
        NP message is deleted and a response host is stripped back to its own
        embeds. Called by cleanup() after the progress/loop tasks are cancelled,
        so no tick can race the retire."""
        host = self._np_host_message
        own = self._np_host_own_embeds
        dedicated = self._np_host_dedicated
        if host is None:
            return
        self._release_np_host()
        await self._retire_np_host(host, own, dedicated)

    async def send_with_np(
        self,
        content: Optional[str] = None,
        *,
        embed: Optional[discord.Embed] = None,
    ) -> discord.Message:
        """Player-initiated channel sends that bypass ctx.send (and therefore
        MusicContext's attach hook) but must still keep the NP block at the
        bottom — same splice-send-adopt sequence as MusicContext.send."""
        own = [embed] if embed is not None else []
        song = self.current_song  # the song the block below is built for
        block = self.np_embed_block()
        embeds = block + own
        if embeds:
            message = await self._channel.send(content, embeds=embeds)
        else:
            message = await self._channel.send(content)
        if block:
            self._adopt_np_host_if_current(message, own, song)
        return message

    async def update_activity(self, song: Optional[YTDL] = None) -> None:
        if song is not None:
            timestamps: dict = {}
            vc = self._guild.voice_client
            is_paused = isinstance(vc, discord.VoiceClient) and vc.is_paused()
            if not is_paused:
                # Backdated by the true audio position (not always "now") so
                # that resuming mid-song still lands `end` the correct remaining
                # duration in the future, and a -ss/crash-recovered song's
                # tooltip agrees with the progress bar (both read position_secs).
                now_ms = int(time.time() * 1000)
                position_ms = int(song.position_secs * 1000)
                timestamps["start"] = now_ms - position_ms
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
        if self.store is not None:
            await self.store.on_pause(time.time())
        self.mark_paused()

    async def resume(self, vc: discord.VoiceClient) -> None:
        vc.resume()
        if self.store is not None:
            await self.store.on_resume(time.time())
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
        if self._progress_task is not None and self._np_host_message is not None:
            self._spawn_background(self._edit_now_playing_once())
        self._spawn_background(self.update_activity(self.current_song))

    # ── -playnow interjection ─────────────────────────────────────────────────

    @_tracer.start_as_current_span("player.interject")
    async def interject(
        self, qobj: QueueObject, vc: discord.VoiceClient
    ) -> Optional[InterjectOutcome]:
        """Play `qobj` immediately; the interrupted song returns afterwards.

        Mechanism (docs/PLAYNOW_PROPOSAL.md §4): capture the current song's
        exact position (frame-counted position_secs — frozen if paused),
        front-insert [qobj, resume-entry(ts=position)] on the queue, and stop
        the current song. The playback loop's ordinary dequeue → FFmpeg -ss →
        play cycle does the rest, and because both entries are persisted
        (LPUSHed to Redis), crash recovery mid-interjection works unchanged.

        Replace semantics: when the interrupted song is itself an interjection
        (current.interjected), no resume entry is built for it — the ORIGINAL
        song's resume entry, still at the queue front, is untouched.

        Returns None when there is no current song (or it ended during the
        prefetch neutralization) — the command falls back to a plain
        front-enqueue. Residual race, documented not defended: if the current
        song ends naturally while put_front below awaits, the front-inserted
        entries still play next (after whatever the loop already committed),
        and a just-finished song's resume entry replays its final seconds.
        The widest variant of that window is the loop awaiting a STILL-RUNNING
        prefetch it claimed before this method ran — put_front then executes
        against a real in-flight head, which its rebuild branch handles (see
        GuildQueue.put_front: that branch is load-bearing here, not
        defensive).
        """
        current = self.current_song
        if current is None:
            return None
        span = trace.get_current_span()
        span.set_attribute("discord.guild_id", str(self._guild.id))
        span.set_attribute("song.interjected_title", qobj.title or "")

        # A completed prefetch bypasses the queue and would play INSTEAD of
        # the front-inserted qobj — take it off the board first.
        await self._neutralize_prefetch()

        # Re-check after the awaits above (cancellation can block up to
        # yt-dlp's socket timeout): if the song ended and the loop moved on,
        # there is nothing to interrupt — bail to the command's fallback
        # rather than building a resume entry for a finished song.
        if self.current_song is not current:
            return None

        was_paused = vc.is_paused()
        replaced = current.interjected
        position = int(current.position_secs)
        resume: Optional[QueueObject] = None
        if not replaced and current.webpage_url:
            # Near-end check on the RAW position — the EOF cap below pulls the
            # position back by its margin, which would mask "almost over".
            near_end = (
                current.duration_secs > 0
                and current.duration_secs - position < _MIN_RESUME_REMAINING_SECS
            )
            if current.duration_secs > 0:
                # EOF guard, matching the crash-recovery cap: imprecise
                # duration metadata must not make FFmpeg seek past the end.
                position = min(
                    position,
                    max(0, current.duration_secs - _RESUME_EOF_MARGIN_SECS),
                )
            if not near_end:
                resume = QueueObject(
                    current.webpage_url,
                    current.title or "",
                    current.requester or self._last_author,
                    ts=position,
                    duration=current.duration_secs or None,
                    uploader=current.uploader,
                    thumbnail=current.thumbnail,
                    is_resume=True,
                    start_paused=was_paused,
                )

        items = [qobj] if resume is None else [qobj, resume]
        await self.queue.put_front(items)

        # Stop only if the song we measured is still the one playing — if the
        # loop already moved on, the front-inserted entries play next anyway
        # and stopping would kill the WRONG (next) song.
        if self.current_song is current:
            if resume is not None:
                # The interrupted song returns — record it in history once,
                # when its tail finishes, not also now. A replaced
                # interjection (no resume) keeps its entry, matching -skip.
                self._skip_history_for = current
            vc.stop()

        span.set_attribute("interject.replaced", replaced)
        span.set_attribute("interject.resume_position", position if resume else -1)
        return InterjectOutcome(
            interrupted_title=current.title or "Unknown",
            resume_position=position if resume is not None else None,
            was_paused=was_paused,
            replaced=replaced,
        )

    async def _neutralize_prefetch(self) -> None:
        """Take the in-flight prefetch (if any) off the board so the loop's
        next dequeue comes from the queue head.

        Claim-then-settle: _prefetch_task is nulled synchronously before any
        await, and the loop's matching read in its prefetch-await step is also
        a synchronous read-and-null — so exactly one of interject()/loop()
        consumes any given prefetch result.

        - running task → cancel; its CancelledError handler returns the
          dequeued item to the pending front (requeue_front), exactly as the
          bulk mutations rely on.
        - completed task → rebuild an equivalent QueueObject from the resolved
          song, return it to the pending front, and kill its FFmpeg
          subprocess. The display/Redis legs never moved for the prefetch's
          dequeue, so the rebuilt item re-aligns all three legs
          (requeue_front's documented "resolved form" tolerance).
        - completed-with-None → the prefetch failed and already retired its
          own dequeue (finish_failed_dequeue); nothing to undo.
        """
        task = self._prefetch_task
        self._prefetch_task = None
        if task is None:
            return
        if not task.done():
            await cancel_task(task)
            return
        try:
            song = task.result()
        except asyncio.CancelledError, Exception:
            song = None
        if song is None:
            return
        # Carry the -ss offset and every -playnow flag through the rebuild —
        # dropping them here would make a neutralized resume entry restart its
        # song from 0:00 (unpaused, unannounced) after the nested interjection,
        # and lose the ?t= offset of an ordinary prefetched song.
        rebuilt = QueueObject(
            song.webpage_url or "",
            song.title or "",
            song.requester or self._last_author,
            ts=song.start_offset or None,
            duration=song.duration_secs or None,
            uploader=song.uploader,
            thumbnail=song.thumbnail,
            interjected=song.interjected,
            is_resume=song.is_resume,
            start_paused=song.start_paused,
        )
        self.queue.requeue_front(rebuilt)
        song.cleanup()

    async def _announce_resume(self, song: YTDL) -> None:
        """One-line notice when an interrupted song returns. Sent from the
        loop's start path — yt_stream's construction-time "Starting song at…"
        notice is suppressed for resume entries because prefetch constructs
        them while the interjected song is still playing. Plain channel send,
        NOT send_with_np: this song's NP host hasn't been sent yet, and
        send_with_np would adopt the notice as host only for
        _send_now_playing to immediately retire it."""
        position = _fmt_duration(int(song.position_secs))
        if song.start_paused:
            text = (
                f"⏮ Returned to **{song.title}** at `{position}` — still paused. "
                f"Use `-resume` to continue."
            )
        else:
            text = f"⏮ Resuming **{song.title}** at `{position}`"
        try:
            await self._channel.send(embed=notice_embed(text, discord.Color.blue()))
        except Exception as e:
            log.warning(f"Failed to send resume notice in guild {self._guild.id}: {e}")

    # ── Playback pipeline helpers ─────────────────────────────────────────────

    async def _resolve_source(
        self, source: Union[QueueObject, YTSource]
    ) -> QueueObject:
        if isinstance(source, YTSource):
            return await YTDL.yt_source(
                self._last_author,
                source.ytsearch or "",
                source.process or False,
                redis=self.store.redis if self.store is not None else None,
            )
        return source

    async def _stream_source(self, source: QueueObject) -> Optional[YTDL]:
        try:
            return await YTDL.yt_stream(
                source,
                self._channel,
                volume=self.volume,
                redis=self.store.redis if self.store is not None else None,
            )
        except Exception as e:
            log.error(f"Error processing song: {type(e).__name__}: {e}", exc_info=True)
            return None

    async def _send_np_host_message(
        self, *, now_playing: Optional[discord.Embed] = None
    ) -> Optional[discord.Message]:
        """Send a dedicated NP host message (its embeds are only the NP block)
        and adopt it — the adopt retires whatever hosted the block before.
        Returns None when there is no live song to describe, or when the song
        changed while the send was in flight (the stale message is deleted
        instead of adopted)."""
        song = self.current_song
        block = self.np_embed_block(now_playing=now_playing)
        if not block:
            return None
        message = await self._channel.send(embeds=block)
        if not self._adopt_np_host_if_current(message, [], song, dedicated=True):
            return None
        return message

    async def repin_now_playing(self) -> bool:
        """-now: re-host the NP block at the bottom of the channel as a fresh
        dedicated message. Does NOT touch _progress_task — the running updater
        follows the host pointer and picks up the new message on its next tick.
        Returns False when no song is live (including a song that ended while
        the send was in flight) so the command can respond another way."""
        return await self._send_np_host_message() is not None

    async def rehost_np_after_resume(self) -> None:
        """-resume: if a command response currently hosts the block — typically
        the -pause confirmation — re-host onto a fresh dedicated message. The
        old response is strip-retired back to its own embeds, so a "⏸️ Paused
        at…" line becomes plain history instead of being re-rendered beneath a
        live, advancing bar by every tick for the rest of the song. A dedicated
        host has no stale state to shed, so it is left alone."""
        if self._np_host_message is None or self._np_host_dedicated:
            return
        await self._send_np_host_message()

    async def _send_now_playing(self, song: YTDL) -> None:
        # Release before attempting the send (not after failure) so a failed/
        # partial send never leaves the host pointing at the *previous* song's
        # message — a stale host would let a later mark_paused()/mark_resumed()
        # on the new song silently overwrite the old song's already-sent embed.
        self._release_np_host()
        try:
            self.play_message = self._build_now_playing_embed(song)
            message = await self._send_np_host_message(now_playing=self.play_message)
            if message is None:
                return
            if song.duration_secs >= 5:
                self._progress_task = asyncio.create_task(self._progress_updater(song))
        except Exception as e:
            log.error(f"embed error: {e}")

    async def _push_np_edit(
        self,
        song: YTDL,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        position_override: Optional[float] = None,
    ) -> bool:
        """Rebuild the host's embeds — a fresh NP block followed by the host's
        cached (static) own embeds — and push a single edit. Shared by the
        periodic tick, the debounced pause/resume refresh, and the song-end
        finalize edit. Returns False if the message no longer exists (deleted)
        so callers can release the host; the finalize path ignores the return
        value.
        """
        try:
            embed = self._build_now_playing_embed(
                song, position_override=position_override
            )
            next_up = self._build_next_up_embed()
            embeds = [embed] + ([next_up] if next_up else []) + own_embeds
            # ≤10 is Discord's per-message embed cap: an attach accepted at the
            # cap can overflow here if a next-up embed appears later. Drop the
            # own-embeds tail, never the block (parity with MusicContext.send's
            # attach guard; unreachable with current commands, max own = 1).
            embeds = embeds[:10]
            await message.edit(embeds=embeds)
            return True
        except discord.NotFound:
            return False
        except discord.HTTPException as e:
            log.warning(f"Now-playing edit failed for guild {self._guild.id}: {e}")
            return True

    async def _edit_now_playing_once(self) -> None:
        """Rebuild and push a single embed edit outside the periodic tick — used
        for the debounced pause/resume refresh (mark_paused()/mark_resumed()).
        Holds the edit lock and re-reads the host inside it: an edit landing
        after a retire's strip would resurrect the NP block on the old host."""
        song = self.current_song
        if song is None:
            return
        async with self._np_edit_lock:
            host = self._np_host_message
            if host is None:
                return
            if not await self._push_np_edit(song, host, self._np_host_own_embeds):
                # Adopt is lock-free, so a command response may have swapped in
                # a new host while this PATCH was in flight — releasing then
                # would orphan the new host's block. Only release OUR host.
                if self._np_host_message is host:
                    self._release_np_host()

    async def _finalize_now_playing(
        self,
        song: YTDL,
        message: discord.Message,
        own_embeds: list[discord.Embed],
    ) -> None:
        """One last embed edit once a song has actually ended, showing the bar
        fully completed (position == duration) rather than left frozen wherever
        the last periodic tick happened to land. song/message/own_embeds are
        captured by the caller rather than read off self.current_song/
        self._np_host_message at run time, since both may already point at the
        next song by the time this (fire-and-forget) task actually runs. The
        host has already been released by loop() before this fires, so no tick
        or retire can START against the message — but a debounce-spawned
        _edit_now_playing_once that captured the host before the release can
        still have a PATCH in flight (resume ≤ debounce window before song
        end); it holds _np_edit_lock across its edit, so taking the lock here
        orders this completed-bar write after it (last write wins).
        """
        if song.duration_secs <= 0:
            return  # no bar was ever shown for this song — nothing to finalize
        async with self._np_edit_lock:
            await self._push_np_edit(
                song, message, own_embeds, position_override=song.duration_secs
            )

    def _spawn_background(self, coro: Any) -> asyncio.Task:
        """Create a fire-and-forget task tracked in _background_tasks."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _fire_finalize_now_playing(
        self, song: YTDL, message: discord.Message, own_embeds: list[discord.Embed]
    ) -> None:
        self._spawn_background(self._finalize_now_playing(song, message, own_embeds))

    async def _progress_updater(self, song: YTDL) -> None:
        interval = config.NOW_PLAYING_UPDATE_INTERVAL_SECS
        try:
            while True:
                await asyncio.sleep(interval)
                vc = self._guild.voice_client
                if not isinstance(vc, discord.VoiceClient) or vc.source is not song:
                    return  # song changed under us; loop() owns cancellation but guard defensively
                if vc.is_paused():
                    continue  # frozen — mark_resumed() below fires a debounced edit instead
                async with self._np_edit_lock:
                    host = self._np_host_message  # re-read INSIDE the lock: a
                    # host swap during this tick's sleep must not leave this
                    # edit targeting the old, about-to-be-stripped message
                    if host is None:
                        continue  # dormant: no visible NP until re-hosted
                    if not await self._push_np_edit(
                        song, host, self._np_host_own_embeds
                    ):
                        # host deleted by a user — go dormant rather than die;
                        # the next command response (or -now) re-hosts the
                        # block. Adopt is lock-free, so if a new host was
                        # swapped in while this PATCH was in flight, releasing
                        # would orphan it — only release OUR host.
                        if self._np_host_message is host:
                            self._release_np_host()
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
        Accounts for its own dequeue on every non-success path: cancellation
        returns the item to the front of the line (requeue_front — a bulk
        mutation is about to drain/reorder it with everything else), and
        failure retires the item on all three queue legs (finish_failed_dequeue
        — leaving the display/Redis heads in place would make the next
        commit retire the wrong entry). On success the dequeue stays open and
        the main loop's commit/task_done() closes it.
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
            song = await self._stream_source(source)
        except asyncio.CancelledError:
            self.queue.requeue_front(source)
            raise
        except Exception as e:
            record_span_error(trace.get_current_span(), e)
            log.error(f"Prefetch error: {type(e).__name__}: {e}", exc_info=True)
            await self.queue.finish_failed_dequeue(source, context="prefetch failure")
            return None
        if song is None:
            # _stream_source swallowed a stream failure — retire the dequeue
            # the same way the raise path above does, or the display/Redis
            # heads would sit one entry ahead of _pending indefinitely.
            await self.queue.finish_failed_dequeue(source, context="prefetch failure")
            return None
        return song

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
            # True while this iteration holds a dequeue it has not yet balanced
            # with task_done() — lets the outer exception handler close the
            # books when a failure lands between the dequeue and the normal
            # song-end task_done() (e.g. the voice client vanished during
            # resolve), instead of drifting the queue's task counter forever.
            dequeue_owed = False
            # Each iteration spans the full song duration (3–5 min typically).
            # This is expected — the span stays open across play_next.wait().
            with _tracer.start_as_current_span(
                "player.loop.iteration",
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    queue_was_cleared = self.queue.consume_cleared_flag()
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
                        dequeue_owed = True  # the prefetch's get_nowait() is now ours
                        # Prefetched items always came through queue_get(), so
                        # they were always real, Redis-mirrored queue entries.
                        # source stays None: redis_pop_for(None) treats the
                        # dequeue as persisted, matching that guarantee.
                        source = None
                        should_pop_queue = True
                    else:
                        source = None
                        try:
                            async with async_timeout.timeout(300):
                                source = await self.queue_get()
                                dequeue_owed = True
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
                                await self.queue.finish_failed_dequeue(
                                    source, context="resolve failure"
                                )
                                dequeue_owed = False
                            raise
                        self.current_song = await self._stream_source(source)
                        should_pop_queue = is_persisted(source)

                    if self.current_song is None:
                        await self.queue.finish_failed_dequeue(
                            source, context="failed-song pop"
                        )
                        dequeue_owed = False
                        try:
                            await self.send_with_np(
                                embed=notice_embed(
                                    "Failed to load the next song, skipping.",
                                    discord.Color.red(),
                                )
                            )
                        except Exception as e:
                            log.warning(
                                f"Failed to send skip-notification in guild {self._guild.id}: {e}"
                            )
                        continue

                    span.set_attribute("song.title", self.current_song.title or "")

                    if not await self.queue.try_commit_dequeue():
                        # The queue was cleared while this song was being resolved
                        # (e.g. during the async yt_stream call). Discard without
                        # playing; task_done() balances the queue.get() above.
                        # cleanup() terminates the FFmpeg subprocess that yt_stream
                        # already spawned — omitting it would leak the process.
                        self.queue.task_done()
                        dequeue_owed = False
                        self.current_song.cleanup()
                        self.current_song = None
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
                    if song.start_paused:
                        # Park the player thread SYNCHRONOUSLY — before any
                        # await — so a song returning paused leaks at most a
                        # frame or two of audio, not a Redis round-trip's
                        # worth. Idempotent with the full pause() below, which
                        # runs after the start transaction so its
                        # pause_start_epoch write isn't clobbered by the
                        # transaction's HDEL.
                        vc.pause()
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
                    if self.store is not None:
                        # play_start_epoch is backdated by the FFmpeg -ss start
                        # offset so the recovery position math
                        # (now - play_start_epoch - pauses) yields the true audio
                        # position, not merely time-since-vc.play(). Without this,
                        # ?t= songs and double-crash recoveries resume
                        # start_offset seconds early.
                        backdated_start = play_start - song.start_offset
                        current = SongQueueEntry.from_song(song)
                        now_playing = NowPlayingData.from_song(song)
                        if should_pop_queue:
                            await self.store.pop_queue_and_start_song(
                                current, backdated_start, now_playing=now_playing
                            )
                        else:
                            await self.store.set_current_song_state(
                                current, backdated_start, now_playing=now_playing
                            )

                    if song.start_paused:
                        # -playnow interrupted this song while it was paused —
                        # it returns parked at the same spot (the player thread
                        # was already paused synchronously at vc.play above).
                        # The full pause() entry point runs here so the Redis
                        # pause epochs and the debounced embed/Activity refresh
                        # all engage; the activity/NP builds below then render
                        # the paused state.
                        await self.pause(vc)
                    if song.is_resume:
                        await self._announce_resume(song)

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

                    # Song has actually ended (naturally or via -skip) — capture
                    # the host, release it (the finished bar stays behind as a
                    # historical record, and the next song's adopt then retires
                    # nothing), then fire one last fire-and-forget edit so the
                    # bar always ends up showing fully completed instead of
                    # frozen at the last periodic tick's position.
                    finished_host = self._np_host_message
                    finished_own = self._np_host_own_embeds
                    self._release_np_host()
                    if self.current_song is not None and finished_host is not None:
                        self._fire_finalize_now_playing(
                            self.current_song, finished_host, finished_own
                        )

                    # Claim-then-await: interject() may have neutralized (and
                    # nulled) the task while this iteration sat in
                    # play_next.wait(). Both sides read-and-null synchronously,
                    # so exactly one consumer sees any given prefetch result;
                    # a task interject() cancelled resolves here to None.
                    prefetch_task = self._prefetch_task
                    self._prefetch_task = None
                    prefetched_song = None
                    if prefetch_task is not None:
                        try:
                            prefetched_song = await prefetch_task
                        except asyncio.CancelledError:
                            prefetched_song = None

                    if self.current_song is not None:
                        # interject() stopped this song with a resume entry
                        # pending — history records it when the tail ends.
                        # Identity match, and the marker clears either way: a
                        # marker left for a song that ended naturally during
                        # interject()'s awaits must not eat this (different)
                        # song's entry.
                        skip_history = self._skip_history_for is self.current_song
                        self._skip_history_for = None
                        if not skip_history:
                            await self.history.add(
                                HistoryEntry.from_song(
                                    self.current_song, played_at=time.time()
                                )
                            )

                    if self.store is not None:
                        await self.store.clear_song_end_state()

                    self.queue.task_done()
                    dequeue_owed = False
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
                    if dequeue_owed:
                        self.queue.task_done()
                    if self._prefetch_task and not self._prefetch_task.done():
                        self._prefetch_task.cancel()
                    self._prefetch_task = None
                    await self._cancel_progress_task()
                    await self._cancel_pause_debounce()
                    # No finalize for a song that errored — the host is simply
                    # released so the next song starts from a clean slate.
                    self._release_np_host()
                    prefetched_song = None
                    self._skip_history_for = None
                    self.current_song = None
                    self.play_message = None
                    if self.store is not None:
                        await self.store.clear_song_end_state()
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
