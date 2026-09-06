import asyncio
import contextlib
import datetime
import time
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
    cast,
)
from collections.abc import AsyncGenerator, Coroutine, Sequence
from zoneinfo import ZoneInfo

import async_timeout
import discord
from discord.ext import commands

import redis.asyncio as aioredis

from opentelemetry import trace
from opentelemetry.context import Context

from src import config
from src.guild_history import GuildHistory
from src.guild_queue import (
    GuildQueue,
    RemoveOutcome,
    ShuffleOutcome,
    is_persisted,
    remove_matcher,
)
from src.guild_state import (
    DEFAULT_TIMEZONE,
    HistoryEntry,
    NowPlayingData,
    SongQueueEntry,
)
from src.redis_client import GuildRedisStore
from src.sources import YTSource
from src.telemetry import get_tracer
from src.util import (
    cancel_task,
    spawn_background,
    fmt_duration,
    notice_embed,
    pluralize,
    record_span_error,
    trace_footer,
    traceparent_context,
    safe_label,
    truncate,
    truncate_embed_title,
    get_logger,
)
from src.youtube import YTDL, NpHostRef, QueueObject, invalidate_stream_cache

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports MusicPlayer).
    from src.musicbot import MusicBot
    from src.main import MusicBotApp

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# A resolved QueueObject, or an unresolved YTSource (e.g. a Spotify playlist
# track awaiting YouTube search).
QueueItem = Union[QueueObject, YTSource]


@dataclass(frozen=True)
class EtaWalk:
    """Accumulator for the queue's ETA walk; `now_pst` is invariant across a walk
    and passed alongside. Frozen: advancing is `replace()` + rebind."""

    cumulative_secs: int
    uncertain: bool

    def advance(self, remaining: Optional[int]) -> EtaWalk:
        """The next walk state after an item whose remaining time is `remaining`,
        or None when its duration is unknown."""
        if remaining is None:
            return replace(self, uncertain=True)
        return replace(self, cumulative_secs=self.cumulative_secs + remaining)


# TODO: every guild's ETAs still render in DEFAULT_TIMEZONE, and in one zone per
# guild rather than per viewer. queue_embed()'s "Est. playing at" and the NP
# "Estimated finish" read GuildConfig.timezone, but nothing writes it (set_timezone
# has no caller). Owed: a write path, then per-viewer rendering (<t:epoch:R>).


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
    """A wall-clock time with the zone it is in, read off the datetime."""
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    # tzname(), not strftime("%Z"): same output, ~50x cheaper, and this runs on
    # every NP tick. None is possible for a naive datetime, hence the `or ""`.
    return f"{hour}:{dt.minute:02d} {ampm} {dt.tzname() or ''}".rstrip()


def _fmt_eta(est_dt: datetime.datetime, uncertain: bool) -> str:
    prefix = "~" if uncertain else ""
    return f"{prefix}**{_fmt_clock_time(est_dt)}**"


def _requester_mention(
    requester: Optional[Union[discord.User, discord.Member]],
) -> str:
    return requester.mention if requester else "Unknown"


# Square emoji blocks: the played portion renders in a different colour from the
# remainder. Width is low because each block glyph is much wider than a dash.
_BAR_WIDTH = 10
_BAR_FILL_DONE = "🟦"
_BAR_FILL_REMAINING = "⬜"
_BAR_HEAD = "🔘"

# Collapses rapid -pause/-resume toggling into one trailing embed edit + Activity
# refresh.
_PAUSE_DEBOUNCE_SECS = 0.5

# ── -playnow interjection ──────────────────────────
# Below this many seconds remaining, an interjected song gets no resume entry.
_MIN_RESUME_REMAINING_SECS = 5
# EOF guard for the resume seek (duration metadata is imprecise), matching the
# crash-recovery position cap in _restore_state().
_RESUME_EOF_MARGIN_SECS = 10

# ── Progress-bar finalize ─────────────────
# Tolerance for "this song reached its end", absorbing drift between yt-dlp's
# duration metadata and the real stream length.
_SONG_COMPLETE_MARGIN_SECS = 5

# ── Playback gate ────────────────────────────────
# How long loop() waits for a voice connection before tearing the player down.
# Matches the idle queue_get() timeout.
_PLAYBACK_GATE_TIMEOUT = 300

# Ceiling on the start transaction, the one Redis write under the queue's bulk
# mutex; the pool sets no socket_timeout. Past it the song plays unpersisted and
# the queue notes its mirror as stale (GuildQueue.note_mirror_write).
_START_WRITE_TIMEOUT = 5.0

# How long a command waits for wait_for_restore() before giving up and saying so.
# Bounded because the pool sets no socket_timeout, so a server that accepts the
# connection then stalls would hang the command outright.
RESTORE_WAIT_SECS = 5.0

# How long a warm -play waits for its restore before reading the ask-time queue
# depth. Short because a timeout here only costs an approximate analytics field.
DEPTH_RESTORE_WAIT_SECS = 1.0


@dataclass(frozen=True)
class InterjectOutcome:
    """What MusicPlayer.interject() did — everything -playnow needs for its
    confirmation wording."""

    interrupted_title: str
    # None → no resume entry (the interrupted song was nearly finished, or had no
    # webpage_url to rebuild from).
    resume_position: Optional[int]
    was_paused: bool  # the OBSERVED state when it was interrupted
    # Whether the resume entry comes back PAUSED — distinct from was_paused, since
    # -playnow restores what it interrupted while -play brings it back playing.
    # Wording keys off this.
    returns_paused: bool = False

    @property
    def resume_position_str(self) -> str:
        return fmt_duration(self.resume_position or 0)


@dataclass(frozen=True)
class StreamFailure:
    """Why a song's stream failed to resolve, captured at the failure point so the
    skip notice can name the cause and the trace carrying the full exception."""

    detail: str  # "<ExceptionType>: <message>"
    trace_id: str  # 32-hex OTel trace id, or "unavailable" when no span is active


def _reached_end(song: YTDL) -> bool:
    """Did this song play through to its end — the only case where the bar finalizes
    to 100%? Answered by position, not cause, so it covers -skip, interjection and a
    mid-song stream death. No known duration → False."""
    if song.duration_secs <= 0:
        return False
    return song.position_secs >= song.duration_secs - _SONG_COMPLETE_MARGIN_SECS


def _remaining_secs(item: QueueObject) -> Optional[int]:
    """A queued item's expected playtime: full duration, minus the resume offset
    for a -playnow resume entry, which plays only its tail."""
    if item.duration is None:
        return None
    if item.is_resume and item.ts:
        return max(0, item.duration - item.ts)
    return item.duration


def _queue_runtime(items: list[QueueItem]) -> tuple[int, bool]:
    """Total remaining playtime of queued items, and whether any duration was
    unknown (the total is then a lower bound, flagged with "~"). Shared by
    queue_embed() and the resume notices so they can't disagree."""
    total_secs = 0
    partial = False
    for item in items:
        remaining = _remaining_secs(item) if isinstance(item, QueueObject) else None
        if remaining is not None:
            total_secs += remaining
        else:
            partial = True
    return total_secs, partial


def _build_progress_bar(
    elapsed_secs: float, duration_secs: int, width: int = _BAR_WIDTH
) -> str:
    if duration_secs <= 0:
        return ""
    # Clamp before formatting: imprecise metadata plus an FFmpeg -ss offset can
    # push the raw position past the reported duration.
    elapsed_secs = max(0.0, min(elapsed_secs, float(duration_secs)))
    ratio = elapsed_secs / duration_secs
    head_pos = min(width - 1, int(ratio * width))
    bar = (
        _BAR_FILL_DONE * head_pos
        + _BAR_HEAD
        + _BAR_FILL_REMAINING * (width - head_pos - 1)
    )
    return f"`{fmt_duration(int(elapsed_secs))}` {bar} `{fmt_duration(duration_secs)}`"


def _fmt_finish_time(duration_secs: int, tz: ZoneInfo) -> str:
    """Clock time `duration_secs` from now. No uncertainty prefix: a playing song's
    remaining duration is known."""
    finish_dt = datetime.datetime.now(tz=tz) + datetime.timedelta(seconds=duration_secs)
    return _fmt_clock_time(finish_dt)


# Discord rejects an empty embed field value (400), which fails the entire
# send/edit. Anything that can legitimately be missing goes through here.
_FIELD_PLACEHOLDER = "—"


# Nothing bounds a yt-dlp uploader string; a field over 1024 characters is a 400,
# and every embed of a message shares one 6000-char budget with the response the
# NP block is prepended to.
_FIELD_VALUE_MAX = 200
# Same budget, for the one queue line the "Up next" embed renders.
_NEXT_UP_TITLE_MAX = 200


def _field_value(value: str) -> str:
    return truncate(value, _FIELD_VALUE_MAX) if value else _FIELD_PLACEHOLDER


def _build_now_playing_base_embed(
    *,
    title: str,
    description: str,
    webpage_url: str,
    uploader: str,
    views: str,
    likes: str,
    abr: str,
    asr: str,
    acodec: str,
    thumbnail: str,
) -> discord.Embed:
    """Shared field layout for the live (YTDL-backed) and recovered
    (NowPlayingData-backed) now-playing embeds. Duration is not a field (the bar or
    the description carries it), nor is the URL (the title links to it). Any value
    can be blank, hence _field_value on every one."""
    title = truncate_embed_title(title)
    embed = (
        discord.Embed(
            title=title,
            url=webpage_url,
            description=description,
            color=discord.Color.green(),
        )
        .add_field(name="Channel", value=_field_value(uploader))
        .add_field(name="Views", value=_field_value(views))
        .add_field(name="Likes", value=_field_value(likes))
        .set_footer(text=f"Avg Bitrate: {abr} | Avg Sampling: {asr} | Acodec: {acodec}")
    )
    if thumbnail:
        embed.set_thumbnail(url=thumbnail)
    return embed


def _link_stream_provenance(span: trace.Span, song: YTDL) -> None:
    """Link this song's trace to the extraction that minted its stream URL (stamped
    on the cache entry by _cache_stream). A URL extracted in band is already in this
    trace and links to nothing. See docs/ARCHITECTURE.md#observability."""
    ctx = traceparent_context(song.data.get("traceparent", ""))
    if ctx is None or ctx.trace_id == span.get_span_context().trace_id:
        return
    span.add_link(ctx, {"link.kind": "stream_extraction"})


class MusicPlayer:
    __slots__ = (
        "bot",
        "_guild",
        "_channel",
        "_last_author",
        "_cog",
        "current_song",
        "_playback_span",
        "play_next",
        "queue",
        "play_message",
        "history",
        "volume",
        "timezone",
        "_player",
        "_prefetch_task",
        "store",
        "_restore_task",
        "_restore_complete",
        "_restore_read_failed",
        "_stopped_deliberately",
        "_playback_gate",
        "_playback_holds",
        "_background_tasks",
        "_progress_task",
        "_heartbeat_task",
        "_np_last_rendered",
        "_np_last_id",
        "_np_host_message",
        "_np_host_own_embeds",
        "_np_host_dedicated",
        "_np_edit_lock",
        "_pause_debounce_task",
        "_skip_history_for",
        "_pending_resume_tail",
        "_ended_song",
        "_last_stream_error",
    )

    def __init__(
        self,
        bot: commands.Bot,
        guild: discord.Guild,
        channel: discord.TextChannel,
        cog: MusicBot,
        redis: Optional[aioredis.Redis] = None,
    ) -> None:
        self.bot = bot
        self._guild = guild
        self._channel = channel
        # Nullable: guild.me is None until the member cache fills and guild.owner
        # can be uncached; discord.py's stub declares Guild.me as Member and hides
        # that. from_context()/set_context() overwrite it before any command path
        # reads it; _require_requester() covers the rest.
        self._last_author: Optional[Union[discord.User, discord.Member]] = (
            guild.me or guild.owner
        )
        self._cog = cog

        # Live-song state. _playback_span is the playing song's trace, read by every
        # NP-block render (docs/ARCHITECTURE.md#debug-footer-seams); _last_stream_error
        # is set by _stream_source() on a failed resolve and read by the loop's skip
        # notice.
        self.current_song: Optional[YTDL] = None
        self._playback_span: Optional[trace.Span] = None
        self._last_stream_error: Optional[StreamFailure] = None
        self.play_next = asyncio.Event()
        self.play_message: Optional[discord.Embed] = None
        self.volume = 1.0
        # Replaced at restore from GuildConfig.
        self.timezone = ZoneInfo(DEFAULT_TIMEZONE)

        self.store = (
            GuildRedisStore(redis, self._guild.id) if redis is not None else None
        )
        self.queue = GuildQueue(guild, self.store)
        # Only the DRAINER is wired in: history writes nudge it, nothing here reads
        # Postgres back. It lives on the app, present exactly when
        # HISTORY_ARCHIVE_ENABLED, so a None drainer wires the None notify
        # GuildHistory's constructor demands be explicit. The cast is a runtime no-op.
        app = cast("MusicBotApp", bot)
        self.history = GuildHistory(
            self.store,
            on_outbox_push=(
                app.history_drainer.notify if app.history_drainer is not None else None
            ),
        )

        # Tasks. _prefetch_task stays parameterized: _neutralize_prefetch reads
        # fields off its result, and a bare Task makes result() Any.
        self._player: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task[Optional[YTDL]]] = None
        self._restore_task: Optional[asyncio.Task] = None
        self._progress_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._pause_debounce_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Restore: the event loop() waits on before its first dequeue, and whether
        # the restore could not READ the store (an empty queue is then "unknown",
        # not "nothing was saved").
        self._restore_complete = asyncio.Event()
        self._restore_read_failed = False
        # Set by whoever calls vc.stop() on the live song, cleared at each vc.play():
        # zero frames alone cannot tell a stream that never opened from one we
        # stopped before its first frame.
        self._stopped_deliberately = False
        # The gate stays shut until a command establishes a voice connection, so a
        # player built by a command that never connects cannot walk the queue and
        # discard it. holds > 0 while an in-flight command owns the opening.
        self._playback_gate = asyncio.Event()
        self._playback_holds = 0

        # NP host state: the message carrying the block, its own cached embeds that
        # follow it, whether it is a dedicated NP message (deleted on retire) or a
        # command response (strip-edited), and the last payload pushed + its host
        # for _push_np_edit's no-op-edit guard (Embed.to_dict() is a TypedDict and
        # list is invariant, so the element type stays Any).
        self._np_last_rendered: Optional[list[Any]] = None
        self._np_last_id: Optional[int] = None
        self._np_host_message: Optional[discord.Message] = None
        self._np_host_own_embeds: list[discord.Embed] = []
        self._np_host_dedicated: bool = False
        self._np_edit_lock = asyncio.Lock()

        # Interjection bookkeeping. _skip_history_for is the song interject()
        # stopped with a resume entry pending, so it is recorded once, when its
        # tail finishes — the song's identity, not a flag, because a stale boolean
        # would eat the next song's entry. _pending_resume_tail is that song's
        # resume entry, awaiting the NP-card ids that only exist at the fragment's
        # iteration end; set and cleared wherever _skip_history_for is.
        self._skip_history_for: Optional[YTDL] = None
        self._pending_resume_tail: Optional[QueueObject] = None
        # The song whose playback ended but whose history row is not written yet;
        # keeps the play claimable across the prefetch await — see
        # claim_current_song_for_history.
        self._ended_song: Optional[YTDL] = None

    @classmethod
    def from_context(
        cls,
        bot: commands.Bot,
        ctx: commands.Context,
        redis: Optional[aioredis.Redis] = None,
    ) -> MusicPlayer:
        assert ctx.guild is not None
        assert isinstance(ctx.channel, discord.TextChannel)
        assert ctx.cog is not None
        # ctx.cog is Optional[Cog] to discord.py; MusicBot is the only cog owning
        # the commands that reach here.
        mp = cls(bot, ctx.guild, ctx.channel, cast("MusicBot", ctx.cog), redis=redis)
        mp._last_author = ctx.author
        return mp

    def start(self) -> None:
        """Start the playback loop and, with Redis, the state restore task. The gate
        opens here when the guild already has a voice client (restore_guild connects
        before calling start()); otherwise -join / -play open it."""
        if self._guild.voice_client is not None:
            self.open_playback_gate()
        if self.store is not None:
            self._restore_task = self.bot.loop.create_task(self._restore_state())
        else:
            # No restore runs — signal now so loop() never waits.
            self._restore_complete.set()
        self._player = self.bot.loop.create_task(self.loop())

    # ── Playback gate ─────────────────────────────────────────────────────────

    def open_playback_gate(self) -> None:
        """Let loop() start consuming the queue. No-op while a hold is
        outstanding — the holder is responsible for the opening."""
        if self._playback_holds == 0:
            self._playback_gate.set()

    @contextlib.asynccontextmanager
    async def defer_playback(self) -> AsyncGenerator[None]:
        """Hold the playback gate shut for the duration of the block: -play's join
        opens the gate the moment the handshake completes, while -play is still
        resolving its input, and the restored head would start in that window. The
        gate opens on the way out even when the block raised."""
        self._playback_holds += 1
        try:
            yield
        finally:
            self._playback_holds -= 1
            if self._playback_holds == 0:
                self.open_playback_gate()

    @property
    def playback_holds(self) -> int:
        """How many commands hold the gate shut. Nonzero means someone else is
        driving this player toward playback and owns the teardown decision."""
        return self._playback_holds

    def can_rejoin_cold(self) -> bool:
        """True in the parked state `-resume`'s rejoin path assumes. Failing it means
        the player outlived its voice client (an eject on_voice_state_update never
        saw), so its legs and gate are untrustworthy: rebuild, don't reuse."""
        return self.current_song is None and not self._playback_gate.is_set()

    @property
    def restore_read_failed(self) -> bool:
        """True when the last restore could not read the store: an empty queue then
        means "unknown", not "nothing was left"."""
        return self._restore_read_failed

    async def wait_for_restore(self, timeout: Optional[float] = None) -> bool:
        """Block until _restore_state() has finished (or failed); False when
        `timeout` elapsed first. Inserting before the restore has read its snapshot
        double-queues: put_front() LPUSHes the mirror, while restore_entries() is
        in-memory only because its entries are already on that list."""
        if timeout is None:
            await self._restore_complete.wait()
            return True
        try:
            async with async_timeout.timeout(timeout):
                await self._restore_complete.wait()
        except asyncio.TimeoutError:
            return False
        return True

    def set_context(self, ctx: commands.Context) -> None:
        assert isinstance(ctx.channel, discord.TextChannel)
        self._channel = ctx.channel
        self._last_author = ctx.author

    def _require_requester(self) -> Union[discord.User, discord.Member]:
        """The fallback requester, for paths that must have one (QueueObject.requester
        is non-optional because persistence reads `requester.id`). Unset only on a
        player whose guild has both bot member AND owner uncached, and never after a
        command has run."""
        if self._last_author is None:
            raise RuntimeError(
                f"No requester available for guild {self._guild.id}: neither the "
                "bot member nor the guild owner is cached"
            )
        return self._last_author

    def _queue_eta_seed(self) -> tuple[datetime.datetime, EtaWalk]:
        """Seed state for walking ETAs across queued songs: (now_pst, walk).
        cumulative_secs starts at the current song's total duration as a proxy for
        its remaining time; uncertain flags an unknown duration."""
        uncertain = False
        cumulative_secs = 0
        if self.current_song is not None:
            secs = getattr(self.current_song, "duration_secs", 0)
            if secs:
                cumulative_secs = secs
            else:
                uncertain = True
        return datetime.datetime.now(tz=self.timezone), EtaWalk(
            cumulative_secs, uncertain
        )

    def _format_queue_line(
        self,
        item: QueueItem,
        index: int,
        now_pst: datetime.datetime,
        walk: EtaWalk,
    ) -> tuple[str, EtaWalk]:
        """Format one -queue page row with its "Est. playing at" ETA. Returns
        (line, updated walk) so the page can chain across consecutive items."""
        est_dt = now_pst + datetime.timedelta(seconds=walk.cumulative_secs)
        est_str = _fmt_eta(est_dt, walk.uncertain)

        if isinstance(item, QueueObject):
            # Capped (ten of these share one 4096-char description) and sanitized:
            # a "]" in a masked link's label would close it early.
            title = safe_label(item.title, _NEXT_UP_TITLE_MAX) or "Unknown"
            requester = _requester_mention(item.requester)
            dur = fmt_duration(item.duration) if item.duration is not None else "?:??"
            channel = truncate(item.uploader or "", _FIELD_VALUE_MAX) or (
                "Unknown channel"
            )
            if item.is_resume and item.ts:
                ts_note = f"  ·  ⏮ resumes at `{fmt_duration(item.ts)}`"
            elif item.ts:
                ts_note = f"  ·  starts at `{item.ts}s`"
            else:
                ts_note = ""
            line = (
                f"`{index}` [**{title}**]({item.webpage_url}) · `{dur}`{ts_note} · Est. playing at {est_str}\n"
                f"{channel} · {requester}"
            )
            walk = walk.advance(_remaining_secs(item))
        else:
            search = safe_label(
                (item.ytsearch or item.url or "?").removeprefix("ytsearch:"),
                _NEXT_UP_TITLE_MAX,
            )
            line = f"`{index}` {search} · *resolving...*"
            walk = walk.advance(None)

        return line, walk

    def _eta_walk_to(self, index: int) -> tuple[datetime.datetime, EtaWalk]:
        """Seed the ETA walk and advance it over every item ahead of 1-based
        `index`. An index past the queue walks all of it — the ETA a song appended
        now would earn."""
        now_pst, walk = self._queue_eta_seed()
        for earlier in self.queue.display_items()[: index - 1]:
            walk = walk.advance(
                _remaining_secs(earlier) if isinstance(earlier, QueueObject) else None
            )
        return now_pst, walk

    def queue_embed(self) -> discord.Embed:
        items = self.queue.display_items()
        total = len(items)

        total_secs, duration_partial = _queue_runtime(items)

        now_pst, walk = self._queue_eta_seed()

        lines = []
        for i, item in enumerate(items[:10], start=1):
            line, walk = self._format_queue_line(item, i, now_pst, walk)
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

    def _resume_left_off_field(self) -> Optional[tuple[str, str]]:
        """(name, value) for the resume notice's "where the last session got to"
        field, or None when nothing recorded it. Call before the front insertion,
        while the queue head is still the restored one. A crash re-queues the
        mid-play song at the head (persisted=False); after a -stop the interrupted
        song is recorded nowhere, and the newest history entry is the last song that
        ran to its END — hence "Last played", not a claim about where playback
        stopped."""
        head = self.queue.peek_next()
        if isinstance(head, QueueObject) and not is_persisted(head) and head.title:
            value = f"**{truncate_embed_title(head.title)}**"
            if head.ts:
                value += f"\n`{fmt_duration(head.ts)}`"
                if head.duration:
                    value += f" / `{fmt_duration(head.duration)}`"
            return "Left off on", value

        last = self.history.latest
        if last is None or not last.title:
            return None
        value = f"**{truncate_embed_title(last.title)}**"
        value += f"\n`{fmt_duration(last.played_secs)}`"
        if last.duration_secs > 0:
            value += f" / `{fmt_duration(last.duration_secs)}`"
        # played_at == 0 means unknown; <t:0:R> would render "56 years ago".
        if last.played_at:
            value += f"\n<t:{int(last.played_at)}:R>"
        return "Last played", value

    def build_resume_notice_embed(
        self, started: QueueObject
    ) -> Optional[discord.Embed]:
        """Heads-up that `-play` on a disconnected bot woke a persisted queue. Build
        before front-inserting, while the queue holds only restored entries; None
        when nothing was restored. `started` is named because this response hosts
        no NP block: the gate is held shut across the enqueue, so current_song is
        still None and the real NP message lands seconds later."""
        items = self.queue.display_items()
        if not items:
            return None

        count = len(items)
        songs = pluralize(count, "song")
        verb = "resume" if count != 1 else "resumes"
        embed = discord.Embed(
            title="❗ Resumed from queue",
            description=(
                f"Playing now: {started.title} - ({started.webpage_url})\n\n"
                f"**{count}** {songs} from the previous session "
                f"{verb} after it."
            ),
            color=discord.Color.orange(),
        )
        # The song being started: the thumbnail sits next to "Playing now".
        if started.thumbnail:
            embed.set_thumbnail(url=started.thumbnail)

        self._add_resume_fields(embed, items)
        return embed

    def build_rejoin_resume_embed(self) -> Optional[discord.Embed]:
        """Heads-up that `-resume` on a disconnected bot rejoined voice and woke a
        persisted queue. Build while the queue head is still the restored one: once
        the gate opens the loop pops it. No song is named: nothing was inserted, so
        the head IS the song the Now Playing card names seconds later. None when the
        restore found nothing."""
        items = self.queue.display_items()
        if not items:
            return None

        count = len(items)
        songs = pluralize(count, "song")
        verb = "resume" if count != 1 else "resumes"
        embed = discord.Embed(
            title="▶️ Resumed from queue",
            description=(
                f"Rejoined voice — **{count}** {songs} from the previous "
                f"session {verb} now."
            ),
            color=discord.Color.green(),
        )
        self._add_resume_fields(embed, items)
        return embed

    def _add_resume_fields(self, embed: discord.Embed, items: list[QueueItem]) -> None:
        """The "what the restore found" fields both resume notices carry: where the
        previous session got to, how much queue came back, and how long it runs."""
        left_off = self._resume_left_off_field()
        if left_off is not None:
            embed.add_field(name=left_off[0], value=left_off[1], inline=True)

        count = len(items)
        songs = pluralize(count, "song")
        embed.add_field(name="Queued", value=f"**{count}** {songs}", inline=True)
        total_secs, partial = _queue_runtime(items)
        if total_secs > 0:
            prefix = "~" if partial else ""
            embed.add_field(
                name="Runtime",
                value=f"{prefix}{_fmt_total_duration(total_secs)}",
                inline=True,
            )

    def claim_current_song_for_history(self) -> Optional[HistoryEntry]:
        """Take the playing song's history entry so a teardown can record it: its
        queue entry was LPOPed at start, clear_connection() drops the parked state
        copy, and the loop is cancelled while parked in play_next.wait().

        SYNCHRONOUS: it reads the song, decides, and takes the _skip_history_for
        marker with no await between, and the loop reads that marker after its
        prefetch await, so exactly one of the two writes. The _ended_song fallback
        covers that await, where current_song is already None. None when there is
        nothing to record."""
        song = self.current_song or self._ended_song
        if song is None:
            return None
        if self._skip_history_for is song:
            # An interjection parked this song's tail, and a teardown leaves the
            # queue intact under its 24h TTL — so -resume plays it and records it.
            return None
        if not song.produced_audio:
            # ffmpeg exited without a frame: nobody heard it.
            return None
        # Captured before cleanup()'s retire_np_host_on_stop() disposes of it.
        host = self._np_host_message
        entry = HistoryEntry.from_song(
            song,
            guild_id=self._guild.id,
            message_id=host.id if host is not None else 0,
            channel_id=host.channel.id if host is not None else 0,
        )
        self._skip_history_for = song
        return entry

    async def stop(self) -> None:
        await self._cog.cleanup(self._guild)

    # ── State restore ─────────────────────────────────────────────────────────

    async def _restore_state(self) -> None:
        """Restore queue, history, and volume from Redis after a restart. Runs as a
        background task; waits for bot ready so guild members are cached. loop()
        waits on _restore_complete before its first queue_get(): the crash-recovered
        head injected here was never on the Redis list, so an LPOP for it would
        delete an unrelated, still-queued song."""
        if self.store is None:
            self._restore_read_failed = True
            self._restore_complete.set()
            return
        try:
            await self.bot.wait_until_ready()
            with _tracer.start_as_current_span(
                "player.state_restore",
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    # One pipelined read: state hash, pending queue, now-playing
                    # snapshot, newest history.
                    snapshot = await self.store.get_playback_snapshot()
                    if snapshot is None:
                        # Read failed — abort rather than proceed with fabricated
                        # defaults. `finally` still sets _restore_complete.
                        self._restore_read_failed = True
                        log.warning(
                            f"State restore aborted for guild {self._guild.id}: "
                            f"Redis unavailable"
                        )
                        return
                    guild_state = snapshot.state

                    # Unconditional: tzinfo() already degrades to the default for
                    # an unset or unusable name.
                    self.timezone = snapshot.config.tzinfo()

                    stored_volume = snapshot.stored_volume
                    # Only when a value was stored: an unconditional assign would
                    # clobber a concurrent -volume with the default.
                    if stored_volume is not None:
                        self.volume = stored_volume
                        # Seed a pre-move value from the 24h-TTL state hash into
                        # config, once. migrate_volume (HSETNX), NOT set_volume: a
                        # -volume that landed since this snapshot was read must not
                        # be overwritten by the older value.
                        if snapshot.config.volume is None and self.store is not None:
                            await self.store.migrate_volume(stored_volume)

                    # Display snapshot, so -now works if a song was playing.
                    if snapshot.now_playing is not None:
                        self.play_message = self._build_now_playing_embed_from_data(
                            snapshot.now_playing
                        )

                    # Re-queue the song that was playing at the crash: current_song_url
                    # is set atomically with the LPOP, so a non-empty value means the
                    # bot died between that transaction and the song's end.
                    if guild_state.has_crashed_song:
                        # The recorded position, straight off the snapshot — no
                        # clock, no IO, so downtime is never credited.
                        position = guild_state.crashed_position_at(time.time())
                        if position is not None:
                            # Cap at duration − 10s so FFmpeg cannot seek past EOF.
                            # Falsy covers both unknown and a livestream's 0: no cap.
                            duration = guild_state.current_song_duration
                            if duration:
                                position = min(position, max(0, duration - 10))
                            log.info(
                                f"Computed recovery position {position}s for "
                                f"'{guild_state.current_song_title}'"
                            )

                        # The crashed current_song_* fields ARE the queue entry the
                        # start transaction LPOPed; rebuild it through the same
                        # rehydration path as everything else.
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
                        # Always clear, re-queued or not: leaving current_song_url
                        # set makes every later restart re-enter this block.
                        await self.store.clear_song_end_state()

                    # After the crashed head, so the interrupted song plays first.
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
                    # Partial restore: what landed stands, but the queue is no longer
                    # known complete, so an empty one is not "nothing was saved".
                    self._restore_read_failed = True
                    record_span_error(span, e)
                    log.error(
                        f"State restore failed for guild {self._guild.id}: {e}",
                        exc_info=True,
                    )
                    return

                await self.store.refresh_ttl()
        finally:
            # Always signal finished-or-failed so loop() never blocks forever.
            self._restore_complete.set()

    async def repark_crashed_head(self) -> bool:
        """Write a crash-recovered queue head back into the state hash it came from;
        True when something was re-parked. _restore_state clears current_song_* as
        soon as it re-queues that song, so this player's memory is its only copy.
        Call AFTER cleanup(): its clear_connection() HDELs these same fields."""
        head = self.queue.peek_next()
        if self.store is None or not isinstance(head, QueueObject):
            return False
        if is_persisted(head):
            # Already on the Redis list: parking it would re-queue a second copy.
            return False
        # Backdated by the resume offset as the loop does at vc.play, and seeded as
        # the recorded position: the hash carries no `ts`.
        await self.store.set_current_song_state(
            SongQueueEntry.from_queue_object(head),
            time.time() - (head.ts or 0),
            start_offset=head.ts or 0,
        )
        return True

    # ── Queue operations ──────────────────────────────────────────────────────

    def enqueue_depth(self) -> int:
        """Songs a new arrival waits behind: everything queued, plus the one playing
        — unless its resume tail is already queued, since that entry is the same
        play. Read once at dispatch, so it is approximate against the insert (±1:
        over while the loop resolves a stream, under when the live song has a
        parked tail from an EARLIER play of the same URL)."""
        depth = self.queue.display_size()
        current = self.current_song
        if current is not None and not self.queue.has_resume_tail(current.webpage_url):
            depth += 1
        return depth

    async def queue_put(
        self,
        obj: Union[QueueItem, Sequence[QueueItem]],
        *,
        prefetch: bool = True,
    ) -> None:
        """Enqueue and, optionally, kick off stream prefetch. prefetch=False for bulk
        playlist enqueues: one batch round-trip, and no per-item tasks — N
        concurrent prefetches mint stream URLs that expire before playback reaches
        them. _prefetch_next_song covers one-ahead prefetch as songs play."""
        items: list[QueueItem] = (
            [obj] if isinstance(obj, (QueueObject, YTSource)) else list(obj)
        )
        items = await self.queue.put(items, batch=not prefetch)
        if prefetch and self.store is not None:
            for item in items:
                if isinstance(item, QueueObject):
                    self._spawn_background(
                        YTDL.prefetch_stream(item, redis=self.store.redis)
                    )

    async def queue_put_front(
        self,
        obj: Union[QueueItem, Sequence[QueueItem]],
        *,
        prefetch: bool = True,
    ) -> None:
        """Insert at the front of the queue, then optionally prefetch. Same contract
        as queue_put(); used when -play runs on a disconnected bot with a persisted
        queue, so the requested song plays now and the persisted entries resume
        behind it."""
        items: list[QueueItem] = (
            [obj] if isinstance(obj, (QueueObject, YTSource)) else list(obj)
        )
        items = await self.queue.put_front(items)
        if prefetch and self.store is not None:
            for item in items:
                if isinstance(item, QueueObject):
                    self._spawn_background(
                        YTDL.prefetch_stream(item, redis=self.store.redis)
                    )

    async def queue_get(self) -> QueueItem:
        return await self.queue.get()

    async def _cancel_prefetch(self) -> None:
        """Cancel any in-flight prefetch task and wait for it. Must run before any
        bulk queue mutation, so the item it dequeued is back at the front
        (requeue_front, in its CancelledError handler) before the drain. A prefetch
        blocked inside run_in_executor cannot be interrupted, so this await can sit
        until the worker exits."""
        await cancel_task(self._prefetch_task)

    async def _flush_played(self, items: Sequence[QueueItem]) -> None:
        """Record every item that already played and is now leaving the queue for
        good: -playnow resume tails, whose remaining tail -clear/-remove discards,
        so nothing else records them. `played_at > 0.0` is the whole test — the loop
        stamps it at vc.play() and the tail inherits it. -stop and the idle
        disconnect are not covered: they leave the queue intact, so a later -resume
        still plays those tails. Accepted crash window: the mirror is destroyed
        inside the bulk mutex and this runs after it."""
        played = [
            item
            for item in items
            if isinstance(item, QueueObject)
            and isinstance(item.played_at, (int, float))
            and not isinstance(item.played_at, bool)
            and item.played_at > 0.0
        ]
        if not played:
            return
        entries = []
        for item in played:
            try:
                entries.append(
                    HistoryEntry.from_queue_object(item, guild_id=self._guild.id)
                )
            except Exception as e:
                # __post_init__ raises rather than coercing, and the mirror is
                # already gone — one malformed wire value must not drop the batch.
                log.warning(
                    f"history flush skipped a malformed entry in guild "
                    f"{self._guild.id}: {type(e).__name__}: {e}"
                )
        if not entries:
            return
        # Concurrently: each write is an independent MULTI plus an outbox push.
        await asyncio.gather(*(self.history.add(entry) for entry in entries))

    async def _dispose_orphaned_cards(self, items: Sequence[QueueItem]) -> None:
        """Retire the frozen NP card of every played tail leaving the queue. A tail
        disposes of its fragment's card when it STARTS, so one destroyed before it
        plays takes the only pointer with it. Fire-and-forget per item: rate-limited
        Discord calls the command must not wait on."""
        for item in items:
            if isinstance(item, QueueObject) and item.is_resume:
                self._spawn_background(self._dispose_previous_np_card(item))

    async def _retire_failed_dequeue(
        self, item: Optional[QueueItem], *, context: str
    ) -> None:
        """Retire a dequeue that will never play, and record it if a listener already
        heard part of it. For a -playnow resume TAIL the flush is the only writer
        left: the interrupted fragment declined to record itself."""
        await self.queue.finish_failed_dequeue(item, context=context)
        if item is not None:
            await self._flush_played([item])

    async def queue_clear(self) -> list[str]:
        await self._cancel_prefetch()  # before the drain — see _cancel_prefetch
        cleared_items = await self.queue.clear()
        # Before the return: a flush failure must surface as a command error
        # rather than a "queue cleared" reply that silently dropped plays.
        await self._flush_played(cleared_items)
        await self._dispose_orphaned_cards(cleared_items)
        return [
            (
                item.title
                if isinstance(item, QueueObject)
                else (item.ytsearch or item.url or "?").removeprefix("ytsearch:")
            )
            for item in cleared_items
        ]

    async def queue_shuffle(self) -> str:
        # Neutralize rather than cancel: cancel_task() no-ops on a COMPLETED
        # prefetch, whose claim would pin its song to the front of the reorder.
        await self._neutralize_prefetch()
        outcome = await self.queue.shuffle()
        if outcome is ShuffleOutcome.TOO_FEW_SONGS:
            return "There must be at least 4 songs to shuffle the queue"
        # The neutralized prefetch took the resolve of the next song with it.
        if self.current_song is not None and self._prefetch_task is None:
            self._prefetch_task = asyncio.create_task(self._prefetch_next_song())
        return "Shuffled!"

    async def queue_remove(self, needle: str) -> RemoveOutcome:
        """Remove every queued item matching `needle` — the resolved yt-dlp URL, or
        what the user originally typed (see remove_matcher)."""
        await self._cancel_prefetch()
        outcome = await self.queue.remove(remove_matcher(needle))
        await self._flush_played(outcome.removed)
        await self._dispose_orphaned_cards(outcome.removed)
        return outcome

    # ── Embed building ────────────────────────────────────────────────────────

    def _decorate_for_debug(
        self, embeds: Sequence[discord.Embed], *, span: Optional[trace.Span] = None
    ) -> None:
        """Add the debug footer to what the player sends or edits itself, which
        MusicContext.send never sees. Freshly built embeds only: the cached
        _np_host_own_embeds keep their send-time footer. NP-block callers pass the
        playback span. See docs/ARCHITECTURE.md#debug-footer-seams."""
        self._cog.debug_settings.decorate(embeds, self._guild, span=span)

    def _build_now_playing_embed(
        self, song: YTDL, *, position_override: Optional[float] = None
    ) -> discord.Embed:
        """position_override renders the bar at a given position instead of
        song.position_secs (used by _finalize_now_playing() for the complete bar)."""
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
                lines.append(bar)
                lines.append("")
        requester_line = f"Requester: [{_requester_mention(song.requester)}]"
        if song.duration_secs > 0:
            # Remaining, not total: a song started mid-stream finishes sooner.
            remaining = max(0, song.duration_secs - int(position))
            requester_line += (
                f"  ·  Estimated finish: {_fmt_finish_time(remaining, self.timezone)}"
            )
        lines.append(requester_line)
        description = "\n".join(lines)
        fields = NowPlayingData.from_song(song)
        return _build_now_playing_base_embed(
            # No markdown: Discord renders embed titles literally.
            title=f"Now playing: {song.title}",
            description=description,
            webpage_url=fields.webpage_url,
            uploader=fields.uploader,
            views=fields.view_count,
            likes=fields.like_count,
            abr=fields.abr,
            asr=fields.asr,
            acodec=fields.acodec,
            thumbnail=fields.thumbnail,
        )

    def build_pause_confirmation_embed(self) -> Optional[discord.Embed]:
        """Slim -pause confirmation: just the pause position, since the response
        hosts the live NP block right below. position_secs is frozen while paused.
        None when no song is live."""
        song = self.current_song
        if song is None:
            return None
        position = int(song.position_secs)
        duration_secs = song.duration_secs
        paused_at = (
            f"{fmt_duration(position)} / {fmt_duration(duration_secs)}"
            if duration_secs > 0
            else fmt_duration(position)
        )
        return discord.Embed(
            title=f"⏸️ Paused: {song.title}",
            description=f"Paused at: `{paused_at}`",
            color=discord.Color.orange(),
        )

    @staticmethod
    def _build_now_playing_embed_from_data(data: NowPlayingData) -> discord.Embed:
        """Reconstruct a now-playing embed from the recovered Redis snapshot.
        Duration goes in the description, in the slot the bar occupies live (no live
        position exists until loop() starts). Rendered as stored."""
        lines = []
        if data.duration:
            lines.append(f"Duration: `{data.duration}`")
            lines.append("")
        lines.append(f"Requester: [{data.requester_mention}]")
        return _build_now_playing_base_embed(
            title=f"Now playing: {data.title}",  # literal, as above
            description="\n".join(lines),
            webpage_url=data.webpage_url,
            uploader=data.uploader,
            views=data.view_count,
            likes=data.like_count,
            abr=data.abr,
            asr=data.asr,
            acodec=data.acodec,
            thumbnail=data.thumbnail,
        )

    def _queue_entry_description(self, item: QueueItem, index: int) -> str:
        """The body both single-entry cards render: one labelled fact per line,
        ending with the ETA position `index` earns."""
        now_pst, walk = self._eta_walk_to(index)
        eta = _fmt_eta(
            now_pst + datetime.timedelta(seconds=walk.cumulative_secs), walk.uncertain
        )
        if not isinstance(item, QueueObject):
            # Unresolved Spotify-playlist entry: only the search term exists yet.
            search = safe_label(
                (item.ytsearch or item.url or "?").removeprefix("ytsearch:"),
                _NEXT_UP_TITLE_MAX,
            )
            return f"{search}\n*resolving...*"
        # Sanitized and capped: a "]" in a masked link's label would close it early.
        title = safe_label(item.title, _NEXT_UP_TITLE_MAX) or "Unknown"
        channel = truncate(item.uploader or "", _FIELD_VALUE_MAX) or "Unknown channel"
        duration = fmt_duration(item.duration) if item.duration is not None else "?:??"
        detail = [f"Channel: {channel}", f"Duration: `{duration}`"]
        if item.is_resume and item.ts:
            detail.append(f"⏮ Resumes at `{fmt_duration(item.ts)}`")
        elif item.ts:
            detail.append(f"Starts at `{item.ts}s`")
        return "\n".join(
            [
                f"Requested by: [{_requester_mention(item.requester)}]",
                f"[**{title}**]({item.webpage_url})",
                "  ·  ".join(detail),
                f"Est. playing at {eta}",
            ]
        )

    def _build_queue_entry_embed(
        self,
        item: QueueItem,
        *,
        index: int,
        title: str,
        warning: Optional[str] = None,
    ) -> discord.Embed:
        """One queue entry as a card, shared by the block's "Up next" and the -play
        confirmation so the two bodies are equal for the same entry."""
        description = self._queue_entry_description(item, index)
        if warning:
            description += f"\n\n{warning}"
        embed = discord.Embed(
            title=title, description=description, color=discord.Color.blue()
        )
        if isinstance(item, QueueObject) and item.thumbnail:
            embed.set_thumbnail(url=item.thumbnail)
        return embed

    def build_queued_song_embed(
        self, item: QueueItem, *, warning: Optional[str] = None
    ) -> discord.Embed:
        """The -play confirmation. `item` is located by identity, so the ETA is the
        one its real position earns; an entry a concurrent -clear removed renders at
        the tail."""
        items = self.queue.display_items()
        index = next(
            (i for i, queued in enumerate(items, 1) if queued is item), len(items) + 1
        )
        return self._build_queue_entry_embed(
            item, index=index, title=f"Queued song — #{index}", warning=warning
        )

    def _build_next_up_embed(self) -> Optional[discord.Embed]:
        item = self.queue.peek_next()
        if item is None:
            return None
        return self._build_queue_entry_embed(item, index=1, title="Up next")

    # ── Now-playing host management ───────────────────────────────────────────
    # The NP block lives in exactly one "host" message at a time — always the newest
    # bot message, so the bar is never buried. Command responses adopt it by
    # prepending at send time (MusicContext.send). The previous host is retired:
    # deleted if it was a dedicated NP message, strip-edited otherwise.

    def np_embed_block(
        self, *, now_playing: Optional[discord.Embed] = None
    ) -> list[discord.Embed]:
        """The [now_playing, next_up?] block, or [] when no song is live. A caller
        that already built this song's embed supplies it as `now_playing`.
        Decorated here, so every attach site gets it from one place; decorate
        replaces rather than appends, so a cached play_message decorated more than
        once is safe."""
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
        self._decorate_for_debug(block, span=self._playback_span)
        return block

    def _adopt_np_host(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        dedicated: bool = False,
    ) -> None:
        """Pointer-first host swap: the pointer update is synchronous, so any tick
        starting after this targets the new host. Retiring the old one is
        fire-and-forget; _retire_np_host's lock orders it after any in-flight tick
        edit against that message."""
        old_msg = self._np_host_message
        old_own = self._np_host_own_embeds
        old_dedicated = self._np_host_dedicated
        if old_msg is not None and message.id < old_msg.id:
            # Overlapping sends complete out of order (channel position is
            # send-START order, adopts run in send-RETURN order). Keep the newer
            # host and shed the older message's block instead.
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
        """Adopt gate for every attach site. The block in `message` was built for
        `song` before the send's await; if the song ended or was replaced in flight,
        adopting would install a stale block as host, so the just-sent message sheds
        it instead. True when adopted."""
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
        """Remove the NP block from a message that is no longer the host. The STRIP
        takes the edit lock so an in-flight tick edit finishes first (a tick landing
        after the strip would resurrect the block). The DELETE does not: nothing can
        resurrect a deleted message, and deletion is a stricter ratelimit bucket."""
        try:
            if dedicated:
                await message.delete()  # pure NP message → remove entirely
            else:
                async with self._np_edit_lock:
                    # response → strip NP block, keep its own embeds
                    await message.edit(embeds=own_embeds)
        except discord.NotFound:
            pass  # user already deleted it — nothing to retire
        except discord.HTTPException as e:
            log.warning(f"NP host retire failed for guild {self._guild.id}: {e}")

    async def _dispose_previous_np_card(self, song: YTDL | QueueObject) -> None:
        """Remove the frozen card the previous fragment of this song left behind.
        Takes either form of the same tail (both carry the np_* fields). With the
        live ref this is _retire_np_host verbatim; after a restart only the ids
        survive and own_embeds cannot be reconstructed, so the by-id delete is
        gated to DEDICATED cards. Never a re-adopt: the live bar belongs at the
        channel bottom. See docs/ARCHITECTURE.md#now-playing-host-invariants."""
        ref = song.np_host_ref
        if ref is not None:
            await self._retire_np_host(ref.message, ref.own_embeds, ref.dedicated)
            return
        mid, cid = song.np_message_id, song.np_channel_id
        # Three wire values reach this DESTRUCTIVE call unchecked. np_dedicated is
        # the authorization, so `is True` and not truthiness (a wire "false" is a
        # truthy string); bool is excluded from the ids because isinstance(True,
        # int) holds and would render "True" into the REST route.
        if song.np_dedicated is not True or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (mid, cid)
        ):
            return
        # Scoped to THIS guild first: a PartialMessageable validates nothing, so a
        # corrupted channel id would delete a message wherever it resolves.
        if self._guild.get_channel_or_thread(cid) is None:
            return
        # Issues the DELETE without the channel being cached, which it may not be
        # on the restart path.
        channel = self.bot.get_partial_messageable(cid, guild_id=self._guild.id)
        try:
            await channel.get_partial_message(mid).delete()
        except discord.NotFound:
            pass  # channel or message gone — nothing to clean up either way
        except discord.Forbidden:
            pass  # permissions changed since the card was posted
        except discord.HTTPException as e:
            log.warning(f"NP card cleanup failed for guild {self._guild.id}: {e}")
        except Exception as e:
            # discord.py surfaces aiohttp.ClientError and asyncio.TimeoutError once
            # its retries are spent; this runs fire-and-forget.
            log.warning(
                f"NP card cleanup errored for guild {self._guild.id}: "
                f"{type(e).__name__}: {e}"
            )

    def _release_np_host(self) -> None:
        """Clear host state without retiring the message. Used at song end: the
        completed bar stays in the channel as a record."""
        self._np_host_message = None
        self._np_host_own_embeds = []
        self._np_host_dedicated = False
        # Retiring can strip-edit the message outside _push_np_edit, so the cache
        # goes with the host or a stale entry suppresses a needed edit.
        self._np_last_rendered = None
        self._np_last_id = None

    async def retire_np_host_on_stop(self) -> None:
        """-stop / alone-disconnect teardown: dispose of the host so no message keeps
        a live-looking bar for a player that no longer exists. cleanup() calls this
        after the progress/loop tasks are cancelled."""
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
        """Player-initiated sends that bypass ctx.send but must still keep the NP
        block at the bottom — the same splice-send-adopt sequence as
        MusicContext.send."""
        own = [embed] if embed is not None else []
        self._decorate_for_debug(own, span=trace.get_current_span())
        song = self.current_song  # the song the block below is built for
        block = self.np_embed_block()  # decorates its own embeds
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
            timestamps: dict[str, int] = {}
            vc = self._guild.voice_client
            is_paused = isinstance(vc, discord.VoiceClient) and vc.is_paused()
            if not is_paused:
                # Backdated by the true audio position, so a -ss/crash-recovered
                # song's tooltip agrees with the bar.
                now_ms = int(time.time() * 1000)
                position_ms = int(song.position_secs * 1000)
                timestamps["start"] = now_ms - position_ms
                if song.duration_secs > 0:
                    timestamps["end"] = timestamps["start"] + song.duration_secs * 1000
            # Paused: no timestamps is the Activity schema's only "frozen"
            # representation.

            # Bot activities reliably render only `name`, so the uploader is packed
            # in as a suffix. `timestamps` still works in the hover tooltip.
            title = song.title or "a song"
            uploader = song.uploader
            raw_name = f"{title} · {uploader}" if uploader else title
            name = raw_name if len(raw_name) <= 128 else raw_name[:127] + "…"

            # `state` renders in both hover and click card for bot activities;
            # details/details_url do not.
            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=name,
                state=song.duration,
                state_url=song.webpage_url,  # discord.py >= 2.6; silent no-op if downgraded
                timestamps=timestamps,
            )
        else:
            # Only reset when no OTHER guild is playing: cleanup() cancels the loop
            # before disconnecting, so this guild's client is still connected here.
            active = any(
                vc.is_playing()
                for vc in self.bot.voice_clients
                if isinstance(vc, discord.VoiceClient) and vc.guild.id != self._guild.id
            )
            if active:
                return
            activity = discord.Game(name="music")
        try:
            await self.bot.change_presence(activity=activity)
        except Exception as e:
            log.warning(f"Failed to update bot activity: {e}", exc_info=True)

    async def pause(self, vc: discord.VoiceClient) -> None:
        """Pause playback and sync the Redis crash-recovery accounting and the
        progress-bar/Activity refresh in one place."""
        vc.pause()
        if self.store is not None:
            # One instant for both writes, so the legacy wall-clock math cannot
            # count the gap between them as playback.
            paused_at = time.time()
            # The exact pause point: the ticking task skips paused songs.
            if self.current_song is not None:
                await self.store.heartbeat(self.current_song.position_secs, paused_at)
            # Still written this release for a rollback; goes one release after.
            await self.store.on_pause(paused_at)
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
        """Debounced refresh of the now-playing embed and the Activity presence:
        nothing rate-limits -pause/-resume, and both targets are rate-limited."""
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
        self,
        qobj: QueueObject,
        vc: discord.VoiceClient,
        *,
        resume_paused: bool = True,
    ) -> Optional[InterjectOutcome]:
        """Play `qobj` immediately; the interrupted song returns afterwards. Capture
        the current song's frame-counted position, front-insert [qobj,
        resume-entry(ts=position)], stop the current song; both entries are
        persisted, so crash recovery mid-interjection works unchanged.

        Interjections STACK: interrupting an interjection parks it in front of the
        tails already waiting, so the queue unwinds LIFO; `ts` is absolute at every
        level. resume_paused decides whether a song interrupted while PAUSED comes
        back paused: True (-playnow) restores it, False (-play) brings it back
        playing.

        None when there is no current song, or it ended during prefetch
        neutralization — the caller falls back to a plain front-enqueue. Residual
        race: a song ending naturally while put_front awaits still gets its resume
        entry, replaying its final seconds."""
        current = self.current_song
        if current is None:
            return None
        span = trace.get_current_span()
        span.set_attribute("discord.guild_id", str(self._guild.id))
        span.set_attribute("song.interjected_title", qobj.title or "")

        # A completed prefetch bypasses the queue and would play INSTEAD of the
        # front-inserted qobj.
        await self._neutralize_prefetch()

        # Re-check after those awaits (cancellation can block up to yt-dlp's socket
        # timeout): if the song ended, bail to the command's fallback rather than
        # build a resume entry for a finished song.
        if self.current_song is not current:
            return None

        was_paused = vc.is_paused()
        position = int(current.position_secs)
        resume: Optional[QueueObject] = None
        if current.webpage_url:
            # On the RAW position: the EOF cap below would mask "almost over".
            near_end = (
                current.duration_secs > 0
                and current.duration_secs - position < _MIN_RESUME_REMAINING_SECS
            )
            if current.duration_secs > 0:
                # EOF guard matching the crash-recovery cap.
                position = min(
                    position,
                    max(0, current.duration_secs - _RESUME_EOF_MARGIN_SECS),
                )
            if not near_end:
                # The tail is the same play, so it keeps the interrupted song's
                # flags and stamps: interjected (read at every stack level by the
                # span attribute below), played_at (files the whole play under its
                # first fragment's start), query_source (the tail writes the ONLY
                # row for this play, and the classification is not recoverable from
                # webpage_url) and user_input (what -remove matches on).
                resume = QueueObject(
                    current.webpage_url,
                    current.title or "",
                    current.requester or self._require_requester(),
                    ts=position,
                    duration=current.duration_secs or None,
                    uploader=current.uploader,
                    thumbnail=current.thumbnail,
                    is_resume=True,
                    interjected=current.interjected,
                    start_paused=was_paused and resume_paused,
                    analytics=current.analytics,
                    played_at=current.played_at,
                    query_source=current.query_source,
                    user_input=current.user_input,
                )

        # The interjection carries depth 0 from its own dispatch — it plays
        # immediately by definition.
        items: list[QueueItem] = [qobj]
        if resume is not None:
            items.append(resume)
            # The song returns, so it is recorded once — when its tail finishes. A
            # song with no resume entry keeps its own entry, matching -skip. One
            # marker suffices at any depth: each interjection stops exactly one
            # song, whose iteration consumes the marker before the next -playnow
            # can resolve. Taken BEFORE the put_front await: everything from the
            # guard above to here is synchronous, so nothing can record the play
            # in between.
            self._skip_history_for = current
            # The tail inherits this fragment's NP card, but which message that is
            # is only settled at the fragment's iteration end.
            self._pending_resume_tail = resume
        await self.queue.put_front(items)

        # Only if the song we measured is still playing: if the loop moved on,
        # stopping would kill the NEXT song.
        if self.current_song is current:
            self.note_deliberate_stop()
            vc.stop()

        # After the insert, so the tail just built is counted.
        span.set_attribute("interject.depth", self.queue.resume_tail_depth())
        # Attribution only: did this cut in front of another -playnow song.
        span.set_attribute("interject.over_interjection", current.interjected)
        span.set_attribute("interject.resume_position", position if resume else -1)
        return InterjectOutcome(
            interrupted_title=current.title or "Unknown",
            resume_position=position if resume is not None else None,
            was_paused=was_paused,
            returns_paused=resume is not None and resume.start_paused,
        )

    async def _neutralize_prefetch(self) -> None:
        """Take the in-flight prefetch off the board so the loop's next dequeue comes
        from the queue head. Claim-then-settle: _prefetch_task is nulled
        synchronously before any await, and the loop's matching read is also a
        synchronous read-and-null, so exactly one consumer sees any given result.
        Running → cancel (its handler requeues the dequeued item). Completed →
        rebuild an equivalent QueueObject, return it to the front, kill its FFmpeg
        subprocess; the rebuild must carry EVERY field. Completed-with-None → the
        prefetch already retired its own dequeue."""
        task = self._prefetch_task
        self._prefetch_task = None
        if task is None:
            return
        if not task.done():
            await cancel_task(task)
            return
        try:
            song = task.result()
        # This reads a *done* task's result, where a cancelled prefetch surfaces as
        # CancelledError and means "no song" — not this coroutine's own
        # cancellation. Listed explicitly because it is not an Exception subclass;
        # the bare tuple form is PEP 758.
        except asyncio.CancelledError, Exception:
            song = None
        if song is None:
            return
        # Dropping a field here restarts a neutralized resume entry from 0:00,
        # loses a ?t= offset, or zeroes the ask this play was queued against.
        rebuilt = QueueObject(
            song.webpage_url or "",
            song.title or "",
            song.requester or self._require_requester(),
            ts=song.start_offset or None,
            duration=song.duration_secs or None,
            uploader=song.uploader,
            thumbnail=song.thumbnail,
            interjected=song.interjected,
            is_resume=song.is_resume,
            start_paused=song.start_paused,
            analytics=song.analytics,
            query_source=song.query_source,
            user_input=song.user_input,
            persisted=song.persisted,
            played_at=song.played_at,
            np_message_id=song.np_message_id,
            np_channel_id=song.np_channel_id,
            np_dedicated=song.np_dedicated,
            np_host_ref=song.np_host_ref,
        )
        self.queue.requeue_front(rebuilt)
        song.cleanup()

    async def _announce_start_offset(self, song: YTDL) -> None:
        """One-line notice for a song starting partway in (a `?t=` link). Sent from
        the loop's start path: at YTDL construction a prefetched song would announce
        itself while the previous one is still playing."""
        try:
            await self._channel.send(
                embed=self._notice(
                    f"Starting song at {song.start_offset} seconds",
                    discord.Color.blue(),
                )
            )
        except Exception as e:
            log.warning(
                f"Failed to send start-offset notice in guild {self._guild.id}: {e}"
            )

    def _notice(self, text: str, color: discord.Color) -> discord.Embed:
        """A notice embed carrying debug mode's footer when the guild has it on."""
        embed = notice_embed(text, color)
        self._decorate_for_debug([embed], span=trace.get_current_span())
        return embed

    async def _announce_resume(self, song: YTDL) -> None:
        """One-line notice when an interrupted song returns, sent from the loop's
        start path like _announce_start_offset. Plain channel send, not
        send_with_np: this song's NP host is not sent yet, so send_with_np would
        adopt the notice only for _send_now_playing to retire it."""
        position = fmt_duration(int(song.position_secs))
        if song.start_paused:
            text = (
                f"⏮ Returned to **{song.title}** at `{position}` — still paused. "
                f"Use `-resume` to continue."
            )
        else:
            text = f"⏮ Resuming **{song.title}** at `{position}`"
        try:
            await self._channel.send(embed=self._notice(text, discord.Color.blue()))
        except Exception as e:
            log.warning(f"Failed to send resume notice in guild {self._guild.id}: {e}")

    # ── Playback pipeline helpers ─────────────────────────────────────────────

    async def _resolve_source(self, source: QueueItem) -> QueueObject:
        if isinstance(source, YTSource):
            return await YTDL.yt_source(
                self._require_requester(),
                source.ytsearch or "",
                redis=self.store.redis if self.store is not None else None,
                query_source=source.query_source,
                analytics=source.analytics,
                user_input=source.user_input,
            )
        return source

    async def _stream_source(
        self, source: QueueObject, *, allow_reextract: bool = True
    ) -> Optional[YTDL]:
        self._last_stream_error = None
        try:
            return await YTDL.yt_stream(
                source,
                self._channel,
                volume=self.volume,
                redis=self.store.redis if self.store is not None else None,
                allow_reextract=allow_reextract,
            )
        except Exception as e:
            ctx = trace.get_current_span().get_span_context()
            trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else "unavailable"
            self._last_stream_error = StreamFailure(
                detail=f"{type(e).__name__}: {e}", trace_id=trace_id
            )
            log.error(
                f"Error processing song: {type(e).__name__}: {e} [trace_id={trace_id}]",
                exc_info=True,
            )
            return None

    def note_deliberate_stop(self) -> None:
        """Record that the live song is about to be stopped by us, not by ffmpeg.
        Call BEFORE vc.stop(); the loop clears it at each vc.play(). A stop we
        initiate reaches `after` as error=None — indistinguishable, on frame count
        alone, from a dead stream."""
        self._stopped_deliberately = True

    async def _drop_unplayable_stream_cache(self, song: YTDL) -> None:
        """Drop the cached stream URL of a song that ended with no frame, no error,
        and no deliberate stop. A BACKSTOP: discord.py reports a failing ffmpeg
        through `after` (the `stream_failed` path _handle_dead_stream owns); what
        reaches here is the window where `_check_process_returncode` sees poll()
        still None. Cache only: being wrong costs one re-extraction, while widening
        `stream_failed` on the same evidence would suppress a real history entry.
        A zero-frame song reaching the loop's iteration end IS still recorded, while
        claim_current_song_for_history refuses one — do not "align" them without
        deciding which record the archive is supposed to hold."""
        if self.store is None or not song.webpage_url:
            return
        dropped = await invalidate_stream_cache(self.store.redis, song.webpage_url)
        # Report the outcome, not the intent: the common case has nothing cached.
        if dropped:
            log.warning(
                "stream ended with no audio and no error, dropped its cached URL: "
                f"{song.webpage_url}"
            )
        else:
            log.info(
                "stream ended with no audio and no error; nothing was cached for "
                f"{song.webpage_url}"
            )

    async def _handle_dead_stream(self, song: YTDL) -> None:
        """Recover from a song whose stream never opened — revoked between
        yt_stream()'s probe and the first read. Drop the cached URL and say so in
        the channel: a failure ffmpeg swallows is invisible to the listener."""
        log.error(
            f"stream produced no audio, treating as failed playback: {song.webpage_url}"
        )
        if self.store is not None and song.webpage_url:
            await invalidate_stream_cache(self.store.redis, song.webpage_url)
        embed = self._notice(
            f"Could not play **{song.title}** — YouTube refused the audio "
            "stream. Queue it again to retry.",
            discord.Color.red(),
        )
        try:
            await self._channel.send(embed=embed)
        except Exception as e:
            log.warning(
                f"Failed to send playback-failure notice in guild {self._guild.id}: {e}"
            )

    async def _send_np_host_message(
        self, *, now_playing: Optional[discord.Embed] = None
    ) -> Optional[discord.Message]:
        """Send a dedicated NP host message (its embeds are only the block) and
        adopt it. None when there is no live song, or the song changed while the
        send was in flight (the stale message is deleted instead of adopted)."""
        song = self.current_song
        block = self.np_embed_block(now_playing=now_playing)
        if not block:
            return None
        message = await self._channel.send(embeds=block)
        if not self._adopt_np_host_if_current(message, [], song, dedicated=True):
            return None
        return message

    @property
    def home_channel(self) -> discord.TextChannel:
        """The text channel this player posts in — where the Now Playing host
        lives. Decides whether a reply may carry the live block or has to be a
        static copy; cog_before_invoke re-points it."""
        return self._channel

    def now_playing_snapshot(self, song: YTDL) -> discord.Embed:
        """A static Now Playing card for `song`, its bar frozen where the song is —
        what `-now` answers with outside home_channel, since a copy sent elsewhere
        cannot be updated."""
        return self._build_now_playing_embed(song)

    async def repin_now_playing(self) -> bool:
        """-now: re-host the NP block at the bottom as a fresh dedicated message.
        The updater follows the host pointer and picks up the new message next
        tick. False when no song is live."""
        return await self._send_np_host_message() is not None

    async def rehost_np_after_resume(self) -> None:
        """-resume: when a command response hosts the block (typically the -pause
        confirmation), re-host onto a fresh dedicated message so "⏸️ Paused at…"
        is not re-rendered beneath a live bar every tick. A dedicated host is left
        alone."""
        if self._np_host_message is None or self._np_host_dedicated:
            return
        await self._send_np_host_message()

    async def _send_now_playing(self, song: YTDL) -> None:
        # Release before the send, so a partial send never leaves the host pointing
        # at the previous song's message, which a later mark_paused()/mark_resumed()
        # would overwrite.
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
        span: Optional[trace.Span] = None,
    ) -> bool:
        """Rebuild the host's embeds — a fresh NP block, then its cached own embeds
        — and push one edit. Shared by the tick, the debounced pause/resume refresh
        and the song-end finalize. False when the message no longer exists.
        `span` follows `song` and is passed by the caller: the finalize awaits
        _np_edit_lock, and by then _playback_span may name the next song."""
        try:
            embed = self._build_now_playing_embed(
                song, position_override=position_override
            )
            next_up = self._build_next_up_embed()
            block = [embed] + ([next_up] if next_up else [])
            self._decorate_for_debug(block, span=span)
            embeds = block + own_embeds
            # Discord's per-message cap: drop the own-embeds tail, never the block.
            embeds = embeds[:10]
            # Skip the PATCH when the payload is identical to the last one pushed
            # to this host. See docs/ARCHITECTURE.md#now-playing-host-model
            rendered = [e.to_dict() for e in embeds]
            if rendered == self._np_last_rendered and message.id == self._np_last_id:
                return True
            await message.edit(embeds=embeds)
            # Recorded only after a successful edit: caching a payload we failed
            # to push would suppress the retry that fixes it.
            self._np_last_rendered = rendered
            self._np_last_id = message.id
            return True
        except discord.NotFound:
            return False
        except discord.HTTPException as e:
            log.warning(f"Now-playing edit failed for guild {self._guild.id}: {e}")
            return True

    async def _edit_now_playing_once(self) -> None:
        """One embed edit outside the periodic tick, for the debounced pause/resume
        refresh. Holds the edit lock and re-reads the host inside it: an edit
        landing after a retire's strip would resurrect the block."""
        song = self.current_song
        if song is None:
            return
        async with self._np_edit_lock:
            host = self._np_host_message
            if host is None:
                return
            if not await self._push_np_edit(
                song, host, self._np_host_own_embeds, span=self._playback_span
            ):
                # Adopt is lock-free, so a command response may have swapped in a
                # new host during this PATCH — releasing would orphan its block.
                if self._np_host_message is host:
                    self._release_np_host()

    async def _finalize_now_playing(
        self,
        song: YTDL,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        completed: bool = True,
        span: Optional[trace.Span] = None,
    ) -> None:
        """One last embed edit once a song has stopped, so the bar lands on its true
        final state rather than wherever the last tick fell. completed=False renders
        where it actually stopped: a 100% bar for a skipped song would be a false
        record. Every argument is captured by the CALLER: all may point at the next
        song by the time this fire-and-forget task runs. The lock orders this write
        after a debounce-spawned edit still in flight."""
        if song.duration_secs <= 0:
            return  # no bar was ever shown for this song — nothing to finalize
        async with self._np_edit_lock:
            await self._push_np_edit(
                song,
                message,
                own_embeds,
                # None → the live position_secs, frozen at the stop point.
                position_override=song.duration_secs if completed else None,
                span=span,
            )

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Fire-and-forget task tracked in _background_tasks."""
        return spawn_background(coro, self._background_tasks)

    def _fire_finalize_now_playing(
        self,
        song: YTDL,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        completed: bool = True,
    ) -> None:
        # _playback_span read here, synchronously: the task this spawns can wake
        # after the next song has claimed the slot.
        self._spawn_background(
            self._finalize_now_playing(
                song,
                message,
                own_embeds,
                completed=completed,
                span=self._playback_span,
            )
        )

    async def _progress_updater(self, song: YTDL) -> None:
        interval = config.NOW_PLAYING_UPDATE_INTERVAL_SECS
        while True:
            await asyncio.sleep(interval)
            vc = self._guild.voice_client
            if not isinstance(vc, discord.VoiceClient) or vc.source is not song:
                return  # song changed under us; loop() owns cancellation
            if vc.is_paused():
                continue  # frozen — mark_resumed() fires a debounced edit
            async with self._np_edit_lock:
                # Re-read inside the lock: a host swap during the sleep must not
                # leave the edit targeting the about-to-be-stripped message.
                host = self._np_host_message
                if host is None:
                    continue  # dormant: no visible NP until re-hosted
                if not await self._push_np_edit(
                    song, host, self._np_host_own_embeds, span=self._playback_span
                ):
                    # Host deleted by a user — go dormant; the next command
                    # response (or -now) re-hosts. Adopt is lock-free, so only
                    # release OUR host.
                    if self._np_host_message is host:
                        self._release_np_host()

    async def _heartbeat_updater(self, song: YTDL) -> None:
        """Record the playback position to Redis on a fixed cadence. Separate from
        _progress_updater, which is display-gated: a song with no visible bar must
        still be recoverable."""
        while True:
            await asyncio.sleep(config.HEARTBEAT_INTERVAL_SECS)
            vc = self._guild.voice_client
            if not isinstance(vc, discord.VoiceClient) or vc.source is not song:
                return  # song changed under us; loop() owns cancellation
            if vc.is_paused():
                # Frames are frozen and pause() already recorded the exact point.
                continue
            if self.store is not None:
                try:
                    await self.store.heartbeat(song.position_secs, time.time())
                except Exception as e:
                    # @_guild_op already swallows Redis failures, so this is a
                    # defect that would recur every tick; cancel_task never awaits
                    # a task that ended on its own, so log it here.
                    log.error(
                        f"playback heartbeat stopped: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    return

    async def _cancel_heartbeat_task(self) -> None:
        await cancel_task(self._heartbeat_task)
        self._heartbeat_task = None

    async def _cancel_progress_task(self) -> None:
        """Await before the next song's _send_now_playing(), so no concurrent edit
        for the old song races the new message send."""
        await cancel_task(self._progress_task)
        self._progress_task = None

    async def _cancel_pause_debounce(self) -> None:
        await cancel_task(self._pause_debounce_task)
        self._pause_debounce_task = None

    @_tracer.start_as_current_span("player.prefetch")
    async def _prefetch_next_song(self) -> Optional[YTDL]:
        """Pre-resolve and stream the next queued song while the current one plays.
        Accounts for its own dequeue on every non-success path: cancellation gives
        the claim back with the item, failure settles it on both legs. On success
        the claim stays open and loop()'s commit settles it."""
        if self.queue.empty():
            return None
        try:
            source = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        trace.get_current_span().set_attribute("discord.guild_id", str(self._guild.id))
        try:
            source = await self._resolve_source(source)
            # No re-extraction here: _cancel_prefetch() awaits this task, and an
            # executor job cannot be interrupted. The play-time resolve decides.
            song = await self._stream_source(source, allow_reextract=False)
        except asyncio.CancelledError:
            self.queue.requeue_front(source)
            raise
        except Exception as e:
            record_span_error(trace.get_current_span(), e)
            log.error(f"Prefetch error: {type(e).__name__}: {e}", exc_info=True)
            await self._retire_failed_dequeue(source, context="prefetch failure")
            return None
        if song is None:
            # _stream_source swallowed a failure — retire the dequeue as the raise
            # path does, or the display/Redis heads sit one entry ahead forever.
            await self._retire_failed_dequeue(source, context="prefetch failure")
            return None
        return song

    # ── Main playback loop ────────────────────────────────────────────────────

    async def loop(self) -> None:
        await self.bot.wait_until_ready()
        # Before the first dequeue: an LPOP for the crash-recovered head, which was
        # never on the Redis list, would delete a still-queued song.
        await self._restore_complete.wait()
        # Wait for a voice connection. The timeout is not optional: a player parked
        # here is not in queue_get(), so the idle disconnect below cannot fire, and
        # a player that never connects would leak its mps entry and task forever.
        while True:
            try:
                async with async_timeout.timeout(_PLAYBACK_GATE_TIMEOUT):
                    await self._playback_gate.wait()
                break
            except asyncio.TimeoutError:
                if self._playback_holds or self._playback_gate.is_set():
                    # A hold means a command is mid-join and owns the teardown
                    # decision. The gate check is needed too: this handler runs a
                    # tick after the timer fires, so a release can land in between
                    # and leave holds 0 with the gate already open.
                    continue
                log.info(
                    f"Playback gate timed out for guild {self._guild.id} "
                    f"(never connected to voice), tearing down player"
                )
                asyncio.create_task(self.stop())
                return
        prefetched_song: Optional[YTDL] = None

        while not self.bot.is_closed():
            self.play_next.clear()
            # True while this iteration holds a claim the commit has not settled, so
            # the outer handler can settle it. Cleared at the commit, not at song
            # end — the release it guards deletes an item.
            claim_outstanding = False
            # Whether that claim has an entry on the Redis list. Carried rather than
            # re-derived from `source`, which the prefetched branch leaves None.
            claim_persisted = True
            # Each iteration spans a full song and roots its own trace, so one song
            # is one trace. See docs/ARCHITECTURE.md#observability.
            with _tracer.start_as_current_span(
                "player.loop.iteration",
                context=Context(),
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    prefetch_used = prefetched_song is not None
                    span.set_attribute("prefetch.used", prefetch_used)
                    # Captured where each path takes its item and handed to the
                    # commit below: a clear() in between voids this dequeue even if
                    # a put() has since refilled the display.
                    commit_generation = self.queue.generation
                    if prefetched_song is not None:
                        self.current_song = prefetched_song
                        prefetched_song = None
                        claim_outstanding = True  # the prefetch's get_nowait() is ours
                        # Read off the song: a prefetch CAN claim a persisted=False
                        # item (a cold-start -play front-inserts AHEAD of the
                        # crash-recovered head). Popping for one that was never on
                        # the list deletes the next real entry.
                        claim_persisted = self.current_song.persisted
                        # `source` stays None because a YTDL is not a QueueItem.
                        should_pop_queue = claim_persisted
                        source = None
                    else:
                        source = None
                        try:
                            async with async_timeout.timeout(300):
                                source = await self.queue_get()
                                claim_outstanding = True
                                # Safe before the resolve: a YTSource is persisted
                                # and yt_source() builds a QueueObject that
                                # defaults the same way.
                                claim_persisted = is_persisted(source)
                                # Re-read: a clear() during the blocking get
                                # belongs to the queue this item came from.
                                commit_generation = self.queue.generation
                                source = await self._resolve_source(source)
                        except asyncio.TimeoutError:
                            log.warning("Queue timed out, disconnecting")
                            asyncio.create_task(self.stop())
                            return
                        except Exception:
                            # _resolve_source() raised after queue_get() already
                            # dequeued `source` — balance that dequeue, then
                            # re-raise into the outer handler.
                            if source is not None:
                                await self._retire_failed_dequeue(
                                    source, context="resolve failure"
                                )
                                claim_outstanding = False
                            raise
                        self.current_song = await self._stream_source(source)
                        should_pop_queue = is_persisted(source)

                    if self.current_song is None:
                        await self._retire_failed_dequeue(
                            source, context="failed-song pop"
                        )
                        claim_outstanding = False
                        failure = self._last_stream_error
                        if failure is not None:
                            message = (
                                "Failed to load the next song, skipping.\n"
                                f"**Reason:** `{failure.detail}`\n"
                                f"**Trace ID:** `{failure.trace_id}`"
                            )
                        else:
                            message = "Failed to load the next song, skipping."
                        try:
                            await self.send_with_np(
                                embed=notice_embed(message, discord.Color.red())
                            )
                        except Exception as e:
                            log.warning(
                                f"Failed to send skip-notification in guild {self._guild.id}: {e}"
                            )
                        continue

                    span.set_attribute("song.title", self.current_song.title or "")
                    _link_stream_provenance(span, self.current_song)
                    # Advances with the song, not the iteration: a failed resolve
                    # renders no card, so the previous song's tail keeps naming its
                    # own trace.
                    self._playback_span = span

                    # The commit and the start transaction under ONE mutex hold —
                    # see GuildQueue.commit_dequeue. The store dispatch is the only
                    # await in here and _START_WRITE_TIMEOUT bounds it; vc.play()
                    # and vc.pause() are synchronous and never precede the commit.
                    vc: discord.VoiceClient
                    song: YTDL
                    discarded: Optional[YTDL] = None
                    landed = False
                    async with self.queue.commit_dequeue(commit_generation) as ok:
                        if not ok:
                            # Cleared while this song resolved: the clear() reset
                            # the cursor, so the claim is already settled. The
                            # FFmpeg reap waits until the hold is released —
                            # cleanup() blocks on the subprocess.
                            claim_outstanding = False
                            discarded, self.current_song = self.current_song, None
                        else:
                            try:
                                # The gate opens only once channel.connect() has
                                # completed the handshake, so an assert suffices.
                                voice = self._guild.voice_client
                                assert isinstance(voice, discord.VoiceClient)
                                assert self.current_song is not None
                                vc = voice
                                # Local binding: pyright's narrowing doesn't survive
                                # the awaits below, and every write in this
                                # iteration stays on the same song.
                                song = self.current_song

                                # The commit settled the claim. Cleared here, not at
                                # song end: left standing across the song it would
                                # release whatever sits at the head by then — the
                                # next song once the prefetch below claims it.
                                claim_outstanding = False

                                # Written from the player thread, read after
                                # play_next.wait(); call_soon_threadsafe orders the
                                # write before the wait returns.
                                play_error: list[Optional[Exception]] = [None]

                                def _after_play(
                                    error: Optional[Exception],
                                    _title: str = song.title or "",
                                ) -> None:
                                    # discord.py hands ffmpeg's failure here and
                                    # nowhere else. A deliberate vc.stop() arrives
                                    # as error=None.
                                    if error is not None:
                                        play_error[0] = error
                                        log.error(
                                            f"playback error for {_title}: {error}"
                                        )
                                    self.bot.loop.call_soon_threadsafe(
                                        self.play_next.set
                                    )

                                self._stopped_deliberately = False
                                vc.play(song, after=_after_play)
                                if song.start_paused:
                                    # Park the player thread SYNCHRONOUSLY, before
                                    # any await, so a song returning paused leaks
                                    # a frame or two. Idempotent with the full
                                    # pause() below, which runs after the start
                                    # transaction so its pause_start_epoch survives
                                    # that transaction's HDEL.
                                    vc.pause()
                                play_start = (
                                    time.time()
                                )  # capture immediately before any awaits
                                # Stamped once and inherited by every later fragment.
                                # Before the state write, or the parked entry
                                # persists 0.0 and a crash recovers no start.
                                song.played_at = song.played_at or play_start

                                # One MULTI/EXEC: every state field, the display
                                # snapshot, and the list leg. Clean and persisted →
                                # LPOP. Stale mirror → the list is REPLACED from
                                # memory, since an LPOP would retire the wrong
                                # entry. A crash-recovered head was never on the
                                # list, so only state is written.
                                if self.store is not None:
                                    # The -ss offset twice: backdated into the epoch
                                    # the legacy fallback extrapolates from, and
                                    # passed as the seed the heartbeat has not
                                    # written yet.
                                    backdated_start = play_start - song.start_offset
                                    current = SongQueueEntry.from_song(song)
                                    now_playing = NowPlayingData.from_song(song)
                                    try:
                                        async with asyncio.timeout(
                                            _START_WRITE_TIMEOUT
                                        ):
                                            if self.queue.mirror_dirty:
                                                landed = await self.store.rebuild_queue_and_start_song(
                                                    current,
                                                    self.queue.mirror_entries(),
                                                    backdated_start,
                                                    now_playing=now_playing,
                                                    start_offset=song.start_offset,
                                                )
                                            elif should_pop_queue:
                                                landed = await self.store.pop_queue_and_start_song(
                                                    current,
                                                    backdated_start,
                                                    now_playing=now_playing,
                                                    start_offset=song.start_offset,
                                                )
                                            else:
                                                landed = await self.store.set_current_song_state(
                                                    current,
                                                    backdated_start,
                                                    now_playing=now_playing,
                                                    start_offset=song.start_offset,
                                                )
                                    except TimeoutError:
                                        log.error(
                                            f"start transaction timed out after "
                                            f"{_START_WRITE_TIMEOUT}s in guild "
                                            f"{self._guild.id}; playing without persisting"
                                        )
                                    # In this block because the store is the
                                    # ticker's only writer. After the seed above,
                                    # so the first tick cannot race it; create_task
                                    # is synchronous, so it does not extend the hold.
                                    self._heartbeat_task = asyncio.create_task(
                                        self._heartbeat_updater(song)
                                    )
                            finally:
                                # A raise between the settle and the write —
                                # vc.play() refusing a dropped voice client — is
                                # recorded like a write that did not land.
                                if self.store is not None:
                                    self.queue.note_mirror_write(
                                        landed=landed, retired=should_pop_queue
                                    )

                    if discarded is not None:
                        discarded.cleanup()
                        continue

                    if self.store is not None and not landed:
                        # Memory dropped the entry; the list still holds it. The
                        # next start replaces the list instead of LPOPing it; a
                        # crash before then replays this song from the stale entry.
                        log.error(
                            f"start transaction did not land in guild "
                            f"{self._guild.id}; the queue mirror is stale until the "
                            "next song start rebuilds it"
                        )

                    if song.start_paused:
                        # The player thread was already paused at vc.play; the full
                        # pause() engages the Redis pause epochs and the debounced
                        # embed/Activity refresh.
                        await self.pause(vc)
                    if song.is_resume:
                        await self._announce_resume(song)
                    elif song.start_offset > 0:
                        await self._announce_start_offset(song)

                    await self.update_activity(song)
                    await self._send_now_playing(song)
                    # Strictly after the new card is up, so the bar is never absent
                    # from the channel. _send_now_playing releases the host before
                    # sending and swallows failures, so a 403 leaves no card and
                    # disposing would delete the only bar in the channel.
                    if song.is_resume and self._np_host_message is not None:
                        self._spawn_background(self._dispose_previous_np_card(song))

                    self._prefetch_task = asyncio.create_task(
                        self._prefetch_next_song()
                    )

                    await self.play_next.wait()

                    # Zero frames AND an ffmpeg error means the stream never opened.
                    # Zero frames alone also describes a song parked paused by
                    # -playnow or stopped the instant it started; an error alone
                    # also describes a mid-song death that earns its history entry.
                    # A THIRD case — zero frames, no error, no deliberate stop — is
                    # handled below by _drop_unplayable_stream_cache.
                    stream_failed = (
                        not song.produced_audio and play_error[0] is not None
                    )
                    span.set_attribute("song.stream_failed", stream_failed)

                    # Must fully retire before the next iteration's
                    # _send_now_playing(), or an in-flight edit for this song could
                    # race the new message being sent.
                    await self._cancel_progress_task()
                    await self._cancel_pause_debounce()

                    # Capture the host, release it (the finished bar stays behind as
                    # a record), then fire one last edit so the bar shows its true
                    # final state.
                    finished_host = self._np_host_message
                    finished_own = self._np_host_own_embeds
                    finished_dedicated = self._np_host_dedicated
                    self._release_np_host()
                    # The id that lands in play_history.message_id, on the span
                    # because 0 is ambiguous in the stored row.
                    span.set_attribute(
                        "song.np_host_id",
                        str(finished_host.id) if finished_host is not None else "",
                    )
                    if finished_host is not None:
                        if stream_failed:
                            # This song delivered nothing, so dispose of the block
                            # rather than finalize it to 100% above the failure
                            # notice.
                            self._spawn_background(
                                self._retire_np_host(
                                    finished_host, finished_own, finished_dedicated
                                )
                            )
                        elif self.current_song is not None:
                            # A skipped, interjected or dead song finalizes at its
                            # true position, never 100%. The edit fires either way:
                            # the tick would leave the bar frozen a tick before the
                            # interruption.
                            self._fire_finalize_now_playing(
                                self.current_song,
                                finished_host,
                                finished_own,
                                completed=_reached_end(self.current_song),
                            )

                    # Stop advertising this song as current before the prefetch
                    # await below, the first point another coroutine can interleave:
                    # MusicContext.send's attach gate is `current_song is not None`,
                    # and left set, a command response would prepend a block for an
                    # ended song and adopt ITSELF as host, which the next
                    # _send_now_playing() releases without retiring. `song` is this
                    # iteration's copy, and what the history entry is built from.
                    self.current_song = None
                    self.play_message = None  # -now must not serve a finished song
                    # The play is no longer current but its row is not written yet,
                    # and a teardown can land during the prefetch await.
                    self._ended_song = song
                    # Below current_song = None: this task always exists, so awaiting
                    # it yields, and the block above must stay synchronous.
                    await self._cancel_heartbeat_task()

                    # Claim-then-await: interject() may have neutralized (and nulled)
                    # the task while this iteration sat in play_next.wait(). Both
                    # sides read-and-null synchronously, so exactly one consumer
                    # sees any given result.
                    prefetch_task = self._prefetch_task
                    self._prefetch_task = None
                    prefetched_song = None
                    if prefetch_task is not None:
                        try:
                            prefetched_song = await prefetch_task
                        except asyncio.CancelledError:
                            prefetched_song = None

                    # interject() stopped this song with a resume entry pending —
                    # history records it when the tail ends. Identity match, and
                    # the marker clears either way: one left for a song that ended
                    # during interject()'s awaits must not eat this song's entry.
                    skip_history = self._skip_history_for is song
                    self._skip_history_for = None
                    # Same identity, same clear-either-way rule: hand the tail the
                    # card this fragment is leaving frozen. Late-bound because THIS
                    # is where the host settles — an id taken at interjection time
                    # can name a message the confirmation's own adopt already
                    # retired. Not mirrored to Redis here, but any later
                    # rebuild_queue re-serializes this object, so the live ids do
                    # reach the wire — which is why _dispose_previous_np_card
                    # guards them as hostile input.
                    pending_tail = self._pending_resume_tail
                    self._pending_resume_tail = None
                    if skip_history and pending_tail is not None:
                        pending_tail.np_host_ref = (
                            NpHostRef(finished_host, finished_own, finished_dedicated)
                            if finished_host is not None
                            else None
                        )
                        pending_tail.np_message_id = (
                            finished_host.id if finished_host is not None else 0
                        )
                        pending_tail.np_channel_id = (
                            finished_host.channel.id if finished_host is not None else 0
                        )
                        pending_tail.np_dedicated = finished_dedicated
                    # stream_failed means THIS fragment never opened a stream —
                    # "nobody heard it" for a fresh song, but not for a resume tail,
                    # whose offset is audio heard under the fragment that parked it
                    # and declined to record.
                    heard_before = song.is_resume and song.start_offset > 0
                    if not skip_history and (not stream_failed or heard_before):
                        await self.history.add(
                            HistoryEntry.from_song(
                                song,
                                guild_id=self._guild.id,
                                # The host captured at song end, not
                                # _np_host_message, which was nulled above. Both
                                # ids come off that one message — never the home
                                # channel, which commands reassign. 0 = nothing
                                # hosted it.
                                message_id=(
                                    finished_host.id if finished_host is not None else 0
                                ),
                                channel_id=(
                                    finished_host.channel.id
                                    if finished_host is not None
                                    else 0
                                ),
                            )
                        )

                    # Written, or declined — either way a teardown from here has
                    # nothing left to claim.
                    self._ended_song = None

                    if self.store is not None:
                        await self.store.clear_song_end_state()

                    await self.update_activity(None)

                    # Last: current_song is already cleared, so the notice goes out
                    # alone rather than re-hosting an NP block for a song that never
                    # played.
                    if stream_failed:
                        await self._handle_dead_stream(song)
                    elif (
                        not song.produced_audio
                        and not self._stopped_deliberately
                        and not song.start_paused
                    ):
                        # Zero frames, no error, nobody stopped it. start_paused is
                        # the one other ending that looks identical and says
                        # nothing about the URL.
                        await self._drop_unplayable_stream_cache(song)
                except asyncio.CancelledError:
                    span.set_attribute("loop.cancelled", True)
                    await self._cancel_progress_task()
                    await self._cancel_heartbeat_task()
                    await self._cancel_pause_debounce()
                    await self.update_activity(None)
                    raise
                except Exception as e:
                    record_span_error(span, e)
                    log.error(
                        f"Unhandled error in playback loop: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    # A claim reaches here only from the window between the dequeue
                    # and the commit. finish_failed_dequeue, not release: release
                    # drops the item from memory alone, leaving its mirror entry for
                    # the next LPOP to retire in its place. persisted= travels
                    # because `source` is None for a prefetched claim.
                    if claim_outstanding:
                        await self.queue.finish_failed_dequeue(
                            source,
                            context="unhandled loop error",
                            persisted=claim_persisted,
                        )
                    # Awaited: the prefetch returns its item through requeue_front,
                    # and claims settle by POSITION, so letting that land after this
                    # handler's own settle swaps the two songs.
                    await cancel_task(self._prefetch_task)
                    self._prefetch_task = None
                    await self._cancel_progress_task()
                    await self._cancel_heartbeat_task()
                    await self._cancel_pause_debounce()
                    # No finalize for a song that errored — just release the host.
                    self._release_np_host()
                    prefetched_song = None
                    self._skip_history_for = None
                    # A tail left holding this slot would receive a LATER fragment's
                    # card ids and delete the wrong message.
                    self._pending_resume_tail = None
                    self._ended_song = None
                    self.current_song = None
                    self.play_message = None
                    if self.store is not None:
                        await self.store.clear_song_end_state()
                    try:
                        # Inside the try: a raise here escapes both handlers and
                        # kills the playback task. Hand-built rather than send_embed
                        # so the debug footer lands before the send.
                        error_embed = discord.Embed(
                            title="Playback error — skipping song",
                            description=f"**{type(e).__name__}:** {e}",
                            color=discord.Color.red(),
                        )
                        error_embed.set_footer(text=trace_footer(span))
                        self._decorate_for_debug([error_embed], span=span)
                        await self._channel.send(embed=error_embed)
                    except Exception as e:
                        log.warning(
                            f"Failed to send playback-error embed in guild {self._guild.id}: {e}"
                        )
