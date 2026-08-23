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

from src import config
from src.debug import decorate_embeds, strip_debug_footers
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
from src.redis_client import GuildRedisStore, cache_get
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
    safe_label,
    truncate,
    truncate_embed_title,
    get_logger,
)
from src.youtube import YTDL, NpHostRef, QueueObject, invalidate_stream_cache

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports MusicPlayer); the
    # cog is only named in annotations here. Same guard main.py uses for MusicPlayer.
    from src.musicbot import MusicBot
    from src.main import MusicBotApp

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# What a song queue holds: a resolved QueueObject, or an unresolved YTSource
# (e.g. a Spotify playlist track awaiting YouTube search).
QueueItem = Union[QueueObject, YTSource]


@dataclass(frozen=True)
class EtaWalk:
    """Accumulator for the queue's ETA-walking fold; `now_pst` is invariant across a
    walk and passed alongside instead of bundled in. Frozen, so advancing means
    `replace()` + rebind — a mutable seed, snapshotted or reused, would read
    silently wrong ETAs."""

    cumulative_secs: int
    uncertain: bool

    def advance(self, remaining: Optional[int]) -> "EtaWalk":
        """The next walk state after a queue item whose remaining time is
        `remaining`, or None when its duration is unknown."""
        if remaining is None:
            return replace(self, uncertain=True)
        return replace(self, cumulative_secs=self.cumulative_secs + remaining)


# TODO: every guild's ETAs still render in DEFAULT_TIMEZONE, and in one zone per
# guild rather than per viewer.
# queue_embed()'s "Est. playing at" and the now-playing "Estimated finish" read
# GuildConfig.timezone, but nothing WRITES it — set_timezone has no caller and the
# `-options key value` command it exists for does not exist yet — so the field is
# always absent and every guild is still quoted Pacific time. Two debts, not one:
# a write path, and then the fact that a guild-wide zone still shows every member
# the same clock. Discord relative timestamps (<t:epoch:R>) would render in each
# VIEWER's locale and need no setting at all; they were not used here because the
# surrounding text reads as a wall clock ("Est. playing at 4:15 PM"), which a
# relative chip does not express.


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
    """A wall-clock time with the zone it is in.

    The suffix comes from the datetime, not a literal. It used to be a hardcoded
    "PST", which a configurable zone would turn into an outright lie — and which was
    already wrong for the ~8 months a year US/Pacific spends in PDT.
    """
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    # tzname(), not strftime("%Z"): identical output, ~50x cheaper (strftime builds a
    # whole timetuple and calls into the C library for one field). This runs on the
    # NP progress tick and on every reply while a song is live. Zones with no
    # abbreviation return a UTC offset, which is still unambiguous; None is possible
    # for a naive datetime, hence the `or ""`.
    return f"{hour}:{dt.minute:02d} {ampm} {dt.tzname() or ''}".rstrip()


def _fmt_eta(est_dt: datetime.datetime, uncertain: bool) -> str:
    prefix = "~" if uncertain else ""
    return f"{prefix}**{_fmt_clock_time(est_dt)}**"


def _requester_mention(
    requester: Optional[Union[discord.User, discord.Member]],
) -> str:
    return requester.mention if requester else "Unknown"


# Square emoji blocks read thicker and higher-contrast than a thin dash, and let
# the played portion render in a visibly different colour from the remainder.
# Width is low because each block glyph is much wider than a dash.
_BAR_WIDTH = 10
_BAR_FILL_DONE = "🟦"
_BAR_FILL_REMAINING = "⬜"
_BAR_HEAD = "🔘"

# How long mark_paused()/mark_resumed() wait before the embed edit + Activity
# refresh — collapses rapid -pause/-resume toggling into one trailing update
# instead of an API call pair per toggle.
_PAUSE_DEBOUNCE_SECS = 0.5

# ── -playnow interjection ──────────────────────────
# Below this many seconds remaining, an interjected song gets no resume entry —
# there is nothing meaningful to return to.
_MIN_RESUME_REMAINING_SECS = 5
# EOF guard for the resume seek (duration metadata is imprecise), matching the
# crash-recovery position cap in _restore_state().
_RESUME_EOF_MARGIN_SECS = 10

# ── Progress-bar finalize ─────────────────
# Tolerance for "this song reached its end", absorbing drift between yt-dlp's
# duration metadata and the real stream length. Anything that stopped short —
# skipped, interjected, dead stream — keeps the bar where it actually reached.
_SONG_COMPLETE_MARGIN_SECS = 5

# ── Playback gate ────────────────────────────────
# How long loop() waits for a voice connection before tearing the player down.
# Matches the idle queue_get() timeout, so a player that never connects and one
# that connects but is never given a song disconnect on the same schedule.
_PLAYBACK_GATE_TIMEOUT = 300


@dataclass(frozen=True)
class InterjectOutcome:
    """What MusicPlayer.interject() did — everything -playnow needs for its
    confirmation wording."""

    interrupted_title: str
    # None → no resume entry (the interrupted song was nearly finished, or had no
    # webpage_url to rebuild from). Every other interruption parks one, however
    # many are already parked behind it.
    resume_position: Optional[int]
    was_paused: bool  # the OBSERVED state when it was interrupted
    # Whether the resume entry comes back PAUSED — distinct from was_paused, since
    # -playnow restores what it interrupted while -play brings it back playing.
    # Wording must key off this, or a -play interjection announces "will return
    # paused" for a song that returns playing.
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
    mid-song stream death that produced audio (which `stream_failed` deliberately
    does not call a failure). No known duration → False."""
    if song.duration_secs <= 0:
        return False
    return song.position_secs >= song.duration_secs - _SONG_COMPLETE_MARGIN_SECS


def _remaining_secs(item: QueueObject) -> Optional[int]:
    """A queued item's expected playtime: full duration, minus the resume offset
    for a -playnow resume entry — it plays only its tail, so counting the full
    duration would overestimate everything behind it."""
    if item.duration is None:
        return None
    if item.is_resume and item.ts:
        return max(0, item.duration - item.ts)
    return item.duration


def _queue_runtime(items: list[QueueItem]) -> tuple[int, bool]:
    """Total remaining playtime of queued items, and whether any duration was
    unknown — an unresolved YTSource or a QueueObject with no duration makes the
    total a lower bound, which callers flag with "~". Shared by queue_embed() and
    the resume notices (via _add_resume_fields) so they can't disagree."""
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
    # Clamp before formatting so the elapsed label can't overshoot the duration
    # label — imprecise metadata plus an FFmpeg -ss offset can push the raw
    # position past the reported duration.
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
    """Clock time `duration_secs` from now. No uncertainty prefix: a song's own
    remaining duration, unlike a queued song's ETA, is known once it's playing."""
    finish_dt = datetime.datetime.now(tz=tz) + datetime.timedelta(seconds=duration_secs)
    return _fmt_clock_time(finish_dt)


# Discord rejects an empty embed field value (400, "This field is required"),
# which fails the entire send/edit, not just that field. Anything that can
# legitimately be missing goes through here.
_FIELD_PLACEHOLDER = "—"


# Nothing bounds a yt-dlp uploader string, and Discord counts every character in
# every embed of a message against one 6000-char budget — which the NP block
# shares with whatever response MusicContext.send prepends it to (a full
# -leaderboard is ~2.5 KB of that). A field over 1024 characters is a 400 on its
# own. Both are bounded here rather than at the call sites.
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
    """Shared field layout for both now-playing builders — live (YTDL-backed) and
    Redis-recovery (NowPlayingData-backed). Channel/Views/Likes are exactly Discord's
    per-row cap of three inline fields. Duration is not a field (the live bar carries
    it, the recovered embed puts it in the description), nor is the webpage URL — the
    title links to it. Views/likes are routinely absent and a partial Redis hash can
    blank any value, hence _field_value on every one."""
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

    bot: commands.Bot
    _guild: discord.Guild
    _channel: discord.TextChannel
    _last_author: Optional[Union[discord.User, discord.Member]]
    _cog: "MusicBot"
    current_song: Optional[YTDL]
    play_next: asyncio.Event
    queue: GuildQueue
    play_message: Optional[discord.Embed]
    history: GuildHistory
    volume: float
    _player: Optional[asyncio.Task]
    # Parameterized, unlike its siblings: _neutralize_prefetch reads fields off
    # this task's result, and a bare Task makes result() Any — so a field YTDL
    # does not carry would raise at runtime with pyright reporting nothing.
    _prefetch_task: Optional[asyncio.Task[Optional[YTDL]]]
    store: Optional[GuildRedisStore]
    _restore_task: Optional[asyncio.Task]
    _restore_complete: asyncio.Event
    _restore_read_failed: bool
    _stopped_deliberately: bool
    _playback_gate: asyncio.Event
    _playback_holds: int
    _background_tasks: set[asyncio.Task[Any]]
    _progress_task: Optional[asyncio.Task]
    _heartbeat_task: Optional[asyncio.Task]
    # Last payload pushed and the host it went to, for the no-op-edit guard in
    # _push_np_edit. Compared only for equality; Embed.to_dict() is a TypedDict
    # and list is invariant, so the element type stays Any.
    _np_last_rendered: Optional[list[Any]]
    _np_last_id: Optional[int]
    _np_host_message: Optional[discord.Message]
    _np_host_own_embeds: list[discord.Embed]
    _np_host_dedicated: bool
    _np_edit_lock: asyncio.Lock
    _pause_debounce_task: Optional[asyncio.Task]
    _skip_history_for: Optional[YTDL]
    _pending_resume_tail: Optional[QueueObject]
    _ended_song: Optional[YTDL]
    _last_stream_error: Optional[StreamFailure]

    def __init__(
        self,
        bot: commands.Bot,
        guild: discord.Guild,
        channel: discord.TextChannel,
        cog: "MusicBot",
        redis: Optional[aioredis.Redis] = None,
    ) -> None:
        self.bot = bot
        self._guild = guild
        self._channel = channel
        # Genuinely nullable: guild.me is None until the member cache fills, and
        # guild.owner can be uncached. discord.py's stub declares Guild.me as Member,
        # collapsing the `or` and hiding that — hence the explicit annotation.
        # from_context()/set_context() overwrite this before any command path reads
        # it; _require_requester() covers the rest.
        _fallback: Union[discord.Member, discord.User, None] = guild.me or guild.owner
        self._last_author = _fallback
        self._cog = cog

        self.current_song = None
        # Set by _stream_source() on a failed resolve; read by the loop to build a
        # descriptive skip notice, then cleared.
        self._last_stream_error: Optional[StreamFailure] = None
        self.play_next = asyncio.Event()

        self.play_message = None
        self.volume = 1.0
        # Replaced at restore from GuildConfig; the default is what a guild that
        # has never set one renders in.
        self.timezone = ZoneInfo(DEFAULT_TIMEZONE)

        self.store = (
            GuildRedisStore(redis, self._guild.id) if redis is not None else None
        )
        # All queue state (one deque + cursor, Redis mirror, bulk mutex, wake
        # Event, cleared-flag) lives behind this one object — see guild_queue.py.
        self.queue = GuildQueue(guild, self.store)
        # Played-song history (in-memory ring + Redis mirror) — guild_history.py.
        # Only the DRAINER is wired in: history writes nudge it, nothing here reads
        # Postgres back. It lives on the app, one per process, present exactly when
        # HISTORY_ARCHIVE_ENABLED, so a None drainer wires the None notify that
        # GuildHistory's constructor demands be explicit. The cast is a runtime no-op.
        app = cast("MusicBotApp", bot)
        self.history = GuildHistory(
            self.store,
            on_outbox_push=(
                app.history_drainer.notify if app.history_drainer is not None else None
            ),
        )
        self._player: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task[Optional[YTDL]]] = None
        self._restore_task: Optional[asyncio.Task] = None
        self._restore_complete = asyncio.Event()
        # Set when the restore could not READ the store. Without it an empty queue is
        # ambiguous — nothing saved vs nothing readable — and a command reporting the
        # first when it means the second tells a guild its queue is gone.
        self._restore_read_failed = False
        # Set by whoever calls vc.stop() on the live song, cleared at each vc.play().
        # Zero frames alone cannot tell a stream that never opened from one we stopped
        # before its first frame; this is the half that says which.
        self._stopped_deliberately = False
        # Restore and *play* are separate concerns: the gate stays shut until a
        # command establishes a voice connection, so a player built by a command that
        # never connects — every command that reaches the cog hook builds one, and
        # only the join opens this — cannot walk the queue and discard it.
        self._playback_gate = asyncio.Event()
        # >0 while an in-flight command owns the opening — -play holds the gate
        # across the join it triggers, so the restored head can't start before the
        # requested song is inserted in front of it.
        self._playback_holds = 0
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._progress_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._np_last_rendered: Optional[list[Any]] = None
        self._np_last_id: Optional[int] = None
        # NP host state: the message carrying the block, its own cached embeds that
        # follow it, and whether it is a dedicated NP message (deleted on retire) or
        # a command response (strip-edited).
        self._np_host_message: Optional[discord.Message] = None
        self._np_host_own_embeds: list[discord.Embed] = []
        self._np_host_dedicated: bool = False
        self._np_edit_lock = asyncio.Lock()
        self._pause_debounce_task: Optional[asyncio.Task] = None
        # Set by interject() to the song it stopped with a resume entry pending, so
        # the stop transition skips its history add and it is recorded once, when its
        # tail finishes. Holds the song's identity, not a flag: the song can end
        # during interject()'s awaits and a stale boolean would eat the next entry.
        self._skip_history_for: Optional[YTDL] = None
        # Its companion: the resume entry built for that same song, awaiting the
        # NP-card ids that only exist at the fragment's iteration end. Set and
        # cleared wherever _skip_history_for is, for the same staleness reason.
        self._pending_resume_tail: Optional[QueueObject] = None
        # The song whose playback ended but whose history row is not written yet.
        # current_song is nulled before the prefetch await; this keeps the play
        # claimable across it — see claim_current_song_for_history.
        self._ended_song: Optional[YTDL] = None

    @classmethod
    def from_context(
        cls,
        bot: commands.Bot,
        ctx: commands.Context,
        redis: Optional[aioredis.Redis] = None,
    ) -> "MusicPlayer":
        assert ctx.guild is not None
        assert isinstance(ctx.channel, discord.TextChannel)
        assert ctx.cog is not None
        # ctx.cog is Optional[Cog] to discord.py, but MusicBot is the only cog owning
        # the commands that reach here.
        mp = cls(bot, ctx.guild, ctx.channel, cast("MusicBot", ctx.cog), redis=redis)
        mp._last_author = ctx.author
        return mp

    def start(self) -> None:
        """Start the playback loop and, with Redis, the state restore task.

        loop() blocks on _restore_complete before consuming the queue (see
        _restore_state); with no store the event is set immediately. It then blocks on
        the playback gate, opened here when the guild already has a voice client —
        restore_guild() connects before calling start(), so recovery resumes from the
        head with no extra call site. Otherwise -join / -play open it."""
        if self._guild.voice_client is not None:
            self.open_playback_gate()
        if self.store is not None:
            self._restore_task = self.bot.loop.create_task(self._restore_state())
        else:
            # No Redis, so no restore runs — signal now so loop() never waits.
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
        """Hold the playback gate shut for the duration of the block.

        -play calls -join, which opens the gate the moment the handshake completes —
        while -play is still resolving its input (a 1-4s extraction). Without this
        hold, the restored head starts playing in that window. The gate opens on the
        way out even when the block raised: -play's error path calls cleanup(), which
        makes the gate moot, and resuming the queue beats stranding it."""
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
        """Block until _restore_state() has finished (or failed). Inserting before
        restore has read its snapshot double-queues: put_front() LPUSHes the mirror,
        while restore_entries() is in-memory only precisely because its entries are
        already on that list.

        False when `timeout` elapsed first. The pool sets no socket_timeout, so a
        Redis that accepts the connection and then stalls hangs the read — and with
        it any command that waits here — until the server answers.
        """
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
        """The fallback requester, for paths that must have one. QueueObject.requester
        is non-optional because persistence reads `requester.id`. _last_author is
        unset only on a player whose guild has both bot member AND owner uncached, and
        never after a command has run — so raising here names the cause instead of
        surfacing as an AttributeError on None during serialization."""
        if self._last_author is None:
            raise RuntimeError(
                f"No requester available for guild {self._guild.id}: neither the "
                "bot member nor the guild owner is cached"
            )
        return self._last_author

    def _queue_eta_seed(self) -> tuple[datetime.datetime, EtaWalk]:
        """Seed state for walking ETAs across queued songs: (now_pst, walk).
        cumulative_secs starts at the current song's total duration as a proxy for
        its remaining time — an overestimate that avoids showing "now" for
        everything; uncertain flags an unknown duration anywhere behind."""
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
        """Format one queue line with its "Est. playing at" ETA. Returns (line,
        updated walk) so callers can chain across consecutive items.
        """
        est_dt = now_pst + datetime.timedelta(seconds=walk.cumulative_secs)
        est_str = _fmt_eta(est_dt, walk.uncertain)

        if isinstance(item, QueueObject):
            # Capped for the same reason _field_value is: a -queue page renders
            # ten of these into one 4096-char description, and the "Up next"
            # embed renders one into a block sharing a message-wide budget.
            # Sanitized too — this lands inside a masked link's LABEL, where a
            # "]" in the title would close it early and re-point the link.
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

    def estimated_playing_at(self) -> str:
        """ETA text for a song appended right now — after the current song and
        everything queued. Same seed as queue_embed()/_build_next_up_embed() so all
        three stay consistent.
        """
        now_pst, walk = self._queue_eta_seed()
        for item in self.queue.display_items():
            walk = walk.advance(
                _remaining_secs(item) if isinstance(item, QueueObject) else None
            )
        est_dt = now_pst + datetime.timedelta(seconds=walk.cumulative_secs)
        return _fmt_eta(est_dt, walk.uncertain)

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
        while the queue head is still the restored one.

        The two restores landing here know different things:
        * A crash re-queues the mid-play song at the head (persisted=False,
          recovery offset in `ts`). That song is where the session stopped.
        * A `-stop` cancels the loop mid-song before its history bookkeeping, so the
          interrupted song is recorded nowhere and clear_connection() dropped its
          state fields. The newest history entry is the last song that ran to its
          END, which is older than the stop — hence "Last played", not a claim about
          where playback stopped.
        """
        head = self.queue.peek_next()
        if isinstance(head, QueueObject) and not is_persisted(head) and head.title:
            value = f"**{truncate_embed_title(head.title)}**"
            if head.ts:
                value += f"\n`{fmt_duration(head.ts)}`"
                if head.duration:
                    value += f" / `{fmt_duration(head.duration)}`"
            return "Left off on", value

        # Absent when history was never populated (no Redis, or a guild whose first
        # song is still its current one) — the queue half stands on its own.
        last = self.history.latest
        if last is None or not last.title:
            return None
        value = f"**{truncate_embed_title(last.title)}**"
        value += f"\n`{fmt_duration(last.played_secs)}`"
        if last.duration_secs > 0:
            value += f" / `{fmt_duration(last.duration_secs)}`"
        # played_at == 0 means unknown (absent on the wire); <t:0:R> would render
        # "56 years ago", so omit the line — same rule as history_embeds().
        if last.played_at:
            value += f"\n<t:{int(last.played_at)}:R>"
        return "Last played", value

    def build_resume_notice_embed(
        self, started: QueueObject
    ) -> Optional[discord.Embed]:
        """Heads-up that `-play` on a disconnected bot woke a persisted queue. Build
        before front-inserting, while the queue holds only restored entries; None
        when nothing was restored (the common first-`-play` case).

        `started` must be named here because this response hosts no NP block: the
        gate is held shut across the enqueue, so current_song is still None and
        MusicContext._np_player() returns None — the real NP message lands seconds
        later. Without the title, the first thing the user sees after `-play` is an
        embed about a *different* song. It also adds what only the restore knows:
        which song the previous session left off on, and how much queue waits behind
        the one starting now.
        """
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
        # The song being started, not the one resumed from: the thumbnail sits
        # next to "Playing now" and has to match it.
        if started.thumbnail:
            embed.set_thumbnail(url=started.thumbnail)

        self._add_resume_fields(embed, items)
        return embed

    def build_rejoin_resume_embed(self) -> Optional[discord.Embed]:
        """Heads-up that `-resume` on a disconnected bot rejoined voice and woke a
        persisted queue. Build while the queue head is still the restored one:
        once the gate opens the loop would pop that head out from under this.

        No song is named, unlike build_resume_notice_embed: nothing was inserted
        here, so the head this describes IS the song the Now Playing card names
        seconds later. None when the restore found nothing, which the caller
        reports instead of joining a channel to sit silent in.
        """
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
        """Take the playing song's history entry so a teardown can record it.

        A teardown abandons a song mid-play and nothing else records it: its queue
        entry was LPOPed when it started, clear_connection() drops the parked state
        copy, and the loop is cancelled while parked in play_next.wait().

        SYNCHRONOUS by design — it reads the song, decides, and takes the
        _skip_history_for marker with no await between, and the loop reads that
        marker after its prefetch await, so exactly one of the two writes.

        The _ended_song fallback covers that await: current_song is nulled before it
        and history written after, so current_song alone is blind for as long as a
        cold extraction takes. None when there is nothing to record.
        """
        song = self.current_song or self._ended_song
        if song is None:
            return None
        if self._skip_history_for is song:
            # An interjection parked this song's tail, and a teardown leaves the
            # queue intact under its 24h TTL — so -resume plays it and records it.
            return None
        if not song.produced_audio:
            # ffmpeg exited without a frame: nobody heard it (see produced_audio).
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
        background task; waits for bot ready so guild members are cached.

        loop() waits on _restore_complete before its first queue_get(): otherwise it
        races ahead, dequeues the crash-recovered "current song" injected below and
        pop_queue()s as normal bookkeeping — but that song was never on the Redis
        queue list (it lives in current_song_url state), so the LPOP silently deletes
        an unrelated, still-queued song."""
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
                    # One pipelined read covers the whole playback aggregate: state
                    # hash, pending queue, now-playing snapshot, newest history.
                    snapshot = await self.store.get_playback_snapshot()
                    if snapshot is None:
                        # Read failed — abort rather than proceed with fabricated
                        # defaults. `finally` still sets _restore_complete, so
                        # loop() is never blocked.
                        self._restore_read_failed = True
                        log.warning(
                            f"State restore aborted for guild {self._guild.id}: "
                            f"Redis unavailable"
                        )
                        return
                    guild_state = snapshot.state

                    # Unconditional is right here: tzinfo() already degrades to the
                    # default for an unset or unusable name, so there is no "absent"
                    # case for this assignment to skip.
                    self.timezone = snapshot.config.tzinfo()

                    stored_volume = snapshot.stored_volume
                    # Only when a value was actually stored: an unconditional assign
                    # would clobber a concurrent -volume with the default.
                    if stored_volume is not None:
                        self.volume = stored_volume
                        # Seed a pre-move value forward, once. Volume lived in the
                        # state hash, which expires in 24h, so leaving it there means
                        # a guild quiet for a day loses the setting. migrate_volume,
                        # NOT set_volume: this snapshot was read an arbitrary number
                        # of awaits ago, and a -volume that landed since must not be
                        # overwritten by the older value it is carrying.
                        if snapshot.config.volume is None and self.store is not None:
                            await self.store.migrate_volume(stored_volume)

                    # Display snapshot, so -now works if a song was playing.
                    if snapshot.now_playing is not None:
                        self.play_message = self._build_now_playing_embed_from_data(
                            snapshot.now_playing
                        )

                    # Re-queue the song that was playing at the crash. current_song_url
                    # is set atomically with the LPOP, so a non-empty value means the
                    # bot died after that transaction committed but before the song
                    # finished (at-most-once delivery).
                    if guild_state.has_crashed_song:
                        # Approximate position at crash time — pure math on the
                        # snapshot, no IO. crashed_position_at documents why it is
                        # only approximate (downtime counts as playback).
                        position = guild_state.crashed_position_at(time.time())
                        if position is not None:
                            # Cap at duration − 10s so FFmpeg can't seek past EOF.
                            # Narrow try/except: a malformed cached duration must
                            # degrade to "no cap", not abort the whole restore.
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

                        # The crashed current_song_* fields ARE a queue entry — the
                        # one the start transaction LPOPed. Rebuild and re-queue it
                        # through the same rehydration path as everything else;
                        # the requester chain falls back to guild.me then owner.
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
                        # set makes every later restart re-enter this block until
                        # the TTL expires.
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

                # Refresh TTL on all guild keys after successful restore.
                await self.store.refresh_ttl()
        finally:
            # Always signal finished-or-failed so loop() never blocks forever.
            self._restore_complete.set()

    async def repark_crashed_head(self) -> bool:
        """Write a crash-recovered queue head back into the state hash it came from.
        True when something was re-parked.

        _restore_state clears current_song_* as soon as it re-queues that song, so
        this player's memory is its only copy — dropping it loses the song silently.
        Call AFTER cleanup(): its clear_connection() HDELs these same fields.
        """
        head = self.queue.peek_next()
        if self.store is None or not isinstance(head, QueueObject):
            return False
        if is_persisted(head):
            # Already on the Redis list: parking it would re-queue a second copy.
            return False
        # Backdated by the resume offset as the loop does at vc.play, and seeded as
        # the recorded position: the hash carries no `ts`, so the offset is the only
        # record of how far this song had reached.
        await self.store.set_current_song_state(
            SongQueueEntry.from_queue_object(head),
            time.time() - (head.ts or 0),
            start_offset=head.ts or 0,
        )
        return True

    # ── Queue operations ──────────────────────────────────────────────────────

    def enqueue_depth(self) -> int:
        """Songs a new arrival waits behind: everything queued, plus the one playing.

        Read once at command dispatch, so the loop's continuous dequeuing leaves
        queue_position approximate against the insert.

        The live song is NOT counted when its resume tail is already queued: after
        an interjection that entry is the same play as current_song, and counting
        both puts a new arrival one song too deep.

        Two accepted ±1 windows:

        - OVER by one while the loop resolves a stream, since current_song is
          assigned before try_commit_dequeue() settles the claim, so the play is
          both current_song and still queued for the length of a probe
          (100ms-seconds).
        - UNDER by one when the live song has a parked tail from an EARLIER play
          of the same URL (-play X, -playnow Y, -playnow X), which
          has_resume_tail matches by URL."""
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
        playlist enqueues: the mirror is written in one batch round-trip and no
        per-item tasks spawn, since N concurrent prefetches saturate the pool and
        mint stream URLs that expire before playback reaches them.
        _prefetch_next_song covers one-ahead prefetch as songs play."""
        items: list[QueueItem]
        if isinstance(obj, (QueueObject, YTSource)):
            items = [obj]
        else:
            items = list(obj)
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
        as queue_put(), used when -play runs on a disconnected bot with a persisted
        queue: the requested song plays now, the persisted entries resume behind it.
        Playlists insert in full and in order, with prefetch=False for the same
        reason as queue_put()."""
        items: list[QueueItem]
        if isinstance(obj, (QueueObject, YTSource)):
            items = [obj]
        else:
            items = list(obj)
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
        """Cancel any in-flight prefetch task and wait for it to finish. Must run
        before any bulk queue mutation, so the item the prefetch dequeued via
        get_nowait() is returned to the front (requeue_front, in its CancelledError
        handler) before the drain — the mutation then handles it with everything else
        instead of stranding it. A prefetch blocked inside run_in_executor cannot be
        interrupted, so this await can sit until the worker exits (socket_timeout)."""
        await cancel_task(self._prefetch_task)

    async def _flush_played(self, items: Sequence[QueueItem]) -> None:
        """Record every item that already played and is now leaving the queue for
        good — the -clear/-remove half of "a song is recorded exactly once, when
        its queue object exits the queue".

        These are -playnow resume tails: songs a listener heard part of, whose
        remaining tail is discarded, so nothing else records them (the loop's single
        write site only fires for a song it played to its end). `played_at > 0.0` is
        the whole test — the loop stamps it at vc.play() and the tail inherits it;
        the isinstance guard is real, since a YTSource has no such field.

        -stop and the idle disconnect are deliberately not covered: they leave the
        queue intact under its 24h TTL, so a later -resume still plays those tails.

        Accepted crash window: clear()/remove() destroy the Redis mirror inside the
        bulk mutex and this runs after it, so a crash in between loses these records
        with no row and no log line — the same at-most-once posture as the dequeue
        LPOP (see redis_client.pop_queue).
        """
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
                # __post_init__ raises rather than coercing, and clear() has already
                # destroyed the mirror by now — so one malformed wire value must not
                # abort the batch and drop every other play with it.
                log.warning(
                    f"history flush skipped a malformed entry in guild "
                    f"{self._guild.id}: {type(e).__name__}: {e}"
                )
        if not entries:
            return
        # Concurrently: each write is an independent MULTI plus an outbox push, and
        # a deep stack would pay the round trip per level before -clear can reply.
        await asyncio.gather(*(self.history.add(entry) for entry in entries))

    async def _dispose_orphaned_cards(self, items: Sequence[QueueItem]) -> None:
        """Retire the frozen NP card of every played tail leaving the queue.

        A tail disposes of the card its interrupted fragment left behind when it
        STARTS, so a tail destroyed before it ever plays takes the only pointer with
        it and the dead partial bar stays in the channel — -clear on a 3-deep stack
        strands three.

        Fire-and-forget per item: these are rate-limited Discord calls the command
        must not wait on, and a failure is cosmetic."""
        for item in items:
            if isinstance(item, QueueObject) and item.is_resume:
                self._spawn_background(self._dispose_previous_np_card(item))

    async def _retire_failed_dequeue(
        self, item: Optional[QueueItem], *, context: str
    ) -> None:
        """Retire a dequeue that will never play, and record it if a listener
        already heard part of it.

        For an ordinary song the flush is a no-op — nobody heard it. For a -playnow
        resume TAIL it is the difference between one record and none: the
        interrupted fragment declined to record itself (_skip_history_for), so the
        tail is the only writer left for a play that may have run for minutes. A
        tail behind a deep stack waits minutes while ytdl:stream:* caps at 30, so
        every level of a stack is an independent chance to lose one."""
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
        # Cancel before shuffle()'s too-few guard: a prefetch holding a dequeued
        # item must be accounted for even when the shuffle is a no-op.
        await self._cancel_prefetch()
        outcome = await self.queue.shuffle()
        if outcome is ShuffleOutcome.TOO_FEW_SONGS:
            return "There must be at least 3 songs to shuffle the queue"
        return "Shuffled!"

    async def queue_remove(self, needle: str) -> RemoveOutcome:
        """Remove every queued item matching `needle` — the resolved yt-dlp URL, or
        what the user originally typed (see remove_matcher). Returns the whole
        outcome, since the command reports the positions and names the mode."""
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
        _np_host_own_embeds keep their send-time footer. No elapsed_ms — no command
        here took any time. NP-block callers pass no span, so a re-rendered block
        cannot alternate trace ids. See docs/ARCHITECTURE.md#debug-footer-seams.
        """
        if not self._cog.debug_settings.enabled(self._guild.id):
            # Strip rather than return: play_message outlives a mid-song --disable.
            strip_debug_footers(embeds)
            return
        decorate_embeds(
            embeds,
            span=span,
            shard_id=self._guild.shard_id,
            runtime=self._cog.debug_settings.snapshot,
        )

    def _build_now_playing_embed(
        self, song: YTDL, *, position_override: Optional[float] = None
    ) -> discord.Embed:
        """position_override renders the bar at a given position instead of
        song.position_secs — used by _finalize_now_playing() to show a complete bar
        once the song has actually ended."""
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
                # Bar sits under the title, above the requester line, blank line
                # between for separation.
                lines.append(bar)
                lines.append("")
        requester_line = f"Requester: [{_requester_mention(song.requester)}]"
        if song.duration_secs > 0:
            # Remaining, not total: a song started mid-stream (?t=, crash
            # recovery, -playnow resume) finishes sooner than its full length.
            remaining = max(0, song.duration_secs - int(position))
            requester_line += (
                f"  ·  Estimated finish: {_fmt_finish_time(remaining, self.timezone)}"
            )
        lines.append(requester_line)
        description = "\n".join(lines)
        fields = NowPlayingData.from_song(song)
        return _build_now_playing_base_embed(
            # No markdown: Discord renders embed titles literally, so
            # "**Now playing:**" would show its asterisks inside the link text.
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
        """Slim -pause confirmation: just the pause position. The response hosts the
        live NP block right below, so repeating the bar/requester/thumbnail would
        render them twice — the paused state is the one thing the block does not
        show. position_secs is frozen while paused, so it is the exact point (-ss
        offset included). None when no song is live."""
        song = self.current_song
        if song is None:
            return None
        position = int(song.position_secs)
        duration_secs = song.duration_secs
        if duration_secs > 0:
            paused_at = f"{fmt_duration(position)} / {fmt_duration(duration_secs)}"
        else:
            paused_at = fmt_duration(position)
        return discord.Embed(
            title=f"⏸️ Paused: {song.title}",
            description=f"Paused at: `{paused_at}`",
            color=discord.Color.orange(),
        )

    @staticmethod
    def _build_now_playing_embed_from_data(data: NowPlayingData) -> discord.Embed:
        """Reconstruct a now-playing embed from the recovered Redis snapshot.
        Duration goes in the description, in the slot the bar occupies live: the base
        builder drops the Duration field because the bar's label carries it, and
        there is no bar here (no live position until loop() starts). Rendered as
        stored, not re-parsed."""
        lines = []
        if data.duration:
            # Blank line after, matching the live embed's bar/requester spacing.
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

    def _build_next_up_embed(self) -> Optional[discord.Embed]:
        item = self.queue.peek_next()
        if item is None:
            return None
        now_pst, walk = self._queue_eta_seed()
        description, _ = self._format_queue_line(item, 1, now_pst, walk)
        return discord.Embed(
            title="Up next",
            description=description,
            color=discord.Color.blue(),
        )

    # ── Now-playing host management ───────────────────────────────────────────
    # The NP block lives in exactly one "host" message at a time — always the newest
    # bot message, so the bar is never buried. Command responses adopt it by
    # prepending at send time (MusicContext.send). The previous host is retired:
    # deleted if it was a dedicated NP message, strip-edited otherwise.

    def np_embed_block(
        self, *, now_playing: Optional[discord.Embed] = None
    ) -> list[discord.Embed]:
        """The [now_playing, next_up?] block, or [] when no song is live — the one
        place encoding its internal order. `now_playing` lets a caller that already
        built this song's embed supply it instead of building an identical one.

        Decoration happens here, not at each caller, so every attach site gets it
        from one place. A supplied `now_playing` may be the cached play_message,
        decorated more than once over its life; decorate_embeds replaces rather than
        appends, which is what makes that safe."""
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
        self._decorate_for_debug(block)
        return block

    def _adopt_np_host(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        dedicated: bool = False,
    ) -> None:
        """Pointer-first host swap. The pointer update is synchronous (atomic on the
        event loop), so any tick starting after this targets the new host. Retiring
        the old one is fire-and-forget; _retire_np_host's lock orders it after any
        in-flight tick edit against that message."""
        old_msg = self._np_host_message
        old_own = self._np_host_own_embeds
        old_dedicated = self._np_host_dedicated
        if old_msg is not None and message.id < old_msg.id:
            # Overlapping sends can complete out of order: channel position is
            # send-START order, adopts run in send-RETURN order. Adopting the older
            # message would pull the block up from the true bottom — keep the newer
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
        `song` before the send's await, and the song may have ended or been replaced
        in flight; adopting then installs a stale block as host and delete-retires
        the next song's NP message (or leaves a frozen block nothing cleans up), so
        the just-sent message sheds it instead. True when adopted."""
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
        """Remove the NP block from a message that is no longer the host.

        The STRIP takes the edit lock so an in-flight tick edit finishes first:
        concurrent PATCHes resolve last-write-wins server-side, and a tick landing
        after the strip would resurrect the block on the retired host.

        The DELETE does not. Nothing can resurrect a deleted message, while message
        deletion is its own stricter ratelimit bucket — held across it, one 429
        stalled every NP edit for the NEW song for the retry-after."""
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

        Takes either form of the same tail — the YTDL the loop is about to play, or
        the QueueObject a bulk mutation is destroying — since both carry the np_*
        fields and the card outlives whichever goes away. Without it a -playnow
        stack accumulates one dead partial bar per interjection.

        With the live ref this is _retire_np_host verbatim, so a card hosted by a
        command response is strip-edited back to its own embeds. After a restart
        only the ids survive and own_embeds cannot be reconstructed from them, so
        the by-id delete is gated to DEDICATED cards. Never a re-adopt: the live bar
        belongs at the channel bottom. See
        docs/ARCHITECTURE.md#now-playing-host-invariants.
        """
        ref = song.np_host_ref
        if ref is not None:
            await self._retire_np_host(ref.message, ref.own_embeds, ref.dedicated)
            return
        mid, cid = song.np_message_id, song.np_channel_id
        # parse_queue_entry coerces nothing, so three wire values reach this
        # DESTRUCTIVE call unchecked. np_dedicated is the authorization between
        # "delete this message" and "leave a user's reply alone", so `is True` and
        # not truthiness — a wire "false" is a truthy string.
        if song.np_dedicated is not True:
            return
        # bool excluded: isinstance(True, int) is True, so a wire `true` would
        # render "True" into the REST route.
        if isinstance(mid, bool) or isinstance(cid, bool):
            return
        if not (isinstance(mid, int) and isinstance(cid, int)):
            return
        if not (mid > 0 and cid > 0):
            return
        # Scoped to THIS guild first: a PartialMessageable validates nothing, so a
        # stale or corrupted channel id would delete a message wherever it happens
        # to resolve, including another guild or a DM.
        if self._guild.get_channel_or_thread(cid) is None:
            return
        # get_partial_messageable: issues the DELETE without the channel being
        # cached, which it may not be on the restart path that reaches this branch.
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
            # its retries are spent, and this runs fire-and-forget — unhandled they
            # land outside structlog as "Task exception was never retrieved".
            log.warning(
                f"NP card cleanup errored for guild {self._guild.id}: "
                f"{type(e).__name__}: {e}"
            )

    def _release_np_host(self) -> None:
        """Clear host state without retiring the message. Used at song end: the
        completed bar stays in the channel as a record, and the next song's adopt
        sees no old host to retire."""
        self._np_host_message = None
        self._np_host_own_embeds = []
        self._np_host_dedicated = False
        # Retiring can strip-edit the message outside _push_np_edit, so the cache
        # goes with the host or a stale entry suppresses a needed edit.
        self._np_last_rendered = None
        self._np_last_id = None

    async def retire_np_host_on_stop(self) -> None:
        """-stop / alone-disconnect teardown: dispose of the host so no message keeps
        a live-looking bar for a player that no longer exists. Song end RELEASES
        instead — a completed bar is a truthful record, a mid-song frozen one is not.
        cleanup() calls this after the progress/loop tasks are cancelled."""
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
        """Player-initiated sends that bypass ctx.send (and MusicContext's attach
        hook) but must still keep the NP block at the bottom — the same
        splice-send-adopt sequence as MusicContext.send."""
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
                # Backdated by the true audio position, not "now", so resuming
                # mid-song still lands `end` the correct remaining duration ahead
                # and a -ss/crash-recovered song's tooltip agrees with the bar
                # (both read position_secs).
                now_ms = int(time.time() * 1000)
                position_ms = int(song.position_secs * 1000)
                timestamps["start"] = now_ms - position_ms
                if song.duration_secs > 0:
                    timestamps["end"] = timestamps["start"] + song.duration_secs * 1000
            # else: paused — timestamps stays {} so Discord shows static text
            # instead of a bar animating through the pause. Omitting them is the
            # Activity schema's only "frozen" representation.

            # Bot opcode-3 activities reliably render only `name` (Rich Presence
            # needs the Discord RPC/SDK, which server bots cannot use), so the
            # uploader is packed into `name` as a suffix. `timestamps` still works
            # in the hover tooltip.
            title = song.title or "a song"
            uploader = song.uploader
            raw_name = f"{title} · {uploader}" if uploader else title
            name = raw_name if len(raw_name) <= 128 else raw_name[:127] + "…"

            # `state` renders in both hover and click card for bot activities;
            # state_url is forward-compat (the URL may become clickable).
            # details/details_url are confirmed non-rendering for bots.
            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=name,
                state=song.duration,
                state_url=song.webpage_url,  # discord.py >= 2.6; silent no-op if downgraded
                timestamps=timestamps,
            )
        else:
            # Only reset when no OTHER guild is playing: cleanup() cancels the loop
            # before disconnecting, so the CancelledError handler reaches here while
            # this guild's client is still connected, and counting it would strand
            # the presence on the stopped song.
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
        """Pause playback and sync all pause-tracking state in one place: the Redis
        crash-recovery epoch accounting and the progress-bar/Activity refresh. One
        entry point, so a future call site can't forget either side effect."""
        vc.pause()
        if self.store is not None:
            # One instant for both writes, so the legacy wall-clock math cannot
            # count the gap between them as playback the heartbeat never saw.
            paused_at = time.time()
            # The exact pause point: the ticking task skips paused songs, so
            # without this the position sits an interval behind for the whole pause.
            if self.current_song is not None:
                await self.store.heartbeat(self.current_song.position_secs, paused_at)
            # Still written this release: a rollback to the previous build
            # reads the wall-clock fields; they go one release after this.
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
        """Debounced trigger for pause()/resume(): refreshes the now-playing embed
        and the Activity presence. Debounced because nothing rate-limits how fast
        -pause/-resume can be invoked, and both targets are rate-limited Discord
        endpoints."""
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
        """Play `qobj` immediately; the interrupted song returns afterwards.

        Capture the current song's exact position (frame-counted, frozen if paused),
        front-insert [qobj, resume-entry(ts=position)], stop the current song; the
        loop's ordinary dequeue → -ss → play cycle does the rest. Both entries are
        persisted, so crash recovery mid-interjection works unchanged.

        Interjections STACK: interrupting an interjection parks it in front of the
        tails already waiting, so the queue unwinds LIFO. Depth is unbounded, and
        `ts` is absolute at every level, so a tail of a tail resumes at the position
        actually reached rather than at its own fragment's start.

        resume_paused decides whether a song interrupted while PAUSED comes back
        paused: True (-playnow) restores what it interrupted, False (-play) brings it
        back playing. No effect on a song that wasn't paused.

        None when there is no current song, or it ended during prefetch
        neutralization — the caller falls back to a plain front-enqueue. Residual
        race: a song ending naturally while put_front awaits still gets its resume
        entry, replaying its final seconds. The widest variant is the loop awaiting a
        still-running prefetch claimed before this ran, so put_front executes against
        a real in-flight head and takes its rebuild branch.
        """
        current = self.current_song
        if current is None:
            return None
        span = trace.get_current_span()
        span.set_attribute("discord.guild_id", str(self._guild.id))
        span.set_attribute("song.interjected_title", qobj.title or "")

        # A completed prefetch bypasses the queue and would play INSTEAD of the
        # front-inserted qobj — take it off the board first.
        await self._neutralize_prefetch()

        # Re-check after those awaits (cancellation can block up to yt-dlp's socket
        # timeout): if the song ended and the loop moved on, bail to the command's
        # fallback rather than build a resume entry for a finished song.
        if self.current_song is not current:
            return None

        was_paused = vc.is_paused()
        position = int(current.position_secs)
        resume: Optional[QueueObject] = None
        if current.webpage_url:
            # On the RAW position: the EOF cap below pulls it back by its margin,
            # which would mask "almost over".
            near_end = (
                current.duration_secs > 0
                and current.duration_secs - position < _MIN_RESUME_REMAINING_SECS
            )
            if current.duration_secs > 0:
                # EOF guard matching the crash-recovery cap: imprecise duration
                # metadata must not make FFmpeg seek past the end.
                position = min(
                    position,
                    max(0, current.duration_secs - _RESUME_EOF_MARGIN_SECS),
                )
            if not near_end:
                resume = QueueObject(
                    current.webpage_url,
                    current.title or "",
                    current.requester or self._require_requester(),
                    ts=position,
                    duration=current.duration_secs or None,
                    uploader=current.uploader,
                    thumbnail=current.thumbnail,
                    is_resume=True,
                    start_paused=was_paused and resume_paused,
                    # The tail is the same play, so it keeps the interrupted song's
                    # stamps — played_at included, which files the whole play under
                    # the moment its first fragment started.
                    analytics=current.analytics,
                    played_at=current.played_at,
                    # The tail writes the ONLY row for this play, and the
                    # classification is not recoverable from webpage_url — a Spotify
                    # link, a search and a pasted link all archive as youtube.com.
                    query_source=current.query_source,
                    # -remove matches on this: without it the parked tail is the
                    # one track a playlist link cannot take back out.
                    user_input=current.user_input,
                )

        # The interjection arrives carrying depth 0 from its own command dispatch
        # — it plays immediately by definition — while the tail keeps the
        # interrupted song's analytics, unknown ones included.
        items: list[QueueItem] = [qobj]
        if resume is not None:
            items.append(resume)
            # The song returns, so it is recorded once — when its tail finishes. A
            # song with no resume entry (nearly over) keeps its own entry, matching
            # -skip. One marker is enough at any depth: each interjection stops
            # exactly one song, and that song's iteration consumes the marker before
            # the next -playnow can finish resolving.
            #
            # Taken BEFORE the put_front await and unconditionally: everything from
            # the guard above to here is synchronous, so there is no window in which
            # a teardown or the loop's own iteration end could record the play too.
            self._skip_history_for = current
            # The tail inherits this fragment's NP card, but which message that is
            # is only settled at the fragment's iteration end — the confirmation
            # this command is about to send can still adopt a different host.
            self._pending_resume_tail = resume
        await self.queue.put_front(items)

        # Only if the song we measured is still playing: if the loop moved on, the
        # inserted entries play next anyway and stopping would kill the NEXT song.
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
        from the queue head.

        Claim-then-settle: _prefetch_task is nulled synchronously before any await,
        and the loop's matching read is also a synchronous read-and-null — exactly
        one of interject()/loop() consumes any given result.

        - running → cancel; its CancelledError handler returns the dequeued item to
          the pending front, exactly as bulk mutations rely on.
        - completed → rebuild an equivalent QueueObject, return it to the front, kill
          its FFmpeg subprocess. Neither the deque slot nor the mirror moved, so
          requeue_front() rewrites the slot with the resolved form. The rebuild must
          carry EVERY field — a dropped one is silently gone from the queue entry.
        - completed-with-None → the prefetch already retired its own dequeue.
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
        # CancelledError is deliberate: this reads a *done* task's result, where a
        # cancelled prefetch surfaces as CancelledError and means "no song" like any
        # failure — not this coroutine swallowing its own cancellation. Listed
        # explicitly because it is not an Exception subclass; the bare tuple form is
        # PEP 758 (see guild_state._b_float).
        except asyncio.CancelledError, Exception:
            song = None
        if song is None:
            return
        # Carry the -ss offset, every -playnow flag and the analytics through the
        # rebuild: dropping them restarts a neutralized resume entry from 0:00
        # (unpaused, unannounced), loses a prefetched song's ?t= offset, and zeroes
        # the ask this play was queued against, which nothing re-mints.
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
        the loop's start path, like _announce_resume and for the same reason: at
        YTDL construction, where it used to live, a prefetched song announces itself
        while the previous one is still playing."""
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
        start path — the same path and the same reason as _announce_start_offset.
        Plain channel send, not send_with_np: this song's NP host is not sent yet, so
        send_with_np would adopt the notice only for _send_now_playing to immediately
        retire it."""
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
        Call BEFORE vc.stop(); the loop clears it at each vc.play(). A stop we initiate
        ends the player thread without another source.read(), so `after` gets
        error=None — indistinguishable, on frame count alone, from a dead stream."""
        self._stopped_deliberately = True

    async def _drop_unplayable_stream_cache(self, song: YTDL) -> None:
        """Drop the cached stream URL of a song that ended with no frame, no error, and
        no deliberate stop.

        A BACKSTOP, not the main path. discord.py reports a failing ffmpeg through
        `after` (read() -> _check_process_returncode), which is the `stream_failed` path
        _handle_dead_stream owns. What reaches HERE is the window that check declines to
        judge: it returns early while `self._process.poll()` is still None, so a child
        that closed stdout but has not been reaped loses its error.

        Cache only, on purpose: being wrong costs one re-extraction, while widening
        `stream_failed` on the same evidence would suppress a real history entry.

        A zero-frame song reaching the loop's iteration end IS still recorded, at
        played_secs=0, while `claim_current_song_for_history` refuses one. That lone
        disagreement is about FRAMES, not about teardown: a song a teardown abandons
        mid-play is recorded at its true position whenever it produced audio. The two
        rules differ on purpose — do not "align" them without deciding which record
        the archive is supposed to hold."""
        if self.store is None or not song.webpage_url:
            return
        dropped = await invalidate_stream_cache(self.store.redis, song.webpage_url)
        # Report the outcome, not the intent: the common case here has nothing cached,
        # and a deletion announced unconditionally reads as a YouTube early warning.
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
        """Recover from a song whose stream never opened. yt_stream() probes before
        handing the URL to ffmpeg, so reaching here means it was revoked between the
        probe and the first read. Drop the cached URL (else the next -play replays
        the dead one) and say so in the channel — a failure ffmpeg swallows is
        invisible to the listener."""
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
        """Send a dedicated NP host message (its embeds are only the block) and adopt
        it, retiring whatever hosted the block before. None when there is no live
        song, or the song changed while the send was in flight (the stale message is
        deleted instead of adopted)."""
        song = self.current_song
        block = self.np_embed_block(now_playing=now_playing)
        if not block:
            return None
        message = await self._channel.send(embeds=block)
        if not self._adopt_np_host_if_current(message, [], song, dedicated=True):
            return None
        return message

    async def repin_now_playing(self) -> bool:
        """-now: re-host the NP block at the bottom as a fresh dedicated message.
        Does not touch _progress_task — the updater follows the host pointer and
        picks up the new message next tick. False when no song is live (including one
        that ended mid-send) so the command can respond another way."""
        return await self._send_np_host_message() is not None

    async def rehost_np_after_resume(self) -> None:
        """-resume: when a command response hosts the block (typically the -pause
        confirmation), re-host onto a fresh dedicated message and strip-retire the
        old one, so "⏸️ Paused at…" becomes history instead of being re-rendered
        beneath a live bar every tick. A dedicated host is left alone."""
        if self._np_host_message is None or self._np_host_dedicated:
            return
        await self._send_np_host_message()

    async def _send_now_playing(self, song: YTDL) -> None:
        # Release before the send, not after a failure, so a partial send never
        # leaves the host pointing at the *previous* song's message — a stale host
        # would let a later mark_paused()/mark_resumed() on the new song overwrite
        # the old song's already-sent embed.
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
        """Rebuild the host's embeds — a fresh NP block, then its cached own embeds —
        and push one edit. Shared by the periodic tick, the debounced pause/resume
        refresh and the song-end finalize. False when the message no longer exists,
        so callers can release the host; finalize ignores it.

        Rebuilding and re-decorating the block each tick is what keeps the debug
        footer's metrics moving with the bar. `own_embeds` is excluded: cached, and
        already decorated at send time."""
        try:
            embed = self._build_now_playing_embed(
                song, position_override=position_override
            )
            next_up = self._build_next_up_embed()
            block = [embed] + ([next_up] if next_up else [])
            self._decorate_for_debug(block)
            embeds = block + own_embeds
            # Discord's per-message cap: an attach accepted at the cap can overflow
            # here if a next-up embed appears later. Drop the own-embeds tail, never
            # the block (parity with MusicContext.send's guard; unreachable today).
            embeds = embeds[:10]
            # Skip the PATCH when the payload is identical to the last one pushed
            # to this host. The bar changes ~10 times in a 4-minute song while the
            # 3s tick fires ~80, so most edits carry nothing new.
            # See docs/ARCHITECTURE.md#now-playing-host-model
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
        """Push one embed edit outside the periodic tick, for the debounced
        pause/resume refresh. Holds the edit lock and re-reads the host inside it:
        an edit landing after a retire's strip would resurrect the block."""
        song = self.current_song
        if song is None:
            return
        async with self._np_edit_lock:
            host = self._np_host_message
            if host is None:
                return
            if not await self._push_np_edit(song, host, self._np_host_own_embeds):
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
    ) -> None:
        """One last embed edit once a song has stopped, so the bar lands on its true
        final state rather than wherever the last tick fell (up to
        NOW_PLAYING_UPDATE_INTERVAL_SECS stale). completed=True renders the bar full;
        completed=False renders where it actually stopped, since a 100% bar for a
        skipped or interjected song would be a false record.

        song/message/own_embeds are captured by the CALLER, not read off self: both
        may already point at the next song by the time this fire-and-forget task
        runs. loop() released the host before this fires, so no tick or retire can
        START against the message — but a debounce-spawned _edit_now_playing_once
        that captured the host before the release can still have a PATCH in flight.
        It holds _np_edit_lock across its edit, so taking the lock here orders this
        write after it (last write wins).
        """
        if song.duration_secs <= 0:
            return  # no bar was ever shown for this song — nothing to finalize
        async with self._np_edit_lock:
            await self._push_np_edit(
                song,
                message,
                own_embeds,
                # None → falls back to the live position_secs, frozen at the stop
                # point (and, for a paused song, at the pause point).
                position_override=song.duration_secs if completed else None,
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
        self._spawn_background(
            self._finalize_now_playing(song, message, own_embeds, completed=completed)
        )

    async def _progress_updater(self, song: YTDL) -> None:
        interval = config.NOW_PLAYING_UPDATE_INTERVAL_SECS
        try:
            while True:
                await asyncio.sleep(interval)
                vc = self._guild.voice_client
                if not isinstance(vc, discord.VoiceClient) or vc.source is not song:
                    # song changed under us; loop() owns cancellation, but guard
                    # defensively
                    return
                if vc.is_paused():
                    continue  # frozen — mark_resumed() fires a debounced edit
                async with self._np_edit_lock:
                    host = self._np_host_message  # re-read inside the lock: a host
                    # swap during this tick's sleep must not leave the edit
                    # targeting the old, about-to-be-stripped message
                    if host is None:
                        continue  # dormant: no visible NP until re-hosted
                    if not await self._push_np_edit(
                        song, host, self._np_host_own_embeds
                    ):
                        # Host deleted by a user — go dormant rather than die; the
                        # next command response (or -now) re-hosts. Adopt is
                        # lock-free, so only release OUR host or a newly swapped-in
                        # one would be orphaned.
                        if self._np_host_message is host:
                            self._release_np_host()
        except asyncio.CancelledError:
            raise

    async def _heartbeat_updater(self, song: YTDL) -> None:
        """Record the playback position to Redis on a fixed cadence.

        Separate from _progress_updater because that task is display-gated: it
        skips songs too short for a bar and goes dormant when the host message is
        gone. A song with no visible bar must still be recoverable.
        """
        while True:
            await asyncio.sleep(config.HEARTBEAT_INTERVAL_SECS)
            vc = self._guild.voice_client
            if not isinstance(vc, discord.VoiceClient) or vc.source is not song:
                return  # song changed under us; loop() owns cancellation
            if vc.is_paused():
                # Frames are frozen, so the position is not moving and pause()
                # already recorded the exact point. Writing here would only
                # rewrite the same value once per interval, forever.
                continue
            if self.store is not None:
                try:
                    await self.store.heartbeat(song.position_secs, time.time())
                except Exception as e:
                    # Redis failures are already swallowed by @_guild_op, so anything
                    # here is a defect that will recur every tick. Stop, but say so:
                    # cancel_task never awaits a task that ended on its own, so the
                    # exception would otherwise surface at GC with no guild attached,
                    # and recovery would quietly fall back to the seeded position.
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
        Runs only when an item is already queued, and accounts for its own dequeue on
        every non-success path: cancellation gives the claim back with the item (a
        bulk mutation is about to take it with the rest), failure settles it on both
        legs (leaving the mirror entry would make the next commit retire the wrong
        one). On success the claim stays open and loop()'s commit settles it."""
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
            # executor job cannot be interrupted, so every bulk mutation would wait
            # on it. The play-time resolve decides instead.
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
        # Wait for _restore_state() to populate self.queue before dequeuing — see
        # its docstring for the race this prevents (an erroneous pop_queue() for a
        # crash-recovered song that was never on the Redis list).
        await self._restore_complete.wait()
        # Queue populated; wait for a voice connection before playing any of it. The
        # timeout is not optional: a player blocked here is not blocked in
        # queue_get(), so the 300s idle-disconnect below can never fire, and a player
        # that never connects would leak its mps entry and task forever.
        while True:
            try:
                async with async_timeout.timeout(_PLAYBACK_GATE_TIMEOUT):
                    await self._playback_gate.wait()
                break
            except asyncio.TimeoutError:
                if self._playback_holds or self._playback_gate.is_set():
                    # A hold means a command is mid-join: tearing down would pop this
                    # player from mps while it still drives it. Every hold is released
                    # by an `async with`, raise or not.
                    # The gate check is NOT redundant — this handler runs a tick after
                    # the timer fires, so a release can land in between and leave holds
                    # 0 with the gate already open. Re-waiting is free.
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
            # True while this iteration holds a claim the commit has not settled,
            # so the outer handler can settle it when a failure lands in that
            # window. Cleared at the commit, not at song end — the release it
            # guards deletes an item.
            claim_outstanding = False
            # Whether that claim has an entry on the Redis list. Carried rather
            # than re-derived from `source`, which the prefetched branch leaves
            # None — and None defaults to popping. Written beside the flag above,
            # so it never describes a different item than the one claimed.
            claim_persisted = True
            # Each iteration spans a full song (3–5 min). Expected — the span stays
            # open across play_next.wait().
            with _tracer.start_as_current_span(
                "player.loop.iteration",
                attributes={"discord.guild_id": str(self._guild.id)},
            ) as span:
                try:
                    queue_was_cleared = self.queue.consume_cleared_flag()
                    prefetch_used = prefetched_song is not None
                    span.set_attribute("prefetch.used", prefetch_used)
                    if prefetched_song is not None and queue_was_cleared:
                        # Cleared while _prefetch_next_song ran: clear() reset the
                        # cursor, so the prefetch's claim is already settled — only
                        # the FFmpeg subprocess is left to reap, and discarding the
                        # result without cleanup() would leak it.
                        prefetched_song.cleanup()
                        prefetched_song = None
                    # Captured where each path takes its item, and handed back to
                    # try_commit_dequeue() below: a clear() in between voids this
                    # dequeue even if a put() has since refilled the display.
                    commit_generation = self.queue.generation
                    if prefetched_song is not None:
                        self.current_song = prefetched_song
                        prefetched_song = None
                        claim_outstanding = True  # the prefetch's get_nowait() is ours
                        # Read off the song: a prefetch CAN claim a persisted=False
                        # item — a cold-start `-play` front-inserts at cursor 0,
                        # AHEAD of the crash-recovered head, so the prefetch behind
                        # it takes that head. Popping for one that was never on the
                        # list deletes the next real entry.
                        claim_persisted = self.current_song.persisted
                        # One read for both settle paths — the start transaction's
                        # LPOP here, the outer handler's below. `source` stays None
                        # because a YTDL is not a QueueItem.
                        should_pop_queue = claim_persisted
                        source = None
                    else:
                        source = None
                        try:
                            async with async_timeout.timeout(300):
                                source = await self.queue_get()
                                claim_outstanding = True
                                # Taken beside the claim it describes. Reading it
                                # before the resolve is safe: a YTSource is
                                # persisted and yt_source() builds a QueueObject
                                # that defaults the same way.
                                claim_persisted = is_persisted(source)
                                # Re-read: queue_get() can block, and a clear()
                                # during that wait belongs to the queue this item
                                # came from, not to the one we sampled above.
                                commit_generation = self.queue.generation
                                source = await self._resolve_source(source)
                        except asyncio.TimeoutError:
                            log.warning("Queue timed out, disconnecting")
                            asyncio.create_task(self.stop())
                            return
                        except Exception:
                            # _resolve_source() raised after queue_get() already
                            # dequeued `source` — balance that dequeue as the
                            # "current_song is None" branch does, then re-raise into
                            # the outer handler's logging/error-embed path.
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

                    if not await self.queue.try_commit_dequeue(commit_generation):
                        # Cleared while this song resolved (e.g. inside yt_stream).
                        # Discard without playing: the clear() that refused this
                        # commit reset the cursor, so the claim is settled, and
                        # cleanup() terminates the FFmpeg subprocess yt_stream
                        # already spawned.
                        claim_outstanding = False
                        self.current_song.cleanup()
                        self.current_song = None
                        continue

                    # The commit settled the claim. Cleared here, not at song end:
                    # the flag guards a release, and left standing across the song
                    # that release would delete whatever sits at the head by then —
                    # the next song, once the prefetch below claims it.
                    claim_outstanding = False

                    # Safe to assert rather than await readiness: the gate above
                    # opens only once a voice connection is established
                    # (channel.connect() awaits the full handshake), so loop() cannot
                    # reach vc.play() mid-handshake.
                    vc = self._guild.voice_client
                    assert isinstance(vc, discord.VoiceClient)
                    assert self.current_song is not None
                    # Local binding: pyright's narrowing doesn't survive the awaits
                    # below, and it keeps every write in this iteration on the same
                    # song even if current_song is reassigned.
                    song = self.current_song

                    # Written from the player thread, read after play_next.wait();
                    # call_soon_threadsafe orders the write before the wait returns.
                    play_error: list[Optional[Exception]] = [None]

                    def _after_play(
                        error: Optional[Exception], _title: str = song.title or ""
                    ) -> None:
                        # discord.py hands ffmpeg's failure here and nowhere else —
                        # a `lambda _:` would drop it, making a stream that never
                        # opened indistinguishable from a song that ended. A
                        # deliberate vc.stop() arrives as error=None.
                        if error is not None:
                            play_error[0] = error
                            log.error(f"playback error for {_title}: {error}")
                        self.bot.loop.call_soon_threadsafe(self.play_next.set)

                    self._stopped_deliberately = False
                    vc.play(song, after=_after_play)
                    if song.start_paused:
                        # Park the player thread SYNCHRONOUSLY, before any await, so
                        # a song returning paused leaks a frame or two rather than a
                        # Redis round-trip of audio. Idempotent with the full pause()
                        # below, which runs after the start transaction so its
                        # pause_start_epoch survives that transaction's HDEL.
                        vc.pause()
                    play_start = time.time()  # capture immediately before any awaits
                    # Stamped once and inherited by every later fragment (a resume
                    # tail arrives carrying it). Before the state write below, or
                    # the parked entry persists 0.0 and a crash recovers no start.
                    song.played_at = song.played_at or play_start

                    # Mirror the now-playing song to Redis. should_pop_queue=True →
                    # one MULTI/EXEC atomically LPOPs the queue and writes every state
                    # field plus the display snapshot, closing the at-most-once
                    # window. A crash-recovered "current song" was never on the Redis
                    # list, so only state is written — an LPOP would drop a queued one.
                    if self.store is not None:
                        # The -ss offset twice: backdated into the epoch the legacy
                        # fallback extrapolates from, and passed as the seed the
                        # heartbeat has not written yet. Without it a `?t=` song
                        # crashing inside the first interval resumes at 0:00.
                        backdated_start = play_start - song.start_offset
                        current = SongQueueEntry.from_song(song)
                        now_playing = NowPlayingData.from_song(song)
                        if should_pop_queue:
                            await self.store.pop_queue_and_start_song(
                                current,
                                backdated_start,
                                now_playing=now_playing,
                                start_offset=song.start_offset,
                            )
                        else:
                            await self.store.set_current_song_state(
                                current,
                                backdated_start,
                                now_playing=now_playing,
                                start_offset=song.start_offset,
                            )
                        # In this block because the store is the ticker's only
                        # writer: a Redis-less guild would otherwise tick for the
                        # whole song to reach a no-op. After the seed above, so the
                        # first tick cannot race it.
                        self._heartbeat_task = asyncio.create_task(
                            self._heartbeat_updater(song)
                        )

                    if song.start_paused:
                        # Returns parked where -playnow interrupted it (the player
                        # thread was already paused synchronously at vc.play). The
                        # full pause() runs here so the Redis pause epochs and the
                        # debounced embed/Activity refresh all engage.
                        await self.pause(vc)
                    if song.is_resume:
                        await self._announce_resume(song)
                    elif song.start_offset > 0:
                        await self._announce_start_offset(song)

                    await self.update_activity(song)
                    await self._send_now_playing(song)
                    # Strictly after the new card is up, so the bar is never absent
                    # from the channel. The host check is what makes that true:
                    # _send_now_playing releases the host before sending and
                    # swallows every failure, so a 403 leaves no card and disposing
                    # would delete the only bar in the channel.
                    if song.is_resume and self._np_host_message is not None:
                        self._spawn_background(self._dispose_previous_np_card(song))

                    self._prefetch_task = asyncio.create_task(
                        self._prefetch_next_song()
                    )

                    await self.play_next.wait()

                    # Zero frames AND an ffmpeg error means the stream never opened
                    # (typically a 403 on a revoked URL). Both conditions matter:
                    # zero frames alone also describes a song parked paused by
                    # -playnow or stopped the instant it started (vc.stop() reports
                    # no error), and an error alone also describes a mid-song death
                    # that delivered real audio and earns its history entry.
                    # A THIRD case joins these below: zero frames, no error, and no
                    # deliberate stop. discord.py surfaces a failing ffmpeg as an
                    # error, so that case is the narrow window where the child had not
                    # been reaped yet — see _drop_unplayable_stream_cache.
                    stream_failed = (
                        not song.produced_audio and play_error[0] is not None
                    )
                    span.set_attribute("song.stream_failed", stream_failed)

                    # Must fully retire before the next iteration's
                    # _send_now_playing(), or an in-flight edit for this song could
                    # resolve concurrently with the new message being sent.
                    await self._cancel_progress_task()
                    await self._cancel_pause_debounce()

                    # Song has ended (naturally or via -skip): capture the host,
                    # release it (the finished bar stays behind as a record, so the
                    # next song's adopt retires nothing), then fire one last edit so
                    # the bar shows its true final state instead of the last tick's.
                    finished_host = self._np_host_message
                    finished_own = self._np_host_own_embeds
                    finished_dedicated = self._np_host_dedicated
                    self._release_np_host()
                    # The id that lands in play_history.message_id, recorded on the
                    # span because 0 is ambiguous in the stored row (send failed,
                    # host deleted mid-song, pre-message_id build, or backfill all
                    # read as 0) and nothing else surfaces it.
                    span.set_attribute(
                        "song.np_host_id",
                        str(finished_host.id) if finished_host is not None else "",
                    )
                    if finished_host is not None:
                        if stream_failed:
                            # A completed bar is truthful only for a song that
                            # played. This one delivered nothing, so dispose of the
                            # block (as retire_np_host_on_stop does) rather than
                            # finalize it to 100% right above the failure notice.
                            self._spawn_background(
                                self._retire_np_host(
                                    finished_host, finished_own, finished_dedicated
                                )
                            )
                        elif self.current_song is not None:
                            # completed=_reached_end(): a skipped, interjected or
                            # dead song finalizes at its true position, never 100%.
                            # The edit fires either way — the 3s tick would leave
                            # the bar frozen a tick before the interruption.
                            self._fire_finalize_now_playing(
                                self.current_song,
                                finished_host,
                                finished_own,
                                completed=_reached_end(self.current_song),
                            )

                    # Ordering: stop advertising this song as current before the
                    # prefetch await below, which can sit for seconds on a yt-dlp
                    # extraction. Everything above is synchronous, so this is the
                    # first point another coroutine can interleave, and
                    # MusicContext.send's attach gate is `current_song is not None`:
                    # left set, a command response in the home channel prepends a
                    # block for an ended song and adopts ITSELF as host, which the
                    # next _send_now_playing() RELEASES without retiring — orphaning
                    # a frozen bar nothing can clean up. `song` is this iteration's
                    # copy, and what the history entry below is built from.
                    self.current_song = None
                    self.play_message = None  # -now must not serve a finished song
                    # The play is no longer current but its row is not written yet,
                    # and the prefetch await below is long enough for a teardown to
                    # land in between. Cleared once this iteration settles it.
                    self._ended_song = song
                    # Below current_song = None, unlike the other task cancels: this
                    # task always exists, so awaiting it yields, and the block above
                    # must stay synchronous for the reason stated there.
                    await self._cancel_heartbeat_task()

                    # Claim-then-await: interject() may have neutralized (and nulled)
                    # the task while this iteration sat in play_next.wait(). Both
                    # sides read-and-null synchronously, so exactly one consumer sees
                    # any given result; a task interject() cancelled resolves to None.
                    prefetch_task = self._prefetch_task
                    self._prefetch_task = None
                    prefetched_song = None
                    if prefetch_task is not None:
                        try:
                            prefetched_song = await prefetch_task
                        except asyncio.CancelledError:
                            prefetched_song = None

                    # interject() stopped this song with a resume entry pending —
                    # history records it when the tail ends. Identity match, and the
                    # marker clears either way: a marker left for a song that ended
                    # during interject()'s awaits must not eat this song's entry.
                    # stream_failed → never heard, so never recorded.
                    skip_history = self._skip_history_for is song
                    self._skip_history_for = None
                    # Same identity, same clear-either-way rule: hand the tail the
                    # card this fragment is leaving frozen. Late-bound because THIS
                    # is where the host settles — an id taken at interjection time
                    # can name a message the confirmation's own adopt already
                    # retired. Mutating the queued object is safe: single event
                    # loop, and the queue legs hold this reference.
                    #
                    # Not mirrored to Redis here, but the wire fields do not stay
                    # zero: any later rebuild_queue re-serializes this object
                    # through SongQueueEntry.from_queue_object, so -shuffle, a
                    # matching -remove or an in-flight-head put_front persist the
                    # live ids — which is why _dispose_previous_np_card guards them
                    # as hostile input.
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
                    # and declined to record. from_song's played_secs is
                    # start_offset-based, so it records what was actually heard.
                    heard_before = song.is_resume and song.start_offset > 0
                    if not skip_history and (not stream_failed or heard_before):
                        await self.history.add(
                            HistoryEntry.from_song(
                                song,
                                guild_id=self._guild.id,
                                # The host captured at song end, not
                                # _np_host_message, which _release_np_host() nulled
                                # above. Both ids come off that one message — never
                                # the home channel, which commands reassign.
                                # 0 = nothing hosted it.
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

                    # Written, or deliberately not — either way a teardown from here
                    # has nothing left to claim.
                    self._ended_song = None

                    if self.store is not None:
                        await self.store.clear_song_end_state()

                    await self.update_activity(None)

                    # Deliberately last: current_song is already cleared, so the
                    # notice goes out alone rather than re-hosting an NP block for a
                    # song that never played.
                    if stream_failed:
                        await self._handle_dead_stream(song)
                    elif (
                        not song.produced_audio
                        and not self._stopped_deliberately
                        and not song.start_paused
                    ):
                        # Zero frames, no error, nobody stopped it. start_paused is
                        # the one other ending that looks identical and says nothing
                        # about the URL: a song parked at vc.pause() and torn down
                        # before it ever played.
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
                    # and the commit — every other path settles its own.
                    # finish_failed_dequeue, not release: release drops the item
                    # from memory alone, leaving its mirror entry for the next LPOP
                    # to retire in its place. persisted= travels because `source` is
                    # None for a prefetched claim, which would default to popping.
                    if claim_outstanding:
                        await self.queue.finish_failed_dequeue(
                            source,
                            context="unhandled loop error",
                            persisted=claim_persisted,
                        )
                    # Awaited, matching _cancel_prefetch: the prefetch returns its
                    # item through requeue_front, and claims settle by POSITION, so
                    # letting that land after this handler's own settle swaps the
                    # two songs.
                    await cancel_task(self._prefetch_task)
                    self._prefetch_task = None
                    await self._cancel_progress_task()
                    await self._cancel_heartbeat_task()
                    await self._cancel_pause_debounce()
                    # No finalize for a song that errored — just release the host so
                    # the next song starts clean.
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
                        # Inside the try: this is the loop's own except block, so a
                        # raise here escapes both handlers and kills the playback
                        # task. Hand-built rather than send_embed so the debug footer
                        # lands before the send; skip_trace dedups the trace id.
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
