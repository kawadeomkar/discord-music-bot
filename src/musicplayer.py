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
from src.guild_history import GuildHistory
from src.guild_queue import GuildQueue, ShuffleOutcome, is_persisted
from src.guild_state import HistoryEntry, NowPlayingData, SongQueueEntry
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
    send_embed,
    trace_footer,
    truncate,
    truncate_embed_title,
    get_logger,
)
from src.youtube import YTDL, QueueObject, invalidate_stream_cache

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


# HACK: Every guild's ETAs are hardcoded to Pacific time.
# queue_embed()'s "Est. playing at" and the now-playing "Estimated finish" render in
# US/Pacific for everyone, quoting users elsewhere a clock time that is not theirs;
# the "PST" suffix is all that stops it misleading. Fix: a per-guild timezone, or
# Discord relative timestamps (<t:epoch:R>), which render in each viewer's locale.
_PST = ZoneInfo("America/Los_Angeles")


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
    # None → no resume entry (the interrupted song was itself an interjection,
    # was nearly finished, or had no webpage_url to rebuild from).
    resume_position: Optional[int]
    was_paused: bool  # the OBSERVED state when it was interrupted
    replaced: bool  # the interrupted song was itself a -playnow interjection
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
    build_resume_notice_embed() so they can't disagree."""
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


def _fmt_finish_time(duration_secs: int) -> str:
    """Clock time `duration_secs` from now. No uncertainty prefix: a song's own
    remaining duration, unlike a queued song's ETA, is known once it's playing."""
    finish_dt = datetime.datetime.now(tz=_PST) + datetime.timedelta(
        seconds=duration_secs
    )
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
        "_player",
        "_prefetch_task",
        "store",
        "_restore_task",
        "_restore_complete",
        "_playback_gate",
        "_playback_holds",
        "_background_tasks",
        "_progress_task",
        "_np_host_message",
        "_np_host_own_embeds",
        "_np_host_dedicated",
        "_np_edit_lock",
        "_pause_debounce_task",
        "_skip_history_for",
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
    _prefetch_task: Optional[asyncio.Task]
    store: Optional[GuildRedisStore]
    _restore_task: Optional[asyncio.Task]
    _restore_complete: asyncio.Event
    _playback_gate: asyncio.Event
    _playback_holds: int
    _background_tasks: set[asyncio.Task[Any]]
    _progress_task: Optional[asyncio.Task]
    _np_host_message: Optional[discord.Message]
    _np_host_own_embeds: list[discord.Embed]
    _np_host_dedicated: bool
    _np_edit_lock: asyncio.Lock
    _pause_debounce_task: Optional[asyncio.Task]
    _skip_history_for: Optional[YTDL]
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

        self.store = (
            GuildRedisStore(redis, self._guild.id) if redis is not None else None
        )
        # All queue state (asyncio queue, display order, Redis mirror, bulk mutex,
        # cleared-flag) lives behind this one object — see guild_queue.py.
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
        self._prefetch_task: Optional[asyncio.Task] = None
        self._restore_task: Optional[asyncio.Task] = None
        self._restore_complete = asyncio.Event()
        # Restore and *play* are separate concerns: the gate stays shut until a
        # command establishes a voice connection, so a player built by a command that
        # never connects (cog_before_invoke runs before validate_commands, so even a
        # rejected command builds one) cannot walk the queue and discard it.
        self._playback_gate = asyncio.Event()
        # >0 while an in-flight command owns the opening — -play holds the gate
        # across the join it triggers, so the restored head can't start before the
        # requested song is inserted in front of it.
        self._playback_holds = 0
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._progress_task: Optional[asyncio.Task] = None
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
        _restore_guild connects before calling start(), so recovery resumes from the
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

    async def wait_for_restore(self) -> None:
        """Block until _restore_state() has finished (or failed). Inserting before
        restore has read its snapshot double-queues: put_front() LPUSHes the mirror,
        while restore_entries() is in-memory only precisely because its entries are
        already on that list."""
        await self._restore_complete.wait()

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
        return datetime.datetime.now(tz=_PST), EtaWalk(cumulative_secs, uncertain)

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
            title = truncate(item.title, _NEXT_UP_TITLE_MAX) or "Unknown"
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
            search = (item.ytsearch or item.url or "?").removeprefix("ytsearch:")
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
        while the display head is still the restored one.

        The two restores landing here know different things:
        * A crash re-queues the mid-play song as the display head (persisted=False,
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

        left_off = self._resume_left_off_field()
        if left_off is not None:
            embed.add_field(name=left_off[0], value=left_off[1], inline=True)

        embed.add_field(name="Queued", value=f"**{count}** {songs}", inline=True)
        total_secs, partial = _queue_runtime(items)
        if total_secs > 0:
            prefix = "~" if partial else ""
            embed.add_field(
                name="Runtime",
                value=f"{prefix}{_fmt_total_duration(total_secs)}",
                inline=True,
            )
        return embed

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
                        log.warning(
                            f"State restore aborted for guild {self._guild.id}: "
                            f"Redis unavailable"
                        )
                        return
                    guild_state = snapshot.state

                    # Only when a value was actually stored: an unconditional
                    # assign would clobber a concurrent -volume with the default.
                    if guild_state.volume is not None:
                        self.volume = guild_state.volume

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

    # ── Queue operations ──────────────────────────────────────────────────────

    def _stamp_enqueue(self, items: list[QueueItem], *, base: int) -> list[QueueItem]:
        """Stamp queued_at/queue_position on items entering a queue for the first
        time, `base` songs behind the front.

        Stamp-once: a non-zero queued_at is left alone. No path re-enqueues an
        already-stamped item through here today — requeue_front, restore and the
        neutralized prefetch all bypass it — so the guard is what keeps a future
        one from silently renumbering a song against a depth it never waited.

        QueueObject is mutated in place, as prefetch already does; YTSource is
        frozen, so it is replaced — callers must use the returned list.
        """
        now = time.time()
        stamped: list[QueueItem] = []
        for offset, item in enumerate(items):
            if item.queued_at:
                stamped.append(item)
            elif isinstance(item, YTSource):
                stamped.append(
                    replace(item, queued_at=now, queue_position=base + offset)
                )
            else:
                item.queued_at = now
                item.queue_position = base + offset
                stamped.append(item)
        return stamped

    def _stamp_at_depth(
        self, items: Sequence[QueueItem], queued_ahead: int
    ) -> list[QueueItem]:
        """GuildQueue stamp hook: songs a new arrival waits behind are the entries
        already ahead of the insert point plus the one playing. `queued_ahead`
        comes from the queue under its bulk-mutation mutex, so a concurrent
        enqueue cannot shift the depth between reading it and the insert landing.

        The live song is NOT counted when its resume tail is already queued: after
        an interjection that entry is the same play as current_song, and counting
        both puts a new arrival one song too deep."""
        base = queued_ahead
        current = self.current_song
        if current is not None and not self.queue.has_resume_tail(current.webpage_url):
            base += 1
        return self._stamp_enqueue(list(items), base=base)

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
        items = await self.queue.put(
            items, batch=not prefetch, stamp=self._stamp_at_depth
        )
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
        # Front insertion waits only on the live song and an in-flight head — the
        # persisted entries it jumps ahead of are behind it, not in front.
        items = await self.queue.put_front(items, stamp=self._stamp_at_depth)
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

    async def queue_clear(self) -> list[str]:
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
        # Cancel before shuffle()'s too-few guard: a prefetch holding a dequeued
        # item must be accounted for even when the shuffle is a no-op.
        await self._cancel_prefetch()
        outcome = await self.queue.shuffle()
        if outcome is ShuffleOutcome.TOO_FEW_SONGS:
            return "There must be at least 3 songs to shuffle the queue"
        return "Shuffled!"

    async def queue_remove(self, url: str) -> list[int]:
        """Remove all queued items matching url (webpage_url for QueueObject, url for
        YTSource). Returns the 1-indexed queue positions removed.
        """
        await self._cancel_prefetch()
        return await self.queue.remove(url)

    # ── Embed building ────────────────────────────────────────────────────────

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
            requester_line += f"  ·  Estimated finish: {_fmt_finish_time(remaining)}"
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
        built this song's embed supply it instead of building an identical one."""
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
        """Remove the NP block from a message that is no longer the host. Holds the
        edit lock so an in-flight tick edit finishes first: concurrent PATCHes
        resolve last-write-wins server-side, and a tick landing after the strip
        would resurrect the block on the retired host with nothing to clean it up."""
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
        """Clear host state without retiring the message. Used at song end: the
        completed bar stays in the channel as a record, and the next song's adopt
        sees no old host to retire."""
        self._np_host_message = None
        self._np_host_own_embeds = []
        self._np_host_dedicated = False

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
        persisted, so crash recovery mid-interjection works unchanged. Interrupting
        an interjection builds NO resume entry — the ORIGINAL song's entry, still at
        the queue front, is untouched (replace semantics).

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
        replaced = current.interjected
        position = int(current.position_secs)
        resume: Optional[QueueObject] = None
        if not replaced and current.webpage_url:
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
                    # The tail is the same play, so it keeps the interrupted
                    # song's stamps rather than earning new ones here.
                    queued_at=current.queued_at,
                    queue_position=current.queue_position,
                )

        # Stamped alone, at position 0: it plays immediately by definition, while
        # the tail keeps the interrupted song's stamps — unknown ones included.
        items = self._stamp_enqueue([qobj], base=0)
        if resume is not None:
            items.append(resume)
        await self.queue.put_front(items)

        # Only if the song we measured is still playing: if the loop moved on, the
        # inserted entries play next anyway and stopping would kill the NEXT song.
        if self.current_song is current:
            if resume is not None:
                # The song returns — record it in history once, when its tail
                # finishes. A replaced interjection (no resume) keeps its entry,
                # matching -skip.
                self._skip_history_for = current
            vc.stop()

        span.set_attribute("interject.replaced", replaced)
        span.set_attribute("interject.resume_position", position if resume else -1)
        return InterjectOutcome(
            interrupted_title=current.title or "Unknown",
            resume_position=position if resume is not None else None,
            was_paused=was_paused,
            replaced=replaced,
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
          its FFmpeg subprocess. The display/Redis legs never moved, so the rebuilt
          item re-aligns all three (requeue_front's "resolved form" tolerance).
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
        # Carry the -ss offset, every -playnow flag and the enqueue stamps through
        # the rebuild: dropping them makes a neutralized resume entry restart from
        # 0:00 (unpaused, unannounced), loses an ordinary prefetched song's ?t=
        # offset, and would let the rebuild be restamped as freshly queued.
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
            queued_at=song.queued_at,
            queue_position=song.queue_position,
        )
        self.queue.requeue_front(rebuilt)
        song.cleanup()

    async def _announce_resume(self, song: YTDL) -> None:
        """One-line notice when an interrupted song returns, sent from the loop's
        start path because yt_stream suppresses its "Starting song at Xs" notice for
        resume entries. Plain channel send, not send_with_np: this song's NP host is
        not sent yet, so send_with_np would adopt the notice only for
        _send_now_playing to immediately retire it."""
        position = fmt_duration(int(song.position_secs))
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

    async def _resolve_source(self, source: QueueItem) -> QueueObject:
        if isinstance(source, YTSource):
            qobj = await YTDL.yt_source(
                self._require_requester(),
                source.ytsearch or "",
                redis=self.store.redis if self.store is not None else None,
            )
            # yt_source builds the QueueObject, so the search's enqueue stamps are
            # copied onto it here rather than threaded through its signature.
            qobj.queued_at = source.queued_at
            qobj.queue_position = source.queue_position
            return qobj
        return source

    async def _stream_source(self, source: QueueObject) -> Optional[YTDL]:
        self._last_stream_error = None
        try:
            return await YTDL.yt_stream(
                source,
                self._channel,
                volume=self.volume,
                redis=self.store.redis if self.store is not None else None,
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
        try:
            await self._channel.send(
                embed=notice_embed(
                    f"Could not play **{song.title}** — YouTube refused the audio "
                    "stream. Queue it again to retry.",
                    discord.Color.red(),
                )
            )
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
        so callers can release the host; finalize ignores it."""
        try:
            embed = self._build_now_playing_embed(
                song, position_override=position_override
            )
            next_up = self._build_next_up_embed()
            embeds = [embed] + ([next_up] if next_up else []) + own_embeds
            # Discord's per-message cap: an attach accepted at the cap can overflow
            # here if a next-up embed appears later. Drop the own-embeds tail, never
            # the block (parity with MusicContext.send's guard; unreachable today).
            embeds = embeds[:10]
            await message.edit(embeds=embeds)
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
        every non-success path: cancellation returns the item to the front (a bulk
        mutation is about to drain it with the rest), failure retires it on all three
        legs (leaving the display/Redis heads would make the next commit retire the
        wrong entry). On success the dequeue stays open and loop()'s
        commit/task_done() closes it."""
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
            # _stream_source swallowed a failure — retire the dequeue as the raise
            # path does, or the display/Redis heads sit one entry ahead forever.
            await self.queue.finish_failed_dequeue(source, context="prefetch failure")
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
        try:
            async with async_timeout.timeout(_PLAYBACK_GATE_TIMEOUT):
                await self._playback_gate.wait()
        except asyncio.TimeoutError:
            log.info(
                f"Playback gate timed out for guild {self._guild.id} "
                f"(never connected to voice), tearing down player"
            )
            asyncio.create_task(self.stop())
            return
        prefetched_song: Optional[YTDL] = None

        while not self.bot.is_closed():
            self.play_next.clear()
            # True while this iteration holds a dequeue not yet balanced with
            # task_done(), so the outer handler can close the books when a failure
            # lands between the dequeue and the normal song-end task_done() instead
            # of drifting the queue's task counter forever.
            dequeue_owed = False
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
                        # Cleared while _prefetch_next_song ran: the completed task
                        # consumed a get_nowait(), so balance it with task_done()
                        # and cleanup() the FFmpeg subprocess so discarding the
                        # result doesn't leak it.
                        self.queue.task_done()
                        prefetched_song.cleanup()
                        prefetched_song = None
                    if prefetched_song is not None:
                        self.current_song = prefetched_song
                        prefetched_song = None
                        dequeue_owed = True  # the prefetch's get_nowait() is ours
                        # Prefetched items always came through queue_get(), so they
                        # are real Redis-mirrored entries. source stays None:
                        # redis_pop_for(None) treats the dequeue as persisted.
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
                            # _resolve_source() raised after queue_get() already
                            # dequeued `source` — balance that dequeue as the
                            # "current_song is None" branch does, then re-raise into
                            # the outer handler's logging/error-embed path.
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

                    if not await self.queue.try_commit_dequeue():
                        # Cleared while this song resolved (e.g. Inside yt_stream).
                        # Discard without playing: task_done() balances the get()
                        # above, and cleanup() terminates the FFmpeg subprocess
                        # yt_stream already spawned, which would otherwise leak.
                        self.queue.task_done()
                        dequeue_owed = False
                        self.current_song.cleanup()
                        self.current_song = None
                        continue

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

                    vc.play(song, after=_after_play)
                    if song.start_paused:
                        # Park the player thread SYNCHRONOUSLY, before any await, so
                        # a song returning paused leaks a frame or two rather than a
                        # Redis round-trip of audio. Idempotent with the full pause()
                        # below, which runs after the start transaction so its
                        # pause_start_epoch survives that transaction's HDEL.
                        vc.pause()
                    play_start = time.time()  # capture immediately before any awaits

                    # Mirror the now-playing song to Redis. should_pop_queue=True →
                    # one MULTI/EXEC atomically LPOPs the queue and writes every state
                    # field plus the display snapshot, closing the at-most-once
                    # window. A crash-recovered "current song" was never on the Redis
                    # list, so only state is written — an LPOP would drop a queued one.
                    if self.store is not None:
                        # Backdated by the FFmpeg -ss offset so recovery math (now -
                        # play_start_epoch - pauses) yields true audio position, not
                        # time-since-vc.play(). Without it, ?t= songs and
                        # double-crash recoveries resume start_offset seconds early.
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
                        # Returns parked where -playnow interrupted it (the player
                        # thread was already paused synchronously at vc.play). The
                        # full pause() runs here so the Redis pause epochs and the
                        # debounced embed/Activity refresh all engage.
                        await self.pause(vc)
                    if song.is_resume:
                        await self._announce_resume(song)

                    await self.update_activity(song)
                    await self._send_now_playing(song)

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
                    if not skip_history and not stream_failed:
                        await self.history.add(
                            HistoryEntry.from_song(
                                song,
                                guild_id=self._guild.id,
                                played_at=time.time(),
                                # The host captured at song end, not
                                # _np_host_message, which _release_np_host() nulled
                                # above. Clearing current_song before the prefetch
                                # await makes this complete: nothing can adopt a
                                # later host for this song. 0 = nothing hosted it.
                                message_id=(
                                    finished_host.id if finished_host is not None else 0
                                ),
                            )
                        )

                    if self.store is not None:
                        await self.store.clear_song_end_state()

                    self.queue.task_done()
                    dequeue_owed = False
                    await self.update_activity(None)

                    # Deliberately last: current_song is already cleared, so the
                    # notice goes out alone rather than re-hosting an NP block for a
                    # song that never played.
                    if stream_failed:
                        await self._handle_dead_stream(song)
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
                    # No finalize for a song that errored — just release the host so
                    # the next song starts clean.
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
