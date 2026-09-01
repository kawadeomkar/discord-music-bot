"""How a `-play` is parsed, gated, admitted and placed: the flag grammar, the
voice gate, and the per-guild registry whose lock makes "check, then insert"
atomic against `-clear`/`-stop`. See docs/ARCHITECTURE.md#play-placement.
"""

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Optional, Union
from collections.abc import AsyncGenerator, Callable, Coroutine

import discord
from discord.ext import commands

from opentelemetry import trace

from src.config import PLAY_INFLIGHT_MAX, PLAY_RESOLVE_CONCURRENCY
from src.musicplayer import MusicPlayer
from src.util import get_logger, record_span_error, spawn_background

log = get_logger(__name__)

# Bound on a -play's place section: the wait for the guild's lock plus one Redis
# round trip. The pool sets no socket_timeout, so this is the only bound on it.
# Above musicplayer._START_WRITE_TIMEOUT (5s), which a song start holds the queue
# mutex for against the same stall: equal bounds expire together, so a request that
# could have landed the moment the mutex freed reports a stall instead.
PLACE_TIMEOUT_SECS = 7.0


# ── The flag grammar ──────────────────────────────────────────────────────────

NOW_FLAG: Final[str] = "--now"
NEXT_FLAG: Final[str] = "--next"


class PlayMode(Enum):
    """Where a `-play` invocation puts its song. One field, so `now and next` is
    unrepresentable."""

    NORMAL = "normal"
    NOW = "now"
    NEXT = "next"


_FLAG_MODES: Final[dict[str, PlayMode]] = {
    NOW_FLAG: PlayMode.NOW,
    NEXT_FLAG: PlayMode.NEXT,
}


class Placement(Enum):
    """Where an enqueue puts its songs, and which confirmation says so. COLD_FRONT
    and NEXT both front-insert, but only a disconnected bot waking a persisted
    queue earns the resume notice."""

    TAIL = "tail"
    COLD_FRONT = "cold_front"
    NEXT = "next"


class ResolveMode(Enum):
    """Whether a resolve may stop at search metadata. FULL is for a head that has to
    be playable before it is used: an interjection stops the current song, and a cold
    start has nothing queued behind it. Every other placement is FLAT_OK; a lazy entry
    resolving at dequeue passes no mode. See docs/ARCHITECTURE.md#resolve-mode."""

    FLAT_OK = "flat_ok"
    FULL = "full"


def resolve_mode_for(placement: Placement) -> ResolveMode:
    """FULL for a cold start, whose song plays immediately and so pays the stream
    extraction either way; FLAT_OK for every other placement. Enumerated rather than
    defaulted, so a Placement added later cannot inherit FLAT_OK in silence."""
    if placement is Placement.COLD_FRONT:
        return ResolveMode.FULL
    return ResolveMode.FLAT_OK


# Every dash Unicode offers that a keyboard or a paste substitutes for ASCII `-`.
# iOS turns a typed `--` into a single em dash.
_DASHES: Final[str] = "-‐‑‒–—―−"
# Built from _FLAG_MODES' keys, so a renamed flag cannot leave a stale near-miss.
# The group is the flag minus its dashes; split_play_args re-attaches them.
_NEAR_FLAG_RE: Final[re.Pattern[str]] = re.compile(
    f"[{_DASHES}]{{1,2}}({'|'.join(flag[2:] for flag in _FLAG_MODES)})"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayArgs:
    """`-play`'s argument, split into the placement flag and the query. kw_only:
    `query` and `dash_typo` are adjacent strings and one is echoed into an embed.
    `dash_typo` names the flag a misspelt leading token meant; it only accompanies
    PlayMode.NORMAL."""

    mode: PlayMode
    query: str
    dash_typo: Optional[str] = None


def split_play_args(argument: str) -> PlayArgs:
    """Split a leading `--now`/`--next` off `-play`'s argument. Only the FIRST token
    counts, so a flag further along stays part of the search and of the origin
    `-remove` matches on; one flag, never a run. A leading token one dash off a
    flag (`-now`, an autocorrected `—next`) sets `dash_typo`; the exact match runs
    first, since a real `--now` also fits the near-miss pattern. A bare `now`/`next`
    is a search (`-p next to me`)."""
    stripped = argument.strip()
    parts = stripped.split(maxsplit=1)
    if not parts:
        return PlayArgs(mode=PlayMode.NORMAL, query="")
    head = parts[0].lower()
    # No strip on the tail: `stripped` had none, and split() eats the separator run.
    rest = parts[1] if len(parts) > 1 else ""
    mode = _FLAG_MODES.get(head)
    if mode is not None:
        return PlayArgs(mode=mode, query=rest)
    typo = _NEAR_FLAG_RE.fullmatch(head)
    if typo is not None:
        return PlayArgs(
            mode=PlayMode.NORMAL, query=stripped, dash_typo=f"--{typo.group(1)}"
        )
    return PlayArgs(mode=PlayMode.NORMAL, query=stripped)


# ── The voice gate ────────────────────────────────────────────────────────────


def play_takes_the_queue(
    ctx: commands.Context, voice_client: Optional[discord.VoiceClient]
) -> bool:
    """Whether this -play decides what a channel hears next (`--now` stops the
    song, `--next` takes the front) rather than appending, and so is gated on the
    same channel every other queue command is. Reads the PARSED argument:
    Command.prepare() parses before call_before_hooks. A paused client counts
    without checking for a current song — the gate cannot build a player."""
    if voice_client is None:
        return False
    if voice_client.is_paused():
        return True
    return split_play_args(str(ctx.kwargs.get("url", ""))).mode is not PlayMode.NORMAL


def check_voice_permissions(
    author: Union[discord.Member, discord.User],
    voice_client: Optional[discord.VoiceClient],
    command_name: str,
    *,
    queue_control: bool = False,
) -> Optional[str]:
    """Returns an error message if validation fails, None if OK. Plain -play is
    exempt from the same-channel rule (appending costs listeners elsewhere
    nothing); queue control is gated like -skip/-shuffle/-remove/-clear."""
    if isinstance(author, discord.User):
        return f"You must be a member of this channel {author}"
    if not author.voice or not author.voice.channel:
        return f"You are not connected to a voice channel, you silly baka {author}"
    if (
        (command_name != "play" or queue_control)
        and voice_client is not None
        and voice_client.channel != author.voice.channel
    ):
        return f"Bot is already being used in channel {voice_client.channel}"
    return None


def voice_refusal(ctx: commands.Context, *, queue_control: bool) -> Optional[str]:
    """validate_commands' check, re-run once the resolve is over: the author can
    leave voice during a 99s extraction. queue_control is passed, not defaulted —
    defaulting re-enters plain -play's same-channel exemption, and a `--now` whose
    author walked to another channel would stop what the first one is hearing."""
    vc = ctx.voice_client
    return check_voice_permissions(
        ctx.author,
        vc if isinstance(vc, discord.VoiceClient) else None,
        ctx.command.name if ctx.command is not None else "",
        queue_control=queue_control,
    )


# ── The placement ─────────────────────────────────────────────────────────────


class PlaceVerdict(Enum):
    """What place() found when a request reached the lock. Every value but PLACE
    is reported to the author by the caller, after the lock is released."""

    PLACE = "place"
    SESSION_ENDED = "session_ended"
    CLEARED = "cleared"
    VOICE = "voice"


@dataclass(frozen=True, slots=True)
class PlaceResult:
    """What the lock decided. `refusal` accompanies VOICE and only VOICE, and is
    the message that verdict is reported with."""

    verdict: PlaceVerdict
    refusal: str = ""

    @property
    def placed(self) -> bool:
        return self.verdict is PlaceVerdict.PLACE


class PlaceStalled(Exception):
    """PLACE_TIMEOUT_SECS elapsed waiting for the place lock or inside the put.
    `before_the_put` decides what the caller may claim: past the lock the put may
    already have appended to the deque."""

    def __init__(self, *, before_the_put: bool) -> None:
        super().__init__(f"place stalled after {PLACE_TIMEOUT_SECS}s")
        self.before_the_put = before_the_put


@dataclass(slots=True, eq=False)
class PlayRequest:
    """One -play between dispatch and reply. `mp` and `generation` are the world at
    dispatch, which place() checks for a retired player and a bumped generation;
    `dropped_by` names the command that failed that check. eq=False: the registry
    holds requests by identity."""

    ctx: commands.Context
    guild_id: int
    query: str
    mp: MusicPlayer
    generation: int
    mode: PlayMode
    # Read at dispatch and carried, not re-derived: place() re-runs the voice check
    # after the resolve, where a -pause landing in between would change the answer.
    queue_control: bool = False
    dropped_by: str = ""
    # Set under the place lock. A placed request is past the point any command can
    # drop it, but it stays in the registry until its reply is sent.
    placed: bool = False


class _GuildPlays:
    """A guild's -play requests in flight, the lock their placements take, and the
    cold-start join they share."""

    __slots__ = ("lock", "inflight", "join", "resolves")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        # Arrival order — the drop reports read it back as sent order. Capped at
        # PLAY_INFLIGHT_MAX, so a retire's linear removal is bounded.
        self.inflight: list[PlayRequest] = []
        self.join: Optional[asyncio.Task[Any]] = None
        # PLAY_INFLIGHT_MAX bounds what this guild holds in memory; this bounds what
        # it holds of the shared yt-dlp pool. Requests wait rather than being refused.
        self.resolves = asyncio.Semaphore(PLAY_RESOLVE_CONCURRENCY)

    def idle(self) -> bool:
        return not self.inflight and self.join is None


def play_key(ctx: commands.Context) -> int:
    """The guild whose place lock this request takes. validate_commands refuses a
    DM before any body runs, so the fallback only keeps the key an int."""
    return ctx.guild.id if ctx.guild else 0


class PlayRegistry:
    """Every guild's in-flight `-play` requests and the place lock its insertions
    take; one per cog. Per-guild state is created on the first request and dropped
    when the guild goes idle."""

    def __init__(self) -> None:
        self._guilds: dict[int, _GuildPlays] = {}

    def register(
        self, ctx: commands.Context, *, query: str, mp: MusicPlayer, mode: PlayMode
    ) -> PlayRequest:
        """Admit a -play to the guild's in-flight set, or decline it past
        PLAY_INFLIGHT_MAX. Synchronous from the cap check to the insert, so two
        dispatches in one tick cannot both pass."""
        span = trace.get_current_span()
        key = play_key(ctx)
        plays = self._guilds.get(key)
        if plays is None:
            plays = self._guilds[key] = _GuildPlays()
        # Recorded before the cap check: a declined request carries the count it
        # would have joined, and the span is the only place declines are counted.
        span.set_attribute("play.inflight", len(plays.inflight) + 1)
        if len(plays.inflight) >= PLAY_INFLIGHT_MAX:
            span.set_attribute("play.declined", True)
            raise commands.MaxConcurrencyReached(
                PLAY_INFLIGHT_MAX, commands.BucketType.guild
            )
        req = PlayRequest(
            ctx=ctx,
            guild_id=key,
            query=query,
            mp=mp,
            generation=mp.queue.generation,
            mode=mode,
            queue_control=mode is not PlayMode.NORMAL
            or (
                isinstance(ctx.voice_client, discord.VoiceClient)
                and ctx.voice_client.is_paused()
            ),
        )
        plays.inflight.append(req)
        return req

    def retire(self, req: PlayRequest) -> None:
        plays = self._guilds.get(req.guild_id)
        if plays is None:
            return
        with contextlib.suppress(ValueError):
            plays.inflight.remove(req)
        if plays.idle():
            self._guilds.pop(req.guild_id, None)

    def resolve_slot(self, req: PlayRequest) -> asyncio.Semaphore:
        """The guild's bound on how many of its requests hold a yt-dlp worker."""
        return self._guilds[req.guild_id].resolves

    def sibling_placed(self, req: PlayRequest) -> bool:
        """Whether another of this guild's in-flight -plays has already landed — what
        the cold-start resume notice's "previous session" wording depends on."""
        plays = self._guilds.get(req.guild_id)
        if plays is None:
            return False
        return any(other is not req and other.placed for other in plays.inflight)

    def join_in_flight(self, guild_id: int) -> bool:
        """Whether a cold-start join is running for this guild right now."""
        plays = self._guilds.get(guild_id)
        return plays is not None and plays.join is not None

    async def retire_player(self, guild_id: int, mp: MusicPlayer) -> None:
        """Mark a player retired without landing mid-placement: the put writes the
        deque and then the mirror, and a flag set between the two leaves the song
        in one leg only. Taken under the place lock, so a put in progress finishes
        and every later request finds the player retired. Retired anyway once the
        put's own bound expires — a stalled Redis must not hold a teardown."""
        plays = self._guilds.get(guild_id)
        if plays is None:
            mp.mark_retired()
            return
        try:
            async with asyncio.timeout(PLACE_TIMEOUT_SECS), plays.lock:
                mp.mark_retired()
        except TimeoutError:
            mp.mark_retired()

    def inflight(
        self,
        guild_id: int,
        by: str,
        matches: Callable[[PlayRequest], bool] = lambda _r: True,
    ) -> list[PlayRequest]:
        """The guild's resolving -play requests, each stamped with the command that
        failed its placement; the stamp is itself a verdict in place(). Synchronous,
        so no request can place between the caller's act and this."""
        plays = self._guilds.get(guild_id)
        if plays is None:
            return []
        # Not the ones that already placed: retire() runs in play()'s finally, so a
        # request whose song is in the queue is still listed here.
        dropped = [r for r in plays.inflight if not r.placed and matches(r)]
        for req in dropped:
            req.dropped_by = by
        return dropped

    def cold_join(
        self,
        req: PlayRequest,
        *,
        joiner: Callable[[], Coroutine[Any, Any, Any]],
        tracked: set[asyncio.Task[Any]],
    ) -> tuple[asyncio.Task[Any], bool]:
        """The guild's one cold-start join, and whether THIS request created it: the
        first request to find no voice client spawns it, later ones get the same
        task (awaited through asyncio.shield, so one waiter's cancellation stays its
        own). join() reports failure on the creator's context only."""
        plays = self._guilds[req.guild_id]
        # cancelling(): a creator that failed alone cancelled its join, but the task
        # settles on a later tick; nobody may be handed a join about to raise.
        if plays.join is None or plays.join.cancelling():
            task = spawn_background(joiner(), tracked)
            plays.join = task

            def _done(_: asyncio.Task[Any]) -> None:
                if plays.join is task:
                    plays.join = None
                if plays.idle() and self._guilds.get(req.guild_id) is plays:
                    self._guilds.pop(req.guild_id)

            task.add_done_callback(_done)
            return task, True
        return plays.join, False

    @contextlib.asynccontextmanager
    async def place(self, req: PlayRequest) -> AsyncGenerator[PlaceResult]:
        """The guild's place lock, with the checks a resolved request passes before
        it may insert: ① the player was retired, ② clear() bumped the generation,
        ③ a command stamped `dropped_by`, ④ the author's voice state, re-read after
        the resolve. The body must be the put alone — no Discord call, no resolve, no
        rendering — and the caller reports every verdict after the block, catching
        PlaceStalled around it. See docs/ARCHITECTURE.md#play-placement."""
        plays = self._guilds[req.guild_id]
        span = trace.get_current_span()
        waited = time.monotonic()
        acquired = False
        try:
            async with asyncio.timeout(PLACE_TIMEOUT_SECS) as bound, plays.lock:
                acquired = True
                span.set_attribute(
                    "play.place_wait_secs", round(time.monotonic() - waited, 3)
                )
                if req.mp.retired:
                    span.set_attribute("play.verdict", PlaceVerdict.SESSION_ENDED.value)
                    span.set_attribute("play.dropped_by", req.dropped_by or "session")
                    yield PlaceResult(PlaceVerdict.SESSION_ENDED)
                    return
                if req.mp.queue.generation != req.generation:
                    span.set_attribute("play.verdict", PlaceVerdict.CLEARED.value)
                    span.set_attribute("play.dropped_by", req.dropped_by or "clear")
                    yield PlaceResult(PlaceVerdict.CLEARED)
                    return
                if req.dropped_by:
                    # Stamped by a command that had nothing to retire or bump (-stop
                    # before the join lands), so the stamp has to BE an invalidation.
                    span.set_attribute("play.verdict", PlaceVerdict.SESSION_ENDED.value)
                    span.set_attribute("play.dropped_by", req.dropped_by)
                    yield PlaceResult(PlaceVerdict.SESSION_ENDED)
                    return
                # The dispatch-time reading, carried on the request: this answers
                # exactly what play_takes_the_queue answered when it was admitted.
                refusal = voice_refusal(req.ctx, queue_control=req.queue_control)
                if refusal is not None:
                    # dropped_by names a COMMAND, and no command did this — the
                    # verdict is what a "did not place" query has to read.
                    span.set_attribute("play.verdict", PlaceVerdict.VOICE.value)
                    yield PlaceResult(PlaceVerdict.VOICE, refusal)
                    return
                span.set_attribute("play.verdict", PlaceVerdict.PLACE.value)
                yield PlaceResult(PlaceVerdict.PLACE)
                # After the body: a put that raised or was cut short queued nothing,
                # and a placed request is one no command will stamp or report.
                req.placed = True
        except TimeoutError as e:
            if not bound.expired():
                # Someone else's deadline (an inner wait_for). Reporting it as a
                # stalled queue would name the wrong subsystem and swallow the error.
                raise
            # The only record a stall leaves — both callers turn PlaceStalled into a
            # notice. Re-read here: the attribute above is set INSIDE the lock.
            span.set_attribute("play.verdict", "stalled")
            span.set_attribute(
                "play.place_wait_secs", round(time.monotonic() - waited, 3)
            )
            record_span_error(span, e)
            log.warning(f"Place stalled after {PLACE_TIMEOUT_SECS}s: {req.query}")
            raise PlaceStalled(before_the_put=not acquired) from e
