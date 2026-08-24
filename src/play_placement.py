"""How a `-play` is parsed, gated, admitted and placed.

Requests resolve concurrently and serialize only at the insert, so this module
owns three things the cog composes: the flag grammar that decides where a song
goes, the voice gate that decides whether it may go there, and the per-guild
registry whose lock makes "check, then insert" atomic against `-clear`/`-stop`.

See docs/ARCHITECTURE.md#play-placement.
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

# Bound on a -play's place section: the wait for the guild's place lock plus
# the one Redis round trip inside it. The pool sets no socket_timeout, so a Redis
# that accepts and then stalls would otherwise park every -play in the guild
# behind the first to reach the lock.
PLACE_TIMEOUT_SECS = 5.0


# ── The flag grammar ──────────────────────────────────────────────────────────

NOW_FLAG: Final[str] = "--now"
NEXT_FLAG: Final[str] = "--next"


class PlayMode(Enum):
    """Where a `-play` invocation puts its song.

    One field, so `now and next` is unrepresentable.
    """

    NORMAL = "normal"
    NOW = "now"
    NEXT = "next"


_FLAG_MODES: Final[dict[str, PlayMode]] = {
    NOW_FLAG: PlayMode.NOW,
    NEXT_FLAG: PlayMode.NEXT,
}


class Placement(Enum):
    """Where an enqueue puts its songs, and which confirmation says so.

    Two decisions, not one: build_resume_notice_embed ("N songs from the previous
    session resume after it") is true for a disconnected bot waking a persisted
    queue and false for a warm front-insert, and it renders only when the queue is
    non-empty — exactly the case that would be wrong.
    """

    TAIL = "tail"
    COLD_FRONT = "cold_front"
    NEXT = "next"


# Every dash Unicode offers that a keyboard or a paste substitutes for ASCII `-`:
# hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar. iOS
# turns a typed `--` into a single em dash.
_DASHES: Final[str] = "-‐‑‒–—―−"
# Alternation built from _FLAG_MODES' own keys, so a renamed flag cannot leave this
# branch offering one that no longer exists. The group is a flag minus its dashes;
# split_play_args re-attaches them for the reply.
_NEAR_FLAG_RE: Final[re.Pattern[str]] = re.compile(
    f"[{_DASHES}]{{1,2}}({'|'.join(flag[2:] for flag in _FLAG_MODES)})"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayArgs:
    """`-play`'s argument, split into the placement flag and the query behind it.

    kw_only because `query` and `dash_typo` are adjacent and both `str`-ish, and one
    of them is echoed into an embed: transposed, the bot asks "did you mean `<the
    user's whole search>`?".

    `dash_typo` names the flag a misspelt leading token meant. It only ever
    accompanies `PlayMode.NORMAL`, since which flag was intended is unknown.
    """

    mode: PlayMode
    query: str
    dash_typo: Optional[str] = None


def split_play_args(argument: str) -> PlayArgs:
    """Split a leading `--now`/`--next` off `-play`'s argument.

    Only the FIRST token is considered, so a flag further along stays part of the
    search text and the origin `-remove` matches on stays what the user typed. One
    flag, never a run: `-p --now --next x` takes `--now` and searches for "--next x".

    Hand-parsed: a FlagConverter's grammar is `--flag value`, which cannot express a
    valueless switch, and it matches flags anywhere in the line.

    A leading token one dash away from a flag gets `dash_typo` — `-now`, or an
    autocorrected `—next`. The exact-match lookup runs first, since a real `--now`
    also satisfies the near-miss pattern. A bare leading `now`/`next` does not
    qualify: `-p next to me` is a real search.
    """
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


def join_succeeded(ctx: commands.Context) -> bool:
    """Did the join a cold-start command just ran leave a USABLE voice client?

    is_connected(), not just the type: discord.py registers the client on the guild
    BEFORE the handshake completes, and vc.play() on a still-connecting one raises
    once per restored song. join also swallows its own failures, so a failed one
    arrives here as an absent client rather than an exception. Shared by -play and
    -resume — the two checks must never diverge, since a type-only check is exactly
    the bug this guards.
    """
    vc = ctx.voice_client
    return isinstance(vc, discord.VoiceClient) and vc.is_connected()


def play_takes_the_queue(
    ctx: commands.Context, voice_client: Optional[discord.VoiceClient]
) -> bool:
    """Whether this -play decides what a channel hears next, rather than adding to
    the end of what it is already hearing.

    `--now` stops the current song and `--next` takes the front of the queue, so
    both are gated on the same channel every other queue command is.

    Reads the PARSED argument: Command.prepare() runs _parse_arguments before
    call_before_hooks, so ctx.kwargs is filled by the time the gate runs. Other
    commands carry no `url` and fall out at the `.get`.

    A paused voice client counts without checking for a current song — the gate
    cannot ask for one without building a player.
    """
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
    """Returns an error message string if validation fails, None if OK.

    -play alone is exempt from the same-channel rule: queueing into a session
    running elsewhere costs its listeners nothing. Queue control is gated like every
    other order-changing command (-skip, -shuffle, -remove, -clear) even when it
    arrives as -play.
    """
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
    """validate_commands' check, re-run for a request whose resolve is over: the
    author can leave voice during a 99s extraction, and join() reads
    ctx.author.voice.channel behind an assert.

    queue_control has to be passed, not defaulted. Dropping it re-enters the
    same-channel EXEMPTION -play alone gets, so a `--now` whose author walked to
    another channel mid-resolve would stop what the channel they left is hearing —
    the one thing the gate at dispatch exists to refuse."""
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
    """One value for what the lock decided, rather than a verdict beside fields the
    caller has to remember to read: `refusal` accompanies VOICE and only VOICE, and
    is the message that verdict is reported with."""

    verdict: PlaceVerdict
    refusal: str = ""

    @property
    def placed(self) -> bool:
        return self.verdict is PlaceVerdict.PLACE


class PlaceStalled(Exception):
    """PLACE_TIMEOUT_SECS elapsed waiting for the place lock or inside the put.

    `before_the_put` says which half ran out, because it decides what the caller
    may claim: waiting for the lock nothing was written, while past it the put may
    already have appended to the deque.
    """

    def __init__(self, *, before_the_put: bool) -> None:
        super().__init__(f"place stalled after {PLACE_TIMEOUT_SECS}s")
        self.before_the_put = before_the_put


@dataclass(slots=True, eq=False)
class PlayRequest:
    """One -play between dispatch and reply.

    `mp` and `generation` are the world as it was at dispatch; place() checks
    whether that player was retired and that generation bumped since. `dropped_by`
    names the command that made that check fail, when one did.

    eq=False keeps identity equality, which is what the registry holds it by: two
    requests for the same query from the same author are two requests.
    """

    ctx: commands.Context
    guild_id: int
    query: str
    mp: MusicPlayer
    generation: int
    mode: PlayMode
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
        # A list, in arrival order: the drop reports read it back as the order the
        # requests were sent in, and PLAY_INFLIGHT_MAX caps it at 16, so the linear
        # removal a retire costs is bounded at sixteen identity comparisons.
        self.inflight: list[PlayRequest] = []
        self.join: Optional[asyncio.Task[Any]] = None
        # Admission (PLAY_INFLIGHT_MAX) bounds what this guild holds in memory;
        # this bounds what it holds of the shared yt-dlp pool. Requests wait here
        # rather than being refused — the queue is fair within the guild and the
        # bound is what keeps it fair between guilds.
        self.resolves = asyncio.Semaphore(PLAY_RESOLVE_CONCURRENCY)

    def idle(self) -> bool:
        return not self.inflight and self.join is None


def play_key(ctx: commands.Context) -> int:
    """The guild whose place lock this request takes.

    validate_commands refuses a DM before any caller's body runs — a discord.User
    has no voice channel — so the fallback keeps the key an int rather than serving
    a reachable case.
    """
    return ctx.guild.id if ctx.guild else 0


class PlayRegistry:
    """Every guild's in-flight `-play` requests, and the place lock each guild's
    insertions take. One per cog.

    Per-guild state is created on the first request and dropped when the guild goes
    idle, so a bot in 10,000 guilds holds entries for the handful playing.
    """

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

    def join_in_flight(self, guild_id: int) -> bool:
        """Whether a cold-start join is running for this guild right now."""
        plays = self._guilds.get(guild_id)
        return plays is not None and plays.join is not None

    async def retire_player(self, guild_id: int, mp: MusicPlayer) -> None:
        """Mark a player retired without landing in the middle of a placement.

        Verdict ① of place() reads this flag, and the put that follows it writes the
        deque first and the Redis mirror second. Setting the flag between those two
        leaves the song in one leg and not the other. Taken under the same lock, the
        flag orders against whole puts: one already holding the lock finishes, and
        every request arriving from here on queues behind this acquisition and finds
        the player retired. Bounded by the put's own 5s and retired anyway on
        expiry — a stalled Redis must not hold a teardown open.
        """
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
        made its placement fail. The stamp is itself a verdict in place(), so this
        holds even when the caller had nothing to retire or bump. Synchronous: the
        caller reads it with no await between its own act and this, so no request
        can place between."""
        plays = self._guilds.get(guild_id)
        if plays is None:
            return []
        # Not the ones that already placed: retire() runs in play()'s finally,
        # after the confirmation is sent, so a request whose song is in the queue is
        # still listed here. Stamping it names it in the dropping command's own
        # "dropped" field beside the song it just cleared, and leaves a stale
        # dropped_by to mis-attribute a later teardown.
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
        """The guild's one cold-start join, and whether THIS request created it.

        The first request to find no voice client creates the task; every request
        that does so while it runs gets the same one. Await it through
        asyncio.shield — `await task` carries the awaiting task's cancellation into
        the awaited one, and another request may be waiting on it. Tracked in the
        caller's set, so the cog's teardown owns it.

        `joiner` builds the coroutine rather than being one: the shared-join branch
        never spawns, and a coroutine created for it would be collected un-awaited.

        The flag is who join() reports a failure to: it runs on the creator's
        context, so a request that only waited has to report for itself."""
        plays = self._guilds[req.guild_id]
        # cancelling(): a creator that failed alone has cancelled its join but the
        # task only settles on a later tick; a request arriving in that window
        # must not be handed a join that is about to raise at it.
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
        """The guild's place lock, with the four checks a resolved request has to
        pass before it may insert. The body runs under the lock and must be the
        put alone: no Discord call, no resolve, no confirmation to render. The
        caller sends every verdict's message after the block, and catches
        PlaceStalled around the whole of it.

        ① The player first: -stop, a kick or the alone-watchdog retired `mp`, and
        a put into it would land in the Redis mirror alone, to be resurrected by
        the next restore. ② Generation: clear() bumped it since this request was
        admitted — the same signal the loop's own commit refuses on. ③ A command's
        own stamp, which invalidates on its own. ④ The author's voice state,
        re-read after the resolve.

        The hold is one Redis round trip long, so a guild bursting to
        PLAY_INFLIGHT_MAX serializes that many: ~40ms at the 2.4ms p50 the start
        transaction measures, and past ~300ms per trip the tail spends its whole
        budget waiting. That is reported as a busy queue, which is what it is.
        """
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
                    # A command stamped this request and had nothing to retire or
                    # bump: -stop before the join lands cleans up no player, since
                    # there is no voice client to find. The stamp has to BE an
                    # invalidation, or the channel is told the request was dropped
                    # and it places, joins and plays anyway.
                    span.set_attribute("play.verdict", PlaceVerdict.SESSION_ENDED.value)
                    span.set_attribute("play.dropped_by", req.dropped_by)
                    yield PlaceResult(PlaceVerdict.SESSION_ENDED)
                    return
                # From the request's own parsed mode, not a re-parse: the
                # pair must answer exactly what play_takes_the_queue answered at
                # dispatch, and a paused song is queue control however it arrived.
                vc = req.ctx.voice_client
                refusal = voice_refusal(
                    req.ctx,
                    queue_control=req.mode is not PlayMode.NORMAL
                    or (isinstance(vc, discord.VoiceClient) and vc.is_paused()),
                )
                if refusal is not None:
                    # dropped_by names a COMMAND, and no command did this — the
                    # verdict is what a "did not place" query has to read.
                    span.set_attribute("play.verdict", PlaceVerdict.VOICE.value)
                    yield PlaceResult(PlaceVerdict.VOICE, refusal)
                    return
                span.set_attribute("play.verdict", PlaceVerdict.PLACE.value)
                req.placed = True
                yield PlaceResult(PlaceVerdict.PLACE)
        except TimeoutError as e:
            if not bound.expired():
                # Someone else's deadline — an inner wait_for, a guard added to the
                # body later. Reporting it as a stalled queue names the wrong
                # subsystem and swallows a real error.
                raise
            # The only record a stall leaves: both callers turn PlaceStalled into
            # a notice, so it never reaches _command_error's log-and-record. The
            # wait is re-read here because the attribute above is set INSIDE the
            # lock, which a request that never acquired it did not reach.
            span.set_attribute("play.verdict", "stalled")
            span.set_attribute(
                "play.place_wait_secs", round(time.monotonic() - waited, 3)
            )
            record_span_error(span, e)
            log.warning(f"Place stalled after {PLACE_TIMEOUT_SECS}s: {req.query}")
            raise PlaceStalled(before_the_put=not acquired) from e
