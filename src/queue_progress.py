"""The live card a slow collection enqueue shows while it is still resolving.

Not a pure module, and it does not claim to be: it owns the delay before the card
appears, the task that drives it, the send, N edits and the delete. `ping.py` can
be pure because `dashboard.py` sequences it; this has no equivalent driver to hand
the work to, because its producer is one opaque `await` that emits increments.

What it does NOT own is the queue, Redis or the player. The card renders four
scalars, and the track list it is about does not exist while it is on screen.
See docs/ARCHITECTURE.md#queue-progress-card.
"""

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import discord
from discord.ext import commands
from opentelemetry import trace

from src import config
from src.dashboard import LiveMessage
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    is_mix,
    timestamp_warning,
)
from src.util import (
    BAR_WIDTH,
    channel_claim,
    fmt_duration,
    get_logger,
    join_task,
    pluralize,
    progress_line,
    set_when_set,
    set_within,
)

log = get_logger(__name__)

Source = Union[SpotifySource, YTSource, SoundcloudSource]

_TITLE = "Working on that playlist…"
# Past the ceiling. Still true, and deliberately not "failed": nothing has.
_STALLED_TITLE = "Still working on that playlist…"

# The elapsed line moves in 5s steps. Per second is 0.5 edits/s against a 1.0/s
# per-channel budget the Now Playing bar already spends a third of.
_ELAPSED_STEP_SECS = 5

# A YouTube list id, rendered bare inside a code span: nothing that matches can
# close the span, style the card or forge a link.
_LIST_ID = re.compile(r"[A-Za-z0-9_-]{2,64}")

# Bound on the teardown's wait for the driver: one Discord round trip plus
# margin. The driver owns its own retraction, so expiring here abandons a delete
# already in flight.
QUEUE_PROGRESS_JOIN_SECS = 10.0

# The claim kind. PLAY_RESOLVE_CONCURRENCY cannot stand in for it: that is taken
# inside the extraction, below where the card is entered, so requests 3..16 of a
# burst park there and are guaranteed to cross the display threshold.
_CARD_CLAIM = "queue-progress-card"


class EnqueuePhase(Enum):
    """What the card can show while the resolve runs."""

    FETCHING = "fetching"
    STALLED = "stalled"  # past the card's ceiling; the card stops editing


@dataclass(slots=True, kw_only=True)
class EnqueueProgress:
    """What the card shows. Mutated by the resolve, read by the driver."""

    phase: EnqueuePhase = EnqueuePhase.FETCHING
    done: int = 0
    total: Optional[int] = None
    started_at: float = field(default_factory=time.monotonic)

    def update(self, done: int, total: Optional[int]) -> None:
        """The ProgressFn the resolve calls. Synchronous and allocation-free: it
        runs inside the extraction, and an await here would put a Discord round
        trip on that path. `done` is absolute, so max() makes a dropped report
        self-correct on the next one and a replayed stream idempotent."""
        self.done = max(self.done, done)
        if total is not None:
            self.total = total


@dataclass(frozen=True, slots=True, kw_only=True)
class CardDetails:
    """What the card knows before the resolve returns. Everything else a playlist
    confirmation shows — the skipped count, the first ten titles — is minted
    inside queue_source and does not exist while this is on screen."""

    requester: str
    playlist_label: str = ""
    warning: str = ""
    placement_note: str = ""
    debug_suffix: str = ""


def _count(n: float) -> str:
    """A progress_line label for a song count."""
    return str(int(n))


def render_progress_card(
    progress: EnqueueProgress, *, elapsed_secs: float, details: CardDetails
) -> list[discord.Embed]:
    """The card as a pure function of what is known. `total` decides the shape:
    with one, the Now Playing bar labelled in songs; without one, an elapsed line.
    For a YouTube Mix, which YouTube marks infinite, no total is the STEADY state,
    not a transient one."""
    lines = [f"Requested by: [{details.requester}]"]
    if details.playlist_label:
        lines.append(details.playlist_label)
    total = progress.total
    if total is not None and total >= 1:
        # The count shown is the smallest that fills the cells drawn, so the number
        # never disagrees with the bar and moves only when a cell does: a whole
        # enqueue costs at most BAR_WIDTH edits. No elapsed line beside it, since
        # that moves every tick. See docs/ARCHITECTURE.md#queue-progress-card.
        cells = min(BAR_WIDTH, progress.done * BAR_WIDTH // total)
        shown = -(-cells * total // BAR_WIDTH)
        bar = progress_line(shown, total, label=_count)
        lines.append(f"{bar} {pluralize(total, 'song')}")
    else:
        # The card is its delay old at its first render, so the
        # step rounds: "0:00" on a message that took 2.5s to appear reads as
        # broken. Not round() — that is banker's, and sends exactly 2.5 to zero.
        step = int(elapsed_secs / _ELAPSED_STEP_SECS + 0.5) * _ELAPSED_STEP_SECS
        # `done` is real here — a YouTube playlist_index, a Spotify item count —
        # and it is the only thing on an indeterminate card that moves with the
        # work rather than with the clock.
        # The count is entries checked: a Mix walk counts repeats _playlist_tracks
        # then drops, so it can exceed the songs the confirmation reports.
        walked = f"`{progress.done}` checked" if progress.done else ""
        elapsed = f"`{fmt_duration(step)}` elapsed"
        lines.append(f"{walked} · {elapsed}" if walked else elapsed)
    if details.placement_note:
        lines.append(details.placement_note)
    if details.warning:
        lines.append(f"\n{details.warning}")
    embed = discord.Embed(
        title=_STALLED_TITLE if progress.phase is EnqueuePhase.STALLED else _TITLE,
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )
    if details.debug_suffix:
        embed.set_footer(text=details.debug_suffix)
    return [embed]


def card_ceiling(delay: float, tick: float, max_secs: float) -> float:
    """How long a card edits before it stalls. The card sends after its delay and
    checks the ceiling after each tick, so under delay + 2 ticks it stalls at its
    first check without one ordinary edit. The three are set separately, so no
    one of them can hold this."""
    return max(max_secs, delay + 2 * tick)


def is_collection(source: Source) -> bool:
    """Whether this input resolves to many tracks. The card owns these and
    slow_resolve_notice owns the rest: two messages for one -play is worse than
    either alone."""
    if isinstance(source, SpotifySource):
        return source.type in (SpotifyType.PLAYLIST, SpotifyType.ALBUM)
    if isinstance(source, YTSource):
        return source.type is YTType.PLAYLIST
    return False


def _playlist_label(source: Source) -> str:
    """What the card calls the collection, or "". Built from the parsed list id, not
    the pasted link, which overruns the row and is cut inside the id. An id that is
    not a YouTube id is not rendered, and a Spotify source carries no name."""
    if not isinstance(source, YTSource) or source.type is not YTType.PLAYLIST:
        return ""
    list_id = source.list_id or ""
    if not _LIST_ID.fullmatch(list_id):
        return ""
    return "YouTube Mix" if is_mix(list_id) else f"Playlist `{list_id}`"


@contextlib.asynccontextmanager
async def enqueue_progress(
    ctx: commands.Context,
    source: Source,
    *,
    delay: float,
    placement_note: str = "",
    debug_suffix: Optional[str] = None,
    request_settled: Optional[asyncio.Event] = None,
) -> AsyncGenerator[EnqueueProgress]:
    """Show a live card once a collection has resolved for `delay`, and take it
    back when `request_settled` is set or the block exits, whichever comes first.
    `delay` is required, so every call site reads the server's
    queue-progress-delay when it enters.

    The card is a SECOND message, deleted on every exit path; it never becomes
    the confirmation. `_reply` sends that through MusicContext.send, which adopts
    it as the Now Playing host, and a message an edit loop owns must not be that
    host, so every exit deletes this one and the confirmation is sent elsewhere.

    Enter it on the outer exit stack BEFORE any gate hold: the delete awaits a
    Discord call, and on a cold start that await may not sit between the teardown
    decision and the hold release.
    """
    progress = EnqueueProgress(started_at=time.monotonic())
    # Read once: every wait and the stall log name the values this card ran with.
    tick = config.queue_progress_tick_secs()
    ceiling = card_ceiling(delay, tick, config.queue_progress_max_secs())
    details = CardDetails(
        requester=ctx.author.mention,
        playlist_label=_playlist_label(source),
        warning=timestamp_warning(source) or "",
        placement_note=placement_note,
        debug_suffix=debug_suffix or "",
    )
    settled = asyncio.Event()

    def _render() -> list[discord.Embed]:
        return render_progress_card(
            progress,
            elapsed_secs=time.monotonic() - progress.started_at,
            details=details,
        )

    async def _run_card() -> None:
        if await set_within(settled, delay):
            # The common case: the enqueue landed inside the delay and the
            # channel sees exactly what it sees today.
            return
        span = trace.get_current_span()
        while True:
            # Held until the card is DELETED, not until the loop ends: a stalled card
            # stops editing and stays on screen, and releasing there would let a
            # sibling post a second one beside it.
            with channel_claim(_CARD_CLAIM, ctx.channel.id) as claimed:
                if claimed:
                    await _show_card(span)
                    return
            # Another request's card holds the channel. Asked again each tick, so
            # this card appears once that one is taken back.
            span.set_attribute("play.progress_card", "claimed_elsewhere")
            if await set_within(settled, tick):
                return

    async def _show_card(span: trace.Span) -> None:
        live = LiveMessage(tick)
        try:
            try:
                await live.start(ctx, _render)
            except (discord.HTTPException, OSError, RuntimeError) as e:
                # A channel that refuses the send leaves the enqueue untouched: the
                # card is advisory and must never take the command down.
                span.set_attribute("play.progress_card", "send_failed")
                log.warning("queue progress card send failed", error=repr(e))
                return
            span.set_attribute("play.progress_card", "shown")
            deadline = progress.started_at + ceiling
            while True:
                if await set_within(settled, tick):
                    return
                if time.monotonic() >= deadline:
                    progress.phase = EnqueuePhase.STALLED
                    span.set_attribute("play.progress_card", "stalled")
                    log.warning(
                        "queue progress card stalled",
                        max_secs=ceiling,
                        done=progress.done,
                        total=progress.total,
                    )
                    # finish(), not tick(): the terminal render must not be lost
                    # to the floor. Then no more edits, but the card and the claim
                    # stay until the enqueue settles.
                    with contextlib.suppress(discord.HTTPException):
                        await live.finish()
                    if live.message is not None:
                        await settled.wait()
                    return
                if not await live.tick():
                    return  # the user deleted it
        finally:
            # The task that sent the card takes it back, in its own finally: a
            # caller cancelled mid-send never learns the handle. Suppressed because
            # this unwinds while an ExtractionError may be propagating, and must
            # not replace it.
            message = live.message
            if message is not None:
                with contextlib.suppress(discord.HTTPException, OSError, RuntimeError):
                    await message.delete()

    driver = asyncio.create_task(_run_card())
    relay = (
        asyncio.create_task(set_when_set(request_settled, settled))
        if request_settled
        else None
    )
    try:
        yield progress
    finally:
        settled.set()
        if relay is not None:
            relay.cancel()
        # Signalled, never cancelled, and joined so the retraction above lands
        # first. Bounded because a -play unwinds through here: past it the
        # retraction is left to the driver's own finally.
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(QUEUE_PROGRESS_JOIN_SECS):
                await join_task(driver)
