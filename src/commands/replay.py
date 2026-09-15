"""`-replay` — play the live song again from its beginning."""

import asyncio
import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import discord
from discord.ext import commands
from opentelemetry import trace

from src.commands._common import NOTHING_PLAYING
from src.guild_queue import is_replay_of
from src.guild_state import Analytics
from src.musicplayer import MusicPlayer
from src.telemetry import get_tracer
from src.util import (
    ECHO_MAX,
    background_typing,
    fmt_duration,
    notice_embed,
    refund_cooldown,
    safe_label,
)
from src.youtube import YTDL, QueueObject

_tracer = get_tracer(__name__)


# Below this position -replay refuses: the song is at its beginning. Also bounds
# history churn — every replay writes an entry to a list LTRIMmed to
# HISTORY_CACHE_LIMIT.
MIN_REPLAY_POSITION_SECS = 1

# How long -replay waits for its copy to resolve. Past it the live song is left
# playing: the resolve continues, and the copy it holds plays when the song ends.
_REPLAY_RESOLVE_TIMEOUT = 8.0


class ReplayResult(Enum):
    """How a -replay that front-inserted its copy ended. Only REPLAYING stopped the
    live song; see docs/ARCHITECTURE.md#-replay."""

    REPLAYING = "replaying"
    # The song ended on its own before the stop. The copy is next.
    ENDED_FIRST = "ended_first"
    # A -skip or --now stopped the song before the stop. The copy is still queued.
    STOPPED_ELSEWHERE = "stopped_elsewhere"
    # The player was torn down. The copy waits in the saved queue for -resume.
    TORN_DOWN = "torn_down"
    # The resolve outlived _REPLAY_RESOLVE_TIMEOUT. The copy plays after the song.
    STILL_LOADING = "still_loading"
    # Another command cancelled the resolve. The verdict runs before that command
    # changes the queue, so where the copy ends up is not known here.
    INTERRUPTED = "interrupted"
    # A -shuffle or --next put something ahead of the resolved copy, still queued.
    QUEUED_LATER = "queued_later"
    # The resolve came back with no song, and retired the copy.
    FAILED = "failed"
    # A -clear or -remove took the copy out of the queue.
    DROPPED = "dropped"


@dataclass(frozen=True)
class ReplayOutcome:
    """What replay_current() did, for -replay's confirmation wording."""

    title: str
    position: int  # where the live song was when -replay ran
    result: ReplayResult = ReplayResult.REPLAYING

    @property
    def stopped(self) -> bool:
        return self.result is ReplayResult.REPLAYING

    @property
    def position_str(self) -> str:
        return fmt_duration(self.position)


@_tracer.start_as_current_span("player.replay")
async def replay_current(
    mp: MusicPlayer,
    vc: discord.VoiceClient,
    *,
    requester: Union[discord.User, discord.Member],
    analytics: Analytics,
) -> Optional[ReplayOutcome]:
    """Play `mp`'s live song again from `0:00`: front-insert a copy with no `ts`,
    persisted like any front insert, resolve it through the loop's prefetch, then stop
    the song. `requester` and `analytics` are the caller's. None when nothing is live,
    there is no URL to rebuild from, or the song stopped being live while the copy
    resolved. See docs/ARCHITECTURE.md#-replay."""
    current = mp.current_song
    if current is None or not current.webpage_url:
        return None
    # Before the neutralize as well as after it: the neutralize destroys the
    # next song's resolved source, and a song already gone needs neither.
    if not mp.still_live(current):
        return None
    span = trace.get_current_span()
    span.set_attribute("discord.guild_id", str(mp.guild_id))

    replay = QueueObject(
        current.webpage_url,
        current.title or "",
        requester,
        duration=current.duration_secs or None,
        uploader=current.uploader,
        thumbnail=current.thumbnail,
        analytics=analytics,
        # Classifies how the song was found, which a replay does not change,
        # and webpage_url cannot rebuild it — a Spotify link, a search and a
        # pasted link all archive as youtube.com.
        query_source=current.query_source,
        user_input=current.user_input,  # -remove matches on this
        # Renders the queue card as a replay for the window before the loop
        # dequeues it.
        is_replay=True,
    )
    # A completed prefetch bypasses the queue and would play instead of the
    # front-inserted replay — take it off the board first.
    await mp.settle_prefetch()
    # Re-check after that await.
    if not mp.still_live(current):
        return None

    position = int(current.position_secs)
    await mp.queue.put_front([replay])
    # Synchronous from this check to the slot write: a song that ended inside the
    # LPUSH has had its slot read, and the loop dequeues the copy on its own.
    if not mp.still_live(current):
        result = _why_not_live(mp)
    else:
        # Resolved through the loop's own prefetch, so the extraction, probe and
        # FFmpeg spawn are paid while the interrupted song still plays.
        # asyncio.wait bounds the wait without cancelling the task or re-raising
        # a teardown's cancellation of it — see docs/ARCHITECTURE.md#-replay.
        resolving = mp.ensure_prefetch()
        await asyncio.wait({resolving}, timeout=_REPLAY_RESOLVE_TIMEOUT)
        result = _replay_result(mp, current, replay, resolving)
        # Nothing awaits from the verdict to the stop, so the loop cannot move
        # between them; the hand-over is recorded after the stop for the same reason.
        if result is ReplayResult.REPLAYING and mp.stop_if_live(current, vc):
            replayed = mp.hand_over_to_replay(current)
            if replayed is not None:
                span.add_link(replayed, {"link.kind": "replayed_song"})
    span.set_attribute("replay.position", position)
    span.set_attribute("replay.outcome", result.value)
    span.set_attribute("replay.stopped", result is ReplayResult.REPLAYING)
    return ReplayOutcome(
        title=current.title or "Unknown", position=position, result=result
    )


def _replay_result(
    mp: MusicPlayer,
    current: YTDL,
    replay: QueueObject,
    resolving: asyncio.Task[Optional[YTDL]],
) -> ReplayResult:
    """Whether the live song may be stopped for `replay`, and why not. Only a
    copy still claimed at the head by the task in the slot is certain to play
    next: a failed resolve retires it, a neutralize hands it back rebuilt, and a
    -clear empties the deque around it."""
    if not mp.still_live(current):
        return _why_not_live(mp)
    if resolving.cancelled():
        return ReplayResult.INTERRUPTED
    head = mp.queue.peek_next()
    # By identity or as a rebuilt copy: a neutralize requeues a new QueueObject.
    if head is replay or is_replay_of(head, replay.webpage_url):
        held = (
            head is replay
            and mp.holds_prefetch(resolving)
            and mp.queue.claim_outstanding()
        )
        if held and resolving.done():
            return ReplayResult.REPLAYING
        return ReplayResult.STILL_LOADING
    if any(
        item is replay or is_replay_of(item, replay.webpage_url)
        for item in mp.queue.display_items()
    ):
        return ReplayResult.QUEUED_LATER
    failed = (
        resolving.done()
        and not resolving.cancelled()
        and resolving.exception() is None
        and resolving.result() is None
    )
    return ReplayResult.FAILED if failed else ReplayResult.DROPPED


def _why_not_live(mp: MusicPlayer) -> ReplayResult:
    """What ended the live song before -replay could stop it, read in the order
    the causes can stack: a teardown also stops, and a stop also ends it."""
    if mp.torn_down:
        return ReplayResult.TORN_DOWN
    if mp.stopped_deliberately:
        return ReplayResult.STOPPED_ELSEWHERE
    return ReplayResult.ENDED_FIRST


# What each outcome that queued a copy says. The copy plays and is recorded, so
# these keep the cooldown.
_REPLAYED = {
    ReplayResult.REPLAYING: "🔁 Replaying **{title}** from `0:00` — was at `{position}`.",
    ReplayResult.ENDED_FIRST: (
        "🔁 **{title}** ended first — queued it again from `0:00`, playing next."
    ),
    ReplayResult.STOPPED_ELSEWHERE: (
        "🔁 Another command stopped **{title}** first — its replay from `0:00` is "
        "still queued."
    ),
    ReplayResult.TORN_DOWN: (
        "🔁 The bot left before **{title}** could replay — the copy waits in the "
        "saved queue for `-resume`."
    ),
    ReplayResult.STILL_LOADING: (
        "🔁 **{title}** is taking a while to load, so it plays again from `0:00` "
        "once this play ends."
    ),
    ReplayResult.INTERRUPTED: (
        "🔁 Another command changed the queue while **{title}** was loading, so it "
        "was not replayed now."
    ),
    ReplayResult.QUEUED_LATER: (
        "🔁 Another request went ahead of **{title}**'s replay — it plays again "
        "from `0:00` once the queue reaches it."
    ),
}
_NOT_REPLAYED = {
    ReplayResult.FAILED: "Couldn't load **{title}** again, so it keeps playing.",
    ReplayResult.DROPPED: (
        "**{title}**'s replay was taken out of the queue while it loaded, so it "
        "keeps playing."
    ),
}


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-replay` — replay the live song from `0:00`, keeping the queue behind it.

    Every reply that replayed nothing refunds the guild cooldown: it exists to bound
    history churn, and a refusal writes no history. See docs/ARCHITECTURE.md#-replay.
    """
    # Every exit names itself here, refusals included: player.replay records only
    # the replays that reached the player.
    span = trace.get_current_span()
    async with background_typing(ctx):
        vc = ctx.voice_client
        song = mp.current_song
        if (
            song is None
            or not isinstance(vc, discord.VoiceClient)
            or not (vc.is_playing() or vc.is_paused())
        ):
            # A connected bot with a queue is between songs, not idle. A claimed
            # head counts: the prefetch holds the last song as a claim, not pending.
            loading = (
                isinstance(vc, discord.VoiceClient)
                and vc.is_connected()
                and (mp.queue.claim_outstanding() or not mp.queue.empty())
            )
            span.set_attribute("replay.outcome", "loading" if loading else "idle")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    "The next song is still loading, so there is nothing to replay yet."
                    if loading
                    else NOTHING_PLAYING,
                    discord.Color.orange(),
                )
            )
            return
        if int(song.position_secs) < MIN_REPLAY_POSITION_SECS:
            # Nothing to rewind, and a repeat here mints history entries nobody
            # heard.
            span.set_attribute("replay.outcome", "at_beginning")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    f"**{safe_label(song.title or 'That song', ECHO_MAX)}** is "
                    "already at the beginning.",
                    discord.Color.orange(),
                )
            )
            return
        outcome = await replay_current(
            mp,
            vc,
            # The replay is this caller's ask: the requester column and the ask-time
            # analytics both name them.
            requester=ctx.author,
            analytics=Analytics(
                queued_at=ctx.message.created_at.timestamp(), queue_position=0
            ),
        )
        if outcome is None:
            # The song stopped being live while the replay resolved — distinct from
            # the guard above, where nothing was playing.
            span.set_attribute("replay.outcome", "not_live")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    "That song is no longer playing — nothing was replayed.",
                    discord.Color.orange(),
                )
            )
            return
        span.set_attribute("replay.outcome", outcome.result.value)
        title = safe_label(outcome.title, ECHO_MAX)
        if outcome.result in (ReplayResult.FAILED, ReplayResult.DROPPED):
            # Nothing was replayed and nothing is queued, so nothing will be
            # recorded: the cooldown is handed back.
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    _NOT_REPLAYED[outcome.result].format(title=title),
                    discord.Color.orange(),
                )
            )
            return
        text = _REPLAYED[outcome.result].format(
            title=title, position=outcome.position_str
        )

        async def _react() -> None:
            # Swallowed here: the replay is committed, so a guild without Add
            # Reactions must not also be told it failed.
            with contextlib.suppress(discord.HTTPException):
                await ctx.message.add_reaction("🔁")

        # Together, so the reaction does not wait out the send. A failed send
        # still raises into the command's except.
        await asyncio.gather(
            ctx.send(embed=notice_embed(text, discord.Color.blue())), _react()
        )
