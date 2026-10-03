"""`-restart` — play the live song again from its beginning."""

import asyncio
import contextlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Union

import discord
from discord.ext import commands
from opentelemetry import trace

from src.commands._common import NOTHING_PLAYING
from src.guild_queue import is_restart_of
from src.queue_item import QueueObject
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
from src.youtube import YTDL

_tracer = get_tracer(__name__)


# Below this position -restart refuses: the song is at its beginning. Also bounds
# history churn — every restart writes an entry to a list LTRIMmed to
# HISTORY_CACHE_LIMIT.
MIN_RESTART_POSITION_SECS = 1

# How long -restart waits for its copy to resolve. Past it the live song is left
# playing: the resolve continues, and the copy it holds plays when the song ends.
_RESTART_RESOLVE_TIMEOUT = 8.0


class RestartResult(Enum):
    """How a -restart that front-inserted its copy ended. Only RESTARTING stopped the
    live song; see docs/ARCHITECTURE.md#-restart."""

    RESTARTING = "restarting"
    # The song ended on its own before the stop. The copy is next.
    ENDED_FIRST = "ended_first"
    # A -skip or --now stopped the song before the stop. The copy is still queued.
    STOPPED_ELSEWHERE = "stopped_elsewhere"
    # The player was torn down. The copy waits in the saved queue for -resume.
    TORN_DOWN = "torn_down"
    # The resolve outlived _RESTART_RESOLVE_TIMEOUT. The copy plays after the song.
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


@dataclass(frozen=True, slots=True, kw_only=True)
class RestartOutcome:
    """What restart_current() did, for -restart's confirmation wording."""

    title: str
    position: int  # where the live song was when -restart ran
    result: RestartResult = RestartResult.RESTARTING

    @property
    def stopped(self) -> bool:
        return self.result is RestartResult.RESTARTING

    @property
    def position_str(self) -> str:
        return fmt_duration(self.position)


@_tracer.start_as_current_span("player.restart")
async def restart_current(
    mp: MusicPlayer,
    vc: discord.VoiceClient,
    *,
    requester: Union[discord.User, discord.Member],
    queued_at: float,
    queue_position: int,
) -> Optional[RestartOutcome]:
    """Play `mp`'s live song again from `0:00`: front-insert a copy with no `ts`,
    persisted like any front insert, resolve it through the loop's prefetch, then stop
    the song. `requester`, `queued_at` and `queue_position` are the caller's. None
    when nothing is live, there is no URL to rebuild from, or the song stopped being
    live while the copy resolved. See docs/ARCHITECTURE.md#-restart."""
    current = mp.current_song
    if current is None or not current.webpage_url:
        return None
    # Before the neutralize as well as after it: the neutralize destroys the
    # next song's resolved source, and a song already gone needs neither.
    if not mp.still_live(current):
        return None
    span = trace.get_current_span()
    span.set_attribute("discord.guild_id", str(mp.guild_id))

    # The same song asked for again: the origin comes along off current.queued —
    # query_source, which webpage_url cannot rebuild (a Spotify link, a search and
    # a pasted link all archive as youtube.com), and user_input, what -remove
    # matches on — while everything the interrupted play accumulated is dropped.
    restart = replace(
        current.queued,
        webpage_url=current.webpage_url,
        title=current.title or "",
        requester=requester,
        duration=current.duration_secs or None,
        uploader=current.uploader,
        thumbnail=current.thumbnail,
        queued_at=queued_at,
        queue_position=queue_position,
        # Renders the queue card as a restart for the window before the loop
        # dequeues it.
        is_restart=True,
        # A fresh play, from the top: no offset, no interjection flags, no start
        # stamp, not the card the fragment it copies is holding, and the full
        # retry budget — the attempts the live play spent are its own.
        ts=None,
        persisted=True,
        interjected=False,
        is_resume=False,
        start_paused=False,
        played_at=0.0,
        stream_attempts=0,
        failed_format_ids=frozenset(),
        np_card=None,
    )
    # A completed prefetch bypasses the queue and would play instead of the
    # front-inserted restart — take it off the board first.
    await mp.settle_prefetch()
    # Re-check after that await.
    if not mp.still_live(current):
        return None

    position = int(current.position_secs)
    await mp.queue.put_front([restart])
    # Synchronous from this check to the slot write: a song that ended inside the
    # LPUSH has had its slot read, and the loop dequeues the copy on its own.
    if not mp.still_live(current):
        result = _why_not_live(mp)
    else:
        # Resolved through the loop's own prefetch, so the extraction, probe and
        # FFmpeg spawn are paid while the interrupted song still plays.
        # asyncio.wait bounds the wait without cancelling the task or re-raising
        # a teardown's cancellation of it — see docs/ARCHITECTURE.md#-restart.
        resolving = mp.ensure_prefetch()
        await asyncio.wait({resolving}, timeout=_RESTART_RESOLVE_TIMEOUT)
        result = _restart_result(mp, current, restart, resolving)
        # Nothing awaits from the verdict to the stop, so the loop cannot move
        # between them; the hand-over is recorded after the stop for the same reason.
        if result is RestartResult.RESTARTING and mp.stop_if_live(current, vc):
            restarted = mp.hand_over_to_restart(current)
            if restarted is not None:
                span.add_link(restarted, {"link.kind": "restarted_song"})
    span.set_attribute("restart.position", position)
    span.set_attribute("restart.outcome", result.value)
    span.set_attribute("restart.stopped", result is RestartResult.RESTARTING)
    return RestartOutcome(
        title=current.title or "Unknown", position=position, result=result
    )


def _restart_result(
    mp: MusicPlayer,
    current: YTDL,
    restart: QueueObject,
    resolving: asyncio.Task[Optional[YTDL]],
) -> RestartResult:
    """Whether the live song may be stopped for `restart`, and why not. Only a
    copy still claimed at the head by the task in the slot is certain to play
    next: a failed resolve retires it, a neutralize hands it back rebuilt, and a
    -clear empties the deque around it."""
    if not mp.still_live(current):
        return _why_not_live(mp)
    if resolving.cancelled():
        return RestartResult.INTERRUPTED
    head = mp.queue.peek_next()
    # By identity or as a rebuilt copy: a neutralize requeues a new QueueObject.
    if head is restart or is_restart_of(head, restart.webpage_url):
        held = (
            head is restart
            and mp.holds_prefetch(resolving)
            and mp.queue.claim_outstanding()
        )
        if held and resolving.done():
            return RestartResult.RESTARTING
        return RestartResult.STILL_LOADING
    if any(
        item is restart or is_restart_of(item, restart.webpage_url)
        for item in mp.queue.display_items()
    ):
        return RestartResult.QUEUED_LATER
    failed = (
        resolving.done()
        and not resolving.cancelled()
        and resolving.exception() is None
        and resolving.result() is None
    )
    return RestartResult.FAILED if failed else RestartResult.DROPPED


def _why_not_live(mp: MusicPlayer) -> RestartResult:
    """What ended the live song before -restart could stop it, read in the order
    the causes can stack: a teardown also stops, and a stop also ends it."""
    if mp.torn_down:
        return RestartResult.TORN_DOWN
    if mp.stopped_deliberately:
        return RestartResult.STOPPED_ELSEWHERE
    return RestartResult.ENDED_FIRST


# What each outcome that queued a copy says. The copy plays and is recorded, so
# these keep the cooldown.
_RESTARTED = {
    RestartResult.RESTARTING: "🔁 Restarting **{title}** from `0:00` — was at `{position}`.",
    RestartResult.ENDED_FIRST: (
        "🔁 **{title}** ended first — queued it again from `0:00`, playing next."
    ),
    RestartResult.STOPPED_ELSEWHERE: (
        "🔁 Another command stopped **{title}** first — its restart from `0:00` is "
        "still queued."
    ),
    RestartResult.TORN_DOWN: (
        "🔁 The bot left before **{title}** could restart — the copy waits in the "
        "saved queue for `-resume`."
    ),
    RestartResult.STILL_LOADING: (
        "🔁 **{title}** is taking a while to load, so it plays again from `0:00` "
        "once this play ends."
    ),
    RestartResult.INTERRUPTED: (
        "🔁 Another command changed the queue while **{title}** was loading, so it "
        "was not restarted now."
    ),
    RestartResult.QUEUED_LATER: (
        "🔁 Another request went ahead of **{title}**'s restart — it plays again "
        "from `0:00` once the queue reaches it."
    ),
}
_NOT_RESTARTED = {
    RestartResult.FAILED: "Couldn't load **{title}** again, so it keeps playing.",
    RestartResult.DROPPED: (
        "**{title}**'s restart was taken out of the queue while it loaded, so it "
        "keeps playing."
    ),
}


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-restart` — restart the live song from `0:00`, keeping the queue behind it.

    Every reply that restarted nothing refunds the guild cooldown: it exists to bound
    history churn, and a refusal writes no history. See docs/ARCHITECTURE.md#-restart.
    """
    # Every exit names itself here, refusals included: player.restart records only
    # the restarts that reached the player.
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
            span.set_attribute("restart.outcome", "loading" if loading else "idle")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    "The next song is still loading, so there is nothing to restart yet."
                    if loading
                    else NOTHING_PLAYING,
                    discord.Color.orange(),
                )
            )
            return
        if int(song.position_secs) < MIN_RESTART_POSITION_SECS:
            # Nothing to rewind, and a repeat here mints history entries nobody
            # heard.
            span.set_attribute("restart.outcome", "at_beginning")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    f"**{safe_label(song.title or 'That song', ECHO_MAX)}** is "
                    "already at the beginning.",
                    discord.Color.orange(),
                )
            )
            return
        outcome = await restart_current(
            mp,
            vc,
            # The restart is this caller's ask: the requester column and the ask-time
            # analytics both name them.
            requester=ctx.author,
            queued_at=ctx.message.created_at.timestamp(),
            queue_position=0,
        )
        if outcome is None:
            # The song stopped being live while the restart resolved — distinct from
            # the guard above, where nothing was playing.
            span.set_attribute("restart.outcome", "not_live")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    "That song is no longer playing — nothing was restarted.",
                    discord.Color.orange(),
                )
            )
            return
        span.set_attribute("restart.outcome", outcome.result.value)
        title = safe_label(outcome.title, ECHO_MAX)
        if outcome.result in (RestartResult.FAILED, RestartResult.DROPPED):
            # Nothing was restarted and nothing is queued, so nothing will be
            # recorded: the cooldown is handed back.
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(
                    _NOT_RESTARTED[outcome.result].format(title=title),
                    discord.Color.orange(),
                )
            )
            return
        text = _RESTARTED[outcome.result].format(
            title=title, position=outcome.position_str
        )

        async def _react() -> None:
            # Swallowed here: the restart is committed, so a guild without Add
            # Reactions must not also be told it failed.
            with contextlib.suppress(discord.HTTPException):
                await ctx.message.add_reaction("🔁")

        # Together, so the reaction does not wait out the send. A failed send
        # still raises into the command's except.
        await asyncio.gather(
            ctx.send(embed=notice_embed(text, discord.Color.blue())), _react()
        )
