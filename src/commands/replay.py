"""`-replay` — play the live song again from its beginning."""

import asyncio
import contextlib

import discord
from discord.ext import commands
from opentelemetry import trace

from src.commands._common import NOTHING_PLAYING
from src.guild_state import Analytics
from src.musicplayer import MusicPlayer, ReplayResult
from src.util import (
    ECHO_MAX,
    background_typing,
    notice_embed,
    refund_cooldown,
    safe_label,
)


# Below this position -replay refuses: the song is at its beginning. Also bounds
# history churn — every replay writes an entry to a list LTRIMmed to
# HISTORY_CACHE_LIMIT.
MIN_REPLAY_POSITION_SECS = 1


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
        outcome = await mp.replay_current(
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
