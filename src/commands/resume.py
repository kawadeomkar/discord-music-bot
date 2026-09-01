"""`-resume` — un-pause, or rejoin and pick the saved queue back up."""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.musicplayer import RESTORE_WAIT_SECS
from src.recovery import abandon_cold_start, join_succeeded
from src.util import background_typing, notice_embed

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, *, cog: MusicBot) -> None:
    """`-resume` — un-pause, or rejoin and pick the saved queue back up.

    Takes the cog for the disconnected arm below, which tears a wedged player down
    and runs -join through discord.py.
    """
    vc = ctx.voice_client
    if not isinstance(vc, discord.VoiceClient):
        # Not in voice: the paused song went with the voice client, but the
        # queue outlives it in Redis. Join and pick that up instead.
        await _resume_disconnected(ctx, cog)
        return
    if vc.is_playing():
        await ctx.send(
            embed=notice_embed(
                "Already playing — nothing is paused.", discord.Color.orange()
            )
        )
        return
    if not vc.is_paused():
        # No queue advice here: this branch also covers the seconds
        # between two songs, where the queue is not empty at all.
        await ctx.send(embed=notice_embed("Nothing is paused.", discord.Color.orange()))
        return
    mp = cog.get_mp(ctx)
    await mp.resume(vc)
    await ctx.message.add_reaction("⏭️")
    # If the -pause confirmation hosts the block, re-host it so
    # "⏸️ Paused at…" becomes history instead of sitting beneath a
    # live, advancing bar for the rest of the song.
    await mp.rehost_np_after_resume()


async def _resume_disconnected(ctx: commands.Context, cog: MusicBot) -> None:
    """`-resume` with the bot out of voice: join the author's channel and let the
    persisted queue play again.

    A `-stop`, an eject and a crash all leave `guild:{id}:queue` intact, so there
    is something to come back to; only a crash also leaves the song that was
    playing, which restore re-queues at its position (cleanup() scrubs those
    state fields on the other two).
    """
    assert ctx.guild is not None  # validate_commands rejects DMs before this
    async with background_typing(ctx):
        mp = cog.get_mp(ctx)
        if not mp.can_rejoin_cold():
            # An eject that never reached on_voice_state_update, so cleanup never
            # ran. Rejoining around it announces a resume its wedged loop cannot
            # deliver.
            await cog.cleanup(ctx.guild)
            mp = cog.get_mp(ctx)
        # Restore first, unlike -play: there is no extraction to hide the join
        # behind, and joining first parks the bot in a channel for an empty queue.
        if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
            await ctx.send(
                embed=notice_embed(
                    "Still loading this server's saved queue — try `-resume` "
                    "again in a moment.",
                    discord.Color.orange(),
                )
            )
            return
        # Built before the gate opens, while the queue head is still the
        # restored one — the loop pops it out from under this.
        embed = mp.build_rejoin_resume_embed()
        if embed is None:
            # A failed read lands here too, with a queue it never filled. Saying
            # "nothing was left" would assert what it cannot know.
            detail = (
                "Nothing to resume — no queue was left from a previous "
                "session. Use `-play` to start one."
                if mp.store is not None and not mp.restore_read_failed
                else "Can't reach the queue store, so there is nothing to "
                "resume from. Use `-play` to start a new queue."
            )
            await ctx.send(embed=notice_embed(detail, discord.Color.orange()))
            return

        # The hold -play takes across its join: without it the head starts playing,
        # and posts its NP card, before the reply explaining the join lands.
        async with mp.defer_playback():
            try:
                await ctx.invoke(cog.join)
                joined = join_succeeded(ctx)
            except BaseException:
                # join swallows Exceptions, so an escape means its error REPORTING
                # failed, or the command was cancelled. Same wreckage, same exit.
                await abandon_cold_start(cog, ctx, mp)
                raise
            if not joined:
                # join already told the user why; nothing to add here.
                await abandon_cold_start(cog, ctx, mp)
                return
            await ctx.send(embed=embed)
