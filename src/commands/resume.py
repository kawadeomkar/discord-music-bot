"""`-resume` — un-pause, or rejoin and pick the saved queue back up."""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.musicplayer import RESTORE_WAIT_SECS
from src.recovery import abandon_cold_start, join_succeeded
from src.util import background_typing, notice_embed

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, *, cog: MusicBot) -> None:
    """`-resume` — un-pause, or rejoin and pick the saved queue back up. Takes
    the cog for the disconnected arm, which tears a wedged player down and runs
    -join through discord.py."""
    vc = ctx.voice_client
    if not isinstance(vc, discord.VoiceClient):
        # The paused song went with the voice client; the queue outlives it in Redis.
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
        # No queue advice: this also covers the seconds between two songs.
        await ctx.send(embed=notice_embed("Nothing is paused.", discord.Color.orange()))
        return
    mp = cog.get_mp(ctx)
    await mp.resume(vc)
    await ctx.message.add_reaction("⏭️")
    # A -pause confirmation hosting the block would otherwise sit beneath a
    # live, advancing bar for the rest of the song.
    await mp.rehost_np_after_resume()


async def _resume_disconnected(ctx: commands.Context, cog: MusicBot) -> None:
    """`-resume` with the bot out of voice: join the author's channel and let
    the persisted queue play again. -stop, an eject and a crash all leave
    `guild:{id}:queue` intact; only a crash also leaves the playing song,
    which restore re-queues at its position."""
    assert ctx.guild is not None  # validate_commands rejects DMs before this
    async with background_typing(ctx):
        mp = cog.get_mp(ctx)
        if not mp.can_rejoin_cold():
            # An eject that never reached on_voice_state_update; its wedged
            # loop cannot deliver the resume.
            await cog.cleanup(ctx.guild)
            mp = cog.get_mp(ctx)
        # Restore BEFORE joining, unlike -play: no extraction to hide the join
        # behind, and joining first parks the bot for an empty queue.
        if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
            await ctx.send(
                embed=notice_embed(
                    "Still loading this server's saved queue — try `-resume` "
                    "again in a moment.",
                    discord.Color.orange(),
                )
            )
            return
        # Before the gate opens: the loop pops the head out from under this.
        embed = mp.build_rejoin_resume_embed()
        if embed is None:
            # A failed read lands here too; "nothing was left" would assert
            # what it cannot know.
            detail = (
                "Nothing to resume — no queue was left from a previous "
                "session. Use `-play` to start one."
                if mp.store is not None and not mp.restore_read_failed
                else "Can't reach the queue store, so there is nothing to "
                "resume from. Use `-play` to start a new queue."
            )
            await ctx.send(embed=notice_embed(detail, discord.Color.orange()))
            return

        # Without the hold the head starts, and posts its NP card, before the
        # reply explaining the join lands.
        async with mp.defer_playback():
            try:
                await ctx.invoke(cog.join)
                joined = join_succeeded(ctx)
            except BaseException:
                # join swallows Exceptions: an escape is a failed error report
                # or a cancellation.
                await abandon_cold_start(cog, ctx, mp)
                raise
            if not joined:
                # join already told the user why.
                await abandon_cold_start(cog, ctx, mp)
                return
            await ctx.send(embed=embed)
