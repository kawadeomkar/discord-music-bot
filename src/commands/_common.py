"""Helpers shared by more than one command body."""

import discord
from discord.ext import commands

from src.musicplayer import RESTORE_WAIT_SECS, MusicPlayer
from src.util import notice_embed


async def await_restore(ctx: commands.Context, mp: MusicPlayer) -> bool:
    """Wait for the saved queue to be replayed into memory; tell the user and
    answer False when it does not arrive. Every command that REBUILDS the Redis
    mirror from the deque clears this first, or a cold player's rebuild writes
    an empty queue over the saved one."""
    if await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
        return True
    await ctx.send(
        embed=notice_embed(
            "Still loading this server's saved queue — try again in a moment.",
            discord.Color.orange(),
        )
    )
    return False
