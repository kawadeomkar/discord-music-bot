"""Helpers shared by more than one command body."""

import discord
from discord.ext import commands

from src.musicplayer import RESTORE_WAIT_SECS, MusicPlayer
from src.util import (
    notice_embed,
)


async def await_restore(ctx: commands.Context, mp: MusicPlayer) -> bool:
    """Wait for the guild's saved queue to be replayed into memory, telling the user
    and answering False when it does not arrive in time.

    Every command below that REBUILDS the Redis mirror from the in-memory deque has
    to clear this first: rebuilding from a deque the restore has not filled writes an
    empty queue over the saved one and deletes the persisted entries. validate_commands
    only requires the AUTHOR in voice, so a cold player reaches these commands.
    """
    if await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
        return True
    await ctx.send(
        embed=notice_embed(
            "Still loading this server's saved queue — try again in a moment.",
            discord.Color.orange(),
        )
    )
    return False
