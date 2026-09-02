"""`-shuffle` — randomly reorder what is pending, leaving the playing song."""

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import (
    background_typing,
    notice_embed,
)
from src.commands._common import await_restore


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-shuffle` — randomly reorder what is pending, leaving the playing song."""
    if not await await_restore(ctx, mp):
        return
    async with background_typing(ctx):
        await ctx.send(
            embed=notice_embed("Please wait... shuffling", discord.Color.blue())
        )
        msg = await mp.queue_shuffle()
        await ctx.message.add_reaction("🔀")
        await ctx.send(embed=notice_embed(msg, discord.Color.blue()))
