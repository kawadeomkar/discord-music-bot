"""`-stop` — end the session: stop the song, retire the card, leave the channel."""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, *, cog: MusicBot) -> None:
    """`-stop` — end the session: stop the song, retire the card, leave the
    channel. Takes the cog because cleanup() is the registry's."""
    # No skip before cleanup: voice_client.stop()'s after callback gives the loop
    # a window to start the next song. cleanup() cancels _player first.
    vc = discord.utils.get(cog.bot.voice_clients, guild=ctx.guild)
    if vc is not None and ctx.guild is not None:
        await ctx.message.add_reaction("👋")
        await cog.cleanup(ctx.guild)
