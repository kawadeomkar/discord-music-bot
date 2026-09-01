"""`-jump` — skip ahead to a queue position. Not implemented yet."""

import discord
from discord.ext import commands

from src.util import (
    notice_embed,
)


async def run(ctx: commands.Context) -> None:
    """`-jump` — a stub. See the marker below."""
    # TODO: Implement -jump or remove it from the command list.
    # The help text advertises it while the body only replies "currently in
    # development", so the bot promises a feature it does not have.
    # Implementing it is a drain/rotate over GuildQueue, shaped like run_remove().
    await ctx.send(embed=notice_embed("currently in development", discord.Color.blue()))
