"""`-jump` — skip ahead to a queue position. Not implemented yet."""

import discord
from discord.ext import commands

from src.util import notice_embed


async def run(ctx: commands.Context) -> None:
    """`-jump` — a stub."""
    # TODO: Implement -jump or remove it from the command list. The help text
    # advertises it while the body only replies "currently in development".
    await ctx.send(embed=notice_embed("currently in development", discord.Color.blue()))
