"""`-queue` — the pending songs, as the player renders them."""

from discord.ext import commands

from src.musicplayer import MusicPlayer


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-queue` — the pending songs, as the player renders them."""
    await ctx.send(embed=mp.queue_embed())
