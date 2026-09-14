"""`-volume` — set playback level 0-100, applied from the next song."""

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import notice_embed


async def run(ctx: commands.Context, volume: str, *, mp: MusicPlayer) -> None:
    """`-volume` — set playback level 0-100, applied from the next song. The
    reply claims persistence only when the write landed."""
    try:
        volume_pct = int(volume)
    except ValueError:
        await ctx.send(
            embed=notice_embed(
                "Volume must be a number between 0 and 100",
                discord.Color.red(),
            )
        )
        return
    if not 0 <= volume_pct <= 100:
        await ctx.send(
            embed=notice_embed("Volume must be between 0 and 100", discord.Color.red())
        )
        return
    mp.volume = volume_pct / 100
    persisted = False
    if mp.store is not None:
        persisted = await mp.store.set_volume(mp.volume)
    durability = (
        "It is saved for this server."
        if persisted
        else "⚠️ It could not be saved (Redis is unavailable), so it "
        "applies until the bot restarts."
    )
    await ctx.send(
        embed=notice_embed(
            f"Set volume to {volume_pct}% (takes effect on next song). " + durability,
            discord.Color.blue(),
        )
    )
