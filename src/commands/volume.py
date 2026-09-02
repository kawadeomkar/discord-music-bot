"""`-volume` — set playback level 0-100, applied from the next song."""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import notice_embed

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    pass


async def run(ctx: commands.Context, volume: str, *, mp: MusicPlayer) -> None:
    """`-volume` — set playback level 0-100, applied from the next song.

    The level is persisted per guild, and the reply says so only when the write
    landed: the help promises it survives a restart, and a level that quietly
    reverts reads as the bot ignoring the request.
    """
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
