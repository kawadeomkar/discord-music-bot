"""`-volume` — set playback level 0-100, applied from the next song."""

import discord
from discord.ext import commands

from src.guild_state import GuildConfig
from src.musicplayer import MusicPlayer
from src.settings import GuildSettings
from src.util import notice_embed


async def run(
    ctx: commands.Context,
    volume: str,
    *,
    mp: MusicPlayer,
    guild_settings: GuildSettings,
) -> None:
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
    assert ctx.guild is not None  # validate_commands admits guild members only
    # The commit assigns mp.volume, after the store call: a restore checking in
    # between finds the write's stamp rather than overwriting the new level.
    result = await guild_settings.write(
        ctx.guild.id, GuildConfig(volume=volume_pct / 100), player=mp
    )
    persisted = result.persisted
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
