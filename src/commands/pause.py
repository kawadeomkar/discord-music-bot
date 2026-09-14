"""`-pause` — hold the song where it is and post the exact position."""

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-pause` — hold the song where it is and post the exact position."""
    vc = ctx.voice_client
    if isinstance(vc, discord.VoiceClient) and vc.is_playing():
        await mp.pause(vc)
        await ctx.message.add_reaction("⏸️")
        embed = mp.build_pause_confirmation_embed()
        if embed is not None:
            await ctx.send(embed=embed)
