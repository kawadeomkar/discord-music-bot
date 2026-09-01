"""`-join` — connect to the author's voice channel and report latency."""

import asyncio
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.ping import send_latency_line

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    pass


async def run(ctx: commands.Context, *, mp: MusicPlayer, bot_latency: float) -> None:
    """`-join` — connect to the author's voice channel and report latency."""
    assert isinstance(ctx.author, discord.Member) and ctx.author.voice is not None
    assert ctx.guild is not None
    channel = ctx.author.voice.channel
    assert channel is not None

    if not ctx.voice_client:
        await channel.connect(timeout=10.0)
    vc = ctx.voice_client
    if isinstance(vc, discord.VoiceClient) and vc.channel != channel:
        await vc.move_to(channel)
    await ctx.guild.change_voice_state(channel=channel, self_mute=False, self_deaf=True)

    if mp.store is not None and isinstance(ctx.channel, discord.TextChannel):
        await mp.store.set_connection(channel.id, ctx.channel.id)

    # Voice is up — release the loop so a persisted queue resumes. No-op while -play
    # holds the gate: it front-inserts its song first, then opens.
    mp.open_playback_gate()

    await asyncio.gather(
        ctx.message.add_reaction("👋"),
        # Not ctx.invoke(ping): that runs the full ~3s dashboard on every
        # join/cold-play AND skips prepare(), losing ping's max_concurrency guard.
        # Cheap one-liner only.
        send_latency_line(ctx, bot_latency),
    )
