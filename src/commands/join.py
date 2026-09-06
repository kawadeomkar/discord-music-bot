"""`-join` — connect to the author's voice channel and report latency."""

import asyncio

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.ping import send_latency_line


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

    # Release the loop so a persisted queue resumes. No-op while -play holds
    # the gate: it front-inserts first, then opens.
    mp.open_playback_gate()

    await asyncio.gather(
        ctx.message.add_reaction("👋"),
        # Not ctx.invoke(ping): the full dashboard, minus its max_concurrency guard.
        send_latency_line(ctx, bot_latency),
    )
