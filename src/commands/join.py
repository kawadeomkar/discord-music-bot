"""`-join` — connect to the author's voice channel and report latency."""

import asyncio

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.ping import send_latency_line
from src.util import get_logger, spawn_background

log = get_logger(__name__)


async def _greet(ctx: commands.Context, latency: float) -> None:
    """`-join`'s acknowledgement, spawned so the handshake it follows is what a
    waiting `-play` waits for. Failures are logged, not raised: the bot is in the
    channel either way, and this task has no caller to report to."""
    try:
        await asyncio.gather(
            ctx.message.add_reaction("👋"), send_latency_line(ctx, latency)
        )
    except Exception as e:
        log.warning(f"join acknowledgement failed after the bot joined: {e!r}")


async def run(
    ctx: commands.Context,
    *,
    mp: MusicPlayer,
    bot_latency: float,
    tracked: set[asyncio.Task],
) -> None:
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

    # Spawned, not awaited: every cold-start -play in the guild waits out this whole
    # task before it may place, and two Discord round trips are no part of the
    # handshake it is waiting for.
    # Not ctx.invoke(ping): that runs the full ~3s dashboard on every join/cold-play
    # AND skips prepare(), losing ping's max_concurrency guard. Cheap one-liner only.
    spawn_background(_greet(ctx, bot_latency), tracked)
