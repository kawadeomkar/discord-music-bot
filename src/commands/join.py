"""`-join` — connect to the author's voice channel and report latency."""

import asyncio

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.ping import send_latency_line
from src.recovery import discord_holds_voice_state, join_succeeded, leave_voice
from src.util import get_logger, notice_embed, spawn_background

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

    parked = ctx.voice_client
    if parked is not None and not join_succeeded(ctx):
        # Registered, but its handshake never landed: it answers truthy and would
        # skip the connect below. Unregistered first so this join makes a fresh one.
        # Forced only while Discord still has the bot in a channel — see
        # docs/ARCHITECTURE.md#voice-teardown.
        log.warning(
            f"join replacing an unconnected voice client in guild {ctx.guild.id}"
        )
        await leave_voice(parked, force=discord_holds_voice_state(ctx.guild))
    if not ctx.voice_client:
        try:
            await channel.connect(timeout=10.0)
        except asyncio.CancelledError:
            # Op 4 may be out with its VOICE_STATE_UPDATE still in flight: the bot
            # member has no voice state yet, so cleanup()'s disconnect would go
            # unforced and send no clear. Forced here, where the connect is known
            # to have started.
            if ctx.voice_client is not None:
                await leave_voice(ctx.voice_client, force=True)
            raise
    # Ahead of move_to, which on an unconnected client sends op 4 and then waits
    # 30s for a state nothing drives; and ahead of set_connection, so on_ready
    # cannot recover a guild this never joined. The next -join replaces the client.
    if not join_succeeded(ctx):
        log.warning(f"join found an unconnected voice client in guild {ctx.guild.id}")
        await ctx.send(
            embed=notice_embed(
                "Couldn't finish connecting to your voice channel. Run `-join` "
                "again to retry with a fresh connection.",
                discord.Color.red(),
            )
        )
        return
    vc = ctx.voice_client
    if isinstance(vc, discord.VoiceClient) and vc.channel != channel:
        await vc.move_to(channel)
    await ctx.guild.change_voice_state(channel=channel, self_mute=False, self_deaf=True)

    if mp.store is not None and isinstance(ctx.channel, discord.TextChannel):
        await mp.store.set_connection(channel.id, ctx.channel.id)

    # Release the loop so a persisted queue resumes. No-op while -play holds
    # the gate: it front-inserts first, then opens.
    mp.open_playback_gate()

    # Spawned, not awaited: every cold-start -play in the guild waits out this whole
    # task before it may place, and two Discord round trips are no part of the
    # handshake it is waiting for.
    # Not ctx.invoke(ping): that runs the full ~3s dashboard on every join/cold-play
    # AND skips prepare(), losing ping's max_concurrency guard. Cheap one-liner only.
    spawn_background(_greet(ctx, bot_latency), tracked)
