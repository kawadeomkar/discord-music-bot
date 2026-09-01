"""`-now` — what is playing, with a live progress bar where one can live."""

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import (
    notice_embed,
)


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-now` — what is playing, with a live progress bar where one can live."""
    vc = ctx.guild.voice_client if ctx.guild else None
    song = mp.current_song
    if (
        vc is not None
        and isinstance(vc, discord.VoiceClient)
        and (vc.is_playing() or vc.is_paused())
        and song is not None
    ):
        if ctx.channel.id != mp.home_channel.id:
            # Outside the player's home channel: the host never leaves home, so
            # answer HERE with a static snapshot (MusicContext's channel guard
            # keeps it unattached).
            await ctx.send(embed=mp.now_playing_snapshot(song))
            return
        # Re-host the live block at the bottom (retiring the old host) rather than
        # sending a snapshot that immediately goes stale.
        if await mp.repin_now_playing():
            return
        # Song ended between the liveness check and the repin — fall through to the
        # static/none responses instead of silence.
    if mp.play_message is not None:
        # Crash-recovery window: current_song isn't live yet but a snapshot survived
        # the restart. Static embed (no bar) until loop() starts.
        await ctx.send(embed=mp.play_message)
    else:
        await ctx.send(
            embed=notice_embed(
                "No songs are currently playing.", discord.Color.orange()
            )
        )
