"""`-now` — what is playing, with a live progress bar where one can live."""

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import notice_embed


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
            # The host never leaves home; answer here with a static snapshot.
            await ctx.send(embed=mp.now_playing_snapshot(song))
            return
        # Re-host the live block at the bottom rather than send a stale snapshot.
        if await mp.repin_now_playing():
            return
        # Song ended between the check and the repin: fall through.
    if mp.play_message is not None:
        # Crash-recovery window: a snapshot survived the restart, no bar yet.
        await ctx.send(embed=mp.play_message)
    else:
        await ctx.send(
            embed=notice_embed(
                "No songs are currently playing.", discord.Color.orange()
            )
        )
