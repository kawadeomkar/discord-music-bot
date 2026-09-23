"""`-pause` — hold the song where it is, and say so when there is nothing to."""

import discord
from discord.ext import commands

from src.commands._common import NOTHING_PLAYING
from src.musicplayer import MusicPlayer
from src.util import notice_embed


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-pause` — hold the song where it is, and say so when there is nothing to.
    It always answers: a silent no-op left the user unable to tell "the bot
    ignored me" from "the bot is not running"."""
    vc = ctx.voice_client
    if isinstance(vc, discord.VoiceClient) and vc.is_playing():
        await mp.pause(vc, by=ctx.author)
        await ctx.message.add_reaction("⏸️")
        # The paused card rides in the block, so the reply IS the block: a
        # dedicated re-pin, which puts it at the bottom of the channel and leaves
        # nothing behind to strip when the next host takes over.
        await mp.repin_now_playing()
        return
    # A paused song is not "nothing playing" — it is loaded, positioned and
    # resumable — so it gets its own line, mirroring what -resume says about an
    # already-playing one. Neither line carries queue advice: the other covers the
    # seconds between two songs, where the queue is not empty at all.
    notice = (
        "Already paused — `-resume` to carry on."
        if isinstance(vc, discord.VoiceClient) and vc.is_paused()
        else NOTHING_PLAYING
    )
    await ctx.send(embed=notice_embed(notice, discord.Color.orange()))
