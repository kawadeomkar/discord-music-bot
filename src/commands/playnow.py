"""`-playnow` — play this now and put the interrupted song back where it was."""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.sources import unquote_argument
from src.util import background_typing

# Stage functions resolve through the module per call: the test seam is the name
# on play_pipeline, which a from-import would bind here at import time.
from src import play_pipeline

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-playnow` — interrupt what is playing, then put it back where it was.
    Falls through to -play (which joins first) when nothing live is playing;
    playlists then enqueue in full."""
    url = unquote_argument(url.strip())  # consume-rest, as -play
    async with background_typing(ctx):
        mp = cog.get_mp(ctx)
        vc = ctx.voice_client
        if (
            mp.current_song is None
            or not isinstance(vc, discord.VoiceClient)
            or not (vc.is_playing() or vc.is_paused())
        ):
            return await ctx.invoke(cog.play, url=url)

        await play_pipeline.interject_flow(ctx, url, mp, vc, cog=cog)
