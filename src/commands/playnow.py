"""`-playnow` — play this now and put the interrupted song back where it was.

Takes the cog: it runs `-play` through discord.py when there is nothing to interject.
"""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.sources import (
    unquote_argument,
)
from src.util import (
    background_typing,
)

# The stage functions are reached through the MODULE, never from-imported: a
# from-import binds them here at import time, and the seam the tests stub is
# the name on play_pipeline. Same reason youtube.py resolves _ytdlp_extract
# per call.
from src import play_pipeline

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-playnow` — interrupt what is playing, then put it back where it was.

    Falls through to -play when there is nothing live to interrupt, which also
    covers not-connected: -play joins first.
    """
    url = unquote_argument(url.strip())  # consume-rest, as -play — see there
    async with background_typing(ctx):
        mp = cog.get_mp(ctx)
        vc = ctx.voice_client
        # Nothing live to interrupt → equivalent to -play (which also
        # covers not-connected, since play joins first). Playlists enqueue
        # in full here: interjection semantics don't apply to an idle
        # player.
        if (
            mp.current_song is None
            or not isinstance(vc, discord.VoiceClient)
            or not (vc.is_playing() or vc.is_paused())
        ):
            return await ctx.invoke(cog.play, url=url)

        await play_pipeline.interject_flow(ctx, url, mp, vc, cog=cog)
