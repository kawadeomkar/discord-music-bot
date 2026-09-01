"""`-ping` — the live dependency-health dashboard."""

from typing import TYPE_CHECKING

from discord.ext import commands

from src.ping import run_health_dashboard

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, *, cog: MusicBot) -> None:
    """`-ping` — probe every dependency and render the result, editing as answers land.

    Takes the cog because the probes' inputs are its handles. Reached only by a
    top-level -ping — the internal join/play path uses send_latency_line.
    """
    await run_health_dashboard(
        ctx,
        bot_latency=cog.bot.latency,
        redis=cog.redis,
        spotify=cog.spotify,
        # The startup validation outcome, not just "is a client configured": lets
        # the Spotify row say *why* the source is unusable without spending a
        # doomed API call (see probe_spotify).
        spotify_status=cog.spotify_status,
        archive=cog.history_archive,
        debug_suffix=cog.debug_suffix(ctx),
    )
