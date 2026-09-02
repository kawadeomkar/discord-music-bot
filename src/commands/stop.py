"""`-stop` — end the session: stop the song, retire the card, leave the channel."""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.play_placement import play_key
from src.util import send_embed
from src.commands._common import dropped_request_field


if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, *, cog: MusicBot) -> None:
    """`-stop` — end the session: stop the song, retire the card, leave the channel.

    Takes the cog because cleanup() is the registry's, not one player's: it pops the
    player, cancels its tasks and clears the persisted connection so on_ready skips
    recovery for this guild.
    """
    # Don't skip before cleanup: skip fires voice_client.stop(), whose after callback
    # (play_next.set) gives the loop a window to start the next song before it is
    # cancelled. cleanup() cancels _player first and disconnect() stops the audio
    # subprocess, so no skip is needed.
    vc = discord.utils.get(cog.bot.voice_clients, guild=ctx.guild)
    if vc is not None and ctx.guild is not None:
        await ctx.message.add_reaction("👋")
        await cog.cleanup(ctx.guild)
    # After the teardown, with no await between, so no request places into
    # a player that no longer exists. Unconditional: a cold-start -play may
    # be resolving before there is a client to find.
    dropped = cog._plays.inflight(play_key(ctx), "stop")
    if dropped:
        await send_embed(
            ctx,
            "Stopped",
            "Play requests still resolving were dropped.",
            discord.Color.red(),
            fields=dropped_request_field(dropped),
        )
