"""`-clear` — empty the queue and list what was dropped."""

import asyncio
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.play_placement import play_key
from src.util import (
    ECHO_ROW_MAX,
    notice_embed,
    queue_message,
    safe_label,
    send_embed,
)
from src.commands._common import await_restore, cleared_title, dropped_request_field


if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, *, mp: MusicPlayer, cog: MusicBot) -> None:
    """`-clear` — empty the queue and list what was dropped. The playing song
    keeps going: only what is pending is removed."""
    if not await await_restore(ctx, mp):
        return
    cleared = await mp.queue_clear()
    # Right after the clear, with no await between: every request still
    # resolving fails its generation check at the insert and reports
    # itself; this names them here too.
    dropped = cog._plays.inflight(play_key(ctx), "clear")
    if not cleared and not dropped:
        await ctx.send(
            embed=notice_embed("The queue is already empty.", discord.Color.orange())
        )
        return
    description = (
        queue_message([safe_label(t, ECHO_ROW_MAX) for t in cleared])
        if cleared
        else "The queue was already empty."
    )
    await asyncio.gather(
        ctx.message.add_reaction("🗑️"),
        send_embed(
            ctx,
            cleared_title(len(cleared), len(dropped)),
            description,
            discord.Color.red(),
            fields=dropped_request_field(dropped),
        ),
    )
