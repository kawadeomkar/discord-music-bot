"""`-clear` — empty the queue and list what was dropped."""

import asyncio

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import (
    ECHO_ROW_MAX,
    notice_embed,
    pluralize,
    queue_message,
    safe_label,
    send_embed,
)
from src.commands._common import await_restore


async def run(ctx: commands.Context, *, mp: MusicPlayer) -> None:
    """`-clear` — empty the queue and list what was dropped. The playing song
    keeps going: only what is pending is removed."""
    if not await await_restore(ctx, mp):
        return
    cleared = await mp.queue_clear()
    if not cleared:
        await ctx.send(
            embed=notice_embed("The queue is already empty.", discord.Color.orange())
        )
        return
    description = queue_message([safe_label(t, ECHO_ROW_MAX) for t in cleared])
    await asyncio.gather(
        ctx.message.add_reaction("🗑️"),
        send_embed(
            ctx,
            f"Queue cleared — {len(cleared)} {pluralize(len(cleared), 'song')} removed",
            description,
            discord.Color.red(),
        ),
    )
