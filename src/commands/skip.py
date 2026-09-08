"""`-skip` — stop the current song so the next one starts."""

import asyncio
from typing import Any, Optional
from collections.abc import Coroutine

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import INLINE_TITLE_MAX, fmt_duration, notice_embed, safe_label


async def run(ctx: commands.Context, *, mp: Optional[MusicPlayer]) -> None:
    """`-skip` — stop the current song so the next one starts. `mp` is whatever
    the guild ALREADY has: this command must not manufacture a player, and one
    lookup keeps the deliberate-stop mark and the paused read on one object."""
    vc = ctx.voice_client
    if not isinstance(vc, discord.VoiceClient):
        return
    # is_playing() is False while paused.
    if not (vc.is_playing() or vc.is_paused()):
        return

    # Before stop(), which clears current_song. Primitives, not the object —
    # the player thread calls cleanup() on it.
    skipped_title: Optional[str] = None
    skipped_position = ""
    if vc.is_paused() and mp is not None:
        song = mp.current_song
        if song is not None:
            # Rendered in bold beside a code span; a title can close either.
            skipped_title = safe_label(song.title or "", INLINE_TITLE_MAX)
            # position_secs is frozen while paused: the exact leave point.
            skipped_position = fmt_duration(int(song.position_secs))

    # Before vc.stop(): a skip inside ffmpeg's startup window otherwise looks
    # like a stream that never opened.
    if mp is not None:
        mp.note_deliberate_stop()
    vc.stop()

    coros: list[Coroutine[Any, Any, Any]] = []
    if not ctx.invoked_parents:
        coros.append(ctx.message.add_reaction("⏭"))
    if skipped_title is not None:
        # A paused song gives no audible cue that it was skipped.
        coros.append(
            ctx.send(
                embed=notice_embed(
                    f"⏭ Skipped **{skipped_title}** — was paused at "
                    f"`{skipped_position}`.",
                    discord.Color.blue(),
                )
            )
        )
    if coros:
        await asyncio.gather(*coros)
