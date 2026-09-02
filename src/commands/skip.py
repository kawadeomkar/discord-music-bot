"""`-skip` — stop the current song so the next one starts."""

import asyncio
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Coroutine

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.util import fmt_duration, notice_embed

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    pass


async def run(ctx: commands.Context, *, mp: Optional[MusicPlayer]) -> None:
    """`-skip` — stop the current song so the next one starts.

    `mp` is whatever the guild ALREADY has, never a freshly built player: this
    command must not manufacture one, and one lookup keeps the deliberate-stop mark
    and the paused read on the same object — cog_before_invoke can rebuild a player
    mid-command.
    """
    vc = ctx.voice_client
    if not isinstance(vc, discord.VoiceClient):
        return
    # is_playing() is False while paused, so gating on it alone made -skip a total
    # no-op on a paused song — not even the reaction.
    if not (vc.is_playing() or vc.is_paused()):
        return

    # Capture before stop(): the loop's song-end bookkeeping clears current_song, and
    # the notice must name the song actually skipped. Primitives, not the object —
    # the player thread calls cleanup() on it.
    skipped_title: Optional[str] = None
    skipped_position = ""
    if vc.is_paused() and mp is not None:
        song = mp.current_song
        if song is not None:
            skipped_title = song.title
            # position_secs is frozen while paused: the exact leave point.
            skipped_position = fmt_duration(int(song.position_secs))

    # Before vc.stop(): a skip inside ffmpeg's startup window otherwise looks exactly
    # like a stream that never opened.
    if mp is not None:
        mp.note_deliberate_stop()
    vc.stop()

    coros: list[Coroutine[Any, Any, Any]] = []
    if not ctx.invoked_parents:
        coros.append(ctx.message.add_reaction("⏭"))
    if skipped_title is not None:
        # A paused song makes no sound, so stopping it gives no audible cue — unlike
        # an ordinary skip, where the music changing is it.
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
