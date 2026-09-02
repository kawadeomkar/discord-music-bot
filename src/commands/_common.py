"""Helpers shared by more than one command body."""

from typing import Optional

import discord
from discord.ext import commands

from src.musicplayer import RESTORE_WAIT_SECS, MusicPlayer
from src.play_placement import PlayRequest
from src.util import (
    ECHO_MAX,
    ECHO_ROW_MAX,
    notice_embed,
    pluralize,
    queue_message,
    safe_label,
)


def echo(text: str, limit: int = ECHO_MAX) -> str:
    """A needle safe to put in an embed — see util.safe_label."""
    return safe_label(text, limit)


def cleared_title(cleared: int, dropped: int) -> str:
    """`-clear`'s heading, counting both halves: with nothing queued, the dropped
    requests ARE the work the command did."""
    songs = f"{cleared} {pluralize(cleared, 'song')} removed"
    if not dropped:
        return f"Queue cleared — {songs}"
    requests = f"{dropped} play {pluralize(dropped, 'request')} dropped"
    return f"Cleared — {songs}, {requests}" if cleared else f"Cleared — {requests}"


def dropped_request_field(
    dropped: list[PlayRequest],
) -> Optional[list[tuple[str, str, bool]]]:
    """The resolving play requests a command dropped, as one embed field. None when
    it dropped none, which is what send_embed takes for no field at all."""
    if not dropped:
        return None
    return [
        (
            f"{len(dropped)} play {pluralize(len(dropped), 'request')} dropped",
            queue_message([safe_label(r.query, ECHO_ROW_MAX) for r in dropped]),
            False,
        )
    ]


async def await_restore(ctx: commands.Context, mp: MusicPlayer) -> bool:
    """Wait for the guild's saved queue to be replayed into memory, telling the user
    and answering False when it does not arrive in time.

    Every command below that REBUILDS the Redis mirror from the in-memory deque has
    to clear this first: rebuilding from a deque the restore has not filled writes an
    empty queue over the saved one and deletes the persisted entries. validate_commands
    only requires the AUTHOR in voice, so a cold player reaches these commands.
    """
    if await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
        return True
    await ctx.send(
        embed=notice_embed(
            "Still loading this server's saved queue — try again in a moment.",
            discord.Color.orange(),
        )
    )
    return False
