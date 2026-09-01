"""`-remove` — drop queued songs matching what the user typed, and
say which ones went."""

from typing import Optional

import discord
from discord.ext import commands

from src.guild_queue import QueueItem, RemoveMode, RemoveOutcome
from src.musicplayer import MusicPlayer
from src.sources import QUERY_SOURCE_SEARCH
from src.util import (
    ECHO_MAX,
    ECHO_ROW_MAX,
    EMBED_FIELD_LIMIT,
    notice_embed,
    pluralize,
    queue_message,
    safe_label,
    send_embed,
    truncate,
)
from src.youtube import QueueObject
from src.commands._common import await_restore


# The most dropped positions worth spelling out; past this the list says nothing
# the count above it did not.
_MAX_SHOWN_POSITIONS = 60


def _echo(text: str, limit: int = ECHO_MAX) -> str:
    """A needle safe to put in an embed — see util.safe_label."""
    return safe_label(text, limit)


def _removed_label(item: QueueItem) -> str:
    """A removed queue item's name for the reply, as MusicPlayer.queue_clear
    renders it: `YTSource` has no title, so an unresolved Spotify-playlist track
    would otherwise show as `?`."""
    if isinstance(item, QueueObject):
        return item.title or "?"
    return (item.ytsearch or item.url or "?").removeprefix("ytsearch:")


def _field(value: str) -> str:
    """An embed field value that cannot 400 the send. The callers below build from
    lists whose length is the user's to choose, and the send happens AFTER the
    queue has been mutated."""
    return truncate(value, EMBED_FIELD_LIMIT)


def _matched_label(outcome: RemoveOutcome, needle: str) -> str:
    """How the removal matched, for the reply's "Matched" field. An origin match
    names which of the user's own inputs did it, since one argument can take out a
    whole playlist."""
    # Not wrapped in a code span: inside one Discord renders safe_label's
    # backslashes literally, so `-remove foo_bar` comes back as `foo\_bar`.
    shown = _echo(needle)
    if outcome.mode is not RemoveMode.ORIGIN:
        return shown
    kinds = {item.query_source for item in outcome.removed if item.query_source}
    # Only when every removed item agrees — a mixed set has no one kind to name.
    kind = kinds.pop() if len(kinds) == 1 else ""
    them = "them" if len(outcome.removed) > 1 else "it"
    if kind == QUERY_SOURCE_SEARCH:
        return f"{shown} — the search you queued {them} with"
    return f"{shown} — the {kind + ' ' if kind else ''}link you queued {them} with"


async def run(ctx: commands.Context, needle: Optional[str], *, mp: MusicPlayer) -> None:
    """`-remove` — drop every queued song matching a link or the text it was queued
    with, then report the positions that went and the queue that is left."""
    if needle is None:
        await ctx.send(
            embed=notice_embed(
                "`-remove <link or search text>` — removes every queued "
                "song that matches. Give it the YouTube link from the "
                "**Now Playing** card, or the search text or link you "
                "queued with; a collection link removes every track it "
                "added.",
                discord.Color.blue(),
            )
        )
        return
    if not await await_restore(ctx, mp):
        return
    outcome = await mp.queue_remove(needle)
    positions = outcome.positions
    if not positions:
        await send_embed(
            ctx,
            "",
            f"No queued songs found matching: {_echo(needle)}",
            discord.Color.red(),
        )
        return
    count = len(positions)
    noun = pluralize(count, "song")
    pos_label = pluralize(count, "Position")
    # Capped by count: one -remove of a collection link drops as many
    # positions as the collection had, and a raw join passes the 1024-char
    # field limit at 227 of them — a 400 for a removal that already
    # happened.
    shown = positions[:_MAX_SHOWN_POSITIONS]
    pos_str = ", ".join(str(p) for p in shown)
    if len(positions) > len(shown):
        pos_str += f", …and {len(positions) - len(shown)} more"
    await send_embed(
        ctx,
        f"Removed {count} {noun} from the queue",
        "",
        discord.Color.orange(),
        fields=[
            ("Matched", _field(_matched_label(outcome, needle)), False),
            (f"{pos_label} removed", _field(pos_str), False),
            # Titles, like -clear reports: one argument can take out a whole
            # playlist, and there is no undo, so a bare count is not enough
            # to tell whether it took what the user meant.
            (
                "Songs",
                _field(
                    queue_message(
                        [
                            _echo(_removed_label(i), ECHO_ROW_MAX)
                            # Sliced before the echo: queue_message keeps 10.
                            for i in outcome.removed[:10]
                        ]
                    )
                ),
                False,
            ),
        ],
    )
    await ctx.send(embed=mp.queue_embed())
    await ctx.message.add_reaction("🗑️")
