"""`-replay` — play the last song that finished again, from its beginning."""

from typing import TYPE_CHECKING, Optional, Union

import discord
from discord.ext import commands
from opentelemetry import trace

from src import play_pipeline
from src.commands import play as play_cmd
from src.commands._common import await_restore
from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.play_placement import Placement, PlaceStalled, PlayMode
from src.queue_item import QueueObject
from src.util import background_typing, notice_embed

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


NOTHING_TO_REPLAY = "Nothing has finished playing yet, so there is nothing to replay."


def last_played(history: GuildHistory, *, skip_url: str = "") -> Optional[HistoryEntry]:
    """The newest history entry with a link to play, passing over `skip_url`: the
    live song, whose own fragment a -restart or -skip may already have recorded.
    Cache-only, so no Redis round trip; the restore fills the cache."""
    for entry in reversed(history):
        if entry.webpage_url and entry.webpage_url != skip_url:
            return entry
    return None


def replay_item(
    entry: HistoryEntry,
    *,
    requester: Union[discord.User, discord.Member],
    queued_at: float,
) -> QueueObject:
    """A queue item for `entry`, already resolved: the history row carries every
    display field, so nothing is searched and the stream cache is keyed by its
    `webpage_url`. A fresh ask by `requester`, from `0:00`; `query_source` stays
    with the song — it classifies how the song was found, and webpage_url cannot
    rebuild it."""
    return QueueObject(
        webpage_url=entry.webpage_url,
        title=entry.title or entry.webpage_url,
        requester=requester,
        # What a `-play <link>` of it stores, so `-remove <link>` takes it back out.
        user_input=entry.webpage_url,
        duration=entry.duration_secs or None,
        uploader=entry.uploader or None,
        thumbnail=entry.thumbnail or None,
        queued_at=queued_at,
        queue_position=0,
        query_source=entry.query_source,
    )


async def run(ctx: commands.Context, *, cog: MusicBot) -> None:
    """`-replay` — play the newest song in history again: now, parking a live song
    to resume after it; next, when nothing is live; through `-play`'s cold path when
    the bot is out of voice. See docs/ARCHITECTURE.md#-replay."""
    span = trace.get_current_span()
    mp = cog.get_mp(ctx)
    async with background_typing(ctx):
        # The history cache is the restore's to fill, and a player this command
        # just built is still reading it.
        if not await await_restore(ctx, mp):
            span.set_attribute("replay.outcome", "restore_pending")
            return
        vc = ctx.voice_client
        current = mp.current_song
        live_vc = (
            vc
            if isinstance(vc, discord.VoiceClient)
            and current is not None
            and (vc.is_playing() or vc.is_paused())
            else None
        )
        skip_url = current.webpage_url if live_vc is not None and current else ""
        entry = last_played(mp.history, skip_url=skip_url)
        if entry is None:
            span.set_attribute("replay.outcome", "no_history")
            await ctx.send(
                embed=notice_embed(NOTHING_TO_REPLAY, discord.Color.orange())
            )
            return
        span.set_attribute("replay.url", entry.webpage_url)

        # Out of voice, or a join still in flight: -play's cold path owns the join,
        # the gate hold and the teardown on failure. It resolves the link, which the
        # voice handshake runs alongside.
        if not ctx.voice_client or cog._plays.join_in_flight(mp.guild_id):
            span.set_attribute("replay.outcome", "cold_start")
            await play_cmd.run(ctx, entry.webpage_url, cog=cog)
            return

        qobj = replay_item(
            entry,
            requester=ctx.author,
            queued_at=ctx.message.created_at.timestamp(),
        )
        # Registered like a -play, so it takes the place lock its siblings take,
        # and -clear, -stop or -remove can still drop it until it lands.
        req = cog._plays.register(
            ctx,
            query=entry.webpage_url,
            mp=mp,
            mode=PlayMode.NOW if live_vc is not None else PlayMode.NEXT,
        )
        try:
            if live_vc is not None:
                span.set_attribute("replay.outcome", "now")
                # resume_paused=False: a paused song comes back playing, on -play's
                # precedent — the command asked for music.
                await play_pipeline.interject_resolved(
                    ctx,
                    qobj,
                    mp,
                    live_vc,
                    req,
                    origin=entry.webpage_url,
                    resume_paused=False,
                    lead="🔁 Replaying",
                    cog=cog,
                )
            else:
                span.set_attribute("replay.outcome", "next")
                await play_pipeline.enqueue_single(
                    ctx, qobj, mp, req, placement=Placement.NEXT, cog=cog
                )
        except PlaceStalled as stall:
            await ctx.send(
                embed=play_cmd.place_stalled_notice(before_the_put=stall.before_the_put)
            )
        finally:
            cog._plays.retire(req)
