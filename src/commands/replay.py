"""`-replay` — play the last song that finished again, from its beginning."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

import discord
from discord.ext import commands
from opentelemetry import trace

from src import play_pipeline
from src.commands._common import await_restore
from src.guild_state import HistoryEntry
from src.musicplayer import MusicPlayer
from src.play_placement import Placement, PlaceStalled, PlayMode, place_stalled_notice
from src.queue_item import QueueObject
from src.util import background_typing, notice_embed, refund_cooldown

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


NOTHING_TO_REPLAY = "There's no earlier song to replay."


def last_played(
    history: Iterable[HistoryEntry],
    *,
    pending: Optional[HistoryEntry] = None,
    skip_urls: frozenset[str] = frozenset(),
) -> Optional[HistoryEntry]:
    """The newest played song with a link, passing over `skip_urls`: the live song,
    whose own fragment a -restart records, and songs parked to resume, which play
    again anyway. `pending` is a play that ended but is not in `history` yet, so it
    is the newest. `history` is oldest-first, as GuildHistory iterates."""
    for entry in (pending, *reversed(list(history))):
        if (
            entry is not None
            and entry.webpage_url
            and entry.webpage_url not in skip_urls
        ):
            return entry
    return None


def songs_to_pass_over(mp: MusicPlayer, *, live: bool) -> frozenset[str]:
    """The live song, and every song parked to resume. Without the parked ones a
    repeated -replay flips between two songs, parking each in turn."""
    urls = {item.webpage_url for item in mp.queue.display_items() if item.is_resume}
    current = mp.current_song
    if live and current is not None:
        urls.add(current.webpage_url)
    return frozenset(urls)


def pending_entry(mp: MusicPlayer) -> Optional[HistoryEntry]:
    """The ended play the loop has yet to record, as the row it will write. Without
    it a -replay right after a -skip finds the song before the skipped one."""
    song = mp.ended_unrecorded
    if song is None:
        return None
    return HistoryEntry.from_song(
        song, guild_id=mp.guild_id, message_id=0, channel_id=0
    )


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
    """`-replay` — play the last song that finished again: now, parking a live song
    to resume after it; next, when nothing is live; through `-playnext` when the bot
    is out of voice. Refusals hand back the cooldown. See docs/ARCHITECTURE.md#-replay.
    """
    span = trace.get_current_span()
    mp = cog.get_mp(ctx)
    async with background_typing(ctx):
        # The history cache is the restore's to fill, and a player this command
        # just built is still reading it.
        if not await await_restore(ctx, mp):
            span.set_attribute("replay.refused", "restore_pending")
            refund_cooldown(ctx)
            return
        vc = ctx.voice_client
        live_vc = (
            vc
            if isinstance(vc, discord.VoiceClient)
            and mp.current_song is not None
            and (vc.is_playing() or vc.is_paused())
            else None
        )
        entry = last_played(
            mp.history,
            pending=pending_entry(mp),
            skip_urls=songs_to_pass_over(mp, live=live_vc is not None),
        )
        if entry is None:
            span.set_attribute("replay.refused", "no_history")
            refund_cooldown(ctx)
            await ctx.send(
                embed=notice_embed(NOTHING_TO_REPLAY, discord.Color.orange())
            )
            return
        span.set_attribute("replay.url", entry.webpage_url)

        # Out of voice, or a join still in flight: -playnext's cold path owns the
        # join, the gate hold and the teardown on failure, and its placement is the
        # front whichever way the join race lands.
        if not ctx.voice_client or cog._plays.join_in_flight(mp.guild_id):
            span.set_attribute("replay.route", "cold_start")
            await ctx.invoke(cog.playnext, url=entry.webpage_url)
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
                span.set_attribute("replay.route", "now")
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
                span.set_attribute("replay.route", "next")
                await play_pipeline.enqueue_single(
                    ctx, qobj, mp, req, placement=Placement.NEXT, cog=cog
                )
        except PlaceStalled as stall:
            await ctx.send(
                embed=place_stalled_notice(before_the_put=stall.before_the_put)
            )
        finally:
            cog._plays.retire(req)
