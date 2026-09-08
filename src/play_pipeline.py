"""The machinery behind `-play`: resolve the input, place it, and — for an
interjection — put the interrupted song back where it was.

Three stages in the order they run: `queue_source` turns a parsed source into
something enqueueable; `enqueue_playlist` / `enqueue_single` place it and send
the confirmation; `interject_flow` is the `--now` path, shared with `-play` on
a paused song. The playlist errors and the two Resolved* shapes live here
because nothing outside this pipeline constructs them.
"""

import asyncio
from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING, Any, Optional, Union, assert_never
from collections.abc import Coroutine, Sequence

import discord
from discord.ext import commands

from src.guild_queue import QueueItem
from src.guild_state import Analytics
from src.musicplayer import MusicPlayer
from src.play_placement import Placement
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    query_source_of,
    spotify_playlist_to_ytsearch,
    timestamp_warning,
)
from src.telemetry import get_tracer
from src.util import (
    ECHO_MAX,
    ECHO_ROW_MAX,
    get_logger,
    notice_embed,
    pluralize,
    queue_message,
    safe_label,
    send_embed,
    truncate_embed_title,
)
from src.youtube import YTDL, QueueObject

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module).
    from src.musicbot import MusicBot

log = get_logger(__name__)
_tracer = get_tracer(__name__)


class PlaylistInputError(ValueError):
    """A playlist link the user can fix by editing it. A ValueError so nothing
    already catching one changes behaviour; `user_message` lets _command_error
    render actionable copy, and one base class keeps its tuple to the concept."""

    def __init__(self, log_message: str, user_message: str) -> None:
        super().__init__(log_message)
        self.user_message = user_message


class PlaylistIndexError(PlaylistInputError):
    """`index=` names a position past the end of the playlist. Both numbers are
    in the message: the user cannot fix the link without the real one."""

    def __init__(self, index: int, total: int) -> None:
        self.index = index
        self.total = total
        # A one-song playlist has no range to offer.
        fix = (
            "Drop the `&index=` from the link to queue it."
            if total == 1
            else (
                f"Pick a position from 1 to {total}, or drop the `&index=` "
                f"from the link to queue the whole playlist."
            )
        )
        super().__init__(
            f"playlist index {index} past end ({total} tracks)",
            f"That link starts the playlist at **#{index}**, but the playlist "
            f"only has **{total} {pluralize(total, 'song')}** — nothing was "
            f"queued.\n\n{fix}",
        )


class EmptyPlaylistError(PlaylistInputError):
    """A playlist that resolved to nothing queueable. Vague about the cause:
    yt-dlp drops unavailable entries before this code sees them, so "empty" and
    "every video is private" are indistinguishable here."""

    def __init__(self) -> None:
        super().__init__(
            "playlist resolved to no tracks",
            "That playlist has no songs I can queue — it may be empty, or every "
            "video in it may be private or unavailable.",
        )


@dataclass
class ResolvedSpotifyPlaylist:
    """A Spotify playlist resolved to track titles, still needing per-title
    YouTube search resolution."""

    titles: list[str]


@dataclass
class ResolvedYoutubePlaylist:
    """A YouTube playlist resolved to playable QueueObjects. `skipped` is how
    many leading tracks the URL's `index=` dropped, for the enqueue embed alone:
    `tracks` is already sliced."""

    tracks: list[QueueObject]
    skipped: int = 0


def _apply_playlist_index(
    tracks: list[QueueObject],
    index: Optional[int],
) -> tuple[list[QueueObject], int]:
    """Drop the tracks ahead of YouTube's 1-based `index=` (a share link copied
    mid-playlist carries the position it was copied at), returning what is left
    and how many went. An index past the end raises rather than queueing
    nothing. The empty-playlist guard lives here so both callers get it."""
    if not tracks:
        raise EmptyPlaylistError
    if index is None or index <= 1:
        return tracks, 0
    if index > len(tracks):
        raise PlaylistIndexError(index, len(tracks))
    kept = tracks[index - 1 :]
    dropped = index - 1
    # Positions were assigned before this slice; the dropped tracks never
    # enqueue, so rebase or every kept track records N-1 too deep.
    for track in kept:
        track.analytics = replace(
            track.analytics,
            queue_position=track.analytics.queue_position - dropped,
        )
    return kept, dropped


def _apply_playlist_timestamp(tracks: list[QueueObject], source: YTSource) -> None:
    """Start the first queued track at the link's `t=` offset, only when that
    track is the `v=` video the link names — without a matching `index=` the
    queue starts at track 1, usually a different song."""
    if not source.ts or not source.video_id or not tracks:
        return
    # Substring, not equality: yt_playlist takes the entry's own `url` when it
    # has one, so the shape is not guaranteed.
    if source.video_id in tracks[0].webpage_url:
        tracks[0].ts = source.ts


@_tracer.start_as_current_span("bot.queue_source")
def plays_after_note(
    mp: MusicPlayer, voice_client: Optional[discord.VoiceProtocol]
) -> str:
    """What a `--next` confirmation says about when the song plays: the song it
    waits behind, and — since `--next` does not interject a paused song — that
    playback is still paused."""
    current = mp.current_song
    if current is None:
        # A claim with no current_song is loop() between taking the prefetch
        # result and starting it: the insert lands behind that song.
        if mp.queue.claim_outstanding():
            return "Plays after the song starting now."
        return "Nothing is playing, so it starts now."
    note = f"Plays after **{current.title or 'the current song'}**."
    if isinstance(voice_client, discord.VoiceClient) and voice_client.is_paused():
        note += " Playback is paused — `-resume` to carry on."
    return note


def with_queue_position(item: QueueItem, position: int) -> QueueItem:
    """Re-mint one item's `queue_position`. A QueueObject is stamped in place, a
    frozen YTSource returns a copy — use the return value either way."""
    analytics = replace(item.analytics, queue_position=position)
    if isinstance(item, QueueObject):
        item.analytics = analytics
        return item
    return replace(item, analytics=analytics)


def collection_note(
    url: str, queued: int, *, returns: str = "", head_playing: bool
) -> str:
    """What a `-play` that queued a whole collection tells the user: how many
    tracks landed, when the interrupted song returns, and the `-remove` undo.
    `head_playing` changes the undo: a playing song has no queue object for
    -remove to reach, so it names -skip."""
    undo = (
        "the queued ones back out; the one playing needs `-skip`."
        if head_playing
        else "the whole playlist back out."
    )
    return (
        f"\n\nQueued **{queued}** {pluralize(queued, 'song')} from the playlist."
        f"{returns}\nNot what you wanted? `-remove {safe_label(url, ECHO_MAX)}` "
        f"takes {undo}"
    )


def front_insert_depth(mp: MusicPlayer) -> int:
    """Ask-time `queue_position` for a song going to the FRONT: it waits behind the
    playing song and nothing else. An outstanding claim counts as that song even
    while current_song is None (loop() between taking the prefetch result and
    starting it). ±1 like enqueue_depth(): two `--next` in a row both record 1."""
    return 1 if mp.current_song is not None or mp.queue.claim_outstanding() else 0


async def _warm_front_track(
    tracks: Sequence[QueueItem], placement: Placement, *, cog: MusicBot
) -> None:
    """Warm the stream URL of a playlist's head when it is about to play. Bulk
    enqueues pass prefetch=False, and under `--next` queue_put_next killed the
    loop's one-ahead prefetch, so the head is left with no warm at all. A lazy
    Spotify entry has no URL yet; it resolves at dequeue."""
    if placement is not Placement.NEXT or not tracks:
        return
    head = tracks[0]
    if isinstance(head, QueueObject):
        await YTDL.prefetch_stream(head, redis=cog.redis)


async def send_playing_next(
    ctx: commands.Context,
    qobj: QueueObject,
    *,
    note: str,
    reaction: str = "👍",
) -> None:
    """The "Playing next" confirmation, for the two paths that make that promise
    — `-play --next`, and the interjection whose song ended before it could be
    interrupted. `note` is the only difference: why this song is next.
    """
    await asyncio.gather(
        send_embed(
            ctx,
            truncate_embed_title(f"▶️ Playing next: {qobj.title}"),
            f"Requested by: [{ctx.author.mention}]\n{note}",
            discord.Color.blue(),
            thumbnail=qobj.thumbnail,
        ),
        ctx.message.add_reaction(reaction),
    )


async def queue_source(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    *,
    analytics: Analytics,
    origin: str,
    cog: MusicBot,
) -> Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]:
    """Resolve a parsed source into something enqueueable. `analytics` is the
    command's ask-time head value; playlist tracks derive per-track positions
    from it. `origin` is the raw command argument, carried onto every item —
    for a collection the link, not the per-track search its expansion made."""
    if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
        # Titles, not QueueObjects — enqueue_playlist mints the YTSources.
        return ResolvedSpotifyPlaylist(await cog._require_spotify().playlist(source.id))
    if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
        if source.list_id is None:
            raise ValueError("YTSource with type=PLAYLIST must have list_id set")
        tracks = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
        )
        tracks, skipped = _apply_playlist_index(tracks, source.index)
        _apply_playlist_timestamp(tracks, source)
        return ResolvedYoutubePlaylist(tracks, skipped=skipped)
    ts: Optional[int] = None
    search: str
    if isinstance(source, SpotifySource):
        search = await cog._require_spotify().track(source.id)
    elif isinstance(source, YTSource):
        search = source.ytsearch or source.url or ""
        ts = source.ts
    elif isinstance(source, SoundcloudSource):
        search = source.url
    else:
        assert_never(source)
    return await YTDL.yt_source(
        ctx.author,
        search,
        ts=ts,
        redis=cog.redis,
        query_source=query_source_of(source),
        analytics=analytics,
        user_input=origin,
    )


@_tracer.start_as_current_span("bot.enqueue_playlist")
async def enqueue_playlist(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    qobj: Union[ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist],
    mp: MusicPlayer,
    *,
    analytics: Analytics,
    origin: str,
    placement: Placement = Placement.TAIL,
    cog: MusicBot,
) -> None:
    """Queue a resolved playlist and notify the channel."""
    # A playlist front-inserts in full, in order, under either flag. NEXT uses
    # queue_put_next: the loop's prefetch holds a claim a plain front-insert
    # lands behind. COLD_FRONT has no prefetch — the gate is shut.
    enqueue = {
        Placement.TAIL: mp.queue_put,
        Placement.COLD_FRONT: mp.queue_put_front,
        Placement.NEXT: mp.queue_put_next,
    }[placement]
    warning = timestamp_warning(source)
    warning_line = f"\n\n{warning}" if warning else ""
    # "Queued playlist" on its own reads as "at the back".
    next_suffix = " — plays next" if placement is Placement.NEXT else ""
    if isinstance(qobj, ResolvedSpotifyPlaylist):
        titles = qobj.titles
        qobjs_yt = spotify_playlist_to_ytsearch(
            titles, analytics=analytics, origin=origin
        )
        log.info(f"ytsearch qobjs: {qobjs_yt}")
        shown_titles = queue_message([safe_label(t, ECHO_ROW_MAX) for t in titles])
        await asyncio.gather(
            send_embed(
                ctx,
                "Queued playlist" + next_suffix,
                f"Requested by: [{ctx.author.mention}]\n\n{shown_titles}{warning_line}",
                discord.Color.blue(),
            ),
            enqueue(qobjs_yt, prefetch=False),
            _warm_front_track(qobjs_yt, placement, cog=cog),
            ctx.message.add_reaction("👍"),
        )
    else:
        # HACK: this assert stands in for a correlation the signature cannot
        # express — a ResolvedYoutubePlaylist always arrives with a YTSource,
        # but they are separate parameters. `python -O` strips it, leaving the
        # attribute reads unguarded. Fix: have the Resolved*Playlist dataclasses
        # carry their own source.
        assert isinstance(source, YTSource)
        playlist_url = source.playlist_url
        tracks = qobj.tracks
        count = len(tracks)
        log.info(f"yt playlist track count: {count}")
        # Stated: only the `index=` in the user's own URL explains fewer songs.
        skipped_line = (
            f"Starting at #{qobj.skipped + 1} — skipped {qobj.skipped} "
            f"earlier {pluralize(qobj.skipped, 'song')}\n"
            if qobj.skipped
            else ""
        )
        shown_titles = queue_message(
            [safe_label(q.title, ECHO_ROW_MAX) for q in islice(tracks, 10)]
        )
        await asyncio.gather(
            send_embed(
                ctx,
                f"Queued playlist — {count} {pluralize(count, 'song')}{next_suffix}",
                f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n"
                f"{skipped_line}\n{shown_titles}{warning_line}",
                discord.Color.blue(),
            ),
            enqueue(tracks, prefetch=False),
            _warm_front_track(tracks, placement, cog=cog),
            ctx.message.add_reaction("👍"),
        )


@_tracer.start_as_current_span("bot.enqueue_single")
async def enqueue_single(
    ctx: commands.Context,
    qobj: QueueObject,
    mp: MusicPlayer,
    *,
    placement: Placement = Placement.TAIL,
    note: str = "",
    warning: Optional[str] = None,
) -> None:
    """Queue one song and confirm. `warning` is about what the user typed, so
    every exit sends it — the confirmation embed is conditional, the warning is
    not."""
    vc = ctx.voice_client
    if placement is Placement.COLD_FRONT:
        # The "Est. playing at" embed below would be wrong: a restored queue is
        # non-empty but its entries sit BEHIND this song. The resume notice
        # names the song starting now (the gate is shut, so no NP block does).
        # Built before the insert, while the queue holds only restored entries.
        resume_notice = mp.build_resume_notice_embed(qobj)
        coros: list[Coroutine[Any, Any, Any]] = [
            mp.queue_put_front(qobj),
            ctx.message.add_reaction("👍"),
        ]
        if resume_notice is not None:
            coros.append(ctx.send(embed=resume_notice))
        if warning is not None:
            coros.append(ctx.send(embed=notice_embed(warning, discord.Color.orange())))
        await asyncio.gather(*coros)
        log.info(f"play (front) qsize: {mp.queue.qsize()}")
        return

    if placement is Placement.NEXT:
        # No "Est. playing at": the ETA walk seeds from the current song's
        # FULL duration as a proxy for what is left of it, which is badly
        # wrong for the very next slot. It names the song it waits behind.
        next_coros: list[Coroutine[Any, Any, Any]] = [
            mp.queue_put_next(qobj),
            send_playing_next(ctx, qobj, note=plays_after_note(mp, vc)),
        ]
        if warning is not None:
            next_coros.append(
                ctx.send(embed=notice_embed(warning, discord.Color.orange()))
            )
        await asyncio.gather(*next_coros)
        log.info(f"play (next) qsize: {mp.queue.qsize()}")
        return

    # A note is the only word the user gets about tracks queued behind this
    # one, so an empty queue does not suppress the field.
    should_show_queued = (
        bool(note)
        or mp.queue.qsize() > 0
        or (isinstance(vc, discord.VoiceClient) and vc.is_playing())
    )
    # Awaited ahead of the reply: its shape depends on whether this song became
    # the queue head, which the put decides.
    await asyncio.gather(mp.queue_put(qobj), ctx.message.add_reaction("👍"))
    log.info(f"play qsize: {mp.queue.qsize()}")

    if should_show_queued:
        # When this song IS the head, the block's "Up next" card and the
        # confirmation are one card printed twice. Re-host the live block and
        # let its card be the confirmation — dedicated, because a response host
        # with no own embeds strip-edits to a blank message on retire.
        if mp.queue.peek_next() is qobj and await mp.repin_now_playing():
            # What the card would have carried: the note and the warning.
            said = "\n\n".join(text for text in (note, warning) if text)
            if said:
                color = discord.Color.orange() if warning else discord.Color.blue()
                await ctx.send(embed=notice_embed(said, color))
            return
        await ctx.send(
            embed=mp.build_queued_song_embed(qobj, note=note, warning=warning)
        )
        return
    if warning is not None:
        # The song starts now and the NP card speaks for it, so the warning
        # needs its own message.
        await ctx.send(embed=notice_embed(warning, discord.Color.orange()))


async def _resolve_interjection_source(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    *,
    origin: str,
    cog: MusicBot,
) -> tuple[QueueObject, list[QueueItem]]:
    """Resolve an interjection's input into (head, everything behind it). The
    head must be a resolved QueueObject to interrupt with; the tail may hold
    lazy YTSources. The interrupted song returns after the whole playlist, and
    one `-remove <the link>` takes it all back out. `origin` is the raw command
    argument — for a playlist the link, not the generated titles."""
    # Ask-time analytics: the snowflake time, depth 0 for the head. Tracks behind
    # it derive 1, 2, … from this base; the caller re-mints the head's own depth.
    analytics = Analytics(
        queued_at=ctx.message.created_at.timestamp(), queue_position=0
    )
    if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
        titles = await cog._require_spotify().playlist(source.id)
        if not titles:
            raise EmptyPlaylistError()
        yts = spotify_playlist_to_ytsearch(titles, analytics=analytics, origin=origin)
        # Only the head is resolved — it has to be playable to interrupt with.
        # The rest stay lazy searches resolved at dequeue, so a 100-track album
        # does not pay 100 searches up front.
        head = await YTDL.yt_source(
            ctx.author,
            yts[0].ytsearch or "",
            redis=cog.redis,
            query_source=query_source_of(yts[0]),
            analytics=analytics,
            user_input=origin,
        )
        return head, list(yts[1:])
    if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
        tracks = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
        )
        # Indexed here too: `--now` on a link copied mid-playlist starts at the
        # track the user was looking at, not the playlist's first.
        tracks, skipped = _apply_playlist_index(tracks, source.index)
        _apply_playlist_timestamp(tracks, source)
        if skipped:
            await ctx.send(
                embed=notice_embed(
                    f"Starting at **#{skipped + 1}** — skipped {skipped} "
                    f"earlier {pluralize(skipped, 'song')}.",
                    discord.Color.orange(),
                )
            )
        return tracks[0], list(tracks[1:])
    qobj = await queue_source(ctx, source, analytics=analytics, origin=origin, cog=cog)
    assert isinstance(qobj, QueueObject)
    return qobj, []


@_tracer.start_as_current_span("bot.interject_flow")
async def interject_flow(
    ctx: commands.Context,
    url: str,
    mp: MusicPlayer,
    vc: discord.VoiceClient,
    *,
    resume_paused: bool = True,
    require_paused: bool = False,
    cog: MusicBot,
) -> None:
    """Resolve `url` to one song, interrupt what is playing, and report.

    Shared by `-play --now` and by `-play` on a paused song; they differ in
    resume_paused (`--now` restores paused-in → paused-out, `-play` brings it
    back playing). require_paused re-reads the pause state after resolution:
    `-play` interjects only because the song is paused, so a `-resume` landing
    during the 1–4s extraction removes the reason and the track is appended.
    """
    source = parse_input(url)
    qobj, follow_on = await _resolve_interjection_source(
        ctx, source, origin=url, cog=cog
    )
    # The head only: `interjected` is attribution, which song cut the line.
    qobj.interjected = True

    # Warm the stream-URL cache before interrupting, or a miss at dequeue puts
    # seconds of dead air between the interrupt and the new song. Awaited: the
    # current song plays through the wait. Also back-fills duration/thumbnail.
    #
    # The head only: warming N tracks would be N concurrent extractions minting
    # URLs that expire before playback reaches them.
    await YTDL.prefetch_stream(qobj, redis=cog.redis)

    if require_paused and not vc.is_paused():
        # Resumed during the resolve: append rather than interrupt a song the
        # user just chose to keep playing, and re-mint the depth now that the
        # queue moved.
        qobj.interjected = False
        # An ordinary append now, behind the whole queue, so replace the 0
        # minted for the interjection. Read here: the queue moved during the
        # resolve.
        depth = mp.enqueue_depth()
        qobj.analytics = replace(qobj.analytics, queue_position=depth)
        note = ""
        if follow_on:
            # The head went to the tail, so these follow it there. Their
            # ask-time depths were minted for a front insert and are re-minted
            # from the head's: play_history keeps whatever number is on them.
            follow_on = [
                with_queue_position(item, depth + offset)
                for offset, item in enumerate(follow_on, start=1)
            ]
            await mp.queue_put(follow_on, prefetch=False)
            note = collection_note(url, len(follow_on) + 1, head_playing=False)
        await enqueue_single(
            ctx, qobj, mp, note=note, warning=timestamp_warning(source)
        )
        return

    outcome = await mp.interject(
        qobj, vc, resume_paused=resume_paused, follow_on=follow_on
    )
    if outcome is None:
        # The song ended during the resolve. Front-insert directly: the user
        # asked for "now", the window can be seconds long with songs queued
        # behind, and re-invoking -play would re-resolve (and for a playlist
        # enqueue every track after the first-track notice). It interrupted
        # nothing, so the marker comes off.
        qobj.interjected = False
        # interject() also returns None when the loop moved on to a
        # DIFFERENT song, which this insert waits behind. One, never the
        # queue depth: it goes to the front.
        qobj.analytics = replace(qobj.analytics, queue_position=front_insert_depth(mp))
        # queue_put_next: the embed below promises "play next", and the
        # loop's prefetch holds a claim a bare front-insert would land behind.
        # interject() returned None without reaching its own neutralize.
        # prefetch=False — the stream URL was warmed above.
        await mp.queue_put_next([qobj, *follow_on], prefetch=False)
        note = "The song being interrupted already ended — queued to play next instead."
        if follow_on:
            # Nothing was interrupted, so the head is QUEUED rather than
            # playing: it counts, and -remove reaches it.
            note += collection_note(url, len(follow_on) + 1, head_playing=False)
        await send_playing_next(ctx, qobj, note=note, reaction="⏯️")
        return

    if outcome.resume_position is None:
        desc = (
            f"**{outcome.interrupted_title}** was nearly finished and will not resume."
        )
    elif outcome.returns_paused:
        # returns_paused, not was_paused: with resume_paused=False a paused
        # song comes back playing.
        desc = (
            f"**{outcome.interrupted_title}** will return paused at "
            f"`{outcome.resume_position_str}`."
        )
    elif outcome.was_paused:
        desc = (
            f"**{outcome.interrupted_title}** was paused at "
            f"`{outcome.resume_position_str}` and will resume from there."
        )
    else:
        desc = (
            f"**{outcome.interrupted_title}** will resume at "
            f"`{outcome.resume_position_str}`."
        )
    if follow_on:
        # The interrupted song waits behind the whole playlist, so the reply says
        # so and names the undo (`-remove <the link>` matches user_input).
        desc += collection_note(
            url,
            len(follow_on),
            returns=(
                f" **{outcome.interrupted_title}** returns after the last of them."
                if outcome.resume_position is not None
                else ""
            ),
            head_playing=True,
        )
    await asyncio.gather(
        send_embed(
            ctx,
            truncate_embed_title(f"▶️ Playing now: {qobj.title}"),
            f"Requested by: [{ctx.author.mention}]\n{desc}",
            discord.Color.blue(),
            thumbnail=qobj.thumbnail,
        ),
        ctx.message.add_reaction("⏯️"),
    )
