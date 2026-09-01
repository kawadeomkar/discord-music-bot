"""The machinery behind `-play`: resolve the input, place it, and — for an
interjection — put the interrupted song back where it was.

Three stages, in the order they run. `queue_source` turns a parsed source into
something enqueueable; `enqueue_playlist` / `enqueue_single` place it and send the
confirmation; `interject_flow` is the `--now` path, shared with `-play` on a paused
song. The playlist errors and the two Resolved* shapes live here because nothing
outside this pipeline constructs them — musicbot.py imports PlaylistInputError back
for the one error-embed branch that renders its user_message.
"""

import asyncio
from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING, Optional, Union, assert_never
from collections.abc import Sequence

import discord
from discord.ext import commands

from src.guild_queue import QueueItem
from src.guild_state import Analytics
from src.musicplayer import InterjectOutcome, MusicPlayer
from src.play_placement import Placement, PlayRequest, ResolveMode
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
    build_embed,
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
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot

log = get_logger(__name__)
_tracer = get_tracer(__name__)


class PlaylistInputError(ValueError):
    """A playlist link the user can fix by editing it, rather than a bot failure.

    Subclasses ValueError so nothing that already catches one changes behaviour,
    and carries user_message so _command_error renders actionable copy instead of
    a "ValueError: …" line. One base class so _command_error's tuple names the
    concept rather than growing an entry per case.
    """

    def __init__(self, log_message: str, user_message: str) -> None:
        super().__init__(log_message)
        self.user_message = user_message


class PlaylistIndexError(PlaylistInputError):
    """`index=` names a position past the end of the playlist. Both numbers are
    in the message: the user cannot fix the link without knowing the real one."""

    def __init__(self, index: int, total: int) -> None:
        self.index = index
        self.total = total
        # A one-song playlist has no range to offer — "from 1 to 1" reads as a
        # bug in the message rather than advice.
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
    """A playlist that resolved to nothing queueable. Deliberately vague about
    which cause: yt-dlp drops unavailable entries before this code sees them, so
    "empty" and "every video is private" are indistinguishable here."""

    def __init__(self) -> None:
        super().__init__(
            "playlist resolved to no tracks",
            "That playlist has no songs I can queue — it may be empty, or every "
            "video in it may be private or unavailable.",
        )


@dataclass
class ResolvedSpotifyPlaylist:
    """A Spotify playlist resolved to track titles — still needs per-title
    YouTube search resolution before it can be queued."""

    titles: list[str]


@dataclass
class ResolvedYoutubePlaylist:
    """A YouTube playlist already resolved to playable QueueObjects.

    `skipped` is how many leading tracks the URL's `index=` dropped, carried for
    the enqueue embed alone: tracks is already sliced, so nothing downstream has
    to know the playlist started anywhere but its own first entry.
    """

    tracks: list[QueueObject]
    skipped: int = 0


def _apply_playlist_index(
    tracks: list[QueueObject],
    index: Optional[int],
) -> tuple[list[QueueObject], int]:
    """Drop the tracks ahead of YouTube's 1-based `index=`, returning what is left
    and how many went. A share link copied mid-playlist carries the position it was
    copied at, so playing it starts there rather than back at track 1.

    An index past the end raises PlaylistIndexError rather than queueing nothing:
    the user named a position this playlist does not have, and an empty enqueue
    reports success. The empty-playlist guard lives here too, so both callers
    get it.
    """
    if not tracks:
        raise EmptyPlaylistError
    if index is None or index <= 1:
        return tracks, 0
    if index > len(tracks):
        raise PlaylistIndexError(index, len(tracks))
    kept = tracks[index - 1 :]
    dropped = index - 1
    # Positions were assigned at construction, before this slice. Rebase to
    # kept-relative: the dropped tracks never enqueue, so an &index=N link would
    # otherwise record every kept track N-1 too deep.
    for track in kept:
        track.analytics = replace(
            track.analytics,
            queue_position=track.analytics.queue_position - dropped,
        )
    return kept, dropped


def _apply_playlist_timestamp(tracks: list[QueueObject], source: YTSource) -> None:
    """Start the first queued track at the link's `t=` offset — but only when
    that track is the video the link actually names.

    A playlist link carries one offset and N tracks, so the offset belongs to the
    `v=` video alone. Without a matching `index=` the queue starts at track 1,
    which is usually a different song, and seeking that one would be wrong. The
    offset was previously parsed and then dropped on this path.
    """
    if not source.ts or not source.video_id or not tracks:
        return
    # Substring, not equality: yt_playlist takes the entry's own `url` when it
    # has one, so the shape it built is not guaranteed. An 11-char video id
    # matching some other part of a YouTube URL is not a case that arises.
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


def _rebase_positions(
    tracks: Sequence[QueueItem], minted_from: int, base: int
) -> Sequence[QueueItem]:
    """Move a resolved collection's `queue_position`s from one head depth to
    another; a no-op when the head has not moved, so the O(N) copy (milliseconds at
    5,000 tracks) runs only when another request placed in between, not under the
    lock on every enqueue."""
    if base == minted_from:
        return tracks
    return [with_queue_position(t, base + offset) for offset, t in enumerate(tracks)]


def _head_depth(mp: MusicPlayer, placement: Placement) -> int:
    """`queue_position` for the first song an insert adds, read at the insert: the
    slot it actually takes. A cold start takes 0 — it plays ahead of everything at the
    moment it lands, and concurrent cold starts each record 0 and each are right, the
    same drift enqueue_depth() carries against a queue the loop keeps moving."""
    if placement is Placement.COLD_FRONT:
        return 0
    if placement is Placement.NEXT:
        return front_insert_depth(mp)
    return mp.enqueue_depth()


async def _reply(
    ctx: commands.Context, embeds: Sequence[discord.Embed], reaction: str = "👍"
) -> None:
    """Confirm a placement that already happened. return_exceptions: the song is IN
    the queue, and a missing Add Reactions permission or a deleted invoking message
    must not render "Failed to queue song" over a song that plays."""
    results = await asyncio.gather(
        ctx.message.add_reaction(reaction),
        *(ctx.send(embed=embed) for embed in embeds),
        return_exceptions=True,
    )
    for failed in (r for r in results if isinstance(r, BaseException)):
        log.warning(f"Confirmation leg failed after the song was queued: {failed!r}")


def front_insert_depth(mp: MusicPlayer) -> int:
    """Ask-time `queue_position` for a song going to the FRONT: it waits behind the
    playing song and nothing else. An outstanding claim counts as that song even
    while current_song is None (loop() between taking the prefetch result and
    starting it). ±1 like enqueue_depth(): two `--next` in a row both record 1."""
    return 1 if mp.current_song is not None or mp.queue.claim_outstanding() else 0


@_tracer.start_as_current_span("bot.warm_front_track")
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


def playing_next_embed(
    ctx: commands.Context, qobj: QueueObject, *, note: str
) -> discord.Embed:
    """The "Playing next" confirmation for `-play --next` and for an interjection
    whose song ended first; `note` says why the song is next. With nothing
    playing the title reads "Playing now", so it agrees with the note."""
    starts_now = note.startswith("Nothing is playing")
    lead = "▶️ Playing now" if starts_now else "▶️ Playing next"
    return build_embed(
        truncate_embed_title(f"{lead}: {qobj.title}"),
        f"Requested by: [{ctx.author.mention}]\n{note}",
        discord.Color.blue(),
        thumbnail=qobj.thumbnail,
    )


async def queue_source(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    *,
    analytics: Analytics,
    origin: str,
    mode: ResolveMode,
    pool_slot: Optional[asyncio.Semaphore] = None,
    cog: MusicBot,
) -> Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]:
    """Resolve a parsed URL/search source into something enqueueable: a
    ResolvedSpotifyPlaylist (titles still needing per-title YouTube resolution),
    a ResolvedYoutubePlaylist (already resolved), or a bare QueueObject.

    `analytics` is the command's ask-time head value, minted at dispatch;
    playlist tracks derive their per-track positions from it. `origin` is the
    raw command argument, carried onto every resulting item — for a collection
    the link, not the per-track search its expansion generated.

    `mode` is required and has no default: interjection resolves through this
    same helper, so "a search may go flat" cannot be decided from the input
    shape — only the caller knows whether the song must be playable on arrival.

    `pool_slot` is the guild's resolve bound, handed down rather than held around
    this call: it is taken at the extraction itself, so a cache hit and the pure
    HTTP of a Spotify playlist do not queue behind two in-flight lookups. See
    docs/ARCHITECTURE.md#where-the-resolve-bound-is-taken."""
    if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
        # Titles, not QueueObjects — enqueue_playlist mints the YTSources
        # they become, carrying this command's analytics.
        titles = await cog._require_spotify().playlist(source.id)
        if not titles:
            # Otherwise the enqueue below confirms "Queued playlist" with 👍
            # over nothing queued, which reads exactly like success.
            raise EmptyPlaylistError()
        return ResolvedSpotifyPlaylist(titles)
    elif isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
        if source.list_id is None:
            raise ValueError("YTSource with type=PLAYLIST must have list_id set")
        tracks = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
            redis=cog.redis,
            pool_slot=pool_slot,
        )
        tracks, skipped = _apply_playlist_index(tracks, source.index)
        _apply_playlist_timestamp(tracks, source)
        return ResolvedYoutubePlaylist(tracks, skipped=skipped)
    else:
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
        # Only a search has a cheap mode; a link pays the watch page either way.
        flat = mode is ResolveMode.FLAT_OK and (
            isinstance(source, SpotifySource)
            or (isinstance(source, YTSource) and source.ytsearch is not None)
        )
        return await YTDL.yt_source(
            ctx.author,
            search,
            ts=ts,
            redis=cog.redis,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
            flat=flat,
            pool_slot=pool_slot,
        )


@_tracer.start_as_current_span("bot.enqueue_playlist")
async def enqueue_playlist(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    qobj: Union[ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist],
    mp: MusicPlayer,
    req: PlayRequest,
    *,
    analytics: Analytics,
    origin: str,
    placement: Placement = Placement.TAIL,
    cog: MusicBot,
) -> None:
    """Queue a resolved playlist under the place lock and notify the channel.
    Spotify playlists arrive as titles needing YouTube search, YouTube playlists
    pre-resolved. Positions are minted at the insert: `analytics` carries the
    ask time and its depth is replaced by the one the head takes."""
    # A playlist front-inserts in full, in order, under either flag. NEXT goes
    # through queue_put_next, for the claim the loop's prefetch holds.
    enqueue = {
        Placement.TAIL: mp.queue_put,
        Placement.COLD_FRONT: mp.queue_put_front,
        Placement.NEXT: mp.queue_put_next,
    }[placement]
    if placement is Placement.NEXT:
        # Off the lock, for the reason enqueue_single spells out: both branches
        # below take the place lock, and queue_put_next neutralizes inside it.
        await mp.settle_prefetch()
    warning = timestamp_warning(source)
    warning_line = f"\n\n{warning}" if warning else ""
    # "Queued playlist" on its own reads as "at the back".
    next_suffix = " — plays next" if placement is Placement.NEXT else ""
    tracks: Sequence[QueueItem]
    if isinstance(qobj, ResolvedSpotifyPlaylist):
        titles = qobj.titles
        shown_titles = queue_message([safe_label(t, ECHO_ROW_MAX) for t in titles])
        embed = build_embed(
            "Queued playlist" + next_suffix,
            f"Requested by: [{ctx.author.mention}]\n\n{shown_titles}{warning_line}",
            discord.Color.blue(),
        )
        # Built outside the lock: one YTSource per title, and the depth it
        # is minted against is almost always still the depth at the insert.
        provisional = _head_depth(mp, placement)
        tracks = spotify_playlist_to_ytsearch(
            titles,
            analytics=replace(analytics, queue_position=provisional),
            origin=origin,
        )
        log.info(f"spotify playlist track count: {len(tracks)}")
        async with cog._plays.place(req) as verdict:
            if verdict.placed:
                tracks = _rebase_positions(
                    tracks, provisional, _head_depth(mp, placement)
                )
                await enqueue(tracks, prefetch=False)
    else:
        # HACK: this assert stands in for a correlation the signature cannot
        # express — `source` and `qobj` are separate parameters, but a
        # ResolvedYoutubePlaylist always arrives with a YTSource. `python -O`
        # strips it, leaving the attribute reads below unguarded. Fix: have the
        # Resolved*Playlist dataclasses carry their own source.
        assert isinstance(source, YTSource)
        playlist_url = source.playlist_url
        # Mirrors the Spotify branch: a YTSource playlist resolves via
        # yt_playlist() to fully-formed QueueObjects.
        tracks = qobj.tracks
        count = len(tracks)
        log.info(f"yt playlist track count: {count}")
        # Stated, not silent: the user pasted a link and got fewer songs than
        # the playlist holds, and only the `index=` in their own URL explains
        # it.
        skipped_line = (
            f"Starting at #{qobj.skipped + 1} — skipped {qobj.skipped} "
            f"earlier {pluralize(qobj.skipped, 'song')}\n"
            if qobj.skipped
            else ""
        )
        shown_titles = queue_message(
            [safe_label(q.title, ECHO_ROW_MAX) for q in islice(tracks, 10)]
        )
        embed = build_embed(
            f"Queued playlist — {count} {pluralize(count, 'song')}{next_suffix}",
            f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n"
            f"{skipped_line}\n{shown_titles}{warning_line}",
            discord.Color.blue(),
        )
        # Minted before the lock: at 5,000 tracks the pass is milliseconds of
        # event-loop time every sibling -play would wait out (_rebase_positions).
        provisional = _head_depth(mp, placement)
        tracks = _rebase_positions(tracks, 0, provisional)
        async with cog._plays.place(req) as verdict:
            if verdict.placed:
                tracks = _rebase_positions(
                    tracks, provisional, _head_depth(mp, placement)
                )
                await enqueue(tracks, prefetch=False)
    if not verdict.placed:
        await cog._report_dropped(req, verdict)
        return
    await asyncio.gather(
        _reply(ctx, [embed]), _warm_front_track(tracks, placement, cog=cog)
    )


@_tracer.start_as_current_span("bot.enqueue_single")
async def enqueue_single(
    ctx: commands.Context,
    qobj: QueueObject,
    mp: MusicPlayer,
    req: PlayRequest,
    *,
    placement: Placement = Placement.TAIL,
    note: str = "",
    warning: Optional[str] = None,
    follow_on: Sequence[QueueItem] = (),
    cog: MusicBot,
) -> None:
    """Insert one resolved song under the place lock, then confirm. Under the
    lock: the put and the `queue_position` minted on it. The embeds are built
    and sent off the lock (the tail confirmation after the put, so it names the
    slot taken and re-hosts the live block when that slot is the head); see
    docs/ARCHITECTURE.md#play-placement. `warning` rides the confirmation or
    gets its own message; `follow_on` is a collection's tail behind the head."""
    vc = ctx.voice_client
    embeds: list[discord.Embed] = []
    # Carried to the end unless a branch folds it into its own embed instead.
    warning_embed = (
        notice_embed(warning, discord.Color.orange()) if warning is not None else None
    )
    should_show_queued = False
    if placement is Placement.COLD_FRONT:
        # The resume notice calls what sits behind this song "the previous
        # session", true only while the queue holds restored entries alone. A
        # sibling cold start that already placed put its own song in there.
        sibling_landed = cog._plays.sibling_placed(req)
        resume_notice = None if sibling_landed else mp.build_resume_notice_embed(qobj)
        if resume_notice is not None:
            embeds.append(resume_notice)
        elif sibling_landed:
            # Every cold start in a burst but the first joins a queue that is
            # partly its own, so it gets the ordinary slot confirmation.
            should_show_queued = True
    elif placement is Placement.NEXT:
        # No ETA: the walk seeds from the current song's FULL duration, which is
        # badly wrong one slot out. The note names the song it waits behind.
        embeds.append(playing_next_embed(ctx, qobj, note=plays_after_note(mp, vc)))
    else:
        # A note is the only word the user gets about tracks queued behind
        # this one, so an empty queue does not suppress the field.
        should_show_queued = (
            bool(note)
            or mp.queue.qsize() > 0
            or (isinstance(vc, discord.VoiceClient) and vc.is_playing())
        )
        if should_show_queued:
            warning_embed = None  # it rides the confirmation, built below
        # Otherwise the song starts now and the NP card speaks for it, so the
        # warning needs its own message.
    if warning_embed is not None:
        embeds.append(warning_embed)
    if placement is Placement.NEXT:
        # Outside the lock: this cancel can wait out a whole yt-dlp extraction
        # (an executor call is not interruptible), and every sibling -play in the
        # guild spends its place bound waiting on the lock.
        await mp.settle_prefetch()
    async with cog._plays.place(req) as verdict:
        if verdict.placed:
            depth = _head_depth(mp, placement)
            qobj.analytics = replace(qobj.analytics, queue_position=depth)
            if placement is Placement.COLD_FRONT:
                await mp.queue_put_front(qobj)
            elif placement is Placement.NEXT:
                await mp.queue_put_next(qobj)
            else:
                await mp.queue_put(qobj)
                if follow_on:
                    # Behind the head, in its order, re-minted from the head's
                    # depth: play_history keeps whatever number is on them.
                    await mp.queue_put(
                        [
                            with_queue_position(item, depth + offset)
                            for offset, item in enumerate(follow_on, start=1)
                        ],
                        prefetch=False,
                    )
            log.info(f"play ({placement.value}) qsize: {mp.queue.qsize()}")
    if not verdict.placed:
        await cog._report_dropped(req, verdict)
        return
    if should_show_queued:
        # After the put, off the lock, so the card names the slot taken. At the
        # head the NP block's "Up next" IS this card: re-host the live one,
        # dedicated — a response host with no own embeds strip-edits to blank.
        if mp.queue.peek_next() is qobj and await mp.repin_now_playing():
            # What the card would have carried: the note and the warning.
            said = "\n\n".join(text for text in (note, warning) if text)
            if said:
                color = discord.Color.orange() if warning else discord.Color.blue()
                embeds.append(notice_embed(said, color))
        else:
            embeds.append(mp.build_queued_song_embed(qobj, note=note, warning=warning))
    await _reply(ctx, embeds)


async def _resolve_interjection_source(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    *,
    origin: str,
    pool_slot: Optional[asyncio.Semaphore] = None,
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
        # The head takes the full path — it has to be playable to interrupt
        # with. The rest stay lazy searches, resolved at dequeue.
        head = await YTDL.yt_source(
            ctx.author,
            yts[0].ytsearch or "",
            redis=cog.redis,
            query_source=query_source_of(yts[0]),
            analytics=analytics,
            user_input=origin,
            pool_slot=pool_slot,
        )
        return head, list(yts[1:])
    if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
        tracks = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
            redis=cog.redis,
            pool_slot=pool_slot,
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
    # FULL, not the placement default: interject() stops the current song, so this
    # head has to be playable before anything is stopped.
    qobj = await queue_source(
        ctx,
        source,
        analytics=analytics,
        origin=origin,
        mode=ResolveMode.FULL,
        pool_slot=pool_slot,
        cog=cog,
    )
    assert isinstance(qobj, QueueObject)
    return qobj, []


@_tracer.start_as_current_span("bot.interject_flow")
async def interject_flow(
    ctx: commands.Context,
    url: str,
    mp: MusicPlayer,
    vc: discord.VoiceClient,
    req: PlayRequest,
    *,
    resume_paused: bool = True,
    require_paused: bool = False,
    cog: MusicBot,
) -> None:
    """Resolve `url` to one song, interrupt what is playing, and report.

    Shared by `-play --now` and by `-play` on a paused song; they differ only in
    resume_paused (`--now` restores paused-in -> paused-out, plain `-play` brings
    it back playing). require_paused re-reads the pause state after resolution,
    before committing: `-play` interjects only *because* the song is paused, so a
    `-resume` landing during the 1-4s extraction removes the reason and the track
    is appended instead. Reading it here rather than at command entry also means
    a song that fails to resolve never stops the paused song.
    """
    source = parse_input(url)
    qobj, follow_on = await _resolve_interjection_source(
        ctx,
        source,
        origin=url,
        pool_slot=cog._plays.resolve_slot(req),
        cog=cog,
    )
    # The head only: `interjected` is attribution, which song cut the line.
    qobj.interjected = True

    # The head only, awaited: a cache miss at dequeue is yt-dlp dead air between
    # the interrupt and the new song, and the current song plays through the wait.
    # A gate, not a hint — this flow stops what is playing, so a head that could
    # not be extracted must not get that far. Also back-fills the embed fields.
    if not await YTDL.prefetch_stream(qobj, redis=cog.redis):
        raise RuntimeError(
            "Could not get a playable stream for that song, so the current "
            "song was left alone."
        )

    # Before the lock: the neutralize can wait on a prefetch pinned in the
    # yt-dlp executor, which under _place would hold the guild's lock.
    await mp.settle_prefetch()

    outcome: Optional[InterjectOutcome] = None
    resumed = False
    async with cog._plays.place(req) as verdict:
        if not verdict.placed:
            pass
        elif require_paused and not vc.is_paused():
            # Resumed during the resolve, so the reason to interject is gone:
            # append instead. The append takes the lock again on its own.
            resumed = True
        else:
            outcome = await mp.interject(
                qobj, vc, resume_paused=resume_paused, follow_on=follow_on
            )
            if outcome is None:
                # The song ended during the resolve, so this interrupted nothing:
                # the marker comes off and the song front-inserts instead.
                qobj.interjected = False
                # interject() also returns None when the loop moved on to a
                # DIFFERENT song, which this insert waits behind: depth 1.
                qobj.analytics = replace(
                    qobj.analytics, queue_position=front_insert_depth(mp)
                )
                # queue_put_next, for the claim the loop's prefetch holds.
                # prefetch=False — the stream URL was warmed above.
                await mp.queue_put_next([qobj, *follow_on], prefetch=False)
    if not verdict.placed:
        await cog._report_dropped(req, verdict)
        return

    if resumed:
        # Clear the marker: a queued song must not trigger replace semantics
        # later. The interjection's 0 is replaced at the insert.
        qobj.interjected = False
        note = (
            collection_note(url, len(follow_on) + 1, head_playing=False)
            if follow_on
            else ""
        )
        await enqueue_single(
            ctx,
            qobj,
            mp,
            req,
            note=note,
            warning=timestamp_warning(source),
            follow_on=follow_on,
            cog=cog,
        )
        return

    if outcome is None:
        note = "The song being interrupted already ended — queued to play next instead."
        if follow_on:
            # Nothing was interrupted, so the head is QUEUED rather than
            # playing: it counts, and -remove reaches it.
            note += collection_note(url, len(follow_on) + 1, head_playing=False)
        await asyncio.gather(
            ctx.send(embed=playing_next_embed(ctx, qobj, note=note)),
            ctx.message.add_reaction("⏯️"),
        )
        return

    if outcome.resume_position is None:
        desc = (
            f"**{outcome.interrupted_title}** was nearly finished and will not resume."
        )
    elif outcome.returns_paused:
        # returns_paused, not was_paused: with resume_paused=False the song was
        # paused but comes back playing, so "will return paused" would lie.
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
