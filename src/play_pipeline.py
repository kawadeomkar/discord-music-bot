"""The machinery behind `-play`: resolve the input, place it, and — for an
interjection — put the interrupted song back where it was.

Three stages in the order they run: `queue_source` turns a parsed source into
something enqueueable; `enqueue_playlist` / `enqueue_single` place it and send
the confirmation; `interject_flow` is the `--now` path, shared with `-play` on
a paused song. The playlist errors and the two Resolved* shapes live here
because nothing outside this pipeline constructs them.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, TypeGuard, Union, assert_never
from collections.abc import Awaitable, Callable, Sequence

import discord
from discord.ext import commands

from src.guild_state import Analytics
from src.musicplayer import InterjectOutcome, MusicPlayer
from src.play_placement import (
    Placement,
    PlayRequest,
    ResolveMode,
    TIMESTAMP_FLAG,
    slow_resolve_notice,
)
from src.sources import (
    QUERY_SOURCE_SPOTIFY,
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    CollectionNoun,
    collection_noun,
    parse_input,
    query_source_of,
    timestamp_warning,
)
from src.spotify import SpotifyPlaylist, SpotifyTrack
from src.telemetry import get_tracer
from src.queue_progress import enqueue_progress, is_collection
from src.queue_rows import queue_runtime
from src.util import (
    ECHO_MAX,
    EMBED_DESCRIPTION_LIMIT,
    ProgressFn,
    ECHO_ROW_MAX,
    build_embed,
    fmt_duration,
    get_logger,
    notice_embed,
    pluralize,
    safe_label,
    send_embed,
    truncate_embed_title,
    verbatim_code,
)
from src.youtube import YTDL, QueueObject

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module).
    from src.musicbot import MusicBot

log = get_logger(__name__)

# Searches built per event-loop turn for a Spotify playlist. Measured ~7ms a
# thousand, which is how long each chunk holds the event loop.
_SEARCH_BUILD_CHUNK = 1000

# Items re-minted per event-loop turn by _rebase_positions. Measured 0.43 ms a
# thousand, so 2,000 is the chunk that holds the event loop for about a
# millisecond: 10,000 tracks pay five of those instead of one slice of 4.3 ms.
_REBASE_CHUNK = 2000

# Blank lines and the "... and N more" tail the rows add around themselves.
_DESCRIPTION_MARGIN = 64
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
    """A collection that resolved to nothing queueable. Vague about the cause:
    yt-dlp drops unavailable entries before this code sees them, so "empty" and
    "every video is private" are indistinguishable here. An album's tracks are
    Spotify's, so its copy names no video."""

    def __init__(self, noun: CollectionNoun = "playlist") -> None:
        tracks = "track on it" if noun == "album" else "video in it"
        super().__init__(
            f"{noun} resolved to no tracks",
            f"That {noun} has no songs I can queue — it may be empty, or every "
            f"{tracks} may be private or unavailable.",
        )


# kw_only: `title` and `link` are adjacent Optional[str]s that transpose silently.
# frozen: nothing writes a field back; the enqueue re-mints the ITEMS in place
# (with_queue_position) and rebinds its own local for the list.
@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedPlaylist:
    """A collection resolved to queue items. A Spotify collection's items are
    still searches, resolved at dequeue; a YouTube walk's are playable already —
    one type, because the enqueue does the same thing with either.

    `skipped` is how many leading tracks the link's `index=` dropped, for the
    enqueue embed alone: `tracks` is already sliced. `link` is what the heading
    points at, taken where the source is still in hand. `artists` and `thumbnail`
    are an album's, and `short` is a walk Spotify ended early."""

    tracks: list[QueueObject]
    title: Optional[str] = None
    link: Optional[str] = None
    skipped: int = 0
    unavailable: int = 0
    artists: list[str] = field(default_factory=list)
    thumbnail: Optional[str] = None
    short: bool = False


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


def _apply_playlist_timestamp(
    tracks: list[QueueObject], source: YTSource, ts: Optional[int]
) -> None:
    """Start the first queued track at `ts` — the link's own `t=`, or the
    `--timestamp` that overrode it — only when that track is the `v=` video the
    link names: without a matching `index=` the queue starts at track 1, usually
    a different song."""
    if not ts or not source.video_id or not tracks:
        return
    # Substring, not equality: yt_playlist takes the entry's own `url` when it
    # has one, so the shape is not guaranteed.
    if source.video_id in tracks[0].webpage_url:
        tracks[0].ts = ts


def effective_start_offset(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    start_offset: Optional[int],
) -> Optional[int]:
    """Where a resolve starts the song: the `--timestamp` flag, else the link's
    own `t=`. The flag wins — it is the more explicit of the two, and a link
    whose `t=` did not take is the likeliest reason to reach for it."""
    if start_offset is not None:
        return start_offset
    return None if isinstance(source, SpotifySource) else source.ts


def start_offset_refusal(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> Optional[str]:
    """Why a `--timestamp` cannot start this input, or None. One offset over N
    songs names none of them, so a collection is refused — except a watch link
    carrying `&list=`, which names its `v=` video and is taken exactly like the
    `t=` on the same link. Answered off the PARSED source, so the refusal lands
    before the join and before any extraction."""
    if not is_collection(source) or (isinstance(source, YTSource) and source.video_id):
        return None
    return (
        f"⚠️ `{TIMESTAMP_FLAG}` starts one song, and that link queues a "
        f"{collection_noun(source)}. Play the song on its own to start it partway in."
    )


def past_end_refusal(
    qobj: Union[QueueObject, ResolvedPlaylist],
    start_offset: Optional[int],
) -> Optional[str]:
    """Why `start_offset` cannot start the resolved song, or None. A duration of
    None or 0 rules nothing out (livestreams report both), so an offset that
    cannot be checked stands and ffmpeg judges it.

    A collection is measured against the track the offset landed on — the `v=`
    head _apply_playlist_timestamp stamped — and not at all when it landed on
    none, so the interjection and the ordinary placement answer one link alike."""
    if start_offset is None:
        return None
    if isinstance(qobj, QueueObject):
        head = qobj
    elif qobj.tracks:
        head = qobj.tracks[0]
        # Only when the offset actually landed on the head: a Spotify
        # collection's items are searches _apply_playlist_timestamp never
        # stamps, so their ts never matches and ffmpeg judges the offset.
        if head.ts != start_offset:
            return None
    else:
        return None
    if not head.duration or start_offset < head.duration:
        return None
    return (
        f"⚠️ `{fmt_duration(start_offset)}` is past the end of "
        f"**{safe_label(head.title, ECHO_ROW_MAX)}** "
        f"(`{fmt_duration(head.duration)}`) — nothing was queued."
    )


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


def with_queue_position(item: QueueObject, position: int) -> QueueObject:
    """Re-mint one item's `queue_position`, in place."""
    item.analytics = replace(item.analytics, queue_position=position)
    return item


def _is_spotify_collection(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> TypeGuard[SpotifySource]:
    return isinstance(source, SpotifySource) and source.type in (
        SpotifyType.PLAYLIST,
        SpotifyType.ALBUM,
    )


async def _spotify_collection(
    source: SpotifySource, *, on_progress: Optional[ProgressFn], cog: MusicBot
) -> SpotifyPlaylist:
    """Walk a Spotify playlist or album. Raises on an empty one: the enqueue would
    otherwise confirm it queued with 👍 over nothing queued."""
    spotify = cog._require_spotify()
    walk = spotify.album if source.type is SpotifyType.ALBUM else spotify.playlist
    collection = await walk(source.id, on_progress=on_progress)
    if not collection.titles:
        raise EmptyPlaylistError(collection_noun(source))
    return collection


def short_walk_notice(noun: CollectionNoun) -> discord.Embed:
    """Said when Spotify ended a walk before its own count of the collection: the
    confirmation's song count is what was queued, not what the link holds."""
    return notice_embed(
        f"Spotify stopped sending this {noun} early, so some of its songs may be "
        "missing from the queue.",
        discord.Color.orange(),
    )


def collection_note(
    url: str,
    queued: int,
    *,
    returns: str = "",
    head_playing: bool,
    noun: CollectionNoun,
) -> str:
    """What a `-play` that queued a whole collection tells the user: how many
    tracks landed, when the interrupted song returns, and the `-remove` undo.
    `head_playing` changes the undo: a playing song has no queue object for
    -remove to reach, so it names -skip."""
    undo = (
        "the queued ones back out; the one playing needs `-skip`."
        if head_playing
        else f"the whole {noun} back out."
    )
    # -remove compares links literally, so the command has to be copyable as-is.
    command = verbatim_code(f"-remove {url}", ECHO_MAX)
    if command is None:
        command = "`-remove` followed by the link you pasted"
    return (
        f"\n\nQueued **{queued}** {pluralize(queued, 'song')} from the {noun}."
        f"{returns}\nNot what you wanted? {command} takes {undo}"
    )


async def _searches_for(
    titles: Sequence[str],
    *,
    requester: Union[discord.User, discord.Member],
    analytics: Analytics,
    origin: str,
    rows: Sequence[SpotifyTrack] = (),
) -> list[QueueObject]:
    """A Spotify collection's titles as queue items still to resolve, yielding the
    loop every chunk so 10,000 of them do not hold it. Positions count on from
    `analytics` across chunks, as they would in one pass. The Spotify token, the
    pasted collection link and the requester are set here, the last point that
    knows where these came from.

    `rows` is the walk's display rows, one per title, which is what a listing shows
    until the search resolves; a set that does not pair up is dropped whole, because
    a row read against the wrong title is worse than no row."""
    if rows and len(rows) != len(titles):
        log.warning(
            f"spotify display rows do not pair with titles "
            f"({len(rows)} rows, {len(titles)} titles); queueing without them"
        )
        rows = ()
    # One joined byline per distinct artist tuple: a collection is usually one
    # artist, and a fresh join per track is the only string this pass keeps for the
    # life of the queue (measured ~800 KiB over 10,000 tracks).
    bylines: dict[tuple[str, ...], Optional[str]] = {}

    def byline(row: SpotifyTrack) -> Optional[str]:
        key = tuple(row.artists)
        if key not in bylines:
            bylines[key] = ", ".join(key) or None
        return bylines[key]

    tracks: list[QueueObject] = []
    for offset, title in enumerate(titles):
        if offset and offset % _SEARCH_BUILD_CHUNK == 0:
            await asyncio.sleep(0)
        row = rows[offset] if rows else None
        tracks.append(
            QueueObject(
                # The track's own page until the search resolves: what the listing
                # links, and what -remove accepts for it.
                row.url or "" if row else "",
                # Empty without a row: every renderer falls back to the term.
                row.name if row else "",
                requester,
                search=f"ytsearch:{title}",
                user_input=origin,
                query_source=QUERY_SOURCE_SPOTIFY,
                uploader=byline(row) if row else None,
                duration=row.duration_secs if row else None,
                analytics=replace(
                    analytics, queue_position=analytics.queue_position + offset
                ),
            )
        )
    return tracks


async def _rebase_positions(
    tracks: Sequence[QueueObject], minted_from: int, base: int
) -> Sequence[QueueObject]:
    """Move a resolved collection's `queue_position`s from one head depth to
    another, returning `tracks` unchanged when the head has not moved and yielding
    the loop every chunk otherwise.

    Called twice per collection enqueue. The first is unconditional: the resolve
    mints depths against the ASK, which is 0 for every collection, so the pass is
    what buys one minting rule for both collection kinds. The second, under the
    lock, moves only when another request placed in between."""
    if base == minted_from:
        return tracks
    rebased: list[QueueObject] = []
    for offset, track in enumerate(tracks):
        if offset and offset % _REBASE_CHUNK == 0:
            await asyncio.sleep(0)
        rebased.append(with_queue_position(track, base + offset))
    return rebased


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


def _songs_ahead(mp: MusicPlayer, placement: Placement) -> int:
    """Queued songs a playlist's first song plays after, read under the place lock.
    Both front placements go ahead of the whole queue."""
    return mp.queue.display_size() if placement is Placement.TAIL else 0


def _playlist_heading(name: Optional[str], link: Optional[str]) -> str:
    """The playlist's name linked to it, as the queue cards link a song; the bare
    link when the resolve found no name. Stripped: Discord renders `**name **`
    literally, and Spotify returns names with trailing spaces."""
    name = (name or "").strip()
    if not name:
        return link or ""
    label = f"**{safe_label(name, ECHO_ROW_MAX)}**"
    return f"[{label}]({link})" if link else label


async def _nothing_to_release() -> None:
    """Default `release_hold` for the enqueue helpers. A warm placement holds no
    playback gate, so there is nothing to release when its put lands."""


async def _reply(
    ctx: commands.Context,
    embeds: Sequence[discord.Embed],
    reaction: str = "👍",
    *,
    together: bool = False,
) -> None:
    """Confirm a placement that already happened. return_exceptions: the song is IN
    the queue, and a missing Add Reactions permission or a deleted invoking message
    must not render "Failed to queue song" over a song that plays. `together` sends
    several embeds as one message, in order; separate sends race."""
    sends = (
        [ctx.send(embeds=list(embeds))]
        if together and len(embeds) > 1
        else [ctx.send(embed=embed) for embed in embeds]
    )
    results = await asyncio.gather(
        ctx.message.add_reaction(reaction), *sends, return_exceptions=True
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
    tracks: Sequence[QueueObject], placement: Placement, *, cog: MusicBot
) -> None:
    """Warm the stream URL of a playlist's head when it is about to play. Bulk
    enqueues pass prefetch=False, and under `--next` queue_put_next killed the
    loop's one-ahead prefetch, so the head is left with no warm at all. An item
    that has not resolved has no URL yet; it resolves at dequeue."""
    if placement is not Placement.NEXT or not tracks:
        return
    head = tracks[0]
    if not head.unresolved:
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


@_tracer.start_as_current_span("bot.queue_source")
async def queue_source(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    *,
    analytics: Analytics,
    origin: str,
    mode: ResolveMode,
    on_progress: Optional[ProgressFn] = None,
    pool_slot: Optional[contextlib.AbstractAsyncContextManager[Any]] = None,
    start_offset: Optional[int] = None,
    cog: MusicBot,
) -> Union[QueueObject, ResolvedPlaylist]:
    """Resolve a parsed source into something enqueueable. `analytics` is the
    command's ask-time head value; playlist tracks derive per-track positions
    from it. `origin` is the raw command argument, carried onto every item —
    for a collection the link, not the per-track search its expansion made.

    `mode` is required and has no default: interjection resolves through this
    same helper, so "a search may go flat" cannot be decided from the input
    shape — only the caller knows whether the song must be playable on arrival.

    `on_progress` is how a collection reports its walk to the live card, handed
    down as a bare callback for the same reason `pool_slot` is: youtube.py and
    spotify.py know nothing about guilds, and less about embeds. Absent for
    everything that is not a collection, and for a cache hit, which finishes far
    under the threshold that would show a card.

    `pool_slot` is the guild's resolve bound, handed down rather than held around
    this call: it is taken at the extraction itself, so a cache hit and the pure
    HTTP of a Spotify playlist do not queue behind two in-flight lookups. See
    docs/ARCHITECTURE.md#where-the-resolve-bound-is-taken.

    `start_offset` is `--timestamp`, applied here so a link's `t=` and the flag
    reach the song by one route."""
    if _is_spotify_collection(source):
        playlist = await _spotify_collection(source, on_progress=on_progress, cog=cog)
        return ResolvedPlaylist(
            # Items that are still searches: a collection nobody plays to the end
            # never resolves its tail, and the ask depth they are minted against
            # is rebased onto the head's at the insert.
            tracks=await _searches_for(
                playlist.titles,
                requester=ctx.author,
                analytics=analytics,
                origin=origin,
                rows=playlist.tracks,
            ),
            title=playlist.name,
            link=source.url,
            unavailable=playlist.unavailable,
            artists=playlist.artists,
            thumbnail=playlist.thumbnail,
            short=playlist.short,
        )
    if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
        if source.list_id is None:
            raise ValueError("YTSource with type=PLAYLIST must have list_id set")
        playlist = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
            redis=cog.redis,
            on_progress=on_progress,
            pool_slot=pool_slot,
        )
        tracks, skipped = _apply_playlist_index(playlist.tracks, source.index)
        _apply_playlist_timestamp(
            tracks, source, effective_start_offset(source, start_offset)
        )
        return ResolvedPlaylist(
            tracks=tracks,
            title=playlist.title,
            link=source.playlist_url,
            skipped=skipped,
            unavailable=playlist.unavailable,
        )
    ts = effective_start_offset(source, start_offset)
    search: str
    if isinstance(source, SpotifySource):
        search = await cog._require_spotify().track(source.id)
    elif isinstance(source, YTSource):
        search = source.ytsearch or source.url or ""
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
    qobj: ResolvedPlaylist,
    mp: MusicPlayer,
    req: PlayRequest,
    *,
    analytics: Analytics,
    origin: str,
    placement: Placement = Placement.TAIL,
    release_hold: Callable[[], Awaitable[None]] = _nothing_to_release,
    cog: MusicBot,
) -> None:
    """Queue a resolved playlist under the place lock and notify the channel.
    Every collection arrives as queue items, whether or not they are resolved
    yet. Positions are minted at the resolve and rebased here: `analytics`
    carries the ask time, and its depth is replaced by the one the head takes."""
    # A collection front-inserts in full, in order, under either flag. NEXT goes
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
    # "Queued playlist" on its own reads as "at the back".
    next_suffix = " — plays next" if placement is Placement.NEXT else ""
    tracks: Sequence[QueueObject] = qobj.tracks
    count = len(tracks)
    ahead = 0
    log.info(f"{collection_noun(source)} track count: {count}")
    heading = [_playlist_heading(qobj.title, qobj.link)]
    if qobj.artists:
        heading.append(f"by {safe_label(', '.join(qobj.artists), ECHO_ROW_MAX)}")
    # Stated: only the `index=` in the user's own URL explains fewer songs.
    if qobj.skipped:
        heading.append(
            f"Starting at #{qobj.skipped + 1} — skipped {qobj.skipped} "
            f"earlier {pluralize(qobj.skipped, 'song')}"
        )
    # The same sum -queue shows for these tracks: the lengths on the items, added
    # as whole seconds, so the two totals cannot drift apart.
    runtime = queue_runtime(tracks)
    # Rebased before the lock: the depths the resolve minted are the ask's, not
    # this head's, and off the lock no sibling -play spends its place bound on the
    # pass. The one under the lock moves only when a sibling placed in between.
    provisional = _head_depth(mp, placement)
    tracks = await _rebase_positions(tracks, analytics.queue_position, provisional)
    async with cog._plays.place(req) as verdict:
        if verdict.placed:
            ahead = _songs_ahead(mp, placement)
            tracks = await _rebase_positions(
                tracks, provisional, _head_depth(mp, placement)
            )
            await enqueue(tracks, prefetch=False)
    if not verdict.placed:
        await cog._report_dropped(req, verdict)
        return
    # The songs are in the queue, which is all the cold-start gate hold was
    # waiting for. What follows is a Discord send and a stream warm.
    await release_hold()
    # After the put, like the single-song card, so the facts describe the slot taken.
    # One read of the queue backs both numbers: "Songs ahead" and the first row's
    # index disagreed when the loop dequeued between them.
    first = mp.queued_slot(tracks, ahead=ahead)
    header = [f"Requested by: [{ctx.author.mention}]", *heading]
    # Every row carries its own start time, so the facts line does not.
    header.append(mp.playlist_facts(ahead=first - 1, runtime=runtime, eta=False))
    body = [line for line in header if line]
    # One bound over the whole description, not just the rows: the heading, the
    # facts and a timestamp warning are each capped on their own, but their SUM
    # is what Discord rejects.
    around = "\n".join(body) + (f"\n\n{warning}" if warning else "")
    room = EMBED_DESCRIPTION_LIMIT - len(around) - _DESCRIPTION_MARGIN
    # The rows -queue will show for these tracks, read after the put.
    rows = mp.queued_rows(tracks, first=first, budget=max(0, room))
    description = "\n".join(body) + f"\n\n{rows}"
    if warning:
        description += f"\n\n{warning}"
    noun = collection_noun(source)
    embeds = [
        build_embed(
            f"Queued {noun} — {count} {pluralize(count, 'song')}{next_suffix}",
            description,
            discord.Color.blue(),
            thumbnail=qobj.thumbnail,
        )
    ]
    if qobj.short:
        embeds.insert(0, short_walk_notice(noun))
    if qobj.unavailable:
        n = qobj.unavailable
        embeds.insert(
            0,
            notice_embed(
                f"Skipped **{n}** unavailable {pluralize(n, 'song')} from this {noun}.",
                discord.Color.red(),
            ),
        )
    await asyncio.gather(
        _reply(ctx, embeds, together=True),
        _warm_front_track(tracks, placement, cog=cog),
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
    follow_on: Sequence[QueueObject] = (),
    release_hold: Callable[[], Awaitable[None]] = _nothing_to_release,
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
        # display_size(): a song the loop has claimed and is still resolving is
        # neither pending nor playing, and this one queues behind it all the same.
        should_show_queued = (
            bool(note)
            or mp.queue.display_size() > 0
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
    # Minted before the lock, like the playlist branch's: `--now`/`--next` take a
    # collection in full, so this is up to 9,999 entries of event-loop time every
    # sibling -play in the guild would spend its place bound waiting out.
    # _rebase_positions inside is a no-op unless another request placed between.
    provisional = _head_depth(mp, placement) + 1
    follow_on = [
        with_queue_position(item, provisional + offset)
        for offset, item in enumerate(follow_on)
    ]
    async with cog._plays.place(req) as verdict:
        if verdict.placed:
            depth = _head_depth(mp, placement)
            qobj.analytics = replace(qobj.analytics, queue_position=depth)
            if placement is Placement.COLD_FRONT:
                await mp.queue_put_front(qobj)
            elif placement is Placement.NEXT:
                await mp.queue_put_next(qobj)
            elif follow_on:
                # One put, so a -remove waiting on the queue mutex takes the head
                # and its tail together or neither. The tail is re-minted from the
                # head's depth: play_history keeps whatever number is on it. Only
                # interject_flow passes follow_on, and it already warmed the head.
                rebased = await _rebase_positions(follow_on, provisional, depth + 1)
                await mp.queue_put([qobj, *rebased], prefetch=False)
            else:
                await mp.queue_put(qobj)
            log.info(f"play ({placement.value}) qsize: {mp.queue.qsize()}")
    if not verdict.placed:
        await cog._report_dropped(req, verdict)
        return
    # The song is in the queue, which is the whole of what the cold-start gate
    # hold was waiting for (see MusicPlayer.defer_playback). Everything below is
    # presentation — an embed send, a reaction, possibly a re-host edit — and on
    # a cold start those Discord round trips sat between the front insert and the
    # first note. Idempotent, so every path that does NOT place still releases
    # through the stack that owns the hold.
    await release_hold()
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
    on_progress: Optional[ProgressFn] = None,
    pool_slot: Optional[contextlib.AbstractAsyncContextManager[Any]] = None,
    start_offset: Optional[int] = None,
    cog: MusicBot,
) -> tuple[QueueObject, list[QueueObject]]:
    """Resolve an interjection's input into (head, everything behind it). The
    head must be resolved to interrupt with; the tail may hold items that are
    still searches. The interrupted song returns after the whole playlist, and
    one `-remove <the link>` takes it all back out. `origin` is the raw command
    argument — for a playlist the link, not the generated titles."""
    # Ask-time analytics: the snowflake time, depth 0 for the head. Tracks behind
    # it derive 1, 2, … from this base; the caller re-mints the head's own depth.
    analytics = Analytics(
        queued_at=ctx.message.created_at.timestamp(), queue_position=0
    )
    if _is_spotify_collection(source):
        playlist = await _spotify_collection(source, on_progress=on_progress, cog=cog)
        if playlist.short:
            await ctx.send(embed=short_walk_notice(collection_noun(source)))
        tracks = await _searches_for(
            playlist.titles,
            requester=ctx.author,
            analytics=analytics,
            origin=origin,
            rows=playlist.tracks,
        )
        # The head takes the full path — it has to be playable to interrupt
        # with. The rest stay searches, resolved at dequeue.
        head = await YTDL.yt_source(
            ctx.author,
            tracks[0].search,
            redis=cog.redis,
            query_source=tracks[0].query_source,
            analytics=analytics,
            user_input=origin,
            pool_slot=pool_slot,
        )
        return head, list(tracks[1:])
    if isinstance(source, YTSource) and source.type is YTType.PLAYLIST:
        playlist = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
            redis=cog.redis,
            on_progress=on_progress,
            pool_slot=pool_slot,
        )
        # Indexed here too: `--now` on a link copied mid-playlist starts at the
        # track the user was looking at, not the playlist's first.
        tracks, skipped = _apply_playlist_index(playlist.tracks, source.index)
        _apply_playlist_timestamp(
            tracks, source, effective_start_offset(source, start_offset)
        )
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
        start_offset=start_offset,
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
    start_offset: Optional[int] = None,
    cog: MusicBot,
) -> None:
    """Resolve `url` to one song, interrupt what is playing, and report.

    Shared by `-play --now` and by `-play` on a paused song; they differ in
    resume_paused (`--now` restores paused-in → paused-out, `-play` brings it
    back playing). require_paused re-reads the pause state after resolution:
    `-play` interjects only because the song is paused, so a `-resume` landing
    during the 1–4s extraction removes the reason and the track is appended.

    Every refusal `start_offset` can draw is settled before interject(), so a
    rejected offset leaves the current song playing.
    """
    source = parse_input(url)
    if start_offset is not None:
        refusal = start_offset_refusal(source)
        if refusal is not None:
            await ctx.send(embed=notice_embed(refusal, discord.Color.red()))
            return
    # The second entry point for both, covering `-playnow <playlist>` and plain
    # `-play` over a paused song. A plain `async with`: no gate hold here, so the
    # LIFO constraint on the other entry point does not apply. It spans the whole
    # flow, so the retraction lands after vc.stop() — the delete is a Discord
    # round trip and this command stops what is playing.
    async with contextlib.AsyncExitStack() as stack:
        progress = None
        if is_collection(source):
            delay = cog.guild_settings.queue_progress_delay_secs(req.guild_id)
            progress = await stack.enter_async_context(
                enqueue_progress(
                    ctx,
                    source,
                    delay=delay,
                    placement_note="Interrupts the current song once it's queued.",
                    debug_suffix=cog.debug_suffix(ctx),
                    request_settled=req.settled,
                )
            )
        else:
            delay = cog.guild_settings.slow_notice_secs(req.guild_id)
            await stack.enter_async_context(
                slow_resolve_notice(
                    ctx,
                    query=req.query,
                    delay=delay,
                    debug_suffix=cog.debug_suffix(ctx),
                    request_settled=req.settled,
                )
            )
        qobj, follow_on = await _resolve_interjection_source(
            ctx,
            source,
            origin=url,
            on_progress=progress.update if progress else None,
            pool_slot=cog._plays.resolve_slot(req),
            start_offset=start_offset,
            cog=cog,
        )
        # Before the prefetch, which every remaining source reaches with a
        # duration already set: a refusal must not pay an extraction it discards.
        refusal = past_end_refusal(qobj, start_offset)
        if refusal is not None:
            await ctx.send(embed=notice_embed(refusal, discord.Color.red()))
            return
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
                # append instead. The append takes the lock again on its own, and
                # until it places, -remove or -clear may still drop the request.
                resumed = True
                req.placed = False
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
                collection_note(
                    url,
                    len(follow_on) + 1,
                    head_playing=False,
                    noun=collection_noun(source),
                )
                if follow_on
                else ""
            )
            await enqueue_single(
                ctx,
                qobj,
                mp,
                req,
                note=note,
                # None when the flag set the offset — see the -play call site.
                warning=(
                    None if start_offset is not None else timestamp_warning(source)
                ),
                follow_on=follow_on,
                cog=cog,
            )
            return

        if outcome is None:
            note = "The song being interrupted already ended — queued to play next instead."
            if follow_on:
                # Nothing was interrupted, so the head is QUEUED rather than
                # playing: it counts, and -remove reaches it.
                note += collection_note(
                    url,
                    len(follow_on) + 1,
                    head_playing=False,
                    noun=collection_noun(source),
                )
            await asyncio.gather(
                ctx.send(embed=playing_next_embed(ctx, qobj, note=note)),
                ctx.message.add_reaction("⏯️"),
            )
            return

        if outcome.replay_pending:
            desc = (
                f"**{outcome.interrupted_title}** was about to replay, so it plays "
                "again from `0:00` after this."
            )
        elif outcome.resume_position is None:
            desc = f"**{outcome.interrupted_title}** was nearly finished and will not resume."
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
            # The interrupted song waits behind the whole collection, so the reply says
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
                noun=collection_noun(source),
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
