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
    """Resolve a parsed URL/search source into something enqueueable: a
    ResolvedSpotifyPlaylist (titles still needing per-title YouTube resolution),
    a ResolvedYoutubePlaylist (already resolved), or a bare QueueObject.

    `analytics` is the command's ask-time head value, minted at dispatch;
    playlist tracks derive their per-track positions from it. `origin` is the
    raw command argument, carried onto every resulting item — for a collection
    the link, not the per-track search its expansion generated."""
    if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
        # Titles, not QueueObjects — enqueue_playlist mints the YTSources
        # they become, carrying this command's analytics.
        return ResolvedSpotifyPlaylist(await cog._require_spotify().playlist(source.id))
    elif isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
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
    """Queue a resolved playlist and notify the channel — branches on the
    resolved shape since Spotify playlists arrive as titles needing YouTube
    search resolution while YouTube playlists arrive pre-resolved."""
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
    """`warning` rides the confirmation embed when there is one. Every exit
    below sends it either way: the embed is conditional (a song that starts
    immediately gets none) and the warning is about what the user typed, so
    losing it on the quietest path would hide it in the common case of
    queueing the first song."""
    vc = ctx.voice_client
    if placement is Placement.COLD_FRONT:
        # The "Est. playing at" embed below would be wrong: a restored queue is
        # non-empty but its entries sit BEHIND this song. The resume notice
        # replaces it — it names the song starting now (nothing else does; the
        # gate is shut, so there is no NP block to host). Built before the
        # insert, while the queue holds only the restored entries.
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
    # Awaited ahead of the reply rather than gathered with it: the reply's
    # shape depends on whether this song became the queue head, which the put
    # decides. One RPUSH under the queue mutex (p50 ~2.4ms).
    await asyncio.gather(mp.queue_put(qobj), ctx.message.add_reaction("👍"))
    log.info(f"play qsize: {mp.queue.qsize()}")

    if should_show_queued:
        # The block's "Up next" card renders the queue head in the same layout
        # the confirmation uses, so when this song IS the head the two are one
        # card printed twice in one message. Re-host the live block instead and
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
        # Nothing else is being sent on this path — the song starts now and
        # the NP card speaks for it — so the warning needs its own message.
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
        ctx, source, origin=url, cog=cog
    )
    # The head only: `interjected` is attribution, which song cut the line.
    qobj.interjected = True

    # Warm the stream-URL cache before interrupting: a cache miss at dequeue puts
    # seconds of yt-dlp dead air between the interrupt and the new song. Awaited,
    # not spawned — the current song plays through the wait. No-op without Redis;
    # also back-fills duration/thumbnail for the embeds below.
    #
    # The head only: warming N tracks would be N concurrent extractions
    # minting URLs that expire before playback reaches them.
    await YTDL.prefetch_stream(qobj, redis=cog.redis)

    if require_paused and not vc.is_paused():
        # Resumed during the resolve — the reason to interject is gone, so
        # append rather than interrupt a song the user just chose to keep
        # playing. Clear the marker: a normally queued song must not trigger
        # replace semantics later.
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
        # The song ended during the resolve — nothing left to interrupt. Insert
        # qobj directly rather than re-invoking -play, which would re-parse,
        # re-resolve and enqueue every track a second time. Front, not append:
        # the user asked for "now", and this window can be seconds long with
        # songs queued behind.
        # It interrupted nothing, so keeping the marker would attribute an
        # interjection that never happened.
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
