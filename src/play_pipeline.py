"""The machinery behind `-play` and `-playnow`: resolve the input, place it, and — for
an interjection — put the interrupted song back where it was.

Three stages, in the order they run. `queue_source` turns a parsed source into
something enqueueable; `enqueue_playlist` / `enqueue_single` place it and send the
confirmation; `interject_flow` is the -playnow path, shared with `-play` on a paused
song. The playlist errors and the two Resolved* shapes live here because nothing
outside this pipeline constructs them — musicbot.py imports PlaylistInputError back
for the one error-embed branch that renders its user_message.
"""

import asyncio
from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING, Any, Optional, Union, assert_never
from collections.abc import Coroutine

import discord
from discord.ext import commands

from src.guild_state import Analytics
from src.musicplayer import MusicPlayer
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
    ECHO_ROW_MAX,
    get_logger,
    notice_embed,
    pluralize,
    queue_message,
    safe_label,
    send_embed,
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
    *,
    keep_first_only: bool = False,
) -> tuple[list[QueueObject], int]:
    """Drop the tracks ahead of YouTube's 1-based `index=`, returning what is left
    and how many went. A share link copied mid-playlist carries the position it was
    copied at, so playing it starts there rather than back at track 1.

    An index past the end raises PlaylistIndexError rather than queueing nothing:
    the user named a position this playlist does not have, and an empty enqueue
    reports success. The empty-playlist guard lives here too, so both callers
    get it.

    keep_first_only trims to the one track -playnow interjects, which also keeps
    the rebase below off the tracks that caller discards (~1ms for a 1000-track
    link).
    """
    if not tracks:
        raise EmptyPlaylistError
    if index is None or index <= 1:
        return (tracks[:1] if keep_first_only else tracks), 0
    if index > len(tracks):
        raise PlaylistIndexError(index, len(tracks))
    kept = tracks[index - 1 :]
    dropped = index - 1
    if keep_first_only:
        kept = kept[:1]
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
    front: bool = False,
) -> None:
    """Queue a resolved playlist and notify the channel — branches on the
    resolved shape since Spotify playlists arrive as titles needing YouTube
    search resolution while YouTube playlists arrive pre-resolved."""
    # A playlist front-inserts in full, in order — unlike -playnow, which
    # collapses it to the first track to bound how long an interrupted song
    # waits. Nothing is playing to interrupt on this path.
    enqueue = mp.queue_put_front if front else mp.queue_put
    warning = timestamp_warning(source)
    warning_line = f"\n\n{warning}" if warning else ""
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
                "Queued playlist",
                f"Requested by: [{ctx.author.mention}]\n\n{shown_titles}{warning_line}",
                discord.Color.blue(),
            ),
            enqueue(qobjs_yt, prefetch=False),
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
                f"Queued playlist — {count} {pluralize(count, 'song')}",
                f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n"
                f"{skipped_line}\n{shown_titles}{warning_line}",
                discord.Color.blue(),
            ),
            enqueue(tracks, prefetch=False),
            ctx.message.add_reaction("👍"),
        )


@_tracer.start_as_current_span("bot.enqueue_single")
async def enqueue_single(
    ctx: commands.Context,
    qobj: QueueObject,
    mp: MusicPlayer,
    *,
    front: bool = False,
    warning: Optional[str] = None,
) -> None:
    """`warning` rides the confirmation embed when there is one. Every exit
    below sends it either way: the embed is conditional (a song that starts
    immediately gets none) and the warning is about what the user typed, so
    losing it on the quietest path would hide it in the common case of
    queueing the first song."""
    vc = ctx.voice_client
    if front:
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

    should_show_queued = mp.queue.qsize() > 0 or (
        isinstance(vc, discord.VoiceClient) and vc.is_playing()
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
            if warning is not None:
                await ctx.send(embed=notice_embed(warning, discord.Color.orange()))
            return
        await ctx.send(embed=mp.build_queued_song_embed(qobj, warning=warning))
        return
    if warning is not None:
        # Nothing else is being sent on this path — the song starts now and
        # the NP card speaks for it — so the warning needs its own message.
        await ctx.send(embed=notice_embed(warning, discord.Color.orange()))


async def _resolve_playnow_source(
    ctx: commands.Context,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    *,
    origin: str,
    cog: MusicBot,
) -> QueueObject:
    """Resolve -playnow input to exactly one QueueObject. Playlists collapse to
    their first track — interjecting a whole one would delay the interrupted
    song's return indefinitely (use -play).

    `origin` is the raw command argument, passed down by every branch — for a
    collapsed playlist it is the link, not the title the expansion generated."""
    playlist_notice = notice_embed(
        "Playlists can't be interjected — playing the **first track** now. "
        "Use `-play` for the full playlist.",
        discord.Color.orange(),
    )
    # Ask-time analytics: the message's snowflake time, and depth 0 — an
    # interjection plays immediately. The caller re-mints the depth on the
    # two paths where it ends up queueing instead.
    analytics = Analytics(
        queued_at=ctx.message.created_at.timestamp(), queue_position=0
    )
    if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
        titles = await cog._require_spotify().playlist(source.id)
        if not titles:
            raise ValueError("Playlist has no tracks")
        await ctx.send(embed=playlist_notice)
        yts = spotify_playlist_to_ytsearch(
            titles[:1], analytics=analytics, origin=origin
        )[0]
        # Both playlist branches resolve directly rather than through
        # queue_source, so each passes its own metadata.
        return await YTDL.yt_source(
            ctx.author,
            yts.ytsearch or "",
            redis=cog.redis,
            query_source=query_source_of(yts),
            analytics=analytics,
            user_input=origin,
        )
    if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
        tracks = await YTDL.yt_playlist(
            source.playlist_url,
            ctx.author,
            query_source=query_source_of(source),
            analytics=analytics,
            user_input=origin,
        )
        # Indexed here too: -playnow on a link copied mid-playlist should
        # interject the track the user was looking at, not the playlist's
        # first. The slice makes tracks[0] that track.
        tracks, skipped = _apply_playlist_index(
            tracks, source.index, keep_first_only=True
        )
        _apply_playlist_timestamp(tracks, source)
        if skipped:
            await ctx.send(
                embed=notice_embed(
                    f"Playlists can't be interjected — playing **#"
                    f"{skipped + 1}** now. Use `-play` for the full "
                    f"playlist.",
                    discord.Color.orange(),
                )
            )
        else:
            await ctx.send(embed=playlist_notice)
        return tracks[0]
    qobj = await queue_source(ctx, source, analytics=analytics, origin=origin, cog=cog)
    assert isinstance(qobj, QueueObject)
    return qobj


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

    Shared by `-playnow` and by `-play` on a paused song; they differ only in
    resume_paused (`-playnow` restores paused-in → paused-out, `-play` brings it
    back playing). require_paused re-reads the pause state after resolution,
    before committing: `-play` interjects only *because* the song is paused, so a
    `-resume` landing during the 1–4s extraction removes the reason and the track
    is appended instead. Reading it here rather than at command entry also means
    a song that fails to resolve never stops the paused song.
    """
    source = parse_input(url, ctx.message.content)
    qobj = await _resolve_playnow_source(ctx, source, origin=url, cog=cog)
    qobj.interjected = True

    # Warm the stream-URL cache before interrupting: a cache miss at dequeue puts
    # seconds of yt-dlp dead air between the interrupt and the new song. Awaited,
    # not spawned — the current song plays through the wait. No-op without Redis;
    # also back-fills duration/thumbnail for the embeds below.
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
        qobj.analytics = replace(qobj.analytics, queue_position=mp.enqueue_depth())
        await enqueue_single(ctx, qobj, mp, warning=timestamp_warning(source))
        return

    outcome = await mp.interject(qobj, vc, resume_paused=resume_paused)
    if outcome is None:
        # The song ended during the resolve — nothing left to interrupt. Insert
        # qobj directly rather than re-invoking -play, which would re-parse,
        # re-resolve and (for a playlist) enqueue all tracks right after the
        # first-track-only notice above. Front, not append: the user asked for
        # "now", and this window can be seconds long with songs queued behind.
        # It interrupted nothing, so keeping the marker would attribute an
        # interjection that never happened.
        qobj.interjected = False
        # interject() also returns None when the loop moved on to a
        # DIFFERENT song, which this insert waits behind. One, never the
        # queue depth: it goes to the front.
        qobj.analytics = replace(
            qobj.analytics,
            queue_position=1 if mp.current_song is not None else 0,
        )
        # The player's wrapper, not queue.put_front directly — the same
        # item-vs-list plumbing as every other user-facing insert.
        # prefetch=False — the stream URL was warmed above.
        await mp.queue_put_front(qobj, prefetch=False)
        await asyncio.gather(
            send_embed(
                ctx,
                f"▶️ Playing next: {qobj.title}",
                f"Requested by: [{ctx.author.mention}]\n"
                "The song being interrupted already ended — "
                "queued to play next instead.",
                discord.Color.blue(),
                thumbnail=qobj.thumbnail,
            ),
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
    await asyncio.gather(
        send_embed(
            ctx,
            f"▶️ Playing now: {qobj.title}",
            f"Requested by: [{ctx.author.mention}]\n{desc}",
            discord.Color.blue(),
            thumbnail=qobj.thumbnail,
        ),
        ctx.message.add_reaction("⏯️"),
    )
