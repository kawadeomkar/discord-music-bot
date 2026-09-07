"""The machinery behind `-play` and `-playnow`: resolve the input, place it,
and — for an interjection — put the interrupted song back where it was.

Three stages in the order they run: `queue_source` turns a parsed source into
something enqueueable; `enqueue_playlist` / `enqueue_single` place it and send
the confirmation; `interject_flow` is the -playnow path, shared with `-play` on
a paused song. The playlist errors and the two Resolved* shapes live here
because nothing outside this pipeline constructs them.
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
    *,
    keep_first_only: bool = False,
) -> tuple[list[QueueObject], int]:
    """Drop the tracks ahead of YouTube's 1-based `index=` (a share link copied
    mid-playlist carries the position it was copied at), returning what is left
    and how many went. An index past the end raises rather than queueing
    nothing. The empty-playlist guard lives here so both callers get it.
    keep_first_only trims to the one track -playnow interjects."""
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
    front: bool = False,
) -> None:
    """Queue a resolved playlist and notify the channel."""
    # A playlist front-inserts in full, in order: nothing is playing to
    # interrupt on this path.
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
    """Queue one song and confirm. `warning` is about what the user typed, so
    every exit sends it — the confirmation embed is conditional, the warning is
    not."""
    vc = ctx.voice_client
    if front:
        # The "Est. playing at" embed would be wrong here: a restored queue is
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

    should_show_queued = mp.queue.qsize() > 0 or (
        isinstance(vc, discord.VoiceClient) and vc.is_playing()
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
            if warning is not None:
                await ctx.send(embed=notice_embed(warning, discord.Color.orange()))
            return
        await ctx.send(embed=mp.build_queued_song_embed(qobj, warning=warning))
        return
    if warning is not None:
        # The song starts now and the NP card speaks for it, so the warning
        # needs its own message.
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
    song's return indefinitely. `origin` is the raw command argument: for a
    collapsed playlist the link, not the generated title."""
    playlist_notice = notice_embed(
        "Playlists can't be interjected — playing the **first track** now. "
        "Use `-play` for the full playlist.",
        discord.Color.orange(),
    )
    # Depth 0: an interjection plays immediately. The caller re-mints it on the
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
        # Indexed here too: a link copied mid-playlist interjects the track
        # the user was looking at. The slice makes tracks[0] that track.
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

    Shared by `-playnow` and by `-play` on a paused song; they differ in
    resume_paused (`-playnow` restores paused-in → paused-out, `-play` brings it
    back playing). require_paused re-reads the pause state after resolution:
    `-play` interjects only because the song is paused, so a `-resume` landing
    during the 1–4s extraction removes the reason and the track is appended.
    """
    source = parse_input(url)
    qobj = await _resolve_playnow_source(ctx, source, origin=url, cog=cog)
    qobj.interjected = True

    # Warm the stream-URL cache before interrupting, or a miss at dequeue puts
    # seconds of dead air between the interrupt and the new song. Awaited: the
    # current song plays through the wait. Also back-fills duration/thumbnail.
    await YTDL.prefetch_stream(qobj, redis=cog.redis)

    if require_paused and not vc.is_paused():
        # Resumed during the resolve: append rather than interrupt a song the
        # user just chose to keep playing, and re-mint the depth now that the
        # queue moved.
        qobj.interjected = False
        qobj.analytics = replace(qobj.analytics, queue_position=mp.enqueue_depth())
        await enqueue_single(ctx, qobj, mp, warning=timestamp_warning(source))
        return

    outcome = await mp.interject(qobj, vc, resume_paused=resume_paused)
    if outcome is None:
        # The song ended during the resolve. Front-insert directly: the user
        # asked for "now", the window can be seconds long with songs queued
        # behind, and re-invoking -play would re-resolve (and for a playlist
        # enqueue every track after the first-track notice). It interrupted
        # nothing, so the marker comes off.
        qobj.interjected = False
        # interject() also returns None when the loop moved on to a DIFFERENT
        # song, which this insert waits behind: one, never the queue depth.
        qobj.analytics = replace(
            qobj.analytics,
            queue_position=1 if mp.current_song is not None else 0,
        )
        # prefetch=False: the stream URL was warmed above.
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
