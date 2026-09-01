"""`-play` — queue a song or playlist, joining the author's channel if needed.

Takes the cog: it runs `-join` through discord.py on the cold path, and tears the
player down when that join produces no usable client.
"""

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Optional, Union

import discord
from discord.ext import commands

from opentelemetry import trace

from src.guild_state import Analytics
from src.musicplayer import MusicPlayer
from src.play_placement import (
    NEXT_FLAG,
    NOW_FLAG,
    PlaceStalled,
    resolve_mode_for,
    Placement,
    PlayArgs,
    PlayMode,
    PlayRequest,
    split_play_args,
)
from src.musicplayer import RESTORE_WAIT_SECS
from src.recovery import abandon_cold_start, join_succeeded
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    YTSource,
    parse_input,
    timestamp_warning,
    unquote_argument,
)
from src.util import (
    background_typing,
    get_logger,
    notice_embed,
)
from src.commands._common import echo
from src.youtube import QueueObject

# The stage functions are reached through the MODULE, never from-imported: a
# from-import binds them here at import time, and the seam the tests stub is
# the name on play_pipeline. Same reason youtube.py resolves _ytdlp_extract
# per call.
from src import play_pipeline
from src.play_pipeline import (
    ResolvedSpotifyPlaylist,
    ResolvedYoutubePlaylist,
)

log = get_logger(__name__)


class InterjectionFailed(Exception):
    """A failure on the interjection branch. `-play` and `-play --now` are one
    command with two failure titles, and only the body knows which branch it took;
    the cog's wrapper reads this to pick the wording and renders the cause."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-play` — split the placement flag off, then resolve, join if needed, queue.

    Takes the cog: the cold path runs -join through discord.py, tears the player
    down when that join produces no usable client, and resolves the player itself.
    """
    # Consume-rest, so a multi-word search arrives whole — it is what -remove
    # matches on. The strip covers callers that bypass discord.py's parser.
    args = split_play_args(url.strip())
    # The player is bound here, not after the resolve: every failure path hands
    # this exact one to abandon_cold_start. The query is unquoted to match what
    # _run_placed stores as the entry's origin, which is what -remove matches on.
    req = cog._plays.register(
        ctx,
        query=unquote_argument(args.query),
        mp=cog.get_mp(ctx),
        mode=args.mode,
    )
    try:
        await _run_placed(ctx, args, req, cog=cog)
    except PlaceStalled as stall:
        # The interject route only: _resolve_and_place reports its own, and
        # there is no gate hold to unwind here and nothing to abandon.
        await ctx.send(embed=_place_stalled_notice(before_the_put=stall.before_the_put))
    finally:
        cog._plays.retire(req)


async def _run_placed(
    ctx: commands.Context, args: PlayArgs, req: PlayRequest, *, cog: MusicBot
) -> None:
    """The body behind -play, taking the argument already split and the request
    the registry admitted."""
    trace.get_current_span().set_attribute("play.mode", args.mode.value)
    # ONE rebind, so every `origin=url` below is the flag-free, unquoted query —
    # the form -remove matches against.
    url = unquote_argument(args.query)
    async with background_typing(ctx):
        if args.dash_typo is not None:
            await ctx.send(
                embed=notice_embed(
                    f"Did you mean `{args.dash_typo}`? Options take two dashes.",
                    discord.Color.orange(),
                )
            )
            return
        if not url:
            await ctx.send(
                embed=notice_embed(
                    f"Missing argument: `url`. Usage: `{ctx.prefix}play "
                    f"[{NOW_FLAG}|{NEXT_FLAG}] <url|search>`",
                    discord.Color.red(),
                )
            )
            return

        # The player the registry admitted this request against, so every
        # failure path below tears down the one abandon_cold_start was given.
        mp = req.mp
        vc = ctx.voice_client
        # A narrowed Optional: VoiceProtocol carries neither is_playing nor
        # is_paused, so a bool would leave vc unnarrowed at both use sites.
        live_vc = (
            vc
            if isinstance(vc, discord.VoiceClient)
            and mp.current_song is not None
            and (vc.is_playing() or vc.is_paused())
            else None
        )
        # Decided BEFORE the resolve, which is where whether a song is live
        # can change.
        if live_vc is not None:
            if args.mode is PlayMode.NOW:
                return await _interject(ctx, url, mp, live_vc, req, cog=cog)
            if live_vc.is_paused() and args.mode is not PlayMode.NEXT:
                # Paused -> interject: appending would bury the request behind
                # a paused song. The interrupted song returns PLAYING. `--next`
                # is excluded — it is next either way.
                return await _interject(
                    ctx,
                    url,
                    mp,
                    live_vc,
                    req,
                    cog=cog,
                    resume_paused=False,
                    require_paused=True,
                )

        source = parse_input(url)

        notice = await _resolve_and_place(ctx, args, req, mp, source, url, cog=cog)
        if notice is not None:
            await ctx.send(embed=notice)


async def _resolve_and_place(
    ctx: commands.Context,
    args: PlayArgs,
    req: PlayRequest,
    mp: MusicPlayer,
    source: Union[SpotifySource, YTSource, SoundcloudSource],
    url: str,
    *,
    cog: MusicBot,
) -> Optional[discord.Embed]:
    """Resolve, then insert under the place lock. Returns the notice for a
    request that did not insert, or None. Returned rather than sent: on the cold
    path a teardown decision (abandon_cold_start reads the hold count) must not
    be followed by an await before the gate hold is released."""
    qobj: Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]
    async with contextlib.AsyncExitStack() as stack:
        # The cold-start gate hold lives on its own stack, so the path that PLACES
        # can release it the moment the put lands rather than holding the first note
        # behind a confirmation embed. aclose() is idempotent — an already-unwound
        # stack unwinds nothing — so every other exit still releases through the
        # outer stack exactly as it did.
        hold = await stack.enter_async_context(contextlib.AsyncExitStack())
        # Not connected: this song goes ahead of any queue restored from Redis. A
        # running join counts as cold — discord.py registers the client BEFORE
        # the handshake completes.
        in_flight_join = cog._plays.join_in_flight(req.guild_id)
        cold_start = not ctx.voice_client or in_flight_join
        if cold_start:
            placement = Placement.COLD_FRONT
        elif args.mode is not PlayMode.NORMAL:
            # Both flags: `--now` reaches here only with nothing to interrupt,
            # and that state lasts the length of every resolve.
            placement = Placement.NEXT
        else:
            placement = Placement.TAIL
        # The message's snowflake time, so the wait covers gateway delivery.
        # The depth is minted at the insert, under the place lock.
        analytics = Analytics(
            queued_at=ctx.message.created_at.timestamp(), queue_position=0
        )
        resolve_started = time.monotonic()
        if cold_start:
            # Held across the join, which opens the gate the moment the
            # handshake lands. Released at the insert on the placed path, and
            # with the stack on every other.
            await hold.enter_async_context(mp.defer_playback())
            # One join per guild, concurrent with this resolve: voice
            # handshake and yt-dlp extraction have no data dependency.
            join, owns_join = cog._plays.cold_join(
                req,
                joiner=lambda: ctx.invoke(cog.join),
                tracked=cog._restore_tasks,
            )
            try:
                qobj = await play_pipeline.queue_source(
                    ctx,
                    source,
                    analytics=analytics,
                    origin=url,
                    mode=resolve_mode_for(placement),
                    pool_slot=cog._plays.resolve_slot(req),
                    cog=cog,
                )
                # Stamped before the join wait below, so play.resolve_secs times
                # the extraction alone and not the voice handshake beside it.
                trace.get_current_span().set_attribute(
                    "play.resolve_secs",
                    round(time.monotonic() - resolve_started, 3),
                )
            except BaseException:
                # Alone on this cold start (the hold count is this command's), so
                # the join is cancelled first, before the teardown removes the
                # player it would rebuild. Another holder owns its own join.
                if mp.playback_holds == 1 and not join.done():
                    join.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await join
                # Full cleanup: cog_before_invoke already started a loop(), which
                # otherwise parks 300s on queue.get() with clear_connection()
                # unfired, and the next restart recovers a guild that stopped.
                await abandon_cold_start(cog, ctx, mp)
                raise
            join_started = time.monotonic()
            with contextlib.suppress(Exception):
                # The join task runs on the CREATOR's context, so a waiter's
                # trace has to say the join was somebody else's.
                if not owns_join:
                    trace.get_current_span().set_attribute("play.join_shared", True)
                await asyncio.shield(join)
            trace.get_current_span().set_attribute(
                "play.join_wait_secs", round(time.monotonic() - join_started, 3)
            )
            # Inserting onto a join that produced no usable client hands the
            # loop a song it can only raise on.
            if not join_succeeded(ctx):
                # join() swallows its own error, so this span and this line are
                # the only record of the drop. Synchronous through the return.
                trace.get_current_span().set_attribute("play.dropped_by", "join_failed")
                log.warning(f"Cold-start join left no voice client: {url}")
                await abandon_cold_start(cog, ctx, mp)
                # The creator was told by join() itself. Returned, not sent: no
                # await may land between the teardown decision and the release.
                return None if owns_join else _join_failed_notice(url)
        else:
            qobj = await play_pipeline.queue_source(
                ctx,
                source,
                analytics=analytics,
                origin=url,
                mode=resolve_mode_for(placement),
                pool_slot=cog._plays.resolve_slot(req),
                cog=cog,
            )
            # The cold branch stamps its own above, before the join wait.
            trace.get_current_span().set_attribute(
                "play.resolve_secs",
                round(time.monotonic() - resolve_started, 3),
            )

        log.info(f"Voice client: {ctx.voice_client}")

        # Every placement waits: put_front LPUSHes the list restore_entries
        # replays in memory, and a put() before the replay lands ahead of entries
        # Redis lists behind it. Bounded — the pool sets no socket_timeout.
        if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
            # Cold start ONLY: abandon_cold_start cancels tasks and
            # disconnects, which on a warm player stops the music mid-song.
            if cold_start:
                await abandon_cold_start(cog, ctx, mp)
                return _restore_unreachable_notice()
            return notice_embed(
                "Still loading this server's saved queue — try again in a moment.",
                discord.Color.orange(),
            )

        try:
            if isinstance(qobj, QueueObject):
                await play_pipeline.enqueue_single(
                    ctx,
                    qobj,
                    mp,
                    req,
                    placement=placement,
                    warning=timestamp_warning(source),
                    release_hold=hold.aclose,
                    cog=cog,
                )
            else:
                await play_pipeline.enqueue_playlist(
                    ctx,
                    source,
                    qobj,
                    mp,
                    req,
                    placement=placement,
                    analytics=analytics,
                    origin=url,
                    release_hold=hold.aclose,
                    cog=cog,
                )
        except PlaceStalled as stall:
            if cold_start and not _cold_start_left_something_playable(ctx, mp):
                await abandon_cold_start(cog, ctx, mp)
            return _place_stalled_notice(before_the_put=stall.before_the_put)
        if cold_start and not req.placed:
            # Refused at the lock, with the join already in the channel: tear
            # down like the other exits, unless a sibling or a restore left
            # something playable there.
            if not _cold_start_left_something_playable(ctx, mp):
                await abandon_cold_start(cog, ctx, mp)
            return None
    return None


def _cold_start_left_something_playable(ctx: commands.Context, mp: MusicPlayer) -> bool:
    """Whether a cold start refused AFTER its join should leave the session up:
    with the connection up and the queue not empty, the songs are a sibling's
    or a restored queue's, and a teardown would disconnect a working session.
    Synchronous, so the caller decides before its gate release."""
    return join_succeeded(ctx) and mp.queue.display_size() > 0


def _join_failed_notice(query: str) -> discord.Embed:
    """A waiting request's own report of a failed cold-start join: join() reports
    only into the context of the request that created it."""
    return notice_embed(
        f"Couldn't join the voice channel, so your song wasn't queued: {echo(query)}",
        discord.Color.red(),
    )


def _restore_unreachable_notice() -> discord.Embed:
    """The cold-start restore read never landed, so nothing was inserted."""
    return notice_embed(
        "Couldn't reach this server's saved queue, so your song wasn't queued — "
        "try again in a moment.",
        discord.Color.red(),
    )


def _place_stalled_notice(*, before_the_put: bool) -> discord.Embed:
    """What a stalled placement may claim. Waiting for the lock nothing was written
    and the song is honestly absent; inside the put the deque is appended before the
    mirror write, so the song may be queued and a "try again" would duplicate it."""
    if before_the_put:
        return notice_embed(
            "This server's queue is busy right now, so your song wasn't queued — "
            "try again in a moment.",
            discord.Color.red(),
        )
    return notice_embed(
        "This server's queue is busy right now, so your song may not have been "
        "queued — check `-queue` before trying again.",
        discord.Color.red(),
    )


async def _interject(
    ctx: commands.Context,
    url: str,
    mp: MusicPlayer,
    vc: discord.VoiceClient,
    req: PlayRequest,
    *,
    cog: MusicBot,
    resume_paused: bool = True,
    require_paused: bool = False,
) -> None:
    """Run the interjection branch, tagging anything it raises so the cog titles
    the embed for the branch the user asked for rather than the command they
    typed. Only the body knows which branch it took."""
    try:
        await play_pipeline.interject_flow(
            ctx,
            url,
            mp,
            vc,
            req,
            resume_paused=resume_paused,
            require_paused=require_paused,
            cog=cog,
        )
    except PlaceStalled:
        # A stall is not a failed interjection: run() reports it as a busy queue,
        # and wrapping it here would title that "Failed to play song now".
        raise
    except Exception as e:
        raise InterjectionFailed(e) from e
