"""`-play` — queue a song or playlist, joining the author's channel if needed."""

import asyncio
import contextlib
from typing import TYPE_CHECKING, Union
from collections.abc import AsyncGenerator

import discord
from discord.ext import commands

from opentelemetry import trace

from src.guild_state import Analytics
from src.musicplayer import MusicPlayer
from src.play_placement import (
    NEXT_FLAG,
    NOW_FLAG,
    Placement,
    PlayArgs,
    PlayMode,
    split_play_args,
)
from src.musicplayer import DEPTH_RESTORE_WAIT_SECS, RESTORE_WAIT_SECS
from src.recovery import abandon_cold_start, join_succeeded
from src.sources import parse_input, timestamp_warning, unquote_argument
from src.util import background_typing, get_logger, notice_embed
from src.youtube import QueueObject

# Stage functions resolve through the module per call: the test seam is the name
# on play_pipeline, which a from-import would bind here at import time.
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
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-play` — split the placement flag off, then resolve, join if needed, queue.
    Takes the cog: the cold path runs -join through discord.py and tears the player
    down when that join produces no usable client."""
    # Consume-rest, so a multi-word search arrives whole — it is what -remove
    # matches on. The strip covers callers that bypass discord.py's parser.
    args = split_play_args(url.strip())
    async with play_bucket(cog, ctx, args.mode):
        await _run_placed(ctx, args, cog=cog)


async def run_now(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-playnow` — the same request as `-play --now`, kept as its own command.

    The placement is re-parsed and forced rather than prepended: a user who spells
    a flag here anyway (`-playnow --next x`) would otherwise have it searched for
    as part of the query. One entry point behind both spellings, so the placement
    grammar, the admission and the confirmation cannot fork.
    """
    await run(ctx, f"{NOW_FLAG} {split_play_args(url.strip()).query}", cog=cog)


async def run_next(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-playnext` — the same request as `-play --next`, kept as its own command.

    The `--now` sibling's reasoning, one flag over: the placement is re-parsed and
    forced so a flag the user spells here anyway is not searched for as text.
    """
    await run(ctx, f"{NEXT_FLAG} {split_play_args(url.strip()).query}", cog=cog)


@contextlib.asynccontextmanager
async def play_bucket(
    cog: MusicBot, ctx: commands.Context, mode: PlayMode
) -> AsyncGenerator[None]:
    """One -play in flight per guild PER PLACEMENT, declining rather than
    queueing: a -play resolving a large playlist holds its bucket for the whole
    flat extraction (99.3s measured on 5,547 tracks), and a `-p --now` must not
    wait behind it. Not max_concurrency — prepare() acquires before the flag is
    parsed; MaxConcurrencyReached is raised by hand so the wording does not fork."""
    key = (ctx.guild.id if ctx.guild else 0, mode)
    if key in cog._play_inflight:
        raise commands.MaxConcurrencyReached(1, commands.BucketType.guild)
    # No await between the check and the claim, so the pair is atomic on one
    # event loop and two invocations in the same tick cannot both pass.
    cog._play_inflight.add(key)
    try:
        yield
    finally:
        cog._play_inflight.discard(key)


async def _run_placed(ctx: commands.Context, args: PlayArgs, *, cog: MusicBot) -> None:
    """The body behind -play, taking the argument already split."""
    trace.get_current_span().set_attribute("play.mode", args.mode.value)
    # ONE rebind, so every `origin=url` below is the query with the flag off: a
    # leaked flag persists a user_input -remove cannot match. read_rest hands the
    # quotes through, so a quoted origin would need a literal match.
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

        # Bound before the join: every failure path hands this exact player
        # to abandon_cold_start, and a later get_mp() would build a new one.
        mp = cog.get_mp(ctx)
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
                return await _interject(ctx, url, mp, live_vc, cog=cog)
            if live_vc.is_paused() and args.mode is not PlayMode.NEXT:
                # Paused -> interject: appending would bury the request behind
                # a paused song. The interrupted song returns PLAYING. `--next`
                # is excluded — it is next either way.
                return await _interject(
                    ctx,
                    url,
                    mp,
                    live_vc,
                    cog=cog,
                    resume_paused=False,
                    require_paused=True,
                )

        source = parse_input(url)

        qobj: Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]
        async with contextlib.AsyncExitStack() as stack:
            # Not connected: this song jumps ahead of any queue restored from
            # Redis. The flag decides the analytics shortcut and the join
            # dance; the insert position is `placement`.
            cold_start = not ctx.voice_client
            if cold_start:
                placement = Placement.COLD_FRONT
            elif args.mode is not PlayMode.NORMAL:
                # Both flags: `--now` reaches here only when there was
                # nothing to interrupt — connected, no song live — and the
                # interruption is the only part of it that needs one. That
                # state lasts the length of every resolve.
                placement = Placement.NEXT
            else:
                placement = Placement.TAIL
            # Ask-time analytics, read ONCE at dispatch: the command
            # message's snowflake time, so the wait covers gateway
            # delivery and the resolve below. Cold => depth 0, the
            # cold-start song plays ahead of the restored queue.
            if cold_start:
                position = 0
            elif placement is Placement.NEXT:
                # No restore wait for the depth here — this number does not
                # come from the queue at all, so a queue still replaying
                # cannot make it wrong.
                position = play_pipeline.front_insert_depth(mp)
            else:
                await mp.wait_for_restore(timeout=DEPTH_RESTORE_WAIT_SECS)
                position = mp.enqueue_depth()
            analytics = Analytics(
                queued_at=ctx.message.created_at.timestamp(),
                queue_position=position,
            )
            if cold_start:
                # Hold the gate across the join below: join opens it the
                # moment the handshake lands, which would start the
                # restored head while queue_source is still extracting.
                # Released on exiting the stack, after the front insertion.
                await stack.enter_async_context(mp.defer_playback())
                # Concurrent with queue_source (no data dependency); awaited
                # after it so the voice client is ready before the insert.
                join_task = asyncio.create_task(ctx.invoke(cog.join))
                try:
                    qobj = await play_pipeline.queue_source(
                        ctx, source, analytics=analytics, origin=url, cog=cog
                    )
                    await join_task
                except BaseException:
                    if not join_task.done():
                        join_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await join_task
                    # Full cleanup: the started loop() would otherwise zombie on
                    # queue.get() with clear_connection() never firing.
                    await abandon_cold_start(cog, ctx, mp)
                    raise
                # A song inserted onto a failed join is one the loop can only raise on.
                if not join_succeeded(ctx):
                    await abandon_cold_start(cog, ctx, mp)
                    return
            else:
                qobj = await play_pipeline.queue_source(
                    ctx, source, analytics=analytics, origin=url, cog=cog
                )

            log.info(f"Voice client: {ctx.voice_client}")

            if placement is not Placement.TAIL:
                # put_front LPUSHes the mirror while restore_entries replays
                # already-listed entries in memory only, so inserting against
                # an unread snapshot double-queues this song. Every front
                # insert: connected-and-idle during crash recovery is the
                # window. Bounded, since the pool sets no socket_timeout.
                if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                    # Cold start ONLY: abandon_cold_start cancels the
                    # player's tasks and disconnects it, which on a warm
                    # player would stop the music over a Redis blink.
                    if cold_start:
                        await abandon_cold_start(cog, ctx, mp)
                    await ctx.send(
                        embed=notice_embed(
                            "Couldn't reach this server's saved queue, so "
                            "your song wasn't queued — try again in a "
                            "moment.",
                            discord.Color.red(),
                        )
                    )
                    return

            if isinstance(qobj, QueueObject):
                await play_pipeline.enqueue_single(
                    ctx,
                    qobj,
                    mp,
                    placement=placement,
                    warning=timestamp_warning(source),
                )
            else:
                await play_pipeline.enqueue_playlist(
                    ctx,
                    source,
                    qobj,
                    mp,
                    placement=placement,
                    analytics=analytics,
                    origin=url,
                    cog=cog,
                )


async def _interject(
    ctx: commands.Context,
    url: str,
    mp: MusicPlayer,
    vc: discord.VoiceClient,
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
            resume_paused=resume_paused,
            require_paused=require_paused,
            cog=cog,
        )
    except Exception as e:
        raise InterjectionFailed(e) from e
