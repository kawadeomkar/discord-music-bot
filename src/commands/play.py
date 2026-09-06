"""`-play` — queue a song or playlist, joining the author's channel if needed."""

import asyncio
import contextlib
from typing import TYPE_CHECKING, Union

import discord
from discord.ext import commands

from src.guild_state import Analytics
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

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-play` — resolve the input, join if needed, and queue it. Takes the cog:
    the cold path runs -join through discord.py and tears the player down when
    that join produces no usable client."""
    # `origin` is stamped from this value and -remove matches on it; read_rest
    # hands the quotes through, so a quoted origin would need a literal match.
    url = unquote_argument(url.strip())
    async with background_typing(ctx):
        # Paused → interject, not append: the interrupted song returns PLAYING,
        # unlike -playnow. Before parse_input, so the paused path parses once.
        paused_vc = ctx.voice_client
        if isinstance(paused_vc, discord.VoiceClient) and paused_vc.is_paused():
            paused_mp = cog.get_mp(ctx)
            if paused_mp.current_song is not None:
                return await play_pipeline.interject_flow(
                    ctx,
                    url,
                    paused_mp,
                    paused_vc,
                    resume_paused=False,
                    require_paused=True,
                    cog=cog,
                )

        source = parse_input(url, ctx.message.content)

        qobj: Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]
        async with contextlib.AsyncExitStack() as stack:
            # Not connected: this song jumps ahead of a queue restored from
            # Redis — -play on a disconnected bot means "play this".
            front = not ctx.voice_client
            # Bound before the join: every failure path hands this exact player
            # to abandon_cold_start, and a get_mp() after its cleanup() would
            # build a fresh one.
            mp = cog.get_mp(ctx)
            # Ask-time analytics, read once at dispatch (the message's snowflake
            # time). front ⇒ depth 0; otherwise wait out an in-flight restore, or
            # the depth reads 0 behind a queue about to reappear.
            if front:
                position = 0
            else:
                await mp.wait_for_restore(timeout=DEPTH_RESTORE_WAIT_SECS)
                position = mp.enqueue_depth()
            analytics = Analytics(
                queued_at=ctx.message.created_at.timestamp(),
                queue_position=position,
            )
            if front:
                # Hold the gate across the join: join opens it the moment the
                # handshake lands, which would start the restored head while
                # queue_source is still extracting. Released on exiting the stack.
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

            if front:
                # wait_for_restore BEFORE put_front: put_front LPUSHes what
                # restore_entries replays in memory, so inserting first
                # double-queues this song. A restore that never lands means no insert.
                if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
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
                    front=front,
                    warning=timestamp_warning(source),
                )
            else:
                await play_pipeline.enqueue_playlist(
                    ctx,
                    source,
                    qobj,
                    mp,
                    front=front,
                    analytics=analytics,
                    origin=url,
                )
