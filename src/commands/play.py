"""`-play` — queue a song or playlist, joining the author's channel if needed.

Takes the cog: it runs `-join` through discord.py on the cold path, and tears the
player down when that join produces no usable client.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING, Union

import discord
from discord.ext import commands

from src.guild_state import Analytics
from src.musicplayer import RESTORE_WAIT_SECS
from src.recovery import abandon_cold_start, join_succeeded
from src.sources import (
    parse_input,
    timestamp_warning,
    unquote_argument,
)
from src.util import (
    background_typing,
    get_logger,
    notice_embed,
)
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

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot


async def run(ctx: commands.Context, url: str, *, cog: MusicBot) -> None:
    """`-play` — resolve the input, join if needed, and queue it.

    Takes the cog: the cold path runs -join through discord.py, tears the player
    down when that join produces no usable client, and resolves the player itself.
    """
    # Consume-rest, so a multi-word search arrives whole: this value is what
    # `origin` is stamped from, and -remove matches on that. read_rest hands
    # the quotes through, hence the unquote — a quoted origin is one -remove
    # would have to match literally.
    url = unquote_argument(url.strip())
    async with background_typing(ctx):
        # Paused → interject, not append: appending leaves the bot silent
        # with the request buried behind a paused song. The interrupted song
        # returns PLAYING, unlike -playnow. Checked before parse_input so the
        # paused path parses once, inside interject_flow.
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
            # front: not connected, so this song jumps ahead of any queue
            # restored from Redis (a -stop leaves its queue persisted).
            # -play on a disconnected bot means "play this", not "play
            # the leftovers".
            front = not ctx.voice_client
            # Bound before the join, not after: every failure path below
            # hands this exact player to _abandon_cold_start, and a get_mp()
            # issued after its cleanup() would build and start a fresh one.
            mp = cog.get_mp(ctx)
            # Ask-time analytics, read ONCE at dispatch: the command
            # message's snowflake time, so the wait covers gateway
            # delivery and the resolve below. front ⇒ depth 0, the
            # cold-start song plays ahead of the restored queue.
            if front:
                position = 0
            else:
                # Wait out any in-flight restore: restore_entries() appends,
                # so a put() landing first leaves the deque holding this
                # song ahead of entries Redis lists behind it, and every
                # later commit-time LPOP then retires the wrong one.
                if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                    await ctx.send(
                        embed=notice_embed(
                            "Still loading this server's saved queue — try "
                            "again in a moment.",
                            discord.Color.orange(),
                        )
                    )
                    return
                position = mp.enqueue_depth()
            analytics = Analytics(
                queued_at=ctx.message.created_at.timestamp(),
                queue_position=position,
            )
            if front:
                # Hold the gate across the join below: join opens it the
                # moment the handshake lands, which would start the
                # restored head while queue_source is still extracting.
                # Released on exiting the stack, after the front insertion.
                await stack.enter_async_context(mp.defer_playback())
                # Concurrent with queue_source: both are pure I/O (voice
                # handshake vs yt-dlp extraction) with no data dependency.
                # Awaiting join_task after queue_source guarantees the voice
                # client is ready before queue_put fires.
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
                    # Full cleanup, not just disconnect: cog_before_invoke
                    # already started a MusicPlayer's loop(), which would
                    # zombie for up to 300s on queue.get() with
                    # clear_connection() never firing — spurious crash
                    # recovery on restart.
                    await abandon_cold_start(cog, ctx, mp)
                    raise
                # Inserting onto a join that produced no usable client hands
                # the loop a song it can only raise on.
                if not join_succeeded(ctx):
                    await abandon_cold_start(cog, ctx, mp)
                    return
            else:
                qobj = await play_pipeline.queue_source(
                    ctx, source, analytics=analytics, origin=url, cog=cog
                )

            log.info(f"Voice client: {ctx.voice_client}")

            if front:
                # Order matters: put_front LPUSHes the mirror while
                # restore_entries replays already-listed entries in memory
                # only, so inserting first double-queues this song. A restore
                # that never lands is therefore a reason NOT to insert — and
                # the wait is bounded, since the pool sets no socket_timeout.
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
