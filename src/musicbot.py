import asyncio
import contextlib
from itertools import islice
from typing import Any, Coroutine, List, Optional, Union, assert_never

import discord
from discord.ext import commands

import redis.asyncio as aioredis

from src.musicplayer import MusicPlayer
from src.redis_client import GuildRedisStore
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    spotify_playlist_to_ytsearch,
)
from src.spotify import Spotify
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.telemetry import get_tracer
from src.util import (
    cancel_task,
    latency_color,
    notice_embed,
    queue_message,
    record_span_error,
    send_embed,
    trace_footer,
    get_logger,
)

log = get_logger(__name__)
_tracer = get_tracer(__name__)

from src.youtube import YTDL, QueueObject


def _check_voice_permissions(
    author: Union[discord.Member, discord.User],
    voice_client: Optional[discord.VoiceClient],
    command_name: str,
) -> Optional[str]:
    """Returns an error message string if validation fails, None if OK."""
    if isinstance(author, discord.User):
        return f"You must be a member of this channel {author}"
    if not author.voice or not author.voice.channel:
        return f"You are not connected to a voice channel, you silly baka {author}"
    if (
        command_name != "play"
        and voice_client is not None
        and voice_client.channel != author.voice.channel
    ):
        return f"Bot is already being used in channel {voice_client.channel}"
    return None


class MusicBot(commands.Cog):
    """
    class for music bot
    """

    __slots__ = (
        "bot",
        "mps",
        "spotify",
        "redis",
        "_active_spans",
        "_alone_timers",
        "_restore_tasks",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.redis: Optional[aioredis.Redis] = getattr(bot, "redis", None)
        self.spotify = Spotify(redis=self.redis)
        self.mps = {}
        self._active_spans: dict = {}  # id(ctx) → (Span, context_token)
        self._alone_timers: dict[int, asyncio.Task] = {}
        self._restore_tasks: set[asyncio.Task] = set()

    def get_mp(self, ctx: commands.Context) -> MusicPlayer:
        assert ctx.guild is not None
        if ctx.guild.id in self.mps:
            mp = self.mps[ctx.guild.id]
            mp.set_context(ctx)
            return mp
        mp = MusicPlayer.from_context(self.bot, ctx, redis=self.redis)
        mp.start()
        self.mps[ctx.guild.id] = mp
        return mp

    @_tracer.start_as_current_span("bot.cleanup")
    async def cleanup(self, guild: discord.Guild) -> None:
        # Cancel any pending alone-disconnect timer before the atomic gate so it
        # cannot fire after cleanup completes and attempt a second cleanup.
        existing = self._alone_timers.pop(guild.id, None)
        if existing and not existing.done() and existing is not asyncio.current_task():
            existing.cancel()

        # Atomic pop: only the first caller proceeds; any concurrent call (e.g., from
        # on_voice_state_update firing while stop's disconnect is in-flight) gets None
        # and returns immediately, preventing the KeyError TOCTOU race.
        mp = self.mps.pop(guild.id, None)
        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        if mp is None:
            return
        log.info("going to cleanup/disconnect")
        try:
            # Cancel tasks before disconnecting so the playback loop cannot
            # wake up and start the next song between voice_client.stop() and
            # the task cancellation. VoiceClient.disconnect() calls stop()
            # internally, so audio is silenced when we disconnect below.
            await asyncio.gather(
                cancel_task(mp._prefetch_task),
                cancel_task(mp._progress_task),
                cancel_task(mp._pause_debounce_task),
                cancel_task(mp._player),
                cancel_task(mp._restore_task),
            )
            # Tasks are down — no tick can race this. Dispose of the NP host so
            # no message keeps a mid-song bar frozen by the stop (dedicated NP
            # message → deleted; command-response host → stripped back to its
            # own embeds).
            await mp.retire_np_host_on_stop()
            if guild.voice_client:
                await guild.voice_client.disconnect(force=False)
            if mp.store is not None:
                # Intentional stop — clear channel IDs and now-playing state so
                # on_ready does not attempt to recover this guild after restart.
                await mp.store.clear_connection()
                await mp.store.refresh_ttl()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            record_span_error(trace.get_current_span(), e)
            log.error(f"cleanup error: {type(e).__name__}: {e}", exc_info=True)

    async def cog_before_invoke(self, ctx: commands.Context):
        from structlog.contextvars import bind_contextvars

        bind_contextvars(
            guild_id=str(ctx.guild.id) if ctx.guild else "none",
            user_id=str(ctx.author.id),
            command=ctx.command.name if ctx.command else "unknown",
        )

        cmd_name = ctx.command.name if ctx.command else "unknown"
        span = _tracer.start_span(
            f"command.{cmd_name}",
            attributes={
                "discord.guild_id": str(ctx.guild.id) if ctx.guild else "",
                "discord.user_id": str(ctx.author.id),
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._active_spans[id(ctx)] = (span, token)

        try:
            if ctx.guild is None:
                return
            old_channel = (
                self.mps[ctx.guild.id]._channel if ctx.guild.id in self.mps else None
            )
            mp = self.get_mp(ctx)
            if (
                isinstance(ctx.channel, discord.TextChannel)
                and old_channel != ctx.channel
                and mp.store is not None
                and ctx.guild is not None
            ):
                vc = ctx.guild.voice_client
                if isinstance(vc, discord.VoiceClient) and vc.channel is not None:
                    await mp.store.set_connection(vc.channel.id, ctx.channel.id)
        except Exception as e:
            # cog_after_invoke won't fire if cog_before_invoke raises — end span now.
            self._active_spans.pop(id(ctx))
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, "before_invoke failed")
            span.end()
            otel_context.detach(token)
            raise

    async def cog_after_invoke(self, ctx: commands.Context):
        from structlog.contextvars import clear_contextvars

        clear_contextvars()
        pair = self._active_spans.pop(id(ctx), None)
        if pair:
            span, token = pair
            span.end()
            otel_context.detach(token)

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
        # Peek (don't pop) — cog_after_invoke runs in the finally block after this
        # and is responsible for ending the span.
        pair = self._active_spans.get(id(ctx))
        if pair:
            span, _ = pair
            span.record_exception(error)
            span.set_status(StatusCode.ERROR, str(error))

        # validate_commands already sends its own message before raising CommandError,
        # so only handle errors that produce no user-visible output.
        if isinstance(error, commands.MissingRequiredArgument):
            cmd = ctx.command
            usage = f"`{ctx.prefix}{cmd.name} {cmd.signature}`" if cmd else ""
            await ctx.send(
                embed=notice_embed(
                    f"Missing argument: `{error.param.name}`."
                    + (f" Usage: {usage}" if usage else ""),
                    discord.Color.red(),
                )
            )

    async def validate_commands(self, ctx: commands.Context) -> None:
        vc = ctx.voice_client
        voice_client = vc if isinstance(vc, discord.VoiceClient) else None
        command_name = ctx.command.name if ctx.command is not None else ""
        msg = _check_voice_permissions(ctx.author, voice_client, command_name)
        if msg:
            await ctx.send(embed=notice_embed(msg, discord.Color.red()))
            raise commands.CommandError(msg)

    async def _command_error(
        self,
        ctx: commands.Context,
        e: Exception,
        title: str = "Command failed",
    ) -> None:
        span = trace.get_current_span()
        record_span_error(span, e)
        await send_embed(
            ctx,
            title,
            f"**{type(e).__name__}:** {e}",
            discord.Color.red(),
            footer=trace_footer(span),
        )

    @_tracer.start_as_current_span("bot.queue_source")
    async def queue_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> Union[QueueObject, List[str], List[QueueObject]]:
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            return await self.spotify.playlist(source.id)
        elif isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            if source.list_id is None:
                raise ValueError("YTSource with type=PLAYLIST must have list_id set")
            playlist_url = (
                source.url or f"https://www.youtube.com/playlist?list={source.list_id}"
            )
            return await YTDL.yt_playlist(playlist_url, ctx.author)
        else:
            ts: Optional[int] = None
            search: str
            if isinstance(source, SpotifySource):
                search = await self.spotify.track(source.id)
            elif isinstance(source, YTSource):
                search = source.ytsearch or source.url or ""
                ts = source.ts
            elif isinstance(source, SoundcloudSource):
                search = source.url
            else:
                assert_never(source)
            return await YTDL.yt_source(
                ctx.author, search, source.process or False, ts=ts, redis=self.redis
            )

    @_tracer.start_as_current_span("bot.enqueue_playlist")
    async def _enqueue_playlist(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        qobj: Union[List[str], List[QueueObject]],
        mp: MusicPlayer,
    ) -> None:
        if isinstance(source, SpotifySource):
            titles: List[str] = qobj  # type: ignore[assignment]
            qobjs_yt = spotify_playlist_to_ytsearch(titles)
            log.info(f"ytsearch qobjs: {qobjs_yt}")
            await asyncio.gather(
                send_embed(
                    ctx,
                    "Queued playlist",
                    f"Requested by: [{ctx.author.mention}]\n\n{queue_message(titles)}",
                    discord.Color.blue(),
                ),
                mp.queue_put(qobjs_yt, prefetch=False),
                ctx.message.add_reaction("👍"),
            )
        else:
            assert isinstance(source, YTSource)
            playlist_url = (
                source.url or f"https://www.youtube.com/playlist?list={source.list_id}"
            )
            tracks: List[QueueObject] = qobj  # type: ignore[assignment]
            count = len(tracks)
            log.info(f"yt playlist track count: {count}")
            await asyncio.gather(
                send_embed(
                    ctx,
                    f"Queued playlist — {count} song{'s' if count != 1 else ''}",
                    f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n\n{queue_message([q.title for q in islice(tracks, 10)])}",
                    discord.Color.blue(),
                ),
                mp.queue_put(tracks, prefetch=False),  # type: ignore[arg-type]
                ctx.message.add_reaction("👍"),
            )

    @_tracer.start_as_current_span("bot.enqueue_single")
    async def _enqueue_single(
        self, ctx: commands.Context, qobj: QueueObject, mp: MusicPlayer
    ) -> None:
        vc = ctx.voice_client
        should_show_queued = mp.queue.qsize() > 0 or (
            isinstance(vc, discord.VoiceClient) and vc.is_playing()
        )
        coros: list[Coroutine[Any, Any, Any]] = [
            mp.queue_put(qobj),
            ctx.message.add_reaction("👍"),
        ]
        if should_show_queued:
            coros.append(
                send_embed(
                    ctx,
                    "Queued song",
                    (
                        f"Requested by: [{ctx.author.mention}]\n"
                        f"{qobj.title} - ({qobj.webpage_url})\n"
                        f"Est. playing at {mp.estimated_playing_at()}"
                    ),
                    discord.Color.blue(),
                    thumbnail=qobj.thumbnail,
                )
            )
        await asyncio.gather(*coros)
        log.info(f"play qsize: {mp.queue.qsize()}")

    @commands.command(name="play", aliases=["p", "sing"], help="play a youtube song")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.play")
    async def play(self, ctx: commands.Context, url):
        async with ctx.typing():
            try:
                source = parse_input(url, ctx.message.content)

                qobj: Union[QueueObject, List[str], List[QueueObject]]
                if not ctx.voice_client:
                    # Launch join concurrently with queue_source — both are pure I/O
                    # (Discord WebSocket handshake vs yt-dlp extraction) with no data
                    # dependency between them. await join_task after queue_source
                    # guarantees the voice client is ready before queue_put fires.
                    join_task = asyncio.create_task(ctx.invoke(self.join))
                    try:
                        qobj = await self.queue_source(ctx, source)
                        await join_task
                    except BaseException:
                        if not join_task.done():
                            join_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await join_task
                        # Full cleanup (not just disconnect) — cog_before_invoke already
                        # created a MusicPlayer and started its loop() task. Without
                        # cleanup() that task runs as a zombie for up to 300s waiting on
                        # queue.get(), and store.clear_connection() is never called,
                        # which would trigger spurious crash recovery on restart.
                        if ctx.guild is not None:
                            with contextlib.suppress(Exception):
                                await self.cleanup(ctx.guild)
                        raise
                else:
                    qobj = await self.queue_source(ctx, source)

                mp = self.get_mp(ctx)
                log.info(f"Voice client: {ctx.voice_client}")

                if isinstance(qobj, list):
                    await self._enqueue_playlist(ctx, source, qobj, mp)
                else:
                    assert isinstance(qobj, QueueObject)
                    qobj.user_input = url
                    await self._enqueue_single(ctx, qobj, mp)

            except Exception as e:
                log.error(f"play failed: {type(e).__name__}: {e}", exc_info=True)
                await self._command_error(ctx, e, title="Failed to queue song")

    async def _resolve_playnow_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> QueueObject:
        """Resolve -playnow input to exactly ONE QueueObject. Playlists
        collapse to their first track (interjecting a whole playlist would
        delay the interrupted song's return indefinitely — use -play)."""
        playlist_notice = notice_embed(
            "Playlists can't be interjected — playing the **first track** now. "
            "Use `-play` for the full playlist.",
            discord.Color.orange(),
        )
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            titles = await self.spotify.playlist(source.id)
            if not titles:
                raise ValueError("Playlist has no tracks")
            await ctx.send(embed=playlist_notice)
            yts = spotify_playlist_to_ytsearch(titles[:1])[0]
            return await YTDL.yt_source(
                ctx.author, yts.ytsearch or "", yts.process or False, redis=self.redis
            )
        if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            playlist_url = (
                source.url or f"https://www.youtube.com/playlist?list={source.list_id}"
            )
            tracks = await YTDL.yt_playlist(playlist_url, ctx.author)
            if not tracks:
                raise ValueError("Playlist has no tracks")
            await ctx.send(embed=playlist_notice)
            return tracks[0]
        qobj = await self.queue_source(ctx, source)
        assert isinstance(qobj, QueueObject)
        return qobj

    @commands.command(
        name="playnow",
        aliases=["pn"],
        help="play a song immediately; the current song resumes after",
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.playnow")
    async def playnow(self, ctx: commands.Context, url):
        async with ctx.typing():
            try:
                mp = self.get_mp(ctx)
                vc = ctx.voice_client
                # Nothing live to interrupt → equivalent to -play (also covers
                # not-connected: play joins first). Playlists enqueue in full
                # on this path — interjection semantics don't apply to an
                # idle player (docs/PLAYNOW_PROPOSAL.md §3).
                if (
                    mp.current_song is None
                    or not isinstance(vc, discord.VoiceClient)
                    or not (vc.is_playing() or vc.is_paused())
                ):
                    return await ctx.invoke(self.play, url=url)

                source = parse_input(url, ctx.message.content)
                qobj = await self._resolve_playnow_source(ctx, source)
                qobj.user_input = url
                qobj.interjected = True

                # Warm the stream-URL cache BEFORE interrupting the current
                # song — a cache miss at dequeue would otherwise put seconds
                # of yt-dlp dead air between the interrupt and the playnow
                # song starting. Awaited (not spawned like queue_put's
                # warm-up): the current song keeps playing through the wait,
                # which beats stopping it into silence. No-op without Redis;
                # also back-fills duration/thumbnail for the embeds below.
                await YTDL.prefetch_stream(qobj, redis=self.redis)

                outcome = await mp.interject(qobj, vc)
                if outcome is None:
                    # The song ended while the input was resolving — nothing
                    # to interrupt anymore. The input is already parsed and
                    # resolved, so insert the qobj directly rather than
                    # re-invoking -play, which would re-parse and re-resolve —
                    # and, for a playlist, enqueue ALL tracks right after the
                    # first-track-only notice above. FRONT insert, not append:
                    # the user asked for "now", and this window can be seconds
                    # long (the loop mid-resolve on the next song) with more
                    # songs queued behind it. Reset the marker: a normally
                    # queued song must not trigger replace semantics when a
                    # later -playnow interrupts it.
                    qobj.interjected = False
                    await mp.queue.put_front([qobj])
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

                if outcome.replaced:
                    desc = (
                        f"Replaced **{outcome.interrupted_title}** (also played "
                        f"via `-playnow` — it will not return)."
                    )
                elif outcome.resume_position is None:
                    desc = (
                        f"**{outcome.interrupted_title}** was nearly finished "
                        f"and will not resume."
                    )
                elif outcome.was_paused:
                    desc = (
                        f"**{outcome.interrupted_title}** will return paused at "
                        f"`{outcome.resume_position_str}`."
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
            except Exception as e:
                log.error(f"playnow failed: {type(e).__name__}: {e}", exc_info=True)
                await self._command_error(ctx, e, title="Failed to play song now")

    @commands.command(name="skip", aliases=["sk"], help="skips current song")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.skip")
    async def skip(self, ctx: commands.Context):
        try:
            vc = ctx.voice_client
            if isinstance(vc, discord.VoiceClient) and vc.is_playing():
                vc.stop()
                if not ctx.invoked_parents:
                    await ctx.message.add_reaction("⏭")
        except Exception as e:
            log.error(f"skip failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(name="stop", aliases=["st"], help="stops current song")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.stop")
    async def stop(self, ctx: commands.Context):
        try:
            # Do not call skip before cleanup: skip fires voice_client.stop() which
            # triggers the after callback (play_next.set), giving the playback loop a
            # window to start the next song before the loop task is cancelled.
            # cleanup() cancels _player first, then disconnects — disconnect()
            # internally stops the audio subprocess, so no explicit skip is needed.
            vc = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
            if vc is not None and ctx.guild is not None:
                await ctx.message.add_reaction("👋")
                await self.cleanup(ctx.guild)
        except Exception as e:
            log.error(f"stop failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(name="pause", aliases=["po"], help="pause the current song")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.pause")
    async def pause(self, ctx: commands.Context):
        try:
            vc = ctx.voice_client
            if isinstance(vc, discord.VoiceClient) and vc.is_playing():
                mp = self.get_mp(ctx)
                await mp.pause(vc)
                await ctx.message.add_reaction("⏸️")
                embed = mp.build_pause_confirmation_embed()
                if embed is not None:
                    await ctx.send(embed=embed)
        except Exception as e:
            log.error(f"pause failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(name="resume", aliases=["r"], help="resume the current song")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.resume")
    async def resume(self, ctx: commands.Context):
        try:
            vc = ctx.voice_client
            if (
                isinstance(vc, discord.VoiceClient)
                and not vc.is_playing()
                and vc.is_paused()
            ):
                mp = self.get_mp(ctx)
                await mp.resume(vc)
                await ctx.message.add_reaction("⏭️")
                # If the -pause confirmation hosts the block, re-host it so
                # "⏸️ Paused at…" becomes plain history instead of sitting
                # beneath a live, advancing bar for the rest of the song.
                await mp.rehost_np_after_resume()
        except Exception as e:
            log.error(f"resume failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(name="shuffle", help="shuffles the songs in the queue (3+ songs)")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.shuffle")
    async def shuffle(self, ctx: commands.Context):
        try:
            mp = self.get_mp(ctx)
            async with ctx.typing():
                await ctx.send(
                    embed=notice_embed("Please wait... shuffling", discord.Color.blue())
                )
                msg = await mp.queue_shuffle()
                await ctx.message.add_reaction("🔀")
                await ctx.send(embed=notice_embed(msg, discord.Color.blue()))
        except Exception as e:
            log.error(f"shuffle failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(name="join", aliases=["summon"], help="join the channel")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.join")
    async def join(self, ctx: commands.Context):
        try:
            assert (
                isinstance(ctx.author, discord.Member) and ctx.author.voice is not None
            )
            assert ctx.guild is not None
            channel = ctx.author.voice.channel
            assert channel is not None

            if not ctx.voice_client:
                await channel.connect(timeout=10.0)
            vc = ctx.voice_client
            if isinstance(vc, discord.VoiceClient) and vc.channel != channel:
                await vc.move_to(channel)
            await ctx.guild.change_voice_state(
                channel=channel, self_mute=False, self_deaf=True
            )

            mp = self.get_mp(ctx)
            if mp.store is not None and isinstance(ctx.channel, discord.TextChannel):
                await mp.store.set_connection(channel.id, ctx.channel.id)

            await asyncio.gather(
                ctx.message.add_reaction("👋"),
                ctx.invoke(self.ping),
            )
        except Exception as e:
            log.error(f"join failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(name="clear", aliases=["c"], help="clears the queue")
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.clear")
    async def clear(self, ctx: commands.Context):
        try:
            mp = self.get_mp(ctx)
            cleared = await mp.queue_clear()
            if not cleared:
                await ctx.send(
                    embed=notice_embed(
                        "The queue is already empty.", discord.Color.orange()
                    )
                )
                return
            description = queue_message(cleared)
            await asyncio.gather(
                ctx.message.add_reaction("🗑️"),
                send_embed(
                    ctx,
                    f"Queue cleared — {len(cleared)} song{'s' if len(cleared) != 1 else ''} removed",
                    description,
                    discord.Color.red(),
                ),
            )
        except Exception as e:
            log.error(f"clear failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="remove",
        aliases=["rm"],
        help="remove all queued songs matching a YouTube URL",
    )
    @commands.before_invoke(validate_commands)
    async def remove(self, ctx: commands.Context, url: Optional[str] = None):
        if url is None:
            await ctx.send(
                embed=notice_embed(
                    "`-remove <url>` — removes all songs matching the given URL from the queue. "
                    "The URL must match the YouTube link shown in the **Now Playing** embed.",
                    discord.Color.blue(),
                )
            )
            return
        mp = self.get_mp(ctx)
        positions = await mp.queue_remove(url)
        if not positions:
            await send_embed(
                ctx, "", f"No queued songs found matching: <{url}>", discord.Color.red()
            )
            return
        count = len(positions)
        noun = "song" if count == 1 else "songs"
        pos_label = "Position" if count == 1 else "Positions"
        pos_str = ", ".join(str(p) for p in positions)
        await send_embed(
            ctx,
            f"Removed {count} {noun} from the queue",
            "",
            discord.Color.orange(),
            fields=[
                ("URL", f"<{url}>", False),
                (f"{pos_label} removed", pos_str, False),
            ],
        )
        await ctx.send(embed=mp.queue_embed())
        await ctx.message.add_reaction("🗑️")

    @commands.command(
        name="now", aliases=["np", "rn", "nowplaying"], help="display current song"
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.now")
    async def now(self, ctx: commands.Context):
        try:
            mp = self.get_mp(ctx)
            vc = ctx.guild.voice_client if ctx.guild else None
            song = mp.current_song
            if (
                vc is not None
                and isinstance(vc, discord.VoiceClient)
                and (vc.is_playing() or vc.is_paused())
                and song is not None
            ):
                if ctx.channel.id != mp._channel.id:
                    # Invoked outside the player's home channel: the host never
                    # leaves home, so answer HERE with a static snapshot (the
                    # MusicContext channel guard keeps it unattached).
                    await ctx.send(embed=mp._build_now_playing_embed(song))
                    return
                # Re-host the live NP block at the bottom of the channel (the
                # old host is retired) instead of sending a static snapshot
                # that immediately goes stale.
                if await mp.repin_now_playing():
                    return
                # Song ended between the liveness check and the repin — fall
                # through to the static/none responses instead of silence.
            if mp.play_message is not None:
                # Crash-recovery window: current_song isn't live yet, but a
                # now-playing snapshot survived the restart. Best-effort static
                # embed (no live progress bar) until loop() starts real playback.
                await ctx.send(embed=mp.play_message)
            else:
                await ctx.send(
                    embed=notice_embed(
                        "No songs are currently playing.", discord.Color.orange()
                    )
                )
        except Exception as e:
            log.error(f"now failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="history", aliases=["h"], help="display history of songs played"
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.history")
    async def history(self, ctx: commands.Context):
        try:
            mp = self.get_mp(ctx)
            if mp and mp.history:
                q_history = queue_message(list(mp.history)[:10])
                await ctx.send(
                    embed=notice_embed(q_history, discord.Color.blue(), title="History")
                )
        except Exception as e:
            log.error(f"history failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="jump", aliases=["j"], help="jumps to a specific position in queue"
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.jump")
    async def jump(self, ctx: commands.Context):
        try:
            await ctx.send(
                embed=notice_embed("currently in development", discord.Color.blue())
            )
        except Exception as e:
            log.error(f"jump failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="queue", aliases=["q"], help="displays current songs in queue"
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.queue")
    async def queue(self, ctx: commands.Context):
        try:
            mp = self.get_mp(ctx)
            await ctx.send(embed=mp.queue_embed())
        except Exception as e:
            log.error(f"queue failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="volume",
        aliases=["v", "vol", "sound"],
        help="volume level between 0 and 100",
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.volume")
    async def volume(self, ctx: commands.Context, volume):
        try:
            if isinstance(volume, str):
                try:
                    volume = int(volume)
                except ValueError:
                    await ctx.send(
                        embed=notice_embed(
                            "Volume must be a number between 0 and 100",
                            discord.Color.red(),
                        )
                    )
                    return
            if not 0 <= volume <= 100:
                return await ctx.send(
                    embed=notice_embed(
                        "Volume must be between 0 and 100", discord.Color.red()
                    )
                )
            mp = self.get_mp(ctx)
            mp.volume = volume / 100
            if mp.store is not None:
                await mp.store.set_volume(mp.volume)
            await ctx.send(
                embed=notice_embed(
                    f"Set volume to {volume}% (takes effect on next song)",
                    discord.Color.blue(),
                )
            )
        except Exception as e:
            log.error(f"volume failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="ping", aliases=["latency", "l", "delay"], help="latency in milliseconds"
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.ping")
    async def ping(self, ctx: commands.Context):
        try:
            ms = self.bot.latency * 1000
            await send_embed(
                ctx,
                "Ping - latency in ms",
                f"Ping: **{round(ms)}** milliseconds!",
                latency_color(ms),
            )
        except Exception as e:
            log.error(f"ping failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    # ── Alone-channel disconnect ──────────────────────────────────────────────

    async def _alone_countdown(self, guild: discord.Guild) -> None:
        try:
            mp = self.mps.get(guild.id)

            if mp is not None:
                try:
                    # send_with_np (not a bare channel send): this can fire
                    # mid-song, and a bare send would bury the NP host message.
                    embed = discord.Embed(
                        title="No users remaining in voice channel",
                        description="All users have disconnected. The bot will disconnect in **10 seconds** unless someone rejoins.",
                        color=discord.Color.orange(),
                    )
                    await mp.send_with_np(embed=embed)
                except Exception as e:
                    log.warning(
                        f"Failed to send alone-countdown notice in guild {guild.id}: {e}"
                    )

            await asyncio.sleep(10)

            # Span covers only the post-sleep decision so it doesn't stay open for
            # the full 10 seconds (which confuses OTLP exporters and leaks OTel context).
            with _tracer.start_as_current_span(
                "bot.alone_countdown",
                attributes={"discord.guild_id": str(guild.id)},
            ):
                vc = guild.voice_client
                if (
                    isinstance(vc, discord.VoiceClient)
                    and vc.channel is not None
                    and not any(not m.bot for m in vc.channel.members)
                ):
                    log.info(
                        f"Bot still alone in guild {guild.id} after 10s — disconnecting"
                    )
                    await self.cleanup(guild)
        except asyncio.CancelledError:
            pass  # user rejoined or explicit stop; timer was cancelled
        except Exception as e:
            log.error(f"_alone_countdown error in guild {guild.id}: {e}", exc_info=True)
        finally:
            self._alone_timers.pop(guild.id, None)

    # ── Restart recovery listeners ────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self):
        """Fires on cold start or session loss (NOT on WebSocket resume).
        Spawns a recovery task per guild so we don't block the event loop."""
        if self.redis is None:
            return
        for guild in self.bot.guilds:
            task = asyncio.create_task(self._restore_guild(guild))
            self._restore_tasks.add(task)
            task.add_done_callback(self._restore_tasks.discard)

    @_tracer.start_as_current_span("guild.restore")
    async def _restore_guild(self, guild: discord.Guild) -> None:
        """Attempt to rejoin voice and restore queue for one guild after restart."""
        if self.redis is None:
            return
        if guild.id in self.mps:
            return

        store = GuildRedisStore(self.redis, guild.id)

        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        # Distributed lock prevents two bot instances from racing on the same guild.
        # Acquired inside the span so the SET NX EX Redis call is a child span.
        if not await store.acquire_recovery_lock():
            trace.get_current_span().set_attribute("restore.skipped_lock", True)
            log.info(
                f"Recovery lock held by another instance for guild {guild.id}, skipping"
            )
            return
        try:
            # One pipelined read serves every gate below: the connection gate
            # (state hash) and the anything-to-restore gate (queue length +
            # crashed song). Only the queue *length* is read here — the actual
            # queue/now-playing/history payload is re-read by _restore_state
            # after a successful connect, so a stopped guild's leftover queue
            # never rides the wire on the common "nothing to do" path.
            gate = await store.get_recovery_gate()
            if gate is None:
                # Redis read failed — do NOT treat as "nothing to restore". Skip
                # this attempt; the recovery lock expires in 60s and the next
                # on_ready (session re-establishment) retries.
                log.warning(f"Recovery skipped for guild {guild.id}: state read failed")
                return
            guild_state = gate.state
            # Equivalent to `not guild_state.has_active_connection`, spelled as
            # explicit None checks so the channel IDs narrow to int below.
            vc_id = guild_state.voice_channel_id
            tc_id = guild_state.text_channel_id
            if vc_id is None or tc_id is None:
                return

            voice_channel = guild.get_channel(vc_id)
            text_channel = guild.get_channel(tc_id)
            voice_ok = isinstance(voice_channel, discord.VoiceChannel)
            text_ok = isinstance(text_channel, discord.TextChannel)

            if not voice_ok or not text_ok:
                # Clear stale channel IDs so this guild is not re-attempted on every reconnect.
                await store.clear_connection()
                trace.get_current_span().set_attribute("restore.channel_missing", True)
                log.warning(
                    f"Recovery skipped for guild {guild.id}: "
                    f"voice_channel_id={vc_id} (resolved={voice_ok}) "
                    f"text_channel_id={tc_id} (resolved={text_ok})"
                )

                notify_channel: Optional[discord.TextChannel] = None
                if text_ok:
                    notify_channel = text_channel  # type: ignore[assignment]
                elif guild.me is not None:
                    if (
                        guild.system_channel is not None
                        and guild.system_channel.permissions_for(guild.me).send_messages
                    ):
                        notify_channel = guild.system_channel
                    else:
                        notify_channel = next(
                            (
                                ch
                                for ch in guild.text_channels
                                if ch.permissions_for(guild.me).send_messages
                            ),
                            None,
                        )

                if notify_channel is not None:
                    deleted: List[str] = []
                    if not voice_ok:
                        deleted.append("voice channel")
                    if not text_ok:
                        deleted.append("text channel")
                    what = " and ".join(deleted)
                    verb = "was" if len(deleted) == 1 else "were"
                    try:
                        await notify_channel.send(
                            embed=notice_embed(
                                f"⚠️ I came back online but the {what} I was playing in "
                                f"{verb} deleted. Use `-play` in a voice channel to start fresh.",
                                discord.Color.orange(),
                            )
                        )
                    except Exception as notify_err:
                        log.warning(
                            f"Failed to send channel-deleted notification for "
                            f"guild {guild.id}: {notify_err}"
                        )
                return

            # Check there is something to restore before connecting.
            if not gate.has_restorable_playback:
                return

            trace.get_current_span().set_attribute(
                "restore.queue_count", gate.pending_count
            )
            trace.get_current_span().set_attribute(
                "restore.crashed_song", guild_state.has_crashed_song
            )

            try:
                await voice_channel.connect(timeout=30.0, reconnect=True)
                await guild.change_voice_state(
                    channel=voice_channel, self_mute=False, self_deaf=True
                )
            except Exception as e:
                trace.get_current_span().set_attribute(
                    "restore.voice_connect_failed", True
                )
                trace.get_current_span().record_exception(e)
                trace.get_current_span().set_status(
                    StatusCode.ERROR, f"voice connect failed: {e}"
                )
                log.warning(f"Could not rejoin voice for guild {guild.id}: {e}")
                return

            mp = MusicPlayer(self.bot, guild, text_channel, self, redis=self.redis)
            mp.start()
            self.mps[guild.id] = mp

            log.info(
                f"Restored guild {guild.id} in #{text_channel.name} / {voice_channel.name}"
            )
        except Exception as e:
            record_span_error(trace.get_current_span(), e)
            log.error(f"_restore_guild failed for guild {guild.id}: {e}", exc_info=True)
        finally:
            await store.release_recovery_lock()

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        guild = member.guild

        # ── Case A: bot itself was disconnected or moved ──────────────────────
        if self.bot.user is not None and member.id == self.bot.user.id:
            if before.channel is not None and after.channel is None:
                # Bot ejected — full cleanup.
                if guild.id in self.mps:
                    with _tracer.start_as_current_span(
                        "bot.voice_state_update",
                        attributes={"discord.guild_id": str(guild.id)},
                    ):
                        log.info(
                            f"Bot disconnected from voice in guild {guild.id}, cleaning up"
                        )
                        await self.cleanup(guild)
            elif before.channel is not None and after.channel is not None:
                # Bot moved to a different channel — cancel any stale alone-timer
                # that was counting down for the old channel.
                existing = self._alone_timers.pop(guild.id, None)
                if existing and not existing.done():
                    existing.cancel()
            return

        # ── Case B: a human member's voice state changed ──────────────────────
        if guild.id not in self.mps:
            return  # bot isn't active in this guild

        vc = guild.voice_client
        if not isinstance(vc, discord.VoiceClient) or vc.channel is None:
            return

        # Skip mute/deafen/server-deafen events — channel is unchanged.
        if before.channel == after.channel:
            return

        # Only care about events that affect the bot's current channel.
        if before.channel != vc.channel and after.channel != vc.channel:
            return

        human_members = [m for m in vc.channel.members if not m.bot]

        if len(human_members) == 0:
            # Bot is now alone — start (or restart) the 10-second countdown.
            existing = self._alone_timers.pop(guild.id, None)
            if existing and not existing.done():
                existing.cancel()
            log.info(f"Bot is alone in guild {guild.id}, starting 10s disconnect timer")
            self._alone_timers[guild.id] = asyncio.create_task(
                self._alone_countdown(guild)
            )
        else:
            # A human is present — cancel any running alone-timer.
            existing = self._alone_timers.pop(guild.id, None)
            if existing and not existing.done():
                log.info(f"User rejoined guild {guild.id}, cancelling alone timer")
                existing.cancel()


async def setup(bot):
    await bot.add_cog(MusicBot(bot))
