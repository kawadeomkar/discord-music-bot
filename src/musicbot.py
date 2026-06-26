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
from src.util import queue_message, send_queue_phrases, get_logger

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


def _latency_color(ms: float) -> int:
    if ms <= 50:
        return 0x44FF44
    if ms <= 100:
        return 0xFFD000
    if ms <= 200:
        return 0xFF6600
    return 0x990000


class MusicBot(commands.Cog):
    """
    class for music bot
    """

    __slots__ = ("bot", "mps", "spotify", "redis", "_active_spans")

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.redis: Optional[aioredis.Redis] = getattr(bot, "redis", None)
        self.spotify = Spotify(redis=self.redis)
        self.mps = {}
        self._active_spans: dict = {}  # id(ctx) → (Span, context_token)

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
        # Atomic pop: only the first caller proceeds; any concurrent call (e.g., from
        # on_voice_state_update firing while stop's disconnect is in-flight) gets None
        # and returns immediately, preventing the KeyError TOCTOU race.
        mp = self.mps.pop(guild.id, None)
        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        if mp is None:
            return
        log.info("going to cleanup/disconnect")
        if guild.voice_client:
            await guild.voice_client.disconnect(force=False)
        try:
            if mp._prefetch_task and not mp._prefetch_task.done():
                mp._prefetch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await mp._prefetch_task
            if mp._restore_task and not mp._restore_task.done():
                mp._restore_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await mp._restore_task
            if mp._player and not mp._player.done():
                mp._player.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await mp._player
            if mp._store is not None:
                # Intentional stop — clear channel IDs and now-playing state so
                # on_ready does not attempt to recover this guild after restart.
                await mp._store.clear_connection()
                await mp._store.refresh_ttl()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            trace.get_current_span().record_exception(e)
            trace.get_current_span().set_status(
                StatusCode.ERROR, f"{type(e).__name__}: {e}"
            )
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
            self.get_mp(ctx)
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
                f"Missing argument: `{error.param.name}`."
                + (f" Usage: {usage}" if usage else "")
            )

    async def validate_commands(self, ctx: commands.Context) -> None:
        vc = ctx.voice_client
        voice_client = vc if isinstance(vc, discord.VoiceClient) else None
        command_name = ctx.command.name if ctx.command is not None else ""
        msg = _check_voice_permissions(ctx.author, voice_client, command_name)
        if msg:
            await ctx.send(msg)
            raise commands.CommandError(msg)

    async def _command_error(
        self,
        ctx: commands.Context,
        e: Exception,
        title: str = "Command failed",
    ) -> None:
        span = trace.get_current_span()
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
        span_ctx = span.get_span_context()
        embed = discord.Embed(
            title=title,
            description=f"**{type(e).__name__}:** {e}",
            color=discord.Color.red(),
        )
        if span_ctx.is_valid:
            embed.set_footer(text=f"trace: {format(span_ctx.trace_id, '032x')}")
        await ctx.send(embed=embed)

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
            embed = discord.Embed(
                title="Queued playlist",
                description=f"Requested by: [{ctx.author.mention}]\n\n{queue_message(titles)}",
                color=discord.Color.blue(),
            )
            await asyncio.gather(
                ctx.send(embed=embed),
                mp.queue_put(qobjs_yt, prefetch=False),
                ctx.message.add_reaction("👍"),
                send_queue_phrases(ctx),
            )
        else:
            assert isinstance(source, YTSource)
            playlist_url = (
                source.url or f"https://www.youtube.com/playlist?list={source.list_id}"
            )
            tracks: List[QueueObject] = qobj  # type: ignore[assignment]
            count = len(tracks)
            log.info(f"yt playlist track count: {count}")
            embed = discord.Embed(
                title=f"Queued playlist — {count} song{'s' if count != 1 else ''}",
                description=(
                    f"Requested by: [{ctx.author.mention}]\n"
                    f"{playlist_url}\n\n{queue_message([q.title for q in islice(tracks, 10)])}"
                ),
                color=discord.Color.blue(),
            )
            await asyncio.gather(
                ctx.send(embed=embed),
                mp.queue_put(tracks, prefetch=False),  # type: ignore[arg-type]
                ctx.message.add_reaction("👍"),
                send_queue_phrases(ctx),
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
            send_queue_phrases(ctx),
        ]
        if should_show_queued:
            coros.append(
                ctx.send(
                    embed=discord.Embed(
                        title="Queued song",
                        description=(
                            f"Requested by: [{ctx.author.mention}]\n"
                            f"{qobj.title} - ({qobj.webpage_url})"
                        ),
                        color=discord.Color.blue(),
                    )
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
                    await self._enqueue_single(ctx, qobj, mp)

            except Exception as e:
                log.error(f"play failed: {type(e).__name__}: {e}", exc_info=True)
                await self._command_error(ctx, e, title="Failed to queue song")

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
            await ctx.invoke(self.skip)
            if ctx.voice_client and ctx.guild is not None:
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
                vc.pause()
                await ctx.message.add_reaction("⏸️")
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
                vc.resume()
                await ctx.message.add_reaction("⏭️")
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
                await ctx.send("Please wait... shuffling")
                msg = await mp.queue_shuffle()
                await ctx.message.add_reaction("🔀")
                await ctx.send(msg)
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
            if mp._store is not None and isinstance(ctx.channel, discord.TextChannel):
                await mp._store.set_connection(channel.id, ctx.channel.id)

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
                await ctx.send("The queue is already empty.")
                return
            description = queue_message(cleared)
            await asyncio.gather(
                ctx.message.add_reaction("🗑️"),
                ctx.send(
                    embed=discord.Embed(
                        title=f"Queue cleared — {len(cleared)} song{'s' if len(cleared) != 1 else ''} removed",
                        description=description,
                        color=discord.Color.red(),
                    )
                ),
            )
        except Exception as e:
            log.error(f"clear failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    @commands.command(
        name="now", aliases=["np", "rn", "nowplaying"], help="display current song"
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.now")
    async def now(self, ctx: commands.Context):
        try:
            mp = self.get_mp(ctx)
            vc = ctx.guild.voice_client if ctx.guild else None
            if (
                vc is not None
                and isinstance(vc, discord.VoiceClient)
                and vc.is_playing()
                and mp.play_message
            ):
                await ctx.send(embed=mp.play_message)
            else:
                await ctx.send("No songs are currently playing.")
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
                await ctx.send(q_history)
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
            await ctx.send("currently in development")
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
            if mp and len(mp.song_queue) > 0:
                q_songs = mp.get_queue()
                await ctx.send(q_songs)
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
                    await ctx.send("Volume must be a number between 0 and 100")
                    return
            if not 0 <= volume <= 100:
                return await ctx.send("Volume must be between 0 and 100")
            mp = self.get_mp(ctx)
            mp.volume = volume / 100
            await mp.redis_set_state("volume", str(mp.volume))
            await ctx.send(f"Set volume to {volume}% (takes effect on next song)")
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
            embed = discord.Embed(
                title="Ping - latency in ms",
                description=f"Ping: **{round(ms)}** milliseconds!",
            )
            embed.color = _latency_color(ms)
            await ctx.send(embed=embed)
        except Exception as e:
            log.error(f"ping failed: {type(e).__name__}: {e}", exc_info=True)
            await self._command_error(ctx, e)

    # ── Restart recovery listeners ────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self):
        """Fires on cold start or session loss (NOT on WebSocket resume).
        Spawns a recovery task per guild so we don't block the event loop."""
        if self.redis is None:
            return
        for guild in self.bot.guilds:
            asyncio.create_task(self._restore_guild(guild))

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
            vc_id, tc_id = await store.get_connection()
            if vc_id is None or tc_id is None:
                return

            voice_channel = guild.get_channel(vc_id)
            text_channel = guild.get_channel(tc_id)
            if not isinstance(voice_channel, discord.VoiceChannel) or not isinstance(
                text_channel, discord.TextChannel
            ):
                return

            # Check there is something to restore before connecting.
            queue_items = await store.get_queue()
            state = await store.get_state()
            has_crashed_song = bool(state.get(b"current_song_url", b""))
            if not queue_items and not has_crashed_song:
                return

            trace.get_current_span().set_attribute(
                "restore.queue_count", len(queue_items)
            )
            trace.get_current_span().set_attribute(
                "restore.crashed_song", has_crashed_song
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
            trace.get_current_span().record_exception(e)
            trace.get_current_span().set_status(
                StatusCode.ERROR, f"{type(e).__name__}: {e}"
            )
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
        """Detect when the bot itself is disconnected from a voice channel."""
        if self.bot.user is None or member.id != self.bot.user.id:
            return
        if before.channel is not None and after.channel is None:
            # Bot was removed from voice — clean up in-memory state.
            guild = member.guild
            if guild.id in self.mps:
                with _tracer.start_as_current_span(
                    "bot.voice_state_update",
                    attributes={"discord.guild_id": str(guild.id)},
                ):
                    log.info(
                        f"Bot disconnected from voice in guild {guild.id}, cleaning up"
                    )
                    await self.cleanup(guild)


async def setup(bot):
    await bot.add_cog(MusicBot(bot))
