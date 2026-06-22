import asyncio
import contextlib
import random
from typing import List, Optional, Union

import discord
from discord.ext import commands

import redis.asyncio as aioredis

from src.musicplayer import MusicPlayer
from src.redis_client import GuildRedisStore
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    YTSource,
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

    async def cleanup(self, guild: discord.Guild) -> None:
        # Atomic pop: only the first caller proceeds; any concurrent call (e.g., from
        # on_voice_state_update firing while stop's disconnect is in-flight) gets None
        # and returns immediately, preventing the KeyError TOCTOU race.
        mp = self.mps.pop(guild.id, None)
        if mp is None:
            return

        with _tracer.start_as_current_span(
            "bot.cleanup",
            attributes={"discord.guild_id": str(guild.id)},
        ) as span:
            log.info("going to cleanup/disconnect")
            try:
                # Cancel tasks before disconnecting so the playback loop cannot
                # wake up and start the next song between voice_client.stop() and
                # the task cancellation. VoiceClient.disconnect() calls stop()
                # internally, so audio is silenced when we disconnect below.
                if mp._prefetch_task and not mp._prefetch_task.done():
                    mp._prefetch_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await mp._prefetch_task
                if mp._player and not mp._player.done():
                    mp._player.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await mp._player
                if mp._restore_task and not mp._restore_task.done():
                    mp._restore_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await mp._restore_task
                if guild.voice_client:
                    await guild.voice_client.disconnect(force=False)
                if mp._store is not None:
                    # Intentional stop — clear channel IDs and now-playing state so
                    # on_ready does not attempt to recover this guild after restart.
                    await mp._store.clear_connection()
                    await mp._store.refresh_ttl()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
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

    async def queue_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> Union[QueueObject, List[str]]:
        if isinstance(source, SpotifySource) and source.type == "playlist":
            return await self.spotify.playlist(source.id)
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
                raise ValueError(f"Unknown source type: {type(source)}")
            return await YTDL.yt_source(
                ctx.author, search, source.process or False, ts=ts, redis=self.redis
            )

    @commands.command(
        name="play", aliases=["p", "pl", "pla", "sing"], help="play a youtube song"
    )
    @commands.before_invoke(validate_commands)
    async def play(self, ctx: commands.Context, url):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if not voice_client:
            await ctx.invoke(self.join)
            voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        mp = self.get_mp(ctx)
        log.info(f"Voice client: {voice_client}")

        async with ctx.typing():
            try:
                source = parse_input(url, ctx.message.content)
                qobj: Union[QueueObject, List[str]] = await self.queue_source(
                    ctx, source
                )

                if (
                    isinstance(qobj, list)
                    and isinstance(source, SpotifySource)
                    and source.type == "playlist"
                ):
                    qobjs = spotify_playlist_to_ytsearch(qobj)
                    log.info(f"ytsearch qobjs: {qobjs}")

                    description = queue_message(qobj)
                    embed_description = (
                        f"Requested by: [{ctx.author.mention}]\n\n" + description
                    )
                    title = f"Queued playlist"

                    await ctx.send(
                        embed=discord.Embed(
                            title=title,
                            description=embed_description,
                            color=discord.Color.blue(),
                        )
                    )
                    await mp.queue_put(qobjs)

                else:
                    assert isinstance(qobj, QueueObject)
                    qobj.user_input = url
                    vc = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
                    if mp.queue.qsize() > 0 or (
                        vc is not None
                        and isinstance(vc, discord.VoiceClient)
                        and vc.is_playing()
                    ):
                        title = f"Queued song"
                        description = (
                            f"Requested by: [{ctx.author.mention}]\n"
                            f"{qobj.title} - ({qobj.webpage_url})"
                        )
                        await ctx.send(
                            embed=discord.Embed(
                                title=title,
                                description=description,
                                color=discord.Color.blue(),
                            )
                        )
                    await mp.queue_put(qobj)
                    log.info(f"play qsize: {mp.queue.qsize()}")

                await ctx.message.add_reaction("👍")
                await send_queue_phrases(ctx)
            except Exception as e:
                log.error(f"play failed: {type(e).__name__}: {e}", exc_info=True)
                span_ctx = trace.get_current_span().get_span_context()
                embed = discord.Embed(
                    title="Failed to queue song",
                    description=f"**{type(e).__name__}:** {e}",
                    color=discord.Color.red(),
                )
                if span_ctx.is_valid:
                    embed.set_footer(text=f"trace: {format(span_ctx.trace_id, '032x')}")
                await ctx.send(embed=embed)

    @commands.command(name="skip", aliases=["sk"], help="skips current song")
    @commands.before_invoke(validate_commands)
    async def skip(self, ctx: commands.Context):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if (
            voice_client is not None
            and isinstance(voice_client, discord.VoiceClient)
            and voice_client.is_playing()
        ):
            voice_client.stop()
            if not ctx.invoked_parents:
                await ctx.message.add_reaction("⏭")

    @commands.command(name="stop", aliases=["st"], help="stops current song")
    @commands.before_invoke(validate_commands)
    async def stop(self, ctx: commands.Context):
        # Do not call skip before cleanup: skip fires voice_client.stop() which
        # triggers the after callback (play_next.set), giving the playback loop a
        # window to start the next song before the loop task is cancelled.
        # cleanup() cancels _player first, then disconnects — disconnect()
        # internally stops the audio subprocess, so no explicit skip is needed.
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice_client and ctx.guild is not None:
            await ctx.message.add_reaction("👋")
            await self.cleanup(ctx.guild)

    @commands.command(name="pause", aliases=["po"], help="pause the current song")
    @commands.before_invoke(validate_commands)
    async def pause(self, ctx: commands.Context):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if (
            voice_client is not None
            and isinstance(voice_client, discord.VoiceClient)
            and voice_client.is_playing()
        ):
            voice_client.pause()
            await ctx.message.add_reaction("⏸️")

    @commands.command(name="resume", aliases=["r"], help="resume the current song")
    @commands.before_invoke(validate_commands)
    async def resume(self, ctx: commands.Context):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if (
            voice_client is not None
            and isinstance(voice_client, discord.VoiceClient)
            and not voice_client.is_playing()
            and voice_client.is_paused()
        ):
            voice_client.resume()
            await ctx.message.add_reaction("⏭️")

    @commands.command(name="shuffle", help="shuffles the songs in the queue (3+ songs)")
    @commands.before_invoke(validate_commands)
    async def shuffle(self, ctx: commands.Context):
        mp = self.get_mp(ctx)
        async with ctx.typing():
            await ctx.send("Please wait... shuffling")
            msg = await mp.queue_shuffle()
            await ctx.message.add_reaction("🔀")
            await ctx.send(msg)

    @commands.command(name="join", aliases=["summon"], help="join the channel")
    @commands.before_invoke(validate_commands)
    async def join(self, ctx: commands.Context):
        assert isinstance(ctx.author, discord.Member) and ctx.author.voice is not None
        assert ctx.guild is not None
        channel = ctx.author.voice.channel
        assert channel is not None
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if not voice_client:
            await channel.connect(timeout=10.0)
        vc = ctx.voice_client
        if (
            vc is not None
            and isinstance(vc, discord.VoiceClient)
            and vc.channel != channel
        ):
            await vc.move_to(channel)
        await ctx.guild.change_voice_state(
            channel=channel, self_mute=False, self_deaf=True
        )

        mp = self.get_mp(ctx)
        if mp._store is not None and isinstance(ctx.channel, discord.TextChannel):
            await mp._store.set_connection(channel.id, ctx.channel.id)

        await ctx.message.add_reaction("👋")
        await ctx.invoke(self.ping)

    @commands.command(
        name="clear", aliases=["c"], help="clears the queue, in development"
    )
    @commands.before_invoke(validate_commands)
    async def clear(self, ctx: commands.Context):
        await ctx.send(f"in development, use -stop and -join to clear")

    @commands.command(
        name="remove",
        aliases=["rm"],
        help="remove all queued songs matching a YouTube URL",
    )
    @commands.before_invoke(validate_commands)
    async def remove(self, ctx: commands.Context, url: Optional[str] = None):
        if url is None:
            await ctx.send(
                "`-remove <url>` — removes all songs matching the given URL from the queue. "
                "The URL must match the YouTube link shown in the **Now Playing** embed."
            )
            return
        mp = self.get_mp(ctx)
        positions = await mp.queue_remove(url)
        if not positions:
            await ctx.send(f"No queued songs found matching: <{url}>")
            return
        count = len(positions)
        noun = "song" if count == 1 else "songs"
        pos_label = "Position" if count == 1 else "Positions"
        pos_str = ", ".join(str(p) for p in positions)
        removal_embed = (
            discord.Embed(
                title=f"Removed {count} {noun} from the queue",
                color=discord.Color.orange(),
            )
            .add_field(name="URL", value=f"<{url}>", inline=False)
            .add_field(name=f"{pos_label} removed", value=pos_str, inline=False)
        )
        await ctx.send(embed=removal_embed)
        await ctx.send(embed=mp.get_queue())
        await ctx.message.add_reaction("🗑️")

    @commands.command(
        name="now", aliases=["np", "rn", "nowplaying"], help="display current song"
    )
    @commands.before_invoke(validate_commands)
    async def now(self, ctx: commands.Context):
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

    @commands.command(
        name="history", aliases=["h"], help="display history of songs played"
    )
    @commands.before_invoke(validate_commands)
    async def history(self, ctx: commands.Context):
        mp = self.get_mp(ctx)
        if mp and mp.history:
            q_history = queue_message(list(mp.history)[:10])
            await ctx.send(q_history)

    @commands.command(
        name="jump", aliases=["j"], help="jumps to a specific position in queue"
    )
    @commands.before_invoke(validate_commands)
    async def jump(self, ctx: commands.Context):
        await ctx.send("currently in development")

    @commands.command(
        name="queue", aliases=["q"], help="displays current songs in queue"
    )
    @commands.before_invoke(validate_commands)
    async def queue(self, ctx: commands.Context):
        mp = self.get_mp(ctx)
        await ctx.send(embed=mp.get_queue())

    @commands.command(
        name="volume",
        aliases=["v", "vol", "sound"],
        help="volume level between 0 and 100",
    )
    @commands.before_invoke(validate_commands)
    async def volume(self, ctx: commands.Context, volume):
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

    @commands.command(
        name="ping", aliases=["latency", "l", "delay"], help="latency in milliseconds"
    )
    @commands.before_invoke(validate_commands)
    async def ping(self, ctx: commands.Context):
        ms = self.bot.latency * 1000
        embed = discord.Embed(
            title="Ping - latency in ms",
            description=f"Ping: **{round(ms)}** milliseconds!",
        )
        embed.color = _latency_color(ms)
        await ctx.send(embed=embed)

    # ── Restart recovery listeners ────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self):
        """Fires on cold start or session loss (NOT on WebSocket resume).
        Spawns a recovery task per guild so we don't block the event loop."""
        if self.redis is None:
            return
        for guild in self.bot.guilds:
            asyncio.create_task(self._restore_guild(guild))

    async def _restore_guild(self, guild: discord.Guild) -> None:
        """Attempt to rejoin voice and restore queue for one guild after restart."""
        if self.redis is None:
            return
        if guild.id in self.mps:
            return

        store = GuildRedisStore(self.redis, guild.id)

        with _tracer.start_as_current_span(
            "guild.restore",
            attributes={"discord.guild_id": str(guild.id)},
        ) as span:
            # Distributed lock prevents two bot instances from racing on the same guild.
            # Acquired inside the span so the SET NX EX Redis call is a child span.
            if not await store.acquire_recovery_lock():
                span.set_attribute("restore.skipped_lock", True)
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
                if not isinstance(
                    voice_channel, discord.VoiceChannel
                ) or not isinstance(text_channel, discord.TextChannel):
                    return

                # Check there is something to restore before connecting.
                queue_items = await store.get_queue()
                state = await store.get_state()
                has_crashed_song = bool(state.get(b"current_song_url", b""))
                if not queue_items and not has_crashed_song:
                    return

                span.set_attribute("restore.queue_count", len(queue_items))
                span.set_attribute("restore.crashed_song", has_crashed_song)

                try:
                    await voice_channel.connect(timeout=30.0, reconnect=True)
                    await guild.change_voice_state(
                        channel=voice_channel, self_mute=False, self_deaf=True
                    )
                except Exception as e:
                    span.set_attribute("restore.voice_connect_failed", True)
                    log.warning(f"Could not rejoin voice for guild {guild.id}: {e}")
                    return

                mp = MusicPlayer(self.bot, guild, text_channel, self, redis=self.redis)
                mp.start()
                self.mps[guild.id] = mp

                log.info(
                    f"Restored guild {guild.id} in #{text_channel.name} / {voice_channel.name}"
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                log.error(
                    f"_restore_guild failed for guild {guild.id}: {e}", exc_info=True
                )
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
