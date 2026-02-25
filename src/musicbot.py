import asyncio
import os
import random
from typing import List, Union

import discord
from discord.ext import commands

from src.musicplayer import MusicPlayer
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    URLSource,
    YTSource,
    parse_url,
    spotify_playlist_to_ytsearch,
)
from src.spotify import Spotify
from src.util import queue_message, send_queue_phrases, get_logger

log = get_logger(__name__)

# music players
from src.youtube import YTDL, QueueObject


class MusicBot(commands.Cog):
    """
    class for music bot
    """

    __slots__ = ("bot", "mps")

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.mps = {}
        self.spotify = Spotify()

    def get_mp(self, ctx: commands.Context) -> MusicPlayer:
        if ctx.guild.id in self.mps:
            return self.mps[ctx.guild.id]
        self.mps[ctx.guild.id] = MusicPlayer(self.bot, ctx)
        return self.mps[ctx.guild.id]

    async def cleanup(self, guild: discord.Guild) -> None:
        log.info("going to cleanup/disconnect")
        if guild.voice_client:
            await guild.voice_client.disconnect()
        if guild.id in self.mps:
            try:
                del self.mps[guild.id]
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.error(f"cleanup error: {type(e).__name__}: {e}")

    async def cog_before_invoke(self, ctx: commands.Context):
        self.get_mp(ctx)

    async def validate_commands(self, ctx: commands.Context) -> None:
        if isinstance(ctx.author, discord.User):
            await ctx.send(f"You must be a member of this channel {ctx.author}")
            raise commands.CommandError(
                f"User {ctx.author} must be a member of this channel."
            )

        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send(
                f"You are not connected to a voice channel, you silly baka {ctx.author}"
            )
            raise commands.CommandError(
                f"User {ctx.author} is not connected to a voice channel."
            )

        if (
            not ctx.command == "play"
            and ctx.voice_client
            and ctx.voice_client.channel != ctx.author.voice.channel
        ):
            await ctx.send(
                f"Bot is already being used in channel {ctx.voice_client.channel}"
            )
            raise commands.CommandError("Bot is already in a voice channel.")

    async def queue_source(
        self,
        ctx: commands.Context,
        loop: asyncio.BaseEventLoop,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> Union[QueueObject, List[str]]:
        if source.stype == URLSource.SPOTIFY and source.type == "playlist":
            return await self.spotify.playlist(source.id)
        else:
            ts = None
            if source.stype == URLSource.SPOTIFY:
                search = await self.spotify.track(source.id)
            elif source.stype == URLSource.YOUTUBE:
                if source.ytsearch:
                    search = source.ytsearch
                elif source.url:
                    search = source.url
                ts = source.ts
            elif source.stype == URLSource.SOUNDCLOUD:
                search = source.url
            return await YTDL.yt_source(ctx, search, source.process, loop=loop, ts=ts)

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

        # only support youtube link for now
        async with ctx.typing():
            try:
                source = parse_url(url, ctx.message.content)
                qobj: Union[QueueObject, List[str]] = await self.queue_source(
                    ctx, self.bot.loop, source
                )

                if (
                    isinstance(qobj, list)
                    and source.stype == URLSource.SPOTIFY
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
                    for title in qobj:
                        mp.song_queue.append(title)

                else:
                    if mp.queue.qsize() > 0 or (
                        voice_client and voice_client.is_playing()
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
                    mp.song_queue.append(f"{qobj.title} - {qobj.webpage_url}")
                    log.info(f"play qsize: {mp.queue.qsize()}")

                await ctx.message.add_reaction("👍")
                await send_queue_phrases(ctx)
            except Exception as e:
                await ctx.send(f"Exception caught: {e}")

    @commands.command(name="skip", aliases=["sk"], help="skips current song")
    @commands.before_invoke(validate_commands)
    async def skip(self, ctx: commands.Context):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice_client and voice_client.is_playing():
            ctx.message.guild.voice_client.stop()
            if not ctx.invoked_parents:
                await ctx.message.add_reaction("⏭")

    @commands.command(name="stop", aliases=["st"], help="stops current song")
    @commands.before_invoke(validate_commands)
    async def stop(self, ctx: commands.Context):
        await ctx.invoke(self.skip)
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice_client:
            await ctx.message.add_reaction("👋")
            await self.cleanup(ctx.guild)

    @commands.command(name="pause", aliases=["po"], help="pause the current song")
    @commands.before_invoke(validate_commands)
    async def pause(self, ctx: commands.Context):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if voice_client and voice_client.is_playing():
            await voice_client.pause()
            await ctx.message.add_reaction("⏸️")

    @commands.command(name="resume", aliases=["r"], help="resume the current song")
    @commands.before_invoke(validate_commands)
    async def pause(self, ctx: commands.Context):
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if voice_client and not voice_client.is_playing() and voice_client.is_paused():
            await voice_client.resume()
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
        channel = ctx.message.author.voice.channel
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if not voice_client:
            await channel.connect(timeout=10.0)
        if not ctx.author.voice.channel == ctx.voice_client.channel:
            await ctx.voice_client.move_to(channel)
        await ctx.guild.change_voice_state(
            channel=channel, self_mute=False, self_deaf=True
        )
        await ctx.message.add_reaction("👋")
        await ctx.invoke(self.ping)

    @commands.command(
        name="clear", aliases=["c"], help="clears the queue, in development"
    )
    @commands.before_invoke(validate_commands)
    async def clear(self, ctx: commands.Context):
        await ctx.send(f"in development, use -stop and -join to clear")

    @commands.command(
        name="now", aliases=["np", "rn", "nowplaying"], help="display current song"
    )
    @commands.before_invoke(validate_commands)
    async def now(self, ctx: commands.Context):
        mp = self.get_mp(ctx)
        if ctx.message.guild.voice_client.is_playing() and mp.play_message:
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
            q_history = queue_message(mp.history[:10])
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
        if mp and len(mp.song_queue) > 0:
            q_songs = mp.get_queue()
            await ctx.send(q_songs)

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
            except ValueError as e:
                await ctx.send("Volume must be a number between 0 and 100")
                return
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice_client and voice_client.is_playing():
            if not 0 <= volume <= 100:
                return await ctx.send("Volume must be between 0 and 100")
            volume = volume / 100
            voice_client.source.volume = volume
            ctx.send(f"Set volume of music player to {volume}")
        else:
            await ctx.send("No songs are currently playing.")

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
        if ms <= 50:
            embed.color = 0x44FF44
        elif ms <= 100:
            embed.color = 0xFFD000
        elif ms <= 200:
            embed.color = 0xFF6600
        else:
            embed.color = 0x990000
        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(MusicBot(bot))
