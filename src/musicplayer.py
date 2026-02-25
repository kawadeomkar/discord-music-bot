import asyncio
import functools
import random
import time
from collections import deque
from typing import List, Optional, Union

import async_timeout
import discord
from discord.ext import commands

from src.sources import YTSource
from src.util import queue_message, get_logger

log = get_logger(__name__)
from src.youtube import YTDL, QueueObject


class MusicPlayer:
    __slots__ = (
        "bot",
        "_ctx",
        "_guild",
        "_channel",
        "_cog",
        "current_song",
        "play_next",
        "queue",
        "mutex",
        "play_message",
        "history",
        "song_queue",
        # "volume",
        "_player",
    )

    def __init__(self, bot: commands.Bot, ctx: commands.Context):
        self.bot = bot

        self._ctx = ctx
        self._guild: discord.Guild = ctx.guild
        self._channel = ctx.channel
        self._cog = ctx.cog

        self.current_song: YTDL = None
        self.play_next: asyncio.Event = asyncio.Event()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.mutex: asyncio.Lock = asyncio.Lock()

        self.play_message: discord.Embed = None
        self.history: List[str] = []
        self.song_queue: deque = deque()
        # self.volume = 0.5

        self._player = bot.loop.create_task(self.loop())

    def __del__(self):
        log.info("cancelling player task")
        try:
            self._player.cancel()
        except Exception as e:
            log.error(f"error cancelling player task: {e}")
        return

    def get_queue(self) -> str:
        return queue_message(list(self.song_queue)[:10])

    async def stop(self):
        await self._cog.cleanup(self._guild)

    async def queue_put(self, obj: Union[QueueObject, YTSource, List[YTSource]]):
        async with self.mutex:
            if isinstance(obj, list):
                for o in obj:
                    await self.queue.put(o)
            else:
                await self.queue.put(obj)

    async def queue_get(self) -> Union[QueueObject, YTSource]:
        while self.mutex.locked():
            # now this is real hacky but cannot use mutex lock here due to race condition
            time.sleep(0.25)
        return await self.queue.get()

    async def queue_clear(self) -> None:
        async with self.mutex:
            for _ in range(self.queue.qsize()):
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()  # eventually will use join
                except asyncio.QueueEmpty:
                    break
            self.song_queue.clear()

    async def queue_shuffle(self) -> str:
        shuffled = []
        squeue = []

        if self.queue.qsize() < 4:
            return "There must be at least 3 songs to shuffle the queue"

        async with self.mutex:
            for _ in range(self.queue.qsize()):
                try:
                    song = self.queue.get_nowait()
                    self.queue.task_done()
                    shuffled.append(song)
                except asyncio.QueueEmpty:
                    break
            random.shuffle(shuffled)
            for song in shuffled:
                try:
                    self.queue.put_nowait(song)
                    squeue.append(f"{song.title} - [{song.webpage_url}]")
                except asyncio.QueueFull:
                    break
            self.song_queue = deque(squeue)
        return "Shuffled!"

    def _build_now_playing_embed(self, song: YTDL) -> discord.Embed:
        return (
            discord.Embed(
                title=f"**Now playing:** {song.title}",
                description=f"Requester: [{song.requester.mention}]",
                color=discord.Color.green(),
            )
            .add_field(name="Youtube link", value=song.webpage_url, inline=False)
            .add_field(name="Duration", value=song.duration)
            .add_field(name="Channel", value=song.uploader)
            .add_field(name="Views", value=str(song.views))
            .add_field(name="Likes", value=str(song.likes))
            .add_field(name="Dislikes", value=str(song.dislikes))
            .set_thumbnail(url=song.thumbnail)
            .set_footer(
                text=f"Avg Bitrate: {song.abr} | Avg Sampling: {song.asr} | Acodec: {song.acodec}"
            )
        )

    async def update_activity(self):
        # TODO
        # stream_activity = discord.Streaming()
        pass

    async def _resolve_source(
        self, source: Union[QueueObject, YTSource]
    ) -> QueueObject:
        if isinstance(source, YTSource):
            return await YTDL.yt_source(
                self._ctx, source.ytsearch, source.process, loop=self.bot.loop
            )
        return source

    async def _stream_source(self, source: QueueObject) -> Optional[YTDL]:
        try:
            return await YTDL.yt_stream(source, self._ctx, loop=self.bot.loop)
        except Exception as e:
            log.error(f"Error processing song: {e}")
            return None

    async def _send_now_playing(self, song: YTDL) -> None:
        try:
            embed = self._build_now_playing_embed(song)
            self.play_message = embed
            await self._ctx.send(embed=embed)
        except Exception as e:
            log.error(f"embed error: {e}")

    async def loop(self):
        await self.bot.wait_until_ready()

        while not self.bot.is_closed():
            self.play_next.clear()

            try:
                async with async_timeout.timeout(300):
                    source = await self.queue_get()
                    source = await self._resolve_source(source)
            except asyncio.TimeoutError:
                log.warning("Queue timed out, disconnecting")
                asyncio.create_task(self.stop())
                return

            self.current_song = await self._stream_source(source)
            if self.current_song is None:
                continue

            await self._send_now_playing(self.current_song)

            self.song_queue.popleft()
            self._guild.voice_client.play(
                self.current_song,
                after=lambda _: self.bot.loop.call_soon_threadsafe(self.play_next.set),
            )

            await asyncio.sleep(10.0)
            await self.play_next.wait()
            self.history.append(
                f"{self.current_song.title} - {self.current_song.webpage_url}"
            )
            self.queue.task_done()
            self.current_song = None
