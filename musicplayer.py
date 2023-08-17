import random
import time
from collections import deque
from discord.ext import commands
from youtube import QueueObject, YTDL
from sources import YTSource
from typing import List, Union
from util import queue_message

import asyncio
import async_timeout
import functools
import discord


class MusicPlayer:
    __slots__ = ('bot', '_ctx', '_guild', '_channel', '_cog', 'current_song', 'play_next', 'queue',
                 'mutex', 'play_message', 'history', 'song_queue', 'volume', '_player')

    def __init__(self, bot: commands.Bot, ctx: commands.Context):
        self.bot = bot

        self._ctx = ctx
        self._guild = ctx.guild
        self._channel = ctx.channel
        self._cog = ctx.cog

        self.current_song: YTDL = None
        self.play_next: asyncio.Event = asyncio.Event()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.mutex: asyncio.Lock = asyncio.Lock()

        self.play_message: discord.Embed = None
        self.history: List[str] = []
        self.song_queue: deque = deque()
        self.volume = 0.5

        self._player = bot.loop.create_task(self.loop())

    def __del__(self):
        print("__del__ cancelling task")
        try:
            self._player.cancel()
        except Exception as e:
            print("del caught error")
            print(str(e))
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

    async def update_activity(self):
        # TODO
        # stream_activity = discord.Streaming()
        pass

    async def loop(self):
        await self.bot.wait_until_ready()

        while not self.bot.is_closed():
            self.play_next.clear()
            print(f"q size: {str(self.queue.qsize())}")
            try:
                async with async_timeout.timeout(300):
                    # mutex lock queue get
                    source: Union[QueueObject, YTSource] = await self.queue_get()
                    if isinstance(source, YTSource):
                        source = await YTDL.yt_source(self._ctx, source.ytsearch, source.process,
                                                      loop=self.bot.loop)
            except asyncio.TimeoutError as e:
                # TOOD: send message to leave
                print("timed out " + str(e))
                self.bot.loop.create_task(self.stop())
                return

            print(f"ingested from queue: {source}")
            # TODO: Exception handle on error processing
            self.current_song = await YTDL.yt_stream(source, self._ctx, loop=self.bot.loop)
            self.current_song.volume = self.volume

            try:
                embed = discord.Embed(title=f"**Now playing:** {self.current_song.title}",
                                      description=f"Requester: [{self.current_song.requester.mention}]",
                                      color=discord.Color.green()) \
                    .add_field(name="Youtube link", value=self.current_song.webpage_url,
                               inline=False) \
                    .add_field(name="Duration", value=self.current_song.duration) \
                    .add_field(name="Channel", value=self.current_song.uploader) \
                    .add_field(name="Views", value=str(self.current_song.views)) \
                    .add_field(name="Likes", value=str(self.current_song.likes)) \
                    .add_field(name="Dislikes", value=str(self.current_song.dislikes)) \
                    .set_thumbnail(url=self.current_song.thumbnail) \
                    .set_footer(text=f"Avg Bitrate: {self.current_song.abr} | "
                                     f"Avg Sampling: {self.current_song.asr} | "
                                     f"Acodec: {self.current_song.acodec}")
                self.play_message = embed

            except Exception as e:
                print(f"embed error {str(e)}")
            await self._ctx.send(embed=embed)

            print(f"guild voice client: {self._guild.voice_client}")

            self.song_queue.popleft()
            self._guild.voice_client.play(self.current_song,
                                          after=lambda _: self.bot.loop.call_soon_threadsafe(
                                              self.play_next.set))
            await self.play_next.wait()
            self.history.append(f"{self.current_song.title} - {self.current_song.webpage_url}")
            self.queue.task_done()
            self.current_song.cleanup()
            self.current_song = None
