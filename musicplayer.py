from discord.ext import commands
from youtube import YTDL

import asyncio
import async_timeout
import discord


class MusicPlayer:
    __slots__ = ('bot', '_ctx', '_guild', '_channel', '_cog', 'current_song', 'play_next', 'queue',
                 'play_message', 'volume', '_player')

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

        self.play_message = None
        self.volume = 0.5

        self._player = bot.loop.create_task(self.loop())

    def __del__(self):
        self._player.cancel()

    async def stop(self):
        print("stopping")
        await self._cog.cleanup(self._guild)

    async def update_activity(self):
        stream_activity = discord.Streaming()
        pass

    async def loop(self):
        await self.bot.wait_until_ready()

        while not self.bot.is_closed():
            self.play_next.clear()
            print(f"q size: {str(self.queue.qsize())}")
            try:
                async with async_timeout.timeout(300):
                    source = await self.queue.get()
            except asyncio.TimeoutError as e:
                # TOOD: send message to leave
                print("timed out")
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

            self._guild.voice_client.play(self.current_song,
                                          after=lambda _: self.bot.loop.call_soon_threadsafe(
                                              self.play_next.set))
            await self.play_next.wait()
            self.queue.task_done()
            self.current_song.cleanup()
            self.current_song = None
