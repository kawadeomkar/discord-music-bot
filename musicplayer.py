from discord.ext import commands
from youtube import YTDL

import asyncio
import async_timeout
import discord


class MusicPlayer:
    __slots__ = ('bot', '_ctx', '_guild', '_channel', '_cog', 'current_song', 'next_song', 'queue',
                 'play_message', 'volume', '_player')

    def __init__(self, bot: commands.Bot, ctx: commands.Context):
        self.bot = bot

        self._ctx = ctx
        self._guild = ctx.guild
        self._channel = ctx.channel
        self._cog = ctx.cog

        self.current_song: YTDL = None
        self.next_song = asyncio.Event()
        self.queue = asyncio.Queue()

        self.play_message = None
        self.volume = 0.5

        self._player = bot.loop.create_task(self.loop())

    def __del__(self):
        self._player.cancel()

    async def stop(self):
        await self._cog.cleanup(self._guild)

    async def loop(self):
        await self.bot.wait_until_ready()

        while not self.bot.is_closed():
            self.next_song.clear()
            try:
                async with async_timeout.timeout(300):
                    source = await self.queue.get()
            except asyncio.TimeoutError as e:
                # TOOD: send message to leave
                print("timed out")
                self.bot.loop.create_task(self.stop())

            print(source)
            # TODO: Exception handle on error processing
            self.current_song = await YTDL.yt_stream(source, self._ctx, loop=self.bot.loop)
            self.current_song.volume = self.volume

            embed = discord.Embed(title="**Now playing**",
                                  description=f"{self.current_song.title} - {self.current_song.webpage_url} "
                                              f"[{self.current_song.requester.mention}]",
                                  color=discord.Color.green())
            self.play_message = await self._ctx.send(embed=embed)

            self._guild.voice_client.play(self.current_song,
                                          after=lambda _: self.bot.loop.call_soon_threadsafe(
                                              self.next_song.set))
            await self.next_song.wait()
            self.current_song.cleanup()
            self.current_song = None
