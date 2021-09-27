from discord.ext import commands

import asyncio
import async_timeout


class MusicPlayer:
    __slots__ = ('bot', '_guild', '_channel', '_cog', 'current_song', 'next_song', 'songs',
                 'play_message', 'volume')

    def __init__(self, bot: commands.Bot, ctx: commands.Context):
        self.bot = bot

        self._ctx = ctx
        self._guild = ctx.guild
        self._channel = ctx.channel
        self._cog = ctx.cog

        self.current_song = None
        self.next_song = asyncio.Event()
        self.songs = asyncio.Queue()

        self.play_message = None
        self._volume = 0.5

        self._player = bot.loop.create_task(self.loop())

    async def stop(self):
        self.songs.clear()
        self._cog.cleanup(self._guild)

    async def loop(self):
        await self.bot.wait_until_ready()

        while not self.bot.is_closed():
            self.next.clear()
            try:
                async with async_timeout.timeout(300):
                    self.current_song = self.songs.get()
            except asyncio.TimeoutError as e:
                self.bot.loop.create_task(self.stop())

            self.current_song.volume = self.volume
