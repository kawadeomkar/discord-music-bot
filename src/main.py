import os

import discord
from discord.ext import commands

from src.redis_client import close_redis_pool, create_redis_pool, get_redis
from src.util import get_logger

log = get_logger(__name__)

intents = discord.Intents.all()
intents.message_content = True
EXTENSIONS = ("src.musicbot",)


# Issue #5: AutoShardedBot handles multi-shard within a single process.
# Discord requires sharding at 2500 guilds; plan migration at ~1500.
# shard_count=None lets Discord auto-assign the correct number.
#
# setup_hook is a method override on the Bot subclass, NOT a @bot.event dispatcher.
# In discord.py 2.x, setup_hook is invoked by the library before the bot connects.
class MusicBotApp(commands.AutoShardedBot):
    def __init__(self):
        super().__init__(
            command_prefix="-",
            intents=intents,
            description="music bot",
            strip_after_prefix=True,
        )
        self._redis_pool = None
        self.redis = None

    async def setup_hook(self) -> None:
        self._redis_pool = create_redis_pool()
        self.redis = get_redis(self._redis_pool)
        for extension in EXTENSIONS:
            await self.load_extension(extension)

    async def on_ready(self):
        activity = discord.Game(name="music", type=3)
        await self.change_presence(status=discord.Status.online, activity=activity)
        if self.user:
            log.info(f"Bot: {self.user.name} # {self.user.id}")
        log.info(f"Bot cogs: {list(self.cogs.keys())}")
        log.info(f"Bot guilds: {len(self.guilds)} | latency: {self.latency:.2f}s")
        log.info(f"Bot commands: {self.intents.voice_states}")

    async def close(self) -> None:
        if self._redis_pool is not None:
            await close_redis_pool(self._redis_pool)
        await super().close()


bot = MusicBotApp()


def main():
    token = os.getenv("DISCORD_TOKEN")
    assert token is not None
    assert os.getenv("SPOTIFY_CLIENT_ID") is not None
    assert os.getenv("SPOTIFY_CLIENT_SECRET") is not None
    bot.run(token)


if __name__ == "__main__":
    main()
