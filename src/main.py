import os

import discord
from discord.ext import commands

from src.util import get_logger

log = get_logger(__name__)


intents = discord.Intents.all()
intents.message_content = True
EXTENSIONS = ("src.musicbot",)


bot = commands.Bot(
    command_prefix="-",
    intents=discord.Intents().all(),  # TODO: narrow down
    description="music bot",
    strip_after_prefix=True,
)


@bot.event
async def setup_hook() -> None:
    for extension in EXTENSIONS:
        await bot.load_extension(extension)


@bot.event
async def on_ready():
    activity = discord.Game(name="music", type=3)
    await bot.change_presence(status=discord.Status.online, activity=activity)
    log.info(f"Bot :{bot.user.name} # {bot.user.id}")
    log.info(f"Bot cogs: {list(bot.cogs.keys())}")
    log.info(f"Bot guilds: {len(bot.guilds)} | latency: {bot.latency:.2f}s")
    log.info(f"Bot commands: {bot.intents.voice_states}")


def main():
    assert os.getenv("DISCORD_TOKEN") is not None
    assert os.getenv("SPOTIFY_CLIENT_ID") is not None
    assert os.getenv("SPOTIFY_CLIENT_SECRET") is not None

    bot.run(os.getenv("DISCORD_TOKEN"))


if __name__ == "__main__":
    main()
