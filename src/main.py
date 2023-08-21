from discord.ext import commands
from musicplayer import MusicPlayer
from sources import spotify_playlist_to_ytsearch, parse_url, URLSource, \
    SoundcloudSource, SpotifySource, YTSource
from spotify import Spotify
from typing import List, Union
from util import queue_message, send_queue_phrases

# music players
from youtube import QueueObject, YTDL

import asyncio
import discord
import os
import random


intents = discord.Intents.all()
intents.message_content = True
EXTENSIONS = ("musicbot",)


bot = commands.Bot(
    command_prefix='-',
    intents=discord.Intents().all(),  # TODO: narrow down
    description='omkars bad music bot lol',
    strip_after_prefix=True)

@bot.event
async def setup_hook() -> None:
    for extension in EXTENSIONS:
        await bot.load_extension(extension)

@bot.event
async def on_ready():
    activity = discord.Game(name="music", type=3)
    await bot.change_presence(status=discord.Status.online, activity=activity)
    print(f'Bot :{bot.user.name} # {bot.user.id}')
    print(f'Bot cogs: {bot.cogs}')
    print(f'Bot state: {bot._get_state()}')
    print(f'Bot commands: {bot.intents.voice_states}')


if __name__ == '__main__':
    assert os.getenv("DISCORD_TOKEN") is not None
    assert os.getenv("SPOTIFY_CLIENT_ID") is not None
    assert os.getenv("SPOTIFY_CLIENT_SECRET") is not None
    import inspect
    print(inspect.iscoroutinefunction(bot.add_cog))

    bot.run(os.getenv("DISCORD_TOKEN"))