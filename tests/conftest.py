"""Shared fixtures for the discord-music-bot test suite."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest



@pytest.fixture
def mock_guild():
    guild = MagicMock(spec=discord.Guild)
    guild.id = 111111111111111111
    guild.voice_client = MagicMock(spec=discord.VoiceClient)
    guild.voice_client.is_playing.return_value = False
    guild.voice_client.is_paused.return_value = False
    return guild


@pytest.fixture
def mock_author():
    member = MagicMock(spec=discord.Member)
    member.id = 222222222222222222
    member.name = "testuser"
    member.mention = "<@222222222222222222>"
    member.voice = MagicMock()
    member.voice.channel = MagicMock(spec=discord.VoiceChannel)
    return member


@pytest.fixture
def mock_channel():
    channel = MagicMock(spec=discord.TextChannel)
    channel.send = AsyncMock()
    return channel


@pytest.fixture
def mock_message(mock_author, mock_channel):
    message = MagicMock(spec=discord.Message)
    message.author = mock_author
    message.channel = mock_channel
    message.content = "-play test song"
    message.add_reaction = AsyncMock()
    return message


@pytest.fixture
def mock_ctx(mock_guild, mock_author, mock_channel, mock_message):
    ctx = MagicMock()
    ctx.guild = mock_guild
    ctx.author = mock_author
    ctx.channel = mock_channel
    ctx.message = mock_message
    ctx.cog = MagicMock()
    ctx.send = AsyncMock()
    ctx.typing = MagicMock()
    ctx.typing.return_value.__aenter__ = AsyncMock(return_value=None)
    ctx.typing.return_value.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.fixture
def mock_bot(mock_guild):
    bot = MagicMock()
    bot.guilds = [mock_guild]
    bot.latency = 0.05
    bot.is_closed.return_value = False
    bot.wait_until_ready = AsyncMock()
    # Prevent the background loop task from actually running
    bot.loop.create_task = MagicMock(
        side_effect=lambda coro: coro.close() or MagicMock()
    )
    return bot
