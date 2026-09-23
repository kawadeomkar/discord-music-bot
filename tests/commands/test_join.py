"""Tests for `-join` (src/commands/join.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
from redis.asyncio import Redis

from src.musicbot import MusicBot
from tests.helpers import (
    command_callback,
)


class TestJoinChannelPersistence:
    async def test_join_writes_channel_ids_to_redis(
        self,
        music_bot_with_redis: MusicBot,
        mock_ctx: MagicMock,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:
        """Calling join should persist voice and text channel IDs to Redis."""
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 777000000000000001
        voice_channel.connect = AsyncMock()
        mock_ctx.author.voice.channel = voice_channel
        mock_guild.change_voice_state = AsyncMock()
        mock_guild.voice_client = None

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.id = 777000000000000002
        mock_ctx.channel = text_channel

        mp = MagicMock()
        mp.store = MagicMock()
        mp.store.set_connection = AsyncMock()
        music_bot_with_redis.mps[mock_guild.id] = mp

        # join is a @commands.command — call the underlying callback directly.
        mock_ctx.voice_client = None  # bot not yet in channel
        with (
            patch.object(discord.VoiceChannel, "connect", new=AsyncMock()),
            patch.object(mock_ctx, "invoke", new=AsyncMock()),
        ):
            music_bot_with_redis.get_mp = MagicMock(return_value=mp)
            await command_callback(MusicBot.join)(music_bot_with_redis, mock_ctx)

        mp.store.set_connection.assert_awaited_once_with(
            voice_channel.id, text_channel.id
        )
        # Voice is up — a queue persisted by a previous -stop resumes.
        mp.open_playback_gate.assert_called_once()
