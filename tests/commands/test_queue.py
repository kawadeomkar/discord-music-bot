"""Tests for `-queue` (src/commands/queue.py)."""

from unittest.mock import MagicMock

import discord

from src.musicbot import MusicBot
from tests.helpers import command_callback


class TestQueueCommand:
    async def test_always_sends_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        embed = discord.Embed(
            title="Queue", description="Songs: **0**\n\n*The queue is empty.*"
        )
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=embed)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.queue)(music_bot, mock_ctx)

        mock_ctx.send.assert_awaited_once()
        call_kwargs = mock_ctx.send.call_args[1]
        assert "embed" in call_kwargs
        assert call_kwargs["embed"] is embed

    async def test_sends_embed_when_queue_is_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        embed = discord.Embed(
            title="Queue", description="Songs: **0**\n\n*The queue is empty.*"
        )
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=embed)
        mp.song_queue = []
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.queue)(music_bot, mock_ctx)

        mock_ctx.send.assert_awaited_once()

    async def test_delegates_to_mp_get_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.queue)(music_bot, mock_ctx)

        mp.queue_embed.assert_called_once()
