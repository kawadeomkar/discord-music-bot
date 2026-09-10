"""Tests for `-pause` (src/commands/pause.py)."""

from unittest.mock import AsyncMock, MagicMock

import discord

from src.musicbot import MusicBot
from tests.helpers import (
    command_callback,
)


class TestPauseCommand:
    async def test_pauses_when_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_awaited_once_with(vc)

    async def test_sends_confirmation_embed_when_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        embed = discord.Embed(title="⏸️ Paused")
        mp.build_pause_confirmation_embed = MagicMock(return_value=embed)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once_with(embed=embed)

    async def test_no_confirmation_sent_when_embed_is_none(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        mp.build_pause_confirmation_embed = MagicMock(return_value=None)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mock_ctx.send.assert_not_awaited()

    async def test_noop_when_not_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_not_awaited()
