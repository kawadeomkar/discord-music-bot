"""Tests for `-pause` (src/commands/pause.py)."""

from unittest.mock import AsyncMock, MagicMock

import discord

from src.commands._common import NOTHING_PLAYING
from src.musicbot import MusicBot
from tests.helpers import (
    command_callback,
)


class TestPauseCommand:
    @staticmethod
    def _playing(mock_ctx: MagicMock) -> MagicMock:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        mp.repin_now_playing = AsyncMock(return_value=True)
        return mp

    async def test_pauses_naming_the_member_who_asked(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The paused card credits the pause, so the author has to reach pause()."""
        mp = self._playing(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_awaited_once_with(mock_ctx.voice_client, by=mock_ctx.author)

    async def test_repins_the_block_rather_than_posting_its_own_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The paused card rides in the block now. A ctx.send with no embed of its
        own would host the block on a response that strip-edits back to blank."""
        mp = self._playing(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()
        mock_ctx.send.assert_not_awaited()

    async def test_reacts_to_the_invoking_message(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._playing(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏸️")

    async def test_notice_when_nothing_is_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=False)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed.description == NOTHING_PLAYING

    async def test_notice_when_not_in_voice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed.description == NOTHING_PLAYING

    async def test_notice_when_already_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        # Not "no song is playing": a paused song is loaded and resumable.
        mp.pause.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Already paused" in embed.description
