"""Tests for `-volume` (src/commands/volume.py)."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock


from src.musicbot import MusicBot
from tests.helpers import (
    command_callback,
)


class TestVolumeCommand:
    @staticmethod
    def _description(ctx: MagicMock) -> str:
        return cast(str, ctx.send.await_args.kwargs["embed"].description)

    async def test_sets_player_volume(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.store.set_volume = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        mp.store.set_volume.assert_awaited_once_with(0.5)
        mock_ctx.send.assert_awaited()

    async def test_volume_persists_nothing_without_store(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.store = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        mock_ctx.send.assert_awaited()
        assert "could not be saved" in self._description(mock_ctx)

    async def test_a_successful_write_says_it_is_saved(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.store.set_volume = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        description = self._description(mock_ctx)
        assert "saved for this server" in description
        assert "could not be saved" not in description

    async def test_a_failed_write_is_reported_not_claimed(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """set_volume returns False when the write did not land, and the help
        promises the level survives a restart. Confirming it anyway is the exact
        failure the debug toggle fixed: a setting that quietly reverts."""
        mp = MagicMock()
        mp.store.set_volume = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        description = self._description(mock_ctx)
        assert "could not be saved" in description
        # Still applied to this process's player, as the debug toggle is.
        assert mp.volume == 0.5

    async def test_rejects_non_numeric_string(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "loud")
        mock_ctx.send.assert_awaited()

    async def test_rejects_out_of_range(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "150")
        mock_ctx.send.assert_awaited()
