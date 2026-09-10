"""Tests for `-shuffle` (src/commands/shuffle.py)."""

from unittest.mock import AsyncMock, MagicMock


from src.musicbot import MusicBot
from tests.helpers import command_callback, no_typing


class TestShuffleWaitsForTheRestore:
    """-shuffle waits for the restore like every other queue-mutating command:
    shuffle() REBUILDS the mirror from memory, so running it before
    restore_entries() has replayed the saved queue writes an unrestored deque over
    it and deletes the persisted entries outright."""

    async def test_it_refuses_until_the_restore_lands(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=False)
        mp.queue_shuffle = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        with no_typing("src.commands.shuffle.background_typing"):
            await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_not_awaited()
        assert "Still loading" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_it_shuffles_once_the_restore_has_landed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_shuffle = AsyncMock(return_value="Shuffled!")
        music_bot.get_mp = MagicMock(return_value=mp)

        with no_typing("src.commands.shuffle.background_typing"):
            await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_awaited_once()
