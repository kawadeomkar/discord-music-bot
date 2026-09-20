"""Tests for `-clear` (src/commands/clear.py)."""

from unittest.mock import AsyncMock, MagicMock


from src.musicbot import MusicBot
from tests.helpers import command_callback


class TestClearCommand:
    async def test_sends_empty_message_when_queue_already_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=[])
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        mp.queue_clear.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "The queue is already empty."
        )

    async def test_an_unrestored_queue_is_never_cleared(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """clear() destroys the Redis mirror while reading the IN-MEMORY display,
        so against an unrestored player it deletes a saved queue it cannot see —
        and a -playnow stack loses its rows too, because _flush_played records
        from that same empty display. validate_commands only requires the AUTHOR
        in voice, so a cold player is reachable."""
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=[])
        mp.wait_for_restore = AsyncMock(return_value=False)  # snapshot not read
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.clear)(music_bot, mock_ctx)

        mp.queue_clear.assert_not_awaited()
        assert "Still loading" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_sends_embed_with_cleared_songs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        cleared = ["Song A - https://yt.com/1", "Song B - https://yt.com/2"]
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=cleared)
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.message.add_reaction = AsyncMock()
        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        mp.queue_clear.assert_awaited_once()
        mock_ctx.message.add_reaction.assert_awaited_once_with("🗑️")
        call_kwargs = mock_ctx.send.call_args[1]
        embed = call_kwargs["embed"]
        assert "2 songs removed" in embed.title
        assert "Song A" in embed.description
