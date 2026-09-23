"""Tests for `-volume` (src/commands/volume.py)."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from redis.asyncio import Redis

from src.guild_state import GuildConfig
from src.musicbot import MusicBot
from src.redis_client import GuildRedisStore
from tests.helpers import (
    command_callback,
)


class TestVolumeCommand:
    """-volume writes through GuildSettings, the one writer of guild:{id}:config,
    against a real (fake) Redis: a mocked store would let a reply claim a write
    that never happened."""

    @staticmethod
    def _description(ctx: MagicMock) -> str:
        return cast(str, ctx.send.await_args.kwargs["embed"].description)

    @staticmethod
    def _player(cog: MusicBot) -> MagicMock:
        mp = MagicMock()
        mp.volume = 1.0
        cog.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_sets_the_player_and_both_stored_copies(
        self,
        music_bot_with_redis: MusicBot,
        mock_ctx: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:
        mp = self._player(music_bot_with_redis)
        await command_callback(MusicBot.volume)(music_bot_with_redis, mock_ctx, "50")

        assert mp.volume == 0.5
        store = GuildRedisStore(fake_redis_bot, mock_ctx.guild.id)
        assert await store.read_config() == GuildConfig(volume=0.5)
        assert (await fake_redis_bot.hget(store.state_key(), "volume")) == b"0.5"
        assert music_bot_with_redis.guild_settings.peek(mock_ctx.guild.id) == (
            GuildConfig(volume=0.5)
        )

    async def test_a_successful_write_says_it_is_saved(
        self, music_bot_with_redis: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._player(music_bot_with_redis)
        await command_callback(MusicBot.volume)(music_bot_with_redis, mock_ctx, "50")
        description = self._description(mock_ctx)
        assert "saved for this server" in description
        assert "could not be saved" not in description

    async def test_a_failed_write_is_reported_not_claimed(
        self, music_bot_with_redis: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The help promises the level survives a restart. Confirming a write that
        did not land is the exact failure the debug toggle fixed: a setting that
        quietly reverts."""
        mp = self._player(music_bot_with_redis)
        with patch.object(
            GuildRedisStore, "set_volume", new=AsyncMock(return_value=False)
        ):
            await command_callback(MusicBot.volume)(
                music_bot_with_redis, mock_ctx, "50"
            )
        assert "could not be saved" in self._description(mock_ctx)
        # Still applied to this process's player, as the debug toggle is.
        assert mp.volume == 0.5
        assert not music_bot_with_redis.guild_settings.is_persisted(
            mock_ctx.guild.id, "volume"
        )

    async def test_without_redis_it_applies_and_says_so(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._player(music_bot)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        assert "could not be saved" in self._description(mock_ctx)

    async def test_rejects_non_numeric_string(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._player(music_bot)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "loud")
        mock_ctx.send.assert_awaited()
        assert mp.volume == 1.0

    async def test_rejects_out_of_range(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._player(music_bot)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "150")
        mock_ctx.send.assert_awaited()
        assert mp.volume == 1.0
        assert cast(Any, music_bot.guild_settings.peek(mock_ctx.guild.id)) is None
