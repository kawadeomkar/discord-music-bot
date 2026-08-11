"""Tests for src/recovery.py — crash recovery (restore_guild) and the
alone-disconnect watchdog.

These drive the MusicBot cog, which owns the redis handle and the mps registry
restore_guild reads; the split follows test_leaderboard.py, where the new file
owns both the extracted module and the cog surface that reaches it.
"""

from typing import Any, Optional
from collections.abc import Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
import redis.asyncio as aioredis
from redis.asyncio import Redis

from src.musicbot import MusicBot
from src.recovery import restore_guild
from src.redis_client import GuildRedisStore
from tests.helpers import stub_create_task


class TestEagerRestore:
    async def test_restore_guild_skips_if_already_in_mps(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """restore_guild is a no-op if the guild already has a MusicPlayer."""
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()
        # Should not raise or create another player
        await restore_guild(music_bot_with_redis, mock_guild)
        assert len(music_bot_with_redis.mps) == 1

    async def test_restore_guild_skips_when_no_channel_ids(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:
        """restore_guild exits early when no connection was persisted."""
        await restore_guild(music_bot_with_redis, mock_guild)
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_restore_guild_skips_when_queue_empty_and_no_crash(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:
        """No queue items + no crashed song → skip restore even if channel IDs exist."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)
        # No queue items, no current_song_url in state

        await restore_guild(music_bot_with_redis, mock_guild)
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_restore_guild_gates_without_reading_queue_payload(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """NIT-7: a -stop'ped guild keeps its (possibly long) queue list, so the
        recovery gate must never pull the full playback aggregate just to
        conclude "nothing to do" — it reads state + LLEN via get_recovery_gate,
        not get_playback_snapshot."""
        from src.guild_state import SongQueueEntry

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        # Connection cleared (stopped) but a leftover queue survives by design.
        for i in range(3):
            await store.push_queue(
                SongQueueEntry(
                    webpage_url=f"https://yt.com/v={i}", title=f"S{i}", requester_id=i
                )
            )

        snapshot_spy = AsyncMock(wraps=store.get_playback_snapshot)
        with patch.object(GuildRedisStore, "get_playback_snapshot", snapshot_spy):
            await restore_guild(music_bot_with_redis, mock_guild)

        snapshot_spy.assert_not_awaited()
        assert mock_guild.id not in music_bot_with_redis.mps


class TestOnReady:
    async def test_noop_when_redis_is_none(self, music_bot: MusicBot) -> None:
        music_bot.redis = None
        await music_bot.on_ready()  # must not raise, no tasks created

    async def test_creates_restore_task_per_guild(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        guilds = list(music_bot_with_redis.bot.guilds)
        stub = stub_create_task()
        passed_guilds = []

        async def _noop() -> None:
            pass

        # Patched where on_ready LOOKS IT UP — musicbot's module globals — not where
        # it is defined, or the cog keeps calling the real one. Capture happens
        # synchronously in the spy (before stub_create_task closes the coroutine,
        # which would prevent the body running).
        def _spy(cog: MusicBot, guild: MagicMock) -> Coroutine[Any, Any, None]:
            passed_guilds.append(guild)
            return _noop()

        with (
            patch("asyncio.create_task", stub),
            patch("src.musicbot.restore_guild", _spy),
        ):
            await music_bot_with_redis.on_ready()

        # One per guild, plus the one-off config hydration.
        assert stub.call_count == len(guilds) + 1
        assert passed_guilds == guilds


class TestRestoreGuildLock:
    async def test_skips_when_lock_already_held(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(100, 200)
        # Pre-hold the lock so acquire fails
        await fake_redis_bot.set(
            f"lock:guild:{mock_guild.id}:recovery", "1", nx=True, ex=60
        )
        await restore_guild(music_bot_with_redis, mock_guild)
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_restore_creates_player_when_queue_exists(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(100, 200)
        await fake_redis_bot.rpush(
            store.queue_key(),
            orjson.dumps(
                {
                    "webpage_url": "https://yt.com/v=1",
                    "title": "Song",
                    "requester_id": 1,
                    "ts": None,
                }
            ),
        )

        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 100
        voice_channel.connect = AsyncMock()
        voice_channel.name = "general"

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.id = 200
        text_channel.name = "general"

        mock_guild.get_channel = MagicMock(
            side_effect=lambda cid: voice_channel if cid == 100 else text_channel
        )
        mock_guild.change_voice_state = AsyncMock()

        mock_mp = MagicMock()
        mock_mp.start = MagicMock()

        with patch("src.recovery.MusicPlayer", return_value=mock_mp):
            await restore_guild(music_bot_with_redis, mock_guild)

        assert mock_guild.id in music_bot_with_redis.mps


# ── restore_guild: Redis-failure gate ────────────────────────────────────────


class TestRestoreGuildStateReadFailed:
    async def test_recovery_skipped_when_state_read_fails(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """get_recovery_gate() returning None (Redis unavailable) must not read as
        "nothing to restore": recovery skips with a WARNING and no channel or player
        work — distinguishable from the empty-gate case, which skips silently."""

        with patch.object(
            GuildRedisStore, "get_recovery_gate", new=AsyncMock(return_value=None)
        ):
            with caplog.at_level("WARNING", logger="src.recovery"):
                await restore_guild(music_bot_with_redis, mock_guild)

        assert "state read failed" in caplog.text
        mock_guild.get_channel.assert_not_called()
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_recovery_skipped_silently_when_nothing_stored(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Empty state hash (zero-value snapshot, no connection) skips recovery
        without the failure warning."""
        with caplog.at_level("WARNING", logger="src.recovery"):
            await restore_guild(music_bot_with_redis, mock_guild)

        assert "state read failed" not in caplog.text
        mock_guild.get_channel.assert_not_called()
        assert mock_guild.id not in music_bot_with_redis.mps


# ── restore_guild Gap 3: channel-deleted notification ────────────────────────


class TestRestoreGuildChannelDeleted:
    async def test_clears_connection_when_both_channels_deleted(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """When both stored channels are gone, Redis state is cleared so the
        guild is not retried on the next on_ready."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None  # both resolved to None
        mock_guild.system_channel.send = AsyncMock()
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await restore_guild(music_bot_with_redis, mock_guild)

        state = await store.get_guild_state()
        assert state is not None
        assert state.voice_channel_id is None
        assert state.text_channel_id is None
        assert not state.has_active_connection

    async def test_sends_notification_via_system_channel(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """Notification is sent via system_channel when both stored channels are deleted."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.send = AsyncMock()
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await restore_guild(music_bot_with_redis, mock_guild)

        mock_guild.system_channel.send.assert_awaited_once()
        msg = mock_guild.system_channel.send.call_args.kwargs["embed"].description
        assert "⚠️" in msg
        assert "voice channel" in msg
        assert "text channel" in msg
        assert "were deleted" in msg

    async def test_the_notification_is_decorated_in_debug_mode(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """No player exists on this path, so the cog decorates directly."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)
        music_bot_with_redis._debug_overrides[mock_guild.id] = True

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.send = AsyncMock()
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await restore_guild(music_bot_with_redis, mock_guild)

        embed = mock_guild.system_channel.send.call_args.kwargs["embed"]
        assert "🐞" in (embed.footer.text or "")

    async def test_falls_back_to_text_channels_when_system_channel_no_perms(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """When system_channel denies send_messages, falls back to guild.text_channels."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=False
        )

        fallback = MagicMock(spec=discord.TextChannel)
        fallback.send = AsyncMock()
        fallback.permissions_for = MagicMock(
            return_value=discord.Permissions(send_messages=True)
        )
        mock_guild.text_channels = [fallback]

        await restore_guild(music_bot_with_redis, mock_guild)

        fallback.send.assert_awaited_once()
        mock_guild.system_channel.send.assert_not_called()

    async def test_notifies_via_text_channel_when_only_voice_deleted(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """When only the voice channel is gone, notify via the still-valid text channel."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()

        def _get_channel(ch_id: int) -> Optional[MagicMock]:
            if ch_id == 888000000000000001:
                return None  # voice deleted
            return text_channel  # text still exists

        mock_guild.get_channel.side_effect = _get_channel

        await restore_guild(music_bot_with_redis, mock_guild)

        text_channel.send.assert_awaited_once()
        msg = text_channel.send.call_args.kwargs["embed"].description
        assert "voice channel" in msg
        assert "was deleted" in msg

    async def test_swallows_notify_send_failure(
        self,
        music_bot_with_redis: MusicBot,
        mock_guild: MagicMock,
        fake_redis_bot: aioredis.Redis,
    ) -> None:
        """A failure sending the notification must not propagate out of restore_guild."""

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.send = AsyncMock(
            side_effect=Exception("channel gone")
        )
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await restore_guild(music_bot_with_redis, mock_guild)  # must not raise
