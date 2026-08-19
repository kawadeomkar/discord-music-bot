"""Tests for src/recovery.py — crash recovery (restore_guild) and the
alone-disconnect watchdog.

These drive the MusicBot cog, which owns the redis handle and the mps registry
restore_guild reads; the split follows test_leaderboard.py, where the new file
owns both the extracted module and the cog surface that reaches it.
"""

import asyncio
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
from tests.helpers import make_mock_task, mocked, stub_create_task


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
        music_bot_with_redis.debug_settings._overrides[mock_guild.id] = True

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


class TestVoiceStateConsistency:
    @staticmethod
    def _wire_bot_user(cog: MusicBot) -> None:
        mock_user = MagicMock()
        mock_user.id = 999999999999999999
        mocked(cog.bot).user = mock_user

    async def test_bot_disconnect_triggers_cleanup(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """on_voice_state_update cleans up when the bot itself leaves a channel."""
        self._wire_bot_user(music_bot_with_redis)

        mp = MagicMock()
        mp.store = None
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        music_bot_with_redis.mps[mock_guild.id] = mp

        member = MagicMock(spec=discord.Member)
        member.id = 999999999999999999
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()  # was in a channel
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None  # now disconnected

        mock_guild.voice_client = None
        with patch.object(
            music_bot_with_redis, "cleanup", new=AsyncMock()
        ) as mock_cleanup:
            await music_bot_with_redis.on_voice_state_update(member, before, after)
        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_bot_moved_cancels_stale_alone_timer(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Bot moved to a new channel (not ejected) cancels any running alone-timer."""
        self._wire_bot_user(music_bot_with_redis)

        timer = make_mock_task()
        music_bot_with_redis.voice_watchdog._timers[mock_guild.id] = timer

        member = MagicMock(spec=discord.Member)
        member.id = 999999999999999999
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()
        after = MagicMock(spec=discord.VoiceState)
        after.channel = MagicMock()  # moved to a new channel, not ejected

        with patch.object(music_bot_with_redis, "cleanup", new=AsyncMock()):
            await music_bot_with_redis.on_voice_state_update(member, before, after)

        timer.cancel.assert_called_once()
        assert mock_guild.id not in music_bot_with_redis.voice_watchdog._timers

    async def test_member_in_inactive_guild_ignored(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Non-bot member event in a guild where the bot has no active player is a noop."""
        self._wire_bot_user(music_bot_with_redis)
        # mps is empty — guild is not active

        member = MagicMock(spec=discord.Member)
        member.id = 123456789
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None

        with patch.object(
            music_bot_with_redis, "cleanup", new=AsyncMock()
        ) as mock_cleanup:
            await music_bot_with_redis.on_voice_state_update(member, before, after)
        mock_cleanup.assert_not_called()

    async def test_last_human_leaves_starts_alone_timer(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """When the last human leaves the bot's channel, an alone-timer is started."""
        self._wire_bot_user(music_bot_with_redis)
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [bot_member]  # only the bot remains
        mock_guild.voice_client = vc

        member = MagicMock(spec=discord.Member)
        member.id = 123456789
        member.bot = False
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = vc.channel
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None

        task_created = []

        def _capture_and_close(coro: Coroutine[Any, Any, Any]) -> MagicMock:
            task_created.append(True)
            coro.close()  # prevent "coroutine was never awaited" ResourceWarning
            return MagicMock(spec=asyncio.Task)

        with patch("asyncio.create_task", side_effect=_capture_and_close):
            await music_bot_with_redis.on_voice_state_update(member, before, after)

        assert mock_guild.id in music_bot_with_redis.voice_watchdog._timers
        assert len(task_created) == 1

    async def test_human_rejoins_cancels_alone_timer(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """When a human joins the bot's channel while a timer is running, the timer is cancelled."""
        self._wire_bot_user(music_bot_with_redis)
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()

        timer = make_mock_task()
        music_bot_with_redis.voice_watchdog._timers[mock_guild.id] = timer

        human = MagicMock(spec=discord.Member)
        human.bot = False

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [human]  # a human is now present
        mock_guild.voice_client = vc

        member = MagicMock(spec=discord.Member)
        member.id = 123456789
        member.bot = False
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = None
        after = MagicMock(spec=discord.VoiceState)
        after.channel = vc.channel  # user joined the bot's channel

        await music_bot_with_redis.on_voice_state_update(member, before, after)

        timer.cancel.assert_called_once()
        assert mock_guild.id not in music_bot_with_redis.voice_watchdog._timers

    async def test_two_rapid_leaves_produce_one_timer(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Two members leaving in quick succession cancels the first timer and starts one new one."""
        self._wire_bot_user(music_bot_with_redis)
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [bot_member]
        mock_guild.voice_client = vc

        tasks_created = []
        first_task = MagicMock(spec=asyncio.Task)
        first_task.done.return_value = False
        first_task.cancel = MagicMock()

        def _capture_and_close(coro: Coroutine[Any, Any, Any]) -> MagicMock:
            coro.close()
            task = MagicMock(spec=asyncio.Task)
            task.done.return_value = False
            task.cancel = MagicMock()
            tasks_created.append(task)
            return task

        def _make_member() -> tuple[MagicMock, MagicMock, MagicMock]:
            m = MagicMock(spec=discord.Member)
            m.id = 123456789
            m.bot = False
            m.guild = mock_guild
            before = MagicMock(spec=discord.VoiceState)
            before.channel = vc.channel
            after = MagicMock(spec=discord.VoiceState)
            after.channel = None
            return m, before, after

        with patch("asyncio.create_task", side_effect=_capture_and_close):
            m1, b1, a1 = _make_member()
            await music_bot_with_redis.on_voice_state_update(m1, b1, a1)
            m2, b2, a2 = _make_member()
            await music_bot_with_redis.on_voice_state_update(m2, b2, a2)

        assert len(tasks_created) == 2
        tasks_created[
            0
        ].cancel.assert_called_once()  # first timer cancelled by second event
        assert (
            music_bot_with_redis.voice_watchdog._timers[mock_guild.id]
            is tasks_created[1]
        )

    async def test_member_change_in_unrelated_channel_ignored(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Member moving between two channels that aren't the bot's channel → no timer action."""
        self._wire_bot_user(music_bot_with_redis)
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()

        bot_channel = MagicMock()
        other_channel_a = MagicMock()
        other_channel_b = MagicMock()

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = bot_channel
        mock_guild.voice_client = vc

        member = MagicMock(spec=discord.Member)
        member.id = 123456789
        member.bot = False
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = other_channel_a
        after = MagicMock(spec=discord.VoiceState)
        after.channel = other_channel_b

        with patch("asyncio.create_task") as mock_create_task:
            await music_bot_with_redis.on_voice_state_update(member, before, after)

        mock_create_task.assert_not_called()
        assert mock_guild.id not in music_bot_with_redis.voice_watchdog._timers


class TestAloneCountdownNotice:
    async def test_notice_routes_through_send_with_np(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The countdown notice can fire mid-song — it must go through
        mp.send_with_np so it can't bury the NP host message."""
        mp = MagicMock()
        mp.send_with_np = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        mock_guild.voice_client = None  # post-sleep check: nothing to disconnect

        with patch("asyncio.sleep", new=AsyncMock()):
            await music_bot.voice_watchdog._countdown(mock_guild)

        mp.send_with_np.assert_awaited_once()
        embed = mp.send_with_np.call_args.kwargs["embed"]
        assert "disconnect" in embed.description


class TestAloneCountdown:
    def _make_vc(self, members: list[MagicMock]) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = members
        return vc

    def _setup_mp(self, music_bot: MusicBot, mock_guild: MagicMock) -> MagicMock:
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()
        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp
        return text_channel

    async def test_calls_cleanup_when_still_alone(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """After the sleep, if no humans remain, cleanup is called."""
        self._setup_mp(music_bot, mock_guild)

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot.voice_watchdog._countdown(mock_guild)

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_user_rejoined(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """After the sleep, if a human is present, cleanup is not called."""
        self._setup_mp(music_bot, mock_guild)

        human = MagicMock(spec=discord.Member)
        human.bot = False
        mock_guild.voice_client = self._make_vc([human])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot.voice_watchdog._countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_cancelled_before_sleep_skips_cleanup(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """CancelledError raised at sleep does not call cleanup."""
        self._setup_mp(music_bot, mock_guild)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot.voice_watchdog._countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_send_failure_does_not_abort_countdown(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """A failed text_channel.send is swallowed; the countdown still fires cleanup."""
        text_channel = self._setup_mp(music_bot, mock_guild)
        text_channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "forbidden")
        )

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot.voice_watchdog._countdown(mock_guild)

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_voice_client_gone(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """If the voice client is None when the countdown wakes, cleanup is not called."""
        self._setup_mp(music_bot, mock_guild)

        mock_guild.voice_client = None  # bot already disconnected mid-sleep

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot.voice_watchdog._countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_timer_removed_from_dict_on_completion(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The timer entry is removed in the finally block regardless of outcome."""
        self._setup_mp(music_bot, mock_guild)
        music_bot.voice_watchdog._timers[mock_guild.id] = MagicMock()  # sentinel

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()):
                await music_bot.voice_watchdog._countdown(mock_guild)

        assert mock_guild.id not in music_bot.voice_watchdog._timers
