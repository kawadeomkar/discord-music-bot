"""Tests for src/recovery.py — crash recovery (restore_guild) and the
alone-disconnect watchdog.

These drive the MusicBot cog, which owns the redis handle and the mps registry
restore_guild reads; the split follows test_leaderboard.py, where the new file
owns both the extracted module and the cog surface that reaches it.
"""

import asyncio
from typing import Any, Optional, cast
from collections.abc import Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
import redis.asyncio as aioredis
from discord.ext import commands
from redis.asyncio import Redis

from src.guild_state import GuildConfig
from src.musicbot import MusicBot
from src.recovery import (
    _COUNTDOWN_BAR_LEFT,
    _COUNTDOWN_BAR_WIDTH,
    VoiceWatchdog,
    _countdown_bar,
    _countdown_embed,
    _seconds_left,
    join_succeeded,
    restore_guild,
)
from src.redis_client import GuildRedisStore
from tests.helpers import make_mock_task, mocked, stalled_config_reads, stub_create_task


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

    async def test_spawns_one_recovery_task_and_only_the_first_sweeps(
        self, music_bot_with_redis: MusicBot
    ) -> None:
        stub = stub_create_task()
        with (
            patch("asyncio.create_task", stub),
            patch.object(music_bot_with_redis, "_recover_after_ready") as recover,
        ):
            await music_bot_with_redis.on_ready()
            await music_bot_with_redis.on_ready()
        assert stub.call_count == 2
        assert [c.kwargs for c in recover.call_args_list] == [
            {"sweep": True},
            {"sweep": False},
        ]

    async def test_recovery_hydrates_before_the_first_restore(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The hydrate's batches hold one pool connection at a time; the restore
        fan-out is what reaches the pool's cap, so it starts only after the pass."""
        guilds = list(music_bot_with_redis.bot.guilds)
        order: list[str] = []
        passed_guilds = []

        async def _hydrate(ids: Any = None) -> None:
            # Suspends, as the real read does: spawned restores would run here.
            await asyncio.sleep(0)
            order.append("hydrate")

        async def _noop() -> None:
            pass

        def _restore(cog: MusicBot, guild: MagicMock) -> Coroutine[Any, Any, None]:
            # Recorded when the task is spawned, not when it runs.
            order.append("restore")
            passed_guilds.append(guild)
            return _noop()

        # Patched where the cog LOOKS IT UP — musicbot's module globals — not
        # where it is defined, or the cog keeps calling the real one.
        with (
            patch.object(music_bot_with_redis, "_hydrate_configs", _hydrate),
            patch("src.musicbot.restore_guild", _restore),
        ):
            await music_bot_with_redis._recover_after_ready()
            await asyncio.gather(*music_bot_with_redis._restore_tasks)

        assert order == ["hydrate", *["restore"] * len(guilds)]
        assert passed_guilds == guilds


class TestOrphanSweepRunsAfterRecovery:
    """The sweep needs a complete guild cache and must not add a connection to the
    restore fan-out, so it runs once, after every restore it spawned returned."""

    @staticmethod
    def _gate(
        cog: MusicBot, *, shard_ids: Any = None, application_id: Any = 42
    ) -> None:
        bot = cast(Any, cog.bot)
        bot.shard_ids = shard_ids
        bot.application_id = application_id

    async def test_it_runs_only_after_every_restore_returned(
        self, music_bot_with_redis: MusicBot
    ) -> None:
        self._gate(music_bot_with_redis)
        order: list[str] = []
        release = asyncio.Event()

        async def _restore(cog: MusicBot, guild: MagicMock) -> None:
            await release.wait()
            order.append("restore")

        async def _sweep(**kwargs: Any) -> None:
            order.append("sweep")

        with (
            patch.object(music_bot_with_redis, "_hydrate_configs", AsyncMock()),
            patch("src.musicbot.restore_guild", _restore),
            patch.object(music_bot_with_redis.guild_settings, "sweep_orphans", _sweep),
        ):
            recovery = asyncio.create_task(
                music_bot_with_redis._recover_after_ready(sweep=True)
            )
            await asyncio.sleep(0.01)
            assert order == []
            release.set()
            await recovery

        assert order == ["restore"] * len(music_bot_with_redis.bot.guilds) + ["sweep"]

    @pytest.mark.parametrize(
        ("sweep", "shard_ids", "application_id"),
        [(False, None, 42), (True, [0], 42), (True, None, None)],
        ids=["not-the-first-ready", "a-shard-subset", "no-application-id"],
    )
    async def test_it_does_not_run(
        self,
        music_bot_with_redis: MusicBot,
        sweep: bool,
        shard_ids: Any,
        application_id: Any,
    ) -> None:
        self._gate(
            music_bot_with_redis, shard_ids=shard_ids, application_id=application_id
        )
        sweeper = AsyncMock()
        with (
            patch.object(music_bot_with_redis, "_hydrate_configs", AsyncMock()),
            patch("src.musicbot.restore_guild", AsyncMock()),
            patch.object(music_bot_with_redis.guild_settings, "sweep_orphans", sweeper),
        ):
            await music_bot_with_redis._recover_after_ready(sweep=sweep)
        sweeper.assert_not_awaited()

    async def test_membership_is_the_bots_guild_cache(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        self._gate(music_bot_with_redis)
        bot = cast(Any, music_bot_with_redis.bot)
        bot.get_guild = lambda guild_id: (
            mock_guild if guild_id == mock_guild.id else None
        )
        sweeper = AsyncMock()
        with (
            patch.object(music_bot_with_redis, "_hydrate_configs", AsyncMock()),
            patch("src.musicbot.restore_guild", AsyncMock()),
            patch.object(music_bot_with_redis.guild_settings, "sweep_orphans", sweeper),
        ):
            await music_bot_with_redis._recover_after_ready(sweep=True)
        call = sweeper.await_args
        assert call is not None
        assert call.kwargs["is_member"](mock_guild.id) is True
        assert call.kwargs["is_member"](1) is False
        assert call.kwargs["application_id"] == 42


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
        """The other half of _signal_rejoin: a timer registered without a rejoin
        event to set is cancelled outright, so the two dicts falling out of step
        can never leave a countdown running past a rejoin."""
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

    async def test_human_rejoins_signals_instead_of_cancelling(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """With a countdown listening, a rejoin SETS its event rather than cancelling
        it: a cancelled task cannot edit its card to say the bot stayed. The timer
        stays put — the countdown clears it from the dict itself, on its way out."""
        self._wire_bot_user(music_bot_with_redis)
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()
        watchdog = music_bot_with_redis.voice_watchdog

        timer = make_mock_task()
        rejoined = asyncio.Event()
        watchdog._timers[mock_guild.id] = timer
        watchdog._rejoins[mock_guild.id] = rejoined

        human = MagicMock(spec=discord.Member)
        human.bot = False

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [human]
        mock_guild.voice_client = vc

        member = MagicMock(spec=discord.Member)
        member.id = 123456789
        member.bot = False
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = None
        after = MagicMock(spec=discord.VoiceState)
        after.channel = vc.channel

        await music_bot_with_redis.on_voice_state_update(member, before, after)

        assert rejoined.is_set()
        timer.cancel.assert_not_called()
        assert watchdog._timers[mock_guild.id] is timer

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

    async def test_a_restarted_countdown_is_still_cancelled_by_a_rejoin(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Real tasks: a patched create_task never runs the countdown it replaces.
        The alone branch stores the replacement before that countdown unwinds, so
        an unwinding countdown that removed the guild's entry unconditionally would
        leave the replacement running where no rejoin could cancel it. Another bot
        joining is enough to restart it."""
        self._wire_bot_user(music_bot)
        mp = MagicMock()
        mp.send_with_np = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        watchdog = music_bot.voice_watchdog

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [bot_member]
        mock_guild.voice_client = vc

        def _move(*, bot: bool, joined: bool) -> tuple[MagicMock, MagicMock, MagicMock]:
            member = MagicMock(spec=discord.Member)
            member.id = 123456789
            member.bot = bot
            member.guild = mock_guild
            before = MagicMock(spec=discord.VoiceState)
            before.channel = None if joined else vc.channel
            after = MagicMock(spec=discord.VoiceState)
            after.channel = vc.channel if joined else None
            return member, before, after

        started: list[asyncio.Task[Any]] = []
        with patch.object(music_bot, "cleanup", new=AsyncMock()) as cleanup:
            try:
                await music_bot.on_voice_state_update(*_move(bot=False, joined=False))
                first = watchdog._timers[mock_guild.id]
                started.append(first)
                await asyncio.sleep(0)  # into its sleep

                await music_bot.on_voice_state_update(*_move(bot=True, joined=True))
                replacement = watchdog._timers[mock_guild.id]
                started.append(replacement)
                assert replacement is not first
                await asyncio.wait({first}, timeout=1.0)
                assert first.done()
                assert watchdog._timers.get(mock_guild.id) is replacement

                human = MagicMock(spec=discord.Member)
                human.bot = False
                vc.channel.members = [bot_member, human]
                await music_bot.on_voice_state_update(*_move(bot=False, joined=True))
                done, _ = await asyncio.wait({replacement}, timeout=1.0)
                assert replacement in done
                assert mock_guild.id not in watchdog._timers
                cleanup.assert_not_awaited()
            finally:
                for task in started:
                    task.cancel()
                await asyncio.gather(*started, return_exceptions=True)

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


class TestAloneTimeoutSetting:
    """The server's alone-timeout, read once when the countdown starts: the arm log,
    the card's opening frame, its deadline and the disconnect log all quote that one
    number."""

    @staticmethod
    def _alone(
        cog: MusicBot, guild: MagicMock
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        """The last human leaves the bot's channel."""
        bot_user = MagicMock()
        bot_user.id = 999999999999999999
        mocked(cog.bot).user = bot_user
        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [bot_member]
        guild.voice_client = vc
        member = MagicMock(spec=discord.Member)
        member.id = 123456789
        member.bot = False
        member.guild = guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = vc.channel
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None
        return member, before, after

    async def test_the_countdown_starts_with_the_servers_value(
        self,
        music_bot: MusicBot,
        mock_guild: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        music_bot.mps[mock_guild.id] = MagicMock()
        guild_settings = music_bot.guild_settings
        with guild_settings.reading() as started:
            guild_settings.seed(
                mock_guild.id, GuildConfig(alone_timeout_secs=120.0), started=started
            )
        countdown = AsyncMock()
        with (
            patch.object(VoiceWatchdog, "_countdown", new=countdown),
            caplog.at_level("INFO", logger="src.recovery"),
        ):
            await music_bot.on_voice_state_update(*self._alone(music_bot, mock_guild))
            await music_bot.voice_watchdog._timers[mock_guild.id]
        assert countdown.await_args is not None
        guild_arg, secs_arg, rejoined_arg = countdown.await_args.args
        assert guild_arg is mock_guild
        assert secs_arg == 120.0
        assert isinstance(rejoined_arg, asyncio.Event)
        assert "starting 120s disconnect timer" in caplog.text

    @pytest.mark.parametrize("secs", [10.0, 120.0])
    async def test_card_and_log_quote_the_number_passed_in(
        self,
        music_bot: MusicBot,
        mock_guild: MagicMock,
        caplog: pytest.LogCaptureFixture,
        secs: float,
    ) -> None:
        """The opening frame counts from the server's own value, not a fixed one.

        Driven with `rejoined` already set so the wait ends on its first pass: the
        opening frame is rendered from the FULL `secs` before any of it elapses, so
        the number is asserted without spending it. Nothing here is patched to
        produce that number - a countdown that ignored `secs` would fail.
        """
        card = MagicMock(spec=discord.Message)
        card.edit = AsyncMock()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock(return_value=card)
        mp = MagicMock()
        mp.home_channel = channel
        mp.repin_now_playing = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = [bot_member]
        mock_guild.voice_client = vc
        rejoined = asyncio.Event()
        rejoined.set()
        with (
            patch.object(music_bot, "cleanup", new=AsyncMock()) as cleanup,
            caplog.at_level("INFO", logger="src.recovery"),
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, secs, rejoined)
        opening = channel.send.call_args.kwargs["embed"]
        assert f"**{int(secs)} seconds**" in (opening.description or "")
        # The whole of `secs` was still on the clock, so nothing disconnected.
        cleanup.assert_not_awaited()
        assert "disconnecting" not in caplog.text

    async def test_the_bot_still_leaves_when_the_store_stalls(
        self, music_bot_with_redis: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The pool has no socket_timeout. The countdown's number comes from the
        cache, so a Redis that accepts and never answers cannot keep the bot in
        its channel."""
        cog = music_bot_with_redis
        mp = MagicMock()
        mp.home_channel = MagicMock(spec=discord.TextChannel)
        mp.home_channel.send = AsyncMock(return_value=MagicMock(spec=discord.Message))
        mp.repin_now_playing = AsyncMock()
        cog.mps[mock_guild.id] = mp
        countdown = AsyncMock()
        with (
            stalled_config_reads(),
            patch.object(VoiceWatchdog, "_countdown", new=countdown),
        ):
            await cog.on_voice_state_update(*self._alone(cog, mock_guild))
            async with asyncio.timeout(1):
                await cog.voice_watchdog._timers[mock_guild.id]
        # The stalled read fell back to the default rather than hanging the arm.
        assert countdown.await_args is not None
        assert countdown.await_args.args[1] == 10.0

        # And a countdown armed that way still reaches the disconnect.
        with patch.object(cog, "cleanup", new=AsyncMock()) as cleanup:
            await cog.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())
        cleanup.assert_awaited_once_with(mock_guild)


class TestCountdownRendering:
    """The pure half of the card: what a frame says, given seconds left."""

    def test_seconds_left_rounds_up(self) -> None:
        # Up, not down: a frame rendered 0.2s into the last second must still read
        # 1, or the card sits on zero while the bot is audibly still there.
        assert _seconds_left(deadline=10.0, now=1.2) == 9
        assert _seconds_left(deadline=10.0, now=9.99) == 1

    def test_seconds_left_floors_at_zero(self) -> None:
        assert _seconds_left(deadline=10.0, now=10.0) == 0
        assert _seconds_left(deadline=10.0, now=42.0) == 0

    def test_frame_quotes_its_own_number(self) -> None:
        assert "**7 seconds**" in (_countdown_embed(7, 10.0).description or "")

    def test_frame_singularizes_one_second(self) -> None:
        assert "**1 second**" in (_countdown_embed(1, 10.0).description or "")

    def test_bar_drains(self) -> None:
        full = _countdown_bar(10.0, 10.0)
        half = _countdown_bar(5.0, 10.0)
        empty = _countdown_bar(0.0, 10.0)
        assert full.count(_COUNTDOWN_BAR_LEFT) == _COUNTDOWN_BAR_WIDTH
        assert half.count(_COUNTDOWN_BAR_LEFT) == _COUNTDOWN_BAR_WIDTH // 2
        assert empty.count(_COUNTDOWN_BAR_LEFT) == 0
        assert len(full) == len(half) == len(empty)

    def test_bar_tolerates_a_zero_window(self) -> None:
        # The window is the guild's own alone-timeout, and a test drives the
        # countdown with 0.0; a zero would otherwise divide by it.
        assert _countdown_bar(0.0, 0.0).count(_COUNTDOWN_BAR_LEFT) == 0


class TestAloneCountdownCard:
    """The live card: one plain send, then an edit per tick, then a final frame."""

    @staticmethod
    def _make_vc(members: list[MagicMock]) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = members
        return vc

    @staticmethod
    def _bot_only() -> list[MagicMock]:
        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        return [bot_member]

    @staticmethod
    def _setup_mp(
        music_bot: MusicBot, mock_guild: MagicMock
    ) -> tuple[MagicMock, MagicMock]:
        """A player whose home channel hands back an editable card. (player, card)."""
        card = MagicMock(spec=discord.Message)
        card.edit = AsyncMock()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock(return_value=card)
        mp = MagicMock()
        mp.home_channel = channel
        mp.repin_now_playing = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        return mp, card

    @staticmethod
    def _titles(card: MagicMock) -> list[str]:
        return [c.kwargs["embed"].title for c in card.edit.await_args_list]

    async def test_card_is_a_plain_send_and_never_the_np_host(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """send_with_np would adopt this message as the Now Playing host, and the
        progress tick rebuilds a host from its cached send-time embeds — which would
        undo every countdown frame. Same reason -ping and -debug send plainly."""
        mp, _ = self._setup_mp(music_bot, mock_guild)
        mock_guild.voice_client = None

        await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mp.home_channel.send.assert_awaited_once()
        mp.send_with_np.assert_not_called()

    async def test_card_is_edited_repeatedly_while_it_waits(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The point of the feature: the card moves on its own. Only the cadence is
        asserted here — what a frame SAYS is pinned by TestCountdownRendering, which
        needs no clock."""
        _, card = self._setup_mp(music_bot, mock_guild)
        mock_guild.voice_client = self._make_vc(self._bot_only())

        with (
            patch("src.recovery._COUNTDOWN_TICK_SECS", 0.02),
            patch.object(music_bot, "cleanup", new=AsyncMock()),
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, 0.3, asyncio.Event())

        # Deliberately loose: the tick is real time, and a loaded runner delivers
        # fewer frames without the behaviour being wrong.
        assert card.edit.await_count >= 3

    async def test_last_frame_reports_the_disconnect(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        _, card = self._setup_mp(music_bot, mock_guild)
        mock_guild.voice_client = self._make_vc(self._bot_only())

        with (
            patch.object(music_bot, "cleanup", new=AsyncMock()),
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        assert self._titles(card)[-1] == "Disconnected from voice channel"

    async def test_last_frame_reports_a_rejoin_and_repins_the_np_block(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """A frozen "disconnect in 1 second" would read as a bot that left. The card
        has to say it stayed — and the Now Playing block it buried goes back to the
        bottom of the channel."""
        mp, card = self._setup_mp(music_bot, mock_guild)
        human = MagicMock(spec=discord.Member)
        human.bot = False
        mock_guild.voice_client = self._make_vc([human])

        with (
            patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup,
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        assert self._titles(card)[-1] == "Someone rejoined"
        mp.repin_now_playing.assert_awaited_once()
        mock_cleanup.assert_not_awaited()

    async def test_rejoin_event_ends_the_countdown_early(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The event is what makes the card prompt; the membership re-check is what
        actually holds the disconnect back."""
        mp, card = self._setup_mp(music_bot, mock_guild)
        human = MagicMock(spec=discord.Member)
        human.bot = False
        mock_guild.voice_client = self._make_vc([human])
        rejoined = asyncio.Event()

        with (
            # Long enough that finishing at all proves the event cut it short.
            patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup,
        ):
            task = asyncio.create_task(
                music_bot.voice_watchdog._countdown(mock_guild, 60.0, rejoined)
            )
            await asyncio.sleep(0.02)
            rejoined.set()
            async with asyncio.timeout(5):
                await task

        assert self._titles(card)[-1] == "Someone rejoined"
        mock_cleanup.assert_not_awaited()
        mp.repin_now_playing.assert_awaited_once()

    async def test_deleted_card_stops_the_edits_but_not_the_disconnect(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        _, card = self._setup_mp(music_bot, mock_guild)
        card.edit = AsyncMock(side_effect=discord.NotFound(MagicMock(), "gone"))
        mock_guild.voice_client = self._make_vc(self._bot_only())

        with (
            patch("src.recovery._COUNTDOWN_TICK_SECS", 0.02),
            patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup,
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, 0.3, asyncio.Event())

        assert card.edit.await_count == 1  # the loop stops spending edits on it
        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_failing_edits_do_not_stop_the_disconnect(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Anything but a deleted message keeps the card: a stalled countdown card
        must not strand the bot in an empty channel."""
        _, card = self._setup_mp(music_bot, mock_guild)
        card.edit = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "boom"))
        mock_guild.voice_client = self._make_vc(self._bot_only())

        with (
            patch("src.recovery._COUNTDOWN_TICK_SECS", 0.02),
            patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup,
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, 0.3, asyncio.Event())

        assert card.edit.await_count > 1
        mock_cleanup.assert_awaited_once_with(mock_guild)


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
        mp.home_channel = text_channel
        mp.repin_now_playing = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        return text_channel

    async def test_calls_cleanup_when_still_alone(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Once the clock runs out, if no humans remain, cleanup is called."""
        self._setup_mp(music_bot, mock_guild)

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_user_rejoined(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Once the clock runs out, if a human is present, cleanup is not called."""
        self._setup_mp(music_bot, mock_guild)

        human = MagicMock(spec=discord.Member)
        human.bot = False
        mock_guild.voice_client = self._make_vc([human])

        with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mock_cleanup.assert_not_awaited()

    async def test_cancellation_mid_countdown_skips_cleanup(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Teardown cancels the task; it must not disconnect on the way out."""
        self._setup_mp(music_bot, mock_guild)

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with (
            patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup,
        ):
            task = asyncio.create_task(
                music_bot.voice_watchdog._countdown(mock_guild, 60.0, asyncio.Event())
            )
            await asyncio.sleep(0.02)
            task.cancel()
            async with asyncio.timeout(5):
                await task

        mock_cleanup.assert_not_awaited()

    async def test_send_failure_does_not_abort_countdown(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """A failed card send is swallowed; the countdown still fires cleanup."""
        text_channel = self._setup_mp(music_bot, mock_guild)
        text_channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "forbidden")
        )

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_no_player_still_disconnects(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """No player means no channel to post a card in — the disconnect is not
        conditional on the card."""
        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_voice_client_gone(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """If the voice client is None when the countdown ends, cleanup is not called."""
        self._setup_mp(music_bot, mock_guild)

        mock_guild.voice_client = None  # bot already disconnected mid-countdown

        with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mock_cleanup.assert_not_awaited()

    async def test_timer_and_event_removed_on_completion(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The finally block clears both dicts regardless of outcome."""
        self._setup_mp(music_bot, mock_guild)
        watchdog = music_bot.voice_watchdog
        rejoined = asyncio.Event()
        watchdog._rejoins[mock_guild.id] = rejoined

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with (
            patch.object(music_bot, "cleanup", new=AsyncMock()),
        ):
            task = asyncio.create_task(watchdog._countdown(mock_guild, 0.0, rejoined))
            watchdog._timers[mock_guild.id] = task
            async with asyncio.timeout(5):
                await task

        assert mock_guild.id not in watchdog._timers
        assert mock_guild.id not in watchdog._rejoins

    async def test_finishing_countdown_leaves_a_replacement_alone(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """A restart registers the replacement before the old task processes its
        cancellation. An unguarded pop in the finally would drop the NEW countdown
        out of both dicts, leaving it running with no way to stop it — so a rejoin
        could no longer prevent the disconnect."""
        self._setup_mp(music_bot, mock_guild)
        watchdog = music_bot.voice_watchdog
        mock_guild.voice_client = None  # nothing to disconnect; ends immediately

        old = asyncio.create_task(watchdog._countdown(mock_guild, 0.0, asyncio.Event()))
        replacement_timer = make_mock_task()
        replacement_event = asyncio.Event()
        watchdog._timers[mock_guild.id] = replacement_timer
        watchdog._rejoins[mock_guild.id] = replacement_event
        async with asyncio.timeout(5):
            await old

        assert watchdog._timers[mock_guild.id] is replacement_timer
        assert watchdog._rejoins[mock_guild.id] is replacement_event


class TestAloneCountdownFailsSafe:
    """The countdown is the only thing that ends an empty voice session, so nothing
    inside it may raise out or leave the guild silently un-disconnected."""

    async def test_unexpected_error_is_logged_not_raised(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        vc.channel.members = [bot_member]
        mock_guild.voice_client = vc

        with (
            patch.object(
                music_bot, "cleanup", new=AsyncMock(side_effect=RuntimeError("boom"))
            ),
            patch("src.recovery.log") as mock_log,
        ):
            await music_bot.voice_watchdog._countdown(mock_guild, 0.0, asyncio.Event())

        mock_log.error.assert_called_once()

    async def test_repin_skipped_when_the_player_is_already_gone(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """cleanup() can pop the player between the final frame and the repin."""
        await music_bot.voice_watchdog._repin_now_playing(mock_guild)  # no raise


class TestJoinSucceeded:
    """The check both cold-start commands gate their insert on. Its whole reason to
    exist is the still-connecting case: a type-only check passes there and hands the
    loop a client vc.play() raises on, once per restored song."""

    @staticmethod
    def _ctx(voice_client: object) -> MagicMock:
        ctx = MagicMock(spec=commands.Context)
        ctx.voice_client = voice_client
        return ctx

    def test_connected_client_succeeds(self) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_connected.return_value = True
        assert join_succeeded(self._ctx(vc)) is True

    def test_still_connecting_client_fails(self) -> None:
        # discord.py registers the client on the guild BEFORE the handshake, so this
        # is a real state a concurrent cold -play leaves behind — not a mock artifact.
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_connected.return_value = False
        assert join_succeeded(self._ctx(vc)) is False

    def test_absent_client_fails(self) -> None:
        # join swallows its own failures, so a failed join arrives as None.
        assert join_succeeded(self._ctx(None)) is False

    def test_non_voice_client_fails(self) -> None:
        assert join_succeeded(self._ctx(MagicMock())) is False
