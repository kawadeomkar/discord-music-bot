"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""

import asyncio
import contextlib
import orjson
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import fakeredis
import pytest
from discord.ext import commands

from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.musicbot import (
    HistoryFlags,
    MusicBot,
    _check_voice_permissions,
    background_typing,
)
from src.util import latency_color
from src.sources import SpotifySource, SpotifyType, YTSource, YTType
from src.youtube import QueueObject
from tests.helpers import make_mock_task, stub_create_task


@pytest.fixture
def music_bot(mock_bot):
    """Minimal MusicBot instance bypassing __init__ Discord registration."""
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog.redis = None
    cog._active_spans = {}
    cog._alone_timers = {}
    cog._restore_tasks = set()
    return cog


class TestCheckVoicePermissions:
    def test_rejects_non_member_user(self):
        user = MagicMock(spec=discord.User)
        assert _check_voice_permissions(user, None, "play") is not None

    def test_rejects_member_not_in_voice_channel(self):
        member = MagicMock(spec=discord.Member)
        member.voice = None
        assert _check_voice_permissions(member, None, "play") is not None

    def test_rejects_wrong_voice_channel_for_non_play(self):
        member = MagicMock(spec=discord.Member)
        channel_a = MagicMock()
        channel_b = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel_a
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel_b
        assert _check_voice_permissions(member, vc, "skip") is not None

    def test_allows_play_in_different_channel(self):
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()  # different from member's channel — OK for play
        assert _check_voice_permissions(member, vc, "play") is None

    def test_passes_valid_member_in_correct_channel(self):
        member = MagicMock(spec=discord.Member)
        channel = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel
        assert _check_voice_permissions(member, vc, "skip") is None

    def test_passes_when_no_voice_client(self):
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        assert _check_voice_permissions(member, None, "skip") is None


class TestBackgroundTyping:
    """docs/PERFORMANCE_PLAN.md §2.2 — typing indicator must never delay the command body."""

    async def test_body_runs_while_first_typing_post_is_in_flight(self, mock_ctx):
        post_started = asyncio.Event()
        release_post = asyncio.Event()

        async def slow_post():
            post_started.set()
            await release_post.wait()

        mock_ctx.typing.return_value.__aenter__ = AsyncMock(side_effect=slow_post)

        async with background_typing(mock_ctx):
            # The body is executing while the POST is still blocked — the ~500ms
            # first POST no longer serializes ahead of the work.
            await asyncio.wait_for(post_started.wait(), timeout=1)
            assert not release_post.is_set()

    async def test_cancel_during_first_post_never_enters_typing_cm(self, mock_ctx):
        """Exiting while the first POST is in flight must not leak the keepalive:
        the CM never entered, so __aexit__ is never called (the AttributeError
        hazard of driving __aenter__/__aexit__ manually cannot occur)."""
        post_started = asyncio.Event()

        async def hung_post():
            post_started.set()
            await asyncio.sleep(3600)

        mock_ctx.typing.return_value.__aenter__ = AsyncMock(side_effect=hung_post)

        async with background_typing(mock_ctx):
            await asyncio.wait_for(post_started.wait(), timeout=1)
        await asyncio.sleep(0)  # let the cancelled keepalive task unwind
        mock_ctx.typing.return_value.__aexit__.assert_not_awaited()

    async def test_typing_cm_exited_after_body_completes(self, mock_ctx):
        exited = asyncio.Event()
        mock_ctx.typing.return_value.__aexit__ = AsyncMock(
            side_effect=lambda *a: exited.set()
        )
        entered = asyncio.Event()
        mock_ctx.typing.return_value.__aenter__ = AsyncMock(
            side_effect=lambda: entered.set()
        )

        async with background_typing(mock_ctx):
            await asyncio.wait_for(entered.wait(), timeout=1)
        # Cancellation unwinds through the async with → indicator dropped promptly.
        await asyncio.wait_for(exited.wait(), timeout=1)
        mock_ctx.typing.return_value.__aexit__.assert_awaited_once()

    async def test_typing_failure_never_surfaces_into_command_body(self, mock_ctx):
        mock_ctx.typing.side_effect = RuntimeError("typing endpoint down")

        async with background_typing(mock_ctx):
            await asyncio.sleep(0)  # let the keepalive task hit the failure
            await asyncio.sleep(0)
        # No exception propagates; the command body is unaffected.

    async def test_body_exception_still_cancels_keepalive(self, mock_ctx):
        entered = asyncio.Event()
        exited = asyncio.Event()
        mock_ctx.typing.return_value.__aenter__ = AsyncMock(
            side_effect=lambda: entered.set()
        )
        mock_ctx.typing.return_value.__aexit__ = AsyncMock(
            side_effect=lambda *a: exited.set()
        )

        with pytest.raises(ValueError):
            async with background_typing(mock_ctx):
                await asyncio.wait_for(entered.wait(), timeout=1)
                raise ValueError("command body blew up")
        await asyncio.wait_for(exited.wait(), timeout=1)


class TestLatencyColor:
    def test_excellent_latency_is_green(self):
        assert latency_color(30).value == 0x44FF44

    def test_boundary_50ms_is_green(self):
        assert latency_color(50).value == 0x44FF44

    def test_good_latency_is_yellow(self):
        assert latency_color(75).value == 0xFFD000

    def test_boundary_100ms_is_yellow(self):
        assert latency_color(100).value == 0xFFD000

    def test_acceptable_latency_is_orange(self):
        assert latency_color(150).value == 0xFF6600

    def test_boundary_200ms_is_orange(self):
        assert latency_color(200).value == 0xFF6600

    def test_poor_latency_is_red(self):
        assert latency_color(300).value == 0x990000


class TestQueueSource:
    async def test_spotify_playlist_returns_list(self, music_bot, mock_ctx):
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        music_bot.spotify.playlist = AsyncMock(return_value=["Song A", "Song B"])
        result = await music_bot.queue_source(mock_ctx, source)
        assert result == ["Song A", "Song B"]

    async def test_spotify_track_calls_yt_source(self, music_bot, mock_ctx):
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        fake_qobj = QueueObject("https://yt.com/v=1", "My Track", mock_ctx.author)
        music_bot.spotify.track = AsyncMock(return_value="My Track Artist")
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_bot.queue_source(mock_ctx, source)
        assert isinstance(result, QueueObject)

    async def test_youtube_url_calls_yt_source(self, music_bot, mock_ctx):
        source = YTSource(url="https://yt.com/watch?v=abc", process=False)
        fake_qobj = QueueObject(
            "https://yt.com/watch?v=abc", "YT Song", mock_ctx.author
        )
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_bot.queue_source(mock_ctx, source)
        assert isinstance(result, QueueObject)

    async def test_youtube_search_uses_ytsearch(self, music_bot, mock_ctx):
        source = YTSource(ytsearch="ytsearch:test song", process=True)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ) as mock_yt:
            await music_bot.queue_source(mock_ctx, source)
        call_args = mock_yt.call_args
        assert call_args[0][1] == "ytsearch:test song"

    async def test_youtube_playlist_calls_yt_playlist(self, music_bot, mock_ctx):
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest123",
            process=False,
            type=YTType.PLAYLIST,
            list_id="PLtest123",
        )
        fake_qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author),
            QueueObject("https://yt.com/watch?v=2", "Track 2", mock_ctx.author),
        ]
        with patch(
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=fake_qobjs)
        ) as mock_playlist:
            result = await music_bot.queue_source(mock_ctx, source)
        mock_playlist.assert_awaited_once_with(
            "https://www.youtube.com/playlist?list=PLtest123", mock_ctx.author
        )
        assert result == fake_qobjs

    async def test_youtube_playlist_raises_if_list_id_missing(
        self, music_bot, mock_ctx
    ):
        """queue_source raises ValueError (not AssertionError) when list_id is None."""
        source = YTSource(
            url="https://www.youtube.com/watch?v=abc",
            process=False,
            type=YTType.PLAYLIST,
            list_id=None,
        )
        with pytest.raises(ValueError, match="list_id"):
            await music_bot.queue_source(mock_ctx, source)

    async def test_youtube_playlist_preserves_full_url(self, music_bot, mock_ctx):
        full_url = "https://www.youtube.com/watch?v=XfHbPIx42uo&list=RDXfHbPIx42uo&start_radio=1"
        source = YTSource(
            url=full_url,
            process=False,
            type=YTType.PLAYLIST,
            list_id="RDXfHbPIx42uo",
        )
        fake_qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)
        ]
        with patch(
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=fake_qobjs)
        ) as mock_playlist:
            await music_bot.queue_source(mock_ctx, source)
        mock_playlist.assert_awaited_once_with(full_url, mock_ctx.author)


class TestEnqueuePlaylist:
    @staticmethod
    def _make_enqueue_mp(mock_ctx) -> MagicMock:
        mp = MagicMock()
        mp.queue_put = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()
        return mp

    # ── YouTube playlist path ─────────────────────────────────────────────────

    async def test_yt_sends_embed_with_song_count_and_playlist_url(
        self, music_bot, mock_ctx
    ):
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author),
            QueueObject("https://yt.com/watch?v=2", "Track 2", mock_ctx.author),
        ]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(mock_ctx, source, qobjs, mp)

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "2 songs" in embed.title
        assert source.url in embed.description
        assert "Track 1" in embed.description

    async def test_yt_singular_song_count_in_title(self, music_bot, mock_ctx):
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Only Track", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(mock_ctx, source, qobjs, mp)

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "1 song" in embed.title
        assert "1 songs" not in embed.title

    async def test_yt_calls_queue_put_with_prefetch_false(self, music_bot, mock_ctx):
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(mock_ctx, source, qobjs, mp)

        mp.queue_put.assert_awaited_once()
        _, call_kwargs = mp.queue_put.call_args
        assert call_kwargs.get("prefetch") is False

    # ── Spotify playlist path ─────────────────────────────────────────────────

    async def test_spotify_sends_queued_playlist_embed(self, music_bot, mock_ctx):
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        titles = ["Song A", "Song B", "Song C"]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(mock_ctx, source, titles, mp)

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Queued playlist" in embed.title
        assert "Song A" in embed.description

    async def test_spotify_calls_queue_put_with_prefetch_false(
        self, music_bot, mock_ctx
    ):
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        titles = ["Song A", "Song B"]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(mock_ctx, source, titles, mp)

        mp.queue_put.assert_awaited_once()
        _, call_kwargs = mp.queue_put.call_args
        assert call_kwargs.get("prefetch") is False


@pytest.fixture
async def fake_redis_bot():
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def music_bot_with_redis(mock_bot, fake_redis_bot):
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog.redis = fake_redis_bot
    cog._active_spans = {}
    cog._alone_timers = {}
    cog._restore_tasks = set()
    return cog


class TestJoinChannelPersistence:
    async def test_join_writes_channel_ids_to_redis(
        self, music_bot_with_redis, mock_ctx, mock_guild, fake_redis_bot
    ):
        """Calling join should persist voice and text channel IDs to Redis."""
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 777000000000000001
        voice_channel.connect = AsyncMock()
        mock_ctx.author.voice.channel = voice_channel
        mock_guild.change_voice_state = AsyncMock()
        mock_guild.voice_client = None

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.id = 777000000000000002
        mock_ctx.channel = text_channel

        mp = MagicMock()
        mp.store = MagicMock()
        mp.store.set_connection = AsyncMock()
        music_bot_with_redis.mps[mock_guild.id] = mp

        # join is a @commands.command — call the underlying callback directly.
        mock_ctx.voice_client = None  # bot not yet in channel
        with (
            patch.object(discord.VoiceChannel, "connect", new=AsyncMock()),
            patch.object(mock_ctx, "invoke", new=AsyncMock()),
        ):
            music_bot_with_redis.get_mp = MagicMock(return_value=mp)
            await MusicBot.join.callback(music_bot_with_redis, mock_ctx)

        mp.store.set_connection.assert_awaited_once_with(
            voice_channel.id, text_channel.id
        )
        # Voice is up — a queue persisted by a previous -stop resumes.
        mp.open_playback_gate.assert_called_once()


class TestEagerRestore:
    async def test_restore_guild_skips_if_already_in_mps(
        self, music_bot_with_redis, mock_guild
    ):
        """_restore_guild is a no-op if the guild already has a MusicPlayer."""
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()
        # Should not raise or create another player
        await music_bot_with_redis._restore_guild(mock_guild)
        assert len(music_bot_with_redis.mps) == 1

    async def test_restore_guild_skips_when_no_channel_ids(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """_restore_guild exits early when no connection was persisted."""
        await music_bot_with_redis._restore_guild(mock_guild)
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_restore_guild_skips_when_queue_empty_and_no_crash(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """No queue items + no crashed song → skip restore even if channel IDs exist."""
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)
        # No queue items, no current_song_url in state

        await music_bot_with_redis._restore_guild(mock_guild)
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_restore_guild_gates_without_reading_queue_payload(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """NIT-7: a -stop'ped guild keeps its (possibly long) queue list, so the
        recovery gate must never pull the full playback aggregate just to
        conclude "nothing to do" — it reads state + LLEN via get_recovery_gate,
        not get_playback_snapshot."""
        from src.guild_state import SongQueueEntry
        from src.redis_client import GuildRedisStore

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
            await music_bot_with_redis._restore_guild(mock_guild)

        snapshot_spy.assert_not_awaited()
        assert mock_guild.id not in music_bot_with_redis.mps


class TestVoiceStateConsistency:
    @staticmethod
    def _wire_bot_user(cog) -> None:
        mock_user = MagicMock()
        mock_user.id = 999999999999999999
        cog.bot.user = mock_user

    async def test_bot_disconnect_triggers_cleanup(
        self, music_bot_with_redis, mock_guild
    ):
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
        self, music_bot_with_redis, mock_guild
    ):
        """Bot moved to a new channel (not ejected) cancels any running alone-timer."""
        self._wire_bot_user(music_bot_with_redis)

        timer = make_mock_task()
        music_bot_with_redis._alone_timers[mock_guild.id] = timer

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
        assert mock_guild.id not in music_bot_with_redis._alone_timers

    async def test_member_in_inactive_guild_ignored(
        self, music_bot_with_redis, mock_guild
    ):
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
        self, music_bot_with_redis, mock_guild
    ):
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

        def _capture_and_close(coro):
            task_created.append(True)
            coro.close()  # prevent "coroutine was never awaited" ResourceWarning
            return MagicMock(spec=asyncio.Task)

        with patch("asyncio.create_task", side_effect=_capture_and_close):
            await music_bot_with_redis.on_voice_state_update(member, before, after)

        assert mock_guild.id in music_bot_with_redis._alone_timers
        assert len(task_created) == 1

    async def test_human_rejoins_cancels_alone_timer(
        self, music_bot_with_redis, mock_guild
    ):
        """When a human joins the bot's channel while a timer is running, the timer is cancelled."""
        self._wire_bot_user(music_bot_with_redis)
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()

        timer = make_mock_task()
        music_bot_with_redis._alone_timers[mock_guild.id] = timer

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
        assert mock_guild.id not in music_bot_with_redis._alone_timers

    async def test_two_rapid_leaves_produce_one_timer(
        self, music_bot_with_redis, mock_guild
    ):
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

        def _capture_and_close(coro):
            coro.close()
            task = MagicMock(spec=asyncio.Task)
            task.done.return_value = False
            task.cancel = MagicMock()
            tasks_created.append(task)
            return task

        def _make_member():
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
        assert music_bot_with_redis._alone_timers[mock_guild.id] is tasks_created[1]

    async def test_member_change_in_unrelated_channel_ignored(
        self, music_bot_with_redis, mock_guild
    ):
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
        assert mock_guild.id not in music_bot_with_redis._alone_timers


# ── New coverage: __init__, get_mp, cleanup, validate_commands, commands, on_ready ──


class TestMusicBotInit:
    @pytest.fixture(autouse=True)
    def _spotify_env(self, monkeypatch):
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "x")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "y")

    def test_sets_bot_attribute(self, mock_bot):
        assert MusicBot(mock_bot).bot is mock_bot

    def test_mps_starts_empty(self, mock_bot):
        assert MusicBot(mock_bot).mps == {}

    def test_alone_timers_starts_empty(self, mock_bot):
        assert MusicBot(mock_bot)._alone_timers == {}

    def test_reads_redis_from_bot(self, mock_bot):
        mock_redis = MagicMock()
        mock_bot.redis = mock_redis
        assert MusicBot(mock_bot).redis is mock_redis


class TestGetMp:
    def test_returns_existing_player_and_sets_context(
        self, music_bot, mock_ctx, mock_guild
    ):
        mp = MagicMock()
        mp.set_context = MagicMock()
        music_bot.mps[mock_guild.id] = mp
        result = music_bot.get_mp(mock_ctx)
        assert result is mp
        mp.set_context.assert_called_once_with(mock_ctx)

    def test_creates_new_player_for_unknown_guild(
        self, music_bot, mock_ctx, mock_guild
    ):
        mock_mp = MagicMock()
        mock_mp.start = MagicMock()
        with patch("src.musicbot.MusicPlayer.from_context", return_value=mock_mp):
            result = music_bot.get_mp(mock_ctx)
        assert result is mock_mp
        mock_mp.start.assert_called_once()
        assert mock_guild.id in music_bot.mps


class TestCleanup:
    @staticmethod
    def _make_minimal_mp(music_bot, mock_guild, **overrides) -> MagicMock:
        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        mp.store = None
        mp._progress_task = None
        mp._pause_debounce_task = None
        mp.retire_np_host_on_stop = AsyncMock()
        for attr, val in overrides.items():
            setattr(mp, attr, val)
        music_bot.mps[mock_guild.id] = mp
        return mp

    async def test_does_not_cancel_current_task(self, music_bot, mock_guild):
        """cleanup() skips cancellation when the alone-timer IS the running task (self-cancel guard)."""
        current = asyncio.current_task()
        assert current is not None, "test must run inside an asyncio.Task"
        music_bot._alone_timers[mock_guild.id] = (
            current  # simulate countdown calling cleanup on itself
        )

        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        # If the guard were missing, current_task().cancel() would have been called
        # and this coroutine would receive CancelledError at the next await.
        assert not current.cancelled()
        assert mock_guild.id not in music_bot._alone_timers

    async def test_disconnects_voice_client(self, music_bot, mock_guild):
        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client.disconnect = AsyncMock()
        await music_bot.cleanup(mock_guild)
        mock_guild.voice_client.disconnect.assert_awaited_once()

    async def test_removes_guild_from_mps(self, music_bot, mock_guild):
        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        assert mock_guild.id not in music_bot.mps

    async def test_cancels_in_flight_prefetch_task(self, music_bot, mock_guild):
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _prefetch_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_cancels_in_flight_progress_task(self, music_bot, mock_guild):
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _progress_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_cancels_in_flight_pause_debounce_task(self, music_bot, mock_guild):
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _pause_debounce_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_retires_np_host_after_task_cancellation(self, music_bot, mock_guild):
        """-stop / alone-disconnect must dispose of the NP host (delete a
        dedicated NP message, strip a response host) so no message keeps a
        mid-song bar frozen by the stop — and only after the progress/loop
        tasks are down, so no tick can race the retire."""
        call_order: list[str] = []

        class _AwaitableTask:
            def done(self):
                return False

            def cancel(self, msg=None):
                call_order.append("cancel")

            def __await__(self):
                return iter([])  # completes immediately, no exception

        mp = self._make_minimal_mp(
            music_bot, mock_guild, _progress_task=_AwaitableTask()
        )
        mp.retire_np_host_on_stop = AsyncMock(
            side_effect=lambda: call_order.append("retire")
        )
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        assert call_order == ["cancel", "retire"]

    async def test_cancels_running_alone_timer(self, music_bot, mock_guild):
        timer = make_mock_task()
        music_bot._alone_timers[mock_guild.id] = timer

        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        timer.cancel.assert_called_once()
        assert mock_guild.id not in music_bot._alone_timers

    async def test_noop_cleanup_does_not_error_without_timer(
        self, music_bot, mock_guild
    ):
        # No timer in _alone_timers — cleanup must not raise KeyError.
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)  # guild not in mps either — pure noop

    async def test_clears_store_connection_on_cleanup(self, music_bot, mock_guild):
        store = MagicMock()
        store.clear_connection = AsyncMock()
        store.refresh_ttl = AsyncMock()
        mp = self._make_minimal_mp(music_bot, mock_guild, store=store)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        mp.store.clear_connection.assert_awaited_once()
        mp.store.refresh_ttl.assert_awaited_once()

    async def test_noop_when_guild_not_in_mps(self, music_bot, mock_guild):
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)  # must not raise

    async def test_cancels_player_task_before_disconnect(self, music_bot, mock_guild):
        """_player must be cancelled before disconnect() so the loop cannot wake up
        and start the next song between voice_client.stop() firing and the loop
        being cancelled (the root cause of the brief-next-song-on-stop bug)."""
        call_order: list[str] = []

        class _AwaitableTask:
            """Minimal awaitable task double: done()=False, cancel() tracked, await=noop."""

            def done(self):
                return False

            def cancel(self, msg=None):
                call_order.append("cancel")

            def __await__(self):
                return iter([])  # completes immediately, no exception

        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = _AwaitableTask()
        mp.store = None
        mp.retire_np_host_on_stop = AsyncMock()
        music_bot.mps[mock_guild.id] = mp

        async def _disconnect(**_kw):
            call_order.append("disconnect")

        mock_guild.voice_client.disconnect = AsyncMock(side_effect=_disconnect)

        await music_bot.cleanup(mock_guild)

        assert call_order.index("cancel") < call_order.index("disconnect"), (
            "player task must be cancelled before voice disconnect"
        )


class TestStopCommand:
    async def test_stop_adds_wave_reaction(self, music_bot, mock_ctx, mock_guild):
        music_bot.cleanup = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.stop.callback(music_bot, mock_ctx)
        mock_ctx.message.add_reaction.assert_awaited_once_with("👋")

    async def test_stop_calls_cleanup(self, music_bot, mock_ctx, mock_guild):
        music_bot.cleanup = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.stop.callback(music_bot, mock_ctx)
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)

    async def test_stop_does_not_call_skip(self, music_bot, mock_ctx, mock_guild):
        """stop must not invoke skip — skip fires voice_client.stop() which triggers
        the after callback and gives the playback loop a window to start the next song.
        """
        music_bot.cleanup = AsyncMock()
        music_bot.skip = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.stop.callback(music_bot, mock_ctx)
        music_bot.skip.assert_not_called()

    async def test_stop_noop_when_no_voice_client(
        self, music_bot, mock_ctx, mock_guild
    ):
        music_bot.cleanup = AsyncMock()
        with patch("discord.utils.get", return_value=None):
            await MusicBot.stop.callback(music_bot, mock_ctx)
        music_bot.cleanup.assert_not_awaited()


class TestCogBeforeInvoke:
    async def test_calls_get_mp(self, music_bot, mock_ctx):
        mock_mp = MagicMock()
        mock_mp.store = None  # skip the channel-persistence branch
        music_bot.get_mp = MagicMock(return_value=mock_mp)
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_called_once_with(mock_ctx)

    async def test_persists_text_channel_when_channel_changes(
        self, music_bot, mock_ctx, mock_guild
    ):
        """set_connection is called when the command arrives from a new text channel."""
        old_channel = MagicMock(spec=discord.TextChannel)
        new_channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = new_channel

        store = MagicMock()
        store.set_connection = AsyncMock()

        mp = MagicMock()
        mp._channel = old_channel
        mp.store = store
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.id = 555
        mock_guild.voice_client = vc

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_awaited_once_with(vc.channel.id, new_channel.id)

    async def test_no_persist_when_channel_unchanged(
        self, music_bot, mock_ctx, mock_guild
    ):
        """set_connection is NOT called when the text channel hasn't changed."""
        channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = channel

        store = MagicMock()
        store.set_connection = AsyncMock()

        mp = MagicMock()
        mp._channel = channel  # same object → no change
        mp.store = store
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_not_awaited()

    async def test_no_persist_when_no_voice_client(
        self, music_bot, mock_ctx, mock_guild
    ):
        """set_connection is NOT called when the bot isn't in a voice channel yet."""
        old_channel = MagicMock(spec=discord.TextChannel)
        new_channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = new_channel

        store = MagicMock()
        store.set_connection = AsyncMock()

        mp = MagicMock()
        mp._channel = old_channel
        mp.store = store
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        mock_guild.voice_client = None  # not connected yet

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_not_awaited()

    async def test_returns_early_when_guild_is_none(self, music_bot, mock_ctx):
        """cog_before_invoke must not call get_mp (which asserts guild is not None) in a DM."""
        mock_ctx.guild = None
        music_bot.get_mp = MagicMock()
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_not_called()


class TestValidateCommands:
    async def test_raises_command_error_when_not_in_voice(self, music_bot, mock_ctx):
        mock_ctx.voice_client = None
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "skip"
        mock_ctx.author.voice = None
        mock_ctx.send = AsyncMock()
        with pytest.raises(commands.CommandError):
            await music_bot.validate_commands(mock_ctx)
        mock_ctx.send.assert_awaited_once()

    async def test_passes_when_member_in_voice(self, music_bot, mock_ctx):
        mock_ctx.voice_client = None
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "play"
        # mock_ctx.author has voice set by conftest
        await music_bot.validate_commands(mock_ctx)  # must not raise


class TestSkipCommand:
    async def test_stops_voice_client_if_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        await MusicBot.skip.callback(music_bot, mock_ctx)
        vc.stop.assert_called_once()

    async def test_noop_when_not_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.stop = MagicMock()
        mock_ctx.voice_client = vc
        await MusicBot.skip.callback(music_bot, mock_ctx)
        vc.stop.assert_not_called()


class TestPauseCommand:
    async def test_pauses_when_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.pause.callback(music_bot, mock_ctx)
        mp.pause.assert_awaited_once_with(vc)

    async def test_sends_confirmation_embed_when_paused(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        embed = discord.Embed(title="⏸️ Paused")
        mp.build_pause_confirmation_embed = MagicMock(return_value=embed)
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.pause.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once_with(embed=embed)

    async def test_no_confirmation_sent_when_embed_is_none(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        mp.build_pause_confirmation_embed = MagicMock(return_value=None)
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.pause.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_not_awaited()

    async def test_noop_when_not_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.pause.callback(music_bot, mock_ctx)
        mp.pause.assert_not_awaited()


class TestResumeCommand:
    async def test_resumes_when_paused(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.resume = AsyncMock()
        mp.rehost_np_after_resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.resume.callback(music_bot, mock_ctx)
        mp.resume.assert_awaited_once_with(vc)

    async def test_rehosts_np_block_after_resume(self, music_bot, mock_ctx):
        """If the -pause confirmation hosts the block, resume re-hosts it so
        "⏸️ Paused at…" becomes plain history instead of sitting beneath a
        live, advancing bar (branch review M3)."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.resume = AsyncMock()
        mp.rehost_np_after_resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.resume.callback(music_bot, mock_ctx)
        mp.rehost_np_after_resume.assert_awaited_once()

    async def test_noop_when_not_paused(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=False)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.resume.callback(music_bot, mock_ctx)
        mp.resume.assert_not_awaited()


class TestVolumeCommand:
    async def test_sets_player_volume(self, music_bot, mock_ctx, mock_guild):
        mp = MagicMock()
        mp.store.set_volume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.volume.callback(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        mp.store.set_volume.assert_awaited_once_with(0.5)
        mock_ctx.send.assert_awaited()

    async def test_volume_persists_nothing_without_store(
        self, music_bot, mock_ctx, mock_guild
    ):
        mp = MagicMock()
        mp.store = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.volume.callback(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        mock_ctx.send.assert_awaited()

    async def test_rejects_non_numeric_string(self, music_bot, mock_ctx):
        await MusicBot.volume.callback(music_bot, mock_ctx, "loud")
        mock_ctx.send.assert_awaited()

    async def test_rejects_out_of_range(self, music_bot, mock_ctx):
        await MusicBot.volume.callback(music_bot, mock_ctx, "150")
        mock_ctx.send.assert_awaited()


def _history_entries(n: int) -> list[HistoryEntry]:
    """n entries, oldest-first (the order GuildHistory stores them)."""
    return [
        HistoryEntry(
            title=f"Song {i}",
            webpage_url=f"https://yt.com/v={i}",
            duration_secs=200,
            played_secs=200,
            requester_id=i + 1,
            requester_name=f"user{i}",
            played_at=1000.0 + i,
        )
        for i in range(n)
    ]


def _flags(limit: int = 10):
    """Stand-in for a parsed HistoryFlags (FlagConverter can't be constructed
    directly; the command body only reads .limit)."""
    return SimpleNamespace(limit=limit)


class TestHistoryCommand:
    def _mp_with_history(self, music_bot, entries):
        mp = MagicMock()
        history = GuildHistory(None)
        history.restore(list(reversed(entries)))  # restore takes newest-first
        mp.history = history
        music_bot.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_empty_history_sends_notice(self, music_bot, mock_ctx):
        self._mp_with_history(music_bot, [])
        await MusicBot.history.callback(music_bot, mock_ctx, flags=_flags())
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "No songs have been played yet" in embed.description

    async def test_shows_most_recent_newest_first(self, music_bot, mock_ctx):
        self._mp_with_history(music_bot, _history_entries(15))
        await MusicBot.history.callback(music_bot, mock_ctx, flags=_flags(limit=3))
        mock_ctx.send.assert_awaited_once()
        embeds = mock_ctx.send.call_args[1]["embeds"]
        # Most recent 3 of 15, newest first — not the oldest 3.
        assert [e.title for e in embeds] == [
            "1. Song 14",
            "2. Song 13",
            "3. Song 12",
        ]

    async def test_default_limit_chunks_at_eight_embeds(self, music_bot, mock_ctx):
        # 10 embeds + the ≤2-embed NP block must stay under Discord's 10-embed
        # cap, so the response is chunked 8 + 2, every chunk via ctx.send.
        self._mp_with_history(music_bot, _history_entries(12))
        await MusicBot.history.callback(music_bot, mock_ctx, flags=_flags())
        assert mock_ctx.send.await_count == 2
        first, second = mock_ctx.send.await_args_list
        assert len(first.kwargs["embeds"]) == 8
        assert len(second.kwargs["embeds"]) == 2

    async def test_limit_smaller_than_history_returns_that_many(
        self, music_bot, mock_ctx
    ):
        self._mp_with_history(music_bot, _history_entries(5))
        await MusicBot.history.callback(music_bot, mock_ctx, flags=_flags(limit=50))
        embeds = mock_ctx.send.call_args[1]["embeds"]
        assert len(embeds) == 5

    @pytest.mark.parametrize("bad_limit", [0, -3, 51])
    async def test_out_of_range_limit_rejected(self, music_bot, mock_ctx, bad_limit):
        self._mp_with_history(music_bot, _history_entries(5))
        await MusicBot.history.callback(
            music_bot, mock_ctx, flags=_flags(limit=bad_limit)
        )
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "--limit must be between 1 and 50" in embed.description
        music_bot.get_mp.assert_not_called()

    async def test_song_embeds_carry_thumbnail_and_metadata(self, music_bot, mock_ctx):
        entry = HistoryEntry(
            title="Rich Song",
            webpage_url="https://yt.com/v=rich",
            duration_secs=242,
            played_secs=225,
            requester_id=42,
            requester_name="Omkar",
            thumbnail="https://i.ytimg.com/t.jpg",
            played_at=1752530000.0,
        )
        self._mp_with_history(music_bot, [entry])
        await MusicBot.history.callback(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embeds"][0]
        assert embed.thumbnail.url == "https://i.ytimg.com/t.jpg"
        lines = embed.description.splitlines()
        assert lines[0] == "https://yt.com/v=rich"
        assert lines[1] == "3:45 / 4:02 · requested by <@42> · <t:1752530000:f>"

    def test_flag_defaults(self):
        # -h with no flags must parse to limit=10.
        assert HistoryFlags.get_flags()["limit"].default == 10


class TestPingCommand:
    async def test_sends_embed_with_latency(self, music_bot, mock_ctx):
        await MusicBot.ping.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once()
        call_kwargs = mock_ctx.send.call_args[1]
        assert "embed" in call_kwargs


class TestClearCommand:
    async def test_sends_empty_message_when_queue_already_empty(
        self, music_bot, mock_ctx
    ):
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=[])
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.clear.callback(music_bot, mock_ctx)
        mp.queue_clear.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "The queue is already empty."
        )

    async def test_sends_embed_with_cleared_songs(self, music_bot, mock_ctx):
        cleared = ["Song A - https://yt.com/1", "Song B - https://yt.com/2"]
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=cleared)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.message.add_reaction = AsyncMock()
        await MusicBot.clear.callback(music_bot, mock_ctx)
        mp.queue_clear.assert_awaited_once()
        mock_ctx.message.add_reaction.assert_awaited_once_with("🗑️")
        call_kwargs = mock_ctx.send.call_args[1]
        embed = call_kwargs["embed"]
        assert "2 songs removed" in embed.title
        assert "Song A" in embed.description


def _no_typing():
    """Stub play()'s background_typing wrapper with an inert async CM.

    TestPlayCommand patches asyncio.create_task as a join-task spy; without this
    stub the typing keepalive would also hit the patched create_task, polluting
    call counts and receiving the fake join future. The wrapper itself is
    covered by TestBackgroundTyping."""
    return patch(
        "src.musicbot.background_typing",
        MagicMock(return_value=contextlib.nullcontext()),
    )


def _mock_mp(qsize: int = 0) -> MagicMock:
    """MusicPlayer stand-in for the -play cold path, with the playback-gate
    hooks awaitable: play() takes defer_playback() as an async context manager
    and awaits wait_for_restore() before front-inserting."""
    mp = MagicMock()
    mp.defer_playback = MagicMock(return_value=contextlib.nullcontext())
    mp.wait_for_restore = AsyncMock()
    mp.queue_put_front = AsyncMock()
    mp.queue_put = AsyncMock()
    mp.queue.qsize = MagicMock(return_value=qsize)
    return mp


class TestPlayCommand:
    """Tests for the play() cold-join parallelism (Change A).

    asyncio.Future is used as the join_task stand-in: unlike AsyncMock,
    a Future is directly awaitable via __await__, matching how the real
    asyncio.Task behaves when the code does `await join_task`.
    """

    async def test_cold_join_creates_task_and_awaits_after_queue_source(
        self, music_bot, mock_ctx
    ):
        """join is launched as a task; join_task is awaited after queue_source."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        # Resolved Future: done() is True, await returns immediately.
        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        def fake_create_task(coro):
            coro.close()
            return join_task

        with (
            _no_typing(),
            patch("asyncio.create_task", side_effect=fake_create_task) as mock_create,
        ):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        mock_create.assert_called_once()
        music_bot.queue_source.assert_awaited_once()
        music_bot._enqueue_single.assert_awaited_once()

    async def test_warm_path_skips_join_task(self, music_bot, mock_ctx):
        """When already in voice, no join task is created and queue_source runs directly."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        with _no_typing(), patch("asyncio.create_task") as mock_create:
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        mock_create.assert_not_called()
        music_bot.queue_source.assert_awaited_once()

    async def test_cold_join_cancels_inflight_join_when_queue_source_fails(
        self, music_bot, mock_ctx
    ):
        """queue_source fails while join is still running → join task cancelled, then cleanup()."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = None

        # Pending Future: done() is False; cancel() marks it cancelled so the
        # subsequent `await join_task` in the guard raises CancelledError (suppressed).
        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=MagicMock())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro):
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()  # error embed shown

    async def test_cold_join_cleans_up_when_join_done_before_queue_source_fails(
        self, music_bot, mock_ctx
    ):
        """join completes first, then queue_source fails → cleanup() called (handles ghost connection)."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = MagicMock(
            spec=discord.VoiceClient
        )  # join already established voice

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)  # done() is True
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=MagicMock())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro):
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        cancel_spy.assert_not_called()  # already done, nothing to cancel
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()

    async def test_cold_join_cancels_and_cleans_up_partial_connection(
        self, music_bot, mock_ctx
    ):
        """join in-flight but voice partially established → cancel join task, then cleanup()."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = MagicMock(spec=discord.VoiceClient)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()  # pending, done() is False
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=MagicMock())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro):
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()


class TestPlayFrontInsertion:
    """-play on a disconnected bot means "play this", not "play whatever was
    left over": the requested song jumps ahead of the queue persisted by a
    previous -stop, which resumes behind it. docs/PLAYBACK_GATE_PLAN.md."""

    async def test_cold_path_enqueues_at_front(self, music_bot, mock_ctx):
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro):
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        assert music_bot._enqueue_single.await_args.kwargs["front"] is True

    async def test_warm_path_enqueues_at_back(self, music_bot, mock_ctx):
        """Regression guard: a -play on a connected bot keeps append semantics."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        with _no_typing(), patch("asyncio.create_task"):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        assert music_bot._enqueue_single.await_args.kwargs["front"] is False
        # No hold and no restore wait on the warm path — the gate is already open.
        mp.wait_for_restore.assert_not_awaited()

    async def test_cold_path_waits_for_restore_before_enqueueing(
        self, music_bot, mock_ctx
    ):
        """Load-bearing ordering: put_front LPUSHes the Redis mirror while
        restore_entries replays already-listed entries in memory only, so
        inserting before restore reads its snapshot double-queues the song."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        calls: list[str] = []
        mp = _mock_mp()
        mp.wait_for_restore = AsyncMock(side_effect=lambda: calls.append("restore"))
        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock(
            side_effect=lambda *a, **kw: calls.append("enqueue")
        )
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro):
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        assert calls == ["restore", "enqueue"]

    async def test_cold_path_holds_playback_gate_across_join(self, music_bot, mock_ctx):
        """join opens the gate as soon as the handshake lands — the hold is what
        stops the restored head from starting while queue_source is still
        extracting."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = _mock_mp()
        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro):
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        mp.defer_playback.assert_called_once()

    async def test_front_single_uses_queue_put_front_and_announces_resume(
        self, music_bot, mock_ctx
    ):
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = _mock_mp(qsize=3)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(mock_ctx, qobj, mp, front=True)

        mp.queue_put_front.assert_awaited_once_with(qobj)
        mp.queue_put.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Playing now" in embed.title
        assert "3 songs from the previous queue" in embed.description

    async def test_front_single_omits_resume_line_when_nothing_persisted(
        self, music_bot, mock_ctx
    ):
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = _mock_mp(qsize=0)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(mock_ctx, qobj, mp, front=True)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "resume after" not in embed.description

    async def test_front_playlist_inserts_all_tracks_in_order(
        self, music_bot, mock_ctx
    ):
        """Unlike -playnow (first track only), -play front-inserts a playlist in
        full — nothing is playing here to delay the return of."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = _mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_playlist(mock_ctx, source, tracks, mp, front=True)

        mp.queue_put_front.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()

    async def test_front_insert_after_restore_orders_both_legs(
        self, music_bot, mock_ctx, music_player, fake_redis, mock_author
    ):
        """End to end against a real GuildQueue and fake Redis: the requested
        song leads, the persisted entries follow in their original order, and
        the in-memory and Redis legs agree.

        Also the double-queue regression (docs/PLAYBACK_GATE_PLAN.md §5.8): the
        new song must appear exactly ONCE. put_front LPUSHes it onto the same
        Redis list restore_entries replays from, so an insert sequenced before
        the snapshot read lands in both legs and gets queued twice.
        """
        for title in ("Persisted One", "Persisted Two"):
            await fake_redis.rpush(
                music_player.store.queue_key(),
                orjson.dumps(
                    {
                        "webpage_url": f"https://yt.com/v={title}",
                        "title": title,
                        "requester_id": mock_author.id,
                        "ts": None,
                    }
                ),
            )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        await music_player._restore_state()
        assert music_player.queue.qsize() == 2

        qobj = QueueObject("https://yt.com/v=new", "New Song", mock_author)
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("src.youtube.YTDL.prefetch_stream", new=AsyncMock()):
            await music_bot._enqueue_single(mock_ctx, qobj, music_player, front=True)

        titles = [item.title for item in music_player.queue.display_items()]
        assert titles == ["New Song", "Persisted One", "Persisted Two"]
        assert titles.count("New Song") == 1

        stored = [
            orjson.loads(raw)["title"]
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert stored == ["New Song", "Persisted One", "Persisted Two"]


class TestEnqueueSingle:
    async def test_shows_queued_embed_with_eta_when_song_playing(
        self, music_bot, mock_ctx
    ):
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.estimated_playing_at.assert_called_once()
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Est. playing at **7:42 PM PST**" in embed.description

    async def test_no_queued_embed_when_nothing_playing(self, music_bot, mock_ctx):
        mock_ctx.voice_client = None
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.estimated_playing_at.assert_not_called()
        mock_ctx.send.assert_not_awaited()

    async def test_queued_embed_has_thumbnail_when_present(self, music_bot, mock_ctx):
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject(
            "https://yt.com/v=1",
            "Test Song",
            mock_ctx.author,
            thumbnail="https://img.youtube.com/vi/1/0.jpg",
        )

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert embed.thumbnail.url == "https://img.youtube.com/vi/1/0.jpg"

    async def test_queued_embed_has_no_thumbnail_when_absent(self, music_bot, mock_ctx):
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert embed.thumbnail.url is None


class TestAloneCountdownNotice:
    async def test_notice_routes_through_send_with_np(self, music_bot, mock_guild):
        """The countdown notice can fire mid-song — it must go through
        mp.send_with_np so it can't bury the NP host message."""
        mp = MagicMock()
        mp.send_with_np = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        mock_guild.voice_client = None  # post-sleep check: nothing to disconnect

        with patch("asyncio.sleep", new=AsyncMock()):
            await music_bot._alone_countdown(mock_guild)

        mp.send_with_np.assert_awaited_once()
        embed = mp.send_with_np.call_args.kwargs["embed"]
        assert "disconnect" in embed.description


class TestNowCommand:
    async def test_repins_now_playing_when_playing(
        self, music_bot, mock_ctx, mock_guild
    ):
        """-now re-hosts the live NP block at the bottom of the channel (the
        old host is retired) instead of sending a static snapshot embed."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = mock_ctx.channel  # invoked from the player's home channel
        mp.repin_now_playing = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.now.callback(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()
        mock_ctx.send.assert_not_awaited()

    async def test_repins_live_block_when_paused(self, music_bot, mock_ctx, mock_guild):
        """Design review (2026-07-01): -now while paused used to reply "No songs
        are currently playing." — this is an intentional behavior change, not an
        incidental side effect of making the embed live."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = mock_ctx.channel
        mp.repin_now_playing = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.now.callback(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()

    async def test_cross_channel_sends_static_embed_where_invoked(
        self, music_bot, mock_ctx, mock_guild
    ):
        """-now from a channel other than the player's home channel must
        answer THERE with a static snapshot — the host never leaves home, so
        repinning would leave the invoking channel with no response at all
        (branch review M2)."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = MagicMock()  # distinct from ctx.channel → distinct .id
        static = discord.Embed(title="NP snapshot")
        mp._build_now_playing_embed = MagicMock(return_value=static)
        mp.repin_now_playing = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.now.callback(music_bot, mock_ctx)
        mp.repin_now_playing.assert_not_awaited()
        mp._build_now_playing_embed.assert_called_once_with(mp.current_song)
        mock_ctx.send.assert_awaited_once_with(embed=static)

    async def test_falls_back_when_repin_reports_no_song(
        self, music_bot, mock_ctx, mock_guild
    ):
        """The song can end between the liveness check and the repin —
        repin_now_playing() returns False and -now must still respond
        instead of silently doing nothing (branch review L3)."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = mock_ctx.channel
        mp.play_message = None
        mp.repin_now_playing = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.now.callback(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "No songs are currently playing."
        )

    async def test_sends_not_playing_when_no_song(
        self, music_bot, mock_ctx, mock_guild
    ):
        mock_guild.voice_client = None
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.play_message = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.now.callback(music_bot, mock_ctx)
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "No songs are currently playing."
        )

    async def test_now_reports_nothing_playing_after_song_ends(
        self, music_bot, mock_ctx, mock_guild
    ):
        """After a song finishes, loop() nulls both current_song and
        play_message — the recovery-snapshot elif must not serve the finished
        song's embed as "Now playing"."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.current_song = None
        mp.play_message = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.now.callback(music_bot, mock_ctx)
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "No songs are currently playing."
        )

    async def test_sends_restored_snapshot_during_recovery_window(
        self, music_bot, mock_ctx, mock_guild
    ):
        """current_song isn't live yet (crash-recovery window), but a
        now-playing snapshot survived the restart via play_message."""
        mock_guild.voice_client = None
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.current_song = None
        mp.play_message = discord.Embed(title="Now Playing")
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.now.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once_with(embed=mp.play_message)


class TestOnReady:
    async def test_noop_when_redis_is_none(self, music_bot):
        music_bot.redis = None
        await music_bot.on_ready()  # must not raise, no tasks created

    async def test_creates_restore_task_per_guild(
        self, music_bot_with_redis, mock_guild
    ):
        guilds = list(music_bot_with_redis.bot.guilds)
        stub = stub_create_task()
        passed_guilds = []

        async def _noop():
            pass

        # MusicBot uses __slots__, so we must patch at the class level.
        # Capture happens synchronously in the spy (before stub_create_task
        # closes the coroutine, which would prevent the body from running).
        def _spy(self_inner, guild):
            passed_guilds.append(guild)
            return _noop()

        with (
            patch("asyncio.create_task", stub),
            patch.object(type(music_bot_with_redis), "_restore_guild", _spy),
        ):
            await music_bot_with_redis.on_ready()

        assert stub.call_count == len(guilds)
        assert passed_guilds == guilds


class TestRestoreGuildLock:
    async def test_skips_when_lock_already_held(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(100, 200)
        # Pre-hold the lock so acquire fails
        await fake_redis_bot.set(
            f"lock:guild:{mock_guild.id}:recovery", "1", nx=True, ex=60
        )
        await music_bot_with_redis._restore_guild(mock_guild)
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_restore_creates_player_when_queue_exists(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        from src.redis_client import GuildRedisStore

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

        with patch("src.musicbot.MusicPlayer", return_value=mock_mp):
            await music_bot_with_redis._restore_guild(mock_guild)

        assert mock_guild.id in music_bot_with_redis.mps


class TestAloneCountdown:
    def _make_vc(self, members):
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.members = members
        return vc

    def _setup_mp(self, music_bot, mock_guild) -> MagicMock:
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()
        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp
        return text_channel

    async def test_calls_cleanup_when_still_alone(self, music_bot, mock_guild):
        """After the sleep, if no humans remain, cleanup is called."""
        self._setup_mp(music_bot, mock_guild)

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_user_rejoined(self, music_bot, mock_guild):
        """After the sleep, if a human is present, cleanup is NOT called."""
        self._setup_mp(music_bot, mock_guild)

        human = MagicMock(spec=discord.Member)
        human.bot = False
        mock_guild.voice_client = self._make_vc([human])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_cancelled_before_sleep_skips_cleanup(self, music_bot, mock_guild):
        """CancelledError raised at sleep does not call cleanup."""
        self._setup_mp(music_bot, mock_guild)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_send_failure_does_not_abort_countdown(self, music_bot, mock_guild):
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
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_voice_client_gone(self, music_bot, mock_guild):
        """If the voice client is None when the countdown wakes, cleanup is not called."""
        self._setup_mp(music_bot, mock_guild)

        mock_guild.voice_client = None  # bot already disconnected mid-sleep

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_timer_removed_from_dict_on_completion(self, music_bot, mock_guild):
        """_alone_timers entry is removed in the finally block regardless of outcome."""
        self._setup_mp(music_bot, mock_guild)
        music_bot._alone_timers[mock_guild.id] = MagicMock()  # sentinel

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()):
                await music_bot._alone_countdown(mock_guild)

        assert mock_guild.id not in music_bot._alone_timers


class TestSetup:
    @pytest.fixture(autouse=True)
    def _spotify_env(self, monkeypatch):
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "x")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "y")

    async def test_adds_music_bot_cog(self):
        from src.musicbot import setup

        mock_bot = AsyncMock()
        await setup(mock_bot)
        mock_bot.add_cog.assert_awaited_once()


# ── _restore_guild: Redis-failure gate ────────────────────────────────────────


class TestRestoreGuildStateReadFailed:
    async def test_recovery_skipped_when_state_read_fails(
        self, music_bot_with_redis, mock_guild, fake_redis_bot, caplog
    ):
        """get_recovery_gate() returning None (Redis unavailable) must NOT
        be treated as "nothing to restore": recovery is skipped with a warning
        and no channel resolution or player creation is attempted.
        Distinguishable from the empty-gate case, which also skips but
        silently."""
        from src.redis_client import GuildRedisStore

        with patch.object(
            GuildRedisStore, "get_recovery_gate", new=AsyncMock(return_value=None)
        ):
            with caplog.at_level("WARNING", logger="src.musicbot"):
                await music_bot_with_redis._restore_guild(mock_guild)

        assert "state read failed" in caplog.text
        mock_guild.get_channel.assert_not_called()
        assert mock_guild.id not in music_bot_with_redis.mps

    async def test_recovery_skipped_silently_when_nothing_stored(
        self, music_bot_with_redis, mock_guild, fake_redis_bot, caplog
    ):
        """Empty state hash (zero-value snapshot, no connection) skips recovery
        without the failure warning."""
        with caplog.at_level("WARNING", logger="src.musicbot"):
            await music_bot_with_redis._restore_guild(mock_guild)

        assert "state read failed" not in caplog.text
        mock_guild.get_channel.assert_not_called()
        assert mock_guild.id not in music_bot_with_redis.mps


# ── _restore_guild Gap 3: channel-deleted notification ────────────────────────


class TestRestoreGuildChannelDeleted:
    async def test_clears_connection_when_both_channels_deleted(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """When both stored channels are gone, Redis state is cleared so the
        guild is not retried on the next on_ready."""
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None  # both resolved to None
        mock_guild.system_channel.send = AsyncMock()
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await music_bot_with_redis._restore_guild(mock_guild)

        state = await store.get_guild_state()
        assert state is not None
        assert state.voice_channel_id is None
        assert state.text_channel_id is None
        assert not state.has_active_connection

    async def test_sends_notification_via_system_channel(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """Notification is sent via system_channel when both stored channels are deleted."""
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.send = AsyncMock()
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await music_bot_with_redis._restore_guild(mock_guild)

        mock_guild.system_channel.send.assert_awaited_once()
        msg = mock_guild.system_channel.send.call_args.kwargs["embed"].description
        assert "⚠️" in msg
        assert "voice channel" in msg
        assert "text channel" in msg
        assert "were deleted" in msg

    async def test_falls_back_to_text_channels_when_system_channel_no_perms(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """When system_channel denies send_messages, falls back to guild.text_channels."""
        from src.redis_client import GuildRedisStore

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

        await music_bot_with_redis._restore_guild(mock_guild)

        fallback.send.assert_awaited_once()
        mock_guild.system_channel.send.assert_not_called()

    async def test_notifies_via_text_channel_when_only_voice_deleted(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """When only the voice channel is gone, notify via the still-valid text channel."""
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()

        def _get_channel(ch_id):
            if ch_id == 888000000000000001:
                return None  # voice deleted
            return text_channel  # text still exists

        mock_guild.get_channel.side_effect = _get_channel

        await music_bot_with_redis._restore_guild(mock_guild)

        text_channel.send.assert_awaited_once()
        msg = text_channel.send.call_args.kwargs["embed"].description
        assert "voice channel" in msg
        assert "was deleted" in msg

    async def test_swallows_notify_send_failure(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """A failure sending the notification must not propagate out of _restore_guild."""
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.send = AsyncMock(
            side_effect=Exception("channel gone")
        )
        mock_guild.system_channel.permissions_for.return_value = discord.Permissions(
            send_messages=True
        )

        await music_bot_with_redis._restore_guild(mock_guild)  # must not raise


# ── Queue command ─────────────────────────────────────────────────────────────


class TestQueueCommand:
    async def test_always_sends_embed(self, music_bot, mock_ctx):
        embed = discord.Embed(
            title="Queue", description="Songs: **0**\n\n*The queue is empty.*"
        )
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=embed)
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.queue.callback(music_bot, mock_ctx)

        mock_ctx.send.assert_awaited_once()
        call_kwargs = mock_ctx.send.call_args[1]
        assert "embed" in call_kwargs
        assert call_kwargs["embed"] is embed

    async def test_sends_embed_when_queue_is_empty(self, music_bot, mock_ctx):
        embed = discord.Embed(
            title="Queue", description="Songs: **0**\n\n*The queue is empty.*"
        )
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=embed)
        mp.song_queue = []
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.queue.callback(music_bot, mock_ctx)

        mock_ctx.send.assert_awaited_once()

    async def test_delegates_to_mp_get_queue(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.queue.callback(music_bot, mock_ctx)

        mp.queue_embed.assert_called_once()


# ── Remove command ────────────────────────────────────────────────────────────


class TestRemoveCommand:
    async def test_no_url_sends_usage_message(self, music_bot, mock_ctx):
        await MusicBot.remove.callback(music_bot, mock_ctx, None)

        mock_ctx.send.assert_awaited_once()
        msg = mock_ctx.send.call_args.kwargs["embed"].description
        assert "-remove" in msg

    async def test_no_match_sends_not_found_embed(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[])
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.remove.callback(
            music_bot, mock_ctx, "https://yt.com/watch?v=notfound"
        )

        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "No queued songs found" in embed.description

    async def test_match_sends_removal_embed(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[2])
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.remove.callback(
            music_bot, mock_ctx, "https://yt.com/watch?v=abc"
        )

        calls = mock_ctx.send.await_args_list
        # First call: removal embed
        first_kwargs = calls[0][1]
        assert "embed" in first_kwargs
        removal_embed = first_kwargs["embed"]
        assert "Removed" in removal_embed.title

    async def test_match_sends_updated_queue_embed(self, music_bot, mock_ctx):
        queue_embed = discord.Embed(title="Queue")
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[1])
        mp.queue_embed = MagicMock(return_value=queue_embed)
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.remove.callback(
            music_bot, mock_ctx, "https://yt.com/watch?v=abc"
        )

        calls = mock_ctx.send.await_args_list
        assert len(calls) == 2
        second_kwargs = calls[1][1]
        assert "embed" in second_kwargs
        assert second_kwargs["embed"] is queue_embed

    async def test_match_adds_trash_reaction(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[1])
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.remove.callback(
            music_bot, mock_ctx, "https://yt.com/watch?v=abc"
        )

        mock_ctx.message.add_reaction.assert_awaited_once_with("🗑️")

    async def test_removal_embed_contains_url_field(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[3])
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        url = "https://yt.com/watch?v=abc"

        await MusicBot.remove.callback(music_bot, mock_ctx, url)

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        field_names = [f.name for f in removal_embed.fields]
        assert "URL" in field_names

    async def test_removal_embed_shows_positions(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[1, 4])
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.remove.callback(
            music_bot, mock_ctx, "https://yt.com/watch?v=abc"
        )

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        field_values = [f.value for f in removal_embed.fields]
        assert any("1" in v and "4" in v for v in field_values)

    async def test_removal_embed_color_is_orange(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.queue_remove = AsyncMock(return_value=[1])
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.remove.callback(
            music_bot, mock_ctx, "https://yt.com/watch?v=abc"
        )

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        assert removal_embed.colour == discord.Color.orange()


# ── -playnow ──────────────────────────────────────────────────────────────────


class TestPlaynow:
    @pytest.fixture
    def live_mp(self):
        """A MusicPlayer mock with a song currently playing."""
        from src.musicplayer import InterjectOutcome

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=False,
                replaced=False,
            )
        )
        return mp

    @pytest.fixture
    def live_vc(self):
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return vc

    async def test_idle_delegates_to_play(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        mock_ctx.invoke.assert_awaited_once_with(music_bot.play, url="test")

    async def test_no_voice_client_delegates_to_play(
        self, music_bot, mock_ctx, live_mp
    ):
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = None
        mock_ctx.invoke = AsyncMock()

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        mock_ctx.invoke.assert_awaited_once_with(music_bot.play, url="test")

    async def test_live_song_interjects(self, music_bot, mock_ctx, live_mp, live_vc):
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.content = "-playnow test song"
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        assert qobj.interjected is True
        assert qobj.user_input == "test"
        live_mp.interject.assert_awaited_once_with(qobj, live_vc)
        # Confirmation embed names both songs and the resume position.
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Urgent" in embed.title
        assert "Original Song" in embed.description
        assert "2:31" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_paused_wording(self, music_bot, mock_ctx, live_mp, live_vc):
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=True,
                replaced=False,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "return paused" in embed.description

    async def test_replaced_wording(self, music_bot, mock_ctx, live_mp, live_vc):
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Old Interjection",
                resume_position=None,
                was_paused=False,
                replaced=True,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Replaced" in embed.description
        assert "will not return" in embed.description

    async def test_interject_none_front_enqueues_with_confirmation(
        self, music_bot, mock_ctx, live_mp, live_vc
    ):
        """Song ended mid-resolve: the already-resolved qobj is front-inserted
        directly (the user asked for "now" and the window can be seconds long)
        — NOT by re-invoking -play, which would re-parse, re-resolve, and
        (for playlists) enqueue every track right after the first-track-only
        notice — and the user always gets a confirmation embed."""
        live_mp.interject = AsyncMock(return_value=None)
        live_mp.queue.put_front = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.invoke = AsyncMock()
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        mock_ctx.invoke.assert_not_awaited()
        live_mp.queue.put_front.assert_awaited_once_with([qobj])
        # The interjection marker must not leak onto a normally queued song —
        # a later -playnow would otherwise "replace" it without a resume entry.
        assert qobj.interjected is False
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Playing next" in embed.title
        assert "already ended" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_spotify_playlist_interjects_first_track_only(
        self, music_bot, mock_ctx, live_mp, live_vc
    ):
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        mock_ctx.message.content = f"-playnow {url}"
        music_bot.spotify.playlist = AsyncMock(return_value=["First Song", "Second"])
        qobj = QueueObject("https://yt.com/v=first", "First Song", mock_ctx.author)

        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=qobj)
        ) as ys:
            await MusicBot.playnow.callback(music_bot, mock_ctx, url)

        music_bot.spotify.playlist.assert_awaited_once_with("37i9dQZF1DXcBWIGoYBM5M")
        ys.assert_awaited_once()
        assert ys.call_args.args[1] == "ytsearch:First Song"
        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is qobj
        # First-track notice + confirmation.
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        assert any("first track" in d for d in notices)

    async def test_yt_playlist_interjects_first_track_only(
        self, music_bot, mock_ctx, live_mp, live_vc
    ):
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://www.youtube.com/playlist?list=PLtest123"
        mock_ctx.message.content = f"-playnow {url}"
        first = QueueObject("https://yt.com/v=1", "Track One", mock_ctx.author)
        second = QueueObject("https://yt.com/v=2", "Track Two", mock_ctx.author)

        with patch(
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=[first, second])
        ):
            await MusicBot.playnow.callback(music_bot, mock_ctx, url)

        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is first

    async def test_error_shows_command_error(
        self, music_bot, mock_ctx, live_mp, live_vc
    ):
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

        await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        live_mp.interject.assert_not_awaited()
        mock_ctx.send.assert_awaited()  # error embed

    async def test_warms_stream_cache_before_interjecting(
        self, music_bot, mock_ctx, live_mp, live_vc
    ):
        """The stream-URL cache is warmed BEFORE interject stops the current
        song — a cache miss at dequeue would otherwise put yt-dlp dead air
        between the interrupt and the playnow song starting."""
        from src.musicplayer import InterjectOutcome

        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)

        order: list[str] = []
        prefetch = AsyncMock(side_effect=lambda *a, **k: order.append("prefetch"))
        outcome = InterjectOutcome(
            interrupted_title="Original Song",
            resume_position=151,
            was_paused=False,
            replaced=False,
        )

        def _interject_effect(*args, **kwargs):
            order.append("interject")
            return outcome

        live_mp.interject = AsyncMock(side_effect=_interject_effect)

        with patch("src.musicbot.YTDL.prefetch_stream", new=prefetch):
            await MusicBot.playnow.callback(music_bot, mock_ctx, "test")

        prefetch.assert_awaited_once_with(qobj, redis=music_bot.redis)
        assert order == ["prefetch", "interject"]
