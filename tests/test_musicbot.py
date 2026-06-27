"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""

import asyncio
import time
import orjson
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import fakeredis
import pytest
from discord.ext import commands

from src.musicbot import MusicBot, _check_voice_permissions, _latency_color
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


class TestLatencyColor:
    def test_excellent_latency_is_green(self):
        assert _latency_color(30) == 0x44FF44

    def test_boundary_50ms_is_green(self):
        assert _latency_color(50) == 0x44FF44

    def test_good_latency_is_yellow(self):
        assert _latency_color(75) == 0xFFD000

    def test_boundary_100ms_is_yellow(self):
        assert _latency_color(100) == 0xFFD000

    def test_acceptable_latency_is_orange(self):
        assert _latency_color(150) == 0xFF6600

    def test_boundary_200ms_is_orange(self):
        assert _latency_color(200) == 0xFF6600

    def test_poor_latency_is_red(self):
        assert _latency_color(300) == 0x990000


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
        mp._store = MagicMock()
        mp._store.set_connection = AsyncMock()
        music_bot_with_redis.mps[mock_guild.id] = mp

        # join is a @commands.command — call the underlying callback directly.
        mock_ctx.voice_client = None  # bot not yet in channel
        with (
            patch.object(discord.VoiceChannel, "connect", new=AsyncMock()),
            patch.object(mock_ctx, "invoke", new=AsyncMock()),
        ):
            music_bot_with_redis.get_mp = MagicMock(return_value=mp)
            await MusicBot.join.callback(music_bot_with_redis, mock_ctx)

        mp._store.set_connection.assert_awaited_once_with(
            voice_channel.id, text_channel.id
        )


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
        mp._store = None
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
    def test_sets_bot_attribute(self, mock_bot):
        with patch.dict(
            "os.environ",
            {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"},
        ):
            cog = MusicBot(mock_bot)
        assert cog.bot is mock_bot

    def test_mps_starts_empty(self, mock_bot):
        with patch.dict(
            "os.environ",
            {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"},
        ):
            cog = MusicBot(mock_bot)
        assert cog.mps == {}

    def test_alone_timers_starts_empty(self, mock_bot):
        with patch.dict(
            "os.environ",
            {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"},
        ):
            cog = MusicBot(mock_bot)
        assert cog._alone_timers == {}

    def test_reads_redis_from_bot(self, mock_bot):
        mock_redis = MagicMock()
        mock_bot.redis = mock_redis
        with patch.dict(
            "os.environ",
            {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"},
        ):
            cog = MusicBot(mock_bot)
        assert cog.redis is mock_redis


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
        mp._store = None
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
        mp = self._make_minimal_mp(music_bot, mock_guild, _store=store)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        mp._store.clear_connection.assert_awaited_once()
        mp._store.refresh_ttl.assert_awaited_once()

    async def test_noop_when_guild_not_in_mps(self, music_bot, mock_guild):
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)  # must not raise


class TestCogBeforeInvoke:
    async def test_calls_get_mp(self, music_bot, mock_ctx):
        mock_mp = MagicMock()
        mock_mp._store = None  # skip both redis_set_state and channel-persistence branches
        music_bot.get_mp = MagicMock(return_value=mock_mp)
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_called_once_with(mock_ctx)

    async def test_writes_last_author_id_when_store_present(
        self, music_bot, mock_ctx
    ):
        """When the player has a Redis store, cog_before_invoke persists the author ID."""
        mock_mp = MagicMock()
        mock_mp._store = MagicMock()  # non-None
        mock_mp.redis_set_state = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp)
        mock_ctx.guild.voice_client = None  # skip set_connection branch

        await music_bot.cog_before_invoke(mock_ctx)

        mock_mp.redis_set_state.assert_awaited_once_with(
            "last_author_id", str(mock_ctx.author.id)
        )

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
        mp._store = store
        mp.redis_set_state = AsyncMock()
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
        mp._store = store
        mp.redis_set_state = AsyncMock()
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
        mp._store = store
        mp.redis_set_state = AsyncMock()
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
        vc.pause = MagicMock()
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        await MusicBot.pause.callback(music_bot, mock_ctx)
        vc.pause.assert_called_once()

    async def test_noop_when_not_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.pause = MagicMock()
        mock_ctx.voice_client = vc
        await MusicBot.pause.callback(music_bot, mock_ctx)
        vc.pause.assert_not_called()

    async def test_pause_calls_on_pause_on_store(self, music_bot, mock_ctx):
        """pause command forwards the wall-clock epoch to the player's store."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.pause = MagicMock()
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        mock_store = MagicMock()
        mock_store.on_pause = AsyncMock()
        mock_mp = MagicMock()
        mock_mp._store = mock_store
        music_bot.mps[mock_ctx.guild.id] = mock_mp

        before = time.time()
        await MusicBot.pause.callback(music_bot, mock_ctx)
        after = time.time()

        mock_store.on_pause.assert_awaited_once()
        call_epoch = mock_store.on_pause.call_args[0][0]
        assert before <= call_epoch <= after


class TestResumeCommand:
    async def test_resumes_when_paused(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        vc.resume = MagicMock()
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        await MusicBot.resume.callback(music_bot, mock_ctx)
        vc.resume.assert_called_once()

    async def test_resume_calls_on_resume_on_store(self, music_bot, mock_ctx):
        """resume command forwards the wall-clock epoch to the player's store."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        vc.resume = MagicMock()
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        mock_store = MagicMock()
        mock_store.on_resume = AsyncMock()
        mock_mp = MagicMock()
        mock_mp._store = mock_store
        music_bot.mps[mock_ctx.guild.id] = mock_mp

        before = time.time()
        await MusicBot.resume.callback(music_bot, mock_ctx)
        after = time.time()

        mock_store.on_resume.assert_awaited_once()
        call_epoch = mock_store.on_resume.call_args[0][0]
        assert before <= call_epoch <= after


class TestVolumeCommand:
    async def test_sets_player_volume(self, music_bot, mock_ctx, mock_guild):
        mp = MagicMock()
        mp.redis_set_state = AsyncMock()
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
        mock_ctx.send.assert_awaited_once_with("The queue is already empty.")

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
        music_bot.get_mp = MagicMock(return_value=MagicMock())

        def fake_create_task(coro):
            coro.close()
            return join_task

        with patch("asyncio.create_task", side_effect=fake_create_task) as mock_create:
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
        music_bot.get_mp = MagicMock(return_value=MagicMock())

        with patch("asyncio.create_task") as mock_create:
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

        with patch("asyncio.create_task", side_effect=fake_create_task):
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

        with patch("asyncio.create_task", side_effect=fake_create_task):
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

        with patch("asyncio.create_task", side_effect=fake_create_task):
            await MusicBot.play.callback(music_bot, mock_ctx, "test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()


class TestNowCommand:
    async def test_sends_embed_when_playing(self, music_bot, mock_ctx, mock_guild):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.play_message = discord.Embed(title="Now Playing")
        music_bot.get_mp = MagicMock(return_value=mp)

        await MusicBot.now.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once()

    async def test_sends_not_playing_when_no_song(
        self, music_bot, mock_ctx, mock_guild
    ):
        mock_guild.voice_client = None
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.play_message = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await MusicBot.now.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_with("No songs are currently playing.")


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
    async def test_adds_music_bot_cog(self):
        from src.musicbot import setup

        mock_bot = AsyncMock()
        with patch.dict(
            "os.environ",
            {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"},
        ):
            await setup(mock_bot)
        mock_bot.add_cog.assert_awaited_once()


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

        await music_bot_with_redis._restore_guild(mock_guild)

        vc_id, tc_id = await store.get_connection()
        assert vc_id is None
        assert tc_id is None

    async def test_sends_notification_via_system_channel(
        self, music_bot_with_redis, mock_guild, fake_redis_bot
    ):
        """Notification is sent when both channels are deleted."""
        from src.redis_client import GuildRedisStore

        store = GuildRedisStore(fake_redis_bot, mock_guild.id)
        await store.set_connection(888000000000000001, 888000000000000002)

        mock_guild.get_channel.return_value = None
        mock_guild.system_channel.send = AsyncMock()

        await music_bot_with_redis._restore_guild(mock_guild)

        mock_guild.system_channel.send.assert_awaited_once()
        msg = mock_guild.system_channel.send.call_args[0][0]
        assert "⚠️" in msg
        assert "voice channel" in msg
        assert "text channel" in msg
        assert "were deleted" in msg

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
        msg = text_channel.send.call_args[0][0]
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

        await music_bot_with_redis._restore_guild(mock_guild)  # must not raise
