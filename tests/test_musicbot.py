"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""
import orjson
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import fakeredis
import pytest

from src.musicbot import MusicBot, _check_voice_permissions, _latency_color
from src.sources import SpotifySource, YTSource
from src.youtube import QueueObject


@pytest.fixture
def music_bot(mock_bot):
    """Minimal MusicBot instance bypassing __init__ Discord registration."""
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog.redis = None
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
        source = SpotifySource(type="playlist", id="pid123")
        music_bot.spotify.playlist = AsyncMock(return_value=["Song A", "Song B"])
        result = await music_bot.queue_source(mock_ctx, source)
        assert result == ["Song A", "Song B"]

    async def test_spotify_track_calls_yt_source(self, music_bot, mock_ctx):
        source = SpotifySource(type="track", id="tid123")
        fake_qobj = QueueObject("https://yt.com/v=1", "My Track", mock_ctx.author)
        music_bot.spotify.track = AsyncMock(return_value="My Track Artist")
        with patch("src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)):
            result = await music_bot.queue_source(mock_ctx, source)
        assert isinstance(result, QueueObject)

    async def test_youtube_url_calls_yt_source(self, music_bot, mock_ctx):
        source = YTSource(url="https://yt.com/watch?v=abc", process=False)
        fake_qobj = QueueObject("https://yt.com/watch?v=abc", "YT Song", mock_ctx.author)
        with patch("src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)):
            result = await music_bot.queue_source(mock_ctx, source)
        assert isinstance(result, QueueObject)

    async def test_youtube_search_uses_ytsearch(self, music_bot, mock_ctx):
        source = YTSource(ytsearch="ytsearch:test song", process=True)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        with patch("src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)) as mock_yt:
            await music_bot.queue_source(mock_ctx, source)
        call_args = mock_yt.call_args
        assert call_args[0][1] == "ytsearch:test song"


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
        with (
            patch("discord.utils.get", return_value=None),
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
    async def test_bot_disconnect_triggers_cleanup(
        self, music_bot_with_redis, mock_guild
    ):
        """on_voice_state_update cleans up when the bot itself leaves a channel."""
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user

        mp = MagicMock()
        mp._store = None
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        music_bot_with_redis.mps[mock_guild.id] = mp

        member = MagicMock(spec=discord.Member)
        member.id = mock_bot_user.id
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()  # was in a channel
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None  # now disconnected

        mock_guild.voice_client = None
        with patch.object(music_bot_with_redis, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot_with_redis.on_voice_state_update(member, before, after)
        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_other_member_disconnect_ignored(self, music_bot_with_redis, mock_guild):
        """on_voice_state_update does nothing when a non-bot member disconnects."""
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user

        member = MagicMock(spec=discord.Member)
        member.id = 123456789  # different from bot
        member.guild = mock_guild
        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None

        with patch.object(music_bot_with_redis, "cleanup", new=AsyncMock()) as mock_cleanup:
            await music_bot_with_redis.on_voice_state_update(member, before, after)
        mock_cleanup.assert_not_called()
