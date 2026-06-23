"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""

import asyncio
import orjson
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import fakeredis
import pytest
from discord.ext import commands

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
    cog._active_spans = {}
    cog._alone_timers = {}
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
        with patch.object(
            music_bot_with_redis, "cleanup", new=AsyncMock()
        ) as mock_cleanup:
            await music_bot_with_redis.on_voice_state_update(member, before, after)
        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_bot_moved_cancels_stale_alone_timer(
        self, music_bot_with_redis, mock_guild
    ):
        """Bot moved to a new channel (not ejected) cancels any running alone-timer."""
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user

        timer = MagicMock(spec=asyncio.Task)
        timer.done.return_value = False
        timer.cancel = MagicMock()
        music_bot_with_redis._alone_timers[mock_guild.id] = timer

        member = MagicMock(spec=discord.Member)
        member.id = mock_bot_user.id
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
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user
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
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user
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
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user
        music_bot_with_redis.mps[mock_guild.id] = MagicMock()

        timer = MagicMock(spec=asyncio.Task)
        timer.done.return_value = False
        timer.cancel = MagicMock()
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

    async def test_member_change_in_unrelated_channel_ignored(
        self, music_bot_with_redis, mock_guild
    ):
        """Member moving between two channels that aren't the bot's channel → no timer action."""
        mock_bot_user = MagicMock()
        mock_bot_user.id = 999999999999999999
        music_bot_with_redis.bot.user = mock_bot_user
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
    async def test_disconnects_voice_client(self, music_bot, mock_guild):
        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        mp._store = None
        music_bot.mps[mock_guild.id] = mp
        mock_guild.voice_client.disconnect = AsyncMock()
        await music_bot.cleanup(mock_guild)
        mock_guild.voice_client.disconnect.assert_awaited_once()

    async def test_removes_guild_from_mps(self, music_bot, mock_guild):
        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        mp._store = None
        music_bot.mps[mock_guild.id] = mp
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        assert mock_guild.id not in music_bot.mps

    async def test_cancels_in_flight_prefetch_task(self, music_bot, mock_guild):
        mp = MagicMock()
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        mp._prefetch_task = task
        mp._restore_task = None
        mp._player = None
        mp._store = None
        music_bot.mps[mock_guild.id] = mp
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_cancels_running_alone_timer(self, music_bot, mock_guild):
        timer = MagicMock(spec=asyncio.Task)
        timer.done.return_value = False
        timer.cancel = MagicMock()
        music_bot._alone_timers[mock_guild.id] = timer

        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        mp._store = None
        music_bot.mps[mock_guild.id] = mp
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
        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        mp._store = MagicMock()
        mp._store.clear_connection = AsyncMock()
        mp._store.refresh_ttl = AsyncMock()
        music_bot.mps[mock_guild.id] = mp
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        mp._store.clear_connection.assert_awaited_once()
        mp._store.refresh_ttl.assert_awaited_once()

    async def test_noop_when_guild_not_in_mps(self, music_bot, mock_guild):
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)  # must not raise


class TestCogBeforeInvoke:
    async def test_calls_get_mp(self, music_bot, mock_ctx):
        mp = MagicMock()
        mp._store = None  # skip the channel-persistence branch
        music_bot.get_mp = MagicMock(return_value=mp)
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
        mp._store = store
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
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        mock_guild.voice_client = None  # not connected yet

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_not_awaited()


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
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.skip.callback(music_bot, mock_ctx)
        vc.stop.assert_called_once()

    async def test_noop_when_not_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.stop = MagicMock()
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.skip.callback(music_bot, mock_ctx)
        vc.stop.assert_not_called()


class TestPauseCommand:
    async def test_pauses_when_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.pause = MagicMock()
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.pause.callback(music_bot, mock_ctx)
        vc.pause.assert_called_once()

    async def test_noop_when_not_playing(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.pause = MagicMock()
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.pause.callback(music_bot, mock_ctx)
        vc.pause.assert_not_called()


class TestResumeCommand:
    async def test_resumes_when_paused(self, music_bot, mock_ctx):
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        vc.resume = MagicMock()
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("discord.utils.get", return_value=vc):
            await MusicBot.resume.callback(music_bot, mock_ctx)
        vc.resume.assert_called_once()


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
    async def test_sends_in_development_message(self, music_bot, mock_ctx):
        await MusicBot.clear.callback(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once()
        msg = mock_ctx.send.call_args[0][0]
        assert "development" in msg.lower()


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
        created = []
        with patch("asyncio.create_task", side_effect=lambda c: created.append(c)):
            await music_bot_with_redis.on_ready()
        assert len(created) == len(music_bot_with_redis.bot.guilds)


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

    async def test_calls_cleanup_when_still_alone(self, music_bot, mock_guild):
        """After the sleep, if no humans remain, cleanup is called."""
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()

        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_skips_cleanup_when_user_rejoined(self, music_bot, mock_guild):
        """After the sleep, if a human is present, cleanup is NOT called."""
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()

        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp

        human = MagicMock(spec=discord.Member)
        human.bot = False
        mock_guild.voice_client = self._make_vc([human])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_cancelled_before_sleep_skips_cleanup(self, music_bot, mock_guild):
        """CancelledError raised at sleep does not call cleanup."""
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()

        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp

        with patch("asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_not_awaited()

    async def test_send_failure_does_not_abort_countdown(self, music_bot, mock_guild):
        """A failed text_channel.send is swallowed; the countdown still fires cleanup."""
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "forbidden")
        )

        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp

        bot_member = MagicMock(spec=discord.Member)
        bot_member.bot = True
        mock_guild.voice_client = self._make_vc([bot_member])

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch.object(music_bot, "cleanup", new=AsyncMock()) as mock_cleanup:
                await music_bot._alone_countdown(mock_guild)

        mock_cleanup.assert_awaited_once_with(mock_guild)

    async def test_timer_removed_from_dict_on_completion(self, music_bot, mock_guild):
        """_alone_timers entry is removed in the finally block regardless of outcome."""
        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.send = AsyncMock()

        mp = MagicMock()
        mp._channel = text_channel
        music_bot.mps[mock_guild.id] = mp
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
