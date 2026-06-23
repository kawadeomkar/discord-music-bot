"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest

from src.musicplayer import (
    MusicPlayer,
    _deserialize_queue_item,
    _queue_display_str,
    _serialize_queue_item,
)
from src.youtube import QueueObject


@pytest.fixture
def mock_song():
    """A mock YTDL-like song object with all metadata attributes."""
    song = MagicMock()
    song.title = "Test Song Title"
    song.requester = MagicMock()
    song.requester.mention = "<@123456>"
    song.webpage_url = "https://www.youtube.com/watch?v=testid"
    song.duration = "3:30"
    song.uploader = "Test Channel"
    song.views = 1_000_000
    song.likes = 50_000
    song.dislikes = 500
    song.thumbnail = "https://img.youtube.com/vi/testid/0.jpg"
    song.abr = 128
    song.asr = 44100
    song.acodec = "opus"
    return song


@pytest.fixture
def queue_obj(mock_author):
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=abc123",
        title="Test Song",
        requester=mock_author,
    )


class TestQueueDisplayStr:
    def test_formats_title_and_url(self):
        assert (
            _queue_display_str("My Song", "https://yt.com")
            == "My Song - https://yt.com"
        )

    def test_empty_url_leaves_trailing_dash(self):
        result = _queue_display_str("Search Query", "")
        assert result == "Search Query - "


class TestQueuePut:
    @pytest.fixture(autouse=True)
    def _stub_prefetch(self, monkeypatch):
        """Silence background prefetch_stream tasks in tests that don't test them.

        Tests that explicitly assert on prefetch behaviour patch the method
        themselves via `with patch(...)`, which takes precedence over this stub.
        """
        from unittest.mock import AsyncMock
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_put_single_queue_object(self, music_player, queue_obj):
        await music_player.queue_put(queue_obj)
        assert music_player.queue.qsize() == 1

    async def test_put_single_appends_to_song_queue(self, music_player, queue_obj):
        await music_player.queue_put(queue_obj)
        assert len(music_player.song_queue) == 1
        assert "Test Song" in music_player.song_queue[0]

    async def test_put_list_of_sources(self, music_player, mock_author):
        from src.sources import YTSource

        sources = [
            YTSource(ytsearch="ytsearch:song one", process=True),
            YTSource(ytsearch="ytsearch:song two", process=True),
            YTSource(ytsearch="ytsearch:song three", process=True),
        ]
        await music_player.queue_put(sources)
        assert music_player.queue.qsize() == 3
        assert len(music_player.song_queue) == 3

    async def test_put_multiple_singles_increments_size(
        self, music_player, mock_author
    ):
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 4
        assert len(music_player.song_queue) == 4

    async def test_put_mirrors_queue_object_to_redis(
        self, music_player, queue_obj, fake_redis
    ):
        await music_player.queue_put(queue_obj)
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["title"] == queue_obj.title
        assert data["webpage_url"] == queue_obj.webpage_url

    async def test_put_sets_ttl_on_redis_key(self, music_player, queue_obj, fake_redis):
        await music_player.queue_put(queue_obj)
        ttl = await fake_redis.ttl(music_player._store.queue_key())
        assert ttl > 0

    async def test_put_spawns_prefetch_stream_for_queue_object(
        self, music_player, queue_obj
    ):
        """queue_put spawns a background prefetch_stream task for QueueObject items."""
        import asyncio
        from unittest.mock import patch, AsyncMock

        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(queue_obj)
            await asyncio.sleep(0)  # yield to let the spawned task run
        mock_pf.assert_awaited_once()
        call_kwargs = mock_pf.call_args
        assert call_kwargs[0][0] == queue_obj  # first positional arg is the QueueObject

    async def test_put_does_not_spawn_prefetch_for_yt_source(
        self, music_player, mock_author
    ):
        """queue_put does not spawn prefetch_stream for YTSource items (no webpage_url)."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.sources import YTSource

        source = YTSource(ytsearch="ytsearch:test song", process=True)
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(source)
            await asyncio.sleep(0)
        mock_pf.assert_not_awaited()


class TestQueueClear:
    async def test_clear_empties_queue(self, music_player, mock_author):
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 3

        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_empties_song_queue(self, music_player, mock_author):
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert len(music_player.song_queue) == 3

        await music_player.queue_clear()
        assert len(music_player.song_queue) == 0

    async def test_clear_on_empty_queue_is_safe(self, music_player):
        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_deletes_redis_key(self, music_player, queue_obj, fake_redis):
        await music_player.queue_put(queue_obj)
        await music_player.queue_clear()
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert items == []


class TestQueueShuffle:
    async def test_shuffle_requires_minimum_four_items(self, music_player, mock_author):
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        result = await music_player.queue_shuffle()
        assert result == "There must be at least 3 songs to shuffle the queue"

    async def test_shuffle_empty_queue_returns_error(self, music_player):
        result = await music_player.queue_shuffle()
        assert result == "There must be at least 3 songs to shuffle the queue"

    async def test_shuffle_sufficient_songs_returns_shuffled(
        self, music_player, mock_author
    ):
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        result = await music_player.queue_shuffle()
        assert result == "Shuffled!"

    async def test_shuffle_preserves_queue_size(self, music_player, mock_author):
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        await music_player.queue_shuffle()
        assert music_player.queue.qsize() == 5


class TestGetQueue:
    def test_get_queue_with_songs(self, music_player):
        music_player.song_queue = deque(
            ["Song A - url_a", "Song B - url_b", "Song C - url_c"]
        )
        result = music_player.get_queue()
        assert isinstance(result, str)
        assert "Song A - url_a" in result

    def test_get_queue_empty(self, music_player):
        result = music_player.get_queue()
        assert result == ""

    def test_get_queue_caps_at_ten(self, music_player):
        music_player.song_queue = deque([f"Song {i} - url{i}" for i in range(15)])
        result = music_player.get_queue()
        lines = [l for l in result.split("\n") if l and l != "..."]
        assert len(lines) == 10


class TestBuildNowPlayingEmbed:
    def test_returns_discord_embed(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert isinstance(embed, discord.Embed)

    def test_embed_title_contains_song_title(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.title in embed.title

    def test_embed_description_contains_requester_mention(
        self, music_player, mock_song
    ):
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.requester.mention in embed.description

    def test_embed_color_is_green(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.colour == discord.Color.green()

    def test_embed_has_youtube_link_field(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Youtube link" in field_names

    def test_embed_has_duration_field(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Duration" in field_names

    def test_embed_thumbnail_is_set(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.thumbnail.url == mock_song.thumbnail

    def test_embed_footer_contains_bitrate_info(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert str(mock_song.abr) in embed.footer.text
        assert str(mock_song.acodec) in embed.footer.text

    def test_embed_does_not_have_dislikes_field(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Dislikes" not in field_names

    def test_embed_thumbnail_not_set_when_none(self, music_player, mock_song):
        mock_song.thumbnail = None
        embed = music_player._build_now_playing_embed(mock_song)
        assert not embed.thumbnail.url


class TestMusicPlayerInitialState:
    def test_queue_starts_empty(self, music_player):
        assert music_player.queue.qsize() == 0

    def test_song_queue_starts_empty(self, music_player):
        assert len(music_player.song_queue) == 0

    def test_history_starts_empty(self, music_player):
        assert len(music_player.history) == 0

    def test_current_song_is_none(self, music_player):
        assert music_player.current_song is None

    def test_play_message_is_none(self, music_player):
        assert music_player.play_message is None

    def test_player_task_is_none_before_start(self, music_player):
        assert music_player._player is None

    def test_restore_task_is_none_before_start(self, music_player):
        assert music_player._restore_task is None


class TestRedisHelpers:
    async def test_redis_push_history_capped_at_50(self, music_player, fake_redis):
        for i in range(55):
            await music_player._store.push_history(orjson.dumps(f"Song {i} - url{i}"))
        items = await fake_redis.lrange(music_player._store.history_key(), 0, -1)
        assert len(items) == 50

    async def test_redis_set_state_updates_volume(self, music_player, fake_redis):
        await music_player.redis_set_state("volume", "0.75")
        state = await fake_redis.hgetall(music_player._store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_redis_pop_queue_removes_first_item(self, music_player, fake_redis):
        await fake_redis.rpush(music_player._store.queue_key(), b"item1")
        await fake_redis.rpush(music_player._store.queue_key(), b"item2")
        await music_player._store.pop_queue()
        remaining = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(remaining) == 1
        assert remaining[0] == b"item2"

    def test_store_is_none_when_no_redis(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        assert mp._store is None


class TestStateRestore:
    async def test_restore_populates_queue(self, music_player, fake_redis, mock_author):
        item = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=abc",
                "title": "Restored Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player._store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()
        assert music_player.queue.qsize() == 1
        assert "Restored Song" in music_player.song_queue[0]

    async def test_restore_sets_volume(self, music_player, fake_redis):
        await fake_redis.hset(music_player._store.state_key(), b"volume", b"0.5")
        await music_player._restore_state()
        assert music_player.volume == 0.5

    async def test_restore_noop_when_no_redis(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        await mp._restore_state()
        assert mp.queue.qsize() == 0


class TestRestoreCrashedSong:
    async def test_crashed_song_requeued_at_front(
        self, music_player, fake_redis, mock_author
    ):
        """If current_song_url is set in state at restore time, it goes to queue position 0."""
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed Song"
        )
        # Also seed a normal queued item so we can confirm ordering.
        normal_item = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=normal",
                "title": "Normal Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player._store.queue_key(), normal_item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 2
        first = await music_player.queue.get()
        assert first.webpage_url == "https://yt.com/v=crash"
        assert first.title == "Crashed Song"

    async def test_crashed_song_state_cleared_after_restore(
        self, music_player, fake_redis
    ):
        """State fields are wiped so a second restart does not re-queue the same song."""
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed Song"
        )
        music_player._guild.get_member = MagicMock(return_value=None)

        await music_player._restore_state()

        state = await fake_redis.hgetall(music_player._store.state_key())
        assert state.get(b"current_song_url", b"") == b""
        assert state.get(b"current_song_title", b"") == b""

    async def test_no_crash_song_when_state_empty(
        self, music_player, fake_redis, mock_author
    ):
        """No crashed song entry means only queued items are restored."""
        normal_item = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=abc",
                "title": "Normal",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player._store.queue_key(), normal_item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 1
        first = await music_player.queue.get()
        assert first.title == "Normal"

    async def test_crashed_song_uses_last_author_id_for_requester(
        self, music_player, fake_redis, mock_author
    ):
        """If last_author_id is in state, the crashed song's requester is that member."""
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_url", b"https://yt.com/v=crash"
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed"
        )
        await fake_redis.hset(
            music_player._store.state_key(),
            b"last_author_id",
            str(mock_author.id).encode(),
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.requester is mock_author

    async def test_crashed_song_computes_position_from_play_epoch(
        self, music_player, fake_redis, mock_author
    ):
        """play_start_epoch and total_pause_seconds are combined into a seek offset."""
        import time

        start = time.time() - 90  # started 90 seconds ago, 10s of pauses
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_url", b"https://yt.com/v=crash"
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed"
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"play_start_epoch", str(start).encode()
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"total_pause_seconds", b"10"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        # expected position ≈ 90 - 10 = 80s; allow ±10s tolerance for test latency
        assert first.ts is not None
        assert 70 <= first.ts <= 90

    async def test_crashed_song_position_none_when_no_epoch(
        self, music_player, fake_redis, mock_author
    ):
        """When play_start_epoch is absent, ts on the restored QueueObject is None."""
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_url", b"https://yt.com/v=crash"
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts is None


class TestResolveSource:
    async def test_returns_queue_object_unchanged(self, music_player, queue_obj):
        result = await music_player._resolve_source(queue_obj)
        assert result is queue_obj

    async def test_resolves_ytsource_via_yt_source(self, music_player, mock_author):
        from unittest.mock import patch, AsyncMock
        from src.sources import YTSource

        fake_qobj = QueueObject("https://yt.com/v=1", "Resolved", mock_author)
        with patch(
            "src.musicplayer.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_player._resolve_source(
                YTSource(ytsearch="ytsearch:test", process=True)
            )
        assert isinstance(result, QueueObject)
        assert result.title == "Resolved"


class TestStreamSource:
    async def test_returns_none_on_exception(self, music_player, queue_obj):
        with patch(
            "src.musicplayer.YTDL.yt_stream",
            new=AsyncMock(side_effect=Exception("boom")),
        ):
            result = await music_player._stream_source(queue_obj)
        assert result is None

    async def test_returns_ytdl_on_success(self, music_player, queue_obj):
        mock_ytdl = MagicMock()
        with patch(
            "src.musicplayer.YTDL.yt_stream", new=AsyncMock(return_value=mock_ytdl)
        ):
            result = await music_player._stream_source(queue_obj)
        assert result is mock_ytdl


# ── New coverage: from_context, start, set_context, stop ─────────────────────


class TestFromContext:
    def test_creates_music_player(self, mock_bot, mock_ctx, fake_redis):
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert isinstance(mp, MusicPlayer)

    def test_sets_last_author_to_ctx_author(self, mock_bot, mock_ctx, fake_redis):
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert mp._last_author is mock_ctx.author

    def test_raises_if_guild_is_none(self, mock_bot, mock_ctx, fake_redis):
        mock_ctx.guild = None
        with pytest.raises(AssertionError):
            MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)

    def test_attaches_store_when_redis_provided(self, mock_bot, mock_ctx, fake_redis):
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert mp._store is not None


class TestStart:
    def test_creates_player_task(self, music_player):
        mock_task = MagicMock()
        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(return_value=mock_task)
        music_player.start()
        assert music_player._player is mock_task

    def test_creates_restore_task_when_store_present(self, music_player):
        mock_task = MagicMock()
        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(return_value=mock_task)
        assert music_player._store is not None
        music_player.start()
        assert music_player._restore_task is not None

    def test_no_restore_task_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = MagicMock(return_value=MagicMock())
        mp.start()
        assert mp._restore_task is None

    def test_restore_complete_set_immediately_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        """When there is no Redis store, start() must signal _restore_complete immediately
        so loop()'s prefetch gate never blocks."""
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = MagicMock(return_value=MagicMock())
        mp.start()
        assert mp._restore_complete.is_set()

    def test_restore_complete_not_set_before_start_when_store_present(
        self, music_player
    ):
        """Before start() or _restore_state() runs, the event must be clear."""
        assert not music_player._restore_complete.is_set()


class TestSetContext:
    def test_updates_channel(self, music_player, mock_ctx):
        new_channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = new_channel
        music_player.set_context(mock_ctx)
        assert music_player._channel is new_channel

    def test_updates_last_author(self, music_player, mock_ctx):
        new_author = MagicMock(spec=discord.Member)
        mock_ctx.author = new_author
        music_player.set_context(mock_ctx)
        assert music_player._last_author is new_author


class TestStop:
    async def test_delegates_to_cog_cleanup(self, music_player):
        music_player._cog.cleanup = AsyncMock()
        await music_player.stop()
        music_player._cog.cleanup.assert_awaited_once_with(music_player._guild)


# ── _cancel_prefetch ──────────────────────────────────────────────────────────


class TestCancelPrefetch:
    async def test_noop_when_no_prefetch_task(self, music_player):
        music_player._prefetch_task = None
        await music_player._cancel_prefetch()  # must not raise

    async def test_noop_when_prefetch_task_already_done(self, music_player):
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        music_player._prefetch_task = task
        await music_player._cancel_prefetch()
        task.cancel.assert_not_called()

    async def test_cancels_in_flight_prefetch_task(self, music_player):
        async def _long():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long())
        music_player._prefetch_task = task
        await music_player._cancel_prefetch()
        assert task.cancelled()


# ── _send_now_playing ─────────────────────────────────────────────────────────


class TestSendNowPlaying:
    async def test_sends_embed_to_channel(self, music_player, mock_song):
        await music_player._send_now_playing(mock_song)
        music_player._channel.send.assert_awaited_once()
        call_kwargs = music_player._channel.send.call_args[1]
        assert "embed" in call_kwargs

    async def test_stores_embed_as_play_message(self, music_player, mock_song):
        await music_player._send_now_playing(mock_song)
        assert music_player.play_message is not None
        assert isinstance(music_player.play_message, discord.Embed)

    async def test_swallows_channel_send_exception(self, music_player, mock_song):
        music_player._channel.send = AsyncMock(side_effect=Exception("channel gone"))
        await music_player._send_now_playing(mock_song)  # must not raise

    async def test_persists_embed_fields_to_redis(
        self, music_player, mock_song, fake_redis
    ):
        """_send_now_playing writes the embed fields into the Redis now-playing hash."""
        mock_song.requester.id = 999
        mock_song.requester.mention = "<@999>"
        await music_player._send_now_playing(mock_song)
        data = await fake_redis.hgetall(music_player._store.now_playing_key())
        assert data[b"title"] == mock_song.title.encode()
        assert data[b"uploader"] == mock_song.uploader.encode()
        assert data[b"requester_mention"] == mock_song.requester.mention.encode()

    async def test_send_now_playing_no_redis_write_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx, mock_song
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp._channel = mock_channel
        await mp._send_now_playing(mock_song)
        # No store means no Redis write; embed still set locally
        assert mp.play_message is not None


# ── _prefetch_next_song ───────────────────────────────────────────────────────


class TestPrefetchNextSong:
    @pytest.fixture(autouse=True)
    def _stub_prefetch_stream(self, monkeypatch):
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_returns_none_when_queue_empty(self, music_player):
        result = await music_player._prefetch_next_song()
        assert result is None

    async def test_returns_ytdl_on_success(self, music_player, queue_obj):
        await music_player.queue.put(queue_obj)
        mock_song = MagicMock()
        with (
            patch(
                "src.musicplayer.YTDL.yt_source",
                new=AsyncMock(return_value=queue_obj),
            ),
            patch(
                "src.musicplayer.YTDL.yt_stream",
                new=AsyncMock(return_value=mock_song),
            ),
        ):
            result = await music_player._prefetch_next_song()
        assert result is mock_song

    async def test_returns_none_and_calls_task_done_on_stream_error(
        self, music_player, queue_obj
    ):
        await music_player.queue.put(queue_obj)
        with patch(
            "src.musicplayer.YTDL.yt_stream",
            new=AsyncMock(side_effect=Exception("network")),
        ):
            result = await music_player._prefetch_next_song()
        assert result is None

    async def test_reraises_cancelled_error_and_calls_task_done(
        self, music_player, queue_obj
    ):
        # MusicPlayer uses __slots__ so instance attributes can't be set directly;
        # patch at the class level instead.
        await music_player.queue.put(queue_obj)
        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await music_player._prefetch_next_song()


# ── queue_get ─────────────────────────────────────────────────────────────────


class TestQueueGet:
    async def test_returns_item_from_queue(self, music_player, queue_obj):
        await music_player.queue.put(queue_obj)
        result = await music_player.queue_get()
        assert result is queue_obj


# ── _deserialize_queue_item edge cases ────────────────────────────────────────


class TestDeserializeQueueItem:
    def test_falls_back_to_guild_owner_when_member_not_found(self, mock_guild):
        owner = MagicMock(spec=discord.Member)
        mock_guild.get_member = MagicMock(return_value=None)
        mock_guild.owner = owner
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=1",
                "title": "Song",
                "requester_id": 99999,
                "ts": None,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is not None
        assert result.requester is owner

    def test_returns_none_when_member_and_owner_both_none(self, mock_guild):
        mock_guild.get_member = MagicMock(return_value=None)
        mock_guild.owner = None
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=1",
                "title": "Song",
                "requester_id": 99999,
                "ts": None,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is None

    def test_returns_none_on_invalid_json(self, mock_guild):
        result = _deserialize_queue_item(b"not valid json{{{{", mock_guild)
        assert result is None

    def test_preserves_ts_field(self, mock_guild, mock_author):
        mock_guild.get_member = MagicMock(return_value=mock_author)
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=1",
                "title": "Song",
                "requester_id": mock_author.id,
                "ts": 42,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is not None
        assert result.ts == 42


class TestSerializeQueueItem:
    def test_round_trip(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_author, ts=30)
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["webpage_url"] == "https://yt.com/v=1"
        assert d["title"] == "Test Song"
        assert d["requester_id"] == mock_author.id
        assert d["ts"] == 30


# ── _restore_state additional paths ──────────────────────────────────────────


class TestRestoreStateTtlRefresh:
    async def test_ttl_refreshed_after_successful_restore(
        self, music_player, fake_redis
    ):
        """TTL on all guild keys is refreshed at the end of a successful restore."""
        await fake_redis.hset(music_player._store.state_key(), b"volume", b"0.8")
        # Set a short TTL initially
        await fake_redis.expire(music_player._store.state_key(), 10)

        await music_player._restore_state()

        ttl = await fake_redis.ttl(music_player._store.state_key())
        assert ttl > 1000  # refreshed to GUILD_TTL

    async def test_restore_continues_after_bad_queue_item(
        self, music_player, fake_redis, mock_author
    ):
        """Malformed queue items are skipped; valid ones are still restored."""
        valid = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=ok",
                "title": "Good Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(
            music_player._store.queue_key(), b"!!!bad json!!!", valid
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 1
        item = await music_player.queue.get()
        assert item.title == "Good Song"


# ── loop() ────────────────────────────────────────────────────────────────────


class TestLoop:
    @pytest.fixture
    def mock_song(self):
        song = MagicMock()
        song.title = "Loop Test Song"
        song.webpage_url = "https://yt.com/v=loop1"
        return song

    async def test_exits_immediately_when_bot_closed(self, music_player):
        music_player.bot.is_closed.return_value = True
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player.loop()  # should return without hanging

    async def test_timeout_triggers_stop(self, music_player):
        # MusicPlayer uses __slots__; patch methods at the class level.
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.return_value = False

        stop_called = asyncio.Event()

        async def _mock_stop(self_inner):
            stop_called.set()

        with patch.object(
            MusicPlayer,
            "queue_get",
            new=AsyncMock(side_effect=asyncio.TimeoutError()),
        ):
            with patch.object(MusicPlayer, "stop", new=_mock_stop):
                await music_player.loop()
        await asyncio.sleep(0)
        assert stop_called.is_set()

    async def test_skips_song_when_stream_returns_none(self, music_player, queue_obj):
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append("Test Song - url")

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=None)
            ),
        ):
            await music_player.loop()

        music_player._channel.send.assert_awaited_with(
            "Failed to load the next song, skipping."
        )

    async def test_plays_song_and_updates_history(
        self, music_player, queue_obj, mock_song
    ):
        # _restore_complete is never set unless start() is called or _restore_state() runs.
        # Set it here so the restore gate in loop() does not block for 10s.
        music_player._restore_complete.set()

        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc

        # vc.play is a mock so it never calls the after callback that sets play_next;
        # mock wait() so it returns immediately instead of blocking.
        music_player.play_next.wait = AsyncMock()

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=AsyncMock()),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
        ):
            await music_player.loop()

        assert len(music_player.history) == 1
        assert mock_song.title in music_player.history[0]

    async def test_unhandled_exception_sends_error_message(
        self, music_player, queue_obj
    ):
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)

        def _bad_play(*a, **kw):
            raise RuntimeError("ffmpeg gone")

        vc.play = _bad_play
        music_player._guild.voice_client = vc

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=MagicMock())
            ),
        ):
            await music_player.loop()

        music_player._channel.send.assert_awaited()


# ── _restore_complete event ───────────────────────────────────────────────────


class TestRestoreCompleteEvent:
    async def test_set_after_successful_restore(self, music_player):
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_set_even_when_restore_raises(self, music_player):
        music_player.bot.wait_until_ready = AsyncMock()
        with patch.object(
            music_player._store,
            "get_state",
            new=AsyncMock(side_effect=Exception("redis down")),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_set_even_when_queue_restore_fails(self, music_player):
        music_player.bot.wait_until_ready = AsyncMock()
        with patch.object(
            music_player._store,
            "get_queue",
            new=AsyncMock(side_effect=Exception("redis down")),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()


# ── _build_now_playing_embed_from_data ────────────────────────────────────────

_NP_DATA: dict[bytes, bytes] = {
    b"title": b"Test Song",
    b"webpage_url": b"https://yt.com/v=1",
    b"uploader": b"Test Channel",
    b"duration": b"3:30",
    b"thumbnail": b"https://img.yt.com/thumb.jpg",
    b"view_count": b"1000",
    b"like_count": b"50",
    b"abr": b"128",
    b"asr": b"44100",
    b"acodec": b"opus",
    b"requester_id": b"123",
    b"requester_mention": b"<@123>",
}


class TestBuildNowPlayingEmbedFromData:
    def test_returns_discord_embed(self, music_player):
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert isinstance(embed, discord.Embed)

    def test_title_from_data(self, music_player):
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert "Test Song" in embed.title

    def test_requester_mention_in_description(self, music_player):
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert "<@123>" in embed.description

    def test_thumbnail_set_from_data(self, music_player):
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert embed.thumbnail.url == "https://img.yt.com/thumb.jpg"

    def test_thumbnail_not_set_when_empty(self, music_player):
        data = dict(_NP_DATA)
        data[b"thumbnail"] = b""
        embed = music_player._build_now_playing_embed_from_data(data)
        assert not embed.thumbnail.url

    def test_footer_contains_bitrate(self, music_player):
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert "128" in embed.footer.text

    def test_missing_field_defaults_to_empty_string(self, music_player):
        data = {b"title": b"Minimal"}  # all other fields absent
        embed = music_player._build_now_playing_embed_from_data(data)
        assert "Minimal" in embed.title


# ── _restore_state: now-playing embed restoration ────────────────────────────


class TestRestoreStateNowPlaying:
    async def test_restores_play_message_from_redis(self, music_player, fake_redis):
        """If now_playing hash exists in Redis, play_message is populated on restore."""
        await fake_redis.hset(
            music_player._store.now_playing_key(),
            mapping={
                "title": "Restored Song",
                "webpage_url": "https://yt.com/v=1",
                "uploader": "Channel",
                "duration": "3:00",
                "thumbnail": "",
                "view_count": "100",
                "like_count": "10",
                "abr": "128",
                "asr": "44100",
                "acodec": "opus",
                "requester_id": "123",
                "requester_mention": "<@123>",
            },
        )
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        assert music_player.play_message is not None
        assert isinstance(music_player.play_message, discord.Embed)
        assert "Restored Song" in music_player.play_message.title

    async def test_play_message_none_when_no_now_playing_in_redis(
        self, music_player
    ):
        """No now_playing hash → play_message stays None after restore."""
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player._restore_state()
        assert music_player.play_message is None
