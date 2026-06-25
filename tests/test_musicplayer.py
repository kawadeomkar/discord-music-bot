"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import asyncio
import time
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
from tests.helpers import stub_create_task


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
    song.duration_secs = 210
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


@pytest.fixture()
def _stub_queue_put_tasks(monkeypatch):
    """Prevent prefetch_stream tasks in queue_put from doing real yt-dlp work."""
    from src import youtube

    monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())


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

    async def test_put_with_prefetch_false_skips_prefetch_task(
        self, music_player, queue_obj
    ):
        """queue_put(prefetch=False) never spawns a background prefetch_stream task."""
        from unittest.mock import patch, AsyncMock

        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(queue_obj, prefetch=False)
            await asyncio.sleep(0)
        mock_pf.assert_not_awaited()


class TestQueueClear:
    @pytest.fixture(autouse=True)
    def _setup(self, _stub_queue_put_tasks):
        pass

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

    async def test_clear_returns_list_of_cleared_display_strings(
        self, music_player, mock_author
    ):
        """queue_clear() returns the song_queue display strings for the cleared songs."""
        qobjs = [
            QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            for i in range(3)
        ]
        for q in qobjs:
            await music_player.queue_put(q)
        cleared = await music_player.queue_clear()
        assert len(cleared) == 3
        assert all("Song" in s for s in cleared)

    async def test_clear_returns_empty_list_when_queue_was_empty(self, music_player):
        cleared = await music_player.queue_clear()
        assert cleared == []


class TestQueueShuffle:
    @pytest.fixture(autouse=True)
    def _setup(self, _stub_queue_put_tasks):
        pass

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


class TestUpdateActivity:
    async def test_sets_playing_activity_when_song_playing(
        self, music_player, mock_song
    ):
        music_player.bot.change_presence = AsyncMock()
        await music_player.update_activity(mock_song)
        music_player.bot.change_presence.assert_awaited_once()
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert isinstance(activity, discord.Activity)
        assert activity.type == discord.ActivityType.listening
        # name encodes uploader as suffix since bot activities only render name
        assert activity.name == f"{mock_song.title} · {mock_song.uploader}"
        assert activity.state == mock_song.duration
        assert activity.state_url == mock_song.webpage_url
        assert "start" in activity.timestamps
        now_ms = int(time.time() * 1000)
        assert activity.timestamps["start"] <= now_ms
        assert activity.timestamps["start"] >= now_ms - 2000
        assert "end" in activity.timestamps
        assert (
            abs(
                activity.timestamps["end"]
                - (activity.timestamps["start"] + mock_song.duration_secs * 1000)
            )
            < 1000
        )

    async def test_omits_end_timestamp_when_duration_unknown(
        self, music_player, mock_song
    ):
        music_player.bot.change_presence = AsyncMock()
        mock_song.duration_secs = 0
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert "start" in activity.timestamps
        assert "end" not in activity.timestamps

    async def test_truncates_name_to_128_chars(self, music_player, mock_song):
        music_player.bot.change_presence = AsyncMock()
        mock_song.title = "A" * 125
        mock_song.uploader = "B"
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert len(activity.name) == 128
        assert activity.name.endswith("…")

    async def test_resets_to_game_activity_when_idle(self, music_player):
        music_player.bot.change_presence = AsyncMock()
        music_player.bot.voice_clients = []
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_awaited_once()
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert isinstance(activity, discord.Game)
        assert activity.name == "music"

    async def test_skips_reset_when_another_guild_is_playing(self, music_player):
        music_player.bot.change_presence = AsyncMock()
        active_vc = MagicMock(spec=discord.VoiceClient)
        active_vc.is_playing.return_value = True
        music_player.bot.voice_clients = [active_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_not_awaited()

    async def test_resets_when_voice_clients_present_but_not_playing(
        self, music_player
    ):
        music_player.bot.change_presence = AsyncMock()
        idle_vc = MagicMock(spec=discord.VoiceClient)
        idle_vc.is_playing.return_value = False
        music_player.bot.voice_clients = [idle_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_awaited_once()

    async def test_falls_back_to_a_song_when_title_is_none(
        self, music_player, mock_song
    ):
        music_player.bot.change_presence = AsyncMock()
        mock_song.title = None
        mock_song.uploader = None
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.name == "a song"

    async def test_swallows_change_presence_exception(self, music_player, mock_song):
        music_player.bot.change_presence = AsyncMock(
            side_effect=Exception("rate limited")
        )
        # Must not raise — playback loop must not be interrupted by a presence failure
        await music_player.update_activity(mock_song)


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
    def test_start_creates_player_and_restore_tasks(self, music_player):
        player_task = MagicMock(name="player_task")
        restore_task = MagicMock(name="restore_task")
        returns = [player_task, restore_task]

        def _create(coro):
            coro.close()
            return returns.pop(0)

        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(side_effect=_create)
        assert music_player._store is not None
        music_player.start()

        assert music_player._player is player_task
        assert music_player._restore_task is restore_task

    def test_no_restore_task_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = stub_create_task()
        mp.start()
        assert mp._player is not None
        assert mp._restore_task is None


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
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
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

    async def test_update_activity_called_at_song_start_and_end(
        self, music_player, queue_obj, mock_song
    ):
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
        music_player.play_next.wait = AsyncMock()

        activity_mock = AsyncMock()

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
            patch.object(MusicPlayer, "update_activity", activity_mock),
        ):
            await music_player.loop()

        assert activity_mock.await_count == 2
        assert activity_mock.call_args_list[0].args[0] is mock_song
        assert activity_mock.call_args_list[1].args[0] is None

    async def test_prefetched_song_cleaned_up_when_queue_was_cleared(
        self, music_player, queue_obj, mock_song
    ):
        """When _queue_cleared is set while a prefetch is in-flight, the loop
        discards the prefetched song and calls cleanup() so the FFmpeg subprocess
        is not leaked.

        Flow:
          Iteration 1 — song 1 plays normally; prefetch dequeues song 2, sets
          _queue_cleared = True, and returns a YTDL mock.
          Iteration 2 — guard fires: task_done() + cleanup() + discard; then
          queue_get() raises TimeoutError so the loop exits cleanly.
        """
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        queue_obj2 = QueueObject(
            "https://yt.com/watch?v=2", "Song 2", queue_obj.requester
        )
        await music_player.queue.put(queue_obj)
        await music_player.queue.put(queue_obj2)
        music_player.song_queue.append("Song 1 - url")
        music_player.song_queue.append("Song 2 - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
        music_player.play_next.wait = AsyncMock()

        prefetched = MagicMock()
        prefetched.cleanup = MagicMock()

        async def _prefetch_with_clear(_self):
            try:
                music_player.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            music_player._queue_cleared = True
            return prefetched

        async def _stop_noop(_self):
            pass

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=AsyncMock()),
            patch.object(MusicPlayer, "_prefetch_next_song", new=_prefetch_with_clear),
            patch.object(
                MusicPlayer,
                "queue_get",
                new=AsyncMock(side_effect=[queue_obj, asyncio.TimeoutError()]),
            ),
            patch.object(MusicPlayer, "stop", new=_stop_noop),
        ):
            await music_player.loop()

        prefetched.cleanup.assert_called_once()

    async def test_discards_song_and_calls_cleanup_when_song_queue_cleared_mid_stream(
        self, music_player, queue_obj, mock_song
    ):
        """If song_queue is cleared while _stream_source runs, the YTDL object is
        discarded without playing and its FFmpeg subprocess is terminated via cleanup().
        """
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc

        mock_song.cleanup = MagicMock()

        async def _stream_and_clear(_self, source):
            # Simulate queue_clear() racing with stream resolution
            music_player.song_queue.clear()
            return mock_song

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(MusicPlayer, "_stream_source", new=_stream_and_clear),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
        ):
            await music_player.loop()

        vc.play.assert_not_called()
        mock_song.cleanup.assert_called_once()
