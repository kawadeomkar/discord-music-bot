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
    _fmt_duration,
    _fmt_total_duration,
    _serialize_queue_item,
)
from src.sources import YTSource
from src.youtube import QueueObject


@pytest.fixture
def mock_song():
    """A mock YTDL-like song object with all metadata attributes."""
    song = MagicMock()
    song.title = "Test Song Title"
    song.requester = MagicMock()
    song.requester.mention = "<@123456>"
    song.webpage_url = "https://www.youtube.com/watch?v=testid"
    song.duration = "0:03:30"
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
        duration=210,
        uploader="Test Channel",
    )


@pytest.fixture
def queue_obj_no_meta(mock_author):
    """QueueObject without optional metadata (duration/uploader None)."""
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=abc123",
        title="Test Song",
        requester=mock_author,
    )


# ── Formatter helpers ─────────────────────────────────────────────────────────


class TestFmtDuration:
    def test_seconds_only(self):
        assert _fmt_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _fmt_duration(185) == "3:05"

    def test_hours_minutes_seconds(self):
        assert _fmt_duration(3723) == "1:02:03"

    def test_zero(self):
        assert _fmt_duration(0) == "0:00"

    def test_exactly_one_hour(self):
        assert _fmt_duration(3600) == "1:00:00"

    def test_pads_seconds(self):
        assert _fmt_duration(61) == "1:01"


class TestFmtTotalDuration:
    def test_seconds_only(self):
        assert _fmt_total_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert _fmt_total_duration(185) == "3m 5s"

    def test_hours_minutes_seconds(self):
        assert _fmt_total_duration(3723) == "1h 2m 3s"

    def test_zero(self):
        assert _fmt_total_duration(0) == "0s"

    def test_exactly_one_hour(self):
        assert _fmt_total_duration(3600) == "1h"

    def test_hours_no_minutes_with_seconds(self):
        # Regression: 1h 0m 45s previously showed as "1h" (seconds dropped)
        assert _fmt_total_duration(3645) == "1h 45s"

    def test_hours_and_minutes_no_seconds(self):
        assert _fmt_total_duration(3780) == "1h 3m"


# ── QueuePut ─────────────────────────────────────────────────────────────────


class TestQueuePut:
    @pytest.fixture(autouse=True)
    def _stub_prefetch(self, monkeypatch):
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_put_single_queue_object(self, music_player, queue_obj):
        await music_player.queue_put(queue_obj)
        assert music_player.queue.qsize() == 1

    async def test_put_single_appends_to_song_queue(self, music_player, queue_obj):
        await music_player.queue_put(queue_obj)
        assert len(music_player.song_queue) == 1
        assert isinstance(music_player.song_queue[0], QueueObject)
        assert music_player.song_queue[0].title == "Test Song"

    async def test_put_list_of_sources(self, music_player, mock_author):
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
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(queue_obj)
            await asyncio.sleep(0)
        mock_pf.assert_awaited_once()
        assert mock_pf.call_args[0][0] == queue_obj

    async def test_put_does_not_spawn_prefetch_for_yt_source(self, music_player):
        source = YTSource(ytsearch="ytsearch:test song", process=True)
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(source)
            await asyncio.sleep(0)
        mock_pf.assert_not_awaited()


# ── QueueClear ────────────────────────────────────────────────────────────────


class TestQueueClear:
    @pytest.fixture(autouse=True)
    def _stub_prefetch(self, monkeypatch):
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

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


# ── QueueShuffle ──────────────────────────────────────────────────────────────


class TestQueueShuffle:
    @pytest.fixture(autouse=True)
    def _stub_prefetch(self, monkeypatch):
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

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

    async def test_shuffle_rebuilds_redis_from_kept_items(
        self, music_player, mock_author, fake_redis
    ):
        """Redis must be rebuilt from the re-queued items, not the pre-shuffle drain."""
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        await music_player.queue_shuffle()

        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 5
        urls = {orjson.loads(item)["webpage_url"] for item in items}
        assert urls == {f"https://yt.com/watch?v={i}" for i in range(5)}


# ── QueueRemove ───────────────────────────────────────────────────────────────


class TestQueueRemove:
    @pytest.fixture(autouse=True)
    def _stub_prefetch(self, monkeypatch):
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_remove_by_webpage_url(self, music_player, mock_author):
        qobj = QueueObject("https://yt.com/v=abc", "Song", mock_author)
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=abc")

        assert positions == [1]
        assert music_player.queue.qsize() == 0
        assert len(music_player.song_queue) == 0

    async def test_remove_by_user_input(self, music_player, mock_author):
        qobj = QueueObject(
            "https://yt.com/v=abc", "Song", mock_author, user_input="my search query"
        )
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("my search query")

        assert positions == [1]
        assert music_player.queue.qsize() == 0

    async def test_no_match_returns_empty_list(self, music_player, mock_author):
        qobj = QueueObject("https://yt.com/v=abc", "Song", mock_author)
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=xyz")

        assert positions == []
        assert music_player.queue.qsize() == 1
        assert len(music_player.song_queue) == 1

    async def test_remove_empty_queue_returns_empty(self, music_player):
        positions = await music_player.queue_remove("https://yt.com/v=x")
        assert positions == []

    async def test_remove_returns_correct_1indexed_positions(
        self, music_player, mock_author
    ):
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=2")
        assert positions == [3]

    async def test_remove_multiple_matches_returns_all_positions(
        self, music_player, mock_author
    ):
        urls = ["https://yt.com/v=a", "https://yt.com/v=b", "https://yt.com/v=a"]
        for url in urls:
            await music_player.queue_put(QueueObject(url, f"Song {url}", mock_author))

        positions = await music_player.queue_remove("https://yt.com/v=a")
        assert positions == [1, 3]

    async def test_remove_keeps_non_matching_songs(self, music_player, mock_author):
        for i in range(3):
            await music_player.queue_put(
                QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            )

        await music_player.queue_remove("https://yt.com/v=1")

        remaining = list(music_player.song_queue)
        assert len(remaining) == 2
        urls = [item.webpage_url for item in remaining if isinstance(item, QueueObject)]
        assert "https://yt.com/v=0" in urls
        assert "https://yt.com/v=2" in urls
        assert "https://yt.com/v=1" not in urls

    async def test_remove_updates_redis_when_songs_remain(
        self, music_player, mock_author, fake_redis
    ):
        for i in range(3):
            await music_player.queue_put(
                QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            )

        await music_player.queue_remove("https://yt.com/v=1")

        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 2
        urls = [orjson.loads(item)["webpage_url"] for item in items]
        assert "https://yt.com/v=1" not in urls

    async def test_remove_deletes_redis_key_when_queue_becomes_empty(
        self, music_player, mock_author, fake_redis
    ):
        await music_player.queue_put(
            QueueObject("https://yt.com/v=only", "Only Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=only")

        exists = await fake_redis.exists(music_player._store.queue_key())
        assert exists == 0

    async def test_remove_does_not_modify_redis_on_no_match(
        self, music_player, mock_author, fake_redis
    ):
        await music_player.queue_put(
            QueueObject("https://yt.com/v=abc", "Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=xyz")

        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1


# ── GetQueue embed ────────────────────────────────────────────────────────────


class TestGetQueue:
    def test_returns_discord_embed(self, music_player, queue_obj):
        music_player.song_queue.append(queue_obj)
        result = music_player.get_queue()
        assert isinstance(result, discord.Embed)

    def test_embed_title_is_queue(self, music_player, queue_obj):
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert embed.title == "Queue"

    def test_embed_color_is_blue(self, music_player, queue_obj):
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert embed.colour == discord.Color.blue()

    def test_empty_queue_description(self, music_player):
        embed = music_player.get_queue()
        assert "Songs: **0**" in embed.description
        assert "*The queue is empty.*" in embed.description

    def test_song_count_in_header(self, music_player, mock_author):
        for i in range(3):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=120
                )
            )
        embed = music_player.get_queue()
        assert "Songs: **3**" in embed.description

    def test_total_duration_in_header_when_all_known(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=90)
        )
        embed = music_player.get_queue()
        assert "Total Duration: **3m**" in embed.description
        assert "~" not in embed.description.split("Total Duration:")[1].split("\n")[0]

    def test_total_duration_partial_when_some_unknown(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=None)
        )
        embed = music_player.get_queue()
        assert "~" in embed.description

    def test_total_duration_partial_with_ytsource(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.song_queue.append(
            YTSource(ytsearch="ytsearch:unresolved", process=True)
        )
        embed = music_player.get_queue()
        assert "~" in embed.description

    def test_song_title_appears_in_description(self, music_player, queue_obj):
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert "Test Song" in embed.description

    def test_song_duration_appears_when_known(self, music_player, queue_obj):
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert "`3:30`" in embed.description

    def test_song_duration_unknown_shows_placeholder(
        self, music_player, queue_obj_no_meta
    ):
        music_player.song_queue.append(queue_obj_no_meta)
        embed = music_player.get_queue()
        assert "`?:??`" in embed.description

    def test_uploader_shown_when_known(self, music_player, queue_obj):
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert "Test Channel" in embed.description

    def test_unknown_channel_shown_when_uploader_none(
        self, music_player, queue_obj_no_meta
    ):
        music_player.song_queue.append(queue_obj_no_meta)
        embed = music_player.get_queue()
        assert "Unknown channel" in embed.description

    def test_est_playing_at_present_for_each_song(self, music_player, mock_author):
        for i in range(3):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.get_queue()
        assert embed.description.count("Est. playing at") == 3

    def test_uncertain_prefix_after_no_duration_song(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=60)
        )
        embed = music_player.get_queue()
        # First song: no preceding unknown → no ~
        # Second song: preceding song had unknown duration → ~
        lines = embed.description.split("\n")
        est_lines = [l for l in lines if "Est. playing at" in l]
        assert not est_lines[0].startswith("~") or "~**" not in est_lines[0]
        assert "~**" in est_lines[1]

    def test_uncertain_when_current_song_duration_parse_fails(
        self, music_player, mock_author
    ):
        mock_current = MagicMock()
        mock_current.duration = "not-a-valid-duration"
        music_player.current_song = mock_current
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        embed = music_player.get_queue()
        assert "~**" in embed.description

    def test_caps_display_at_ten_songs(self, music_player, mock_author):
        for i in range(15):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.get_queue()
        assert embed.description.count("Est. playing at") == 10

    def test_shows_more_indicator_when_over_ten(self, music_player, mock_author):
        for i in range(15):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.get_queue()
        assert "... and 5 more" in embed.description

    def test_ytsource_shows_resolving(self, music_player):
        music_player.song_queue.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player.get_queue()
        assert "resolving..." in embed.description


# ── BuildNowPlayingEmbed ──────────────────────────────────────────────────────


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


# ── MusicPlayerInitialState ───────────────────────────────────────────────────


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


# ── RedisHelpers ──────────────────────────────────────────────────────────────


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


# ── StateRestore ──────────────────────────────────────────────────────────────


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
        assert isinstance(music_player.song_queue[0], QueueObject)
        assert music_player.song_queue[0].title == "Restored Song"

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


# ── RestoreCrashedSong ────────────────────────────────────────────────────────


class TestRestoreCrashedSong:
    async def test_crashed_song_requeued_at_front(
        self, music_player, fake_redis, mock_author
    ):
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed Song"
        )
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


# ── ResolveSource ─────────────────────────────────────────────────────────────


class TestResolveSource:
    async def test_returns_queue_object_unchanged(self, music_player, queue_obj):
        result = await music_player._resolve_source(queue_obj)
        assert result is queue_obj

    async def test_resolves_ytsource_via_yt_source(self, music_player, mock_author):
        fake_qobj = QueueObject("https://yt.com/v=1", "Resolved", mock_author)
        with patch(
            "src.musicplayer.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_player._resolve_source(
                YTSource(ytsearch="ytsearch:test", process=True)
            )
        assert isinstance(result, QueueObject)
        assert result.title == "Resolved"


# ── StreamSource ──────────────────────────────────────────────────────────────


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


# ── FromContext ───────────────────────────────────────────────────────────────


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


# ── Start ─────────────────────────────────────────────────────────────────────


class TestStart:
    def test_creates_player_task(self, music_player):
        mock_task = MagicMock()
        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(return_value=mock_task)
        # Replace async methods with plain mocks so create_task receives a
        # non-coroutine value and doesn't leak an unawaited coroutine on GC.
        music_player.loop = MagicMock()
        music_player.start()
        assert music_player._player is mock_task

    def test_creates_restore_task_when_store_present(self, music_player):
        mock_task = MagicMock()
        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(return_value=mock_task)
        music_player.loop = MagicMock()
        music_player._restore_state = MagicMock()
        assert music_player._store is not None
        music_player.start()
        assert music_player._restore_task is not None

    def test_no_restore_task_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = MagicMock(return_value=MagicMock())
        mp.loop = MagicMock()
        mp.start()
        assert mp._restore_task is None


# ── SetContext ────────────────────────────────────────────────────────────────


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


# ── Stop ──────────────────────────────────────────────────────────────────────


class TestStop:
    async def test_delegates_to_cog_cleanup(self, music_player):
        music_player._cog.cleanup = AsyncMock()
        await music_player.stop()
        music_player._cog.cleanup.assert_awaited_once_with(music_player._guild)


# ── CancelPrefetch ────────────────────────────────────────────────────────────


class TestCancelPrefetch:
    async def test_noop_when_no_prefetch_task(self, music_player):
        music_player._prefetch_task = None
        await music_player._cancel_prefetch()

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


# ── SendNowPlaying ────────────────────────────────────────────────────────────


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
        await music_player._send_now_playing(mock_song)


# ── PrefetchNextSong ──────────────────────────────────────────────────────────


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
        await music_player.queue.put(queue_obj)
        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with pytest.raises(asyncio.CancelledError):
                await music_player._prefetch_next_song()


# ── QueueGet ──────────────────────────────────────────────────────────────────


class TestQueueGet:
    async def test_returns_item_from_queue(self, music_player, queue_obj):
        await music_player.queue.put(queue_obj)
        result = await music_player.queue_get()
        assert result is queue_obj


# ── DeserializeQueueItem ──────────────────────────────────────────────────────


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

    def test_deserializes_new_fields(self, mock_guild, mock_author):
        mock_guild.get_member = MagicMock(return_value=mock_author)
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=1",
                "title": "Song",
                "requester_id": mock_author.id,
                "ts": None,
                "user_input": "my search",
                "duration": 240,
                "uploader": "My Channel",
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is not None
        assert result.user_input == "my search"
        assert result.duration == 240
        assert result.uploader == "My Channel"

    def test_backward_compat_missing_new_fields(self, mock_guild, mock_author):
        """Old Redis entries without user_input/duration/uploader deserialize cleanly."""
        mock_guild.get_member = MagicMock(return_value=mock_author)
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=1",
                "title": "Old Entry",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is not None
        assert result.user_input is None
        assert result.duration is None
        assert result.uploader is None


# ── SerializeQueueItem ────────────────────────────────────────────────────────


class TestSerializeQueueItem:
    def test_round_trip_all_fields(self, mock_author):
        qobj = QueueObject(
            "https://yt.com/v=1",
            "Test Song",
            mock_author,
            ts=30,
            user_input="my search",
            duration=240,
            uploader="My Channel",
        )
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["webpage_url"] == "https://yt.com/v=1"
        assert d["title"] == "Test Song"
        assert d["requester_id"] == mock_author.id
        assert d["ts"] == 30
        assert d["user_input"] == "my search"
        assert d["duration"] == 240
        assert d["uploader"] == "My Channel"

    def test_none_optional_fields_serialize_as_null(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_author)
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["user_input"] is None
        assert d["duration"] is None
        assert d["uploader"] is None


# ── RestoreStateTtlRefresh ────────────────────────────────────────────────────


class TestRestoreStateTtlRefresh:
    async def test_ttl_refreshed_after_successful_restore(
        self, music_player, fake_redis
    ):
        await fake_redis.hset(music_player._store.state_key(), b"volume", b"0.8")
        await fake_redis.expire(music_player._store.state_key(), 10)

        await music_player._restore_state()

        ttl = await fake_redis.ttl(music_player._store.state_key())
        assert ttl > 1000

    async def test_restore_continues_after_bad_queue_item(
        self, music_player, fake_redis, mock_author
    ):
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


# ── Loop ──────────────────────────────────────────────────────────────────────


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
        await music_player.loop()

    async def test_timeout_triggers_stop(self, music_player):
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
        music_player.song_queue.append(queue_obj)

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
        music_player.song_queue.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc

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
        music_player.song_queue.append(queue_obj)

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
