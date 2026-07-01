"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import asyncio
import contextlib
import re
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest

from src.musicplayer import (
    MusicPlayer,
    _deserialize_queue_item,
    _fmt_duration,
    _fmt_finish_time,
    _fmt_total_duration,
    _requester_mention,
    _serialize_queue_item,
)
from src.sources import YTSource
from src.youtube import QueueObject
from tests.helpers import stub_create_task


@pytest.fixture(autouse=True)
def _stub_prefetch(monkeypatch):
    """Stub YTDL.prefetch_stream for every test in this module.

    queue_put() spawns asyncio.create_task(YTDL.prefetch_stream(...)) for every
    QueueObject. Without this stub, any test that calls queue_put with a yt.com
    test URL would trigger a real yt-dlp network request in a background task.
    Tests that specifically assert on prefetch behaviour override this via their
    own patch() context manager, which takes precedence.
    """
    from src import youtube

    monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())


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


@pytest.fixture()
def _stub_queue_put_tasks(monkeypatch):
    """Prevent prefetch_stream tasks in queue_put from doing real yt-dlp work."""
    from src import youtube

    monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())


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


class TestRequesterMention:
    def test_returns_mention_when_present(self, mock_author):
        assert _requester_mention(mock_author) == mock_author.mention

    def test_returns_unknown_when_none(self):
        assert _requester_mention(None) == "Unknown"


class TestFmtFinishTime:
    def test_matches_clock_format(self):
        assert re.match(r"^\d{1,2}:\d{2} (AM|PM) PST$", _fmt_finish_time(90))

    def test_no_uncertainty_prefix(self):
        # Unlike _fmt_eta(), a song's own remaining duration is never
        # uncertain — no "~" prefix and no bold markdown wrapping.
        result = _fmt_finish_time(90)
        assert not result.startswith("~")
        assert "**" not in result


# ── QueuePut ─────────────────────────────────────────────────────────────────


class TestQueuePut:
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
        assert data["type"] == "qobj"
        assert data["title"] == queue_obj.title
        assert data["webpage_url"] == queue_obj.webpage_url

    async def test_put_mirrors_yt_source_to_redis(self, music_player, fake_redis):
        src = YTSource(ytsearch="ytsearch:Never Gonna Give You Up", process=True)
        await music_player.queue_put(src)
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["type"] == "ytsource"
        assert data["ytsearch"] == "ytsearch:Never Gonna Give You Up"

    async def test_put_yt_source_does_not_spawn_prefetch(
        self, music_player, fake_redis, mock_author
    ):
        from unittest.mock import patch, AsyncMock

        src = YTSource(ytsearch="ytsearch:test", process=True)
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(src)
            await asyncio.sleep(0)
        mock_pf.assert_not_awaited()
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1

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


# ── QueueClear ────────────────────────────────────────────────────────────────


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


# ── QueueShuffle ──────────────────────────────────────────────────────────────


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

    async def test_shuffle_excludes_non_persisted_item_from_redis(
        self, music_player, mock_author, fake_redis
    ):
        """A crash-recovered (persisted=False) item mid-queue must never be
        written to Redis by a shuffle — it was never RPUSHed there."""
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.queue.put(crashed)
        music_player.song_queue.append(crashed)
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        await music_player.queue_shuffle()

        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        urls = {orjson.loads(item)["webpage_url"] for item in items}
        assert "https://yt.com/v=crashed" not in urls
        assert len(items) == 4


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

    async def test_remove_by_user_input_not_supported(self, music_player, mock_author):
        # user_input is not a match key — only webpage_url is used.
        qobj = QueueObject(
            "https://yt.com/v=abc", "Song", mock_author, user_input="my search query"
        )
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("my search query")

        assert positions == []
        assert music_player.queue.qsize() == 1

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

    async def test_remove_excludes_non_persisted_item_from_redis(
        self, music_player, mock_author, fake_redis
    ):
        """A crash-recovered (persisted=False) item kept after a remove must
        never be written to Redis — it was never RPUSHed there."""
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.queue.put(crashed)
        music_player.song_queue.append(crashed)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=a", "Song A", mock_author)
        )
        await music_player.queue_put(
            QueueObject("https://yt.com/v=b", "Song B", mock_author)
        )

        positions = await music_player.queue_remove("https://yt.com/v=a")

        assert positions == [2]  # crashed(1), a(2), b(3) — 1-indexed
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        urls = {orjson.loads(item)["webpage_url"] for item in items}
        assert "https://yt.com/v=crashed" not in urls
        assert urls == {"https://yt.com/v=b"}

    async def test_remove_deletes_redis_key_when_only_non_persisted_item_kept(
        self, music_player, mock_author, fake_redis
    ):
        """If removal leaves only a non-persisted item, Redis's queue key
        should end up empty/deleted, not populated with a phantom entry."""
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.queue.put(crashed)
        music_player.song_queue.append(crashed)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=only", "Only Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=only")

        exists = await fake_redis.exists(music_player._store.queue_key())
        assert exists == 0

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

    def test_uncertain_when_current_song_has_no_duration_secs(
        self, music_player, mock_author
    ):
        mock_current = MagicMock()
        mock_current.duration_secs = 0
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


# ── EstimatedPlayingAt ────────────────────────────────────────────────────────


class TestEstimatedPlayingAt:
    def test_matches_clock_format(self, music_player):
        result = music_player.estimated_playing_at()
        assert re.match(r"^\*\*\d{1,2}:\d{2} (AM|PM) PST\*\*$", result)

    def test_uncertain_when_current_song_has_no_duration_secs(self, music_player):
        mock_current = MagicMock()
        mock_current.duration_secs = 0
        music_player.current_song = mock_current
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_accounts_for_already_queued_songs(self, music_player, mock_song):
        music_player.current_song = mock_song  # duration_secs = 210
        empty_eta = music_player.estimated_playing_at()

        music_player.song_queue.append(
            QueueObject(
                "https://yt.com/v=1", "Song 1", mock_song.requester, duration=600
            )
        )
        later_eta = music_player.estimated_playing_at()

        assert empty_eta != later_eta

    def test_uncertain_when_queued_song_duration_unknown(
        self, music_player, mock_song, mock_author
    ):
        music_player.current_song = mock_song
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_matches_last_queue_line_eta(self, music_player, mock_song, mock_author):
        """estimated_playing_at() should reflect the same seed used by
        get_queue()/_build_next_up_embed() for consistency across embeds."""
        music_player.current_song = mock_song
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        eta = music_player.estimated_playing_at()

        # A song appended now would start right where the last queued line's
        # ETA ends up, so re-derive it via the same line formatter for index 2.
        now_pst, cumulative_secs, uncertain = music_player._queue_eta_seed()
        _, cumulative_secs, uncertain = music_player._format_queue_line(
            music_player.song_queue[0], 1, now_pst, cumulative_secs, uncertain
        )
        expected_line, _, _ = music_player._format_queue_line(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=60),
            2,
            now_pst,
            cumulative_secs,
            uncertain,
        )
        assert eta in expected_line


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

    def test_embed_does_not_have_dislikes_field(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Dislikes" not in field_names

    def test_embed_thumbnail_not_set_when_none(self, music_player, mock_song):
        mock_song.thumbnail = None
        embed = music_player._build_now_playing_embed(mock_song)
        assert not embed.thumbnail.url

    def test_description_has_estimated_finish_when_duration_known(
        self, music_player, mock_song
    ):
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish:" in embed.description

    def test_estimated_finish_appears_after_requester_on_same_line(
        self, music_player, mock_song
    ):
        embed = music_player._build_now_playing_embed(mock_song)
        assert "\n" not in embed.description
        assert re.search(
            r"Requester: \[.*\].*Estimated finish: \d{1,2}:\d{2} (AM|PM) PST$",
            embed.description,
        )

    def test_no_estimated_finish_when_duration_unknown(self, music_player, mock_song):
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish" not in embed.description


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

    async def test_restore_fetches_queue_and_history_concurrently(
        self, music_player, fake_redis
    ):
        """Guard against a future edit reintroducing a hidden ordering
        dependency between get_queue() and get_history() — they're gathered
        specifically because neither depends on the other's result."""
        get_queue_spy = AsyncMock(wraps=music_player._store.get_queue)
        get_history_spy = AsyncMock(wraps=music_player._store.get_history)
        with (
            patch.object(music_player._store, "get_queue", get_queue_spy),
            patch.object(music_player._store, "get_history", get_history_spy),
        ):
            await music_player._restore_state()
        get_queue_spy.assert_awaited_once()
        get_history_spy.assert_awaited_once()


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

    async def test_crashed_song_url_cleared_even_when_requester_unresolvable(
        self, music_player, fake_redis
    ):
        """When guild.me and guild.owner are both None, the crashed song cannot be
        re-queued — but current_song_url must still be cleared to avoid an infinite
        retry loop on every subsequent restart."""
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Ghost Song"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player._guild.me = None
        music_player._guild.owner = None

        await music_player._restore_state()

        state = await fake_redis.hgetall(music_player._store.state_key())
        assert state.get(b"current_song_url", b"") == b""
        assert state.get(b"current_song_title", b"") == b""
        # Song was not re-queued since requester was unresolvable.
        assert music_player.queue.empty()

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

    async def test_crashed_song_uses_last_author_id_for_requester(
        self, music_player, fake_redis, mock_author
    ):
        """If last_author_id is in state, the crashed song's requester is that member."""
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
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
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
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
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts is None

    async def test_crashed_song_position_accounts_for_active_pause(
        self, music_player, fake_redis, mock_author
    ):
        """When the bot crashed while paused, pause_start_epoch contributes to total pause
        time and is subtracted from the seek position alongside total_pause_seconds."""
        import time

        play_start = time.time() - 90  # song started 90 s ago
        pause_start = time.time() - 20  # paused 20 s ago (still paused at crash)
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Paused Crash"
        )
        await fake_redis.hset(
            music_player._store.state_key(),
            b"play_start_epoch",
            str(play_start).encode(),
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"total_pause_seconds", b"10"
        )
        await fake_redis.hset(
            music_player._store.state_key(),
            b"pause_start_epoch",
            str(pause_start).encode(),
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        # elapsed=90s, prior_pause=10s, active_pause≈20s → position ≈ 90-10-20 = 60s
        assert first.ts is not None
        assert 50 <= first.ts <= 70


# ── RestoreCompleteEvent (loop guard) ─────────────────────────────────────────
# Regression coverage for a race where loop() could dequeue the crash-recovered
# "current song" _restore_state() injects and call pop_queue() (Redis LPOP) for
# it — silently deleting an unrelated, still-queued song from Redis, since the
# crashed song was never itself on the Redis queue list. loop() now waits on
# self._restore_complete, which _restore_state() sets only once it has finished.


class TestRestoreCompleteLoopGuard:
    async def test_restore_state_sets_restore_complete_on_success(
        self, music_player, fake_redis
    ):
        music_player._restore_complete.clear()
        await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_restore_state_sets_restore_complete_on_failure(self, music_player):
        music_player._restore_complete.clear()
        with patch.object(
            music_player._store,
            "get_state",
            new=AsyncMock(side_effect=Exception("redis down")),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_restore_state_sets_restore_complete_when_no_store(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        await mp._restore_state()
        assert mp._restore_complete.is_set()

    async def test_loop_waits_for_restore_before_dequeuing(
        self, music_player, fake_redis, mock_author
    ):
        """loop() must not call pop_queue() for the crash-recovered song until
        _restore_state() has fully populated the queue from Redis."""
        music_player._restore_complete.clear()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.return_value = False
        music_player.bot.loop = asyncio.get_running_loop()

        loop_task = asyncio.create_task(music_player.loop())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not loop_task.done()
        assert music_player.queue.qsize() == 0  # loop() hasn't dequeued anything yet

        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

    async def test_pop_queue_not_called_for_crash_recovered_song_before_restore_reads_queue(
        self, music_player, fake_redis, mock_author
    ):
        """End-to-end guard for the original bug: seed Redis with a crashed
        song plus 2 still-queued songs. After restore populates the queue and
        loop() processes exactly the crash-recovered song (its stream fails
        here, taking the "skip" path that also calls pop_queue()), both real
        queued songs must still be present in Redis — pop_queue() must not
        fire for the crashed song's own dequeue.
        """
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed Song"
        )
        for i in range(2):
            item = orjson.dumps(
                {
                    "webpage_url": f"https://yt.com/v={i}",
                    "title": f"Queued {i}",
                    "requester_id": mock_author.id,
                    "ts": None,
                }
            )
            await fake_redis.rpush(music_player._store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player._restore_complete.clear()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player._restore_state()
        assert music_player.queue.qsize() == 3  # crashed + 2 real queued songs

        # Exactly one loop() iteration — enough to process the crashed song.
        music_player.bot.is_closed.side_effect = [False, True]
        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(side_effect=lambda s: s)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=None)
            ),
        ):
            await music_player.loop()

        remaining = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(remaining) == 2

    async def test_shuffle_during_restore_window_does_not_orphan_redis_entry(
        self, music_player, fake_redis, mock_author
    ):
        """End-to-end guard for Issue 1: if a user runs -shuffle while the
        crash-recovered song is still sitting in song_queue (before loop()
        has dequeued it), Redis's queue list must still end up with exactly
        the real queued songs — no phantom entry for the crashed song."""
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed Song"
        )
        for i in range(4):
            item = orjson.dumps(
                {
                    "webpage_url": f"https://yt.com/v={i}",
                    "title": f"Queued {i}",
                    "requester_id": mock_author.id,
                    "ts": None,
                }
            )
            await fake_redis.rpush(music_player._store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()
        assert music_player.queue.qsize() == 5  # crashed + 4 real queued songs

        # Simulates a -shuffle command running before loop() ever dequeues anything.
        await music_player.queue_shuffle()

        remaining = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        urls = {orjson.loads(item)["webpage_url"] for item in remaining}
        assert "https://yt.com/v=crash" not in urls
        assert len(remaining) == 4


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
    def test_start_creates_player_and_restore_tasks(self, music_player):
        # _restore_state() is scheduled before loop() — loop() waits on
        # self._restore_complete before its first dequeue, so restore must be
        # in flight first. See _restore_state()'s docstring for why.
        restore_task = MagicMock(name="restore_task")
        player_task = MagicMock(name="player_task")
        returns = [restore_task, player_task]

        def _create(coro):
            coro.close()
            return returns.pop(0)

        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(side_effect=_create)
        assert music_player._store is not None
        music_player.start()

        assert music_player._restore_task is restore_task
        assert music_player._player is player_task

    def test_no_restore_task_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = stub_create_task()
        mp.start()
        assert mp._player is not None
        assert mp._restore_task is None

    def test_restore_complete_set_immediately_when_store_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        """When there is no Redis store, start() must signal _restore_complete immediately
        so loop()'s prefetch gate never blocks."""
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = stub_create_task()
        mp.start()
        assert mp._restore_complete.is_set()

    def test_restore_complete_not_set_before_start_when_store_present(
        self, mock_bot, mock_guild, mock_channel, mock_ctx, fake_redis
    ):
        """Before start() or _restore_state() runs, the event must be clear."""
        mp = MusicPlayer(
            mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis
        )
        assert not mp._restore_complete.is_set()


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
        assert "embeds" in call_kwargs

    async def test_stores_embed_as_play_message(self, music_player, mock_song):
        await music_player._send_now_playing(mock_song)
        assert music_player.play_message is not None
        assert isinstance(music_player.play_message, discord.Embed)

    async def test_swallows_channel_send_exception(self, music_player, mock_song):
        music_player._channel.send = AsyncMock(side_effect=Exception("channel gone"))
        await music_player._send_now_playing(mock_song)

    async def test_sends_only_now_playing_embed_when_queue_empty(
        self, music_player, mock_song
    ):
        await music_player._send_now_playing(mock_song)
        call_kwargs = music_player._channel.send.call_args[1]
        assert len(call_kwargs["embeds"]) == 1
        assert call_kwargs["embeds"][0].colour == discord.Color.green()

    async def test_sends_next_up_embed_when_queue_has_song(
        self, music_player, mock_song, mock_author
    ):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        await music_player._send_now_playing(mock_song)
        call_kwargs = music_player._channel.send.call_args[1]
        embeds = call_kwargs["embeds"]
        assert len(embeds) == 2
        assert embeds[1].colour == discord.Color.blue()
        assert embeds[1].title == "Up next"
        assert "Next Song" in embeds[1].description

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


# ── BuildNextUpEmbed ──────────────────────────────────────────────────────────


class TestBuildNextUpEmbed:
    def test_returns_none_when_queue_empty(self, music_player):
        assert music_player._build_next_up_embed() is None

    def test_returns_blue_embed_with_song_details(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert embed.colour == discord.Color.blue()
        assert embed.title == "Up next"
        assert "Next Song" in embed.description
        assert "https://yt.com/v=next" in embed.description
        assert "`1:30`" in embed.description
        assert mock_author.mention in embed.description

    def test_shows_resolving_for_unresolved_ytsource(self, music_player):
        music_player.song_queue.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "resolving..." in embed.description

    def test_shows_placeholder_duration_when_unknown(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "`?:??`" in embed.description

    def test_only_uses_first_queued_song(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "First", mock_author, duration=60)
        )
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=2", "Second", mock_author, duration=60)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "First" in embed.description
        assert "Second" not in embed.description

    def test_includes_est_playing_at_eta(self, music_player, mock_author):
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "Est. playing at" in embed.description
        assert re.search(r"\*\*\d{1,2}:\d{2} (AM|PM) PST\*\*", embed.description)

    def test_eta_matches_current_song_estimated_finish(self, music_player, mock_song):
        """The next song's ETA should line up with the current song's finish time,
        since both derive from the same cumulative_secs seed."""
        music_player.current_song = mock_song
        music_player.song_queue.append(
            QueueObject(
                "https://yt.com/v=next", "Next Song", mock_song.requester, duration=90
            )
        )
        now_playing_embed = music_player._build_now_playing_embed(mock_song)
        next_up_embed = music_player._build_next_up_embed()
        assert next_up_embed is not None
        finish_time = now_playing_embed.description.split("Estimated finish: ")[1]
        assert finish_time in next_up_embed.description


# ── PrefetchNextSong ──────────────────────────────────────────────────────────


class TestPrefetchNextSong:
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
                "thumbnail": "https://img.youtube.com/vi/1/0.jpg",
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is not None
        assert result.user_input == "my search"
        assert result.duration == 240
        assert result.uploader == "My Channel"
        assert result.thumbnail == "https://img.youtube.com/vi/1/0.jpg"
        assert result.persisted is True

    def test_deserializes_persisted_false(self, mock_guild, mock_author):
        mock_guild.get_member = MagicMock(return_value=mock_author)
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=1",
                "title": "Song",
                "requester_id": mock_author.id,
                "ts": None,
                "persisted": False,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert result is not None
        assert result.persisted is False

    def test_backward_compat_missing_new_fields(self, mock_guild, mock_author):
        """Old Redis entries without user_input/duration/uploader/thumbnail/persisted
        deserialize cleanly, defaulting persisted to True (a pre-fix entry can only
        ever have been a real, Redis-mirrored queue item)."""
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
        assert result.thumbnail is None
        assert result.persisted is True


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
            thumbnail="https://img.youtube.com/vi/1/0.jpg",
        )
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["type"] == "qobj"
        assert d["webpage_url"] == "https://yt.com/v=1"
        assert d["title"] == "Test Song"
        assert d["requester_id"] == mock_author.id
        assert d["ts"] == 30
        assert d["user_input"] == "my search"
        assert d["duration"] == 240
        assert d["uploader"] == "My Channel"
        assert d["thumbnail"] == "https://img.youtube.com/vi/1/0.jpg"
        assert d["persisted"] is True

    def test_none_optional_fields_serialize_as_null(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_author)
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["user_input"] is None
        assert d["duration"] is None
        assert d["uploader"] is None
        assert d["thumbnail"] is None

    def test_persisted_false_is_serialized(self, mock_author):
        qobj = QueueObject(
            "https://yt.com/v=1", "Test Song", mock_author, persisted=False
        )
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["persisted"] is False

    def test_ytsource_round_trip(self):
        src = YTSource(ytsearch="ytsearch:Never Gonna Give You Up", process=True, ts=10)
        data = _serialize_queue_item(src)
        d = orjson.loads(data)
        assert d["type"] == "ytsource"
        assert d["ytsearch"] == "ytsearch:Never Gonna Give You Up"
        assert d["process"] is True
        assert d["ts"] == 10
        assert "requester_id" not in d

    def test_ytsource_url_preserved(self):
        src = YTSource(url="https://www.youtube.com/watch?v=abc", process=False)
        data = _serialize_queue_item(src)
        d = orjson.loads(data)
        assert d["type"] == "ytsource"
        assert d["url"] == "https://www.youtube.com/watch?v=abc"


class TestDeserializeQueueItemYTSource:
    def test_ytsource_deserialized_correctly(self, mock_guild):
        data = orjson.dumps(
            {
                "type": "ytsource",
                "ytsearch": "ytsearch:Bohemian Rhapsody Queen",
                "url": None,
                "process": True,
                "ts": None,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:Bohemian Rhapsody Queen"
        assert result.process is True

    def test_ytsource_with_url(self, mock_guild):
        data = orjson.dumps(
            {
                "type": "ytsource",
                "ytsearch": None,
                "url": "https://www.youtube.com/watch?v=abc",
                "process": False,
                "ts": 15,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert isinstance(result, YTSource)
        assert result.url == "https://www.youtube.com/watch?v=abc"
        assert result.ts == 15

    def test_legacy_entry_without_type_field_deserializes_as_qobj(
        self, mock_guild, mock_author
    ):
        mock_guild.get_member = MagicMock(return_value=mock_author)
        data = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=legacy",
                "title": "Legacy Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        result = _deserialize_queue_item(data, mock_guild)
        assert isinstance(result, QueueObject)
        assert result.webpage_url == "https://yt.com/v=legacy"


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
        song.duration_secs = 210
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

    async def test_resolve_failure_balances_queue_and_redis(
        self, music_player, queue_obj, fake_redis
    ):
        """If _resolve_source() raises after queue_get() already dequeued the
        item, the dequeue must still be balanced (song_queue popped, Redis
        popped for a persisted item, queue.task_done() called exactly once)
        and the outer handler's error embed must still be sent."""
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player._store.push_queue(_serialize_queue_item(queue_obj))
        await music_player.queue.put(queue_obj)
        music_player.song_queue.append(queue_obj)

        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=Exception("yt-dlp lookup failed")),
        ):
            await music_player.loop()

        assert len(music_player.song_queue) == 0
        remaining = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(remaining) == 0
        assert music_player.queue._unfinished_tasks == 0  # task_done() balanced get()
        sent_embed = music_player._channel.send.call_args.kwargs["embed"]
        assert sent_embed.title == "Playback error — skipping song"

    async def test_resolve_failure_for_non_persisted_item_does_not_pop_redis(
        self, music_player, mock_author, fake_redis
    ):
        """A crash-recovered (persisted=False) item that fails to resolve
        must not trigger a Redis pop — it was never RPUSHed there."""
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player._store.push_queue(
            _serialize_queue_item(
                QueueObject("https://yt.com/v=real", "Real Song", mock_author)
            )
        )
        await music_player.queue.put(crashed)
        music_player.song_queue.append(crashed)

        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=Exception("yt-dlp lookup failed")),
        ):
            await music_player.loop()

        remaining = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        urls = {orjson.loads(item)["webpage_url"] for item in remaining}
        assert urls == {"https://yt.com/v=real"}

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

    async def test_error_path_clears_current_song_url(
        self, music_player, queue_obj, fake_redis
    ):
        """When loop() hits an unhandled exception, current_song_url must be cleared so
        a later process restart does not ghost-replay the failed song."""
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=RuntimeError("ffmpeg gone"))
        music_player._guild.voice_client = vc

        # Seed Redis so a restart would see a crashed song.
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=MagicMock())
            ),
        ):
            await music_player.loop()

        state = await fake_redis.hgetall(music_player._store.state_key())
        assert state.get(b"current_song_url", b"") == b""
        assert state.get(b"current_song_title", b"") == b""


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

    async def test_play_message_none_when_no_now_playing_in_redis(self, music_player):
        """No now_playing hash → play_message stays None after restore."""
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player._restore_state()
        assert music_player.play_message is None


# ── loop() additional coverage from main branch ───────────────────────────────


class TestLoopAdditional:
    @pytest.fixture
    def mock_song(self):
        song = MagicMock()
        song.title = "Loop Test Song"
        song.webpage_url = "https://yt.com/v=loop1"
        song.duration_secs = 210
        return song

    async def test_update_activity_called_at_song_start_and_end(
        self, music_player, queue_obj, mock_song
    ):
        music_player._restore_complete.set()
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
        music_player._restore_complete.set()
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
