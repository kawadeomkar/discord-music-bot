"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import asyncio
import contextlib
import re
import time
from collections import deque
from typing import Any, AsyncIterator, Callable, Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from redis.asyncio import Redis

from src.musicplayer import (
    MusicPlayer,
    _build_progress_bar,
    _deserialize_queue_item,
    _fmt_duration,
    _fmt_finish_time,
    _fmt_total_duration,
    _requester_mention,
    _serialize_queue_item,
)
from src.sources import YTSource
from src.youtube import QueueObject, YTDL
from tests.helpers import stub_create_task


@pytest.fixture(autouse=True)
def _stub_prefetch(monkeypatch: pytest.MonkeyPatch) -> None:
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
def mock_song() -> MagicMock:
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
    song.elapsed_secs = 0.0
    song.abr = 128
    song.asr = 44100
    song.acodec = "opus"
    return song


@pytest.fixture
def queue_obj(mock_author: MagicMock) -> QueueObject:
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=abc123",
        title="Test Song",
        requester=mock_author,
        duration=210,
        uploader="Test Channel",
    )


@pytest.fixture
def queue_obj_no_meta(mock_author: MagicMock) -> QueueObject:
    """QueueObject without optional metadata (duration/uploader None)."""
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=abc123",
        title="Test Song",
        requester=mock_author,
    )


@pytest.fixture()
def _stub_queue_put_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent prefetch_stream tasks in queue_put from doing real yt-dlp work."""
    from src import youtube

    monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())


# ── Formatter helpers ─────────────────────────────────────────────────────────


class TestFmtDuration:
    def test_seconds_only(self) -> None:
        assert _fmt_duration(45) == "0:45"

    def test_minutes_and_seconds(self) -> None:
        assert _fmt_duration(185) == "3:05"

    def test_hours_minutes_seconds(self) -> None:
        assert _fmt_duration(3723) == "1:02:03"

    def test_zero(self) -> None:
        assert _fmt_duration(0) == "0:00"

    def test_exactly_one_hour(self) -> None:
        assert _fmt_duration(3600) == "1:00:00"

    def test_pads_seconds(self) -> None:
        assert _fmt_duration(61) == "1:01"


class TestFmtTotalDuration:
    def test_seconds_only(self) -> None:
        assert _fmt_total_duration(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert _fmt_total_duration(185) == "3m 5s"

    def test_hours_minutes_seconds(self) -> None:
        assert _fmt_total_duration(3723) == "1h 2m 3s"

    def test_zero(self) -> None:
        assert _fmt_total_duration(0) == "0s"

    def test_exactly_one_hour(self) -> None:
        assert _fmt_total_duration(3600) == "1h"

    def test_hours_no_minutes_with_seconds(self) -> None:
        # Regression: 1h 0m 45s previously showed as "1h" (seconds dropped)
        assert _fmt_total_duration(3645) == "1h 45s"

    def test_hours_and_minutes_no_seconds(self) -> None:
        assert _fmt_total_duration(3780) == "1h 3m"


class TestRequesterMention:
    def test_returns_mention_when_present(self, mock_author: MagicMock) -> None:
        assert _requester_mention(mock_author) == mock_author.mention

    def test_returns_unknown_when_none(self) -> None:
        assert _requester_mention(None) == "Unknown"


class TestFmtFinishTime:
    def test_matches_clock_format(self) -> None:
        assert re.match(r"^\d{1,2}:\d{2} (AM|PM) PST$", _fmt_finish_time(90))

    def test_no_uncertainty_prefix(self) -> None:
        # Unlike _fmt_eta(), a song's own remaining duration is never
        # uncertain — no "~" prefix and no bold markdown wrapping.
        result = _fmt_finish_time(90)
        assert not result.startswith("~")
        assert "**" not in result


# ── QueuePut ─────────────────────────────────────────────────────────────────


class TestQueuePut:
    async def test_put_single_queue_object(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        await music_player.queue_put(queue_obj)
        assert music_player.queue.qsize() == 1

    async def test_put_single_appends_to_song_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        await music_player.queue_put(queue_obj)
        assert len(music_player.song_queue) == 1
        assert isinstance(music_player.song_queue[0], QueueObject)
        assert music_player.song_queue[0].title == "Test Song"

    async def test_put_list_of_sources(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        sources = [
            YTSource(ytsearch="ytsearch:song one", process=True),
            YTSource(ytsearch="ytsearch:song two", process=True),
            YTSource(ytsearch="ytsearch:song three", process=True),
        ]
        await music_player.queue_put(sources)
        assert music_player.queue.qsize() == 3
        assert len(music_player.song_queue) == 3

    async def test_put_multiple_singles_increments_size(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
    ) -> None:
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 4
        assert len(music_player.song_queue) == 4

    async def test_put_mirrors_queue_object_to_redis(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: Redis,
    ) -> None:
        await music_player.queue_put(queue_obj)
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["type"] == "qobj"
        assert data["title"] == queue_obj.title
        assert data["webpage_url"] == queue_obj.webpage_url

    async def test_put_mirrors_yt_source_to_redis(
        self, music_player: MusicPlayer, fake_redis: Redis
    ) -> None:
        src = YTSource(ytsearch="ytsearch:Never Gonna Give You Up", process=True)
        await music_player.queue_put(src)
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["type"] == "ytsource"
        assert data["ytsearch"] == "ytsearch:Never Gonna Give You Up"

    async def test_put_yt_source_does_not_spawn_prefetch(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
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

    async def test_put_sets_ttl_on_redis_key(
        self, music_player: MusicPlayer, queue_obj: QueueObject, fake_redis: Redis
    ) -> None:
        await music_player.queue_put(queue_obj)
        ttl = await fake_redis.ttl(music_player._store.queue_key())
        assert ttl > 0

    async def test_put_spawns_prefetch_stream_for_queue_object(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
    ) -> None:
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(queue_obj)
            await asyncio.sleep(0)
        mock_pf.assert_awaited_once()
        assert mock_pf.call_args[0][0] == queue_obj

    async def test_put_does_not_spawn_prefetch_for_yt_source(
        self, music_player: MusicPlayer
    ) -> None:
        source = YTSource(ytsearch="ytsearch:test song", process=True)
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(source)
            await asyncio.sleep(0)
        mock_pf.assert_not_awaited()

    async def test_put_with_prefetch_false_skips_prefetch_task(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
    ) -> None:
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
    def _setup(self, _stub_queue_put_tasks: None) -> None:
        pass

    async def test_clear_empties_queue(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 3

        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_empties_song_queue(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert len(music_player.song_queue) == 3

        await music_player.queue_clear()
        assert len(music_player.song_queue) == 0

    async def test_clear_on_empty_queue_is_safe(
        self, music_player: MusicPlayer
    ) -> None:
        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_deletes_redis_key(
        self, music_player: MusicPlayer, queue_obj: QueueObject, fake_redis: Redis
    ) -> None:
        await music_player.queue_put(queue_obj)
        await music_player.queue_clear()
        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert items == []

    async def test_clear_returns_list_of_cleared_display_strings(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
    ) -> None:
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

    async def test_clear_returns_empty_list_when_queue_was_empty(
        self, music_player: MusicPlayer
    ) -> None:
        cleared = await music_player.queue_clear()
        assert cleared == []


# ── QueueShuffle ──────────────────────────────────────────────────────────────


class TestQueueShuffle:
    @pytest.fixture(autouse=True)
    def _setup(self, _stub_queue_put_tasks: None) -> None:
        pass

    async def test_shuffle_requires_minimum_four_items(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        result = await music_player.queue_shuffle()
        assert result == "There must be at least 3 songs to shuffle the queue"

    async def test_shuffle_empty_queue_returns_error(
        self, music_player: MusicPlayer
    ) -> None:
        result = await music_player.queue_shuffle()
        assert result == "There must be at least 3 songs to shuffle the queue"

    async def test_shuffle_sufficient_songs_returns_shuffled(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
    ) -> None:
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        result = await music_player.queue_shuffle()
        assert result == "Shuffled!"

    async def test_shuffle_preserves_queue_size(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        await music_player.queue_shuffle()
        assert music_player.queue.qsize() == 5

    async def test_shuffle_rebuilds_redis_from_kept_items(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
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
    def _stub_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_remove_by_webpage_url(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=abc", "Song", mock_author)
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=abc")

        assert positions == [1]
        assert music_player.queue.qsize() == 0
        assert len(music_player.song_queue) == 0

    async def test_remove_by_user_input_not_supported(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # user_input is not a match key — only webpage_url is used.
        qobj = QueueObject(
            "https://yt.com/v=abc", "Song", mock_author, user_input="my search query"
        )
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("my search query")

        assert positions == []
        assert music_player.queue.qsize() == 1

    async def test_no_match_returns_empty_list(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=abc", "Song", mock_author)
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=xyz")

        assert positions == []
        assert music_player.queue.qsize() == 1
        assert len(music_player.song_queue) == 1

    async def test_remove_empty_queue_returns_empty(
        self, music_player: MusicPlayer
    ) -> None:
        positions = await music_player.queue_remove("https://yt.com/v=x")
        assert positions == []

    async def test_remove_returns_correct_1indexed_positions(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
    ) -> None:
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=2")
        assert positions == [3]

    async def test_remove_multiple_matches_returns_all_positions(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
    ) -> None:
        urls = ["https://yt.com/v=a", "https://yt.com/v=b", "https://yt.com/v=a"]
        for url in urls:
            await music_player.queue_put(QueueObject(url, f"Song {url}", mock_author))

        positions = await music_player.queue_remove("https://yt.com/v=a")
        assert positions == [1, 3]

    async def test_remove_keeps_non_matching_songs(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
        await music_player.queue_put(
            QueueObject("https://yt.com/v=only", "Only Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=only")

        exists = await fake_redis.exists(music_player._store.queue_key())
        assert exists == 0

    async def test_remove_does_not_modify_redis_on_no_match(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
        await music_player.queue_put(
            QueueObject("https://yt.com/v=abc", "Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=xyz")

        items = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(items) == 1


# ── GetQueue embed ────────────────────────────────────────────────────────────


class TestGetQueue:
    def test_returns_discord_embed(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.song_queue.append(queue_obj)
        result = music_player.get_queue()
        assert isinstance(result, discord.Embed)

    def test_embed_title_is_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert embed.title == "Queue"

    def test_embed_color_is_blue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert embed.colour == discord.Color.blue()

    def test_empty_queue_description(self, music_player: MusicPlayer) -> None:
        embed = music_player.get_queue()
        assert "Songs: **0**" in embed.description
        assert "*The queue is empty.*" in embed.description

    def test_song_count_in_header(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=120
                )
            )
        embed = music_player.get_queue()
        assert "Songs: **3**" in embed.description

    def test_total_duration_in_header_when_all_known(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=90)
        )
        embed = music_player.get_queue()
        assert "Total Duration: **3m**" in embed.description
        assert "~" not in embed.description.split("Total Duration:")[1].split("\n")[0]

    def test_total_duration_partial_when_some_unknown(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=None)
        )
        embed = music_player.get_queue()
        assert "~" in embed.description

    def test_total_duration_partial_with_ytsource(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.song_queue.append(
            YTSource(ytsearch="ytsearch:unresolved", process=True)
        )
        embed = music_player.get_queue()
        assert "~" in embed.description

    def test_song_title_appears_in_description(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert "Test Song" in embed.description

    def test_song_duration_appears_when_known(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert "`3:30`" in embed.description

    def test_song_duration_unknown_shows_placeholder(
        self,
        music_player: MusicPlayer,
        queue_obj_no_meta: QueueObject,
    ) -> None:
        music_player.song_queue.append(queue_obj_no_meta)
        embed = music_player.get_queue()
        assert "`?:??`" in embed.description

    def test_uploader_shown_when_known(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.song_queue.append(queue_obj)
        embed = music_player.get_queue()
        assert "Test Channel" in embed.description

    def test_unknown_channel_shown_when_uploader_none(
        self,
        music_player: MusicPlayer,
        queue_obj_no_meta: QueueObject,
    ) -> None:
        music_player.song_queue.append(queue_obj_no_meta)
        embed = music_player.get_queue()
        assert "Unknown channel" in embed.description

    def test_est_playing_at_present_for_each_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.get_queue()
        assert embed.description.count("Est. playing at") == 3

    def test_uncertain_prefix_after_no_duration_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
    ) -> None:
        mock_current = MagicMock()
        mock_current.duration_secs = 0
        music_player.current_song = mock_current
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        embed = music_player.get_queue()
        assert "~**" in embed.description

    def test_caps_display_at_ten_songs(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(15):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.get_queue()
        assert embed.description.count("Est. playing at") == 10

    def test_shows_more_indicator_when_over_ten(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(15):
            music_player.song_queue.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.get_queue()
        assert "... and 5 more" in embed.description

    def test_ytsource_shows_resolving(self, music_player: MusicPlayer) -> None:
        music_player.song_queue.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player.get_queue()
        assert "resolving..." in embed.description


# ── EstimatedPlayingAt ────────────────────────────────────────────────────────


class TestEstimatedPlayingAt:
    def test_matches_clock_format(self, music_player: MusicPlayer) -> None:
        result = music_player.estimated_playing_at()
        assert re.match(r"^\*\*\d{1,2}:\d{2} (AM|PM) PST\*\*$", result)

    def test_uncertain_when_current_song_has_no_duration_secs(
        self, music_player: MusicPlayer
    ) -> None:
        mock_current = MagicMock()
        mock_current.duration_secs = 0
        music_player.current_song = mock_current
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_accounts_for_already_queued_songs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        music_player.current_song = mock_song
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_matches_last_queue_line_eta(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """estimated_playing_at() should reflect the same seed used by
        get_queue()/_build_next_up_embed() for consistency across embeds."""
        music_player.current_song = mock_song
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        eta = music_player.estimated_playing_at()

        # A song appended now would start right where the last queued line's
        # ETA ends up, so re-derive it via the same line formatter for index 2.
        now_pst, walk = music_player._queue_eta_seed()
        _, walk = music_player._format_queue_line(
            music_player.song_queue[0], 1, now_pst, walk
        )
        expected_line, _ = music_player._format_queue_line(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=60),
            2,
            now_pst,
            walk,
        )
        assert eta in expected_line


# ── BuildNowPlayingEmbed ──────────────────────────────────────────────────────


class TestBuildProgressBar:
    def test_empty_string_when_duration_unknown(self) -> None:
        assert _build_progress_bar(0.0, 0) == ""
        assert _build_progress_bar(10.0, -1) == ""

    def test_head_at_start_when_elapsed_zero(self) -> None:
        bar = _build_progress_bar(0.0, 200, width=10)
        assert bar.count("🔘") == 1
        # head is the first bar character after the leading `elapsed` code span
        assert "`0:00`" in bar

    def test_head_at_end_when_elapsed_equals_duration(self) -> None:
        bar = _build_progress_bar(200.0, 200, width=10)
        assert bar.count("🔘") == 1
        # clamped to width - 1: fully "done" up to the head, nothing remaining
        assert bar.count("🟦") == 9
        assert bar.count("⬜") == 0

    def test_head_roughly_midpoint_at_half_duration(self) -> None:
        bar = _build_progress_bar(100.0, 200, width=10)
        # head_pos = int(0.5 * 10) = 5 done blocks before the head, 4 remaining after
        middle = bar.split("`")[2]  # text between the two backtick-wrapped times
        head_index = middle.index("🔘")
        assert middle[:head_index].count("🟦") == 5
        assert middle[head_index + 1 :].count("⬜") == 4

    def test_clamped_when_elapsed_exceeds_duration(self) -> None:
        """Involuntary drift (e.g. a stale duration_secs) must not overflow the bar."""
        bar = _build_progress_bar(500.0, 200, width=10)
        assert bar.count("🔘") == 1
        assert bar.count("🟦") == 9
        assert bar.count("⬜") == 0

    def test_head_clamped_to_start_when_elapsed_negative(self) -> None:
        """elapsed_secs is never negative in practice (Design §1's read()-counter
        starts at 0 and only increments), but ratio clamping must not crash or
        push the head off the bar if it ever were."""
        bar = _build_progress_bar(-5.0, 200, width=10)
        assert bar.count("🔘") == 1
        assert bar.count("🟦") == 0
        middle = bar.split("`")[2].strip()
        assert middle.startswith(
            "🔘"
        )  # head pinned to the start, no done blocks before it

    def test_width_is_customizable(self) -> None:
        bar = _build_progress_bar(0.0, 200, width=5)
        assert bar.count("🟦") + bar.count("🔘") + bar.count("⬜") == 5

    def test_includes_formatted_elapsed_and_duration(self) -> None:
        bar = _build_progress_bar(65.0, 200)
        assert "`1:05`" in bar
        assert "`3:20`" in bar


class TestBuildNowPlayingEmbed:
    def test_returns_discord_embed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert isinstance(embed, discord.Embed)

    def test_embed_title_contains_song_title(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.title in embed.title

    def test_embed_description_contains_requester_mention(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.requester.mention in embed.description

    def test_embed_color_is_green(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.colour == discord.Color.green()

    def test_embed_has_youtube_link_field(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Youtube link" in field_names

    def test_embed_has_duration_field(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Duration" in field_names

    def test_embed_thumbnail_is_set(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.thumbnail.url == mock_song.thumbnail

    def test_embed_footer_contains_bitrate_info(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert str(mock_song.abr) in embed.footer.text
        assert str(mock_song.acodec) in embed.footer.text

    def test_description_has_estimated_finish_when_duration_known(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish:" in embed.description

    def test_estimated_finish_appears_after_requester_on_same_line(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """The requester/finish-time line stays on one line — the progress bar
        (Design §2 of the progress-bar plan) sits above it as its own line, not
        interleaved with it."""
        embed = music_player._build_now_playing_embed(mock_song)
        requester_line = embed.description.split("\n")[-1]
        assert re.search(
            r"Requester: \[.*\].*Estimated finish: \d{1,2}:\d{2} (AM|PM) PST$",
            requester_line,
        )

    def test_progress_bar_appears_above_requester_line(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """UI update: the bar sits directly under the title, above the
        requester/finish-time line — not the other way around."""
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        lines = embed.description.split("\n")
        assert "🔘" in lines[0]
        assert lines[2].startswith("Requester:")

    def test_blank_line_separates_bar_from_requester_line(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        lines = embed.description.split("\n")
        assert lines[1] == ""

    def test_no_estimated_finish_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish" not in embed.description

    def test_progress_bar_line_present_when_duration_known(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "🔘" in embed.description
        assert "\n" in embed.description  # progress bar is on its own line

    def test_progress_bar_reflects_elapsed_secs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 105.0  # roughly halfway through 210s
        embed = music_player._build_now_playing_embed(mock_song)
        assert _fmt_duration(105) in embed.description

    def test_no_progress_bar_line_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "🔘" not in embed.description

    def test_elapsed_override_replaces_song_elapsed_secs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Used by _finalize_now_playing() to render the bar fully completed
        once a song has ended, regardless of song.elapsed_secs's live value."""
        mock_song.elapsed_secs = 30.0
        mock_song.duration_secs = 210
        embed = music_player._build_now_playing_embed(mock_song, elapsed_override=210.0)
        assert _fmt_duration(210) in embed.description
        assert _fmt_duration(30) not in embed.description

    def test_no_override_falls_back_to_song_elapsed_secs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert _fmt_duration(30) in embed.description


class TestUpdateActivity:
    async def test_sets_playing_activity_when_song_playing(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        mock_song.duration_secs = 0
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert "start" in activity.timestamps
        assert "end" not in activity.timestamps

    async def test_truncates_name_to_128_chars(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        mock_song.title = "A" * 125
        mock_song.uploader = "B"
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert len(activity.name) == 128
        assert activity.name.endswith("…")

    async def test_resets_to_game_activity_when_idle(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        music_player.bot.voice_clients = []
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_awaited_once()
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert isinstance(activity, discord.Game)
        assert activity.name == "music"

    async def test_skips_reset_when_another_guild_is_playing(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        active_vc = MagicMock(spec=discord.VoiceClient)
        active_vc.is_playing.return_value = True
        music_player.bot.voice_clients = [active_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_not_awaited()

    async def test_resets_when_voice_clients_present_but_not_playing(
        self,
        music_player: MusicPlayer,
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        idle_vc = MagicMock(spec=discord.VoiceClient)
        idle_vc.is_playing.return_value = False
        music_player.bot.voice_clients = [idle_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_awaited_once()

    async def test_falls_back_to_a_song_when_title_is_none(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        mock_song.title = None
        mock_song.uploader = None
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.name == "a song"

    async def test_swallows_change_presence_exception(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.bot.change_presence = AsyncMock(
            side_effect=Exception("rate limited")
        )
        # Must not raise — playback loop must not be interrupted by a presence failure
        await music_player.update_activity(mock_song)

    async def test_backdates_start_by_elapsed_secs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """start must be backdated by elapsed time, not always "now" — otherwise
        resuming a song already 60s in would make `end` land a full duration_secs
        in the future instead of the correct remaining time (Design §6)."""
        music_player.bot.change_presence = AsyncMock()
        mock_song.elapsed_secs = 60.0
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        now_ms = int(time.time() * 1000)
        assert activity.timestamps["start"] <= now_ms - 60_000 + 1000
        assert activity.timestamps["start"] >= now_ms - 60_000 - 2000
        # end still lands duration_secs after the (backdated) start, not "now"
        assert (
            abs(
                activity.timestamps["end"]
                - (activity.timestamps["start"] + mock_song.duration_secs * 1000)
            )
            < 1000
        )


class TestUpdateActivityPause:
    """Design review (2026-07-01): update_activity() previously set timestamps
    once at song start and never accounted for pause state at all."""

    async def test_omits_timestamps_entirely_while_paused(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        music_player._guild.voice_client.is_paused.return_value = True
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.timestamps == {}

    async def test_still_sets_name_and_state_while_paused(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """Only the ticking timestamps are dropped — the rest of the activity
        (title/uploader/state) still renders while paused."""
        music_player.bot.change_presence = AsyncMock()
        music_player._guild.voice_client.is_paused.return_value = True
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.name == f"{mock_song.title} · {mock_song.uploader}"
        assert activity.state == mock_song.duration

    async def test_resumed_timestamps_reflect_elapsed_not_full_duration(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """On resume, elapsed_secs already reflects time played before the pause
        (Design §1 — YTDL.read() counting freezes during a pause), so a normal
        (non-paused) update_activity() call after resume must still backdate
        `start` by that elapsed time rather than restarting the countdown."""
        music_player.bot.change_presence = AsyncMock()
        music_player._guild.voice_client.is_paused.return_value = False
        mock_song.elapsed_secs = 60.0  # paused at 1:00 into a 3:30 track, now resumed
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        remaining_ms = activity.timestamps["end"] - int(time.time() * 1000)
        expected_remaining_ms = (mock_song.duration_secs - 60) * 1000
        assert abs(remaining_ms - expected_remaining_ms) < 2000


class TestMusicPlayerInitialState:
    def test_queue_starts_empty(self, music_player: MusicPlayer) -> None:
        assert music_player.queue.qsize() == 0

    def test_song_queue_starts_empty(self, music_player: MusicPlayer) -> None:
        assert len(music_player.song_queue) == 0

    def test_history_starts_empty(self, music_player: MusicPlayer) -> None:
        assert len(music_player.history) == 0

    def test_current_song_is_none(self, music_player: MusicPlayer) -> None:
        assert music_player.current_song is None

    def test_play_message_is_none(self, music_player: MusicPlayer) -> None:
        assert music_player.play_message is None

    def test_player_task_is_none_before_start(self, music_player: MusicPlayer) -> None:
        assert music_player._player is None

    def test_restore_task_is_none_before_start(self, music_player: MusicPlayer) -> None:
        assert music_player._restore_task is None


# ── RedisHelpers ──────────────────────────────────────────────────────────────


class TestRedisHelpers:
    async def test_redis_push_history_capped_at_50(
        self, music_player: MusicPlayer, fake_redis: Redis
    ) -> None:
        for i in range(55):
            await music_player._store.push_history(orjson.dumps(f"Song {i} - url{i}"))
        items = await fake_redis.lrange(music_player._store.history_key(), 0, -1)
        assert len(items) == 50

    async def test_redis_set_state_updates_volume(
        self, music_player: MusicPlayer, fake_redis: Redis
    ) -> None:
        await music_player.redis_set_state("volume", "0.75")
        state = await fake_redis.hgetall(music_player._store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_redis_pop_queue_removes_first_item(
        self, music_player: MusicPlayer, fake_redis: Redis
    ) -> None:
        await fake_redis.rpush(music_player._store.queue_key(), b"item1")
        await fake_redis.rpush(music_player._store.queue_key(), b"item2")
        await music_player._store.pop_queue()
        remaining = await fake_redis.lrange(music_player._store.queue_key(), 0, -1)
        assert len(remaining) == 1
        assert remaining[0] == b"item2"

    def test_store_is_none_when_no_redis(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        assert mp._store is None


# ── StateRestore ──────────────────────────────────────────────────────────────


class TestStateRestore:
    async def test_restore_populates_queue(
        self, music_player: MusicPlayer, fake_redis: Redis, mock_author: MagicMock
    ) -> None:
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

    async def test_restore_sets_volume(
        self, music_player: MusicPlayer, fake_redis: Redis
    ) -> None:
        await fake_redis.hset(music_player._store.state_key(), b"volume", b"0.5")
        await music_player._restore_state()
        assert music_player.volume == 0.5

    async def test_restore_noop_when_no_redis(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        await mp._restore_state()
        assert mp.queue.qsize() == 0

    async def test_restore_fetches_queue_and_history_concurrently(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
    ) -> None:
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

    async def test_crashed_song_restores_duration_and_uploader(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
        await fake_redis.hset(
            music_player._store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_title", b"Crashed Song"
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_duration", b"240"
        )
        await fake_redis.hset(
            music_player._store.state_key(), b"current_song_uploader", b"Test Channel"
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.duration == 240
        assert first.uploader == "Test Channel"

    async def test_no_crash_song_when_state_empty(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
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


# ── RestoredEvent ─────────────────────────────────────────────────────────────
# Regression coverage for a race where loop() could dequeue the crash-recovered
# "current song" _restore_state() injects and call pop_queue() (Redis LPOP) for
# it — silently deleting an unrelated, still-queued song from Redis, since the
# crashed song was never itself on the Redis queue list. loop() now waits on
# self._restored, which _restore_state() sets only once it has finished.


class TestRestoredEvent:
    async def test_restore_state_sets_restored_on_success(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
    ) -> None:
        music_player._restored.clear()
        await music_player._restore_state()
        assert music_player._restored.is_set()

    async def test_restore_state_sets_restored_on_failure(
        self, music_player: MusicPlayer
    ) -> None:
        music_player._restored.clear()
        with patch.object(
            music_player._store,
            "get_state",
            new=AsyncMock(side_effect=Exception("redis down")),
        ):
            await music_player._restore_state()
        assert music_player._restored.is_set()

    async def test_restore_state_sets_restored_when_no_store(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        await mp._restore_state()
        assert mp._restored.is_set()

    async def test_loop_waits_for_restore_before_dequeuing(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
        """loop() must not call pop_queue() for the crash-recovered song until
        _restore_state() has fully populated the queue from Redis."""
        music_player._restored.clear()
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
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
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
        music_player._restored.clear()
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
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
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
    async def test_returns_queue_object_unchanged(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        result = await music_player._resolve_source(queue_obj)
        assert result is queue_obj

    async def test_resolves_ytsource_via_yt_source(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
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
    async def test_returns_none_on_exception(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        with patch(
            "src.musicplayer.YTDL.yt_stream",
            new=AsyncMock(side_effect=Exception("boom")),
        ):
            result = await music_player._stream_source(queue_obj)
        assert result is None

    async def test_returns_ytdl_on_success(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        mock_ytdl = MagicMock()
        with patch(
            "src.musicplayer.YTDL.yt_stream", new=AsyncMock(return_value=mock_ytdl)
        ):
            result = await music_player._stream_source(queue_obj)
        assert result is mock_ytdl


# ── FromContext ───────────────────────────────────────────────────────────────


class TestFromContext:
    def test_creates_music_player(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert isinstance(mp, MusicPlayer)

    def test_sets_last_author_to_ctx_author(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert mp._last_author is mock_ctx.author

    def test_raises_if_guild_is_none(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        mock_ctx.guild = None
        with pytest.raises(AssertionError):
            MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)

    def test_attaches_store_when_redis_provided(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert mp._store is not None


# ── Start ─────────────────────────────────────────────────────────────────────


class TestStart:
    def test_start_creates_player_and_restore_tasks(
        self, music_player: MusicPlayer
    ) -> None:
        # _restore_state() is scheduled before loop() — loop() waits on
        # self._restored before its first dequeue, so restore must be
        # in flight first. See _restore_state()'s docstring for why.
        restore_task = MagicMock(name="restore_task")
        player_task = MagicMock(name="player_task")
        returns = [restore_task, player_task]

        def _create(coro: Coroutine[Any, Any, Any]) -> MagicMock:
            coro.close()
            return returns.pop(0)

        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(side_effect=_create)
        assert music_player._store is not None
        music_player.start()

        assert music_player._restore_task is restore_task
        assert music_player._player is player_task

    def test_no_restore_task_when_store_absent(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = stub_create_task()
        mp.start()
        assert mp._player is not None
        assert mp._restore_task is None


# ── SetContext ────────────────────────────────────────────────────────────────


class TestSetContext:
    def test_updates_channel(
        self, music_player: MusicPlayer, mock_ctx: MagicMock
    ) -> None:
        new_channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = new_channel
        music_player.set_context(mock_ctx)
        assert music_player._channel is new_channel

    def test_updates_last_author(
        self, music_player: MusicPlayer, mock_ctx: MagicMock
    ) -> None:
        new_author = MagicMock(spec=discord.Member)
        mock_ctx.author = new_author
        music_player.set_context(mock_ctx)
        assert music_player._last_author is new_author


# ── Stop ──────────────────────────────────────────────────────────────────────


class TestStop:
    async def test_delegates_to_cog_cleanup(self, music_player: MusicPlayer) -> None:
        music_player._cog.cleanup = AsyncMock()
        await music_player.stop()
        music_player._cog.cleanup.assert_awaited_once_with(music_player._guild)


# ── CancelPrefetch ────────────────────────────────────────────────────────────


class TestCancelPrefetch:
    async def test_noop_when_no_prefetch_task(self, music_player: MusicPlayer) -> None:
        music_player._prefetch_task = None
        await music_player._cancel_prefetch()

    async def test_noop_when_prefetch_task_already_done(
        self, music_player: MusicPlayer
    ) -> None:
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        music_player._prefetch_task = task
        await music_player._cancel_prefetch()
        task.cancel.assert_not_called()

    async def test_cancels_in_flight_prefetch_task(
        self, music_player: MusicPlayer
    ) -> None:
        async def _long() -> None:
            await asyncio.sleep(100)

        task = asyncio.create_task(_long())
        music_player._prefetch_task = task
        await music_player._cancel_prefetch()
        assert task.cancelled()


# ── SendNowPlaying ────────────────────────────────────────────────────────────


class TestSendNowPlaying:
    @pytest.fixture(autouse=True)
    async def _cleanup_progress_task(
        self, music_player: MusicPlayer
    ) -> AsyncIterator[None]:
        """_send_now_playing() may spawn a real _progress_task (Design §4). Tests
        in this class don't drive loop() to retire it themselves, so clean it up
        here rather than leaking a pending asyncio.sleep() task past the test."""
        yield
        await music_player._cancel_progress_task()

    async def test_sends_embed_to_channel(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        await music_player._send_now_playing(mock_song)
        music_player._channel.send.assert_awaited_once()
        call_kwargs = music_player._channel.send.call_args[1]
        assert "embeds" in call_kwargs

    async def test_stores_embed_as_play_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        await music_player._send_now_playing(mock_song)
        assert music_player.play_message is not None
        assert isinstance(music_player.play_message, discord.Embed)

    async def test_swallows_channel_send_exception(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player._channel.send = AsyncMock(side_effect=Exception("channel gone"))
        await music_player._send_now_playing(mock_song)

    async def test_resets_stale_now_playing_message_on_send_failure(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """Regression (code review): a failed/partial send must not leave
        _now_playing_message pointing at the *previous* song's message —
        otherwise a later mark_paused()/mark_resumed() on the new song would
        silently edit the wrong (old, already-finished) song's embed."""
        stale_message = MagicMock(spec=discord.Message)
        music_player._now_playing_message = stale_message
        music_player._channel.send = AsyncMock(side_effect=Exception("channel gone"))
        await music_player._send_now_playing(mock_song)
        assert music_player._now_playing_message is None

    async def test_sends_only_now_playing_embed_when_queue_empty(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        await music_player._send_now_playing(mock_song)
        call_kwargs = music_player._channel.send.call_args[1]
        assert len(call_kwargs["embeds"]) == 1
        assert call_kwargs["embeds"][0].colour == discord.Color.green()

    async def test_sends_next_up_embed_when_queue_has_song(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        mock_author: MagicMock,
    ) -> None:
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

    async def test_stores_sent_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        sent_message = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent_message)
        await music_player._send_now_playing(mock_song)
        assert music_player._now_playing_message is sent_message

    async def test_starts_progress_task_for_normal_duration_song(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        mock_song.duration_secs = 210
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is not None
        assert not music_player._progress_task.done()

    async def test_no_progress_task_for_sub_5s_song(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 4
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is None

    async def test_no_progress_task_for_zero_duration_song(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        mock_song.duration_secs = 0
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is None

    async def test_progress_task_starts_for_exactly_5s_song(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        mock_song.duration_secs = 5
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is not None


# ── FinalizeNowPlaying ────────────────────────────────────────────────────────


class TestFinalizeNowPlaying:
    """A song freezing mid-bar (e.g. `3:04 / 3:07`) after it ends — because the
    last periodic tick landed before the true end — is fixed by one last,
    fire-and-forget edit showing the bar fully completed."""

    async def test_edits_message_with_full_duration(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 184.0  # song ended mid-tick, e.g. 3:04 / 3:07
        mock_song.duration_secs = 210
        message = AsyncMock(spec=discord.Message)

        await music_player._finalize_now_playing(mock_song, message)

        message.edit.assert_awaited_once()
        embed = message.edit.call_args.kwargs["embeds"][0]
        assert _fmt_duration(210) in embed.description
        assert _fmt_duration(184) not in embed.description

    async def test_noop_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message)
        message.edit.assert_not_awaited()

    async def test_includes_next_up_embed_when_queue_has_song(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message)
        embeds = message.edit.call_args.kwargs["embeds"]
        assert len(embeds) == 2
        assert embeds[1].title == "Up next"

    async def test_swallows_not_found(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.NotFound(MagicMock(), "message deleted")
        await music_player._finalize_now_playing(mock_song, message)  # must not raise

    async def test_swallows_and_logs_http_exception(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")
        await music_player._finalize_now_playing(mock_song, message)  # must not raise

    async def test_operates_on_captured_song_and_message_args(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """Must use the song/message passed in, not self.current_song /
        self._now_playing_message — those may already point at the next song
        by the time this fire-and-forget task actually runs."""
        other_message = AsyncMock(spec=discord.Message)
        music_player.current_song = MagicMock()  # a different, "next" song
        music_player._now_playing_message = other_message

        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message)

        message.edit.assert_awaited_once()
        other_message.edit.assert_not_awaited()


class TestFireFinalizeNowPlaying:
    async def test_spawns_tracked_background_task(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        music_player._fire_finalize_now_playing(mock_song, message)
        task = next(iter(music_player._background_tasks))
        assert task in music_player._background_tasks
        await task
        message.edit.assert_awaited_once()
        assert task not in music_player._background_tasks


# ── ProgressUpdater ───────────────────────────────────────────────────────────


class TestProgressUpdater:
    @staticmethod
    def _make_sleep(n_ticks: int) -> Callable[[float], Coroutine[Any, Any, None]]:
        """asyncio.sleep double that lets the loop run n_ticks times, then raises
        CancelledError — deterministic without waiting on the real interval."""
        calls = 0

        async def _sleep(_secs: float) -> None:
            nonlocal calls
            calls += 1
            if calls > n_ticks:
                raise asyncio.CancelledError()

        return _sleep

    async def test_ticks_and_edits_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = AsyncMock(spec=discord.Message)

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song, message)

        message.edit.assert_awaited_once()
        assert "embeds" in message.edit.call_args.kwargs

    async def test_skips_edit_while_paused(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = True
        music_player._guild.voice_client = vc
        message = AsyncMock(spec=discord.Message)

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song, message)

        message.edit.assert_not_awaited()

    async def test_returns_when_song_changed_under_it(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """loop() owns cancellation on song transition, but this guard protects
        against a stray tick landing after the song changed (Design §4)."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = MagicMock()  # a different song than the one passed in
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = AsyncMock(spec=discord.Message)

        with patch("asyncio.sleep", new=AsyncMock()):
            await music_player._progress_updater(
                mock_song, message
            )  # returns, no raise

        message.edit.assert_not_awaited()

    async def test_stops_cleanly_on_message_not_found(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.NotFound(MagicMock(), "message deleted")

        with patch("asyncio.sleep", new=AsyncMock()):
            await music_player._progress_updater(
                mock_song, message
            )  # returns, no raise

        message.edit.assert_awaited_once()

    async def test_logs_and_continues_on_http_exception(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song, message)

        assert message.edit.await_count == 2  # kept ticking despite the failure


# ── CancelProgressTask ────────────────────────────────────────────────────────


class TestCancelProgressTask:
    async def test_noop_when_no_progress_task(self, music_player: MusicPlayer) -> None:
        music_player._progress_task = None
        await music_player._cancel_progress_task()

    async def test_noop_when_progress_task_already_done(
        self, music_player: MusicPlayer
    ) -> None:
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        music_player._progress_task = task
        await music_player._cancel_progress_task()
        task.cancel.assert_not_called()

    async def test_cancels_and_awaits_in_flight_progress_task(
        self, music_player: MusicPlayer
    ) -> None:
        async def _long() -> None:
            await asyncio.sleep(100)

        task = asyncio.create_task(_long())
        music_player._progress_task = task
        await music_player._cancel_progress_task()
        assert task.cancelled()
        assert music_player._progress_task is None

    async def test_song_transition_retires_task_before_next_send(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """Closes the song-transition race found in design review: the previous
        song's progress task must be fully retired — not just .cancel()'d —
        before the next song's _send_now_playing() sends a new message."""
        call_order: list[str] = []

        async def _never_finishes() -> None:
            try:
                await asyncio.sleep(100)
            finally:
                call_order.append("old_task_retired")

        music_player._progress_task = asyncio.create_task(_never_finishes())
        await asyncio.sleep(0)  # let the task actually start before cancelling it

        original_send = music_player._channel.send

        async def _tracked_send(*a: Any, **kw: Any) -> Any:
            call_order.append("new_message_sent")
            return await original_send(*a, **kw)

        music_player._channel.send = AsyncMock(side_effect=_tracked_send)

        await music_player._cancel_progress_task()
        await music_player._send_now_playing(mock_song)

        assert call_order == ["old_task_retired", "new_message_sent"]
        await music_player._cancel_progress_task()  # clean up the new song's task


# ── Pause/resume debounce ─────────────────────────────────────────────────────


class TestPauseDebounce:
    @pytest.fixture(autouse=True)
    async def _cleanup(self, music_player: MusicPlayer) -> AsyncIterator[None]:
        yield
        await music_player._cancel_pause_debounce()
        # _progress_task in these tests is a bare MagicMock sentinel (truthy for
        # the "is not None" check), not a real awaitable task — reset directly
        # rather than going through _cancel_progress_task()'s await.
        music_player._progress_task = None

    async def test_noop_when_no_current_song(self, music_player: MusicPlayer) -> None:
        music_player.current_song = None
        music_player.mark_paused()
        assert music_player._pause_debounce_task is None

    async def test_single_call_fires_after_debounce_window(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.current_song = mock_song
        music_player._now_playing_message = AsyncMock(spec=discord.Message)
        music_player._progress_task = MagicMock(spec=asyncio.Task)
        music_player._progress_task.done.return_value = False
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        assert music_player._pause_debounce_task is not None
        await music_player._pause_debounce_task

        music_player._now_playing_message.edit.assert_awaited_once()
        music_player.bot.change_presence.assert_awaited_once()

    async def test_rapid_toggling_collapses_to_one_trailing_update(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.current_song = mock_song
        music_player._now_playing_message = AsyncMock(spec=discord.Message)
        music_player._progress_task = MagicMock(spec=asyncio.Task)
        music_player._progress_task.done.return_value = False
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        music_player.mark_resumed()
        music_player.mark_paused()
        music_player.mark_resumed()
        # Only the last debounce task should still be alive/pending.
        final_task = music_player._pause_debounce_task
        assert final_task is not None
        await final_task

        music_player._now_playing_message.edit.assert_awaited_once()
        music_player.bot.change_presence.assert_awaited_once()

    async def test_no_embed_edit_when_no_progress_task_or_message(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.current_song = mock_song
        music_player._now_playing_message = None
        music_player._progress_task = None
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        await music_player._pause_debounce_task

        music_player.bot.change_presence.assert_awaited_once()


# ── MarkPausedResumed ──────────────────────────────────────────────────────────


class TestMarkPausedResumed:
    @pytest.fixture(autouse=True)
    async def _cleanup(self, music_player: MusicPlayer) -> AsyncIterator[None]:
        yield
        await music_player._cancel_pause_debounce()
        music_player._progress_task = None

    async def test_mark_paused_schedules_debounced_update(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.current_song = mock_song
        music_player.mark_paused()
        assert music_player._pause_debounce_task is not None

    async def test_mark_resumed_schedules_debounced_update(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        music_player.current_song = mock_song
        music_player.mark_resumed()
        assert music_player._pause_debounce_task is not None

    async def test_scheduled_tasks_tracked_via_background_tasks(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
    ) -> None:
        """The debounce task itself, and the embed-edit/activity tasks it spawns,
        must be tracked via _background_tasks (not bare create_task() calls) —
        design review flagged this as the same GC-pending-task risk the codebase
        already guards against elsewhere (musicplayer.py:511-512)."""
        music_player.current_song = mock_song
        music_player._now_playing_message = AsyncMock(spec=discord.Message)
        music_player._progress_task = MagicMock(spec=asyncio.Task)
        music_player._progress_task.done.return_value = False
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        assert music_player._pause_debounce_task in music_player._background_tasks
        await music_player._pause_debounce_task
        # Debounce task itself is discarded from the set once done (done_callback).
        assert music_player._pause_debounce_task not in music_player._background_tasks


# ── BuildNextUpEmbed ──────────────────────────────────────────────────────────


class TestBuildNextUpEmbed:
    def test_returns_none_when_queue_empty(self, music_player: MusicPlayer) -> None:
        assert music_player._build_next_up_embed() is None

    def test_returns_blue_embed_with_song_details(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
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

    def test_shows_resolving_for_unresolved_ytsource(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.song_queue.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "resolving..." in embed.description

    def test_shows_placeholder_duration_when_unknown(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "`?:??`" in embed.description

    def test_only_uses_first_queued_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
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

    def test_includes_est_playing_at_eta(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.song_queue.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "Est. playing at" in embed.description
        assert re.search(r"\*\*\d{1,2}:\d{2} (AM|PM) PST\*\*", embed.description)

    def test_eta_matches_current_song_estimated_finish(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
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
        # Last line only — the progress bar sits above it as its own line and
        # isn't part of the finish-time text being compared here.
        requester_line = now_playing_embed.description.split("\n")[-1]
        finish_time = requester_line.split("Estimated finish: ")[1]
        assert finish_time in next_up_embed.description


# ── PrefetchNextSong ──────────────────────────────────────────────────────────


class TestPrefetchNextSong:
    async def test_returns_none_when_queue_empty(
        self, music_player: MusicPlayer
    ) -> None:
        result = await music_player._prefetch_next_song()
        assert result is None

    async def test_returns_ytdl_on_success(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
    ) -> None:
        await music_player.queue.put(queue_obj)
        with patch(
            "src.musicplayer.YTDL.yt_stream",
            new=AsyncMock(side_effect=Exception("network")),
        ):
            result = await music_player._prefetch_next_song()
        assert result is None

    async def test_reraises_cancelled_error_and_calls_task_done(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
    ) -> None:
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
    async def test_returns_item_from_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        await music_player.queue.put(queue_obj)
        result = await music_player.queue_get()
        assert result is queue_obj


# ── DeserializeQueueItem ──────────────────────────────────────────────────────


class TestDeserializeQueueItem:
    def test_falls_back_to_guild_owner_when_member_not_found(
        self, mock_guild: MagicMock
    ) -> None:
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

    def test_returns_none_when_member_and_owner_both_none(
        self, mock_guild: MagicMock
    ) -> None:
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

    def test_returns_none_on_invalid_json(self, mock_guild: MagicMock) -> None:
        result = _deserialize_queue_item(b"not valid json{{{{", mock_guild)
        assert result is None

    def test_preserves_ts_field(
        self, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
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

    def test_deserializes_new_fields(
        self, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
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

    def test_deserializes_persisted_false(
        self, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
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

    def test_backward_compat_missing_new_fields(
        self, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
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
    def test_round_trip_all_fields(self, mock_author: MagicMock) -> None:
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

    def test_none_optional_fields_serialize_as_null(
        self, mock_author: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_author)
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["user_input"] is None
        assert d["duration"] is None
        assert d["uploader"] is None
        assert d["thumbnail"] is None

    def test_persisted_false_is_serialized(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            "https://yt.com/v=1", "Test Song", mock_author, persisted=False
        )
        data = _serialize_queue_item(qobj)
        d = orjson.loads(data)
        assert d["persisted"] is False

    def test_ytsource_round_trip(self) -> None:
        src = YTSource(ytsearch="ytsearch:Never Gonna Give You Up", process=True, ts=10)
        data = _serialize_queue_item(src)
        d = orjson.loads(data)
        assert d["type"] == "ytsource"
        assert d["ytsearch"] == "ytsearch:Never Gonna Give You Up"
        assert d["process"] is True
        assert d["ts"] == 10
        assert "requester_id" not in d

    def test_ytsource_url_preserved(self) -> None:
        src = YTSource(url="https://www.youtube.com/watch?v=abc", process=False)
        data = _serialize_queue_item(src)
        d = orjson.loads(data)
        assert d["type"] == "ytsource"
        assert d["url"] == "https://www.youtube.com/watch?v=abc"


class TestDeserializeQueueItemYTSource:
    def test_ytsource_deserialized_correctly(self, mock_guild: MagicMock) -> None:
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

    def test_ytsource_with_url(self, mock_guild: MagicMock) -> None:
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
        self,
        mock_guild: MagicMock,
        mock_author: MagicMock,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
    ) -> None:
        await fake_redis.hset(music_player._store.state_key(), b"volume", b"0.8")
        await fake_redis.expire(music_player._store.state_key(), 10)

        await music_player._restore_state()

        ttl = await fake_redis.ttl(music_player._store.state_key())
        assert ttl > 1000

    async def test_restore_continues_after_bad_queue_item(
        self,
        music_player: MusicPlayer,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
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
    def mock_song(self) -> MagicMock:
        song = MagicMock()
        song.title = "Loop Test Song"
        song.webpage_url = "https://yt.com/v=loop1"
        song.duration_secs = 210
        return song

    async def test_exits_immediately_when_bot_closed(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.bot.is_closed.return_value = True
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player.loop()

    async def test_timeout_triggers_stop(self, music_player: MusicPlayer) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.return_value = False

        stop_called = asyncio.Event()

        async def _mock_stop(self_inner: MusicPlayer) -> None:
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

    async def test_skips_song_when_stream_returns_none(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: Redis,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
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

    async def test_fires_finalize_task_when_song_ends(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
        """When a song ends, loop() must fire the finalize-embed task with the
        song/message that just finished, before current_song/_now_playing_message
        get overwritten for the next iteration."""
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
        music_player.play_next.wait = AsyncMock()

        sent_message = MagicMock(spec=discord.Message)

        async def _fake_send_now_playing(_self: MusicPlayer, song: YTDL) -> None:
            _self._now_playing_message = sent_message

        finalize_mock = MagicMock()

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=_fake_send_now_playing),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
            patch.object(MusicPlayer, "_fire_finalize_now_playing", new=finalize_mock),
        ):
            await music_player.loop()

        finalize_mock.assert_called_once_with(mock_song, sent_message)

    async def test_unhandled_exception_sends_error_message(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
    ) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue.put(queue_obj)
        music_player.song_queue.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)

        def _bad_play(*a: Any, **kw: Any) -> None:
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
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
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
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
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

        async def _prefetch_with_clear(_self: MusicPlayer) -> MagicMock:
            try:
                music_player.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            music_player._queue_cleared = True
            return prefetched

        async def _stop_noop(_self: MusicPlayer) -> None:
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
            # Unrelated to this test (prefetch/cleanup) — the bare object.__new__()
            # VoiceClient double below has no real _player, so the real
            # update_activity() would crash calling vc.is_paused() on it.
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        prefetched.cleanup.assert_called_once()

    async def test_discards_song_and_calls_cleanup_when_song_queue_cleared_mid_stream(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
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

        async def _stream_and_clear(
            _self: MusicPlayer, source: QueueObject
        ) -> MagicMock:
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
