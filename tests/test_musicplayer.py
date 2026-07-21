"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import redis.asyncio as aioredis
import asyncio
import contextlib
import dataclasses
import re
import time
from typing import Any, Never, cast
from collections.abc import AsyncGenerator, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import orjson
import pytest

from src.guild_state import HistoryEntry, NowPlayingData, SongQueueEntry
from src.musicplayer import (
    MusicPlayer,
    _BAR_WIDTH,
    _build_progress_bar,
    _reached_end,
    _fmt_finish_time,
    _fmt_total_duration,
    _requester_mention,
)
from src.sources import YTSource
from src.util import fmt_duration
from src.youtube import QueueObject
from tests.helpers import described, mocked, queue_object, stub_create_task


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
    song.requester.id = 123456
    song.requester.display_name = "TestUser"
    song.webpage_url = "https://www.youtube.com/watch?v=testid"
    song.duration = "0:03:30"
    song.uploader = "Test Channel"
    song.views = 1_000_000
    song.likes = 50_000
    song.dislikes = 500
    song.thumbnail = "https://img.youtube.com/vi/testid/0.jpg"
    song.duration_secs = 210
    song.elapsed_secs = 0.0
    song.start_offset = 0
    song.abr = 128
    song.asr = 44100
    song.acodec = "opus"
    # -playnow flags a real YTDL always carries — as bare MagicMock attributes
    # they'd read truthy and trip the loop's start_paused/is_resume gates.
    song.interjected = False
    song.is_resume = False
    song.start_paused = False
    # Mirror the real YTDL.position_secs property (start_offset + elapsed_secs)
    # so tests that set either attribute get the derived position automatically.
    type(song).position_secs = PropertyMock(
        side_effect=lambda: song.start_offset + song.elapsed_secs
    )
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
        assert len(music_player.queue._display) == 1
        assert isinstance(music_player.queue._display[0], QueueObject)
        assert music_player.queue._display[0].title == "Test Song"

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
        assert len(music_player.queue._display) == 3

    async def test_put_multiple_singles_increments_size(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 4
        assert len(music_player.queue._display) == 4

    async def test_put_mirrors_queue_object_to_redis(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        await music_player.queue_put(queue_obj)
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["type"] == "qobj"
        assert data["title"] == queue_obj.title
        assert data["webpage_url"] == queue_obj.webpage_url

    async def test_put_mirrors_yt_source_to_redis(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        src = YTSource(ytsearch="ytsearch:Never Gonna Give You Up", process=True)
        await music_player.queue_put(src)
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["type"] == "ytsource"
        assert data["ytsearch"] == "ytsearch:Never Gonna Give You Up"

    async def test_put_yt_source_does_not_spawn_prefetch(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        assert music_player.store is not None
        from unittest.mock import patch, AsyncMock

        src = YTSource(ytsearch="ytsearch:test", process=True)
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put(src)
            await asyncio.sleep(0)
        mock_pf.assert_not_awaited()
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1

    async def test_put_sets_ttl_on_redis_key(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        await music_player.queue_put(queue_obj)
        ttl = await fake_redis.ttl(music_player.store.queue_key())
        assert ttl > 0

    async def test_put_spawns_prefetch_stream_for_queue_object(
        self, music_player: MusicPlayer, queue_obj: QueueObject
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
        self, music_player: MusicPlayer, queue_obj: QueueObject
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
        assert len(music_player.queue._display) == 3

        await music_player.queue_clear()
        assert len(music_player.queue._display) == 0

    async def test_clear_on_empty_queue_is_safe(
        self, music_player: MusicPlayer
    ) -> None:
        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_deletes_redis_key(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        await music_player.queue_put(queue_obj)
        await music_player.queue_clear()
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert items == []

    async def test_clear_returns_list_of_cleared_display_strings(
        self, music_player: MusicPlayer, mock_author: MagicMock
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
        self, music_player: MusicPlayer, mock_author: MagicMock
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
        fake_redis: aioredis.Redis,
    ) -> None:
        """Redis must be rebuilt from the re-queued items, not the pre-shuffle drain."""
        assert music_player.store is not None
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        await music_player.queue_shuffle()

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 5
        urls = {orjson.loads(item)["webpage_url"] for item in items}
        assert urls == {f"https://yt.com/watch?v={i}" for i in range(5)}

    async def test_shuffle_excludes_non_persisted_item_from_redis(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """A crash-recovered (persisted=False) item mid-queue must never be
        written to Redis by a shuffle — it was never RPUSHed there."""
        assert music_player.store is not None
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.queue._pending.put(crashed)
        music_player.queue._display.append(crashed)
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        await music_player.queue_shuffle()

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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
        assert len(music_player.queue._display) == 0

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
        assert len(music_player.queue._display) == 1

    async def test_remove_empty_queue_returns_empty(
        self, music_player: MusicPlayer
    ) -> None:
        positions = await music_player.queue_remove("https://yt.com/v=x")
        assert positions == []

    async def test_remove_returns_correct_1indexed_positions(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=2")
        assert positions == [3]

    async def test_remove_multiple_matches_returns_all_positions(
        self, music_player: MusicPlayer, mock_author: MagicMock
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

        remaining = list(music_player.queue._display)
        assert len(remaining) == 2
        urls = [item.webpage_url for item in remaining if isinstance(item, QueueObject)]
        assert "https://yt.com/v=0" in urls
        assert "https://yt.com/v=2" in urls
        assert "https://yt.com/v=1" not in urls

    async def test_remove_updates_redis_when_songs_remain(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        for i in range(3):
            await music_player.queue_put(
                QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            )

        await music_player.queue_remove("https://yt.com/v=1")

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 2
        urls = [orjson.loads(item)["webpage_url"] for item in items]
        assert "https://yt.com/v=1" not in urls

    async def test_remove_excludes_non_persisted_item_from_redis(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """A crash-recovered (persisted=False) item kept after a remove must
        never be written to Redis — it was never RPUSHed there."""
        assert music_player.store is not None
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.queue._pending.put(crashed)
        music_player.queue._display.append(crashed)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=a", "Song A", mock_author)
        )
        await music_player.queue_put(
            QueueObject("https://yt.com/v=b", "Song B", mock_author)
        )

        positions = await music_player.queue_remove("https://yt.com/v=a")

        assert positions == [2]  # crashed(1), a(2), b(3) — 1-indexed
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        urls = {orjson.loads(item)["webpage_url"] for item in items}
        assert "https://yt.com/v=crashed" not in urls
        assert urls == {"https://yt.com/v=b"}

    async def test_remove_deletes_redis_key_when_only_non_persisted_item_kept(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """If removal leaves only a non-persisted item, Redis's queue key
        should end up empty/deleted, not populated with a phantom entry."""
        assert music_player.store is not None
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.queue._pending.put(crashed)
        music_player.queue._display.append(crashed)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=only", "Only Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=only")

        exists = await fake_redis.exists(music_player.store.queue_key())
        assert exists == 0

    async def test_remove_deletes_redis_key_when_queue_becomes_empty(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        await music_player.queue_put(
            QueueObject("https://yt.com/v=only", "Only Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=only")

        exists = await fake_redis.exists(music_player.store.queue_key())
        assert exists == 0

    async def test_remove_does_not_modify_redis_on_no_match(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        await music_player.queue_put(
            QueueObject("https://yt.com/v=abc", "Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=xyz")

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1


# ── GetQueue embed ────────────────────────────────────────────────────────────


class TestGetQueue:
    def test_returns_discord_embed(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj)
        result = music_player.queue_embed()
        assert isinstance(result, discord.Embed)

    def test_embed_title_is_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert embed.title == "Queue"

    def test_embed_color_is_blue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert embed.colour == discord.Color.blue()

    def test_empty_queue_description(self, music_player: MusicPlayer) -> None:
        embed = music_player.queue_embed()
        assert "Songs: **0**" in described(embed)
        assert "*The queue is empty.*" in described(embed)

    def test_song_count_in_header(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=120
                )
            )
        embed = music_player.queue_embed()
        assert "Songs: **3**" in described(embed)

    def test_total_duration_in_header_when_all_known(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=90)
        )
        embed = music_player.queue_embed()
        assert "Total Duration: **3m**" in described(embed)
        assert "~" not in described(embed).split("Total Duration:")[1].split("\n")[0]

    def test_total_duration_partial_when_some_unknown(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=None)
        )
        embed = music_player.queue_embed()
        assert "~" in described(embed)

    def test_total_duration_partial_with_ytsource(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.queue._display.append(
            YTSource(ytsearch="ytsearch:unresolved", process=True)
        )
        embed = music_player.queue_embed()
        assert "~" in described(embed)

    def test_song_title_appears_in_description(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert "Test Song" in described(embed)

    def test_song_duration_appears_when_known(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert "`3:30`" in described(embed)

    def test_song_duration_unknown_shows_placeholder(
        self, music_player: MusicPlayer, queue_obj_no_meta: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj_no_meta)
        embed = music_player.queue_embed()
        assert "`?:??`" in described(embed)

    def test_uploader_shown_when_known(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert "Test Channel" in described(embed)

    def test_unknown_channel_shown_when_uploader_none(
        self, music_player: MusicPlayer, queue_obj_no_meta: QueueObject
    ) -> None:
        music_player.queue._display.append(queue_obj_no_meta)
        embed = music_player.queue_embed()
        assert "Unknown channel" in described(embed)

    def test_est_playing_at_present_for_each_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.queue_embed()
        assert described(embed).count("Est. playing at") == 3

    def test_uncertain_prefix_after_no_duration_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=60)
        )
        embed = music_player.queue_embed()
        # First song: no preceding unknown → no ~
        # Second song: preceding song had unknown duration → ~
        lines = described(embed).split("\n")
        est_lines = [line for line in lines if "Est. playing at" in line]
        assert not est_lines[0].startswith("~") or "~**" not in est_lines[0]
        assert "~**" in est_lines[1]

    def test_uncertain_when_current_song_has_no_duration_secs(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        mock_current = MagicMock()
        mock_current.duration_secs = 0
        music_player.current_song = mock_current
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        embed = music_player.queue_embed()
        assert "~**" in described(embed)

    def test_caps_display_at_ten_songs(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(15):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.queue_embed()
        assert described(embed).count("Est. playing at") == 10

    def test_shows_more_indicator_when_over_ten(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(15):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.queue_embed()
        assert "... and 5 more" in described(embed)

    def test_ytsource_shows_resolving(self, music_player: MusicPlayer) -> None:
        music_player.queue._display.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player.queue_embed()
        assert "resolving..." in described(embed)


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

        music_player.queue._display.append(
            QueueObject(
                "https://yt.com/v=1", "Song 1", mock_song.requester, duration=600
            )
        )
        later_eta = music_player.estimated_playing_at()

        assert empty_eta != later_eta

    def test_uncertain_when_queued_song_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_matches_last_queue_line_eta(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """estimated_playing_at() should reflect the same seed used by
        queue_embed()/_build_next_up_embed() for consistency across embeds."""
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        eta = music_player.estimated_playing_at()

        # A song appended now would start right where the last queued line's
        # ETA ends up, so re-derive it via the same line formatter for index 2.
        now_pst, walk = music_player._queue_eta_seed()
        _, walk = music_player._format_queue_line(
            music_player.queue._display[0], 1, now_pst, walk
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

    def test_default_width_is_bar_width_constant(self) -> None:
        # Pins the default to the constant rather than a literal, so changing
        # _BAR_WIDTH stays a one-line edit but an accidental drift in the
        # signature's default doesn't go unnoticed.
        bar = _build_progress_bar(0.0, 200)
        assert bar.count("🟦") + bar.count("🔘") + bar.count("⬜") == _BAR_WIDTH

    def test_includes_formatted_elapsed_and_duration(self) -> None:
        bar = _build_progress_bar(65.0, 200)
        assert "`1:05`" in bar
        assert "`3:20`" in bar

    def test_elapsed_label_clamped_to_duration(self) -> None:
        """The left time label must never overshoot the right one — imprecise
        duration metadata plus a -ss start offset can push the raw position
        past the reported duration (e.g. `4:05 … 4:02`)."""
        bar = _build_progress_bar(250.0, 200, width=10)
        assert bar.startswith("`3:20`")
        assert "`4:10`" not in bar

    def test_elapsed_label_clamped_to_zero_when_negative(self) -> None:
        bar = _build_progress_bar(-5.0, 200, width=10)
        assert bar.startswith("`0:00`")


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
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.requester.mention in embed.description

    def test_embed_color_is_green(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.colour == discord.Color.green()

    def test_embed_title_links_to_youtube(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        # The title carries the URL, so no separate "Youtube link" field.
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.url == mock_song.webpage_url
        assert "Youtube link" not in [f.name for f in embed.fields]

    def test_embed_title_has_no_markdown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        # Discord renders embed titles literally — "**Now playing:**" would
        # show its asterisks, inside the title's link text.
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.title is not None
        assert "*" not in embed.title
        assert embed.title.startswith("Now playing: ")

    def test_embed_title_truncated_to_discord_limit(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        # An over-length title 400s the whole send, not just the title.
        mock_song.title = "x" * 400
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.title is not None
        assert len(embed.title) == 256
        assert embed.title.endswith("…")

    def test_embed_fields_are_exactly_one_inline_row(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        # Three inline fields — Discord's per-row cap — so they render as one
        # clean row. Duration is not among them: the progress bar's right-hand
        # label already shows it.
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert field_names == ["Channel", "Views", "Likes"]
        assert all(f.inline for f in embed.fields)

    def test_empty_field_values_get_placeholder(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        # Discord rejects an empty field value with a 400 that fails the whole
        # send. Views/likes are routinely absent (livestreams, hidden counts).
        mock_song.views = None
        mock_song.likes = None
        mock_song.uploader = None
        embed = music_player._build_now_playing_embed(mock_song)
        assert [f.value for f in embed.fields] == ["—", "—", "—"]
        assert all(f.value for f in embed.fields)

    def test_embed_thumbnail_is_set(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.thumbnail.url == mock_song.thumbnail

    def test_embed_footer_contains_bitrate_info(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.footer.text is not None
        assert str(mock_song.abr) in embed.footer.text
        assert str(mock_song.acodec) in embed.footer.text

    def test_embed_does_not_have_dislikes_field(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Dislikes" not in field_names

    def test_zero_views_and_likes_render_as_zero_not_blank(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A legitimate 0 must render as "0", not collapse to an empty field
        (the `str(x or "")` bug this shared extraction fixed)."""
        mock_song.views = 0
        mock_song.likes = 0
        embed = music_player._build_now_playing_embed(mock_song)
        fields_by_name = {f.name: f.value for f in embed.fields}
        assert fields_by_name["Views"] == "0"
        assert fields_by_name["Likes"] == "0"

    def test_embed_thumbnail_not_set_when_none(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.thumbnail = None
        embed = music_player._build_now_playing_embed(mock_song)
        assert not embed.thumbnail.url

    def test_description_has_estimated_finish_when_duration_known(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish:" in described(embed)

    def test_estimated_finish_appears_after_requester_on_same_line(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The requester/finish-time line stays on one line — the progress bar
        (Design §2 of the progress-bar plan) sits above it as its own line, not
        interleaved with it."""
        embed = music_player._build_now_playing_embed(mock_song)
        requester_line = described(embed).split("\n")[-1]
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
        lines = described(embed).split("\n")
        assert "🔘" in lines[0]
        assert lines[2].startswith("Requester:")

    def test_blank_line_separates_bar_from_requester_line(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        lines = described(embed).split("\n")
        assert lines[1] == ""

    def test_no_estimated_finish_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish" not in described(embed)

    def test_progress_bar_line_present_when_duration_known(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "🔘" in described(embed)
        assert "\n" in described(embed)  # progress bar is on its own line

    def test_progress_bar_reflects_elapsed_secs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 105.0  # roughly halfway through 210s
        embed = music_player._build_now_playing_embed(mock_song)
        assert fmt_duration(105) in described(embed)

    def test_no_progress_bar_line_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "🔘" not in described(embed)

    def test_position_override_replaces_live_position(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Used by _finalize_now_playing() to render the bar fully completed
        once a song has ended, regardless of song.position_secs's live value."""
        mock_song.elapsed_secs = 30.0
        mock_song.duration_secs = 210
        embed = music_player._build_now_playing_embed(
            mock_song, position_override=210.0
        )
        assert fmt_duration(210) in described(embed)
        assert fmt_duration(30) not in described(embed)

    def test_no_override_falls_back_to_live_position(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert fmt_duration(30) in described(embed)

    def test_progress_bar_includes_start_offset(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A ?t= song or a crash-recovered song resumed mid-stream via FFmpeg
        -ss renders its true audio position (start_offset + elapsed_secs) —
        all position surfaces read YTDL.position_secs, so the bar can't
        disagree with the pause embed or the Activity tooltip."""
        mock_song.start_offset = 60
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert fmt_duration(90) in described(embed)
        assert fmt_duration(30) not in described(embed)


class TestBuildPauseConfirmationEmbed:
    """Slim by design: the -pause response message hosts the live NP block
    directly below this embed (MusicContext attach), so the bar, requester,
    link fields, and thumbnail would all render twice if repeated here. The
    embed carries only what the NP block doesn't: the paused state and the
    exact pause position."""

    def test_returns_none_when_no_current_song(self, music_player: MusicPlayer) -> None:
        music_player.current_song = None
        assert music_player.build_pause_confirmation_embed() is None

    def test_returns_discord_embed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert isinstance(embed, discord.Embed)

    def test_title_contains_song_title(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert mock_song.title in embed.title

    def test_color_is_orange(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert embed.colour == discord.Color.orange()

    def test_paused_at_reflects_elapsed_secs(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 65.0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        # position 1:05 of total 3:30
        assert "Paused at: `1:05 / 3:30`" in described(embed)

    def test_paused_at_includes_start_offset(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A song resumed mid-stream via FFmpeg -ss reports true audio position
        (YTDL.position_secs), not just elapsed_secs."""
        mock_song.start_offset = 60
        mock_song.elapsed_secs = 65.0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        # position = 60 + 65 = 125s = 2:05
        assert "Paused at: `2:05 / 3:30`" in described(embed)

    def test_paused_at_omits_total_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 65.0
        mock_song.duration_secs = 0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert "Paused at: `1:05`" in described(embed)
        assert "/" not in described(embed).split("Paused at:")[1].split("\n")[0]

    def test_no_progress_bar(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.elapsed_secs = 65.0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert "🔘" not in described(embed)

    def test_no_requester_line(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert mock_song.requester.mention not in described(embed)

    def test_no_fields(self, music_player: MusicPlayer, mock_song: MagicMock) -> None:
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert embed.fields == []

    def test_no_thumbnail(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed is not None
        assert not embed.thumbnail.url


class TestUpdateActivity:
    async def test_sets_playing_activity_when_song_playing(
        self, music_player: MusicPlayer, mock_song: MagicMock
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
        self, music_player: MusicPlayer, mock_song: MagicMock
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
        mocked(music_player.bot).voice_clients = []
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
        mocked(music_player.bot).voice_clients = [active_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_not_awaited()

    async def test_resets_when_voice_clients_present_but_not_playing(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        idle_vc = MagicMock(spec=discord.VoiceClient)
        idle_vc.is_playing.return_value = False
        mocked(music_player.bot).voice_clients = [idle_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_awaited_once()

    async def test_resets_while_own_client_is_still_playing(
        self, music_player: MusicPlayer
    ) -> None:
        """-stop reached here with the presence stuck on the stopped song:
        cleanup() cancels the playback loop *before* it disconnects, so the
        loop's CancelledError handler calls update_activity(None) while this
        guild's own client is still connected and playing. The "another guild is
        playing" gate must not count our own client, or the reset never fires.
        """
        music_player.bot.change_presence = AsyncMock()
        own_vc = MagicMock(spec=discord.VoiceClient)
        own_vc.is_playing.return_value = True
        own_vc.guild = music_player._guild
        mocked(music_player.bot).voice_clients = [own_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_awaited_once()
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert isinstance(activity, discord.Game)

    async def test_skips_reset_when_own_client_stops_but_another_guild_plays(
        self, music_player: MusicPlayer
    ) -> None:
        """The own-guild exclusion must not go so far as to reset the presence
        out from under a different guild that is still playing.
        """
        music_player.bot.change_presence = AsyncMock()
        own_vc = MagicMock(spec=discord.VoiceClient)
        own_vc.is_playing.return_value = True
        own_vc.guild = music_player._guild
        other_vc = MagicMock(spec=discord.VoiceClient)
        other_vc.is_playing.return_value = True
        mocked(music_player.bot).voice_clients = [own_vc, other_vc]
        await music_player.update_activity(None)
        music_player.bot.change_presence.assert_not_awaited()

    async def test_falls_back_to_a_song_when_title_is_none(
        self, music_player: MusicPlayer, mock_song: MagicMock
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

    async def test_backdate_includes_start_offset(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A ?t=/crash-recovered song's tooltip must agree with the progress
        bar: start is backdated by position_secs (start_offset + elapsed), so
        Discord shows e.g. 1:30 elapsed, not 0:30."""
        music_player.bot.change_presence = AsyncMock()
        mock_song.start_offset = 60
        mock_song.elapsed_secs = 30.0
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        now_ms = int(time.time() * 1000)
        assert activity.timestamps["start"] <= now_ms - 90_000 + 1000
        assert activity.timestamps["start"] >= now_ms - 90_000 - 2000


class TestUpdateActivityPause:
    """Design review (2026-07-01): update_activity() previously set timestamps
    once at song start and never accounted for pause state at all."""

    async def test_omits_timestamps_entirely_while_paused(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.bot.change_presence = AsyncMock()
        mocked(music_player._guild.voice_client).is_paused.return_value = True
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.timestamps == {}

    async def test_still_sets_name_and_state_while_paused(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Only the ticking timestamps are dropped — the rest of the activity
        (title/uploader/state) still renders while paused."""
        music_player.bot.change_presence = AsyncMock()
        mocked(music_player._guild.voice_client).is_paused.return_value = True
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.name == f"{mock_song.title} · {mock_song.uploader}"
        assert activity.state == mock_song.duration

    async def test_resumed_timestamps_reflect_elapsed_not_full_duration(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """On resume, elapsed_secs already reflects time played before the pause
        (Design §1 — YTDL.read() counting freezes during a pause), so a normal
        (non-paused) update_activity() call after resume must still backdate
        `start` by that elapsed time rather than restarting the countdown."""
        music_player.bot.change_presence = AsyncMock()
        mocked(music_player._guild.voice_client).is_paused.return_value = False
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
        assert len(music_player.queue._display) == 0

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
    async def test_redis_push_history_unbounded(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        # Full history retention: the Redis list must never be trimmed
        # (docs/HISTORY_OVERHAUL_PLAN.md §4).
        assert music_player.store is not None
        for i in range(55):
            await music_player.store.push_history(
                HistoryEntry(title=f"Song {i}", webpage_url=f"url{i}")
            )
        items = await fake_redis.lrange(music_player.store.history_key(), 0, -1)
        assert len(items) == 55

    async def test_store_set_volume_updates_volume(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await music_player.store.set_volume(0.75)
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_redis_pop_queue_removes_first_item(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await fake_redis.rpush(music_player.store.queue_key(), b"item1")
        await fake_redis.rpush(music_player.store.queue_key(), b"item2")
        await music_player.store.pop_queue()
        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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
        assert mp.store is None


# ── PlaybackGate ──────────────────────────────────────────────────────────────


class TestReachedEnd:
    """_reached_end decides whether the Now Playing bar is finalized to 100%.
    Answering by position covers every early-termination cause at once — -skip,
    interjection, and mid-song stream death (docs/PLAY_WHILE_PAUSED_PLAN.md §5)."""

    def _song(self, position: float, duration: int) -> MagicMock:
        song = MagicMock()
        song.position_secs = position
        song.duration_secs = duration
        return song

    def test_played_to_the_end(self) -> None:
        assert _reached_end(self._song(210.0, 210)) is True

    def test_within_margin_counts_as_complete(self) -> None:
        """yt-dlp's duration metadata drifts from real stream length; a song
        that played out fully must still render a full bar."""
        assert _reached_end(self._song(206.0, 210)) is True

    def test_just_outside_margin_is_incomplete(self) -> None:
        assert _reached_end(self._song(204.0, 210)) is False

    def test_skipped_early_is_incomplete(self) -> None:
        assert _reached_end(self._song(20.0, 210)) is False

    def test_overshoot_is_complete(self) -> None:
        """position can exceed duration slightly when metadata understates."""
        assert _reached_end(self._song(212.0, 210)) is True

    def test_unknown_duration_is_incomplete(self) -> None:
        """No bar was ever shown — nothing to complete."""
        assert _reached_end(self._song(50.0, 0)) is False


class TestFinalizeCompletion:
    """The finalize edit fires either way; only the rendered position differs.
    Skipping the edit entirely would leave the bar frozen up to one 3s progress
    tick BEFORE the interruption, rather than at the true stop point."""

    async def test_completed_renders_full_bar(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = MagicMock(spec=discord.Message)
        with patch.object(MusicPlayer, "_push_np_edit", new=AsyncMock()) as push:
            await music_player._finalize_now_playing(
                mock_song, message, [], completed=True
            )
        push_call = push.await_args
        assert push_call is not None
        assert push_call.kwargs["position_override"] == mock_song.duration_secs

    async def test_incomplete_renders_true_position(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """position_override=None makes _build_now_playing_embed fall back to
        the live position_secs — frozen at the stop (or pause) point."""
        message = MagicMock(spec=discord.Message)
        with patch.object(MusicPlayer, "_push_np_edit", new=AsyncMock()) as push:
            await music_player._finalize_now_playing(
                mock_song, message, [], completed=False
            )
        push_call = push.await_args
        assert push_call is not None
        assert push_call.kwargs["position_override"] is None

    async def test_defaults_to_completed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = MagicMock(spec=discord.Message)
        with patch.object(MusicPlayer, "_push_np_edit", new=AsyncMock()) as push:
            await music_player._finalize_now_playing(mock_song, message, [])
        push_call = push.await_args
        assert push_call is not None
        assert push_call.kwargs["position_override"] == mock_song.duration_secs

    async def test_no_edit_without_duration(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        message = MagicMock(spec=discord.Message)
        with patch.object(MusicPlayer, "_push_np_edit", new=AsyncMock()) as push:
            await music_player._finalize_now_playing(
                mock_song, message, [], completed=False
            )
        push.assert_not_awaited()


class TestPlaybackGate:
    """Restoring the persisted queue and playing it are separate concerns —
    docs/PLAYBACK_GATE_PLAN.md."""

    async def test_gate_closed_at_construction(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        assert not mp._playback_gate.is_set()

    async def test_start_opens_gate_when_already_connected(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        """Crash recovery connects to voice BEFORE start() — that path must keep
        resuming from the head with no extra call site."""
        mock_guild.voice_client = MagicMock(spec=discord.VoiceClient)
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        # Stub loop() at the class: a real coroutine handed to a MagicMock
        # create_task is never awaited, and the "coroutine was never awaited"
        # finalizer surfaces as an unraisable warning in a later test.
        with (
            patch.object(mock_bot, "loop", MagicMock()),
            patch.object(MusicPlayer, "loop", MagicMock()),
        ):
            mp.start()
        assert mp._playback_gate.is_set()

    async def test_start_leaves_gate_closed_when_disconnected(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mock_guild.voice_client = None
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        with (
            patch.object(mock_bot, "loop", MagicMock()),
            patch.object(MusicPlayer, "loop", MagicMock()),
        ):
            mp.start()
        assert not mp._playback_gate.is_set()

    async def test_hold_suppresses_open(self, music_player: MusicPlayer) -> None:
        """-join opens the gate the moment the handshake lands; -play holds it
        shut across that join so the restored head cannot start before the
        requested song is inserted in front of it."""
        music_player._playback_gate.clear()
        async with music_player.defer_playback():
            music_player.open_playback_gate()  # join's call, while play holds
            assert not music_player._playback_gate.is_set()
        assert music_player._playback_gate.is_set()

    async def test_hold_opens_gate_even_when_block_raises(
        self, music_player: MusicPlayer
    ) -> None:
        """Fallback: resume the persisted queue rather than strand it behind a
        closed gate if play's error path ever skips cleanup()."""
        music_player._playback_gate.clear()
        with pytest.raises(ValueError):
            async with music_player.defer_playback():
                raise ValueError("boom")
        assert music_player._playback_gate.is_set()

    async def test_nested_holds_open_only_on_last_release(
        self, music_player: MusicPlayer
    ) -> None:
        music_player._playback_gate.clear()
        async with music_player.defer_playback():
            async with music_player.defer_playback():
                pass
            assert not music_player._playback_gate.is_set()
        assert music_player._playback_gate.is_set()

    async def test_loop_does_not_dequeue_while_gate_closed(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """The bug this exists to prevent: cog_before_invoke builds a player for
        every command — including ones validate_commands is about to reject — and
        that player used to walk the persisted queue and discard it entry by
        entry against a nonexistent voice client."""
        music_player._playback_gate.clear()
        await music_player.queue.put(
            [QueueObject("https://yt.com/v=1", "Persisted Song", mock_author)]
        )

        task = asyncio.create_task(music_player.loop())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert not task.done()
        assert music_player.queue.qsize() == 1
        assert music_player.current_song is None

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_wait_for_restore_blocks_until_restore_completes(
        self, music_player: MusicPlayer
    ) -> None:
        """The load-bearing ordering guarantee of the front-insert path: -play
        must not touch the queue before _restore_state() has read its snapshot,
        or put_front's LPUSH lands in that snapshot and gets queued twice."""
        music_player._restore_complete.clear()
        waiter = asyncio.create_task(music_player.wait_for_restore())
        await asyncio.sleep(0)
        assert not waiter.done()

        music_player._restore_complete.set()
        await asyncio.wait_for(waiter, timeout=1)
        assert waiter.done()

    async def test_gate_timeout_tears_down_player(
        self, music_player: MusicPlayer
    ) -> None:
        """A player blocked on the gate is NOT blocked in queue_get(), so the
        idle-disconnect never fires for it — the gate needs its own timeout or
        the mps entry and task leak forever."""
        music_player._playback_gate.clear()
        # stop() itself is a slot-less method on a __slots__ class — patch what
        # it delegates to (cleanup cancels the tasks and drops the mps entry).
        music_player._cog.cleanup = AsyncMock()

        with patch("src.musicplayer._PLAYBACK_GATE_TIMEOUT", 0.01):
            await music_player.loop()

        await asyncio.sleep(0.05)
        music_player._cog.cleanup.assert_awaited_once_with(music_player._guild)


class TestQueuePutFront:
    """MusicPlayer.queue_put_front — the -play-on-a-disconnected-bot path
    (docs/PLAYBACK_GATE_PLAN.md §3.5). The list branch is the playlist case,
    which front-inserts in full rather than collapsing to one track."""

    @pytest.fixture(autouse=True)
    def _stub_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_single_item_goes_to_the_head(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        await music_player.queue_put(
            [QueueObject("https://yt.com/v=old", "Old", mock_author)]
        )
        await music_player.queue_put_front(
            QueueObject("https://yt.com/v=new", "New", mock_author)
        )

        assert [queue_object(i).title for i in music_player.queue.display_items()] == [
            "New",
            "Old",
        ]

    async def test_playlist_preserves_order_on_both_legs(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """LPUSH pushes each successive argument to the head, so a naive batch
        push would reverse the playlist. push_queue_front reverses first —
        this pins that, since a 3+ item front insert had no coverage."""
        assert music_player.store is not None
        await music_player.queue_put(
            [QueueObject("https://yt.com/v=old", "Old", mock_author)]
        )
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_author)
            for i in range(3)
        ]

        await music_player.queue_put_front(tracks, prefetch=False)

        assert [queue_object(i).title for i in music_player.queue.display_items()] == [
            "Track 0",
            "Track 1",
            "Track 2",
            "Old",
        ]
        stored = [
            orjson.loads(raw)["title"]
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert stored == ["Track 0", "Track 1", "Track 2", "Old"]

    async def test_prefetches_each_queue_object(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        from src import youtube

        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_author)
            for i in range(2)
        ]
        with patch.object(
            youtube.YTDL, "prefetch_stream", new=AsyncMock()
        ) as mock_prefetch:
            await music_player.queue_put_front(tracks)
            await asyncio.sleep(0)

        assert mock_prefetch.await_count == 2

    async def test_prefetch_false_spawns_nothing(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """Bulk playlist inserts skip prefetch: N concurrent extractions
        saturate the thread pool and mint URLs that expire before playback."""
        from src import youtube

        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_author)
            for i in range(2)
        ]
        with patch.object(
            youtube.YTDL, "prefetch_stream", new=AsyncMock()
        ) as mock_prefetch:
            await music_player.queue_put_front(tracks, prefetch=False)
            await asyncio.sleep(0)

        mock_prefetch.assert_not_awaited()

    async def test_ytsource_items_are_not_prefetched(
        self, music_player: MusicPlayer
    ) -> None:
        """YTSource has no stable webpage_url at enqueue time — same rule
        queue_put follows."""
        from src import youtube

        with patch.object(
            youtube.YTDL, "prefetch_stream", new=AsyncMock()
        ) as mock_prefetch:
            await music_player.queue_put_front(
                [YTSource(ytsearch="ytsearch:a song", process=True)]
            )
            await asyncio.sleep(0)

        mock_prefetch.assert_not_awaited()


# ── StateRestore ──────────────────────────────────────────────────────────────


class TestStateRestore:
    async def test_restore_populates_queue(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        assert music_player.store is not None
        item = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=abc",
                "title": "Restored Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player.store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()
        assert music_player.queue.qsize() == 1
        assert isinstance(music_player.queue._display[0], QueueObject)
        assert music_player.queue._display[0].title == "Restored Song"

    async def test_restore_sets_volume(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.5")
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

    async def test_restore_reads_everything_in_one_snapshot_call(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        """State, queue, now-playing, and history all ride the single
        pipelined get_playback_snapshot() read — guard against a future edit
        reintroducing per-key reads (recovery was 3 round trips per guild
        before the snapshot absorbed now_playing/history)."""
        assert music_player.store is not None
        snapshot_spy = AsyncMock(wraps=music_player.store.get_playback_snapshot)
        get_np_spy = AsyncMock(wraps=music_player.store.get_now_playing)
        get_history_spy = AsyncMock(wraps=music_player.store.get_history)
        with (
            patch.object(music_player.store, "get_playback_snapshot", snapshot_spy),
            patch.object(music_player.store, "get_now_playing", get_np_spy),
            patch.object(music_player.store, "get_history", get_history_spy),
        ):
            await music_player._restore_state()
        snapshot_spy.assert_awaited_once()
        get_np_spy.assert_not_awaited()
        get_history_spy.assert_not_awaited()

    async def test_restore_populates_history_from_snapshot(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        old = HistoryEntry(title="Old Song", webpage_url="url1", played_at=1.0)
        new = HistoryEntry(title="New Song", webpage_url="url2", played_at=2.0)
        await music_player.store.push_history(old)
        await music_player.store.push_history(new)
        await music_player._restore_state()
        assert list(music_player.history) == [old, new]  # oldest first

    async def test_restore_populates_play_message_from_snapshot(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.now_playing_key(), b"title", b"Crashed Song"
        )
        await music_player._restore_state()
        assert music_player.play_message is not None
        assert music_player.play_message.title is not None
        assert "Crashed Song" in music_player.play_message.title


# ── RestoreCrashedSong ────────────────────────────────────────────────────────


class TestRestoreCrashedSong:
    async def test_crashed_song_requeued_at_front(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed Song"
        )
        normal_item = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=normal",
                "title": "Normal Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player.store.queue_key(), normal_item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 2
        first = await music_player.queue.get()
        assert queue_object(first).webpage_url == "https://yt.com/v=crash"
        assert queue_object(first).title == "Crashed Song"

    async def test_crashed_song_state_cleared_after_restore(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed Song"
        )
        music_player._guild.get_member = MagicMock(return_value=None)

        await music_player._restore_state()

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"current_song_url" not in state
        assert b"current_song_title" not in state

    async def test_crashed_song_restores_duration_and_uploader(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed Song"
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_duration", b"240"
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_uploader", b"Test Channel"
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert queue_object(first).duration == 240
        assert queue_object(first).uploader == "Test Channel"

    async def test_crashed_song_url_cleared_even_when_requester_unresolvable(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        """When guild.me and guild.owner are both None, the crashed song cannot be
        re-queued — but current_song_url must still be cleared to avoid an infinite
        retry loop on every subsequent restart."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Ghost Song"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        mocked(music_player._guild).me = None
        mocked(music_player._guild).owner = None

        await music_player._restore_state()

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"current_song_url" not in state
        assert b"current_song_title" not in state
        # Song was not re-queued since requester was unresolvable.
        assert music_player.queue.empty()

    async def test_no_crash_song_when_state_empty(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        assert music_player.store is not None
        normal_item = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=abc",
                "title": "Normal",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player.store.queue_key(), normal_item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 1
        first = await music_player.queue.get()
        assert queue_object(first).title == "Normal"

    async def test_crashed_song_resolves_requester_from_requester_id(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """current_song_requester_id (persisted atomically with the song at
        start-transaction time) resolves to the guild member who requested it."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed"
        )
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_requester_id",
            str(mock_author.id).encode(),
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        music_player._guild.get_member.assert_called_once_with(mock_author.id)
        first = await music_player.queue.get()
        assert queue_object(first).requester is mock_author

    async def test_crashed_song_falls_back_to_guild_me_without_requester_id(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """State without current_song_requester_id (or a departed member) falls
        back to guild.me so the song is still re-queued."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        bot_member = MagicMock(spec=discord.Member)
        mocked(music_player._guild).me = bot_member
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert queue_object(first).requester is bot_member

    async def test_crashed_song_computes_position_from_play_epoch(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """play_start_epoch and total_pause_seconds are combined into a seek offset."""
        assert music_player.store is not None
        import time

        start = time.time() - 90  # started 90 seconds ago, 10s of pauses
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed"
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"play_start_epoch", str(start).encode()
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"total_pause_seconds", b"10"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        # expected position ≈ 90 - 10 = 80s; allow ±10s tolerance for test latency
        assert first.ts is not None
        assert 70 <= first.ts <= 90

    async def test_crashed_song_position_none_when_no_epoch(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """When play_start_epoch is absent, ts on the restored QueueObject is None."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts is None

    async def test_crashed_song_position_accounts_for_active_pause(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """When the bot crashed while paused, pause_start_epoch contributes to total pause
        time and is subtracted from the seek position alongside total_pause_seconds."""
        assert music_player.store is not None
        import time

        play_start = time.time() - 90  # song started 90 s ago
        pause_start = time.time() - 20  # paused 20 s ago (still paused at crash)
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Paused Crash"
        )
        await fake_redis.hset(
            music_player.store.state_key(),
            b"play_start_epoch",
            str(play_start).encode(),
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"total_pause_seconds", b"10"
        )
        await fake_redis.hset(
            music_player.store.state_key(),
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

    async def test_crashed_song_position_capped_by_cached_stream_duration(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """The recovery position is capped at cached stream duration − 10s so
        FFmpeg never seeks past EOF."""
        assert music_player.store is not None
        import time

        start = time.time() - 90  # computed position ≈ 90s
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed"
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"play_start_epoch", str(start).encode()
        )
        # Cached stream metadata says the song is only 60s long.
        await fake_redis.set(
            "ytdl:stream:https://yt.com/v=crash", orjson.dumps({"duration": 60})
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts == 50  # min(≈90, 60 − 10)

    async def test_crashed_song_position_uncapped_when_cached_duration_malformed(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """A malformed cached stream duration degrades to "no cap" — the
        computed position is kept and the restore still completes (clears the
        crashed-song state) instead of aborting."""
        assert music_player.store is not None
        import time

        start = time.time() - 90
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed"
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"play_start_epoch", str(start).encode()
        )
        await fake_redis.set(
            "ytdl:stream:https://yt.com/v=crash",
            orjson.dumps({"duration": "not-a-number"}),
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts is not None
        assert 80 <= first.ts <= 100  # uncapped ≈90s position survives
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"current_song_url" not in state  # restore completed and cleared


# ── RestoreCompleteEvent (loop guard) ─────────────────────────────────────────
# Regression coverage for a race where loop() could dequeue the crash-recovered
# "current song" _restore_state() injects and call pop_queue() (Redis LPOP) for
# it — silently deleting an unrelated, still-queued song from Redis, since the
# crashed song was never itself on the Redis queue list. loop() now waits on
# self._restore_complete, which _restore_state() sets only once it has finished.


class TestRestoreCompleteLoopGuard:
    async def test_restore_state_sets_restore_complete_on_success(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        music_player._restore_complete.clear()
        await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_restore_state_sets_restore_complete_on_failure(
        self, music_player: MusicPlayer
    ) -> None:
        # get_playback_snapshot() swallows Redis errors and returns None, so
        # the failure path here is the None early-return, not an exception.
        music_player._restore_complete.clear()
        with patch.object(
            music_player.store,
            "get_playback_snapshot",
            new=AsyncMock(return_value=None),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()
        # Restore aborted before touching the queue.
        assert music_player.queue.qsize() == 0

    async def test_restore_state_sets_restore_complete_when_no_store(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        await mp._restore_state()
        assert mp._restore_complete.is_set()

    async def test_loop_waits_for_restore_before_dequeuing(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """loop() must not call pop_queue() for the crash-recovered song until
        _restore_state() has fully populated the queue from Redis."""
        music_player._restore_complete.clear()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).return_value = False
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
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """End-to-end guard for the original bug: seed Redis with a crashed
        song plus 2 still-queued songs. After restore populates the queue and
        loop() processes exactly the crash-recovered song (its stream fails
        here, taking the "skip" path that also calls pop_queue()), both real
        queued songs must still be present in Redis — pop_queue() must not
        fire for the crashed song's own dequeue.
        """
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed Song"
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
            await fake_redis.rpush(music_player.store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player._restore_complete.clear()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player._restore_state()
        assert music_player.queue.qsize() == 3  # crashed + 2 real queued songs

        # Exactly one loop() iteration — enough to process the crashed song.
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(side_effect=lambda s: s)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=None)
            ),
        ):
            await music_player.loop()

        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(remaining) == 2

    async def test_shuffle_during_restore_window_does_not_orphan_redis_entry(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """End-to-end guard for Issue 1: if a user runs -shuffle while the
        crash-recovered song is still sitting in song_queue (before loop()
        has dequeued it), Redis's queue list must still end up with exactly
        the real queued songs — no phantom entry for the crashed song."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Crashed Song"
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
            await fake_redis.rpush(music_player.store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()
        assert music_player.queue.qsize() == 5  # crashed + 4 real queued songs

        # Simulates a -shuffle command running before loop() ever dequeues anything.
        await music_player.queue_shuffle()

        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert isinstance(mp, MusicPlayer)

    def test_sets_last_author_to_ctx_author(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert mp._last_author is mock_ctx.author

    def test_raises_if_guild_is_none(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        mock_ctx.guild = None
        with pytest.raises(AssertionError):
            MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)

    def test_attaches_store_when_redis_provided(
        self, mock_bot: MagicMock, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        mp = MusicPlayer.from_context(mock_bot, mock_ctx, redis=fake_redis)
        assert mp.store is not None


# ── Start ─────────────────────────────────────────────────────────────────────


class TestStart:
    def test_start_creates_player_and_restore_tasks(
        self, music_player: MusicPlayer
    ) -> None:
        # _restore_state() is scheduled before loop() — loop() waits on
        # self._restore_complete before its first dequeue, so restore must be
        # in flight first. See _restore_state()'s docstring for why.
        # Precondition, stated up front like the suite's other narrowing asserts:
        # the fixture wires a store, which is what makes start() take the restore
        # branch at all.
        assert music_player.store is not None
        restore_task = MagicMock(name="restore_task")
        player_task = MagicMock(name="player_task")
        returns = [restore_task, player_task]

        def _create(coro: Any) -> MagicMock:
            coro.close()
            return returns.pop(0)

        music_player.bot.loop = MagicMock()
        music_player.bot.loop.create_task = MagicMock(side_effect=_create)
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

    def test_restore_complete_set_immediately_when_store_absent(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        """When there is no Redis store, start() must signal _restore_complete immediately
        so loop()'s prefetch gate never blocks."""
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mock_bot.loop = MagicMock()
        mock_bot.loop.create_task = stub_create_task()
        mp.start()
        assert mp._restore_complete.is_set()

    def test_restore_complete_not_set_before_start_when_store_present(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """Before start() or _restore_state() runs, the event must be clear."""
        mp = MusicPlayer(
            mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis
        )
        assert not mp._restore_complete.is_set()


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


# ── RequireRequester ──────────────────────────────────────────────────────────


class TestRequireRequester:
    def test_returns_last_author(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player._last_author = mock_author
        assert music_player._require_requester() is mock_author

    def test_raises_when_no_author_resolved(self, music_player: MusicPlayer) -> None:
        """Reached only when guild.me AND guild.owner were both uncached at
        construction and no command has run since — QueueObject.requester is
        non-optional, so this must fail here rather than as an AttributeError
        on None inside serialization."""
        music_player._last_author = None
        with pytest.raises(RuntimeError, match="No requester available"):
            music_player._require_requester()


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
    ) -> AsyncGenerator[None]:
        """_send_now_playing() may spawn a real _progress_task (Design §4). Tests
        in this class don't drive loop() to retire it themselves, so clean it up
        here rather than leaking a pending asyncio.sleep() task past the test."""
        yield
        await music_player._cancel_progress_task()

    @pytest.fixture(autouse=True)
    def _live_song(self, music_player: MusicPlayer, mock_song: MagicMock) -> None:
        """_send_now_playing's embed block is built off current_song (shared
        with the MusicContext attach path) — loop() always sets it before
        calling, so mirror that here."""
        music_player.current_song = mock_song

    async def test_sends_embed_to_channel(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        await music_player._send_now_playing(mock_song)
        mocked(music_player._channel.send).assert_awaited_once()
        call_kwargs = mocked(music_player._channel.send).call_args[1]
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

    async def test_resets_stale_np_host_on_send_failure(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Regression (code review): a failed/partial send must not leave
        the NP host pointing at the *previous* song's message — otherwise a
        later mark_paused()/mark_resumed() on the new song would silently edit
        the wrong (old, already-finished) song's embed."""
        stale_message = MagicMock(spec=discord.Message)
        music_player._np_host_message = stale_message
        music_player._channel.send = AsyncMock(side_effect=Exception("channel gone"))
        await music_player._send_now_playing(mock_song)
        assert music_player._np_host_message is None

    async def test_sends_only_now_playing_embed_when_queue_empty(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        await music_player._send_now_playing(mock_song)
        call_kwargs = mocked(music_player._channel.send).call_args[1]
        assert len(call_kwargs["embeds"]) == 1
        assert call_kwargs["embeds"][0].colour == discord.Color.green()

    async def test_sends_next_up_embed_when_queue_has_song(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        await music_player._send_now_playing(mock_song)
        call_kwargs = mocked(music_player._channel.send).call_args[1]
        embeds = call_kwargs["embeds"]
        assert len(embeds) == 2
        assert embeds[1].colour == discord.Color.blue()
        assert embeds[1].title == "Up next"
        assert "Next Song" in embeds[1].description

    async def test_send_now_playing_works_without_store(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        mock_song: MagicMock,
    ) -> None:
        # The Redis now-playing snapshot is written by the start transaction in
        # loop(), not here — _send_now_playing only builds/sends the embed.
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp._channel = mock_channel
        mp.current_song = mock_song
        await mp._send_now_playing(mock_song)
        assert mp.play_message is not None

    async def test_adopts_sent_message_as_dedicated_host(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        sent_message = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent_message)
        await music_player._send_now_playing(mock_song)
        assert music_player._np_host_message is sent_message
        assert music_player._np_host_own_embeds == []
        assert music_player._np_host_dedicated is True

    async def test_sent_block_reuses_play_message_embed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The NP embed stored as play_message IS the one sent in the block —
        not an identical rebuild (branch review N3)."""
        await music_player._send_now_playing(mock_song)
        embeds = mocked(music_player._channel.send).call_args.kwargs["embeds"]
        assert embeds[0] is music_player.play_message

    async def test_starts_progress_task_for_normal_duration_song(
        self, music_player: MusicPlayer, mock_song: MagicMock
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
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is None

    async def test_progress_task_starts_for_exactly_5s_song(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 5
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is not None


# ── Now-playing host primitives (embed-attach plan §1–§4) ─────────────────────


class TestNpEmbedBlock:
    def test_empty_when_no_song(self, music_player: MusicPlayer) -> None:
        assert music_player.np_embed_block() == []

    def test_now_playing_only_when_queue_empty(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        block = music_player.np_embed_block()
        assert len(block) == 1
        assert block[0].colour == discord.Color.green()

    def test_np_then_next_up_ordering(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        block = music_player.np_embed_block()
        assert len(block) == 2
        assert block[0].colour == discord.Color.green()
        assert block[1].title == "Up next"


class TestNpHostAdoptRetire:
    def test_adopt_updates_state_synchronously(self, music_player: MusicPlayer) -> None:
        msg = MagicMock(spec=discord.Message)
        msg.id = 1
        own = [discord.Embed(title="Queue")]
        music_player._adopt_np_host(msg, own)
        assert music_player._np_host_message is msg
        assert music_player._np_host_own_embeds is own
        assert music_player._np_host_dedicated is False
        assert not music_player._background_tasks  # no old host → no retire

    async def test_adopt_retires_old_dedicated_host_with_delete(
        self, music_player: MusicPlayer
    ) -> None:
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        music_player._adopt_np_host(old, [], dedicated=True)
        new = AsyncMock(spec=discord.Message)
        new.id = 2
        music_player._adopt_np_host(new, [])
        await asyncio.gather(*list(music_player._background_tasks))
        old.delete.assert_awaited_once()
        old.edit.assert_not_awaited()

    async def test_adopt_strips_old_response_host_with_edit(
        self, music_player: MusicPlayer
    ) -> None:
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        old_own = [discord.Embed(title="Queue")]
        music_player._adopt_np_host(old, old_own)
        new = AsyncMock(spec=discord.Message)
        new.id = 2
        music_player._adopt_np_host(new, [], dedicated=True)
        await asyncio.gather(*list(music_player._background_tasks))
        old.edit.assert_awaited_once_with(embeds=old_own)
        old.delete.assert_not_awaited()

    async def test_adopt_same_message_retires_nothing(
        self, music_player: MusicPlayer
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        msg.id = 1
        music_player._adopt_np_host(msg, [])
        music_player._adopt_np_host(msg, [discord.Embed(title="p")])
        assert not music_player._background_tasks
        msg.delete.assert_not_awaited()
        msg.edit.assert_not_awaited()

    async def test_retire_swallows_not_found(self, music_player: MusicPlayer) -> None:
        msg = AsyncMock(spec=discord.Message)
        msg.delete.side_effect = discord.NotFound(MagicMock(), "gone")
        await music_player._retire_np_host(msg, [], True)  # must not raise

    async def test_retire_swallows_and_logs_http_exception(
        self, music_player: MusicPlayer
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        msg.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")
        await music_player._retire_np_host(msg, [], False)  # must not raise

    def test_release_clears_state_without_touching_message(
        self, music_player: MusicPlayer
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        music_player._np_host_message = msg
        music_player._np_host_own_embeds = [discord.Embed(title="p")]
        music_player._np_host_dedicated = True
        music_player._release_np_host()
        assert music_player._np_host_message is None
        assert music_player._np_host_own_embeds == []
        assert music_player._np_host_dedicated is False
        msg.delete.assert_not_awaited()
        msg.edit.assert_not_awaited()

    async def test_adopt_ignores_older_message_and_sheds_its_block(
        self, music_player: MusicPlayer
    ) -> None:
        """Two overlapping sends can return out of order (channel position is
        send-start order, adopts run in send-return order) — an older message
        adopting late would pull the block up from the true bottom. The adopt
        is ignored and the older message sheds the block it carries."""
        newer = AsyncMock(spec=discord.Message)
        newer.id = 2
        music_player._adopt_np_host(newer, [])
        older = AsyncMock(spec=discord.Message)
        older.id = 1
        older_own = [discord.Embed(title="Queue")]
        music_player._adopt_np_host(older, older_own)
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is newer
        older.edit.assert_awaited_once_with(embeds=older_own)
        newer.edit.assert_not_awaited()
        newer.delete.assert_not_awaited()

    async def test_retire_waits_for_lock_holder(
        self, music_player: MusicPlayer
    ) -> None:
        """Plan §4 lock ordering: a retire serializes behind _np_edit_lock, so
        an in-flight tick edit (which holds the lock across its await) always
        completes before the retire's strip/delete — the retire is the final
        write and a late tick can't resurrect the NP block on the old host."""
        order: list[str] = []
        old = AsyncMock(spec=discord.Message)

        async def _delete() -> None:
            order.append("retire")

        old.delete.side_effect = _delete

        async def _hold_lock_like_a_tick() -> None:
            async with music_player._np_edit_lock:
                order.append("edit_started")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                order.append("edit_finished")

        holder = asyncio.create_task(_hold_lock_like_a_tick())
        await asyncio.sleep(0)  # holder acquires the lock
        retire = asyncio.create_task(music_player._retire_np_host(old, [], True))
        await asyncio.gather(holder, retire)
        assert order == ["edit_started", "edit_finished", "retire"]


class TestAdoptNpHostIfCurrent:
    """The adopt gate closing the adopt-after-await race (branch review H1):
    a send crossing a song boundary must shed its now-stale block instead of
    adopting — adopting would delete the next song's freshly sent NP host, or
    leave a bogus frozen block nothing ever cleans up."""

    async def test_adopts_when_song_still_current(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        msg = AsyncMock(spec=discord.Message)
        msg.id = 1
        own = [discord.Embed(title="Queue")]
        assert music_player._adopt_np_host_if_current(msg, own, mock_song) is True
        assert music_player._np_host_message is msg
        msg.edit.assert_not_awaited()

    async def test_sheds_block_when_song_changed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = MagicMock()  # the next song took over
        msg = AsyncMock(spec=discord.Message)
        own = [discord.Embed(title="Queue")]
        assert music_player._adopt_np_host_if_current(msg, own, mock_song) is False
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None
        msg.edit.assert_awaited_once_with(embeds=own)  # strip back to own embeds

    async def test_deletes_stale_dedicated_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = None  # queue emptied while send was in flight
        msg = AsyncMock(spec=discord.Message)
        assert (
            music_player._adopt_np_host_if_current(msg, [], mock_song, dedicated=True)
            is False
        )
        await asyncio.gather(*list(music_player._background_tasks))
        msg.delete.assert_awaited_once()

    async def test_never_adopts_for_none_song(self, music_player: MusicPlayer) -> None:
        """A block can only have been built off a live song; a None song must
        never adopt even if current_song is also None."""
        msg = AsyncMock(spec=discord.Message)
        assert music_player._adopt_np_host_if_current(msg, [], None) is False
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None

    async def test_stale_adopt_does_not_disturb_new_songs_host(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Variant (a) of the race: song B's dedicated host is already up when
        song A's late send returns — B's host must survive untouched."""
        song_b = MagicMock()
        music_player.current_song = song_b
        host_b = AsyncMock(spec=discord.Message)
        host_b.id = 2
        music_player._adopt_np_host(host_b, [], dedicated=True)

        late = AsyncMock(spec=discord.Message)
        late.id = 3  # newer id — only the song gate protects host_b here
        music_player._adopt_np_host_if_current(late, [], mock_song)
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is host_b
        host_b.delete.assert_not_awaited()
        host_b.edit.assert_not_awaited()
        late.edit.assert_awaited_once_with(embeds=[])


class TestSendWithNp:
    async def test_attaches_block_and_adopts_when_song_live(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        sent = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent)
        notice = discord.Embed(title="Notice")

        message = await music_player.send_with_np(embed=notice)

        assert message is sent
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert embeds[0].colour == discord.Color.green()  # NP block leads
        assert embeds[1].title == "Notice"  # own embeds follow the block
        assert music_player._np_host_message is sent
        assert music_player._np_host_own_embeds == [notice]
        assert music_player._np_host_dedicated is False

    async def test_plain_send_when_no_song(self, music_player: MusicPlayer) -> None:
        sent = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent)
        await music_player.send_with_np("hello")
        args, kwargs = music_player._channel.send.call_args
        assert args == ("hello",)
        assert "embeds" not in kwargs
        assert music_player._np_host_message is None

    async def test_embed_send_without_song_does_not_adopt(
        self, music_player: MusicPlayer
    ) -> None:
        sent = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent)
        notice = discord.Embed(title="Notice")
        await music_player.send_with_np(embed=notice)
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert embeds == [notice]
        assert music_player._np_host_message is None

    async def test_content_and_embed_together_when_song_live(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Plain text + embed coexist on one message with the block leading."""
        music_player.current_song = mock_song
        sent = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent)
        notice = discord.Embed(title="Notice")
        await music_player.send_with_np("heads up", embed=notice)
        args, kwargs = music_player._channel.send.call_args
        assert args == ("heads up",)
        assert kwargs["embeds"][0].colour == discord.Color.green()
        assert kwargs["embeds"][-1].title == "Notice"
        assert music_player._np_host_message is sent

    async def test_song_ending_mid_send_sheds_block_instead_of_adopting(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """H1 at the send_with_np attach site: the song ends while the HTTP
        send is in flight — the sent message strips its stale block and the
        host stays released."""
        music_player.current_song = mock_song
        sent = AsyncMock(spec=discord.Message)

        async def _send_crossing_song_boundary(*args: Any, **kwargs: Any) -> MagicMock:
            music_player.current_song = None
            return sent

        music_player._channel.send = AsyncMock(side_effect=_send_crossing_song_boundary)
        await music_player.send_with_np(embed=discord.Embed(title="Notice"))
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None
        sent.edit.assert_awaited_once()  # stripped back to its own embeds


class TestRepinNowPlaying:
    async def test_false_when_no_song(self, music_player: MusicPlayer) -> None:
        assert await music_player.repin_now_playing() is False
        mocked(music_player._channel.send).assert_not_awaited()

    async def test_sends_dedicated_block_and_adopts(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        sent = MagicMock(spec=discord.Message)
        sent.id = 2
        music_player._channel.send = AsyncMock(return_value=sent)

        assert await music_player.repin_now_playing() is True
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert embeds[0].colour == discord.Color.green()
        assert music_player._np_host_message is sent
        assert music_player._np_host_dedicated is True

    async def test_delete_retires_previous_dedicated_host(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        music_player._adopt_np_host(old, [], dedicated=True)
        sent = MagicMock(spec=discord.Message)
        sent.id = 2
        music_player._channel.send = AsyncMock(return_value=sent)

        await music_player.repin_now_playing()
        await asyncio.gather(*list(music_player._background_tasks))
        old.delete.assert_awaited_once()

    async def test_false_when_song_ends_mid_send(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """H1 at the repin attach site: the song ends while the dedicated NP
        send is in flight — the stale message is deleted, nothing is adopted,
        and repin reports False so -now can respond another way."""
        music_player.current_song = mock_song
        sent = AsyncMock(spec=discord.Message)

        async def _send_crossing_song_boundary(*args: Any, **kwargs: Any) -> MagicMock:
            music_player.current_song = None
            return sent

        music_player._channel.send = AsyncMock(side_effect=_send_crossing_song_boundary)
        assert await music_player.repin_now_playing() is False
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None
        sent.delete.assert_awaited_once()

    async def test_does_not_touch_progress_task(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The running updater follows the host pointer — a re-pin must not
        cancel/restart it."""
        music_player.current_song = mock_song
        sentinel = MagicMock(spec=asyncio.Task)
        music_player._progress_task = sentinel
        sent = MagicMock(spec=discord.Message)
        sent.id = 2
        music_player._channel.send = AsyncMock(return_value=sent)
        await music_player.repin_now_playing()
        assert music_player._progress_task is sentinel
        music_player._progress_task = None  # sentinel isn't awaitable — reset directly


class TestRetireNpHostOnStop:
    """-stop / alone-disconnect teardown (branch review L4): the host is
    disposed of — unlike song end, which releases and leaves the completed bar
    as history, a bar frozen mid-song on a stopped player is misleading."""

    async def test_deletes_dedicated_host(self, music_player: MusicPlayer) -> None:
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        music_player._adopt_np_host(host, [], dedicated=True)
        await music_player.retire_np_host_on_stop()
        host.delete.assert_awaited_once()
        assert music_player._np_host_message is None

    async def test_strips_response_host_to_own_embeds(
        self, music_player: MusicPlayer
    ) -> None:
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        own = [discord.Embed(title="Queue")]
        music_player._adopt_np_host(host, own)
        await music_player.retire_np_host_on_stop()
        host.edit.assert_awaited_once_with(embeds=own)
        host.delete.assert_not_awaited()
        assert music_player._np_host_message is None

    async def test_noop_when_no_host(self, music_player: MusicPlayer) -> None:
        await music_player.retire_np_host_on_stop()  # must not raise


class TestRehostNpAfterResume:
    """-resume re-hosting (branch review M3): a command-response host —
    typically the -pause confirmation — is strip-retired in favor of a fresh
    dedicated NP message, so "⏸️ Paused at…" becomes plain history instead of
    being re-rendered beneath a live bar by every tick."""

    async def test_rehosts_when_response_hosts_the_block(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        pause_embed = discord.Embed(title="⏸️ Paused: x")
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        music_player._adopt_np_host(old, [pause_embed])
        sent = MagicMock(spec=discord.Message)
        sent.id = 2
        music_player._channel.send = AsyncMock(return_value=sent)

        await music_player.rehost_np_after_resume()
        await asyncio.gather(*list(music_player._background_tasks))

        assert music_player._np_host_message is sent
        assert music_player._np_host_dedicated is True
        old.edit.assert_awaited_once_with(embeds=[pause_embed])

    async def test_noop_when_host_is_dedicated(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A dedicated NP message has no stale state to shed — no extra send."""
        music_player.current_song = mock_song
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        music_player._adopt_np_host(host, [], dedicated=True)
        await music_player.rehost_np_after_resume()
        mocked(music_player._channel.send).assert_not_awaited()
        assert music_player._np_host_message is host

    async def test_noop_when_no_host(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        await music_player.rehost_np_after_resume()
        mocked(music_player._channel.send).assert_not_awaited()


class TestPushNpEditEmbedCap:
    async def test_truncates_to_ten_embeds_keeping_the_block(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """An attach accepted at Discord's 10-embed cap can overflow if a
        next-up embed appears later — the edit drops the own-embeds tail, never
        the block, instead of 400ing on every tick (branch review L5)."""
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        own = [discord.Embed(title=f"e{i}") for i in range(9)]
        message = AsyncMock(spec=discord.Message)
        assert await music_player._push_np_edit(mock_song, message, own) is True
        embeds = message.edit.call_args.kwargs["embeds"]
        assert len(embeds) == 10
        assert embeds[0].colour == discord.Color.green()  # NP block intact
        assert embeds[1].title == "Up next"
        assert embeds[-1].title == "e7"  # own-embeds tail dropped


class TestEditNowPlayingOnce:
    async def test_edits_host_with_own_embeds(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        host = AsyncMock(spec=discord.Message)
        own = [discord.Embed(title="Queue")]
        music_player._np_host_message = host
        music_player._np_host_own_embeds = own
        await music_player._edit_now_playing_once()
        embeds = host.edit.call_args.kwargs["embeds"]
        assert embeds[0].colour == discord.Color.green()  # NP block leads
        assert embeds[1].title == "Queue"  # host's own embeds follow

    async def test_releases_host_on_not_found(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        host = AsyncMock(spec=discord.Message)
        host.edit.side_effect = discord.NotFound(MagicMock(), "gone")
        music_player._np_host_message = host
        await music_player._edit_now_playing_once()
        assert music_player._np_host_message is None

    async def test_not_found_keeps_host_adopted_mid_edit(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Adopt is lock-free, so a command response can swap in a new host
        while this edit's PATCH is in flight. A NotFound then must not release
        the NEW host — that would permanently orphan its block (branch review
        M1)."""
        music_player.current_song = mock_song
        old_host = AsyncMock(spec=discord.Message)
        new_host = AsyncMock(spec=discord.Message)

        async def _edit_racing_an_adopt(*args: Any, **kwargs: Any) -> Never:
            music_player._np_host_message = new_host  # adopt lands mid-PATCH
            raise discord.NotFound(MagicMock(), "old host deleted")

        old_host.edit.side_effect = _edit_racing_an_adopt
        music_player._np_host_message = old_host
        await music_player._edit_now_playing_once()
        assert music_player._np_host_message is new_host

    async def test_noop_when_no_host(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        await music_player._edit_now_playing_once()  # must not raise

    async def test_noop_when_no_song(self, music_player: MusicPlayer) -> None:
        host = AsyncMock(spec=discord.Message)
        music_player._np_host_message = host
        await music_player._edit_now_playing_once()
        host.edit.assert_not_awaited()


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

        await music_player._finalize_now_playing(mock_song, message, [])

        message.edit.assert_awaited_once()
        embed = message.edit.call_args.kwargs["embeds"][0]
        assert fmt_duration(210) in embed.description
        assert fmt_duration(184) not in embed.description

    async def test_noop_when_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        mock_song.duration_secs = 0
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, [])
        message.edit.assert_not_awaited()

    async def test_includes_next_up_embed_when_queue_has_song(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, [])
        embeds = message.edit.call_args.kwargs["embeds"]
        assert len(embeds) == 2
        assert embeds[1].title == "Up next"

    async def test_preserves_captured_host_own_embeds(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A song that ended while a command response hosted the NP block must
        keep that response's own embeds after the completed bar."""
        own = [discord.Embed(title="Queue")]
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, own)
        embeds = message.edit.call_args.kwargs["embeds"]
        assert fmt_duration(mock_song.duration_secs) in embeds[0].description
        assert embeds[1].title == "Queue"

    async def test_swallows_not_found(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.NotFound(MagicMock(), "message deleted")
        await music_player._finalize_now_playing(
            mock_song, message, []
        )  # must not raise

    async def test_swallows_and_logs_http_exception(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")
        await music_player._finalize_now_playing(
            mock_song, message, []
        )  # must not raise

    async def test_operates_on_captured_song_and_message_args(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Must use the song/message passed in, not self.current_song /
        self._np_host_message — those may already point at the next song
        by the time this fire-and-forget task actually runs."""
        other_message = AsyncMock(spec=discord.Message)
        music_player.current_song = MagicMock()  # a different, "next" song
        music_player._np_host_message = other_message

        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, [])

        message.edit.assert_awaited_once()
        other_message.edit.assert_not_awaited()

    async def test_waits_for_lock_holder(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The finalize's completed-bar write must land AFTER any in-flight
        debounce-spawned edit (which holds _np_edit_lock across its PATCH) —
        otherwise a resume just before song end can freeze the historical bar
        short of 100% (branch review L2)."""
        order: list[str] = []
        message = AsyncMock(spec=discord.Message)

        async def _edit(*args: Any, **kwargs: Any) -> None:
            order.append("finalize")

        message.edit.side_effect = _edit

        async def _hold_lock_like_a_oneshot_edit() -> None:
            async with music_player._np_edit_lock:
                order.append("oneshot_started")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                order.append("oneshot_finished")

        holder = asyncio.create_task(_hold_lock_like_a_oneshot_edit())
        await asyncio.sleep(0)  # holder acquires the lock
        finalize = asyncio.create_task(
            music_player._finalize_now_playing(mock_song, message, [])
        )
        await asyncio.gather(holder, finalize)
        assert order == ["oneshot_started", "oneshot_finished", "finalize"]


class TestFireFinalizeNowPlaying:
    async def test_spawns_tracked_background_task(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        music_player._fire_finalize_now_playing(mock_song, message, [])
        task = next(iter(music_player._background_tasks))
        assert task in music_player._background_tasks
        await task
        message.edit.assert_awaited_once()
        assert task not in music_player._background_tasks


# ── ProgressUpdater ───────────────────────────────────────────────────────────


class TestProgressUpdater:
    @staticmethod
    def _make_sleep(n_ticks: int) -> Callable[[Any], Awaitable[None]]:
        """asyncio.sleep double that lets the loop run n_ticks times, then raises
        CancelledError — deterministic without waiting on the real interval."""
        calls = 0

        async def _sleep(_secs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls > n_ticks:
                raise asyncio.CancelledError()

        return _sleep

    @staticmethod
    def _host(music_player: MusicPlayer) -> AsyncMock:
        """Install an NP host message for the updater to edit."""
        message = AsyncMock(spec=discord.Message)
        music_player._np_host_message = message
        music_player._np_host_own_embeds = []
        return message

    async def test_ticks_and_edits_host_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        message = self._host(music_player)

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        message.edit.assert_awaited_once()
        assert "embeds" in message.edit.call_args.kwargs

    async def test_edits_follow_a_host_swap(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The tick must re-read the host pointer each pass — a -now re-pin or
        a command response adopting the host mid-song redirects the next tick
        to the new message with no updater restart."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        old_host = self._host(music_player)
        new_host = AsyncMock(spec=discord.Message)

        calls = 0

        async def _sleep(_secs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:  # swap between the first and second tick
                music_player._np_host_message = new_host
            if calls > 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", new=_sleep):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        old_host.edit.assert_awaited_once()
        new_host.edit.assert_awaited_once()

    async def test_skips_edit_while_paused(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = True
        mocked(music_player._guild).voice_client = vc
        message = self._host(music_player)

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        message.edit.assert_not_awaited()

    async def test_returns_when_song_changed_under_it(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """loop() owns cancellation on song transition, but this guard protects
        against a stray tick landing after the song changed (Design §4)."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = MagicMock()  # a different song than the one passed in
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        message = self._host(music_player)

        with patch("asyncio.sleep", new=AsyncMock()):
            await music_player._progress_updater(mock_song)  # returns, no raise

        message.edit.assert_not_awaited()

    async def test_goes_dormant_on_message_not_found(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Deleting the host is no longer opt-out: the updater releases the
        host and keeps looping (dormant) so the next command response or -now
        can re-host the block with an accurate bar."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        message = self._host(music_player)
        message.edit.side_effect = discord.NotFound(MagicMock(), "message deleted")

        # Tick 1: NotFound → release + stay alive. Tick 2: dormant no-op.
        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        message.edit.assert_awaited_once()
        assert music_player._np_host_message is None

    async def test_not_found_keeps_host_adopted_mid_tick(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Adopt is lock-free, so a command response can swap in a new host
        while this tick's PATCH is in flight. A NotFound then must not release
        the NEW host — that would permanently orphan its block (branch review
        M1)."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        old_host = self._host(music_player)
        new_host = AsyncMock(spec=discord.Message)

        async def _edit_racing_an_adopt(*args: Any, **kwargs: Any) -> Never:
            music_player._np_host_message = new_host  # adopt lands mid-PATCH
            raise discord.NotFound(MagicMock(), "old host deleted")

        old_host.edit.side_effect = _edit_racing_an_adopt

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        assert music_player._np_host_message is new_host

    async def test_logs_and_continues_on_http_exception(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        message = self._host(music_player)
        message.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

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
        self, music_player: MusicPlayer, mock_song: MagicMock
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

        music_player.current_song = mock_song  # _send_now_playing builds off it
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
    async def _cleanup(self, music_player: MusicPlayer) -> AsyncGenerator[None]:
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
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player._np_host_message = AsyncMock(spec=discord.Message)
        music_player._progress_task = MagicMock(spec=asyncio.Task)
        music_player._progress_task.done.return_value = False
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        assert music_player._pause_debounce_task is not None
        await music_player._pause_debounce_task
        # The debounce task spawns the edit/activity work as separate tracked
        # tasks — drain them before asserting.
        await asyncio.gather(*list(music_player._background_tasks))

        music_player._np_host_message.edit.assert_awaited_once()
        music_player.bot.change_presence.assert_awaited_once()

    async def test_rapid_toggling_collapses_to_one_trailing_update(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player._np_host_message = AsyncMock(spec=discord.Message)
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
        await asyncio.gather(*list(music_player._background_tasks))

        music_player._np_host_message.edit.assert_awaited_once()
        music_player.bot.change_presence.assert_awaited_once()

    async def test_no_embed_edit_when_no_progress_task_or_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player._np_host_message = None
        music_player._progress_task = None
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        assert music_player._pause_debounce_task is not None
        await music_player._pause_debounce_task
        await asyncio.gather(*list(music_player._background_tasks))

        music_player.bot.change_presence.assert_awaited_once()


# ── MarkPausedResumed ──────────────────────────────────────────────────────────


class TestPlayerPauseResume:
    """MusicPlayer.pause()/resume() own all pause-tracking side effects in one
    place: the voice-client call, Redis epoch accounting, and the debounced
    progress-bar/Activity refresh — so a future call site can't forget one."""

    @pytest.fixture(autouse=True)
    async def _cleanup(self, music_player: MusicPlayer) -> AsyncGenerator[None]:
        yield
        await music_player._cancel_pause_debounce()
        music_player._progress_task = None

    async def test_pause_calls_vc_pause(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.pause(vc)
        vc.pause.assert_called_once()

    async def test_pause_writes_to_store(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.pause(vc)
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"pause_start_epoch" in state

    async def test_pause_schedules_debounced_update(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.pause(vc)
        assert music_player._pause_debounce_task is not None

    async def test_pause_skips_store_when_absent(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        mock_song: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await mp.pause(vc)  # must not raise
        vc.pause.assert_called_once()

    async def test_resume_calls_vc_resume(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.resume(vc)
        vc.resume.assert_called_once()

    async def test_resume_writes_to_store(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        assert music_player.store is not None
        music_player.current_song = mock_song
        await music_player.store.on_pause(1000.0)
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.resume(vc)
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"pause_start_epoch" not in state

    async def test_resume_schedules_debounced_update(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.resume(vc)
        assert music_player._pause_debounce_task is not None

    async def test_resume_skips_store_when_absent(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        mock_song: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await mp.resume(vc)  # must not raise
        vc.resume.assert_called_once()


class TestMarkPausedResumed:
    @pytest.fixture(autouse=True)
    async def _cleanup(self, music_player: MusicPlayer) -> AsyncGenerator[None]:
        yield
        await music_player._cancel_pause_debounce()
        music_player._progress_task = None

    async def test_mark_paused_schedules_debounced_update(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player.mark_paused()
        assert music_player._pause_debounce_task is not None

    async def test_mark_resumed_schedules_debounced_update(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        music_player.mark_resumed()
        assert music_player._pause_debounce_task is not None

    async def test_scheduled_tasks_tracked_via_background_tasks(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The debounce task itself, and the embed-edit/activity tasks it spawns,
        must be tracked via _background_tasks (not bare create_task() calls) —
        design review flagged this as the same GC-pending-task risk the codebase
        already guards against elsewhere (musicplayer.py:511-512)."""
        music_player.current_song = mock_song
        music_player._np_host_message = AsyncMock(spec=discord.Message)
        music_player._progress_task = MagicMock(spec=asyncio.Task)
        music_player._progress_task.done.return_value = False
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        assert music_player._pause_debounce_task in music_player._background_tasks
        assert music_player._pause_debounce_task is not None
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
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert embed.colour == discord.Color.blue()
        assert embed.title == "Up next"
        assert "Next Song" in described(embed)
        assert "https://yt.com/v=next" in described(embed)
        assert "`1:30`" in described(embed)
        assert mock_author.mention in embed.description

    def test_shows_resolving_for_unresolved_ytsource(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.queue._display.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "resolving..." in described(embed)

    def test_shows_placeholder_duration_when_unknown(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "`?:??`" in described(embed)

    def test_only_uses_first_queued_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "First", mock_author, duration=60)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Second", mock_author, duration=60)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "First" in described(embed)
        assert "Second" not in described(embed)

    def test_includes_est_playing_at_eta(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "Est. playing at" in described(embed)
        assert re.search(r"\*\*\d{1,2}:\d{2} (AM|PM) PST\*\*", described(embed))

    def test_eta_matches_current_song_estimated_finish(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The next song's ETA should line up with the current song's finish time,
        since both derive from the same cumulative_secs seed."""
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject(
                "https://yt.com/v=next", "Next Song", mock_song.requester, duration=90
            )
        )
        now_playing_embed = music_player._build_now_playing_embed(mock_song)
        next_up_embed = music_player._build_next_up_embed()
        assert next_up_embed is not None
        # Last line only — the progress bar sits above it as its own line and
        # isn't part of the finish-time text being compared here.
        requester_line = described(now_playing_embed).split("\n")[-1]
        finish_time = requester_line.split("Estimated finish: ")[1]
        assert finish_time in described(next_up_embed)


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
        await music_player.queue._pending.put(queue_obj)
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

    async def test_stream_error_retires_dequeue_on_all_three_legs(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        """A prefetch whose resolve/stream raises must retire its dequeue
        everywhere — pending was popped by get_nowait(), so leaving the
        display/Redis heads in place would make the next commit retire the
        wrong entry."""
        assert music_player.store is not None
        await music_player.queue.put([queue_obj])
        with patch(
            "src.musicplayer.YTDL.yt_stream",
            new=AsyncMock(side_effect=Exception("network")),
        ):
            result = await music_player._prefetch_next_song()
        assert result is None
        assert music_player.queue.qsize() == 0
        assert music_player.queue.display_items() == []
        queue_key = music_player.store.queue_key()
        assert await fake_redis.lrange(queue_key, 0, -1) == []

    async def test_swallowed_stream_failure_retires_dequeue(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        """_stream_source catches its own exceptions and returns None — that
        path must retire the dequeue exactly like the raise path."""
        assert music_player.store is not None
        await music_player.queue.put([queue_obj])
        with patch.object(
            MusicPlayer, "_stream_source", new=AsyncMock(return_value=None)
        ):
            result = await music_player._prefetch_next_song()
        assert result is None
        assert music_player.queue.qsize() == 0
        assert music_player.queue.display_items() == []
        queue_key = music_player.store.queue_key()
        assert await fake_redis.lrange(queue_key, 0, -1) == []

    async def test_cancellation_requeues_held_item_at_front(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        queue_obj_no_meta: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        """-clear/-shuffle/-remove cancel the prefetch before mutating; the
        item it holds must return to the front of the pending queue — not be
        dropped — so the mutation drains/reorders it with everything else
        instead of silently losing the next song."""
        await music_player.queue.put([queue_obj, queue_obj_no_meta])
        started = asyncio.Event()
        never_set = asyncio.Event()

        async def hang(self: MusicPlayer, source: Any) -> Any:
            started.set()
            await never_set.wait()
            return source

        with patch.object(MusicPlayer, "_resolve_source", new=hang):
            task = asyncio.create_task(music_player._prefetch_next_song())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert music_player.queue.qsize() == 2
        assert music_player.queue.display_items() == [queue_obj, queue_obj_no_meta]
        # Original order restored, and the transferred task slot balances:
        # consuming both items normally lets join() complete.
        assert music_player.queue.get_nowait() is queue_obj
        music_player.queue.task_done()
        assert music_player.queue.get_nowait() is queue_obj_no_meta
        music_player.queue.task_done()
        await asyncio.wait_for(music_player.queue._pending.join(), timeout=1)


# ── Loop task accounting ──────────────────────────────────────────────────────


class TestLoopTaskAccounting:
    async def test_exception_after_commit_still_balances_task_counter(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """A failure landing between the committed dequeue and the normal
        song-end task_done() (here: the voice client vanished during resolve,
        so the isinstance assert fails before vc.play) must still balance the
        get() in the loop's exception handler — otherwise the queue's task
        counter drifts upward on every such failure."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)
        mocked(music_player._guild).voice_client = None

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        await asyncio.wait_for(music_player.queue._pending.join(), timeout=1)


# ── QueueGet ──────────────────────────────────────────────────────────────────


class TestQueueGet:
    async def test_returns_item_from_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        await music_player.queue._pending.put(queue_obj)
        result = await music_player.queue_get()
        assert result is queue_obj


# ── RestoreStateTtlRefresh ────────────────────────────────────────────────────


class TestRestoreStateTtlRefresh:
    async def test_ttl_refreshed_after_successful_restore(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.8")
        await fake_redis.expire(music_player.store.state_key(), 10)

        await music_player._restore_state()

        ttl = await fake_redis.ttl(music_player.store.state_key())
        assert ttl > 1000

    async def test_restore_continues_after_bad_queue_item(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        assert music_player.store is not None
        valid = orjson.dumps(
            {
                "webpage_url": "https://yt.com/v=ok",
                "title": "Good Song",
                "requester_id": mock_author.id,
                "ts": None,
            }
        )
        await fake_redis.rpush(music_player.store.queue_key(), b"!!!bad json!!!", valid)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 1
        item = await music_player.queue.get()
        assert queue_object(item).title == "Good Song"


# ── Loop ──────────────────────────────────────────────────────────────────────


class TestLoop:
    @pytest.fixture
    def mock_song(self) -> MagicMock:
        # Real (str/int/None) values for every field NowPlayingData.from_song()
        # reads — loop() now serializes the song into the Redis start
        # transaction, and MagicMock attribute values are not HSET-able.
        song = MagicMock()
        song.title = "Loop Test Song"
        song.webpage_url = "https://yt.com/v=loop1"
        song.duration_secs = 210
        song.duration = "0:03:30"
        song.uploader = "Loop Channel"
        song.thumbnail = ""
        song.views = None
        song.likes = None
        song.abr = None
        song.asr = None
        song.acodec = ""
        song.requester = None
        song.start_offset = 0
        # Real number: loop()'s history step feeds this through
        # HistoryEntry.from_song, and round(MagicMock) raises.
        song.position_secs = 195.0
        # -playnow flags a real YTDL always carries — truthy MagicMock
        # attributes would trip the loop's start_paused/is_resume gates.
        song.interjected = False
        song.is_resume = False
        song.start_paused = False
        return song

    async def test_exits_immediately_when_bot_closed(
        self, music_player: MusicPlayer
    ) -> None:
        mocked(music_player.bot.is_closed).return_value = True
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player.loop()

    async def test_timeout_triggers_stop(self, music_player: MusicPlayer) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).return_value = False

        stop_called = asyncio.Event()

        async def _mock_stop(self_inner: Any) -> None:
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
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=None)
            ),
        ):
            await music_player.loop()

        sent_embeds = mocked(music_player._channel.send).call_args.kwargs["embeds"]
        assert sent_embeds[0].description == "Failed to load the next song, skipping."

    async def test_resolve_failure_balances_queue_and_redis(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        """If _resolve_source() raises after queue_get() already dequeued the
        item, the dequeue must still be balanced (song_queue popped, Redis
        popped for a persisted item, queue.task_done() called exactly once)
        and the outer handler's error embed must still be sent."""
        assert music_player.store is not None
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.store.push_queue(SongQueueEntry.from_queue_object(queue_obj))
        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=Exception("yt-dlp lookup failed")),
        ):
            await music_player.loop()

        assert len(music_player.queue._display) == 0
        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(remaining) == 0
        assert (
            music_player.queue._pending._unfinished_tasks == 0  # pyright: ignore[reportAttributeAccessIssue]
        )  # task_done() balanced get()
        sent_embed = mocked(music_player._channel.send).call_args.kwargs["embed"]
        assert sent_embed.title == "Playback error — skipping song"

    async def test_resolve_failure_for_non_persisted_item_does_not_pop_redis(
        self,
        music_player: MusicPlayer,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """A crash-recovered (persisted=False) item that fails to resolve
        must not trigger a Redis pop — it was never RPUSHed there."""
        assert music_player.store is not None
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed Song", mock_author, persisted=False
        )
        await music_player.store.push_queue(
            SongQueueEntry.from_queue_object(
                QueueObject("https://yt.com/v=real", "Real Song", mock_author)
            )
        )
        await music_player.queue._pending.put(crashed)
        music_player.queue._display.append(crashed)

        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=Exception("yt-dlp lookup failed")),
        ):
            await music_player.loop()

        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        urls = {orjson.loads(item)["webpage_url"] for item in remaining}
        assert urls == {"https://yt.com/v=real"}

    async def test_plays_song_and_updates_history(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        # _restore_complete is never set unless start() is called or _restore_state() runs.
        # Set it here so the restore gate in loop() does not block for 10s.
        music_player._restore_complete.set()

        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc

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
        assert music_player.history[0].title == mock_song.title
        assert music_player.history[0].webpage_url == mock_song.webpage_url

    async def test_song_that_produced_no_audio_is_not_treated_as_played(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Regression: a 403 kills ffmpeg instantly, which discord.py reports exactly
        like a song that finished. The bot then advanced in silence, logged nothing, kept
        the dead URL cached, and filed the song in history as if it had been heard."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False  # ffmpeg never delivered a frame

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        # A dead stream is zero frames PLUS the error discord.py hands the after
        # callback when ffmpeg exits non-zero (FFmpegProcessError). Zero frames
        # alone is a song parked or stopped deliberately — see the companion test.
        vc.play = MagicMock(
            side_effect=lambda song, after: after(
                Exception("FFmpeg exited with code 1. Stderr: HTTP error 403")
            )
        )
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()
        music_player._channel.send = AsyncMock()

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
            patch(
                "src.musicplayer.invalidate_stream_cache", new=AsyncMock()
            ) as mock_invalidate,
        ):
            await music_player.loop()

        # The dead URL must not survive to be replayed by the next -play of this song.
        mock_invalidate.assert_awaited_once()
        await_args = mock_invalidate.await_args
        assert await_args is not None
        assert mock_song.webpage_url in await_args.args
        # Nothing was heard, so nothing belongs in history, and the listener is told.
        assert len(music_player.history) == 0
        music_player._channel.send.assert_awaited_once()

    async def test_song_stopped_before_first_frame_is_not_a_dead_stream(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Zero frames WITHOUT an ffmpeg error is a deliberate stop — a -skip or
        interject() landing inside ffmpeg's startup window, or a -playnow resume
        entry parked at vc.pause() (vc.stop() reports error=None to the after
        callback). The stream was never refused: the cached URL must survive, no
        failure notice may be posted, and the song keeps its history entry exactly
        like any other -skip."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False  # stopped before the first frame

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        # after fires with no error, exactly how discord.py reports a vc.stop().
        vc.play = MagicMock(side_effect=lambda song, after: after(None))
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()
        music_player._channel.send = AsyncMock()

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
            patch(
                "src.musicplayer.invalidate_stream_cache", new=AsyncMock()
            ) as mock_invalidate,
        ):
            await music_player.loop()

        mock_invalidate.assert_not_awaited()
        music_player._channel.send.assert_not_awaited()
        assert len(music_player.history) == 1

    async def test_dead_stream_retires_np_host_instead_of_finalizing_bar(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """A bar finalized to 100% directly above the red failure notice would be
        a false record — the song delivered nothing. The host is disposed of like
        retire_np_host_on_stop (dedicated NP message deleted), not finalized."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False
        host = AsyncMock(spec=discord.Message)
        music_player._np_host_message = host
        music_player._np_host_dedicated = True

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(
            side_effect=lambda song, after: after(
                Exception("FFmpeg exited with code 1. Stderr: HTTP error 403")
            )
        )
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()
        music_player._channel.send = AsyncMock()

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
            patch("src.musicplayer.invalidate_stream_cache", new=AsyncMock()),
            patch.object(MusicPlayer, "_fire_finalize_now_playing") as mock_finalize,
        ):
            await music_player.loop()
            await asyncio.gather(*music_player._background_tasks)

        mock_finalize.assert_not_called()
        host.delete.assert_awaited_once()

    async def test_plays_song_writes_duration_uploader_requester_atomically(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """Regression: duration/uploader/requester_id must land in the same
        atomic pop_queue_and_start_song() write as url/title — not via a
        separate, later, non-atomic call that could crash-drop the fields."""
        assert music_player.store is not None
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.duration_secs = 240
        mock_song.uploader = "Test Channel"
        mock_song.requester = mock_author

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        pop_spy = AsyncMock(wraps=music_player.store.pop_queue_and_start_song)

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
            patch.object(music_player.store, "pop_queue_and_start_song", pop_spy),
        ):
            await music_player.loop()

        pop_spy.assert_awaited_once()
        current = pop_spy.call_args.args[0]  # the SongQueueEntry carrier
        assert isinstance(current, SongQueueEntry)
        assert current.duration == 240
        assert current.uploader == "Test Channel"
        assert current.requester_id == mock_author.id

    async def test_loop_clears_play_message_on_song_end(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """After a song finishes, -now must not serve the finished song's embed
        via the crash-recovery elif — play_message is cleared with current_song."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)
        music_player.play_message = discord.Embed(title="stale")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
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

        assert music_player.current_song is None
        assert music_player.play_message is None

    async def test_loop_clears_play_message_on_playback_error(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """The generic exception path must also clear play_message so a failed
        song is never served by -now as still playing."""
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)
        music_player.play_message = discord.Embed(title="stale")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=RuntimeError("ffmpeg gone"))
        mocked(music_player._guild).voice_client = vc

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=MagicMock())
            ),
        ):
            await music_player.loop()

        assert music_player.play_message is None

    async def test_loop_backdates_play_start_epoch_by_start_offset(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """A song started with FFmpeg -ss must persist play_start_epoch backdated
        by the offset, so recovery position math (now - epoch - pauses) yields
        the true audio position rather than time-since-vc.play()."""
        assert music_player.store is not None
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.start_offset = 90

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        pop_spy = AsyncMock(wraps=music_player.store.pop_queue_and_start_song)

        before = time.time()
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
            patch.object(music_player.store, "pop_queue_and_start_song", pop_spy),
        ):
            await music_player.loop()
        after = time.time()

        pop_spy.assert_awaited_once()
        epoch = pop_spy.call_args.args[1]  # play_start_epoch
        assert before - 90 <= epoch <= after - 90

    async def test_now_playing_hash_committed_before_send_now_playing(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """Crash-window regression (the Issue-3 bug): the now_playing snapshot
        must be committed in the start transaction, *before* any Discord I/O —
        by the time _send_now_playing runs, the hash already shows this song."""
        assert music_player.store is not None
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        np_at_send_time: dict = {}

        async def _capture_send(_self: Any, song: Any) -> None:
            assert music_player.store is not None
            np_at_send_time.update(
                await fake_redis.hgetall(music_player.store.now_playing_key())
            )

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=_capture_send),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        assert np_at_send_time.get(b"title") == b"Loop Test Song"
        assert np_at_send_time.get(b"webpage_url") == b"https://yt.com/v=loop1"

    async def test_fires_finalize_task_when_song_ends(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """When a song ends, loop() must capture the host, release it (so the
        next song's adopt retires nothing), and fire the finalize-embed task
        with the song/host/own-embeds that just finished — before current_song
        and the host state get overwritten for the next iteration."""
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        sent_message = MagicMock(spec=discord.Message)

        async def _fake_send_now_playing(_self: Any, song: Any) -> None:
            _self._np_host_message = sent_message
            _self._np_host_own_embeds = []
            _self._np_host_dedicated = True

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

        # The loop fixture stops at 195s of 210s — more than
        # _SONG_COMPLETE_MARGIN_SECS short of the end — so the bar is finalized
        # at its true position rather than 100%. The completed=True/False
        # decision itself is covered by TestFinalizeCompletion.
        finalize_mock.assert_called_once_with(
            mock_song, sent_message, [], completed=False
        )
        assert music_player._np_host_message is None  # released, not retired

    async def test_unhandled_exception_sends_error_message(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)

        def _bad_play(*a: Any, **kw: Any) -> Never:
            raise RuntimeError("ffmpeg gone")

        vc.play = _bad_play
        mocked(music_player._guild).voice_client = vc

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=MagicMock())
            ),
        ):
            await music_player.loop()

        mocked(music_player._channel.send).assert_awaited()

    async def test_error_path_clears_current_song_url(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        """When loop() hits an unhandled exception, current_song_url must be cleared so
        a later process restart does not ghost-replay the failed song."""
        assert music_player.store is not None
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=RuntimeError("ffmpeg gone"))
        mocked(music_player._guild).voice_client = vc

        # Seed Redis so a restart would see a crashed song.
        await fake_redis.hset(
            music_player.store.state_key(),
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

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert state.get(b"current_song_url", b"") == b""
        assert state.get(b"current_song_title", b"") == b""


# ── _restore_complete event ───────────────────────────────────────────────────


class TestRestoreCompleteEvent:
    async def test_set_after_successful_restore(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_set_even_when_restore_raises(
        self, music_player: MusicPlayer
    ) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        with patch.object(
            music_player.store,
            "get_playback_snapshot",
            new=AsyncMock(side_effect=Exception("redis down")),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_set_and_restore_aborted_when_state_read_fails(
        self, music_player: MusicPlayer
    ) -> None:
        """get_playback_snapshot() returning None (Redis unavailable) aborts
        the restore early — nothing is fabricated — but the loop guard event
        is still set."""
        music_player.bot.wait_until_ready = AsyncMock()
        with patch.object(
            music_player.store,
            "get_playback_snapshot",
            new=AsyncMock(return_value=None),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()
        assert music_player.queue.qsize() == 0
        assert len(music_player.history) == 0


# ── _build_now_playing_embed_from_data ────────────────────────────────────────

_NP_DATA = NowPlayingData(
    title="Test Song",
    webpage_url="https://yt.com/v=1",
    uploader="Test Channel",
    duration="3:30",
    thumbnail="https://img.yt.com/thumb.jpg",
    view_count="1000",
    like_count="50",
    abr="128",
    asr="44100",
    acodec="opus",
    requester_id="123",
    requester_mention="<@123>",
)


class TestBuildNowPlayingEmbedFromData:
    def test_returns_discord_embed(self, music_player: MusicPlayer) -> None:
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert isinstance(embed, discord.Embed)

    def test_title_from_data(self, music_player: MusicPlayer) -> None:
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert embed.title is not None
        assert "Test Song" in embed.title

    def test_requester_mention_in_description(self, music_player: MusicPlayer) -> None:
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert "<@123>" in described(embed)

    def test_thumbnail_set_from_data(self, music_player: MusicPlayer) -> None:
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert embed.thumbnail.url == "https://img.yt.com/thumb.jpg"

    def test_thumbnail_not_set_when_empty(self, music_player: MusicPlayer) -> None:
        data = dataclasses.replace(_NP_DATA, thumbnail="")
        embed = music_player._build_now_playing_embed_from_data(data)
        assert not embed.thumbnail.url

    def test_footer_contains_bitrate(self, music_player: MusicPlayer) -> None:
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert embed.footer.text is not None
        assert "128" in embed.footer.text

    def test_duration_in_description(self, music_player: MusicPlayer) -> None:
        # This embed has no progress bar, so the description is the only place
        # duration can appear — the base builder dropped the Duration field on
        # the grounds that the bar's right-hand label shows it.
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert "Duration: `3:30`" in described(embed)
        assert "Duration" not in [f.name for f in embed.fields]

    def test_duration_line_omitted_when_unknown(
        self, music_player: MusicPlayer
    ) -> None:
        data = dataclasses.replace(_NP_DATA, duration="")
        embed = music_player._build_now_playing_embed_from_data(data)
        assert "Duration" not in described(embed)
        assert embed.description == "Requester: [<@123>]"

    def test_default_fields_render_as_empty_strings(
        self, music_player: MusicPlayer
    ) -> None:
        data = NowPlayingData(title="Minimal")  # all other fields defaulted
        embed = music_player._build_now_playing_embed_from_data(data)
        assert embed.title is not None
        assert "Minimal" in embed.title


# ── _restore_state: now-playing embed restoration ────────────────────────────


class TestRestoreStateNowPlaying:
    async def test_restores_play_message_from_redis(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        """If now_playing hash exists in Redis, play_message is populated on restore."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.now_playing_key(),
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
        assert music_player.play_message.title is not None
        assert "Restored Song" in music_player.play_message.title

    async def test_play_message_none_when_no_now_playing_in_redis(
        self, music_player: MusicPlayer
    ) -> None:
        """No now_playing hash → play_message stays None after restore."""
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player._restore_state()
        assert music_player.play_message is None


# ── loop() additional coverage from main branch ───────────────────────────────


class TestLoopAdditional:
    @pytest.fixture
    def mock_song(self) -> MagicMock:
        # See TestLoop.mock_song — real values so the Redis start transaction
        # in loop() can serialize the song.
        song = MagicMock()
        song.title = "Loop Test Song"
        song.webpage_url = "https://yt.com/v=loop1"
        song.duration_secs = 210
        song.duration = "0:03:30"
        song.uploader = "Loop Channel"
        song.thumbnail = ""
        song.views = None
        song.likes = None
        song.abr = None
        song.asr = None
        song.acodec = ""
        song.requester = None
        song.start_offset = 0
        # Real number: loop()'s history step feeds this through
        # HistoryEntry.from_song, and round(MagicMock) raises.
        song.position_secs = 195.0
        # -playnow flags a real YTDL always carries — truthy MagicMock
        # attributes would trip the loop's start_paused/is_resume gates.
        song.interjected = False
        song.is_resume = False
        song.start_paused = False
        return song

    async def test_update_activity_called_at_song_start_and_end(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
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
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
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
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        queue_obj2 = QueueObject(
            "https://yt.com/watch?v=2", "Song 2", queue_obj.requester
        )
        await music_player.queue._pending.put(queue_obj)
        await music_player.queue._pending.put(queue_obj2)
        music_player.queue._display.append("Song 1 - url")
        music_player.queue._display.append("Song 2 - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        prefetched = MagicMock()
        prefetched.cleanup = MagicMock()

        async def _prefetch_with_clear(_self: Any) -> MagicMock:
            try:
                music_player.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            music_player.queue._cleared = True
            return prefetched

        async def _stop_noop(_self: Any) -> None:
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
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """If song_queue is cleared while _stream_source runs, the YTDL object is
        discarded without playing and its FFmpeg subprocess is terminated via cleanup().
        """
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc

        mock_song.cleanup = MagicMock()

        async def _stream_and_clear(_self: Any, source: Any) -> MagicMock:
            # Simulate queue_clear() racing with stream resolution
            music_player.queue._display.clear()
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


# ── -playnow interjection ─────────────────────────────────────────────────────


@pytest.fixture
def mock_vc() -> MagicMock:
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    return vc


@pytest.fixture
def live_song(mock_song: MagicMock) -> MagicMock:
    """mock_song with the -playnow flags a real YTDL carries (a bare MagicMock
    attribute would read as a truthy mock and trip the replace-semantics gate)."""
    mock_song.interjected = False
    mock_song.is_resume = False
    mock_song.start_paused = False
    return mock_song


@pytest.fixture
def playnow_obj(mock_author: MagicMock) -> QueueObject:
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=urgent",
        title="Urgent Song",
        requester=mock_author,
        duration=120,
        interjected=True,
    )


class TestInterject:
    async def test_returns_none_without_current_song(
        self, music_player: MusicPlayer, playnow_obj: QueueObject, mock_vc: MagicMock
    ) -> None:
        music_player.current_song = None
        assert await music_player.interject(playnow_obj, mock_vc) is None
        mock_vc.stop.assert_not_called()

    async def test_front_inserts_playnow_then_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 42.0
        music_player.current_song = live_song
        queued = QueueObject("https://yt.com/v=b", "Queued B", mock_author)
        await music_player.queue.put([queued])

        outcome = await music_player.interject(playnow_obj, mock_vc)

        items = music_player.queue.display_items()
        assert items[0] is playnow_obj
        resume = items[1]
        assert isinstance(resume, QueueObject)
        assert resume.is_resume is True
        assert resume.start_paused is False
        assert resume.ts == 42
        assert resume.webpage_url == live_song.webpage_url
        assert resume.duration == live_song.duration_secs
        assert items[2] is queued

        mock_vc.stop.assert_called_once()
        assert music_player._skip_history_for is live_song
        assert outcome is not None
        assert outcome.interrupted_title == live_song.title
        assert outcome.resume_position == 42
        assert outcome.was_paused is False
        assert outcome.replaced is False

    async def test_paused_song_returns_start_paused(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(playnow_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.start_paused is True
        assert outcome is not None and outcome.was_paused is True
        # -playnow's default: restore exactly what was interrupted.
        assert outcome.returns_paused is True

    async def test_resume_paused_false_returns_song_playing(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """-play on a paused song means "stop being paused, play this" — the
        interrupted song comes back PLAYING at its pause position
        (docs/PLAY_WHILE_PAUSED_PLAN.md §3.1)."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(
            playnow_obj, mock_vc, resume_paused=False
        )

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.start_paused is False
        assert resume.ts == 30  # position preserved even though it returns playing
        assert outcome is not None
        # was_paused is the OBSERVED state and stays True; returns_paused is
        # what the command wording keys off.
        assert outcome.was_paused is True
        assert outcome.returns_paused is False

    async def test_resume_paused_false_is_a_noop_for_a_playing_song(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = False

        outcome = await music_player.interject(
            playnow_obj, mock_vc, resume_paused=False
        )

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.start_paused is False
        assert outcome is not None and outcome.returns_paused is False

    async def test_returns_paused_false_when_no_resume_entry(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """Replaced interjection → no resume entry at all, so nothing returns."""
        live_song.elapsed_secs = 30.0
        live_song.interjected = True
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(playnow_obj, mock_vc)

        assert outcome is not None
        assert outcome.replaced is True
        assert outcome.resume_position is None
        assert outcome.returns_paused is False

    async def test_replace_semantics_skip_resume_for_interjection(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.interjected = True  # the playing song IS a -playnow song
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        assert music_player.queue.display_items() == [playnow_obj]
        assert music_player._skip_history_for is None  # discarded, not returning
        mock_vc.stop.assert_called_once()
        assert outcome is not None
        assert outcome.replaced is True
        assert outcome.resume_position is None

    async def test_near_end_skips_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 207.0  # 3s left of 210 — below the 5s floor
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        assert music_player.queue.display_items() == [playnow_obj]
        assert outcome is not None and outcome.resume_position is None
        assert music_player._skip_history_for is None

    async def test_eof_cap_pulls_position_back(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 205.0  # 5s left: resumable, but capped to 200
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.ts == 200  # duration 210 − 10s EOF margin
        assert outcome is not None and outcome.resume_position == 200

    async def test_no_webpage_url_skips_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.webpage_url = None
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        assert music_player.queue.display_items() == [playnow_obj]
        assert outcome is not None and outcome.resume_position is None

    async def test_stop_skipped_when_song_changed_during_insert(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        async def put_front_and_advance(items: Any) -> None:
            music_player.current_song = MagicMock()  # loop moved on mid-await

        from src.guild_queue import GuildQueue

        # Class-level patch: GuildQueue uses __slots__, so patch.object on the
        # instance can't set the attribute.
        with patch.object(GuildQueue, "put_front", side_effect=put_front_and_advance):
            await music_player.interject(playnow_obj, mock_vc)

        mock_vc.stop.assert_not_called()
        assert music_player._skip_history_for is None

    async def test_neutralizes_running_prefetch_first(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        blocker = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = blocker

        await music_player.interject(playnow_obj, mock_vc)

        assert blocker.cancelled()
        assert music_player._prefetch_task is None


class TestNeutralizePrefetch:
    async def test_no_task_is_noop(self, music_player: MusicPlayer) -> None:
        music_player._prefetch_task = None
        await music_player._neutralize_prefetch()  # must not raise

    async def test_running_task_cancelled_and_cleared(
        self, music_player: MusicPlayer
    ) -> None:
        task = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()
        assert task.cancelled()
        assert music_player._prefetch_task is None

    async def test_completed_task_requeues_rebuilt_item_and_kills_ffmpeg(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        # Simulate the prefetch's own dequeue: pending pops, display keeps the
        # entry (the prefetch commit was still pending).
        original = QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        assert music_player._prefetch_task is None
        live_song.cleanup.assert_called_once()
        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.webpage_url == live_song.webpage_url
        assert rebuilt.title == live_song.title

    async def test_completed_task_rebuild_keeps_offset_and_playnow_flags(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """Nested -playnow regression: the prefetcher resolves the FIRST
        interjection's resume entry within seconds (cache hit), so a second
        -playnow neutralizes a completed prefetch holding a flagged, offset
        entry. A rebuild that drops ts/is_resume/start_paused would restart
        the interrupted song from 0:00, unpaused and unannounced."""
        original = QueueObject(
            "https://yt.com/v=orig",
            "Interrupted Song",
            mock_author,
            ts=151,
            duration=210,
            is_resume=True,
            start_paused=True,
        )
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        # The resolved YTDL for that entry, as yt_stream would build it.
        live_song.start_offset = 151
        live_song.is_resume = True
        live_song.start_paused = True
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.ts == 151
        assert rebuilt.is_resume is True
        assert rebuilt.start_paused is True
        assert rebuilt.interjected is False

    async def test_completed_task_rebuild_keeps_interjected_flag(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """A parked playnow entry must keep its marker through the rebuild —
        losing it would make a later -playnow stack a resume entry for it
        instead of applying replace semantics."""
        original = QueueObject(
            "https://yt.com/v=pn", "Playnow Song", mock_author, interjected=True
        )
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.interjected = True
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.interjected is True
        assert rebuilt.ts is None  # start_offset 0 → no bogus -ss

    async def test_completed_task_with_none_result_is_noop(
        self, music_player: MusicPlayer
    ) -> None:
        async def _done() -> None:
            return None

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()
        assert music_player.queue.qsize() == 0

    async def test_completed_task_that_raised_is_swallowed(
        self, music_player: MusicPlayer
    ) -> None:
        async def _boom() -> Never:
            raise RuntimeError("prefetch exploded")

        task = asyncio.create_task(_boom())
        with contextlib.suppress(RuntimeError):
            await task
        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()  # must not raise
        assert music_player.queue.qsize() == 0


class TestAnnounceResume:
    async def test_playing_wording(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        live_song.elapsed_secs = 42.0
        live_song.is_resume = True
        await music_player._announce_resume(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert "Resuming" in embed.description
        assert "0:42" in embed.description

    async def test_paused_wording(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        live_song.elapsed_secs = 42.0
        live_song.is_resume = True
        live_song.start_paused = True
        await music_player._announce_resume(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert "still paused" in embed.description
        assert "-resume" in embed.description

    async def test_send_failure_swallowed(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        mock_channel.send.side_effect = RuntimeError("channel gone")
        await music_player._announce_resume(live_song)  # must not raise


class TestRemainingSecs:
    def test_normal_item_full_duration(self, queue_obj: QueueObject) -> None:
        from src.musicplayer import _remaining_secs

        assert _remaining_secs(queue_obj) == 210

    def test_resume_entry_counts_only_tail(self, mock_author: MagicMock) -> None:
        from src.musicplayer import _remaining_secs

        item = QueueObject(
            "https://yt.com/v=1", "T", mock_author, ts=150, duration=210, is_resume=True
        )
        assert _remaining_secs(item) == 60

    def test_unknown_duration_is_none(self, queue_obj_no_meta: QueueObject) -> None:
        from src.musicplayer import _remaining_secs

        assert _remaining_secs(queue_obj_no_meta) is None

    def test_non_resume_ts_does_not_shrink_duration(
        self, mock_author: MagicMock
    ) -> None:
        # A ?t= start offset is a playback preference, not a shorter song —
        # only resume entries are known to play just their tail.
        from src.musicplayer import _remaining_secs

        item = QueueObject("https://yt.com/v=1", "T", mock_author, ts=150, duration=210)
        assert _remaining_secs(item) == 210


class TestResumeEntryDisplay:
    async def test_queue_embed_shows_resume_note(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        item = QueueObject(
            "https://yt.com/v=1",
            "Interrupted Song",
            mock_author,
            ts=150,
            duration=210,
            is_resume=True,
        )
        await music_player.queue.put([item])
        embed = music_player.queue_embed()
        assert "⏮ resumes at `2:30`" in described(embed)

    async def test_plain_ts_note_unchanged(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        item = QueueObject("https://yt.com/v=1", "T", mock_author, ts=30, duration=210)
        await music_player.queue.put([item])
        embed = music_player.queue_embed()
        assert "starts at `30s`" in described(embed)


class TestEstimatedFinishUsesRemaining:
    def test_offset_start_finishes_sooner(
        self, music_player: MusicPlayer, live_song: MagicMock
    ) -> None:
        from src.musicplayer import _fmt_finish_time

        live_song.start_offset = 100  # 110s of the 210s song remain
        before = _fmt_finish_time(110)
        embed = music_player._build_now_playing_embed(live_song)
        after = _fmt_finish_time(110)
        assert (before in described(embed)) or (after in described(embed))

    def test_position_override_shrinks_remaining(
        self, music_player: MusicPlayer, live_song: MagicMock
    ) -> None:
        from src.musicplayer import _fmt_finish_time

        before = _fmt_finish_time(10)
        embed = music_player._build_now_playing_embed(
            live_song, position_override=200.0
        )
        after = _fmt_finish_time(10)
        assert (before in described(embed)) or (after in described(embed))


class TestHistorySkipMarker:
    """The _skip_history_for identity marker consumed by loop()'s history step."""

    async def _run_one_song(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
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

    async def test_marker_for_current_song_skips_history_once(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """interject() marked this song (resume entry pending) — its stop
        transition must not record it; the tail's own end will."""
        music_player._skip_history_for = mock_song
        await self._run_one_song(music_player, queue_obj, mock_song)
        assert len(music_player.history) == 0
        assert music_player._skip_history_for is None

    async def test_stale_marker_does_not_eat_next_songs_history(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """A marker left for a song that ended naturally during interject()'s
        awaits (its history step already ran) must not suppress the NEXT
        song's entry — the identity check makes it a no-op that clears."""
        music_player._skip_history_for = MagicMock()  # some other, ended song
        await self._run_one_song(music_player, queue_obj, mock_song)
        assert len(music_player.history) == 1
        assert music_player.history[0].title == mock_song.title
        assert music_player._skip_history_for is None


class TestInterjectPostNeutralizeRecheck:
    async def test_song_changed_during_neutralize_returns_none(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        playnow_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """Neutralize can block up to yt-dlp's socket timeout (cancellation
        can't interrupt the executor thread) — if the song ended and the loop
        moved on in that window, interject bails to the command's fallback
        instead of building a resume entry for a finished song."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        async def neutralize_and_advance(_self: Any) -> None:
            music_player.current_song = MagicMock()

        with patch.object(
            MusicPlayer, "_neutralize_prefetch", new=neutralize_and_advance
        ):
            outcome = await music_player.interject(playnow_obj, mock_vc)

        assert outcome is None
        assert music_player.queue.display_items() == []  # nothing inserted
        mock_vc.stop.assert_not_called()
        assert music_player._skip_history_for is None


class TestPlaynowLoopStart:
    """Loop-level behavior for -playnow entries at song start (review gap):
    start_paused parks the player, is_resume announces from the start path."""

    async def _run_one_song(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        vc: discord.VoiceClient,
    ) -> tuple[AsyncMock, AsyncMock]:
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        mocked(music_player._guild).voice_client = vc
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
            patch.object(MusicPlayer, "pause", new=AsyncMock()) as pause_mock,
            patch.object(
                MusicPlayer, "_announce_resume", new=AsyncMock()
            ) as announce_mock,
        ):
            await music_player.loop()
        return pause_mock, announce_mock

    def _vc(self) -> discord.VoiceClient:
        """A VoiceClient whose play/pause are mocks, built without __init__.

        Deliberately a real instance rather than MagicMock(spec=...): any
        attribute the loop touches beyond these two should fail loudly, not
        hand back a truthy mock that quietly steers the loop down another path.
        Read the mocks back with _mock_call(vc, "pause").
        """
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        vc.pause = MagicMock()
        return vc

    @staticmethod
    def _mock_call(vc: discord.VoiceClient, name: str) -> MagicMock:
        """Return a VoiceClient method that _vc replaced with a MagicMock."""
        return cast(MagicMock, getattr(vc, name))

    async def test_start_paused_parks_synchronously_and_engages_bookkeeping(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        mock_song.start_paused = True
        vc = self._vc()
        pause_mock, _ = await self._run_one_song(music_player, queue_obj, mock_song, vc)
        # Synchronous park right after vc.play (frame-leak guard) …
        self._mock_call(vc, "pause").assert_called_once()
        # … plus the full pause() entry point (Redis epochs, debounced refresh).
        pause_mock.assert_awaited_once()

    async def test_resume_entry_announced_at_start(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        mock_song.is_resume = True
        vc = self._vc()
        _, announce_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        announce_mock.assert_awaited_once_with(mock_song)

    async def test_plain_song_neither_parks_nor_announces(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        vc = self._vc()
        pause_mock, announce_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        self._mock_call(vc, "pause").assert_not_called()
        pause_mock.assert_not_awaited()
        announce_mock.assert_not_awaited()
