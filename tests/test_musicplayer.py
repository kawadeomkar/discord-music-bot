"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import asyncio
import contextlib
import dataclasses
import re
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import orjson
import pytest

from src.guild_state import NowPlayingData, SongQueueEntry
from src.musicplayer import (
    MusicPlayer,
    _build_progress_bar,
    _fmt_duration,
    _fmt_finish_time,
    _fmt_total_duration,
    _requester_mention,
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
        assert len(music_player.queue._display) == 1
        assert isinstance(music_player.queue._display[0], QueueObject)
        assert music_player.queue._display[0].title == "Test Song"

    async def test_put_list_of_sources(self, music_player, mock_author):
        sources = [
            YTSource(ytsearch="ytsearch:song one", process=True),
            YTSource(ytsearch="ytsearch:song two", process=True),
            YTSource(ytsearch="ytsearch:song three", process=True),
        ]
        await music_player.queue_put(sources)
        assert music_player.queue.qsize() == 3
        assert len(music_player.queue._display) == 3

    async def test_put_multiple_singles_increments_size(
        self, music_player, mock_author
    ):
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 4
        assert len(music_player.queue._display) == 4

    async def test_put_mirrors_queue_object_to_redis(
        self, music_player, queue_obj, fake_redis
    ):
        await music_player.queue_put(queue_obj)
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1
        data = orjson.loads(items[0])
        assert data["type"] == "qobj"
        assert data["title"] == queue_obj.title
        assert data["webpage_url"] == queue_obj.webpage_url

    async def test_put_mirrors_yt_source_to_redis(self, music_player, fake_redis):
        src = YTSource(ytsearch="ytsearch:Never Gonna Give You Up", process=True)
        await music_player.queue_put(src)
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1

    async def test_put_sets_ttl_on_redis_key(self, music_player, queue_obj, fake_redis):
        await music_player.queue_put(queue_obj)
        ttl = await fake_redis.ttl(music_player.store.queue_key())
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
        assert len(music_player.queue._display) == 3

        await music_player.queue_clear()
        assert len(music_player.queue._display) == 0

    async def test_clear_on_empty_queue_is_safe(self, music_player):
        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_deletes_redis_key(self, music_player, queue_obj, fake_redis):
        await music_player.queue_put(queue_obj)
        await music_player.queue_clear()
        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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
    def _stub_prefetch(self, monkeypatch):
        from src import youtube

        monkeypatch.setattr(youtube.YTDL, "prefetch_stream", AsyncMock())

    async def test_remove_by_webpage_url(self, music_player, mock_author):
        qobj = QueueObject("https://yt.com/v=abc", "Song", mock_author)
        await music_player.queue_put(qobj)

        positions = await music_player.queue_remove("https://yt.com/v=abc")

        assert positions == [1]
        assert music_player.queue.qsize() == 0
        assert len(music_player.queue._display) == 0

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
        assert len(music_player.queue._display) == 1

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

        remaining = list(music_player.queue._display)
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

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
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
        self, music_player, mock_author, fake_redis
    ):
        """If removal leaves only a non-persisted item, Redis's queue key
        should end up empty/deleted, not populated with a phantom entry."""
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
        self, music_player, mock_author, fake_redis
    ):
        await music_player.queue_put(
            QueueObject("https://yt.com/v=only", "Only Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=only")

        exists = await fake_redis.exists(music_player.store.queue_key())
        assert exists == 0

    async def test_remove_does_not_modify_redis_on_no_match(
        self, music_player, mock_author, fake_redis
    ):
        await music_player.queue_put(
            QueueObject("https://yt.com/v=abc", "Song", mock_author)
        )

        await music_player.queue_remove("https://yt.com/v=xyz")

        items = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(items) == 1


# ── GetQueue embed ────────────────────────────────────────────────────────────


class TestGetQueue:
    def test_returns_discord_embed(self, music_player, queue_obj):
        music_player.queue._display.append(queue_obj)
        result = music_player.queue_embed()
        assert isinstance(result, discord.Embed)

    def test_embed_title_is_queue(self, music_player, queue_obj):
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert embed.title == "Queue"

    def test_embed_color_is_blue(self, music_player, queue_obj):
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert embed.colour == discord.Color.blue()

    def test_empty_queue_description(self, music_player):
        embed = music_player.queue_embed()
        assert "Songs: **0**" in embed.description
        assert "*The queue is empty.*" in embed.description

    def test_song_count_in_header(self, music_player, mock_author):
        for i in range(3):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=120
                )
            )
        embed = music_player.queue_embed()
        assert "Songs: **3**" in embed.description

    def test_total_duration_in_header_when_all_known(self, music_player, mock_author):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=90)
        )
        embed = music_player.queue_embed()
        assert "Total Duration: **3m**" in embed.description
        assert "~" not in embed.description.split("Total Duration:")[1].split("\n")[0]

    def test_total_duration_partial_when_some_unknown(self, music_player, mock_author):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=None)
        )
        embed = music_player.queue_embed()
        assert "~" in embed.description

    def test_total_duration_partial_with_ytsource(self, music_player, mock_author):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90)
        )
        music_player.queue._display.append(
            YTSource(ytsearch="ytsearch:unresolved", process=True)
        )
        embed = music_player.queue_embed()
        assert "~" in embed.description

    def test_song_title_appears_in_description(self, music_player, queue_obj):
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert "Test Song" in embed.description

    def test_song_duration_appears_when_known(self, music_player, queue_obj):
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert "`3:30`" in embed.description

    def test_song_duration_unknown_shows_placeholder(
        self, music_player, queue_obj_no_meta
    ):
        music_player.queue._display.append(queue_obj_no_meta)
        embed = music_player.queue_embed()
        assert "`?:??`" in embed.description

    def test_uploader_shown_when_known(self, music_player, queue_obj):
        music_player.queue._display.append(queue_obj)
        embed = music_player.queue_embed()
        assert "Test Channel" in embed.description

    def test_unknown_channel_shown_when_uploader_none(
        self, music_player, queue_obj_no_meta
    ):
        music_player.queue._display.append(queue_obj_no_meta)
        embed = music_player.queue_embed()
        assert "Unknown channel" in embed.description

    def test_est_playing_at_present_for_each_song(self, music_player, mock_author):
        for i in range(3):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.queue_embed()
        assert embed.description.count("Est. playing at") == 3

    def test_uncertain_prefix_after_no_duration_song(self, music_player, mock_author):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=60)
        )
        embed = music_player.queue_embed()
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
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        embed = music_player.queue_embed()
        assert "~**" in embed.description

    def test_caps_display_at_ten_songs(self, music_player, mock_author):
        for i in range(15):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.queue_embed()
        assert embed.description.count("Est. playing at") == 10

    def test_shows_more_indicator_when_over_ten(self, music_player, mock_author):
        for i in range(15):
            music_player.queue._display.append(
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                )
            )
        embed = music_player.queue_embed()
        assert "... and 5 more" in embed.description

    def test_ytsource_shows_resolving(self, music_player):
        music_player.queue._display.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player.queue_embed()
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

        music_player.queue._display.append(
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
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None)
        )
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_matches_last_queue_line_eta(self, music_player, mock_song, mock_author):
        """estimated_playing_at() should reflect the same seed used by
        queue_embed()/_build_next_up_embed() for consistency across embeds."""
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60)
        )
        eta = music_player.estimated_playing_at()

        # A song appended now would start right where the last queued line's
        # ETA ends up, so re-derive it via the same line formatter for index 2.
        now_pst, cumulative_secs, uncertain = music_player._queue_eta_seed()
        _, cumulative_secs, uncertain = music_player._format_queue_line(
            music_player.queue._display[0], 1, now_pst, cumulative_secs, uncertain
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


class TestBuildProgressBar:
    def test_empty_string_when_duration_unknown(self):
        assert _build_progress_bar(0.0, 0) == ""
        assert _build_progress_bar(10.0, -1) == ""

    def test_head_at_start_when_elapsed_zero(self):
        bar = _build_progress_bar(0.0, 200, width=10)
        assert bar.count("🔘") == 1
        # head is the first bar character after the leading `elapsed` code span
        assert "`0:00`" in bar

    def test_head_at_end_when_elapsed_equals_duration(self):
        bar = _build_progress_bar(200.0, 200, width=10)
        assert bar.count("🔘") == 1
        # clamped to width - 1: fully "done" up to the head, nothing remaining
        assert bar.count("🟦") == 9
        assert bar.count("⬜") == 0

    def test_head_roughly_midpoint_at_half_duration(self):
        bar = _build_progress_bar(100.0, 200, width=10)
        # head_pos = int(0.5 * 10) = 5 done blocks before the head, 4 remaining after
        middle = bar.split("`")[2]  # text between the two backtick-wrapped times
        head_index = middle.index("🔘")
        assert middle[:head_index].count("🟦") == 5
        assert middle[head_index + 1 :].count("⬜") == 4

    def test_clamped_when_elapsed_exceeds_duration(self):
        """Involuntary drift (e.g. a stale duration_secs) must not overflow the bar."""
        bar = _build_progress_bar(500.0, 200, width=10)
        assert bar.count("🔘") == 1
        assert bar.count("🟦") == 9
        assert bar.count("⬜") == 0

    def test_head_clamped_to_start_when_elapsed_negative(self):
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

    def test_width_is_customizable(self):
        bar = _build_progress_bar(0.0, 200, width=5)
        assert bar.count("🟦") + bar.count("🔘") + bar.count("⬜") == 5

    def test_includes_formatted_elapsed_and_duration(self):
        bar = _build_progress_bar(65.0, 200)
        assert "`1:05`" in bar
        assert "`3:20`" in bar

    def test_elapsed_label_clamped_to_duration(self):
        """The left time label must never overshoot the right one — imprecise
        duration metadata plus a -ss start offset can push the raw position
        past the reported duration (e.g. `4:05 … 4:02`)."""
        bar = _build_progress_bar(250.0, 200, width=10)
        assert bar.startswith("`3:20`")
        assert "`4:10`" not in bar

    def test_elapsed_label_clamped_to_zero_when_negative(self):
        bar = _build_progress_bar(-5.0, 200, width=10)
        assert bar.startswith("`0:00`")


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

    def test_zero_views_and_likes_render_as_zero_not_blank(
        self, music_player, mock_song
    ):
        """A legitimate 0 must render as "0", not collapse to an empty field
        (the `str(x or "")` bug this shared extraction fixed)."""
        mock_song.views = 0
        mock_song.likes = 0
        embed = music_player._build_now_playing_embed(mock_song)
        fields_by_name = {f.name: f.value for f in embed.fields}
        assert fields_by_name["Views"] == "0"
        assert fields_by_name["Likes"] == "0"

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
        """The requester/finish-time line stays on one line — the progress bar
        (Design §2 of the progress-bar plan) sits above it as its own line, not
        interleaved with it."""
        embed = music_player._build_now_playing_embed(mock_song)
        requester_line = embed.description.split("\n")[-1]
        assert re.search(
            r"Requester: \[.*\].*Estimated finish: \d{1,2}:\d{2} (AM|PM) PST$",
            requester_line,
        )

    def test_progress_bar_appears_above_requester_line(self, music_player, mock_song):
        """UI update: the bar sits directly under the title, above the
        requester/finish-time line — not the other way around."""
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        lines = embed.description.split("\n")
        assert "🔘" in lines[0]
        assert lines[2].startswith("Requester:")

    def test_blank_line_separates_bar_from_requester_line(
        self, music_player, mock_song
    ):
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        lines = embed.description.split("\n")
        assert lines[1] == ""

    def test_no_estimated_finish_when_duration_unknown(self, music_player, mock_song):
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "Estimated finish" not in embed.description

    def test_progress_bar_line_present_when_duration_known(
        self, music_player, mock_song
    ):
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "🔘" in embed.description
        assert "\n" in embed.description  # progress bar is on its own line

    def test_progress_bar_reflects_elapsed_secs(self, music_player, mock_song):
        mock_song.elapsed_secs = 105.0  # roughly halfway through 210s
        embed = music_player._build_now_playing_embed(mock_song)
        assert _fmt_duration(105) in embed.description

    def test_no_progress_bar_line_when_duration_unknown(self, music_player, mock_song):
        mock_song.duration_secs = 0
        embed = music_player._build_now_playing_embed(mock_song)
        assert "🔘" not in embed.description

    def test_position_override_replaces_live_position(self, music_player, mock_song):
        """Used by _finalize_now_playing() to render the bar fully completed
        once a song has ended, regardless of song.position_secs's live value."""
        mock_song.elapsed_secs = 30.0
        mock_song.duration_secs = 210
        embed = music_player._build_now_playing_embed(
            mock_song, position_override=210.0
        )
        assert _fmt_duration(210) in embed.description
        assert _fmt_duration(30) not in embed.description

    def test_no_override_falls_back_to_live_position(self, music_player, mock_song):
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert _fmt_duration(30) in embed.description

    def test_progress_bar_includes_start_offset(self, music_player, mock_song):
        """A ?t= song or a crash-recovered song resumed mid-stream via FFmpeg
        -ss renders its true audio position (start_offset + elapsed_secs) —
        all position surfaces read YTDL.position_secs, so the bar can't
        disagree with the pause embed or the Activity tooltip."""
        mock_song.start_offset = 60
        mock_song.elapsed_secs = 30.0
        embed = music_player._build_now_playing_embed(mock_song)
        assert _fmt_duration(90) in embed.description
        assert _fmt_duration(30) not in embed.description


class TestBuildPauseConfirmationEmbed:
    """Slim by design: the -pause response message hosts the live NP block
    directly below this embed (MusicContext attach), so the bar, requester,
    link fields, and thumbnail would all render twice if repeated here. The
    embed carries only what the NP block doesn't: the paused state and the
    exact pause position."""

    def test_returns_none_when_no_current_song(self, music_player):
        music_player.current_song = None
        assert music_player.build_pause_confirmation_embed() is None

    def test_returns_discord_embed(self, music_player, mock_song):
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert isinstance(embed, discord.Embed)

    def test_title_contains_song_title(self, music_player, mock_song):
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert mock_song.title in embed.title

    def test_color_is_orange(self, music_player, mock_song):
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed.colour == discord.Color.orange()

    def test_paused_at_reflects_elapsed_secs(self, music_player, mock_song):
        mock_song.elapsed_secs = 65.0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        # position 1:05 of total 3:30
        assert "Paused at: `1:05 / 3:30`" in embed.description

    def test_paused_at_includes_start_offset(self, music_player, mock_song):
        """A song resumed mid-stream via FFmpeg -ss reports true audio position
        (YTDL.position_secs), not just elapsed_secs."""
        mock_song.start_offset = 60
        mock_song.elapsed_secs = 65.0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        # position = 60 + 65 = 125s = 2:05
        assert "Paused at: `2:05 / 3:30`" in embed.description

    def test_paused_at_omits_total_when_duration_unknown(self, music_player, mock_song):
        mock_song.elapsed_secs = 65.0
        mock_song.duration_secs = 0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert "Paused at: `1:05`" in embed.description
        assert "/" not in embed.description.split("Paused at:")[1].split("\n")[0]

    def test_no_progress_bar(self, music_player, mock_song):
        mock_song.elapsed_secs = 65.0
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert "🔘" not in embed.description

    def test_no_requester_line(self, music_player, mock_song):
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert mock_song.requester.mention not in embed.description

    def test_no_fields(self, music_player, mock_song):
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert embed.fields == []

    def test_no_thumbnail(self, music_player, mock_song):
        music_player.current_song = mock_song
        embed = music_player.build_pause_confirmation_embed()
        assert not embed.thumbnail.url


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

    async def test_backdates_start_by_elapsed_secs(self, music_player, mock_song):
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

    async def test_backdate_includes_start_offset(self, music_player, mock_song):
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
        self, music_player, mock_song
    ):
        music_player.bot.change_presence = AsyncMock()
        music_player._guild.voice_client.is_paused.return_value = True
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.timestamps == {}

    async def test_still_sets_name_and_state_while_paused(
        self, music_player, mock_song
    ):
        """Only the ticking timestamps are dropped — the rest of the activity
        (title/uploader/state) still renders while paused."""
        music_player.bot.change_presence = AsyncMock()
        music_player._guild.voice_client.is_paused.return_value = True
        await music_player.update_activity(mock_song)
        activity = music_player.bot.change_presence.call_args.kwargs["activity"]
        assert activity.name == f"{mock_song.title} · {mock_song.uploader}"
        assert activity.state == mock_song.duration

    async def test_resumed_timestamps_reflect_elapsed_not_full_duration(
        self, music_player, mock_song
    ):
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
    def test_queue_starts_empty(self, music_player):
        assert music_player.queue.qsize() == 0

    def test_song_queue_starts_empty(self, music_player):
        assert len(music_player.queue._display) == 0

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
            await music_player.store.push_history(f"Song {i} - url{i}")
        items = await fake_redis.lrange(music_player.store.history_key(), 0, -1)
        assert len(items) == 50

    async def test_store_set_volume_updates_volume(self, music_player, fake_redis):
        await music_player.store.set_volume(0.75)
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_redis_pop_queue_removes_first_item(self, music_player, fake_redis):
        await fake_redis.rpush(music_player.store.queue_key(), b"item1")
        await fake_redis.rpush(music_player.store.queue_key(), b"item2")
        await music_player.store.pop_queue()
        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(remaining) == 1
        assert remaining[0] == b"item2"

    def test_store_is_none_when_no_redis(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        assert mp.store is None


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
        await fake_redis.rpush(music_player.store.queue_key(), item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()
        assert music_player.queue.qsize() == 1
        assert isinstance(music_player.queue._display[0], QueueObject)
        assert music_player.queue._display[0].title == "Restored Song"

    async def test_restore_sets_volume(self, music_player, fake_redis):
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.5")
        await music_player._restore_state()
        assert music_player.volume == 0.5

    async def test_restore_noop_when_no_redis(
        self, mock_bot, mock_guild, mock_channel, mock_ctx
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        await mp._restore_state()
        assert mp.queue.qsize() == 0

    async def test_restore_reads_everything_in_one_snapshot_call(
        self, music_player, fake_redis
    ):
        """State, queue, now-playing, and history all ride the single
        pipelined get_playback_snapshot() read — guard against a future edit
        reintroducing per-key reads (recovery was 3 round trips per guild
        before the snapshot absorbed now_playing/history)."""
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
        self, music_player, fake_redis
    ):
        await music_player.store.push_history("Old Song - url1")
        await music_player.store.push_history("New Song - url2")
        await music_player._restore_state()
        assert list(music_player.history) == ["Old Song - url1", "New Song - url2"]

    async def test_restore_populates_play_message_from_snapshot(
        self, music_player, fake_redis
    ):
        await fake_redis.hset(
            music_player.store.now_playing_key(), b"title", b"Crashed Song"
        )
        await music_player._restore_state()
        assert music_player.play_message is not None
        assert "Crashed Song" in music_player.play_message.title


# ── RestoreCrashedSong ────────────────────────────────────────────────────────


class TestRestoreCrashedSong:
    async def test_crashed_song_requeued_at_front(
        self, music_player, fake_redis, mock_author
    ):
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
        assert first.webpage_url == "https://yt.com/v=crash"
        assert first.title == "Crashed Song"

    async def test_crashed_song_state_cleared_after_restore(
        self, music_player, fake_redis
    ):
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
        self, music_player, fake_redis, mock_author
    ):
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
        assert first.duration == 240
        assert first.uploader == "Test Channel"

    async def test_crashed_song_url_cleared_even_when_requester_unresolvable(
        self, music_player, fake_redis
    ):
        """When guild.me and guild.owner are both None, the crashed song cannot be
        re-queued — but current_song_url must still be cleared to avoid an infinite
        retry loop on every subsequent restart."""
        await fake_redis.hset(
            music_player.store.state_key(),
            b"current_song_url",
            b"https://yt.com/v=crash",
        )
        await fake_redis.hset(
            music_player.store.state_key(), b"current_song_title", b"Ghost Song"
        )
        music_player._guild.get_member = MagicMock(return_value=None)
        music_player._guild.me = None
        music_player._guild.owner = None

        await music_player._restore_state()

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"current_song_url" not in state
        assert b"current_song_title" not in state
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
        await fake_redis.rpush(music_player.store.queue_key(), normal_item)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 1
        first = await music_player.queue.get()
        assert first.title == "Normal"

    async def test_crashed_song_resolves_requester_from_requester_id(
        self, music_player, fake_redis, mock_author
    ):
        """current_song_requester_id (persisted atomically with the song at
        start-transaction time) resolves to the guild member who requested it."""
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
        assert first.requester is mock_author

    async def test_crashed_song_falls_back_to_guild_me_without_requester_id(
        self, music_player, fake_redis, mock_author
    ):
        """State without current_song_requester_id (or a departed member) falls
        back to guild.me so the song is still re-queued."""
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
        music_player._guild.me = bot_member
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.requester is bot_member

    async def test_crashed_song_computes_position_from_play_epoch(
        self, music_player, fake_redis, mock_author
    ):
        """play_start_epoch and total_pause_seconds are combined into a seek offset."""
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
        self, music_player, fake_redis, mock_author
    ):
        """When play_start_epoch is absent, ts on the restored QueueObject is None."""
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
        self, music_player, fake_redis, mock_author
    ):
        """When the bot crashed while paused, pause_start_epoch contributes to total pause
        time and is subtracted from the seek position alongside total_pause_seconds."""
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
        self, music_player, fake_redis, mock_author
    ):
        """The recovery position is capped at cached stream duration − 10s so
        FFmpeg never seeks past EOF."""
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
        self, music_player, fake_redis, mock_author
    ):
        """A malformed cached stream duration degrades to "no cap" — the
        computed position is kept and the restore still completes (clears the
        crashed-song state) instead of aborting."""
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
        self, music_player, fake_redis
    ):
        music_player._restore_complete.clear()
        await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_restore_state_sets_restore_complete_on_failure(self, music_player):
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

        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(remaining) == 2

    async def test_shuffle_during_restore_window_does_not_orphan_redis_entry(
        self, music_player, fake_redis, mock_author
    ):
        """End-to-end guard for Issue 1: if a user runs -shuffle while the
        crash-recovered song is still sitting in song_queue (before loop()
        has dequeued it), Redis's queue list must still end up with exactly
        the real queued songs — no phantom entry for the crashed song."""
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
        assert mp.store is not None


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
        assert music_player.store is not None
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
    @pytest.fixture(autouse=True)
    async def _cleanup_progress_task(self, music_player):
        """_send_now_playing() may spawn a real _progress_task (Design §4). Tests
        in this class don't drive loop() to retire it themselves, so clean it up
        here rather than leaking a pending asyncio.sleep() task past the test."""
        yield
        await music_player._cancel_progress_task()

    @pytest.fixture(autouse=True)
    def _live_song(self, music_player, mock_song):
        """_send_now_playing's embed block is built off current_song (shared
        with the MusicContext attach path) — loop() always sets it before
        calling, so mirror that here."""
        music_player.current_song = mock_song

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

    async def test_resets_stale_np_host_on_send_failure(self, music_player, mock_song):
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
        self, music_player, mock_song
    ):
        await music_player._send_now_playing(mock_song)
        call_kwargs = music_player._channel.send.call_args[1]
        assert len(call_kwargs["embeds"]) == 1
        assert call_kwargs["embeds"][0].colour == discord.Color.green()

    async def test_sends_next_up_embed_when_queue_has_song(
        self, music_player, mock_song, mock_author
    ):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        await music_player._send_now_playing(mock_song)
        call_kwargs = music_player._channel.send.call_args[1]
        embeds = call_kwargs["embeds"]
        assert len(embeds) == 2
        assert embeds[1].colour == discord.Color.blue()
        assert embeds[1].title == "Up next"
        assert "Next Song" in embeds[1].description

    async def test_send_now_playing_works_without_store(
        self, mock_bot, mock_guild, mock_channel, mock_ctx, mock_song
    ):
        # The Redis now-playing snapshot is written by the start transaction in
        # loop(), not here — _send_now_playing only builds/sends the embed.
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp._channel = mock_channel
        mp.current_song = mock_song
        await mp._send_now_playing(mock_song)
        assert mp.play_message is not None

    async def test_adopts_sent_message_as_dedicated_host(self, music_player, mock_song):
        sent_message = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent_message)
        await music_player._send_now_playing(mock_song)
        assert music_player._np_host_message is sent_message
        assert music_player._np_host_own_embeds == []
        assert music_player._np_host_dedicated is True

    async def test_sent_block_reuses_play_message_embed(self, music_player, mock_song):
        """The NP embed stored as play_message IS the one sent in the block —
        not an identical rebuild (branch review N3)."""
        await music_player._send_now_playing(mock_song)
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert embeds[0] is music_player.play_message

    async def test_starts_progress_task_for_normal_duration_song(
        self, music_player, mock_song
    ):
        mock_song.duration_secs = 210
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is not None
        assert not music_player._progress_task.done()

    async def test_no_progress_task_for_sub_5s_song(self, music_player, mock_song):
        mock_song.duration_secs = 4
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is None

    async def test_no_progress_task_for_zero_duration_song(
        self, music_player, mock_song
    ):
        mock_song.duration_secs = 0
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is None

    async def test_progress_task_starts_for_exactly_5s_song(
        self, music_player, mock_song
    ):
        mock_song.duration_secs = 5
        await music_player._send_now_playing(mock_song)
        assert music_player._progress_task is not None


# ── Now-playing host primitives (embed-attach plan §1–§4) ─────────────────────


class TestNpEmbedBlock:
    def test_empty_when_no_song(self, music_player):
        assert music_player.np_embed_block() == []

    def test_now_playing_only_when_queue_empty(self, music_player, mock_song):
        music_player.current_song = mock_song
        block = music_player.np_embed_block()
        assert len(block) == 1
        assert block[0].colour == discord.Color.green()

    def test_np_then_next_up_ordering(self, music_player, mock_song, mock_author):
        music_player.current_song = mock_song
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        block = music_player.np_embed_block()
        assert len(block) == 2
        assert block[0].colour == discord.Color.green()
        assert block[1].title == "Up next"


class TestNpHostAdoptRetire:
    def test_adopt_updates_state_synchronously(self, music_player):
        msg = MagicMock(spec=discord.Message)
        msg.id = 1
        own = [discord.Embed(title="Queue")]
        music_player._adopt_np_host(msg, own)
        assert music_player._np_host_message is msg
        assert music_player._np_host_own_embeds is own
        assert music_player._np_host_dedicated is False
        assert not music_player._background_tasks  # no old host → no retire

    async def test_adopt_retires_old_dedicated_host_with_delete(self, music_player):
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        music_player._adopt_np_host(old, [], dedicated=True)
        new = AsyncMock(spec=discord.Message)
        new.id = 2
        music_player._adopt_np_host(new, [])
        await asyncio.gather(*list(music_player._background_tasks))
        old.delete.assert_awaited_once()
        old.edit.assert_not_awaited()

    async def test_adopt_strips_old_response_host_with_edit(self, music_player):
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

    async def test_adopt_same_message_retires_nothing(self, music_player):
        msg = AsyncMock(spec=discord.Message)
        msg.id = 1
        music_player._adopt_np_host(msg, [])
        music_player._adopt_np_host(msg, [discord.Embed(title="p")])
        assert not music_player._background_tasks
        msg.delete.assert_not_awaited()
        msg.edit.assert_not_awaited()

    async def test_retire_swallows_not_found(self, music_player):
        msg = AsyncMock(spec=discord.Message)
        msg.delete.side_effect = discord.NotFound(MagicMock(), "gone")
        await music_player._retire_np_host(msg, [], True)  # must not raise

    async def test_retire_swallows_and_logs_http_exception(self, music_player):
        msg = AsyncMock(spec=discord.Message)
        msg.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")
        await music_player._retire_np_host(msg, [], False)  # must not raise

    def test_release_clears_state_without_touching_message(self, music_player):
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

    async def test_adopt_ignores_older_message_and_sheds_its_block(self, music_player):
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

    async def test_retire_waits_for_lock_holder(self, music_player):
        """Plan §4 lock ordering: a retire serializes behind _np_edit_lock, so
        an in-flight tick edit (which holds the lock across its await) always
        completes before the retire's strip/delete — the retire is the final
        write and a late tick can't resurrect the NP block on the old host."""
        order: list[str] = []
        old = AsyncMock(spec=discord.Message)

        async def _delete():
            order.append("retire")

        old.delete.side_effect = _delete

        async def _hold_lock_like_a_tick():
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

    async def test_adopts_when_song_still_current(self, music_player, mock_song):
        music_player.current_song = mock_song
        msg = AsyncMock(spec=discord.Message)
        msg.id = 1
        own = [discord.Embed(title="Queue")]
        assert music_player._adopt_np_host_if_current(msg, own, mock_song) is True
        assert music_player._np_host_message is msg
        msg.edit.assert_not_awaited()

    async def test_sheds_block_when_song_changed(self, music_player, mock_song):
        music_player.current_song = MagicMock()  # the next song took over
        msg = AsyncMock(spec=discord.Message)
        own = [discord.Embed(title="Queue")]
        assert music_player._adopt_np_host_if_current(msg, own, mock_song) is False
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None
        msg.edit.assert_awaited_once_with(embeds=own)  # strip back to own embeds

    async def test_deletes_stale_dedicated_message(self, music_player, mock_song):
        music_player.current_song = None  # queue emptied while send was in flight
        msg = AsyncMock(spec=discord.Message)
        assert (
            music_player._adopt_np_host_if_current(msg, [], mock_song, dedicated=True)
            is False
        )
        await asyncio.gather(*list(music_player._background_tasks))
        msg.delete.assert_awaited_once()

    async def test_never_adopts_for_none_song(self, music_player):
        """A block can only have been built off a live song; a None song must
        never adopt even if current_song is also None."""
        msg = AsyncMock(spec=discord.Message)
        assert music_player._adopt_np_host_if_current(msg, [], None) is False
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None

    async def test_stale_adopt_does_not_disturb_new_songs_host(
        self, music_player, mock_song
    ):
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
        self, music_player, mock_song
    ):
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

    async def test_plain_send_when_no_song(self, music_player):
        sent = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent)
        await music_player.send_with_np("hello")
        args, kwargs = music_player._channel.send.call_args
        assert args == ("hello",)
        assert "embeds" not in kwargs
        assert music_player._np_host_message is None

    async def test_embed_send_without_song_does_not_adopt(self, music_player):
        sent = MagicMock(spec=discord.Message)
        music_player._channel.send = AsyncMock(return_value=sent)
        notice = discord.Embed(title="Notice")
        await music_player.send_with_np(embed=notice)
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert embeds == [notice]
        assert music_player._np_host_message is None

    async def test_content_and_embed_together_when_song_live(
        self, music_player, mock_song
    ):
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
        self, music_player, mock_song
    ):
        """H1 at the send_with_np attach site: the song ends while the HTTP
        send is in flight — the sent message strips its stale block and the
        host stays released."""
        music_player.current_song = mock_song
        sent = AsyncMock(spec=discord.Message)

        async def _send_crossing_song_boundary(*args, **kwargs):
            music_player.current_song = None
            return sent

        music_player._channel.send = AsyncMock(side_effect=_send_crossing_song_boundary)
        await music_player.send_with_np(embed=discord.Embed(title="Notice"))
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None
        sent.edit.assert_awaited_once()  # stripped back to its own embeds


class TestRepinNowPlaying:
    async def test_false_when_no_song(self, music_player):
        assert await music_player.repin_now_playing() is False
        music_player._channel.send.assert_not_awaited()

    async def test_sends_dedicated_block_and_adopts(self, music_player, mock_song):
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
        self, music_player, mock_song
    ):
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

    async def test_false_when_song_ends_mid_send(self, music_player, mock_song):
        """H1 at the repin attach site: the song ends while the dedicated NP
        send is in flight — the stale message is deleted, nothing is adopted,
        and repin reports False so -now can respond another way."""
        music_player.current_song = mock_song
        sent = AsyncMock(spec=discord.Message)

        async def _send_crossing_song_boundary(*args, **kwargs):
            music_player.current_song = None
            return sent

        music_player._channel.send = AsyncMock(side_effect=_send_crossing_song_boundary)
        assert await music_player.repin_now_playing() is False
        await asyncio.gather(*list(music_player._background_tasks))
        assert music_player._np_host_message is None
        sent.delete.assert_awaited_once()

    async def test_does_not_touch_progress_task(self, music_player, mock_song):
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

    async def test_deletes_dedicated_host(self, music_player):
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        music_player._adopt_np_host(host, [], dedicated=True)
        await music_player.retire_np_host_on_stop()
        host.delete.assert_awaited_once()
        assert music_player._np_host_message is None

    async def test_strips_response_host_to_own_embeds(self, music_player):
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        own = [discord.Embed(title="Queue")]
        music_player._adopt_np_host(host, own)
        await music_player.retire_np_host_on_stop()
        host.edit.assert_awaited_once_with(embeds=own)
        host.delete.assert_not_awaited()
        assert music_player._np_host_message is None

    async def test_noop_when_no_host(self, music_player):
        await music_player.retire_np_host_on_stop()  # must not raise


class TestRehostNpAfterResume:
    """-resume re-hosting (branch review M3): a command-response host —
    typically the -pause confirmation — is strip-retired in favor of a fresh
    dedicated NP message, so "⏸️ Paused at…" becomes plain history instead of
    being re-rendered beneath a live bar by every tick."""

    async def test_rehosts_when_response_hosts_the_block(self, music_player, mock_song):
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

    async def test_noop_when_host_is_dedicated(self, music_player, mock_song):
        """A dedicated NP message has no stale state to shed — no extra send."""
        music_player.current_song = mock_song
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        music_player._adopt_np_host(host, [], dedicated=True)
        await music_player.rehost_np_after_resume()
        music_player._channel.send.assert_not_awaited()
        assert music_player._np_host_message is host

    async def test_noop_when_no_host(self, music_player, mock_song):
        music_player.current_song = mock_song
        await music_player.rehost_np_after_resume()
        music_player._channel.send.assert_not_awaited()


class TestPushNpEditEmbedCap:
    async def test_truncates_to_ten_embeds_keeping_the_block(
        self, music_player, mock_song, mock_author
    ):
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
    async def test_edits_host_with_own_embeds(self, music_player, mock_song):
        music_player.current_song = mock_song
        host = AsyncMock(spec=discord.Message)
        own = [discord.Embed(title="Queue")]
        music_player._np_host_message = host
        music_player._np_host_own_embeds = own
        await music_player._edit_now_playing_once()
        embeds = host.edit.call_args.kwargs["embeds"]
        assert embeds[0].colour == discord.Color.green()  # NP block leads
        assert embeds[1].title == "Queue"  # host's own embeds follow

    async def test_releases_host_on_not_found(self, music_player, mock_song):
        music_player.current_song = mock_song
        host = AsyncMock(spec=discord.Message)
        host.edit.side_effect = discord.NotFound(MagicMock(), "gone")
        music_player._np_host_message = host
        await music_player._edit_now_playing_once()
        assert music_player._np_host_message is None

    async def test_not_found_keeps_host_adopted_mid_edit(self, music_player, mock_song):
        """Adopt is lock-free, so a command response can swap in a new host
        while this edit's PATCH is in flight. A NotFound then must not release
        the NEW host — that would permanently orphan its block (branch review
        M1)."""
        music_player.current_song = mock_song
        old_host = AsyncMock(spec=discord.Message)
        new_host = AsyncMock(spec=discord.Message)

        async def _edit_racing_an_adopt(*args, **kwargs):
            music_player._np_host_message = new_host  # adopt lands mid-PATCH
            raise discord.NotFound(MagicMock(), "old host deleted")

        old_host.edit.side_effect = _edit_racing_an_adopt
        music_player._np_host_message = old_host
        await music_player._edit_now_playing_once()
        assert music_player._np_host_message is new_host

    async def test_noop_when_no_host(self, music_player, mock_song):
        music_player.current_song = mock_song
        await music_player._edit_now_playing_once()  # must not raise

    async def test_noop_when_no_song(self, music_player):
        host = AsyncMock(spec=discord.Message)
        music_player._np_host_message = host
        await music_player._edit_now_playing_once()
        host.edit.assert_not_awaited()


# ── FinalizeNowPlaying ────────────────────────────────────────────────────────


class TestFinalizeNowPlaying:
    """A song freezing mid-bar (e.g. `3:04 / 3:07`) after it ends — because the
    last periodic tick landed before the true end — is fixed by one last,
    fire-and-forget edit showing the bar fully completed."""

    async def test_edits_message_with_full_duration(self, music_player, mock_song):
        mock_song.elapsed_secs = 184.0  # song ended mid-tick, e.g. 3:04 / 3:07
        mock_song.duration_secs = 210
        message = AsyncMock(spec=discord.Message)

        await music_player._finalize_now_playing(mock_song, message, [])

        message.edit.assert_awaited_once()
        embed = message.edit.call_args.kwargs["embeds"][0]
        assert _fmt_duration(210) in embed.description
        assert _fmt_duration(184) not in embed.description

    async def test_noop_when_duration_unknown(self, music_player, mock_song):
        mock_song.duration_secs = 0
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, [])
        message.edit.assert_not_awaited()

    async def test_includes_next_up_embed_when_queue_has_song(
        self, music_player, mock_song, mock_author
    ):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90)
        )
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, [])
        embeds = message.edit.call_args.kwargs["embeds"]
        assert len(embeds) == 2
        assert embeds[1].title == "Up next"

    async def test_preserves_captured_host_own_embeds(self, music_player, mock_song):
        """A song that ended while a command response hosted the NP block must
        keep that response's own embeds after the completed bar."""
        own = [discord.Embed(title="Queue")]
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, own)
        embeds = message.edit.call_args.kwargs["embeds"]
        assert _fmt_duration(mock_song.duration_secs) in embeds[0].description
        assert embeds[1].title == "Queue"

    async def test_swallows_not_found(self, music_player, mock_song):
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.NotFound(MagicMock(), "message deleted")
        await music_player._finalize_now_playing(
            mock_song, message, []
        )  # must not raise

    async def test_swallows_and_logs_http_exception(self, music_player, mock_song):
        message = AsyncMock(spec=discord.Message)
        message.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")
        await music_player._finalize_now_playing(
            mock_song, message, []
        )  # must not raise

    async def test_operates_on_captured_song_and_message_args(
        self, music_player, mock_song
    ):
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

    async def test_waits_for_lock_holder(self, music_player, mock_song):
        """The finalize's completed-bar write must land AFTER any in-flight
        debounce-spawned edit (which holds _np_edit_lock across its PATCH) —
        otherwise a resume just before song end can freeze the historical bar
        short of 100% (branch review L2)."""
        order: list[str] = []
        message = AsyncMock(spec=discord.Message)

        async def _edit(*args, **kwargs):
            order.append("finalize")

        message.edit.side_effect = _edit

        async def _hold_lock_like_a_oneshot_edit():
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
    async def test_spawns_tracked_background_task(self, music_player, mock_song):
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
    def _make_sleep(n_ticks: int):
        """asyncio.sleep double that lets the loop run n_ticks times, then raises
        CancelledError — deterministic without waiting on the real interval."""
        calls = 0

        async def _sleep(_secs):
            nonlocal calls
            calls += 1
            if calls > n_ticks:
                raise asyncio.CancelledError()

        return _sleep

    @staticmethod
    def _host(music_player) -> AsyncMock:
        """Install an NP host message for the updater to edit."""
        message = AsyncMock(spec=discord.Message)
        music_player._np_host_message = message
        music_player._np_host_own_embeds = []
        return message

    async def test_ticks_and_edits_host_message(self, music_player, mock_song):
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = self._host(music_player)

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        message.edit.assert_awaited_once()
        assert "embeds" in message.edit.call_args.kwargs

    async def test_edits_follow_a_host_swap(self, music_player, mock_song):
        """The tick must re-read the host pointer each pass — a -now re-pin or
        a command response adopting the host mid-song redirects the next tick
        to the new message with no updater restart."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        old_host = self._host(music_player)
        new_host = AsyncMock(spec=discord.Message)

        calls = 0

        async def _sleep(_secs):
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

    async def test_skips_edit_while_paused(self, music_player, mock_song):
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = True
        music_player._guild.voice_client = vc
        message = self._host(music_player)

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        message.edit.assert_not_awaited()

    async def test_returns_when_song_changed_under_it(self, music_player, mock_song):
        """loop() owns cancellation on song transition, but this guard protects
        against a stray tick landing after the song changed (Design §4)."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = MagicMock()  # a different song than the one passed in
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = self._host(music_player)

        with patch("asyncio.sleep", new=AsyncMock()):
            await music_player._progress_updater(mock_song)  # returns, no raise

        message.edit.assert_not_awaited()

    async def test_goes_dormant_on_message_not_found(self, music_player, mock_song):
        """Deleting the host is no longer opt-out: the updater releases the
        host and keeps looping (dormant) so the next command response or -now
        can re-host the block with an accurate bar."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = self._host(music_player)
        message.edit.side_effect = discord.NotFound(MagicMock(), "message deleted")

        # Tick 1: NotFound → release + stay alive. Tick 2: dormant no-op.
        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        message.edit.assert_awaited_once()
        assert music_player._np_host_message is None

    async def test_not_found_keeps_host_adopted_mid_tick(self, music_player, mock_song):
        """Adopt is lock-free, so a command response can swap in a new host
        while this tick's PATCH is in flight. A NotFound then must not release
        the NEW host — that would permanently orphan its block (branch review
        M1)."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        old_host = self._host(music_player)
        new_host = AsyncMock(spec=discord.Message)

        async def _edit_racing_an_adopt(*args, **kwargs):
            music_player._np_host_message = new_host  # adopt lands mid-PATCH
            raise discord.NotFound(MagicMock(), "old host deleted")

        old_host.edit.side_effect = _edit_racing_an_adopt

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        assert music_player._np_host_message is new_host

    async def test_logs_and_continues_on_http_exception(self, music_player, mock_song):
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        music_player._guild.voice_client = vc
        message = self._host(music_player)
        message.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._progress_updater(mock_song)

        assert message.edit.await_count == 2  # kept ticking despite the failure


# ── CancelProgressTask ────────────────────────────────────────────────────────


class TestCancelProgressTask:
    async def test_noop_when_no_progress_task(self, music_player):
        music_player._progress_task = None
        await music_player._cancel_progress_task()

    async def test_noop_when_progress_task_already_done(self, music_player):
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        music_player._progress_task = task
        await music_player._cancel_progress_task()
        task.cancel.assert_not_called()

    async def test_cancels_and_awaits_in_flight_progress_task(self, music_player):
        async def _long():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long())
        music_player._progress_task = task
        await music_player._cancel_progress_task()
        assert task.cancelled()
        assert music_player._progress_task is None

    async def test_song_transition_retires_task_before_next_send(
        self, music_player, mock_song
    ):
        """Closes the song-transition race found in design review: the previous
        song's progress task must be fully retired — not just .cancel()'d —
        before the next song's _send_now_playing() sends a new message."""
        call_order: list[str] = []

        async def _never_finishes():
            try:
                await asyncio.sleep(100)
            finally:
                call_order.append("old_task_retired")

        music_player.current_song = mock_song  # _send_now_playing builds off it
        music_player._progress_task = asyncio.create_task(_never_finishes())
        await asyncio.sleep(0)  # let the task actually start before cancelling it

        original_send = music_player._channel.send

        async def _tracked_send(*a, **kw):
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
    async def _cleanup(self, music_player):
        yield
        await music_player._cancel_pause_debounce()
        # _progress_task in these tests is a bare MagicMock sentinel (truthy for
        # the "is not None" check), not a real awaitable task — reset directly
        # rather than going through _cancel_progress_task()'s await.
        music_player._progress_task = None

    async def test_noop_when_no_current_song(self, music_player):
        music_player.current_song = None
        music_player.mark_paused()
        assert music_player._pause_debounce_task is None

    async def test_single_call_fires_after_debounce_window(
        self, music_player, mock_song
    ):
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
        self, music_player, mock_song
    ):
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
        self, music_player, mock_song
    ):
        music_player.current_song = mock_song
        music_player._np_host_message = None
        music_player._progress_task = None
        music_player.bot.change_presence = AsyncMock()

        music_player.mark_paused()
        await music_player._pause_debounce_task
        await asyncio.gather(*list(music_player._background_tasks))

        music_player.bot.change_presence.assert_awaited_once()


# ── MarkPausedResumed ──────────────────────────────────────────────────────────


class TestPlayerPauseResume:
    """MusicPlayer.pause()/resume() own all pause-tracking side effects in one
    place: the voice-client call, Redis epoch accounting, and the debounced
    progress-bar/Activity refresh — so a future call site can't forget one."""

    @pytest.fixture(autouse=True)
    async def _cleanup(self, music_player):
        yield
        await music_player._cancel_pause_debounce()
        music_player._progress_task = None

    async def test_pause_calls_vc_pause(self, music_player, mock_song):
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.pause(vc)
        vc.pause.assert_called_once()

    async def test_pause_writes_to_store(self, music_player, mock_song, fake_redis):
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.pause(vc)
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"pause_start_epoch" in state

    async def test_pause_schedules_debounced_update(self, music_player, mock_song):
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.pause(vc)
        assert music_player._pause_debounce_task is not None

    async def test_pause_skips_store_when_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx, mock_song
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await mp.pause(vc)  # must not raise
        vc.pause.assert_called_once()

    async def test_resume_calls_vc_resume(self, music_player, mock_song):
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.resume(vc)
        vc.resume.assert_called_once()

    async def test_resume_writes_to_store(self, music_player, mock_song, fake_redis):
        music_player.current_song = mock_song
        await music_player.store.on_pause(1000.0)
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.resume(vc)
        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"pause_start_epoch" not in state

    async def test_resume_schedules_debounced_update(self, music_player, mock_song):
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await music_player.resume(vc)
        assert music_player._pause_debounce_task is not None

    async def test_resume_skips_store_when_absent(
        self, mock_bot, mock_guild, mock_channel, mock_ctx, mock_song
    ):
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)
        await mp.resume(vc)  # must not raise
        vc.resume.assert_called_once()


class TestMarkPausedResumed:
    @pytest.fixture(autouse=True)
    async def _cleanup(self, music_player):
        yield
        await music_player._cancel_pause_debounce()
        music_player._progress_task = None

    async def test_mark_paused_schedules_debounced_update(
        self, music_player, mock_song
    ):
        music_player.current_song = mock_song
        music_player.mark_paused()
        assert music_player._pause_debounce_task is not None

    async def test_mark_resumed_schedules_debounced_update(
        self, music_player, mock_song
    ):
        music_player.current_song = mock_song
        music_player.mark_resumed()
        assert music_player._pause_debounce_task is not None

    async def test_scheduled_tasks_tracked_via_background_tasks(
        self, music_player, mock_song
    ):
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
        await music_player._pause_debounce_task
        # Debounce task itself is discarded from the set once done (done_callback).
        assert music_player._pause_debounce_task not in music_player._background_tasks


# ── BuildNextUpEmbed ──────────────────────────────────────────────────────────


class TestBuildNextUpEmbed:
    def test_returns_none_when_queue_empty(self, music_player):
        assert music_player._build_next_up_embed() is None

    def test_returns_blue_embed_with_song_details(self, music_player, mock_author):
        music_player.queue._display.append(
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
        music_player.queue._display.append(
            YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "resolving..." in embed.description

    def test_shows_placeholder_duration_when_unknown(self, music_player, mock_author):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "`?:??`" in embed.description

    def test_only_uses_first_queued_song(self, music_player, mock_author):
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=1", "First", mock_author, duration=60)
        )
        music_player.queue._display.append(
            QueueObject("https://yt.com/v=2", "Second", mock_author, duration=60)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "First" in embed.description
        assert "Second" not in embed.description

    def test_includes_est_playing_at_eta(self, music_player, mock_author):
        music_player.queue._display.append(
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
        requester_line = now_playing_embed.description.split("\n")[-1]
        finish_time = requester_line.split("Estimated finish: ")[1]
        assert finish_time in next_up_embed.description


# ── PrefetchNextSong ──────────────────────────────────────────────────────────


class TestPrefetchNextSong:
    async def test_returns_none_when_queue_empty(self, music_player):
        result = await music_player._prefetch_next_song()
        assert result is None

    async def test_returns_ytdl_on_success(self, music_player, queue_obj):
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
        self, music_player, queue_obj, fake_redis
    ):
        """A prefetch whose resolve/stream raises must retire its dequeue
        everywhere — pending was popped by get_nowait(), so leaving the
        display/Redis heads in place would make the next commit retire the
        wrong entry."""
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
        self, music_player, queue_obj, fake_redis
    ):
        """_stream_source catches its own exceptions and returns None — that
        path must retire the dequeue exactly like the raise path."""
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
        self, music_player, queue_obj, queue_obj_no_meta, fake_redis
    ):
        """-clear/-shuffle/-remove cancel the prefetch before mutating; the
        item it holds must return to the front of the pending queue — not be
        dropped — so the mutation drains/reorders it with everything else
        instead of silently losing the next song."""
        await music_player.queue.put([queue_obj, queue_obj_no_meta])
        started = asyncio.Event()
        never_set = asyncio.Event()

        async def hang(self, source):
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
        self, music_player, queue_obj, mock_song
    ):
        """A failure landing between the committed dequeue and the normal
        song-end task_done() (here: the voice client vanished during resolve,
        so the isinstance assert fails before vc.play) must still balance the
        get() in the loop's exception handler — otherwise the queue's task
        counter drifts upward on every such failure."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)
        music_player._guild.voice_client = None

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
    async def test_returns_item_from_queue(self, music_player, queue_obj):
        await music_player.queue._pending.put(queue_obj)
        result = await music_player.queue_get()
        assert result is queue_obj


# ── RestoreStateTtlRefresh ────────────────────────────────────────────────────


class TestRestoreStateTtlRefresh:
    async def test_ttl_refreshed_after_successful_restore(
        self, music_player, fake_redis
    ):
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.8")
        await fake_redis.expire(music_player.store.state_key(), 10)

        await music_player._restore_state()

        ttl = await fake_redis.ttl(music_player.store.state_key())
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
        await fake_redis.rpush(music_player.store.queue_key(), b"!!!bad json!!!", valid)
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        assert music_player.queue.qsize() == 1
        item = await music_player.queue.get()
        assert item.title == "Good Song"


# ── Loop ──────────────────────────────────────────────────────────────────────


class TestLoop:
    @pytest.fixture
    def mock_song(self):
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
        # -playnow flags a real YTDL always carries — truthy MagicMock
        # attributes would trip the loop's start_paused/is_resume gates.
        song.interjected = False
        song.is_resume = False
        song.start_paused = False
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

        sent_embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert sent_embeds[0].description == "Failed to load the next song, skipping."

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
            music_player.queue._pending._unfinished_tasks == 0
        )  # task_done() balanced get()
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
        self, music_player, queue_obj, mock_song
    ):
        # _restore_complete is never set unless start() is called or _restore_state() runs.
        # Set it here so the restore gate in loop() does not block for 10s.
        music_player._restore_complete.set()

        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

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

    async def test_song_that_produced_no_audio_is_not_treated_as_played(
        self, music_player, queue_obj, mock_song
    ):
        """Regression: a 403 kills ffmpeg instantly, which discord.py reports exactly
        like a song that finished. The bot then advanced in silence, logged nothing, kept
        the dead URL cached, and filed the song in history as if it had been heard."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False  # ffmpeg never delivered a frame

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
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

    async def test_plays_song_writes_duration_uploader_requester_atomically(
        self, music_player, queue_obj, mock_song, mock_author
    ):
        """Regression: duration/uploader/requester_id must land in the same
        atomic pop_queue_and_start_song() write as url/title — not via a
        separate, later, non-atomic call that could crash-drop the fields."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.duration_secs = 240
        mock_song.uploader = "Test Channel"
        mock_song.requester = mock_author

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
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
        self, music_player, queue_obj, mock_song
    ):
        """After a song finishes, -now must not serve the finished song's embed
        via the crash-recovery elif — play_message is cleared with current_song."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)
        music_player.play_message = discord.Embed(title="stale")

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

        assert music_player.current_song is None
        assert music_player.play_message is None

    async def test_loop_clears_play_message_on_playback_error(
        self, music_player, queue_obj
    ):
        """The generic exception path must also clear play_message so a failed
        song is never served by -now as still playing."""
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)
        music_player.play_message = discord.Embed(title="stale")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=RuntimeError("ffmpeg gone"))
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

        assert music_player.play_message is None

    async def test_loop_backdates_play_start_epoch_by_start_offset(
        self, music_player, queue_obj, mock_song
    ):
        """A song started with FFmpeg -ss must persist play_start_epoch backdated
        by the offset, so recovery position math (now - epoch - pauses) yields
        the true audio position rather than time-since-vc.play()."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.start_offset = 90

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
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
        self, music_player, queue_obj, mock_song, fake_redis
    ):
        """Crash-window regression (the Issue-3 bug): the now_playing snapshot
        must be committed in the start transaction, *before* any Discord I/O —
        by the time _send_now_playing runs, the hash already shows this song."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
        music_player.play_next.wait = AsyncMock()

        np_at_send_time: dict = {}

        async def _capture_send(_self, song):
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
        self, music_player, queue_obj, mock_song
    ):
        """When a song ends, loop() must capture the host, release it (so the
        next song's adopt retires nothing), and fire the finalize-embed task
        with the song/host/own-embeds that just finished — before current_song
        and the host state get overwritten for the next iteration."""
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc
        music_player.play_next.wait = AsyncMock()

        sent_message = MagicMock(spec=discord.Message)

        async def _fake_send_now_playing(_self, song):
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

        finalize_mock.assert_called_once_with(mock_song, sent_message, [])
        assert music_player._np_host_message is None  # released, not retired

    async def test_unhandled_exception_sends_error_message(
        self, music_player, queue_obj
    ):
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

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

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=RuntimeError("ffmpeg gone"))
        music_player._guild.voice_client = vc

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
    async def test_set_after_successful_restore(self, music_player):
        music_player.bot.wait_until_ready = AsyncMock()
        await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_set_even_when_restore_raises(self, music_player):
        music_player.bot.wait_until_ready = AsyncMock()
        with patch.object(
            music_player.store,
            "get_playback_snapshot",
            new=AsyncMock(side_effect=Exception("redis down")),
        ):
            await music_player._restore_state()
        assert music_player._restore_complete.is_set()

    async def test_set_and_restore_aborted_when_state_read_fails(self, music_player):
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
        data = dataclasses.replace(_NP_DATA, thumbnail="")
        embed = music_player._build_now_playing_embed_from_data(data)
        assert not embed.thumbnail.url

    def test_footer_contains_bitrate(self, music_player):
        embed = music_player._build_now_playing_embed_from_data(_NP_DATA)
        assert "128" in embed.footer.text

    def test_default_fields_render_as_empty_strings(self, music_player):
        data = NowPlayingData(title="Minimal")  # all other fields defaulted
        embed = music_player._build_now_playing_embed_from_data(data)
        assert "Minimal" in embed.title


# ── _restore_state: now-playing embed restoration ────────────────────────────


class TestRestoreStateNowPlaying:
    async def test_restores_play_message_from_redis(self, music_player, fake_redis):
        """If now_playing hash exists in Redis, play_message is populated on restore."""
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
        # -playnow flags a real YTDL always carries — truthy MagicMock
        # attributes would trip the loop's start_paused/is_resume gates.
        song.interjected = False
        song.is_resume = False
        song.start_paused = False
        return song

    async def test_update_activity_called_at_song_start_and_end(
        self, music_player, queue_obj, mock_song
    ):
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append("Test Song - url")

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
        await music_player.queue._pending.put(queue_obj)
        await music_player.queue._pending.put(queue_obj2)
        music_player.queue._display.append("Song 1 - url")
        music_player.queue._display.append("Song 2 - url")

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
            music_player.queue._cleared = True
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
            # Unrelated to this test (prefetch/cleanup) — the bare object.__new__()
            # VoiceClient double below has no real _player, so the real
            # update_activity() would crash calling vc.is_paused() on it.
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
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

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append("Test Song - url")

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        music_player._guild.voice_client = vc

        mock_song.cleanup = MagicMock()

        async def _stream_and_clear(_self, source):
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
def mock_vc():
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    return vc


@pytest.fixture
def live_song(mock_song):
    """mock_song with the -playnow flags a real YTDL carries (a bare MagicMock
    attribute would read as a truthy mock and trip the replace-semantics gate)."""
    mock_song.interjected = False
    mock_song.is_resume = False
    mock_song.start_paused = False
    return mock_song


@pytest.fixture
def playnow_obj(mock_author):
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=urgent",
        title="Urgent Song",
        requester=mock_author,
        duration=120,
        interjected=True,
    )


class TestInterject:
    async def test_returns_none_without_current_song(
        self, music_player, playnow_obj, mock_vc
    ):
        music_player.current_song = None
        assert await music_player.interject(playnow_obj, mock_vc) is None
        mock_vc.stop.assert_not_called()

    async def test_front_inserts_playnow_then_resume(
        self, music_player, live_song, playnow_obj, mock_vc, mock_author
    ):
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
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(playnow_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.start_paused is True
        assert outcome is not None and outcome.was_paused is True

    async def test_replace_semantics_skip_resume_for_interjection(
        self, music_player, live_song, playnow_obj, mock_vc
    ):
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
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        live_song.elapsed_secs = 207.0  # 3s left of 210 — below the 5s floor
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        assert music_player.queue.display_items() == [playnow_obj]
        assert outcome is not None and outcome.resume_position is None
        assert music_player._skip_history_for is None

    async def test_eof_cap_pulls_position_back(
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        live_song.elapsed_secs = 205.0  # 5s left: resumable, but capped to 200
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.ts == 200  # duration 210 − 10s EOF margin
        assert outcome is not None and outcome.resume_position == 200

    async def test_no_webpage_url_skips_resume(
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        live_song.webpage_url = None
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        outcome = await music_player.interject(playnow_obj, mock_vc)

        assert music_player.queue.display_items() == [playnow_obj]
        assert outcome is not None and outcome.resume_position is None

    async def test_stop_skipped_when_song_changed_during_insert(
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        async def put_front_and_advance(items):
            music_player.current_song = MagicMock()  # loop moved on mid-await

        from src.guild_queue import GuildQueue

        # Class-level patch: GuildQueue uses __slots__, so patch.object on the
        # instance can't set the attribute.
        with patch.object(GuildQueue, "put_front", side_effect=put_front_and_advance):
            await music_player.interject(playnow_obj, mock_vc)

        mock_vc.stop.assert_not_called()
        assert music_player._skip_history_for is None

    async def test_neutralizes_running_prefetch_first(
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        blocker = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = blocker

        await music_player.interject(playnow_obj, mock_vc)

        assert blocker.cancelled()
        assert music_player._prefetch_task is None


class TestNeutralizePrefetch:
    async def test_no_task_is_noop(self, music_player):
        music_player._prefetch_task = None
        await music_player._neutralize_prefetch()  # must not raise

    async def test_running_task_cancelled_and_cleared(self, music_player):
        task = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()
        assert task.cancelled()
        assert music_player._prefetch_task is None

    async def test_completed_task_requeues_rebuilt_item_and_kills_ffmpeg(
        self, music_player, live_song, mock_author
    ):
        # Simulate the prefetch's own dequeue: pending pops, display keeps the
        # entry (the prefetch commit was still pending).
        original = QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.cleanup = MagicMock()

        async def _done():
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
        self, music_player, live_song, mock_author
    ):
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

        async def _done():
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
        self, music_player, live_song, mock_author
    ):
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

        async def _done():
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.interjected is True
        assert rebuilt.ts is None  # start_offset 0 → no bogus -ss

    async def test_completed_task_with_none_result_is_noop(self, music_player):
        async def _done():
            return None

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()
        assert music_player.queue.qsize() == 0

    async def test_completed_task_that_raised_is_swallowed(self, music_player):
        async def _boom():
            raise RuntimeError("prefetch exploded")

        task = asyncio.create_task(_boom())
        with contextlib.suppress(RuntimeError):
            await task
        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()  # must not raise
        assert music_player.queue.qsize() == 0


class TestAnnounceResume:
    async def test_playing_wording(self, music_player, live_song, mock_channel):
        live_song.elapsed_secs = 42.0
        live_song.is_resume = True
        await music_player._announce_resume(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert "Resuming" in embed.description
        assert "0:42" in embed.description

    async def test_paused_wording(self, music_player, live_song, mock_channel):
        live_song.elapsed_secs = 42.0
        live_song.is_resume = True
        live_song.start_paused = True
        await music_player._announce_resume(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert "still paused" in embed.description
        assert "-resume" in embed.description

    async def test_send_failure_swallowed(self, music_player, live_song, mock_channel):
        mock_channel.send.side_effect = RuntimeError("channel gone")
        await music_player._announce_resume(live_song)  # must not raise


class TestRemainingSecs:
    def test_normal_item_full_duration(self, queue_obj):
        from src.musicplayer import _remaining_secs

        assert _remaining_secs(queue_obj) == 210

    def test_resume_entry_counts_only_tail(self, mock_author):
        from src.musicplayer import _remaining_secs

        item = QueueObject(
            "https://yt.com/v=1", "T", mock_author, ts=150, duration=210, is_resume=True
        )
        assert _remaining_secs(item) == 60

    def test_unknown_duration_is_none(self, queue_obj_no_meta):
        from src.musicplayer import _remaining_secs

        assert _remaining_secs(queue_obj_no_meta) is None

    def test_non_resume_ts_does_not_shrink_duration(self, mock_author):
        # A ?t= start offset is a playback preference, not a shorter song —
        # only resume entries are known to play just their tail.
        from src.musicplayer import _remaining_secs

        item = QueueObject("https://yt.com/v=1", "T", mock_author, ts=150, duration=210)
        assert _remaining_secs(item) == 210


class TestResumeEntryDisplay:
    async def test_queue_embed_shows_resume_note(self, music_player, mock_author):
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
        assert "⏮ resumes at `2:30`" in embed.description

    async def test_plain_ts_note_unchanged(self, music_player, mock_author):
        item = QueueObject("https://yt.com/v=1", "T", mock_author, ts=30, duration=210)
        await music_player.queue.put([item])
        embed = music_player.queue_embed()
        assert "starts at `30s`" in embed.description


class TestEstimatedFinishUsesRemaining:
    def test_offset_start_finishes_sooner(self, music_player, live_song):
        from src.musicplayer import _fmt_finish_time

        live_song.start_offset = 100  # 110s of the 210s song remain
        before = _fmt_finish_time(110)
        embed = music_player._build_now_playing_embed(live_song)
        after = _fmt_finish_time(110)
        assert (before in embed.description) or (after in embed.description)

    def test_position_override_shrinks_remaining(self, music_player, live_song):
        from src.musicplayer import _fmt_finish_time

        before = _fmt_finish_time(10)
        embed = music_player._build_now_playing_embed(
            live_song, position_override=200.0
        )
        after = _fmt_finish_time(10)
        assert (before in embed.description) or (after in embed.description)


class TestHistorySkipMarker:
    """The _skip_history_for identity marker consumed by loop()'s history step."""

    async def _run_one_song(self, music_player, queue_obj, mock_song):
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

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

    async def test_marker_for_current_song_skips_history_once(
        self, music_player, queue_obj, mock_song
    ):
        """interject() marked this song (resume entry pending) — its stop
        transition must not record it; the tail's own end will."""
        music_player._skip_history_for = mock_song
        await self._run_one_song(music_player, queue_obj, mock_song)
        assert len(music_player.history) == 0
        assert music_player._skip_history_for is None

    async def test_stale_marker_does_not_eat_next_songs_history(
        self, music_player, queue_obj, mock_song
    ):
        """A marker left for a song that ended naturally during interject()'s
        awaits (its history step already ran) must not suppress the NEXT
        song's entry — the identity check makes it a no-op that clears."""
        music_player._skip_history_for = MagicMock()  # some other, ended song
        await self._run_one_song(music_player, queue_obj, mock_song)
        assert len(music_player.history) == 1
        assert mock_song.title in music_player.history[0]
        assert music_player._skip_history_for is None


class TestInterjectPostNeutralizeRecheck:
    async def test_song_changed_during_neutralize_returns_none(
        self, music_player, live_song, playnow_obj, mock_vc
    ):
        """Neutralize can block up to yt-dlp's socket timeout (cancellation
        can't interrupt the executor thread) — if the song ended and the loop
        moved on in that window, interject bails to the command's fallback
        instead of building a resume entry for a finished song."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        async def neutralize_and_advance(_self):
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

    async def _run_one_song(self, music_player, queue_obj, mock_song, vc):
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        music_player.bot.is_closed.side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.queue._pending.put(queue_obj)
        music_player.queue._display.append(queue_obj)

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
            patch.object(MusicPlayer, "pause", new=AsyncMock()) as pause_mock,
            patch.object(
                MusicPlayer, "_announce_resume", new=AsyncMock()
            ) as announce_mock,
        ):
            await music_player.loop()
        return pause_mock, announce_mock

    def _vc(self):
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        vc.pause = MagicMock()
        return vc

    async def test_start_paused_parks_synchronously_and_engages_bookkeeping(
        self, music_player, queue_obj, mock_song
    ):
        mock_song.start_paused = True
        vc = self._vc()
        pause_mock, _ = await self._run_one_song(music_player, queue_obj, mock_song, vc)
        # Synchronous park right after vc.play (frame-leak guard) …
        vc.pause.assert_called_once()
        # … plus the full pause() entry point (Redis epochs, debounced refresh).
        pause_mock.assert_awaited_once()

    async def test_resume_entry_announced_at_start(
        self, music_player, queue_obj, mock_song
    ):
        mock_song.is_resume = True
        vc = self._vc()
        _, announce_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        announce_mock.assert_awaited_once_with(mock_song)

    async def test_plain_song_neither_parks_nor_announces(
        self, music_player, queue_obj, mock_song
    ):
        vc = self._vc()
        pause_mock, announce_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        vc.pause.assert_not_called()
        pause_mock.assert_not_awaited()
        announce_mock.assert_not_awaited()
