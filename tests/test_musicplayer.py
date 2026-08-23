"""Tests for src/musicplayer.py — queue operations, embed building, and Redis integration."""

import redis.asyncio as aioredis
import asyncio
import contextlib
import logging
import dataclasses
import re
from zoneinfo import ZoneInfo
import time
from typing import Any, Never, cast
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import orjson
import pytest
from opentelemetry import trace as trace_api

from src.debug import RuntimeSnapshot
from src.guild_queue import GuildQueue, RemoveMode
from src.guild_state import (
    ANALYTICS_ZERO,
    Analytics,
    DEFAULT_TIMEZONE,
    GuildStateData,
    HistoryEntry,
    NowPlayingData,
    SongQueueEntry,
    parse_queue_entry,
)
from src.redis_client import GuildRedisStore
from src.musicplayer import (
    MusicPlayer,
    StreamFailure,
    _BAR_WIDTH,
    _build_progress_bar,
    _reached_end,
    _fmt_finish_time,
    _fmt_total_duration,
    _requester_mention,
)
from src.redis_client import HISTORY_CACHE_LIMIT
from src.sources import YTSource
from src.util import cancel_task, fmt_duration
from src.youtube import NpHostRef, QueueObject, YTDL
from tests.helpers import seed_queue, described, mocked, queue_object, stub_create_task


@pytest.fixture(autouse=True)
def _stub_prefetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub YTDL.prefetch_stream for every test in this module.

    queue_put() spawns a prefetch task per QueueObject, so without this any
    queue_put of a yt.com URL fires a real yt-dlp request in the background. Tests
    asserting on prefetch override this with their own patch(), which wins."""
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
    # Interjection flags a real YTDL always carries — as bare MagicMock attributes
    # they'd read truthy and trip the loop's start_paused/is_resume gates.
    song.interjected = False
    song.is_resume = False
    song.start_paused = False
    # Enqueue analytics: a real (zero) Analytics, since HistoryEntry.from_song
    # clamps its fields into the play_history column domain. query_source likewise
    # a real string — the slug clamp regex-matches it and a MagicMock raises
    # TypeError there, exactly as a MagicMock title would.
    song.analytics = ANALYTICS_ZERO
    song.query_source = ""
    # Same reason: the resume tail an interjection builds carries it, and it is
    # serialized straight to the queue mirror.
    song.user_input = None
    # Unstamped, like a song the loop has not started yet: the loop's or-stamp
    # writes the real clock here, and the epoch clamp raises on a MagicMock.
    song.played_at = 0.0
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


# ── Archive wiring ────────────────────────────────────────────────────────────


class TestOutboxNotifyWiring:
    """MusicPlayer decides GuildHistory's outbox notify from the bot's drainer, and
    both answers must be constructible: enabled, the notify wakes the drain; disabled
    — the SHIP default — there is no drainer and the None is passed explicitly.

    Nothing else can see this. Every other player test builds from `mock_bot`, whose
    auto-vivified attributes are truthy, so the None arm is reachable only in
    production, and coverage reports the one-line conditional covered the moment
    either arm runs. Dropping the guard passed the entire suite while raising
    AttributeError on the first -play of any default deployment."""

    def test_a_running_drainer_is_wired_to_the_history(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        drainer = MagicMock()
        mock_bot.history_drainer = drainer
        mp = MusicPlayer(
            mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis
        )
        assert mp.history._on_outbox_push is drainer.notify

    def test_no_drainer_wires_none_rather_than_crashing(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        # The archive-disabled shape MusicBotApp.__init__ really produces —
        # pinned by TestAppInitDefaults in test_main.py.
        mock_bot.history_drainer = None
        mp = MusicPlayer(
            mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis
        )
        assert mp.history._on_outbox_push is None

    async def test_a_play_is_still_recorded_with_no_drainer(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """Construction is not the whole contract — the queue must still
        work, or the default deployment would have a silent -history."""
        mock_bot.history_drainer = None
        mp = MusicPlayer(
            mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis
        )
        await mp.history.add(
            HistoryEntry(title="Song", webpage_url="https://yt.com/v=1", played_at=1.0)
        )
        assert [e.title for e in await mp.history.recent(10)] == ["Song"]


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
        """PST or PDT: the suffix follows the zone's own abbreviation now, so half
        the year US/Pacific is legitimately PDT."""
        rendered = _fmt_finish_time(90, ZoneInfo(DEFAULT_TIMEZONE))
        assert re.match(r"^\d{1,2}:\d{2} (AM|PM) P[SD]T$", rendered)

    def test_the_suffix_follows_the_configured_zone(self) -> None:
        """A guild set to London must not be quoted London time labelled PST — the
        suffix was a hardcoded literal before the zone became configurable."""
        rendered = _fmt_finish_time(90, ZoneInfo("Europe/London"))
        assert re.match(r"^\d{1,2}:\d{2} (AM|PM) (GMT|BST)$", rendered)

    def test_no_uncertainty_prefix(self) -> None:
        # Unlike _fmt_eta(), a song's own remaining duration is never
        # uncertain — no "~" prefix and no bold markdown wrapping.
        result = _fmt_finish_time(90, ZoneInfo(DEFAULT_TIMEZONE))
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
        assert len(music_player.queue._items) == 1
        assert isinstance(music_player.queue._items[0], QueueObject)
        assert music_player.queue._items[0].title == "Test Song"

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
        assert len(music_player.queue._items) == 3

    async def test_put_multiple_singles_increments_size(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 4
        assert len(music_player.queue._items) == 4

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


# ── QueuePutNext ──────────────────────────────────────────────────────────────


class TestQueuePutNext:
    """ "Next" means next, which put_front alone does not deliver.

    loop() spawns _prefetch_next_song() on every iteration and its get_nowait()
    holds an open claim for the rest of the current song, so with anything queued
    at all a bare put_front lands at _cursor — behind that claim — and the song
    plays second. Every test here is that ordering, or the bookkeeping that makes
    it safe."""

    @staticmethod
    def _titles(mp: MusicPlayer) -> list[str]:
        return [queue_object(item).title for item in mp.queue.display_items()]

    async def test_it_lands_ahead_of_a_running_prefetchs_claim(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The headline case, and the common one: a song is playing and B is
        queued, so the prefetch already owns B. Without the neutralize the new
        song plays after B rather than before it."""
        first = QueueObject("https://yt.com/v=b", "B", mock_author)
        await music_player.queue.put([first])
        claimed = asyncio.Event()

        async def hang(_source: Any) -> QueueObject:
            claimed.set()
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

        # Patched on the class, not the instance: MusicPlayer has __slots__.
        with patch.object(
            MusicPlayer, "_resolve_source", new_callable=AsyncMock, side_effect=hang
        ):
            music_player._prefetch_task = asyncio.create_task(
                music_player._prefetch_next_song()
            )
            await claimed.wait()
            # The claim is real: B has left the pending region entirely.
            assert music_player.queue.qsize() == 0

            newcomer = QueueObject("https://yt.com/v=x", "X", mock_author)
            await music_player.queue_put_next(newcomer, prefetch=False)

        assert self._titles(music_player) == ["X", "B"]
        assert music_player.queue.qsize() == 2

    async def test_a_completed_prefetch_is_rebuilt_behind_the_new_song(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """A finished prefetch bypasses the queue entirely — it would have played
        INSTEAD of the insert. Its rebuilt equivalent goes back behind the
        newcomer, and its FFmpeg subprocess is killed rather than leaked."""
        original = QueueObject("https://yt.com/v=b", "B", mock_author)
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        newcomer = QueueObject("https://yt.com/v=x", "X", mock_author)
        await music_player.queue_put_next(newcomer, prefetch=False)

        live_song.cleanup.assert_called_once()
        assert self._titles(music_player) == ["X", live_song.title]

    async def test_it_never_respawns_the_prefetch(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """_prefetch_task is a single slot with a claim-then-null protocol shared
        with loop(). Re-spawning here would race the loop's own spawn at the next
        song start, orphaning one task with a claim nothing settles and drifting
        _cursor by one permanently. The one-song gap is accepted, as interject()
        accepts it."""
        await music_player.queue.put(
            [QueueObject("https://yt.com/v=b", "B", mock_author)]
        )
        await music_player.queue_put_next(
            QueueObject("https://yt.com/v=x", "X", mock_author), prefetch=False
        )
        assert music_player._prefetch_task is None

    async def test_with_no_prefetch_it_is_a_plain_front_insert(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        music_player._prefetch_task = None
        for title in ("B", "C"):
            await music_player.queue.put(
                [QueueObject(f"https://yt.com/v={title}", title, mock_author)]
            )

        await music_player.queue_put_next(
            QueueObject("https://yt.com/v=x", "X", mock_author), prefetch=False
        )

        assert self._titles(music_player) == ["X", "B", "C"]

    async def test_an_empty_queue_degrades_to_an_append(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """ "Play next" and "play" are the same request when nothing is queued —
        which is what lets --next need no special case for an idle bot."""
        await music_player.queue_put_next(
            QueueObject("https://yt.com/v=x", "X", mock_author), prefetch=False
        )
        assert self._titles(music_player) == ["X"]

    async def test_it_mirrors_the_new_order_to_redis(
        self, music_player: MusicPlayer, mock_author: MagicMock, fake_redis: Any
    ) -> None:
        """The mirror is the queue a restart reads back, so an insert that is
        right in memory and wrong in Redis survives exactly until the next
        crash."""
        assert music_player.store is not None
        await music_player.queue.put(
            [QueueObject("https://yt.com/v=b", "B", mock_author)]
        )

        await music_player.queue_put_next(
            QueueObject("https://yt.com/v=x", "X", mock_author), prefetch=False
        )

        stored = [
            orjson.loads(raw)["title"]
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert stored == ["X", "B"]

    async def test_it_still_warms_the_stream_url(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The prefetch it suppresses is loop()'s queue-claiming one. This one only
        writes ytdl:stream:*, and it is what keeps the neutralize affordable: the
        song about to play is warmed even though no claim is held for it."""
        newcomer = QueueObject("https://yt.com/v=x", "X", mock_author)
        with patch(
            "src.musicplayer.YTDL.prefetch_stream", new_callable=AsyncMock
        ) as mock_pf:
            await music_player.queue_put_next(newcomer)
            await asyncio.sleep(0)
        mock_pf.assert_awaited_once()
        assert mock_pf.call_args[0][0] == newcomer


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
        assert len(music_player.queue._items) == 3

        await music_player.queue_clear()
        assert len(music_player.queue._items) == 0

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


class TestQueueClearFlushesPlayedSongs:
    """A song records exactly once, when its queue object leaves the queue for
    good. An interjection's resume tail cleared before it finishes has already been
    heard and will never reach the loop's write site, so -clear is its only
    chance at a history row."""

    @pytest.fixture(autouse=True)
    def _setup(self, _stub_queue_put_tasks: None) -> None:
        pass

    async def test_played_tail_is_recorded_once(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        tail = QueueObject(
            "https://yt.com/v=heard",
            "Heard Song",
            mock_author,
            ts=95,
            duration=240,
            is_resume=True,
            played_at=1752530000.0,
        )
        await music_player.queue_put(tail)

        await music_player.queue_clear()

        assert len(music_player.history) == 1
        entry = music_player.history[0]
        assert entry.webpage_url == "https://yt.com/v=heard"
        assert entry.played_at == 1752530000.0
        assert entry.played_secs == 95  # ts is absolute, not per-fragment

    async def test_flushed_row_points_at_the_card_that_hosted_the_play(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """End of the chain: the tail's np_* fields become the row's host pair, so
        a play destroyed before its tail could finish is still traceable back to
        the message that carried its bar."""
        tail = QueueObject(
            "https://yt.com/v=heard",
            "Heard",
            mock_author,
            ts=95,
            is_resume=True,
            played_at=1752530000.0,
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
            np_dedicated=True,
        )
        await music_player.queue_put(tail)

        await music_player.queue_clear()

        entry = music_player.history[0]
        assert (entry.message_id, entry.channel_id) == (
            777777777777777777,
            888888888888888888,
        )

    async def test_unplayed_entries_are_not_recorded(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # played_at stays 0.0: an ordinary queued song was never heard, and
        # recording it would invent a play out of a cancelled one.
        await music_player.queue_put(
            QueueObject("https://yt.com/v=never", "Never Played", mock_author)
        )
        await music_player.queue_put(YTSource(ytsearch="ytsearch:some song"))

        cleared = await music_player.queue_clear()

        assert len(cleared) == 2
        assert len(music_player.history) == 0

    async def test_mixed_queue_records_only_what_played(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        played = QueueObject(
            "https://yt.com/v=heard", "Heard", mock_author, ts=30, played_at=1.0
        )
        await music_player.queue_put(played)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=queued", "Queued", mock_author)
        )
        await music_player.queue_put(YTSource(url="https://yt.com/v=lazy"))

        await music_player.queue_clear()

        assert [e.webpage_url for e in music_player.history] == [
            "https://yt.com/v=heard"
        ]

    async def test_flush_failure_surfaces_instead_of_reporting_success(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The flush runs before the return, so a failure reaches the command's
        error path. Swallowed, -clear would reply "queue cleared" over plays it
        dropped — and the queue is already gone by then, so nothing could retry."""
        await music_player.queue_put(
            QueueObject("https://yt.com/v=heard", "Heard", mock_author, played_at=1.0)
        )
        # The class, not the instance: GuildHistory has __slots__.
        with patch.object(
            type(music_player.history),
            "add",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(RuntimeError):
                await music_player.queue_clear()

    async def test_every_tail_of_a_stacked_queue_is_recorded(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """What a 3-deep interjection stack leaves behind: one tail per interrupted
        song, each with its own start and its own absolute position, plus the
        interruption that has not played. Clearing it must produce a row per
        HEARD song and nothing for the one that was only queued.

        Player-level rather than cog-level: -clear's reply is built from titles
        and the cog's tests mock queue_clear outright, so the per-tail rule is
        only observable here."""
        tails = [
            QueueObject(
                f"https://yt.com/v={n}",
                f"Song {n}",
                mock_author,
                ts=30 * n,
                duration=240,
                is_resume=True,
                played_at=1752530000.0 + n,
            )
            for n in (1, 2, 3)
        ]
        for tail in tails:
            await music_player.queue_put(tail)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=next", "Never Heard", mock_author)
        )

        await music_player.queue_clear()

        assert [
            (e.webpage_url, e.played_at, e.played_secs) for e in music_player.history
        ] == [
            ("https://yt.com/v=1", 1752530001.0, 30),
            ("https://yt.com/v=2", 1752530002.0, 60),
            ("https://yt.com/v=3", 1752530003.0, 90),
        ]

    async def test_card_ids_survive_the_full_restore_round_trip(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The three legs each had a test — the wire round trip and the by-id
        delete — but nothing connected them, so the whole restart branch of
        _dispose_previous_np_card could go dead without a failure. Reachable in
        production: the loop stamps a tail in memory, -shuffle or a matching
        -remove re-serializes the stamped object, and the restart path is then the
        only thing holding those ids."""
        tail = QueueObject(
            "https://yt.com/v=tail",
            "Tail",
            mock_author,
            ts=95,
            is_resume=True,
            played_at=1752530001.0,
        )
        tail.np_message_id = 777777777777777777
        tail.np_channel_id = 888888888888888888
        tail.np_dedicated = True

        entry = SongQueueEntry.from_queue_object(tail)
        wire = entry.to_redis()
        parsed = parse_queue_entry(wire)
        assert parsed is not None
        rebuilt = music_player.queue._rehydrate(parsed)

        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.np_message_id == 777777777777777777
        assert rebuilt.np_channel_id == 888888888888888888
        assert rebuilt.np_dedicated is True
        # And a rebuilt tail with a live ref gone takes the by-id branch.
        assert rebuilt.np_host_ref is None

    async def test_clear_disposes_the_cards_of_the_tails_it_destroys(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """A tail disposes of its fragment's frozen card when it STARTS, so a tail
        destroyed first takes the only pointer with it and the dead bar stays in
        the channel forever — the exact accumulation the feature prevents,
        reached through the most-used escape hatch."""
        tails = [
            QueueObject(
                f"https://yt.com/v={n}",
                f"Song {n}",
                mock_author,
                ts=30 * n,
                is_resume=True,
                played_at=1752530000.0 + n,
            )
            for n in (1, 2, 3)
        ]
        for tail in tails:
            await music_player.queue_put(tail)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=plain", "Plain", mock_author)
        )
        disposed: list[str] = []

        async def track(_self: Any, song: Any) -> None:
            disposed.append(song.webpage_url)

        with patch.object(MusicPlayer, "_dispose_previous_np_card", new=track):
            await music_player.queue_clear()
            await asyncio.sleep(0)  # let the fire-and-forget tasks run

        assert disposed == [
            "https://yt.com/v=1",
            "https://yt.com/v=2",
            "https://yt.com/v=3",
        ]

    async def test_one_malformed_entry_does_not_drop_the_whole_batch(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """parse_queue_entry coerces nothing and HistoryEntry raises rather than
        coercing, so one bad wire value used to abort the flush out of
        queue_clear() — AFTER clear() destroyed the mirror. The user got "Failed"
        on a queue that was cleared, and every play in the batch was lost."""
        good = QueueObject(
            "https://yt.com/v=good", "Good", mock_author, ts=30, played_at=1.0
        )
        bad = QueueObject(
            "https://yt.com/v=bad", "Bad", mock_author, ts=30, played_at=2.0
        )
        bad.np_message_id = {"nested": "object"}  # pyright: ignore[reportAttributeAccessIssue]

        await music_player._flush_played([good, bad])

        assert [e.webpage_url for e in music_player.history] == [
            "https://yt.com/v=good"
        ]

    async def test_a_non_numeric_played_at_is_skipped_not_raised(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # The `> 0.0` comparison is itself a raise site on a null or string.
        item = QueueObject("https://yt.com/v=x", "X", mock_author, ts=30)
        item.played_at = None  # pyright: ignore[reportAttributeAccessIssue]

        await music_player._flush_played([item])

        assert list(music_player.history) == []

    async def test_a_failed_tail_dequeue_still_records_the_play(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The third queue exit, alongside -clear and -remove. A tail that fails
        to resolve or stream has left the queue as permanently as one -clear
        destroyed it, and the fragment that parked it already declined to record
        itself — so this is the only writer left for 95s the listener heard."""
        tail = QueueObject(
            "https://yt.com/v=heard",
            "Heard",
            mock_author,
            ts=95,
            duration=240,
            is_resume=True,
            played_at=1752530001.0,
        )
        await music_player.queue_put(tail)
        assert music_player.queue.get_nowait() is tail

        await music_player._retire_failed_dequeue(tail, context="failed-song pop")

        assert [(e.webpage_url, e.played_secs) for e in music_player.history] == [
            ("https://yt.com/v=heard", 95)
        ]

    async def test_a_failed_fresh_dequeue_records_nothing(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The other half of the same gate: a song that never played is not a
        play. played_at == 0.0 is the whole distinction."""
        song = QueueObject("https://yt.com/v=new", "New", mock_author)
        await music_player.queue_put(song)
        assert music_player.queue.get_nowait() is song

        await music_player._retire_failed_dequeue(song, context="failed-song pop")

        assert list(music_player.history) == []

    async def test_in_flight_head_is_flushed_exactly_once(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """A played tail dequeued but not yet committed is still on the display
        leg, so clear() returns it. The loop's own discard path records nothing
        (try_commit_dequeue fails and the song is thrown away), which is what
        makes this the ONE record rather than a duplicate of one."""
        tail = QueueObject(
            "https://yt.com/v=heard", "Heard", mock_author, ts=95, played_at=1.0
        )
        await music_player.queue_put(tail)
        assert music_player.queue.get_nowait() is tail  # dequeued, uncommitted
        generation = music_player.queue.generation  # as the loop captures it

        await music_player.queue_clear()

        assert len(music_player.history) == 1
        # The loop would now find the display empty and discard its song silently.
        assert await music_player.queue.try_commit_dequeue(generation) is False

    async def test_a_put_after_clear_cannot_revive_a_flushed_dequeue(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The emptiness check alone was not enough. -clear flushes the in-flight
        head to history, but a -play landing in the loop's resolve window refills
        the display — and an emptiness-only commit then pops that entry, plays the
        cleared song, and records it a SECOND time at its iteration end. The
        generation is what a refill cannot forge."""
        tail = QueueObject(
            "https://yt.com/v=heard", "Heard", mock_author, ts=95, played_at=1.0
        )
        await music_player.queue_put(tail)
        assert music_player.queue.get_nowait() is tail
        generation = music_player.queue.generation

        await music_player.queue_clear()
        assert len(music_player.history) == 1  # flushed exactly once

        # -play lands while the loop is still inside yt_stream.
        await music_player.queue_put(
            QueueObject("https://yt.com/v=new", "New", mock_author)
        )

        assert await music_player.queue.try_commit_dequeue(generation) is False
        # And the new song's display entry survives for its own iteration.
        [remaining] = music_player.queue.display_items()
        assert isinstance(remaining, QueueObject)
        assert remaining.webpage_url == "https://yt.com/v=new"

    async def test_only_the_generation_refuses_when_the_refill_is_claimed(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The test above does not reach the generation check: clear() resets the
        cursor, so try_release() refuses on its own. A SECOND consumer reaches the
        state only the generation can refuse — the prefetch claims the refill, so
        there IS a claim to settle, and without the check the stale commit would
        settle the new song's."""
        first = QueueObject("https://yt.com/v=first", "First", mock_author)
        await music_player.queue_put(first)
        assert music_player.queue.get_nowait() is first  # the loop claims
        generation = music_player.queue.generation

        await music_player.queue_clear()
        refill = QueueObject("https://yt.com/v=refill", "Refill", mock_author)
        await music_player.queue_put(refill)
        assert music_player.queue.get_nowait() is refill  # the prefetch claims
        assert music_player.queue._cursor == 1  # so try_release() alone would say True

        assert await music_player.queue.try_commit_dequeue(generation) is False

        # The refill's claim is untouched — the stale commit settled nothing.
        assert music_player.queue._cursor == 1
        assert music_player.queue.display_items() == [refill]


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
        seed_queue(music_player.queue, crashed)
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

        positions = (await music_player.queue_remove("https://yt.com/v=abc")).positions

        assert positions == [1]
        assert music_player.queue.qsize() == 0
        assert len(music_player.queue._items) == 0

    async def test_remove_by_the_search_text_that_queued_it(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The resolved yt-dlp URL is not what a user who typed a search has in
        front of them — before this, the only way to remove that song was to read
        the link off the Now Playing card."""
        qobj = QueueObject(
            "https://yt.com/v=abc", "Song", mock_author, user_input="my search query"
        )
        await music_player.queue_put(qobj)

        outcome = await music_player.queue_remove("my search query")

        assert outcome.positions == [1]
        assert outcome.mode is RemoveMode.ORIGIN
        assert music_player.queue.qsize() == 0

    async def test_one_collection_link_removes_every_track_it_queued(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The point of matching on origin: an album expands to N songs sharing
        one link, and removing them one resolved URL at a time is the workflow
        this replaces. The song queued separately stays."""
        album = "https://open.spotify.com/album/abc123"
        await music_player.queue_put(
            [
                QueueObject(
                    f"https://yt.com/v={n}",
                    f"Track {n}",
                    mock_author,
                    user_input=album,
                    query_source="spotify.com",
                )
                for n in range(1, 4)
            ]
            + [
                QueueObject(
                    "https://yt.com/v=other",
                    "Other",
                    mock_author,
                    user_input="unrelated search",
                )
            ]
        )

        outcome = await music_player.queue_remove(album)

        assert outcome.positions == [1, 2, 3]
        assert outcome.mode is RemoveMode.ORIGIN
        survivors = music_player.queue.display_items()
        assert [getattr(i, "title", None) for i in survivors] == ["Other"]

    async def test_no_match_returns_empty_list(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=abc", "Song", mock_author)
        await music_player.queue_put(qobj)

        positions = (await music_player.queue_remove("https://yt.com/v=xyz")).positions

        assert positions == []
        assert music_player.queue.qsize() == 1
        assert len(music_player.queue._items) == 1

    async def test_remove_empty_queue_returns_empty(
        self, music_player: MusicPlayer
    ) -> None:
        positions = (await music_player.queue_remove("https://yt.com/v=x")).positions
        assert positions == []

    async def test_remove_returns_correct_1indexed_positions(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(5):
            qobj = QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        positions = (await music_player.queue_remove("https://yt.com/v=2")).positions
        assert positions == [3]

    async def test_removed_played_tail_is_recorded(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # Same exit, same rule as -clear: -remove destroys the tail, so this is
        # the play's only chance at a row. The positions the command reports are
        # unchanged by the flush.
        tail = QueueObject(
            "https://yt.com/v=heard",
            "Heard",
            mock_author,
            ts=95,
            duration=240,
            is_resume=True,
            played_at=1752530000.0,
        )
        await music_player.queue_put(tail)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=other", "Other", mock_author)
        )

        positions = (
            await music_player.queue_remove("https://yt.com/v=heard")
        ).positions

        assert positions == [1]
        assert len(music_player.history) == 1
        assert music_player.history[0].played_secs == 95

    async def test_removing_an_unplayed_song_records_nothing(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        await music_player.queue_put(
            QueueObject("https://yt.com/v=abc", "Song", mock_author)
        )

        assert (await music_player.queue_remove("https://yt.com/v=abc")).positions == [
            1
        ]
        assert len(music_player.history) == 0

    async def test_remove_multiple_matches_returns_all_positions(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        urls = ["https://yt.com/v=a", "https://yt.com/v=b", "https://yt.com/v=a"]
        for url in urls:
            await music_player.queue_put(QueueObject(url, f"Song {url}", mock_author))

        positions = (await music_player.queue_remove("https://yt.com/v=a")).positions
        assert positions == [1, 3]

    async def test_remove_keeps_non_matching_songs(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            await music_player.queue_put(
                QueueObject(f"https://yt.com/v={i}", f"Song {i}", mock_author)
            )

        await music_player.queue_remove("https://yt.com/v=1")

        remaining = list(music_player.queue._items)
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
        seed_queue(music_player.queue, crashed)
        await music_player.queue_put(
            QueueObject("https://yt.com/v=a", "Song A", mock_author)
        )
        await music_player.queue_put(
            QueueObject("https://yt.com/v=b", "Song B", mock_author)
        )

        positions = (await music_player.queue_remove("https://yt.com/v=a")).positions

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
        seed_queue(music_player.queue, crashed)
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
        seed_queue(music_player.queue, queue_obj)
        result = music_player.queue_embed()
        assert isinstance(result, discord.Embed)

    def test_embed_title_is_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        embed = music_player.queue_embed()
        assert embed.title == "Queue"

    def test_embed_color_is_blue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
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
            seed_queue(
                music_player.queue,
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=120
                ),
            )
        embed = music_player.queue_embed()
        assert "Songs: **3**" in described(embed)

    def test_total_duration_in_header_when_all_known(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90),
        )
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=90),
        )
        embed = music_player.queue_embed()
        assert "Total Duration: **3m**" in described(embed)
        assert "~" not in described(embed).split("Total Duration:")[1].split("\n")[0]

    def test_total_duration_partial_when_some_unknown(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90),
        )
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=None),
        )
        embed = music_player.queue_embed()
        assert "~" in described(embed)

    def test_total_duration_partial_with_ytsource(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90),
        )
        seed_queue(
            music_player.queue, YTSource(ytsearch="ytsearch:unresolved", process=True)
        )
        embed = music_player.queue_embed()
        assert "~" in described(embed)

    def test_song_title_appears_in_description(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        embed = music_player.queue_embed()
        assert "Test Song" in described(embed)

    def test_song_duration_appears_when_known(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        embed = music_player.queue_embed()
        assert "`3:30`" in described(embed)

    def test_song_duration_unknown_shows_placeholder(
        self, music_player: MusicPlayer, queue_obj_no_meta: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj_no_meta)
        embed = music_player.queue_embed()
        assert "`?:??`" in described(embed)

    def test_uploader_shown_when_known(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        embed = music_player.queue_embed()
        assert "Test Channel" in described(embed)

    def test_unknown_channel_shown_when_uploader_none(
        self, music_player: MusicPlayer, queue_obj_no_meta: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj_no_meta)
        embed = music_player.queue_embed()
        assert "Unknown channel" in described(embed)

    def test_est_playing_at_present_for_each_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            seed_queue(
                music_player.queue,
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                ),
            )
        embed = music_player.queue_embed()
        assert described(embed).count("Est. playing at") == 3

    def test_uncertain_prefix_after_no_duration_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None),
        )
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=2", "Song 2", mock_author, duration=60),
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
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60),
        )
        embed = music_player.queue_embed()
        assert "~**" in described(embed)

    def test_caps_display_at_ten_songs(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(15):
            seed_queue(
                music_player.queue,
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                ),
            )
        embed = music_player.queue_embed()
        assert described(embed).count("Est. playing at") == 10

    def test_shows_more_indicator_when_over_ten(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(15):
            seed_queue(
                music_player.queue,
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=60
                ),
            )
        embed = music_player.queue_embed()
        assert "... and 5 more" in described(embed)

    def test_ytsource_shows_resolving(self, music_player: MusicPlayer) -> None:
        seed_queue(
            music_player.queue, YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player.queue_embed()
        assert "resolving..." in described(embed)


# ── Resume notice embed ───────────────────────────────────────────────────────


def _fields(embed: discord.Embed) -> dict[str, str]:
    """An embed's fields as a name → value mapping, both asserted non-empty. Same
    reasoning as described(): name and value are `Optional[str]`, so failing that
    half separately says which of the two broke."""
    fields: dict[str, str] = {}
    for f in embed.fields:
        assert f.name is not None and f.value is not None
        fields[f.name] = f.value
    return fields


@pytest.fixture
def started(mock_author: MagicMock) -> QueueObject:
    """The song `-play` is starting — the one front-inserted ahead of the
    restored queue. Deliberately distinct from `queue_obj` (which stands in for
    restored entries) so a test can tell which song the embed is talking
    about."""
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=started",
        title="The Requested Song",
        requester=mock_author,
        duration=180,
        thumbnail="https://img/started.jpg",
    )


def _crashed(
    mock_author: MagicMock,
    *,
    title: str = "Interrupted Song",
    ts: int | None = 45,
    duration: int | None = 200,
) -> QueueObject:
    """The crash-recovered "current song" as _restore_state re-queues it:
    persisted=False (its LPOP already committed) with the recovery offset in
    `ts`. See GuildQueue.restore_crashed / SongQueueEntry.from_crashed_state."""
    return QueueObject(
        webpage_url="https://yt.com/v=crashed",
        title=title,
        requester=mock_author,
        ts=ts,
        duration=duration,
        persisted=False,
    )


class TestResumeNoticeEmbed:
    """build_resume_notice_embed() — the -play-on-a-disconnected-bot heads-up."""

    def test_returns_none_when_queue_empty(
        self, music_player: MusicPlayer, started: QueueObject
    ) -> None:
        """The gate: nothing was restored, so there is nothing to announce."""
        assert music_player.build_resume_notice_embed(started) is None

    def test_returns_none_when_queue_empty_even_with_history(
        self, music_player: MusicPlayer, started: QueueObject
    ) -> None:
        """History alone is not a resumption — the queue is what gates the send."""
        music_player.history.restore([HistoryEntry(title="Old Song", played_at=1.0)])
        assert music_player.build_resume_notice_embed(started) is None

    def test_highlight_color_is_orange(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        """Orange, not the blue every other -play embed uses: this is an
        attention notice about restored state. Pinned so a refactor can't
        quietly re-blue it."""
        seed_queue(music_player.queue, queue_obj)

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert embed.colour == discord.Color.orange()

    def test_names_the_song_being_started(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        """This embed is the only thing naming the requested song: the playback
        gate is shut across the enqueue, so the response hosts no Now Playing
        block and the real one is seconds away. Dropping the title left the
        user staring at an embed about some other song."""
        seed_queue(music_player.queue, queue_obj)

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert described(embed).startswith(
            "Playing now: The Requested Song "
            "- (https://www.youtube.com/watch?v=started)"
        )

    def test_thumbnail_is_the_started_song_not_the_last_played(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        """The thumbnail sits next to "Playing now" and has to match it."""
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore(
            [HistoryEntry(title="Old Song", thumbnail="https://img/old.jpg")]
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert embed.thumbnail.url == "https://img/started.jpg"

    def test_reports_count_and_runtime(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        for i in range(3):
            seed_queue(
                music_player.queue,
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=90
                ),
            )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert "**3** songs" in described(embed)
        fields = _fields(embed)
        assert fields["Queued"] == "**3** songs"
        assert fields["Runtime"] == "4m 30s"

    def test_runtime_marked_approximate_when_a_duration_is_unknown(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90),
        )
        seed_queue(
            music_player.queue, YTSource(ytsearch="ytsearch:unresolved", process=True)
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert _fields(embed)["Runtime"] == "~1m 30s"

    def test_runtime_field_omitted_when_no_duration_is_known(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None),
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert "Runtime" not in _fields(embed)

    def test_singular_wording_for_one_song(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=90),
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert described(embed).endswith(
            "**1** song from the previous session resumes after it."
        )

    def test_crash_recovered_head_is_named_as_where_playback_stopped(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        """A crash re-queues the mid-play song at the queue head with its
        recovery offset — that song is genuinely where the session left off,
        and it is about to play again."""
        seed_queue(music_player.queue, _crashed(mock_author))

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert _fields(embed)["Left off on"] == (
            "**Interrupted Song**\n`0:45` / `3:20`"
        )

    def test_crash_recovered_head_beats_history(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        """The regression this guards: history's newest entry is the last song
        to run to its *end*, which is older than the crash. Naming it would
        point at a song that already finished while the song it interrupted
        sits at the head of the queue about to resume."""
        seed_queue(music_player.queue, _crashed(mock_author))
        music_player.history.restore(
            [HistoryEntry(title="Finished Earlier", played_at=1721530000.0)]
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        fields = _fields(embed)
        assert "Last played" not in fields
        assert "Finished Earlier" not in fields["Left off on"]

    def test_crash_recovered_head_without_a_position(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        mock_author: MagicMock,
    ) -> None:
        """crashed_position_at() returns None when no play_start_epoch was
        recorded; the song is still the right one to name, just without an
        offset to show."""
        seed_queue(music_player.queue, _crashed(mock_author, ts=None))

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert _fields(embed)["Left off on"] == "**Interrupted Song**"

    def test_names_last_played_song_after_a_stop(
        self,
        music_player: MusicPlayer,
        started: QueueObject,
        queue_obj: QueueObject,
    ) -> None:
        """No crashed head (a -stop leaves none), so history is all there is. The
        field says "Last played", not "Stopped at": -stop cancels the loop before its
        history bookkeeping, so the interrupted song is recorded nowhere and this
        entry is an older song that ran to its end."""
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore(
            [
                HistoryEntry(
                    title="Father - Look At Wrist",
                    duration_secs=233,
                    played_secs=151,
                    thumbnail="https://img/thumb.jpg",
                    played_at=1721530000.0,
                )
            ]
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        fields = _fields(embed)
        assert "Stopped at" not in fields
        assert fields["Last played"] == (
            "**Father - Look At Wrist**\n`2:31` / `3:53`\n<t:1721530000:R>"
        )

    def test_uses_newest_history_entry(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        """restore() takes newest-first and reverses; latest must be the newest."""
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore(
            [
                HistoryEntry(title="Newest", played_at=2.0),
                HistoryEntry(title="Older", played_at=1.0),
            ]
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        last_played = _fields(embed)["Last played"]
        assert "Newest" in last_played
        assert "Older" not in last_played

    def test_omits_duration_when_unknown(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore(
            [HistoryEntry(title="Livestream", played_secs=151, duration_secs=0)]
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert _fields(embed)["Last played"] == "**Livestream**\n`2:31`"

    def test_omits_timestamp_when_played_at_unknown(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        """played_at == 0 is 'absent on the wire'; <t:0:R> would say 1970."""
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore(
            [HistoryEntry(title="Song", played_secs=10, duration_secs=60)]
        )

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert "<t:" not in _fields(embed)["Last played"]

    def test_no_left_off_field_without_history_or_crashed_head(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert embed.title == "❗ Resumed from queue"
        fields = _fields(embed)
        assert "Last played" not in fields
        assert "Left off on" not in fields

    def test_history_entry_without_a_title_is_skipped(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        """An empty title would render a bolded nothing."""
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore([HistoryEntry(title="", played_secs=10)])

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert "Last played" not in _fields(embed)

    def test_long_last_played_title_is_truncated(
        self, music_player: MusicPlayer, started: QueueObject, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        music_player.history.restore([HistoryEntry(title="x" * 400)])

        embed = music_player.build_resume_notice_embed(started)

        assert embed is not None
        assert len(_fields(embed)["Last played"]) <= 1024


class TestRejoinResumeEmbed:
    """build_rejoin_resume_embed() — the -resume-on-a-disconnected-bot heads-up."""

    def test_returns_none_when_queue_empty(self, music_player: MusicPlayer) -> None:
        """The gate: nothing came back, so -resume reports that instead of
        joining a channel to sit silent in."""
        assert music_player.build_rejoin_resume_embed() is None

    def test_returns_none_when_queue_empty_even_with_history(
        self, music_player: MusicPlayer
    ) -> None:
        """History is a record of finished songs, not something to resume."""
        music_player.history.restore([HistoryEntry(title="Old Song", played_at=1.0)])
        assert music_player.build_rejoin_resume_embed() is None

    def test_names_no_song(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """Unlike the -play notice, nothing was inserted: the head this describes
        is the song the Now Playing card names seconds later, so naming it here
        would just be the same title twice."""
        seed_queue(music_player.queue, queue_obj)

        embed = music_player.build_rejoin_resume_embed()

        assert embed is not None
        assert "Playing now" not in described(embed)
        assert embed.thumbnail.url is None

    def test_reports_count_and_runtime(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        for i in range(3):
            seed_queue(
                music_player.queue,
                QueueObject(
                    f"https://yt.com/v={i}", f"Song {i}", mock_author, duration=90
                ),
            )

        embed = music_player.build_rejoin_resume_embed()

        assert embed is not None
        assert "**3** songs" in described(embed)
        fields = _fields(embed)
        assert fields["Queued"] == "**3** songs"
        assert fields["Runtime"] == "4m 30s"

    def test_singular_wording_for_one_song(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)

        embed = music_player.build_rejoin_resume_embed()

        assert embed is not None
        assert "**1** song from the previous session resumes now." in described(embed)

    def test_names_the_crash_recovered_head_it_left_off_on(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """Shares _resume_left_off_field with the -play notice, so a crash's
        mid-play song is named with the offset it will resume from."""
        seed_queue(music_player.queue, _crashed(mock_author))

        embed = music_player.build_rejoin_resume_embed()

        assert embed is not None
        assert _fields(embed)["Left off on"] == (
            "**Interrupted Song**\n`0:45` / `3:20`"
        )

    def test_highlight_color_is_green(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """Green, not the -play notice's orange: that one interrupts a request
        with news about unrelated restored state, while this IS the response to
        what was asked for."""
        seed_queue(music_player.queue, queue_obj)

        embed = music_player.build_rejoin_resume_embed()

        assert embed is not None
        assert embed.colour == discord.Color.green()


class TestClaimCurrentSongForHistory:
    """A teardown abandons the playing song, and nothing else can record it: it is
    not in the queue, clear_connection() drops the parked state copy, and the loop
    is cancelled while parked in play_next.wait()."""

    def _playing(self, music_player: MusicPlayer, mock_song: MagicMock) -> MagicMock:
        mock_song.produced_audio = True
        mock_song.elapsed_secs = 42.0
        mock_song.played_at = 1752530000.0
        music_player.current_song = mock_song
        return mock_song

    def test_records_the_song_being_abandoned(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        song = self._playing(music_player, mock_song)

        entry = music_player.claim_current_song_for_history()

        assert entry is not None
        assert entry.title == song.title
        assert entry.played_at == 1752530000.0  # the play's start, not now
        assert entry.played_secs == 42  # the position it actually reached

    def test_captures_the_np_host_before_it_is_retired(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """cleanup() disposes of the host right after this; the id still records
        which message carried the block, which is what the column is for."""
        self._playing(music_player, mock_song)
        host = AsyncMock(spec=discord.Message)
        host.id = 777777777777777777
        host.channel.id = 888888888888888888
        music_player._np_host_message = host

        entry = music_player.claim_current_song_for_history()

        assert entry is not None
        assert (entry.message_id, entry.channel_id) == (
            777777777777777777,
            888888888888888888,
        )

    def test_nothing_playing_records_nothing(self, music_player: MusicPlayer) -> None:
        music_player.current_song = None
        assert music_player.claim_current_song_for_history() is None

    def test_a_song_that_never_opened_is_not_recorded(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        # ffmpeg exited without a frame — nobody heard it.
        self._playing(music_player, mock_song)
        mock_song.produced_audio = False
        assert music_player.claim_current_song_for_history() is None

    def test_an_interjected_song_is_left_for_its_tail(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The one case that must NOT record. A teardown leaves the queue in Redis
        under its 24h TTL, so the parked tail survives and records the play when
        -resume reaches it. Recording here too would file one play as two rows."""
        song = self._playing(music_player, mock_song)
        music_player._skip_history_for = song  # interject() marked it

        assert music_player.claim_current_song_for_history() is None
        # And the marker is left intact for the tail's own iteration.
        assert music_player._skip_history_for is song

    def test_claiming_stops_the_loop_recording_it_again(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The claim is synchronous precisely so the loop cannot slip its own
        iteration end into a window this opened — it takes the same marker
        interject() uses, so a loop that does reach its end skips the song."""
        song = self._playing(music_player, mock_song)

        assert music_player.claim_current_song_for_history() is not None

        assert music_player._skip_history_for is song  # the loop will skip it


class TestReparkCrashedHead:
    """repark_crashed_head() — putting back the recovered song that only memory
    holds. _restore_state clears current_song_* as soon as it re-queues that song,
    so from then until it plays, dropping the player loses it outright."""

    async def test_writes_the_head_back_to_the_state_hash(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(music_player.queue, _crashed(mock_author))

        assert await music_player.repark_crashed_head() is True

        assert music_player.store is not None
        state = await music_player.store.get_guild_state()
        assert state is not None
        assert state.has_crashed_song
        assert state.current_song_title == "Interrupted Song"

    async def test_position_survives_the_round_trip(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The state hash carries no `ts` field, so the offset a re-parked head had
        already reached survives only as the seeded position — which is what
        recovery reads back out."""
        seed_queue(music_player.queue, _crashed(mock_author, ts=45))

        await music_player.repark_crashed_head()

        assert music_player.store is not None
        state = await music_player.store.get_guild_state()
        assert state is not None
        assert state.last_position_secs == pytest.approx(45)
        assert state.crashed_position_at(time.time()) == 45

    async def test_the_backdated_epoch_carries_the_position_for_a_rollback(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The same offset, down the legacy path an older build takes. That build
        cannot read last_position_secs, so dropping the backdate here would strand
        a re-parked head at 0:00 on any `just up <older-sha>`."""
        seed_queue(music_player.queue, _crashed(mock_author, ts=45))

        await music_player.repark_crashed_head()

        assert music_player.store is not None
        state = await music_player.store.get_guild_state()
        assert state is not None
        rolled_back = dataclasses.replace(state, last_position_secs=None)
        assert rolled_back.crashed_position_at(time.time()) == pytest.approx(45, abs=2)

    async def test_played_at_survives_the_round_trip(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """A double crash: the head recovered from one crash is re-parked by this
        path and must still carry the start of the play it belongs to. Free —
        _now_playing_state_mapping is the single signature both writers use — but
        only while the head carries the field, which is what this pins."""
        head = _crashed(mock_author)
        head.played_at = 1752530000.5
        seed_queue(music_player.queue, head)

        await music_player.repark_crashed_head()

        assert music_player.store is not None
        state = await music_player.store.get_guild_state()
        assert state is not None
        assert state.current_song_played_at == 1752530000.5
        recovered = SongQueueEntry.from_crashed_state(state, position=45)
        assert recovered is not None and recovered.played_at == 1752530000.5

    async def test_leaves_a_persisted_head_alone(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """An ordinary queued song is still on the Redis list. Parking it in
        current_song_* would restore a second copy alongside that list entry."""
        seed_queue(music_player.queue, queue_obj)

        assert await music_player.repark_crashed_head() is False

        assert music_player.store is not None
        state = await music_player.store.get_guild_state()
        assert state is not None
        assert not state.has_crashed_song

    async def test_no_head_writes_nothing(self, music_player: MusicPlayer) -> None:
        assert await music_player.repark_crashed_head() is False

    async def test_without_redis_writes_nothing(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        seed_queue(mp.queue, _crashed(mock_author))

        assert await mp.repark_crashed_head() is False


# ── EstimatedPlayingAt ────────────────────────────────────────────────────────


class TestEstimatedPlayingAt:
    def test_matches_clock_format(self, music_player: MusicPlayer) -> None:
        result = music_player.estimated_playing_at()
        assert re.match(r"^\*\*\d{1,2}:\d{2} (AM|PM) P[SD]T\*\*$", result)

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

        seed_queue(
            music_player.queue,
            QueueObject(
                "https://yt.com/v=1", "Song 1", mock_song.requester, duration=600
            ),
        )
        later_eta = music_player.estimated_playing_at()

        assert empty_eta != later_eta

    def test_uncertain_when_queued_song_duration_unknown(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=None),
        )
        result = music_player.estimated_playing_at()
        assert result.startswith("~")

    def test_matches_last_queue_line_eta(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """estimated_playing_at() should reflect the same seed used by
        queue_embed()/_build_next_up_embed() for consistency across embeds."""
        music_player.current_song = mock_song
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "Song 1", mock_author, duration=60),
        )
        eta = music_player.estimated_playing_at()

        # A song appended now would start right where the last queued line's
        # ETA ends up, so re-derive it via the same line formatter for index 2.
        now_pst, walk = music_player._queue_eta_seed()
        _, walk = music_player._format_queue_line(
            music_player.queue._items[0], 1, now_pst, walk
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
        """elapsed_secs is never negative in practice (the read()-counter
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
        """The requester/finish-time line stays on one line — the progress bar sits
        above it as its own line, not interleaved with it."""
        embed = music_player._build_now_playing_embed(mock_song)
        requester_line = described(embed).split("\n")[-1]
        assert re.search(
            r"Requester: \[.*\].*Estimated finish: \d{1,2}:\d{2} (AM|PM) P[SD]T$",
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
        # Scoped to the bar line, not the whole description: that also carries
        # "Estimated finish: <wall clock>", and fmt_duration(30) is "0:30" — a
        # substring of "10:30 PM PST". Unscoped, this fails for two minutes a day.
        bar_line = next(line for line in described(embed).splitlines() if "🔘" in line)
        assert fmt_duration(210) in bar_line
        assert fmt_duration(30) not in bar_line

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
    """Slim by design: the -pause response hosts the live NP block directly below
    this embed, so a bar, requester, links or thumbnail here would render twice. It
    carries only what the NP block doesn't — paused state and pause position."""

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
        # Name encodes uploader as suffix since bot activities only render name
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
        """cleanup() cancels the playback loop before disconnecting, so the loop's
        CancelledError handler calls update_activity(None) while this guild's own
        client is still connected and playing. The "another guild is playing" gate
        must not count our own client, or the presence stays stuck on the stopped
        song."""
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
        in the future instead of the correct remaining time."""
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
    """Presence timestamps must track pause state, not just be stamped once at
    song start."""

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
        (YTDL.read() counting freezes during a pause), so a normal
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
        assert len(music_player.queue._items) == 0

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
    async def test_redis_push_history_caps_the_list(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        # Bounded retention: the list is a fixed window of the newest plays, not a
        # full record. Postgres keeps everything; this is what -history reads, and
        # it is capped at exactly that command's ceiling.
        assert music_player.store is not None
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await music_player.store.push_history(
                HistoryEntry(title=f"Song {i}", webpage_url=f"url{i}")
            )
        items = await fake_redis.lrange(music_player.store.history_key(), 0, -1)
        assert len(items) == HISTORY_CACHE_LIMIT

    async def test_store_set_volume_updates_volume(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await music_player.store.set_volume(0.75)
        config = await fake_redis.hgetall(music_player.store.config_key())
        assert config[b"volume"] == b"0.75"

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
    interjection, and mid-song stream death."""

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
    tick before the interruption, rather than at the true stop point."""

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
    """Restoring the persisted queue and playing it are separate concerns: the
    gate holds the loop shut until a real voice connection exists."""

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
        """Crash recovery connects to voice before start() — that path must keep
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
        """cog_before_invoke builds a player for every command, including ones
        validate_commands is about to reject; without the gate that player walks the
        persisted queue and discards it entry by entry against no voice client."""
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
        """The ordering guarantee of the front-insert path: -play must not touch
        the queue before _restore_state() has read its snapshot,
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
        """A player blocked on the gate is not blocked in queue_get(), so the
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

    async def test_gate_timeout_waits_out_an_in_flight_hold(
        self, music_player: MusicPlayer
    ) -> None:
        """The timeout exists for a player nobody is coming back for. A hold means a
        command is mid-join and still holds this player — tearing down under it pops
        the mps entry it is driving and hands its join's voice client to nobody."""
        music_player._playback_gate.clear()
        music_player._cog.cleanup = AsyncMock()

        with patch("src.musicplayer._PLAYBACK_GATE_TIMEOUT", 0.01):
            async with music_player.defer_playback():
                loop_task = asyncio.create_task(music_player.loop())
                await asyncio.sleep(0.05)  # several timeouts' worth
                music_player._cog.cleanup.assert_not_awaited()
                assert not loop_task.done()
            # Hold released: the gate opens and the loop proceeds normally.
            await asyncio.sleep(0.05)

        music_player._cog.cleanup.assert_not_awaited()
        await cancel_task(loop_task)

    async def test_a_timeout_landing_after_the_gate_opened_does_not_tear_down(
        self, music_player: MusicPlayer
    ) -> None:
        """The hold check alone is not enough, and the gap is not theoretical.

        defer_playback's exit drops the count and opens the gate in one synchronous
        step, but this handler runs a tick AFTER the timer fired — async_timeout
        cancels the inner wait and the except body reaches us later — so the release
        can land in between. holds is then already 0 and the gate already open, and
        a handler reading holds alone tears down a player whose join just succeeded.

        Timing-free: the state the race produces is set up directly, and a wait that
        never returns forces the handler to run in it. Under CPU contention the
        sibling test above reproduces the same thing about once in 25 runs, which is
        what CI saw.
        """
        music_player._cog.cleanup = AsyncMock()
        music_player._playback_gate.set()
        assert music_player.playback_holds == 0

        never = asyncio.Event()  # never set, so every wait hits the timeout
        with (
            patch("src.musicplayer._PLAYBACK_GATE_TIMEOUT", 0.01),
            patch.object(music_player._playback_gate, "wait", never.wait),
        ):
            loop_task = asyncio.create_task(music_player.loop())
            await asyncio.sleep(0.05)  # several timeouts' worth
            music_player._cog.cleanup.assert_not_awaited()
            assert not loop_task.done()
        await cancel_task(loop_task)

    def test_can_rejoin_cold_on_a_parked_player(
        self, music_player: MusicPlayer
    ) -> None:
        music_player._playback_gate.clear()
        assert music_player.can_rejoin_cold() is True

    def test_cannot_rejoin_cold_once_the_gate_is_open(
        self, music_player: MusicPlayer
    ) -> None:
        """An open gate with no voice client is a loop already walking the queue,
        not a player waiting to be handed a connection."""
        music_player._playback_gate.set()
        assert music_player.can_rejoin_cold() is False

    def test_cannot_rejoin_cold_while_a_song_is_held(
        self, music_player: MusicPlayer
    ) -> None:
        music_player._playback_gate.clear()
        music_player.current_song = MagicMock()
        assert music_player.can_rejoin_cold() is False

    async def test_wait_for_restore_gives_up_at_its_timeout(
        self, music_player: MusicPlayer
    ) -> None:
        """The pool sets no socket_timeout, so a Redis that accepts the connection
        and then stalls never finishes the restore — an unbounded wait here is a
        command that never answers."""
        music_player._restore_complete.clear()

        assert await music_player.wait_for_restore(timeout=0.01) is False

    async def test_wait_for_restore_reports_a_completed_restore(
        self, music_player: MusicPlayer
    ) -> None:
        assert await music_player.wait_for_restore(timeout=1) is True


class TestQueuePutFront:
    """MusicPlayer.queue_put_front — the -play-on-a-disconnected-bot path
    . The list branch is the playlist case,
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


# ── Ask-time analytics ────────────────────────────────────────────────────────


class TestEnqueueDepth:
    """queue_position is depth at ASK: MusicBot reads mp.enqueue_depth() once at
    command dispatch and constructs every queue object complete — nothing stamps
    at insert anymore. The depth is everything queued plus the live song, so 0
    means the song will play immediately."""

    async def test_idle_player_is_depth_zero(self, music_player: MusicPlayer) -> None:
        assert music_player.enqueue_depth() == 0

    async def test_live_song_adds_one(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        assert music_player.enqueue_depth() == 1

    async def test_queued_entries_count(
        self, music_player: MusicPlayer, mock_author: MagicMock, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        await music_player.queue.put(
            [QueueObject("https://yt.com/v=1", "One", mock_author)]
        )
        assert music_player.enqueue_depth() == 2

    async def test_live_song_and_its_resume_tail_count_once(
        self, music_player: MusicPlayer, mock_author: MagicMock, mock_song: MagicMock
    ) -> None:
        # Post-interjection the interrupted song is BOTH current_song and, as its
        # resume tail, an entry on the display. It is one song, so a new arrival
        # waits behind two — counting it twice would report 3.
        mock_song.webpage_url = "https://yt.com/v=live"
        music_player.current_song = mock_song
        await music_player.queue.put_front(
            [
                QueueObject("https://yt.com/v=now", "Now", mock_author),
                QueueObject(
                    "https://yt.com/v=live", "Live", mock_author, is_resume=True
                ),
            ]
        )
        assert music_player.enqueue_depth() == 2

    async def test_in_flight_head_still_counts(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # THE reason the depth reads display_size(): a claimed-but-unsettled head
        # (taken here the way a prefetch takes it, via get_nowait) is past the
        # cursor but still ahead of a new arrival. qsize() reads 0 and would
        # under-report exactly when a -play lands during another song's resolve.
        await music_player.queue.put(
            [QueueObject("https://yt.com/v=1", "One", mock_author)]
        )
        music_player.queue.get_nowait()
        assert music_player.queue.qsize() == 0
        assert music_player.queue.display_size() == 1
        assert music_player.enqueue_depth() == 1

    async def test_the_over_by_one_window_during_a_resolve_is_unchanged(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The first of the two ±1 windows the analytics work documented and
        accepted. current_song is assigned before try_commit_dequeue() settles the
        claim, so for the length of a probe the same play is both current_song and
        still in the queue, and the depth counts it twice. Pinned because the
        single-deque migration had to reproduce it, not close it."""
        song = MagicMock()
        song.webpage_url = "https://yt.com/v=1"
        await music_player.queue.put(
            [QueueObject("https://yt.com/v=2", "Two", mock_author)]
        )
        await music_player.queue.get()  # the loop claimed it
        music_player.current_song = song  # ...and assigned it, not yet committed

        # 1 in the queue (claimed, uncommitted) + 1 for the live song = 2, where
        # only one play is actually ahead of a new arrival.
        assert music_player.enqueue_depth() == 2

    async def test_a_resume_tail_keeps_the_origin_it_was_queued_with(
        self, music_player: MusicPlayer, mock_author: MagicMock, mock_vc: MagicMock
    ) -> None:
        """H5. The tail is a rebuild, and YTDL had no user_input at all — so the
        origin died the moment a song started playing, and `-remove <album link>`
        left the parked track behind while taking every other track."""
        album = "https://open.spotify.com/album/abc123"
        current = MagicMock()
        current.webpage_url = "https://yt.com/v=parked"
        current.title = "Parked"
        current.requester = mock_author
        current.position_secs = 40.0
        current.duration_secs = 300
        current.user_input = album
        current.query_source = "spotify.com"
        current.analytics = ANALYTICS_ZERO
        current.played_at = 1.0
        current.interjected = False
        current.is_resume = False
        current.start_paused = False
        music_player.current_song = current

        await music_player.interject(
            QueueObject("https://yt.com/v=urgent", "Urgent", mock_author), mock_vc
        )

        tails = [
            i
            for i in music_player.queue.display_items()
            if isinstance(i, QueueObject) and i.is_resume
        ]
        assert [t.user_input for t in tails] == [album]

    async def test_the_under_by_one_window_for_a_repeated_url_is_unchanged(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        """The second window: has_resume_tail matches on URL, not identity, so a
        tail parked by an EARLIER play of the same song answers True for the
        current one and the live song stops being counted. Reached by
        -play X, -p --now Y, -p --now X."""
        tail = QueueObject(
            "https://yt.com/v=x", "X", mock_author, is_resume=True, ts=30
        )
        await music_player.queue.put([tail])
        current = MagicMock()
        current.webpage_url = "https://yt.com/v=x"  # same URL, different play
        music_player.current_song = current

        # The tail is counted; the live song is NOT, because they look like one
        # play. Without the collision this would be 2.
        assert music_player.enqueue_depth() == 1

    async def test_resolved_search_passes_its_analytics_through(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # A Spotify playlist track sat in the queue as a search; _resolve_source
        # threads its ask-time analytics into yt_source, which is REQUIRED to
        # take it — there is no post-copy left to forget.
        source = YTSource(
            ytsearch="ytsearch:a song",
            analytics=Analytics(queued_at=1752530000.5, queue_position=4),
        )
        resolved = QueueObject("https://yt.com/v=1", "One", mock_author)
        spy = AsyncMock(return_value=resolved)
        with patch.object(YTDL, "yt_source", new=spy):
            out = await music_player._resolve_source(source)
        assert out is resolved
        assert spy.await_args is not None
        assert spy.await_args.kwargs["analytics"] == source.analytics

    async def test_resolved_search_passes_its_query_source_through(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        # A Spotify playlist track resolves to a YouTube URL here, so this hop is
        # the only thing keeping the archive from recording it as YouTube.
        source = YTSource(ytsearch="ytsearch:a song", query_source="spotify.com")
        resolved = QueueObject("https://youtube.com/watch?v=1", "One", mock_author)
        spy = AsyncMock(return_value=resolved)
        with patch.object(YTDL, "yt_source", new=spy):
            await music_player._resolve_source(source)
        assert spy.await_args is not None
        assert spy.await_args.kwargs["query_source"] == "spotify.com"


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
        assert isinstance(music_player.queue._items[0], QueueObject)
        assert music_player.queue._items[0].title == "Restored Song"

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

    async def test_crash_mid_stack_restores_every_parked_level(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """A crash three deep into an interjection stack.

        Every level is an ordinary persisted SongQueueEntry on the Redis list, so
        recovery needs no stack-specific code — which is exactly the claim worth
        pinning. The crashed song goes in front of the tails it interrupted, and
        each tail comes back with its own absolute position and its own start."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            mapping={
                b"current_song_url": b"https://yt.com/v=cut3",
                b"current_song_title": b"Cut 3",
                b"current_song_interjected": b"1",
                b"current_song_played_at": b"1752530300.0",
            },
        )
        for n, ts in ((2, 90), (1, 60), (0, 30)):
            await fake_redis.rpush(
                music_player.store.queue_key(),
                SongQueueEntry(
                    webpage_url=f"https://yt.com/v=cut{n}",
                    title=f"Cut {n}",
                    requester_id=mock_author.id,
                    ts=ts,
                    is_resume=True,
                    played_at=1752530000.0 + n,
                ).to_redis(),
            )
        music_player._guild.get_member = MagicMock(return_value=mock_author)

        await music_player._restore_state()

        restored = [queue_object(i) for i in music_player.queue.display_items()]
        assert [s.webpage_url for s in restored] == [
            "https://yt.com/v=cut3",  # the crashed song, re-queued at the front
            "https://yt.com/v=cut2",
            "https://yt.com/v=cut1",
            "https://yt.com/v=cut0",
        ]
        assert [s.ts for s in restored[1:]] == [90, 60, 30]
        assert [s.played_at for s in restored[1:]] == [
            1752530002.0,
            1752530001.0,
            1752530000.0,
        ]
        assert restored[0].played_at == 1752530300.0  # from the parked state hash
        assert restored[0].persisted is False  # its LPOP committed before the crash
        assert music_player.queue.resume_tail_depth() == 3

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

    async def test_crashed_song_position_capped_by_the_parked_duration(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """Capped at duration − 10s so FFmpeg never seeks past EOF, read from the
        state hash rather than the stream cache: that key expires in 30 minutes, so
        sourcing it there capped or did not depending on how long the restart took.
        """
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            mapping={
                b"current_song_url": b"https://yt.com/v=crash",
                b"current_song_title": b"Crashed",
                b"current_song_duration": b"60",
                b"last_position_secs": b"90",
                b"last_heartbeat_epoch": b"2000",
                b"play_start_epoch": b"1000",
            },
        )
        # Deliberately absent: the cache this used to read is gone half an hour in,
        # and the cap must not depend on it.
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts == 50  # min(90, 60 − 10)

    async def test_a_livestream_duration_of_zero_does_not_cap_to_zero(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """max(0, 0 - 10) is 0, so treating an unknown duration as a real one would
        restart every livestream from the beginning."""
        assert music_player.store is not None
        await fake_redis.hset(
            music_player.store.state_key(),
            mapping={
                b"current_song_url": b"https://yt.com/v=live",
                b"current_song_title": b"Live",
                b"current_song_duration": b"0",
                b"last_position_secs": b"140",
                b"last_heartbeat_epoch": b"2000",
                b"play_start_epoch": b"1000",
            },
        )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        music_player.bot.wait_until_ready = AsyncMock()

        await music_player._restore_state()

        first = await music_player.queue.get()
        assert first.ts == 140

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
# The crashed song _restore_state() injects was never on the Redis queue list, so a
# loop() that dequeued it before restore finished would LPOP an unrelated, still-
# queued song. loop() waits on _restore_complete, set only once restore is done.


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
        """End-to-end guard: a crashed song plus 2 still-queued songs in Redis. The
        crash-recovered song takes the stream-failure "skip" path, which also calls
        pop_queue(), yet both real queued songs must survive — pop_queue() must not
        fire for the crashed song's own dequeue."""
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
        # _restore_state() is scheduled before loop(), which waits on
        # _restore_complete before its first dequeue. Precondition: the fixture wires
        # a store, which is what makes start() take the restore branch at all.
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
        """_send_now_playing() may spawn a real _progress_task. Tests
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
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
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
        """The NP embed stored as play_message is the one sent in the block —
        not an identical rebuild."""
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


# ── Now-playing host primitives ─────────────────────


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
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
        )
        block = music_player.np_embed_block()
        assert len(block) == 2
        assert block[0].colour == discord.Color.green()
        assert block[1].title == "Up next"


@contextlib.contextmanager
def _current_span() -> Generator[None]:
    """Make a span with a valid, sampled context current. conftest installs no
    TracerProvider, so otherwise every trace assertion passes vacuously."""
    span = trace_api.NonRecordingSpan(
        trace_api.SpanContext(
            trace_id=0x4BF92F3577B34DA6A3CE929D0E0E4736,
            span_id=0x00F067AA0BA902B7,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED),
        )
    )
    with trace_api.use_span(span, end_on_exit=False):
        yield


class TestPlayerDebugDecoration:
    """The player's half of debug mode's footer — what MusicContext.send never sees.

    The cog is a MagicMock pinned off in mock_ctx, so enabling means setting both
    halves: an auto-mock runtime would render garbage.
    """

    @staticmethod
    def _enable(
        music_player: MusicPlayer,
        *,
        cpu: float = 12.0,
        mem: float = 34.0,
        lag: float = 1.0,
    ) -> None:
        cog = mocked(music_player._cog)
        cog.debug_settings.enabled.return_value = True
        cog.debug_settings.snapshot = RuntimeSnapshot(
            cpu_percent=cpu, mem_percent=mem, lag_ms=lag, tasks=7, pool_workers=4
        )

    @staticmethod
    def _footers(embeds: Sequence[discord.Embed]) -> list[str]:
        return [e.footer.text or "" for e in embeds]

    # ── The NP block: one chokepoint, every render site ───────────────────────

    def test_block_is_decorated_when_enabled(
        self, music_player: MusicPlayer, mock_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """Every embed of the block, not just the now-playing one."""
        self._enable(music_player)
        music_player.current_song = mock_song
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
        )
        block = music_player.np_embed_block()
        assert len(block) == 2
        assert all("🐞" in f for f in self._footers(block))

    def test_block_is_clean_when_disabled(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        music_player.current_song = mock_song
        block = music_player.np_embed_block()
        # The NP embed keeps its own stream-metadata footer either way — "clean"
        # means no debug suffix was added to it.
        assert not any("🐞" in f for f in self._footers(block))
        assert "Avg Bitrate" in self._footers(block)[0]

    def test_the_suffix_appends_after_the_np_footer(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The NP embed already carries a stream-metadata footer (bitrate/sampling/
        codec). Decoration must extend it, not overwrite it."""
        self._enable(music_player)
        music_player.current_song = mock_song
        footer = self._footers(music_player.np_embed_block())[0]
        assert "Avg Bitrate" in footer
        assert footer.index("Avg Bitrate") < footer.index("🐞")

    def test_the_block_carries_no_trace_id(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The block re-renders under the command span at attach and the playback
        span on the next tick, so a trace id there would alternate on one message."""
        self._enable(music_player)
        music_player.current_song = mock_song
        with _current_span():
            footer = self._footers(music_player.np_embed_block())[0]
        assert "trace" not in footer
        assert "cpu 12%" in footer and "shard" in footer

    async def test_the_dedicated_host_send_is_decorated(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        self._enable(music_player)
        music_player.current_song = mock_song
        sent = MagicMock(spec=discord.Message)
        sent.id = 1
        music_player._channel.send = AsyncMock(return_value=sent)
        await music_player._send_np_host_message()
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert all("🐞" in f for f in self._footers(embeds))

    # ── The periodic tick: the every-N-seconds half of the requirement ────────

    async def test_the_tick_refreshes_the_metrics(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The footer tracks the sampler rather than freezing at song start."""
        self._enable(music_player, cpu=12.0)
        message = AsyncMock(spec=discord.Message)
        await music_player._push_np_edit(mock_song, message, [])
        first = message.edit.call_args.kwargs["embeds"][0].footer.text or ""

        self._enable(music_player, cpu=91.0)
        await music_player._push_np_edit(mock_song, message, [])
        second = message.edit.call_args.kwargs["embeds"][0].footer.text or ""

        assert "cpu 12%" in first
        assert "cpu 91%" in second

    async def test_the_tick_leaves_cached_own_embeds_alone(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The host's own embeds are cached from send time and already decorated;
        their elapsed-ms records that request, so the tick must leave them alone."""
        self._enable(music_player)
        own = discord.Embed(title="Queue")
        own.set_footer(text="🐞 4 ms · shard 0")
        message = AsyncMock(spec=discord.Message)
        await music_player._push_np_edit(mock_song, message, [own])
        await music_player._push_np_edit(mock_song, message, [own])
        assert own.footer.text == "🐞 4 ms · shard 0"

    async def test_disabling_mid_song_clears_the_footer_next_tick(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The block is rebuilt each tick, so a toggle takes effect on the next one
        in both directions."""
        self._enable(music_player)
        message = AsyncMock(spec=discord.Message)
        await music_player._push_np_edit(mock_song, message, [])
        assert "🐞" in (message.edit.call_args.kwargs["embeds"][0].footer.text or "")

        mocked(music_player._cog).debug_settings.enabled.return_value = False
        await music_player._push_np_edit(mock_song, message, [])
        after = message.edit.call_args.kwargs["embeds"][0].footer.text or ""
        assert "🐞" not in after
        assert "Avg Bitrate" in after  # the embed's own footer survives

    async def test_enabling_mid_song_adds_the_footer_next_tick(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        await music_player._push_np_edit(mock_song, message, [])
        assert "🐞" not in (
            message.edit.call_args.kwargs["embeds"][0].footer.text or ""
        )

        self._enable(music_player)
        await music_player._push_np_edit(mock_song, message, [])
        assert "🐞" in (message.edit.call_args.kwargs["embeds"][0].footer.text or "")

    def test_disabling_strips_a_suffix_from_a_cached_embed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """play_message is built once per song and decorated in place, so it outlives
        a mid-song --disable and -now re-sends that same object. Freshly built embeds
        self-heal on the next tick; this one cannot."""
        self._enable(music_player)
        music_player.current_song = mock_song
        cached = music_player._build_now_playing_embed(mock_song)
        music_player.np_embed_block(now_playing=cached)
        assert "🐞" in (cached.footer.text or "")

        mocked(music_player._cog).debug_settings.enabled.return_value = False
        music_player.np_embed_block(now_playing=cached)
        assert "🐞" not in (cached.footer.text or "")
        assert "Avg Bitrate" in (cached.footer.text or "")  # its own footer survives

    async def test_the_finalize_edit_is_decorated(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        self._enable(music_player)
        message = AsyncMock(spec=discord.Message)
        await music_player._finalize_now_playing(mock_song, message, [])
        assert "🐞" in (message.edit.call_args.kwargs["embeds"][0].footer.text or "")

    # ── Player-initiated one-shot sends ───────────────────────────────────────

    async def test_send_with_np_decorates_its_own_embed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        self._enable(music_player)
        music_player.current_song = mock_song
        sent = MagicMock(spec=discord.Message)
        sent.id = 1
        music_player._channel.send = AsyncMock(return_value=sent)
        await music_player.send_with_np(embed=discord.Embed(title="Notice"))
        embeds = music_player._channel.send.call_args.kwargs["embeds"]
        assert all("🐞" in f for f in self._footers(embeds))

    async def test_the_resume_notice_is_decorated(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        self._enable(music_player)
        mock_song.start_paused = False
        music_player._channel.send = AsyncMock()
        await music_player._announce_resume(mock_song)
        embed = music_player._channel.send.call_args.kwargs["embed"]
        assert "🐞" in (embed.footer.text or "")

    async def test_the_dead_stream_notice_is_decorated(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        self._enable(music_player)
        music_player.store = None
        music_player._channel.send = AsyncMock()
        await music_player._handle_dead_stream(mock_song)
        embed = music_player._channel.send.call_args.kwargs["embed"]
        assert "🐞" in (embed.footer.text or "")

    async def test_the_playback_error_embed_is_decorated(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """The loop's outer handler. This block was rewritten away from send_embed()
        so the footer lands before the send, so a refactor back would undo it."""
        self._enable(music_player)
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)

        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=Exception("yt-dlp lookup failed")),
        ):
            await music_player.loop()

        embed = mocked(music_player._channel.send).call_args.kwargs["embed"]
        assert embed.title == "Playback error — skipping song"
        assert "🐞" in (embed.footer.text or "")

    async def test_the_playback_error_embed_shows_one_trace_id(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """It carries its own `trace: <id>` from trace_footer(span), so skip_trace
        must dedup against it rather than naming the same trace twice."""
        self._enable(music_player)
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)

        with (
            _current_span(),
            patch.object(
                MusicPlayer,
                "_resolve_source",
                new=AsyncMock(side_effect=Exception("boom")),
            ),
        ):
            await music_player.loop()

        footer = (
            mocked(music_player._channel.send).call_args.kwargs["embed"].footer.text
        )
        assert (footer or "").count("4bf92f3577b34da6a3ce929d0e0e4736") == 1

    async def test_a_one_shot_notice_keeps_its_trace_id(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The opposite of the block rule: nothing re-renders a notice, so its trace
        id stays valid."""
        self._enable(music_player)
        mock_song.start_paused = False
        music_player._channel.send = AsyncMock()
        with _current_span():
            await music_player._announce_resume(mock_song)
        embed = music_player._channel.send.call_args.kwargs["embed"]
        assert "trace" in (embed.footer.text or "")


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
        """Lock ordering on the STRIP: an in-flight tick edit (which holds the
        lock across its await) always completes before the strip, so the strip is
        the final write and a late tick cannot resurrect the NP block on the old
        host."""
        order: list[str] = []
        old = AsyncMock(spec=discord.Message)

        async def _edit(**_kw: Any) -> None:
            order.append("retire")

        old.edit.side_effect = _edit

        async def _hold_lock_like_a_tick() -> None:
            async with music_player._np_edit_lock:
                order.append("edit_started")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                order.append("edit_finished")

        holder = asyncio.create_task(_hold_lock_like_a_tick())
        await asyncio.sleep(0)  # holder acquires the lock
        retire = asyncio.create_task(music_player._retire_np_host(old, [], False))
        await asyncio.gather(holder, retire)
        assert order == ["edit_started", "edit_finished", "retire"]

    async def test_a_delete_does_not_queue_behind_the_edit_lock(
        self, music_player: MusicPlayer
    ) -> None:
        """And the asymmetry is deliberate. Nothing can resurrect a DELETED
        message — a late tick edit 404s and is swallowed — while message deletion
        is its own, stricter ratelimit bucket. Held across it, one 429 stalled
        every NP edit for the NEW song, so a burst of interjections serialized the live
        progress bar behind a queue of deletes."""
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

        assert order.index("retire") < order.index("edit_finished")


class TestAdoptNpHostIfCurrent:
    """The adopt gate closing the adopt-after-await race:
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
        """The send_with_np attach site: the song ends while the HTTP
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
        """The repin attach site: the song ends while the dedicated NP
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
    """-stop / alone-disconnect teardown: the host is
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
    """-resume re-hosting: a command-response host —
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
        the block, instead of 400ing on every tick."""
        music_player.current_song = mock_song
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
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
        the new host — that would permanently orphan its block."""
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
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
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
        """The finalize's completed-bar write must land after any in-flight
        debounce-spawned edit (which holds _np_edit_lock across its PATCH) —
        otherwise a resume just before song end can freeze the historical bar
        short of 100%."""
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
        against a stray tick landing after the song changed."""
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
        the new host — that would permanently orphan its block."""
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

    async def test_pause_records_the_exact_position(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """The ticking task skips paused songs, so without this write the recorded
        position sits up to one interval behind for the whole pause — and a crash
        during it replays that much."""
        assert music_player.store is not None
        mock_song.start_offset = 10
        mock_song.elapsed_secs = 32.5
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)

        await music_player.pause(vc)

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert float(state[b"last_position_secs"]) == 42.5

    async def test_pause_with_no_song_records_no_position(
        self,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
    ) -> None:
        """-pause is reachable with nothing playing. position_secs would raise on
        None, and a position written here would name whatever song ran last."""
        assert music_player.store is not None
        music_player.current_song = None
        vc = MagicMock(spec=discord.VoiceClient)

        await music_player.pause(vc)

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert b"last_position_secs" not in state
        assert b"pause_start_epoch" in state  # the legacy leg still runs

    async def test_pause_stamps_both_writes_with_one_instant(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """Two clock reads would leave the sliver between them as elapsed the
        legacy wall-clock math counts as playback and the heartbeat never saw."""
        assert music_player.store is not None
        music_player.current_song = mock_song
        vc = MagicMock(spec=discord.VoiceClient)

        await music_player.pause(vc)

        state = await fake_redis.hgetall(music_player.store.state_key())
        assert float(state[b"last_heartbeat_epoch"]) == float(
            state[b"pause_start_epoch"]
        )

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
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
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
        seed_queue(
            music_player.queue, YTSource(ytsearch="ytsearch:some song", process=True)
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "resolving..." in described(embed)

    def test_shows_placeholder_duration_when_unknown(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author),
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "`?:??`" in described(embed)

    def test_only_uses_first_queued_song(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=1", "First", mock_author, duration=60),
        )
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=2", "Second", mock_author, duration=60),
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "First" in described(embed)
        assert "Second" not in described(embed)

    def test_includes_est_playing_at_eta(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        seed_queue(
            music_player.queue,
            QueueObject("https://yt.com/v=next", "Next Song", mock_author, duration=90),
        )
        embed = music_player._build_next_up_embed()
        assert embed is not None
        assert "Est. playing at" in described(embed)
        assert re.search(r"\*\*\d{1,2}:\d{2} (AM|PM) P[SD]T\*\*", described(embed))

    def test_eta_matches_current_song_estimated_finish(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The next song's ETA should line up with the current song's finish time,
        since both derive from the same cumulative_secs seed."""
        music_player.current_song = mock_song
        seed_queue(
            music_player.queue,
            QueueObject(
                "https://yt.com/v=next", "Next Song", mock_song.requester, duration=90
            ),
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
        seed_queue(music_player.queue, queue_obj)
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
        # Original order restored, and the cancelled prefetch's claim went back
        # with its item — both are claimable again, in order.
        assert music_player.queue._cursor == 0
        assert music_player.queue.get_nowait() is queue_obj
        assert music_player.queue.get_nowait() is queue_obj_no_meta


# ── Loop task accounting ──────────────────────────────────────────────────────


class TestLoopClaimAccounting:
    async def test_exception_after_commit_does_not_settle_the_next_claim(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """A failure between the committed dequeue and the normal song-end
        settling its claim (here the voice client vanished during resolve) must still
        balance the get() in the loop's exception handler, or the task counter
        drifts upward on every such failure."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        seed_queue(music_player.queue, queue_obj)
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

        assert music_player.queue._cursor == 0  # every claim settled

    async def test_a_raise_before_the_commit_settles_the_claim_on_both_legs(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        """The window the handler's release exists for: _stream_source sits outside
        the inner try, so a raise there reaches the outer handler with the claim
        still live. Settling it must reach Redis too — release() alone drops the
        item from memory and leaves its entry for the next LPOP to retire instead
        of its own."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        assert music_player.store is not None
        await music_player.queue_put(queue_obj)  # real put, so Redis has the entry
        popped: list[int] = []
        original = music_player.store.pop_queue

        async def spy_pop() -> None:
            popped.append(1)
            await original()

        music_player.store.pop_queue = spy_pop

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer,
                "_stream_source",
                new=AsyncMock(side_effect=RuntimeError("boom before the commit")),
            ),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        assert music_player.queue._cursor == 0, "the claim was left standing"
        assert popped == [1], "the mirror kept an entry memory dropped"

    async def test_the_commit_uses_the_generation_captured_at_the_claim(
        self, music_player: MusicPlayer, mock_author: MagicMock, mock_song: MagicMock
    ) -> None:
        """Re-reading the generation at commit time instead of using the one
        captured at the claim cannot be caught by a clear() alone — clear() zeroes
        the cursor, so the release refuses either way. It needs a second consumer
        claiming the refill: then a stale commit would settle THAT claim."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        await music_player.queue_put(
            QueueObject("https://yt.com/v=first", "First", mock_author)
        )
        refill = QueueObject("https://yt.com/v=refill", "Refill", mock_author)
        # No voice client, so a commit that wrongly SUCCEEDS falls straight into
        # the outer handler instead of hanging on play_next: this fails by
        # assertion rather than by timeout.
        mocked(music_player._guild).voice_client = None

        async def clear_refill_and_claim(_self: MusicPlayer, source: Any) -> MagicMock:
            # A -clear and a -play land while this song resolves, and the prefetch
            # claims what the -play added.
            await music_player.queue_clear()
            await music_player.queue_put(refill)
            music_player.queue.get_nowait()
            return mock_song

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(side_effect=lambda s: s)
            ),
            patch.object(MusicPlayer, "_stream_source", new=clear_refill_and_claim),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        # The commit was refused, so the refill's claim is untouched and its item
        # is still queued for its own iteration.
        assert music_player.queue._cursor == 1
        assert music_player.queue.display_items() == [refill]

    async def test_a_raise_after_the_commit_does_not_eat_the_next_song(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """try_commit_dequeue settles the claim, so the flag guarding the handler's
        release is cleared there rather than at song end: left standing across the
        song, the release pops index 0, which by then is the NEXT song once the
        prefetch claims it. _send_now_playing is the raiser because it is awaited
        inline after the commit."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        following = QueueObject("https://yt.com/v=next", "Next", queue_obj.requester)
        seed_queue(music_player.queue, queue_obj, following)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc

        async def claim_then_raise(_self: MusicPlayer, _song: Any) -> None:
            # Stand in for the prefetch having claimed the next item by the time
            # something in the post-commit block fails.
            music_player.queue.get_nowait()
            raise RuntimeError("boom after the commit")

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=claim_then_raise),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        titles = [getattr(i, "title", None) for i in music_player.queue.display_items()]
        assert titles == ["Next"], "the handler settled the prefetch's claim"


# ── QueueGet ──────────────────────────────────────────────────────────────────


class TestQueueGet:
    async def test_returns_item_from_queue(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        seed_queue(music_player.queue, queue_obj)
        result = await music_player.queue_get()
        assert result is queue_obj


# ── RestoreStateTtlRefresh ────────────────────────────────────────────────────


class TestRestoreStateTtlRefresh:
    async def test_ttl_refreshed_after_successful_restore(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        # A realistic state hash: volume alone would be MIGRATED out of it by the
        # restore below (see stored_volume), leaving an empty hash that Redis deletes
        # — so the TTL this test is about would have nothing to sit on.
        await fake_redis.hset(
            music_player.store.state_key(), b"voice_channel_id", b"321"
        )
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
        # Interjection flags a real YTDL always carries — truthy MagicMock
        # attributes would trip the loop's start_paused/is_resume gates.
        song.interjected = False
        song.is_resume = False
        song.start_paused = False
        # Enqueue analytics: a real (zero) Analytics, since HistoryEntry.from_song
        # clamps its fields into the play_history column domain — query_source
        # too, which the slug clamp regex-matches.
        song.analytics = ANALYTICS_ZERO
        song.user_input = None
        song.query_source = ""
        # Unstamped: the loop's or-stamp writes the real clock here, and the
        # epoch clamp in HistoryEntry raises on a MagicMock.
        song.played_at = 0.0
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

        seed_queue(music_player.queue, queue_obj)

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

    async def test_skip_notice_includes_reason_and_trace_id(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        seed_queue(music_player.queue, queue_obj)

        failure = StreamFailure(
            detail="RuntimeError: YouTube refused the audio stream",
            trace_id="4b1e1b9c4f66d48943f7aae9b413ee81",
        )

        async def _fail(self_inner: Any, source: Any) -> None:
            self_inner._last_stream_error = failure
            return None

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(MusicPlayer, "_stream_source", new=_fail),
        ):
            await music_player.loop()

        description = (
            mocked(music_player._channel.send).call_args.kwargs["embeds"][0].description
        )
        assert "Failed to load the next song, skipping." in description
        assert failure.detail in description
        assert failure.trace_id in description

    async def test_stream_source_captures_failure(
        self, music_player: MusicPlayer, queue_obj: QueueObject
    ) -> None:
        boom = RuntimeError("YouTube refused the audio stream")
        with patch.object(YTDL, "yt_stream", new=AsyncMock(side_effect=boom)):
            result = await music_player._stream_source(queue_obj)

        assert result is None
        assert music_player._last_stream_error is not None
        assert (
            music_player._last_stream_error.detail
            == "RuntimeError: YouTube refused the audio stream"
        )
        # 32-hex trace id, or the "unavailable" sentinel when no span is active.
        assert music_player._last_stream_error.trace_id

    async def test_resolve_failure_balances_queue_and_redis(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        fake_redis: aioredis.Redis,
    ) -> None:
        """If _resolve_source() raises after queue_get() already dequeued the
        item, the dequeue must still be balanced (song_queue popped, Redis
        popped for a persisted item, the claim settled exactly once)
        and the outer handler's error embed must still be sent."""
        assert music_player.store is not None
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        await music_player.store.push_queue(SongQueueEntry.from_queue_object(queue_obj))
        seed_queue(music_player.queue, queue_obj)

        with patch.object(
            MusicPlayer,
            "_resolve_source",
            new=AsyncMock(side_effect=Exception("yt-dlp lookup failed")),
        ):
            await music_player.loop()

        assert len(music_player.queue._items) == 0
        remaining = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert len(remaining) == 0
        assert music_player.queue._cursor == 0  # the claim settled the get()
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
        seed_queue(music_player.queue, crashed)

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

        seed_queue(music_player.queue, queue_obj)

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
        # _send_now_playing is patched out, so no message ever hosted the block:
        # the play_history "unknown" sentinel, not a forgotten stamp.
        assert music_player.history[0].message_id == 0

    async def test_history_entry_records_the_np_host_message_and_channel(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The history row carries the host of this song, as of song END: not the
        *released* _np_host_message (None by then), and not a host captured before
        this song adopted its own — a hoisted capture yields the PREVIOUS song's id,
        undetectable downstream since every id is a plausible snowflake. The side
        effect on _send_now_playing is what makes the two distinguishable; a bare
        AsyncMock would leave the decoy standing all iteration.

        Both ids come off that ONE message. The hosts here sit in different
        channels, which is what a mid-song host migration looks like: a channel
        read from anywhere else pairs a real message id with a channel that never
        held it, and `channel.get_partial_message(message_id)` then 404s."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        # Decoy: the host left over from the PREVIOUS song, in another channel.
        # Stamping this is the off-by-one-song failure, so neither of its ids may
        # land in history.
        stale_host = AsyncMock(spec=discord.Message)
        stale_host.id = 555555555555555555
        stale_host.channel.id = 111111111111111111
        music_player._np_host_message = stale_host

        this_songs_host = AsyncMock(spec=discord.Message)
        this_songs_host.id = 777777777777777777
        this_songs_host.channel.id = 888888888888888888

        async def adopt_this_songs_host(_song: object) -> None:
            music_player._np_host_message = this_songs_host

        seed_queue(music_player.queue, queue_obj)

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
            patch.object(
                MusicPlayer,
                "_send_now_playing",
                new=AsyncMock(side_effect=adopt_this_songs_host),
            ),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
            # The completed-bar edit is a separate concern and would PATCH the
            # mock host from a background task while this assertion runs.
            patch.object(MusicPlayer, "_fire_finalize_now_playing", new=MagicMock()),
        ):
            await music_player.loop()

        assert music_player._np_host_message is None  # released before the entry
        assert len(music_player.history) == 1
        entry = music_player.history[0]
        assert (entry.message_id, entry.channel_id) == (
            777777777777777777,
            888888888888888888,
        )

    async def test_current_song_is_cleared_before_the_prefetch_await(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The song must stop being 'current' before loop() blocks on prefetch.

        The attach gate is `current_song is not None` and the prefetch await is a
        whole yt-dlp extraction, so a song left current across it lets a command
        response prepend a block for an ended song and adopt ITSELF as host — which
        the next iteration releases without retiring, orphaning a frozen bar. That
        await is loop()'s first suspension point after teardown, so a patched
        prefetch observes exactly that instant."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        observed: list[object] = []

        async def observe_at_prefetch_await(_self: object) -> None:
            observed.append(music_player.current_song)

        seed_queue(music_player.queue, queue_obj)

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
                MusicPlayer, "_prefetch_next_song", new=observe_at_prefetch_await
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        assert observed == [None], (
            "current_song was still set when loop() awaited the prefetch task — "
            "a command response in that window can adopt an orphan NP host"
        )
        # The entry is still built from the iteration's own copy of the song.
        assert len(music_player.history) == 1
        assert music_player.history[0].title == mock_song.title

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

        seed_queue(music_player.queue, queue_obj)

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

    async def test_a_dead_resume_tail_still_records_what_was_heard(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Same dead stream, but on an interjection's resume tail. Zero frames HERE does
        not mean nothing was heard: the offset is audio the interrupted fragment
        played, and that fragment suppressed its own record so this tail would
        carry it. Dropping it on stream_failed lost the whole play."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False
        mock_song.is_resume = True
        mock_song.start_offset = 95  # 95s heard under the fragment that parked it

        seed_queue(music_player.queue, queue_obj)

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
        ):
            await music_player.loop()

        assert [e.webpage_url for e in music_player.history] == [mock_song.webpage_url]

    async def test_song_stopped_before_first_frame_is_not_a_dead_stream(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Zero frames and no ffmpeg error, when the stop was OURS: the stream was
        never refused, so the cached URL survives, no notice is posted, and the song
        keeps its history entry. The marker is the only thing separating this from a
        host that never answered, which MUST drop its cache — see the sibling test."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False  # stopped before the first frame

        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)

        # Marked between vc.play() and `after`, where a real skip lands: the loop
        # clears the marker immediately before vc.play(), so anything earlier is wiped.
        def _play(song: object, after: object) -> None:
            music_player.note_deliberate_stop()
            cast(Any, after)(None)

        vc.play = MagicMock(side_effect=_play)
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

    async def test_stream_that_never_opened_drops_its_cached_url(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The backstop: zero frames, no error, no deliberate stop. discord.py DOES
        report a failing ffmpeg (read() -> _check_process_returncode -> `after`), so
        this is not the ordinary dead stream — it is the window that check declines to
        judge, where the child closed stdout but poll() has not reaped it yet. The
        cached URL must not survive: left in place, every replay reads it and fails the
        same way for the entry's whole TTL. Cache only — history and the absent notice
        stay as in the deliberate-stop case.

        The marker is set BEFORE the loop runs: it must be cleared by the reset that
        precedes vc.play(), or a single earlier -skip would disable this branch for the
        player's whole life."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False  # ffmpeg never opened the input
        mock_song.webpage_url = "https://yt.com/v=blackholed"
        mock_song.start_paused = (
            False  # not a parked song; explicit, not MagicMock-truthy
        )

        # A stale mark from an earlier song. Kills the mutation that deletes the
        # reset before vc.play(): without it this leaks in and suppresses the drop.
        music_player.note_deliberate_stop()

        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)
        # No note_deliberate_stop(): this stop came from ffmpeg dying, not from us.
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

        mock_invalidate.assert_awaited_once()
        assert cast(Any, mock_invalidate.await_args).args[1] == (
            "https://yt.com/v=blackholed"
        )
        # Cache only: no red notice, and the play still earns its history entry.
        music_player._channel.send.assert_not_awaited()
        assert len(music_player.history) == 1

    async def test_drop_unplayable_stream_cache_noops_without_a_store(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Golden rule 5: the in-memory bot keeps working. Without the store guard this
        raises AttributeError on self.store.redis at every zero-frame song end, and the
        loop's outer handler turns it into a misleading "Unhandled error in playback
        loop" — a Redis-less deployment reporting a playback bug."""
        music_player.store = None
        mock_song.webpage_url = "https://yt.com/v=nostore"

        with patch(
            "src.musicplayer.invalidate_stream_cache", new=AsyncMock()
        ) as mock_invalidate:
            await music_player._drop_unplayable_stream_cache(mock_song)

        mock_invalidate.assert_not_awaited()

    async def test_drop_unplayable_stream_cache_noops_without_a_url(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """An info-dict can omit webpage_url; deleting `ytdl:stream:` (the bare prefix)
        would be a write against a key no song owns."""
        mock_song.webpage_url = ""

        with patch(
            "src.musicplayer.invalidate_stream_cache", new=AsyncMock()
        ) as mock_invalidate:
            await music_player._drop_unplayable_stream_cache(mock_song)

        mock_invalidate.assert_not_awaited()

    async def test_parked_paused_song_keeps_its_cached_url(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """A resume tail parked at vc.pause() and torn down before it played has zero
        frames and no error — identical, on those two facts alone, to a stream that
        never opened. It says nothing about the URL, so the entry must survive."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        mock_song.produced_audio = False
        mock_song.webpage_url = "https://yt.com/v=parked"
        mock_song.start_paused = True  # the distinguishing fact

        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=lambda song, after: after(None))
        vc.pause = MagicMock()
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

        seed_queue(music_player.queue, queue_obj)

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

        seed_queue(music_player.queue, queue_obj)

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

    async def _run_one_song(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        pop_spy: AsyncMock,
    ) -> None:
        """One full loop iteration with the start transaction spied on."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        assert music_player.store is not None
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

    async def test_the_handler_waits_for_the_prefetch_it_cancelled(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
        """The outer handler awaits the cancel before it moves on.

        An unawaited cancel leaves the prefetch's requeue_front to whatever awaits
        next, and two live claims settle by POSITION — so each takes the other's
        song. What is pinned is that the prefetch is SETTLED by the time the handler
        continues."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock(side_effect=RuntimeError("boom"))
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        hung = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = hung
        settled: list[bool] = []

        async def _record(_self: MusicPlayer) -> None:
            settled.append(hung.done())

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
            patch.object(MusicPlayer, "_cancel_progress_task", new=_record),
        ):
            await music_player.loop()

        assert settled == [True]

    async def test_a_song_queued_after_a_clear_still_plays(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """queue_get() parks, and a -clear during that wait bumps the generation.
        The commit generation is re-read AFTER the wait for exactly this reason: the
        item finally handed out came from the queue as it is NOW, so comparing it
        against a sample taken before the wait voids a dequeue that is perfectly
        valid — the song queued after the clear is discarded without playing, with
        no error and no log line."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()
        queued_after = QueueObject("https://yt.com/v=after", "After", mock_author)

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queued_after)
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
            runner = asyncio.create_task(music_player.loop())
            await asyncio.sleep(0)  # park it inside queue_get()
            await music_player.queue.clear()
            await music_player.queue.put([queued_after])
            await runner

        vc.play.assert_called_once()

    async def test_a_crash_recovered_head_never_lpops_a_queued_song(
        self,
        music_player: MusicPlayer,
        mock_song: MagicMock,
        mock_author: MagicMock,
        fake_redis: aioredis.Redis,
    ) -> None:
        """persisted=False says this song's entry was LPOPed in the run that
        crashed, so nothing on the list belongs to it. An LPOP here retires the
        HEAD instead — an unrelated, still-queued song, deleted from Redis on every
        restart-with-recovery, with no error and nothing in memory to notice."""
        assert music_player.store is not None
        crashed = QueueObject(
            "https://yt.com/v=crashed", "Crashed", mock_author, persisted=False
        )
        queued = QueueObject("https://yt.com/v=queued", "Queued", mock_author)
        # The production shape restore_crashed leaves behind: the crashed head is
        # in memory only, the queued song is on both legs.
        seed_queue(music_player.queue, crashed)
        await music_player.queue.put([queued])
        assert len(await fake_redis.lrange(music_player.store.queue_key(), 0, -1)) == 1

        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=crashed)
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

        stored = await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        assert stored == [SongQueueEntry.from_queue_object(queued).to_redis()]

    async def test_the_loop_starts_and_retires_the_heartbeat_task(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Nothing else ties the feature to the loop. _heartbeat_updater and
        store.heartbeat are each tested in isolation, so deleting the create_task
        leaves every crash recovering at whatever the start transaction seeded —
        0:00 for an ordinary song — with a green suite.
        """
        observed: list[Any] = []
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc

        async def _capture_mid_song() -> None:
            observed.append(music_player._heartbeat_task)

        music_player.play_next.wait = AsyncMock(side_effect=_capture_mid_song)

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

        assert observed == [observed[0]] and observed[0] is not None, (
            "loop() never started the heartbeat task"
        )
        assert observed[0].get_coro().__name__ == "_heartbeat_updater"
        # Retired by the time the iteration settles. A natural end does NOT clear
        # VoiceClient._player, so vc.source still returns this song and the
        # updater's own guard cannot fire — this cancel is what stops it writing
        # over the fields clear_song_end_state() just deleted.
        assert music_player._heartbeat_task is None

    async def test_a_redis_less_loop_starts_no_heartbeat_task(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The store is the task's only writer, so without the guard a degraded
        guild ticks for every song of its life to reach a no-op."""
        observed: list[Any] = []
        music_player.store = None
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc

        async def _capture_mid_song() -> None:
            observed.append(music_player._heartbeat_task)

        music_player.play_next.wait = AsyncMock(side_effect=_capture_mid_song)

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

        assert observed == [None], "a store-less loop started a heartbeat task"

    async def test_played_at_is_stamped_before_the_start_transaction(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The parked entry must carry the start, not 0.0.

        A playing song has no queue entry anywhere persistent — this transaction's
        own LPOP destroyed it — so the hash is its only at-rest copy. Stamping
        after from_song() persists 0.0 while the in-memory song says otherwise,
        and the crash path then recovers a song with no start at all."""
        assert music_player.store is not None
        pop_spy = AsyncMock(wraps=music_player.store.pop_queue_and_start_song)
        before = time.time()

        await self._run_one_song(music_player, queue_obj, mock_song, pop_spy)

        current = pop_spy.call_args.args[0]
        assert isinstance(current, SongQueueEntry)
        assert before <= current.played_at <= time.time()
        assert current.played_at == mock_song.played_at  # one value, not two clocks

    async def test_played_at_is_the_wall_clock_not_the_backdated_epoch(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """play_start_epoch is deliberately pulled back by the FFmpeg -ss offset so
        recovery math yields true audio position. Reusing it here would claim a
        `?t=60` song started a minute before anyone pressed play."""
        assert music_player.store is not None
        mock_song.start_offset = 60
        pop_spy = AsyncMock(wraps=music_player.store.pop_queue_and_start_song)

        await self._run_one_song(music_player, queue_obj, mock_song, pop_spy)

        current, backdated_start = pop_spy.call_args.args[:2]
        assert current.played_at == pytest.approx(backdated_start + 60)

    async def test_the_start_transaction_seeds_the_heartbeat_from_the_offset(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """No heartbeat has ticked yet, so a crash inside the first interval resumes
        from whatever this wrote. The store defaults the argument to 0, which reads
        as a `?t=60` song having reached 0:00."""
        assert music_player.store is not None
        mock_song.start_offset = 60
        pop_spy = AsyncMock(wraps=music_player.store.pop_queue_and_start_song)

        await self._run_one_song(music_player, queue_obj, mock_song, pop_spy)

        assert pop_spy.call_args.kwargs["start_offset"] == 60

    async def test_inherited_played_at_is_not_restamped(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """An interjection's resume tail arrives already carrying the interrupted song's
        start, and the or-stamp must leave it alone: restamping files the two
        fragments of one play as two plays, minutes apart."""
        assert music_player.store is not None
        mock_song.played_at = 1752530000.5
        mock_song.is_resume = True
        pop_spy = AsyncMock(wraps=music_player.store.pop_queue_and_start_song)

        with patch.object(MusicPlayer, "_announce_resume", new=AsyncMock()):
            await self._run_one_song(music_player, queue_obj, mock_song, pop_spy)

        assert pop_spy.call_args.args[0].played_at == 1752530000.5
        assert music_player.history[0].played_at == 1752530000.5

    async def test_loop_clears_play_message_on_song_end(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """After a song finishes, -now must not serve the finished song's embed
        via the crash-recovery elif — play_message is cleared with current_song."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        seed_queue(music_player.queue, queue_obj)
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

        seed_queue(music_player.queue, queue_obj)
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

        seed_queue(music_player.queue, queue_obj)

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

        seed_queue(music_player.queue, queue_obj)

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

        seed_queue(music_player.queue, queue_obj)

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

        seed_queue(music_player.queue, queue_obj)

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

        seed_queue(music_player.queue, queue_obj)

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
        # Interjection flags a real YTDL always carries — truthy MagicMock
        # attributes would trip the loop's start_paused/is_resume gates.
        song.interjected = False
        song.is_resume = False
        song.start_paused = False
        # Enqueue analytics: a real (zero) Analytics, since HistoryEntry.from_song
        # clamps its fields into the play_history column domain — query_source
        # too, which the slug clamp regex-matches.
        song.analytics = ANALYTICS_ZERO
        song.user_input = None
        song.query_source = ""
        # Unstamped: the loop's or-stamp writes the real clock here, and the
        # epoch clamp in HistoryEntry raises on a MagicMock.
        song.played_at = 0.0
        return song

    async def test_update_activity_called_at_song_start_and_end(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        seed_queue(music_player.queue, queue_obj)

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
        """_queue_cleared set while a prefetch is in-flight: the loop discards the
        prefetched song and cleanup()s it so no FFmpeg subprocess leaks. Iteration 1
        plays song 1 while the prefetch dequeues song 2 and sets the flag; iteration
        2 fires the guard (claim already settled + cleanup + discard), then times out."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        queue_obj2 = QueueObject(
            "https://yt.com/watch?v=2", "Song 2", queue_obj.requester
        )
        seed_queue(music_player.queue, queue_obj, queue_obj2)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        prefetched = MagicMock()
        prefetched.cleanup = MagicMock()

        gets: list[int] = []

        async def _real_get_then_timeout(_self: Any) -> Any:
            if gets:
                raise asyncio.TimeoutError()
            gets.append(1)
            return await music_player.queue.get()

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
            # Claims through the real queue rather than an AsyncMock: the loop
            # commits that claim two lines later, and a commit with nothing
            # claimed is a state production cannot reach.
            patch.object(MusicPlayer, "queue_get", new=_real_get_then_timeout),
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

        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc

        mock_song.cleanup = MagicMock()

        async def _stream_and_clear(_self: Any, source: Any) -> MagicMock:
            # A real clear(), not a hand-emptied deque: the point is that
            # the commit afterwards finds nothing to settle, and only the real
            # one bumps the generation the commit checks.
            await music_player.queue.clear()
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


# ── interjection ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vc() -> MagicMock:
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    return vc


@pytest.fixture
def live_song(mock_song: MagicMock) -> MagicMock:
    """mock_song with the interjection flags a real YTDL carries — bare MagicMock
    attributes would read truthy and trip the loop's is_resume/start_paused
    gates."""
    mock_song.interjected = False
    mock_song.is_resume = False
    mock_song.start_paused = False
    # Both have been silently dropped by the rebuild before now — persisted as an
    # outright AttributeError (YTDL had no such attribute at all), user_input as a
    # quiet default. A bare MagicMock reads truthy for either and hides both.
    mock_song.user_input = "https://open.spotify.com/playlist/live"
    mock_song.persisted = True
    return mock_song


@pytest.fixture
def interject_obj(mock_author: MagicMock) -> QueueObject:
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=urgent",
        title="Urgent Song",
        requester=mock_author,
        duration=120,
        interjected=True,
    )


class TestInterject:
    async def test_follow_on_sits_between_the_head_and_the_resume_tail(
        self,
        music_player: MusicPlayer,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
        live_song: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """`--now <playlist>`: the head interrupts, the rest play in order behind
        it, and the interrupted song comes back after ALL of it.

        The ordering is the decision, not an accident of insertion — a resume tail
        that landed between the head and the tracks would make `--now` on a playlist
        mean "play one track of this now", which is the behaviour this replaced.
        """
        music_player.current_song = live_song
        rest = [
            QueueObject(f"https://yt.com/v=t{i}", f"Track {i}", mock_author)
            for i in (2, 3)
        ]

        outcome = await music_player.interject(interject_obj, mock_vc, follow_on=rest)

        assert outcome is not None
        titles = [
            queue_object(item).title for item in music_player.queue.display_items()
        ]
        assert titles == ["Urgent Song", "Track 2", "Track 3", live_song.title]
        tail = queue_object(music_player.queue.display_items()[-1])
        assert tail.is_resume is True
        # The span attribute describes the same insert: one play is parked. It
        # counted the CONSECUTIVE run behind the head, which the playlist between
        # them empties — so every playlist interjection reported 0.
        assert music_player.queue.resume_tail_depth() == 1

    async def test_returns_none_without_current_song(
        self, music_player: MusicPlayer, interject_obj: QueueObject, mock_vc: MagicMock
    ) -> None:
        music_player.current_song = None
        assert await music_player.interject(interject_obj, mock_vc) is None
        mock_vc.stop.assert_not_called()

    async def test_front_inserts_the_interjection_then_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 42.0
        music_player.current_song = live_song
        queued = QueueObject("https://yt.com/v=b", "Queued B", mock_author)
        await music_player.queue.put([queued])

        outcome = await music_player.interject(interject_obj, mock_vc)

        items = music_player.queue.display_items()
        assert items[0] is interject_obj
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

    async def test_interjection_keeps_its_own_analytics_and_tail_keeps_the_original(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        # interject() no longer stamps anything: the interruption arrives already
        # carrying the depth-0 analytics the command minted at dispatch, and the tail
        # is the same play, so it keeps the interrupted song's.
        interject_obj.analytics = Analytics(queued_at=1752530500.5, queue_position=0)
        live_song.elapsed_secs = 42.0
        live_song.analytics = Analytics(queued_at=1752530000.5, queue_position=5)
        music_player.current_song = live_song

        await music_player.interject(interject_obj, mock_vc)

        assert interject_obj.analytics == Analytics(
            queued_at=1752530500.5, queue_position=0
        )
        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.analytics is live_song.analytics

    async def test_resume_tail_inherits_the_query_source(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """The tail writes the ONLY history row for an interrupted play — the
        interrupted fragment's entry is suppressed by _skip_history_for — and the
        classification cannot be rebuilt from webpage_url, since a Spotify link, a
        plaintext search and a pasted YouTube link all archive as youtube.com.
        Dropped here it is gone: no error, no log line, and the row reads as a
        pre-feature one."""
        live_song.elapsed_secs = 42.0
        live_song.query_source = "spotify.com"
        music_player.current_song = live_song

        await music_player.interject(interject_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.query_source == "spotify.com"

    async def test_resume_tail_inherits_the_interrupted_songs_start(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        # The tail IS the interrupted play, so history files it under when that
        # play began. Left unset, the loop's or-stamp would restamp the tail to
        # when it resumed — one play recorded as if it started twice.
        live_song.elapsed_secs = 42.0
        live_song.played_at = 1752530000.5
        music_player.current_song = live_song

        await music_player.interject(interject_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.played_at == 1752530000.5
        # The interruption is its own play and has not started yet.
        assert interject_obj.played_at == 0.0

    async def test_paused_song_returns_start_paused(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(interject_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.start_paused is True
        assert outcome is not None and outcome.was_paused is True
        # `--now`'s default: restore exactly what was interrupted.
        assert outcome.returns_paused is True

    async def test_resume_paused_false_returns_song_playing(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """-play on a paused song means "stop being paused, play this" — the
                interrupted song comes back PLAYING at its pause position
        ."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(
            interject_obj, mock_vc, resume_paused=False
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
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = False

        outcome = await music_player.interject(
            interject_obj, mock_vc, resume_paused=False
        )

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.start_paused is False
        assert outcome is not None and outcome.returns_paused is False

    async def test_returns_paused_false_when_no_resume_entry(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """Nearly-finished song → no resume entry at all, so nothing returns."""
        live_song.elapsed_secs = 207.0  # 3s left of 210 — below the 5s floor
        music_player.current_song = live_song
        mock_vc.is_paused.return_value = True

        outcome = await music_player.interject(interject_obj, mock_vc)

        assert outcome is not None
        assert outcome.resume_position is None
        assert outcome.returns_paused is False

    async def test_interjection_over_an_interjection_stacks(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """The inverse of the old replace semantics: an interjected song is
        parked like any other. Its own resume tail, if it has one, is already
        behind it and stays there — the queue unwinds LIFO."""
        live_song.interjected = True  # the playing song is itself interjected
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        outcome = await music_player.interject(interject_obj, mock_vc)

        display = music_player.queue.display_items()
        assert display[0] is interject_obj
        parked = display[1]
        assert isinstance(parked, QueueObject)
        assert parked.is_resume is True
        assert parked.ts == 30
        # It returns, so its history entry waits for the tail rather than being
        # written now — the marker is what defers it.
        assert music_player._skip_history_for is live_song
        mock_vc.stop.assert_called_once()
        assert outcome is not None
        assert outcome.resume_position == 30

    async def _stack(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
        depth: int,
    ) -> list[QueueObject]:
        """Interject `depth` times, each over the previous interjection, as a user
        running it repeatedly would. Returns the songs in the order they cut
        in. Each round makes the new song current, since the loop is not running
        here to do it."""
        cut_in: list[QueueObject] = []
        for n in range(1, depth + 1):
            if cut_in:
                # The playback loop consumes the song that just cut in before the
                # next interjection can land. Without this the display keeps entries a
                # running loop would already have dequeued, and the tails stop
                # being contiguous.
                assert music_player.queue.get_nowait() is cut_in[-1]
                await music_player.queue.try_commit_dequeue(
                    music_player.queue.generation
                )
            live_song.elapsed_secs = float(30 * n)
            music_player.current_song = live_song
            # What the loop's vc.play() does: the previous interjection's stop is
            # over once the next song is live.
            music_player._stopped_deliberately = False
            qobj = QueueObject(
                f"https://yt.com/v=cut{n}",
                f"Cut {n}",
                mock_author,
                duration=120,
                interjected=True,
            )
            await music_player.interject(qobj, mock_vc)
            cut_in.append(qobj)
            # It is playing now, so the next interjection interrupts THIS song.
            live_song.webpage_url = qobj.webpage_url
            live_song.title = qobj.title
            live_song.interjected = True
        return cut_in

    async def test_three_deep_stack_unwinds_lifo(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """The whole feature: three interjections leave three parked plays, each
        resuming from where it actually stopped, most recent first."""
        cut_in = await self._stack(music_player, live_song, mock_vc, mock_author, 3)

        display = music_player.queue.display_items()
        assert display[0] is cut_in[2]  # the newest cut is playing next
        tails = [t for t in display[1:] if isinstance(t, QueueObject)]
        assert len(tails) == 3 and all(t.is_resume for t in tails)
        assert [t.webpage_url for t in tails] == [
            "https://yt.com/v=cut2",  # interrupted most recently → returns first
            "https://yt.com/v=cut1",
            "https://www.youtube.com/watch?v=testid",  # the original song, last
        ]
        # ts is absolute at every level: each tail resumes at the position that
        # song had actually reached, not at its own fragment's start.
        assert [t.ts for t in tails] == [90, 60, 30]

    async def test_depth_is_recorded_on_the_span(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """Depth counts parked PLAYS, and it is the only observability on how deep
        real guilds stack — the field it replaced (`interject.replaced`) is gone."""
        await self._stack(music_player, live_song, mock_vc, mock_author, 3)
        assert music_player.queue.resume_tail_depth() == 3

    async def test_a_tail_interjected_again_keeps_its_stamps(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """A resumed song interrupted a second time is still ONE play: it keeps the
        start it was first heard at and the position it was first queued at, so it
        files under one moment however many fragments it ends up in."""
        live_song.is_resume = True  # this fragment is itself a resumed tail
        live_song.elapsed_secs = 95.0
        live_song.played_at = 1752530000.5
        live_song.analytics = Analytics(queued_at=1752529000.5, queue_position=4)
        music_player.current_song = live_song

        await music_player.interject(interject_obj, mock_vc)

        tail = music_player.queue.display_items()[1]
        assert isinstance(tail, QueueObject)
        assert tail.played_at == 1752530000.5
        assert (tail.analytics.queued_at, tail.analytics.queue_position) == (
            1752529000.5,
            4,
        )
        assert tail.ts == 95  # absolute, so the next fragment resumes here

    async def test_pending_resume_tail_is_set_with_the_history_marker(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """The tail cannot be stamped with its NP card here — the confirmation
        this command is about to send can still adopt (and retire) a different
        host. It is parked for the loop's iteration end instead, alongside the
        history marker, and the two are set together or not at all."""
        live_song.elapsed_secs = 42.0
        music_player.current_song = live_song

        await music_player.interject(interject_obj, mock_vc)

        tail = music_player.queue.display_items()[1]
        assert music_player._pending_resume_tail is tail
        assert music_player._skip_history_for is live_song
        # Still unstamped: nothing is known about the card yet.
        assert isinstance(tail, QueueObject)
        assert (tail.np_message_id, tail.np_channel_id, tail.np_host_ref) == (
            0,
            0,
            None,
        )

    async def test_no_resume_entry_parks_no_tail(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        # Nearly-over song: no tail exists, so neither slot may be set — a stale
        # one would receive a later fragment's ids.
        live_song.elapsed_secs = 207.0
        music_player.current_song = live_song

        await music_player.interject(interject_obj, mock_vc)

        assert music_player._pending_resume_tail is None
        assert music_player._skip_history_for is None

    async def test_near_end_skips_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 207.0  # 3s left of 210 — below the 5s floor
        music_player.current_song = live_song

        outcome = await music_player.interject(interject_obj, mock_vc)

        assert music_player.queue.display_items() == [interject_obj]
        assert outcome is not None and outcome.resume_position is None
        assert music_player._skip_history_for is None

    async def test_eof_cap_pulls_position_back(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 205.0  # 5s left: resumable, but capped to 200
        music_player.current_song = live_song

        outcome = await music_player.interject(interject_obj, mock_vc)

        resume = music_player.queue.display_items()[1]
        assert isinstance(resume, QueueObject)
        assert resume.ts == 200  # duration 210 − 10s EOF margin
        assert outcome is not None and outcome.resume_position == 200

    async def test_no_webpage_url_skips_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.webpage_url = None
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        outcome = await music_player.interject(interject_obj, mock_vc)

        assert music_player.queue.display_items() == [interject_obj]
        assert outcome is not None and outcome.resume_position is None

    async def test_stop_skipped_when_song_changed_during_insert(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
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
            await music_player.interject(interject_obj, mock_vc)

        mock_vc.stop.assert_not_called()
        # The marker is taken even though the song moved on: the tail is on the
        # queue whatever the loop did, so it records this play. Leaving it unset
        # would let the loop's own iteration end record it too.
        assert music_player._skip_history_for is live_song

    async def test_depth_and_over_interjection_reach_the_span(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """Asserted on the SPAN, not by calling resume_tail_depth() directly. The
        depth attribute is the only observability on how deep real guilds stack,
        and interject.replaced was deleted — so both can be dropped or misnamed
        with the depth helper still perfectly tested."""
        live_song.elapsed_secs = 30.0
        live_song.interjected = True
        music_player.current_song = live_song
        attrs: dict[str, Any] = {}
        span = MagicMock()
        span.set_attribute = lambda k, v: attrs.__setitem__(k, v)

        with patch("src.musicplayer.trace.get_current_span", return_value=span):
            await music_player.interject(interject_obj, mock_vc)

        assert attrs["interject.depth"] == 1
        assert attrs["interject.over_interjection"] is True

    async def test_interject_marks_the_stop_as_deliberate_before_stopping(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """An interjection stops the live song, so the loop must not read that as a
        stream that never opened — inside ffmpeg's startup window the two are identical
        apart from this marker, and the song would lose its still-good cached URL.
        Ordered: marked BEFORE vc.stop(), since `after` can fire the moment it lands."""
        live_song.elapsed_secs = 30.0
        live_song.produced_audio = False  # stopped inside ffmpeg's startup window
        music_player.current_song = live_song
        order: list[str] = []
        mock_vc.stop = MagicMock(side_effect=lambda: order.append("stop"))

        with patch.object(
            MusicPlayer,
            "note_deliberate_stop",
            side_effect=lambda: order.append("mark"),
        ):
            await music_player.interject(interject_obj, mock_vc)

        assert order == ["mark", "stop"]

    async def test_the_marker_is_taken_before_the_insert_await(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """A teardown landing inside put_front's Redis round trip must find the
        marker already there, or it claims the song for history and the tail
        records it too — one play, two rows. The marker is what declines the
        claim, so it has to exist for as long as the tail does."""
        live_song.elapsed_secs = 30.0
        live_song.produced_audio = True
        music_player.current_song = live_song
        seen: list[Any] = []

        async def claim_mid_await(items: Any) -> None:
            seen.append(music_player.claim_current_song_for_history())

        from src.guild_queue import GuildQueue

        with patch.object(GuildQueue, "put_front", side_effect=claim_mid_await):
            await music_player.interject(interject_obj, mock_vc)

        assert seen == [None]  # the teardown declined; the tail owns the record

    async def test_neutralizes_running_prefetch_first(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        blocker = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = blocker

        await music_player.interject(interject_obj, mock_vc)

        assert blocker.cancelled()
        assert music_player._prefetch_task is None


class TestPrefetchedHeadRespectsPersistence:
    """A prefetched claim can be a crash-recovered head, which is on no Redis list:
    `-play` on a DISCONNECTED bot inserts at cursor 0, AHEAD of the head
    `_restore_state` appended, so the prefetch behind it claims that head. LPOPing
    for an entry that was never on the list deletes the next real one instead —
    at-most-once, surfacing only as a queue one song short after the next restart.

    Driven through two real loop iterations, because that is the only way into the
    branch: iteration 1 spawns the prefetch, iteration 2 consumes its result."""

    async def _run_with_prefetch(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        first_song: MagicMock,
        prefetched: MagicMock,
    ) -> tuple[AsyncMock, AsyncMock]:
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        # TWO items: one for iteration 1, one for the prefetch to claim. The stand-in
        # below takes that claim with get_nowait() exactly as the real
        # _prefetch_next_song does — without it the commit in iteration 2 finds
        # nothing claimed and refuses, and the branch under test never runs.
        seed_queue(music_player.queue, queue_obj)
        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        async def _prefetch() -> MagicMock:
            music_player.queue.get_nowait()
            return prefetched

        assert music_player.store is not None
        pop_spy = AsyncMock()
        set_spy = AsyncMock()
        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=first_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=AsyncMock()),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(side_effect=_prefetch)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
            patch.object(music_player.store, "pop_queue_and_start_song", pop_spy),
            patch.object(music_player.store, "set_current_song_state", set_spy),
        ):
            await music_player.loop()
        return pop_spy, set_spy

    async def test_an_unpersisted_prefetched_head_does_not_lpop(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        live_song: MagicMock,
    ) -> None:
        live_song.persisted = False
        pop_spy, set_spy = await self._run_with_prefetch(
            music_player, queue_obj, mock_song, live_song
        )
        # One LPOP for the first (persisted) song, none for the recovered head.
        assert pop_spy.await_count == 1
        set_spy.assert_awaited_once()

    async def test_the_recovered_head_seeds_the_heartbeat_from_its_offset(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        live_song: MagicMock,
    ) -> None:
        """The double-crash path. This branch writes a song already recovered part
        way through; seeding 0.0 there shadows the legacy fallback (0.0 is not None),
        so a second crash restarts it at 0:00 — worse than not having the heartbeat.
        """
        live_song.persisted = False
        live_song.start_offset = 180
        _, set_spy = await self._run_with_prefetch(
            music_player, queue_obj, mock_song, live_song
        )
        set_spy.assert_awaited_once()
        assert set_spy.call_args.kwargs["start_offset"] == 180

    async def test_a_persisted_prefetched_head_still_lpops(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        live_song: MagicMock,
    ) -> None:
        """The common case must keep its LPOP — a fix that simply stopped popping
        would leave the mirror permanently ahead of memory."""
        live_song.persisted = True
        pop_spy, set_spy = await self._run_with_prefetch(
            music_player, queue_obj, mock_song, live_song
        )
        assert pop_spy.await_count == 2
        set_spy.assert_not_awaited()


class TestFailedPrefetchedClaimRespectsPersistence:
    """The same rule on the FAILURE leg. The start path reads `persisted` off the
    song, and the outer handler cannot re-derive it from `source`, which the
    prefetched branch leaves None — and None defaults to popping, which would
    retire a real entry for a head that never had one. `claim_persisted` is
    carried from the claim to the handler so neither end has to guess."""

    async def _run_until_it_breaks(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        first_song: MagicMock,
        prefetched: MagicMock,
    ) -> AsyncMock:
        """Two iterations: the first plays normally and spawns the prefetch, the
        second claims its result and then dies before committing."""
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
        seed_queue(music_player.queue, queue_obj)

        vc = object.__new__(discord.VoiceClient)
        vc.play = MagicMock()
        mocked(music_player._guild).voice_client = vc
        music_player.play_next.wait = AsyncMock()

        async def _prefetch() -> MagicMock:
            music_player.queue.get_nowait()
            return prefetched

        # Raise inside the claim→commit window, on the prefetched iteration only.
        # try_commit_dequeue is the one await in that window, which is what makes
        # this the narrow-but-real path the handler exists for. Patched on the
        # CLASS: GuildQueue is __slots__ed, so the instance takes no override.
        real_commit = GuildQueue.try_commit_dequeue
        commits = 0

        async def _commit(queue: GuildQueue, generation: int) -> bool:
            nonlocal commits
            commits += 1
            if commits == 2:
                raise RuntimeError("boom inside the claim window")
            return await real_commit(queue, generation)

        assert music_player.store is not None
        lpop_spy = AsyncMock()
        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=first_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=AsyncMock()),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(side_effect=_prefetch)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
            patch.object(MusicPlayer, "send_with_np", new=AsyncMock()),
            patch.object(
                music_player.store, "pop_queue_and_start_song", new=AsyncMock()
            ),
            patch.object(music_player.store, "set_current_song_state", new=AsyncMock()),
            patch.object(GuildQueue, "try_commit_dequeue", new=_commit),
            patch.object(music_player.store, "pop_queue", lpop_spy),
        ):
            await music_player.loop()
        assert commits == 2, "the prefetched iteration never reached the commit"
        return lpop_spy

    async def test_an_unpersisted_claim_is_settled_without_an_lpop(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        live_song: MagicMock,
    ) -> None:
        live_song.persisted = False
        lpop_spy = await self._run_until_it_breaks(
            music_player, queue_obj, mock_song, live_song
        )
        lpop_spy.assert_not_awaited()

    async def test_a_persisted_claim_still_retires_its_entry(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        live_song: MagicMock,
    ) -> None:
        """The other half: a fix that just stopped popping here would strand the
        mirror one entry ahead of memory on every failed claim."""
        live_song.persisted = True
        lpop_spy = await self._run_until_it_breaks(
            music_player, queue_obj, mock_song, live_song
        )
        lpop_spy.assert_awaited_once()


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

    async def test_the_rebuild_reads_a_real_ytdl(
        self,
        music_player: MusicPlayer,
        ytdl_instance: Callable[..., Any],
        mock_author: MagicMock,
    ) -> None:
        """Drives the rebuild off a REAL YTDL, not a MagicMock. Every field it
        reads must exist on YTDL; one that does not raises AttributeError here
        rather than in production, where the claim is already stranded. `persisted`
        reached main missing, and a mock invented it as a truthy Mock."""
        original = QueueObject("https://yt.com/v=next", "Next Song", mock_author)
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        song = ytdl_instance(
            None,
            user_input="https://open.spotify.com/playlist/abc",
            persisted=False,
            interjected=True,
            is_resume=True,
            start_paused=True,
            query_source="spotify.com",
            played_at=1234.5,
            np_message_id=77,
            np_channel_id=88,
            np_dedicated=True,
        )
        song.cleanup = MagicMock()

        async def _done() -> Any:
            return song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        # Asserted together so a field dropped from the rebuild fails here rather
        # than needing to be noticed.
        assert (
            rebuilt.user_input,
            rebuilt.persisted,
            rebuilt.interjected,
            rebuilt.is_resume,
            rebuilt.start_paused,
            rebuilt.query_source,
            rebuilt.played_at,
            rebuilt.np_message_id,
            rebuilt.np_channel_id,
            rebuilt.np_dedicated,
        ) == (
            "https://open.spotify.com/playlist/abc",
            False,
            True,
            True,
            True,
            "spotify.com",
            1234.5,
            77,
            88,
            True,
        )

    async def test_completed_task_rebuild_keeps_offset_and_interjection_flags(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """Nested interjection: the prefetcher resolves the first one's resume
        entry on a cache hit, so a second one neutralizes a completed prefetch
        holding a flagged, offset entry. A rebuild dropping ts/is_resume/start_paused
        restarts the interrupted song from 0:00, unpaused and unannounced."""
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

    async def test_completed_task_rebuild_keeps_enqueue_stamps(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        # Losing them here zeroes the ask this play was queued against, and
        # nothing re-mints it: the archive would read "queued at unknown,
        # played immediately".
        original = QueueObject(
            "https://yt.com/v=orig",
            "Interrupted Song",
            mock_author,
            analytics=Analytics(queued_at=1752530000.5, queue_position=6),
        )
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.analytics = Analytics(queued_at=1752530000.5, queue_position=6)
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert (rebuilt.analytics.queued_at, rebuilt.analytics.queue_position) == (
            1752530000.5,
            6,
        )

    async def test_completed_task_rebuild_keeps_the_np_card_pointer(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        # A prefetched resume tail neutralized by a second interjection: dropping these
        # strands the previous fragment's frozen card with nothing left that knows
        # to delete it. The runtime ref rides along too — it is what allows a
        # strip-edit, which the ids alone cannot do.
        ref = NpHostRef(AsyncMock(spec=discord.Message), [], True)
        original = QueueObject(
            "https://yt.com/v=orig",
            "Interrupted Song",
            mock_author,
            ts=151,
            is_resume=True,
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
            np_dedicated=True,
            np_host_ref=ref,
        )
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.start_offset = 151
        live_song.is_resume = True
        live_song.np_message_id = 777777777777777777
        live_song.np_channel_id = 888888888888888888
        live_song.np_dedicated = True
        live_song.np_host_ref = ref
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert (rebuilt.np_message_id, rebuilt.np_channel_id, rebuilt.np_dedicated) == (
            777777777777777777,
            888888888888888888,
            True,
        )
        assert rebuilt.np_host_ref is ref

    async def test_completed_task_rebuild_keeps_played_at(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        # A resume tail is the only prefetch that carries a start, and a second
        # An interjection neutralizing it is exactly when that happens. Dropped here,
        # loop's or-stamp refiles the tail under whenever it eventually resumes.
        original = QueueObject(
            "https://yt.com/v=orig",
            "Interrupted Song",
            mock_author,
            ts=151,
            is_resume=True,
            played_at=1752530000.5,
        )
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.start_offset = 151
        live_song.is_resume = True
        live_song.played_at = 1752530000.5
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.played_at == 1752530000.5

    async def test_completed_task_rebuild_keeps_query_source(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        # Nothing downstream of the rebuild can recover the classification, so
        # dropping it here would archive an interjected-over song as unknown.
        original = QueueObject(
            "https://yt.com/v=orig",
            "Interrupted Song",
            mock_author,
            query_source="spotify.com",
        )
        await music_player.queue.put([original])
        assert music_player.queue.get_nowait() is original

        live_song.query_source = "spotify.com"
        live_song.cleanup = MagicMock()

        async def _done() -> MagicMock:
            return live_song

        task = asyncio.create_task(_done())
        await task
        music_player._prefetch_task = task

        await music_player._neutralize_prefetch()

        rebuilt = music_player.queue.get_nowait()
        assert isinstance(rebuilt, QueueObject)
        assert rebuilt.query_source == "spotify.com"

    async def test_completed_task_rebuild_keeps_interjected_flag(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_author: MagicMock
    ) -> None:
        """A parked interjected entry must keep its marker through the rebuild —
        losing it would make a later interjection stack a resume entry for it
        instead of applying replace semantics."""
        original = QueueObject(
            "https://yt.com/v=pn", "Interjected Song", mock_author, interjected=True
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

    async def test_already_cancelled_done_task_is_treated_as_no_song(
        self, music_player: MusicPlayer
    ) -> None:
        """A *done and cancelled* prefetch reaches .result() as CancelledError — the
        arm of `except asyncio.CancelledError, Exception` no other test exercises.
        Statement coverage cannot see the gap: the RuntimeError test above marks the
        same `except` line covered, so dropping CancelledError would report 100%
        while raising straight out of _neutralize_prefetch. Reached when the task is
        cancelled without being cleared and settles before interject() runs, so the
        `not task.done()` path above does not apply."""
        task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)  # let it start so cancel() lands on a live task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.done() and task.cancelled()

        music_player._prefetch_task = task
        await music_player._neutralize_prefetch()  # must not raise

        # "no song" — nothing to requeue and nothing to clean up.
        assert music_player.queue.qsize() == 0
        assert music_player._prefetch_task is None

    def test_cancellederror_catch_stays_unreachable_by_own_cancellation(self) -> None:
        """Structural guard: no `await` may follow the CancelledError handler.

        Catching CancelledError is safe for one reason only — the tail of
        _neutralize_prefetch is fully synchronous, so this coroutine's own
        cancellation can never be delivered inside or after the handler; the only
        one it can observe is .result()'s. Making requeue_front async, or awaiting
        anything in the rebuild, silently turns the handler into a cancellation sink.
        No runtime test can catch that (no suspension point exists to cancel at), so
        this asserts on the AST."""
        import ast
        import inspect

        import src.musicplayer as mp_module

        tree = ast.parse(inspect.getsource(mp_module))
        func = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_neutralize_prefetch"
        )

        def catches_cancelled(node: ast.stmt) -> bool:
            if not isinstance(node, ast.Try):
                return False
            for h in node.handlers:
                if h.type is None:
                    continue
                caught = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
                if any(ast.unparse(e).endswith("CancelledError") for e in caught):
                    return True
            return False

        try_node = next((n for n in func.body if catches_cancelled(n)), None)
        if try_node is None:
            # Nothing catches CancelledError any more, so there is no
            # cancellation sink to guard. That is a behaviour change owned by
            # test_already_cancelled_done_task_is_treated_as_no_song, which
            # fails loudly on it — this guard is simply vacuous.
            return
        idx = func.body.index(try_node)

        offenders = [
            ast.unparse(node)
            for stmt in func.body[idx:]
            for node in ast.walk(stmt)
            if isinstance(node, ast.Await)
        ]
        assert not offenders, (
            "an `await` now sits at or after the CancelledError handler in "
            f"_neutralize_prefetch: {offenders}. The handler can now swallow "
            "this coroutine's own cancellation. Narrow the try/except to the "
            "bare .result() call, or split CancelledError into its own handler."
        )


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


class TestStartOffsetAnnounce:
    """The "Starting song at Xs" notice for a `?t=` link. Sent by YTDL.yt_stream at
    construction until this branch, which announced under the wrong song."""

    async def test_wording(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        live_song.start_offset = 90
        await music_player._announce_start_offset(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert "Starting song at 90 seconds" in embed.description

    async def test_send_failure_swallowed(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        live_song.start_offset = 90
        mock_channel.send.side_effect = RuntimeError("channel gone")
        await music_player._announce_start_offset(live_song)  # must not raise

    async def test_it_is_clean_with_debug_off(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        live_song.start_offset = 90
        await music_player._announce_start_offset(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert embed.footer.text is None

    async def test_a_crash_recovered_song_takes_this_arm_not_the_resume_one(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Pre-existing, pinned as-is rather than fixed here.

        from_crashed_state stamps `ts` and leaves is_resume False, so a song
        recovered at 2:17 announces "Starting song at 137 seconds". main does the
        same via the old yt_stream guard. See the FIXME in guild_state.py.
        """
        entry = SongQueueEntry.from_crashed_state(
            GuildStateData(
                current_song_url="https://yt.com/v=crash",
                current_song_title="Interrupted",
                current_song_duration=210,
            ),
            position=137,
        )
        assert entry is not None
        assert entry.is_resume is False  # the stamp that decides the arm
        mock_song.is_resume = entry.is_resume
        mock_song.start_offset = entry.ts
        vc = TestInterjectLoopStart()._vc()
        _, announce_mock, offset_mock = await TestInterjectLoopStart()._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        offset_mock.assert_awaited_once_with(mock_song)
        announce_mock.assert_not_awaited()

    async def test_it_is_decorated_in_debug_mode(
        self, music_player: MusicPlayer, live_song: MagicMock, mock_channel: MagicMock
    ) -> None:
        live_song.start_offset = 90
        TestPlayerDebugDecoration._enable(music_player)
        await music_player._announce_start_offset(live_song)
        embed = mock_channel.send.call_args.kwargs["embed"]
        assert "🐞" in (embed.footer.text or "")


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
        before = _fmt_finish_time(110, music_player.timezone)
        embed = music_player._build_now_playing_embed(live_song)
        after = _fmt_finish_time(110, music_player.timezone)
        assert (before in described(embed)) or (after in described(embed))

    def test_position_override_shrinks_remaining(
        self, music_player: MusicPlayer, live_song: MagicMock
    ) -> None:
        from src.musicplayer import _fmt_finish_time

        before = _fmt_finish_time(10, music_player.timezone)
        embed = music_player._build_now_playing_embed(
            live_song, position_override=200.0
        )
        after = _fmt_finish_time(10, music_player.timezone)
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

        seed_queue(music_player.queue, queue_obj)

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

    async def test_matching_marker_stamps_the_tail_with_the_finished_host(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The late-bound half of the NP-card cleanup: the tail learns which
        message froze this fragment's bar, at the one moment that is settled."""
        host = AsyncMock(spec=discord.Message)
        host.id = 777777777777777777
        host.channel.id = 888888888888888888
        music_player._np_host_message = host
        music_player._np_host_own_embeds = []
        music_player._np_host_dedicated = True

        tail = QueueObject("https://yt.com/v=t", "Tail", MagicMock(), is_resume=True)
        music_player._skip_history_for = mock_song
        music_player._pending_resume_tail = tail

        with patch.object(MusicPlayer, "_fire_finalize_now_playing", new=MagicMock()):
            await self._run_one_song(music_player, queue_obj, mock_song)

        assert (tail.np_message_id, tail.np_channel_id) == (
            777777777777777777,
            888888888888888888,
        )
        assert tail.np_dedicated is True
        assert tail.np_host_ref is not None and tail.np_host_ref.message is host
        assert music_player._pending_resume_tail is None

    async def test_a_stale_tail_is_not_stamped_by_a_later_fragment(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Identity mismatch — the marker names a song that ended during
        interject()'s awaits. Stamping anyway would point this tail at an unrelated
        song's card and delete it."""
        host = AsyncMock(spec=discord.Message)
        host.id = 777777777777777777
        host.channel.id = 888888888888888888
        music_player._np_host_message = host
        music_player._np_host_dedicated = True

        tail = QueueObject("https://yt.com/v=t", "Tail", MagicMock(), is_resume=True)
        music_player._skip_history_for = MagicMock()  # some other, ended song
        music_player._pending_resume_tail = tail

        with patch.object(MusicPlayer, "_fire_finalize_now_playing", new=MagicMock()):
            await self._run_one_song(music_player, queue_obj, mock_song)

        assert (tail.np_message_id, tail.np_channel_id) == (0, 0)
        assert tail.np_host_ref is None
        assert music_player._pending_resume_tail is None  # cleared either way

    async def test_a_song_is_claimable_across_the_post_song_prefetch_await(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The loop nulls current_song BEFORE `await prefetch_task`, which can sit
        seconds on a cold extraction, and only writes history after it. A teardown
        landing in that window used to claim nothing and then cancel the loop past
        its own write site — losing the song outright. A song ending is also when
        the last listener leaves, so the alone-disconnect aims right at it."""
        claimed: list[Any] = []
        mock_song.produced_audio = True

        async def claim_during_prefetch(_self: Any) -> None:
            # Exactly where the teardown lands: current_song is already None.
            assert music_player.current_song is None
            claimed.append(music_player.claim_current_song_for_history())

        # Not _run_one_song: it installs its own _prefetch_next_song, which would
        # win over an outer patch and never reach this window.
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
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
            patch.object(MusicPlayer, "_prefetch_next_song", new=claim_during_prefetch),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        assert len(claimed) == 1 and claimed[0] is not None
        assert claimed[0].webpage_url == mock_song.webpage_url
        # And the loop does not then record it a second time: the claim took the
        # marker, which the loop reads AFTER the prefetch await.
        assert len(music_player.history) == 0

    async def test_a_failed_iteration_releases_the_pending_tail(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The error-path twin of the iteration-end clear. A tail left holding the
        slot receives a LATER fragment's card ids and deletes the wrong message —
        so the exception handler has to release it, and until now nothing asserted
        that it did."""
        tail = QueueObject("https://yt.com/v=t", "Tail", MagicMock(), is_resume=True)
        music_player._pending_resume_tail = tail
        music_player._skip_history_for = mock_song

        # Raise from inside the iteration, BEFORE the iteration-end clear.
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
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
            patch.object(
                MusicPlayer,
                "_send_now_playing",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
        ):
            await music_player.loop()

        assert music_player._pending_resume_tail is None
        assert (tail.np_message_id, tail.np_channel_id) == (0, 0)

    async def test_one_marker_suffices_at_depth(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
        queue_obj: QueueObject,
        mock_song: MagicMock,
    ) -> None:
        """Stacking three deep does not need three markers.

        Each interjection stops exactly ONE song, and that song's loop iteration
        consumes the marker before the next interjection can finish resolving — so the
        single slot never has to hold two identities at once. What makes it safe is
        that it holds an identity rather than a flag: the marker left by the last
        interjection must suppress only the song it named."""
        markers: list[object] = []

        async def record_marker() -> None:
            markers.append(music_player._skip_history_for)

        for n in range(1, 4):
            live_song.elapsed_secs = float(30 * n)
            music_player.current_song = live_song
            # The loop's vc.play() between rounds: the previous stop is over.
            music_player._stopped_deliberately = False
            await music_player.interject(
                QueueObject(f"https://yt.com/v=cut{n}", f"Cut {n}", mock_author),
                mock_vc,
            )
            await record_marker()
            # The loop plays the song that cut in, clearing the marker as it goes.
            music_player._skip_history_for = None

        # Every round set the marker to the song it actually stopped.
        assert markers == [live_song, live_song, live_song]

        # And a marker naming a DIFFERENT song never suppresses this one's entry,
        # which is what stops a deep stack from eating an unrelated record.
        music_player._skip_history_for = MagicMock()
        music_player.current_song = None
        await self._run_one_song(music_player, queue_obj, mock_song)
        assert len(music_player.history) == 1


class TestDisposePreviousNpCard:
    """Cleanup of the card an interrupted fragment left frozen. Without it a
    An interjection stack accumulates one dead partial bar each: song end
    RELEASES the host rather than retiring it, by design."""

    def _song(self, **attrs: Any) -> MagicMock:
        song = MagicMock()
        song.np_message_id = 0
        song.np_channel_id = 0
        song.np_dedicated = False
        song.np_host_ref = None
        for name, value in attrs.items():
            setattr(song, name, value)
        return song

    async def test_dedicated_ref_is_deleted(self, music_player: MusicPlayer) -> None:
        message = AsyncMock(spec=discord.Message)
        song = self._song(np_host_ref=NpHostRef(message, [], True))

        await music_player._dispose_previous_np_card(song)

        message.delete.assert_awaited_once()
        message.edit.assert_not_awaited()

    async def test_a_channel_this_guild_does_not_own_is_never_touched(
        self, music_player: MusicPlayer
    ) -> None:
        """The ids are wire values up to the queue key's 24h TTL old, and a
        PartialMessageable validates nothing — so without scoping, a stale or
        corrupted channel id issues a DELETE wherever it resolves, including
        another guild or a DM."""
        mocked(music_player._guild).get_channel_or_thread = MagicMock(return_value=None)
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated=True,
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
        )

        await music_player._dispose_previous_np_card(song)

        music_player.bot.get_partial_messageable.assert_not_called()

    async def test_a_bool_id_never_reaches_the_route(
        self, music_player: MusicPlayer
    ) -> None:
        # isinstance(True, int) is True in Python, so a wire `true` would render
        # "True" into the REST path. The isinstance check alone did not catch it.
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated=True, np_message_id=True, np_channel_id=888888888888888888
        )

        await music_player._dispose_previous_np_card(song)

        music_player.bot.get_partial_messageable.assert_not_called()

    async def test_a_half_stamped_pair_is_never_issued(
        self, music_player: MusicPlayer
    ) -> None:
        # Both ids come off one message, so a zero on either side means the pair
        # never identified anything — get_partial_message(0) would 404 at best.
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated=True, np_message_id=777777777777777777, np_channel_id=0
        )

        await music_player._dispose_previous_np_card(song)

        music_player.bot.get_partial_messageable.assert_not_called()

    async def test_forbidden_and_http_errors_are_swallowed(
        self, music_player: MusicPlayer
    ) -> None:
        # Fire-and-forget: a permission change after the card was posted is the
        # ordinary case and must not surface as an unretrieved task exception.
        for exc in (
            discord.Forbidden(MagicMock(status=403), "nope"),
            discord.HTTPException(MagicMock(status=500), "boom"),
            TimeoutError("aiohttp gave up"),
        ):
            message = AsyncMock(spec=discord.Message)
            message.delete = AsyncMock(side_effect=exc)
            song = self._song(np_host_ref=None, np_dedicated=True)
            song.np_message_id = 777777777777777777
            song.np_channel_id = 888888888888888888
            music_player.bot.get_partial_messageable = MagicMock(
                return_value=MagicMock(
                    get_partial_message=MagicMock(return_value=message)
                )
            )

            await music_player._dispose_previous_np_card(song)  # must not raise

    async def test_a_truthy_non_bool_dedicated_flag_never_authorizes_a_delete(
        self, music_player: MusicPlayer
    ) -> None:
        """np_dedicated is the AUTHORIZATION, not a target — the only thing
        between deleting the bot's own card and deleting a user's command reply.
        parse_queue_entry coerces nothing, so a wire "false" arrives as a truthy
        string; truthiness would read that as permission to delete."""
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated="false",  # truthy string, e.g. a "1"/"0" writer
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
        )

        await music_player._dispose_previous_np_card(song)

        music_player.bot.get_partial_messageable.assert_not_called()

    async def test_response_ref_is_strip_edited_back_to_its_own_embeds(
        self, music_player: MusicPlayer
    ) -> None:
        """A card hosted by a command response must NOT be deleted — that would
        destroy a user's reply. Only the live ref can do this: own_embeds cannot be
        reconstructed from ids, which is why the by-id path skips non-dedicated."""
        message = AsyncMock(spec=discord.Message)
        own = [discord.Embed(title="the reply's own embed")]
        song = self._song(np_host_ref=NpHostRef(message, own, False))

        await music_player._dispose_previous_np_card(song)

        message.edit.assert_awaited_once_with(embeds=own)
        message.delete.assert_not_awaited()

    async def test_wire_ids_delete_a_dedicated_card_by_id(
        self, music_player: MusicPlayer
    ) -> None:
        """The post-restart path: the ref is gone, the ids survived. No fetch and
        no cache lookup — a partial message issues the DELETE directly."""
        partial = MagicMock()
        partial.delete = AsyncMock()
        messageable = MagicMock()
        messageable.get_partial_message = MagicMock(return_value=partial)
        music_player.bot.get_partial_messageable = MagicMock(return_value=messageable)
        song = self._song(
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
            np_dedicated=True,
        )

        await music_player._dispose_previous_np_card(song)

        # guild_id scopes the route: the ids are wire values up to 24h stale and
        # a PartialMessageable validates nothing on its own.
        mocked(music_player.bot.get_partial_messageable).assert_called_once_with(
            888888888888888888, guild_id=music_player._guild.id
        )
        messageable.get_partial_message.assert_called_once_with(777777777777777777)
        partial.delete.assert_awaited_once()

    async def test_by_id_delete_swallows_not_found(
        self, music_player: MusicPlayer
    ) -> None:
        partial = MagicMock()
        partial.delete = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        messageable = MagicMock()
        messageable.get_partial_message = MagicMock(return_value=partial)
        music_player.bot.get_partial_messageable = MagicMock(return_value=messageable)
        song = self._song(np_message_id=7, np_channel_id=8, np_dedicated=True)

        await music_player._dispose_previous_np_card(song)  # must not raise

        partial.delete.assert_awaited_once()

    async def test_wire_ids_alone_never_touch_a_response_host(
        self, music_player: MusicPlayer
    ) -> None:
        """Deliberately a no-op: a by-id DELETE of a non-dedicated host destroys a
        user's command reply, and a strip-edit needs embeds the ids cannot supply.
        A frozen block left on a response after a crash is accepted noise."""
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(np_message_id=7, np_channel_id=8, np_dedicated=False)

        await music_player._dispose_previous_np_card(song)

        mocked(music_player.bot.get_partial_messageable).assert_not_called()

    async def test_unstamped_song_is_a_noop(self, music_player: MusicPlayer) -> None:
        music_player.bot.get_partial_messageable = MagicMock()
        await music_player._dispose_previous_np_card(self._song())
        mocked(music_player.bot.get_partial_messageable).assert_not_called()

    async def test_a_non_integer_id_never_reaches_the_delete(
        self, music_player: MusicPlayer
    ) -> None:
        """parse_queue_entry coerces nothing, so a corrupt entry can carry a
        non-id here. The guard is at the destructive call rather than in the
        parser — dropping the whole song over a cosmetic field would be worse."""
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_message_id={"nested": "object"}, np_channel_id=8, np_dedicated=True
        )

        await music_player._dispose_previous_np_card(song)

        mocked(music_player.bot.get_partial_messageable).assert_not_called()

    async def test_the_ref_wins_over_the_ids(self, music_player: MusicPlayer) -> None:
        # Both present (no crash, ids stamped anyway): the ref path is strictly
        # better — it can strip-edit — and doing both would double-retire.
        message = AsyncMock(spec=discord.Message)
        music_player.bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_host_ref=NpHostRef(message, [], True),
            np_message_id=7,
            np_channel_id=8,
            np_dedicated=True,
        )

        await music_player._dispose_previous_np_card(song)

        message.delete.assert_awaited_once()
        mocked(music_player.bot.get_partial_messageable).assert_not_called()


class TestInterjectPostNeutralizeRecheck:
    async def test_song_changed_during_neutralize_returns_none(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
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
            outcome = await music_player.interject(interject_obj, mock_vc)

        assert outcome is None
        assert music_player.queue.display_items() == []  # nothing inserted
        mock_vc.stop.assert_not_called()
        assert music_player._skip_history_for is None

    async def test_bailing_leaves_an_existing_stack_untouched(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """The same bail with tails already parked. It must insert nothing and
        disturb nothing: the parked plays belong to interjections that already
        completed, and dropping one loses a song a listener was promised back."""
        parked = [
            QueueObject(
                f"https://yt.com/v={n}",
                f"Parked {n}",
                mock_author,
                ts=30 * n,
                is_resume=True,
                played_at=1752530000.0 + n,
            )
            for n in (2, 1)
        ]
        await music_player.queue.put(parked)
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        async def neutralize_and_advance(_self: Any) -> None:
            music_player.current_song = MagicMock()

        with patch.object(
            MusicPlayer, "_neutralize_prefetch", new=neutralize_and_advance
        ):
            outcome = await music_player.interject(interject_obj, mock_vc)

        assert outcome is None
        assert music_player.queue.display_items() == parked
        assert music_player.queue.resume_tail_depth() == 1  # head is a live tail
        mock_vc.stop.assert_not_called()


class TestInterjectStoppedSong:
    """Between note_deliberate_stop() + vc.stop() and the loop's next vc.play(), the
    stopped song is still current_song. Two interjections landing in that window
    used to park two resume tails for one play."""

    async def test_a_stopped_song_is_not_interjected_twice(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        first = await music_player.interject(interject_obj, mock_vc)
        assert first is not None
        assert music_player._stopped_deliberately  # current is stopped, not replaced
        parked = music_player.queue.display_items()

        second = QueueObject("https://yt.com/v=2", "Second", mock_author)
        outcome = await music_player.interject(second, mock_vc)

        assert outcome is None
        assert music_player.queue.display_items() == parked  # no second tail
        assert mock_vc.stop.call_count == 1

    async def test_a_now_during_a_skip_front_inserts(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        interject_obj: QueueObject,
        mock_vc: MagicMock,
    ) -> None:
        """-skip sets the same flag; an interjection landing mid-skip has nothing
        to park and falls back to the caller's front insert."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        music_player.note_deliberate_stop()

        outcome = await music_player.interject(interject_obj, mock_vc)

        assert outcome is None
        assert music_player.queue.display_items() == []
        mock_vc.stop.assert_not_called()
        assert music_player._skip_history_for is None


class TestInterjectLoopStart:
    """Loop-level behavior for interjected entries at song start (review gap):
    start_paused parks the player, is_resume announces from the start path."""

    async def _run_one_song(
        self,
        music_player: MusicPlayer,
        queue_obj: QueueObject,
        mock_song: MagicMock,
        vc: discord.VoiceClient,
    ) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()

        seed_queue(music_player.queue, queue_obj)

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
            patch.object(
                MusicPlayer, "_announce_start_offset", new=AsyncMock()
            ) as offset_mock,
        ):
            await music_player.loop()
        return pause_mock, announce_mock, offset_mock

    def _vc(self) -> discord.VoiceClient:
        """A VoiceClient whose play/pause are mocks, built without __init__.

        A real instance, not MagicMock(spec=...): any other attribute the loop
        touches must fail loudly rather than hand back a truthy mock that quietly
        steers it elsewhere. Read the mocks back with _mock_call(vc, "pause")."""
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
        pause_mock, _, _ = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        # Synchronous park right after vc.play (frame-leak guard) …
        self._mock_call(vc, "pause").assert_called_once()
        # … plus the full pause() entry point (Redis epochs, debounced refresh).
        pause_mock.assert_awaited_once()

    async def test_resume_entry_announced_at_start(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        mock_song.is_resume = True
        mock_song.start_offset = 42  # a resume entry always carries one
        vc = self._vc()
        _, announce_mock, offset_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        announce_mock.assert_awaited_once_with(mock_song)
        # The two notices are exclusive: a resume already says where it resumed.
        offset_mock.assert_not_awaited()

    async def test_a_resume_disposes_of_the_previous_card_after_its_own_is_up(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The new bar must be in the channel before the old one goes, or the
        channel is momentarily without a Now Playing card."""
        order: list[str] = []
        mock_song.is_resume = True
        mock_song.start_offset = 42
        mock_song.np_host_ref = NpHostRef(AsyncMock(spec=discord.Message), [], True)

        async def track_send(self_inner: Any, _song: object) -> None:
            # The sleep is what makes the ordering OBSERVABLE. _spawn_background
            # cannot run its task until the loop yields, so with a send that never
            # awaits, a dispose spawned FIRST still records second and the
            # assertion below holds for the broken order too.
            await asyncio.sleep(0)
            order.append("send_now_playing")
            self_inner._np_host_message = AsyncMock(spec=discord.Message)

        async def track_dispose(_self: Any, _song: object) -> None:
            order.append("dispose")

        # Not _run_one_song: it installs its own _send_now_playing, which would
        # win over an outer patch and leave the ordering unobservable.
        music_player._restore_complete.set()
        music_player.bot.wait_until_ready = AsyncMock()
        mocked(music_player.bot.is_closed).side_effect = [False, True]
        music_player.bot.loop = asyncio.get_running_loop()
        seed_queue(music_player.queue, queue_obj)
        mocked(music_player._guild).voice_client = self._vc()
        music_player.play_next.wait = AsyncMock()

        with (
            patch.object(
                MusicPlayer, "_resolve_source", new=AsyncMock(return_value=queue_obj)
            ),
            patch.object(
                MusicPlayer, "_stream_source", new=AsyncMock(return_value=mock_song)
            ),
            patch.object(MusicPlayer, "_send_now_playing", new=track_send),
            patch.object(MusicPlayer, "_dispose_previous_np_card", new=track_dispose),
            patch.object(
                MusicPlayer, "_prefetch_next_song", new=AsyncMock(return_value=None)
            ),
            patch.object(MusicPlayer, "update_activity", new=AsyncMock()),
            patch.object(MusicPlayer, "_announce_resume", new=AsyncMock()),
        ):
            await music_player.loop()

        assert order == ["send_now_playing", "dispose"]

    async def test_a_card_that_never_went_up_disposes_of_nothing(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """ "After the new card is up" has to mean it IS up. _send_now_playing
        releases the host before it sends and swallows every failure, so a 403 or
        a rate-limit leaves none — and disposing there deletes the only bar in the
        channel for a song that is playing, which is the inverse of the invariant
        the ordering exists for."""
        mock_song.is_resume = True
        mock_song.start_offset = 42
        dispose = AsyncMock()

        async def failed_send(self_inner: Any, _song: object) -> None:
            self_inner._np_host_message = None  # released, send raised, swallowed

        with (
            patch.object(MusicPlayer, "_dispose_previous_np_card", new=dispose),
            patch.object(MusicPlayer, "_send_now_playing", new=failed_send),
            patch.object(MusicPlayer, "_announce_resume", new=AsyncMock()),
        ):
            await self._run_one_song(music_player, queue_obj, mock_song, self._vc())

        dispose.assert_not_awaited()

    async def test_a_plain_song_disposes_of_nothing(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        # Only a resume tail inherits a card to clean up. Firing for every song
        # would delete the bar of the song that just ended.
        mock_song.is_resume = False
        mock_song.start_offset = 0
        dispose = AsyncMock()
        vc = self._vc()
        with patch.object(MusicPlayer, "_dispose_previous_np_card", new=dispose):
            await self._run_one_song(music_player, queue_obj, mock_song, vc)
        dispose.assert_not_awaited()

    async def test_a_tail_of_a_tail_plays_from_its_absolute_position(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The second fragment of an already-resumed song, as a 2-deep stack
        produces. `ts` is absolute, so the loop seeks to everything heard so far
        rather than to this fragment's own start — and the play keeps the single
        start it was first stamped with, so its history row does not move."""
        mock_song.is_resume = True
        mock_song.start_offset = 137  # not 137 minus the previous fragment
        mock_song.elapsed_secs = 20.0
        mock_song.played_at = 1752530000.5
        vc = self._vc()

        _, announce_mock, offset_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )

        announce_mock.assert_awaited_once_with(mock_song)
        offset_mock.assert_not_awaited()
        # position_secs = start_offset + elapsed, so the row spans the whole play.
        assert music_player.history[0].played_secs == 157
        assert music_player.history[0].played_at == 1752530000.5

    async def test_a_start_offset_entry_announces_from_the_start_path(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """Relocated from YTDL.yt_stream, which ran at construction — prefetch builds
        the next song while this one plays."""
        mock_song.is_resume = False
        mock_song.start_offset = 90
        vc = self._vc()
        _, announce_mock, offset_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        offset_mock.assert_awaited_once_with(mock_song)
        announce_mock.assert_not_awaited()

    async def test_a_zero_offset_announces_nothing(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        """The old guard asked whether a ts was present, not whether it was nonzero,
        so `?t=0` rendered "Starting song at 0 seconds"."""
        mock_song.is_resume = False
        mock_song.start_offset = 0
        vc = self._vc()
        _, _, offset_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        offset_mock.assert_not_awaited()

    async def test_plain_song_neither_parks_nor_announces(
        self, music_player: MusicPlayer, queue_obj: QueueObject, mock_song: MagicMock
    ) -> None:
        mock_song.start_offset = 0
        vc = self._vc()
        pause_mock, announce_mock, offset_mock = await self._run_one_song(
            music_player, queue_obj, mock_song, vc
        )
        self._mock_call(vc, "pause").assert_not_called()
        pause_mock.assert_not_awaited()
        announce_mock.assert_not_awaited()
        offset_mock.assert_not_awaited()


# ── HeartbeatUpdater ──────────────────────────────────────────────────────────


class TestHeartbeatUpdater:
    """Records the playback position so recovery never infers it from a clock."""

    @staticmethod
    def _make_sleep(n_ticks: int) -> Callable[[Any], Awaitable[None]]:
        calls = 0

        async def _sleep(_secs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls > n_ticks:
                raise asyncio.CancelledError()

        return _sleep

    async def test_writes_the_current_position(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        # position_secs is derived (start_offset + elapsed_secs) on the fixture,
        # mirroring the real YTDL property — set the inputs, not the result.
        mock_song.start_offset = 10
        mock_song.elapsed_secs = 32.5
        store = AsyncMock(spec=GuildRedisStore)
        music_player.store = store

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._heartbeat_updater(mock_song)

        store.heartbeat.assert_awaited_once()
        assert store.heartbeat.await_args.args[0] == 42.5

    async def test_skips_while_paused(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Frames are frozen, so the position is not moving and pause() already
        recorded the exact point — writing would rewrite the same value forever."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = True
        mocked(music_player._guild).voice_client = vc
        store = AsyncMock(spec=GuildRedisStore)
        music_player.store = store

        with patch("asyncio.sleep", new=self._make_sleep(2)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._heartbeat_updater(mock_song)

        store.heartbeat.assert_not_awaited()

    async def test_returns_when_the_song_changes(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = MagicMock()  # a different song is playing now
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        store = AsyncMock(spec=GuildRedisStore)
        music_player.store = store

        with patch("asyncio.sleep", new=self._make_sleep(3)):
            await music_player._heartbeat_updater(mock_song)  # returns, no raise

        store.heartbeat.assert_not_awaited()

    async def test_runs_without_a_now_playing_host_message(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """The reason this is a separate task from _progress_updater: that one
        goes dormant when the host message is gone. A song whose bar a user
        deleted must still be recoverable."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        music_player._np_host_message = None
        store = AsyncMock(spec=GuildRedisStore)
        music_player.store = store

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._heartbeat_updater(mock_song)

        store.heartbeat.assert_awaited_once()

    async def test_an_unexpected_error_stops_the_ticker_at_error_level(
        self, music_player: MusicPlayer, mock_song: MagicMock, caplog: Any
    ) -> None:
        """@_guild_op already swallows Redis failures, so anything reaching here is a
        defect that recurs every tick. It must not die silently: cancel_task never
        awaits a task that ended on its own, so the traceback would surface at GC with
        no guild attached while recovery quietly fell back to the seeded position."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        store = AsyncMock(spec=GuildRedisStore)
        store.heartbeat.side_effect = AttributeError("boom")
        music_player.store = store

        with patch("asyncio.sleep", new=self._make_sleep(5)):
            with caplog.at_level(logging.ERROR):
                await music_player._heartbeat_updater(mock_song)

        # Returned rather than raising, and wrote exactly once before giving up.
        assert store.heartbeat.await_count == 1
        assert any("playback heartbeat stopped" in r.message for r in caplog.records)

    async def test_tolerates_no_store(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.source = mock_song
        vc.is_paused.return_value = False
        mocked(music_player._guild).voice_client = vc
        music_player.store = None

        with patch("asyncio.sleep", new=self._make_sleep(1)):
            with pytest.raises(asyncio.CancelledError):
                await music_player._heartbeat_updater(mock_song)  # must not raise


class TestNowPlayingEditDiffing:
    """A 3s tick re-renders an identical payload most of the time; the bar only
    changes ~10 times in a 4-minute song. Per *bot*, those PATCHes are the
    dominant REST rate-limit cost."""

    @staticmethod
    def _host() -> AsyncMock:
        message = AsyncMock(spec=discord.Message)
        message.id = 999
        return message

    async def test_identical_rerender_is_not_pushed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = self._host()
        assert await music_player._push_np_edit(mock_song, message, []) is True
        assert message.edit.await_count == 1

        assert await music_player._push_np_edit(mock_song, message, []) is True
        assert message.edit.await_count == 1  # skipped

    async def test_changed_render_is_pushed(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        message = self._host()
        await music_player._push_np_edit(mock_song, message, [])
        mock_song.elapsed_secs = 120.0  # bar advances
        await music_player._push_np_edit(mock_song, message, [])
        assert message.edit.await_count == 2

    async def test_a_different_host_always_gets_its_own_edit(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """A host swap must not inherit the old host's cache — the new message
        has never been written to, so an identical payload is still needed."""
        first = self._host()
        await music_player._push_np_edit(mock_song, first, [])
        second = self._host()
        second.id = 1000
        await music_player._push_np_edit(mock_song, second, [])
        assert second.edit.await_count == 1

    async def test_failed_edit_is_not_cached(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Caching a payload we failed to push would suppress the retry that
        fixes it."""
        message = self._host()
        message.edit.side_effect = discord.HTTPException(MagicMock(), "boom")
        assert await music_player._push_np_edit(mock_song, message, []) is True

        message.edit.side_effect = None
        await music_player._push_np_edit(mock_song, message, [])
        assert message.edit.await_count == 2  # retried, not suppressed

    async def test_releasing_the_host_drops_the_cache(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Retiring can strip-edit the message outside _push_np_edit, so a
        stale cache entry could suppress a genuinely needed edit."""
        message = self._host()
        await music_player._push_np_edit(mock_song, message, [])
        music_player._release_np_host()
        assert music_player._np_last_rendered is None
        await music_player._push_np_edit(mock_song, message, [])
        assert message.edit.await_count == 2

    async def test_own_embeds_changing_forces_an_edit(
        self, music_player: MusicPlayer, mock_song: MagicMock
    ) -> None:
        """Keyed on the rendered payload, not on position — so every reason an
        embed can differ is covered without enumerating them."""
        message = self._host()
        await music_player._push_np_edit(mock_song, message, [])
        extra = discord.Embed(title="a command response")
        await music_player._push_np_edit(mock_song, message, [extra])
        assert message.edit.await_count == 2


class TestVolumeMigratesForwardOnRestore:
    """The first restore after the deploy SEEDS a pre-move volume into config, so
    no migration script is needed and the 24h state TTL stops being able to take the
    setting with it. Seeds, never overwrites — the snapshot it is working from was
    read an arbitrary number of awaits earlier."""

    async def test_a_legacy_volume_is_restored_and_written_forward(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.30")

        await music_player._restore_state()

        assert music_player.volume == 0.30
        config = await fake_redis.hgetall(music_player.store.config_key())
        assert config[b"volume"] == b"0.3"

    async def test_a_muted_guild_is_restored_and_migrated(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        """0.0 is a real choice, not an absence. A truthiness check here brings a
        muted guild back at 100% AND never migrates it, so it is muted-then-loud on
        every restart until the state hash expires and the setting is gone."""
        assert music_player.store is not None
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.0")

        await music_player._restore_state()

        assert music_player.volume == 0.0
        config = await fake_redis.hgetall(music_player.store.config_key())
        assert config[b"volume"] == b"0.0"

    async def test_a_config_volume_is_not_rewritten(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        """Already migrated: restore reads it and leaves it alone rather than
        spending a write on every recovery for the life of the deployment."""
        assert music_player.store is not None
        await music_player.store.set_volume(0.75)
        with patch.object(
            type(music_player.store), "migrate_volume", new=AsyncMock()
        ) as write:
            await music_player._restore_state()
        assert music_player.volume == 0.75
        write.assert_not_awaited()

    async def test_the_forward_write_cannot_clobber_a_concurrent_volume(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        """Restore reads its snapshot, then writes it back after an arbitrary number
        of awaits. A `-volume` landing in that window used to be overwritten by the
        older stored value — durably, while the user was being told the new one took.
        The seed is HSETNX, so the newer value wins.
        """
        assert music_player.store is not None
        await fake_redis.hset(music_player.store.state_key(), b"volume", b"0.30")

        real_snapshot = type(music_player.store).get_playback_snapshot

        async def snapshot_then_user_sets_volume(store: object) -> object:
            snapshot = await real_snapshot(store)  # pyright: ignore[reportArgumentType]
            # The window: the snapshot is taken, and only then does -volume run.
            assert music_player.store is not None
            await music_player.store.set_volume(0.80)
            return snapshot

        with patch.object(
            type(music_player.store),
            "get_playback_snapshot",
            new=snapshot_then_user_sets_volume,
        ):
            await music_player._restore_state()

        config = await fake_redis.hgetall(music_player.store.config_key())
        assert config[b"volume"] == b"0.8"

    async def test_a_guild_that_never_set_one_keeps_the_default(
        self, music_player: MusicPlayer
    ) -> None:
        before = music_player.volume
        await music_player._restore_state()
        assert music_player.volume == before


class TestGuildTimezoneOnRestore:
    """ETAs follow the guild's stored zone instead of a hardcoded Pacific."""

    async def test_a_stored_zone_is_adopted(
        self, music_player: MusicPlayer, fake_redis: aioredis.Redis
    ) -> None:
        assert music_player.store is not None
        await music_player.store.set_timezone("Europe/London")
        await music_player._restore_state()
        assert music_player.timezone == ZoneInfo("Europe/London")

    async def test_a_guild_with_no_zone_keeps_the_default(
        self, music_player: MusicPlayer
    ) -> None:
        await music_player._restore_state()
        assert music_player.timezone == ZoneInfo(DEFAULT_TIMEZONE)

    async def test_an_unusable_stored_zone_degrades_to_the_default(
        self, music_player: MusicPlayer
    ) -> None:
        """Rendering must not raise on a name the host's tz database lost."""
        assert music_player.store is not None
        await music_player.store.set_timezone("Mars/Olympus")
        await music_player._restore_state()
        assert music_player.timezone == ZoneInfo(DEFAULT_TIMEZONE)

    async def test_the_eta_renders_in_the_guilds_zone(
        self, music_player: MusicPlayer
    ) -> None:
        """End to end: the suffix is the zone's own abbreviation, so a London guild
        is never quoted London time labelled PST."""
        music_player.timezone = ZoneInfo("Europe/London")
        rendered = _fmt_finish_time(90, music_player.timezone)
        assert re.match(r"^\d{1,2}:\d{2} (AM|PM) (GMT|BST)$", rendered)


class TestQueueLinesCannotForgeALink:
    """`_format_queue_line` renders the title inside a masked link's LABEL, and
    yt-dlp titles are uploader-chosen. An unbalanced `]` closes the label early
    and re-points the link at whatever the title puts after it — under the bot's
    name, in a message a member only had to get queued to trigger.

    This was the inconsistency the -remove work introduced: its own Songs field
    escaped, and the queue embed it sends 30ms later did not."""

    def test_a_hostile_title_cannot_close_the_label(
        self, music_player: MusicPlayer, mock_author: MagicMock
    ) -> None:
        item = QueueObject(
            "https://yt.com/v=1",
            "Song](https://evil.example) [FREE NITRO",
            mock_author,
            duration=100,
        )
        now, walk = music_player._queue_eta_seed()
        line, _ = music_player._format_queue_line(item, 1, now, walk)
        assert "](https://evil.example)" not in line
        assert "[FREE NITRO" not in line
        # The real destination is still the one the queue holds.
        assert "](https://yt.com/v=1)" in line

    def test_an_unresolved_search_is_sanitized_too(
        self, music_player: MusicPlayer
    ) -> None:
        """The resolving line renders user-typed text straight into a description."""
        item = YTSource(ytsearch="ytsearch:[click](https://evil.example)", process=True)
        now, walk = music_player._queue_eta_seed()
        line, _ = music_player._format_queue_line(item, 1, now, walk)
        assert "[" not in line and "](" not in line
