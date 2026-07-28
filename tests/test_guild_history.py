"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory display cache and the Redis leg agree on their shared window. The
cache is capped at HISTORY_CACHE_LIMIT; the Redis leg is unbounded (source of
truth for all played songs — docs/HISTORY_OVERHAUL_PLAN.md §4).
"""

import asyncio

import redis.asyncio as aioredis
from typing import Any, cast
from redis.asyncio import Redis
import pytest

from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.redis_client import (
    HISTORY_CACHE_LIMIT,
    HISTORY_OUTBOX_KEY,
    GuildRedisStore,
)


def _entry(n: int) -> HistoryEntry:
    return HistoryEntry(
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=200,
        requester_id=n,
        requester_name=f"user{n}",
        played_at=1000.0 + n,
    )


@pytest.fixture
def store(fake_redis: aioredis.Redis) -> GuildRedisStore:
    return GuildRedisStore(fake_redis, guild_id=42)


class StubArchive:
    """An archive holding exactly what it is told to hold.

    The default stand-in used to be a bare `object()`, which was fine while
    the archive was write-only. recent() reads it now (Phase B), so the
    default has to be something with the protocol's shape — an EMPTY one, so
    every test written against the Redis/cache tiers keeps exercising those
    tiers by falling through.
    """

    def __init__(self, entries: list[HistoryEntry] | None = None) -> None:
        self.entries = entries or []
        self.calls: list[tuple[int, int]] = []

    async def insert_batch(self, entries: Any) -> None:
        self.entries.extend(entries)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        self.calls.append((guild_id, limit))
        return self.entries[:limit]


def _history(
    store: Any,
    *,
    archive: Any = None,
    guild_id: int = 42,
    on_outbox_push: Any = None,
) -> GuildHistory:
    """GuildHistory with the required archive/notify wiring defaulted.

    Both are mandatory on the real constructor (the Postgres tier is not
    optional). Most tests here exercise the display legs, so the default
    archive is an empty StubArchive: recent() consults it, finds nothing, and
    falls through to exactly the tier the test is about. Tests that DO care
    pass their own."""
    return GuildHistory(
        store,
        archive=cast(Any, archive if archive is not None else StubArchive()),
        guild_id=guild_id,
        on_outbox_push=on_outbox_push if on_outbox_push is not None else (lambda: None),
    )


class TestAdd:
    async def test_appends_and_mirrors_to_redis(self, store: GuildRedisStore) -> None:
        h = _history(store)
        await h.add(_entry(1))
        await h.add(_entry(2))
        assert list(h) == [_entry(1), _entry(2)]  # oldest first
        assert await store.get_history() == [_entry(2), _entry(1)]

    async def test_works_without_store(self) -> None:
        h = _history(None)
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]

    async def test_cache_capped_redis_leg_unbounded(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        h = _history(store)
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await h.add(_entry(i))
        # The cache holds only the newest window…
        assert len(h) == HISTORY_CACHE_LIMIT
        assert h[0] == _entry(5)  # oldest cached = first evicted survivor
        # …while every entry landed in Redis.
        raw = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(raw) == HISTORY_CACHE_LIMIT + 5

    async def test_cache_matches_newest_slice_of_redis(
        self, store: GuildRedisStore
    ) -> None:
        h = _history(store)
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await h.add(_entry(i))
        mirrored = await store.get_history()  # newest-first, bounded read
        assert list(h) == list(reversed(mirrored))


class TestAddOutboxRouting:
    """Postgres archive wiring (docs/POSTGRES_HISTORY_PLAN.md §5.4): every
    add() pushes to the outbox and nudges the drainer. Unconditional — the
    archive is a required tier, so there is no no-archive shape to gate on.
    Write path only — the read path is TestRecentReadsPostgresFirst."""

    async def test_add_routes_entry_to_outbox_too(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        h = _history(store)
        await h.add(_entry(1))
        assert await fake_redis.lrange(HISTORY_OUTBOX_KEY, 0, -1) == [
            _entry(1).to_redis()
        ]
        # The display legs carry the same entry, from the same pipeline.
        assert list(h) == [_entry(1)]
        assert await store.get_history() == [_entry(1)]

    async def test_notify_fires_once_per_add(self, store: GuildRedisStore) -> None:
        calls = []
        h = _history(store, on_outbox_push=lambda: calls.append(1))
        await h.add(_entry(1))
        await h.add(_entry(2))
        assert len(calls) == 2

    async def test_no_store_skips_outbox_and_notify(self) -> None:
        # Redis optionality is a SEPARATE axis from the archive: without a
        # store there is nowhere to buffer, so add() degrades to memory-only
        # and must not nudge a drainer that has nothing to drain.
        calls = []
        h = _history(None, on_outbox_push=lambda: calls.append(1))
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]
        assert calls == []


class TestRestore:
    def test_reverses_newest_first_input(self) -> None:
        h = _history(None)
        h.restore([_entry(3), _entry(2), _entry(1)])
        assert list(h) == [_entry(1), _entry(2), _entry(3)]

    def test_restore_respects_cache_limit(self) -> None:
        h = _history(None)
        h.restore([_entry(i) for i in range(HISTORY_CACHE_LIMIT + 10)])
        assert len(h) == HISTORY_CACHE_LIMIT
        assert h[-1] == _entry(0)  # newest entry survives the cap


class TestRecent:
    async def test_newest_first_selection(self) -> None:
        h = _history(None)
        h.restore([_entry(3), _entry(2), _entry(1)])  # newest-first input
        assert await h.recent(2) == [_entry(3), _entry(2)]

    async def test_limit_larger_than_history_returns_all(self) -> None:
        h = _history(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]

    async def test_nonpositive_limit_returns_nothing(self) -> None:
        h = _history(None)
        h.restore([_entry(1)])
        assert await h.recent(0) == []
        assert await h.recent(-1) == []

    async def test_empty_history(self) -> None:
        assert await _history(None).recent(10) == []

    async def test_reads_persisted_when_cache_cold(
        self, store: GuildRedisStore
    ) -> None:
        """After a clean stop+restart the cache is empty but Redis still holds
        history — recent() must surface it from the store."""
        seed = _history(store)
        for i in range(3):
            await seed.add(_entry(i))
        cold = _history(store)  # fresh player: empty in-memory cache
        assert len(cold) == 0
        assert await cold.recent(10) == [_entry(2), _entry(1), _entry(0)]

    async def test_falls_back_to_cache_without_store(self) -> None:
        h = _history(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]


class TestRecentReadsPostgresFirst:
    """Phase B. Postgres holds every play, including ones no Redis window can
    still show — so it has to be the first tier asked, and every failure has to
    fall THROUGH rather than surface, because -history is a display command."""

    async def test_a_full_archive_result_short_circuits_redis(
        self, store: GuildRedisStore
    ) -> None:
        # >= limit rows means the archive can answer on its own; Redis is not
        # read at all. This is the steady state after the backfill.
        h = _history(store, archive=StubArchive([_entry(9), _entry(8)]))
        await h.add(_entry(1))
        assert await h.recent(2) == [_entry(9), _entry(8)]

    async def test_short_archive_result_is_merged_with_redis(
        self, store: GuildRedisStore
    ) -> None:
        """REGRESSION: a non-empty-but-SHORT archive result used to be returned
        as-is, so -history went silently shallow for the whole window between
        deploying the archive and finishing the backfill — a guild with 50
        pre-deploy plays showed 1 as soon as one song drained. The window is
        structural (the backfill cannot run before the schema and outbox exist),
        so it hit every guild on every deploy. It also covers the steady state,
        where the newest plays are still in the outbox and only Redis has them.
        """
        h = _history(store, archive=StubArchive([_entry(9), _entry(8)]))
        await h.add(_entry(1))  # in Redis and the cache, not in the archive
        assert await h.recent(10) == [_entry(9), _entry(8), _entry(1)]

    async def test_merge_dedups_entries_present_in_both(
        self, store: GuildRedisStore
    ) -> None:
        # The overlap is the normal case: an entry is on the Redis list AND
        # archived. Identity is (played_at, webpage_url) — the archive's own
        # dedup key — so it must appear exactly once.
        h = _history(store, archive=StubArchive([_entry(1)]))
        await h.add(_entry(1))
        assert await h.recent(10) == [_entry(1)]

    async def test_merge_still_honours_the_limit(self, store: GuildRedisStore) -> None:
        h = _history(store, archive=StubArchive([_entry(9)]))
        for n in (1, 2, 3):
            await h.add(_entry(n))
        assert await h.recent(2) == [_entry(9), _entry(3)]

    async def test_archive_is_scoped_to_this_guild_and_limit(self) -> None:
        stub = StubArchive([_entry(9)])
        h = _history(None, archive=stub, guild_id=777)
        await h.recent(5)
        assert stub.calls == [(777, 5)]

    async def test_empty_archive_falls_through_to_redis(
        self, store: GuildRedisStore
    ) -> None:
        # A guild whose plays are all still in the outbox has no archive rows
        # yet — rendering "no history" there would be wrong.
        h = _history(store, archive=StubArchive([]))
        await h.add(_entry(1))
        assert await h.recent(10) == [_entry(1)]

    async def test_archive_failure_falls_through_and_warns(
        self, store: GuildRedisStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        class BrokenArchive:
            async def insert_batch(self, entries: Any) -> None: ...

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                raise OSError("pg unreachable")

        h = _history(store, archive=BrokenArchive())
        await h.add(_entry(1))
        assert await h.recent(10) == [_entry(1)]
        assert "fell back to redis" in caplog.text

    async def test_slow_archive_is_bounded_and_falls_through(
        self, store: GuildRedisStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.guild_history as guild_history

        monkeypatch.setattr(guild_history, "_ARCHIVE_READ_TIMEOUT_SECS", 0.02)

        class SlowArchive:
            async def insert_batch(self, entries: Any) -> None: ...

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                await asyncio.Event().wait()
                return []

        h = _history(store, archive=SlowArchive())
        await h.add(_entry(1))
        assert await h.recent(10) == [_entry(1)]

    async def test_archive_failure_with_no_store_still_reaches_the_cache(
        self,
    ) -> None:
        class BrokenArchive:
            async def insert_batch(self, entries: Any) -> None: ...

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                raise OSError("pg unreachable")

        h = _history(None, archive=BrokenArchive())
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]


class TestSequenceProtocol:
    def test_len_iter_getitem(self) -> None:
        # The -history command and tests read the cache as a plain sequence.
        h = _history(None)
        h.restore([_entry(2), _entry(1)])
        assert len(h) == 2
        assert h[0] == _entry(1)
        assert list(h) == [_entry(1), _entry(2)]

    def test_empty_is_falsy(self) -> None:
        assert not _history(None)
