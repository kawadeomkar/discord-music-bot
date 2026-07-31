"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory display cache and the Redis leg agree on their shared window. The
cache is capped at HISTORY_CACHE_LIMIT; the Redis leg is unbounded (source of
truth for all played songs — docs/HISTORY_OVERHAUL_PLAN.md §4).
"""

import redis.asyncio as aioredis
from typing import Any, cast
from redis.asyncio import Redis
import pytest

from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.redis_client import (
    HISTORY_CACHE_LIMIT,
    HISTORY_OUTBOX_KEY,
    OUTBOX_FIELD,
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
    """A stand-in for the Postgres archive.

    Required by the constructor — the archive is not an optional tier, so
    there is no shape of GuildHistory without one — but nothing here awaits
    it: add() buffers to the Redis outbox and the DRAINER is what reaches
    Postgres, so the write path never touches the archive object at all. The
    double therefore only needs the protocol's shape.
    """

    def __init__(self, entries: list[HistoryEntry] | None = None) -> None:
        self.entries = entries or []

    async def insert_batch(self, entries: Any) -> None:
        self.entries.extend(entries)


def _history(
    store: Any,
    *,
    archive: Any = None,
    guild_id: int = 42,
    on_outbox_push: Any = None,
) -> GuildHistory:
    """GuildHistory with the required archive/notify wiring defaulted.

    Both are mandatory on the real constructor (the Postgres tier is not
    optional), and neither is what most tests here are about — they exercise
    the display legs — so both default to inert stand-ins. Tests that DO care
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
    archive is a required tier, so there is no no-archive shape to gate on."""

    async def test_add_routes_entry_to_outbox_too(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        h = _history(store)
        await h.add(_entry(1))
        # XRANGE, not LRANGE: the outbox is a stream. Asserting the payload
        # under OUTBOX_FIELD rather than just the entry count is what keeps
        # this honest about the wire format the drainer reads back.
        # The cast narrows redis-py's XRANGE union, which is wide enough to
        # cover XAUTOCLAIM's 4-tuple rows and the RESP3 dict form.
        entries = cast(
            list[tuple[bytes, dict[bytes, bytes]]],
            await fake_redis.xrange(HISTORY_OUTBOX_KEY),
        )
        assert [fields[OUTBOX_FIELD] for _id, fields in entries] == [
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
