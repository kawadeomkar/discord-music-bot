"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory display cache and the Redis leg agree on their shared window. The
cache is capped at HISTORY_CACHE_LIMIT; the Redis leg is unbounded (source of
truth for all played songs — docs/HISTORY_OVERHAUL_PLAN.md §4).
"""

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
def store(fake_redis):
    return GuildRedisStore(fake_redis, guild_id=42)


class TestAdd:
    async def test_appends_and_mirrors_to_redis(self, store):
        h = GuildHistory(store)
        await h.add(_entry(1))
        await h.add(_entry(2))
        assert list(h) == [_entry(1), _entry(2)]  # oldest first
        assert await store.get_history() == [_entry(2), _entry(1)]

    async def test_works_without_store(self):
        h = GuildHistory(None)
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]

    async def test_both_cache_legs_capped(self, store, fake_redis):
        # Post-cutover both legs are display caches — retention is Postgres's
        # job (docs/POSTGRES_HISTORY_PLAN.md §5.3).
        h = GuildHistory(store)
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await h.add(_entry(i))
        assert len(h) == HISTORY_CACHE_LIMIT
        assert h[0] == _entry(5)  # oldest cached = first evicted survivor
        raw = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(raw) == HISTORY_CACHE_LIMIT

    async def test_cache_matches_newest_slice_of_redis(self, store):
        h = GuildHistory(store)
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await h.add(_entry(i))
        mirrored = await store.get_history()  # newest-first, bounded read
        assert list(h) == list(reversed(mirrored))


class TestAddOutboxGating:
    """Postgres archive wiring (docs/POSTGRES_HISTORY_PLAN.md §5.4): the
    outbox push and drainer notify happen exactly when an archive is
    configured. Write path only — recent() is untouched until the Phase C
    read flip."""

    async def test_no_archive_means_no_outbox(self, store, fake_redis):
        h = GuildHistory(store)
        await h.add(_entry(1))
        assert await fake_redis.exists(HISTORY_OUTBOX_KEY) == 0

    async def test_archive_routes_entry_to_outbox_too(self, store, fake_redis):
        h = GuildHistory(store, archive=object(), guild_id=42)
        await h.add(_entry(1))
        assert await fake_redis.lrange(HISTORY_OUTBOX_KEY, 0, -1) == [
            _entry(1).to_redis()
        ]
        # Display legs behave exactly as without an archive.
        assert list(h) == [_entry(1)]
        assert await store.get_history() == [_entry(1)]

    async def test_notify_fires_once_per_add_with_archive(self, store):
        calls = []
        h = GuildHistory(
            store, archive=object(), guild_id=42, on_outbox_push=lambda: calls.append(1)
        )
        await h.add(_entry(1))
        await h.add(_entry(2))
        assert len(calls) == 2

    async def test_notify_not_fired_without_archive(self, store):
        calls = []
        h = GuildHistory(store, on_outbox_push=lambda: calls.append(1))
        await h.add(_entry(1))
        assert calls == []

    async def test_no_store_skips_outbox_and_notify(self):
        # Without Redis there is nowhere to buffer — degrade to memory-only
        # exactly as before the archive existed.
        calls = []
        h = GuildHistory(None, archive=object(), on_outbox_push=lambda: calls.append(1))
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]
        assert calls == []


class TestRestore:
    def test_reverses_newest_first_input(self):
        h = GuildHistory(None)
        h.restore([_entry(3), _entry(2), _entry(1)])
        assert list(h) == [_entry(1), _entry(2), _entry(3)]

    def test_restore_respects_cache_limit(self):
        h = GuildHistory(None)
        h.restore([_entry(i) for i in range(HISTORY_CACHE_LIMIT + 10)])
        assert len(h) == HISTORY_CACHE_LIMIT
        assert h[-1] == _entry(0)  # newest entry survives the cap


class TestRecent:
    async def test_newest_first_selection(self):
        h = GuildHistory(None)
        h.restore([_entry(3), _entry(2), _entry(1)])  # newest-first input
        assert await h.recent(2) == [_entry(3), _entry(2)]

    async def test_limit_larger_than_history_returns_all(self):
        h = GuildHistory(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]

    async def test_nonpositive_limit_returns_nothing(self):
        h = GuildHistory(None)
        h.restore([_entry(1)])
        assert await h.recent(0) == []
        assert await h.recent(-1) == []

    async def test_empty_history(self):
        assert await GuildHistory(None).recent(10) == []

    async def test_reads_persisted_when_cache_cold(self, store):
        """After a clean stop+restart the cache is empty but Redis still holds
        history — recent() must surface it from the store."""
        seed = GuildHistory(store)
        for i in range(3):
            await seed.add(_entry(i))
        cold = GuildHistory(store)  # fresh player: empty in-memory cache
        assert len(cold) == 0
        assert await cold.recent(10) == [_entry(2), _entry(1), _entry(0)]

    async def test_falls_back_to_cache_without_store(self):
        h = GuildHistory(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]


class _FakeArchive:
    """recent()-only archive fake: serves a canned newest-first list, records
    calls, raises on demand."""

    def __init__(self, entries=None, fail=False):
        self.entries = entries or []  # newest-first, as the real one returns
        self.fail = fail
        self.calls: list[tuple[int, int]] = []

    async def insert_batch(self, entries):  # protocol completeness
        raise AssertionError("recent()-path tests must not insert")

    async def recent(self, guild_id, limit):
        self.calls.append((guild_id, limit))
        if self.fail:
            raise RuntimeError("pg down")
        return self.entries[:limit]


class TestRecentArchivePrimary:
    """Phase C read flip (docs/POSTGRES_HISTORY_PLAN.md §5.4): the archive is
    authoritative, freshness-merged with the deque; the pre-cutover chain
    survives only as the degraded fallback."""

    async def test_archive_serves_reads_with_guild_id(self, store):
        archive = _FakeArchive(entries=[_entry(3), _entry(2), _entry(1)])
        h = GuildHistory(store, archive=archive, guild_id=42)
        assert await h.recent(2) == [_entry(3), _entry(2)]
        assert archive.calls == [(42, 2)]

    async def test_archive_beats_redis_cache(self, store):
        # The Redis list may be a partial cache post-cutover — a non-empty
        # list must NOT shadow the archive (the original v1-review bug).
        h_seed = GuildHistory(store)
        await h_seed.add(_entry(9))  # Redis list now holds only entry 9
        archive = _FakeArchive(entries=[_entry(3), _entry(2), _entry(1)])
        h = GuildHistory(store, archive=archive, guild_id=42)
        assert await h.recent(10) == [_entry(3), _entry(2), _entry(1)]

    async def test_merge_prepends_undrained_deque_entries(self, store):
        # Drain lag: the song that just ended is in the deque but not yet in
        # PG — it must still show, newest first.
        archive = _FakeArchive(entries=[_entry(2), _entry(1)])
        h = GuildHistory(store, archive=archive, guild_id=42)
        h.restore([_entry(3)])  # deque holds the undrained newest entry
        assert await h.recent(10) == [_entry(3), _entry(2), _entry(1)]

    async def test_merge_dedups_by_identity(self, store):
        archive = _FakeArchive(entries=[_entry(2), _entry(1)])
        h = GuildHistory(store, archive=archive, guild_id=42)
        h.restore([_entry(2), _entry(1)])  # fully drained — all dupes
        assert await h.recent(10) == [_entry(2), _entry(1)]

    async def test_merge_respects_limit(self, store):
        archive = _FakeArchive(entries=[_entry(2), _entry(1)])
        h = GuildHistory(store, archive=archive, guild_id=42)
        h.restore([_entry(3)])
        assert await h.recent(2) == [_entry(3), _entry(2)]

    async def test_empty_archive_serves_deque_via_merge(self, store):
        # Fresh PG + undrained entries: the merge alone carries the answer.
        archive = _FakeArchive(entries=[])
        h = GuildHistory(store, archive=archive, guild_id=42)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]

    async def test_archive_error_falls_back_to_redis(self, store, caplog):
        seed = GuildHistory(store)
        await seed.add(_entry(1))
        await seed.add(_entry(2))
        h = GuildHistory(store, archive=_FakeArchive(fail=True), guild_id=42)
        assert await h.recent(10) == [_entry(2), _entry(1)]
        assert "archive read failed" in caplog.text

    async def test_archive_error_without_store_falls_back_to_deque(self):
        h = GuildHistory(None, archive=_FakeArchive(fail=True), guild_id=42)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]

    async def test_nonpositive_limit_short_circuits_archive(self, store):
        archive = _FakeArchive(entries=[_entry(1)])
        h = GuildHistory(store, archive=archive, guild_id=42)
        assert await h.recent(0) == []
        assert archive.calls == []


class TestSequenceProtocol:
    def test_len_iter_getitem(self):
        # The -history command and tests read the cache as a plain sequence.
        h = GuildHistory(None)
        h.restore([_entry(2), _entry(1)])
        assert len(h) == 2
        assert h[0] == _entry(1)
        assert list(h) == [_entry(1), _entry(2)]

    def test_empty_is_falsy(self):
        assert not GuildHistory(None)
