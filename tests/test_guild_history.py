"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory display cache and the Redis leg agree on their shared window. The
cache is capped at HISTORY_CACHE_LIMIT; the Redis leg is unbounded (source of
truth for all played songs — docs/HISTORY_OVERHAUL_PLAN.md §4).
"""

import asyncio
from dataclasses import replace as dataclasses_replace

import redis.asyncio as aioredis
from typing import Any, cast
from redis.asyncio import Redis
import pytest

from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.history_archive import quantized_played_at
from src.redis_client import (
    HISTORY_CACHE_LIMIT,
    HISTORY_OUTBOX_KEY,
    GuildRedisStore,
)


def _entry(n: int, played_at: float | None = None) -> HistoryEntry:
    return HistoryEntry(
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=200,
        requester_id=n,
        requester_name=f"user{n}",
        played_at=1000.0 + n if played_at is None else played_at,
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


class QuantizingArchive(StubArchive):
    """StubArchive that truncates played_at the way timestamptz does.

    StubArchive hands back the very objects it was given, so no test using it
    can observe a round trip at all — which is exactly why the µs-quantization
    defect below lived through the whole suite. Anything asserting on the
    Postgres/Redis MERGE has to read through this one instead.
    """

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        rows = await super().recent(guild_id, limit)
        return [
            dataclasses_replace(e, played_at=quantized_played_at(e.played_at))
            for e in rows
        ]


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

    async def test_merge_dedups_across_the_timestamptz_round_trip(
        self, store: GuildRedisStore
    ) -> None:
        """REGRESSION: the dedup key compared a µs-quantized played_at coming
        back from Postgres against the raw time.time() float still on the Redis
        list. Those are unequal for ~37% of real timestamps, so a play that had
        already drained appeared in BOTH legs, survived the dedup and rendered
        twice.

        The fix (quantized_played_at, restored from docs/ARCHITECTURE.md §783
        after the merge rewrite dropped its only two call sites) puts both sides
        through the same truncation. Every other test in this class uses
        whole-second played_at values, which survive quantization exactly — so
        none of them can see this.
        """
        raw = 1785309912.8701255  # a real time.time() shape, finer than 1us
        assert quantized_played_at(raw) != raw, "premise: this value must be lossy"
        entry = _entry(1, played_at=raw)

        h = _history(store, archive=QuantizingArchive([entry]))
        await h.add(entry)  # the same play, now on both legs

        got = await h.recent(10)
        assert [e.webpage_url for e in got] == [entry.webpage_url]

    async def test_a_round_trip_duplicate_does_not_displace_an_older_song(
        self, store: GuildRedisStore
    ) -> None:
        # The second half of the same defect: merged[:limit] runs AFTER the
        # dedup, so every duplicate that slips through pushes a genuine older
        # play out of the window the user asked for.
        #
        # The archive must come back SHORT of the limit or recent() returns it
        # verbatim (the len(archived) >= limit short-circuit) and never merges
        # at all — which is the shape that made an earlier draft of this test
        # pass against the bug it was written for.
        raw = 1785309912.8701255
        newest = _entry(9, played_at=raw)  # drained: in the archive AND Redis
        older = _entry(3, played_at=raw - 600.0)  # only Redis, still in the outbox

        h = _history(store, archive=QuantizingArchive([newest]))
        await h.add(older)
        await h.add(newest)

        got = await h.recent(2)
        assert [e.webpage_url for e in got] == [newest.webpage_url, older.webpage_url]

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
