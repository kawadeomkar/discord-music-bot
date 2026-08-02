"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory display cache and the Redis leg agree on their shared window. The
cache is capped at HISTORY_CACHE_LIMIT; the Redis leg is unbounded and PERSISTed.

Writes fan out to Postgres (TestAddOutboxRouting); reads do not
(TestRecentIsRedisOnly). TestRecentWindowIsComplete asserts the arithmetic that
makes reading one leg enough.
"""

import asyncio
from unittest.mock import AsyncMock, patch

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


def _history(
    store: Any,
    *,
    on_outbox_push: Any = None,
) -> GuildHistory:
    """GuildHistory with the required notify wiring defaulted.

    on_outbox_push is mandatory on the real constructor (the Postgres tier is
    not optional on the WRITE path). There is no archive parameter any more:
    recent() reads Redis and the in-memory deque, never Postgres.
    """
    return GuildHistory(
        store,
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
        # Non-evictable: no TTL, because -history has no other source.
        assert await fake_redis.ttl(store.history_key()) == -1

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
    Write path only — and it is the ONLY direction Postgres appears in now; the
    read path is TestRecentIsRedisOnly."""

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


class TestRecentIsRedisOnly:
    """-history is served from the Redis list and the in-memory deque, and from
    nothing else.

    The class this replaced (TestRecentReadsPostgresFirst) pinned a three-tier
    merge: archive first, Redis merged in for undrained plays, cache last. Every
    defect it documented — a full archive result short-circuiting the newest
    songs, µs-quantized timestamps duplicating a play across legs, one archived
    row suppressing the whole cache — was a reconciliation bug between tiers that
    hold the same plays at different times. Reading one leg deletes the class of
    bug rather than fixing instances of it.

    What makes that sound is arithmetic, not preference, and it is asserted in
    TestRecentWindowIsTheCap below: push_history caps the list at
    HISTORY_CACHE_LIMIT and musicbot.HISTORY_MAX_LIMIT is pinned to the same
    constant, so the list always holds every play the command can ask for.
    """

    async def test_redis_is_the_source_when_the_cache_is_cold(
        self, store: GuildRedisStore
    ) -> None:
        # The restart shape: entries on the list, nothing in memory yet.
        await store.push_history(_entry(1))
        await store.push_history(_entry(2))
        h = _history(store)
        assert [e.title for e in await h.recent(10)] == ["Song 2", "Song 1"]

    async def test_the_cache_adds_depth_redis_did_not_return(
        self, store: GuildRedisStore
    ) -> None:
        """The deque is MERGED, not a fallback reached only on an empty read.

        REGRESSION: it used to be consulted only when every leg above it came
        back empty, so a single row from a healthier leg hid it entirely —
        reproduced at Redis-erroring / cache-holding-9, where recent(10)
        returned 1 where the pre-archive code returned 9. As a second leg it can
        only add depth.
        """
        h = _history(store)
        await h.add(_entry(1))
        # A play that reached the deque but not the list (the write half of a
        # Redis blip): get_history() cannot see it, recent() still must.
        h.restore([_entry(2)])
        assert [e.title for e in await h.recent(10)] == ["Song 2", "Song 1"]

    async def test_the_two_legs_dedup_exactly(self, store: GuildRedisStore) -> None:
        """Both legs carry the same time.time() float — the deque holds the
        object add() appended, the list holds it through orjson, which
        round-trips a double without loss — so identity is equality.

        This is what the deleted quantized_played_at guarded on the archive leg,
        where timestamptz truncated to µs and ~37% of real timestamps compared
        unequal against their own Redis copy. No such boundary is left.
        """
        h = _history(store)
        raw = 1000.0000014  # finer than µs; nothing truncates it now
        await h.add(_entry(1, played_at=raw))  # lands on BOTH legs
        got = await h.recent(10)
        assert len(got) == 1
        assert got[0].played_at == raw

    async def test_redis_failure_falls_through_to_the_cache_and_warns(
        self, store: GuildRedisStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        h = _history(store)
        await h.add(_entry(1))
        with patch.object(
            store, "get_history", AsyncMock(side_effect=OSError("redis unreachable"))
        ):
            assert await h.recent(10) == [_entry(1)]
        assert "redis read failed" in caplog.text

    async def test_a_slow_redis_is_bounded_and_falls_through(
        self, store: GuildRedisStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound exists even though GuildRedisStore swallows its own errors:
        swallowing turns a FAILURE into [], but a connected-and-unresponsive
        server produces no error to swallow, and the pool sets no socket read
        timeout. Without the timeout this test HANGS rather than failing.
        """
        import src.guild_history as guild_history

        monkeypatch.setattr(guild_history, "_READ_TIMEOUT_SECS", 0.02)
        h = _history(store)
        await h.add(_entry(1))

        async def never() -> list[HistoryEntry]:
            await asyncio.Event().wait()
            return []

        with patch.object(store, "get_history", never):
            assert await h.recent(10) == [_entry(1)]

    async def test_no_store_at_all_still_reaches_the_cache(self) -> None:
        # Redis unconfigured is a different axis from Redis failing: there is no
        # leg to read, and the deque is the whole answer.
        h = _history(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(10) == [_entry(2), _entry(1)]

    async def test_merge_still_honours_the_limit(self, store: GuildRedisStore) -> None:
        h = _history(store)
        for n in (1, 2, 3):
            await h.add(_entry(n))
        assert [e.title for e in await h.recent(2)] == ["Song 3", "Song 2"]

    @pytest.mark.parametrize("limit", [0, -1])
    async def test_a_nonpositive_limit_yields_nothing(self, limit: int) -> None:
        """The CONTRACT, not the early-out that implements it.

        recent()'s `limit <= 0` check is an equivalent mutation away from
        `< 0` — merged[:limit] handles 0 identically — so a test asserting the
        branch would be theatre. What is worth pinning is that a caller asking
        for nothing gets nothing, from the arrangement that used to be able to
        return the whole cache instead: nothing above the cache, cache full.
        """
        h = _history(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(limit) == []


class TestRecentWindowIsComplete:
    """The arithmetic the whole Redis-only read path rests on.

    get_history reads the newest HISTORY_CACHE_LIMIT entries and
    musicbot.HISTORY_MAX_LIMIT is pinned to the same constant — so the list
    provably holds every play -history can be asked for. Break either and the
    command silently starts losing depth, which is why the relationship is
    asserted rather than left to the comments that state it.
    """

    async def test_the_command_ceiling_cannot_outrun_the_window(self) -> None:
        # Imported here rather than at module scope: this is the one assertion
        # in this file that reaches into the command layer, and it is asserting
        # a relationship between two constants, not exercising a command.
        from src.musicbot import HISTORY_MAX_LIMIT

        assert HISTORY_MAX_LIMIT == HISTORY_CACHE_LIMIT

    async def test_the_full_window_is_readable_at_the_ceiling(
        self, store: GuildRedisStore
    ) -> None:
        """A guild that has played more than the window still renders a full
        page — from Redis alone, with no archive behind it."""
        h = _history(store)
        total = HISTORY_CACHE_LIMIT + 10
        for n in range(total):
            await h.add(_entry(n))
        got = await h.recent(HISTORY_CACHE_LIMIT)
        assert len(got) == HISTORY_CACHE_LIMIT
        assert [e.title for e in got] == [
            f"Song {n}" for n in range(total - 1, total - 1 - HISTORY_CACHE_LIMIT, -1)
        ]

    async def test_an_undrained_play_renders_immediately(
        self, store: GuildRedisStore
    ) -> None:
        """The property that makes Postgres unnecessary here: the list is
        written synchronously at song end, so it LEADS the archive. A play the
        drainer has not touched is already the newest row -history shows."""
        h = _history(store)
        await h.add(_entry(1))
        await h.add(_entry(2))  # drained nowhere yet
        assert [e.title for e in await h.recent(10)] == ["Song 2", "Song 1"]


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
