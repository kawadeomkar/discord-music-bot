"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory cache and the Redis leg agree on their shared window. Both are capped
at HISTORY_CACHE_LIMIT; the Redis leg carries no TTL — bounded by length,
retained forever, because it is the only thing -history reads. Writes still fan
out to Postgres (TestAddOutboxRouting); reads do not (TestRecentIsRedisOnly).
"""

import asyncio
import dataclasses
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


_UNSET: Any = object()


def _history(
    store: Any,
    *,
    on_outbox_push: Any = _UNSET,
) -> GuildHistory:
    """GuildHistory with the notify wiring defaulted to a no-op callable.

    A sentinel default, not None: None is MEANINGFUL on the real constructor
    (archive tier disabled), so the helper must be able to pass it through.
    """
    return GuildHistory(
        store,
        on_outbox_push=(lambda: None) if on_outbox_push is _UNSET else on_outbox_push,
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

    async def test_both_legs_are_capped_at_the_same_window(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        h = _history(store)
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await h.add(_entry(i))
        # The cache holds only the newest window…
        assert len(h) == HISTORY_CACHE_LIMIT
        assert h[0] == _entry(5)  # oldest cached = first evicted survivor
        # …and so does Redis: trimmed to the same window on every write, which
        # is what makes it a bounded, permanently-retained key.
        raw = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(raw) == HISTORY_CACHE_LIMIT
        # Non-evictable, still: no TTL, because -history has no other source.
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
    """Write-path archive wiring: with a notify present (the enabled shape and
    the suite default) every add() pushes to the outbox and nudges the drainer;
    with None (archive off, no drainer exists) add() keeps the display legs and
    nudges nothing. The read path is TestRecentIsRedisOnly."""

    async def test_add_routes_entry_to_outbox_too(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        h = _history(store)
        await h.add(_entry(1))
        # XRANGE, not LRANGE: the outbox is a stream. Asserting the payload under
        # OUTBOX_FIELD, not just the entry count, keeps this honest about the wire
        # format the drainer reads back. The cast narrows redis-py's XRANGE union
        # (wide enough for XAUTOCLAIM's 4-tuple rows and the RESP3 dict form).
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
        # Redis optionality is a separate axis from the archive: without a
        # store there is nowhere to buffer, so add() degrades to memory-only
        # and must not nudge a drainer that has nothing to drain.
        calls = []
        h = _history(None, on_outbox_push=lambda: calls.append(1))
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]
        assert calls == []

    async def test_none_notify_keeps_the_display_legs(
        self, store: GuildRedisStore
    ) -> None:
        # The disabled-archive shape: no drainer exists, so the constructor
        # receives None explicitly. The display legs are untouched by that
        # choice — Redis behavior is identical in both modes.
        h = _history(store, on_outbox_push=None)
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]
        assert await store.get_history() == [_entry(1)]

    async def test_none_notify_without_store_degrades_to_memory(self) -> None:
        # The two Optional axes compose: archive disabled AND Redis absent
        # still records the in-memory leg — the bot keeps working.
        h = _history(None, on_outbox_push=None)
        await h.add(_entry(1))
        assert list(h) == [_entry(1)]


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

    async def test_empty_history(self) -> None:
        assert await _history(None).recent(10) == []


class TestRecentIsRedisOnly:
    """-history is served from the Redis list and the in-memory deque, and from
    nothing else.

    Reading one leg deletes a whole class of reconciliation bug (a full archive
    result short-circuiting the newest songs, quantized timestamps duplicating a
    play, one archived row suppressing the cache); the arithmetic that makes it
    sound is asserted in TestRecentWindowIsTheCap below.
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

        Regression: consulted only when every leg above it came back empty, a
        single row from a healthier leg hid it entirely. As a second leg it can
        only add depth.
        """
        h = _history(store)
        await h.add(_entry(1))
        # A play that reached the deque but not the list (the write half of a
        # Redis blip): get_history() cannot see it, recent() still must.
        h.restore([_entry(2)])
        assert [e.title for e in await h.recent(10)] == ["Song 2", "Song 1"]

    async def test_the_two_legs_dedup_exactly(self, store: GuildRedisStore) -> None:
        """Both legs carry the same time.time() float — the deque holds the object
        add() appended, the list holds it through orjson, which round-trips a
        double without loss — so identity is equality. No lossy boundary (the
        archive's µs-truncating timestamptz) is left on this path.
        """
        h = _history(store)
        raw = 1000.0000014  # finer than µs; nothing truncates it now
        await h.add(_entry(1, played_at=raw))  # lands on both legs
        got = await h.recent(10)
        assert len(got) == 1
        assert got[0].played_at == raw

    async def test_a_song_played_twice_survives_the_merge(
        self, store: GuildRedisStore
    ) -> None:
        """The other direction of the dedup property: the `played_at` half of the
        key. Every other builder here mints a distinct URL, so the mutation
        `key = (0.0, entry.webpage_url)` survived the whole suite while collapsing
        every repeat play of a song into its newest occurrence.
        """
        h = _history(store)
        await h.add(_entry(1, played_at=1000.0))
        await h.add(_entry(2, played_at=1001.0))
        # Same URL as the first, played again later.
        await h.add(_entry(1, played_at=1002.0))

        got = await h.recent(10)

        assert [e.played_at for e in got] == [1002.0, 1001.0, 1000.0]
        assert sum(e.webpage_url == _entry(1).webpage_url for e in got) == 2

    async def test_unknown_timestamps_are_not_treated_as_one_identity(
        self, store: GuildRedisStore
    ) -> None:
        """played_at == 0.0 means "never recorded", not "played at the epoch":
        parse_history_entry defaults it and __post_init__ forces it for
        NaN/out-of-range, so every pre-timestamp entry carries the same value.
        Keying on it collapsed distinct plays of one URL into a single row.
        """
        h = _history(store)
        first = _entry(1, played_at=0.0)
        # The same song, played again by someone else — still no timestamp.
        # Keyed on (played_at, webpage_url) these two were indistinguishable
        # and one of them silently vanished.
        again = dataclasses.replace(first, requester_id=99, requester_name="user99")
        await h.add(first)
        await h.add(again)

        got = await h.recent(10)

        assert len(got) == 2
        assert {e.requester_id for e in got} == {1, 99}

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
        """The bound exists even though GuildRedisStore swallows its own errors: a
        connected-but-unresponsive server produces no error to swallow, and the
        pool sets no socket read timeout. Without the timeout this test HANGS
        rather than failing.
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
        """The CONTRACT, not the early-out that implements it: `limit <= 0` is an
        equivalent mutation away from `< 0` (merged[:limit] handles 0 identically),
        so asserting the branch would be theatre. What is pinned is that a caller
        asking for nothing gets nothing, with a full cache behind it.
        """
        h = _history(None)
        h.restore([_entry(2), _entry(1)])
        assert await h.recent(limit) == []


class TestRecentWindowIsTheCap:
    """The arithmetic the Redis-only read path rests on: push_history trims to
    HISTORY_CACHE_LIMIT, get_history reads that whole window, and
    musicbot.HISTORY_MAX_LIMIT is pinned to the same constant. Break any of the
    three and -history silently starts losing depth.
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
