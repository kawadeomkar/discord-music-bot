"""Tests for src/redis_client.py — connection lifecycle, cache helpers, and GuildRedisStore."""

import ast
import inspect
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import redis.asyncio as aioredis
from redis.asyncio.connection import Connection as RedisConnection
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.backoff import ExponentialBackoff, NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import OutOfMemoryError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.exceptions import WatchError

import src.redis_client as redis_client
from src.guild_state import (
    GuildConfig,
    GuildPlaybackSnapshot,
    GuildRecoveryGate,
    GuildStateData,
    HistoryEntry,
    NowPlayingData,
    SongQueueEntry,
)
from tests.helpers import mocked
from src.redis_client import (
    GUILD_TTL,
    HISTORY_CACHE_LIMIT,
    HISTORY_OUTBOX_GROUP,
    HISTORY_OUTBOX_KEY,
    OUTBOX_FIELD,
    GuildRedisStore,
    _PLAYBACK_POSITION_FIELDS,
    _prev_stream_id,
    _TRANSIENT_SONG_FIELDS,
    ack_outbox,
    cache_get,
    cache_set,
    close_redis_pool,
    create_redis_pool,
    ensure_outbox_group,
    get_redis,
    outbox_depth,
    outbox_pending_below,
    outbox_pending_count,
    read_outbox_new,
    read_outbox_pending,
    reclaim_outbox_stale,
    retire_outbox,
    spotify_token_get_with_ttl,
    spotify_token_set,
    trim_outbox_below,
)

# ── Connection lifecycle ──────────────────────────────────────────────────────


class TestCreateRedisPool:
    def test_returns_connection_pool(self) -> None:
        pool = create_redis_pool()
        assert isinstance(pool, aioredis.ConnectionPool)

    def test_pool_has_expected_max_connections(self) -> None:
        pool = create_redis_pool()
        assert pool.max_connections == 20

    def test_redis_errors_are_not_builtin_subclasses(self) -> None:
        """The invariant that made the old config a silent no-op: redis-py's
        ConnectionError and TimeoutError derive from RedisError, not from the
        builtins of the same name — so `retry_on_error=[ConnectionError,
        TimeoutError]` (the builtins) matched nothing redis-py ever raises."""
        assert not issubclass(RedisConnectionError, ConnectionError)
        assert not issubclass(RedisTimeoutError, TimeoutError)

    def test_retry_on_error_uses_redis_exception_classes(self) -> None:
        pool = create_redis_pool()
        configured = pool.connection_kwargs["retry_on_error"]
        assert RedisConnectionError in configured
        assert RedisTimeoutError in configured
        # The builtins are what made this ineffective; they must not come back.
        assert ConnectionError not in configured

    def test_connections_actually_retry_connection_errors(self) -> None:
        """End of the chain: a connection built by this pool must treat a redis-py
        ConnectionError as retryable. Asserted on a real Connection rather than on
        kwargs, because redis-py merges `retry_on_error` into the Retry object at
        construction time — the step the old config silently no-opped."""
        conn = create_redis_pool().make_connection()
        assert conn.retry.get_retries() == 3
        # _supported_errors is private but is the only readable surface for the
        # merged set; the assertions above cover the public config either way.
        assert RedisConnectionError in conn.retry._supported_errors

    def test_backoff_is_not_the_synthesised_no_backoff(self) -> None:
        """Without an explicit Retry, redis-py synthesises Retry(NoBackoff(), 1)
        — one immediate reattempt, which a restarting Redis outlives."""
        conn = create_redis_pool().make_connection()
        assert isinstance(conn.retry._backoff, ExponentialBackoff)

    async def test_a_failing_connect_is_actually_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mutation guard: `redis.retry.Retry` in place of the asyncio one.

        Every test above passes under either class, but the sync one never awaits:
        its call_with_retry hands back an un-awaited coroutine, so the error
        escapes outside the retry loop. Counting attempts on a real connect is the
        only assertion that tells them apart; `_connect` is patched rather than
        aimed at a closed port so no socket is involved.
        """
        attempts = 0

        async def refuse(_self: object) -> None:
            nonlocal attempts
            attempts += 1
            raise RedisConnectionError("connection refused")

        monkeypatch.setattr(RedisConnection, "_connect", refuse)
        # NoBackoff so the assertion is about the retry count, not about waiting
        # out ExponentialBackoff's 8+16+32ms. The backoff itself is asserted above.
        monkeypatch.setattr(
            create_redis_pool.__module__ + ".ExponentialBackoff", NoBackoff
        )
        conn = create_redis_pool().make_connection()
        with pytest.raises(RedisConnectionError):
            await conn.connect()
        # 1 initial + 3 retries. The sync class yields exactly 1.
        assert attempts == 4


class TestGetRedis:
    def test_returns_redis_client(self) -> None:
        pool = create_redis_pool()
        client = get_redis(pool)
        assert isinstance(client, aioredis.Redis)


class TestCloseRedisPool:
    async def test_calls_aclose_on_pool(self) -> None:
        pool = AsyncMock()
        pool.aclose = AsyncMock()
        await close_redis_pool(pool)
        pool.aclose.assert_awaited_once()

    async def test_swallows_exception_on_close(self) -> None:
        pool = AsyncMock()
        pool.aclose = AsyncMock(side_effect=Exception("network gone"))
        await close_redis_pool(pool)  # must not raise


# ── Cache helpers ─────────────────────────────────────────────────────────────


class TestCacheGet:
    async def test_returns_none_when_redis_is_none(self) -> None:
        result = await cache_get(None, "some:key")
        assert result is None

    async def test_returns_none_on_cache_miss(self, fake_redis: aioredis.Redis) -> None:
        result = await cache_get(fake_redis, "nonexistent:key")
        assert result is None

    async def test_returns_decoded_value_on_hit(
        self, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.set("mykey", orjson.dumps({"x": 1}))
        result = await cache_get(fake_redis, "mykey")
        assert result == {"x": 1}

    async def test_returns_none_on_redis_error(self) -> None:
        bad_redis = AsyncMock()
        bad_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        result = await cache_get(bad_redis, "key")
        assert result is None


class TestCacheSet:
    async def test_sets_value_with_ttl(self, fake_redis: aioredis.Redis) -> None:
        await cache_set(fake_redis, "ck", [1, 2, 3], 3600)
        raw = await fake_redis.get("ck")
        assert raw is not None
        assert orjson.loads(raw) == [1, 2, 3]
        ttl = await fake_redis.ttl("ck")
        assert 3595 <= ttl <= 3600

    async def test_noop_when_redis_is_none(self) -> None:
        await cache_set(None, "key", "val", 60)  # must not raise

    async def test_swallows_redis_error(self) -> None:
        bad_redis = AsyncMock()
        bad_redis.set = AsyncMock(side_effect=ConnectionError("down"))
        await cache_set(bad_redis, "k", "v", 60)  # must not raise


# ── GuildRedisStore fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store(fake_redis: aioredis.Redis) -> GuildRedisStore:
    return GuildRedisStore(fake_redis, guild_id=123456789)


@pytest.fixture
def broken_store() -> GuildRedisStore:
    """Store backed by a Redis mock that raises on every operation."""
    r = MagicMock()
    err = ConnectionError("redis down")
    for attr in ("rpush", "lpop", "lrange", "delete", "hset", "hgetall", "hdel", "set"):
        setattr(r, attr, AsyncMock(side_effect=err))
    pipe = MagicMock()
    for attr in ("rpush", "expire", "lpush", "ltrim", "hset", "delete", "hdel", "lpop"):
        setattr(pipe, attr, MagicMock())
    pipe.execute = AsyncMock(side_effect=err)
    r.pipeline = MagicMock(return_value=pipe)
    return GuildRedisStore(r, guild_id=999)


# ── Key helpers ───────────────────────────────────────────────────────────────


class TestKeyHelpers:
    def test_queue_key_includes_guild_id(self, store: GuildRedisStore) -> None:
        assert "123456789" in store.queue_key()

    def test_state_key_includes_guild_id(self, store: GuildRedisStore) -> None:
        assert "123456789" in store.state_key()

    def test_history_key_includes_guild_id(self, store: GuildRedisStore) -> None:
        assert "123456789" in store.history_key()

    def test_now_playing_key_includes_guild_id(self, store: GuildRedisStore) -> None:
        assert "123456789" in store.now_playing_key()

    def test_keys_are_distinct(self, store: GuildRedisStore) -> None:
        keys = [
            store.queue_key(),
            store.state_key(),
            store.history_key(),
            store.now_playing_key(),
        ]
        assert len(set(keys)) == len(keys)


# ── Queue operations ──────────────────────────────────────────────────────────


def _entry(n: int = 1) -> SongQueueEntry:
    return SongQueueEntry(
        webpage_url=f"https://yt.com/v={n}", title=f"Song {n}", requester_id=n
    )


class TestPushQueue:
    async def test_rpush_adds_entry_bytes(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_queue(_entry(1))
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis()]

    async def test_sets_ttl_on_queue_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_queue(_entry(1))
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_refreshes_ttl_on_now_playing_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """_pipe_expire_all must refresh the TTL-managed guild keys, including
        now_playing."""
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        await fake_redis.expire(store.now_playing_key(), 5)
        await store.push_queue(_entry(1))
        ttl = await fake_redis.ttl(store.now_playing_key())
        assert ttl > 5

    async def test_does_not_rearm_history_expiry(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The history key is persistent — _pipe_expire_all must leave it alone."""
        await fake_redis.lpush(store.history_key(), b'"entry"')
        await store.push_queue(_entry(1))
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.push_queue(_entry(1))  # must not raise


class TestPushQueueBatch:
    async def test_rpush_all_entries_in_order(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_queue_batch([_entry(1), _entry(2)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis(), _entry(2).to_redis()]

    async def test_noop_on_empty_sequence(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_queue_batch([])
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.push_queue_batch([_entry(1)])  # must not raise


class TestPushQueueFront:
    async def test_entries_land_at_head_in_given_order(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), _entry(3).to_redis())
        await store.push_queue_front([_entry(1), _entry(2)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [
            _entry(1).to_redis(),
            _entry(2).to_redis(),
            _entry(3).to_redis(),
        ]

    async def test_sets_ttl_on_queue_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_queue_front([_entry(1)])
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_noop_on_empty_sequence(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_queue_front([])
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.push_queue_front([_entry(1)])  # must not raise


class TestPopQueue:
    async def test_removes_first_item(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"first", b"second")
        await store.pop_queue()
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"second"]

    async def test_noop_on_empty_queue(self, store: GuildRedisStore) -> None:
        await store.pop_queue()  # must not raise

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.pop_queue()  # must not raise


class TestDeleteQueue:
    async def test_deletes_queue_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"x")
        await store.delete_queue()
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == []

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.delete_queue()  # must not raise


class TestRebuildQueue:
    async def test_atomically_replaces_queue(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"old")
        await store.rebuild_queue([_entry(1), _entry(2)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis(), _entry(2).to_redis()]

    async def test_sets_ttl_after_rebuild(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.rebuild_queue([_entry(1)])
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.rebuild_queue([_entry(1)])  # must not raise


class TestRemoveQueueEntries:
    """The LREM path GuildQueue takes for a small removal instead of rewriting
    the whole list. See tests/test_redis_integration.py for the real server —
    fakeredis is not authoritative about LREM."""

    async def test_removes_only_the_named_entries(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.rebuild_queue([_entry(n) for n in range(1, 5)])
        await store.remove_queue_entries([_entry(2), _entry(4)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis(), _entry(3).to_redis()]

    async def test_order_of_the_survivors_is_untouched(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The whole licence for LREM over a rebuild: it leaves the rest where
        they were, so the queue a user is looking at does not reshuffle."""
        await store.rebuild_queue([_entry(n) for n in range(1, 7)])
        await store.remove_queue_entries([_entry(1), _entry(6)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(n).to_redis() for n in (2, 3, 4, 5)]

    async def test_a_duplicate_is_removed_once_per_copy_taken(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """LREM ... 0 would remove every identical entry. Two enqueues of one song
        usually differ on the wire, but when they do not, removing "all matching"
        takes out a copy still queued."""
        await store.rebuild_queue([_entry(1), _entry(1), _entry(1)])
        await store.remove_queue_entries([_entry(1)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis(), _entry(1).to_redis()]

    async def test_refreshes_the_ttl(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The queue key is TTL'd like every other guild key, and a removal is a
        write — leaving it to age off its old expiry would expire a live queue."""
        await store.rebuild_queue([_entry(1), _entry(2)])
        await fake_redis.persist(store.queue_key())
        await store.remove_queue_entries([_entry(1)])
        assert await fake_redis.ttl(store.queue_key()) > 0

    async def test_returns_how_many_were_removed(self, store: GuildRedisStore) -> None:
        """Exactly the LREM counts — the pipeline's trailing EXPIRE reply is a 1
        of its own, and summing it in would report a phantom removal."""
        await store.rebuild_queue([_entry(n) for n in range(1, 5)])
        assert await store.remove_queue_entries([_entry(2), _entry(4)]) == 2

    async def test_returns_short_when_the_mirror_no_longer_holds_them(
        self, store: GuildRedisStore
    ) -> None:
        """The count is GuildQueue's only signal that the mirror and memory
        disagree: LREM raises nothing when what it was told to drop is absent."""
        await store.rebuild_queue([_entry(1)])
        assert await store.remove_queue_entries([_entry(1), _entry(2)]) == 1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        # 0 rather than None: it cannot match what the caller asked for, so a
        # failed LREM routes it to the rebuild like any other short count.
        assert await broken_store.remove_queue_entries([_entry(1)]) == 0


# ── History operations ────────────────────────────────────────────────────────


def _hentry(n: int = 1) -> HistoryEntry:
    return HistoryEntry(
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200 + n,
        played_secs=100 + n,
        requester_id=n,
        requester_name=f"user{n}",
        played_at=1000.0 + n,
    )


class TestPushHistory:
    async def test_prepends_item_newest_first(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_history(_hentry(1))
        await store.push_history(_hentry(2))
        items = await fake_redis.lrange(store.history_key(), 0, -1)
        assert items[0] == _hentry(2).to_redis()  # newest first

    async def test_the_trim_keeps_the_NEWEST_window(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # Direction matters and LTRIM's argument order is easy to invert: the list
        # is newest-first, so `0, LIMIT-1` keeps the head. Keeping the tail instead
        # would leave -history rendering a guild's oldest 50 plays forever —
        # bounded, PERSISTed, and wrong, which no length assertion can see.
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await store.push_history(_hentry(i))
        items = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(items) == HISTORY_CACHE_LIMIT
        assert items[0] == _hentry(HISTORY_CACHE_LIMIT + 4).to_redis()

    async def test_history_key_is_persistent(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_history(_hentry(1))
        assert await fake_redis.ttl(store.history_key()) == -1  # no expiry

    async def test_persist_heals_pre_migration_ttl(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # A key written by an old build carries the 24h idle expiry; the first
        # new-build push must remove it, not let history evaporate.
        await fake_redis.lpush(store.history_key(), orjson.dumps("old entry"))
        await fake_redis.expire(store.history_key(), 3600)
        await store.push_history(_hentry(1))
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.push_history(_hentry(1))  # must not raise


class TestHistoryTtl:
    """The read behind -debug's PERSIST check row. Redis's own vocabulary: -1 is
    no expiry, -2 is no such key."""

    async def test_absent_key(self, store: GuildRedisStore) -> None:
        assert await store.history_ttl() == -2

    async def test_persisted_key_reports_no_expiry(
        self, store: GuildRedisStore
    ) -> None:
        await store.push_history(_hentry(1))
        assert await store.history_ttl() == -1

    async def test_reports_a_ttl_that_should_not_be_there(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.push_history(_hentry(1))
        await fake_redis.expire(store.history_key(), 3600)
        assert await store.history_ttl() == 3600

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        """On the @_guild_op side of golden rule 5's split: None means Redis did
        not answer, which the check row renders as unknown rather than as a
        violation of the invariant."""
        assert await broken_store.history_ttl() is None


class TestPushHistoryOutbox:
    """The outbox leg of the push pipeline, in both archive modes. The suite
    default is archive-enabled (conftest pins the flag), so tests that don't touch
    it assert the full wiring; the disabled tests monkeypatch it false per case
    and pin the consent property — the outbox key is never even created.
    """

    async def test_every_push_reaches_the_outbox(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # The enabled shape: a drainer exists behind the outbox, so no push
        # may skip it — an entry that misses the stream is a play the archive
        # never hears about.
        await store.push_history(_hentry(1))
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 1

    async def test_disabled_archive_never_creates_the_outbox(
        self, store: GuildRedisStore, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The ship default. EXISTS, not XLEN: the consent property is that
        # nothing accumulates for Postgres AT all — the non-evictable key is
        # never created, not just left empty.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        await store.push_history(_hentry(1))
        assert await fake_redis.exists(HISTORY_OUTBOX_KEY) == 0

    async def test_disabled_archive_keeps_the_display_invariants(
        self, store: GuildRedisStore, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The retention policy is identical in both modes: LTRIM to the window,
        # PERSIST so nothing expires it. Only the outbox leg is gated.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await store.push_history(_hentry(i))
        items = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(items) == HISTORY_CACHE_LIMIT
        assert items[0] == _hentry(HISTORY_CACHE_LIMIT + 4).to_redis()
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_outbox_gets_same_wire_bytes_under_the_agreed_field(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # The field name is a contract between push_history and the drainer's
        # reader. Asserting the payload by name rather than by "the only value
        # in the dict" is what makes a rename fail here instead of at runtime.
        await store.push_history(_hentry(1))
        entries = await _stream_entries(fake_redis)
        assert [f[OUTBOX_FIELD] for _id, f in entries] == [_hentry(1).to_redis()]

    async def test_display_leg_unchanged_by_outbox_flag(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # The display list still gets the entry, untrimmed, PERSISTed.
        await store.push_history(_hentry(1))
        assert await fake_redis.lrange(store.history_key(), 0, -1) == [
            _hentry(1).to_redis()
        ]
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_outbox_is_global_and_interleaves_guilds(
        self, fake_redis: Redis
    ) -> None:
        # One outbox for all guilds — entries carry guild_id on the wire.
        a = GuildRedisStore(fake_redis, guild_id=1)
        b = GuildRedisStore(fake_redis, guild_id=2)
        await a.push_history(_hentry(1))
        await b.push_history(_hentry(2))
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 2

    async def test_outbox_key_has_no_ttl(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # Not-yet-durable entries must never be eviction candidates under
        # volatile-lru — the same property that protects history today, and it
        # survives the list→stream change unchanged (golden rule 12).
        await store.push_history(_hentry(1))
        assert await fake_redis.ttl(HISTORY_OUTBOX_KEY) == -1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        # The producer stays on the SWALLOWING side of golden rule 5, unlike
        # every drain helper below. The playback loop must not die because
        # Redis blinked.
        await broken_store.push_history(_hentry(1))  # must not raise


async def _push(fake_redis: Redis, *ns: int) -> None:
    store = GuildRedisStore(fake_redis, guild_id=42)
    for n in ns:
        await store.push_history(_hentry(n))


async def _stream_entries(fake_redis: Redis) -> list[tuple[bytes, dict[bytes, bytes]]]:
    """The outbox stream, oldest first, narrowed to the ordinary-entry shape.

    redis-py types XRANGE's reply as a union wide enough for XAUTOCLAIM's 4-tuple
    rows and RESP3 dicts, so every unpack would otherwise need its own ignore.
    """
    return cast(
        list[tuple[bytes, dict[bytes, bytes]]],
        await fake_redis.xrange(HISTORY_OUTBOX_KEY),
    )


async def _stream_ids(fake_redis: Redis) -> list[bytes]:
    return [i for i, _ in await _stream_entries(fake_redis)]


class TestOperatorRecipeMatchesTheSchema:
    """`just outbox` hardcodes the key and group names — a shell recipe cannot
    import them. Drift fails open: rename either constant and the recipe keeps
    working against a key nothing writes, reporting "nothing buffered, nothing to
    do" during the incident it exists for.
    """

    @staticmethod
    def _recipe() -> str:
        text = (Path(__file__).resolve().parent.parent / "justfile").read_text()
        start = text.index("\noutbox IDLE_MS=")
        # Recipes end at the first line that is neither blank nor indented.
        body: list[str] = []
        for line in text[start + 1 :].splitlines()[1:]:
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
        return "\n".join(body)

    def test_the_recipe_targets_the_key_the_producer_writes(self) -> None:
        # Whole assignment lines, not `in`: the substring form passes for
        # `key=history:outbox_old`, which is precisely the drift being guarded.
        assigned = {ln.strip() for ln in self._recipe().splitlines()}
        assert f"key={HISTORY_OUTBOX_KEY}" in assigned

    def test_the_recipe_targets_the_group_the_drainer_reads(self) -> None:
        assigned = {ln.strip() for ln in self._recipe().splitlines()}
        assert f"group={HISTORY_OUTBOX_GROUP}" in assigned


class TestEnsureOutboxGroup:
    async def test_creates_the_group_and_the_key(self, fake_redis: Redis) -> None:
        await ensure_outbox_group(fake_redis)
        groups = await fake_redis.xinfo_groups(HISTORY_OUTBOX_KEY)
        assert [g["name"] for g in groups] == [HISTORY_OUTBOX_GROUP.encode()]

    async def test_repeat_call_is_tolerated(self, fake_redis: Redis) -> None:
        await ensure_outbox_group(fake_redis)
        await ensure_outbox_group(fake_redis)  # BUSYGROUP, swallowed

    async def test_group_sees_entries_that_predate_it(self, fake_redis: Redis) -> None:
        """mutation guard: xgroup_create(id="0") → redis-py's default id="$".

        "$" delivers only what arrives after the call, so a group created after any
        window where push_history ran first would silently skip every entry already
        in the stream — never delivered, never inserted, never logged.
        """
        await _push(fake_redis, 1, 2)  # XADD auto-creates the key, no group
        await ensure_outbox_group(fake_redis)
        batch = await read_outbox_new(fake_redis, 10)
        assert [e.wire for e in batch] == [
            _hentry(1).to_redis(),
            _hentry(2).to_redis(),
        ]

    async def test_repeat_call_does_not_rewind_an_existing_group(
        self, fake_redis: Redis
    ) -> None:
        # Bootstrap runs at startup AND from the NOGROUP heal, so a repeat
        # against a live group must not re-deliver everything already settled.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        batch = await read_outbox_new(fake_redis, 10)
        await retire_outbox(fake_redis, [e.id for e in batch])
        await ensure_outbox_group(fake_redis)
        assert await read_outbox_new(fake_redis, 10) == []
        assert await read_outbox_pending(fake_redis, 10) == []

    async def test_wrongtype_propagates(self, fake_redis: Redis) -> None:
        """mutation guard: `except ResponseError: pass` for a BUSYGROUP check.

        BUSYGROUP, WRONGTYPE and NOGROUP are all bare ResponseError, so a
        class-level catch swallows the two meaning "this bot cannot record
        history" along with the one meaning "already fine". A leftover list here
        must abort startup: @_guild_op would otherwise silence the producer's
        failed XADD, taking guild:{id}:history down in the same transaction.
        """
        await fake_redis.lpush(HISTORY_OUTBOX_KEY, b"pre-R1 list entry")
        with pytest.raises(aioredis.ResponseError, match="WRONGTYPE"):
            await ensure_outbox_group(fake_redis)


class TestOutboxDrainHelpers:
    @pytest.mark.parametrize(
        "reader", [read_outbox_pending, read_outbox_new], ids=["pending", "new"]
    )
    async def test_both_reads_pin_noack_false(
        self,
        reader: Any,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mutation guard: inheriting redis-py's `noack` default, which happens to
        be correct — so it can flip with nothing going red. noack=True delivers
        without entering the PEL, turning at-least-once into at-most-once, and
        every behavioural assertion in this class still passes under it. Only the
        emitted kwargs can catch it.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        seen: list[Any] = []
        real = fake_redis.xreadgroup

        async def recording(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return await real(*args, **kwargs)

        monkeypatch.setattr(fake_redis, "xreadgroup", recording)
        await reader(fake_redis, 10)
        assert seen and seen[0].get("noack") is False

    async def test_new_read_returns_oldest_first(self, fake_redis: Redis) -> None:
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        batch = await read_outbox_new(fake_redis, 10)
        assert [e.wire for e in batch] == [
            _hentry(1).to_redis(),
            _hentry(2).to_redis(),
            _hentry(3).to_redis(),
        ]

    async def test_new_read_caps_at_count(self, fake_redis: Redis) -> None:
        """mutation guard: dropping `count` from XREADGROUP. Without it the whole
        backlog lands in the PEL in one read, and the PEL's bound (BATCH_SIZE x
        readers) is what keeps a Postgres outage from being paid for twice in
        Redis.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        batch = await read_outbox_new(fake_redis, 2)
        assert [e.wire for e in batch] == [
            _hentry(1).to_redis(),
            _hentry(2).to_redis(),
        ]
        assert await outbox_pending_count(fake_redis) == 2

    async def test_new_read_is_not_repeatable(self, fake_redis: Redis) -> None:
        # `>` consumes: a second read gets nothing new, which is what makes two
        # live drainers disjoint rather than duplicating.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        assert len(await read_outbox_new(fake_redis, 10)) == 2
        assert await read_outbox_new(fake_redis, 10) == []

    async def test_two_consumers_get_disjoint_entries(self, fake_redis: Redis) -> None:
        # The server-side property the design rests on. Named consumers
        # here rather than the module constant, because this asserts what Redis
        # guarantees about a group, not what our code chose to call itself.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3, 4)

        async def claim(consumer: str) -> set[bytes]:
            reply = cast(
                list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]],
                await fake_redis.xreadgroup(
                    HISTORY_OUTBOX_GROUP,
                    consumer,
                    {HISTORY_OUTBOX_KEY: ">"},
                    count=2,
                ),
            )
            return {i for i, _ in reply[0][1]}

        ids_a = await claim("a")
        ids_b = await claim("b")
        assert ids_a and ids_b and not (ids_a & ids_b)

    async def test_pending_read_replays_unacked_entries(
        self, fake_redis: Redis
    ) -> None:
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        first = await read_outbox_new(fake_redis, 10)
        replay = await read_outbox_pending(fake_redis, 10)
        assert [e.id for e in replay] == [e.id for e in first]
        assert [e.wire for e in replay] == [e.wire for e in first]

    async def test_pending_read_inherits_a_dead_predecessors_batch(
        self, fake_redis: Redis
    ) -> None:
        """mutation guard: a per-process consumer name (uuid4().hex). The PEL
        belongs to the name, not the process: a successor recovers a dead
        predecessor's batch by reading `0` under the same name — no lease, no TTL,
        no XAUTOCLAIM. A random name per process orphans that PEL instead.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        delivered = await read_outbox_new(fake_redis, 10)  # "predecessor" dies here
        recovered = await read_outbox_pending(fake_redis, 10)  # fresh process
        assert [e.wire for e in recovered] == [e.wire for e in delivered]

    async def test_pending_read_is_empty_once_everything_is_settled(
        self, fake_redis: Redis
    ) -> None:
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        batch = await read_outbox_new(fake_redis, 10)
        await retire_outbox(fake_redis, [e.id for e in batch])
        assert await read_outbox_pending(fake_redis, 10) == []

    async def test_reads_on_an_empty_outbox(self, fake_redis: Redis) -> None:
        # The two empty replies have different shapes — `>` gives [] and `0`
        # gives [[key, []]] — so both arms of the parser are exercised.
        await ensure_outbox_group(fake_redis)
        assert await read_outbox_new(fake_redis, 10) == []
        assert await read_outbox_pending(fake_redis, 10) == []

    async def test_retire_acks_and_deletes(self, fake_redis: Redis) -> None:
        """mutation guard: dropping either half of retire_outbox. Without XACK the
        entry is redelivered forever, which the archive's dedup index HIDES (every
        replay a silent no-op insert while the PEL grows); without XDEL the body
        stays in a key golden rule 12 makes non-evictable. Either assertion alone
        would miss one of the two.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        batch = await read_outbox_new(fake_redis, 10)
        await retire_outbox(fake_redis, [e.id for e in batch[:2]])
        assert await outbox_pending_count(fake_redis) == 1  # XACK ran
        assert await outbox_depth(fake_redis) == 1  # XDEL ran

    async def test_retire_settles_only_the_named_ids(self, fake_redis: Redis) -> None:
        """mutation guard: acking only the first ID of the batch. Unlike the
        positional retire this replaced ("drop the oldest N"), this one means
        "drop exactly these" — a push landing between the read and the retire
        must be untouched.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        batch = await read_outbox_new(fake_redis, 10)
        await _push(fake_redis, 3)  # concurrent arrival, never delivered to us
        await retire_outbox(fake_redis, [e.id for e in batch])
        assert [e.wire for e in await read_outbox_new(fake_redis, 10)] == [
            _hentry(3).to_redis()
        ]

    async def test_retire_is_idempotent(self, fake_redis: Redis) -> None:
        # Re-send safety: this is why the drain path needs no special
        # no-retry pool. XACK and XDEL both return 0 for an ID already settled.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        batch = await read_outbox_new(fake_redis, 10)
        ids = [e.id for e in batch]
        await retire_outbox(fake_redis, ids)
        await retire_outbox(fake_redis, ids)  # must not raise or disturb anything
        assert await outbox_depth(fake_redis) == 0
        assert await outbox_pending_count(fake_redis) == 0

    async def test_retire_of_nothing_is_a_noop(self, fake_redis: Redis) -> None:
        # XACK/XDEL with zero IDs is a syntax error on real Redis, so the guard
        # is required.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        await retire_outbox(fake_redis, [])
        assert await outbox_depth(fake_redis) == 1

    async def test_retire_acks_before_it_deletes(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL + mutation guard: the XACK→XDEL order inside the MULTI. A crash
        between the two is only safe this way round: XACK-then-crash leaves an
        acked-but-undeleted entry the MINID trim reclaims; XDEL-then-crash leaves
        an unrecoverable TOMBSTONE (an ID pending with no body). Asserted by
        command sequence because both orders produce identical end state.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        batch = await read_outbox_new(fake_redis, 10)
        seen: list[str] = []
        real_execute = Pipeline.execute

        async def recording_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.extend(cmd[0] for cmd, _ in self.command_stack)
            return await real_execute(self, *args, **kwargs)

        monkeypatch.setattr(Pipeline, "execute", recording_execute)
        await retire_outbox(fake_redis, [e.id for e in batch])
        assert seen == ["XACK", "XDEL"]

    async def test_retire_is_transactional(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-transactional, the ack and the delete can be split by a crash in
        # EITHER direction, which reintroduces the tombstone window the ordering
        # above exists to close.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        batch = await read_outbox_new(fake_redis, 10)
        seen: list[bool] = []
        real_execute = Pipeline.execute

        async def recording_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(self.is_transaction)
            return await real_execute(self, *args, **kwargs)

        monkeypatch.setattr(Pipeline, "execute", recording_execute)
        await retire_outbox(fake_redis, [e.id for e in batch])
        assert seen == [True]

    async def test_tombstone_reads_back_with_no_payload(
        self, fake_redis: Redis
    ) -> None:
        """An entry delivered, then deleted while still pending — XTRIM and any
        operator XDEL both produce this, neither consulting the PEL. The reader
        must surface it as wire=None: a literal fields[b"e"] raises KeyError,
        which is neither a ResponseError nor a parse failure, so it escapes every
        handler the drain path has and replays that entry forever.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        batch = await read_outbox_new(fake_redis, 10)
        await fake_redis.xdel(HISTORY_OUTBOX_KEY, batch[0].id)  # body only
        replay = await read_outbox_pending(fake_redis, 10)
        assert [e.wire for e in replay] == [None, _hentry(2).to_redis()]
        assert [e.id for e in replay] == [batch[0].id, batch[1].id]

    async def test_a_tombstone_can_be_acked(self, fake_redis: Redis) -> None:
        # Self-healing depends on this: the drainer acks tombstones
        # unconditionally, so the PEL record must actually clear.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        batch = await read_outbox_new(fake_redis, 10)
        await fake_redis.xdel(HISTORY_OUTBOX_KEY, batch[0].id)
        await retire_outbox(fake_redis, [batch[0].id])
        assert await outbox_pending_count(fake_redis) == 0
        assert await read_outbox_pending(fake_redis, 10) == []

    async def test_depth(self, fake_redis: Redis) -> None:
        assert await outbox_depth(fake_redis) == 0
        await _push(fake_redis, 1, 2)
        assert await outbox_depth(fake_redis) == 2

    async def test_helpers_raise_on_redis_error(self) -> None:
        # Unlike the cache helpers and unlike push_history, errors must
        # PROPAGATE — the drainer's backoff loop is the handler, and a swallowed
        # error would read as an empty outbox and silently stall the drain. The
        # "deliberately DO raise" half of golden rule 5, asserted not inherited.
        boom = aioredis.ConnectionError("down")

        class _DeadPipeline:
            def xack(self, *a: Any, **k: Any) -> None: ...
            def xdel(self, *a: Any, **k: Any) -> None: ...

            async def execute(self) -> Any:
                raise boom

            async def __aenter__(self) -> "_DeadPipeline":
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

        dead = MagicMock()
        for name in (
            "xreadgroup",
            "xlen",
            "xpending",
            "xtrim",
            "xautoclaim",
            "xgroup_create",
        ):
            setattr(dead, name, AsyncMock(side_effect=boom))
        dead.pipeline = MagicMock(return_value=_DeadPipeline())
        with pytest.raises(aioredis.ConnectionError):
            await read_outbox_new(dead, 10)
        with pytest.raises(aioredis.ConnectionError):
            await read_outbox_pending(dead, 10)
        with pytest.raises(aioredis.ConnectionError):
            await retire_outbox(dead, [b"1-0"])
        with pytest.raises(aioredis.ConnectionError):
            await outbox_depth(dead)
        with pytest.raises(aioredis.ConnectionError):
            await outbox_pending_count(dead)
        with pytest.raises(aioredis.ConnectionError):
            await trim_outbox_below(dead, b"1-0")
        with pytest.raises(aioredis.ConnectionError):
            await ensure_outbox_group(dead)


class TestOutboxPendingBelow:
    async def test_lists_delivered_ids_older_than_the_cut(
        self, fake_redis: Redis
    ) -> None:
        # The set a cap trim is about to destroy while a drainer holds it.
        # Trimming those bodies without acking leaves tombstones.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3, 4)
        batch = await read_outbox_new(fake_redis, 3)  # 1..3 delivered, 4 is not
        cut = batch[2].id
        assert await outbox_pending_below(fake_redis, cut) == [
            batch[0].id,
            batch[1].id,
        ]

    async def test_the_cut_itself_is_excluded(self, fake_redis: Redis) -> None:
        # MINID keeps entries >= its argument, so the entry AT the cut survives
        # the trim and must not be acked. XPENDING's range is inclusive at both
        # ends, which is exactly the off-by-one this guards.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        batch = await read_outbox_new(fake_redis, 10)
        assert await outbox_pending_below(fake_redis, batch[0].id) == []

    async def test_settled_entries_are_not_listed(self, fake_redis: Redis) -> None:
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        batch = await read_outbox_new(fake_redis, 10)
        await retire_outbox(fake_redis, [batch[0].id])
        assert await outbox_pending_below(fake_redis, batch[1].id) == []

    async def test_undelivered_entries_are_not_listed(self, fake_redis: Redis) -> None:
        # A trim that destroys never-delivered entries needs no ack — there is
        # no PEL record to leave behind.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        ids = await _stream_ids(fake_redis)
        assert await outbox_pending_below(fake_redis, ids[1]) == []

    async def test_a_missing_group_lists_nothing(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An operator DEL racing the cap. With no group nothing is pending by
        # definition, so there is nothing to ack — and the drain cycle's
        # NOGROUP heal rebuilds it on the next tick.
        monkeypatch.setattr(
            fake_redis,
            "xpending_range",
            AsyncMock(side_effect=aioredis.ResponseError("NOGROUP gone")),
        )
        assert await outbox_pending_below(fake_redis, b"5-0") == []

    async def test_other_response_errors_propagate(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fake_redis,
            "xpending_range",
            AsyncMock(side_effect=aioredis.ResponseError("WRONGTYPE nope")),
        )
        with pytest.raises(aioredis.ResponseError, match="WRONGTYPE"):
            await outbox_pending_below(fake_redis, b"5-0")


class TestPrevStreamId:
    """The inclusive→exclusive conversion behind outbox_pending_below. XPENDING's
    range is inclusive at both ends and MINID's lower bound is exclusive, so
    "everything the trim will destroy" is the set below the ID one step down. Both
    halves of a stream ID are integers, so the step is exact rather than epsilon.
    """

    def test_steps_the_sequence_when_it_can(self) -> None:
        assert _prev_stream_id(b"1700000000000-5") == b"1700000000000-4"

    def test_borrows_from_the_millisecond_half_at_sequence_zero(self) -> None:
        # 2^64-1, not 0: the sequence half is 64-bit, so the predecessor of
        # <ms>-0 is the last id of the previous millisecond. Borrowing to
        # <ms-1>-0 would leave that millisecond's other entries outside the range.
        assert _prev_stream_id(b"7-0") == b"6-%d" % ((1 << 64) - 1)

    def test_zero_zero_is_its_own_floor(self) -> None:
        # b"0-0" has nothing below it, and Redis rejects a negative id outright.
        # Returning it unchanged makes the range empty, which is the truth.
        assert _prev_stream_id(b"0-0") == b"0-0"


class TestAckOutbox:
    async def test_clears_the_pel_without_deleting_the_body(
        self, fake_redis: Redis
    ) -> None:
        """Deliberately not retire_outbox: the cap removes bodies with a single
        MINID trim, so naming every doomed ID in an XDEL would reintroduce the
        unbounded-command problem the list transport had. This only clears PEL
        records, a set bounded by BATCH_SIZE x live drainers.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        batch = await read_outbox_new(fake_redis, 10)
        await ack_outbox(fake_redis, [e.id for e in batch])
        assert await outbox_pending_count(fake_redis) == 0
        assert await outbox_depth(fake_redis) == 2  # bodies still present

    async def test_acking_nothing_is_a_noop(self, fake_redis: Redis) -> None:
        # XACK with zero IDs is a syntax error on real Redis.
        await ensure_outbox_group(fake_redis)
        await ack_outbox(fake_redis, [])


class TestTrimOutboxBelow:
    async def test_trims_strictly_below_the_id(self, fake_redis: Redis) -> None:
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        ids = await _stream_ids(fake_redis)
        assert await trim_outbox_below(fake_redis, ids[2]) == 2
        assert await _stream_ids(fake_redis) == [ids[2]]

    async def test_returns_the_number_actually_destroyed(
        self, fake_redis: Redis
    ) -> None:
        # The caller logs this, not a derived depth-minus-cap: XLEN over-counts
        # acked-but-undeleted entries and the clamp may trim fewer than asked.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        ids = await _stream_ids(fake_redis)
        assert await trim_outbox_below(fake_redis, ids[0]) == 0  # nothing older

    async def test_is_idempotent(self, fake_redis: Redis) -> None:
        """mutation guard: XTRIM MINID → XTRIM MAXLEN. MINID names an absolute ID,
        so a re-send removes nothing; MAXLEN names a length, so a re-send after
        concurrent XADDs destroys a second tranche of un-archived plays. The drain
        path runs on the pool with retries enabled, so re-sends really happen.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        ids = await _stream_ids(fake_redis)
        assert await trim_outbox_below(fake_redis, ids[2]) == 2
        await _push(fake_redis, 4, 5)  # concurrent arrivals, as after a blip
        assert await trim_outbox_below(fake_redis, ids[2]) == 0  # re-send: no-op
        assert await outbox_depth(fake_redis) == 3

    async def test_passes_approximate_false(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mutation guard: relying on redis-py's `approximate` default, which is
        True — trimming to radix-node boundaries, so on a small real stream it
        removes NOTHING while returning success. fakeredis models it as EXACT, so
        end-state assertions pass either way; only the emitted call can catch it.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        seen: list[Any] = []
        real_xtrim = fake_redis.xtrim

        async def recording_xtrim(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return await real_xtrim(*args, **kwargs)

        monkeypatch.setattr(fake_redis, "xtrim", recording_xtrim)
        await trim_outbox_below(fake_redis, b"9999999999999-0")
        assert seen and seen[0].get("approximate") is False
        assert "maxlen" not in seen[0]

    async def test_trim_does_not_clear_the_pel(self, fake_redis: Redis) -> None:
        """The reason the cap must XACK what it destroys before it trims: XTRIM is
        blind TO the PEL. Deleting a delivered-but-unacked entry destroys a play a
        drainer is holding — no Postgres row, no play_history_rejected row, no
        error naming it — and leaves a tombstone behind.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        await read_outbox_new(fake_redis, 10)  # all three delivered, none acked
        ids = await _stream_ids(fake_redis)
        await trim_outbox_below(fake_redis, ids[2])
        assert await outbox_depth(fake_redis) == 1  # bodies gone
        assert await outbox_pending_count(fake_redis) == 3  # PEL untouched


class TestReclaimOutboxStale:
    async def test_purges_tombstones_from_the_pel(self, fake_redis: Redis) -> None:
        # Besides XACK, this is the only thing that clears a tombstone on
        # Redis 7 — which is why the sweep is not optional.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        batch = await read_outbox_new(fake_redis, 10)
        await fake_redis.xdel(HISTORY_OUTBOX_KEY, batch[0].id)
        reclaimed, purged = await reclaim_outbox_stale(
            fake_redis, min_idle_ms=0, count=10, max_passes=5
        )
        assert purged == 1
        assert reclaimed == 1  # the surviving entry, claimed into our name

    async def test_reclaims_a_foreign_consumers_pel(self, fake_redis: Redis) -> None:
        # The case the shared-name pending read cannot reach: a consumer name
        # nothing live reads, which a deploy that renamed it produces.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        await fake_redis.xreadgroup(
            HISTORY_OUTBOX_GROUP, "some-old-name", {HISTORY_OUTBOX_KEY: ">"}, count=10
        )
        assert await read_outbox_pending(fake_redis, 10) == []  # not ours yet
        reclaimed, _ = await reclaim_outbox_stale(
            fake_redis, min_idle_ms=0, count=10, max_passes=5
        )
        assert reclaimed == 1
        assert [e.wire for e in await read_outbox_pending(fake_redis, 10)] == [
            _hentry(1).to_redis()
        ]

    async def test_min_idle_time_protects_a_live_siblings_batch(
        self, fake_redis: Redis
    ) -> None:
        # Idle is measured from last delivery under a shared name, so a sweep
        # with too small a min_idle_time reclaims a peer mid-insert. The
        # drainer's SWEEP_MIN_IDLE_MS is what keeps this from firing.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)
        await fake_redis.xreadgroup(
            HISTORY_OUTBOX_GROUP, "live-peer", {HISTORY_OUTBOX_KEY: ">"}, count=10
        )
        reclaimed, purged = await reclaim_outbox_stale(
            fake_redis, min_idle_ms=600_000, count=10, max_passes=5
        )
        assert (reclaimed, purged) == (0, 0)

    async def test_terminates_on_an_empty_sweep(self, fake_redis: Redis) -> None:
        """The cursor loop must not depend on cursor == b"0-0" alone: fakeredis
        returns the last-SCANNED id where real Redis returns "0-0", so a
        cursor-only termination runs correctly in production and spins forever
        here. Empty-pass termination works on both.
        """
        await ensure_outbox_group(fake_redis)
        assert await reclaim_outbox_stale(
            fake_redis, min_idle_ms=0, count=10, max_passes=5
        ) == (0, 0)

    async def test_max_passes_bounds_the_loop(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, *range(1, 8))
        calls = {"n": 0}
        real_xautoclaim = fake_redis.xautoclaim

        async def counting(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return await real_xautoclaim(*args, **kwargs)

        monkeypatch.setattr(fake_redis, "xautoclaim", counting)
        await reclaim_outbox_stale(fake_redis, min_idle_ms=0, count=1, max_passes=3)
        # Bounded above by max_passes; it may also stop early once a pass finds
        # nothing new, which is the termination condition that works on both
        # fakeredis' and real Redis' cursor conventions.
        assert 0 < calls["n"] <= 3


class TestPushHistoryAtomicity:
    """The display push and the outbox push must ride one transactional
    pipeline. Every existing test checks only end state, which is identical
    whether they share a pipeline or not."""

    async def test_both_pushes_ride_a_single_transaction(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failure mode if they split: the connection drops between the two
        round-trips, the display list gains the play and the outbox does not, and
        no drainer ever sees it — absent from Postgres forever. -history reads the
        Redis list, so nothing surfaces the gap until the archive is queried."""
        queued: list[str] = []
        direct: list[str] = []
        real_pipeline = fake_redis.pipeline

        def spy_pipeline(*args: Any, **kwargs: Any) -> Any:
            pipe = real_pipeline(*args, **kwargs)
            for name in ("lpush", "persist", "xadd", "ltrim", "expire"):
                original = getattr(pipe, name)

                def record(
                    *a: Any, _n: str = name, _o: Any = original, **k: Any
                ) -> Any:
                    queued.append(_n)
                    return _o(*a, **k)

                monkeypatch.setattr(pipe, name, record)
            return pipe

        monkeypatch.setattr(fake_redis, "pipeline", spy_pipeline)

        async def record_direct(*a: Any, **k: Any) -> Any:
            direct.append("direct")

        monkeypatch.setattr(fake_redis, "lpush", record_direct)
        monkeypatch.setattr(fake_redis, "xadd", record_direct)

        store = GuildRedisStore(fake_redis, guild_id=1)
        await store.push_history(_hentry(1))

        # Display LPUSH and outbox XADD, both on the pipeline, neither direct.
        assert queued.count("lpush") == 1
        assert queued.count("xadd") == 1
        assert direct == []


class TestHistoryRetention:
    """The whole retention policy for guild:{id}:history, which is two commands
    and no configuration: LTRIM to the window, PERSIST so nothing expires it."""

    async def test_the_list_is_capped_and_never_expires(
        self, fake_redis: Redis
    ) -> None:
        """Both halves together, because either alone is a defect. Capped without
        PERSIST is a 24h display cache, and -history has no second source, so a
        guild idle for a day answers with silence. PERSISTed without the cap is an
        unbounded non-evictable key — the OOM shape the ISSUE header describes.
        """
        store = GuildRedisStore(fake_redis, guild_id=42)
        for n in range(HISTORY_CACHE_LIMIT + 5):
            await store.push_history(_hentry(n))
        assert await fake_redis.llen(store.history_key()) == HISTORY_CACHE_LIMIT
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_an_inherited_ttl_is_cleared_by_the_next_write(
        self, fake_redis: Redis
    ) -> None:
        # A key written by a build that DID expire this list must not keep that
        # TTL: one song end per guild is the whole migration, and it is the
        # PERSIST above that performs it.
        store = GuildRedisStore(fake_redis, guild_id=42)
        await fake_redis.lpush(store.history_key(), _hentry(0).to_redis())
        await fake_redis.expire(store.history_key(), GUILD_TTL)
        assert await fake_redis.ttl(store.history_key()) > 0
        await store.push_history(_hentry(1))
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_an_oom_refusal_trims_then_retries(
        self,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """regression: a full Redis could not self-heal.

        LPUSH carries Redis' `denyoom` flag and is queued first, so the server
        refuses it at queue time and EXEC aborts with the LTRIM never running;
        reordering inside the MULTI does not help, and @_guild_op swallows the
        OutOfMemoryError into one warning per song. A bare LTRIM is not denyoom,
        so the one command that can make room is what can still run.
        """
        store = GuildRedisStore(fake_redis, guild_id=42)
        for n in range(HISTORY_CACHE_LIMIT + 10):
            await fake_redis.lpush(store.history_key(), _hentry(n).to_redis())

        # The order is the assertion, not the end state: the retried pipeline
        # carries an LTRIM of its own, so the list ends at the cap either way.
        # Only a real server tells them apart, so what is pinned is that a
        # standalone LTRIM happened BETWEEN the refusal and the retry.
        seq: list[str] = []
        real_execute = Pipeline.execute
        real_ltrim = fake_redis.ltrim

        async def failing_once(self: Any, *a: Any, **k: Any) -> Any:
            seq.append("execute")
            if seq.count("execute") == 1:
                raise OutOfMemoryError(
                    "OOM command not allowed when used memory > 'maxmemory'."
                )
            return await real_execute(self, *a, **k)

        async def recording_ltrim(*a: Any, **k: Any) -> Any:
            seq.append("bare-ltrim")
            return await real_ltrim(*a, **k)

        monkeypatch.setattr(Pipeline, "execute", failing_once)
        monkeypatch.setattr(fake_redis, "ltrim", recording_ltrim)
        await store.push_history(_hentry(999))

        assert seq == ["execute", "bare-ltrim", "execute"]
        # …and the play landed rather than being lost to the refusal.
        assert await fake_redis.llen(store.history_key()) == HISTORY_CACHE_LIMIT
        assert "at maxmemory" in caplog.text
        # The message names what the trim will actually achieve — this branch
        # is the one where it achieves something.
        assert "retrying" in caplog.text

    async def test_an_oom_refusal_at_the_cap_says_the_trim_frees_nothing(
        self,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The steady state, where the self-heal cannot heal.

        push_history LPUSHes and LTRIMs in one transaction, so every list this
        build wrote already sits at exactly the cap and a recovery LTRIM frees
        nothing (measured on redis:7-alpine at maxmemory 3mb: the retry aborted
        again, the play was dropped). Hence no pointless LTRIM round trip on every
        song end, and a warning that says the play is likely lost.
        """
        store = GuildRedisStore(fake_redis, guild_id=42)
        for n in range(HISTORY_CACHE_LIMIT):  # exactly at the cap, not over
            await fake_redis.lpush(store.history_key(), _hentry(n).to_redis())

        seq: list[str] = []
        real_execute = Pipeline.execute
        real_ltrim = fake_redis.ltrim

        async def failing_once(self: Any, *a: Any, **k: Any) -> Any:
            seq.append("execute")
            if seq.count("execute") == 1:
                raise OutOfMemoryError(
                    "OOM command not allowed when used memory > 'maxmemory'."
                )
            return await real_execute(self, *a, **k)

        async def recording_ltrim(*a: Any, **k: Any) -> Any:
            seq.append("bare-ltrim")
            return await real_ltrim(*a, **k)

        monkeypatch.setattr(Pipeline, "execute", failing_once)
        monkeypatch.setattr(fake_redis, "ltrim", recording_ltrim)
        await store.push_history(_hentry(999))

        # No standalone trim: it provably frees nothing at the cap.
        assert seq == ["execute", "execute"]
        assert "would free nothing" in caplog.text
        assert "likely LOST" in caplog.text

    async def test_trimming_the_list_never_affects_the_outbox(
        self, fake_redis: Redis
    ) -> None:
        # The outbox is what makes entries durable; trimming the DISPLAY list
        # must not drop anything on its way to Postgres. With the list capped
        # this is the only copy an un-drained play has.
        store = GuildRedisStore(fake_redis, guild_id=42)
        for n in range(HISTORY_CACHE_LIMIT + 5):
            await store.push_history(_hentry(n))
        assert await outbox_depth(fake_redis) == HISTORY_CACHE_LIMIT + 5
        assert await fake_redis.ttl(HISTORY_OUTBOX_KEY) == -1


class TestGetHistory:
    async def test_returns_up_to_cache_limit_items(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The read is bounded independently of the write-side cap, so a list
        # left long by an older build still reads as one window.
        for i in range(HISTORY_CACHE_LIMIT + 10):
            await fake_redis.lpush(store.history_key(), _hentry(i).to_redis())
        items = await store.get_history()
        assert len(items) == HISTORY_CACHE_LIMIT
        assert all(isinstance(item, HistoryEntry) for item in items)

    async def test_round_trips_push(self, store: GuildRedisStore) -> None:
        await store.push_history(_hentry(1))
        assert await store.get_history() == [_hentry(1)]

    async def test_drops_corrupt_entries(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.lpush(store.history_key(), _hentry(1).to_redis())
        await fake_redis.lpush(store.history_key(), b"not json")
        await fake_redis.lpush(store.history_key(), orjson.dumps([1, 2]))
        await fake_redis.lpush(store.history_key(), orjson.dumps("a bare string"))
        assert await store.get_history() == [_hentry(1)]

    async def test_returns_empty_list_when_missing(
        self, store: GuildRedisStore
    ) -> None:
        items = await store.get_history()
        assert items == []

    async def test_returns_empty_list_on_error(
        self, broken_store: GuildRedisStore
    ) -> None:
        result = await broken_store.get_history()
        assert result == []

    async def test_error_fallback_is_a_fresh_list_per_guild(
        self, broken_store: GuildRedisStore
    ) -> None:
        """Two failing reads must not share one list object. A decorator argument
        is evaluated once at class-body execution, so `@_guild_op(default=[])`
        returns the *same* list to every guild on every failure: one in-place
        mutation poisons "empty history" process-wide.
        """
        other = GuildRedisStore(broken_store.redis, guild_id=1234)

        first = await broken_store.get_history()
        second = await other.get_history()

        assert first == [] and second == []
        assert first is not second, (
            "both guilds got the same list object — the fallback is a shared "
            "mutable default, not a per-call factory"
        )

        first.append(_hentry(99))  # in-place mutation, the poisoning case
        assert await other.get_history() == []


class TestGuildOpDefaults:
    """Structural guard over every @_guild_op default in GuildRedisStore.

    The fallback is a decorator argument, which makes two mistakes cheap and
    invisible: a mutable literal (shared across guilds for the process lifetime)
    and a default contradicting the declared return type. Neither is caught by
    pyright — `default` is typed Any, so `default=None` cannot collapse the TypeVar.
    """

    @staticmethod
    def _decorated() -> list[tuple[str, ast.Call, Optional[str]]]:
        """(method name, the _guild_op call node, return annotation) per method."""
        import src.redis_client as rc_module

        tree = ast.parse(inspect.getsource(rc_module))
        cls = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "GuildRedisStore"
        )
        out: list[tuple[str, ast.Call, Optional[str]]] = []
        for node in cls.body:
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "_guild_op"
                ):
                    returns = (
                        ast.unparse(node.returns) if node.returns is not None else None
                    )
                    out.append((node.name, dec, returns))
        return out

    def test_finds_every_decorated_method(self) -> None:
        """Guard against the guard silently matching nothing."""
        assert len(self._decorated()) >= 24

    def test_mutable_defaults_use_a_factory(self) -> None:
        """No `default=` may be a mutable literal — those need default_factory."""
        offenders = []
        for name, call, _ in self._decorated():
            for kw in call.keywords:
                if kw.arg == "default" and isinstance(
                    kw.value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp)
                ):
                    offenders.append(f"{name}: default={ast.unparse(kw.value)}")
        assert not offenders, (
            "mutable @_guild_op default(s) shared across every guild for the "
            f"process lifetime: {offenders}. Use default_factory=... instead."
        )

    def test_every_default_matches_its_return_type(self) -> None:
        """A fallback that contradicts the annotation is a silent type lie."""
        mismatches = []
        for name, call, returns in self._decorated():
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            factory = kwargs.get("default_factory")
            default = kwargs.get("default")
            rendered = ast.unparse(default) if default is not None else None

            if returns is None:
                mismatches.append(f"{name}: no return annotation")
            elif factory is not None:
                # A factory implies a concrete container return, never Optional.
                if returns.startswith("Optional[") or returns == "None":
                    mismatches.append(
                        f"{name}: default_factory with Optional return {returns}"
                    )
            elif returns == "bool":
                if rendered != "False":
                    mismatches.append(f"{name}: -> bool but default={rendered}")
            elif returns == "None" or returns.startswith("Optional["):
                if rendered not in (None, "None"):
                    mismatches.append(f"{name}: -> {returns} but default={rendered}")
            elif rendered == "None":
                # Non-Optional, non-bool returning a None fallback: the caller's
                # annotation promises a value the error path does not deliver.
                mismatches.append(
                    f"{name}: -> {returns} (not Optional) but default=None"
                )
        assert not mismatches, "\n".join(mismatches)


# ── State operations ──────────────────────────────────────────────────────────


class TestSetVolume:
    async def test_writes_volume_to_the_config_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        assert await store.set_volume(0.75) is True
        config = await fake_redis.hgetall(store.config_key())
        assert config[b"volume"] == b"0.75"

    async def test_the_volume_never_expires(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """It used to live in the 24h state hash, so a guild that went a day without
        playing came back at 100% with nothing to explain why."""
        await store.set_volume(1.0)
        assert await fake_redis.ttl(store.config_key()) == -1
        await store.refresh_ttl()
        assert await fake_redis.ttl(store.config_key()) == -1

    async def test_the_legacy_state_field_is_kept_fresh_not_deleted(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """A rollback to a build that reads only :state must still find the current
        value. Deleting the field made `just up <older-sha>` silently reset every
        migrated guild to 100% — no log line, no user-visible cause."""
        await fake_redis.hset(store.state_key(), "volume", "0.10")
        await store.set_volume(0.75)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_the_legacy_write_cannot_leave_the_state_key_immortal(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The legacy write can CREATE the state hash on a guild whose 24h TTL has
        lapsed, and a state key created without an EXPIRE never expires again — a
        permanently non-evictable key that golden rule 12 does not sanction."""
        assert await fake_redis.exists(store.state_key()) == 0
        await store.set_volume(0.5)
        assert await fake_redis.ttl(store.state_key()) > 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        assert await broken_store.set_volume(0.5) is False  # must not raise


class TestSetTimezone:
    """Write half of the planned `-options timezone`. No command calls it yet, so
    these are the only thing holding its contract."""

    async def test_a_real_zone_is_stored(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        assert await store.set_timezone("Europe/London") is True
        assert (await fake_redis.hgetall(store.config_key()))[b"timezone"] == (
            b"Europe/London"
        )

    async def test_the_write_clears_an_expiry_it_finds(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """PERSIST, like every config write. Seeded with a TTL first on purpose:
        asserting `ttl == -1` on a key nothing ever expires passes whether or not
        the PERSIST is there, so it would pin nothing. A real expiry can be left by
        an older build or a hand-edit, and under volatile-lru it makes the key an
        eviction candidate — a setting silently reverting with no log line."""
        await fake_redis.hset(store.config_key(), "debug_mode", "1")
        await fake_redis.expire(store.config_key(), 60)
        assert await fake_redis.ttl(store.config_key()) > 0

        assert await store.set_timezone("Europe/London") is True

        assert await fake_redis.ttl(store.config_key()) == -1

    @pytest.mark.parametrize(
        "name", ["Mars/Olympus", "", "../../../etc/passwd", "zone.tab", "x" * 200]
    )
    async def test_an_unusable_name_is_refused_rather_than_stored(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis, name: str
    ) -> None:
        """This is the first user-typed string that will reach guild:{id}:config —
        PERSISTed, excluded from every TTL path, non-evictable. Storing a bad one
        fails silently: the write succeeds, the command says so, and the guild's
        ETAs stay on the default forever."""
        assert await store.set_timezone(name) is False
        assert await fake_redis.hgetall(store.config_key()) == {}

    async def test_a_refusal_does_not_disturb_an_existing_choice(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_timezone("Europe/London")
        assert await store.set_timezone("Mars/Olympus") is False
        assert (await fake_redis.hgetall(store.config_key()))[b"timezone"] == (
            b"Europe/London"
        )

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        assert await broken_store.set_timezone("UTC") is False  # must not raise


class TestConfigWritesUseTheSchemaEncoder:
    """Every config setter encodes through GuildConfig.to_redis rather than by
    hand, so the wire format lives in one place and a setter cannot drift from what
    from_redis expects. Asserted by round-tripping through the real parser."""

    async def test_each_setter_round_trips_through_from_redis(
        self, store: GuildRedisStore
    ) -> None:
        await store.set_debug_mode(False)
        await store.set_volume(0.25)
        await store.set_timezone("Europe/London")

        config = await store.get_config()

        # False, not None: an explicit opt-out must survive the round trip.
        assert config.debug_mode is False
        assert config.volume == 0.25
        assert config.timezone == "Europe/London"

    async def test_a_setter_writes_only_its_own_field(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """to_redis omits unset fields, which is what makes per-field setters safe:
        a setter that wrote a whole config would clobber the other two with the
        defaults of whatever it happened to construct."""
        await store.set_debug_mode(True)
        assert set(await fake_redis.hgetall(store.config_key())) == {b"debug_mode"}


class TestReadGuildConfigs:
    """The multi-guild read hydration uses. Its contract is the return SHAPE: a
    guild is present only if its read happened."""

    async def test_reads_every_guild_across_batch_boundaries(
        self, fake_redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain per-guild fan-out silently failed every guild past the pool's
        connection cap, so this batches. The batch loop's boundaries are the part
        that can drop a guild."""
        monkeypatch.setattr(redis_client, "_CONFIG_READ_BATCH", 2)
        ids = list(range(1, 8))  # 7 guilds over batches of 2 — last batch partial
        for guild_id in ids:
            await GuildRedisStore(fake_redis, guild_id).set_debug_mode(True)

        configs = await redis_client.read_guild_configs(fake_redis, ids)

        assert sorted(configs) == ids
        assert all(c.debug_mode is True for c in configs.values())

    async def test_the_work_is_actually_batched(
        self, fake_redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """fakeredis has no connection cap, so the reason for batching is invisible
        here — this pins the shape instead. One pipeline per batch, never one
        awaited command per guild: the real pool RAISES rather than queueing once
        its connections are in use, so a per-guild fan-out fails every guild past
        the cap. tests/test_redis_integration.py proves that against a real server.
        """
        monkeypatch.setattr(redis_client, "_CONFIG_READ_BATCH", 2)
        real = fake_redis.pipeline
        with patch.object(fake_redis, "pipeline", side_effect=real) as pipe:
            await redis_client.read_guild_configs(fake_redis, list(range(5)))
        assert pipe.call_count == 3  # ceil(5 / 2)

    async def test_a_guild_with_nothing_stored_is_present_and_all_unset(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """Present-but-unset is a real answer: "this guild never chose"."""
        configs = await redis_client.read_guild_configs(fake_redis, [99])
        assert configs[99] == GuildConfig()

    async def test_a_failed_read_is_reported_by_OMISSION_not_a_zero_value(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """The distinction the whole function exists for. Handing back an all-unset
        GuildConfig would be indistinguishable from "never chose", and the caller
        caches that — which is how a Redis blink came to DELETE stored settings."""
        with patch.object(fake_redis, "pipeline", side_effect=RuntimeError("down")):
            configs = await redis_client.read_guild_configs(fake_redis, [1, 2])
        assert configs == {}

    async def test_one_failed_batch_does_not_lose_the_others(
        self, fake_redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(redis_client, "_CONFIG_READ_BATCH", 2)
        for guild_id in (1, 2, 3, 4):
            await GuildRedisStore(fake_redis, guild_id).set_debug_mode(True)
        real = fake_redis.pipeline
        calls = {"n": 0}

        def flaky(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("down")
            return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        with patch.object(fake_redis, "pipeline", side_effect=flaky):
            configs = await redis_client.read_guild_configs(fake_redis, [1, 2, 3, 4])

        assert sorted(configs) == [3, 4]

    async def test_no_guilds_issues_no_commands(
        self, fake_redis: aioredis.Redis
    ) -> None:
        with patch.object(fake_redis, "pipeline", side_effect=AssertionError) as pipe:
            assert await redis_client.read_guild_configs(fake_redis, []) == {}
        pipe.assert_not_called()


class TestMigrateVolume:
    """Seeding config from the legacy field, which must never overwrite."""

    async def test_seeds_config_when_nothing_is_stored(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        assert await store.migrate_volume(0.30) is True
        assert (await fake_redis.hgetall(store.config_key()))[b"volume"] == b"0.3"

    async def test_never_overwrites_a_value_already_there(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The whole reason this is HSETNX. Restore reads its snapshot and writes it
        back an arbitrary number of awaits later, so a -volume that landed in that
        window would otherwise be destroyed by the older value — durably, while the
        user is being told the new one took."""
        await store.set_volume(0.50)
        assert await store.migrate_volume(0.30) is True
        assert (await fake_redis.hgetall(store.config_key()))[b"volume"] == b"0.5"

    async def test_the_seeded_volume_never_expires(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.migrate_volume(0.30)
        assert await fake_redis.ttl(store.config_key()) == -1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        assert await broken_store.migrate_volume(0.5) is False  # must not raise


class TestGetGuildState:
    async def test_returns_typed_snapshot(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"volume", b"0.5")
        await fake_redis.hset(store.state_key(), b"current_song_url", b"https://x")
        state = await store.get_guild_state()
        assert state == GuildStateData(volume=0.5, current_song_url="https://x")

    async def test_returns_zero_value_snapshot_when_missing(
        self, store: GuildRedisStore
    ) -> None:
        state = await store.get_guild_state()
        assert state == GuildStateData()

    async def test_returns_none_on_error_not_defaults(
        self, broken_store: GuildRedisStore
    ) -> None:
        # None (read failed) must be distinguishable from GuildStateData()
        # (nothing stored) — restore_guild() relies on this to avoid silently
        # skipping recovery during a Redis outage.
        result = await broken_store.get_guild_state()
        assert result is None


class TestGetPlaybackSnapshot:
    async def test_returns_state_and_queue_together(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"current_song_url", b"https://x")
        await fake_redis.rpush(
            store.queue_key(), _entry(1).to_redis(), _entry(2).to_redis()
        )
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert snap.state.current_song_url == "https://x"
        assert snap.queue == (_entry(1), _entry(2))
        assert snap.pending_count == 2
        assert snap.has_restorable_playback

    async def test_empty_guild_yields_empty_snapshot_not_none(
        self, store: GuildRedisStore
    ) -> None:
        snap = await store.get_playback_snapshot()
        assert snap == GuildPlaybackSnapshot(state=GuildStateData())
        assert snap is not None
        assert not snap.has_restorable_playback

    async def test_corrupt_queue_entries_dropped(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(
            store.queue_key(), b"not json", _entry(1).to_redis(), b"{}"
        )
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert snap.queue == (_entry(1),)

    async def test_includes_now_playing_and_history(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        await store.push_history(_hentry(1))
        await store.push_history(_hentry(2))
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert snap.now_playing is not None
        assert snap.now_playing.title == "Song"
        assert snap.history == (_hentry(2), _hentry(1))  # newest first

    async def test_history_read_bounded_at_cache_limit(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The list is unbounded; the snapshot read must stay O(cache limit).
        for i in range(HISTORY_CACHE_LIMIT + 10):
            await store.push_history(_hentry(i))
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert len(snap.history) == HISTORY_CACHE_LIMIT

    async def test_empty_guild_has_no_now_playing_and_empty_history(
        self, store: GuildRedisStore
    ) -> None:
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert snap.now_playing is None
        assert snap.history == ()

    async def test_corrupt_history_entries_dropped(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.lpush(store.history_key(), _hentry(1).to_redis())
        await fake_redis.lpush(store.history_key(), b"not json")
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert snap.history == (_hentry(1),)

    async def test_returns_none_on_error(self, broken_store: GuildRedisStore) -> None:
        assert await broken_store.get_playback_snapshot() is None

    async def test_single_pipeline_round_trip(self, fake_redis: aioredis.Redis) -> None:
        """State HGETALL and queue LRANGE ride one pipeline execute()."""
        store = GuildRedisStore(fake_redis, guild_id=42)
        real_pipeline = fake_redis.pipeline
        execute_counts = []

        def counting_pipeline(*args: Any, **kwargs: Any) -> Any:
            pipe = real_pipeline(*args, **kwargs)
            original_execute = pipe.execute

            async def counted_execute() -> Any:
                execute_counts.append(1)
                return await original_execute()

            mocked(pipe).execute = counted_execute
            return pipe

        fake_redis.pipeline = counting_pipeline
        try:
            snap = await store.get_playback_snapshot()
        finally:
            fake_redis.pipeline = real_pipeline
        assert snap is not None
        assert len(execute_counts) == 1


class TestGetRecoveryGate:
    async def test_returns_state_and_queue_length(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"current_song_url", b"https://x")
        await fake_redis.rpush(
            store.queue_key(), _entry(1).to_redis(), _entry(2).to_redis()
        )
        gate = await store.get_recovery_gate()
        assert gate is not None
        assert gate.state.current_song_url == "https://x"
        assert gate.pending_count == 2
        assert gate.has_restorable_playback

    async def test_empty_guild_yields_zero_gate_not_none(
        self, store: GuildRedisStore
    ) -> None:
        gate = await store.get_recovery_gate()
        assert gate == GuildRecoveryGate(state=GuildStateData())
        assert gate is not None
        assert gate.pending_count == 0
        assert not gate.has_restorable_playback

    async def test_does_not_transfer_queue_contents(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The gate reads LLEN, never LRANGE, so the queue payload never rides
        the wire on the recovery path."""
        real_lrange = fake_redis.lrange
        lrange_keys = []

        async def spy_lrange(key: str, *args: Any, **kwargs: Any) -> Any:
            lrange_keys.append(key)
            return await real_lrange(key, *args, **kwargs)

        await fake_redis.rpush(store.queue_key(), _entry(1).to_redis())
        mocked(fake_redis).lrange = spy_lrange
        try:
            gate = await store.get_recovery_gate()
        finally:
            fake_redis.lrange = real_lrange
        assert gate is not None and gate.pending_count == 1
        assert store.queue_key() not in lrange_keys

    async def test_crashed_song_makes_empty_queue_restorable(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"current_song_url", b"https://x")
        gate = await store.get_recovery_gate()
        assert gate is not None
        assert gate.pending_count == 0
        assert gate.has_restorable_playback  # crashed song, empty queue

    async def test_returns_none_on_error(self, broken_store: GuildRedisStore) -> None:
        assert await broken_store.get_recovery_gate() is None

    async def test_single_pipeline_round_trip(self, fake_redis: aioredis.Redis) -> None:
        """State HGETALL and queue LLEN ride one pipeline execute()."""
        store = GuildRedisStore(fake_redis, guild_id=42)
        real_pipeline = fake_redis.pipeline
        execute_counts = []

        def counting_pipeline(*args: Any, **kwargs: Any) -> Any:
            pipe = real_pipeline(*args, **kwargs)
            original_execute = pipe.execute

            async def counted_execute() -> Any:
                execute_counts.append(1)
                return await original_execute()

            mocked(pipe).execute = counted_execute
            return pipe

        fake_redis.pipeline = counting_pipeline
        try:
            gate = await store.get_recovery_gate()
        finally:
            fake_redis.pipeline = real_pipeline
        assert gate is not None
        assert len(execute_counts) == 1


# ── TTL management ────────────────────────────────────────────────────────────


class TestRefreshTtl:
    async def test_refreshes_ttl_managed_guild_keys(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        managed_keys = [
            store.queue_key(),
            store.state_key(),
            store.now_playing_key(),
        ]
        for key in managed_keys:
            await fake_redis.set(key, b"x")
            await fake_redis.expire(key, 10)  # short initial TTL
        await store.refresh_ttl()
        for key in managed_keys:
            ttl = await fake_redis.ttl(key)
            assert ttl > 1000  # refreshed to GUILD_TTL

    async def test_never_rearms_history_expiry(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The history key is persistent (unbounded retention) — refresh_ttl
        # re-arming an idle expiry on it would silently destroy full history.
        await fake_redis.lpush(store.history_key(), b'"entry"')
        await store.refresh_ttl()
        assert await fake_redis.ttl(store.history_key()) == -1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.refresh_ttl()  # must not raise


# ── Connection persistence ────────────────────────────────────────────────────


class TestSetConnection:
    async def test_persists_channel_ids(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_connection(111, 222)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"voice_channel_id"] == b"111"
        assert state[b"text_channel_id"] == b"222"

    async def test_sets_ttl(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_connection(111, 222)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.set_connection(1, 2)  # must not raise


class TestConnectionViaGuildState:
    async def test_returns_channel_ids_when_set(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"voice_channel_id", b"111")
        await fake_redis.hset(store.state_key(), b"text_channel_id", b"222")
        state = await store.get_guild_state()
        assert state is not None
        assert state.voice_channel_id == 111
        assert state.text_channel_id == 222
        assert state.has_active_connection

    async def test_no_active_connection_when_not_set(
        self, store: GuildRedisStore
    ) -> None:
        state = await store.get_guild_state()
        assert state is not None
        assert state.voice_channel_id is None
        assert state.text_channel_id is None
        assert not state.has_active_connection


class TestClearConnection:
    async def test_removes_all_transient_fields(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """clear_connection removes all transient state fields and the now-playing hash."""
        transient_fields = {
            b"voice_channel_id": b"111",
            b"text_channel_id": b"222",
            b"current_song_url": b"https://yt.com/v=1",
            b"current_song_title": b"Test Song",
            b"current_song_duration": b"210",
            b"current_song_uploader": b"Some Channel",
            b"current_song_requester_id": b"42",
            b"last_author_id": b"999",
            b"play_start_epoch": b"1000.0",
            b"total_pause_seconds": b"30",
            b"pause_start_epoch": b"1200.0",
        }
        for field, value in transient_fields.items():
            await fake_redis.hset(store.state_key(), field, value)
        await fake_redis.hset(store.now_playing_key(), b"title", b"Some Song")

        await store.clear_connection()

        state = await fake_redis.hgetall(store.state_key())
        for field in transient_fields:
            assert field not in state, f"expected {field!r} to be cleared"
        np_data = await fake_redis.hgetall(store.now_playing_key())
        assert np_data == {}

    async def test_preserves_non_transient_fields(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """clear_connection must not delete persistent fields like volume."""
        await fake_redis.hset(store.state_key(), b"volume", b"0.8")
        await store.clear_connection()
        state = await fake_redis.hgetall(store.state_key())
        assert state.get(b"volume") == b"0.8"

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.clear_connection()  # must not raise


# ── Recovery lock ─────────────────────────────────────────────────────────────


class TestRecoveryLock:
    async def test_acquire_returns_true_first_time(
        self, store: GuildRedisStore
    ) -> None:
        acquired = await store.acquire_recovery_lock()
        assert acquired is True

    async def test_acquire_returns_false_when_already_held(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.set(store._recovery_lock_key(), "1", nx=True, ex=60)
        acquired = await store.acquire_recovery_lock()
        assert acquired is False

    async def test_acquire_returns_false_on_error(
        self, broken_store: GuildRedisStore
    ) -> None:
        result = await broken_store.acquire_recovery_lock()
        assert result is False

    async def test_release_deletes_a_lock_this_store_acquired(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        assert await store.acquire_recovery_lock() is True
        await store.release_recovery_lock()
        assert await fake_redis.get(store._recovery_lock_key()) is None

    async def test_acquire_stores_a_unique_token_per_acquisition(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.acquire_recovery_lock()
        first = store._recovery_lock_token
        assert first is not None
        assert await fake_redis.get(store._recovery_lock_key()) == first.encode()

        await store.release_recovery_lock()
        await store.acquire_recovery_lock()
        assert store._recovery_lock_token != first

    async def test_release_does_not_delete_a_lock_another_acquirer_holds(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """Attempt #1 acquires; its lease expires mid-recovery; #2 acquires; #1
        finishes and releases. #1 must NOT delete #2's lock — that would let a #3
        acquire while #2 is still restoring, which is the double restore the lock
        exists to prevent. Both are restore_guild tasks, not separate processes."""
        store_a = store
        assert await store_a.acquire_recovery_lock() is True

        # A's lock expires and B takes it (simulated: overwrite with B's value).
        await fake_redis.delete(store_a._recovery_lock_key())
        store_b = GuildRedisStore(fake_redis, guild_id=store_a.guild_id)
        assert await store_b.acquire_recovery_lock() is True
        b_token = store_b._recovery_lock_token
        assert b_token is not None

        await store_a.release_recovery_lock()  # A finishes, late

        assert await fake_redis.get(store_a._recovery_lock_key()) == b_token.encode()
        # ...and B can still release its own lock afterwards.
        await store_b.release_recovery_lock()
        assert await fake_redis.get(store_a._recovery_lock_key()) is None

    async def test_release_without_acquire_is_a_noop(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """A store that never won the lock has no business deleting it — that is
        the cross-instance delete, just arriving by a different route."""
        other = GuildRedisStore(fake_redis, guild_id=store.guild_id)
        assert await other.acquire_recovery_lock() is True

        await store.release_recovery_lock()  # never acquired

        assert await fake_redis.get(store._recovery_lock_key()) is not None

    async def test_release_is_idempotent(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The token is consumed on release, so a second release cannot delete a
        lock a later acquirer has since taken."""
        assert await store.acquire_recovery_lock() is True
        await store.release_recovery_lock()

        later = GuildRedisStore(fake_redis, guild_id=store.guild_id)
        assert await later.acquire_recovery_lock() is True

        await store.release_recovery_lock()  # second release from the old owner

        assert await fake_redis.get(store._recovery_lock_key()) is not None

    async def test_release_tolerates_a_lock_that_simply_expired(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        assert await store.acquire_recovery_lock() is True
        await fake_redis.delete(store._recovery_lock_key())  # TTL elapsed
        await store.release_recovery_lock()  # must not raise

    async def test_release_leaves_the_key_when_exec_aborts(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """The compare passed but the key changed hands before EXEC, so the
        transaction aborts and the delete never lands. Driven here by raising
        WatchError directly: whether fakeredis invalidates a WATCH across
        connections is not something to rely on — the real-server proof of this
        path is in the redis tier."""
        assert await store.acquire_recovery_lock() is True
        token = store._recovery_lock_token
        assert token is not None

        real_pipeline = fake_redis.pipeline

        def _aborting_pipeline(*args: Any, **kwargs: Any) -> Any:
            pipe = real_pipeline(*args, **kwargs)
            pipe.execute = AsyncMock(side_effect=WatchError("changed hands"))
            return pipe

        with patch.object(fake_redis, "pipeline", _aborting_pipeline):
            await store.release_recovery_lock()  # must not raise

        assert await fake_redis.get(store._recovery_lock_key()) == token.encode()

    async def test_release_swallows_redis_error(
        self, broken_store: GuildRedisStore
    ) -> None:
        # Set the token directly: broken_store's acquire raises, so it would
        # otherwise return early and never reach the code under test.
        broken_store._recovery_lock_token = "deadbeef"
        await broken_store.release_recovery_lock()  # must not raise

    async def test_ttl_outlasts_the_voice_connect_cap(
        self, store: GuildRedisStore
    ) -> None:
        """_restore_guild's connect(timeout=30.0) caps the whole retry loop, and a
        few Redis reads follow it. The TTL must clear that with room to spare."""
        assert store._RECOVERY_LOCK_TTL >= 60

    async def test_lock_key_includes_guild_id(self, store: GuildRedisStore) -> None:
        assert "123456789" in store._recovery_lock_key()


# ── Spotify token cache ───────────────────────────────────────────────────────


class TestSpotifyTokenCache:
    async def test_token_get_with_ttl_returns_none_on_miss(
        self, fake_redis: aioredis.Redis
    ) -> None:
        result = await spotify_token_get_with_ttl(fake_redis)
        assert result is None

    async def test_token_get_with_ttl_returns_token_and_remaining_life(
        self, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.set("spotify:auth:token", b"my_bearer_token", ex=3570)
        result = await spotify_token_get_with_ttl(fake_redis)
        assert result is not None
        token, ttl = result
        assert token == "my_bearer_token"
        assert 3560 <= ttl <= 3570

    async def test_token_get_with_ttl_returns_none_when_redis_is_none(self) -> None:
        result = await spotify_token_get_with_ttl(None)
        assert result is None

    async def test_token_get_with_ttl_returns_none_for_key_without_expiry(
        self, fake_redis: aioredis.Redis
    ) -> None:
        # TTL of -1 (no expiry) must be treated as unusable, not as a live token.
        await fake_redis.set("spotify:auth:token", b"stale_token")
        result = await spotify_token_get_with_ttl(fake_redis)
        assert result is None

    async def test_token_get_with_ttl_swallows_error(self) -> None:
        bad_redis = MagicMock()
        bad_redis.pipeline = MagicMock(side_effect=ConnectionError("down"))
        result = await spotify_token_get_with_ttl(bad_redis)
        assert result is None

    async def test_token_set_stores_raw_string_with_ttl(
        self, fake_redis: aioredis.Redis
    ) -> None:
        await spotify_token_set(fake_redis, "token_abc", 3600)
        val = await fake_redis.get("spotify:auth:token")
        assert val == b"token_abc"
        ttl = await fake_redis.ttl("spotify:auth:token")
        assert 3560 <= ttl <= 3570  # 3600 - 30 = 3570

    async def test_token_set_skips_cache_for_short_lived_token(
        self, fake_redis: aioredis.Redis
    ) -> None:
        # A margin that *raised* the TTL would serve an expired token to other
        # processes — short-lived tokens must simply not be cached.
        await spotify_token_set(fake_redis, "token", 20)
        assert await fake_redis.get("spotify:auth:token") is None

    async def test_token_set_boundary_just_above_margin(
        self, fake_redis: aioredis.Redis
    ) -> None:
        await spotify_token_set(fake_redis, "token", 31)  # 31 - 30 = 1s
        ttl = await fake_redis.ttl("spotify:auth:token")
        assert ttl == 1

    async def test_token_set_boundary_at_margin_not_written(
        self, fake_redis: aioredis.Redis
    ) -> None:
        await spotify_token_set(fake_redis, "token", 30)  # 30 - 30 = 0 → skip
        assert await fake_redis.get("spotify:auth:token") is None

    async def test_token_set_noop_when_redis_is_none(self) -> None:
        await spotify_token_set(None, "token", 3600)  # must not raise

    async def test_token_set_swallows_error(self) -> None:
        bad_redis = AsyncMock()
        bad_redis.set = AsyncMock(side_effect=ConnectionError("down"))
        await spotify_token_set(bad_redis, "token", 3600)  # must not raise


# ── pop_queue_and_start_song ──────────────────────────────────────────────────


def _current(url: str = "url", title: str = "title", **kwargs: Any) -> SongQueueEntry:
    """The start-transaction carrier: the queue-entry view of the song that is
    about to play (its fields get parked in the state hash as current_song_*)."""
    return SongQueueEntry(
        webpage_url=url,
        title=title,
        requester_id=kwargs.pop("requester_id", None),
        **kwargs,
    )


class TestPopQueueAndStartSong:
    async def test_lpop_removes_first_item_only(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"first", b"second")
        assert await store.pop_queue_and_start_song(_current(), 1000.0) is True
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"second"]

    async def test_a_swallowed_failure_reports_that_the_lpop_did_not_land(
        self,
    ) -> None:
        """@_guild_op logs and returns the default, so this return value is the
        caller's ONLY signal. Defaulted to True it reads as success, the loop skips
        its resync, and the mirror keeps an entry memory already dropped — the next
        start retires the wrong one and a crash inside the drift records a song
        twice."""
        bad_redis = MagicMock()
        bad_redis.pipeline = MagicMock(side_effect=ConnectionError("down"))
        bad_store = GuildRedisStore(cast(aioredis.Redis, bad_redis), guild_id=1)

        assert await bad_store.pop_queue_and_start_song(_current(), 1000.0) is False

    async def test_writes_now_playing_fields_atomically(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(
            _current("https://yt.com/v=1", "Test Song"), 1000.5
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_url"] == b"https://yt.com/v=1"
        assert state[b"current_song_title"] == b"Test Song"
        assert state[b"play_start_epoch"] == b"1000.5"
        assert state[b"total_pause_seconds"] == b"0"

    async def test_writes_duration_uploader_and_requester_id(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(
            _current(duration=210, uploader="Some Channel", requester_id=42),
            1000.0,
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_duration"] == b"210"
        assert state[b"current_song_uploader"] == b"Some Channel"
        assert state[b"current_song_requester_id"] == b"42"

    async def test_omitted_duration_uploader_requester_id_write_empty(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_duration"] == b""
        assert state[b"current_song_uploader"] == b""
        assert state[b"current_song_requester_id"] == b""

    async def test_clears_pause_epoch_on_start(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"999.0")
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert b"pause_start_epoch" not in state

    async def test_writes_interjected_flag(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(interjected=True), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_interjected"] == b"1"

    async def test_interjected_false_writes_empty(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_interjected"] == b""

    async def test_parks_the_enqueue_stamps(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The state hash is the parked queue entry, so a crash-recovered song
        # must find its original enqueue position waiting for it.
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(
            _current(queued_at=1752530000.5, queue_position=3), 1000.0
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_queued_at"] == b"1752530000.5"
        assert state[b"current_song_queue_position"] == b"3"
        restored = GuildStateData.from_redis(cast(dict[bytes, bytes], state))
        assert restored.current_song_queued_at == 1752530000.5
        assert restored.current_song_queue_position == 3

    def test_every_parked_field_is_on_a_clear_list(
        self, store: GuildRedisStore
    ) -> None:
        """The start transaction and the two clear paths must name the same
        fields. A field added to the mapping but to neither tuple survives its
        own song: nothing reads it while `current_song_url` is absent, so it is
        invisible until some later change starts trusting it. Asserted as a
        relationship rather than a list, because enumerating the fields is what
        already failed — `query_source` was added to both and neither had a test.
        """
        parked = set(store._now_playing_state_mapping(_current(), 1000.0))
        assert parked <= {*_TRANSIENT_SONG_FIELDS, *_PLAYBACK_POSITION_FIELDS}

    async def test_parks_the_query_source(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # Only knowable at parse time, so the state hash is the sole place a
        # crash-recovered song can read it back from.
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(
            _current(query_source="spotify.com"), 1000.0
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_query_source"] == b"spotify.com"
        restored = GuildStateData.from_redis(cast(dict[bytes, bytes], state))
        assert restored.current_song_query_source == "spotify.com"

    async def test_parks_the_played_at(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The LPOP in this very transaction destroys the song's queue-list entry,
        # so from here until song end the hash holds the only copy of when the
        # play started. It rides the same MULTI, leaving no window where the song
        # is playing and its start is unrecorded — and it is NOT play_start_epoch,
        # which is backdated by the -ss offset.
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(played_at=1752530000.5), 990.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_played_at"] == b"1752530000.5"
        assert state[b"play_start_epoch"] == b"990.0"
        restored = GuildStateData.from_redis(cast(dict[bytes, bytes], state))
        assert restored.current_song_played_at == 1752530000.5

    async def test_sets_ttl_on_state(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_empty_queue_lpop_is_noop_state_still_written(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """LPOP on an empty list returns nil, but the HSET still runs atomically."""
        await store.pop_queue_and_start_song(
            _current("https://yt.com/v=1", "No Queue Song"), 500.0
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_url"] == b"https://yt.com/v=1"
        assert state[b"current_song_title"] == b"No Queue Song"
        assert state[b"play_start_epoch"] == b"500.0"
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == []

    async def test_now_playing_fields_written_in_same_transaction(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(
            _current(),
            1000.0,
            now_playing=NowPlayingData(title="Song", uploader="Channel"),
        )
        np_data = await fake_redis.hgetall(store.now_playing_key())
        assert np_data[b"title"] == b"Song"
        assert np_data[b"uploader"] == b"Channel"
        ttl = await fake_redis.ttl(store.now_playing_key())
        assert ttl > 0

    async def test_now_playing_untouched_when_fields_omitted(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        assert await fake_redis.exists(store.now_playing_key()) == 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.pop_queue_and_start_song(
            _current(), 1000.0
        )  # must not raise


class TestSetCurrentSongState:
    """Mirrors TestPopQueueAndStartSong minus the LPOP — used for restarting a
    crash-recovered "current song" that was never RPUSHed to the queue list."""

    async def test_writes_now_playing_fields(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_current_song_state(
            _current(
                "https://yt.com/v=1",
                "Test Song",
                duration=210,
                uploader="Some Channel",
                requester_id=42,
            ),
            1000.5,
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_url"] == b"https://yt.com/v=1"
        assert state[b"current_song_title"] == b"Test Song"
        assert state[b"play_start_epoch"] == b"1000.5"
        assert state[b"total_pause_seconds"] == b"0"
        assert state[b"current_song_duration"] == b"210"
        assert state[b"current_song_uploader"] == b"Some Channel"
        assert state[b"current_song_requester_id"] == b"42"

    async def test_does_not_touch_queue(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.rpush(store.queue_key(), b"untouched")
        await store.set_current_song_state(_current(), 1000.0)
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"untouched"]

    async def test_clears_pause_epoch(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"999.0")
        await store.set_current_song_state(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert b"pause_start_epoch" not in state

    async def test_now_playing_fields_written_in_same_transaction(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_current_song_state(
            _current(),
            1000.0,
            now_playing=NowPlayingData(title="Song", uploader="Channel"),
        )
        np_data = await fake_redis.hgetall(store.now_playing_key())
        assert np_data[b"title"] == b"Song"
        assert np_data[b"uploader"] == b"Channel"
        ttl = await fake_redis.ttl(store.now_playing_key())
        assert ttl > 0

    async def test_now_playing_untouched_when_fields_omitted(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_current_song_state(_current(), 1000.0)
        assert await fake_redis.exists(store.now_playing_key()) == 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.set_current_song_state(_current(), 1000.0)  # must not raise


# ── Now-playing operations ────────────────────────────────────────────────────
# (Writes are covered above via now_playing_fields on the start-transaction
#  methods; only the read side has a standalone method.)


class TestNowPlayingOperations:
    async def test_get_now_playing_returns_typed_snapshot(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        data = await store.get_now_playing()
        assert data is not None
        assert data.title == "Song"

    async def test_get_now_playing_returns_none_when_missing(
        self, store: GuildRedisStore
    ) -> None:
        data = await store.get_now_playing()
        assert data is None

    async def test_get_now_playing_returns_none_on_error(
        self, broken_store: GuildRedisStore
    ) -> None:
        result = await broken_store.get_now_playing()
        assert result is None


# ── Playback position tracking ────────────────────────────────────────────────


class TestPlaybackPosition:
    async def test_on_pause_writes_epoch(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.on_pause(1234.5)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"pause_start_epoch"] == b"1234.5"

    async def test_on_pause_sets_ttl(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.on_pause(1234.5)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_on_resume_accumulates_pause_seconds(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # paused at t=1000, resuming at t=1030 → 30s of pause to add
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"1000.0")
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"60")
        await store.on_resume(1030.0)
        state = await fake_redis.hgetall(store.state_key())
        assert float(state[b"total_pause_seconds"]) == 90.0  # 60 + 30
        assert b"pause_start_epoch" not in state

    async def test_on_resume_preserves_fractional_seconds(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """Repeated short pauses must not each lose their fractional second."""
        total = 0.0
        for i in range(10):
            pause_start = 1000.0 + i * 10
            resume_at = pause_start + 4.9
            await fake_redis.hset(
                store.state_key(), b"pause_start_epoch", str(pause_start).encode()
            )
            await store.on_resume(resume_at)
            state = await fake_redis.hgetall(store.state_key())
            total = float(state[b"total_pause_seconds"])
        assert abs(total - 49.0) < 0.01

    async def test_on_resume_clamps_negative_elapsed_pause(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """A backward clock step between pause and resume must not decrease the total."""
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"2000.0")
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"60")
        await store.on_resume(1000.0)  # resume_epoch is *before* pause_start_epoch
        state = await fake_redis.hgetall(store.state_key())
        assert float(state[b"total_pause_seconds"]) == 60.0  # unchanged, not decreased

    async def test_on_resume_no_op_when_not_paused(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"60")
        # no pause_start_epoch set
        await store.on_resume(1030.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state.get(b"total_pause_seconds") == b"60"  # unchanged

    async def test_set_playback_start_writes_epoch_and_resets_pauses(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"999.0")
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"30")
        await store.set_playback_start(5000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"play_start_epoch"] == b"5000.0"
        assert state[b"total_pause_seconds"] == b"0"
        assert b"pause_start_epoch" not in state

    async def test_on_pause_swallows_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.on_pause(1234.5)  # must not raise

    async def test_on_resume_swallows_error(
        self, broken_store: GuildRedisStore
    ) -> None:
        await broken_store.on_resume(1234.5)  # must not raise


class TestClearSongEndState:
    async def test_clears_current_song_fields(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(
            store.state_key(),
            mapping={
                b"current_song_url": b"https://yt.com/v=1",
                b"current_song_title": b"Song",
                b"current_song_duration": b"210",
                b"current_song_uploader": b"Some Channel",
                b"current_song_requester_id": b"42",
                b"current_song_interjected": b"1",
                b"current_song_queued_at": b"1752530000.5",
                b"current_song_queue_position": b"3",
            },
        )
        await store.clear_song_end_state()
        state = await fake_redis.hgetall(store.state_key())
        assert b"current_song_url" not in state
        assert b"current_song_title" not in state
        assert b"current_song_duration" not in state
        assert b"current_song_uploader" not in state
        assert b"current_song_requester_id" not in state
        assert b"current_song_interjected" not in state
        assert b"current_song_queued_at" not in state
        assert b"current_song_queue_position" not in state

    async def test_deletes_now_playing_hash(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        await store.clear_song_end_state()
        assert await fake_redis.exists(store.now_playing_key()) == 0

    async def test_removes_playback_position_fields(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.hset(
            store.state_key(),
            mapping={
                b"play_start_epoch": b"1000.0",
                b"total_pause_seconds": b"5",
                b"pause_start_epoch": b"995.0",
            },
        )
        await store.clear_song_end_state()
        state = await fake_redis.hgetall(store.state_key())
        assert b"play_start_epoch" not in state
        assert b"total_pause_seconds" not in state
        assert b"pause_start_epoch" not in state

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.clear_song_end_state()  # must not raise


class TestHeartbeat:
    async def test_the_write_is_not_wrapped_in_a_transaction(
        self, store: GuildRedisStore
    ) -> None:
        """HSET then EXPIRE on one key needs no atomicity, and unlike every other
        state writer this runs once every few seconds per PLAYING guild — the
        MULTI/EXEC pair measured 12% of the latency and 14% of the bytes for nothing.
        """
        spy = MagicMock(wraps=store.redis.pipeline)
        with patch.object(store.redis, "pipeline", spy):
            await store.heartbeat(1.0, 2.0)
        assert spy.call_args.kwargs.get("transaction") is False

    async def test_writes_both_position_fields(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.heartbeat(42.5, 1700000000.0)
        raw = await fake_redis.hgetall(store.state_key())
        assert raw[b"last_position_secs"] == b"42.500"
        assert raw[b"last_heartbeat_epoch"] == b"1700000000.0"

    async def test_the_position_is_written_without_a_float_error_tail(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """position_secs accumulates 20ms frames, so repr renders 13.5% of values
        as '153.42000000000002'. The reader truncates to whole seconds, so the tail
        is noise on every write."""
        await store.heartbeat(7671 * 0.02, 1700000000.0)  # 153.42000000000002

        raw = await fake_redis.hgetall(store.state_key())
        assert raw[b"last_position_secs"] == b"153.420"

    async def test_refreshes_the_state_ttl(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """Otherwise a long song on an otherwise-idle guild could let the state
        key expire mid-playback."""
        await fake_redis.hset(store.state_key(), b"x", b"1")
        await fake_redis.expire(store.state_key(), 5)
        await store.heartbeat(1.0, 2.0)
        assert await fake_redis.ttl(store.state_key()) > 5

    async def test_negative_position_is_floored(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.heartbeat(-3.0, 1.0)
        raw = await fake_redis.hgetall(store.state_key())
        assert float(raw[b"last_position_secs"]) == 0.0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.heartbeat(1.0, 2.0)  # must not raise

    async def test_start_transaction_seeds_the_heartbeat_from_start_offset(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """Closes the window before the first tick — and makes a crash-recovered
        song that crashes again immediately resume at its -ss offset, not 0:00."""
        await store.pop_queue_and_start_song(_entry(1), 1000.0, start_offset=30.0)
        raw = await fake_redis.hgetall(store.state_key())
        assert float(raw[b"last_position_secs"]) == 30.0

    async def test_start_transaction_seeds_zero_without_an_offset(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.pop_queue_and_start_song(_entry(1), 1000.0)
        raw = await fake_redis.hgetall(store.state_key())
        assert float(raw[b"last_position_secs"]) == 0.0

    async def test_clear_song_end_state_wipes_the_heartbeat(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        """A stale position outliving its song would be read as the next
        crash's resume point."""
        await store.heartbeat(42.0, 1.0)
        await store.clear_song_end_state()
        raw = await fake_redis.hgetall(store.state_key())
        assert b"last_position_secs" not in raw
        assert b"last_heartbeat_epoch" not in raw


class TestGuildConfigStore:
    """The durable half of the per-guild debug toggle."""

    async def test_a_stored_choice_survives_a_round_trip(self, fake_redis: Any) -> None:
        store = GuildRedisStore(fake_redis, 42)
        assert await store.set_debug_mode(True) is True
        assert (await store.get_config()).debug_mode is True
        assert await store.set_debug_mode(False) is True
        assert (await store.get_config()).debug_mode is False

    async def test_a_guild_that_never_chose_reads_unset(self, fake_redis: Any) -> None:
        """Not False. Unset means "follow the host default", which is what lets
        DEBUG_MODE still move a guild that has no opinion."""
        assert (await GuildRedisStore(fake_redis, 7).get_config()).debug_mode is None

    async def test_the_config_key_carries_no_ttl(self, fake_redis: Any) -> None:
        """Under volatile-lru only TTL-bearing keys are eviction candidates, so an
        expiring config is a setting that reverts with no log line and no error."""
        store = GuildRedisStore(fake_redis, 42)
        await store.set_debug_mode(True)
        assert await fake_redis.ttl(store.config_key()) == -1

    async def test_a_ttl_refresh_does_not_reach_the_config_key(
        self, fake_redis: Any
    ) -> None:
        """refresh_ttl sweeps the runtime keys. Config must not be swept in with
        them — that is the whole reason it is not a field on guild:{id}:state."""
        store = GuildRedisStore(fake_redis, 42)
        await store.set_debug_mode(True)
        await store.refresh_ttl()
        assert await fake_redis.ttl(store.config_key()) == -1

    async def test_a_write_failure_is_reported_rather_than_swallowed(
        self, fake_redis: Any
    ) -> None:
        """@_guild_op turns the raise into False, and the command turns that False
        into "this applies until restart" instead of claiming a durable change."""
        store = GuildRedisStore(fake_redis, 42)
        with patch.object(
            fake_redis, "pipeline", side_effect=RuntimeError("redis down")
        ):
            assert await store.set_debug_mode(True) is False

    async def test_an_unreachable_redis_reads_as_unset(self, fake_redis: Any) -> None:
        """Degrades to the host default rather than to an arbitrary one."""
        store = GuildRedisStore(fake_redis, 42)
        with patch.object(
            fake_redis, "hgetall", side_effect=RuntimeError("redis down")
        ):
            assert (await store.get_config()).debug_mode is None

    async def test_a_timezone_round_trips(self, fake_redis: Any) -> None:
        store = GuildRedisStore(fake_redis, 42)
        assert await store.set_timezone("Europe/London") is True
        config = await store.get_config()
        assert config.timezone == "Europe/London"
        assert config.tzinfo() == ZoneInfo("Europe/London")

    async def test_the_timezone_never_expires(self, fake_redis: Any) -> None:
        store = GuildRedisStore(fake_redis, 42)
        await store.set_timezone("Europe/London")
        assert await fake_redis.ttl(store.config_key()) == -1

    async def test_settings_do_not_clobber_each_other(self, fake_redis: Any) -> None:
        """Three fields, one hash: each write is an HSET of its own field, so
        setting a timezone must not drop the guild's debug choice or volume."""
        store = GuildRedisStore(fake_redis, 42)
        await store.set_debug_mode(True)
        await store.set_volume(0.4)
        await store.set_timezone("Europe/London")
        config = await store.get_config()
        assert config.debug_mode is True
        assert config.volume == 0.4
        assert config.timezone == "Europe/London"

    async def test_clearing_removes_the_key(self, fake_redis: Any) -> None:
        store = GuildRedisStore(fake_redis, 42)
        await store.set_debug_mode(True)
        assert await store.clear_config() is True
        assert await fake_redis.exists(store.config_key()) == 0
