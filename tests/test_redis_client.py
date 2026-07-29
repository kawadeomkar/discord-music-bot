"""Tests for src/redis_client.py — connection lifecycle, cache helpers, and GuildRedisStore."""

import ast
import inspect
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.guild_state import (
    GuildPlaybackSnapshot,
    GuildRecoveryGate,
    GuildStateData,
    HistoryEntry,
    NowPlayingData,
    SongQueueEntry,
)
from tests.helpers import mocked
from src import redis_client
from src.redis_client import (
    DRAINER_LEASE_KEY,
    DRAINER_LEASE_MS,
    HISTORY_CACHE_LIMIT,
    HISTORY_OUTBOX_KEY,
    GuildRedisStore,
    cache_get,
    cache_set,
    close_redis_pool,
    create_redis_pool,
    get_redis,
    hold_drainer_lease,
    outbox_depth,
    peek_outbox_oldest,
    release_drainer_lease,
    retire_outbox,
    spotify_token_get_with_ttl,
    spotify_token_set,
    trim_outbox_oldest,
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
        """The invariant that made the old config a silent no-op, pinned here
        because nothing else can catch it: redis-py's ConnectionError and
        TimeoutError derive from RedisError, NOT from the builtins of the same
        name — so `retry_on_error=[ConnectionError, TimeoutError]` (the
        builtins) matched nothing redis-py ever raises."""
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
        """End of the chain: a connection built by this pool must treat a
        redis-py ConnectionError as retryable. Asserted on a real Connection
        rather than on kwargs, because redis-py merges `retry_on_error` into
        the Retry object at construction time — that merge is the step the old
        config silently no-opped."""
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

    async def test_no_trim_history_is_unbounded(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The Redis list is the source of truth for ALL played songs — a trim
        # here would silently discard history (docs/HISTORY_OVERHAUL_PLAN.md §4).
        for i in range(HISTORY_CACHE_LIMIT + 5):
            await store.push_history(_hentry(i))
        items = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(items) == HISTORY_CACHE_LIMIT + 5

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


class TestPushHistoryOutbox:
    async def test_every_push_reaches_the_outbox(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # Unconditional: the archive is a required tier, so there is always a
        # drainer behind the outbox and no shape of push_history that skips it.
        await store.push_history(_hentry(1))
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1

    async def test_outbox_gets_same_wire_bytes(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        await store.push_history(_hentry(1))
        assert await fake_redis.lrange(HISTORY_OUTBOX_KEY, 0, -1) == [
            _hentry(1).to_redis()
        ]

    async def test_display_leg_unchanged_by_outbox_flag(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # Phase A: the display list still gets the entry, untrimmed, PERSISTed.
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
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 2

    async def test_outbox_key_has_no_ttl(
        self, store: GuildRedisStore, fake_redis: Redis
    ) -> None:
        # Not-yet-durable entries must never be eviction candidates under
        # volatile-lru — the same property that protects history today.
        await store.push_history(_hentry(1))
        assert await fake_redis.ttl(HISTORY_OUTBOX_KEY) == -1

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.push_history(_hentry(1))  # must not raise


class TestOutboxDrainHelpers:
    async def _push(self, fake_redis: Redis, *ns: int) -> None:
        store = GuildRedisStore(fake_redis, guild_id=42)
        for n in ns:
            await store.push_history(_hentry(n))

    async def test_peek_returns_oldest_first(self, fake_redis: Redis) -> None:
        await self._push(fake_redis, 1, 2, 3)
        raw = await peek_outbox_oldest(fake_redis, 10)
        assert raw == [
            _hentry(1).to_redis(),
            _hentry(2).to_redis(),
            _hentry(3).to_redis(),
        ]

    async def test_peek_caps_at_count_and_leaves_entries_in_place(
        self, fake_redis: Redis
    ) -> None:
        await self._push(fake_redis, 1, 2, 3)
        raw = await peek_outbox_oldest(fake_redis, 2)
        assert raw == [_hentry(1).to_redis(), _hentry(2).to_redis()]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 3  # non-destructive

    async def test_peek_empty_outbox(self, fake_redis: Redis) -> None:
        assert await peek_outbox_oldest(fake_redis, 10) == []

    async def test_retire_drops_oldest_only(self, fake_redis: Redis) -> None:
        await self._push(fake_redis, 1, 2, 3)
        await retire_outbox(fake_redis, 2)
        assert await peek_outbox_oldest(fake_redis, 10) == [_hentry(3).to_redis()]

    async def test_retire_zero_is_noop(self, fake_redis: Redis) -> None:
        await self._push(fake_redis, 1)
        await retire_outbox(fake_redis, 0)
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1

    async def test_retire_never_touches_concurrent_head_pushes(
        self, fake_redis: Redis
    ) -> None:
        # A push landing between peek and retire goes to the HEAD; RPOP from
        # the tail must retire only what was peeked.
        await self._push(fake_redis, 1, 2)
        peeked = await peek_outbox_oldest(fake_redis, 10)
        await self._push(fake_redis, 3)  # concurrent new entry
        await retire_outbox(fake_redis, len(peeked))
        assert await peek_outbox_oldest(fake_redis, 10) == [_hentry(3).to_redis()]

    async def test_depth(self, fake_redis: Redis) -> None:
        assert await outbox_depth(fake_redis) == 0
        await self._push(fake_redis, 1, 2)
        assert await outbox_depth(fake_redis) == 2

    async def test_helpers_raise_on_redis_error(self) -> None:
        # Unlike the cache helpers, errors must PROPAGATE — the drainer's
        # backoff loop is the handler, and a swallowed error would read as an
        # empty outbox and silently stall the drain.
        dead = MagicMock()
        dead.lrange = AsyncMock(side_effect=aioredis.ConnectionError("down"))
        dead.rpop = AsyncMock(side_effect=aioredis.ConnectionError("down"))
        dead.llen = AsyncMock(side_effect=aioredis.ConnectionError("down"))
        with pytest.raises(aioredis.ConnectionError):
            await peek_outbox_oldest(dead, 10)
        with pytest.raises(aioredis.ConnectionError):
            await retire_outbox(dead, 1)
        with pytest.raises(aioredis.ConnectionError):
            await outbox_depth(dead)

    async def test_trim_drops_oldest(self, fake_redis: Redis) -> None:
        # The opt-in cap's mechanism. Same shape as retire, different meaning:
        # these entries never reached Postgres.
        await self._push(fake_redis, 1, 2, 3)
        await trim_outbox_oldest(fake_redis, 2)
        assert await peek_outbox_oldest(fake_redis, 10) == [_hentry(3).to_redis()]

    async def test_trim_zero_is_noop(self, fake_redis: Redis) -> None:
        await self._push(fake_redis, 1)
        await trim_outbox_oldest(fake_redis, 0)
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1

    async def test_trim_pops_in_bounded_slices(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REGRESSION: one `RPOP key <count>` for the whole drop. RPOP with a
        count returns what it popped, so at 490k entries the 206 MB / 5.3s
        reply blew redis-py's default socket timeout and retry_on_timeout
        re-issued the destructive pop — emptying a 500k outbox while logging
        "dropped 490,000". Each command must stay bounded no matter how far
        the backlog ran."""
        monkeypatch.setattr(redis_client, "_TRIM_SLICE", 2)
        await self._push(fake_redis, *range(1, 8))  # 7 entries
        calls: list[int] = []
        real_rpop = fake_redis.rpop

        async def counting_rpop(name: str, count: int) -> Any:
            calls.append(count)
            return await real_rpop(name, count)

        monkeypatch.setattr(fake_redis, "rpop", counting_rpop)
        await trim_outbox_oldest(fake_redis, 5)
        assert calls == [2, 2, 1]  # never one 5-wide pop
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 2
        # And it dropped the OLDEST 5, leaving the two newest.
        assert await peek_outbox_oldest(fake_redis, 10) == [
            _hentry(6).to_redis(),
            _hentry(7).to_redis(),
        ]

    async def test_trim_stops_when_the_outbox_empties_underneath_it(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A concurrent drain can retire the tail between slices; the loop must
        # notice the short/empty reply rather than spinning on a gone key.
        monkeypatch.setattr(redis_client, "_TRIM_SLICE", 2)
        await self._push(fake_redis, 1, 2)
        await trim_outbox_oldest(fake_redis, 1000)
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0


class TestDrainerLease:
    async def test_acquire_then_renew_by_the_same_holder(
        self, fake_redis: Redis
    ) -> None:
        assert await hold_drainer_lease(fake_redis, "a") is True
        assert await hold_drainer_lease(fake_redis, "a") is True  # renew
        assert await fake_redis.get(DRAINER_LEASE_KEY) == b"a"

    async def test_second_holder_is_refused(self, fake_redis: Redis) -> None:
        assert await hold_drainer_lease(fake_redis, "a") is True
        assert await hold_drainer_lease(fake_redis, "b") is False
        assert await fake_redis.get(DRAINER_LEASE_KEY) == b"a"

    async def test_renew_extends_the_ttl(self, fake_redis: Redis) -> None:
        await hold_drainer_lease(fake_redis, "a")
        await fake_redis.pexpire(DRAINER_LEASE_KEY, 50)
        assert await hold_drainer_lease(fake_redis, "a") is True
        ttl = await fake_redis.pttl(DRAINER_LEASE_KEY)
        assert ttl > DRAINER_LEASE_MS // 2

    async def test_lapsed_lease_is_reacquirable_by_anyone(
        self, fake_redis: Redis
    ) -> None:
        await hold_drainer_lease(fake_redis, "a")
        await fake_redis.delete(DRAINER_LEASE_KEY)  # stand-in for expiry
        assert await hold_drainer_lease(fake_redis, "b") is True

    async def test_release_only_by_the_owner(self, fake_redis: Redis) -> None:
        await hold_drainer_lease(fake_redis, "a")
        await release_drainer_lease(fake_redis, "b")  # not ours
        assert await fake_redis.get(DRAINER_LEASE_KEY) == b"a"
        await release_drainer_lease(fake_redis, "a")
        assert await fake_redis.get(DRAINER_LEASE_KEY) is None

    async def test_release_on_a_missing_lease_is_safe(self, fake_redis: Redis) -> None:
        await release_drainer_lease(fake_redis, "a")

    async def test_an_aborted_watch_is_retried_and_re_reads(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The WATCH-abort branch, forced.

        It cannot be reached naturally under fakeredis: measured against real
        redis:7-alpine, TTL-expiry of a WATCHed key DOES abort EXEC, and under
        fakeredis it does NOT. That divergence hid the one branch that makes
        renewal safe — a drainer stalled past DRAINER_LEASE_MS whose lease
        lapses between our GET and our EXEC. On real Redis the transaction
        aborts and only the retry re-reads and reports the truth; without the
        retry the caller is told it still owns a lease somebody else can now
        take, which is the two-drainers state the lease exists to prevent.
        """
        await fake_redis.set(DRAINER_LEASE_KEY, "a", px=DRAINER_LEASE_MS)
        calls = {"n": 0}
        real_execute_transaction = Pipeline._execute_transaction

        # Patched BELOW execute(), not over it: execute() ends in
        # `finally: await self.reset()`, and that reset is exactly what lets the
        # retry re-WATCH — without it the second lap dies on "Cannot issue a
        # WATCH after a MULTI". Replacing execute() outright would skip it and
        # test a pipeline redis-py never actually hands us.
        async def aborting_transaction(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                # What real Redis does when the watched key changed under us —
                # plus the lapse that made it change.
                await fake_redis.delete(DRAINER_LEASE_KEY)
                raise aioredis.WatchError("watched key changed")
            return await real_execute_transaction(self, *args, **kwargs)

        monkeypatch.setattr(Pipeline, "_execute_transaction", aborting_transaction)
        # Must NOT report continued ownership of a lease that is now gone.
        assert await hold_drainer_lease(fake_redis, "a") is False
        assert calls["n"] == 1  # retry short-circuited at the re-read, no 2nd EXEC

    async def test_not_the_owner_releases_the_watch(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Leaving a WATCH set on the connection would make the NEXT transaction
        # on it abort for a key this call had no business watching.
        await fake_redis.set(DRAINER_LEASE_KEY, "a", px=DRAINER_LEASE_MS)
        unwatched = MagicMock()
        real_unwatch = Pipeline.unwatch

        async def counting_unwatch(self: Any) -> Any:
            unwatched()
            return await real_unwatch(self)

        monkeypatch.setattr(Pipeline, "unwatch", counting_unwatch)
        assert await hold_drainer_lease(fake_redis, "b") is False
        unwatched.assert_called_once()


class TestPushHistoryAtomicity:
    """The display push and the outbox push must ride ONE transactional
    pipeline. Every existing test checks only end state, which is identical
    whether they share a pipeline or not."""

    async def test_both_pushes_ride_a_single_transaction(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failure mode if they split: the connection drops between the two
        round-trips, the display list gains the play and the outbox does not,
        and no drainer ever sees it — absent from Postgres forever. -history
        still shows it from Redis, so nobody notices until the Phase C cutover
        trims the list."""
        queued: list[str] = []
        direct: list[str] = []
        real_pipeline = fake_redis.pipeline

        def spy_pipeline(*args: Any, **kwargs: Any) -> Any:
            pipe = real_pipeline(*args, **kwargs)
            for name in ("lpush", "persist", "ltrim", "expire"):
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
            direct.append("lpush")

        monkeypatch.setattr(fake_redis, "lpush", record_direct)

        store = GuildRedisStore(fake_redis, guild_id=1)
        await store.push_history(_hentry(1))

        # Two LPUSHes (display + outbox), both on the pipeline, none direct.
        assert queued.count("lpush") == 2
        assert direct == []


class TestGetHistory:
    async def test_returns_up_to_cache_limit_items(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        # The read stays bounded even though the list is unbounded.
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
        """Two failing reads must not share one list object.

        A decorator argument is evaluated once at class-body execution, so
        `@_guild_op(default=[])` would return the *same* list to every guild on
        every failure — one in-place mutation would poison "empty history"
        process-wide, and a guild that never played a song would start
        reporting another guild's songs. `default_factory=list` builds one per
        failure.
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

    The decorator takes its fallback as an argument, which makes two mistakes
    cheap and invisible: a mutable literal (shared across guilds for the
    process lifetime) and a default that contradicts the declared return type
    (e.g. None from a `-> bool`). Neither is caught by pyright — `default` is
    typed Any on purpose so `default=None` cannot collapse the return TypeVar.
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
    async def test_writes_volume_to_hash(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_volume(0.75)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_sets_ttl_on_state_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await store.set_volume(1.0)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store: GuildRedisStore) -> None:
        await broken_store.set_volume(0.5)  # must not raise


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
        # (nothing stored) — _restore_guild relies on this to avoid silently
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
        """The whole point of the gate: it reads LLEN, never LRANGE, so the
        queue payload never rides the wire on the recovery path."""
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

    async def test_release_deletes_lock_key(
        self, store: GuildRedisStore, fake_redis: aioredis.Redis
    ) -> None:
        await fake_redis.set(store._recovery_lock_key(), "1", nx=True, ex=60)
        await store.release_recovery_lock()
        val = await fake_redis.get(store._recovery_lock_key())
        assert val is None

    async def test_release_swallows_redis_error(
        self, broken_store: GuildRedisStore
    ) -> None:
        await broken_store.release_recovery_lock()  # must not raise

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
        await store.pop_queue_and_start_song(_current(), 1000.0)
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"second"]

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
