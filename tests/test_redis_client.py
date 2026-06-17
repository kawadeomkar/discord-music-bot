"""Tests for src/redis_client.py — connection lifecycle, cache helpers, and GuildRedisStore."""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import redis.asyncio as aioredis

from src.redis_client import (
    GUILD_TTL,
    GuildRedisStore,
    cache_get,
    cache_set,
    close_redis_pool,
    create_redis_pool,
    get_redis,
)


# ── Connection lifecycle ──────────────────────────────────────────────────────


class TestCreateRedisPool:
    def test_returns_connection_pool(self):
        pool = create_redis_pool()
        assert isinstance(pool, aioredis.ConnectionPool)

    def test_pool_has_expected_max_connections(self):
        pool = create_redis_pool()
        assert pool.max_connections == 20


class TestGetRedis:
    def test_returns_redis_client(self):
        pool = create_redis_pool()
        client = get_redis(pool)
        assert isinstance(client, aioredis.Redis)


class TestCloseRedisPool:
    async def test_calls_aclose_on_pool(self):
        pool = AsyncMock()
        pool.aclose = AsyncMock()
        await close_redis_pool(pool)
        pool.aclose.assert_awaited_once()

    async def test_swallows_exception_on_close(self):
        pool = AsyncMock()
        pool.aclose = AsyncMock(side_effect=Exception("network gone"))
        await close_redis_pool(pool)  # must not raise


# ── Cache helpers ─────────────────────────────────────────────────────────────


class TestCacheGet:
    async def test_returns_none_when_redis_is_none(self):
        result = await cache_get(None, "some:key")
        assert result is None

    async def test_returns_none_on_cache_miss(self, fake_redis):
        result = await cache_get(fake_redis, "nonexistent:key")
        assert result is None

    async def test_returns_decoded_value_on_hit(self, fake_redis):
        await fake_redis.set("mykey", orjson.dumps({"x": 1}))
        result = await cache_get(fake_redis, "mykey")
        assert result == {"x": 1}

    async def test_returns_none_on_redis_error(self):
        bad_redis = AsyncMock()
        bad_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        result = await cache_get(bad_redis, "key")
        assert result is None


class TestCacheSet:
    async def test_sets_value_with_ttl(self, fake_redis):
        await cache_set(fake_redis, "ck", [1, 2, 3], 3600)
        raw = await fake_redis.get("ck")
        assert orjson.loads(raw) == [1, 2, 3]
        ttl = await fake_redis.ttl("ck")
        assert 3595 <= ttl <= 3600

    async def test_noop_when_redis_is_none(self):
        await cache_set(None, "key", "val", 60)  # must not raise

    async def test_swallows_redis_error(self):
        bad_redis = AsyncMock()
        bad_redis.set = AsyncMock(side_effect=ConnectionError("down"))
        await cache_set(bad_redis, "k", "v", 60)  # must not raise


# ── GuildRedisStore fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store(fake_redis):
    return GuildRedisStore(fake_redis, guild_id=123456789)


@pytest.fixture
def broken_store():
    """Store backed by a Redis mock that raises on every operation."""
    r = MagicMock()
    err = ConnectionError("redis down")
    for attr in ("rpush", "lpop", "lrange", "delete", "hset", "hgetall", "hdel", "set"):
        setattr(r, attr, AsyncMock(side_effect=err))
    pipe = MagicMock()
    for attr in ("rpush", "expire", "lpush", "ltrim", "hset", "delete"):
        setattr(pipe, attr, MagicMock())
    pipe.execute = AsyncMock(side_effect=err)
    r.pipeline = MagicMock(return_value=pipe)
    return GuildRedisStore(r, guild_id=999)


# ── Key helpers ───────────────────────────────────────────────────────────────


class TestKeyHelpers:
    def test_queue_key_includes_guild_id(self, store):
        assert "123456789" in store.queue_key()

    def test_state_key_includes_guild_id(self, store):
        assert "123456789" in store.state_key()

    def test_history_key_includes_guild_id(self, store):
        assert "123456789" in store.history_key()

    def test_keys_are_distinct(self, store):
        assert store.queue_key() != store.state_key()
        assert store.state_key() != store.history_key()


# ── Queue operations ──────────────────────────────────────────────────────────


class TestPushQueue:
    async def test_rpush_adds_item(self, store, fake_redis):
        await store.push_queue(b"item1")
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [b"item1"]

    async def test_sets_ttl_on_queue_key(self, store, fake_redis):
        await store.push_queue(b"item1")
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.push_queue(b"item")  # must not raise


class TestPopQueue:
    async def test_removes_first_item(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"first", b"second")
        await store.pop_queue()
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"second"]

    async def test_noop_on_empty_queue(self, store):
        await store.pop_queue()  # must not raise

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.pop_queue()  # must not raise


class TestGetQueue:
    async def test_returns_all_items_oldest_first(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"a", b"b", b"c")
        items = await store.get_queue()
        assert items == [b"a", b"b", b"c"]

    async def test_returns_empty_list_when_key_missing(self, store):
        items = await store.get_queue()
        assert items == []

    async def test_returns_empty_list_on_error(self, broken_store):
        result = await broken_store.get_queue()
        assert result == []


class TestDeleteQueue:
    async def test_deletes_queue_key(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"x")
        await store.delete_queue()
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == []

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.delete_queue()  # must not raise


class TestRebuildQueue:
    async def test_atomically_replaces_queue(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"old")
        await store.rebuild_queue([b"new1", b"new2"])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [b"new1", b"new2"]

    async def test_sets_ttl_after_rebuild(self, store, fake_redis):
        await store.rebuild_queue([b"item"])
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.rebuild_queue([b"x"])  # must not raise


# ── History operations ────────────────────────────────────────────────────────


class TestPushHistory:
    async def test_prepends_item_newest_first(self, store, fake_redis):
        await store.push_history(b"song1")
        await store.push_history(b"song2")
        items = await fake_redis.lrange(store.history_key(), 0, -1)
        assert items[0] == b"song2"  # newest first

    async def test_trims_to_50(self, store, fake_redis):
        for i in range(55):
            await store.push_history(orjson.dumps(f"song {i}"))
        items = await fake_redis.lrange(store.history_key(), 0, -1)
        assert len(items) == 50

    async def test_sets_ttl(self, store, fake_redis):
        await store.push_history(b"entry")
        ttl = await fake_redis.ttl(store.history_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.push_history(b"item")  # must not raise


class TestGetHistory:
    async def test_returns_up_to_50_items(self, store, fake_redis):
        for i in range(60):
            await fake_redis.lpush(store.history_key(), f"song{i}".encode())
        items = await store.get_history()
        assert len(items) == 50

    async def test_returns_empty_list_when_missing(self, store):
        items = await store.get_history()
        assert items == []

    async def test_returns_empty_list_on_error(self, broken_store):
        result = await broken_store.get_history()
        assert result == []


# ── State operations ──────────────────────────────────────────────────────────


class TestSetState:
    async def test_writes_field_to_hash(self, store, fake_redis):
        await store.set_state("volume", "0.75")
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_sets_ttl_on_state_key(self, store, fake_redis):
        await store.set_state("volume", "1.0")
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.set_state("field", "val")  # must not raise


class TestGetState:
    async def test_returns_hash_fields(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"volume", b"0.5")
        state = await store.get_state()
        assert state[b"volume"] == b"0.5"

    async def test_returns_empty_dict_when_missing(self, store):
        state = await store.get_state()
        assert state == {}

    async def test_returns_empty_dict_on_error(self, broken_store):
        result = await broken_store.get_state()
        assert result == {}


# ── TTL management ────────────────────────────────────────────────────────────


class TestRefreshTtl:
    async def test_refreshes_all_guild_keys(self, store, fake_redis):
        for key in [store.queue_key(), store.state_key(), store.history_key()]:
            await fake_redis.set(key, b"x")
            await fake_redis.expire(key, 10)  # short initial TTL
        await store.refresh_ttl()
        for key in [store.queue_key(), store.state_key(), store.history_key()]:
            ttl = await fake_redis.ttl(key)
            assert ttl > 1000  # refreshed to GUILD_TTL

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.refresh_ttl()  # must not raise


# ── Connection persistence ────────────────────────────────────────────────────


class TestSetConnection:
    async def test_persists_channel_ids(self, store, fake_redis):
        await store.set_connection(111, 222)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"voice_channel_id"] == b"111"
        assert state[b"text_channel_id"] == b"222"

    async def test_sets_ttl(self, store, fake_redis):
        await store.set_connection(111, 222)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.set_connection(1, 2)  # must not raise


class TestGetConnection:
    async def test_returns_channel_ids_when_set(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"voice_channel_id", b"111")
        await fake_redis.hset(store.state_key(), b"text_channel_id", b"222")
        vc_id, tc_id = await store.get_connection()
        assert vc_id == 111
        assert tc_id == 222

    async def test_returns_none_none_when_not_set(self, store):
        vc_id, tc_id = await store.get_connection()
        assert vc_id is None
        assert tc_id is None

    async def test_returns_none_none_on_error(self, broken_store):
        vc_id, tc_id = await broken_store.get_connection()
        assert vc_id is None
        assert tc_id is None


class TestClearConnection:
    async def test_removes_channel_and_song_fields(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"voice_channel_id", b"111")
        await fake_redis.hset(store.state_key(), b"text_channel_id", b"222")
        await fake_redis.hset(store.state_key(), b"current_song_url", b"url")
        await fake_redis.hset(store.state_key(), b"current_song_title", b"title")
        await store.clear_connection()
        state = await fake_redis.hgetall(store.state_key())
        assert b"voice_channel_id" not in state
        assert b"text_channel_id" not in state
        assert b"current_song_url" not in state
        assert b"current_song_title" not in state

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.clear_connection()  # must not raise


# ── Recovery lock ─────────────────────────────────────────────────────────────


class TestRecoveryLock:
    async def test_acquire_returns_true_first_time(self, store):
        acquired = await store.acquire_recovery_lock()
        assert acquired is True

    async def test_acquire_returns_false_when_already_held(self, store, fake_redis):
        await fake_redis.set(store._recovery_lock_key(), "1", nx=True, ex=60)
        acquired = await store.acquire_recovery_lock()
        assert acquired is False

    async def test_acquire_returns_false_on_error(self, broken_store):
        result = await broken_store.acquire_recovery_lock()
        assert result is False

    async def test_release_deletes_lock_key(self, store, fake_redis):
        await fake_redis.set(store._recovery_lock_key(), "1", nx=True, ex=60)
        await store.release_recovery_lock()
        val = await fake_redis.get(store._recovery_lock_key())
        assert val is None

    async def test_release_swallows_redis_error(self, broken_store):
        await broken_store.release_recovery_lock()  # must not raise

    async def test_lock_key_includes_guild_id(self, store):
        assert "123456789" in store._recovery_lock_key()
