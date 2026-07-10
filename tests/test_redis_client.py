"""Tests for src/redis_client.py — connection lifecycle, cache helpers, and GuildRedisStore."""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import redis.asyncio as aioredis

from src.guild_state import (
    GuildPlaybackSnapshot,
    GuildStateData,
    NowPlayingData,
    SongQueueEntry,
)
from src.redis_client import (
    GUILD_TTL,
    GuildRedisStore,
    cache_get,
    cache_set,
    close_redis_pool,
    create_redis_pool,
    get_redis,
    spotify_token_get_with_ttl,
    spotify_token_set,
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
    for attr in ("rpush", "expire", "lpush", "ltrim", "hset", "delete", "hdel", "lpop"):
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

    def test_now_playing_key_includes_guild_id(self, store):
        assert "123456789" in store.now_playing_key()

    def test_keys_are_distinct(self, store):
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
    async def test_rpush_adds_entry_bytes(self, store, fake_redis):
        await store.push_queue(_entry(1))
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis()]

    async def test_sets_ttl_on_queue_key(self, store, fake_redis):
        await store.push_queue(_entry(1))
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_refreshes_ttl_on_now_playing_key(self, store, fake_redis):
        """_pipe_expire_all must refresh all four guild keys, including now_playing."""
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        await fake_redis.expire(store.now_playing_key(), 5)
        await store.push_queue(_entry(1))
        ttl = await fake_redis.ttl(store.now_playing_key())
        assert ttl > 5

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.push_queue(_entry(1))  # must not raise


class TestPushQueueBatch:
    async def test_rpush_all_entries_in_order(self, store, fake_redis):
        await store.push_queue_batch([_entry(1), _entry(2)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis(), _entry(2).to_redis()]

    async def test_noop_on_empty_sequence(self, store, fake_redis):
        await store.push_queue_batch([])
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.push_queue_batch([_entry(1)])  # must not raise


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
        await store.rebuild_queue([_entry(1), _entry(2)])
        items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert items == [_entry(1).to_redis(), _entry(2).to_redis()]

    async def test_sets_ttl_after_rebuild(self, store, fake_redis):
        await store.rebuild_queue([_entry(1)])
        ttl = await fake_redis.ttl(store.queue_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.rebuild_queue([_entry(1)])  # must not raise


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


class TestSetVolume:
    async def test_writes_volume_to_hash(self, store, fake_redis):
        await store.set_volume(0.75)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"volume"] == b"0.75"

    async def test_sets_ttl_on_state_key(self, store, fake_redis):
        await store.set_volume(1.0)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.set_volume(0.5)  # must not raise


class TestGetGuildState:
    async def test_returns_typed_snapshot(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"volume", b"0.5")
        await fake_redis.hset(store.state_key(), b"current_song_url", b"https://x")
        state = await store.get_guild_state()
        assert state == GuildStateData(volume=0.5, current_song_url="https://x")

    async def test_returns_zero_value_snapshot_when_missing(self, store):
        state = await store.get_guild_state()
        assert state == GuildStateData()

    async def test_returns_none_on_error_not_defaults(self, broken_store):
        # None (read failed) must be distinguishable from GuildStateData()
        # (nothing stored) — _restore_guild relies on this to avoid silently
        # skipping recovery during a Redis outage.
        result = await broken_store.get_guild_state()
        assert result is None


class TestGetPlaybackSnapshot:
    async def test_returns_state_and_queue_together(self, store, fake_redis):
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

    async def test_empty_guild_yields_empty_snapshot_not_none(self, store):
        snap = await store.get_playback_snapshot()
        assert snap == GuildPlaybackSnapshot(state=GuildStateData())
        assert not snap.has_restorable_playback

    async def test_corrupt_queue_entries_dropped(self, store, fake_redis):
        await fake_redis.rpush(
            store.queue_key(), b"not json", _entry(1).to_redis(), b"{}"
        )
        snap = await store.get_playback_snapshot()
        assert snap is not None
        assert snap.queue == (_entry(1),)

    async def test_returns_none_on_error(self, broken_store):
        assert await broken_store.get_playback_snapshot() is None

    async def test_single_pipeline_round_trip(self, fake_redis):
        """State HGETALL and queue LRANGE ride one pipeline execute()."""
        store = GuildRedisStore(fake_redis, guild_id=42)
        real_pipeline = fake_redis.pipeline
        execute_counts = []

        def counting_pipeline(*args, **kwargs):
            pipe = real_pipeline(*args, **kwargs)
            original_execute = pipe.execute

            async def counted_execute():
                execute_counts.append(1)
                return await original_execute()

            pipe.execute = counted_execute
            return pipe

        fake_redis.pipeline = counting_pipeline
        try:
            snap = await store.get_playback_snapshot()
        finally:
            fake_redis.pipeline = real_pipeline
        assert snap is not None
        assert len(execute_counts) == 1


# ── TTL management ────────────────────────────────────────────────────────────


class TestRefreshTtl:
    async def test_refreshes_all_guild_keys(self, store, fake_redis):
        all_keys = [
            store.queue_key(),
            store.state_key(),
            store.history_key(),
            store.now_playing_key(),
        ]
        for key in all_keys:
            await fake_redis.set(key, b"x")
            await fake_redis.expire(key, 10)  # short initial TTL
        await store.refresh_ttl()
        for key in all_keys:
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


class TestConnectionViaGuildState:
    async def test_returns_channel_ids_when_set(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"voice_channel_id", b"111")
        await fake_redis.hset(store.state_key(), b"text_channel_id", b"222")
        state = await store.get_guild_state()
        assert state is not None
        assert state.voice_channel_id == 111
        assert state.text_channel_id == 222
        assert state.has_active_connection

    async def test_no_active_connection_when_not_set(self, store):
        state = await store.get_guild_state()
        assert state is not None
        assert state.voice_channel_id is None
        assert state.text_channel_id is None
        assert not state.has_active_connection


class TestClearConnection:
    async def test_removes_all_transient_fields(self, store, fake_redis):
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

    async def test_preserves_non_transient_fields(self, store, fake_redis):
        """clear_connection must not delete persistent fields like volume."""
        await fake_redis.hset(store.state_key(), b"volume", b"0.8")
        await store.clear_connection()
        state = await fake_redis.hgetall(store.state_key())
        assert state.get(b"volume") == b"0.8"

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


# ── Spotify token cache ───────────────────────────────────────────────────────


class TestSpotifyTokenCache:
    async def test_token_get_with_ttl_returns_none_on_miss(self, fake_redis):
        result = await spotify_token_get_with_ttl(fake_redis)
        assert result is None

    async def test_token_get_with_ttl_returns_token_and_remaining_life(
        self, fake_redis
    ):
        await fake_redis.set("spotify:auth:token", b"my_bearer_token", ex=3570)
        result = await spotify_token_get_with_ttl(fake_redis)
        assert result is not None
        token, ttl = result
        assert token == "my_bearer_token"
        assert 3560 <= ttl <= 3570

    async def test_token_get_with_ttl_returns_none_when_redis_is_none(self):
        result = await spotify_token_get_with_ttl(None)
        assert result is None

    async def test_token_get_with_ttl_returns_none_for_key_without_expiry(
        self, fake_redis
    ):
        # TTL of -1 (no expiry) must be treated as unusable, not as a live token.
        await fake_redis.set("spotify:auth:token", b"stale_token")
        result = await spotify_token_get_with_ttl(fake_redis)
        assert result is None

    async def test_token_get_with_ttl_swallows_error(self):
        bad_redis = MagicMock()
        bad_redis.pipeline = MagicMock(side_effect=ConnectionError("down"))
        result = await spotify_token_get_with_ttl(bad_redis)
        assert result is None

    async def test_token_set_stores_raw_string_with_ttl(self, fake_redis):
        await spotify_token_set(fake_redis, "token_abc", 3600)
        val = await fake_redis.get("spotify:auth:token")
        assert val == b"token_abc"
        ttl = await fake_redis.ttl("spotify:auth:token")
        assert 3560 <= ttl <= 3570  # 3600 - 30 = 3570

    async def test_token_set_skips_cache_for_short_lived_token(self, fake_redis):
        # A margin that *raised* the TTL would serve an expired token to other
        # processes — short-lived tokens must simply not be cached.
        await spotify_token_set(fake_redis, "token", 20)
        assert await fake_redis.get("spotify:auth:token") is None

    async def test_token_set_boundary_just_above_margin(self, fake_redis):
        await spotify_token_set(fake_redis, "token", 31)  # 31 - 30 = 1s
        ttl = await fake_redis.ttl("spotify:auth:token")
        assert ttl == 1

    async def test_token_set_boundary_at_margin_not_written(self, fake_redis):
        await spotify_token_set(fake_redis, "token", 30)  # 30 - 30 = 0 → skip
        assert await fake_redis.get("spotify:auth:token") is None

    async def test_token_set_noop_when_redis_is_none(self):
        await spotify_token_set(None, "token", 3600)  # must not raise

    async def test_token_set_swallows_error(self):
        bad_redis = AsyncMock()
        bad_redis.set = AsyncMock(side_effect=ConnectionError("down"))
        await spotify_token_set(bad_redis, "token", 3600)  # must not raise


# ── pop_queue_and_start_song ──────────────────────────────────────────────────


def _current(url: str = "url", title: str = "title", **kwargs) -> SongQueueEntry:
    """The start-transaction carrier: the queue-entry view of the song that is
    about to play (its fields get parked in the state hash as current_song_*)."""
    return SongQueueEntry(
        webpage_url=url,
        title=title,
        requester_id=kwargs.pop("requester_id", None),
        **kwargs,
    )


class TestPopQueueAndStartSong:
    async def test_lpop_removes_first_item_only(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"first", b"second")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"second"]

    async def test_writes_now_playing_fields_atomically(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(
            _current("https://yt.com/v=1", "Test Song"), 1000.5
        )
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_url"] == b"https://yt.com/v=1"
        assert state[b"current_song_title"] == b"Test Song"
        assert state[b"play_start_epoch"] == b"1000.5"
        assert state[b"total_pause_seconds"] == b"0"

    async def test_writes_duration_uploader_and_requester_id(self, store, fake_redis):
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
        self, store, fake_redis
    ):
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"current_song_duration"] == b""
        assert state[b"current_song_uploader"] == b""
        assert state[b"current_song_requester_id"] == b""

    async def test_clears_pause_epoch_on_start(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"999.0")
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert b"pause_start_epoch" not in state

    async def test_sets_ttl_on_state(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_empty_queue_lpop_is_noop_state_still_written(
        self, store, fake_redis
    ):
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
        self, store, fake_redis
    ):
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

    async def test_now_playing_untouched_when_fields_omitted(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"song")
        await store.pop_queue_and_start_song(_current(), 1000.0)
        assert await fake_redis.exists(store.now_playing_key()) == 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.pop_queue_and_start_song(
            _current(), 1000.0
        )  # must not raise


class TestSetCurrentSongState:
    """Mirrors TestPopQueueAndStartSong minus the LPOP — used for restarting a
    crash-recovered "current song" that was never RPUSHed to the queue list."""

    async def test_writes_now_playing_fields(self, store, fake_redis):
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

    async def test_does_not_touch_queue(self, store, fake_redis):
        await fake_redis.rpush(store.queue_key(), b"untouched")
        await store.set_current_song_state(_current(), 1000.0)
        remaining = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert remaining == [b"untouched"]

    async def test_clears_pause_epoch(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"999.0")
        await store.set_current_song_state(_current(), 1000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert b"pause_start_epoch" not in state

    async def test_now_playing_fields_written_in_same_transaction(
        self, store, fake_redis
    ):
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

    async def test_now_playing_untouched_when_fields_omitted(self, store, fake_redis):
        await store.set_current_song_state(_current(), 1000.0)
        assert await fake_redis.exists(store.now_playing_key()) == 0

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.set_current_song_state(_current(), 1000.0)  # must not raise


# ── Now-playing operations ────────────────────────────────────────────────────
# (Writes are covered above via now_playing_fields on the start-transaction
#  methods; only the read side has a standalone method.)


class TestNowPlayingOperations:
    async def test_get_now_playing_returns_typed_snapshot(self, store, fake_redis):
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        data = await store.get_now_playing()
        assert data is not None
        assert data.title == "Song"

    async def test_get_now_playing_returns_none_when_missing(self, store):
        data = await store.get_now_playing()
        assert data is None

    async def test_get_now_playing_returns_none_on_error(self, broken_store):
        result = await broken_store.get_now_playing()
        assert result is None


# ── Playback position tracking ────────────────────────────────────────────────


class TestPlaybackPosition:
    async def test_on_pause_writes_epoch(self, store, fake_redis):
        await store.on_pause(1234.5)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"pause_start_epoch"] == b"1234.5"

    async def test_on_pause_sets_ttl(self, store, fake_redis):
        await store.on_pause(1234.5)
        ttl = await fake_redis.ttl(store.state_key())
        assert ttl > 0

    async def test_on_resume_accumulates_pause_seconds(self, store, fake_redis):
        # paused at t=1000, resuming at t=1030 → 30s of pause to add
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"1000.0")
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"60")
        await store.on_resume(1030.0)
        state = await fake_redis.hgetall(store.state_key())
        assert float(state[b"total_pause_seconds"]) == 90.0  # 60 + 30
        assert b"pause_start_epoch" not in state

    async def test_on_resume_preserves_fractional_seconds(self, store, fake_redis):
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

    async def test_on_resume_clamps_negative_elapsed_pause(self, store, fake_redis):
        """A backward clock step between pause and resume must not decrease the total."""
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"2000.0")
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"60")
        await store.on_resume(1000.0)  # resume_epoch is *before* pause_start_epoch
        state = await fake_redis.hgetall(store.state_key())
        assert float(state[b"total_pause_seconds"]) == 60.0  # unchanged, not decreased

    async def test_on_resume_no_op_when_not_paused(self, store, fake_redis):
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"60")
        # no pause_start_epoch set
        await store.on_resume(1030.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state.get(b"total_pause_seconds") == b"60"  # unchanged

    async def test_set_playback_start_writes_epoch_and_resets_pauses(
        self, store, fake_redis
    ):
        await fake_redis.hset(store.state_key(), b"pause_start_epoch", b"999.0")
        await fake_redis.hset(store.state_key(), b"total_pause_seconds", b"30")
        await store.set_playback_start(5000.0)
        state = await fake_redis.hgetall(store.state_key())
        assert state[b"play_start_epoch"] == b"5000.0"
        assert state[b"total_pause_seconds"] == b"0"
        assert b"pause_start_epoch" not in state

    async def test_on_pause_swallows_error(self, broken_store):
        await broken_store.on_pause(1234.5)  # must not raise

    async def test_on_resume_swallows_error(self, broken_store):
        await broken_store.on_resume(1234.5)  # must not raise


class TestClearSongEndState:
    async def test_clears_current_song_fields(self, store, fake_redis):
        await fake_redis.hset(
            store.state_key(),
            mapping={
                b"current_song_url": b"https://yt.com/v=1",
                b"current_song_title": b"Song",
                b"current_song_duration": b"210",
                b"current_song_uploader": b"Some Channel",
                b"current_song_requester_id": b"42",
            },
        )
        await store.clear_song_end_state()
        state = await fake_redis.hgetall(store.state_key())
        assert b"current_song_url" not in state
        assert b"current_song_title" not in state
        assert b"current_song_duration" not in state
        assert b"current_song_uploader" not in state
        assert b"current_song_requester_id" not in state

    async def test_deletes_now_playing_hash(self, store, fake_redis):
        await fake_redis.hset(store.now_playing_key(), b"title", b"Song")
        await store.clear_song_end_state()
        assert await fake_redis.exists(store.now_playing_key()) == 0

    async def test_removes_playback_position_fields(self, store, fake_redis):
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

    async def test_swallows_redis_error(self, broken_store):
        await broken_store.clear_song_end_state()  # must not raise
