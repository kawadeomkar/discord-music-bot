import os
from typing import Any, Optional

import orjson
import redis.asyncio as aioredis

from src.util import get_logger

log = get_logger(__name__)

GUILD_QUEUE_KEY = "guild:{guild_id}:queue"
GUILD_STATE_KEY = "guild:{guild_id}:state"
GUILD_HISTORY_KEY = "guild:{guild_id}:history"
GUILD_TTL = 86400  # 24h idle expiry


# ── Connection lifecycle ──────────────────────────────────────────────────────


def create_redis_pool() -> aioredis.ConnectionPool:
    """Create the application-wide connection pool. Call once at startup."""
    return aioredis.ConnectionPool.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        max_connections=20,
        decode_responses=False,
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_timeout=True,
        retry_on_error=[ConnectionError, TimeoutError],
        socket_connect_timeout=5,
    )


def get_redis(pool: aioredis.ConnectionPool) -> aioredis.Redis:
    """Return a Redis client backed by the given pool."""
    return aioredis.Redis(connection_pool=pool)


async def close_redis_pool(pool: aioredis.ConnectionPool) -> None:
    """Gracefully close the connection pool. Call once at shutdown."""
    try:
        await pool.aclose()
    except Exception as e:
        log.warning(f"Failed to close Redis connection pool: {e}")


# ── Generic cache helpers ─────────────────────────────────────────────────────


async def cache_get(redis: Optional[aioredis.Redis], key: str) -> Any:
    """Get and orjson-decode a cached value. Returns None on miss, error, or when redis is None."""
    if redis is None:
        return None
    try:
        val = await redis.get(key)
        return orjson.loads(val) if val is not None else None
    except Exception as e:
        log.warning(f"cache_get failed [{key}]: {e}")
        return None


async def cache_set(
    redis: Optional[aioredis.Redis], key: str, value: Any, ttl: int
) -> None:
    """orjson-encode and set a value with TTL. No-ops when redis is None; silently ignores errors."""
    if redis is None:
        return
    try:
        await redis.set(key, orjson.dumps(value), ex=ttl)
    except Exception as e:
        log.warning(f"cache_set failed [{key}]: {e}")


# ── Guild-scoped Redis store ──────────────────────────────────────────────────


class GuildRedisStore:
    """Encapsulates all Redis IO for a single guild. All methods log errors and never raise."""

    def __init__(self, redis: aioredis.Redis, guild_id: int) -> None:
        self.redis = redis
        self.guild_id = guild_id

    # Key helpers

    def queue_key(self) -> str:
        return GUILD_QUEUE_KEY.format(guild_id=self.guild_id)

    def state_key(self) -> str:
        return GUILD_STATE_KEY.format(guild_id=self.guild_id)

    def history_key(self) -> str:
        return GUILD_HISTORY_KEY.format(guild_id=self.guild_id)

    def _pipe_expire_all(self, pipe) -> None:
        """Queue expire commands for all three guild keys onto an existing pipeline."""
        pipe.expire(self.queue_key(), GUILD_TTL)
        pipe.expire(self.state_key(), GUILD_TTL)
        pipe.expire(self.history_key(), GUILD_TTL)

    # Queue operations

    async def push_queue(self, data: bytes) -> None:
        """RPUSH one serialized item and refresh TTL on all guild keys."""
        try:
            pipe = self.redis.pipeline()
            pipe.rpush(self.queue_key(), data)
            self._pipe_expire_all(pipe)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_queue failed: {e}")

    async def push_queue_batch(self, items: list[bytes]) -> None:
        """RPUSH all items in one pipeline round-trip and refresh TTL on all guild keys."""
        if not items:
            return
        try:
            pipe = self.redis.pipeline()
            pipe.rpush(self.queue_key(), *items)
            self._pipe_expire_all(pipe)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_queue_batch failed: {e}")

    async def pop_queue(self) -> None:
        # At-most-once: LPOP removes the item immediately with no ack.
        # If the bot crashes after this call, the song is lost from Redis.
        # This is acceptable in Phase 2 (asyncio.Queue is source of truth).
        # Phase 3b migrates to Redis Streams + XACK for at-least-once.
        try:
            await self.redis.lpop(self.queue_key())  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis pop_queue failed: {e}")

    async def get_queue(self) -> list[bytes]:
        """Return all queued items oldest-first."""
        try:
            return await self.redis.lrange(self.queue_key(), 0, -1)  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis get_queue failed: {e}")
            return []

    async def delete_queue(self) -> None:
        """DELETE the queue key."""
        try:
            await self.redis.delete(self.queue_key())
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis delete_queue failed: {e}")

    async def rebuild_queue(self, items: list[bytes]) -> None:
        """Atomically DELETE + RPUSH all items. Uses MULTI/EXEC to avoid empty-window race."""
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.delete(self.queue_key())
            pipe.rpush(self.queue_key(), *items)
            pipe.expire(self.queue_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis rebuild_queue failed: {e}")

    # History operations

    async def push_history(self, data: bytes) -> None:
        """LPUSH one serialized entry, trim list to 50, and refresh TTL."""
        try:
            pipe = self.redis.pipeline()
            pipe.lpush(self.history_key(), data)
            pipe.ltrim(self.history_key(), 0, 49)
            pipe.expire(self.history_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_history failed: {e}")

    async def get_history(self) -> list[bytes]:
        """Return up to 50 history items newest-first."""
        try:
            return await self.redis.lrange(self.history_key(), 0, 49)  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis get_history failed: {e}")
            return []

    # State operations

    async def set_state(self, field: str, value: str) -> None:
        """HSET a single field and refresh TTL."""
        try:
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), field, value)
            pipe.expire(self.state_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(
                f"[guild:{self.guild_id}] Redis set_state [{field}={value}] failed: {e}"
            )

    async def get_state(self) -> dict:
        """HGETALL the state hash. Returns empty dict on error."""
        try:
            return await self.redis.hgetall(self.state_key())  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis get_state failed: {e}")
            return {}

    # TTL management

    async def refresh_ttl(self) -> None:
        """Refresh GUILD_TTL on all guild keys."""
        try:
            pipe = self.redis.pipeline()
            for key in [self.queue_key(), self.state_key(), self.history_key()]:
                pipe.expire(key, GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis refresh_ttl failed: {e}")

    # Connection persistence

    async def set_connection(self, voice_channel_id: int, text_channel_id: int) -> None:
        """Persist active voice and text channel IDs into the state hash."""
        try:
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), "voice_channel_id", str(voice_channel_id))
            pipe.hset(self.state_key(), "text_channel_id", str(text_channel_id))
            pipe.expire(self.state_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] set_connection failed: {e}")

    async def get_connection(self) -> tuple[Optional[int], Optional[int]]:
        """Return (voice_channel_id, text_channel_id) or (None, None) if not stored."""
        try:
            state = await self.redis.hgetall(self.state_key())  # type: ignore[misc]
            vc_raw = state.get(b"voice_channel_id")
            tc_raw = state.get(b"text_channel_id")
            vc_id = int(vc_raw) if vc_raw else None
            tc_id = int(tc_raw) if tc_raw else None
            return vc_id, tc_id
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] get_connection failed: {e}")
            return None, None

    async def clear_connection(self) -> None:
        """Remove channel IDs and now-playing state from the hash on intentional cleanup."""
        try:
            await self.redis.hdel(  # type: ignore[misc]
                self.state_key(),
                "voice_channel_id",
                "text_channel_id",
                "current_song_url",
                "current_song_title",
            )
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] clear_connection failed: {e}")

    # Recovery lock (distributed, for rolling-restart safety)

    _RECOVERY_LOCK_TTL = 60  # seconds

    def _recovery_lock_key(self) -> str:
        return f"lock:guild:{self.guild_id}:recovery"

    async def acquire_recovery_lock(self) -> bool:
        """SET NX EX — True if this instance won the lock, False if another holds it."""
        try:
            result = await self.redis.set(
                self._recovery_lock_key(), "1", nx=True, ex=self._RECOVERY_LOCK_TTL
            )
            return result is True
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] acquire_recovery_lock failed: {e}")
            return False

    async def release_recovery_lock(self) -> None:
        try:
            await self.redis.delete(self._recovery_lock_key())
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] release_recovery_lock failed: {e}")
