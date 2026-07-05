import os
from dataclasses import dataclass
from typing import Any, Optional

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import Pipeline

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


@dataclass
class GuildState:
    """Typed, decoded view of a guild's `guild:{id}:state` Redis hash.
    All fields default to their "not set" value so a missing/empty hash
    round-trips as GuildState() with no special-casing at call sites."""

    volume: Optional[float] = None
    current_song_title: str = ""
    current_song_url: str = ""
    current_song_duration: Optional[int] = None
    current_song_uploader: Optional[str] = None
    voice_channel_id: Optional[int] = None
    text_channel_id: Optional[int] = None


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

    def _pipe_expire_all(self, pipe: Pipeline) -> None:
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

    async def get_state(self) -> GuildState:
        """HGETALL the state hash and decode it into a GuildState. Returns
        GuildState() (all-default) on a missing hash, any Redis error, or
        malformed field data."""
        try:
            raw: dict[bytes, bytes] = await self.redis.hgetall(self.state_key())  # type: ignore[misc]
            if not raw:
                return GuildState()

            def _int(key: bytes) -> Optional[int]:
                v = raw.get(key)
                return int(v) if v else None

            def _str(key: bytes) -> str:
                v = raw.get(key)
                return v.decode() if v else ""

            volume_raw = raw.get(b"volume")
            return GuildState(
                volume=float(volume_raw) if volume_raw else None,
                current_song_title=_str(b"current_song_title"),
                current_song_url=_str(b"current_song_url"),
                current_song_duration=_int(b"current_song_duration"),
                current_song_uploader=_str(b"current_song_uploader") or None,
                voice_channel_id=_int(b"voice_channel_id"),
                text_channel_id=_int(b"text_channel_id"),
            )
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis get_state failed: {e}")
            return GuildState()

    # TTL management

    async def refresh_ttl(self) -> None:
        """Refresh GUILD_TTL on all guild keys."""
        try:
            pipe = self.redis.pipeline()
            self._pipe_expire_all(pipe)
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

    async def set_current_song(
        self, title: str, url: str, duration: str, uploader: str
    ) -> None:
        """HSET all four 'current song' fields in one pipeline and refresh TTL.
        Pass all-empty strings to clear them (used by MusicPlayer._clear_current_song).
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), "current_song_title", title)
            pipe.hset(self.state_key(), "current_song_url", url)
            pipe.hset(self.state_key(), "current_song_duration", duration)
            pipe.hset(self.state_key(), "current_song_uploader", uploader)
            pipe.expire(self.state_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] set_current_song failed: {e}")

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
