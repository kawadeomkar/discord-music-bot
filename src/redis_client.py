import os
from collections.abc import Sequence
from typing import Any, Optional, cast

import orjson
import redis.asyncio as aioredis

from src.guild_state import (
    GuildPlaybackSnapshot,
    GuildRecoveryGate,
    GuildStateData,
    NowPlayingData,
    QueueEntry,
    SongQueueEntry,
    StateField,
    parse_history_entry,
    parse_queue_entry,
    serialize_history_entry,
)
from src.util import get_logger

log = get_logger(__name__)

GUILD_QUEUE_KEY = "guild:{guild_id}:queue"
GUILD_STATE_KEY = "guild:{guild_id}:state"
GUILD_HISTORY_KEY = "guild:{guild_id}:history"
GUILD_NOW_PLAYING_KEY = "guild:{guild_id}:now_playing"
GUILD_TTL = 86400  # 24h idle expiry
HISTORY_LIMIT = 50  # entries kept on both history legs (deque maxlen + LTRIM)

# Transient per-song fields cleared together on song end / disconnect, and the
# playback-position fields cleared together alongside them. Shared here so
# clear_song_end_state()/clear_connection() can't drift out of sync with each
# other by hand-editing one and forgetting the other.
_TRANSIENT_SONG_FIELDS = (
    StateField.CURRENT_SONG_URL,
    StateField.CURRENT_SONG_TITLE,
    StateField.CURRENT_SONG_DURATION,
    StateField.CURRENT_SONG_UPLOADER,
    StateField.CURRENT_SONG_REQUESTER_ID,
    StateField.CURRENT_SONG_INTERJECTED,
)
_PLAYBACK_POSITION_FIELDS = (
    StateField.PLAY_START_EPOCH,
    StateField.TOTAL_PAUSE_SECONDS,
    StateField.PAUSE_START_EPOCH,
)


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


# ── Spotify auth token cache ──────────────────────────────────────────────────
# Intentionally does not use cache_get/cache_set: the token is a raw string
# scalar, not JSON. Using orjson here would double-encode it as a JSON string.

_SPOTIFY_TOKEN_KEY = "spotify:auth:token"


async def spotify_token_set(
    redis: Optional[aioredis.Redis], token: str, expires_in: int
) -> None:
    """Store a Spotify bearer token as a raw string with TTL = expires_in − 30s.

    Skips caching entirely when the margin would consume the token's remaining
    life — a floor that *raised* the TTL would serve other processes a token
    that has already expired.
    """
    if redis is None:
        return
    ttl = expires_in - 30
    if ttl <= 0:
        # Token too short-lived to share safely — let each process fetch its own.
        return
    try:
        await redis.set(_SPOTIFY_TOKEN_KEY, token, ex=ttl)
    except Exception as e:
        log.warning(f"spotify_token_set failed: {e}")


async def spotify_token_get_with_ttl(
    redis: Optional[aioredis.Redis],
) -> Optional[tuple[str, int]]:
    """Return (token, seconds_remaining) for the cached Spotify bearer token, or
    None on miss/error/an already-expired key. Mirrors GET+TTL in one round trip
    so the caller can size its local expiry to the token's actual remaining
    life instead of a flat guess."""
    if redis is None:
        return None
    try:
        pipe = redis.pipeline()
        pipe.get(_SPOTIFY_TOKEN_KEY)
        pipe.ttl(_SPOTIFY_TOKEN_KEY)
        val, ttl = await pipe.execute()
        if val is None or ttl is None or ttl <= 0:
            return None
        return val.decode(), int(ttl)
    except Exception as e:
        log.warning(f"spotify_token_get_with_ttl failed: {e}")
        return None


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

    def now_playing_key(self) -> str:
        return GUILD_NOW_PLAYING_KEY.format(guild_id=self.guild_id)

    def _pipe_expire_all(self, pipe) -> None:
        """Queue expire commands for all four guild keys onto an existing pipeline."""
        pipe.expire(self.queue_key(), GUILD_TTL)
        pipe.expire(self.state_key(), GUILD_TTL)
        pipe.expire(self.history_key(), GUILD_TTL)
        pipe.expire(self.now_playing_key(), GUILD_TTL)

    async def _exec_with_state_ttl(self, pipe) -> None:
        """Append the state-key TTL refresh and execute the pipeline.

        EXPIRE must come after the write commands already queued on the pipe —
        an EXPIRE on a not-yet-created key is a no-op and would leave the key
        persistent-until-eviction.
        """
        pipe.expire(self.state_key(), GUILD_TTL)
        await pipe.execute()

    # Queue operations

    async def push_queue(self, entry: QueueEntry) -> None:
        """RPUSH one queue entry and refresh TTL on all guild keys."""
        try:
            pipe = self.redis.pipeline()
            pipe.rpush(self.queue_key(), entry.to_redis())
            self._pipe_expire_all(pipe)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_queue failed: {e}")

    async def push_queue_batch(self, entries: Sequence[QueueEntry]) -> None:
        """RPUSH all entries in one pipeline round-trip and refresh TTL on all guild keys."""
        if not entries:
            return
        try:
            pipe = self.redis.pipeline()
            pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
            self._pipe_expire_all(pipe)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_queue_batch failed: {e}")

    async def push_queue_front(self, entries: Sequence[QueueEntry]) -> None:
        """LPUSH entries so entries[0] ends up at the queue head, and refresh
        TTL on all guild keys — the -playnow front insert. LPUSH pushes each
        successive argument to the head, so the batch is reversed first to
        preserve the given order."""
        if not entries:
            return
        try:
            pipe = self.redis.pipeline()
            pipe.lpush(self.queue_key(), *[e.to_redis() for e in reversed(entries)])
            self._pipe_expire_all(pipe)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_queue_front failed: {e}")

    async def pop_queue(self) -> None:
        # At-most-once: LPOP removes the item immediately with no ack.
        # If the bot crashes after this call, the song is lost from Redis.
        # This is acceptable in Phase 2 (asyncio.Queue is source of truth).
        # Phase 3b migrates to Redis Streams + XACK for at-least-once.
        try:
            await self.redis.lpop(self.queue_key())  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis pop_queue failed: {e}")

    def _now_playing_state_mapping(
        self, current: SongQueueEntry, play_start_epoch: float
    ) -> dict[str, str]:
        """The current_song_* state fields ARE a parked queue entry — this one
        signature enforces the identity that SongQueueEntry.from_song() /
        from_crashed_state() rely on for crash recovery."""
        return {
            StateField.CURRENT_SONG_URL: current.webpage_url,
            StateField.CURRENT_SONG_TITLE: current.title,
            StateField.CURRENT_SONG_DURATION: (
                str(current.duration) if current.duration else ""
            ),
            StateField.CURRENT_SONG_UPLOADER: current.uploader or "",
            StateField.CURRENT_SONG_REQUESTER_ID: (
                str(current.requester_id) if current.requester_id else ""
            ),
            StateField.CURRENT_SONG_INTERJECTED: ("1" if current.interjected else ""),
            StateField.PLAY_START_EPOCH: str(play_start_epoch),
            StateField.TOTAL_PAUSE_SECONDS: "0",
        }

    async def pop_queue_and_start_song(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
    ) -> None:
        """Atomically LPOP the queue and park `current`'s fields in the state hash.

        Uses MULTI/EXEC so the song is always in one of two consistent states:
          (a) still in guild:{id}:queue, current_song_url empty  — transaction not executed
          (b) not in queue, all now-playing fields set           — transaction executed

        Eliminates the crash window where the song was absent from both the queue
        and current_song_url (the at-most-once gap from the prior pop_queue() pattern).

        When now_playing is given, the now_playing display snapshot is
        written inside the same transaction — a crash can never leave state
        pointing at song B while the snapshot still shows song A.
        """
        try:
            mapping = self._now_playing_state_mapping(current, play_start_epoch)
            pipe = self.redis.pipeline(transaction=True)
            pipe.lpop(self.queue_key())
            pipe.hset(self.state_key(), mapping=mapping)  # type: ignore[misc]
            pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
            pipe.expire(self.state_key(), GUILD_TTL)
            if now_playing is not None:
                pipe.hset(self.now_playing_key(), mapping=now_playing.to_redis_mapping())  # type: ignore[misc]
                pipe.expire(self.now_playing_key(), GUILD_TTL)
            await pipe.execute()
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] pop_queue_and_start_song failed: {e}")

    async def set_current_song_state(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
    ) -> None:
        """Same fields as pop_queue_and_start_song, in one transaction, but
        without the LPOP — for restarting a crash-recovered "current song" that
        was never RPUSHed to the Redis queue list in the first place.
        """
        try:
            mapping = self._now_playing_state_mapping(current, play_start_epoch)
            pipe = self.redis.pipeline(transaction=True)
            pipe.hset(self.state_key(), mapping=mapping)  # type: ignore[misc]
            pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
            pipe.expire(self.state_key(), GUILD_TTL)
            if now_playing is not None:
                pipe.hset(self.now_playing_key(), mapping=now_playing.to_redis_mapping())  # type: ignore[misc]
                pipe.expire(self.now_playing_key(), GUILD_TTL)
            await pipe.execute()
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] set_current_song_state failed: {e}")

    async def delete_queue(self) -> None:
        """DELETE the queue key."""
        try:
            await self.redis.delete(self.queue_key())
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis delete_queue failed: {e}")

    async def rebuild_queue(self, entries: Sequence[QueueEntry]) -> None:
        """Atomically DELETE + RPUSH all entries. Uses MULTI/EXEC to avoid empty-window race."""
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.delete(self.queue_key())
            pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
            pipe.expire(self.queue_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis rebuild_queue failed: {e}")

    # History operations

    async def push_history(self, entry: str) -> None:
        """LPUSH one entry, trim list to HISTORY_LIMIT, and refresh TTL."""
        try:
            pipe = self.redis.pipeline()
            pipe.lpush(self.history_key(), serialize_history_entry(entry))
            pipe.ltrim(self.history_key(), 0, HISTORY_LIMIT - 1)
            pipe.expire(self.history_key(), GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis push_history failed: {e}")

    async def get_history(self) -> list[str]:
        """Return up to HISTORY_LIMIT history entries newest-first. Corrupt
        entries are dropped (parse_history_entry warns per entry)."""
        try:
            raw: list[bytes] = await self.redis.lrange(  # type: ignore[misc]
                self.history_key(), 0, HISTORY_LIMIT - 1
            )
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis get_history failed: {e}")
            return []
        return [e for e in map(parse_history_entry, raw) if e is not None]

    # Now-playing operations
    # (Writes happen inside pop_queue_and_start_song()/set_current_song_state()
    #  via the now_playing value object, atomically with the rest of the start state.)

    async def get_now_playing(self) -> Optional[NowPlayingData]:
        """HGETALL the now_playing hash. Returns None on miss or error.

        Miss and error collapse to None deliberately: the only caller uses this
        to optionally restore a display embed, and "no embed" is the correct
        degraded behavior in both cases.
        """
        try:
            raw = cast(
                dict[bytes, bytes], await self.redis.hgetall(self.now_playing_key())
            )
            return NowPlayingData.from_redis(raw)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] get_now_playing failed: {e}")
            return None

    # Playback position tracking

    async def set_playback_start(self, epoch: float) -> None:
        """Record that playback started at `epoch`. Resets all pause accounting.

        Kept for unit tests and standalone use. In loop(), position fields are
        written atomically via pop_queue_and_start_song() instead.
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), StateField.PLAY_START_EPOCH, str(epoch))
            pipe.hset(self.state_key(), StateField.TOTAL_PAUSE_SECONDS, "0")
            pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
            await self._exec_with_state_ttl(pipe)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] set_playback_start failed: {e}")

    async def on_pause(self, epoch: float) -> None:
        """Record the epoch when the voice client was paused."""
        try:
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), StateField.PAUSE_START_EPOCH, str(epoch))
            await self._exec_with_state_ttl(pipe)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] on_pause failed: {e}")

    async def on_resume(self, resume_epoch: float) -> None:
        """Accumulate elapsed pause time into total_pause_seconds and clear pause_start_epoch.

        Non-atomic read-modify-write: assumes a single writer per guild (true
        for the current one-process-per-guild command flow). Under
        multi-process sharding this must become a Lua script or WATCH/MULTI
        retry loop — see docs/REDIS_MIGRATION_PLAN.md.
        """
        try:
            vals = await self.redis.hmget(
                self.state_key(),
                StateField.PAUSE_START_EPOCH,
                StateField.TOTAL_PAUSE_SECONDS,
            )
            pause_start_raw = vals[0] or b""
            if not pause_start_raw:
                return
            total_raw = vals[1] if vals[1] is not None else b"0"
            elapsed_pause = max(0.0, resume_epoch - float(pause_start_raw))
            new_total = float(total_raw) + elapsed_pause
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), StateField.TOTAL_PAUSE_SECONDS, str(new_total))
            pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
            await self._exec_with_state_ttl(pipe)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] on_resume failed: {e}")

    async def clear_song_end_state(self) -> None:
        """Pipeline that clears all transient song state in one round-trip.

        HDELs all current_song_* and playback-position fields and DELETEs the
        now_playing hash — the same idiom clear_connection() uses, so absent
        (not empty-string) is the one representation of "no song". Called on
        both normal song end and the error-path skip in loop().
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hdel(
                self.state_key(),
                *_TRANSIENT_SONG_FIELDS,
                *_PLAYBACK_POSITION_FIELDS,
            )
            pipe.delete(self.now_playing_key())
            await pipe.execute()
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] clear_song_end_state failed: {e}")

    # State operations

    async def get_guild_state(self) -> Optional[GuildStateData]:
        """HGETALL the state hash and return a typed snapshot.

        Returns GuildStateData() (zero values) when the hash is missing/empty
        and None when the read itself failed — callers can distinguish "nothing
        stored" from "Redis unavailable" (see _restore_guild).

        Does NOT refresh TTL — pure read. TTL is refreshed by refresh_ttl() at
        the end of _restore_state(), which covers the recovery window.
        """
        try:
            raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.state_key()))
            return GuildStateData.from_redis(raw)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] get_guild_state failed: {e}")
            return None

    async def get_recovery_gate(self) -> Optional[GuildRecoveryGate]:
        """State hash + pending-queue *length* in one pipeline — the lightweight
        connection/restorable gate for `_restore_guild`.

        Deliberately does NOT transfer the queue contents, now-playing, or
        history: a `-stop`ped guild keeps a possibly-long queue list by design,
        and gating on LLEN keeps that payload off the wire on every `on_ready`.
        `_restore_state` re-reads the full snapshot after a successful connect,
        so the contents are fetched exactly once, only when they are used.

        Same RTT count as get_playback_snapshot (one pipeline) but a fixed,
        tiny payload. Returns None on read failure (same error-vs-empty
        contract as get_guild_state).
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hgetall(self.state_key())
            pipe.llen(self.queue_key())
            raw_state, queue_len = await pipe.execute()
            return GuildRecoveryGate(
                state=GuildStateData.from_redis(raw_state),
                pending_count=int(queue_len),
            )
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] get_recovery_gate failed: {e}")
            return None

    async def get_playback_snapshot(self) -> Optional[GuildPlaybackSnapshot]:
        """Read the complete playback aggregate — state hash, pending queue,
        now-playing snapshot, and history — in one pipeline round-trip.

        Returns None when the read failed (same error-vs-empty contract as
        get_guild_state: an empty guild yields a snapshot with zero-value
        state and empty queue/history, never None). Because all four reads
        ride one pipeline, a failure aborts the whole snapshot — the caller
        restores everything or nothing, rather than a partially-fabricated
        state. Corrupt queue/history entries are dropped with a warning by
        their parsers.

        Not MULTI: the reads are not atomic relative to a concurrent writer,
        which matches the previous back-to-back reads exactly — recovery
        holds the guild recovery lock during the window that matters.
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hgetall(self.state_key())
            pipe.lrange(self.queue_key(), 0, -1)
            pipe.hgetall(self.now_playing_key())
            pipe.lrange(self.history_key(), 0, HISTORY_LIMIT - 1)
            raw_state, raw_queue, raw_np, raw_history = await pipe.execute()
            entries = tuple(
                entry
                for entry in (parse_queue_entry(item) for item in raw_queue)
                if entry is not None
            )
            history = tuple(
                entry
                for entry in (parse_history_entry(item) for item in raw_history)
                if entry is not None
            )
            return GuildPlaybackSnapshot(
                state=GuildStateData.from_redis(raw_state),
                queue=entries,
                now_playing=NowPlayingData.from_redis(raw_np),
                history=history,
            )
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] get_playback_snapshot failed: {e}")
            return None

    async def set_volume(self, volume: float) -> None:
        """Persist the guild volume setting."""
        try:
            pipe = self.redis.pipeline()
            pipe.hset(self.state_key(), StateField.VOLUME, str(volume))
            await self._exec_with_state_ttl(pipe)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] set_volume failed: {e}")

    # TTL management

    async def refresh_ttl(self) -> None:
        """Refresh GUILD_TTL on all guild keys."""
        try:
            pipe = self.redis.pipeline()
            for key in [
                self.queue_key(),
                self.state_key(),
                self.history_key(),
                self.now_playing_key(),
            ]:
                pipe.expire(key, GUILD_TTL)
            await pipe.execute()  # type: ignore[misc]
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] Redis refresh_ttl failed: {e}")

    # Connection persistence

    async def set_connection(self, voice_channel_id: int, text_channel_id: int) -> None:
        """Persist active voice and text channel IDs into the state hash."""
        try:
            pipe = self.redis.pipeline()
            pipe.hset(
                self.state_key(), StateField.VOICE_CHANNEL_ID, str(voice_channel_id)
            )
            pipe.hset(
                self.state_key(), StateField.TEXT_CHANNEL_ID, str(text_channel_id)
            )
            await self._exec_with_state_ttl(pipe)
        except Exception as e:
            log.warning(f"[guild:{self.guild_id}] set_connection failed: {e}")

    async def clear_connection(self) -> None:
        """Remove all transient state on intentional disconnect.

        Clears voice/text channel IDs so on_ready skips recovery for this guild.
        Also clears now-playing display, requester attribution, and all playback
        position tracking fields.
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hdel(
                self.state_key(),
                StateField.VOICE_CHANNEL_ID,
                StateField.TEXT_CHANNEL_ID,
                *_TRANSIENT_SONG_FIELDS,
                # last_author_id is no longer written; the HDEL scrubs hashes
                # left by older builds and is removable after one release. It
                # is intentionally a literal, not a StateField — it is not part
                # of the schema.
                "last_author_id",
                *_PLAYBACK_POSITION_FIELDS,
            )
            pipe.delete(self.now_playing_key())
            await pipe.execute()
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
