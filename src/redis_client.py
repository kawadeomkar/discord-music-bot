from __future__ import annotations

import os
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, Concatenate, Final, Optional, ParamSpec, TypeVar, cast

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import Pipeline
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import OutOfMemoryError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.exceptions import WatchError
from redis.asyncio.retry import Retry
from redis.typing import EncodableT, FieldT

from src import config
from src.guild_state import (
    ConfigField,
    GuildConfig,
    GuildPlaybackSnapshot,
    GuildRecoveryGate,
    GuildStateData,
    HistoryEntry,
    NowPlayingData,
    QueueEntry,
    SongQueueEntry,
    StateField,
    parse_history_entry,
    parse_queue_entry,
    serialize_history_entry,
    valid_timezone,
)
from src.util import get_logger

log = get_logger(__name__)

GUILD_QUEUE_KEY = "guild:{guild_id}:queue"
GUILD_STATE_KEY = "guild:{guild_id}:state"
GUILD_HISTORY_KEY = "guild:{guild_id}:history"
GUILD_NOW_PLAYING_KEY = "guild:{guild_id}:now_playing"
# Durable per-guild preferences: NOT one of the TTL-managed keys above, since a
# setting that expires after a day idle reverts for reasons the user cannot
# see. See GuildConfig and _pipe_expire_all.
GUILD_CONFIG_KEY = "guild:{guild_id}:config"
# Global write-ahead buffer for the Postgres history archive: every guild's
# entries interleave (each carries its guild_id), XADDed beside the display
# list and drained oldest-first by HistoryOutboxDrainer. NO TTL — it holds
# not-yet-durable entries and must never be an eviction candidate. A STREAM
# with a consumer group, because XACK settles by ID.
HISTORY_OUTBOX_KEY = "history:outbox"
HISTORY_OUTBOX_GROUP = "drainers"
# Stable, not per-process: the PEL belongs to the name, so a starting process's
# `XREADGROUP ... 0` inherits what a predecessor or live sibling left in flight.
# Two live drainers are safe: `>` hands them disjoint entries and `0` replays a
# shared set that ON CONFLICT DO NOTHING collapses.
HISTORY_OUTBOX_CONSUMER = "drainer"
# The single stream field holding one serialize_history_entry blob, opaque to
# transport so every wire rule stays in guild_state.py.
OUTBOX_FIELD = b"e"
# 24h idle expiry. Never applied to the history key: that list is capped rather
# than expired (HISTORY_CACHE_LIMIT, push_history) and is the only thing
# -history reads, so a guild quiet for a day must still answer.
GUILD_TTL = 86400
# The retention cap, the display cap and the -history ceiling at once.
# push_history LTRIMs to this on every write and PERSISTs the list;
# guild_history.HISTORY_MAX_LIMIT is pinned to it. "Slots", not "plays": a
# corrupt or duplicate entry shortens the answer by one. Raising it costs ~625 B
# per entry per guild, permanently. See docs/ARCHITECTURE.md#history-read-path.
HISTORY_CACHE_LIMIT = 50

# Transient per-song fields and the playback-position fields, cleared together
# on song end / disconnect (clear_song_end_state, clear_connection). An older
# image's copy of this tuple cannot name fields added since, so after `just up
# <older-sha>` from_crashed_state can read a value from a song that finished
# under the old build; rewritten on every song start, so the exposure is one
# restore read.
_TRANSIENT_SONG_FIELDS = (
    StateField.CURRENT_SONG_URL,
    StateField.CURRENT_SONG_TITLE,
    StateField.CURRENT_SONG_DURATION,
    StateField.CURRENT_SONG_UPLOADER,
    StateField.CURRENT_SONG_REQUESTER_ID,
    StateField.CURRENT_SONG_INTERJECTED,
    StateField.CURRENT_SONG_IS_RESUME,
    StateField.CURRENT_SONG_START_PAUSED,
    StateField.CURRENT_SONG_QUEUED_AT,
    StateField.CURRENT_SONG_QUEUE_POSITION,
    StateField.CURRENT_SONG_QUERY_SOURCE,
    StateField.CURRENT_SONG_USER_INPUT,
    StateField.CURRENT_SONG_PLAYED_AT,
)
_PLAYBACK_POSITION_FIELDS = (
    StateField.PLAY_START_EPOCH,
    StateField.TOTAL_PAUSE_SECONDS,
    StateField.PAUSE_START_EPOCH,
    StateField.LAST_POSITION_SECS,
    StateField.LAST_HEARTBEAT_EPOCH,
)


def _hset_mapping(mapping: dict[str, str]) -> Mapping[FieldT, EncodableT]:
    """Variance workaround only: Mapping's key parameter is invariant, so
    dict[str, str] is not assignable to Mapping[FieldT, EncodableT]."""
    return cast(Mapping[FieldT, EncodableT], mapping)


def _fmt_position(secs: float) -> str:
    """Encode a playback position for LAST_POSITION_SECS — the one definition
    for the seed and the heartbeat. Milliseconds: position_secs accumulates
    20ms frames, so repr carries float-error tails, and the reader truncates
    to whole seconds anyway."""
    return f"{max(0.0, secs):.3f}"


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
        # redis-py's OWN exception classes: `redis.exceptions.ConnectionError`
        # does not derive from builtins.ConnectionError, so listing the builtin
        # would match nothing redis-py raises — invisibly, since every store
        # method logs-and-swallows.
        retry_on_error=[RedisConnectionError, RedisTimeoutError],
        # 3 attempts over ExponentialBackoff's 8ms→512ms covers an ordinary
        # restart. `redis.asyncio.retry.Retry`, not `redis.retry.Retry`: same
        # name and constructor, but only the async one awaits, and only attempt
        # counts tell them apart. docs/ARCHITECTURE.md#redis-connection-retry.
        retry=Retry(ExponentialBackoff(), 3),
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
    """Get and orjson-decode a cached value. None on miss, error, or redis=None."""
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
    """orjson-encode and set a value with TTL. No-ops when redis is None; errors
    are logged and swallowed."""
    if redis is None:
        return
    try:
        await redis.set(key, orjson.dumps(value), ex=ttl)
    except Exception as e:
        log.warning(f"cache_set failed [{key}]: {e}")


async def cache_del(redis: Optional[aioredis.Redis], key: str) -> bool:
    """Drop a cached value. Returns whether an entry was actually removed, so a
    caller never announces a deletion that did not occur; False on a no-op or
    an error."""
    if redis is None:
        return False
    try:
        return bool(await redis.delete(key))
    except Exception as e:
        log.warning(f"cache_del failed [{key}]: {e}")
        return False


# ── Spotify auth token cache ──────────────────────────────────────────────────
# Not cache_get/cache_set: the token is a raw string scalar, and orjson would
# double-encode it.

_SPOTIFY_TOKEN_KEY = "spotify:auth:token"


async def spotify_token_set(
    redis: Optional[aioredis.Redis], token: str, expires_in: int
) -> None:
    """Store a Spotify bearer token with TTL = expires_in − 30s. Skips caching
    when that margin would consume the token's life — a floor that raised the
    TTL would serve other processes an expired token."""
    if redis is None:
        return
    ttl = expires_in - 30
    if ttl <= 0:
        return
    try:
        await redis.set(_SPOTIFY_TOKEN_KEY, token, ex=ttl)
    except Exception as e:
        log.warning(f"spotify_token_set failed: {e}")


async def spotify_token_get_with_ttl(
    redis: Optional[aioredis.Redis],
) -> Optional[tuple[str, int]]:
    """(token, seconds_remaining) for the cached bearer token, or None on
    miss/error/expired. GET+TTL in one round trip so the caller can size its
    local expiry to the token's real remaining life."""
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


# ── -analytics rendered chart cache ───────────────────────────────────────────
# Raw bytes, not cache_get/cache_set (orjson would base64 a PNG); the pool is
# decode_responses=False, so redis.get() hands back bytes. Both keys are TTL'd,
# so they stay volatile-lru candidates and rule 12's non-evictable keys are
# untouched. See docs/ARCHITECTURE.md#analytics-rendering.


async def analytics_png_set(
    redis: Optional[aioredis.Redis], key: str, png: bytes, ttl: int
) -> None:
    """Store a rendered chart. No-ops when redis is None, when the TTL is not
    positive (the aggregate straddled a midnight), or on any Redis error."""
    if redis is None or ttl <= 0:
        return
    try:
        await redis.set(key, png, ex=ttl)
    except Exception as e:
        log.warning(f"analytics_png_set failed [{key}]: {e}")


async def analytics_png_get(
    redis: Optional[aioredis.Redis], key: str
) -> Optional[bytes]:
    """The cached chart, or None on miss/error. The key carries a digest of the
    aggregate it was rendered FROM, so a stale entry misses."""
    if redis is None:
        return None
    try:
        value = await redis.get(key)
        return value if isinstance(value, bytes) else None
    except Exception as e:
        log.warning(f"analytics_png_get failed [{key}]: {e}")
        return None


# ── History outbox (drain side) ───────────────────────────────────────────────
# Consumed only by HistoryOutboxDrainer (history_archive.py). Unlike the cache
# helpers above, these DO raise on Redis failure: the drainer's backoff loop is
# the error handler, and a swallowed error would look like an empty outbox and
# silently stall the drain. Raw bytes in/out; wire parsing stays in
# guild_state.py. Every command is idempotent under re-send (XACK and XDEL
# return 0 for a settled ID, XTRIM MINID names an absolute ID), which is what
# makes the drain path safe on a pool with retries enabled. `XTRIM MAXLEN` is
# not — "keep the newest n" re-sent after concurrent XADDs destroys a second
# tranche — and must not be introduced.


@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxEntry:
    """One delivered stream entry: its ID, and its payload if it still has one.
    kw_only because both fields are bytes-shaped and a transposition would
    type-check.

    `wire is None` is a TOMBSTONE: the body was deleted while the entry was
    still pending (XTRIM and an operator XDEL both ignore the PEL), so the read
    returns the ID with an empty field map. It must not reach
    parse_history_entry; the drainer acks it and logs a lost play, since left
    pending it replays every cycle forever on a key volatile-lru cannot reclaim.
    """

    id: bytes
    wire: Optional[bytes]


def _parse_outbox_reply(reply: Any) -> list[OutboxEntry]:
    """Flatten redis-py's XREADGROUP reply, [[stream, [(id, {field: value}),
    ...]], ...]. The two empty cases differ in shape (`>` returns `[]`, `0`
    with an empty PEL returns `[[key, []]]`), hence the tolerant loop.
    `.get(OUTBOX_FIELD)` is the tombstone rule: a KeyError is neither a
    ResponseError nor a parse failure, so it would reach the generic backoff
    and replay the same entry forever."""
    out: list[OutboxEntry] = []
    for _key, entries in cast(list[Any], reply) or []:
        for entry_id, fields in cast(list[Any], entries):
            out.append(
                OutboxEntry(
                    id=cast(bytes, entry_id),
                    wire=cast(Optional[bytes], fields.get(OUTBOX_FIELD)),
                )
            )
    return out


async def ensure_outbox_group(redis: aioredis.Redis) -> None:
    """Create the consumer group if it is missing, tolerating only BUSYGROUP.
    BUSYGROUP, WRONGTYPE and NOGROUP are all plain ResponseError, and WRONGTYPE
    (a pre-stream list at the key) must abort startup: push_history is a
    @_guild_op method, so its XADD failure would be one warning per song, and
    it takes guild:{id}:history down with it (one MULTI/EXEC). Enabled mode
    only: with the archive off setup_hook never calls this, since MKSTREAM
    would create the non-evictable key. id="0", not redis-py's "$", which
    skips every entry already in the stream."""
    try:
        await redis.xgroup_create(
            HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, id="0", mkstream=True
        )
    except ResponseError as e:
        if not str(e).startswith("BUSYGROUP"):
            raise


async def read_outbox_pending(redis: aioredis.Redis, count: int) -> list[OutboxEntry]:
    """Re-deliver this consumer name's still-unacked entries, oldest first.
    Runs every cycle: under a shared name this is what recovers a peer
    SIGKILLed mid-batch. noack=False is spelled out although it is the default:
    noack=True would deliver without entering the PEL, turning at-least-once
    into at-most-once, and no fakeredis assertion can see the difference."""
    return _parse_outbox_reply(
        await redis.xreadgroup(
            HISTORY_OUTBOX_GROUP,
            HISTORY_OUTBOX_CONSUMER,
            {HISTORY_OUTBOX_KEY: "0"},
            count=count,
            noack=False,
        )
    )


async def read_outbox_new(redis: aioredis.Redis, count: int) -> list[OutboxEntry]:
    """Claim never-before-delivered entries into this consumer's PEL. Two live
    drainers get DISJOINT sets here — the server guarantees it."""
    return _parse_outbox_reply(
        await redis.xreadgroup(
            HISTORY_OUTBOX_GROUP,
            HISTORY_OUTBOX_CONSUMER,
            {HISTORY_OUTBOX_KEY: ">"},
            count=count,
            noack=False,  # see read_outbox_pending — at-least-once depends on it
        )
    )


async def retire_outbox(redis: aioredis.Redis, ids: Sequence[bytes]) -> None:
    """Settle entries by ID — call only after their Postgres INSERT committed.
    XACK then XDEL, transactionally, in that order: XACK alone leaves the entry
    on a key with no TTL and no eviction, and a crash between them leaves an
    acked-but-undeleted entry the cap's MINID trim reclaims. The reverse order
    leaves an unrecoverable tombstone (OutboxEntry)."""
    if not ids:
        return
    async with redis.pipeline(transaction=True) as pipe:
        pipe.xack(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, *ids)
        pipe.xdel(HISTORY_OUTBOX_KEY, *ids)
        await pipe.execute()


async def outbox_depth(redis: aioredis.Redis) -> int:
    """Entries present in the stream — the drainer's backlog metric. It
    over-reports after a crash between XACK and XDEL (harmless) and
    UNDER-reports when entries were trimmed while still pending, since the
    bodies are gone while the PEL records survive. XINFO GROUPS' `lag` is no
    better: nil whenever entries were deleted in a way Redis cannot reconcile,
    which retire_outbox does every cycle. The exact measure is XPENDING + XLEN."""
    return await redis.xlen(HISTORY_OUTBOX_KEY)


async def outbox_pending_count(redis: aioredis.Redis) -> int:
    """Entries delivered but not yet acked, across all consumers in the group."""
    summary = cast(
        dict[str, Any], await redis.xpending(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP)
    )
    return int(summary["pending"])


# Ceiling on one outbox_pending_below scan. The PEL is bounded by BATCH_SIZE x
# live drainers, so this is a runaway guard, not a page.
_PENDING_SCAN_LIMIT = 10_000


async def outbox_pending_below(redis: aioredis.Redis, minid: bytes) -> list[bytes]:
    """Delivered-but-unacked IDs strictly older than `minid` — the set a trim
    is about to destroy while a drainer still holds it. XTRIM is blind to the
    PEL, so without this the cap leaves tombstones that replay forever. Bounded
    by BATCH_SIZE x live drainers, since the drain cycle never reads `>` with a
    non-empty PEL. [] when the group has vanished (an operator DEL racing the
    cap): nothing is pending, so there is nothing to ack."""
    try:
        detail = cast(
            list[dict[str, Any]],
            await redis.xpending_range(
                HISTORY_OUTBOX_KEY,
                HISTORY_OUTBOX_GROUP,
                min="-",
                max=_prev_stream_id(minid),
                count=_PENDING_SCAN_LIMIT,
            ),
        )
    except ResponseError as e:
        if not str(e).startswith("NOGROUP"):
            raise
        return []
    return [cast(bytes, row["message_id"]) for row in detail]


async def ack_outbox(redis: aioredis.Redis, ids: Sequence[bytes]) -> None:
    """Clear PEL records without archiving the entries — the cap only. Separate
    from retire_outbox so that name keeps meaning "these reached Postgres";
    this one means "these are being destroyed on purpose". XACK only: the
    bodies go with the caller's single MINID trim."""
    if not ids:
        return
    await redis.xack(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, *ids)


def _prev_stream_id(entry_id: bytes) -> bytes:
    """The ID immediately before `entry_id`: XPENDING's range is inclusive at
    both ends while MINID's is exclusive below. Both halves of `<ms>-<seq>` are
    64-bit, so at seq 0 it borrows from the millisecond half; b"0-0" has
    nothing below it."""
    ms, _, seq = entry_id.partition(b"-")
    ms_i, seq_i = int(ms), int(seq)
    if seq_i:
        return b"%d-%d" % (ms_i, seq_i - 1)
    if ms_i:
        return b"%d-%d" % (ms_i - 1, (1 << 64) - 1)
    return b"0-0"


async def trim_outbox_below(redis: aioredis.Redis, minid: bytes) -> int:
    """Drop every entry older than `minid` without archiving it, returning the
    number destroyed — the opt-in HISTORY_OUTBOX_MAX cap only. MINID, never
    MAXLEN (see the section comment). approximate=False is required: redis-py
    defaults it to True, which trims to node boundaries and on a small stream
    trims NOTHING while reporting success, and fakeredis models it as exact.
    The caller logs this value rather than `depth - cap`: XLEN over-counts
    acked-but-undeleted entries."""
    return await redis.xtrim(HISTORY_OUTBOX_KEY, minid=minid, approximate=False)


async def reclaim_outbox_stale(
    redis: aioredis.Redis, *, min_idle_ms: int, count: int, max_passes: int
) -> tuple[int, int]:
    """Sweep the group's PEL: reclaim long-idle entries, purge tombstones.
    Returns (reclaimed, purged). Besides XACK this is the only thing that
    clears a tombstone on Redis 7, and the only thing that reaches an ORPHANED
    PEL under a consumer name no live process reads. Claims into our OWN name
    too — a no-op for live entries we hold, but what purges tombstones.

    min_idle_ms must exceed the drain deadline: under a shared name "idle" is
    measured from last delivery, so a shorter value reclaims a live sibling's
    in-flight batch. justid=True is not used: redis-py then returns only the
    claimed-ID list, discarding the cursor AND the deleted-ID list. Requires
    Redis 7.0+ for the 3-element reply.

    The loop terminates on "this pass found nothing new" and counts distinct
    ids, not on `cursor == b"0-0"` alone: fakeredis returns the last-scanned ID
    for a completed scan, which fed back as an inclusive start re-delivers
    entries already counted."""
    cursor: bytes = b"0-0"
    seen_claimed: set[bytes] = set()
    seen_purged: set[bytes] = set()
    for _ in range(max_passes):
        next_cursor, claimed, deleted = cast(
            tuple[bytes, list[Any], list[bytes]],
            await redis.xautoclaim(
                HISTORY_OUTBOX_KEY,
                HISTORY_OUTBOX_GROUP,
                HISTORY_OUTBOX_CONSUMER,
                min_idle_time=min_idle_ms,
                start_id=cursor,
                count=count,
            ),
        )
        fresh = {cast(bytes, mid) for mid, _fields in claimed} - seen_claimed
        fresh_deleted = {cast(bytes, mid) for mid in deleted} - seen_purged
        if not fresh and not fresh_deleted:
            break
        seen_claimed |= fresh
        seen_purged |= fresh_deleted
        cursor = next_cursor
        if cursor == b"0-0":
            break
    return len(seen_claimed), len(seen_purged)


# ── Multi-guild config reads ─────────────────────────────────────────────────

# Commands per pipeline, so one enormous bot cannot buffer every guild's reply
# into a single response.
_CONFIG_READ_BATCH: Final[int] = 250


async def read_guild_configs(
    redis: aioredis.Redis, guild_ids: Sequence[int]
) -> dict[int, GuildConfig]:
    """Read many guilds' stored configs, batched onto pipelines. Returns an
    entry ONLY for a guild whose read happened: a missing guild means "could
    not read", not the all-unset GuildConfig an absent hash yields, and a
    caller that caches this must not treat the two alike or a Redis blink reads
    as every guild un-choosing everything. Pipelined rather than one awaited
    HGETALL each: the pool RAISES rather than queueing past its cap, so a plain
    fan-out fails every guild past it."""
    configs: dict[int, GuildConfig] = {}
    ids = list(guild_ids)
    for start in range(0, len(ids), _CONFIG_READ_BATCH):
        batch = ids[start : start + _CONFIG_READ_BATCH]
        try:
            # transaction=False: independent reads with nothing to make atomic.
            pipe = redis.pipeline(transaction=False)
            for guild_id in batch:
                pipe.hgetall(GUILD_CONFIG_KEY.format(guild_id=guild_id))
            replies = await pipe.execute()
        except Exception as e:  # noqa: BLE001 — reported by omission, see above
            log.warning(f"config read failed for {len(batch)} guilds: {e}")
            continue
        for guild_id, raw in zip(batch, replies):
            configs[guild_id] = GuildConfig.from_redis(cast(dict[bytes, bytes], raw))
    return configs


# ── Guild-scoped Redis store ──────────────────────────────────────────────────

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _guild_op(
    default: Any = None,
    default_factory: Optional[Callable[[], Any]] = None,
) -> Callable[
    [Callable[Concatenate[GuildRedisStore, _P], Awaitable[_R]]],
    Callable[Concatenate[GuildRedisStore, _P], Awaitable[_R]],
]:
    """GuildRedisStore's 'log, never raise' contract: on any exception, log
    `[guild:{id}] {method} failed: {e}` and return `default`. Pass `default`
    for immutable fallbacks and `default_factory` for anything mutable — a
    decorator argument is evaluated once at class-body time, so `default=[]`
    hands the same list to every guild on every failure. `default` is typed
    Any, not `_R`: pinning it would let `default=None` collapse `_R` to `None`
    for the Optional-returning readers."""

    def decorator(
        func: Callable[Concatenate[GuildRedisStore, _P], Awaitable[_R]],
    ) -> Callable[Concatenate[GuildRedisStore, _P], Awaitable[_R]]:
        @wraps(func)
        async def wrapper(
            self: GuildRedisStore, *args: _P.args, **kwargs: _P.kwargs
        ) -> _R:
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                log.warning(f"[guild:{self.guild_id}] {func.__name__} failed: {e}")
                return default_factory() if default_factory is not None else default

        return wrapper

    return decorator


class GuildRedisStore:
    """All Redis IO for a single guild. Every method logs errors and never
    raises — @_guild_op applies the try/except, so each body is its Redis
    happy path."""

    def __init__(self, redis: aioredis.Redis, guild_id: int) -> None:
        self.redis = redis
        self.guild_id = guild_id
        # Set by acquire_recovery_lock, consumed by release_recovery_lock: the
        # scope is one store object, which restore_guild builds per attempt.
        self._recovery_lock_token: Optional[str] = None

    # Key helpers

    def queue_key(self) -> str:
        return GUILD_QUEUE_KEY.format(guild_id=self.guild_id)

    def state_key(self) -> str:
        return GUILD_STATE_KEY.format(guild_id=self.guild_id)

    def history_key(self) -> str:
        return GUILD_HISTORY_KEY.format(guild_id=self.guild_id)

    def now_playing_key(self) -> str:
        return GUILD_NOW_PLAYING_KEY.format(guild_id=self.guild_id)

    def config_key(self) -> str:
        return GUILD_CONFIG_KEY.format(guild_id=self.guild_id)

    def _pipe_expire_all(self, pipe: Pipeline) -> None:
        """Queue EXPIREs for the TTL-managed guild keys. Two keys are absent:
        history is bounded by length and is the only thing -history reads, so
        a TTL here answers a guild that played hundreds of songs with silence;
        config holds choices, and a choice that expires is one the user is
        never told was undone."""
        pipe.expire(self.queue_key(), GUILD_TTL)
        pipe.expire(self.state_key(), GUILD_TTL)
        pipe.expire(self.now_playing_key(), GUILD_TTL)

    async def _exec_with_state_ttl(self, pipe: Pipeline) -> None:
        """Append the state-key TTL refresh and execute. EXPIRE must follow the
        queued writes: on a not-yet-created key it is a no-op."""
        pipe.expire(self.state_key(), GUILD_TTL)
        await pipe.execute()

    # Queue operations

    @_guild_op(default=False)
    async def push_queue(self, entry: QueueEntry) -> bool:
        """RPUSH one queue entry and refresh TTL on all guild keys. Reports whether
        it landed: the failure is swallowed here, and the caller has no other way to
        learn its mirror is now short of the deque."""
        pipe = self.redis.pipeline()
        pipe.rpush(self.queue_key(), entry.to_redis())
        self._pipe_expire_all(pipe)
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def push_queue_batch(self, entries: Sequence[QueueEntry]) -> bool:
        """RPUSH all entries in one round-trip and refresh TTL on all guild keys.
        Reports whether it landed, like push_queue; nothing to write is a landed
        write, since the mirror already agrees with the deque."""
        if not entries:
            return True
        pipe = self.redis.pipeline()
        pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
        self._pipe_expire_all(pipe)
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def push_queue_front(self, entries: Sequence[QueueEntry]) -> bool:
        """LPUSH entries so entries[0] ends up at the queue head — the interjection
        front insert; reversed first because LPUSH sends each successive
        argument to the head. A swallowed failure here leaves memory
        len(entries) ahead of Redis at the HEAD, so later LPOPs retire other
        songs' entries — the landed bool is what lets put_front mark the mirror
        stale instead."""
        if not entries:
            return True
        pipe = self.redis.pipeline()
        pipe.lpush(self.queue_key(), *[e.to_redis() for e in reversed(entries)])
        self._pipe_expire_all(pipe)
        await pipe.execute()
        return True

    @_guild_op(default=None)
    async def pop_queue(self) -> None:
        # At-most-once: LPOP has no ack, so a crash after this loses the song
        # from Redis. Accepted — the in-memory deque is the source of truth.
        await self.redis.lpop(self.queue_key())

    def _now_playing_state_mapping(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        start_offset: float = 0.0,
    ) -> dict[str, str]:
        """The current_song_* state fields ARE a parked queue entry — the one
        signature enforcing the identity SongQueueEntry.from_song()/
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
            StateField.CURRENT_SONG_IS_RESUME: ("1" if current.is_resume else ""),
            StateField.CURRENT_SONG_START_PAUSED: ("1" if current.start_paused else ""),
            StateField.CURRENT_SONG_QUEUED_AT: str(current.queued_at),
            StateField.CURRENT_SONG_QUEUE_POSITION: str(current.queue_position),
            StateField.CURRENT_SONG_QUERY_SOURCE: current.query_source,
            StateField.CURRENT_SONG_USER_INPUT: current.user_input or "",
            StateField.CURRENT_SONG_PLAYED_AT: str(current.played_at),
            StateField.PLAY_START_EPOCH: str(play_start_epoch),
            StateField.TOTAL_PAUSE_SECONDS: "0",
            # Seeded so a position exists before the first tick: a crash inside
            # that interval would otherwise resume at 0:00, not the -ss offset.
            StateField.LAST_POSITION_SECS: _fmt_position(start_offset),
            StateField.LAST_HEARTBEAT_EPOCH: str(play_start_epoch + start_offset),
        }

    def _start_song_pipeline(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData],
        start_offset: float,
    ) -> Pipeline:
        """The state and snapshot legs every song start writes, in one MULTI;
        the caller adds its queue leg and executes. `now_playing` rides the same
        transaction, so a crash can never leave state pointing at song B while
        the snapshot shows song A."""
        mapping = self._now_playing_state_mapping(
            current, play_start_epoch, start_offset
        )
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(self.state_key(), mapping=_hset_mapping(mapping))
        pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
        pipe.expire(self.state_key(), GUILD_TTL)
        if now_playing is not None:
            pipe.hset(
                self.now_playing_key(),
                mapping=_hset_mapping(now_playing.to_redis_mapping()),
            )
            pipe.expire(self.now_playing_key(), GUILD_TTL)
        return pipe

    @_guild_op(default=False)
    async def pop_queue_and_start_song(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
        start_offset: float = 0.0,
    ) -> bool:
        """Atomically LPOP the queue and park `current`'s fields in the state
        hash, so a crash observes the song either still queued or dequeued with
        every now-playing field set, never absent from both. Returns whether
        the transaction landed, and THE CALLER MUST CHECK: the in-memory settle
        already happened, so a swallowed failure leaves the list holding an
        entry memory does not (GuildQueue.note_mirror_write)."""
        pipe = self._start_song_pipeline(
            current, play_start_epoch, now_playing, start_offset
        )
        pipe.lpop(self.queue_key())
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def rebuild_queue_and_start_song(
        self,
        current: SongQueueEntry,
        entries: Sequence[QueueEntry],
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
        start_offset: float = 0.0,
    ) -> bool:
        """pop_queue_and_start_song with the list REPLACED by `entries` instead
        of LPOPed — for a mirror the caller knows is stale. DELETE + RPUSH ride
        the same MULTI as the state fields. Returns whether it landed."""
        pipe = self._start_song_pipeline(
            current, play_start_epoch, now_playing, start_offset
        )
        pipe.delete(self.queue_key())
        if entries:
            pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
            pipe.expire(self.queue_key(), GUILD_TTL)
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def set_current_song_state(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
        start_offset: float = 0.0,
    ) -> bool:
        """pop_queue_and_start_song without the LPOP — for a crash-recovered
        "current song" that was never RPUSHed. Returns whether it landed."""
        pipe = self._start_song_pipeline(
            current, play_start_epoch, now_playing, start_offset
        )
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def delete_queue(self) -> bool:
        """DELETE the queue key. Returns whether it landed."""
        await self.redis.delete(self.queue_key())
        return True

    @_guild_op(default=False)
    async def rebuild_queue(self, entries: Sequence[QueueEntry]) -> bool:
        """DELETE + RPUSH all entries in one MULTI, so a concurrent LPOP never
        sees an empty window. Returns whether it landed."""
        pipe = self.redis.pipeline(transaction=True)
        pipe.delete(self.queue_key())
        pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
        pipe.expire(self.queue_key(), GUILD_TTL)
        await pipe.execute()
        return True

    @_guild_op(default=0)
    async def remove_queue_entries(self, entries: Sequence[QueueEntry]) -> int:
        """LREM the given entries out of the list, leaving the rest in place.
        Returns HOW MANY were removed — the caller must check it: LREM matches
        exact serialized bytes, so a queued object mutated after its entry was
        written matches nothing, and a short count means only a rebuild can be
        trusted (a Redis failure returns 0 and takes the same path). Counted per
        distinct serialization, never `LREM ... 0`: two enqueues of one song can
        serialize identically, and "all matching" would take out the copy still
        queued."""
        counts: dict[bytes, int] = {}
        for entry in entries:
            blob = entry.to_redis()
            counts[blob] = counts.get(blob, 0) + 1
        pipe = self.redis.pipeline(transaction=True)
        for blob, count in counts.items():
            # redis-py's stub types `value` as str; its encoder takes bytes.
            pipe.lrem(self.queue_key(), count, blob)  # pyright: ignore[reportArgumentType]
        pipe.expire(self.queue_key(), GUILD_TTL)
        replies = await pipe.execute()
        return sum(cast(list[int], replies[:-1]))  # the last reply is the EXPIRE

    # History operations

    # ISSUE: non-evictable keys can exhaust Redis and stall ALL writes.
    # Three kinds of key carry no TTL (guild:{id}:history, guild:{id}:config,
    # HISTORY_OUTBOX_KEY), so under volatile-lru they are never evicted; once
    # they fill maxmemory Redis rejects every write with OOM, and each store
    # method swallows it, so persistence degrades silently. Only the OUTBOX can
    # get there by growing: history lists are capped per guild (~24 KB each)
    # and config is a handful of fields, but the outbox grows for the whole of
    # a Postgres outage at ~625 B per play (256mb holds ~429k; see
    # HistoryOutboxDrainer.CAP_PAGE for the listpack cliff behind that figure).
    # HISTORY_OUTBOX_MAX is the opt-in bound and dropping there is real data
    # loss. The history trim is lazy — it runs inside push_history only — so a
    # dormant guild keeps whatever oversized list it already had until its next
    # play or a manual DEL. A memory/eviction alarm is still owed. Do not switch
    # to allkeys-lru: an evicted outbox entry is a play that vanishes with no
    # error and no log line. See docs/ARCHITECTURE.md#redis-memory-bounds.
    @_guild_op(default=None)
    async def push_history(self, entry: HistoryEntry) -> None:
        """LPUSH one entry, cap and PERSIST the list, and — while the archive is
        enabled — mirror it onto the Postgres outbox, in one transaction.

        LTRIM + PERSIST are the whole retention policy, unconditional in both
        archive modes: bounded because an unbounded non-evictable key is the OOM
        shape above, permanent because -history reads this list and nothing
        else (the PERSIST also self-heals an older build's 24h expiry). See
        docs/ARCHITECTURE.md#history-read-path. With the archive disabled the
        XADD leg is absent and the outbox key is never created.

        On the SWALLOWING side of the split, unlike the drain helpers: the
        playback loop must never die because Redis blinked. So the producer can
        never report a mis-shaped outbox, which is why ensure_outbox_group()
        aborts at STARTUP instead."""
        wire = serialize_history_entry(entry)
        try:
            await self._push_history_pipeline(wire)
        except OutOfMemoryError:
            # A full Redis cannot self-heal without this: LPUSH is denyoom and
            # queued first, so with nothing evictable the server refuses it at
            # queue time and EXEC aborts before the LTRIM runs. A bare LTRIM is
            # not denyoom, so the one command that can make room can still
            # run — but it frees something only for an oversized legacy list;
            # at the steady state the retry covers memory freed elsewhere. LLEN
            # first (reads are not denyoom) to skip a trim that frees nothing.
            length = await self.redis.llen(self.history_key())
            if length > HISTORY_CACHE_LIMIT:
                log.warning(
                    f"guild {self.guild_id}: Redis is at maxmemory and refused "
                    f"the history write; trimming {length} entries to "
                    f"{HISTORY_CACHE_LIMIT} and retrying"
                )
                await self.redis.ltrim(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
            else:
                log.warning(
                    f"guild {self.guild_id}: Redis is at maxmemory and refused "
                    f"the history write. This guild's list is already at the "
                    f"{HISTORY_CACHE_LIMIT}-entry cap, so trimming would free "
                    f"nothing — retrying once, but this play is likely LOST. "
                    f"Free memory (usually: drain history:outbox by restoring "
                    f"Postgres) or raise maxmemory."
                )
            await self._push_history_pipeline(wire)

    async def _push_history_pipeline(self, wire: bytes) -> None:
        """The one transactional write, so the OOM path can re-issue it."""
        pipe = self.redis.pipeline()
        pipe.lpush(self.history_key(), wire)
        # Unpaged: every push after the first trims 51→50, so this is O(1) in
        # the steady state; only a list from a build that did not cap pays more,
        # once (~22ms at 500k entries).
        pipe.ltrim(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
        # No EXPIRE anywhere near this key — length bounds it, time never does.
        # PERSIST because -history has no other source, and it clears an older
        # build's inherited TTL in the same write.
        pipe.persist(self.history_key())
        # The consent gate for long-term storage: disabled, nothing may
        # accumulate for a drainer that does not exist, and the XADD would
        # create the non-evictable key. Read per call through the module so
        # tests can patch the function or the environment. setup_hook reads the
        # flag first, since @_guild_op would swallow the parser's ValueError
        # into one warning per song.
        if config.history_archive_enabled():
            pipe.xadd(HISTORY_OUTBOX_KEY, {OUTBOX_FIELD: wire})
        await pipe.execute()

    @_guild_op(default=None)
    async def history_ttl(self) -> Optional[int]:
        """This guild's history list TTL in redis's vocabulary: -1 is no expiry
        (the invariant golden rule 12 protects), -2 is no such key. For -debug's
        check row; None means Redis did not answer."""
        return int(await self.redis.ttl(self.history_key()))

    @_guild_op(default_factory=list)
    async def get_history(self) -> list[HistoryEntry]:
        """Up to HISTORY_CACHE_LIMIT entries, most recently RECORDED first —
        song-end order, not played_at order (GuildHistory.recent sorts). Corrupt
        entries are dropped with a warning."""
        raw = await self.redis.lrange(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
        return [e for e in map(parse_history_entry, raw) if e is not None]

    # Now-playing operations (writes happen inside pop_queue_and_start_song()/
    # set_current_song_state(), atomically with the rest of the start state)

    @_guild_op(default=None)
    async def get_now_playing(self) -> Optional[NowPlayingData]:
        """HGETALL the now_playing hash. None on miss or error alike: the only
        caller restores a display embed, and "no embed" is right for both."""
        # Bytes, not str: create_redis_pool() sets decode_responses=False, which
        # redis-py's return type cannot express. from_redis() decodes, so a
        # decoded pool breaks it at runtime.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.now_playing_key()))
        return NowPlayingData.from_redis(raw)

    # Playback position tracking

    @_guild_op(default=None)
    async def set_playback_start(self, epoch: float) -> None:
        """Record that playback started at `epoch`, resetting pause accounting.
        For unit tests and standalone use; loop() writes these fields via
        pop_queue_and_start_song()."""
        pipe = self.redis.pipeline()
        pipe.hset(self.state_key(), StateField.PLAY_START_EPOCH, str(epoch))
        pipe.hset(self.state_key(), StateField.TOTAL_PAUSE_SECONDS, "0")
        pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
        await self._exec_with_state_ttl(pipe)

    @_guild_op(default=None)
    async def on_pause(self, epoch: float) -> None:
        """Record the epoch when the voice client was paused. LEGACY: feeds only
        _legacy_wall_clock_position_at (MusicPlayer.pause records the exact
        position through heartbeat()). Drop with on_resume and the three
        wall-clock StateFields one release after the heartbeat ships."""
        pipe = self.redis.pipeline()
        pipe.hset(self.state_key(), StateField.PAUSE_START_EPOCH, str(epoch))
        await self._exec_with_state_ttl(pipe)

    @_guild_op(default=None)
    async def on_resume(self, resume_epoch: float) -> None:
        """Accumulate elapsed pause time into total_pause_seconds and clear
        pause_start_epoch. LEGACY, like on_pause. Non-atomic read-modify-write:
        assumes one writer per guild, and must become a Lua script or a
        WATCH/MULTI loop under multi-process sharding."""
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

    @_guild_op(default=None)
    async def clear_song_end_state(self) -> None:
        """HDEL every current_song_*/position field and DELETE the now_playing
        hash in one round-trip, so *absent* is the one representation of "no
        song" (as in clear_connection)."""
        pipe = self.redis.pipeline()
        pipe.hdel(
            self.state_key(),
            *_TRANSIENT_SONG_FIELDS,
            *_PLAYBACK_POSITION_FIELDS,
        )
        pipe.delete(self.now_playing_key())
        await pipe.execute()

    # State operations

    @_guild_op(default=None)
    async def get_guild_state(self) -> Optional[GuildStateData]:
        """HGETALL the state hash: zero-value GuildStateData when the hash is
        missing, None when the read itself failed, so callers can tell "nothing
        stored" from "Redis unavailable" (recovery.restore_guild). Pure read;
        refresh_ttl() at the end of _restore_state() covers the recovery window."""
        # Same decode_responses=False invariant as get_now_playing() above.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.state_key()))
        return GuildStateData.from_redis(raw)

    @_guild_op(default=None)
    async def get_recovery_gate(self) -> Optional[GuildRecoveryGate]:
        """State hash + pending-queue LENGTH in one pipeline, for
        recovery.restore_guild's connect/restorable gate. Transfers no queue
        contents: a -stopped guild keeps a possibly-long queue, and gating on
        LLEN keeps it off the wire on every on_ready. None on read failure."""
        pipe = self.redis.pipeline()
        pipe.hgetall(self.state_key())
        pipe.llen(self.queue_key())
        raw_state, queue_len = await pipe.execute()
        return GuildRecoveryGate(
            state=GuildStateData.from_redis(raw_state),
            pending_count=int(queue_len),
        )

    @_guild_op(default=None)
    async def get_playback_snapshot(self) -> Optional[GuildPlaybackSnapshot]:
        """The complete playback aggregate — state, queue, now-playing, history,
        config — in one pipeline, so a failure aborts the whole snapshot and
        the caller restores everything or nothing. Same error-vs-empty contract
        as get_guild_state. Not MULTI: recovery holds the guild lock during
        the window that matters."""
        pipe = self.redis.pipeline()
        pipe.hgetall(self.state_key())
        pipe.lrange(self.queue_key(), 0, -1)
        pipe.hgetall(self.now_playing_key())
        pipe.lrange(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
        pipe.hgetall(self.config_key())
        raw_state, raw_queue, raw_np, raw_history, raw_config = await pipe.execute()
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
            config=GuildConfig.from_redis(raw_config),
        )

    @_guild_op(default=False)
    async def set_volume(self, volume: float) -> bool:
        """Persist the guild volume. True when it landed. guild:{id}:config is
        the source of truth; the legacy state field is written TOO, and not
        deleted, for one release: `just up <older-sha>` reads only
        StateField.VOLUME and would otherwise reset every migrated guild to
        100%. Drop that leg with StateField.VOLUME and GuildStateData.volume."""
        # Encoded by GuildConfig.to_redis, never by hand, so the wire format
        # for volume has one definition.
        mapping = GuildConfig(volume=volume).to_redis()
        pipe = self.redis.pipeline()
        pipe.hset(self.config_key(), mapping=_hset_mapping(mapping))
        pipe.persist(self.config_key())
        pipe.hset(self.state_key(), StateField.VOLUME, mapping[ConfigField.VOLUME])
        # The legacy write can CREATE the state hash on a guild whose TTL has
        # lapsed, and a key created without an EXPIRE never expires.
        await self._exec_with_state_ttl(pipe)
        return True

    @_guild_op(default=False)
    async def migrate_volume(self, volume: float) -> bool:
        """Seed :config's volume from the legacy state field. HSETNX, never an
        overwrite: restore writes this back after an arbitrary number of awaits,
        and a `-volume` landing in that window would otherwise be clobbered by
        the older value, durably."""
        mapping = GuildConfig(volume=volume).to_redis()
        pipe = self.redis.pipeline()
        pipe.hsetnx(self.config_key(), ConfigField.VOLUME, mapping[ConfigField.VOLUME])
        pipe.persist(self.config_key())
        await pipe.execute()
        return True

    # Durable per-guild config

    @_guild_op(default_factory=GuildConfig)
    async def get_config(self) -> GuildConfig:
        """This guild's stored preferences; all-unset when nothing is stored OR
        Redis is unreachable — the same answer, since unset means "follow the
        host default" and an outage should degrade to the host's configuration."""
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.config_key()))
        return GuildConfig.from_redis(raw)

    @_guild_op(default=False)
    async def set_debug_mode(self, enabled: bool) -> bool:
        """Persist this guild's debug-mode choice. True when it landed; the
        command uses False to tell the user the setting applies to this process
        only. PERSIST because this key must never be an eviction candidate: an
        evicted config is a setting silently reverting."""
        pipe = self.redis.pipeline()
        pipe.hset(
            self.config_key(),
            mapping=_hset_mapping(GuildConfig(debug_mode=enabled).to_redis()),
        )
        pipe.persist(self.config_key())
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def set_timezone(self, name: str) -> bool:
        """Persist the IANA zone this guild renders ETAs in. True when it
        landed. Stores the name as given (GuildConfig.tzinfo resolves at read
        time). No caller yet: the write half of the planned `-options` command.
        Validated here because an unusable name stored in a PERSISTed,
        non-evictable key fails silently; `-options` should still call
        valid_timezone itself so it can tell the user WHY, since False here
        cannot say whether the name was bad or Redis was down."""
        if not valid_timezone(name):
            log.warning(f"[guild:{self.guild_id}] refusing unusable timezone {name!r}")
            return False
        pipe = self.redis.pipeline()
        pipe.hset(
            self.config_key(),
            mapping=_hset_mapping(GuildConfig(timezone=name).to_redis()),
        )
        pipe.persist(self.config_key())
        await pipe.execute()
        return True

    @_guild_op(default=False)
    async def clear_config(self) -> bool:
        """Drop this guild's stored preferences when the bot leaves it, so a
        departed guild stops occupying a key nothing will ever expire."""
        await self.redis.delete(self.config_key())
        return True

    # TTL management

    @_guild_op(default=None)
    async def refresh_ttl(self) -> None:
        """Refresh GUILD_TTL on the TTL-managed guild keys (_pipe_expire_all)."""
        pipe = self.redis.pipeline()
        self._pipe_expire_all(pipe)
        await pipe.execute()

    # Connection persistence

    @_guild_op(default=None)
    async def set_connection(self, voice_channel_id: int, text_channel_id: int) -> None:
        """Persist active voice and text channel IDs into the state hash."""
        pipe = self.redis.pipeline()
        pipe.hset(self.state_key(), StateField.VOICE_CHANNEL_ID, str(voice_channel_id))
        pipe.hset(self.state_key(), StateField.TEXT_CHANNEL_ID, str(text_channel_id))
        await self._exec_with_state_ttl(pipe)

    @_guild_op(default=None)
    async def clear_connection(self) -> None:
        """Remove all transient state on intentional disconnect: channel IDs
        (so on_ready skips recovery), now-playing display, and every song and
        position field."""
        pipe = self.redis.pipeline()
        pipe.hdel(
            self.state_key(),
            StateField.VOICE_CHANNEL_ID,
            StateField.TEXT_CHANNEL_ID,
            *_TRANSIENT_SONG_FIELDS,
            # HACK: last_author_id is dead schema still scrubbed on every disconnect.
            # Only cleans hashes left by older builds (hence the bare literal). Safe
            # to delete once no pre-migration hash can be live — one release, given
            # the 24h TTL.
            "last_author_id",
            *_PLAYBACK_POSITION_FIELDS,
        )
        pipe.delete(self.now_playing_key())
        await pipe.execute()

    @_guild_op(default=None)
    async def heartbeat(self, position_secs: float, epoch: float) -> None:
        """Record where the audio is, so recovery never has to infer it. Once
        per HEARTBEAT_INTERVAL_SECS per PLAYING guild, each an AOF append
        (~230 B). Writes the legacy fields' successor without removing them, so
        a rollback still reads. transaction=False: HSET then EXPIRE on one key
        needs no atomicity, and at this frequency MULTI/EXEC costs 12% latency
        for nothing."""
        pipe = self.redis.pipeline(transaction=False)
        pipe.hset(
            self.state_key(),
            mapping=_hset_mapping(
                {
                    StateField.LAST_POSITION_SECS: _fmt_position(position_secs),
                    StateField.LAST_HEARTBEAT_EPOCH: str(epoch),
                }
            ),
        )
        await self._exec_with_state_ttl(pipe)

    # Recovery lock — one restore_guild per guild at a time

    # Must outlast the guarded section: the voice connect is capped at 30s plus
    # a few Redis reads. Expiry is safe (release compares before deleting), so
    # an early one costs duplicate work, against a dead holder blocking recovery.
    _RECOVERY_LOCK_TTL = 60  # seconds

    def _recovery_lock_key(self) -> str:
        return f"lock:guild:{self.guild_id}:recovery"

    @_guild_op(default=False)
    async def acquire_recovery_lock(self) -> bool:
        """SET NX EX — True if this store won the lock. The value is a
        per-acquisition random token, so release can prove the lock it deletes
        is still the one this store acquired."""
        token = secrets.token_hex(16)
        result = await self.redis.set(
            self._recovery_lock_key(), token, nx=True, ex=self._RECOVERY_LOCK_TTL
        )
        if result is True:
            self._recovery_lock_token = token
            return True
        return False

    @_guild_op(default=None)
    async def release_recovery_lock(self) -> None:
        """Delete the lock only if this store still owns it. A lock that can
        expire must never be deleted blind: a holder whose lock expired
        mid-recovery would DEL its successor's, admitting a double restore.
        WATCH/MULTI, not a Lua CAS — fakeredis has no Lua interpreter, so this
        stays covered by real tests. The compare is against bytes
        (decode_responses=False)."""
        token = self._recovery_lock_token
        if token is None:
            # Never held it (or a different store object did): deleting would be
            # the foreign-lock delete this method exists to stop.
            return
        self._recovery_lock_token = None
        key = self._recovery_lock_key()
        async with self.redis.pipeline() as pipe:
            try:
                await pipe.watch(key)
                if await pipe.get(key) != token.encode():
                    # Expired, or re-acquired by someone else: not ours.
                    await pipe.unwatch()
                    return
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
            except WatchError:
                # Changed hands between the read and EXEC — same conclusion.
                pass
