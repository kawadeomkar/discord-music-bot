from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, Concatenate, Optional, ParamSpec, TypeVar, cast

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import Pipeline
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import OutOfMemoryError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.asyncio.retry import Retry
from redis.typing import EncodableT, FieldT

from src import config
from src.guild_state import (
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
)
from src.util import get_logger

log = get_logger(__name__)

GUILD_QUEUE_KEY = "guild:{guild_id}:queue"
GUILD_STATE_KEY = "guild:{guild_id}:state"
GUILD_HISTORY_KEY = "guild:{guild_id}:history"
GUILD_NOW_PLAYING_KEY = "guild:{guild_id}:now_playing"
# Global (not per-guild) write-ahead buffer for the Postgres history archive:
# entries from all guilds interleave (each carries its guild_id on the wire),
# XADDed alongside the display list and drained oldest-first by
# HistoryOutboxDrainer. NO TTL — it holds not-yet-durable entries, so under
# volatile-lru it must never be an eviction candidate; normally near-empty, it
# grows only while Postgres is unreachable. A STREAM with a consumer group, not a
# list: a list retire is `RPOP <count>` — "remove the oldest N", not "remove the N
# I just archived" — while XACK names IDs, so it cannot mean anything else.
HISTORY_OUTBOX_KEY = "history:outbox"
HISTORY_OUTBOX_GROUP = "drainers"
# Stable, not per-process: the PEL belongs to the name, so a
# starting process's `XREADGROUP ... 0` inherits whatever a predecessor or live
# sibling left in flight — no lease, no TTL, no housekeeping. Two live drainers are
# safe by construction: `>` hands them disjoint entries and `0` replays a shared
# set that ON CONFLICT DO NOTHING collapses on the Postgres side, so concurrency
# costs duplicated work, never lost data.
HISTORY_OUTBOX_CONSUMER = "drainer"
# The single stream field holding one serialize_history_entry blob. Keeping the
# orjson payload opaque leaves every wire rule in guild_state.py (parsing, domain
# clamping, compatibility) untouched by transport.
OUTBOX_FIELD = b"e"
# 24h idle expiry. Never applied to the history key: that list is capped rather
# than expired (see HISTORY_CACHE_LIMIT and push_history) and is the only thing
# -history reads, so a guild quiet for a day must still answer the command. Every
# TTL path here excludes it unconditionally.
GUILD_TTL = 86400
# The retention cap, the display cap and the -history ceiling at once — the same
# number on purpose. push_history LTRIMs the list to this on every write and
# PERSISTs it, and musicbot.HISTORY_MAX_LIMIT is pinned to this constant, so the
# command can never ask for more slots than the window holds. "Slots", not "plays":
# the equality leaves no headroom, so anything that costs a slot without yielding a
# renderable play shortens the answer by one — a corrupt entry (get_history drops
# it) or a duplicate (recent() dedups it; retry_on_error re-sends a non-idempotent
# LPUSH after the server applied EXEC). Raising it costs ~487 B per entry per guild
# in all three roles at once, permanently, since nothing expires it.
# See docs/ARCHITECTURE.md#history-read-path.
HISTORY_CACHE_LIMIT = 50

# Transient per-song fields and the playback-position fields, cleared together on
# song end / disconnect. Shared so clear_song_end_state() and clear_connection()
# can't drift by hand-editing one and forgetting the other.
_TRANSIENT_SONG_FIELDS = (
    StateField.CURRENT_SONG_URL,
    StateField.CURRENT_SONG_TITLE,
    StateField.CURRENT_SONG_DURATION,
    StateField.CURRENT_SONG_UPLOADER,
    StateField.CURRENT_SONG_REQUESTER_ID,
    StateField.CURRENT_SONG_INTERJECTED,
    StateField.CURRENT_SONG_QUEUED_AT,
    StateField.CURRENT_SONG_QUEUE_POSITION,
    StateField.CURRENT_SONG_QUERY_SOURCE,
)
_PLAYBACK_POSITION_FIELDS = (
    StateField.PLAY_START_EPOCH,
    StateField.TOTAL_PAUSE_SECONDS,
    StateField.PAUSE_START_EPOCH,
)


def _hset_mapping(mapping: dict[str, str]) -> Mapping[FieldT, EncodableT]:
    """Adapt a guild_state str→str mapping to redis-py's HSET mapping type.

    Mapping's key parameter is invariant, so dict[str, str] is not assignable to
    Mapping[FieldT, EncodableT] even though str is a member of FieldT. A variance
    workaround only — it widens nothing at runtime."""
    return cast(Mapping[FieldT, EncodableT], mapping)


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
        # redis-py's OWN exception classes, not the builtins of the same name:
        # `redis.exceptions.ConnectionError` derives from RedisError, not from
        # builtins.ConnectionError, so listing the builtin matches nothing redis-py
        # raises and connection errors get no retry at all — invisible, because every
        # store method logs-and-swallows. (The builtin TimeoutError is redundant:
        # retry_on_timeout=True already appends socket.timeout, which is
        # builtins.TimeoutError on 3.10+.)
        retry_on_error=[RedisConnectionError, RedisTimeoutError],
        # Without an explicit Retry, redis-py synthesises `Retry(NoBackoff(), 1)` for
        # a non-empty retry_on_error: one immediate reattempt, no backoff, while a
        # restarting Redis is usually gone for longer. 3 attempts over
        # ExponentialBackoff's default 8ms→512ms ceiling covers an ordinary restart.
        #
        # `redis.asyncio.retry.Retry`, not `redis.retry.Retry` — different classes,
        # same name, same constructor, same attributes, and only the async one awaits.
        # Measured against a closed port: sync = 1 connection attempt, async = 4.
        # Every assertion on the configuration passes under both classes, so only
        # attempt counts tell them apart. docs/ARCHITECTURE.md#redis-connection-retry.
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


async def cache_del(redis: Optional[aioredis.Redis], key: str) -> None:
    """Drop a cached value. No-ops when redis is None; silently ignores errors."""
    if redis is None:
        return
    try:
        await redis.delete(key)
    except Exception as e:
        log.warning(f"cache_del failed [{key}]: {e}")


# ── Spotify auth token cache ──────────────────────────────────────────────────
# Intentionally does not use cache_get/cache_set: the token is a raw string
# scalar, not JSON. Using orjson here would double-encode it as a JSON string.

_SPOTIFY_TOKEN_KEY = "spotify:auth:token"


async def spotify_token_set(
    redis: Optional[aioredis.Redis], token: str, expires_in: int
) -> None:
    """Store a Spotify bearer token as a raw string with TTL = expires_in − 30s.

    Skips caching entirely when the margin would consume the token's remaining life
    — a floor that *raised* the TTL would serve other processes an expired token.
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


# ── History outbox (drain side) ───────────────────────────────────────────────
# Consumed only by HistoryOutboxDrainer (history_archive.py). Unlike the cache
# helpers above, these DO raise on Redis failure — the drainer's backoff loop is the
# error handler, and a swallowed error here would look like an empty outbox and
# silently stall the drain. Raw bytes in/out; wire parsing stays in guild_state.py.
# There is no lease and no single-consumer requirement (see
# HISTORY_OUTBOX_CONSUMER); exactly-once `play_history_rejected` recording, which
# the lease silently provided, moved into the SQL as an ON CONFLICT clause. Every
# command below is idempotent under re-send, which is what makes the drain path safe
# on the application pool with retries enabled: XACK and XDEL return 0 for an
# already-settled ID, and XTRIM MINID names an absolute ID. `XTRIM MAXLEN` is not —
# it means "keep the newest n", so a re-send after two concurrent XADDs destroys a
# second tranche. It is not used here, and must not be introduced.


@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxEntry:
    """One delivered stream entry: its ID, and its payload if it still has one.
    kw_only because both fields are Optional[bytes]-shaped and transposing them
    would type-check silently.

    `wire is None` is a TOMBSTONE — the entry was delivered and then had its body
    deleted while still pending, which XTRIM and any operator XDEL can both produce
    because neither consults the PEL; the ID survives in the PEL, so the read
    returns it with an empty field map. It is not corrupt and must not reach
    parse_history_entry: the bytes are gone and nothing can reproduce them. The
    drainer acks it unconditionally and logs a lost play — left pending it replays
    every cycle forever, on a key volatile-lru can never reclaim.
    """

    id: bytes
    wire: Optional[bytes]


def _parse_outbox_reply(reply: Any) -> list[OutboxEntry]:
    """Flatten redis-py's XREADGROUP reply into IDs and payloads.

    The reply is [[stream_name, [(id, {field: value}), ...]], ...]. The two empty
    cases have different shapes, hence the tolerant loop rather than indexing: `>`
    with nothing new returns `[]`, while `0` with an empty PEL returns `[[key, []]]`.

    `.get(OUTBOX_FIELD)` rather than `[OUTBOX_FIELD]` is the tombstone rule: a
    KeyError is neither a ResponseError nor a parse failure, so it escapes every
    handler the drain path has and reaches the generic backoff — where the same
    entry replays forever while the outbox grows unbounded behind it.
    """
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
    """Create the consumer group if it is missing. Tolerates only BUSYGROUP.

    Not `except ResponseError: pass`. BUSYGROUP, WRONGTYPE and NOGROUP are all
    plain ResponseError — redis-py has no subclass for any of them — so a
    class-level catch swallows the two that mean "this bot cannot record history":

      BUSYGROUP   repeat create                → fine, ignore
      WRONGTYPE   a pre-stream list at the key → fatal, must abort startup
      NOGROUP     the key was deleted          → healed by the drain cycle

    Swallowing WRONGTYPE is total, silent history loss: push_history is a @_guild_op
    method, so its XADD failure is one warning per song and takes
    guild:{id}:history down with it (both legs share one MULTI/EXEC), and XLEN would
    raise too, so the backlog alarm can never fire either — hence the startup abort.
    Enabled mode only: with the archive off setup_hook never calls this, since
    creating the group would MKSTREAM the non-evictable key into existence.

    id="0" is not redis-py's default ("$" silently skips every entry already in the
    stream). MKSTREAM removes any separate create-the-key step, and repeating this
    call does not rewind an existing group.
    """
    try:
        await redis.xgroup_create(
            HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, id="0", mkstream=True
        )
    except ResponseError as e:
        if not str(e).startswith("BUSYGROUP"):
            raise


async def read_outbox_pending(redis: aioredis.Redis, count: int) -> list[OutboxEntry]:
    """Re-deliver this consumer name's still-unacked entries, oldest first.

    Runs every cycle, not only at startup: under a shared consumer name this is
    what recovers a peer SIGKILLed mid-batch, with no lease TTL to wait out.

    noack=False is redis-py's default and is spelled out anyway. noack=True would
    deliver without ever entering the PEL, turning the design's at-least-once into
    at-most-once: this "0" read would find nothing to re-deliver and a drainer that
    died mid-batch would lose its plays outright. No behavioural test can catch it
    — every fakeredis assertion still passes — so the defence is writing it down.
    """
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
    """Claim never-before-delivered entries into this consumer's PEL.

    Two live drainers get DISJOINT sets here — the server guarantees it — which is
    the half of the design that makes a lease unnecessary.
    """
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

    Both commands are required, in this order. XACK clears the
    pending record but does not remove the entry (a stream is a log, not a queue),
    and ack-without-delete grows a key that carries no TTL and is never an eviction
    candidate; XDEL frees the memory and is idempotent (missing ID → 0).

    Transactionally, XACK strictly first, a crash between them leaves an
    acked-but-undeleted entry: invisible to XREADGROUP, harmless, reclaimed by the
    cap's MINID trim. The other order — or no transaction — inverts the window to
    XDEL landing without XACK, an unrecoverable tombstone (OutboxEntry). Crashing
    before the ack simply leaves the batch pending → redelivered → deduped by the
    archive's unique index.
    """
    if not ids:
        return
    async with redis.pipeline(transaction=True) as pipe:
        pipe.xack(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, *ids)
        pipe.xdel(HISTORY_OUTBOX_KEY, *ids)
        await pipe.execute()


async def outbox_depth(redis: aioredis.Redis) -> int:
    """Entries present in the stream — the drainer's backlog watchdog metric.

    An approximation of the un-archived backlog that diverges in both directions:
    it over-reports after a crash between XACK and XDEL (harmless, reconciled by
    the MINID trim) and UNDER-reports when entries were trimmed or XDELed while
    still pending, since the bodies are gone while the PEL records survive — it
    reads low at exactly the moment plays are being lost. The cap's ack-before-trim
    rule (_enforce_cap in history_archive.py) is what keeps that rare.

    XINFO GROUPS' `lag` is not a usable exact alternative: it is nil whenever
    entries have been deleted in a way Redis cannot reconcile, and retire_outbox
    XDELs on every successful cycle. The honest exact measure is XPENDING + XLEN.
    """
    return await redis.xlen(HISTORY_OUTBOX_KEY)


async def outbox_pending_count(redis: aioredis.Redis) -> int:
    """Entries delivered but not yet acked, across all consumers in the group."""
    summary = cast(
        dict[str, Any], await redis.xpending(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP)
    )
    return int(summary["pending"])


async def outbox_pending_below(redis: aioredis.Redis, minid: bytes) -> list[bytes]:
    """Delivered-but-unacked IDs strictly older than `minid` — the set a trim is
    about to destroy while a drainer is still holding it.

    XTRIM is blind TO the PEL — verified on redis:7-alpine (7.4.9): five entries
    delivered and unacked, `XTRIM MAXLEN 2` deleted three, and XPENDING still read
    5 afterwards. Without this the cap leaves a pending record whose body is gone:
    a tombstone, which unacked replays every cycle forever on a non-evictable key.

    Bounded by BATCH_SIZE x live drainers however large the trim is, because the
    drain cycle never reads `>` with a non-empty PEL — which is what lets the caller
    ack this set by ID while trimming the bodies with a single MINID command.

    Returns [] when the group has vanished (an operator DEL racing the cap); with
    no group nothing is pending, so there is nothing to ack.
    """
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
    """Clear PEL records without archiving the entries — the cap only.

    Separate from retire_outbox (which acks AND deletes, meaning "these reached
    Postgres") so `grep retire_outbox` keeps its meaning; this one means "these are
    being destroyed on purpose". Only XACK: the bodies go with the caller's single
    MINID trim, orders of magnitude cheaper than naming every doomed ID in an XDEL.
    """
    if not ids:
        return
    await redis.xack(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, *ids)


# Ceiling on one outbox_pending_below scan. The PEL is bounded by BATCH_SIZE x live
# drainers, so this sits far above any reachable size: a runaway guard, not a page.
_PENDING_SCAN_LIMIT = 10_000


def _prev_stream_id(entry_id: bytes) -> bytes:
    """The ID immediately before `entry_id`, for an inclusive→exclusive bound.

    XPENDING's range is inclusive at both ends while MINID's is exclusive below.
    Stream IDs are `<ms>-<seq>` with both halves 64-bit, so stepping the sequence is
    exact rather than an epsilon; at seq 0 it borrows from the millisecond half, and
    b"0-0" has nothing below it.
    """
    ms, _, seq = entry_id.partition(b"-")
    ms_i, seq_i = int(ms), int(seq)
    if seq_i:
        return b"%d-%d" % (ms_i, seq_i - 1)
    if ms_i:
        return b"%d-%d" % (ms_i - 1, (1 << 64) - 1)
    return b"0-0"


async def trim_outbox_below(redis: aioredis.Redis, minid: bytes) -> int:
    """Drop every entry older than `minid` without archiving it, returning the
    number actually destroyed — the opt-in HISTORY_OUTBOX_MAX cap only
    (history_archive.py). A separate name from retire_outbox so that grep still
    means "entries that reached Postgres" and this one always reads as data loss.

    MINID, never MAXLEN. MINID names an ID, so a re-send is a no-op; MAXLEN names a
    length, so its effect depends on stream state at execution time and a re-send
    after concurrent XADDs destroys a second tranche of unarchived plays.

    approximate=False is not optional. redis-py defaults it to True, which trims to
    node boundaries — on a small stream that trims NOTHING while reporting success,
    and fakeredis models it as an exact trim, so a unit test cannot see the
    difference. The same default sits on xadd(maxlen=...).

    The caller logs this value rather than a derived `depth - cap`: XLEN over-counts
    acked-but-undeleted entries, and a retire racing the cap can settle part of the
    doomed range first. Measured on a real 490,000-entry stream, XTRIM removed
    489,000 in 37ms against the 206 MB / 5.3s a single `RPOP key 490000` cost the
    list implementation; LIMIT cannot be combined with approximate=False anyway.
    """
    return await redis.xtrim(HISTORY_OUTBOX_KEY, minid=minid, approximate=False)


async def reclaim_outbox_stale(
    redis: aioredis.Redis, *, min_idle_ms: int, count: int, max_passes: int
) -> tuple[int, int]:
    """Sweep the group's PEL: reclaim long-idle entries, purge tombstones.

    Returns (reclaimed, purged). Besides XACK this is the only thing that clears a
    tombstone from the PEL on Redis 7, and the only thing that reaches an ORPHANED
    PEL (left under a consumer name no live process reads — what a deploy that
    changed the name produces); a peer killed mid-batch is already covered by the
    every-cycle pending read, the name being shared. Claims into our OWN name too:
    a no-op for live entries we hold, but exactly what purges tombstones.

    min_idle_ms must exceed the drain deadline: under a shared name "idle" is
    measured from last delivery, so a shorter value reclaims a live sibling's
    in-flight batch mid-insert. The caller asserts it.

    justid=True is not used: redis-py's parser returns only the claimed-ID list for
    that form, discarding both the cursor AND the deleted-ID list. Requires Redis
    7.0+ — the 3-element reply carrying deleted IDs is 7.0, and on 6.2 redis-py
    hands back a literal (None, None), so the tombstone is never purged.

    The loop terminates on "this pass found nothing new" and counts distinct ids,
    not on `cursor == b"0-0"` alone: real Redis returns b"0-0" for a completed scan
    while fakeredis returns the last-scanned ID, which fed back as an inclusive
    start re-delivers entries already counted. max_passes bounds the work either way.
    """
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


# ── Guild-scoped Redis store ──────────────────────────────────────────────────

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _guild_op(
    default: Any = None,
    default_factory: Optional[Callable[[], Any]] = None,
) -> Callable[
    [Callable[Concatenate["GuildRedisStore", _P], Awaitable[_R]]],
    Callable[Concatenate["GuildRedisStore", _P], Awaitable[_R]],
]:
    """Enforce GuildRedisStore's 'log, never raise' contract in one place.

    On any exception, logs `[guild:{id}] {method} failed: {e}` and returns
    `default`, so each store method body is just its Redis happy path.

    Pass `default` for immutable fallbacks and `default_factory` for anything
    mutable: a decorator argument is evaluated ONCE at class-body execution, so
    `default=[]` would hand the *same* list to every guild on every failure and one
    in-place mutation would poison "empty" process-wide.

    `default` is typed Any, not `_R`: pinning it to the return TypeVar would let
    `default=None` collapse `_R` to `None` for the Optional-returning readers. The
    cost is that nothing type-checks `default` against the return type.
    """

    def decorator(
        func: Callable[Concatenate["GuildRedisStore", _P], Awaitable[_R]],
    ) -> Callable[Concatenate["GuildRedisStore", _P], Awaitable[_R]]:
        @wraps(func)
        async def wrapper(
            self: "GuildRedisStore", *args: _P.args, **kwargs: _P.kwargs
        ) -> _R:
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                log.warning(f"[guild:{self.guild_id}] {func.__name__} failed: {e}")
                return default_factory() if default_factory is not None else default

        return wrapper

    return decorator


class GuildRedisStore:
    """Encapsulates all Redis IO for a single guild. All methods log errors and
    never raise — the try/except/log is applied by the @_guild_op decorator so
    each method body is just its Redis happy path."""

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

    def _pipe_expire_all(self, pipe: Pipeline) -> None:
        """Queue expire commands for the TTL-managed guild keys onto an existing
        pipeline. The history key is absent: push_history PERSISTs it and bounds it
        by length instead, and -history reads that list and nothing else, so a TTL
        added here answers a guild that has played hundreds of songs with silence."""
        pipe.expire(self.queue_key(), GUILD_TTL)
        pipe.expire(self.state_key(), GUILD_TTL)
        pipe.expire(self.now_playing_key(), GUILD_TTL)

    async def _exec_with_state_ttl(self, pipe: Pipeline) -> None:
        """Append the state-key TTL refresh and execute the pipeline. EXPIRE must
        come after the queued writes — on a not-yet-created key it is a no-op and
        would leave the key persistent-until-eviction."""
        pipe.expire(self.state_key(), GUILD_TTL)
        await pipe.execute()

    # Queue operations

    @_guild_op(default=None)
    async def push_queue(self, entry: QueueEntry) -> None:
        """RPUSH one queue entry and refresh TTL on all guild keys."""
        pipe = self.redis.pipeline()
        pipe.rpush(self.queue_key(), entry.to_redis())
        self._pipe_expire_all(pipe)
        await pipe.execute()

    @_guild_op(default=None)
    async def push_queue_batch(self, entries: Sequence[QueueEntry]) -> None:
        """RPUSH all entries in one pipeline round-trip and refresh TTL on all guild keys."""
        if not entries:
            return
        pipe = self.redis.pipeline()
        pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
        self._pipe_expire_all(pipe)
        await pipe.execute()

    @_guild_op(default=None)
    async def push_queue_front(self, entries: Sequence[QueueEntry]) -> None:
        """LPUSH entries so entries[0] ends up at the queue head, and refresh TTL on
        all guild keys — the -playnow front insert. LPUSH sends each successive
        argument to the head, so the batch is reversed first to preserve order.

        A swallowed failure degrades WORSE than a tail-push failure: the in-memory
        legs end up len(entries) ahead of Redis at the HEAD, so the next commit-time
        LPOPs retire other songs' entries, and a crash before the mismatch drains
        restores a queue shifted by up to that many songs."""
        if not entries:
            return
        pipe = self.redis.pipeline()
        pipe.lpush(self.queue_key(), *[e.to_redis() for e in reversed(entries)])
        self._pipe_expire_all(pipe)
        await pipe.execute()

    @_guild_op(default=None)
    async def pop_queue(self) -> None:
        # At-most-once: LPOP removes with no ack, so a crash after this loses the
        # song from Redis. Accepted — the in-memory asyncio.Queue is the source of
        # truth; at-least-once would need a stream and an XACK.
        await self.redis.lpop(self.queue_key())

    def _now_playing_state_mapping(
        self, current: SongQueueEntry, play_start_epoch: float
    ) -> dict[str, str]:
        """The current_song_* state fields ARE a parked queue entry — one signature
        enforcing the identity SongQueueEntry.from_song()/from_crashed_state()
        rely on for crash recovery."""
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
            StateField.CURRENT_SONG_QUEUED_AT: str(current.queued_at),
            StateField.CURRENT_SONG_QUEUE_POSITION: str(current.queue_position),
            StateField.CURRENT_SONG_QUERY_SOURCE: current.query_source,
            StateField.PLAY_START_EPOCH: str(play_start_epoch),
            StateField.TOTAL_PAUSE_SECONDS: "0",
        }

    @_guild_op(default=None)
    async def pop_queue_and_start_song(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
    ) -> None:
        """Atomically LPOP the queue and park `current`'s fields in the state hash.

        MULTI/EXEC leaves the song in one of two consistent states — still queued
        with current_song_url empty, or dequeued with all now-playing fields set —
        closing the crash window where it was absent from both. `now_playing` rides
        the same transaction, so a crash can never leave state pointing at song B
        while the snapshot still shows song A.
        """
        mapping = self._now_playing_state_mapping(current, play_start_epoch)
        pipe = self.redis.pipeline(transaction=True)
        pipe.lpop(self.queue_key())
        pipe.hset(self.state_key(), mapping=_hset_mapping(mapping))
        pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
        pipe.expire(self.state_key(), GUILD_TTL)
        if now_playing is not None:
            pipe.hset(
                self.now_playing_key(),
                mapping=_hset_mapping(now_playing.to_redis_mapping()),
            )
            pipe.expire(self.now_playing_key(), GUILD_TTL)
        await pipe.execute()

    @_guild_op(default=None)
    async def set_current_song_state(
        self,
        current: SongQueueEntry,
        play_start_epoch: float,
        now_playing: Optional[NowPlayingData] = None,
    ) -> None:
        """pop_queue_and_start_song without the LPOP, in one transaction — for
        restarting a crash-recovered "current song" that was never RPUSHed to the
        queue list.
        """
        mapping = self._now_playing_state_mapping(current, play_start_epoch)
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
        await pipe.execute()

    @_guild_op(default=None)
    async def delete_queue(self) -> None:
        """DELETE the queue key."""
        await self.redis.delete(self.queue_key())

    @_guild_op(default=None)
    async def rebuild_queue(self, entries: Sequence[QueueEntry]) -> None:
        """Atomically DELETE + RPUSH all entries. Uses MULTI/EXEC to avoid empty-window race."""
        pipe = self.redis.pipeline(transaction=True)
        pipe.delete(self.queue_key())
        pipe.rpush(self.queue_key(), *[e.to_redis() for e in entries])
        pipe.expire(self.queue_key(), GUILD_TTL)
        await pipe.execute()

    # History operations

    # ISSUE: non-evictable keys can exhaust Redis and stall ALL writes.
    # Two kinds of key carry no TTL, so under `maxmemory-policy volatile-lru` they are
    # never eviction candidates: guild:{id}:history and HISTORY_OUTBOX_KEY. Once they fill
    # maxmemory with no TTL-bearing key left to evict, Redis rejects every write with
    # OOM — state, queue and cache alike — and each store method swallows it and logs, so
    # persistence degrades silently rather than crashing.
    #
    # Only the OUTBOX can get there BY GROWING: the history lists are bounded at
    # HISTORY_CACHE_LIMIT entries per guild, trimmed on every write, so their total scales
    # with guild count (~24 KB each), not with runtime. The outbox is near-empty whenever
    # the drainer keeps up and grows for the whole duration of a Postgres outage, at ~487
    # bytes per play. HISTORY_OUTBOX_MAX is the opt-in bound; dropping entries there is
    # real data loss, since a capped list leaves no second copy. A Redis memory/eviction
    # alarm is still owed.
    #
    # CAVEAT: the trim is lazy — it runs inside push_history and nowhere else, so a guild
    # that stops playing keeps whatever oversized list it already had, forever, and no TTL
    # path touches it. Upgrading from a build that never trimmed, a dormant guild holding
    # 100k entries is ~49 MB of the bundled 256mb, permanently, in a key volatile-lru can
    # never evict; only a further play or a manual DEL reclaims it.
    # See docs/ARCHITECTURE.md#redis-memory-bounds.
    #
    # Don't switch to allkeys-lru as a workaround: it makes the outbox evictable, and an
    # evicted entry is a play that vanishes with no error, no rejected-table row and no
    # log line (see docker-compose.yml redis command).
    @_guild_op(default=None)
    async def push_history(self, entry: HistoryEntry) -> None:
        """LPUSH one entry, cap and PERSIST the list, and — while the archive is
        enabled — mirror the entry onto the Postgres outbox: one transaction, one
        round trip, one branch.

        LTRIM + PERSIST are the whole retention policy, unconditional in both
        archive modes (the display list is runtime state; the flag gates only the
        durable tier). Bounded because an unbounded non-evictable key is the OOM
        shape the note above describes; permanent because -history reads this list
        and nothing else, so an idle TTL would answer a quiet guild with silence —
        the PERSIST also self-heals an older build's 24h expiry. See
        HISTORY_CACHE_LIMIT and docs/ARCHITECTURE.md#history-read-path.

        With the archive disabled the XADD leg is absent and the outbox key is never
        created. Both legs share one serialize_history_entry call so they cannot
        drift; the outbox leg is idempotent under a re-sent transaction (a duplicate
        collapses on the archive's unique index) while the LPUSH above it is not.

        Stays on the SWALLOWING side of the split (@_guild_op), unlike the drain
        helpers, which raise: the playback loop must never die because Redis
        blinked. So the producer can never report a mis-shaped outbox — a pre-stream
        list fails the XADD with WRONGTYPE, swallowed into one warning per song —
        which is why ensure_outbox_group() aborts at STARTUP instead.
        """
        wire = serialize_history_entry(entry)
        try:
            await self._push_history_pipeline(wire)
        except OutOfMemoryError:
            # AN ALREADY-full REDIS CANNOT SELF-HEAL without this. LPUSH carries
            # Redis' `denyoom` flag and is queued first, so once used_memory exceeds
            # maxmemory with nothing evictable the server refuses it at queue time,
            # the CAS is dirtied and EXEC aborts — measured on redis:7-alpine: llen
            # unchanged at 386, TTL still -1, the LTRIM never ran. Reordering inside
            # the MULTI does not help.
            #
            # A bare LTRIM is not denyoom and frees memory immediately, so the one
            # command that could make room is the one that can still run — but at
            # the steady state it frees NOTHING (measured at maxmemory 3mb: llen
            # stayed 50, the retry aborted again, the play was dropped). The
            # recovery is real once per guild per upgrade (an oversized legacy
            # list); after that the retry only covers memory freed elsewhere. Log
            # which case happened. Reachable despite the cap because the outbox
            # grows unbounded during a Postgres outage and fills the instance.
            #
            # LLEN first, not LTRIM first: reads are not denyoom, and skipping a
            # trim that provably frees nothing removes a round trip from a path that
            # runs per song end for every guild while Redis is full.
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
        """The one transactional write, factored out so the OOM path above can
        re-issue it verbatim rather than restate it."""
        pipe = self.redis.pipeline()
        pipe.lpush(self.history_key(), wire)
        # UNPAGED on purpose, unlike HistoryOutboxDrainer._enforce_cap, which pages
        # its trim against the same hazard class: LTRIM costs O(N) in elements
        # removed and every push after the first trims 51→50, so this is O(1) in the
        # steady state and only a list from a build that did not cap pays more,
        # once. Measured on redis:7-alpine 7.4.9 — 0.24 ms at 10k entries, 6.3 ms at
        # 100k, ~22 ms at 500k, with another guild's playback write blocked for that
        # duration. Nothing here trips on a 30 ms stall (no socket_timeout on the
        # pool, a 5s healthcheck, a 2s bound on the -history read, and audio is
        # never Redis-gated).
        pipe.ltrim(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
        # No EXPIRE anywhere near this key — length bounds it, time never does.
        # PERSIST because -history has no other source, and because it clears an
        # older build's inherited TTL in the same write.
        pipe.persist(self.history_key())
        # The pipeline's one conditional command: the flag is the consent gate for
        # long-term storage, and with the archive disabled nothing may accumulate
        # for a drainer that does not exist — the XADD would create the
        # non-evictable key on first write. Read per call, through the module so
        # tests can patch either the function or the environment. A garbage value
        # must never first surface here (@_guild_op would swallow the parser's
        # ValueError into one warning per song), which is why setup_hook must read
        # the flag before anything else consumes it.
        if config.history_archive_enabled():
            pipe.xadd(HISTORY_OUTBOX_KEY, {OUTBOX_FIELD: wire})
        await pipe.execute()

    @_guild_op(default_factory=list)
    async def get_history(self) -> list[HistoryEntry]:
        """Return up to HISTORY_CACHE_LIMIT history entries newest-first.
        Corrupt entries are dropped (parse_history_entry warns per entry)."""
        raw = await self.redis.lrange(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
        return [e for e in map(parse_history_entry, raw) if e is not None]

    # Now-playing operations
    # (Writes happen inside pop_queue_and_start_song()/set_current_song_state()
    #  via the now_playing value object, atomically with the rest of the start state.)

    @_guild_op(default=None)
    async def get_now_playing(self) -> Optional[NowPlayingData]:
        """HGETALL the now_playing hash. Returns None on miss or error.

        The two collapse to None on purpose: the only caller uses this to optionally
        restore a display embed, and "no embed" is right for both.
        """
        # Bytes, not str: create_redis_pool() sets decode_responses=False, an
        # invariant redis-py's return type cannot express. Don't simplify this
        # away — from_redis() decodes, so a decoded pool breaks it at runtime.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.now_playing_key()))
        return NowPlayingData.from_redis(raw)

    # Playback position tracking

    @_guild_op(default=None)
    async def set_playback_start(self, epoch: float) -> None:
        """Record that playback started at `epoch`. Resets all pause accounting.

        For unit tests and standalone use; loop() writes these fields atomically via
        pop_queue_and_start_song() instead."""
        pipe = self.redis.pipeline()
        pipe.hset(self.state_key(), StateField.PLAY_START_EPOCH, str(epoch))
        pipe.hset(self.state_key(), StateField.TOTAL_PAUSE_SECONDS, "0")
        pipe.hdel(self.state_key(), StateField.PAUSE_START_EPOCH)
        await self._exec_with_state_ttl(pipe)

    @_guild_op(default=None)
    async def on_pause(self, epoch: float) -> None:
        """Record the epoch when the voice client was paused."""
        pipe = self.redis.pipeline()
        pipe.hset(self.state_key(), StateField.PAUSE_START_EPOCH, str(epoch))
        await self._exec_with_state_ttl(pipe)

    @_guild_op(default=None)
    async def on_resume(self, resume_epoch: float) -> None:
        """Accumulate elapsed pause time into total_pause_seconds, clear
        pause_start_epoch.

        Non-atomic read-modify-write, so it assumes one writer per guild (true
        today). Under multi-process sharding this must become a Lua script or a
        WATCH/MULTI retry loop."""
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
        """Clear all transient song state in one round-trip: HDEL every
        current_song_*/position field and DELETE the now_playing hash. Matches
        clear_connection()'s idiom, so *absent* — not empty-string — is the one
        representation of "no song"."""
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
        """HGETALL the state hash and return a typed snapshot.

        Zero-value GuildStateData when the hash is missing/empty, None when the read
        itself failed — so callers can tell "nothing stored" from "Redis
        unavailable" (see _restore_guild). Pure read: does not refresh TTL;
        refresh_ttl() at the end of _restore_state() covers the recovery window.
        """
        # Same decode_responses=False invariant as get_now_playing() above.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.state_key()))
        return GuildStateData.from_redis(raw)

    @_guild_op(default=None)
    async def get_recovery_gate(self) -> Optional[GuildRecoveryGate]:
        """State hash + pending-queue *length* in one pipeline — the lightweight
        connection/restorable gate for `_restore_guild`.

        Transfers no queue contents, now-playing or history: a -stopped guild keeps
        a possibly-long queue by design, and gating on LLEN keeps that payload off
        the wire on every on_ready. _restore_state re-reads the full snapshot after
        a successful connect. None on read failure, as in get_guild_state.
        """
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
        """The complete playback aggregate — state hash, pending queue, now-playing
        snapshot, history — in one pipeline round-trip.

        Same error-vs-empty contract as get_guild_state. All four reads ride one
        pipeline, so a failure aborts the whole snapshot and the caller restores
        everything or nothing rather than a partly-fabricated state. Not MULTI —
        recovery holds the guild lock during the window that matters.
        """
        pipe = self.redis.pipeline()
        pipe.hgetall(self.state_key())
        pipe.lrange(self.queue_key(), 0, -1)
        pipe.hgetall(self.now_playing_key())
        pipe.lrange(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
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

    @_guild_op(default=None)
    async def set_volume(self, volume: float) -> None:
        """Persist the guild volume setting."""
        pipe = self.redis.pipeline()
        pipe.hset(self.state_key(), StateField.VOLUME, str(volume))
        await self._exec_with_state_ttl(pipe)

    # TTL management

    @_guild_op(default=None)
    async def refresh_ttl(self) -> None:
        """Refresh GUILD_TTL on the TTL-managed guild keys. History is excluded for
        the reason _pipe_expire_all gives: it is bounded by length, never by time,
        because it is the only thing -history reads."""
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
        """Remove all transient state on intentional disconnect: voice/text channel
        IDs (so on_ready skips recovery for this guild), now-playing display,
        requester attribution, and every position-tracking field."""
        pipe = self.redis.pipeline()
        pipe.hdel(
            self.state_key(),
            StateField.VOICE_CHANNEL_ID,
            StateField.TEXT_CHANNEL_ID,
            *_TRANSIENT_SONG_FIELDS,
            # HACK: last_author_id is dead schema still scrubbed on every disconnect.
            # Nothing writes it any more; this HDEL only cleans up hashes left by
            # older builds, hence the bare literal rather than a StateField
            # constant — so every guild disconnect pays to delete a field that
            # cannot exist on any hash written since that migration. Safe to delete
            # once no pre-migration hash can be live — guild keys carry a 24h TTL,
            # so one release is already more than enough.
            "last_author_id",
            *_PLAYBACK_POSITION_FIELDS,
        )
        pipe.delete(self.now_playing_key())
        await pipe.execute()

    # Recovery lock (distributed, for rolling-restart safety)

    _RECOVERY_LOCK_TTL = 60  # seconds

    def _recovery_lock_key(self) -> str:
        return f"lock:guild:{self.guild_id}:recovery"

    @_guild_op(default=False)
    async def acquire_recovery_lock(self) -> bool:
        """SET NX EX — True if this instance won the lock, False if another holds it."""
        result = await self.redis.set(
            self._recovery_lock_key(), "1", nx=True, ex=self._RECOVERY_LOCK_TTL
        )
        return result is True

    @_guild_op(default=None)
    async def release_recovery_lock(self) -> None:
        await self.redis.delete(self._recovery_lock_key())
