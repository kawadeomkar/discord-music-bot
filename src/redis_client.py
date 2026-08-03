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
# HistoryOutboxDrainer. Deliberately NO TTL — it holds not-yet-durable entries,
# so under volatile-lru it must never be an eviction candidate; normally
# near-empty, it grows only while Postgres is unreachable.
#
# A STREAM with a consumer group, not a list: a list retire is `RPOP <count>` —
# "remove the oldest N", not "remove the N I just archived" — and those are equal
# only when nothing else touched the key and the command ran exactly once. XACK
# names IDs, so it cannot mean anything else.
HISTORY_OUTBOX_KEY = "history:outbox"
HISTORY_OUTBOX_GROUP = "drainers"
# STABLE, not per-process, and load-bearing: the PEL belongs to the NAME, so a
# starting process's `XREADGROUP ... 0` inherits whatever its predecessor — or a
# live sibling — left in flight, and recovery needs no lease, no TTL and no
# housekeeping. Two live drainers are safe by construction: `>` hands them
# disjoint entries and `0` replays a shared set that ON CONFLICT DO NOTHING
# collapses on the Postgres side, so concurrency costs duplicated work, never
# lost data.
HISTORY_OUTBOX_CONSUMER = "drainer"
# The single stream field holding one serialize_history_entry blob. Keeping the
# orjson payload opaque means parse_history_entry, HistoryEntry's domain clamping
# and every wire-compatibility rule in guild_state.py are untouched by transport.
OUTBOX_FIELD = b"e"
# 24h idle expiry. NEVER applied to the history key: that list is capped rather
# than expired (see HISTORY_CACHE_LIMIT and push_history), and it is the only
# thing -history reads, so a guild that goes quiet for a day must still answer
# the command. Every TTL path here excludes it unconditionally.
GUILD_TTL = 86400
# The retention cap, the display cap and the -history ceiling at once — the same
# number on purpose. push_history LTRIMs the list to this many entries on every
# write and PERSISTs it, so the list is a bounded, permanent window of the newest
# plays, and musicbot.HISTORY_MAX_LIMIT is pinned to this constant: the command
# can never ask for more SLOTS than the window holds.
#
# "Slots", not "plays" — the equality leaves no headroom, so anything that costs
# a slot without yielding a renderable play shortens the answer by one: a corrupt
# entry (get_history drops it) or a duplicate (recent() dedups it). The second is
# reachable — create_redis_pool sets retry_on_error, and the LPUSH is not
# idempotent, so a timeout after the server applied EXEC re-sends the pipeline
# and the entry lands twice. See docs/ARCHITECTURE.md#history-read-path.
#
# Raising it costs Redis memory in all three roles at once — roughly 487 B per
# entry per guild, permanently, since nothing expires it.
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
    workaround only — it widens nothing at runtime.
    """
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
        # redis-py's OWN exception classes, not the builtins of the same name.
        # `redis.exceptions.ConnectionError` does not subclass builtins.ConnectionError
        # (it derives from RedisError), so listing the builtin here matched nothing
        # redis-py ever raises and connection errors got no retry at all — while every
        # store method logs-and-swallows, so the missing retry surfaced as persistence
        # quietly degrading rather than as an error. See test_redis_client.py, which
        # asserts the non-subclass relationship that makes this easy to get wrong.
        #
        # (The builtin TimeoutError was harmless but redundant: retry_on_timeout=True
        # already appends socket.timeout, which IS builtins.TimeoutError on 3.10+.)
        retry_on_error=[RedisConnectionError, RedisTimeoutError],
        # Without an explicit Retry, redis-py synthesises `Retry(NoBackoff(), 1)` for a
        # non-empty retry_on_error: one immediate reattempt, no backoff. A restarting
        # Redis is usually gone for longer than that, and a hammering reconnect is what
        # backoff exists to avoid. 3 attempts over ExponentialBackoff's default 8ms→512ms
        # ceiling covers an ordinary restart without stalling a command for long.
        # `redis.asyncio.retry.Retry`, NOT `redis.retry.Retry` — they are different
        # classes with the same name, the same constructor and the same attributes,
        # and only the async one awaits. The sync version's call_with_retry does
        # `try: return do()`, which under an async connection returns a COROUTINE
        # without awaiting it: no exception is ever raised inside that try, the
        # `except` arm is dead, the failure handler never runs, and the error
        # surfaces from the caller's await — outside the retry loop entirely.
        # Measured against a closed port: sync = 1 connection attempt, async = 4.
        # For a year this pool was configured for 3 retries and performed none.
        #
        # Nothing about the object looks wrong, which is why it survived:
        # get_retries() is 3, _supported_errors and _backoff are exactly as set,
        # so every assertion on the CONFIGURATION passes under both classes. The
        # test below counts attempts instead, because that is the only difference.
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


# ── History outbox (drain side) ───────────────────────────────────────────────
# Consumed only by HistoryOutboxDrainer (history_archive.py). Unlike the cache
# helpers above, these deliberately DO raise on Redis failure — the drainer's
# backoff loop is the error handler, and a swallowed error here would look
# like an empty outbox and silently stall the drain. Raw bytes in/out: wire
# parsing stays in guild_state.py (parse_history_entry), per the schema rule.
#
# There is no lease and no single-consumer requirement. Both were consequences
# of a positional retire; the consumer group replaces them (see
# HISTORY_OUTBOX_CONSUMER). What the lease also silently provided and nothing
# here replaces is exactly-once `play_history_rejected` recording — that moved
# into the SQL as an ON CONFLICT clause rather than into a lock.
#
# Every command below is idempotent under re-send, which is why the drain path
# is safe on the application pool with retries enabled: XACK and XDEL return 0
# for an ID they have already settled, and XTRIM MINID names an absolute ID so
# its effect is a function of its argument. `XTRIM MAXLEN` is NOT — it means
# "keep the newest n", so a re-send after two concurrent XADDs destroys a
# second tranche. It is not used here, and must not be introduced.


@dataclass(frozen=True, slots=True, kw_only=True)
class OutboxEntry:
    """One delivered stream entry: its ID, and its payload if it still has one.

    kw_only because both fields are Optional[bytes]-shaped and transposing them
    would type-check silently (the convention ExtractRequest documents).

    `wire is None` is a TOMBSTONE — the entry was delivered to a consumer and
    then had its body deleted while still pending, which XTRIM and any operator
    XDEL can both produce because neither consults the PEL. The ID survives in
    the PEL, so the read returns it with an EMPTY field map. This is not a
    corrupt entry and must not be routed to parse_history_entry: the bytes are
    gone from Redis and nothing can reproduce them. The drainer acks it
    unconditionally and logs a lost play — leaving it pending replays it every
    cycle forever, on a key the volatile-lru eviction policy can never reclaim
    (it carries no TTL, by design — see push_history).
    """

    id: bytes
    wire: Optional[bytes]


def _parse_outbox_reply(reply: Any) -> list[OutboxEntry]:
    """Flatten redis-py's XREADGROUP reply into IDs and payloads.

    The reply is [[stream_name, [(id, {field: value}), ...]], ...] — one outer
    element per stream, and we always ask for exactly one. The two empty cases
    have DIFFERENT shapes, which is why this tolerates both rather than
    indexing: `>` with nothing new returns `[]`, while `0` with an empty PEL
    returns `[[key, []]]`.

    `.get(OUTBOX_FIELD)` rather than `[OUTBOX_FIELD]` is the tombstone rule.
    A literal subscript raises KeyError, which is not a ResponseError and not a
    parse failure, so it escapes every handler the drain path has and reaches
    the generic backoff — where the same entry replays on the next tick,
    forever, while the outbox grows unbounded behind it.
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

    Deliberately NOT `except ResponseError: pass`. BUSYGROUP, WRONGTYPE and
    NOGROUP are all plain ResponseError — redis-py has no subclass for any of
    them — so a class-level catch swallows the two that mean "this bot cannot
    record history" alongside the one that means "already fine":

      BUSYGROUP   repeat create              → fine, ignore
      WRONGTYPE   a pre-R1 LIST at the key   → fatal, must abort startup
      NOGROUP     the key was deleted        → recoverable, healed by the
                                               drain cycle

    Swallowing WRONGTYPE is total, silent history loss: push_history is a
    @_guild_op method, so its XADD failure is one warning per song and takes
    guild:{id}:history down with it (both legs share one MULTI/EXEC). XLEN
    would raise too, so depth reads -1 and the backlog alarm can never fire
    either. Hence the startup abort — it matches setup_hook's refusal to run
    an ENABLED archive without POSTGRES_URL: a bot the operator opted into
    archiving with must not serve while it cannot durably record what it
    plays. (Enabled mode only. With the archive disabled setup_hook never
    calls this — creating the group would MKSTREAM the non-evictable key into
    existence — and a mis-shaped key is inert, downgraded to a startup
    warning by the leftover-outbox probe.)

    id="0" is NOT redis-py's default (it defaults to "$", which silently skips
    every entry already in the stream). MKSTREAM removes any separate
    create-the-key step. Repeating this call does not rewind an existing group.
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

    Runs EVERY cycle, not only at startup. Under a shared consumer name this is
    what recovers a peer that was SIGKILLed mid-batch — including one killed
    past K8s' terminationGracePeriodSeconds — without any lease TTL to wait out.

    noack=False is redis-py's default and is spelled out anyway, alongside the
    other kwarg this module pins (xgroup_create's id="0"). noack=True would
    deliver without ever entering the PEL, turning the whole design's
    at-least-once into at-most-once: this "0" read would find nothing to
    re-deliver and a drainer that died mid-batch would lose its plays outright.
    No behavioural test can catch it — every fakeredis assertion still passes —
    so the defence is that the value is written down at both call sites.
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

    Two live drainers get DISJOINT sets here — the server guarantees it — so
    this is the half of the design that makes a lease unnecessary.
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

    Both commands are required, and the ORDER IS LOAD-BEARING.

    XACK clears the pending record but does NOT remove the entry: a stream is a
    log, not a queue. Ack-without-delete would grow this key forever, which is
    strictly worse than the list design because this key carries no TTL and so is
    never an eviction candidate. XDEL frees the memory and is idempotent (missing ID → 0).

    Written this way — transactionally, XACK strictly first — a crash between
    them leaves an acked-but-undeleted entry: invisible to XREADGROUP, harmless,
    and reclaimed by the cap's MINID trim. Written the other way, or
    non-transactionally, the window inverts to XDEL landing without XACK, which
    is an unrecoverable tombstone (OutboxEntry). That is the one new loss
    window the stream transport introduces, and it is closed by construction
    here rather than by remembering to be careful at the call site.

    Crash BEFORE the ack and the batch simply stays pending → redelivered →
    deduped by the archive's unique index. At-least-once is preserved, which is
    the same property that makes the Phase C backfill idempotent.
    """
    if not ids:
        return
    async with redis.pipeline(transaction=True) as pipe:
        pipe.xack(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, *ids)
        pipe.xdel(HISTORY_OUTBOX_KEY, *ids)
        await pipe.execute()


async def outbox_depth(redis: aioredis.Redis) -> int:
    """Entries present in the stream — the drainer's backlog watchdog metric.

    An approximation of the un-archived backlog, and it diverges in BOTH
    directions. It over-reports after a crash between XACK and XDEL (harmless,
    reconciled by the MINID trim). It UNDER-reports when entries were trimmed
    or XDELed while still pending, because the bodies are gone while the PEL
    records survive — i.e. it reads low at exactly the moment plays are being
    lost. The cap's ack-before-trim rule (_enforce_cap in history_archive.py)
    is what keeps that case rare — everything a trim destroys is XACKed first,
    so no pending record outlives its body; without that rule, DEPTH_ALARM
    would go quiet during the incident it exists to catch.

    XINFO GROUPS' `lag` is NOT a usable exact alternative: it is nil whenever
    entries have been deleted in a way Redis cannot reconcile, and retire_outbox
    makes XDEL part of every successful cycle, so it is unavailable in precisely
    the state an operator would consult it. The honest exact measure is
    XPENDING's summary count plus XLEN.
    """
    return await redis.xlen(HISTORY_OUTBOX_KEY)


async def outbox_pending_count(redis: aioredis.Redis) -> int:
    """Entries delivered but not yet acked, across all consumers in the group."""
    summary = cast(
        dict[str, Any], await redis.xpending(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP)
    )
    return int(summary["pending"])


async def outbox_pending_below(redis: aioredis.Redis, minid: bytes) -> list[bytes]:
    """Delivered-but-unacked IDs strictly older than `minid`.

    The set a trim is about to destroy while a drainer is still holding it.
    XTRIM IS BLIND TO THE PEL — verified on redis:7-alpine (7.4.9): five entries
    delivered and unacked, `XTRIM MAXLEN 2` deleted three, and XPENDING still
    read 5 afterwards — so without this the cap leaves a pending record whose
    body is gone. That is a tombstone, and a tombstone left unacked replays
    every cycle forever on a non-evictable key.

    Bounded by BATCH_SIZE x live drainers however large the trim is, because
    the drain cycle never reads `>` with a non-empty PEL. That is what lets the
    caller ack this set by ID while trimming the bodies with a single MINID
    command: the expensive half stays O(1) in commands and the precise half
    stays small.

    Returns [] when the group has vanished (an operator DEL racing the cap).
    With no group nothing is pending by definition, so there is nothing to ack.
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
    """Clear PEL records WITHOUT archiving the entries — the cap only.

    Deliberately separate from retire_outbox, which acks AND deletes and means
    "these reached Postgres". This one means "these are being destroyed on
    purpose", and it exists so `grep retire_outbox` keeps its meaning.

    Only XACK: the bodies are removed by the caller's single MINID trim, which
    is orders of magnitude cheaper than naming every doomed ID in an XDEL.
    """
    if not ids:
        return
    await redis.xack(HISTORY_OUTBOX_KEY, HISTORY_OUTBOX_GROUP, *ids)


# Ceiling on one outbox_pending_below scan. The PEL is bounded by
# BATCH_SIZE x live drainers, so this is far above any reachable size; it is a
# runaway guard, not a page size.
_PENDING_SCAN_LIMIT = 10_000


def _prev_stream_id(entry_id: bytes) -> bytes:
    """The ID immediately before `entry_id`, for an inclusive→exclusive bound.

    XPENDING's range is inclusive at both ends while MINID's is exclusive
    below, so scanning "everything the trim will destroy" needs the ID one step
    down. Stream IDs are `<ms>-<seq>` with both halves 64-bit, so stepping the
    sequence is exact rather than an epsilon; at seq 0 it borrows from the
    millisecond half, and b"0-0" has nothing below it.
    """
    ms, _, seq = entry_id.partition(b"-")
    ms_i, seq_i = int(ms), int(seq)
    if seq_i:
        return b"%d-%d" % (ms_i, seq_i - 1)
    if ms_i:
        return b"%d-%d" % (ms_i - 1, (1 << 64) - 1)
    return b"0-0"


async def trim_outbox_below(redis: aioredis.Redis, minid: bytes) -> int:
    """Drop every entry older than `minid` WITHOUT archiving it, returning the
    number actually destroyed — the opt-in HISTORY_OUTBOX_MAX cap only
    (history_archive.py). A separate name from retire_outbox so that
    `grep retire_outbox` still means "entries that reached Postgres" and this
    one always reads as data loss.

    MINID, never MAXLEN. MINID names an ID, so its effect is a function of its
    argument and a re-send is a no-op; MAXLEN names a length, so its effect is a
    function of stream state at execution time and a re-send after concurrent
    XADDs destroys a second tranche of unarchived plays. That is the same
    destructive-retry defect the positional RPOP had, reconstructed.

    approximate=False is not optional. redis-py defaults it to True, which trims
    to node boundaries — on a small stream that trims NOTHING while reporting
    success, and fakeredis models it as an exact trim, so a unit test cannot see
    the difference. The same default sits on xadd(maxlen=...).

    The caller logs THIS value rather than a derived `depth - cap`: XLEN
    over-counts acked-but-undeleted entries, and a retire racing the cap can
    settle part of the doomed range first, so the trim may legitimately remove
    fewer than the caller computed.

    No slicing: XTRIM returns a count, not the entries — measured on a real
    490,000-entry stream it removed 489,000 in 37ms, against the 206 MB / 5.3s a
    single `RPOP key 490000` cost the list implementation. LIMIT cannot be
    combined with approximate=False anyway.
    """
    return await redis.xtrim(HISTORY_OUTBOX_KEY, minid=minid, approximate=False)


async def reclaim_outbox_stale(
    redis: aioredis.Redis, *, min_idle_ms: int, count: int, max_passes: int
) -> tuple[int, int]:
    """Sweep the group's PEL: reclaim long-idle entries, purge tombstones.

    Returns (reclaimed, purged). Not belt-and-braces — besides XACK this is the
    only thing that clears a tombstone from the PEL on Redis 7, and it is the
    only thing at all that reaches an ORPHANED PEL (one left under a consumer
    name no live process reads, which a deploy that changed the name produces).
    The ordinary case — a peer killed mid-batch — is already covered by the
    every-cycle pending read, because the name is shared.

    Claims into our OWN name as well as any other. Claiming an entry we already
    hold is a no-op for live entries, but it is exactly what purges tombstones,
    so scoping the sweep to foreign consumers would make it useless for its main
    job.

    min_idle_ms must exceed the drain deadline: under a shared name "idle" is
    measured from last DELIVERY, so a shorter value reclaims a live sibling's
    in-flight batch mid-insert. The caller asserts it.

    justid=True is deliberately NOT used: redis-py's parser returns only the
    claimed-ID list for that form, discarding both the cursor AND the deleted-ID
    list — the two things this loop needs.

    Requires Redis 7.0+. XAUTOCLAIM exists at 6.2, but the 3-element reply
    carrying the deleted-ID list is 7.0; on 6.2 redis-py hands back a literal
    (None, None) inside the claimed list and the tombstone is never purged.

    The loop terminates on "this pass found nothing NEW", not on
    `cursor == b"0-0"` alone, and the counts are of DISTINCT ids. Both are
    forced by the same divergence: real Redis returns b"0-0" to signal a
    completed scan, while fakeredis returns the last-scanned ID — which, fed
    back as an inclusive start, re-delivers entries this sweep has already
    counted. A cursor-only condition would run correctly in production and spin
    here; a running total would report five reclaims for one entry. Tracking ids
    is correct under both conventions and is what makes the unit tier meaningful
    at all. max_passes bounds the work either way.
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
    `default=[]` would hand the *same* list to every guild on every failure and
    one in-place mutation would poison "empty" process-wide. Do not add a mutable
    `default` — `test_mutable_defaults_use_a_factory` fails if you do.

    `default` is typed Any, not `_R`: pinning it to the return TypeVar would let
    `default=None` collapse `_R` to `None` for the Optional-returning readers. The
    cost is that nothing type-checks `default` against the return type — that onus
    falls on the (obvious) call sites below.
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
        pipeline. The history key is deliberately absent: push_history PERSISTs
        it and bounds it by LENGTH instead, so there is no TTL here to refresh.
        Adding one would break -history quietly — the command reads that list and
        nothing else, so an expired key is an empty answer for a guild that has
        played hundreds of songs."""
        pipe.expire(self.queue_key(), GUILD_TTL)
        pipe.expire(self.state_key(), GUILD_TTL)
        pipe.expire(self.now_playing_key(), GUILD_TTL)

    async def _exec_with_state_ttl(self, pipe: Pipeline) -> None:
        """Append the state-key TTL refresh and execute the pipeline. EXPIRE must
        come after the queued writes — on a not-yet-created key it is a no-op and
        would leave the key persistent-until-eviction.
        """
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
        argument to the head, so the batch is reversed first to preserve the given
        order.

        A swallowed failure here degrades WORSE than a tail-push failure: the
        in-memory legs end up len(entries) ahead of Redis at the HEAD, so the next
        commit-time LPOPs retire other songs' entries, and a crash before the
        mismatch drains restores a queue shifted by up to that many songs."""
        if not entries:
            return
        pipe = self.redis.pipeline()
        pipe.lpush(self.queue_key(), *[e.to_redis() for e in reversed(entries)])
        self._pipe_expire_all(pipe)
        await pipe.execute()

    @_guild_op(default=None)
    async def pop_queue(self) -> None:
        # At-most-once: LPOP removes with no ack, so a crash after this loses the
        # song from Redis. Acceptable in Phase 2 (asyncio.Queue is source of
        # truth); Phase 3b moves to Redis Streams + XACK for at-least-once.
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

        MULTI/EXEC leaves the song in one of two consistent states — (a) still
        queued with current_song_url empty, or (b) dequeued with all now-playing
        fields set — closing the crash window where it was absent from both.

        `now_playing` is written inside the same transaction, so a crash can never
        leave state pointing at song B while the snapshot still shows song A.
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
    # maxmemory with no TTL-bearing key left to evict, Redis rejects EVERY write with
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
    # CAVEAT: the trim is LAZY — it runs inside push_history and nowhere else, so a guild
    # that stops playing keeps whatever oversized list it already had, forever, and no TTL
    # path touches it. Upgrading from a build that never trimmed, a dormant guild holding
    # 100k entries is ~49 MB of the bundled 256mb, permanently, in a key volatile-lru can
    # never evict; only a further play or a manual DEL reclaims it.
    # See docs/ARCHITECTURE.md#redis-memory-bounds.
    #
    # Do NOT switch to allkeys-lru as a workaround: it makes the outbox evictable, and an
    # evicted entry is a play that vanishes with no error, no rejected-table row and no
    # log line (see docker-compose.yml redis command).
    @_guild_op(default=None)
    async def push_history(self, entry: HistoryEntry) -> None:
        """LPUSH one entry, cap and PERSIST the list, and — while the archive is
        enabled — mirror the entry onto the Postgres outbox: one transaction, one
        round trip, one branch.

        LTRIM + PERSIST are the whole retention policy and are unconditional,
        identical whether the archive is enabled or not: the display list is
        runtime state, and HISTORY_ARCHIVE_ENABLED gates only the durable tier.
        BOUNDED because an unbounded non-evictable key is the OOM shape the note
        above describes. PERMANENT because -history reads this list and nothing
        else, so an idle TTL would answer a quiet guild with silence; the PERSIST
        also self-heals keys written by an older build that applied a 24h expiry.

        musicbot's HISTORY_MAX_LIMIT is pinned to HISTORY_CACHE_LIMIT, so the
        window this trim leaves behind always contains every play -history can be
        asked for — raising one without the other is what breaks it. Postgres is
        deliberately NOT on the read path; see
        docs/ARCHITECTURE.md#history-read-path.

        While the archive is enabled the same wire bytes are also XADDed onto the
        HISTORY_OUTBOX_KEY stream in the same transactional pipeline. Disabled —
        the default — that leg is absent and the key is never created. Both legs
        share one serialize_history_entry call so they cannot drift, and song end
        costs exactly one Redis round trip and never awaits Postgres.

        This method stays on the SWALLOWING side of the split (@_guild_op), unlike
        the module-level drain helpers, which raise: the playback loop must never
        die because Redis blinked. The consequence is that the producer can never
        report a mis-shaped outbox — a pre-stream list at this key fails the XADD
        leg with WRONGTYPE and would be swallowed into one warning per song —
        which is why ensure_outbox_group() aborts at STARTUP instead, the only
        place the signal can be loud.

        The outbox leg is idempotent under a re-sent transaction (a duplicate
        collapses on the archive's unique index); the guild:{id}:history LPUSH
        above it is NOT, and nothing dedups it.
        """
        wire = serialize_history_entry(entry)
        try:
            await self._push_history_pipeline(wire)
        except OutOfMemoryError:
            # AN ALREADY-FULL REDIS CANNOT SELF-HEAL WITHOUT THIS. LPUSH carries
            # Redis' `denyoom` flag and is queued FIRST, so once used_memory
            # exceeds maxmemory with nothing evictable the server refuses it at
            # queue time, the CAS is dirtied and EXEC aborts — measured on
            # redis:7-alpine: llen unchanged at 386, TTL still -1, the LTRIM never
            # ran. Reordering inside the MULTI does not help.
            #
            # A bare LTRIM is allowed at the cap (not denyoom) and frees memory
            # immediately, so the one command that could make room is exactly the
            # one that can still run. But at the steady state the trim frees
            # NOTHING — measured at maxmemory 3mb: llen stayed 50, the retry
            # aborted again, the play was dropped. The recovery is real exactly
            # once per guild per upgrade (an oversized legacy list); after that
            # the retry is a second attempt in case memory freed elsewhere, not a
            # guarantee. Log which case happened.
            #
            # Reachable even though the lists are capped: the outbox grows without
            # bound during a Postgres outage and fills the instance for every key.
            #
            # LLEN first, not LTRIM first: reads are not denyoom, and skipping a
            # trim that provably frees nothing removes a round trip from a path
            # that runs per song end for every guild while Redis is full.
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
        # UNPAGED on purpose, unlike HistoryOutboxDrainer._enforce_cap, which
        # pages its trim at CAP_PAGE=10_000 against the same hazard class. The
        # difference is that this one is O(1) in the steady state: LTRIM costs
        # O(N) in elements REMOVED, and every push after the first trims 51→50.
        # Only a list left over from a build that did not cap pays more, once.
        # Measured on redis:7-alpine 7.4.9 — 0.24 ms at 10k entries, 6.3 ms at
        # 100k, ~22 ms at 500k, with another guild's playback write seeing
        # head-of-line blocking equal to that duration. Nothing in the codebase
        # trips on a 30 ms stall (no socket_timeout on the pool, a 5s
        # healthcheck, a 2s bound on the -history read, and audio is never
        # Redis-gated). Paging it would mean carrying cursor state across song
        # ends to save a quarter of a millisecond.
        pipe.ltrim(self.history_key(), 0, HISTORY_CACHE_LIMIT - 1)
        # No EXPIRE anywhere near this key — see the docstring: length bounds it,
        # time never does. PERSIST because -history has no other source, and
        # because it clears an inherited TTL from an older build in one write.
        pipe.persist(self.history_key())
        # The outbox leg is the pipeline's ONE conditional command: the flag is
        # the consent gate for long-term storage, and with the archive disabled
        # nothing may accumulate for a drainer that does not exist — the XADD
        # would even MKSTREAM-like create the non-evictable key on first write.
        # Read per call (config's call-time convention, no hot path: once per
        # song end), through the module so tests can patch either the function
        # or the environment. A garbage value must never FIRST surface here:
        # @_guild_op would swallow the parser's ValueError into one warning per
        # song, so setup_hook is required to read the flag before anything else
        # consumes it — by the time a song ends, the process has already proven
        # the value parses (history_archive_enabled's validation-placement
        # rule).
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

        The two collapse to None deliberately: the only caller uses this to
        optionally restore a display embed, and "no embed" is right for both.
        """
        # bytes, not str: create_redis_pool() sets decode_responses=False, an
        # invariant redis-py's return type cannot express. Do NOT "simplify" this
        # away — from_redis() decodes, so a decoded pool breaks it at runtime.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.now_playing_key()))
        return NowPlayingData.from_redis(raw)

    # Playback position tracking

    @_guild_op(default=None)
    async def set_playback_start(self, epoch: float) -> None:
        """Record that playback started at `epoch`. Resets all pause accounting.

        For unit tests and standalone use; loop() writes these fields atomically
        via pop_queue_and_start_song() instead.
        """
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
        WATCH/MULTI retry loop.
        """
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
        representation of "no song". Used on normal song end and error-path skip.
        """
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

        Zero-value GuildStateData when the hash is missing/empty, None when the
        read itself failed — so callers can tell "nothing stored" from "Redis
        unavailable" (see _restore_guild).

        Pure read: does NOT refresh TTL. refresh_ttl() at the end of
        _restore_state() covers the recovery window.
        """
        # Same decode_responses=False invariant as get_now_playing() above.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.state_key()))
        return GuildStateData.from_redis(raw)

    @_guild_op(default=None)
    async def get_recovery_gate(self) -> Optional[GuildRecoveryGate]:
        """State hash + pending-queue *length* in one pipeline — the lightweight
        connection/restorable gate for `_restore_guild`.

        Deliberately transfers no queue contents, now-playing, or history: a
        -stopped guild keeps a possibly-long queue by design, and gating on LLEN
        keeps that payload off the wire on every on_ready. _restore_state re-reads
        the full snapshot after a successful connect, so contents are fetched once,
        only when used. One pipeline like get_playback_snapshot, but a fixed tiny
        payload. None on read failure (same contract as get_guild_state).
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
        everything or nothing rather than a partly-fabricated state; corrupt entries
        are dropped with a warning by their parsers. Not MULTI — recovery holds the
        guild lock during the window that matters.
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
        """Refresh GUILD_TTL on the TTL-managed guild keys. History is excluded
        for the reason _pipe_expire_all gives: it is bounded by length, never by
        time, because it is the only thing -history reads."""
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
        requester attribution, and every position-tracking field.
        """
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
