from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
from typing import Any, Concatenate, Optional, ParamSpec, TypeVar, cast

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import Pipeline
from redis.typing import EncodableT, FieldT

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
# entries from all guilds will interleave here (each carries its guild_id on
# the wire), oldest-first. Deliberately NO TTL — it is meant to hold
# not-yet-durable entries, so under maxmemory-policy volatile-lru it must never
# be an eviction candidate. See docs/POSTGRES_HISTORY_PLAN.md §3.
#
# NOTHING WRITES TO THIS KEY YET, and that is the point of where this branch
# stops. Only the drain-side helpers below exist here. A producer without a
# consumer would grow a non-evictable key at ~412 bytes per play with nothing
# ever popping it — golden rule 12's exact failure mode, and OOM against the
# compose Redis' 256mb in under two weeks. So push_history's outbox leg and the
# drainer that empties it land together, in the branch that adds the archive.
HISTORY_OUTBOX_KEY = "history:outbox"
# Cross-process drainer ownership (H2). peek → INSERT → retire is only safe
# with a single drainer, and the structure it guards lives in this same Redis,
# so a single-instance lease is the correct failure domain (no Redlock).
DRAINER_LEASE_KEY = "history:drainer"
# INVARIANT: must exceed HistoryOutboxDrainer.DRAIN_DEADLINE_SECS * 1000, so a
# batch can never outlive the ownership it was started under. A test asserts it.
DRAINER_LEASE_MS = 90_000
# How many entries trim_outbox_oldest() drops per RPOP. Bounds both the reply
# size and the time Redis spends inside one command, so capping a backlog of
# any size stays incremental; see that function for what one unbounded pop did.
_TRIM_SLICE = 1000
# 24h idle expiry, never applied to the history key: push_history PERSISTs that
# list and every TTL path below excludes it unconditionally.
GUILD_TTL = 86400
# In-memory/display cap only — NOT a retention cap. The Redis history list is
# unbounded (source of truth for all played songs; Postgres eventually — see
# docs/HISTORY_OVERHAUL_PLAN.md §4/§8); this bounds the GuildHistory deque and
# every history read so startup stays O(50) against an unbounded list.
HISTORY_CACHE_LIMIT = 50

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


def _hset_mapping(mapping: dict[str, str]) -> Mapping[FieldT, EncodableT]:
    """Adapt a guild_state str→str mapping to redis-py's HSET mapping type.

    Mapping's key parameter is invariant, so dict[str, str] is not assignable to
    Mapping[FieldT, EncodableT] even though str is one of FieldT's own members.
    The cast is a variance workaround only — it widens nothing at runtime.
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
# The mechanics, ahead of the tier that uses them: these are the operations the
# archive's drainer will run, with their own tests, and nothing in the bot calls
# them yet (see HISTORY_OUTBOX_KEY above for why the producer waits). Unlike the
# cache helpers above, they deliberately DO raise on Redis failure — the
# drainer's backoff loop is the error handler, and a swallowed error here would
# look like an empty outbox and silently stall the drain. Raw bytes in/out: wire
# parsing stays in guild_state.py (parse_history_entry), per the schema rule.
#
# Single-consumer assumption: peek → INSERT → retire is only safe with one
# drainer per outbox (two would both peek the same tail, then the second
# retire would pop *unprocessed* entries — reproduced as 6 pushed / 3 archived
# / outbox empty). One bot process per Redis is already the deployment's
# operating rule (see the recovery lock and docs/K8S_DEPLOYMENT_PLAN.md), but
# "already the rule" is not an enforcement mechanism, and the rolling deploys
# on that same roadmap break it by construction for the length of a handoff.
# hold_drainer_lease() below turns the assumption into a guarantee.


async def peek_outbox_oldest(redis: aioredis.Redis, count: int) -> list[bytes]:
    """The oldest ≤count outbox entries, oldest first, left in place.
    LPUSH writes at the head, so the tail slice LRANGE -count..-1 is the
    oldest run; Redis returns it in list order (newer→older), hence the
    reversal."""
    # cast, not a bare annotation: decode_responses=False on the pool, so the
    # list is bytes — same convention as the HGETALL readers below.
    raw = cast(list[bytes], await redis.lrange(HISTORY_OUTBOX_KEY, -count, -1))
    raw.reverse()
    return raw


async def retire_outbox(redis: aioredis.Redis, count: int) -> None:
    """Drop the oldest `count` entries — call only after their Postgres
    INSERT committed (crash between insert and retire redelivers; the
    archive's unique index dedups). RPOP pops from the tail (oldest), so
    concurrent LPUSHes at the head are never touched."""
    if count > 0:
        await redis.rpop(HISTORY_OUTBOX_KEY, count)


async def outbox_depth(redis: aioredis.Redis) -> int:
    """Current outbox length — the drainer's backlog watchdog metric."""
    return await redis.llen(HISTORY_OUTBOX_KEY)


async def trim_outbox_oldest(redis: aioredis.Redis, count: int) -> None:
    """Drop the oldest `count` entries WITHOUT archiving them — the opt-in
    HISTORY_OUTBOX_MAX cap only (history_archive.py). Same end of the list as
    retire_outbox; a separate name so `grep retire_outbox` still means
    "entries that reached Postgres" and this one always reads as data loss.

    Popped in _TRIM_SLICE-sized slices, NOT as one `RPOP key <count>`. Unlike
    retire_outbox — whose count is bounded by BATCH_SIZE at its only call
    site — this count is `depth - HISTORY_OUTBOX_MAX`, i.e. however far a
    Postgres outage ran. RPOP with a count *returns what it popped*, so a
    single call carries the whole dropped set back over the socket: measured
    5.3s and 206 MB for 490k entries. That is bad twice over. Redis is
    single-threaded, so it head-of-line-blocks every other guild for the
    duration (a concurrent `SET guild:state` measured 1.46s), and it exceeds
    redis-py's default 5s socket timeout, whereupon retry_on_timeout re-issues
    this DESTRUCTIVE pop against a list Redis has already popped — silently
    emptying a 500k outbox while logging "dropped 490,000". Slicing keeps each
    reply small and each command short regardless of how large the backlog got.
    """
    while count > 0:
        popped = await redis.rpop(HISTORY_OUTBOX_KEY, min(count, _TRIM_SLICE))
        if not popped:
            # Raced to empty by a concurrent push/drain; nothing left to drop.
            return
        count -= len(cast(list[bytes], popped))


async def _if_still_owner(
    redis: aioredis.Redis,
    lease_id: str,
    act: Callable[[Pipeline], object],
) -> bool:
    """Run `act` on the lease key iff `lease_id` still owns it, atomically.

    Renew and release are both compare-and-act, which a bare GET-then-PEXPIRE
    cannot be: between the two, our lease can lapse and another process can
    take it, and we would then extend or delete a lease we no longer hold —
    the exact two-drainers state the lease exists to prevent.

    WATCH/MULTI rather than a Lua script, deliberately: EVAL needs a real Lua
    interpreter, which fakeredis only has with an extra dependency, and a lease
    whose correctness cannot be tested is not worth having. The optimistic
    retry closes the window — if the key changed after WATCH, EXEC aborts
    (WatchError) and we re-read. Key expiry needs no special handling either
    way: an expired-and-untaken key reads as None below, and an
    expired-and-retaken one reads as somebody else's id.
    """
    async with redis.pipeline() as pipe:
        while True:
            try:
                await pipe.watch(DRAINER_LEASE_KEY)
                # decode_responses=False on the pool, so this is bytes (same
                # cast convention as the HGETALL readers).
                current = cast(Optional[bytes], await pipe.get(DRAINER_LEASE_KEY))
                if current is None or current.decode() != lease_id:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                act(pipe)
                await pipe.execute()
                return True
            except aioredis.WatchError:
                continue


async def hold_drainer_lease(redis: aioredis.Redis, lease_id: str) -> bool:
    """True when `lease_id` owns the drain right for the next DRAINER_LEASE_MS.

    Acquire-or-renew in one call: SET NX PX takes a free lease, and the
    compare-and-renew above extends one we already hold. False means somebody
    else owns it — the caller must not touch the outbox this cycle.

    The TTL is what makes a dead holder recoverable: a process that dies
    mid-batch never releases, and the lease simply lapses ≤90s later, at which
    point any survivor's next SET NX takes over.
    """
    if await redis.set(DRAINER_LEASE_KEY, lease_id, nx=True, px=DRAINER_LEASE_MS):
        return True
    return await _if_still_owner(
        redis, lease_id, lambda pipe: pipe.pexpire(DRAINER_LEASE_KEY, DRAINER_LEASE_MS)
    )


async def release_drainer_lease(redis: aioredis.Redis, lease_id: str) -> None:
    """Hand the lease back at shutdown so the next instance starts draining
    immediately instead of waiting out the TTL. Compare-and-delete: a lease
    that already lapsed and was retaken belongs to someone else now."""
    await _if_still_owner(redis, lease_id, lambda pipe: pipe.delete(DRAINER_LEASE_KEY))


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

    Every store method is a `try: <redis IO> except Exception: log.warning(...)`
    with only the body and the default return value differing. This decorator
    factors the try/except/log out: on any exception it logs
    `[guild:{id}] {method} failed: {e}` and returns `default`, so each method
    is just its happy path. The method name comes from the wrapped function, so
    the log line names the operation without repeating it by hand.

    Pass `default` for immutable fallbacks (None, False) and `default_factory`
    for anything mutable. A decorator argument is evaluated ONCE, at class-body
    execution, so `default=[]` would hand the *same* list to every guild on
    every failure — one caller mutating it in place would poison "empty" for
    the whole process. The factory is called per failure instead, so each
    caller gets its own object. Do not add a mutable `default`; the test
    `test_mutable_defaults_use_a_factory` fails if you do.

    `default` is typed Any (not `_R`) on purpose: pinning it to the return
    TypeVar would let `default=None` collapse `_R` to `None` for the
    Optional-returning readers. Inferred from the wrapped function alone, `_R`
    keeps each decorated method's exact caller-facing signature; the onus that
    `default` matches the return type falls to the (obvious) call sites below.
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
        pipeline. The history key is deliberately absent: full history is
        retained indefinitely (PERSISTed by push_history), so it must never be
        re-armed with an idle expiry."""
        pipe.expire(self.queue_key(), GUILD_TTL)
        pipe.expire(self.state_key(), GUILD_TTL)
        pipe.expire(self.now_playing_key(), GUILD_TTL)

    async def _exec_with_state_ttl(self, pipe: Pipeline) -> None:
        """Append the state-key TTL refresh and execute the pipeline.

        EXPIRE must come after the write commands already queued on the pipe —
        an EXPIRE on a not-yet-created key is a no-op and would leave the key
        persistent-until-eviction.
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
        """LPUSH entries so entries[0] ends up at the queue head, and refresh
        TTL on all guild keys — the -playnow front insert. LPUSH pushes each
        successive argument to the head, so the batch is reversed first to
        preserve the given order.

        Failure note (store policy: log, never raise): a swallowed failure
        here degrades WORSE than a tail-push failure. The in-memory legs end
        up len(entries) ahead of Redis at the HEAD, so the next commit-time
        LPOPs retire other songs' entries — a crash before the mismatch
        drains restores a queue shifted by up to that many songs, not just
        missing the entries that failed to push."""
        if not entries:
            return
        pipe = self.redis.pipeline()
        pipe.lpush(self.queue_key(), *[e.to_redis() for e in reversed(entries)])
        self._pipe_expire_all(pipe)
        await pipe.execute()

    @_guild_op(default=None)
    async def pop_queue(self) -> None:
        # At-most-once: LPOP removes the item immediately with no ack.
        # If the bot crashes after this call, the song is lost from Redis.
        # This is acceptable in Phase 2 (asyncio.Queue is source of truth).
        # Phase 3b migrates to Redis Streams + XACK for at-least-once.
        await self.redis.lpop(self.queue_key())

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

    @_guild_op(default=None)
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
        """Same fields as pop_queue_and_start_song, in one transaction, but
        without the LPOP — for restarting a crash-recovered "current song" that
        was never RPUSHed to the Redis queue list in the first place.
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

    # ISSUE: Unbounded, non-evictable history can exhaust Redis and stall all writes.
    # History keys carry no TTL, so under `maxmemory-policy volatile-lru` they are never
    # eviction candidates. Once history fills the 256mb maxmemory and no TTL-bearing key
    # is left to evict, Redis rejects EVERY write with OOM — not just history: state,
    # queue, and cache writes all start failing (each store method swallows the error and
    # logs, so persistence silently degrades rather than crashing). Small entries make
    # this a slow burn (~1M+ entries), but "unbounded" means it does arrive. The fix in
    # progress is migrating full history to Postgres and demoting the Redis list to a
    # bounded cache (docs/POSTGRES_HISTORY_PLAN.md); until then this needs a Redis
    # memory/eviction alarm. Do NOT switch back to allkeys-lru as a workaround — that
    # would make history itself an eviction candidate and defeat the whole
    # persistent-history design (see docker-compose.yml redis command).
    #
    # HISTORY_OUTBOX_KEY is declared with the same no-TTL contract and will be the
    # second, faster route into this same failure — but nothing writes to it in this
    # branch, so today history is the only non-evictable key under memory pressure.
    @_guild_op(default=None)
    async def push_history(self, entry: HistoryEntry) -> None:
        """LPUSH one entry and PERSIST the key — no trim, no TTL: the list is
        the unbounded source of truth for all played songs (write-per-song-end
        is the durability boundary; cadence analysis in
        docs/HISTORY_OVERHAUL_PLAN.md §4). The PERSIST also self-heals
        pre-migration keys still carrying the old 24h idle expiry."""
        pipe = self.redis.pipeline()
        pipe.lpush(self.history_key(), serialize_history_entry(entry))
        pipe.persist(self.history_key())
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

        Miss and error collapse to None deliberately: the only caller uses this
        to optionally restore a display embed, and "no embed" is the correct
        degraded behavior in both cases.
        """
        # bytes keys/values, not str: create_redis_pool() sets
        # decode_responses=False (see :75), an invariant redis-py's own return
        # type cannot express. Do not "simplify" this away — from_redis()
        # decodes, and a decoded pool would break it at runtime, not here.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.now_playing_key()))
        return NowPlayingData.from_redis(raw)

    # Playback position tracking

    @_guild_op(default=None)
    async def set_playback_start(self, epoch: float) -> None:
        """Record that playback started at `epoch`. Resets all pause accounting.

        Kept for unit tests and standalone use. In loop(), position fields are
        written atomically via pop_queue_and_start_song() instead.
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
        """Accumulate elapsed pause time into total_pause_seconds and clear pause_start_epoch.

        Non-atomic read-modify-write: assumes a single writer per guild (true
        for the current one-process-per-guild command flow). Under
        multi-process sharding this must become a Lua script or WATCH/MULTI
        retry loop — see docs/REDIS_MIGRATION_PLAN.md.
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
        """Pipeline that clears all transient song state in one round-trip.

        HDELs all current_song_* and playback-position fields and DELETEs the
        now_playing hash — the same idiom clear_connection() uses, so absent
        (not empty-string) is the one representation of "no song". Called on
        both normal song end and the error-path skip in loop().
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

        Returns GuildStateData() (zero values) when the hash is missing/empty
        and None when the read itself failed — callers can distinguish "nothing
        stored" from "Redis unavailable" (see _restore_guild).

        Does NOT refresh TTL — pure read. TTL is refreshed by refresh_ttl() at
        the end of _restore_state(), which covers the recovery window.
        """
        # Same decode_responses=False invariant as get_now_playing() above.
        raw = cast(dict[bytes, bytes], await self.redis.hgetall(self.state_key()))
        return GuildStateData.from_redis(raw)

    @_guild_op(default=None)
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
        for the same reason as in _pipe_expire_all: the key is persistent."""
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
        """Remove all transient state on intentional disconnect.

        Clears voice/text channel IDs so on_ready skips recovery for this guild.
        Also clears now-playing display, requester attribution, and all playback
        position tracking fields.
        """
        pipe = self.redis.pipeline()
        pipe.hdel(
            self.state_key(),
            StateField.VOICE_CHANNEL_ID,
            StateField.TEXT_CHANNEL_ID,
            *_TRANSIENT_SONG_FIELDS,
            # HACK: last_author_id is dead schema still scrubbed on every disconnect.
            # Nothing writes this field any more; the HDEL exists only to clean up
            # state hashes left behind by older builds that did, which is why it is
            # a bare string literal rather than a StateField constant. Every guild
            # disconnect now pays to delete a field that cannot exist on any hash
            # written since that migration.
            # Safe to delete once no pre-migration hash can still be live — guild
            # keys carry a 24h TTL, so one release is already more than enough.
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
