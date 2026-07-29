"""
Postgres play-history archive — the durable long-term home for every played
song (docs/POSTGRES_HISTORY_PLAN.md).

Three pieces:

- HistoryArchive — the protocol GuildHistory and the drainer program against;
  unit tests substitute an in-memory fake.
- PostgresHistoryArchive — the asyncpg implementation. Lazily connects on first
  use so bot startup never blocks on Postgres being reachable (the drainer's
  backoff loop absorbs failures instead). It no longer applies DDL: the schema
  is owned by migrations/ and src/db_migrate.py, and this verifies the version
  rather than creating it (see _ensure).
- HistoryOutboxDrainer — the single background task that moves entries from
  the Redis outbox list to Postgres: peek oldest batch → INSERT ... ON
  CONFLICT DO NOTHING → retire. At-least-once delivery; a crash between
  insert and retire redelivers, and the play_history_dedup unique index
  collapses the replay. The playback loop never awaits Postgres — add()
  LPUSHes the outbox and notify()s this task.

Row mapping (HistoryEntry ↔ play_history row) lives here, not in
guild_state.py — that module's contract is pure wire schema with no runtime
imports, and asyncpg is very much a runtime import.

Failure-handling shape, and why it is this elaborate: the outbox is
non-evictable by design, so anything that stops the drain grows a Redis list
that eventually refuses every write in the process, not just history's. Each
guard below closes one way that could happen.

  poison entry     an entry Postgres will never accept blocks the batch behind
                   it forever → prevented at construction
                   (HistoryEntry.__post_init__ clamps into the column domain,
                   so the class of entry no longer exists). _isolate remains as
                   the backstop for a validator regression or schema drift, and
                   parks what it cannot insert in play_history_rejected
  hung server      a connected-but-unresponsive Postgres never returns, so
                   there is no exception, no backoff, and no alarm →
                   command_timeout + DRAIN_DEADLINE_SECS
  two drainers     peek/retire is single-consumer; a second one retires
                   entries the first never inserted → the Redis drainer lease
  dead drainer     the task dies and nothing notices until a restart →
                   _on_task_done respawn with exponential damping
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional, Protocol
from uuid import uuid4

import asyncpg
import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import Span

from src import config
from src.db_migrate import EXPECTED_SCHEMA_VERSION
from src.guild_state import HistoryEntry, parse_history_entry, serialize_history_entry
from src.redis_client import (
    hold_drainer_lease,
    outbox_depth,
    peek_outbox_oldest,
    release_drainer_lease,
    retire_outbox,
    trim_outbox_oldest,
)
from src.telemetry import get_tracer
from src.util import get_logger, trace_id_of

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_INSERT_SQL = """
INSERT INTO play_history (guild_id, title, webpage_url, duration_secs,
                          played_secs, requester_id, requester_name,
                          thumbnail, uploader, played_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (guild_id, played_at, webpage_url) DO NOTHING
"""

_RECENT_SQL = """
SELECT guild_id, title, webpage_url, duration_secs, played_secs,
       requester_id, requester_name, thumbnail, uploader, played_at
FROM play_history
WHERE guild_id = $1
ORDER BY played_at DESC, id DESC
LIMIT $2
"""

_REJECT_SQL = """
INSERT INTO play_history_rejected (guild_id, error_type, error_detail, trace_id, payload)
VALUES ($1, $2, $3, $4, $5)
"""

_SCHEMA_VERSION_SQL = "SELECT max(version) FROM schema_migrations"

# Cap on the asyncpg message stored in play_history_rejected.error_detail. Long
# enough to keep the SQLSTATE and the offending value, short enough that a
# pathological message cannot bloat a table meant to stay empty.
_REJECT_DETAIL_MAX = 2000

# How long any single statement may run before asyncpg cancels it. Bounds the
# case timeout=10 does not: a connection that established fine against a server
# that then stops answering. Generous relative to a 100-row executemany
# (milliseconds) — this is a liveness bound, not a latency target.
_COMMAND_TIMEOUT_SECS = 30.0
# How long a caller may wait for a free connection from the pool. The drainer,
# -ping's health probe and Phase B reads share max_size=4, so a stuck consumer
# must not turn into an unbounded wait for everyone else.
_ACQUIRE_TIMEOUT_SECS = 10.0
# How long close() waits for a graceful pool shutdown before terminating it.
# Short on purpose: this runs on the shutdown path, ahead of the Redis pool,
# discord.py's own close and the span flush, and every second spent here is a
# second nothing else in teardown is happening.
_POOL_CLOSE_TIMEOUT_SECS = 5.0


def _entry_to_row(entry: HistoryEntry) -> tuple:
    return (
        entry.guild_id,
        entry.title,
        entry.webpage_url,
        entry.duration_secs,
        entry.played_secs,
        entry.requester_id,
        entry.requester_name,
        entry.thumbnail,
        entry.uploader,
        datetime.fromtimestamp(entry.played_at, tz=timezone.utc),
    )


def quantized_played_at(played_at: float) -> float:
    """`played_at` as it will read back out of Postgres.

    timestamptz is microsecond-granular, but a raw time.time() float is finer
    than that near 2026 (~2.4e-7 s ULP), so a value and its round trip are
    UNEQUAL for roughly a third of real timestamps (measured: 37.7%). Any
    Python-side identity built from both sides of the archive boundary must put
    BOTH through this, or it compares a quantized float against a raw one and
    never matches. The one such identity today is the (played_at, webpage_url)
    dedup key in GuildHistory.recent()'s merge; getting it wrong there showed
    every affected play twice AND, because the merge slices to `limit`
    afterwards, silently dropped a real song per duplicate.

    Deliberately the same datetime pair _entry_to_row/_row_to_entry use, NOT
    round(t * 1e6) / 1e6: those disagree by 1us at ~0.125-ULP boundaries, and
    only this one is what the database actually does. Idempotent, so applying
    it to a value already read from a row is a no-op.

    Total by construction. HistoryEntry.__post_init__ already clamps played_at
    into the timestamptz domain, so the guard below is unreachable through any
    entry — but this takes a bare float, and it runs on the -history read path,
    which must degrade rather than error. Catching what the conversion actually
    raises (ValueError for NaN and for years outside 1..9999, OverflowError for
    values past platform time_t) is more honest than re-deriving the bound.
    """
    try:
        return datetime.fromtimestamp(played_at, tz=timezone.utc).timestamp()
    except ValueError, OverflowError, OSError:
        return played_at


def _row_to_entry(row: asyncpg.Record) -> HistoryEntry:
    return HistoryEntry(
        guild_id=row["guild_id"],
        title=row["title"],
        webpage_url=row["webpage_url"],
        duration_secs=row["duration_secs"],
        played_secs=row["played_secs"],
        requester_id=row["requester_id"],
        requester_name=row["requester_name"],
        thumbnail=row["thumbnail"],
        uploader=row["uploader"],
        played_at=row["played_at"].timestamp(),
    )


class HistoryArchive(Protocol):
    """What GuildHistory (Phase B reads) and the drainer (writes) need from
    the archive — faked in unit tests, implemented by asyncpg below."""

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None: ...

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]: ...

    async def record_rejection(
        self, entry: HistoryEntry, error: BaseException, trace_id: str = ""
    ) -> None: ...


class SchemaVersionError(RuntimeError):
    """The database's schema is older than the code expects.

    Its own type so callers can tell "run the migrations" apart from "Postgres
    is down" — the first is an operator action, the second is a wait.
    """


class PostgresHistoryArchive:
    """asyncpg-backed archive. All methods raise on failure — callers own the
    error policy (the drainer backs off; Phase B's recent() falls back)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _create_pool(self) -> asyncpg.Pool:
        return await asyncpg.create_pool(
            self._url,
            min_size=1,
            max_size=4,
            # Connect bound: a fast connect failure keeps the drainer's backoff
            # loop responsive (asyncpg's default is 60s).
            timeout=10,
            # Statement bound. Without it, a server that accepts the connection
            # and then stops answering hangs executemany forever — which
            # produces no exception, so no backoff, no DEPTH_ALARM and not one
            # log line while the outbox grows. See H1 in the module docstring.
            command_timeout=_COMMAND_TIMEOUT_SECS,
            # Prepared statements are per-connection, so PgBouncer in
            # transaction-pooling mode breaks them. Setting POSTGRES_STATEMENT_CACHE=0
            # turns the cache off for that deployment shape; the default matches
            # asyncpg's own.
            statement_cache_size=config.POSTGRES_STATEMENT_CACHE,
            # Makes this bot's connections identifiable in pg_stat_activity,
            # which is the first thing anyone looks at when the database is the
            # suspect.
            server_settings={"application_name": "musicbot-history"},
        )

    async def _ensure(self) -> asyncpg.Pool:
        """Lazy pool + schema-version check, double-checked under the lock.
        First successful call wins; a failed attempt leaves no half-open pool.

        Refuses after close() so a late caller cannot resurrect a pool that
        nothing will ever close again — see close(). The check is repeated
        under the lock and again after the awaits, because close() can win
        either race: _closed is read before the lock is taken, and the
        connect + version query in between are two more suspension points.
        Every escape from that window routes through the BaseException handler,
        so the just-built pool is always closed by whoever built it.
        """
        if self._closed:
            raise RuntimeError("PostgresHistoryArchive is closed")
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._closed:  # close() may have won the race to the lock
                raise RuntimeError("PostgresHistoryArchive is closed")
            if self._pool is None:
                pool = await self._create_pool()
                try:
                    async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                        await self._assert_schema_version(conn)
                    if self._closed:  # close() ran during our awaits
                        raise RuntimeError("PostgresHistoryArchive is closed")
                except BaseException:
                    # Covers the version check failing, cancellation, and the
                    # re-check above — all three would otherwise leak a pool
                    # that self._pool never received and close() cannot see.
                    await pool.close()
                    raise
                self._pool = pool
        return self._pool

    @staticmethod
    async def _assert_schema_version(conn: Any) -> None:
        """Verify the database carries the schema this build was written for.

        This replaces the CREATE TABLE IF NOT EXISTS the archive used to run on
        first connect. Two reasons it had to go: DDL from the application means
        the bot's role must hold DDL rights permanently, and — the reason it
        was actually blocking — `IF NOT EXISTS` can only ever create the
        original schema, so there was no way to ship a change to it.

        A *newer* database is tolerated with a warning rather than refused.
        Migrations here are additive (expand/contract), so an older bot reads a
        newer schema fine, and refusing would mean a rolled-back deploy cannot
        start — turning a routine rollback into an outage.

        `conn` is Any because the pool hands out a PoolConnectionProxy, not the
        asyncpg.Connection the same call site would get from asyncpg.connect();
        the two are duck-identical for fetchval and the union buys nothing.
        """
        try:
            version = await conn.fetchval(_SCHEMA_VERSION_SQL)
        except asyncpg.exceptions.UndefinedTableError:
            version = None  # never migrated at all
        if version is None or version < EXPECTED_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"play_history schema is at version {version if version else 'none'}, "
                f"this build needs {EXPECTED_SCHEMA_VERSION}. "
                f"Run `just db-migrate` (or `python -m src.db_migrate`) against "
                f"the archive database."
            )
        if version > EXPECTED_SCHEMA_VERSION:
            log.warning(
                f"play_history schema is at version {version}, ahead of this "
                f"build's {EXPECTED_SCHEMA_VERSION}; continuing (migrations are "
                f"additive), but this build is older than the database."
            )

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        """Insert oldest-first; replays and backfill overlap dedup via the
        play_history_dedup unique index (ON CONFLICT DO NOTHING).

        No conversion guard: HistoryEntry.__post_init__ already clamped every
        field into this table's column domain, so _entry_to_row cannot fail and
        asyncpg cannot be handed a value it will not encode. That is the schema
        lock (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md §5), and it is asserted
        directly in the pg tier by
        test_every_constructible_entry_is_insertable."""
        if not entries:
            return
        rows = [_entry_to_row(e) for e in entries]
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            await conn.executemany(_INSERT_SQL, rows)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        """The `limit` most recent entries for one guild, newest first. id is
        the tie-break so epoch-0 (unknown-time) entries order stably."""
        if limit <= 0:
            return []
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            rows = await conn.fetch(_RECENT_SQL, guild_id, limit)
        return [_row_to_entry(r) for r in rows]

    async def record_rejection(
        self, entry: HistoryEntry, error: BaseException, trace_id: str = ""
    ) -> None:
        """Park one refused play in play_history_rejected. Best-effort and
        TERMINAL: never retries, never recurses.

        A rejects table that can itself fail into a retry loop is worse than no
        rejects table — so if this insert fails, the payload goes to the log and
        the caller moves on.

        This is always reachable at the moment it is needed, which is not
        obvious and is the reason the design works: a REJECTION means the server
        is up and said no to this row, so the same connection that carried the
        refusal carries this insert. An OUTAGE is the other case entirely, and it
        never reaches here — the entry stays on the outbox and redelivers. There
        is no third case.

        error_detail is text, and asyncpg messages echo the offending value, so
        it is NUL-scrubbed and capped: the same poison this table exists to
        record would otherwise fail the insert recording it, one layer up.
        payload is bytea and takes the orjson bytes verbatim, which is what
        makes replay exact (migrations/0005 documents why neither jsonb nor text
        can hold them).
        """
        detail = str(error).replace("\x00", "")[:_REJECT_DETAIL_MAX]
        # serialize_history_entry cannot raise here: `entry` was constructed, so
        # __post_init__ already proved it orjson-encodable (findings §3.1).
        payload = serialize_history_entry(entry)
        try:
            pool = await self._ensure()
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                await conn.execute(
                    _REJECT_SQL,
                    entry.guild_id,
                    type(error).__name__,
                    detail,
                    trace_id,
                    payload,
                )
        except Exception as e:
            log.error(
                f"play rejected AND unrecordable ({type(e).__name__}: {e}); "
                f"payload={payload!r}"
            )

    async def health_check(self) -> None:
        """Prove the archive's database is reachable and answering. Raises on
        any failure — the caller (-ping's probe_postgres) times it and turns an
        exception into a red row.

        Connects if nothing has yet: the pool is lazy, so before the first
        song-end it is simply absent, and reporting "not configured" for a
        REQUIRED tier would be a lie an operator acts on. That makes this the
        second _ensure() caller after the drainer, which is what the closed
        guard above exists for. Pool creation is bounded by _ensure's
        timeout=10; -ping's own deadline (PING_DEADLINE_SECS, 3s) is shorter,
        so a hung first connect renders as FAILED rather than DOWN — both red.

        SELECT 1 rather than a table read: this asks "is the server up and
        answering on a real connection", not "is the schema right" — _ensure's
        version check already settled that, and settles it here too, since an
        unmigrated database fails this probe with the actionable message.
        """
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            await conn.execute("SELECT 1")

    async def close(self) -> None:
        """Close the pool. Terminal: _ensure() refuses afterwards.

        _closed is set FIRST so new callers are turned away immediately, then
        the init lock is taken so an _ensure() already inside its connect
        cannot finish and hand a fresh pool to a _pool field nobody will read
        again. Waiting on that lock can cost up to the 10s connect timeout at
        shutdown, which is the price of not leaking a live pool and its
        connections.

        The ordering it used to rely on — MusicBotApp.close() runs
        drainer.stop() strictly before archive.close(), so no _ensure() can
        race a second pool into existence — still holds for the drainer, but
        health_check() added a caller that shutdown does NOT sequence: a -ping
        probe in flight while the bot is closing. Keep the two closes in that
        order anyway; it is what lets the drainer's final drain still reach
        Postgres.
        """
        self._closed = True
        async with self._init_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            try:
                async with asyncio.timeout(_POOL_CLOSE_TIMEOUT_SECS):
                    await pool.close()
            except asyncio.CancelledError:
                pool.terminate()  # don't leave sockets behind on the way out
                raise
            except Exception as e:
                # A graceful close waits for in-flight queries to drain, which
                # a connected-but-unresponsive server never allows: measured at
                # 30s against a hung Postgres, and it RAISED — which aborted
                # every remaining step of MusicBotApp.close(), leaving the
                # Redis pool open, discord.py unclosed, the yt-dlp pool to its
                # 61s atexit join, and no spans flushed, so the very hang that
                # caused it was invisible in Tempo. terminate() is synchronous
                # and unconditional; closing an archive must not be able to
                # take the shutdown down with it.
                pool.terminate()
                log.warning(
                    f"history archive pool close forced: {type(e).__name__}: {e}"
                )


# Errors that mean "this data will never be accepted", as opposed to "try again
# later".
#
# Since the schema lock (HistoryEntry.__post_init__) this set is expected to be
# UNREACHABLE: every entry on the outbox was clamped into the play_history column
# domain at construction. It survives as the backstop for the two ways that can
# stop being true — the validator regressing, or this build talking to a schema
# it was not written for — because dropping a whole batch on either would turn a
# one-row bug into a hundred-play loss.
#
#   DataError            SQLSTATE 22xxx, server-side data rejections:
#                        CharacterNotInRepertoireError for NUL bytes,
#                        NumericValueOutOfRangeError, …
#   CheckViolationError  SQLSTATE 23514, and NOT a DataError — verified: it
#   NotNullViolationError inherits from IntegrityConstraintViolationError.
#                        migrations/0004 adds the CHECK constraints that are the
#                        enforced half of the schema lock, and without these two
#                        arms a violation would propagate straight past the
#                        quarantine path and wedge the drain head permanently on
#                        a non-evictable list, rather than dead-lettering one row.
#
# Deliberately NOT here:
#   - UniqueViolationError. play_history_dedup is the ON CONFLICT target, so it
#     cannot surface; catching it would hide a genuine index bug.
#   - bare ValueError. asyncpg raises it for whole-statement problems like an
#     argument-count overflow, which is not per-entry poison.
#   - bare TypeError. asyncpg's parameter encoders raise it on a value they
#     cannot encode, but so does an ordinary bug anywhere inside insert_batch,
#     and naming it here would dead-letter a healthy batch on a refactor. The
#     encoder case is now unreachable anyway: __post_init__ guarantees every
#     field is in-domain before _entry_to_row runs.
#   - anything OSError-shaped. InterfaceError, ConnectionResetError and
#     TimeoutError are how a Postgres restart or failover presents, and
#     dead-lettering on those would delete healthy history on every outage.
_POISON = (
    asyncpg.exceptions.DataError,
    asyncpg.exceptions.CheckViolationError,
    asyncpg.exceptions.NotNullViolationError,
)


class HistoryOutboxDrainer:
    """The one task per process that drains the Redis outbox into the archive.

    Wakes on notify() (set by every outbox push) with a periodic fallback
    tick, drains in batches until the outbox is empty, and on archive/Redis
    failure backs off exponentially while entries accumulate safely in the
    outbox (persistent, non-evictable — see HISTORY_OUTBOX_KEY).

    Single-consumer by design: the peek → insert → retire cycle is only safe
    with one drainer per outbox (redis_client.py, "History outbox" section).
    Enforced rather than assumed — every cycle runs under a Redis lease, so a
    second instance (a k8s rollout overlapping two pods, a stray process)
    waits instead of retiring entries it never inserted.
    """

    BATCH_SIZE = 100
    TICK_SECS = 30.0
    DEPTH_ALARM = 10_000  # backlog that escalates the retry warning to ERROR
    # Whole-cycle bound. Belt to command_timeout's braces: it covers connection
    # acquisition, DNS re-resolution and anything else asyncpg does not bound
    # itself, so ANY hang becomes a TimeoutError that flows into _run's normal
    # error path — which is what makes DEPTH_ALARM fire for hangs and not only
    # for errors. Must stay below DRAINER_LEASE_MS so a batch can never outlive
    # the ownership it started under (a test asserts the inequality).
    DRAIN_DEADLINE_SECS = 60.0
    # Respawn damping for a drainer task that dies outside its own error
    # handling (M5). Base is short because the first restart usually works;
    # the cap keeps a hard-broken drainer from becoming a log flood.
    RESTART_BASE = 5.0
    RESTART_MAX = 300.0
    # 0 = unbounded, the durability default. See config.HISTORY_OUTBOX_MAX.
    OUTBOX_MAX = config.HISTORY_OUTBOX_MAX
    _BACKOFF_START = 1.0
    _BACKOFF_MAX = 60.0

    def __init__(self, redis: aioredis.Redis, archive: HistoryArchive) -> None:
        self._redis = redis
        self._archive = archive
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        # Per-process identity for the drain lease. Regenerated per instance,
        # never persisted: a restarted process must NOT inherit the lease of
        # the process it replaced, or a lease leaked by a crash would be
        # silently renewed rather than waited out.
        self._lease_id = uuid4().hex
        # Shutdown/supervision state.
        self._stop_lock = asyncio.Lock()
        self._stopped = False
        self._stopping = False
        self._restart_delay = self.RESTART_BASE
        self._respawn_handle: Optional[asyncio.TimerHandle] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stopping = False
        # _stopped too: stop() latches it forever, so a start-after-stop used to
        # spawn a live _run that the next stop() returned early from without
        # cancelling — a leaked task still draining after teardown, holding a
        # lease it never releases. No caller does this today, but stop()'s
        # reentrancy is documented as a correctness property.
        self._stopped = False
        self._restart_delay = self.RESTART_BASE
        self._spawn()

    def _spawn(self) -> None:
        self._task = asyncio.create_task(self._run(), name="history-outbox-drainer")
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: "asyncio.Task[None]") -> None:
        """Supervision. _run only ever exits via cancellation, so any exception
        surfacing here is a bug — but a bug that leaves the non-evictable
        outbox growing with nothing draining it until someone restarts the bot.
        So: log loudly the moment it happens, then restart the task with
        exponential damping rather than staying dead.
        """
        if self._stopping or task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return  # _run never returns normally; defensive
        log.error(
            f"history outbox drainer died unexpectedly "
            f"({type(exc).__name__}: {exc}); restarting in "
            f"{self._restart_delay:.0f}s",
            exc_info=exc,
        )
        self._respawn_handle = asyncio.get_running_loop().call_later(
            self._restart_delay, self._respawn
        )
        self._restart_delay = min(self._restart_delay * 2, self.RESTART_MAX)

    def _respawn(self) -> None:
        self._respawn_handle = None
        if not self._stopping:
            self._spawn()

    def notify(self) -> None:
        """Signal a fresh outbox push — cheap, sync, callable from anywhere."""
        self._wake.set()

    async def stop(self, timeout: float = 5.0) -> None:
        """Cancel the loop, then make one bounded final-drain attempt so a
        clean shutdown ships whatever a healthy Postgres can take. Never
        raises — anything left simply stays in the outbox for next start.

        Reentrancy is a correctness property, not politeness: two concurrent
        stop()s each ran their own peek → insert → retire, and the second
        retired entries the first had not inserted (reproduced: 6 pushed, 3
        archived, outbox empty). discord.py calls close() from run()'s finally
        as well as on demand, so a second call is an ordinary event. The lock
        makes the second caller *wait* for the first to finish rather than
        return early into a still-draining shutdown.
        """
        async with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stopping = True
            if self._respawn_handle is not None:
                self._respawn_handle.cancel()
                self._respawn_handle = None
            task = self._task
            self._task = None
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    # The task can already be FINISHED-with-exception rather
                    # than cancellable: _on_task_done leaves self._task pointing
                    # at the failed task while a respawn is pending, so awaiting
                    # it re-raises whatever killed it. Catching only
                    # CancelledError let that escape stop(), which skipped the
                    # final drain AND the lease release — the successor in a
                    # rolling deploy then idled out the full 90s TTL.
                    log.warning(f"history drainer task ended in error: {e}")
            try:
                async with asyncio.timeout(timeout):
                    # Same gate as the loop: if another instance owns the drain
                    # right, our final flush would be exactly the concurrent
                    # second drainer the lease exists to prevent.
                    if await hold_drainer_lease(self._redis, self._lease_id):
                        while await self._drain_once():
                            pass
            except Exception as e:
                log.warning(f"history outbox final drain incomplete: {e}")
            try:
                await release_drainer_lease(self._redis, self._lease_id)
            except Exception as e:
                # Nothing to do about it: the lease expires on its own.
                log.warning(f"history drainer lease release failed: {e}")

    # ── Drain loop ───────────────────────────────────────────────────────────

    async def _run(self) -> None:
        backoff = self._BACKOFF_START
        while True:
            held = False
            try:
                held = await hold_drainer_lease(self._redis, self._lease_id)
                if not held:
                    # Another instance owns the outbox (rolling deploy, stray
                    # process). Idle rather than compete; the holder's lease
                    # lapses within DRAINER_LEASE_MS if it dies.
                    await asyncio.sleep(self.TICK_SECS)
                    continue
                drained = await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The cap MUST also be enforced here, not only on the success
                # tail of _drain_batch. _enforce_cap there is unreachable for
                # the entire duration of a Postgres outage — every failure
                # raises before it — so a cap evaluated only on cycles that
                # delivered is a cap that never fires while the backlog is
                # actually growing, which is the one scenario it was written
                # for (config.HISTORY_OUTBOX_MAX). Depth is an O(1) LLEN and
                # needs no Postgres. Gated on `held` so a cycle that never won
                # the lease cannot trim entries the real holder is mid-insert on.
                if held:
                    await self._enforce_cap_quietly()
                await self._log_retry(e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._BACKOFF_MAX)
                continue
            backoff = self._BACKOFF_START
            if drained:
                # A cycle that actually delivered proves the drainer is healthy,
                # so the respawn damping starts over from the base delay.
                self._restart_delay = self.RESTART_BASE
                continue  # backlog: keep draining without waiting
            # Idle. clear() only ever runs after wait() returns, so a push
            # racing this window costs at most a TICK_SECS delay, never a lost
            # entry: a notify() landing between wait() resolving and clear()
            # IS dropped, but the next iteration drains unconditionally, so the
            # entry is picked up either by that pass or by the timeout wake.
            try:
                async with asyncio.timeout(self.TICK_SECS):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()

    async def _drain_once(self) -> int:
        """One batch, under the whole-cycle deadline. Returns entries retired."""
        async with asyncio.timeout(self.DRAIN_DEADLINE_SECS):
            return await self._drain_batch()

    async def _drain_batch(self) -> int:
        """Peek oldest, insert, retire.

        No poison taxonomy. Every entry that reaches this outbox was produced by
        HistoryEntry.__post_init__, which clamps into the play_history column
        domain, so a batch cannot contain a row Postgres refuses on the grounds
        of its DATA (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md §3.2, §5). A refusal
        here therefore means a validator regression or schema drift — a bug —
        and is handled as one: recorded to play_history_rejected, dropped, batch
        continues.

        Corrupt entries — bytes that do not parse at all — are still dropped and
        still retired, exactly as before. Leaving them would wedge the queue head
        forever.
        """
        raw = await peek_outbox_oldest(self._redis, self.BATCH_SIZE)
        if not raw:
            return 0
        with _tracer.start_as_current_span("history.drain") as span:
            span.set_attribute("drain.batch", len(raw))
            entries = [e for e in map(parse_history_entry, raw) if e is not None]
            span.set_attribute("drain.parsed", len(entries))
            try:
                if entries:
                    await self._archive.insert_batch(entries)
            except _POISON:
                # _isolate retires each entry as it settles it, so the batch
                # retire in the else-branch must NOT also run — it would pop
                # entries that arrived after this batch was peeked. Expressed as
                # try/except/else rather than the `quarantined` local this
                # replaces, so the two retire paths are structurally exclusive.
                await self._isolate(raw, span)
            else:
                await retire_outbox(self._redis, len(raw))
            await self._enforce_cap()
            return len(raw)

    async def _isolate(self, raw: list[bytes], span: Span) -> None:
        """One batch refused: retry it row by row so one bad row costs one row,
        not the 99 batched with it.

        Expected to be dead code. It exists because the alternative — dropping
        the whole batch on a validator regression — turns a one-row bug into a
        hundred-play loss, and a regression is exactly when you most want the
        other 99 saved.

        Takes RAW BYTES, not parsed entries, and retires EVERY element including
        the ones that do not parse. Both properties are load-bearing and both
        were learned the hard way by the _quarantine this replaces:

        - Retire per entry, never once at the end. A transient error partway
          through raises before an end-of-batch retire would run, redelivering
          rows that were already recorded as rejected (reproduced on the DLQ:
          one poison entry, one blip, two copies). The destination is now a
          Postgres table rather than a Redis list, which changes where the
          duplicates land, not whether they happen. Per entry also makes the
          pass RESUMABLE, which matters because single inserts cost ~22x a batch
          (measured: 15.7ms for executemany(100) vs 343ms for 100 singles), so
          on a degraded server this can exceed DRAIN_DEADLINE_SECS and be
          cancelled.
        - Iterate raw, not entries. A corrupt element parses to None and is
          absent from `entries`; retiring only what parsed would leave it at the
          head of the outbox forever, which is the wedge this path exists to
          prevent.

        Unlike _quarantine there is no head-failure counter, no QUARANTINE_AFTER
        threshold and no Redis DLQ — those existed to tell transient from poison,
        and a transient error still simply raises out of here, leaving the
        current entry and everything after it to redeliver.
        """
        rejected = 0
        trace_id = trace_id_of(trace.get_current_span())
        for wire in raw:
            entry = parse_history_entry(wire)
            if entry is not None:
                try:
                    await self._archive.insert_batch([entry])
                except _POISON as e:
                    await self._archive.record_rejection(entry, e, trace_id)
                    rejected += 1
                    log.error(
                        f"play_history refused a row ({type(e).__name__}) — the "
                        f"HistoryEntry validator regressed or the schema "
                        f"drifted: {entry.title[:60]!r} / "
                        f"{entry.webpage_url[:80]}"
                    )
            # else: corrupt, dropped as it always was — but still retired below.
            await retire_outbox(self._redis, 1)
        span.set_attribute("drain.rejected", rejected)

    async def _enforce_cap(self) -> None:
        """Opt-in outbox ceiling (config.HISTORY_OUTBOX_MAX, default off).

        Dropping un-archived plays is data loss, full stop — which is why the
        default is unbounded and why every drop is an ERROR. It exists for
        operators who would rather lose the oldest history than have a long
        Postgres outage push a non-evictable list into Redis' maxmemory, where
        it stops *every* write in the process, not just history's.
        """
        if not self.OUTBOX_MAX:
            return
        depth = await outbox_depth(self._redis)
        if depth <= self.OUTBOX_MAX:
            return
        dropped = depth - self.OUTBOX_MAX
        await trim_outbox_oldest(self._redis, dropped)
        log.error(
            f"history outbox over cap (depth={depth}, HISTORY_OUTBOX_MAX="
            f"{self.OUTBOX_MAX}); dropped {dropped} oldest entries — those "
            f"plays are lost and will not reach Postgres"
        )

    async def _enforce_cap_quietly(self) -> None:
        """_enforce_cap for the failure path, where Redis may itself be what
        broke. Never raises: the caller is already handling one error, and
        losing its backoff to a second one would turn a Redis blip into a hot
        retry loop."""
        try:
            await self._enforce_cap()
        except Exception as e:
            log.warning(f"history outbox cap check failed: {e}")

    async def _log_retry(self, error: Exception, backoff: float) -> None:
        try:
            depth = await outbox_depth(self._redis)
        except Exception:
            depth = -1  # Redis itself is down; depth unknowable
        emit = log.error if depth >= self.DEPTH_ALARM else log.warning
        emit(
            f"history outbox drain failed (backlog={depth}): "
            f"{type(error).__name__}: {error}; retrying in {backoff:.0f}s"
        )
