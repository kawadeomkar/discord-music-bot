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
                   it forever → _sanitize_entry (prevention) + _quarantine
                   (per-entry isolation + DLQ)
  hung server      a connected-but-unresponsive Postgres never returns, so
                   there is no exception, no backoff, and no alarm →
                   command_timeout + DRAIN_DEADLINE_SECS
  two drainers     peek/retire is single-consumer; a second one retires
                   entries the first never inserted → the Redis drainer lease
  dead drainer     the task dies and nothing notices until a restart →
                   _on_task_done respawn with exponential damping
"""

import asyncio
import dataclasses
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional, Protocol
from uuid import uuid4

import asyncpg
import redis.asyncio as aioredis
from opentelemetry.trace import Span

from src import config
from src.db_migrate import EXPECTED_SCHEMA_VERSION
from src.guild_state import HistoryEntry, parse_history_entry
from src.redis_client import (
    dead_letter_outbox,
    hold_drainer_lease,
    outbox_depth,
    peek_outbox_oldest,
    release_drainer_lease,
    retire_outbox,
    trim_outbox_oldest,
)
from src.telemetry import get_tracer
from src.util import get_logger

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

_SCHEMA_VERSION_SQL = "SELECT max(version) FROM schema_migrations"

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

# Clamp bounds for _sanitize_entry.
_TS_MAX = 253402300799.0  # 9999-12-31T23:59:59Z — the timestamptz ceiling
_INT4_MAX = 2**31 - 1
_INT8_MAX = 2**63 - 1
# Columns declared `integer` in migrations/0001_play_history.sql; everything
# else numeric is bigint.
_INT4_FIELDS = frozenset({"duration_secs", "played_secs"})
# Hoisted: dataclasses.fields() builds a fresh tuple on every call, and this
# runs per entry on the drain path (measured 2.64us of a 7.49us sanitize).
# HistoryEntry is frozen+slots, so the field list cannot change at runtime.
_ENTRY_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(HistoryEntry))


def _sanitize_entry(entry: HistoryEntry) -> HistoryEntry:
    """Clamp or strip anything Postgres (or datetime) would reject, so that
    every *parseable* entry is also insertable.

    The three confirmed poison vectors, all of which pass parse_history_entry
    and then fail at or before the INSERT:

      NUL in text      Postgres text cannot hold \\x00 — the server raises
                       CharacterNotInRepertoireError, which fails the whole
                       executemany batch and every healthy entry with it.
      huge epoch       datetime.fromtimestamp raises OverflowError in
                       _entry_to_row, before any SQL is sent.
      negative epoch   likewise ValueError on some platforms; and a negative
                       duration/played_secs is meaningless anyway.

    Out-of-range played_at collapses to the epoch-0 "unknown" sentinel the wire
    format already uses. The chained comparison is False for NaN, so NaN lands
    there too — which is the point of writing it as a range test rather than
    two explicit bounds checks.

    Returns the *same object* when nothing needs changing: the clean path is
    every real entry, and it should not allocate.
    """
    changes: dict[str, object] = {}
    for name in _ENTRY_FIELD_NAMES:
        value = getattr(entry, name)
        if isinstance(value, str) and "\x00" in value:
            changes[name] = value.replace("\x00", "")
        elif isinstance(value, int) and not isinstance(value, bool):
            ceiling = _INT4_MAX if name in _INT4_FIELDS else _INT8_MAX
            if not 0 <= value <= ceiling:
                changes[name] = min(max(value, 0), ceiling)
    if not 0.0 <= entry.played_at <= _TS_MAX:
        changes["played_at"] = 0.0
    return dataclasses.replace(entry, **changes) if changes else entry


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


class RowConversionError(ValueError):
    """A HistoryEntry that cannot be turned into a play_history row.

    Exists so "this data is unusable" is a TYPE rather than a guess at which
    exception the conversion happened to raise. That guess is not portable:
    `datetime.fromtimestamp(1e18)` raises OverflowError on some platforms and
    `OSError: [Errno 84] Value too large` on macOS — and OSError is exactly
    what a network failure looks like, so a taxonomy that named it would
    dead-letter healthy batches during every Postgres outage. Conversion runs
    before any socket is touched, so wrapping it is unambiguous.
    """


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

        Every entry is sanitized on the way in. That is what makes "parseable"
        and "insertable" the same set, so one bad row can never fail the
        executemany that carries 99 good ones (see _sanitize_entry)."""
        if not entries:
            return
        # Conversion first, and strictly before the pool is touched: it is the
        # only step here that can fail because of the DATA rather than the
        # network, so isolating it is what lets the drainer tell the two apart.
        try:
            rows = [_entry_to_row(_sanitize_entry(e)) for e in entries]
        except Exception as e:
            raise RowConversionError(
                f"history entry cannot be mapped to a play_history row "
                f"({type(e).__name__}: {e})"
            ) from e
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            try:
                await conn.executemany(_INSERT_SQL, rows)
            except TypeError as e:
                # asyncpg's parameter encoders raise TypeError on a value they
                # cannot encode for its column. That is a property of the data,
                # so it is poison — but see _POISON for why it is renamed here
                # instead of being matched by type.
                raise RowConversionError(f"asyncpg could not encode a row: {e}") from e

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        """The `limit` most recent entries for one guild, newest first. id is
        the tie-break so epoch-0 (unknown-time) entries order stably."""
        if limit <= 0:
            return []
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            rows = await conn.fetch(_RECENT_SQL, guild_id, limit)
        return [_row_to_entry(r) for r in rows]

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
#   DataError            server-side rejections: CharacterNotInRepertoireError
#                        for NUL bytes, NumericValueOutOfRangeError, …
#   RowConversionError   the entry could not be turned into a row at all —
#                        raised before any socket is touched, so it can never
#                        be a disguised network failure (see the class).
#   OverflowError        asyncpg's client-side integer codecs on a value that
#                        does not fit the column. Never transient.
#
# Deliberately NOT here:
#   - bare ValueError. asyncpg raises it for whole-statement problems like an
#     argument-count overflow, which is not per-entry poison.
#   - bare TypeError. asyncpg's parameter encoders DO raise it on a value they
#     cannot encode, which is genuine poison — but so does an ordinary bug
#     anywhere inside insert_batch, and naming it here would dead-letter a
#     healthy batch on a refactor. insert_batch converts the encoder case into
#     RowConversionError at the one call that can produce it, so the taxonomy
#     keys on the code path rather than on an exception type shared with bugs.
#   - anything OSError-shaped. InterfaceError, ConnectionResetError and
#     TimeoutError are how a Postgres restart or failover presents, and
#     dead-lettering on those would delete healthy history on every outage.
_POISON = (
    asyncpg.exceptions.DataError,
    RowConversionError,
    OverflowError,
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
    # Consecutive failures of the SAME head batch before the drainer stops
    # trusting the exception taxonomy and isolates the batch entry-by-entry
    # anyway. At 8 that is 1+2+4+8+16+32+60 = ~123s of cumulative backoff, which
    # an ordinary Postgres restart CAN exceed — so the common outcome of a slow
    # restart is one forced entry-by-entry pass. That pass is benign: every
    # entry inserts, nothing is dead-lettered, and the counter resets. Raising
    # this would trade that harmless cost for a longer window in which an
    # unforeseen poison shape wedges the outbox, which is the worse failure.
    QUARANTINE_AFTER = 8
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
        # Poison catch-all state: which head batch keeps failing, and how often.
        self._head_sig: Optional[int] = None
        self._head_fails = 0
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
        """Peek oldest, insert, retire. Corrupt entries are dropped
        (parse_history_entry warns per entry) but still retired — leaving them
        would wedge the queue head forever, which is the same failure the
        quarantine path below exists to prevent for entries that parse but
        that Postgres rejects."""
        raw = await peek_outbox_oldest(self._redis, self.BATCH_SIZE)
        if not raw:
            self._reset_head_failures()
            return 0
        with _tracer.start_as_current_span("history.drain") as span:
            span.set_attribute("drain.batch", len(raw))
            entries = [e for e in map(parse_history_entry, raw) if e is not None]
            span.set_attribute("drain.parsed", len(entries))
            # The catch-all: a batch that has failed QUARANTINE_AFTER times in
            # a row is isolated entry-by-entry even though its exception type
            # says "transient". Covers poison shapes _POISON does not name.
            forced = self._head_fails >= self.QUARANTINE_AFTER
            span.set_attribute("drain.forced_quarantine", forced)
            quarantined = False
            try:
                if entries and not forced:
                    await self._archive.insert_batch(entries)
            except _POISON:
                await self._quarantine(raw, span)
                quarantined = True
            except Exception:
                self._note_head_failure(raw[0])
                raise  # unchanged: backoff in _run
            else:
                if forced:
                    await self._quarantine(raw, span)
                    quarantined = True
            # _quarantine retires each entry as it settles it, so retiring the
            # batch again here would pop entries that arrived since.
            if not quarantined:
                await retire_outbox(self._redis, len(raw))
            self._reset_head_failures()
            await self._enforce_cap()
            return len(raw)

    async def _quarantine(self, raw: list[bytes], span: Optional[Span] = None) -> None:
        """Per-entry isolation: good rows land, individually-failing rows go to
        the DLQ. This is the whole point of C1 — one entry Postgres will never
        accept must cost one entry, not the entire batch behind it and every
        guild's archiving with it.

        Each entry is RETIRED as soon as it is settled — inserted, dead-lettered
        or dropped as corrupt — rather than retiring the batch at the end. Two
        things follow, both of which were defects before:

        - No DLQ duplication. A transient error partway through used to raise
          before the single end-of-batch retire, redelivering entries that had
          already been dead-lettered (reproduced: one poison entry, one blip,
          two DLQ copies). The DLQ carries no TTL and is non-evictable, so a
          poison entry plus a flapping Postgres grew Redis without bound.
        - No wedge. This path costs ~22x a batched insert (measured: 15.7ms for
          executemany(100) vs 343ms for 100 single inserts), so on a degraded
          server it can exceed DRAIN_DEADLINE_SECS. Retiring as we go makes it
          RESUMABLE: the cancelled attempt keeps its progress and the next cycle
          starts from the first unsettled entry, instead of re-running the whole
          batch forever.

        A transient error still raises out, leaving the current entry and
        everything after it on the outbox to redeliver.
        """
        dead = 0
        for wire in raw:
            entry = parse_history_entry(wire)
            if entry is not None:
                try:
                    await self._archive.insert_batch([entry])
                except _POISON as e:
                    await dead_letter_outbox(self._redis, wire)
                    dead += 1
                    log.error(
                        f"history entry quarantined to DLQ ({type(e).__name__}): "
                        f"{entry.title[:60]!r} / {entry.webpage_url[:80]}"
                    )
            # else: corrupt, dropped as it always was — but still retired here.
            await retire_outbox(self._redis, 1)
        if span is not None and dead:
            span.set_attribute("drain.dead_lettered", dead)

    def _note_head_failure(self, head: bytes) -> None:
        """Count consecutive failures of the SAME batch head.

        Keyed on the head rather than being a plain counter because the two
        situations that produce repeated failures look identical from the
        outside: a Postgres outage (head unchanged because nothing drains) and
        an unforeseen poison entry (head unchanged because it cannot drain).
        The head signature at least guarantees the count describes one specific
        stuck batch and resets the moment anything moves — and QUARANTINE_AFTER
        is set well past a normal restart or failover so the outage case
        recovers on its own long before the catch-all engages.
        """
        sig = hash(head)
        if sig == self._head_sig:
            self._head_fails += 1
        else:
            self._head_sig, self._head_fails = sig, 1

    def _reset_head_failures(self) -> None:
        self._head_sig, self._head_fails = None, 0

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
