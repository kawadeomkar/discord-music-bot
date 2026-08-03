"""
Postgres play-history archive — the durable long-term home for every played song.

- HistoryArchive — the protocol the drainer and the backfill tool program against.
- ArchiveReader — the read side MusicBot holds: -ping's liveness probe and
  -leaderboard's aggregate. Deliberately separate, so write-surface fakes do not
  grow a read method they would never call.
- PostgresHistoryArchive — asyncpg. Connects lazily so startup never blocks on
  Postgres; applies no DDL (migrations/ owns the schema, _ensure verifies it).
- HistoryOutboxDrainer — Redis outbox STREAM → Postgres: replay this consumer's
  pending IDs → read new → INSERT ... ON CONFLICT DO NOTHING → XACK+XDEL by ID.
  At-least-once: a crash between insert and ack redelivers and play_history_dedup
  collapses the replay. The playback loop never awaits Postgres. One task per
  process but not one per deployment — the consumer group makes concurrent
  drainers safe with no lease to win (`>` gives them disjoint entries; the
  pending replay a shared set ON CONFLICT collapses).

Row mapping (HistoryEntry ↔ play_history row) lives here, not in guild_state.py:
that module's contract is pure wire schema with no runtime imports.

The outbox is non-evictable, so anything that stalls the drain grows a Redis key
that eventually refuses every write in the process. Each guard closes one door:

  poison entry  blocks the batch behind it forever → prevented at construction
                (HistoryEntry.__post_init__ clamps into the column domain);
                _isolate parks what still will not insert in play_history_rejected
  hung server   a connected-but-unresponsive Postgres never returns, so there is
                no exception and no alarm → command_timeout + DRAIN_DEADLINE_SECS
  two drainers  structural — XACK settles only the IDs this process archived
  tombstone     body deleted while pending: the ID replays with no payload and
                raises out of every handler → _settle_tombstones acks and logs it
  stranded PEL  entries under a consumer name nothing reads → XAUTOCLAIM sweep
  dead drainer  the task dies unnoticed → _on_task_done respawn with damping

See docs/ARCHITECTURE.md#history-archive-tier.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, cast

import asyncpg
import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import Span
from redis.exceptions import ResponseError

from src import config
from src.db_migrate import EXPECTED_SCHEMA_VERSION
from src.guild_state import HistoryEntry, parse_history_entry, serialize_history_entry
from src.redis_client import (
    HISTORY_OUTBOX_KEY,
    OutboxEntry,
    ensure_outbox_group,
    outbox_depth,
    ack_outbox,
    outbox_pending_below,
    outbox_pending_count,
    read_outbox_new,
    read_outbox_pending,
    reclaim_outbox_stale,
    retire_outbox,
    trim_outbox_below,
)
from src.telemetry import get_tracer
from src.util import get_logger, trace_id_of

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_INSERT_SQL = """
INSERT INTO play_history (guild_id, title, webpage_url, duration_secs,
                          played_secs, requester_id, requester_name,
                          thumbnail, uploader, played_at, message_id,
                          queued_at, queue_position)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
ON CONFLICT (guild_id, played_at, webpage_url) DO NOTHING
"""

_RECENT_SQL = """
SELECT guild_id, title, webpage_url, duration_secs, played_secs,
       requester_id, requester_name, thumbnail, uploader, played_at, message_id,
       queued_at, queue_position
FROM play_history
WHERE guild_id = $1
ORDER BY played_at DESC, id DESC
LIMIT $2
"""

# ON CONFLICT against play_history_rejected_dedup makes this exactly-once: two
# drainers can replay the same pending entry and both fail it. The table is
# expected to stay empty forever and `just db-rejects` reports its contents, so
# "3 rows" must read as three distinct failures, not one seen three times.
_REJECT_SQL = """
INSERT INTO play_history_rejected (guild_id, error_type, error_detail, trace_id, payload)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT ON CONSTRAINT play_history_rejected_dedup DO NOTHING
"""

# The -leaderboard aggregates. Sentinel groups are excluded rather than shown:
# requester_id 0 and webpage_url '' both mean "unknown" and would each merge
# unrelated plays into one top-10 row. sum() over integer returns bigint, so
# totals cannot overflow int4. $3 is the period cutoff: to_timestamp(0) for
# all-time — the wire format's floor, so an inclusive compare excludes nothing
# — while any real cutoff also excludes the epoch-0 unknown-time rows.
#
# Two passes, not one. The display name/title is the most recent one recorded
# (titles drift; the id is stable identity), but taking it inline with
# `(array_agg(x ORDER BY played_at DESC, id DESC))[1]` makes it an ORDERED
# aggregate, and an ordered aggregate removes hash aggregation from the
# planner's options entirely. Both boards then plan as GroupAggregate over a
# full sort of every matching row, and array_agg's state is not work_mem-bounded
# and cannot spill — measured at 3M rows: 6.8s, 140MB of external merge, and
# 431MB RSS in one backend for a single large group. Aggregating first and
# resolving the ten winners through LATERAL keeps it a HashAggregate: same rows,
# same order, no temp files, and no per-group state.
#
# The LATERAL leg rides play_history_recent and filters, so it walks back to each
# winner's newest play: cheap for a song still in rotation, proportional to the
# guild's history for one that ranks on old plays alone. Measured at 300k rows,
# 53ms typical against 111ms for that worst case — both against 880ms before. A
# (guild_id, webpage_url, played_at DESC, id DESC) index would make it an exact
# seek, at write amplification on an append-only table; not yet worth it.
# See docs/ARCHITECTURE.md#history-archive-tier.
_TOP_REQUESTERS_SQL = """
WITH top AS (
    SELECT requester_id,
           count(*)         AS plays,
           sum(played_secs) AS played_secs
    FROM play_history
    WHERE guild_id = $1 AND requester_id > 0 AND played_at >= $3
    GROUP BY requester_id
    ORDER BY played_secs DESC, plays DESC, requester_id
    LIMIT $2
)
SELECT t.requester_id, l.requester_name, t.plays, t.played_secs
FROM top t
CROSS JOIN LATERAL (
    SELECT p.requester_name
    FROM play_history p
    WHERE p.guild_id = $1 AND p.requester_id = t.requester_id
    ORDER BY p.played_at DESC, p.id DESC
    LIMIT 1
) l
ORDER BY t.played_secs DESC, t.plays DESC, t.requester_id
"""

_TOP_SONGS_SQL = """
WITH top AS (
    SELECT webpage_url,
           count(*)         AS plays,
           sum(played_secs) AS played_secs
    FROM play_history
    WHERE guild_id = $1 AND webpage_url <> '' AND played_at >= $3
    GROUP BY webpage_url
    ORDER BY played_secs DESC, plays DESC, webpage_url
    LIMIT $2
)
SELECT t.webpage_url, l.title, l.duration_secs, t.plays, t.played_secs
FROM top t
CROSS JOIN LATERAL (
    SELECT p.title, p.duration_secs
    FROM play_history p
    WHERE p.guild_id = $1 AND p.webpage_url = t.webpage_url
    ORDER BY p.played_at DESC, p.id DESC
    LIMIT 1
) l
ORDER BY t.played_secs DESC, t.plays DESC, t.webpage_url
"""

_SCHEMA_VERSION_SQL = "SELECT max(version) FROM schema_migrations"

# Cap on the asyncpg message in play_history_rejected.error_detail: enough for
# the SQLSTATE and the offending value, short enough not to bloat an empty table.
_REJECT_DETAIL_MAX = 2000

# Per-statement bound. Covers what timeout=10 does not: a connection that
# established fine against a server that then stops answering. A liveness bound,
# not a latency target — a 100-row executemany is milliseconds.
_COMMAND_TIMEOUT_SECS = 30.0
# Wait for a free connection. The drainer, -ping's health probe and archive reads
# share max_size=4, so a stuck consumer must not block everyone else unboundedly.
_ACQUIRE_TIMEOUT_SECS = 10.0
# Concurrent -leaderboard reads, against that same max_size=4. Reads are the only
# user-triggered traffic on this pool and are unbounded across guilds
# (max_concurrency serializes per guild, not globally), so without a cap a burst
# takes every connection and the drainer's acquire starts timing out — measured:
# 64 concurrent boards pushed insert_batch from 11.8ms to a 10s TimeoutError and
# into backoff. Two leaves two, one for the drainer and one for -ping.
_READ_CONCURRENCY = 2
# Whole-operation bound for a read, covering the wait for a slot. Deliberately
# well under command_timeout: a leaderboard that cannot answer in this long is
# better failed than left holding a connection the drain needs.
_READ_DEADLINE_SECS = 15.0
# Graceful pool shutdown before terminate(). Short: this runs on the shutdown
# path, ahead of the Redis pool, discord.py's close and the span flush.
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
        entry.message_id,
        datetime.fromtimestamp(entry.queued_at, tz=timezone.utc),
        entry.queue_position,
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
        message_id=row["message_id"],
        queued_at=row["queued_at"].timestamp(),
        queue_position=row["queue_position"],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RequesterLeader:
    """One row of the -leaderboard listeners board. requester_name is the most
    recent one recorded for that id, so a rename shows the current name."""

    requester_id: int
    requester_name: str
    plays: int
    played_secs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SongLeader:
    """One row of the -leaderboard songs board, grouped by webpage_url. title
    and duration_secs are the most recent values seen for that URL."""

    title: str
    webpage_url: str
    duration_secs: int
    plays: int
    played_secs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Leaderboard:
    requesters: list[RequesterLeader]
    songs: list[SongLeader]


class ArchiveReader(Protocol):
    """What MusicBot needs from the archive: liveness for -ping's Postgres row
    and the aggregate behind -leaderboard. Structural, like ping's ArchiveHealth
    (which it satisfies), so the cog stays fake-able in tests."""

    async def health_check(self) -> None: ...

    async def leaderboard(
        self, guild_id: int, limit: int, *, since_epoch: float = 0.0
    ) -> Leaderboard: ...


class HistoryArchive(Protocol):
    """The archive surface the drainer (writes) and the backfill tool program
    against — faked in unit tests, implemented by asyncpg below. recent() reads
    the durable record; -history does not use it (see
    docs/ARCHITECTURE.md#history-read-path)."""

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None: ...

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]: ...

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None: ...


class SchemaVersionError(RuntimeError):
    """The database's schema is older than the code expects.

    Its own type so callers can tell "run the migrations" (an operator action)
    apart from "Postgres is down" (a wait).
    """


class PostgresHistoryArchive:
    """asyncpg-backed archive. All methods raise on failure — callers own the
    error policy (the drainer backs off, the backfill counts the guild failed)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()
        self._closed = False
        # Per instance, not module-level: one archive owns one pool, and a
        # shared counter would leak across tests that build several.
        self._read_slots = asyncio.Semaphore(_READ_CONCURRENCY)

    async def _create_pool(self) -> asyncpg.Pool:
        return await asyncpg.create_pool(
            self._url,
            min_size=1,
            max_size=4,
            # Connect bound: a fast connect failure keeps the drainer's backoff
            # loop responsive (asyncpg's default is 60s).
            timeout=10,
            # Statement bound. Without it a server that accepts the connection
            # then stops answering hangs executemany forever — no exception, so
            # no backoff, no DEPTH_ALARM and not one log line while the outbox
            # grows (the "hung server" guard in the module docstring).
            command_timeout=_COMMAND_TIMEOUT_SECS,
            # Prepared statements are per-connection, so PgBouncer in
            # transaction-pooling mode breaks them; POSTGRES_STATEMENT_CACHE=0
            # turns the cache off for that shape. Default matches asyncpg's own.
            statement_cache_size=config.POSTGRES_STATEMENT_CACHE,
            # Identifies this bot's connections in pg_stat_activity.
            server_settings={"application_name": "musicbot-history"},
        )

    async def _ensure(self) -> asyncpg.Pool:
        """Lazy pool + schema-version check, double-checked under the lock.
        First successful call wins; a failed attempt leaves no half-open pool.

        Refuses after close() so a late caller cannot resurrect a pool nothing
        will ever close again. _closed is re-checked under the lock and after
        the awaits — close() can win any of those suspension points — and every
        escape routes through the BaseException handler, so a just-built pool is
        always closed by whoever built it.
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
                    # Covers the version check, cancellation and the re-check
                    # above — all three would leak a pool self._pool never
                    # received and close() cannot see.
                    await pool.close()
                    raise
                self._pool = pool
        return self._pool

    @staticmethod
    async def _assert_schema_version(conn: Any) -> None:
        """Verify the database carries the schema this build was written for.

        A *newer* database is tolerated with a warning rather than refused:
        migrations are additive, so an older bot reads a newer schema fine and
        refusing would turn a rolled-back deploy into an outage. `conn` is Any
        because the pool hands out a PoolConnectionProxy, not the
        asyncpg.Connection asyncpg.connect() returns — duck-identical here.
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
        field into this table's column domain — the schema lock — so
        _entry_to_row cannot hand asyncpg a value it will not encode."""
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

    async def leaderboard(
        self, guild_id: int, limit: int, *, since_epoch: float = 0.0
    ) -> Leaderboard:
        """Top requesters and songs for one guild, ranked by total played_secs.

        since_epoch 0.0 = all-time; epoch-0 unknown-time rows appear only there,
        since any real cutoff excludes them by definition. Sentinel groups
        (requester 0, url '') are excluded — see the SQL constants.

        Bounded twice, because this is the pool's only user-triggered reader and
        the drainer shares it. _read_slots keeps reads off the last connections
        so a burst of commands cannot starve the writer, and the deadline covers
        waiting for a slot as well as the queries — two statements on one
        connection are otherwise bounded only by 2 x command_timeout, longer
        than the drainer's whole DRAIN_DEADLINE_SECS.
        """
        if limit <= 0:
            return Leaderboard(requesters=[], songs=[])
        cutoff = datetime.fromtimestamp(max(0.0, since_epoch), tz=timezone.utc)
        async with asyncio.timeout(_READ_DEADLINE_SECS), self._read_slots:
            pool = await self._ensure()
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                requester_rows = await conn.fetch(
                    _TOP_REQUESTERS_SQL, guild_id, limit, cutoff
                )
                song_rows = await conn.fetch(_TOP_SONGS_SQL, guild_id, limit, cutoff)
        return Leaderboard(
            requesters=[
                RequesterLeader(
                    requester_id=r["requester_id"],
                    requester_name=r["requester_name"],
                    plays=r["plays"],
                    played_secs=r["played_secs"],
                )
                for r in requester_rows
            ],
            songs=[
                SongLeader(
                    title=r["title"],
                    webpage_url=r["webpage_url"],
                    duration_secs=r["duration_secs"],
                    plays=r["plays"],
                    played_secs=r["played_secs"],
                )
                for r in song_rows
            ],
        )

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None:
        """Park one refused play in play_history_rejected. Best-effort and
        TERMINAL: never retries, never recurses — a rejects table that can fail
        into a retry loop is worse than none, so a failed insert goes to the log
        and the caller moves on. Only reachable on a REJECTION, which means the
        server is up; an outage leaves the entry on the outbox to redeliver.

        error_detail is text and asyncpg messages echo the offending value, so it
        is NUL-scrubbed and capped: the same poison this table exists to record
        would otherwise fail the insert recording it. payload is bytea and takes
        the delivered `wire` bytes verbatim, never a re-serialized entry — in a
        mixed-version rollout an entry written by a NEWER build carries fields
        this parser drops, so the record would be a lossy re-encoding, and since
        the dedup identity is an md5 of payload, one entry seen by two builds
        lands as two rows in a table whose row count must be a failure count.
        Falls back to re-serializing when the caller has no wire bytes.
        """
        detail = str(error).replace("\x00", "")[:_REJECT_DETAIL_MAX]
        # serialize_history_entry cannot raise: `entry` was constructed, so
        # __post_init__ already proved it orjson-encodable.
        payload = wire if wire is not None else serialize_history_entry(entry)
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
        """Prove the archive's database is reachable and answering. Raises on any
        failure — -ping's probe_postgres times it and turns that into a red row.

        Connects if nothing has yet: the pool is lazy, so before the first song
        end it is simply absent and reporting "not configured" for an enabled
        tier would be a lie an operator acts on. _ensure's timeout=10 outlasts
        PING_DEADLINE_SECS (3s), so the first -ping of a cold bot can render one
        red row for a healthy-but-slow connect; it self-corrects next tick, and
        widening the deadline would delay every OTHER row's first render.

        SELECT 1 rather than a table read: this asks "is the server answering",
        not "is the schema right" — _ensure's version check settled that.
        """
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            await conn.execute("SELECT 1")

    async def close(self) -> None:
        """Close the pool. Terminal: _ensure() refuses afterwards.

        _closed is set first so new callers are turned away immediately, then the
        init lock is taken so an _ensure() already inside its connect cannot hand
        a fresh pool to a field nobody will read again. That wait can cost up to
        the 10s connect timeout, which is the price of not leaking a live pool.

        Keep MusicBotApp.close()'s order — drainer.stop() strictly before this —
        so the final drain can still reach Postgres. It no longer implies
        exclusivity: health_check() gave -ping a caller shutdown cannot sequence.
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
                # A graceful close waits for in-flight queries, which a hung
                # server never allows: measured at 30s, and it RAISED — aborting
                # every remaining step of MusicBotApp.close() (Redis pool open,
                # discord.py unclosed, yt-dlp left to its 61s atexit join, no
                # spans flushed). terminate() is synchronous and unconditional;
                # closing an archive must not take shutdown down with it.
                pool.terminate()
                log.warning(
                    f"history archive pool close forced: {type(e).__name__}: {e}"
                )


# Errors meaning "this data will never be accepted", not "try again later".
# Expected UNREACHABLE since the schema lock (HistoryEntry.__post_init__); kept
# as the backstop for a validator regression or an unexpected schema, where
# dropping a whole batch would turn a one-row bug into a hundred-play loss.
#
#   DataError             SQLSTATE 22xxx server-side data rejections
#                         (CharacterNotInRepertoireError for NUL bytes,
#                         NumericValueOutOfRangeError, …)
#   CheckViolationError   23514, and not a DataError — both inherit from
#   NotNullViolationError IntegrityConstraintViolationError; without both arms a
#                         CHECK violation wedges the drain head permanently
#
# Deliberately not here — each would break the drain:
#   - UniqueViolationError: play_history_dedup is the ON CONFLICT target, so it
#     cannot surface; catching it hides a genuine index bug.
#   - bare ValueError / TypeError: asyncpg raises them for whole-statement
#     problems and unencodable values, but so does an ordinary bug anywhere in
#     insert_batch — a refactor would dead-letter a healthy batch.
#   - anything OSError-shaped (InterfaceError, ConnectionResetError,
#     TimeoutError): that is how a restart or failover presents, so it would
#     delete healthy history on every outage.
_POISON = (
    asyncpg.exceptions.DataError,
    asyncpg.exceptions.CheckViolationError,
    asyncpg.exceptions.NotNullViolationError,
)


class HistoryOutboxDrainer:
    """The one task per process that drains the Redis outbox into the archive.

    Wakes on notify() (set by every outbox push) with a periodic fallback tick,
    drains in batches until the outbox is empty, and on archive/Redis failure
    backs off exponentially while entries accumulate safely in the outbox
    (persistent, non-evictable — see HISTORY_OUTBOX_KEY).

    Not single-consumer, and does not need to be: the outbox is a stream consumer
    group under a stable name, so a second instance (an overlapping k8s rollout,
    a developer's bot on a shared REDIS_URL) reads disjoint new entries and
    shares the pending set, which the archive's unique index collapses.
    Exactly-once rejection recording comes from _REJECT_SQL's ON CONFLICT.
    """

    BATCH_SIZE: int = 100
    TICK_SECS: float = 30.0
    DEPTH_ALARM: int = 10_000  # backlog that escalates the retry warning to ERROR
    # Whole-cycle bound, belt to command_timeout's braces: covers connection
    # acquisition, DNS re-resolution and anything else asyncpg does not bound, so
    # any hang becomes a TimeoutError on _run's normal error path — which is what
    # makes DEPTH_ALARM fire for hangs and not only for errors.
    DRAIN_DEADLINE_SECS: float = 60.0
    # Rate limit for the depth watchdog on the productive path. Matched to
    # TICK_SECS so a busy drain reports a growing backlog on an idle one's cadence.
    DEPTH_SAMPLE_INTERVAL_SECS: float = 30.0
    # PEL sweep (reclaim_outbox_stale). Much slower than TICK_SECS: everything it
    # catches is rare by construction, and it costs an XAUTOCLAIM scan.
    SWEEP_INTERVAL_SECS: float = 300.0
    # INVARIANT: must exceed DRAIN_DEADLINE_SECS * 1000. Under a shared consumer
    # name "idle" is measured from last delivery, so a smaller value lets the
    # sweep reclaim a live sibling's batch while it is still inserting.
    SWEEP_MIN_IDLE_MS: int = 300_000
    # Bounds one sweep's work, and terminates the cursor loop under fakeredis,
    # which returns the last-scanned ID where real Redis returns "0-0".
    SWEEP_MAX_PASSES: int = 20
    # Respawn damping for a drainer task that dies outside its own error handling.
    # Short base because the first restart usually works; the cap keeps a
    # hard-broken drainer from becoming a log flood.
    RESTART_BASE: float = 5.0
    RESTART_MAX: float = 300.0
    # 0 = unbounded, the durability default. See config.HISTORY_OUTBOX_MAX.
    OUTBOX_MAX: int = config.HISTORY_OUTBOX_MAX
    # Bounds one _enforce_cap pass's XRANGE. minid discovery needs the ID of the
    # (page+1)-th oldest entry and XRANGE has no ID-only form, so the reply
    # carries bodies: uncapped, a 500k backlog would haul the entire overage over
    # the socket in one ~240 MB reply. 10k entries ≈ 5 MB at the measured ~487 B.
    CAP_PAGE: int = 10_000
    _BACKOFF_START: float = 1.0
    _BACKOFF_MAX: float = 60.0

    def __init__(self, redis: aioredis.Redis, archive: HistoryArchive) -> None:
        self._redis = redis
        self._archive = archive
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        # Shutdown/supervision state.
        self._stop_lock = asyncio.Lock()
        self._stopped = False
        self._stopping = False
        self._restart_delay = self.RESTART_BASE
        self._respawn_handle: Optional[asyncio.TimerHandle] = None
        # Monotonic deadline for the next PEL sweep. None = due now (start()).
        self._next_sweep: Optional[float] = None
        # Same, for the depth watchdog's productive-path rate limit.
        self._next_depth_sample: Optional[float] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stopping = False
        # _stopped too: stop() latches it forever, so a start-after-stop would
        # spawn a live _run that the next stop() returns early from without
        # cancelling — a leaked task draining after teardown, claiming entries
        # into a PEL nothing acks until the sweep reclaims them.
        self._stopped = False
        self._restart_delay = self.RESTART_BASE
        self._spawn()

    def _spawn(self) -> None:
        self._task = asyncio.create_task(self._run(), name="history-outbox-drainer")
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: "asyncio.Task[None]") -> None:
        """Supervision. _run only ever exits via cancellation, so any exception
        here is a bug — one that leaves the non-evictable outbox growing with
        nothing draining it. Log loudly, then restart with exponential damping
        rather than staying dead.
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
        """Cancel the loop, then make one bounded final-drain attempt so a clean
        shutdown ships whatever a healthy Postgres can take. Never raises —
        anything left simply stays in the outbox for the next start.

        Reentrancy is a correctness property: discord.py calls close() from
        run()'s finally as well as on demand, so a second call is ordinary, and
        the lock makes that caller WAIT for the first rather than return early
        into a still-draining shutdown.

        Under a shared consumer name the final drain's pending replay returns a
        live peer's in-flight batch, so a departing process duplicates the
        survivor's work rather than complementing it — harmless (dedup plus an
        idempotent XACK), but shutdown does more work in the overlap window.
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
                    # CancelledError lets that escape stop() and skip the final
                    # drain below.
                    log.warning(f"history drainer task ended in error: {e}")
            try:
                async with asyncio.timeout(timeout):
                    while await self._drain_once():
                        pass
            except Exception as e:
                log.warning(f"history outbox final drain incomplete: {e}")

    # ── Drain loop ───────────────────────────────────────────────────────────

    async def _run(self) -> None:
        backoff = self._BACKOFF_START
        while True:
            try:
                await self._sweep_if_due()
                drained = await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The cap must also be enforced here, not only on _drain_batch's
                # success tail: that one is unreachable for the whole duration of
                # a Postgres outage — every failure raises before it — so a cap
                # evaluated only on cycles that delivered never fires while the
                # backlog is actually growing, the one scenario it exists for.
                # Depth is an O(1) XLEN, no Postgres.
                #
                # This is the failure arm, precisely the state where a batch is
                # delivered and unacked, and _enforce_cap deliberately trims
                # across the PEL (its docstring says why refusing would make the
                # cap a no-op exactly here). ACCEPTED because settlement covers
                # it: everything the cap destroys is XACKed first, so a trimmed
                # in-flight batch is dropped-and-logged, never a tombstone.
                await self._enforce_cap_quietly()
                await self._log_retry(e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._BACKOFF_MAX)
                continue
            backoff = self._BACKOFF_START
            if drained:
                # A cycle that delivered proves the drainer is healthy, so the
                # respawn damping starts over from the base delay.
                self._restart_delay = self.RESTART_BASE
                # Sample here too, or the alarm cannot fire in the case its own
                # message describes: "keeping up but not catching up" is a
                # succeeding drain, which takes this branch every time — so with
                # the sample only below, a backlog growing faster than BATCH_SIZE
                # per cycle logged nothing at all, forever.
                await self._sample_depth_if_due()
                continue  # backlog: keep draining without waiting
            # Idle — the drain settled everything it can. Sample on the success
            # path: _log_retry only fires in the except arm, so without this a
            # healthy drainer sitting on a growing outbox (or a stranded PEL)
            # says nothing at all.
            await self._sample_depth()
            # clear() only ever runs after wait() returns, so a notify() landing
            # in that window is dropped — but the next iteration drains
            # unconditionally, so the entry costs at most a TICK_SECS delay,
            # never a loss.
            try:
                async with asyncio.timeout(self.TICK_SECS):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()

    async def _sweep_if_due(self) -> None:
        """Periodic PEL housekeeping — see reclaim_outbox_stale().

        Every SWEEP_INTERVAL_SECS, and once at the first cycle after start().
        Non-fatal, but not optional either: besides XACK it is the only thing
        that clears a tombstone on Redis 7 — hence the log, not a silent pass.
        """
        loop = asyncio.get_running_loop()
        if self._next_sweep is not None and loop.time() < self._next_sweep:
            return
        self._next_sweep = loop.time() + self.SWEEP_INTERVAL_SECS
        try:
            reclaimed, purged = await reclaim_outbox_stale(
                self._redis,
                min_idle_ms=self.SWEEP_MIN_IDLE_MS,
                count=self.BATCH_SIZE,
                max_passes=self.SWEEP_MAX_PASSES,
            )
        except Exception as e:
            log.warning(f"history outbox PEL sweep failed: {e}")
            return
        if reclaimed or purged:
            # INFO, not debug: a healthy drainer's own work is invisible here
            # (_log_retry only runs on the failure path), so a stranded PEL being
            # reclaimed would leave no trace at all.
            log.info(
                f"history outbox PEL sweep reclaimed {reclaimed} stale "
                f"entries and purged {purged} tombstones"
            )

    async def _drain_once(self) -> int:
        """One batch, under the whole-cycle deadline. Returns entries SETTLED.

        Callers loop on the return value, so it must be forward progress, not
        batch size: "batch was empty" is not a sufficient stop condition, because
        a tombstone the cycle could not ack redelivers on every pass forever.
        """
        async with asyncio.timeout(self.DRAIN_DEADLINE_SECS):
            return await self._drain_batch()

    async def _read_batch(self) -> list[OutboxEntry]:
        """Pending first, then new — and never both in one cycle.

        FIFO within this drainer does not generalise: delivery across two live
        drainers is disjoint, so entry N+1 can commit before entry N. The real reason
        is a memory bound — the PEL stays within BATCH_SIZE x concurrent readers only
        while a cycle never reads `>` with a non-empty PEL. Any future head-of-line-blocking fix,
        whose natural shape is "keep draining new entries while the stuck batch
        retries", breaks that bound and must re-derive the sizing.

        NOGROUP is healed here, not only at startup: deleting a stream key
        destroys its consumer groups, operators are told to DEL this key on
        upgrade, and XADD recreates it groupless — after which every read fails
        identically forever. Rebuild and retry once.
        """
        for attempt in range(2):
            try:
                pending = await read_outbox_pending(self._redis, self.BATCH_SIZE)
                if pending:
                    return pending
                return await read_outbox_new(self._redis, self.BATCH_SIZE)
            except ResponseError as e:
                if attempt or not str(e).startswith("NOGROUP"):
                    raise
                log.warning(
                    "history outbox consumer group is missing (the key was "
                    "deleted); recreating it and retrying"
                )
                await ensure_outbox_group(self._redis)
        return []  # unreachable: the loop either returns or raises

    async def _drain_batch(self) -> int:
        """Read a batch, insert it, settle it by ID.

        No poison taxonomy: every entry here was produced by
        HistoryEntry.__post_init__, which clamps into the play_history column
        domain — except guild_id 0, which __post_init__ deliberately declines to
        fix up. A refusal therefore means a validator regression, schema drift or an
        unstamped guild_id, all bugs, and is handled as one: recorded to
        play_history_rejected, dropped, batch continues. Corrupt entries (bytes
        that do not parse) are dropped and still settled, since leaving them
        wedges the queue head forever. Tombstones are separated out first: they
        have no bytes to be corrupt.
        """
        batch = await self._read_batch()
        if not batch:
            return 0
        with _tracer.start_as_current_span("history.drain") as span:
            span.set_attribute("drain.batch", len(batch))
            settled = await self._settle_tombstones(batch, span)
            live = [e for e in batch if e.wire is not None]
            raw = [e.wire for e in live if e.wire is not None]
            entries = [e for e in map(parse_history_entry, raw) if e is not None]
            span.set_attribute("drain.parsed", len(entries))
            try:
                if entries:
                    await self._archive.insert_batch(entries)
            except _POISON:
                # _isolate settles each entry as it goes, so the batch settle in
                # the else-branch must not also run — try/except/else keeps the
                # two settle paths structurally exclusive.
                await self._isolate(live, span)
            else:
                await retire_outbox(self._redis, [e.id for e in live])
            settled += len(live)
            await self._enforce_cap()
            return settled

    async def _settle_tombstones(self, batch: list[OutboxEntry], span: Span) -> int:
        """Ack and log entries whose body is gone. Returns how many.

        A tombstone is delivered-but-deleted: XTRIM and any operator XDEL remove
        bodies without consulting the PEL, so the ID replays with an empty field
        map. Unrecoverable — a lost PLAY, logged at ERROR and counted apart from
        a parse failure, which is recoverable in principle.

        Unconditional ack is the point. Left pending, a tombstone is re-read
        every cycle forever: the drainer never dies, so respawn supervision never
        fires, nothing escalates past the backoff ceiling, and a non-evictable
        key grows unbounded — a permanent silent stall.
        """
        tombstones = [e.id for e in batch if e.wire is None]
        if not tombstones:
            return 0
        await retire_outbox(self._redis, tombstones)
        span.set_attribute("drain.tombstones", len(tombstones))
        log.error(
            f"history outbox delivered {len(tombstones)} entries whose payload "
            f"had already been deleted (ids "
            f"{b', '.join(tombstones[:5]).decode()}"
            f"{'…' if len(tombstones) > 5 else ''}) — those plays are lost and "
            f"cannot reach Postgres; acked so the drain can make progress"
        )
        return len(tombstones)

    async def _isolate(self, batch: list[OutboxEntry], span: Span) -> None:
        """One batch refused: retry it row by row so one bad row costs one row,
        not the 99 batched with it. Expected dead code, kept because dropping the
        whole batch on a validator regression turns a one-row bug into a
        hundred-play loss.

        Takes the delivered entries, not parsed rows, and settles every element.
        Both matter:

        - Settle per entry, never once at the end. A transient error partway
          through raises before an end-of-batch settle would run, redelivering
          rows already recorded as rejected. Per entry also makes the pass
          RESUMABLE, which matters because single inserts cost ~22x a batch
          (measured: 15.7ms for executemany(100) vs 343ms for 100 singles), so on
          a degraded server this can exceed DRAIN_DEADLINE_SECS and be cancelled.
        - Iterate the batch, not what parsed. A corrupt element parses to None;
          settling only what parsed leaves it delivered-and-unacked forever — the
          wedge this path exists to prevent.

        A transient error raises out of here, leaving the current entry and the
        rest to redeliver; duplicate rejection rows are prevented by _REJECT_SQL's
        ON CONFLICT, not by exclusivity.
        """
        rejected = 0
        trace_id = trace_id_of(trace.get_current_span())
        for item in batch:
            entry = parse_history_entry(item.wire) if item.wire is not None else None
            if entry is not None:
                try:
                    await self._archive.insert_batch([entry])
                except _POISON as e:
                    # item.wire, not the re-serialized entry — see
                    # record_rejection's docstring on mixed-version rollouts.
                    await self._archive.record_rejection(entry, e, trace_id, item.wire)
                    rejected += 1
                    log.error(
                        f"play_history refused a row ({type(e).__name__}) — the "
                        f"HistoryEntry validator regressed or the schema "
                        f"drifted: {entry.title[:60]!r} / "
                        f"{entry.webpage_url[:80]}"
                    )
            # else: corrupt, dropped as it always was — but still settled below.
            await retire_outbox(self._redis, [item.id])
        span.set_attribute("drain.rejected", rejected)

    async def _enforce_cap(self) -> None:
        """Opt-in outbox ceiling (config.HISTORY_OUTBOX_MAX, default off, where
        the trade-off is documented). Dropping un-archived plays is data loss, so
        every drop logs at ERROR.

        Never RUNS while SHUTTING DOWN: a departing process knows least about
        what a live peer is doing. Re-checked each pass, so stop() also halts a
        convergence loop already in progress.

        ACK before TRIM. XTRIM does not consult the PEL: on real Redis 7.4.9,
        five delivered-and-unacked entries survived `XTRIM MAXLEN 2` as pending
        records while three bodies were destroyed, and trimming first leaves an
        ID pending with no body — a tombstone, which replays forever. A crash
        between the two leaves entries acked but not trimmed: invisible to
        readers, reclaimed next pass.

        The cap deliberately crosses the PEL. Refusing to is the one thing that
        would make it useless: during an outage the cycle re-reads the same
        pending batch every tick, so the oldest entries are permanently in flight
        and a clamp below the oldest pending ID trims nothing while the backlog
        grows. Their drainer still holds the parsed rows, so a later successful
        insert archives them anyway.

        PAGED until depth is back at the cap. The trim is one MINID command
        however large the tranche and the ack set is bounded by BATCH_SIZE x live
        drainers; only minid discovery scales with the backlog, since XRANGE has
        no ID-only form and one COUNT=overage fetch would run unbounded past
        DRAIN_DEADLINE_SECS. Hence CAP_PAGE.
        """
        if not self.OUTBOX_MAX:
            return
        while not self._stopping:
            depth = await outbox_depth(self._redis)
            if depth <= self.OUTBOX_MAX:
                return
            over = depth - self.OUTBOX_MAX
            page = min(over, self.CAP_PAGE)
            # The ID of the (page+1)-th oldest entry: everything strictly below
            # it is this pass's tranche. Asking for one extra and taking the last
            # turns a COUNT into an exclusive lower bound.
            oldest = cast(
                list[tuple[bytes, dict[bytes, bytes]]],
                await self._redis.xrange(HISTORY_OUTBOX_KEY, count=page + 1),
            )
            if len(oldest) <= page:
                # Raced shorter than the page by a concurrent drain; the next
                # pass's depth check settles whether anything is genuinely left
                # over the cap.
                return
            minid = cast(bytes, oldest[-1][0])
            in_flight = await outbox_pending_below(self._redis, minid)
            await ack_outbox(self._redis, in_flight)
            dropped = await trim_outbox_below(self._redis, minid)
            if not dropped:
                return
            # XTRIM's RETURNED count, never the derived depth - OUTBOX_MAX: XLEN
            # over-counts acked-but-undeleted entries, and under redis-py's
            # approximate=True default a derived figure would claim drops while
            # removing nothing.
            log.error(
                f"history outbox over cap (depth={depth}, HISTORY_OUTBOX_MAX="
                f"{self.OUTBOX_MAX}); dropped {dropped} oldest entries — those "
                f"plays are lost and will not reach Postgres"
            )
            if in_flight:
                log.error(
                    f"{len(in_flight)} of those entries were already delivered "
                    f"to a drainer; their pending records were cleared so the "
                    f"trim could not leave them replaying forever"
                )

    async def _enforce_cap_quietly(self) -> None:
        """_enforce_cap for the failure path, where Redis may itself be what
        broke. Never raises: the caller is already handling one error, and losing
        its backoff to a second would turn a Redis blip into a hot retry loop."""
        try:
            await self._enforce_cap()
        except Exception as e:
            log.warning(f"history outbox cap check failed: {e}")

    async def _sample_depth_if_due(self) -> None:
        """_sample_depth for the PRODUCTIVE path, rate-limited. An idle cycle can
        sample every time because it costs a TICK_SECS wait, but a cycle that
        drained `continue`s straight into the next batch — sampling there
        unconditionally adds two Redis round trips per BATCH_SIZE entries for a
        whole catch-up, on the Redis that also serves playback."""
        now = asyncio.get_running_loop().time()
        if self._next_depth_sample is not None and now < self._next_depth_sample:
            return
        await self._sample_depth()

    async def _sample_depth(self) -> None:
        """Report a backlog on the success path too. _log_retry is the only other
        reader of depth and runs exclusively in the failure arm, so a
        healthy-but-behind drainer — or one on a stranded PEL — reported nothing
        at all. Best-effort and silent when shallow: a watchdog, not a metric.

        Stamps the rate-limit deadline itself so the idle and productive paths
        share one clock; otherwise a drain alternating idle and busy cycles
        samples on both and doubles the round trips it was limited to avoid.
        """
        self._next_depth_sample = (
            asyncio.get_running_loop().time() + self.DEPTH_SAMPLE_INTERVAL_SECS
        )
        try:
            depth = await outbox_depth(self._redis)
            pending = await outbox_pending_count(self._redis)
        except Exception:
            return
        if depth >= self.DEPTH_ALARM:
            log.error(
                f"history outbox backlog is {depth} entries ({pending} "
                f"delivered and unacked) while the drain is SUCCEEDING — "
                f"Postgres is keeping up but not catching up"
            )

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
