"""
Postgres play-history archive — the durable long-term home for every played
song (docs/POSTGRES_HISTORY_PLAN.md).

Three pieces:

- HistoryArchive — the protocol GuildHistory and the drainer program against;
  unit tests substitute an in-memory fake.
- PostgresHistoryArchive — the SQL + row-mapping repository. Connection
  lifecycle and schema live in src/db.py (`Database`): the pool is lazy and
  migrations run on first acquire, so bot startup never blocks on Postgres
  being reachable (the drainer's backoff loop absorbs failures instead).
- HistoryOutboxDrainer — the single background task that moves entries from
  the Redis outbox list to Postgres: peek oldest batch → INSERT ... ON
  CONFLICT DO NOTHING → retire. At-least-once delivery; a crash between
  insert and retire redelivers, and the play_history_dedup unique index
  collapses the replay. The playback loop never awaits Postgres — add()
  LPUSHes the outbox and notify()s this task.

Row mapping (HistoryEntry ↔ play_history row) lives here, not in
guild_state.py — that module's contract is pure wire schema with no runtime
imports, and the database layer is very much a runtime import. The schema
itself lives in db/migrations/ (docs/POSTGRES_HISTORY_PLAN.md §4).
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol

from src.db import Database
from src.guild_state import HistoryEntry, parse_history_entry
from src.redis_client import outbox_depth, peek_outbox_oldest, retire_outbox
from src.telemetry import get_meter
from src.util import get_logger

log = get_logger(__name__)


# ── -stats query result shapes (docs/POSTGRES_HISTORY_PLAN.md §7.1) ──────────
# Not wire schema (nothing here is persisted), so they live with the queries
# that produce them, not in guild_state.py.


@dataclass(frozen=True, slots=True, kw_only=True)
class TopSong:
    title: str
    webpage_url: str
    plays: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TopRequester:
    requester_id: int
    requester_name: str
    plays: int


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildStats:
    """One guild's aggregate playback statistics, optionally windowed."""

    plays: int
    distinct_songs: int
    seconds_listened: int
    top_songs: tuple[TopSong, ...]
    top_requesters: tuple[TopRequester, ...]


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

_RECENT_BY_USER_SQL = """
SELECT guild_id, title, webpage_url, duration_secs, played_secs,
       requester_id, requester_name, thumbnail, uploader, played_at
FROM play_history
WHERE guild_id = $1 AND requester_id = $2
ORDER BY played_at DESC, id DESC
LIMIT $3
"""

# The optional --days window is one nullable parameter, not a second SQL
# variant: $2 NULL disables the clause. guild_id equality leads every index
# here, so the residual filter cost is negligible at this table's scale.
_STATS_TOTALS_SQL = """
SELECT count(*)                       AS plays,
       count(DISTINCT webpage_url)    AS distinct_songs,
       coalesce(sum(played_secs), 0)  AS seconds_listened
FROM play_history
WHERE guild_id = $1
  AND ($2::int IS NULL OR played_at > now() - make_interval(days => $2::int))
"""

_STATS_TOP_SONGS_SQL = """
SELECT webpage_url, max(title) AS title, count(*) AS plays
FROM play_history
WHERE guild_id = $1
  AND ($2::int IS NULL OR played_at > now() - make_interval(days => $2::int))
GROUP BY webpage_url
ORDER BY plays DESC, max(played_at) DESC
LIMIT $3
"""

# requester_id = 0 rows (unknown requester) are excluded from the leaderboard
# but deliberately included in the totals above — plan §7.1.
_STATS_TOP_REQUESTERS_SQL = """
SELECT requester_id, max(requester_name) AS requester_name, count(*) AS plays
FROM play_history
WHERE guild_id = $1 AND requester_id <> 0
  AND ($2::int IS NULL OR played_at > now() - make_interval(days => $2::int))
GROUP BY requester_id
ORDER BY plays DESC, max(played_at) DESC
LIMIT $3
"""

_STATS_TOP_N = 5


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


def _row_to_entry(row) -> HistoryEntry:
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
    """What GuildHistory (reads) and the drainer (writes) need from the
    archive — faked in unit tests, implemented by asyncpg below."""

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None: ...

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]: ...

    async def recent_by_user(
        self, guild_id: int, requester_id: int, limit: int
    ) -> list[HistoryEntry]: ...

    async def stats(
        self, guild_id: int, *, days: Optional[int] = None
    ) -> GuildStats: ...


class PostgresHistoryArchive:
    """SQL + row-mapping repository over `Database`. All methods raise on
    failure — callers own the error policy (the drainer backs off; Phase C's
    recent() falls back). Lifecycle (pool, migrations, close) belongs to the
    Database, which the app owns."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        """Insert oldest-first; replays and backfill overlap dedup via the
        play_history_dedup unique index (ON CONFLICT DO NOTHING)."""
        if not entries:
            return
        async with self._db.acquire() as conn:
            await conn.executemany(_INSERT_SQL, [_entry_to_row(e) for e in entries])

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        """The `limit` most recent entries for one guild, newest first. id is
        the tie-break so epoch-0 (unknown-time) entries order stably."""
        if limit <= 0:
            return []
        async with self._db.acquire() as conn:
            rows = await conn.fetch(_RECENT_SQL, guild_id, limit)
        return [_row_to_entry(r) for r in rows]

    async def recent_by_user(
        self, guild_id: int, requester_id: int, limit: int
    ) -> list[HistoryEntry]:
        """recent(), filtered to one requester (plan §7.2; 0003 index)."""
        if limit <= 0:
            return []
        async with self._db.acquire() as conn:
            rows = await conn.fetch(_RECENT_BY_USER_SQL, guild_id, requester_id, limit)
        return [_row_to_entry(r) for r in rows]

    async def stats(self, guild_id: int, *, days: Optional[int] = None) -> GuildStats:
        """Aggregate playback stats, optionally windowed to the last `days`
        days (plan §7.1). Three single-table aggregates on one connection."""
        async with self._db.acquire() as conn:
            totals = await conn.fetchrow(_STATS_TOTALS_SQL, guild_id, days)
            songs = await conn.fetch(_STATS_TOP_SONGS_SQL, guild_id, days, _STATS_TOP_N)
            requesters = await conn.fetch(
                _STATS_TOP_REQUESTERS_SQL, guild_id, days, _STATS_TOP_N
            )
        assert totals is not None  # aggregates always return one row
        return GuildStats(
            plays=totals["plays"],
            distinct_songs=totals["distinct_songs"],
            seconds_listened=totals["seconds_listened"],
            top_songs=tuple(
                TopSong(
                    title=r["title"], webpage_url=r["webpage_url"], plays=r["plays"]
                )
                for r in songs
            ),
            top_requesters=tuple(
                TopRequester(
                    requester_id=r["requester_id"],
                    requester_name=r["requester_name"],
                    plays=r["plays"],
                )
                for r in requesters
            ),
        )

    async def count(self, guild_id: int) -> int:
        """Rows stored for one guild — the backfill's inserted/dup accounting
        and --verify comparison (docs/POSTGRES_HISTORY_PLAN.md §5.6)."""
        async with self._db.acquire() as conn:
            return await conn.fetchval(
                "SELECT count(*) FROM play_history WHERE guild_id = $1", guild_id
            )


class HistoryOutboxDrainer:
    """The one task per process that drains the Redis outbox into the archive.

    Wakes on notify() (set by every outbox push) with a periodic fallback
    tick, drains in batches until the outbox is empty, and on archive/Redis
    failure backs off exponentially while entries accumulate safely in the
    outbox (persistent, non-evictable — see HISTORY_OUTBOX_KEY).

    Single-consumer by design: the peek → insert → retire cycle is only safe
    with one drainer per outbox (redis_client.py, "History outbox" section).
    """

    BATCH_SIZE = 100
    TICK_SECS = 30.0
    DEPTH_ALARM = 10_000  # backlog that escalates the retry warning to ERROR
    _BACKOFF_START = 1.0
    _BACKOFF_MAX = 60.0

    def __init__(self, redis, archive: HistoryArchive) -> None:
        self._redis = redis
        self._archive = archive
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        # §8.3 monitoring: the alert line is the depth gauge growing across
        # two scrapes (Postgres down longer than the backoff ceiling). Both
        # instruments are no-ops until setup_telemetry() configures a meter
        # provider, so tests and OTEL_SDK_DISABLED runs pay nothing.
        meter = get_meter(__name__)
        self._drained_counter = meter.create_counter(
            "musicbot.history.outbox.drained",
            unit="{entry}",
            description="Outbox entries retired to the archive "
            "(corrupt-dropped entries included)",
        )
        # TODO: Add the Grafana alert rule for the outbox depth gauge.
        # The metric exports, but nothing consumes it yet: a Postgres outage
        # longer than the drainer's backoff ceiling is visible only in logs
        # until the rule exists. Rule shape: alert when this gauge grows
        # across two consecutive scrapes (the 60s retry-cadence cap refreshes
        # it inside every scrape interval, so growth means Postgres is still
        # down). Dashboard config in the otel-lgtm Grafana, not repo code —
        # this comment is the tracking anchor.
        # See: docs/POSTGRES_HISTORY_PLAN.md §8.3.
        self._depth_gauge = meter.create_gauge(
            "musicbot.history.outbox.depth",
            unit="{entry}",
            description="history:outbox backlog depth "
            "(0 on every drain to empty; refreshed on each failed-drain retry)",
        )

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="history-outbox-drainer")

    def notify(self) -> None:
        """Signal a fresh outbox push — cheap, sync, callable from anywhere."""
        self._wake.set()

    async def stop(self, timeout: float = 5.0) -> None:
        """Cancel the loop, then make one bounded final-drain attempt so a
        clean shutdown ships whatever a healthy Postgres can take. Never
        raises — anything left simply stays in the outbox for next start."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            async with asyncio.timeout(timeout):
                while await self._drain_once():
                    pass
        except Exception as e:
            log.warning(f"history outbox final drain incomplete: {e}")

    async def _run(self) -> None:
        backoff = self._BACKOFF_START
        while True:
            try:
                drained = await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._log_retry(e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._BACKOFF_MAX)
                continue
            backoff = self._BACKOFF_START
            if drained:
                continue  # backlog: keep draining without waiting
            # Idle. clear() only ever runs after wait() returns, so a push
            # racing this window re-sets the event and is never lost — worst
            # case is one spurious extra drain of an empty outbox.
            try:
                async with asyncio.timeout(self.TICK_SECS):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()

    async def _drain_once(self) -> int:
        """One batch: peek oldest, insert, retire. Returns entries retired.
        Corrupt entries are dropped (parse_history_entry warns per entry) but
        still retired — leaving them would wedge the queue head forever."""
        raw = await peek_outbox_oldest(self._redis, self.BATCH_SIZE)
        if not raw:
            # An empty peek is an exact depth reading — no extra LLEN needed.
            # During an outage the gauge is refreshed by _log_retry instead,
            # so "growing across two scrapes" is the §8.3 alert line.
            self._depth_gauge.set(0)
            return 0
        entries = [e for e in map(parse_history_entry, raw) if e is not None]
        if entries:
            await self._archive.insert_batch(entries)
        await retire_outbox(self._redis, len(raw))
        self._drained_counter.add(len(raw))
        return len(raw)

    async def _log_retry(self, error: Exception, backoff: float) -> None:
        try:
            depth = await outbox_depth(self._redis)
        except Exception:
            depth = -1  # Redis itself is down; depth unknowable
        if depth >= 0:
            # Unknowable depth (-1) keeps the last known reading rather than
            # recording a sentinel that would corrupt the growth alert.
            self._depth_gauge.set(depth)
        emit = log.error if depth >= self.DEPTH_ALARM else log.warning
        emit(
            f"history outbox drain failed (backlog={depth}): "
            f"{type(error).__name__}: {error}; retrying in {backoff:.0f}s"
        )
