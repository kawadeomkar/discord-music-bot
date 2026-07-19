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
from datetime import datetime, timezone
from typing import Optional, Protocol

from src.db import Database
from src.guild_state import HistoryEntry, parse_history_entry
from src.redis_client import outbox_depth, peek_outbox_oldest, retire_outbox
from src.util import get_logger

log = get_logger(__name__)

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
    """What GuildHistory (Phase B reads) and the drainer (writes) need from
    the archive — faked in unit tests, implemented by asyncpg below."""

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None: ...

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]: ...


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
            return 0
        entries = [e for e in map(parse_history_entry, raw) if e is not None]
        if entries:
            await self._archive.insert_batch(entries)
        await retire_outbox(self._redis, len(raw))
        return len(raw)

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
