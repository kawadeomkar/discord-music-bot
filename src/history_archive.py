"""
Postgres play-history archive — the durable long-term home for every played
song (docs/POSTGRES_HISTORY_PLAN.md).

Three pieces:

- HistoryArchive — the protocol GuildHistory and the drainer program against;
  unit tests substitute an in-memory fake.
- PostgresHistoryArchive — the asyncpg implementation. Lazily connects and
  applies the schema on first use so bot startup never blocks on Postgres
  being reachable (the drainer's backoff loop absorbs failures instead).
- HistoryOutboxDrainer — the single background task that moves entries from
  the Redis outbox list to Postgres: peek oldest batch → INSERT ... ON
  CONFLICT DO NOTHING → retire. At-least-once delivery; a crash between
  insert and retire redelivers, and the play_history_dedup unique index
  collapses the replay. The playback loop never awaits Postgres — add()
  LPUSHes the outbox and notify()s this task.

Row mapping (HistoryEntry ↔ play_history row) lives here, not in
guild_state.py — that module's contract is pure wire schema with no runtime
imports, and asyncpg is very much a runtime import.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

import asyncpg
import redis.asyncio as aioredis

from src.guild_state import HistoryEntry, parse_history_entry
from src.redis_client import outbox_depth, peek_outbox_oldest, retire_outbox
from src.util import get_logger

log = get_logger(__name__)

# The zero-value convention ("0 / empty string = unknown") carries over from
# the wire format — no NULLs. Deliberate: standard unique indexes treat NULLs
# as distinct, which would break dedup exactly on the unknown-played_at rows
# that need it most. played_at epoch 0 = unknown, same sentinel as the wire.
# The dedup index doubles as the -history read index: its leading
# (guild_id, played_at) columns serve ORDER BY played_at DESC via backward
# scan. Full rationale: docs/POSTGRES_HISTORY_PLAN.md §4.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS play_history (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    guild_id       bigint      NOT NULL,
    title          text        NOT NULL DEFAULT '',
    webpage_url    text        NOT NULL DEFAULT '',
    duration_secs  integer     NOT NULL DEFAULT 0,
    played_secs    integer     NOT NULL DEFAULT 0,
    requester_id   bigint      NOT NULL DEFAULT 0,
    requester_name text        NOT NULL DEFAULT '',
    thumbnail      text        NOT NULL DEFAULT '',
    uploader       text        NOT NULL DEFAULT '',
    played_at      timestamptz NOT NULL DEFAULT to_timestamp(0)
);
CREATE UNIQUE INDEX IF NOT EXISTS play_history_dedup
    ON play_history (guild_id, played_at, webpage_url);
"""

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


def _row_to_entry(row: Any) -> HistoryEntry:
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
    """asyncpg-backed archive. All methods raise on failure — callers own the
    error policy (the drainer backs off; Phase B's recent() falls back)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()

    async def _ensure(self) -> asyncpg.Pool:
        """Lazy pool + idempotent DDL, double-checked under the lock. First
        successful call wins; a failed attempt leaves no half-open pool."""
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._pool is None:
                # timeout=10: a fast connect failure keeps the drainer's
                # backoff loop responsive (default is 60s).
                pool = await asyncpg.create_pool(
                    self._url, min_size=1, max_size=4, timeout=10
                )
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(_SCHEMA_DDL)
                except BaseException:
                    await pool.close()
                    raise
                self._pool = pool
        return self._pool

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        """Insert oldest-first; replays and backfill overlap dedup via the
        play_history_dedup unique index (ON CONFLICT DO NOTHING)."""
        if not entries:
            return
        pool = await self._ensure()
        async with pool.acquire() as conn:
            await conn.executemany(_INSERT_SQL, [_entry_to_row(e) for e in entries])

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        """The `limit` most recent entries for one guild, newest first. id is
        the tie-break so epoch-0 (unknown-time) entries order stably."""
        if limit <= 0:
            return []
        pool = await self._ensure()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_RECENT_SQL, guild_id, limit)
        return [_row_to_entry(r) for r in rows]

    async def close(self) -> None:
        # _pool is nulled before the await deliberately. This is safe ONLY
        # because MusicBotApp.close() runs drainer.stop() (which finishes or
        # cancels all draining) strictly before archive.close(), so no _ensure()
        # can race a second pool into existence during the await. Do not reorder
        # those two closes, or shutdown could leak a freshly-built pool.
        pool = self._pool
        self._pool = None
        if pool is not None:
            await pool.close()


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

    def __init__(self, redis: aioredis.Redis, archive: HistoryArchive) -> None:
        self._redis = redis
        self._archive = archive
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="history-outbox-drainer")
        self._task.add_done_callback(self._on_task_done)

    @staticmethod
    def _on_task_done(task: "asyncio.Task[None]") -> None:
        """Last-resort supervision: _run only ever exits via cancellation, so
        any exception surfacing here is a bug — log it loudly the moment it
        happens (not at shutdown), because a dead drainer means the outbox
        grows silently until restart."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error(
                f"history outbox drainer died unexpectedly "
                f"({type(exc).__name__}: {exc}); entries will accumulate in "
                f"the Redis outbox until the next start"
            )

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
