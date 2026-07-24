"""
Durable-tier substrate (docs/POSTGRES_HISTORY_PLAN.md §5.7) — everything
asyncpg-lifecycle, nothing domain.

One `Database` per process. Lazy: constructing it makes no connection, so bot
startup never blocks on Postgres — the first `acquire()` creates the pool and
runs pending migrations, and callers own the error policy (the outbox drainer
backs off; command reads fall back). Repositories (PostgresHistoryArchive now,
future Phase-D repos) depend on this class, never on asyncpg pools directly.

Migrations (§4.2): numbered forward-only SQL files in db/migrations/, applied
in filename order, each inside its own transaction together with its
schema_migrations ledger row. The whole run holds pg_advisory_lock so two
racing processes can't double-apply. A file changed after it was applied is a
MigrationError (checksum mismatch), never a re-run — restore-from-backup is
the rollback story (§8.1), by design.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import asyncpg

from src.util import get_logger

log = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"

# Session-scoped advisory lock guarding the migration run. Arbitrary but
# stable constant ("music" in hex); anything else touching this database
# must not reuse it.
_MIGRATION_LOCK_KEY = 0x6D75736963

_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text        PRIMARY KEY,
    checksum   text        NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(Exception):
    """A migration file is malformed, misordered, or was edited after being
    applied. Deliberately NOT self-healing: the fix is a human reconciling
    files against the ledger, not code guessing."""


def _discover(migrations_dir: Path) -> list[Path]:
    """Migration files in apply order. Filenames are NNNN_description.sql;
    a duplicated numeric prefix is ambiguous ordering → error."""
    files = sorted(p for p in migrations_dir.glob("*.sql") if p.stem[:4].isdigit())
    seen: dict[str, str] = {}
    for path in files:
        prefix = path.stem[:4]
        if prefix in seen:
            raise MigrationError(
                f"duplicate migration prefix {prefix}: {seen[prefix]} vs {path.name}"
            )
        seen[prefix] = path.name
    return files


async def run_migrations(conn: Any, migrations_dir: Path = MIGRATIONS_DIR) -> int:
    """Apply pending migrations on `conn`; returns how many were applied.
    Caller holds the advisory lock (Database does) — this function only owns
    ledger bookkeeping and per-file transactions."""
    files = _discover(migrations_dir)
    await conn.execute(_LEDGER_DDL)
    rows = await conn.fetch("SELECT version, checksum FROM schema_migrations")
    applied = {r["version"]: r["checksum"] for r in rows}
    count = 0
    for path in files:
        version = path.stem
        sql = path.read_text()
        checksum = hashlib.sha256(sql.encode()).hexdigest()
        if version in applied:
            if applied[version] != checksum:
                raise MigrationError(
                    f"{path.name} was modified after being applied "
                    f"(ledger {applied[version][:12]}… ≠ file {checksum[:12]}…)"
                )
            continue
        # asyncpg runs multi-statement scripts via the simple protocol when
        # no bind args are passed — one .sql file may hold several statements.
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)",
                version,
                checksum,
            )
        log.info(f"db: applied migration {path.name}")
        count += 1
    return count


class Database:
    """Owns the process's one asyncpg pool + the migration run gating it.

    Pool config (plan §2.5): sized for the actual workload — ≤1 drainer write
    batch per song end plus occasional command reads. timeout=10 keeps the
    drainer's backoff loop responsive on connect failure; command_timeout=30
    because no query here has any business running longer; idle connections
    fold back to min_size between listening sessions. server_settings pins
    application_name (pg_stat_activity) and UTC.
    """

    def __init__(self, dsn: str, *, migrations_dir: Path = MIGRATIONS_DIR) -> None:
        self._dsn = dsn
        self._migrations_dir = migrations_dir
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()

    async def _ensure(self) -> asyncpg.Pool:
        """Lazy pool + migrations, double-checked under the lock. First
        successful call wins; a failed attempt leaves no half-open pool."""
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._pool is None:
                pool = await asyncpg.create_pool(
                    self._dsn,
                    min_size=1,
                    max_size=4,
                    timeout=10,
                    command_timeout=30,
                    max_inactive_connection_lifetime=300,
                    server_settings={
                        "application_name": "discord-music-bot",
                        "timezone": "UTC",
                    },
                )
                try:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_KEY
                        )
                        try:
                            await run_migrations(conn, self._migrations_dir)
                        finally:
                            await conn.execute(
                                "SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_KEY
                            )
                except BaseException:
                    await pool.close()
                    raise
                self._pool = pool
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """`async with db.acquire() as conn:` — connection from the (lazily
        created, migrated) pool."""
        pool = await self._ensure()
        async with pool.acquire() as conn:
            yield conn

    async def close(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            await pool.close()
