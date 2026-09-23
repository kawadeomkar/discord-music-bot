"""Schema migration runner for the Postgres play-history tier.

    just db-migrate            # or: python -m src.db_migrate

The app never applies DDL: it reads `schema_migrations` and refuses a version
it was not built for, so its role needs no DDL rights and this module is the
only thing that changes the schema. `migrations/NNNN_name.sql` are applied in
numeric order, each recorded in `schema_migrations` in the same transaction as
its own DDL (Postgres DDL is transactional, so a half-way failure leaves no
version row claiming success). `pg_advisory_xact_lock` serializes concurrent
runners — `just db-migrate` racing the compose one-shot is the normal case —
and is released with the transaction. Re-running is a no-op.

`POSTGRES_MIGRATE_URL` lets the migrating role differ from the bot's, which is
granted only SELECT/INSERT. Falls back to POSTGRES_URL, as compose uses.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import asyncpg

from src.util import get_logger

log = get_logger(__name__)

# The schema version this code is written against. A literal rather than "the
# highest file in migrations/" so the runtime never depends on that directory
# being present in the image; asserted against the files by a test.
EXPECTED_SCHEMA_VERSION = 1

# src/db_migrate.py → src/ → project root.
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_MIGRATION_RE = re.compile(r"^(\d+)_.*\.sql$")

# 'mbt1' as an int32 — arbitrary but stable. Advisory locks share one namespace
# per database, so it only has to not collide with other users of that namespace.
_ADVISORY_LOCK_ID = 0x6D627431

_MIGRATIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    int PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def discover(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """Every migration file as (version, path), ascending. Raises on a duplicate
    version: only one of the pair would ever reach a fresh database, giving a
    schema that differs by deployment history."""
    found: dict[int, Path] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_RE.match(path.name)
        if match is None:
            raise RuntimeError(f"migration filename must be NNNN_name.sql: {path.name}")
        version = int(match.group(1))
        if version in found:
            raise RuntimeError(
                f"duplicate migration version {version}: "
                f"{found[version].name} and {path.name}"
            )
        found[version] = path
    return sorted(found.items())


async def migrate(url: str, directory: Path = MIGRATIONS_DIR) -> int:
    """Apply every unapplied migration. Returns the resulting schema version.
    timeout=10 matches the archive's connect bound: a step that cannot reach
    the database should fail the deploy quickly, not hang the one-shot
    container the bot's `depends_on` waits for."""
    migrations = discover(directory)
    if not migrations:
        raise RuntimeError(f"no migrations found in {directory}")

    conn = await asyncpg.connect(url, timeout=10)
    try:
        # The bootstrap DDL runs inside a session advisory lock: concurrent
        # `CREATE TABLE IF NOT EXISTS` hits a catalog race and all but one
        # runner die with UniqueViolationError on pg_type_typname_nsp_index
        # (8 of 15 trials with 4 runners). Released before the per-migration
        # xact lock is taken.
        await conn.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_ID)
        try:
            await conn.execute(_MIGRATIONS_TABLE_DDL)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_ID)
        applied_count = 0
        for version, path in migrations:
            # One transaction per migration: a failure keeps every earlier
            # migration applied, so the recorded version describes the real schema.
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1)", _ADVISORY_LOCK_ID
                )
                already = await conn.fetchval(
                    "SELECT 1 FROM schema_migrations WHERE version = $1", version
                )
                if already:
                    continue
                await conn.execute(path.read_text())
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)", version
                )
                applied_count += 1
                log.info(f"applied migration {version:04d} {path.name}")
        current = await conn.fetchval("SELECT max(version) FROM schema_migrations")
    finally:
        await conn.close()

    if applied_count == 0:
        log.info(f"schema already at version {current}; nothing to apply")
    return int(current or 0)


def main() -> int:
    url = os.environ.get("POSTGRES_MIGRATE_URL") or os.environ.get("POSTGRES_URL")
    if not url:
        print(
            "Error: neither POSTGRES_MIGRATE_URL nor POSTGRES_URL is set.\n"
            "       Run ./setup_env.sh to populate .env, or point "
            "POSTGRES_MIGRATE_URL at the database to migrate.",
            file=sys.stderr,
        )
        return 1
    version = asyncio.run(migrate(url))
    if version < EXPECTED_SCHEMA_VERSION:
        # Only reachable when migrations/ and this module disagree: a partial
        # checkout or a hand-edited migrations directory.
        print(
            f"Error: schema is at version {version} after migrating, but this "
            f"build expects {EXPECTED_SCHEMA_VERSION}.",
            file=sys.stderr,
        )
        return 1
    if version > EXPECTED_SCHEMA_VERSION:
        # A newer database is not an error (migrations are additive), as in
        # PostgresHistoryArchive._assert_schema_version; deploy_docker.sh gates
        # on this exit status, so refusing would make every rollback an outage.
        print(
            f"Note: schema is at version {version}, ahead of this build's "
            f"{EXPECTED_SCHEMA_VERSION}. Continuing — migrations are additive, "
            f"but this build is older than the database."
        )
        return 0
    print(f"play-history schema at version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
