"""Schema migration runner for the Postgres play-history tier.

    just db-migrate            # or: python -m src.db_migrate

The app never applies DDL: it reads `schema_migrations` and refuses to run
against a version it was not built for, so its database role needs no DDL
rights and this module is the only thing that changes the schema.

Ordering and safety properties:

- Files are `migrations/NNNN_name.sql`, applied in numeric order, each recorded
  in `schema_migrations` in the SAME transaction as its own DDL — Postgres has
  transactional DDL, so a migration that fails half-way leaves nothing behind
  and no version row claiming it succeeded.
- `pg_advisory_xact_lock` serializes concurrent runners. Two pods starting at
  once (or a `just db-migrate` racing the compose one-shot) is the normal case,
  not the exotic one; the loser waits and then finds every version already
  applied. The lock is released with the transaction, including on crash.
- Re-running is a no-op. That is what makes it safe to wire into startup.

`POSTGRES_MIGRATE_URL` exists so the migrating role can differ from the bot's:
the app role is granted only SELECT/INSERT, leaving DDL to a role only this
runner uses. It falls back to
POSTGRES_URL, which is what the bundled compose stack uses.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import asyncpg

from src.util import get_logger

log = get_logger(__name__)

# The schema version this code is written against. Kept as a literal rather
# than "the highest file in migrations/" so the runtime never depends on the
# migrations directory being present in the image — but a test asserts the two
# agree, so adding a file and forgetting this line fails the suite rather than
# shipping a bot that rejects its own schema.
EXPECTED_SCHEMA_VERSION = 1

# src/db_migrate.py → src/ → project root.
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_MIGRATION_RE = re.compile(r"^(\d+)_.*\.sql$")

# 'mbt1' as an int32 — an arbitrary but stable application-chosen lock id.
# Advisory locks share one namespace per database, so the value only has to not
# collide with whatever else uses advisory locks in the same database.
_ADVISORY_LOCK_ID = 0x6D627431

_MIGRATIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    int PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def discover(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """Every migration file as (version, path), ascending.

    Raises on a duplicate version: two files claiming the same number means one
    of them will never be applied on a fresh database while both applied on
    whoever ran them as they landed — a schema that differs by deployment
    history. Better to fail the run than to pick one.
    """
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

    timeout=10 matches the archive's connect bound: a migration step that
    cannot reach the database should fail the deploy quickly, not hang the
    one-shot container that the bot's `depends_on` is waiting for.
    """
    migrations = discover(directory)
    if not migrations:
        raise RuntimeError(f"no migrations found in {directory}")

    conn = await asyncpg.connect(url, timeout=10)
    try:
        # The bootstrap DDL runs INSIDE the advisory lock, not before it.
        # `CREATE TABLE IF NOT EXISTS` is NOT safe to race: concurrent creators
        # hit a catalog race and all but one die with
        #   UniqueViolationError: duplicate key value violates unique
        #   constraint "pg_type_typname_nsp_index"
        # Measured on postgres:18, 4 concurrent runners against a virgin
        # database: 8 of 15 trials killed at least one runner. Harmless for the
        # single compose one-shot, fatal for a K8s init-container per pod.
        #
        # A session-level lock (not xact) because this is outside any
        # transaction; released explicitly below so the per-migration
        # pg_advisory_xact_lock is not taken while we still hold this one.
        await conn.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_ID)
        try:
            await conn.execute(_MIGRATIONS_TABLE_DDL)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_ID)
        applied_count = 0
        for version, path in migrations:
            # One transaction per migration, not one for the whole run: a
            # failure keeps every earlier migration applied, so the retry is
            # short and the recorded version always describes the real schema.
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
    if version != EXPECTED_SCHEMA_VERSION:
        # Only reachable when migrations/ and this module disagree — i.e. a
        # partial checkout or a hand-edited migrations directory.
        print(
            f"Error: schema is at version {version} after migrating, but this "
            f"build expects {EXPECTED_SCHEMA_VERSION}.",
            file=sys.stderr,
        )
        return 1
    print(f"play-history schema at version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
