"""Real-Postgres integration tier (docs/POSTGRES_HISTORY_PLAN.md §9.2).

Opt-in: skipped unless RUN_PG_TESTS=1 (needs Docker). One postgres:18-alpine
testcontainer per session; one throwaway database per test for isolation.

    RUN_PG_TESTS=1 poetry run pytest -m pg --no-cov

Covers exactly what the fakes can't: the migration runner against a real
ledger, ON CONFLICT dedup under concurrency, timestamptz↔epoch round-trips,
recent() ordering with tie-breaks, and the drainer's redelivery dedup.
"""

import asyncio
import itertools
import os

import pytest

from src.backfill_history import run as backfill_run
from src.db import Database, MigrationError
from src.guild_state import HistoryEntry
from src.history_archive import HistoryOutboxDrainer, PostgresHistoryArchive
from src.redis_client import HISTORY_CACHE_LIMIT, HISTORY_OUTBOX_KEY, GuildRedisStore

pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(
        not os.getenv("RUN_PG_TESTS"),
        reason="pg tier is opt-in: set RUN_PG_TESTS=1 (requires Docker)",
    ),
]

_PG_IMAGE = "postgres:18-alpine"
_dbname_counter = itertools.count(1)


@pytest.fixture(scope="session")
def pg_container():
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(_PG_IMAGE, username="test", password="test") as pg:
        yield pg


@pytest.fixture
def admin_dsn(pg_container) -> str:
    host = pg_container.get_container_host_ip()
    port = pg_container.get_exposed_port(5432)
    return f"postgresql://test:test@{host}:{port}/{pg_container.dbname}"


@pytest.fixture
async def pg_dsn(pg_container, admin_dsn):
    """A fresh database per test — full isolation, ~ms to create."""
    import asyncpg

    name = f"t{next(_dbname_counter)}"
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f"CREATE DATABASE {name}")
    finally:
        await conn.close()
    yield admin_dsn.rsplit("/", 1)[0] + f"/{name}"
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f"DROP DATABASE {name} WITH (FORCE)")
    finally:
        await conn.close()


@pytest.fixture
async def db(pg_dsn):
    database = Database(pg_dsn)
    yield database
    await database.close()


@pytest.fixture
def archive(db):
    return PostgresHistoryArchive(db)


def _entry(n: int, guild_id: int = 42, played_at: float | None = None) -> HistoryEntry:
    return HistoryEntry(
        guild_id=guild_id,
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=190,
        requester_id=222222222222222222,  # snowflake magnitude on purpose
        requester_name=f"user{n}",
        thumbnail=f"https://img/{n}.jpg",
        uploader="Chan",
        played_at=1752530000.0 + n if played_at is None else played_at,
    )


class TestMigrations:
    async def test_fresh_database_gets_schema_and_ledger(self, db):
        async with db.acquire() as conn:
            versions = [
                r["version"]
                for r in await conn.fetch(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ]
            assert versions[0] == "0001_play_history"
            # And the baseline objects exist.
            assert await conn.fetchval("SELECT count(*) FROM play_history") == 0
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM pg_indexes WHERE indexname = 'play_history_dedup'"
                )
                == 1
            )

    async def test_second_database_same_dsn_is_noop(self, pg_dsn, db):
        async with db.acquire():
            pass  # first: applies migrations
        second = Database(pg_dsn)
        try:
            async with second.acquire() as conn:
                count = await conn.fetchval("SELECT count(*) FROM schema_migrations")
        finally:
            await second.close()
        assert count == 1  # ledger unchanged — no re-apply

    async def test_tampered_migration_fails_closed(self, pg_dsn, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        f = d / "0001_thing.sql"
        f.write_text("CREATE TABLE thing (id int);")
        first = Database(pg_dsn, migrations_dir=d)
        try:
            async with first.acquire():
                pass
        finally:
            await first.close()
        f.write_text("CREATE TABLE thing (id int, sneaky int);")
        tampered = Database(pg_dsn, migrations_dir=d)
        try:
            with pytest.raises(MigrationError, match="modified after being applied"):
                async with tampered.acquire():
                    pass
        finally:
            await tampered.close()

    async def test_failed_pool_attempt_leaves_no_half_open_pool(self, pg_dsn, tmp_path):
        # A broken migration must not leave a usable-looking Database behind.
        d = tmp_path / "m"
        d.mkdir()
        (d / "0001_bad.sql").write_text("THIS IS NOT SQL;")
        db = Database(pg_dsn, migrations_dir=d)
        try:
            with pytest.raises(Exception):
                async with db.acquire():
                    pass
            # Fixing the dir isn't possible on this instance (path is fixed),
            # but a fresh acquire retries from scratch rather than reusing a
            # half-open pool:
            with pytest.raises(Exception):
                async with db.acquire():
                    pass
        finally:
            await db.close()


class TestArchiveAgainstRealPG:
    async def test_round_trip_preserves_every_field(self, archive):
        await archive.insert_batch([_entry(1)])
        assert await archive.recent(42, 10) == [_entry(1)]

    async def test_epoch_zero_unknown_survives(self, archive):
        e = _entry(1, played_at=0.0)
        await archive.insert_batch([e])
        got = await archive.recent(42, 10)
        assert got == [e] and got[0].played_at == 0.0

    async def test_dedup_on_replay(self, archive):
        await archive.insert_batch([_entry(1), _entry(2)])
        await archive.insert_batch([_entry(1), _entry(2)])  # full replay
        assert len(await archive.recent(42, 10)) == 2

    async def test_dedup_under_concurrent_inserts(self, archive):
        batch = [_entry(n) for n in range(20)]
        await asyncio.gather(*(archive.insert_batch(batch) for _ in range(4)))
        assert len(await archive.recent(42, 50)) == 20

    async def test_recent_orders_newest_first_with_id_tiebreak(self, archive):
        # Same played_at, different songs: later insert (higher id) wins ties.
        a = _entry(1, played_at=5000.0)
        b = _entry(2, played_at=5000.0)
        newer = _entry(3, played_at=6000.0)
        await archive.insert_batch([a, b, newer])
        assert await archive.recent(42, 10) == [newer, b, a]

    async def test_guild_isolation(self, archive):
        await archive.insert_batch([_entry(1, guild_id=1), _entry(2, guild_id=2)])
        assert await archive.recent(1, 10) == [_entry(1, guild_id=1)]

    async def test_limit_applies(self, archive):
        await archive.insert_batch([_entry(n) for n in range(5)])
        assert len(await archive.recent(42, 3)) == 3


class TestBackfillAgainstRealPG:
    async def test_count(self, archive):
        assert await archive.count(42) == 0
        await archive.insert_batch([_entry(1), _entry(2), _entry(3, guild_id=7)])
        assert await archive.count(42) == 2
        assert await archive.count(7) == 1

    async def test_full_backfill_verify_demote_flow(self, fake_redis, archive):
        # Runbook §6 steps 2–5 in miniature: seed two guilds' Redis lists
        # (one oversized), backfill + verify + demote in one invocation.
        for n in range(HISTORY_CACHE_LIMIT + 5):
            await fake_redis.lpush("guild:1:history", _entry(n, guild_id=1).to_redis())
        await fake_redis.lpush("guild:2:history", _entry(1, guild_id=2).to_redis())
        await fake_redis.lpush("guild:2:history", b"corrupt bytes")
        lines: list[str] = []
        assert (
            await backfill_run(
                fake_redis, archive, verify=True, demote=True, out=lines.append
            )
            == 0
        )
        assert await archive.count(1) == HISTORY_CACHE_LIMIT + 5
        assert await archive.count(2) == 1
        assert "verification passed" in lines
        # Demotion trimmed + TTL'd the at-rest lists.
        assert await fake_redis.llen("guild:1:history") == HISTORY_CACHE_LIMIT
        assert await fake_redis.ttl("guild:1:history") > 0
        # Idempotent: a second full run inserts nothing and still passes.
        assert await backfill_run(fake_redis, archive, verify=True) == 0
        assert await archive.count(1) == HISTORY_CACHE_LIMIT + 5


class TestDrainerAgainstRealPG:
    async def test_outbox_to_rows_and_redelivery_dedup(self, fake_redis, archive):
        store = GuildRedisStore(fake_redis, guild_id=42)
        for n in (1, 2, 3):
            await store.push_history(_entry(n), outbox=True)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        assert await drainer._drain_once() == 3
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0
        assert await archive.recent(42, 10) == [_entry(3), _entry(2), _entry(1)]
        # Redelivery (crash-between-insert-and-retire simulation): same
        # entries pushed again drain cleanly and dedup to the same rows.
        for n in (1, 2, 3):
            await store.push_history(_entry(n), outbox=True)
        assert await drainer._drain_once() == 3
        assert len(await archive.recent(42, 10)) == 3
