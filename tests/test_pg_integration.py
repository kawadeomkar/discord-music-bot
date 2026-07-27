"""Real-Postgres integration tier (docs/POSTGRES_HISTORY_PLAN.md §9).

Opt-in, with two ways to supply the server — see `admin_dsn` below:

    just test-pg                        # local: testcontainers, needs Docker
    POSTGRES_TEST_URL=... just test-pg  # CI: an already-running server

A throwaway database per test gives isolation either way.

Covers exactly what the in-memory fakes cannot: PostgresHistoryArchive against a
real server — _SCHEMA_DDL actually executing, the _INSERT_SQL/_RECENT_SQL
parameter binding, ON CONFLICT dedup, the timestamptz<->epoch round-trip, and
recent()'s newest-first ordering with the id tie-break. Until this tier runs,
those SQL constants are validated only by inspection.
"""

import itertools
import os
from collections.abc import AsyncIterator, Iterator
from urllib.parse import urlsplit, urlunsplit

import pytest

from src.guild_state import HistoryEntry
from src.history_archive import PostgresHistoryArchive

# POSTGRES_TEST_URL enables the tier on its own, deliberately. If it only
# *selected the provider* and RUN_PG_TESTS still had to be set as well, a CI job
# that supplied the server but forgot the flag would skip every test in this
# file and report green — a blind job that looks covered is worse than a
# missing one. One variable is enough to say "there is a Postgres here".
_PG_ENABLED = bool(os.getenv("RUN_PG_TESTS") or os.getenv("POSTGRES_TEST_URL"))

pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(
        not _PG_ENABLED,
        reason="pg tier is opt-in: set RUN_PG_TESTS=1 (Docker) or POSTGRES_TEST_URL",
    ),
]

# Must match the postgres service image in .github/workflows/ci.yml's
# pg-integration job — `just pins` asserts they agree, so this is enforcement
# rather than a "keep these in step" comment.
_PG_IMAGE = "postgres:18-alpine"
_dbname_counter = itertools.count(1)


def _with_database(dsn: str, name: str) -> str:
    """The same DSN pointed at a different database.

    urlsplit rather than rsplit("/", 1): an externally supplied DSN may carry a
    query string (?sslmode=require) or a password containing a slash, and a
    naive split corrupts both.
    """
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{name}"))


@pytest.fixture(scope="session")
def admin_dsn() -> Iterator[str]:
    """A Postgres this suite may CREATE and DROP databases on.

    Two providers behind one fixture. POSTGRES_TEST_URL wins when set: CI runs a
    GitHub Actions *service container*, which the runner pulls and health-checks
    during job setup — before the first step executes — so paying testcontainers'
    pull, container start and ryuk-reaper cost again inside the job would buy
    nothing. Unset (the local default) falls back to testcontainers, which is
    what keeps this tier zero-setup for a developer who just has Docker running.

    Session-scoped and synchronous on purpose: the value is a plain string, so
    it survives the per-function event loop that asyncio_default_fixture_loop_scope
    pins us to. A pooled *connection* here would not.
    """
    external = os.getenv("POSTGRES_TEST_URL")
    if external:
        yield external
        return

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(_PG_IMAGE, username="test", password="test") as pg:
        yield (
            f"postgresql://test:test@{pg.get_container_host_ip()}"
            f":{pg.get_exposed_port(5432)}/{pg.dbname}"
        )


@pytest.fixture
async def pg_dsn(admin_dsn: str) -> AsyncIterator[str]:
    """A fresh database per test — full isolation, ~ms to create.

    Connects per operation instead of holding one admin connection open for the
    session: fixture loop scope is "function", so a connection opened in one
    test's loop is unusable in the next. Two extra connects are microseconds
    against CREATE DATABASE — do not "optimize" this into a session fixture.
    """
    import asyncpg

    name = f"t{next(_dbname_counter)}"
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f"CREATE DATABASE {name}")
    finally:
        await conn.close()
    yield _with_database(admin_dsn, name)
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f"DROP DATABASE {name} WITH (FORCE)")
    finally:
        await conn.close()


@pytest.fixture
async def archive(pg_dsn: str) -> AsyncIterator[PostgresHistoryArchive]:
    a = PostgresHistoryArchive(pg_dsn)
    yield a
    await a.close()


def _entry(
    n: int, *, guild_id: int = 42, played_at: float | None = None
) -> HistoryEntry:
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


class TestSchemaBootstrap:
    async def test_ensure_creates_table_and_dedup_index(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # _SCHEMA_DDL runs against a real server for the first time here — a typo
        # in the DDL surfaces as a raised error rather than shipping silently.
        pool = await archive._ensure()
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval("SELECT to_regclass('play_history')::text")
                == "play_history"
            )
            assert (
                await conn.fetchval("SELECT to_regclass('play_history_dedup')::text")
                == "play_history_dedup"
            )

    async def test_ensure_is_idempotent(self, archive: PostgresHistoryArchive) -> None:
        # IF NOT EXISTS + double-checked lock: the second call reuses the pool
        # and never re-runs DDL destructively.
        p1 = await archive._ensure()
        p2 = await archive._ensure()
        assert p1 is p2


class TestInsertAndRecent:
    async def test_roundtrip_all_fields(self, archive: PostgresHistoryArchive) -> None:
        e = _entry(1)
        await archive.insert_batch([e])
        got = await archive.recent(42, 10)
        assert got == [e]  # every column survives insert -> select intact

    async def test_newest_first_ordering(self, archive: PostgresHistoryArchive) -> None:
        entries = [_entry(i) for i in range(5)]  # played_at increases with i
        await archive.insert_batch(entries)
        got = await archive.recent(42, 10)
        assert got == list(reversed(entries))  # newest (Song 4) first

    async def test_limit_caps_result(self, archive: PostgresHistoryArchive) -> None:
        await archive.insert_batch([_entry(i) for i in range(10)])
        got = await archive.recent(42, 3)
        assert [g.title for g in got] == ["Song 9", "Song 8", "Song 7"]

    async def test_recent_filters_by_guild(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(1, guild_id=1), _entry(2, guild_id=2)])
        assert [e.guild_id for e in await archive.recent(1, 10)] == [1]

    async def test_nonpositive_limit_returns_empty(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(1)])
        assert await archive.recent(42, 0) == []
        assert await archive.recent(42, -5) == []

    async def test_empty_insert_is_noop(self, archive: PostgresHistoryArchive) -> None:
        await archive.insert_batch([])
        assert await archive.recent(42, 10) == []


class TestDedupAndPrecision:
    async def test_on_conflict_dedup(self, archive: PostgresHistoryArchive) -> None:
        # Same (guild_id, played_at, webpage_url) inserted twice — the redelivery
        # / backfill-overlap case the play_history_dedup index exists to collapse.
        e = _entry(1)
        await archive.insert_batch([e])
        await archive.insert_batch([e])
        assert len(await archive.recent(42, 10)) == 1

    async def test_timestamptz_preserves_microseconds(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # A sub-second played_at must survive the timestamptz round-trip to µs.
        e = _entry(1, played_at=1752530000.123456)
        await archive.insert_batch([e])
        [got] = await archive.recent(42, 10)
        assert got.played_at == pytest.approx(1752530000.123456, abs=1e-6)

    async def test_epoch_zero_tiebreak_is_stable(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Unknown-time entries all land at played_at=0 (distinct URLs, so no
        # dedup); ORDER BY played_at DESC, id DESC gives newest-inserted first.
        es = [_entry(i, played_at=0.0) for i in range(3)]
        await archive.insert_batch(es)
        got = await archive.recent(42, 10)
        assert [g.title for g in got] == ["Song 2", "Song 1", "Song 0"]


class TestClose:
    async def test_close_is_final_and_data_outlives_the_instance(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        """close() is terminal for the instance, not for the data.

        This used to assert the opposite — that a post-close op lazily rebuilt
        the pool and still read its rows. That reuse was incidental (close() is
        only ever called from MusicBotApp.close(), at shutdown) and became unsafe
        once health_check() gave -ping a path into _ensure() that shutdown does
        not sequence: a probe racing close() would have built a pool nothing was
        left to close. What actually mattered in the old test — the rows survive
        — is asserted here through a fresh archive on the same DSN.
        """
        await archive.insert_batch([_entry(1)])
        await archive.close()
        with pytest.raises(RuntimeError, match="closed"):
            await archive.recent(42, 10)

        reopened = PostgresHistoryArchive(pg_dsn)
        try:
            [got] = await reopened.recent(42, 10)
            assert got.title == "Song 1"
        finally:
            await reopened.close()

    async def test_health_check_against_a_live_server(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # -ping's Postgres row, end to end against a real server: connects on
        # first use (nothing has touched the archive yet in this test) and
        # returns without raising, which is what probe_postgres times.
        await archive.health_check()

    async def test_close_is_idempotent(self, archive: PostgresHistoryArchive) -> None:
        await archive._ensure()
        await archive.close()
        await archive.close()  # must not raise
