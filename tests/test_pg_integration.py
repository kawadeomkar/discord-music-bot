"""Real-Postgres integration tier (docs/POSTGRES_HISTORY_PLAN.md §9).

Opt-in: skipped unless RUN_PG_TESTS=1 (needs Docker). One postgres:18-alpine
testcontainer per session; a throwaway database per test for isolation.

    RUN_PG_TESTS=1 poetry run pytest -m pg --no-cov

Covers exactly what the in-memory fakes cannot: PostgresHistoryArchive against a
real server — _SCHEMA_DDL actually executing, the _INSERT_SQL/_RECENT_SQL
parameter binding, ON CONFLICT dedup, the timestamptz<->epoch round-trip, and
recent()'s newest-first ordering with the id tie-break. Until this tier runs,
those SQL constants are validated only by inspection.
"""

import itertools
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from src.guild_state import HistoryEntry
from src.history_archive import PostgresHistoryArchive

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
def pg_container() -> Iterator[Any]:
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(_PG_IMAGE, username="test", password="test") as pg:
        yield pg


@pytest.fixture
def admin_dsn(pg_container: Any) -> str:
    host = pg_container.get_container_host_ip()
    port = pg_container.get_exposed_port(5432)
    return f"postgresql://test:test@{host}:{port}/{pg_container.dbname}"


@pytest.fixture
async def pg_dsn(pg_container: Any, admin_dsn: str) -> AsyncIterator[str]:
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
    async def test_close_then_reuse_lazily_rebuilds(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(1)])
        await archive.close()
        # A subsequent op rebuilds the pool via _ensure and still sees the data.
        [got] = await archive.recent(42, 10)
        assert got.title == "Song 1"

    async def test_close_is_idempotent(self, archive: PostgresHistoryArchive) -> None:
        await archive._ensure()
        await archive.close()
        await archive.close()  # must not raise
