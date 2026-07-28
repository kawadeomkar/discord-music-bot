"""Real-Postgres integration tier (docs/POSTGRES_HISTORY_PLAN.md §9).

Opt-in, with two ways to supply the server — see `admin_dsn` below:

    just test-pg                        # local: testcontainers, needs Docker
    POSTGRES_TEST_URL=... just test-pg  # CI: an already-running server

A throwaway database per test gives isolation either way.

Covers exactly what the in-memory fakes cannot: the migrations actually
executing, PostgresHistoryArchive's _INSERT_SQL/_RECENT_SQL parameter binding
against a real server, ON CONFLICT dedup, the timestamptz<->epoch round-trip,
recent()'s newest-first ordering with the id tie-break, and the poison-entry
classes (NUL bytes, out-of-range epochs) that only a real Postgres rejects the
way production will. Until this tier runs, those SQL constants and the
sanitizer's premises are validated only by inspection.
"""

import asyncio
import itertools
import os
from collections.abc import AsyncIterator, Iterator
from urllib.parse import urlsplit, urlunsplit

import pytest

from src.db_migrate import EXPECTED_SCHEMA_VERSION, migrate
from src.guild_state import HistoryEntry
from src.history_archive import (
    _RECENT_SQL,
    HistoryOutboxDrainer,
    PostgresHistoryArchive,
    SchemaVersionError,
)
from src.redis_client import HISTORY_DLQ_KEY, HISTORY_OUTBOX_KEY, GuildRedisStore

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
async def raw_pg_dsn(admin_dsn: str) -> AsyncIterator[str]:
    """A fresh, UNMIGRATED database per test — full isolation, ~ms to create.

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
async def pg_dsn(raw_pg_dsn: str) -> str:
    """A fresh database with the schema applied, via the real runner.

    The archive no longer creates its own schema, so the migration step is now
    part of the setup rather than a side effect of first use — which also means
    every run of this tier exercises src/db_migrate.py end to end against a
    real server, not just the code that reads what it produced.
    """
    await migrate(raw_pg_dsn)
    return raw_pg_dsn


@pytest.fixture
async def archive(pg_dsn: str) -> AsyncIterator[PostgresHistoryArchive]:
    a = PostgresHistoryArchive(pg_dsn)
    yield a
    await a.close()


def _entry(
    n: int,
    *,
    guild_id: int = 42,
    played_at: float | None = None,
    title: str | None = None,
) -> HistoryEntry:
    return HistoryEntry(
        guild_id=guild_id,
        title=f"Song {n}" if title is None else title,
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
    async def test_runner_creates_table_and_dedup_index(self, pg_dsn: str) -> None:
        # The migrations run against a real server here — a typo in one of them
        # surfaces as a raised error rather than shipping silently.
        import asyncpg

        conn = await asyncpg.connect(pg_dsn)
        try:
            assert (
                await conn.fetchval("SELECT to_regclass('play_history')::text")
                == "play_history"
            )
            assert (
                await conn.fetchval("SELECT to_regclass('play_history_dedup')::text")
                == "play_history_dedup"
            )
            assert (
                await conn.fetchval("SELECT max(version) FROM schema_migrations")
                == EXPECTED_SCHEMA_VERSION
            )
        finally:
            await conn.close()

    async def test_inserted_at_column_exists_and_defaults(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        # 0002. Not part of HistoryEntry — the default is what fills it, which
        # is exactly what makes it useful for outage forensics.
        import asyncpg

        await archive.insert_batch([_entry(1)])
        conn = await asyncpg.connect(pg_dsn)
        try:
            assert (
                await conn.fetchval("SELECT inserted_at FROM play_history") is not None
            )
        finally:
            await conn.close()

    async def test_rerunning_the_migrator_is_a_noop(self, pg_dsn: str) -> None:
        # `just db-migrate` on an up-to-date database, and the compose one-shot
        # on every `up`, both take this path.
        assert await migrate(pg_dsn) == EXPECTED_SCHEMA_VERSION
        assert await migrate(pg_dsn) == EXPECTED_SCHEMA_VERSION

    async def test_migrator_is_safe_under_concurrent_runs(
        self, raw_pg_dsn: str
    ) -> None:
        """Two pods (or a manual run racing the compose one-shot) starting at
        once is the normal case; the advisory lock is what makes the loser wait
        rather than double-apply.

        raw_pg_dsn, NOT pg_dsn: this used to take the already-migrated fixture,
        so all four racers hit the `IF NOT EXISTS` short-circuit and the test
        passed without ever exercising a concurrent bootstrap. Against a VIRGIN
        database it reproduced a ~50% failure rate — the pg_type catalog race
        that `CREATE TABLE IF NOT EXISTS` does not protect against.
        """
        results = await asyncio.gather(*(migrate(raw_pg_dsn) for _ in range(6)))
        assert results == [EXPECTED_SCHEMA_VERSION] * 6

    async def test_migrator_upgrades_a_legacy_first_use_ddl_database(
        self, raw_pg_dsn: str
    ) -> None:
        # The upgrade path for deployments created by the OLD first-use DDL:
        # the table already exists, so 0001 must record itself as applied
        # without failing.
        import asyncpg

        conn = await asyncpg.connect(raw_pg_dsn)
        try:
            await conn.execute(
                "CREATE TABLE play_history ("
                " id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
                " guild_id bigint NOT NULL, title text NOT NULL DEFAULT '',"
                " webpage_url text NOT NULL DEFAULT '',"
                " duration_secs integer NOT NULL DEFAULT 0,"
                " played_secs integer NOT NULL DEFAULT 0,"
                " requester_id bigint NOT NULL DEFAULT 0,"
                " requester_name text NOT NULL DEFAULT '',"
                " thumbnail text NOT NULL DEFAULT '',"
                " uploader text NOT NULL DEFAULT '',"
                " played_at timestamptz NOT NULL DEFAULT to_timestamp(0))"
            )
        finally:
            await conn.close()
        assert await migrate(raw_pg_dsn) == EXPECTED_SCHEMA_VERSION

    async def test_archive_refuses_an_unmigrated_database(
        self, raw_pg_dsn: str
    ) -> None:
        # The archive stopped creating its own schema; it verifies instead, and
        # the failure has to name the fix.
        archive = PostgresHistoryArchive(raw_pg_dsn)
        try:
            with pytest.raises(SchemaVersionError, match="db-migrate"):
                await archive.insert_batch([_entry(1)])
        finally:
            await archive.close()

    async def test_ensure_is_idempotent(self, archive: PostgresHistoryArchive) -> None:
        # Double-checked lock: the second call reuses the pool.
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

    async def test_recent_is_planned_as_a_plain_index_scan(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        """The -history query must be served by play_history_recent, with NO
        sort node.

        play_history_dedup (guild_id, played_at, webpage_url) only presorts
        played_at, so the id DESC tie-break planned an Incremental Sort that had
        to consume an ENTIRE equal-played_at group before emitting LIMIT 50.
        Those groups are large in practice — every legacy row backfilled from
        Redis lands on the epoch-0 sentinel — and it measured 37x slower (p50
        49.98ms vs 1.34ms) with ~9,900 buffers touched per call.

        Asserting the PLAN, not a duration: a timing threshold on a
        few-hundred-row test table would be noise, and the defect was never
        about absolute speed — it was about the plan degrading with group size.
        """
        import asyncpg

        await archive.insert_batch([_entry(i, played_at=0.0) for i in range(200)])
        conn = await asyncpg.connect(pg_dsn)
        try:
            await conn.execute("ANALYZE play_history")
            rows = await conn.fetch(f"EXPLAIN {_RECENT_SQL}", 42, 50)
        finally:
            await conn.close()
        plan = "\n".join(r["QUERY PLAN"] for r in rows)
        assert "play_history_recent" in plan, plan
        assert "Sort" not in plan, plan  # covers Sort and Incremental Sort


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


class TestPoisonAgainstARealServer:
    """The sanitizer's premises are claims about Postgres, so they are only
    really tested here. Each case is a vector that reproducibly wedged the
    outbox before C1."""

    async def test_nul_in_text_is_stripped_and_inserted(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Unsanitized this raises CharacterNotInRepertoireError server-side and
        # fails the whole executemany.
        entry = _entry(1, title="ab\x00cd")
        await archive.insert_batch([entry])
        [got] = await archive.recent(42, 10)
        assert got.title == "abcd"

    async def test_out_of_range_epoch_lands_on_the_unknown_sentinel(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(1, played_at=1e18)])
        [got] = await archive.recent(42, 10)
        assert got.played_at == 0.0

    async def test_negative_epoch_lands_on_the_unknown_sentinel(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(1, played_at=-1e18)])
        [got] = await archive.recent(42, 10)
        assert got.played_at == 0.0

    async def test_one_poison_entry_does_not_block_a_healthy_batch(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # The actual C1 failure: good/poison/good in one executemany used to
        # deliver nothing at all.
        await archive.insert_batch(
            [
                _entry(1),
                _entry(2, title="bad\x00title"),
                _entry(3),
            ]
        )
        assert len(await archive.recent(42, 10)) == 3


class TestDrainerEndToEnd:
    """The full path a played song takes: GuildRedisStore.push_history →
    Redis outbox → drainer → Postgres. Every layer real except Discord."""

    async def test_entries_reach_postgres_through_the_drainer(
        self, archive: PostgresHistoryArchive, fake_redis: object
    ) -> None:
        store = GuildRedisStore(fake_redis, guild_id=42)  # type: ignore[arg-type]
        entries = [_entry(i) for i in range(3)]
        for entry in entries:
            await store.push_history(entry)

        drainer = HistoryOutboxDrainer(fake_redis, archive)  # type: ignore[arg-type]
        drainer.start()
        try:
            drainer.notify()
            async with asyncio.timeout(10):
                while await fake_redis.llen(HISTORY_OUTBOX_KEY):  # type: ignore[attr-defined]
                    await asyncio.sleep(0.05)
        finally:
            await drainer.stop()

        got = await archive.recent(42, 10)
        assert [e.title for e in got] == ["Song 2", "Song 1", "Song 0"]

    async def test_poison_is_quarantined_while_the_rest_is_delivered(
        self,
        archive: PostgresHistoryArchive,
        fake_redis: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scenario B against a real server: the poison rows land in the DLQ,
        the healthy ones land in Postgres, and the outbox empties.

        Sanitization is neutralized (identity) rather than the archive being
        replaced, so this exercises the real insert_batch — including its
        RowConversionError wrapping, which is what makes the drainer's poison
        taxonomy portable. A hand-rolled stand-in archive skipped that wrapping
        and let a macOS-only `OSError: [Errno 84]` escape as a transient error,
        which is exactly the misclassification the wrapping exists to prevent.
        """
        import src.history_archive as history_archive

        monkeypatch.setattr(history_archive, "_sanitize_entry", lambda e: e)

        store = GuildRedisStore(fake_redis, guild_id=42)  # type: ignore[arg-type]
        for entry in (
            _entry(1),
            _entry(2, title="bad\x00title"),  # server-side DataError
            _entry(3),
            _entry(4, played_at=1e18),  # row conversion blows up
            _entry(5),
        ):
            await store.push_history(entry)

        drainer = HistoryOutboxDrainer(fake_redis, archive)  # type: ignore[arg-type]
        assert await drainer._drain_once() == 5

        assert [e.title for e in await archive.recent(42, 10)] == [
            "Song 5",
            "Song 3",
            "Song 1",
        ]
        assert await fake_redis.llen(HISTORY_DLQ_KEY) == 2  # type: ignore[attr-defined]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0  # type: ignore[attr-defined]
