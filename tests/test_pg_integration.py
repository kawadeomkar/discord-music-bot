"""Real-Postgres integration tier.

Opt-in; two ways to supply the server (see `admin_dsn`):

    just test-pg                        # local: testcontainers, needs Docker
    POSTGRES_TEST_URL=... just test-pg  # CI: an already-running server

A throwaway database per test gives isolation either way.

Covers what the in-memory fakes cannot: the migrations actually executing,
_INSERT_SQL/_RECENT_SQL binding against a real server, ON CONFLICT dedup, the
timestamptz<->epoch round-trip, recent()'s newest-first ordering with the id
tie-break, and the poison classes (NUL bytes, out-of-range epochs) only a real
Postgres rejects the way production will.
"""

import asyncio
import itertools
import os
from dataclasses import replace
from collections.abc import AsyncIterator, Iterator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from redis.asyncio import Redis

from src.db_migrate import EXPECTED_SCHEMA_VERSION, migrate
from src.backfill_history import backfill
from src.guild_state import (
    HistoryEntry,
    parse_history_entry,
    serialize_history_entry,
)
from src.history_archive import (
    _POISON,
    _RECENT_SQL,
    HistoryOutboxDrainer,
    PostgresHistoryArchive,
    SchemaVersionError,
)
from src.redis_client import (
    GUILD_HISTORY_KEY,
    HISTORY_OUTBOX_KEY,
    GuildRedisStore,
    ensure_outbox_group,
)

from tests.helpers import bind_loopback_only, tier_enabled

# POSTGRES_TEST_URL enables the tier on its own: if it only selected the
# provider, a CI job that supplied the server but forgot RUN_PG_TESTS would skip
# every test here and report green, and a blind job that looks covered is worse
# than a missing one.
_PG_ENABLED = tier_enabled("RUN_PG_TESTS", "POSTGRES_TEST_URL")

pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(
        not _PG_ENABLED,
        reason="pg tier is opt-in: set RUN_PG_TESTS=1 (Docker) or POSTGRES_TEST_URL",
    ),
]

# Must match the postgres service image in ci.yml's pg-integration job —
# `just pins` asserts they agree.
_PG_IMAGE = "postgres:18-alpine"
_dbname_counter = itertools.count(1)


def _with_database(dsn: str, name: str) -> str:
    """The same DSN pointed at a different database. urlsplit rather than
    rsplit("/", 1): a supplied DSN may carry a query string (?sslmode=require)
    or a password containing a slash, and a naive split corrupts both."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{name}"))


@pytest.fixture(scope="session")
def admin_dsn() -> Iterator[str]:
    """A Postgres this suite may CREATE and DROP databases on. POSTGRES_TEST_URL
    wins when set (CI's service container is already pulled and health-checked);
    unset, testcontainers keeps the tier zero-setup for anyone running Docker.
    Session-scoped and synchronous: a plain string survives the per-function
    fixture loop scope, a pooled connection would not."""
    external = os.getenv("POSTGRES_TEST_URL")
    if external:
        yield external
        return

    from testcontainers.postgres import PostgresContainer

    pg = PostgresContainer(_PG_IMAGE, username="test", password="test")
    bind_loopback_only(pg, 5432)
    with pg:
        yield (
            f"postgresql://test:test@{pg.get_container_host_ip()}"
            f":{pg.get_exposed_port(5432)}/{pg.dbname}"
        )


@pytest.fixture
async def raw_pg_dsn(admin_dsn: str) -> AsyncIterator[str]:
    """A fresh, UNMIGRATED database per test — full isolation, ~ms to create.
    Connects per operation rather than holding an admin connection open: fixture
    loop scope is "function", so a connection from one test's loop is unusable in
    the next. Do not "optimize" this into a session fixture."""
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
    """A fresh database with the schema applied, via the real runner — so every
    run of this tier exercises src/db_migrate.py end to end against a real
    server, not just the code that reads what it produced."""
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
        queued_at=1752529000.0 + n,
        queue_position=n,
    )


def _play(
    *,
    guild_id: int = 42,
    url: str,
    title: str = "Song",
    requester_id: int = 1,
    requester_name: str = "user",
    played_secs: int = 100,
    duration_secs: int = 200,
    played_at: float,
    query_source: str = "",
) -> HistoryEntry:
    """A leaderboard fixture row: everything the aggregates group or rank by is
    a parameter, and played_at is required because play_history_dedup collapses
    two plays of one URL that share it."""
    return HistoryEntry(
        guild_id=guild_id,
        title=title,
        webpage_url=url,
        duration_secs=duration_secs,
        played_secs=played_secs,
        requester_id=requester_id,
        requester_name=requester_name,
        played_at=played_at,
        query_source=query_source,
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
        # Not part of HistoryEntry — the default is what fills it, which is
        # exactly what makes it useful for outage forensics.
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
        once is normal, and the advisory lock makes the loser wait rather than
        double-apply. raw_pg_dsn, not pg_dsn: on an already-migrated database
        every racer hits the `IF NOT EXISTS` short-circuit and no concurrent
        bootstrap happens; on a VIRGIN one this hits the pg_type catalog race
        `CREATE TABLE IF NOT EXISTS` does not protect against, ~50% of runs."""
        results = await asyncio.gather(*(migrate(raw_pg_dsn) for _ in range(6)))
        assert results == [EXPECTED_SCHEMA_VERSION] * 6

    async def test_migrated_schema_accepts_every_column_the_archive_writes(
        self, raw_pg_dsn: str
    ) -> None:
        """The runner leaving a USABLE table, not just a version number. Asserting
        the return value alone cannot see a schema that drifted from _INSERT_SQL —
        that shape passes the version check and then fails every insert with an
        UndefinedColumnError the drainer treats as transient and retries forever.
        Only executing an insert afterwards catches it."""
        assert await migrate(raw_pg_dsn) == EXPECTED_SCHEMA_VERSION
        archive = PostgresHistoryArchive(raw_pg_dsn)
        try:
            await archive.insert_batch([_entry(7)])
            rows = await archive.recent(42, 1)
        finally:
            await archive.close()
        assert rows[0].queue_position == 7
        assert rows[0].queued_at == 1752529007.0

    async def test_migrator_upgrades_a_legacy_first_use_ddl_database(
        self, raw_pg_dsn: str
    ) -> None:
        # The upgrade path for deployments created by the OLD first-use DDL:
        # the table already exists, so 0001 must record itself as applied
        # without failing. Shape tolerance only — that DDL never shipped in a
        # release, so no such database is reachable, and IF NOT EXISTS leaves the
        # table it created missing every column 0001 grew afterwards.
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
        """The -history query must be served by play_history_recent with NO sort
        node: play_history_dedup only presorts played_at, so the id DESC
        tie-break plans an Incremental Sort that consumes an ENTIRE
        equal-played_at group (large in practice — every row backfilled from
        Redis lands on the epoch-0 sentinel) before emitting LIMIT 50. The PLAN
        is asserted because the defect is degradation with group size, not
        absolute speed, and a timing threshold on a small table would be noise."""
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
        """close() is terminal for the instance, not for the data: a post-close
        op must raise rather than lazily rebuild the pool, because health_check()
        gives -ping a path into _ensure() that shutdown does not sequence — a
        probe racing close() would build a pool nothing is left to close. The
        rows surviving is asserted through a fresh archive on the same DSN."""
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
    """The sanitizer's premises are claims about Postgres, so they can only be
    tested here. Each case is a vector that reproducibly wedged the outbox."""

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
        # good/poison/good in one executemany used to deliver nothing at all.
        await archive.insert_batch(
            [
                _entry(1),
                _entry(2, title="bad\x00title"),
                _entry(3),
            ]
        )
        assert len(await archive.recent(42, 10)) == 3


def _boundary_entries() -> list[HistoryEntry]:
    """Every clamp __post_init__ performs, one entry each. Enumerated rather than
    sampled: this list is the claim that the residual poison set is empty, so
    extend it whenever a column type or a domain tuple changes. Distinct
    webpage_urls keep play_history_dedup from collapsing rows that clamp alike."""
    return [
        HistoryEntry(guild_id=7, webpage_url="u/clean", played_at=1752530000.0),
        # int8 columns fed values orjson accepts but a SIGNED bigint refuses.
        HistoryEntry(guild_id=2**63, webpage_url="u/g63", played_at=1752530001.0),
        HistoryEntry(guild_id=2**64 - 1, webpage_url="u/g64", played_at=1752530002.0),
        HistoryEntry(guild_id=7, requester_id=2**63, webpage_url="u/r63"),
        HistoryEntry(guild_id=7, requester_id=-1, webpage_url="u/rneg"),
        # int4 columns.
        HistoryEntry(guild_id=7, duration_secs=2**31, webpage_url="u/d31"),
        HistoryEntry(guild_id=7, played_secs=-1, webpage_url="u/pneg"),
        # played_at, including the inclusive ceiling.
        HistoryEntry(guild_id=7, webpage_url="u/huge", played_at=1e300),
        HistoryEntry(guild_id=7, webpage_url="u/negts", played_at=-1e12),
        HistoryEntry(guild_id=7, webpage_url="u/nan", played_at=float("nan")),
        HistoryEntry(guild_id=7, webpage_url="u/ceil", played_at=253402300799.0),
        # queued_at rides the same _EPOCH_FIELDS clamp; queue_position is int4.
        HistoryEntry(guild_id=7, webpage_url="u/qhuge", queued_at=1e300),
        HistoryEntry(guild_id=7, webpage_url="u/qnan", queued_at=float("nan")),
        HistoryEntry(guild_id=7, webpage_url="u/qceil", queued_at=253402300799.0),
        HistoryEntry(guild_id=7, webpage_url="u/qpneg", queue_position=-1),
        HistoryEntry(guild_id=7, webpage_url="u/qp31", queue_position=2**31),
        # NUL in each text field.
        HistoryEntry(guild_id=7, webpage_url="u/t", title="a\x00b"),
        HistoryEntry(guild_id=7, webpage_url="u/w\x00url"),
        HistoryEntry(guild_id=7, webpage_url="u/rn", requester_name="a\x00b"),
        HistoryEntry(guild_id=7, webpage_url="u/th", thumbnail="a\x00b"),
        HistoryEntry(guild_id=7, webpage_url="u/up", uploader="a\x00b"),
    ]


class TestSchemaLock:
    """The play_history CHECK constraints plus HistoryEntry.__post_init__ — both
    directions of the claim that possessing a HistoryEntry is the proof Postgres
    accepts it."""

    async def test_every_constructible_entry_is_insertable(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Forward: everything __post_init__ produces, Postgres accepts. Inserted
        # one at a time so a failure names the offending entry instead of
        # failing the whole executemany.
        entries = _boundary_entries()
        for entry in entries:
            await archive.insert_batch([entry])  # must not raise
        landed = len(await archive.recent(7, 100)) + len(
            await archive.recent(2**63 - 1, 100)
        )
        assert landed == len(entries)

    @pytest.mark.parametrize(
        ("field", "value", "constraint"),
        [
            ("guild_id", 0, "play_history_guild_id_valid"),
            ("requester_id", -1, "play_history_requester_valid"),
            ("played_secs", -1, "play_history_played_secs_valid"),
            ("duration_secs", -1, "play_history_duration_valid"),
            ("message_id", -1, "play_history_message_id_valid"),
            ("queue_position", -1, "play_history_queue_position_valid"),
            # The epoch floor -leaderboard's all-time cutoff relies on. Without
            # these CHECKs it is asserted only by the validator, so a pre-epoch
            # played_at would sort ahead of every real play unremarked.
            ("played_at", -1.0, "play_history_played_at_valid"),
            ("queued_at", -1.0, "play_history_queued_at_valid"),
            # Unlike title/uploader this column holds only machine-minted
            # tokens, so an out-of-domain value means the normalizer regressed.
            ("query_source", "NOT A HOST", "play_history_query_source_valid"),
        ],
    )
    async def test_constraints_catch_a_validator_regression(
        self,
        archive: PostgresHistoryArchive,
        field: str,
        value: object,
        constraint: str,
    ) -> None:
        # Backward: bypass the validator and the database still refuses. Every
        # CHECK gets its own case; object.__setattr__ is how a regression would
        # present (a dropped clamp, or a field added to the dataclass but not to
        # a domain tuple). The constraint name is asserted, not just the class —
        # a row can violate several CHECKs at once and the server reports
        # whichever it evaluates first, so a bare raises() would pass even with
        # the constraint under test dropped from the schema entirely.
        bad = HistoryEntry(guild_id=1, webpage_url="https://x", played_at=0.0)
        object.__setattr__(bad, field, value)
        with pytest.raises(asyncpg.exceptions.CheckViolationError) as exc:
            await archive.insert_batch([bad])
        # as_dict() rather than .constraint_name: asyncpg sets the message
        # fields dynamically, so the attribute is real at runtime but invisible
        # to pyright, and the dict is the documented way in.
        assert exc.value.as_dict()["constraint_name"] == constraint

    async def test_guild_id_zero_is_constructible_and_refused(
        self, archive: PostgresHistoryArchive
    ) -> None:
        """The one deliberate hole in "constructible implies insertable":
        __post_init__ floors every integer at 0 while the CHECK is strictly
        `> 0`, so this entry is built without bypassing the validator and still
        refused. Both halves are choices — clamping up to 1 would file an
        unattributable play into a real guild's history, relaxing the CHECK would
        make guild 0 a bucket of orphans no read path excludes. Refusing routes
        it to play_history_rejected, where `just db-rejects` shows an operator
        the actual defect: a producer that stopped stamping guild_id."""
        orphan = HistoryEntry(guild_id=0, webpage_url="https://x", played_at=0.0)
        assert orphan.guild_id == 0  # constructed, not clamped away
        with pytest.raises(asyncpg.exceptions.CheckViolationError) as exc:
            await archive.insert_batch([orphan])
        assert exc.value.as_dict()["constraint_name"] == "play_history_guild_id_valid"

    async def test_a_check_violation_is_classified_as_poison(self) -> None:
        # CheckViolationError is SQLSTATE 23514 under
        # IntegrityConstraintViolationError, not a DataError. Without that arm in
        # _POISON a violation propagates past the quarantine path and wedges the
        # drain head permanently on a non-evictable stream.
        assert issubclass(
            asyncpg.exceptions.CheckViolationError,
            asyncpg.exceptions.IntegrityConstraintViolationError,
        )
        assert not issubclass(
            asyncpg.exceptions.CheckViolationError, asyncpg.exceptions.DataError
        )
        assert isinstance(asyncpg.exceptions.CheckViolationError("x"), _POISON)

    async def test_not_valid_constraints_do_not_reject_legacy_rows(
        self, pg_dsn: str
    ) -> None:
        # The NOT VALID promise against a row that predates the constraint: drop
        # it, write a violating row, re-add it NOT VALID, and confirm both the
        # ADD and a later read succeed. This is what catches a migration that
        # would fail on real data.
        conn = await asyncpg.connect(pg_dsn)
        try:
            await conn.execute(
                "ALTER TABLE play_history DROP CONSTRAINT play_history_guild_id_valid"
            )
            await conn.execute(
                "INSERT INTO play_history (guild_id, title, webpage_url, "
                "duration_secs, played_secs, requester_id, requester_name, "
                "thumbnail, uploader, played_at) VALUES "
                "(0, 'legacy', 'https://legacy', 0, 0, 0, '', '', '', now())"
            )
            await conn.execute(
                "ALTER TABLE play_history ADD CONSTRAINT play_history_guild_id_valid "
                "CHECK (guild_id > 0) NOT VALID"
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM play_history WHERE guild_id = 0"
                )
                == 1
            )
            # And the constraint is live for new rows even while unvalidated.
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO play_history (guild_id, title, webpage_url, "
                    "duration_secs, played_secs, requester_id, requester_name, "
                    "thumbnail, uploader, played_at) VALUES "
                    "(0, 'new', 'https://new', 0, 0, 0, '', '', '', now())"
                )
        finally:
            await conn.close()


class TestRejectsTable:
    """play_history_rejected against a real server. The column-type choice is a claim
    about Postgres, so it can only be settled here."""

    async def test_a_nul_bearing_payload_round_trips_through_bytea(
        self, pg_dsn: str
    ) -> None:
        # the reason payload is bytea. A NUL byte is the first poison vector, and
        # both jsonb and text refuse it — so either would fail to store exactly
        # the class of row this table exists to capture.
        entry = _entry(1)
        raw = b'{"title":"bad\x00title","played_at":1.0}'
        conn = await asyncpg.connect(pg_dsn)
        try:
            await conn.execute(
                "INSERT INTO play_history_rejected (guild_id, error_type, "
                "error_detail, trace_id, payload) VALUES ($1, $2, $3, $4, $5)",
                entry.guild_id,
                "CheckViolationError",
                "detail",
                "trace",
                raw,
            )
            stored = await conn.fetchval("SELECT payload FROM play_history_rejected")
            assert bytes(stored) == raw  # byte-for-byte, NUL included
            # And the documented inspection path works on it.
            escaped = await conn.fetchval(
                "SELECT encode(payload, 'escape') FROM play_history_rejected"
            )
            assert "bad" in escaped
        finally:
            await conn.close()

    async def test_jsonb_and_text_would_both_refuse_that_payload(
        self, pg_dsn: str
    ) -> None:
        # The counterfactual, asserted rather than argued: this is what makes the
        # bytea column choice evidence. Built with chr(0) because a source file
        # containing a real NUL byte cannot be compiled at all.
        nul = chr(0)
        conn = await asyncpg.connect(pg_dsn)
        try:
            with pytest.raises(asyncpg.exceptions.PostgresError):
                await conn.execute("SELECT $1::text", f"bad{nul}title")
            with pytest.raises(asyncpg.exceptions.PostgresError):
                await conn.execute("SELECT $1::jsonb", '{"t":"bad' + nul + '"}')
        finally:
            await conn.close()

    async def test_record_rejection_writes_a_row_against_a_real_server(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        entry = _entry(1, title="rejected song")
        await archive.record_rejection(
            entry, asyncpg.exceptions.CheckViolationError("guild_id > 0"), "tr-1"
        )
        conn = await asyncpg.connect(pg_dsn)
        try:
            row = await conn.fetchrow(
                "SELECT guild_id, error_type, error_detail, trace_id, payload, "
                "rejected_at FROM play_history_rejected"
            )
            assert row["guild_id"] == 42
            assert row["error_type"] == "CheckViolationError"
            assert "guild_id > 0" in row["error_detail"]
            assert row["trace_id"] == "tr-1"
            assert row["rejected_at"] is not None  # server default
            # Replayable: the stored bytes parse straight back into the entry.
            assert parse_history_entry(bytes(row["payload"])) == entry
        finally:
            await conn.close()

    async def test_a_real_check_violation_can_be_recorded_on_the_same_server(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        """The premise that makes the no-DLQ design work: a REJECTION means the
        server is up and said no, so the connection that carried the refusal can
        carry the reject insert. An OUTAGE never reaches here — the entry stays
        on the outbox and redelivers."""
        bad = HistoryEntry(guild_id=1, webpage_url="https://x", played_at=1.0)
        object.__setattr__(bad, "guild_id", 0)
        try:
            await archive.insert_batch([bad])
            raise AssertionError("expected the CHECK constraint to fire")
        except asyncpg.exceptions.CheckViolationError as e:
            await archive.record_rejection(bad, e, "tr-2")
        conn = await asyncpg.connect(pg_dsn)
        try:
            assert (
                await conn.fetchval("SELECT count(*) FROM play_history_rejected") == 1
            )
            assert await conn.fetchval("SELECT count(*) FROM play_history") == 0
        finally:
            await conn.close()


class TestMessageIdColumn:
    """message_id reaches Postgres. The column is `default 0` with a
    `message_id >= 0` CHECK, so a value dropped between the wire and the INSERT
    lands cleanly and is indistinguishable from a song that genuinely had no Now
    Playing host. Only a real round trip catches that; the drainer tests never
    touch SQL."""

    async def test_a_snowflake_survives_the_round_trip(
        self, archive: PostgresHistoryArchive
    ) -> None:
        entry = _entry(1)
        stamped = replace(entry, message_id=1418765432109876543)
        await archive.insert_batch([stamped])
        (got,) = await archive.recent(42, 10)
        # bigint, and read back as a native int — a float route would round it.
        assert got.message_id == 1418765432109876543

    async def test_an_unstamped_entry_reads_back_as_zero(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(2)])
        (got,) = await archive.recent(42, 10)
        assert got.message_id == 0


class TestEnqueueStampColumns:
    """queued_at / queue_position reach Postgres. Both default to the unknown
    zero value, so a value dropped between the wire and the INSERT lands cleanly
    and reads back indistinguishable from a legacy row — only a round trip
    against a real server catches it."""

    async def test_stamps_survive_the_round_trip(
        self, archive: PostgresHistoryArchive
    ) -> None:
        stamped = replace(_entry(1), queued_at=1752529000.25, queue_position=4)
        await archive.insert_batch([stamped])
        (got,) = await archive.recent(42, 10)
        assert got.queued_at == 1752529000.25
        assert got.queue_position == 4

    async def test_unstamped_entry_reads_back_as_the_unknown_sentinel(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # queued_at 0.0 lands as to_timestamp(0), not NULL — the same sentinel
        # played_at uses, so the zero-value convention holds across the tier.
        await archive.insert_batch(
            [replace(_entry(2), queued_at=0.0, queue_position=0)]
        )
        (got,) = await archive.recent(42, 10)
        assert (got.queued_at, got.queue_position) == (0.0, 0)


class TestQuerySourceColumn:
    """The classification reaches Postgres. Its whole value is telling apart
    three inputs that share a webpage_url, so a value dropped between the wire
    and the INSERT reads back as a plain legacy row and looks like nothing
    happened."""

    async def test_token_survives_the_round_trip(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([replace(_entry(1), query_source="spotify.com")])
        (got,) = await archive.recent(42, 10)
        assert got.query_source == "spotify.com"

    async def test_unstamped_entry_reads_back_as_unknown(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([replace(_entry(2), query_source="")])
        (got,) = await archive.recent(42, 10)
        assert got.query_source == ""


class TestLeaderboard:
    """The aggregates against a real server. fakeredis has no Postgres and
    mocking fetch() would only assert the mock, so every SQL semantic —
    grouping, sentinel exclusion, latest-value pick, tie-breaks, the period
    cutoff — is pinned here or nowhere."""

    async def test_requesters_ranked_by_played_secs(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch(
            [
                _play(url="u/1", requester_id=1, played_secs=100, played_at=1000.0),
                _play(url="u/2", requester_id=1, played_secs=50, played_at=1001.0),
                _play(url="u/3", requester_id=2, played_secs=400, played_at=1002.0),
            ]
        )
        board = await archive.leaderboard(42, 10)
        assert [(r.requester_id, r.played_secs, r.plays) for r in board.requesters] == [
            (2, 400, 1),
            (1, 150, 2),
        ]

    async def test_songs_group_by_url_and_show_the_newest_title(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Titles drift (uploader edits, "[Official Video]" suffixes) while the
        # URL is stable identity, so one URL is one row titled by its last play.
        await archive.insert_batch(
            [
                _play(
                    url="u/1",
                    title="Old Title",
                    duration_secs=200,
                    played_secs=10,
                    played_at=1000.0,
                ),
                _play(
                    url="u/1",
                    title="New Title",
                    duration_secs=210,
                    played_secs=20,
                    played_at=2000.0,
                ),
            ]
        )
        (song,) = (await archive.leaderboard(42, 10)).songs
        assert song.title == "New Title"
        assert song.duration_secs == 210
        assert (song.plays, song.played_secs) == (2, 30)

    async def test_songs_show_the_newest_query_source(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # One song is reachable several ways — pasted once, searched the next
        # time. The board resolves it through the same LATERAL that picks the
        # title, so the label means "how it was most recently asked for".
        await archive.insert_batch(
            [
                _play(
                    url="u/1",
                    played_secs=10,
                    played_at=1000.0,
                    query_source="spotify.com",
                ),
                _play(
                    url="u/1", played_secs=20, played_at=2000.0, query_source="search"
                ),
            ]
        )
        (song,) = (await archive.leaderboard(42, 10)).songs
        assert song.query_source == "search"

    async def test_songs_ranked_by_played_secs(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Listening time, not play count: a 10-minute mix played twice outranks a
        # 30-second clip played ten times, which is the whole ranking choice.
        await archive.insert_batch(
            [
                _play(url="u/short", played_secs=30, played_at=1000.0),
                _play(url="u/short", played_secs=30, played_at=1001.0),
                _play(url="u/short", played_secs=30, played_at=1002.0),
                _play(url="u/long", played_secs=600, played_at=1003.0),
            ]
        )
        board = await archive.leaderboard(42, 10)
        assert [(s.webpage_url, s.played_secs, s.plays) for s in board.songs] == [
            ("u/long", 600, 1),
            ("u/short", 90, 3),
        ]

    async def test_song_ties_break_deterministically(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Equal played_secs -> more plays first, then webpage_url ascending.
        # Without a total order the ranking assertions flake on plan changes.
        await archive.insert_batch(
            [
                _play(url="u/b", played_secs=50, played_at=1000.0),
                _play(url="u/b", played_secs=50, played_at=1001.0),
                _play(url="u/c", played_secs=100, played_at=1002.0),
                _play(url="u/a", played_secs=100, played_at=1003.0),
            ]
        )
        board = await archive.leaderboard(42, 10)
        assert [(s.webpage_url, s.plays) for s in board.songs] == [
            ("u/b", 2),  # same 100 seconds, but two plays
            ("u/a", 1),  # tie on both -> lower url first
            ("u/c", 1),
        ]

    async def test_windowed_board_names_a_song_by_its_newest_title(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # The LATERAL that picks the title deliberately does NOT carry the cutoff:
        # the totals are about the window, but the title is the song's current
        # name. A play outside the window still supplies it.
        await archive.insert_batch(
            [
                _play(url="u/1", title="Old Title", played_at=1000.0),
                _play(url="u/1", title="Renamed", played_at=9000.0),
            ]
        )
        (song,) = (await archive.leaderboard(42, 10, since_epoch=8000.0)).songs
        assert song.title == "Renamed"
        (song,) = (await archive.leaderboard(42, 10, since_epoch=500.0)).songs
        assert song.title == "Renamed"

    async def test_newest_requester_name_wins(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch(
            [
                _play(
                    url="u/1", requester_id=1, requester_name="old", played_at=1000.0
                ),
                _play(
                    url="u/2", requester_id=1, requester_name="new", played_at=2000.0
                ),
            ]
        )
        (leader,) = (await archive.leaderboard(42, 10)).requesters
        assert leader.requester_name == "new"

    async def test_unknown_requester_is_excluded_but_still_counts_for_songs(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # requester_id 0 is the unknown sentinel: one merged pseudo-user would
        # pollute a top-10, while the plays are still real listening time.
        await archive.insert_batch(
            [_play(url="u/1", requester_id=0, played_secs=90, played_at=1000.0)]
        )
        board = await archive.leaderboard(42, 10)
        assert board.requesters == ()
        assert [(s.webpage_url, s.played_secs) for s in board.songs] == [("u/1", 90)]

    async def test_unknown_url_is_excluded_but_still_counts_for_requesters(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch(
            [_play(url="", requester_id=1, played_secs=90, played_at=1000.0)]
        )
        board = await archive.leaderboard(42, 10)
        assert board.songs == ()
        assert [(r.requester_id, r.played_secs) for r in board.requesters] == [(1, 90)]

    async def test_boards_are_per_guild(self, archive: PostgresHistoryArchive) -> None:
        await archive.insert_batch(
            [
                _play(guild_id=1, url="u/1", requester_id=1, played_at=1000.0),
                _play(guild_id=2, url="u/2", requester_id=2, played_at=1000.0),
            ]
        )
        board = await archive.leaderboard(1, 10)
        assert [r.requester_id for r in board.requesters] == [1]
        assert [s.webpage_url for s in board.songs] == ["u/1"]

    async def test_limit_caps_both_boards(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch(
            [
                _play(url=f"u/{i}", requester_id=i, played_secs=i, played_at=1000.0 + i)
                for i in range(1, 6)
            ]
        )
        board = await archive.leaderboard(42, 2)
        assert len(board.requesters) == 2
        assert len(board.songs) == 2

    async def test_ties_break_deterministically(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # Equal played_secs -> more plays first, then the group key ascending.
        # Without this the ordering assertions above could flake on plan changes.
        await archive.insert_batch(
            [
                _play(url="u/a", requester_id=9, played_secs=100, played_at=1000.0),
                _play(url="u/b", requester_id=3, played_secs=50, played_at=1001.0),
                _play(url="u/c", requester_id=3, played_secs=50, played_at=1002.0),
                _play(url="u/d", requester_id=5, played_secs=100, played_at=1003.0),
            ]
        )
        board = await archive.leaderboard(42, 10)
        assert [(r.requester_id, r.plays) for r in board.requesters] == [
            (3, 2),  # same 100 seconds, but two plays
            (5, 1),  # tie on both -> lower id first
            (9, 1),
        ]

    async def test_totals_exceed_int4(self, archive: PostgresHistoryArchive) -> None:
        # sum() over integer returns bigint; a client that narrowed it would
        # overflow here rather than in a year's worth of real plays.
        big = 2_000_000_000
        await archive.insert_batch(
            [
                _play(url="u/1", requester_id=1, played_secs=big, played_at=1000.0),
                _play(url="u/2", requester_id=1, played_secs=big, played_at=1001.0),
            ]
        )
        (leader,) = (await archive.leaderboard(42, 10)).requesters
        assert leader.played_secs == 2 * big

    async def test_empty_guild_returns_empty_boards(
        self, archive: PostgresHistoryArchive
    ) -> None:
        board = await archive.leaderboard(42, 10)
        assert board.requesters == () and board.songs == ()

    async def test_cutoff_excludes_older_plays(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch(
            [
                _play(url="u/old", requester_id=1, played_secs=500, played_at=1000.0),
                _play(url="u/new", requester_id=2, played_secs=10, played_at=3000.0),
            ]
        )
        board = await archive.leaderboard(42, 10, since_epoch=2000.0)
        assert [r.requester_id for r in board.requesters] == [2]
        assert [s.webpage_url for s in board.songs] == ["u/new"]

    async def test_cutoff_is_inclusive(self, archive: PostgresHistoryArchive) -> None:
        await archive.insert_batch([_play(url="u/1", requester_id=1, played_at=2000.0)])
        board = await archive.leaderboard(42, 10, since_epoch=2000.0)
        assert [r.requester_id for r in board.requesters] == [1]

    async def test_unknown_time_rows_appear_only_in_all_time_boards(
        self, archive: PostgresHistoryArchive
    ) -> None:
        # played_at 0.0 is the unknown sentinel (backfilled legacy entries), so
        # any real cutoff excludes it by definition — documented, not a bug.
        await archive.insert_batch(
            [_play(url="u/1", requester_id=1, played_secs=90, played_at=0.0)]
        )
        assert (await archive.leaderboard(42, 10)).requesters != ()
        assert (await archive.leaderboard(42, 10, since_epoch=1.0)).requesters == ()


class TestRejectionDedup:
    """play_history_rejected must be exactly-once. The stable stream consumer
    name lets two drainers replay the same pending entry, fail it identically,
    and both write here. The table's diagnostic value is that it is normally
    empty and every row is page-worthy: three rows from `just db-rejects` must
    mean three distinct failures, not one seen three times."""

    async def test_the_same_refusal_recorded_twice_stores_one_row(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        entry = _entry(1)
        error = asyncpg.exceptions.CheckViolationError("play_history_played_secs_valid")
        await archive.record_rejection(entry, error, "trace-a")
        await archive.record_rejection(entry, error, "trace-b")
        conn = await asyncpg.connect(pg_dsn)
        try:
            assert (
                await conn.fetchval("SELECT count(*) FROM play_history_rejected") == 1
            )
            # First writer wins, as ON CONFLICT DO NOTHING implies.
            assert (
                await conn.fetchval("SELECT trace_id FROM play_history_rejected")
                == "trace-a"
            )
        finally:
            await conn.close()

    async def test_distinct_failures_are_still_distinct_rows(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        # The dedup key is (guild_id, error_type, digest of the payload), so a
        # different entry, a different error class, or a different guild each
        # remain separately visible. A key that collapsed those would hide real
        # failures behind the first one reported.
        entry = _entry(1)
        check = asyncpg.exceptions.CheckViolationError("check")
        await archive.record_rejection(entry, check, "t")
        await archive.record_rejection(_entry(2), check, "t")  # other payload
        await archive.record_rejection(
            entry, asyncpg.exceptions.DataError("data"), "t"
        )  # other error class
        await archive.record_rejection(
            _entry(1, guild_id=77), check, "t"
        )  # other guild
        conn = await asyncpg.connect(pg_dsn)
        try:
            assert (
                await conn.fetchval("SELECT count(*) FROM play_history_rejected") == 4
            )
        finally:
            await conn.close()

    async def test_a_nul_bearing_payload_still_dedups(
        self, archive: PostgresHistoryArchive, pg_dsn: str
    ) -> None:
        # The digest is md5(bytea), so it works on exactly the payloads text and
        # jsonb refuse — which are the ones this table exists for.
        entry = _entry(1, title="bad\x00title")
        error = asyncpg.exceptions.DataError("nul")
        await archive.record_rejection(entry, error, "t")
        await archive.record_rejection(entry, error, "t")
        conn = await asyncpg.connect(pg_dsn)
        try:
            assert (
                await conn.fetchval("SELECT count(*) FROM play_history_rejected") == 1
            )
        finally:
            await conn.close()


class TestDrainerEndToEnd:
    """The full path a played song takes: GuildRedisStore.push_history →
    Redis outbox → drainer → Postgres. Every layer real except Discord."""

    async def test_entries_reach_postgres_through_the_drainer(
        self, archive: PostgresHistoryArchive, fake_redis: Redis
    ) -> None:
        await ensure_outbox_group(fake_redis)
        store = GuildRedisStore(fake_redis, guild_id=42)
        entries = [_entry(i) for i in range(3)]
        for entry in entries:
            await store.push_history(entry)

        drainer = HistoryOutboxDrainer(fake_redis, archive)
        drainer.start()
        try:
            drainer.notify()
            async with asyncio.timeout(10):
                while await fake_redis.xlen(HISTORY_OUTBOX_KEY):
                    await asyncio.sleep(0.05)
        finally:
            await drainer.stop()

        got = await archive.recent(42, 10)
        assert [e.title for e in got] == ["Song 2", "Song 1", "Song 0"]

    async def test_the_former_poison_classes_all_land_untouched(
        self,
        archive: PostgresHistoryArchive,
        fake_redis: Redis,
    ) -> None:
        """The former poison classes against a real server: all FIVE land and
        nothing is dead-lettered, because `title="bad\\x00title"` and
        `played_at=1e18` are clamped by __post_init__ at construction. Kept in
        the pg tier because the claim is about what a real Postgres accepts (NUL
        in text, an out-of-range timestamptz) — what fakes cannot answer."""
        await ensure_outbox_group(fake_redis)
        store = GuildRedisStore(fake_redis, guild_id=42)
        for entry in (
            _entry(1),
            _entry(2, title="bad\x00title"),  # was a server-side DataError
            _entry(3),
            _entry(4, played_at=1e18),  # was a row-conversion blow-up
            _entry(5),
        ):
            await store.push_history(entry)

        drainer = HistoryOutboxDrainer(fake_redis, archive)
        assert await drainer._drain_once() == 5

        landed = await archive.recent(42, 10)
        assert len(landed) == 5
        by_url = {e.webpage_url: e for e in landed}
        # Clamped, not rejected: the NUL is stripped and the absurd epoch lands
        # on the unknown sentinel, exactly as __post_init__ promised.
        assert by_url["https://yt.com/v=2"].title == "badtitle"
        assert by_url["https://yt.com/v=4"].played_at == 0.0
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0


class TestBackfillAgainstARealArchive:
    """`src/backfill_history.py` against a real Postgres. Its safety story is
    "just run it again — the dedup index absorbs it", a claim composing four
    things the in-memory double cannot model together: a key-derived guild_id,
    real int8 columns, the `guild_id > 0` CHECK, and executemany's atomicity.
    `CollectingArchive` has no unique index, so the unit tier can only observe
    that the TOOL re-sent the same count."""

    async def test_rerunning_is_idempotent_against_the_dedup_index(
        self, archive: PostgresHistoryArchive, fake_redis: Redis
    ) -> None:
        # Legacy wire shape: guild_id=0 on the entry, real id only in the KEY.
        key = GUILD_HISTORY_KEY.format(guild_id=42)
        for n in range(5):
            await fake_redis.lpush(
                key, serialize_history_entry(replace(_entry(n), guild_id=0))
            )

        first = await backfill(fake_redis, archive)
        second = await backfill(fake_redis, archive)

        assert first.ok and second.ok
        assert first.attempted == second.attempted == 5
        # the assertion the unit tier cannot make: the second run inserted
        # nothing new, because ON CONFLICT collapsed every row.
        rows = await archive.recent(42, 100)
        assert len(rows) == 5
        # …and the key-derived stamp survived the round trip through int8.
        assert {r.guild_id for r in rows} == {42}

    async def test_a_refused_row_is_contained_and_reported(
        self, archive: PostgresHistoryArchive, fake_redis: Redis, pg_dsn: str
    ) -> None:
        """One guild refused must not cost the others, and must not report ok.
        Driven through a real CHECK violation: `guild_id > 0` is the constraint
        __post_init__ clamps toward rather than away from, so it is the realistic
        refusal. The tool skips such keys upstream, so it is injected through a
        patched key parser — the shape a regression in that guard would take."""
        import src.backfill_history as bh

        good = GUILD_HISTORY_KEY.format(guild_id=7)
        bad = GUILD_HISTORY_KEY.format(guild_id=8)
        await fake_redis.lpush(good, serialize_history_entry(_entry(1, guild_id=7)))
        # guild_id=0 on the wire, so the KEY-derived id is what gets stamped —
        # which is the only route by which a poisoned id reaches the insert.
        await fake_redis.lpush(
            bad, serialize_history_entry(replace(_entry(2), guild_id=0))
        )

        real_parse = bh._guild_id_from_key

        def poisoned(key: bytes) -> object:
            # Guild 8's key parses to 0 — constructible, and refused by the CHECK.
            return 0 if b":8:" in key else real_parse(key)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(bh, "_guild_id_from_key", poisoned)
            report = await backfill(fake_redis, archive)

        assert report.failed_guilds == 1
        assert report.guilds == 1  # only the guild that moved in full
        assert not report.ok
        # Guild 7 still landed — containment is real, not just counted.
        assert len(await archive.recent(7, 10)) == 1


class TestArchiveStats:
    """-debug's Postgres block, against a real server.

    Its SQL never runs in the unit tier — those tests fake the archive — so a
    misspelled column or a catalog view that does not answer the way it is
    assumed to would ship silently and render the whole block "unavailable".
    """

    async def test_reports_sizes_connections_and_cache(
        self, archive: PostgresHistoryArchive
    ) -> None:
        await archive.insert_batch([_entry(i) for i in range(5)])
        stats = await archive.stats()
        assert stats.database_bytes > 0
        assert stats.table_bytes > 0
        # Own connection at minimum, and the setting is an int the row divides by.
        assert 1 <= stats.connections <= stats.max_connections
        assert stats.max_connections > 0
        assert stats.shared_buffers  # a human string like "128MB"
        assert 0.0 <= stats.cache_hit_ratio <= 1.0

    async def test_row_counts_are_estimates_that_survive_never_analyzing(
        self, archive: PostgresHistoryArchive
    ) -> None:
        """n_live_tup is NULL until autovacuum has visited the table, and a fresh
        database is exactly that case. COALESCE keeps it 0 rather than crashing
        the block on the deployment most likely to be running -debug."""
        stats = await archive.stats()
        assert stats.rows_estimate >= 0
        # Expected to stay 0 forever: a row here means HistoryEntry's clamp
        # regressed. -debug surfaces it, `just db-rejects` inspects it.
        assert stats.rejected_estimate == 0
