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
from dataclasses import replace
from collections.abc import AsyncIterator, Iterator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from redis.asyncio import Redis

from src.db_migrate import EXPECTED_SCHEMA_VERSION, migrate
from src.guild_state import HistoryEntry, parse_history_entry
from src.history_archive import (
    _POISON,
    _RECENT_SQL,
    HistoryOutboxDrainer,
    PostgresHistoryArchive,
    SchemaVersionError,
    quantized_played_at,
)
from src.redis_client import (
    HISTORY_OUTBOX_KEY,
    GuildRedisStore,
    ensure_outbox_group,
)

from tests.helpers import tier_enabled

# POSTGRES_TEST_URL enables the tier on its own, deliberately. If it only
# *selected the provider* and RUN_PG_TESTS still had to be set as well, a CI job
# that supplied the server but forgot the flag would skip every test in this
# file and report green — a blind job that looks covered is worse than a
# missing one. One variable is enough to say "there is a Postgres here".
_PG_ENABLED = tier_enabled("RUN_PG_TESTS", "POSTGRES_TEST_URL")

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

    @pytest.mark.parametrize(
        "raw",
        [
            1785309912.0000014,
            1785309912.0000026,
            1785309912.0000045,
            1785309912.0000055,
            1785309912.0000074,
            1785309912.0000086,
        ],
    )
    async def test_quantized_played_at_models_the_real_round_trip(
        self, archive: PostgresHistoryArchive, raw: float
    ) -> None:
        """`quantized_played_at` claims to be what timestamptz does. This is the
        only place that claim can be settled, and until it existed nothing
        checked it anywhere.

        Its whole job is to let GuildHistory.recent() build one dedup identity
        out of a raw time.time() float (Redis leg) and a value that has been
        through Postgres (archive leg). If the model is wrong the key misses,
        the play renders twice, and merged[:limit] displaces a genuine older
        song per duplicate.

        EXACT equality, and adversarial inputs. The sibling test above uses
        approx(abs=1e-6) — a tolerance the same size as the error class — so it
        stays green against a wrong model. These six values are ULP-scale
        neighbours chosen because they are exactly where the rejected
        alternative `round(t * 1e6) / 1e6` disagrees with the shipped
        datetime pair; substituting it turns every case here red. Without them
        that substitution passed the entire unit suite AND this tier, while
        missing the identity for 12.5% of real timestamps.
        """
        # __post_init__ must not clamp these — they are ordinary 2026 epochs.
        entry = _entry(1, played_at=raw)
        assert entry.played_at == raw
        await archive.insert_batch([entry])
        [got] = await archive.recent(42, 10)
        assert got.played_at == quantized_played_at(raw)

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


def _boundary_entries() -> list[HistoryEntry]:
    """Every clamp __post_init__ performs, one entry each.

    Enumerated rather than sampled: this list IS the claim that the residual
    poison set is empty (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md §3.2). Extend it
    whenever a column type or a domain tuple changes. Distinct webpage_urls keep
    play_history_dedup from collapsing rows that clamp to the same guild_id.
    """
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
        # NUL in each text field.
        HistoryEntry(guild_id=7, webpage_url="u/t", title="a\x00b"),
        HistoryEntry(guild_id=7, webpage_url="u/w\x00url"),
        HistoryEntry(guild_id=7, webpage_url="u/rn", requester_name="a\x00b"),
        HistoryEntry(guild_id=7, webpage_url="u/th", thumbnail="a\x00b"),
        HistoryEntry(guild_id=7, webpage_url="u/up", uploader="a\x00b"),
    ]


class TestSchemaLock:
    """The play_history CHECK constraints plus HistoryEntry.__post_init__ — both
    directions of the
    claim that possessing a HistoryEntry is the proof Postgres accepts it."""

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
        ],
    )
    async def test_constraints_catch_a_validator_regression(
        self,
        archive: PostgresHistoryArchive,
        field: str,
        value: int,
        constraint: str,
    ) -> None:
        # Backward: bypass the validator and the database still refuses. Every
        # CHECK gets its own case — only guild_id was exercised, so the other
        # four were asserted to exist by reading the migration and nothing more.
        # object.__setattr__ is how a regression would present: a future edit
        # dropping a clamp, or a field added to the dataclass but not to a
        # domain tuple.
        #
        # The constraint NAME is asserted, not just the class. A row can violate
        # several CHECKs at once and the server reports whichever it evaluates
        # first, so a bare `raises(CheckViolationError)` would pass even if the
        # constraint under test had been dropped from the schema entirely.
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
        """The one deliberate hole in "constructible implies insertable".

        __post_init__ floors every integer at 0 while play_history's CHECK is
        strictly `> 0`, so this entry is built WITHOUT bypassing the validator —
        no object.__setattr__ — and Postgres still refuses it. Both halves are
        asserted here because both are choices: clamping up to 1 would file an
        unattributable play into a real guild's history, and relaxing the CHECK
        would make guild 0 a bucket of orphans no read path excludes. Refusing
        sends it to play_history_rejected, where `just db-rejects` shows an
        operator the actual defect — a producer that stopped stamping guild_id.
        """
        orphan = HistoryEntry(guild_id=0, webpage_url="https://x", played_at=0.0)
        assert orphan.guild_id == 0  # constructed, not clamped away
        with pytest.raises(asyncpg.exceptions.CheckViolationError) as exc:
            await archive.insert_batch([orphan])
        assert exc.value.as_dict()["constraint_name"] == "play_history_guild_id_valid"

    async def test_a_check_violation_is_classified_as_poison(self) -> None:
        # The ordering fix: CheckViolationError is SQLSTATE 23514 under
        # IntegrityConstraintViolationError, NOT a DataError. Without the arm in
        # _POISON a violation would propagate past the quarantine path and wedge
        # the drain head permanently on a non-evictable list.
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
        # The NOT VALID promise, against a row that predates the constraint.
        # This is the assertion that would have caught a migration failing on
        # real data: drop the constraint, write a row that violates it, re-add
        # it NOT VALID, and confirm both the ADD and a later read succeed.
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
            # And the constraint is live for NEW rows even while unvalidated.
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
        # THE reason payload is bytea. A NUL byte is the first poison vector, and
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
        # The counterfactual, asserted rather than argued in a comment: this is
        # what makes the bytea column choice evidence instead of opinion.
        # Built with chr(0) rather than written as a literal — a source file
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
    """message_id reaches Postgres.

    It landed on the wire two branches before any INSERT carried it, and the
    column's CHECK is `message_id >= 0` with `DEFAULT 0` — so a missed value
    inserted cleanly and was permanently indistinguishable from a song that
    genuinely had no Now Playing host. Nothing short of a real round trip
    catches that: the fakeredis drainer tests never touch SQL.
    """

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


class TestRejectionDedup:
    """play_history_rejected must be exactly-once.

    Under the drainer lease it was, for free — only one process could hold a
    poisoned batch. A stable stream consumer name lets two drainers replay the
    same pending entry, fail it identically, and both write here. The whole
    diagnostic value of this table is that it is normally empty and every row
    is page-worthy, so `just db-rejects` printing three rows must mean three
    distinct failures rather than one seen three times.
    """

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
        """The former poison classes, against a real server, after the schema
        lock — ALL FIVE land and nothing is dead-lettered.

        This test used to neutralize sanitization and assert that two of the
        five reached the DLQ. That premise is gone: `title="bad\\x00title"` and
        `played_at=1e18` are now clamped by HistoryEntry.__post_init__ at
        construction, so the entries the outbox carries are insertable by
        construction and the class of failure no longer exists. Asserting the
        elimination is strictly stronger than asserting it was handled.

        Kept in the pg tier rather than moved: the claim is about what a real
        Postgres accepts (NUL in text, an out-of-range timestamptz), which is
        precisely what fakes cannot answer.
        """
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
