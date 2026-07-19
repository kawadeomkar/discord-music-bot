"""Tests for src/db.py — migration discovery/runner logic against a scripted
connection fake (ledger, checksums, ordering, per-file transactions are pure
logic; real-Postgres behavior is the pg tier's job — see test_pg_integration).
"""

import hashlib

import pytest

from src.db import MIGRATIONS_DIR, Database, MigrationError, _discover, run_migrations


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FakeConn:
    """Just enough asyncpg Connection for run_migrations: execute/fetch and a
    transaction context manager, all recorded."""

    def __init__(self, ledger=None, fail_on: str | None = None):
        self.ledger = ledger or []  # rows for the SELECT version, checksum
        self.executed: list[tuple[str, tuple]] = []
        self.fail_on = fail_on  # substring: execute raises when it appears
        self.tx_events: list[str] = []

    async def execute(self, sql, *args):
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("scripted failure")
        self.executed.append((sql.strip(), args))

    async def fetch(self, sql, *args):
        return self.ledger

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self):
                conn.tx_events.append("begin")
                return self

            async def __aexit__(self, exc_type, exc, tb):
                conn.tx_events.append("rollback" if exc_type else "commit")
                return False

        return _Tx()


@pytest.fixture
def migrations(tmp_path):
    d = tmp_path / "migrations"
    d.mkdir()
    (d / "0001_first.sql").write_text("CREATE TABLE one (id int);")
    (d / "0002_second.sql").write_text("CREATE TABLE two (id int);")
    return d


class TestDiscover:
    def test_orders_by_filename(self, migrations):
        (migrations / "0010_tenth.sql").write_text("SELECT 10;")
        assert [p.name for p in _discover(migrations)] == [
            "0001_first.sql",
            "0002_second.sql",
            "0010_tenth.sql",
        ]

    def test_ignores_non_numbered_files(self, migrations):
        (migrations / "README.sql").write_text("-- not a migration")
        assert len(_discover(migrations)) == 2

    def test_duplicate_prefix_is_ambiguous(self, migrations):
        (migrations / "0002_other.sql").write_text("SELECT 2;")
        with pytest.raises(MigrationError, match="duplicate migration prefix 0002"):
            _discover(migrations)

    def test_real_repo_baseline_present(self):
        # Guard the DDL's move out of code: the shipped migrations dir must
        # hold the 0001 baseline with the table + dedup index.
        files = _discover(MIGRATIONS_DIR)
        assert files and files[0].name == "0001_play_history.sql"
        sql = files[0].read_text()
        assert "CREATE TABLE IF NOT EXISTS play_history" in sql
        assert "play_history_dedup" in sql


class TestRunMigrations:
    async def test_fresh_database_applies_all_in_order(self, migrations):
        conn = FakeConn()
        assert await run_migrations(conn, migrations) == 2
        sqls = [s for s, _ in conn.executed]
        assert "CREATE TABLE one (id int);" in sqls
        assert "CREATE TABLE two (id int);" in sqls
        assert sqls.index("CREATE TABLE one (id int);") < sqls.index(
            "CREATE TABLE two (id int);"
        )

    async def test_ledger_rows_carry_version_and_checksum(self, migrations):
        conn = FakeConn()
        await run_migrations(conn, migrations)
        inserts = [
            args
            for sql, args in conn.executed
            if "INSERT INTO schema_migrations" in sql
        ]
        assert inserts == [
            ("0001_first", _sha("CREATE TABLE one (id int);")),
            ("0002_second", _sha("CREATE TABLE two (id int);")),
        ]

    async def test_each_file_in_its_own_transaction(self, migrations):
        conn = FakeConn()
        await run_migrations(conn, migrations)
        assert conn.tx_events == ["begin", "commit", "begin", "commit"]

    async def test_applied_versions_skipped(self, migrations):
        conn = FakeConn(
            ledger=[
                {
                    "version": "0001_first",
                    "checksum": _sha("CREATE TABLE one (id int);"),
                }
            ]
        )
        assert await run_migrations(conn, migrations) == 1
        sqls = [s for s, _ in conn.executed]
        assert "CREATE TABLE one (id int);" not in sqls
        assert "CREATE TABLE two (id int);" in sqls

    async def test_rerun_is_noop(self, migrations):
        conn = FakeConn(
            ledger=[
                {
                    "version": "0001_first",
                    "checksum": _sha("CREATE TABLE one (id int);"),
                },
                {
                    "version": "0002_second",
                    "checksum": _sha("CREATE TABLE two (id int);"),
                },
            ]
        )
        assert await run_migrations(conn, migrations) == 0
        assert not any(
            "CREATE TABLE" in s and "schema_migrations" not in s
            for s, _ in conn.executed
        )

    async def test_tampered_file_is_an_error_not_a_rerun(self, migrations):
        conn = FakeConn(
            ledger=[{"version": "0001_first", "checksum": "not-the-real-checksum"}]
        )
        with pytest.raises(MigrationError, match="modified after being applied"):
            await run_migrations(conn, migrations)
        # And nothing after the mismatch ran — fail closed.
        assert not any("CREATE TABLE two" in s for s, _ in conn.executed)

    async def test_failed_file_rolls_back_and_propagates(self, migrations):
        conn = FakeConn(fail_on="CREATE TABLE two")
        with pytest.raises(RuntimeError, match="scripted failure"):
            await run_migrations(conn, migrations)
        # First file committed; the failing one rolled back, no ledger row.
        assert conn.tx_events == ["begin", "commit", "begin", "rollback"]
        inserts = [
            args
            for sql, args in conn.executed
            if "INSERT INTO schema_migrations" in sql
        ]
        assert [v for v, _ in inserts] == ["0001_first"]


class TestDatabaseLifecycle:
    async def test_construct_makes_no_connection(self):
        # A bogus DSN would explode on any connection attempt — constructing
        # must be free (startup never blocks on Postgres).
        Database("postgresql://nope:1/nope")

    async def test_close_before_connect_is_safe(self):
        await Database("postgresql://nope:1/nope").close()
        # Idempotent too.
        db = Database("postgresql://nope:1/nope")
        await db.close()
        await db.close()
