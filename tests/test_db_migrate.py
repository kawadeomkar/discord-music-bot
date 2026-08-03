"""Tests for src/db_migrate.py — migration discovery and the version contract.

SQL execution needs a real server and lives in the pg tier. Here: everything
that can be wrong without one — the file-naming contract, ordering, duplicate
detection, and migrations/ agreeing with EXPECTED_SCHEMA_VERSION.
"""

from pathlib import Path

import pytest

from src.db_migrate import (
    EXPECTED_SCHEMA_VERSION,
    MIGRATIONS_DIR,
    discover,
    main,
)


class TestVersionContract:
    def test_expected_version_matches_the_highest_migration_on_disk(self) -> None:
        # The guard that makes the literal safe: add a migration file and
        # forget the constant, and this fails instead of shipping a bot that
        # rejects its own schema at startup.
        versions = [v for v, _ in discover()]
        assert versions, "migrations/ must not be empty"
        assert max(versions) == EXPECTED_SCHEMA_VERSION

    def test_migrations_are_numbered_from_one_without_gaps(self) -> None:
        # A gap means a migration was deleted after shipping — databases that
        # applied it and databases that never saw it now differ, and the
        # version number stops describing the schema.
        versions = [v for v, _ in discover()]
        assert versions == list(range(1, len(versions) + 1))

    def test_shipped_migrations_are_readable_sql(self) -> None:
        for _, path in discover():
            assert path.read_text().strip()


class TestDiscover:
    def test_orders_numerically_not_lexically(self, tmp_path: Path) -> None:
        # "0010" sorts before "0009" only if you compare as ints — which is the
        # bug this ordering exists to avoid once there are ten migrations.
        for name in ("0010_j.sql", "0009_i.sql", "0002_b.sql"):
            (tmp_path / name).write_text("SELECT 1")
        assert [v for v, _ in discover(tmp_path)] == [2, 9, 10]

    def test_rejects_a_badly_named_file(self, tmp_path: Path) -> None:
        (tmp_path / "add_column.sql").write_text("SELECT 1")
        with pytest.raises(RuntimeError, match="NNNN_name.sql"):
            discover(tmp_path)

    def test_rejects_duplicate_versions(self, tmp_path: Path) -> None:
        # Two files claiming the same number: whichever loses is applied on the
        # databases that saw it land and never on a fresh one.
        (tmp_path / "0001_a.sql").write_text("SELECT 1")
        (tmp_path / "0001_b.sql").write_text("SELECT 1")
        with pytest.raises(RuntimeError, match="duplicate migration version"):
            discover(tmp_path)

    def test_empty_directory_discovers_nothing(self, tmp_path: Path) -> None:
        assert discover(tmp_path) == []

    def test_default_directory_is_the_repo_migrations_dir(self) -> None:
        assert MIGRATIONS_DIR.name == "migrations"
        assert MIGRATIONS_DIR.is_dir()


class TestMain:
    def test_missing_url_exits_nonzero_with_guidance(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("POSTGRES_MIGRATE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert main() == 1
        assert "POSTGRES_MIGRATE_URL" in capsys.readouterr().err

    def test_migrate_url_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The migrating role is allowed to differ from the bot's (the app role
        # loses DDL rights in the documented deployment).
        seen: list[str] = []

        async def fake_migrate(url: str) -> int:
            seen.append(url)
            return EXPECTED_SCHEMA_VERSION

        monkeypatch.setenv("POSTGRES_URL", "postgresql://app@h/db")
        monkeypatch.setenv("POSTGRES_MIGRATE_URL", "postgresql://ddl@h/db")
        monkeypatch.setattr("src.db_migrate.migrate", fake_migrate)
        assert main() == 0
        assert seen == ["postgresql://ddl@h/db"]

    def test_reports_failure_when_the_result_version_disagrees(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async def fake_migrate(url: str) -> int:
            return EXPECTED_SCHEMA_VERSION - 1

        monkeypatch.setenv("POSTGRES_URL", "postgresql://app@h/db")
        monkeypatch.delenv("POSTGRES_MIGRATE_URL", raising=False)
        monkeypatch.setattr("src.db_migrate.migrate", fake_migrate)
        assert main() == 1
        assert "expects" in capsys.readouterr().err

    def test_a_newer_database_succeeds_with_a_note(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A database AHEAD of this build is the rollback case, and it must exit 0.

        Two things depend on that. PostgresHistoryArchive._assert_schema_version
        already tolerates it — migrations are additive, so an older bot reads a
        newer schema fine — and a runner stricter than the bot it serves would
        contradict it. And deploy_docker.sh now gates the deploy on this exit
        status, so returning non-zero would make every rollback fail to deploy:
        a routine rollback turned into an outage, which is precisely what the
        archive's tolerance was written to prevent.
        """

        async def fake_migrate(url: str) -> int:
            return EXPECTED_SCHEMA_VERSION + 1

        monkeypatch.setenv("POSTGRES_URL", "postgresql://app@h/db")
        monkeypatch.delenv("POSTGRES_MIGRATE_URL", raising=False)
        monkeypatch.setattr("src.db_migrate.migrate", fake_migrate)
        assert main() == 0
        captured = capsys.readouterr()
        assert "ahead of this build" in captured.out
        assert captured.err == ""
