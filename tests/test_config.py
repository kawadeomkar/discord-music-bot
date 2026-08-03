"""Tests for src/config.py — the Spotify toggle, the Postgres knobs, and the
default-credential detector."""

import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from collections.abc import Iterator
from types import ModuleType

import pytest

import src.config
from src.config import (
    DEFAULT_POSTGRES_PASSWORD,
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
    _int_env,
    history_archive_enabled,
    postgres_url,
    spotify_enabled,
    using_default_postgres_password,
)


class TestSpotifyEnabled:
    def test_enabled_when_both_credentials_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
        assert spotify_enabled() is True

    def test_disabled_when_both_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
        monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
        assert spotify_enabled() is False

    def test_disabled_when_only_id_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
        monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
        assert spotify_enabled() is False

    def test_disabled_when_only_secret_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
        assert spotify_enabled() is False

    def test_disabled_when_credentials_are_empty_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exported-but-empty var (`SPOTIFY_CLIENT_ID=`) must count as absent —
        otherwise a blank line in .env would 'enable' Spotify and then 400 on auth."""
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "")
        assert spotify_enabled() is False

    def test_read_at_call_time_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate must reflect the live environment on each call, not a value
        frozen at import."""
        monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
        monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
        assert spotify_enabled() is False
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
        assert spotify_enabled() is True


class TestSpotifyConfigConstants:
    def test_probe_track_id_is_the_documented_track(self) -> None:
        # "Never Gonna Give You Up" — a permanent public track used as the
        # startup credential probe. Guards against an accidental edit.
        assert SPOTIFY_TEST_TRACK_ID == "4PTG3Z6ehGkBFwjybzWkR8"

    def test_status_has_three_distinct_states(self) -> None:
        assert {s.value for s in SpotifyStatus} == {"disabled", "invalid", "enabled"}


class TestPostgresUrl:
    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert postgres_url() is None

    def test_returns_the_dsn_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u@h/db")
        assert postgres_url() == "postgresql://u@h/db"

    def test_empty_string_counts_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exported-but-empty `POSTGRES_URL=` must read as absent, same rule as
        the Spotify credentials above: `""` is not None, so without the guard the
        Optional[str] return type is a lie and catching an empty DSN depends on
        each caller spelling its check as truthiness rather than `is None`."""
        monkeypatch.setenv("POSTGRES_URL", "")
        assert postgres_url() is None

    def test_read_at_call_time_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert postgres_url() is None
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u@h/db")
        assert postgres_url() == "postgresql://u@h/db"


class TestHistoryArchiveEnabled:
    """The consent gate for long-term storage. Fail-closed: absence of a choice
    must mean no collection, so unset and empty are False. Strict parse: a
    lenient anything-but-true-is-False rule turns a typo into an operator who
    believes archiving is on while every play goes unrecorded, so garbage raises.
    """

    @pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", "YES"])
    def test_truthy_spellings(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", raw)
        assert history_archive_enabled() is True

    @pytest.mark.parametrize("raw", ["false", "FALSE", "False", "0", "no", "NO"])
    def test_falsy_spellings(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", raw)
        assert history_archive_enabled() is False

    def test_unset_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HISTORY_ARCHIVE_ENABLED", raising=False)
        assert history_archive_enabled() is False

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_empty_reads_as_unset(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`HISTORY_ARCHIVE_ENABLED=` is the bare KEY= shape .env.example
        models for POSTGRES_PASSWORD — same tolerance rule as _int_env."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", raw)
        assert history_archive_enabled() is False

    def test_surrounding_whitespace_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "  true  ")
        assert history_archive_enabled() is True

    @pytest.mark.parametrize("raw", ["on", "enabled", "ture", "2", "y", "t"])
    def test_garbage_raises_naming_the_variable(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Includes near-misses an operator would plausibly type (`on`, `y`,
        `enabled`): every one silently disables archiving under a lenient
        parser, which is the failure the strict parse exists to prevent."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", raw)
        with pytest.raises(ValueError, match="HISTORY_ARCHIVE_ENABLED"):
            history_archive_enabled()

    def test_read_at_call_time_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HISTORY_ARCHIVE_ENABLED", raising=False)
        assert history_archive_enabled() is False
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "true")
        assert history_archive_enabled() is True


class TestIntEnv:
    """The parser behind both archive tunables.

    It runs at import, before structlog or OTel exist, so its failures are stderr
    tracebacks in a compose restart loop — hence empty is tolerated, garbage is not.
    """

    def test_unset_returns_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KNOB", raising=False)
        assert _int_env("KNOB", 7) == 7

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_empty_reads_as_unset(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`KNOB=` is the bare shape .env.example already models for
        POSTGRES_PASSWORD. Raising here would crash-loop the bot before any log
        pipeline exists — same rule postgres_url() applies to a blank DSN."""
        monkeypatch.setenv("KNOB", raw)
        assert _int_env("KNOB", 7) == 7

    def test_parses_a_set_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOB", "5000")
        assert _int_env("KNOB", 0) == 5000

    def test_negative_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """-1 is the universal "no limit" idiom, so it is what an operator reaches
        for to spell out HISTORY_OUTBOX_MAX's default. Downstream it means the
        opposite: the drainer treats it as an active cap, computes
        dropped = depth + 1, and trims the outbox to empty every cycle."""
        monkeypatch.setenv("KNOB", "-1")
        with pytest.raises(ValueError, match="KNOB must be >= 0"):
            _int_env("KNOB", 0)

    @pytest.mark.parametrize("raw", ["abc", "100mb", "1e6", "0x10", "3.5"])
    def test_malformed_raises_naming_the_variable(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A bare "invalid literal for int() with base 10" does not say which of
        # the environment's variables is at fault, and there is no logger
        # attached at import to add that context.
        monkeypatch.setenv("KNOB", raw)
        with pytest.raises(ValueError, match="KNOB must be an integer"):
            _int_env("KNOB", 0)


class TestArchiveTunables:
    """The env -> constant path, which asserting the defaults alone cannot pin.

    `assert HISTORY_OUTBOX_MAX == 0` passes even if the constant stops reading its
    variable (a plain literal is also 0), and fails wherever that variable happens
    to be exported. Reloading under a controlled environment fixes both.
    """

    @pytest.fixture(autouse=True)
    def _restore_config_module(self) -> Iterator[None]:
        # Snapshot rather than rely on monkeypatch teardown ordering: the module
        # must be left holding the constants the rest of the session imported.
        original = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original)
        importlib.reload(src.config)

    @staticmethod
    def _reload(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        # Pin ENVIRONMENT so the reload never shells out to git, and never trips
        # config's detached-HEAD RuntimeWarning that pytest promotes to an error.
        monkeypatch.setenv("ENVIRONMENT", "development")
        return importlib.reload(src.config)

    @pytest.mark.parametrize(
        ("name", "default", "override", "expected"),
        [
            # 0 is the durability contract: an entry only leaves the outbox once
            # Postgres has it. A non-zero default would silently discard plays.
            ("HISTORY_OUTBOX_MAX", 0, "5000", 5000),
            # Matches asyncpg's own default; 0 is the PgBouncer setting.
            ("POSTGRES_STATEMENT_CACHE", 100, "0", 0),
        ],
    )
    def test_default_and_override(
        self,
        name: str,
        default: int,
        override: str,
        expected: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(name, raising=False)
        assert getattr(self._reload(monkeypatch), name) == default
        monkeypatch.setenv(name, override)
        assert getattr(self._reload(monkeypatch), name) == expected

    def test_a_negative_cap_fails_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup refusal is the point: the alternative is a drainer that wipes
        the outbox every cycle while the bot reports healthy."""
        monkeypatch.setenv("HISTORY_OUTBOX_MAX", "-1")
        with pytest.raises(ValueError, match="HISTORY_OUTBOX_MAX must be >= 0"):
            self._reload(monkeypatch)


class TestDefaultPostgresPassword:
    """compose defaults POSTGRES_PASSWORD so `docker compose up` works with only a
    Discord token, and the bot has to be able to tell that it did.

    Scoped to the DSNs this project's tooling assembles from `.env`, the only
    supported place the password is set — shapes asyncpg accepts but compose and
    `just run` cannot emit are deliberately undetected (see the function).
    """

    def test_true_when_the_dsn_carries_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL",
            f"postgresql://musicbot:{DEFAULT_POSTGRES_PASSWORD}@127.0.0.1:5432/musicbot",
        )
        assert using_default_postgres_password() is True

    def test_false_for_a_real_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:9f3a1c@127.0.0.1:5432/musicbot"
        )
        assert using_default_postgres_password() is False

    def test_false_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Nothing configured is not the same as configured badly — and the
        # missing-URL case has its own, louder failure in setup_hook.
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert using_default_postgres_password() is False

    def test_reads_the_dsn_not_the_password_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The bot only ever sees the assembled DSN: compose builds it and the
        # password variable is usually absent from the bot's own environment.
        # A check that read POSTGRES_PASSWORD would report "fine" for a stack
        # that is in fact running on the default.
        monkeypatch.setenv("POSTGRES_PASSWORD", "a-real-secret")
        monkeypatch.setenv(
            "POSTGRES_URL",
            f"postgresql://musicbot:{DEFAULT_POSTGRES_PASSWORD}@127.0.0.1:5432/musicbot",
        )
        assert using_default_postgres_password() is True

    def test_url_encoded_default_is_still_recognised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # asyncpg unquotes the netloc password, so an escaped form is the same
        # credential and must not slip past. The detector has to unquote it
        # ITSELF: SplitResult.password does not percent-decode.
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:%70assword@127.0.0.1:5432/musicbot"
        )
        assert using_default_postgres_password() is True

    @pytest.mark.parametrize(
        "url",
        [
            "not-a-url",
            "postgresql://",
            "postgresql://user@host/db",
            "://[bad",
            # The one that actually reaches the except arm: urlsplit only raises
            # on a malformed IPv6 literal in the NETLOC, and every other case
            # above returns normally. Without it the `except ValueError` branch
            # was dead — deleting it, and making it `return True`, both passed.
            "postgresql://[::1@h/db",
        ],
    )
    def test_never_raises_on_a_malformed_dsn(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        # It feeds a startup warning and a -ping row; a malformed DSN is the
        # archive's problem to report, not this function's.
        monkeypatch.setenv("POSTGRES_URL", url)
        assert using_default_postgres_password() is False


# ── The compose contract ──────────────────────────────────────────────────────
#
# Nothing else in CI or tests/ reads docker-compose.yml, which is how a missed
# `${POSTGRES_PASSWORD:?}` on db-backfill shipped: it broke `docker compose up`
# for a token-only stack while every test, ruff and pyright stayed green.
_COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.yml"


def _compose_directives() -> str:
    """docker-compose.yml with comment lines removed.

    The file DESCRIBES the old mandatory form (`${VAR:?}`) in comments, so a naive
    scan of the raw text reports a violation that does not exist. Whole-line
    stripping is enough — Compose has no inline-comment-after-value form here.
    """
    return "\n".join(
        line
        for line in _COMPOSE.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


def _service_block(name: str) -> str:
    """The directive lines of one top-level compose service (comments already
    stripped). Regex, not a YAML parser, on the same terms as everything else
    in this file: nothing in the repo's dependency set reads YAML, and adding
    a parser for a contract test would be the tail wagging the dog."""
    match = re.search(
        rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)",
        _compose_directives(),
        re.S | re.M,
    )
    assert match is not None, f"service {name} not found in docker-compose.yml"
    return match.group(1)


class TestComposeArchiveProfile:
    """The deployment half of the opt-in archive: postgres and db-migrate exist in
    the model only while the `archive` profile is active, so a token-only
    `docker compose up` deploys no long-term storage. Nothing in CI parses
    compose, so the contract is asserted against the real file."""

    def test_postgres_and_db_migrate_are_archive_profiled(self) -> None:
        # The profile IS the deployment gate: without it, a default `up`
        # deploys a database nobody opted in to and the consent story is
        # app-side only.
        for service in ("postgres", "db-migrate"):
            assert 'profiles: ["archive"]' in _service_block(service), service

    def test_the_bot_does_not_depend_on_the_profiled_services(self) -> None:
        """REGRESSION GUARD: an un-profiled service with depends_on on a profiled
        one makes the WHOLE project invalid while the profile is inactive — even
        `docker compose config` fails ("depends on undefined service") — so
        re-adding either dependency breaks token-only `up` outright."""
        bot = _service_block("discord-music-bot")
        depends = re.search(r"depends_on:\n((?:      .*\n)*)", bot)
        assert depends is not None
        assert "postgres" not in depends.group(1)
        assert "db-migrate" not in depends.group(1)
        # redis stays: it is un-profiled, so the rule does not apply to it.
        assert "redis:" in depends.group(1)

    def test_db_backfill_stays_on_ops_not_archive(self) -> None:
        """`up` starts EVERY service of an active profile, so db-backfill joining
        `archive` would run a full Redis keyspace walk on every enabled `up`. Its
        own profile means it runs only when explicitly targeted (`docker compose
        run` auto-activates a target's own profiles)."""
        backfill = _service_block("db-backfill")
        assert 'profiles: ["ops"]' in backfill
        assert "archive" not in backfill

    def test_just_down_activates_the_archive_profile(self) -> None:
        """`docker compose down` with the profile inactive removes only un-profiled
        containers and leaves a running postgres behind, so the justfile recipe
        must pass --profile archive. Greps the justfile — the same
        cannot-import-a-shell-script reasoning as the preflight test below."""
        justfile = (Path(__file__).resolve().parent.parent / "justfile").read_text()
        assert "docker compose --profile archive down" in justfile


class TestComposeMatchesTheDefault:
    """The two halves of the first-run promise, asserted against the real file:
    `docker compose up` must work with nothing configured but DISCORD_TOKEN, and
    the password it falls back to must be the one the bot warns about. Both are
    invisible to every other check in the repo.
    """

    def test_no_postgres_password_interpolation_is_mandatory(self) -> None:
        """REGRESSION: db-backfill kept `:?` when the other three services moved.
        Compose interpolates the WHOLE document before profile filtering, so an
        `ops`-profiled service with a mandatory variable fails `up`, `ps`, `logs`
        and `config` alike.
        """
        mandatory = re.findall(
            r"\$\{POSTGRES_PASSWORD:\?[^}]*\}", _compose_directives()
        )
        assert mandatory == []

    def test_every_fallback_is_the_password_the_bot_warns_about(self) -> None:
        """The drift check. DEFAULT_POSTGRES_PASSWORD is duplicated across
        config.py, build_common.sh and three compose services with nothing holding
        them together, and drift fails OPEN: change compose's fallback alone and
        the detector goes permanently silent on a known credential.
        """
        fallbacks = set(
            re.findall(r"\$\{POSTGRES_PASSWORD:-([^}]*)\}", _compose_directives())
        )
        # Non-empty guard: a regex that matched nothing would make the equality
        # below trivially true, which is how this kind of test rots.
        assert len(fallbacks) >= 1
        assert fallbacks == {DEFAULT_POSTGRES_PASSWORD}

    def test_the_build_preflight_checks_for_the_same_password(self) -> None:
        """build_common.sh hardcodes the literal too — the fifth copy. Drifting
        compose's fallback would leave the build-time warning checking for a value
        no deployment uses, silently and fail-open. A shell script cannot import
        config.py, so the coupling is asserted here instead of enforced there.
        """
        preflight = (
            Path(__file__).resolve().parent.parent / "build_common.sh"
        ).read_text()
        assert f'= "{DEFAULT_POSTGRES_PASSWORD}"' in preflight

    def test_the_justfile_dsn_uses_the_same_default(self) -> None:
        """The SIXTH copy, and the only one nothing held: `_dotenv`'s
        `${POSTGRES_PASSWORD:-password}` states the coupling in a comment with
        nothing enforcing it. Rotate the default in the three copies the tests
        above cover and every test stays green while `just run`/`db-*` build a
        DSN the database rejects — surfacing later as a drainer backoff loop.
        """
        justfile = (Path(__file__).resolve().parent.parent / "justfile").read_text()
        fallbacks = set(re.findall(r"\$\{POSTGRES_PASSWORD:-([^}]*)\}", justfile))
        assert len(fallbacks) >= 1  # non-empty guard, as above
        assert fallbacks == {DEFAULT_POSTGRES_PASSWORD}

    def test_the_bot_and_the_migration_tiers_all_carry_a_default(self) -> None:
        # Count rather than merely "none mandatory": a service whose
        # POSTGRES_URL was deleted outright would also pass the first test.
        # Four interpolations = the bot, the postgres service, and the two
        # one-shots' DSNs.
        assert len(re.findall(r"\$\{POSTGRES_PASSWORD:", _compose_directives())) >= 4


class TestSetupEnvTightensTheEnvFile:
    """setup_env.sh is the escape hatch from the shared default, so the file it
    writes the replacement into must not be world-readable.

    Run rather than grepped, unlike the drift checks above: the behaviour is an
    interaction between `stat`, `chmod` and `mv` across two platforms' stat
    spellings, which no source-text assertion pins. Copied into a tmpdir first —
    the script `cd`s to its own directory and would otherwise rewrite the real .env.
    """

    @staticmethod
    def _run(mode: int, tmp_path: Path) -> int:
        root = Path(__file__).resolve().parent.parent
        for name in ("setup_env.sh", "build_common.sh", ".env.example"):
            shutil.copy(root / name, tmp_path / name)
        env = tmp_path / ".env"
        env.write_text("DISCORD_TOKEN=x\n")
        env.chmod(mode)
        subprocess.run(
            ["bash", str(tmp_path / "setup_env.sh"), "--force"],
            check=True,
            capture_output=True,
        )
        assert "POSTGRES_PASSWORD=" in env.read_text()
        return env.stat().st_mode & 0o777

    def test_a_hand_made_world_readable_env_is_narrowed(self, tmp_path: Path) -> None:
        # `cp .env.example .env` by hand lands at 644 under the usual umask, and
        # this script is what then writes a freshly generated credential into it.
        assert self._run(0o644, tmp_path) == 0o600

    def test_a_stricter_mode_survives(self, tmp_path: Path) -> None:
        # Narrowing only: go-rwx clears bits and never sets them, so an operator
        # who chose 400 keeps 400.
        assert self._run(0o400, tmp_path) == 0o400
