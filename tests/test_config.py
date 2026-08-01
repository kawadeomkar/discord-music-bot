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
        """An exported-but-empty `POSTGRES_URL=` must read as absent, same rule
        as the Spotify credentials above. `""` is not None, so without the guard
        the Optional[str] return type is a lie and whether an empty DSN is
        caught depends on each caller spelling its check as truthiness rather
        than `is None`."""
        monkeypatch.setenv("POSTGRES_URL", "")
        assert postgres_url() is None

    def test_read_at_call_time_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert postgres_url() is None
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u@h/db")
        assert postgres_url() == "postgresql://u@h/db"


class TestIntEnv:
    """The parser behind both archive tunables.

    It runs at import, before main() has configured structlog or OTel, so its
    failure modes are stderr tracebacks in a compose restart loop rather than
    log lines. That is why empty is tolerated and bad input is not.
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
        """-1 is the universal "no limit" idiom, so it is exactly what an
        operator reaches for to spell out HISTORY_OUTBOX_MAX's default. It means
        the opposite downstream: the drainer's cap check treats it as an active
        cap of -1, computes dropped = depth + 1, and trims the outbox to empty
        on every cycle — on a HEALTHY system, since the cap is enforced on the
        drain success path too. Every un-archived play, gone every ~30s."""
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

    `assert HISTORY_OUTBOX_MAX == 0` passes even if the constant stops reading
    its variable entirely (a plain literal is also 0), and it fails on any
    machine where the documented variable happens to be exported. Reloading
    under a controlled environment fixes both.
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


class TestHistoryRedisCutoverParsing:
    """The only destructive switch in the system, so it refuses to guess.

    Every other boolean read in this repo silently treats an unrecognized value
    as false. That is the right default for a feature toggle and the wrong one
    here: `HISTORY_REDIS_CUTOVER=on` reading as OFF leaves an operator believing
    the migration shipped while the key keeps growing unbounded, and nothing at
    runtime contradicts them. Failing at the moment they are watching is the
    cheapest signal available.
    """

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_accepted_on_spellings(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HISTORY_REDIS_CUTOVER", value)
        assert src.config.history_redis_cutover() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", " off "])
    def test_accepted_off_spellings(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HISTORY_REDIS_CUTOVER", value)
        assert src.config.history_redis_cutover() is False

    def test_unset_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HISTORY_REDIS_CUTOVER", raising=False)
        assert src.config.history_redis_cutover() is False

    @pytest.mark.parametrize("value", ["enabled", "y", "nope", "2", "True!"])
    def test_unrecognized_values_raise_rather_than_read_as_off(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The failure this closes: a plausible-looking value that silently does
        # nothing. `y` and `enabled` are exactly what someone types when they
        # have not read the docs, and the old parser read both as OFF.
        monkeypatch.setenv("HISTORY_REDIS_CUTOVER", value)
        with pytest.raises(ValueError, match="HISTORY_REDIS_CUTOVER"):
            src.config.history_redis_cutover()

    def test_the_error_lists_the_values_that_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An operator who mistyped needs to be told what IS accepted, in the
        # same message — not sent to the source to find out.
        monkeypatch.setenv("HISTORY_REDIS_CUTOVER", "enabled")
        with pytest.raises(ValueError) as exc:
            src.config.history_redis_cutover()
        assert "'1'" in str(exc.value) and "'true'" in str(exc.value)


class TestDefaultPostgresPassword:
    """compose defaults POSTGRES_PASSWORD so `docker compose up` works with only
    a Discord token. The bot has to be able to tell that it did.

    Scoped to DSNs this project's tooling assembles from `.env`, which is the
    only supported place the password is set. Shapes asyncpg accepts but compose
    and `just run` cannot emit (`?password=`, `PGPASSWORD`, an unescaped `@` in
    the password) are deliberately undetected — see using_default_postgres_password
    for the ladder that would have to come back if an external Postgres ever
    becomes supported.
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
        # credential and must not slip past. Note the detector has to unquote it
        # ITSELF: SplitResult.password does not percent-decode, which is the
        # opposite of what an earlier version of this comment claimed.
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
            # The one that actually reaches the except arm. urlsplit only raises
            # on a malformed IPv6 literal in the NETLOC — and "://[bad" has no
            # "//" prefix, so its bracket lands in the path and the check never
            # runs. Every other case above returns normally, which meant the
            # `except ValueError` branch was dead to the whole suite: deleting
            # it, and making it `return True`, both passed 1,681 tests.
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
# `${POSTGRES_PASSWORD:?}` on the db-backfill service shipped: it broke
# `docker compose up` for a token-only stack — the exact thing the default
# exists to enable — while every test, ruff and pyright stayed green.
_COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.yml"


def _compose_directives() -> str:
    """docker-compose.yml with comment lines removed.

    Comments matter here: the file DESCRIBES the old mandatory form
    (`${VAR:?}`) while explaining why the default replaced it, so a naive scan
    of the raw text reports a violation that does not exist. Stripping whole
    comment lines is enough — Compose has no inline-comment-after-value form
    that would need finer handling in this file.
    """
    return "\n".join(
        line
        for line in _COMPOSE.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


class TestComposeMatchesTheDefault:
    """The two halves of the first-run promise, asserted against the real file.

    `docker compose up` must work with nothing configured but DISCORD_TOKEN, and
    the password it falls back to must be the one the bot warns about. Both are
    invisible to every other check in the repo.
    """

    def test_no_postgres_password_interpolation_is_mandatory(self) -> None:
        """REGRESSION: db-backfill kept `:?` when the other three services moved.

        Compose interpolates the WHOLE document before profile filtering, so an
        `ops`-profiled service with a mandatory variable fails `up`, `ps`,
        `logs` and `config` alike. Reproduced under `env -u POSTGRES_PASSWORD`:
        "required variable POSTGRES_PASSWORD is missing a value".
        """
        mandatory = re.findall(
            r"\$\{POSTGRES_PASSWORD:\?[^}]*\}", _compose_directives()
        )
        assert mandatory == []

    def test_every_fallback_is_the_password_the_bot_warns_about(self) -> None:
        """The drift check. DEFAULT_POSTGRES_PASSWORD is duplicated across
        config.py, build_common.sh and three compose services with nothing
        holding them together — and drift here fails OPEN: change compose's
        fallback alone and the detector goes permanently silent while the
        deployment still runs on a known credential.
        """
        fallbacks = set(
            re.findall(r"\$\{POSTGRES_PASSWORD:-([^}]*)\}", _compose_directives())
        )
        # Non-empty guard: a regex that matched nothing would make the equality
        # below trivially true, which is how this kind of test rots.
        assert len(fallbacks) >= 1
        assert fallbacks == {DEFAULT_POSTGRES_PASSWORD}

    def test_the_build_preflight_checks_for_the_same_password(self) -> None:
        """build_common.sh hardcodes the literal too, and it is the fifth copy.

        Nothing sourced it from anywhere, so drifting compose's fallback would
        leave the build-time warning checking for a value no deployment uses —
        silently, and in the fail-open direction, exactly like the detector.
        A shell script cannot import config.py, so the coupling is asserted here
        instead of enforced there.
        """
        preflight = (
            Path(__file__).resolve().parent.parent / "build_common.sh"
        ).read_text()
        assert f'= "{DEFAULT_POSTGRES_PASSWORD}"' in preflight

    def test_the_bot_and_the_migration_tiers_all_carry_a_default(self) -> None:
        # Count rather than merely "none mandatory": a service whose
        # POSTGRES_URL was deleted outright would also pass the first test.
        # Three interpolations = the bot, the postgres service, and the two
        # one-shots' DSNs.
        assert len(re.findall(r"\$\{POSTGRES_PASSWORD:", _compose_directives())) >= 4


class TestSetupEnvTightensTheEnvFile:
    """setup_env.sh is the escape hatch from the shared default, so the file it
    writes the replacement into must not be world-readable.

    Run rather than grepped, unlike the drift checks above. The behaviour is an
    interaction between `stat`, `chmod` and `mv` across two platforms' stat
    spellings, and a source-text assertion would hold just as well for a
    `chmod go-rwx` placed on the wrong side of the `mv` (that one dies with a
    non-zero exit rather than silently, but only because the temp file is gone
    by then — nothing about the grep would have told you which).

    The script `cd`s to its own directory, so it is copied into a tmpdir first:
    pointed at the checkout it would rewrite the developer's real .env and mint
    a password the running Postgres was never initialized with.
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
