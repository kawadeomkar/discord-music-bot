"""Tests for src/config.py — the Spotify feature toggle and archive tunables."""

import importlib
import os
from collections.abc import Iterator
from types import ModuleType

import pytest

import src.config
from src.config import (
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
    _int_env,
    postgres_url,
    spotify_enabled,
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
