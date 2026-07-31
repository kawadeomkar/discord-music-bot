"""Tests for src/config.py — the Spotify feature toggle and archive tunables."""

import pytest

from src.config import (
    HISTORY_OUTBOX_MAX,
    POSTGRES_STATEMENT_CACHE,
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
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
        setup_hook's required-DSN check passes and the real failure surfaces
        later as an asyncpg connect error instead of a startup refusal."""
        monkeypatch.setenv("POSTGRES_URL", "")
        assert postgres_url() is None

    def test_read_at_call_time_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert postgres_url() is None
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u@h/db")
        assert postgres_url() == "postgresql://u@h/db"


class TestArchiveTunables:
    def test_outbox_cap_defaults_to_unbounded(self) -> None:
        # 0 is the durability contract: an entry only leaves the outbox once
        # Postgres has it. A non-zero default would silently discard plays.
        assert HISTORY_OUTBOX_MAX == 0

    def test_statement_cache_defaults_to_asyncpg_default(self) -> None:
        assert POSTGRES_STATEMENT_CACHE == 100
