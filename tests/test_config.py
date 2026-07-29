"""Tests for src/config.py — the Spotify feature toggle and Postgres knobs."""

import pytest

from src.config import (
    DEFAULT_POSTGRES_PASSWORD,
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
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


class TestDefaultPostgresPassword:
    """compose defaults POSTGRES_PASSWORD so `docker compose up` works with only
    a Discord token. The bot has to be able to tell that it did."""

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
        # urlsplit percent-decodes, so an escaped form cannot slip past.
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:%70assword@127.0.0.1:5432/musicbot"
        )
        assert using_default_postgres_password() is True

    @pytest.mark.parametrize(
        "url", ["not-a-url", "postgresql://", "postgresql://user@host/db", "://[bad"]
    )
    def test_never_raises_on_a_malformed_dsn(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        # It feeds a startup warning and a -ping row; a malformed DSN is the
        # archive's problem to report, not this function's.
        monkeypatch.setenv("POSTGRES_URL", url)
        assert using_default_postgres_password() is False
