"""Tests for src/config.py — the Spotify feature toggle."""

import pytest

from src.config import spotify_enabled


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
