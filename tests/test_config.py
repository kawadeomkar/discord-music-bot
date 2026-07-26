"""Tests for src/config.py — the Spotify feature toggle and environment resolution."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from src import config
from src.config import SPOTIFY_TEST_TRACK_ID, SpotifyStatus, spotify_enabled


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


class TestEnvironmentResolution:
    """ENVIRONMENT must be a plain env-var read — no subprocess at import."""

    @pytest.fixture(autouse=True)
    def _restore_config_module(self) -> Iterator[None]:
        """These tests reload src.config, which rebinds a module-level constant
        for the whole session. Put it back so later tests (and any module that
        late-binds config.ENVIRONMENT) see the original value."""
        import importlib

        original = config.ENVIRONMENT
        yield
        importlib.reload(config)
        config.ENVIRONMENT = original

    def test_importing_config_runs_no_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The actual regression: a production process must not depend on a git
        binary and a repo, and the old import-time call raised a RuntimeWarning
        that `filterwarnings = ["error"]` turned into a collection failure in
        any detached worktree."""
        import importlib
        import subprocess

        calls: list[Any] = []

        def _boom(*args: Any, **kwargs: Any) -> None:
            calls.append(args)
            raise AssertionError("config imported must not shell out")

        monkeypatch.setattr(subprocess, "run", _boom)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        importlib.reload(config)
        assert calls == []
        assert config.ENVIRONMENT == "development"

    def test_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(config)
        assert config.ENVIRONMENT == "production"

    def test_empty_env_var_falls_back_to_development(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty string is a misconfiguration, not a valid environment name
        — it would render as `environment: ` in -ping and in every log line."""
        import importlib

        monkeypatch.setenv("ENVIRONMENT", "")
        importlib.reload(config)
        assert config.ENVIRONMENT == "development"


class TestInferEnvironmentFromGit:
    """Dev-only convenience, called from main() and never at import."""

    def _run(self, monkeypatch: pytest.MonkeyPatch, stdout: str, rc: int = 0) -> Any:
        import subprocess

        result = MagicMock()
        result.stdout = stdout
        result.returncode = rc
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)
        return config.infer_environment_from_git()

    def test_main_branch_is_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run(monkeypatch, "main\n") == "production"

    def test_feature_branch_slashes_become_dashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._run(monkeypatch, "task/some-feature\n") == "task-some-feature"

    def test_long_branch_name_is_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert len(self._run(monkeypatch, "x" * 200)) == 50

    def test_detached_head_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`git rev-parse --abbrev-ref HEAD` prints "HEAD" when detached — a
        normal state for a worktree, and previously the trigger for the warning
        that killed the suite."""
        assert self._run(monkeypatch, "HEAD\n") is None

    def test_non_zero_exit_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run(monkeypatch, "", rc=128) is None

    def test_empty_output_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run(monkeypatch, "  \n") is None

    def test_missing_git_binary_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def _missing(*a: Any, **k: Any) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", _missing)
        assert config.infer_environment_from_git() is None

    def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        def _slow(*a: Any, **k: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="git", timeout=2)

        monkeypatch.setattr(subprocess, "run", _slow)
        assert config.infer_environment_from_git() is None
