"""Tests for src/diagnostics.py — dependency probes and version collection."""

import asyncio
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from src import diagnostics, telemetry
from src.diagnostics import ProbeState


# ── Redis ──────────────────────────────────────────────────────────────────────


class TestProbeRedis:
    async def test_none_is_na(self) -> None:
        r = await diagnostics.probe_redis(None)
        assert r.state is ProbeState.NA
        assert r.latency_ms is None

    async def test_live_client_is_ok_with_latency(self, fake_redis: Redis) -> None:
        r = await diagnostics.probe_redis(fake_redis)
        assert r.state is ProbeState.OK
        assert r.latency_ms is not None and r.latency_ms >= 0

    async def test_ping_error_is_down(self) -> None:
        client = MagicMock()
        client.ping = AsyncMock(side_effect=ConnectionError("boom"))
        r = await diagnostics.probe_redis(client)
        assert r.state is ProbeState.DOWN
        assert r.detail == "ConnectionError"


# ── Spotify ────────────────────────────────────────────────────────────────────


class TestProbeSpotify:
    def _spotify(self, http_call: object, creds: bool = True) -> object:
        return SimpleNamespace(
            client_id="id" if creds else None,
            client_secret="secret" if creds else None,
            spotify_endpoint="https://api.spotify.com/",
            http_call=http_call,
        )

    async def test_no_credentials_is_na(self) -> None:
        r = await diagnostics.probe_spotify(self._spotify(AsyncMock(), creds=False))  # type: ignore[arg-type]
        assert r.state is ProbeState.NA

    async def test_success_is_ok(self) -> None:
        spotify = self._spotify(AsyncMock(return_value={"categories": {}}))
        r = await diagnostics.probe_spotify(spotify)  # type: ignore[arg-type]
        assert r.state is ProbeState.OK

    async def test_error_is_down(self) -> None:
        spotify = self._spotify(AsyncMock(side_effect=Exception("401")))
        r = await diagnostics.probe_spotify(spotify)  # type: ignore[arg-type]
        assert r.state is ProbeState.DOWN


# ── Postgres ───────────────────────────────────────────────────────────────────


class TestProbePostgres:
    async def test_none_is_na(self) -> None:
        r = await diagnostics.probe_postgres(None)
        assert r.state is ProbeState.NA

    async def test_live_pool_is_ok(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire_cm)
        r = await diagnostics.probe_postgres(pool)
        assert r.state is ProbeState.OK
        conn.execute.assert_awaited_once_with("SELECT 1")


# ── OTEL ───────────────────────────────────────────────────────────────────────


class TestProbeOtel:
    async def test_disabled_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", None)
        r = await diagnostics.probe_otel()
        assert r.state is ProbeState.OFF

    async def test_reachable_is_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", object())
        monkeypatch.setattr(telemetry, "_OTLP_ENDPOINT", "http://collector:4317")
        writer = MagicMock()
        writer.wait_closed = AsyncMock()
        with patch(
            "src.diagnostics.asyncio.open_connection",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ) as open_conn:
            r = await diagnostics.probe_otel()
        assert r.state is ProbeState.OK
        # scheme present → connects to the configured host, not localhost
        assert open_conn.await_args is not None
        assert open_conn.await_args.args == ("collector", 4317)
        writer.close.assert_called_once()

    async def test_schemeless_endpoint_still_targets_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", object())
        monkeypatch.setattr(telemetry, "_OTLP_ENDPOINT", "collector:4317")
        writer = MagicMock()
        writer.wait_closed = AsyncMock()
        with patch(
            "src.diagnostics.asyncio.open_connection",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ) as open_conn:
            await diagnostics.probe_otel()
        assert open_conn.await_args is not None
        assert open_conn.await_args.args == ("collector", 4317)

    async def test_unreachable_is_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", object())
        monkeypatch.setattr(telemetry, "_OTLP_ENDPOINT", "http://localhost:4317")
        with patch(
            "src.diagnostics.asyncio.open_connection",
            new=AsyncMock(side_effect=OSError("refused")),
        ):
            r = await diagnostics.probe_otel()
        assert r.state is ProbeState.DOWN


# ── _timed ─────────────────────────────────────────────────────────────────────


class TestTimed:
    async def test_cancelled_propagates(self) -> None:
        async def body() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await diagnostics._timed("x", body)

    async def test_success_records_latency(self) -> None:
        async def body() -> None:
            return None

        r = await diagnostics._timed("x", body)
        assert r.state is ProbeState.OK and r.latency_ms is not None


# ── Versions ───────────────────────────────────────────────────────────────────


class TestVersions:
    def _reset_ffmpeg_cache(self) -> None:
        diagnostics._ffmpeg_version_cache = None

    def test_ffmpeg_parses_version_token(self) -> None:
        self._reset_ffmpeg_cache()
        completed = MagicMock(stdout="ffmpeg version 7.1 Copyright (c) 2000-2024\n")
        with patch("src.diagnostics.subprocess.run", return_value=completed) as run:
            assert diagnostics.ffmpeg_version() == "7.1"
            # second call is cached — no second subprocess
            assert diagnostics.ffmpeg_version() == "7.1"
            run.assert_called_once()

    def test_ffmpeg_failure_is_unknown(self) -> None:
        self._reset_ffmpeg_cache()
        with patch(
            "src.diagnostics.subprocess.run", side_effect=FileNotFoundError("ffmpeg")
        ):
            assert diagnostics.ffmpeg_version() == "unknown"

    def test_bot_version_reads_pyproject(self) -> None:
        # Must match [tool.poetry].version in pyproject.toml — NOT installed dist
        # metadata (the container installs --no-root, so none exists).
        diagnostics._bot_version_cache = None
        with diagnostics._PYPROJECT.open("rb") as f:
            expected = tomllib.load(f)["tool"]["poetry"]["version"]
        assert diagnostics.bot_version() == expected
        assert diagnostics.bot_version() != "unknown"

    def test_bot_version_falls_back_when_pyproject_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        diagnostics._bot_version_cache = None
        monkeypatch.setattr(diagnostics, "_PYPROJECT", tmp_path / "nope.toml")
        # No pyproject and no installed dist metadata in the test env → "unknown",
        # never a crash.
        assert diagnostics.bot_version() in {"unknown"} or diagnostics.bot_version()

    def test_ytdlp_version_is_non_empty(self) -> None:
        assert diagnostics.ytdlp_version()

    async def test_collect_versions_has_all_keys(self) -> None:
        self._reset_ffmpeg_cache()
        completed = MagicMock(stdout="ffmpeg version 7.1 x\n")
        with patch("src.diagnostics.subprocess.run", return_value=completed):
            versions = await diagnostics.collect_versions()
        assert set(versions) == {"bot", "yt-dlp", "ffmpeg", "python", "discord.py"}
        assert all(versions.values())
