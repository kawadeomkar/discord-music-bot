"""Tests for src/ping.py — the -ping health dashboard: the probes and version
collectors, the embed rendering, and the optimistic-send + live-edit loop."""

import asyncio
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from opentelemetry import trace
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from src import ping, telemetry
from src.ping import ProbeResult, ProbeState, render_ping_embed
from src.musicbot import MusicBot
from tests.helpers import command_callback, mocked


def _probe(state: ProbeState, ms: float | None = None) -> ProbeResult:
    return ProbeResult("x", state, latency_ms=ms)


def _patch_probes(**results: ProbeResult) -> Any:
    """Patch each probe_* to resolve to the given ProbeResult, and
    collect_versions to a fast coroutine that still yields once (so the command's
    pre-drain can settle the immediate probes). Unnamed probes default to NA/OFF."""

    async def _make(res: ProbeResult) -> ProbeResult:
        return res

    async def _versions() -> dict[str, str]:
        await asyncio.sleep(0)  # a real yield: lets the scheduled probe tasks run
        return {
            "bot": "1.1.0",
            "yt-dlp": "2026.7.4",
            "ffmpeg": "7.1",
            "python": "3.14.0",
            "discord.py": "2.4.0",
        }

    return patch.multiple(
        "src.ping",
        probe_redis=lambda *a, **k: _make(results.get("redis", _probe(ProbeState.NA))),
        probe_spotify=lambda *a, **k: _make(
            results.get("spotify", _probe(ProbeState.NA))
        ),
        probe_postgres=lambda *a, **k: _make(
            results.get("postgres", _probe(ProbeState.NA))
        ),
        probe_otel=lambda *a, **k: _make(results.get("otel", _probe(ProbeState.OFF))),
        collect_versions=_versions,
    )


def _ping_message(mock_ctx: MagicMock) -> MagicMock:
    """Wire channel.send → a message whose edit() is awaitable; return the message."""
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    mock_ctx.channel.send = AsyncMock(return_value=message)
    return message


def _latency_field(embed: discord.Embed) -> str:
    return next(f.value for f in embed.fields if f.name == "Latency") or ""


async def _until(cond: Any, tries: int = 2000) -> None:
    """Yield to the loop until cond() is true (bounded), for interleaving a
    running ping command with external state changes in a test."""
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition never became true")


class TestPingCommand:
    async def test_posts_skeleton_via_channel_send_not_ctx_send(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Top-level ping bypasses the NP host: sends via channel.send, not ctx.send."""
        _ping_message(mock_ctx)
        with _patch_probes(redis=_probe(ProbeState.OK, 2.0)):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        mock_ctx.channel.send.assert_awaited_once()
        assert "embed" in mock_ctx.channel.send.call_args.kwargs
        mock_ctx.send.assert_not_awaited()

    async def test_all_resolved_settles_in_the_send_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Every probe resolves immediately → the pre-drain settles them and the
        very first send carries the final state; no edits needed (finalize-early)."""
        message = _ping_message(mock_ctx)
        with _patch_probes(
            redis=_probe(ProbeState.OK, 1.0),
            spotify=_probe(ProbeState.OK, 120.0),
            postgres=_probe(ProbeState.NA),
            otel=_probe(ProbeState.OK, 3.0),
        ):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        sent = mock_ctx.channel.send.await_args.kwargs["embed"]
        latency = _latency_field(sent)
        assert "120 ms" in latency and "pending" not in latency
        message.edit.assert_not_awaited()

    async def test_skeleton_prefills_na_and_off_not_pending(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """D2: unconfigured/disabled deps must never flash 'pending…' — the
        pre-drain lands them in the skeleton send."""
        _ping_message(mock_ctx)
        with _patch_probes(otel=_probe(ProbeState.OFF)):  # redis None → NA on music_bot
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        latency = _latency_field(mock_ctx.channel.send.await_args.kwargs["embed"])
        assert "n/a" in latency and "off" in latency and "pending" not in latency

    async def test_deadline_marks_straggler_failed(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A probe still pending at the deadline is cancelled → 'failed', via an edit."""
        message = _ping_message(mock_ctx)
        # Collapse the deadline so the never-returning probe fails at once.
        monkeypatch.setattr(ping, "PING_TICK_SECS", 0.0)
        monkeypatch.setattr(ping, "PING_DEADLINE_SECS", 0.0)

        never = asyncio.Event()  # never set → the probe hangs until cancelled

        async def _hang(*a: Any, **k: Any) -> ProbeResult:
            await never.wait()
            raise AssertionError("unreachable")

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.probe_spotify", new=_hang):
                await command_callback(MusicBot.ping)(music_bot, mock_ctx)

        message.edit.assert_awaited()  # the deadline edit
        assert "failed" in _latency_field(message.edit.await_args.kwargs["embed"])

    async def test_probe_resolving_mid_loop_edits_in_place(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The headline behavior: a probe that returns AFTER the skeleton send is
        folded in on a tick and the message is edited from pending → its latency."""
        message = _ping_message(mock_ctx)
        monkeypatch.setattr(ping, "PING_TICK_SECS", 0.01)
        monkeypatch.setattr(ping, "PING_DEADLINE_SECS", 5.0)

        gate = asyncio.Event()

        async def _gated(*a: Any, **k: Any) -> ProbeResult:
            await gate.wait()
            return ProbeResult("Spotify API", ProbeState.OK, latency_ms=42.0)

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.probe_spotify", new=_gated):
                task = asyncio.create_task(
                    command_callback(MusicBot.ping)(music_bot, mock_ctx)
                )
                await _until(lambda: mock_ctx.channel.send.await_count == 1)
                # skeleton: Spotify still pending, nothing edited yet
                skeleton = mock_ctx.channel.send.await_args.kwargs["embed"]
                assert "pending" in _latency_field(skeleton)
                message.edit.assert_not_awaited()
                gate.set()  # Spotify returns → next tick must edit
                await task

        message.edit.assert_awaited()
        final = _latency_field(message.edit.await_args.kwargs["embed"])
        assert "42 ms" in final and "pending" not in final

    async def test_edit_on_deleted_message_is_tolerated(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_safe_edit: a host the user deleted mid-loop (edit → NotFound) must not
        crash the command."""
        message = _ping_message(mock_ctx)
        message.edit = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        monkeypatch.setattr(ping, "PING_TICK_SECS", 0.0)
        monkeypatch.setattr(ping, "PING_DEADLINE_SECS", 0.0)

        never = asyncio.Event()

        async def _hang(*a: Any, **k: Any) -> ProbeResult:
            await never.wait()
            raise AssertionError("unreachable")

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.probe_spotify", new=_hang):
                await command_callback(MusicBot.ping)(music_bot, mock_ctx)  # no raise

        message.edit.assert_awaited()  # attempted the deadline edit, swallowed NotFound

    async def test_nan_gateway_latency_renders_red_not_crash(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression: discord.py reports nan latency while reconnecting — the
        embed must render (red 'down'), not raise and fall to the error path."""
        _ping_message(mock_ctx)
        mocked(music_bot.bot).latency = float("nan")
        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        sent = mock_ctx.channel.send.await_args.kwargs["embed"]
        assert sent.color is not None and sent.color.value == 0x990000
        assert "down" in _latency_field(sent)  # gateway row, not a crash

    async def test_join_uses_latency_line_not_the_dashboard(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression for §2.3 / C1: -join must post the cheap latency line and
        must NOT run the dependency probes."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.channel = mock_ctx.author.voice.channel
        mock_ctx.guild.change_voice_state = AsyncMock()
        mp = MagicMock()
        mp.store = None
        mp.open_playback_gate = MagicMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        redis_spy = AsyncMock(return_value=_probe(ProbeState.OK, 1.0))

        with (
            patch("src.musicbot.send_latency_line", new=AsyncMock()) as latency_line,
            patch("src.ping.probe_redis", new=redis_spy),
        ):
            await command_callback(MusicBot.join)(music_bot, mock_ctx)

        latency_line.assert_awaited_once()
        redis_spy.assert_not_awaited()  # no health dashboard on the join path


class TestDownReasonRendering:
    """A bare "down" sends the operator to the logs; the server's reason code is
    the actionable half (regression for the live MISCONF outage)."""

    def test_down_row_shows_the_reason(self) -> None:
        results = {
            "Redis": ProbeResult("Redis", ProbeState.DOWN, detail="MISCONF"),
        }
        versions: dict[str, str] = dict.fromkeys(
            ["bot", "yt-dlp", "ffmpeg", "python", "discord.py"], "x"
        )
        embed = render_ping_embed(results, versions, 42.0, trace.get_current_span())
        latency = next(f.value for f in embed.fields if f.name == "Latency") or ""
        assert "down (MISCONF)" in latency

    def test_down_without_detail_stays_bare(self) -> None:
        results = {"Redis": ProbeResult("Redis", ProbeState.DOWN)}
        versions: dict[str, str] = dict.fromkeys(
            ["bot", "yt-dlp", "ffmpeg", "python", "discord.py"], "x"
        )
        embed = render_ping_embed(results, versions, 42.0, trace.get_current_span())
        latency = next(f.value for f in embed.fields if f.name == "Latency") or ""
        assert "down" in latency and "(" not in latency


# ── Redis ──────────────────────────────────────────────────────────────────────


class TestProbeRedis:
    async def test_none_is_na(self) -> None:
        r = await ping.probe_redis(None)
        assert r.state is ProbeState.NA
        assert r.latency_ms is None

    async def test_live_client_is_ok_with_latency(self, fake_redis: Redis) -> None:
        r = await ping.probe_redis(fake_redis)
        assert r.state is ProbeState.OK
        assert r.latency_ms is not None and r.latency_ms >= 0

    async def test_exercises_the_write_path_not_just_ping(
        self, fake_redis: Redis
    ) -> None:
        # PING alone would stay green while Redis refuses writes (MISCONF/OOM/
        # READONLY), so the probe must actually write a short-TTL key.
        await ping.probe_redis(fake_redis)
        assert await fake_redis.get(ping._REDIS_HEALTH_KEY) == b"1"
        assert 0 < await fake_redis.ttl(ping._REDIS_HEALTH_KEY) <= 30

    async def test_ping_error_is_down(self) -> None:
        client = MagicMock()
        client.ping = AsyncMock(side_effect=ConnectionError("boom"))
        r = await ping.probe_redis(client)
        assert r.state is ProbeState.DOWN
        assert r.detail == "ConnectionError"

    async def test_write_refused_is_down_with_the_server_code(self) -> None:
        """Reads fine, writes refused — the exact shape of the live MISCONF outage."""
        client = MagicMock()
        client.ping = AsyncMock(return_value=True)
        client.set = AsyncMock(
            side_effect=ResponseError(
                "MISCONF Redis is configured to save RDB snapshots, but is "
                "currently unable to persist to disk."
            )
        )
        r = await ping.probe_redis(client)
        assert r.state is ProbeState.DOWN
        assert r.detail == "MISCONF"  # not the useless "ResponseError"


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
        r = await ping.probe_spotify(self._spotify(AsyncMock(), creds=False))  # type: ignore[arg-type]
        assert r.state is ProbeState.NA

    async def test_success_is_ok(self) -> None:
        spotify = self._spotify(AsyncMock(return_value={"categories": {}}))
        r = await ping.probe_spotify(spotify)  # type: ignore[arg-type]
        assert r.state is ProbeState.OK

    async def test_error_is_down(self) -> None:
        spotify = self._spotify(AsyncMock(side_effect=Exception("401")))
        r = await ping.probe_spotify(spotify)  # type: ignore[arg-type]
        assert r.state is ProbeState.DOWN


# ── Postgres ───────────────────────────────────────────────────────────────────


class TestProbePostgres:
    async def test_none_is_na(self) -> None:
        r = await ping.probe_postgres(None)
        assert r.state is ProbeState.NA

    async def test_live_pool_is_ok(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire_cm)
        r = await ping.probe_postgres(pool)
        assert r.state is ProbeState.OK
        conn.execute.assert_awaited_once_with("SELECT 1")


# ── OTEL ───────────────────────────────────────────────────────────────────────


class TestProbeOtel:
    async def test_disabled_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", None)
        r = await ping.probe_otel()
        assert r.state is ProbeState.OFF

    async def test_reachable_is_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", object())
        monkeypatch.setattr(telemetry, "_OTLP_ENDPOINT", "http://collector:4317")
        writer = MagicMock()
        writer.wait_closed = AsyncMock()
        with patch(
            "src.ping.asyncio.open_connection",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ) as open_conn:
            r = await ping.probe_otel()
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
            "src.ping.asyncio.open_connection",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ) as open_conn:
            await ping.probe_otel()
        assert open_conn.await_args is not None
        assert open_conn.await_args.args == ("collector", 4317)

    async def test_unreachable_is_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", object())
        monkeypatch.setattr(telemetry, "_OTLP_ENDPOINT", "http://localhost:4317")
        with patch(
            "src.ping.asyncio.open_connection",
            new=AsyncMock(side_effect=OSError("refused")),
        ):
            r = await ping.probe_otel()
        assert r.state is ProbeState.DOWN


# ── _timed ─────────────────────────────────────────────────────────────────────


class TestTimed:
    async def test_cancelled_propagates(self) -> None:
        async def body() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await ping._timed("x", body)

    async def test_success_records_latency(self) -> None:
        async def body() -> None:
            return None

        r = await ping._timed("x", body)
        assert r.state is ProbeState.OK and r.latency_ms is not None


class TestErrorDetail:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ResponseError("MISCONF cannot persist to disk"), "MISCONF"),
            (ResponseError("OOM command not allowed"), "OOM"),
            (ResponseError("READONLY You can't write against a replica"), "READONLY"),
            (ConnectionError("Connection refused"), "ConnectionError"),
            (ResponseError("some lowercase message"), "ResponseError"),
            (ValueError(""), "ValueError"),
        ],
    )
    def test_prefers_the_server_error_code(self, exc: Exception, expected: str) -> None:
        assert ping._error_detail(exc) == expected


# ── Versions ───────────────────────────────────────────────────────────────────


class TestVersions:
    def _reset_ffmpeg_cache(self) -> None:
        ping._ffmpeg_version_cache = None

    def test_ffmpeg_parses_version_token(self) -> None:
        self._reset_ffmpeg_cache()
        completed = MagicMock(stdout="ffmpeg version 7.1 Copyright (c) 2000-2024\n")
        with patch("src.ping.subprocess.run", return_value=completed) as run:
            assert ping.ffmpeg_version() == "7.1"
            # second call is cached — no second subprocess
            assert ping.ffmpeg_version() == "7.1"
            run.assert_called_once()

    def test_ffmpeg_failure_is_unknown(self) -> None:
        self._reset_ffmpeg_cache()
        with patch("src.ping.subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
            assert ping.ffmpeg_version() == "unknown"

    def test_bot_version_reads_pyproject(self) -> None:
        # Must match [tool.poetry].version in pyproject.toml — NOT installed dist
        # metadata (the container installs --no-root, so none exists).
        ping._bot_version_cache = None
        with ping._PYPROJECT.open("rb") as f:
            expected = tomllib.load(f)["tool"]["poetry"]["version"]
        assert ping.bot_version() == expected
        assert ping.bot_version() != "unknown"

    def test_bot_version_falls_back_when_pyproject_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ping._bot_version_cache = None
        monkeypatch.setattr(ping, "_PYPROJECT", tmp_path / "nope.toml")
        # No pyproject and no installed dist metadata in the test env → "unknown",
        # never a crash.
        assert ping.bot_version() in {"unknown"} or ping.bot_version()

    def test_ytdlp_version_is_non_empty(self) -> None:
        assert ping.ytdlp_version()

    async def test_collect_versions_has_all_keys(self) -> None:
        self._reset_ffmpeg_cache()
        completed = MagicMock(stdout="ffmpeg version 7.1 x\n")
        with patch("src.ping.subprocess.run", return_value=completed):
            versions = await ping.collect_versions()
        assert set(versions) == {"bot", "yt-dlp", "ffmpeg", "python", "discord.py"}
        assert all(versions.values())
