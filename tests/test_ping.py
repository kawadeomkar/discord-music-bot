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
from src.config import SpotifyStatus
from src.ping import (
    ProbeResult,
    ProbeState,
    default_password_embed,
    render_ping_embed,
)
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


def _health_embed(call: Any) -> discord.Embed:
    """The health card out of a send/edit call. The dashboard sends a list (a
    default-password advisory can ride in front of the card), and the health card
    is always last — which is also what _safe_edit returns to the diffing loop."""
    return call.kwargs["embeds"][-1]


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
        assert "embeds" in mock_ctx.channel.send.call_args.kwargs
        mock_ctx.send.assert_not_awaited()

    async def test_a_probe_that_raises_costs_its_row_not_the_whole_board(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """_timed guards a probe's BODY, not everything a probe does — probe_otel
        parses a URL first. A settle callback runs on the driver's own task, so an
        unguarded raise there took the board down before it was ever sent.
        """
        message = _ping_message(mock_ctx)

        async def exploding() -> ping.ProbeResult:
            raise ValueError("port could not be cast")

        with (
            _patch_probes(redis=_probe(ProbeState.OK, 1.0)),
            patch("src.ping.probe_otel", exploding),
        ):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)

        mock_ctx.channel.send.assert_awaited_once()  # the board still exists
        call = message.edit.await_args or mock_ctx.channel.send.await_args
        rendered = str(_health_embed(call).to_dict())
        assert "OTEL collector" in rendered
        assert ProbeState.FAILED.value in rendered

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
        sent = _health_embed(mock_ctx.channel.send.await_args)
        latency = _latency_field(sent)
        assert "120 ms" in latency and "pending" not in latency
        message.edit.assert_not_awaited()

    async def test_skeleton_prefills_na_and_off_not_pending(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unconfigured/disabled deps must never flash 'pending…' — the pre-drain
        lands them in the skeleton send."""
        _ping_message(mock_ctx)
        with _patch_probes(otel=_probe(ProbeState.OFF)):  # redis None → NA on music_bot
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        latency = _latency_field(_health_embed(mock_ctx.channel.send.await_args))
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
        assert "failed" in _latency_field(_health_embed(message.edit.await_args))

    async def test_probe_resolving_mid_loop_edits_in_place(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The headline behavior: a probe that returns after the skeleton send is
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
                skeleton = _health_embed(mock_ctx.channel.send.await_args)
                assert "pending" in _latency_field(skeleton)
                message.edit.assert_not_awaited()
                gate.set()  # Spotify returns → next tick must edit
                await task

        message.edit.assert_awaited()
        final = _latency_field(_health_embed(message.edit.await_args))
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
        sent = _health_embed(mock_ctx.channel.send.await_args)
        assert sent.color is not None and sent.color.value == 0x990000
        assert "down" in _latency_field(sent)  # gateway row, not a crash

    async def test_join_uses_latency_line_not_the_dashboard(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression: -join must post the cheap latency line and
        must not run the dependency probes."""
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


class TestPingReportsSpotifySource:
    """End-to-end: the Spotify row reports the *source's* startup state — never
    configured, configured-but-rejected, or live — so a user can see why Spotify
    links are declined without reading the logs. probe_spotify is left unpatched
    here: the cog's _spotify_status is what is under test."""

    def _patch_everything_but_spotify(self) -> Any:
        async def _make(res: ProbeResult) -> ProbeResult:
            return res

        async def _versions() -> dict[str, str]:
            await asyncio.sleep(0)  # yield so the immediate probes settle
            return dict.fromkeys(
                ["bot", "yt-dlp", "ffmpeg", "python", "discord.py"], "x"
            )

        return patch.multiple(
            "src.ping",
            probe_redis=lambda *a, **k: _make(_probe(ProbeState.NA)),
            probe_postgres=lambda *a, **k: _make(_probe(ProbeState.NA)),
            probe_otel=lambda *a, **k: _make(_probe(ProbeState.OFF)),
            collect_versions=_versions,
        )

    async def _run(self, music_bot: MusicBot, mock_ctx: MagicMock) -> str:
        _ping_message(mock_ctx)
        with self._patch_everything_but_spotify():
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        return _latency_field(_health_embed(mock_ctx.channel.send.await_args))

    async def test_no_credentials_row_says_not_configured(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.spotify = None
        music_bot._spotify_status = SpotifyStatus.DISABLED
        latency = await self._run(music_bot, mock_ctx)
        assert "Spotify API" in latency
        assert "n/a (not configured)" in latency

    async def test_rejected_credentials_row_says_so(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The startup probe found the credentials invalid: the row is red and
        names the cause rather than reporting a generic outage."""
        music_bot._spotify_status = SpotifyStatus.INVALID
        latency = await self._run(music_bot, mock_ctx)
        assert "down (credentials rejected)" in latency

    async def test_validated_credentials_row_is_a_live_latency(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mocked(music_bot.spotify).http_call = AsyncMock(return_value={"x": 1})
        music_bot._spotify_status = SpotifyStatus.ENABLED
        latency = await self._run(music_bot, mock_ctx)
        assert "ms" in latency
        assert "rejected" not in latency and "not configured" not in latency


class TestPingReportsPostgres:
    """End-to-end for the Postgres row, probe_postgres left UNPATCHED: the wiring
    from the cog's history_archive through the dashboard to health_check is what
    is under test. The probe once read a `pg_pool` attribute nothing ever set, so
    the row was a permanent "n/a" while every unit test of it passed."""

    def _patch_everything_but_postgres(self) -> Any:
        async def _make(res: ProbeResult) -> ProbeResult:
            return res

        async def _versions() -> dict[str, str]:
            await asyncio.sleep(0)  # yield so the immediate probes settle
            return dict.fromkeys(
                ["bot", "yt-dlp", "ffmpeg", "python", "discord.py"], "x"
            )

        return patch.multiple(
            "src.ping",
            probe_redis=lambda *a, **k: _make(_probe(ProbeState.NA)),
            probe_spotify=lambda *a, **k: _make(_probe(ProbeState.NA)),
            probe_otel=lambda *a, **k: _make(_probe(ProbeState.OFF)),
            collect_versions=_versions,
        )

    async def _run(self, music_bot: MusicBot, mock_ctx: MagicMock) -> str:
        """The Postgres ROW, not the whole latency field: the other probes are
        patched to n/a/off, so a field-wide `"n/a" not in ...` would match their
        rows and never fail no matter what Postgres reported."""
        _ping_message(mock_ctx)
        with self._patch_everything_but_postgres():
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        field = _latency_field(_health_embed(mock_ctx.channel.send.await_args))
        return next(line for line in field.splitlines() if "Postgres" in line)

    async def test_healthy_archive_row_is_a_live_latency(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = MagicMock()
        archive.health_check = AsyncMock()
        music_bot.history_archive = archive
        row = await self._run(music_bot, mock_ctx)
        assert "ms" in row
        assert "n/a" not in row and "down" not in row
        archive.health_check.assert_awaited_once_with()

    async def test_unreachable_archive_row_is_down(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = MagicMock()
        archive.health_check = AsyncMock(side_effect=OSError("connection refused"))
        music_bot.history_archive = archive
        row = await self._run(music_bot, mock_ctx)
        assert "down (OSError)" in row

    async def test_no_archive_row_is_na(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Only reachable outside a real MusicBotApp — setup_hook refuses to start
        # without POSTGRES_URL — but it is what keeps the probe renderable.
        music_bot.history_archive = None
        row = await self._run(music_bot, mock_ctx)
        assert "n/a" in row

    async def test_the_disabled_archive_row_explains_itself(
        self, music_bot: MusicBot, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ship default, asserted on the RENDERED row. probe_postgres sets
        detail="archive disabled", but _ping_value once appended a detail only for
        DOWN and NA, so the row read as a bare "off". Asserting the detail on the
        ProbeResult instead pins something no user can see."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        music_bot.history_archive = None
        row = await self._run(music_bot, mock_ctx)
        assert "off (archive disabled)" in row


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

    async def test_none_is_na(self) -> None:
        # spotify is None when the bot was started without Spotify credentials
        # (optional-Spotify feature): the probe reports N/A, not a failure.
        r = await ping.probe_spotify(None, SpotifyStatus.DISABLED)
        assert r.state is ProbeState.NA
        assert r.detail == "not configured"

    async def test_no_credentials_is_na(self) -> None:
        r = await ping.probe_spotify(self._spotify(AsyncMock(), creds=False))  # type: ignore[arg-type]
        assert r.state is ProbeState.NA

    async def test_disabled_status_is_na_without_calling_the_api(self) -> None:
        """disabled wins even if a client object is somehow present: no API call."""
        http_call = AsyncMock()
        r = await ping.probe_spotify(self._spotify(http_call), SpotifyStatus.DISABLED)  # type: ignore[arg-type]
        assert r.state is ProbeState.NA
        http_call.assert_not_awaited()

    async def test_invalid_status_is_down_without_calling_the_api(self) -> None:
        """Startup validation already saw Spotify reject these credentials — the
        row says so instead of spending a doomed request to re-earn the 401."""
        http_call = AsyncMock()
        r = await ping.probe_spotify(self._spotify(http_call), SpotifyStatus.INVALID)  # type: ignore[arg-type]
        assert r.state is ProbeState.DOWN
        assert r.detail == "credentials rejected"
        http_call.assert_not_awaited()

    async def test_success_is_ok(self) -> None:
        spotify = self._spotify(AsyncMock(return_value={"categories": {}}))
        r = await ping.probe_spotify(spotify, SpotifyStatus.ENABLED)  # type: ignore[arg-type]
        assert r.state is ProbeState.OK

    async def test_error_is_down(self) -> None:
        spotify = self._spotify(AsyncMock(side_effect=Exception("401")))
        r = await ping.probe_spotify(spotify, SpotifyStatus.ENABLED)  # type: ignore[arg-type]
        assert r.state is ProbeState.DOWN


# ── Postgres ───────────────────────────────────────────────────────────────────


class TestProbePostgres:
    async def test_none_with_the_flag_on_is_na(self) -> None:
        # The suite default pins HISTORY_ARCHIVE_ENABLED=true, so a None
        # archive here means "constructed without an archive" (tests, a cog
        # built outside MusicBotApp) — NA, not OFF.
        r = await ping.probe_postgres(None)
        assert r.state is ProbeState.NA

    async def test_none_with_the_archive_disabled_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The ship default. OFF is "deliberately disabled" (the OTEL row's
        # state): the operator's choice must read as a choice on the
        # dashboard, not as a fault or a shrug.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        r = await ping.probe_postgres(None)
        assert r.state is ProbeState.OFF
        assert r.detail == "archive disabled"

    async def test_healthy_archive_is_ok(self) -> None:
        archive = MagicMock()
        archive.health_check = AsyncMock()
        r = await ping.probe_postgres(archive)
        assert r.state is ProbeState.OK
        assert r.latency_ms is not None
        archive.health_check.assert_awaited_once_with()

    async def test_unreachable_archive_is_down_with_reason(self) -> None:
        # Why the probe asks the archive rather than a pool: a bot whose
        # Postgres is unreachable reports DOWN, not the "n/a (not configured)"
        # that a lazily-absent pool would have produced.
        archive = MagicMock()
        archive.health_check = AsyncMock(
            side_effect=ConnectionRefusedError("no server")
        )
        r = await ping.probe_postgres(archive)
        assert r.state is ProbeState.DOWN
        assert r.detail == "ConnectionRefusedError"

    async def test_cancellation_propagates(self) -> None:
        # _timed re-raises CancelledError so the dashboard's deadline can flip a
        # straggling probe to FAILED itself. A first connect can outlast that
        # deadline, so Postgres is the probe most likely to take this path.
        archive = MagicMock()
        archive.health_check = AsyncMock(side_effect=asyncio.CancelledError)
        with pytest.raises(asyncio.CancelledError):
            await ping.probe_postgres(archive)


# ── OTEL ───────────────────────────────────────────────────────────────────────


class TestProbeOtel:
    async def test_disabled_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(telemetry, "_tracer_provider", None)
        r = await ping.probe_otel()
        assert r.state is ProbeState.OFF

    @pytest.mark.parametrize(
        "endpoint",
        [
            "grpc://collector:4317 ",  # a trailing space in .env
            "http://collector:99999",  # out of range
        ],
    )
    async def test_an_unparseable_port_fails_the_row_not_the_board(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        """urlparse().port is LAZY — it accepts these and raises on dereference. That
        read sits outside _timed, so before the guard a stray character in .env raised
        through _settle and out of the dashboard driver BEFORE the skeleton send, and
        -ping answered with a bare error embed and no board at all.
        """
        monkeypatch.setattr(telemetry, "_tracer_provider", object())
        monkeypatch.setattr(telemetry, "_OTLP_ENDPOINT", endpoint)
        r = await ping.probe_otel()
        assert r.state is ProbeState.FAILED

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
            # Second call is cached — no second subprocess
            assert ping.ffmpeg_version() == "7.1"
            run.assert_called_once()

    def test_ffmpeg_failure_is_unknown(self) -> None:
        self._reset_ffmpeg_cache()
        with patch("src.ping.subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
            assert ping.ffmpeg_version() == "unknown"

    def test_bot_version_reads_pyproject(self) -> None:
        # Must match [tool.poetry].version in pyproject.toml — not installed dist
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


class TestDefaultPasswordEmbed:
    """The standing advisory. Rendered on every -ping rather than once at
    startup: the startup log is seen once, usually by whoever already knows,
    while -ping is the surface an operator returns to."""

    def test_none_when_the_password_is_real(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:9f3a1c@127.0.0.1:5432/musicbot"
        )
        assert default_password_embed() is None

    def test_none_when_postgres_is_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert default_password_embed() is None

    def test_none_when_the_archive_is_disabled_even_with_the_default_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Compose interpolates POSTGRES_URL into the bot's environment whether or
        # not the archive profile is active, so a disabled bot can hold this DSN
        # with no Postgres behind it. Warning about a database that isn't there
        # trains operators to ignore warnings — same gate as setup_hook's ERROR.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        assert default_password_embed() is None

    def test_warns_and_says_how_to_fix_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        embed = default_password_embed()
        assert embed is not None
        assert "Default database password" in (embed.title or "")
        body = (embed.description or "") + "".join(f.value or "" for f in embed.fields)
        assert "setup_env.sh" in body
        assert "ALTER USER" in body
        assert "up -d" in body  # the container recreate the old remedy omitted

        # Order, not just presence. The detector reads the bot's DSN, so running
        # `setup_env.sh --force` first clears the warning while Postgres still
        # accepts the old password — an unwarned exposure. The server change has
        # to come first and the embed has to say so.
        assert body.index("ALTER USER") < body.index("setup_env.sh")
        assert "order is the point" in body
        # By STEP NUMBER too: relabelling the steps without moving them would
        # leave the text order intact while telling the operator to do the
        # silencing step first.
        assert body.index("1.") < body.index("ALTER USER") < body.index("2.")

    def test_is_static_so_the_edit_loop_still_diffs_only_health(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # run_health_dashboard computes it once and re-attaches the same embed on
        # every edit; if it varied per call the dashboard would either flicker or
        # edit on every tick.
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        first, second = default_password_embed(), default_password_embed()
        assert first is not None and second is not None
        assert first.to_dict() == second.to_dict()


class TestDefaultPasswordWarningReachesTheWire:
    """The advisory on the actual dashboard, not in isolation.

    TestDefaultPasswordEmbed covers default_password_embed() alone, which left the
    whole surface deletable: replacing the dashboard's `warning =
    default_password_embed()` with `None` passed the entire suite. These need a
    pinned POSTGRES_URL — conftest scrubs it, and they set the default explicitly,
    because otherwise the suite reads a developer shell carrying a real password."""

    @pytest.fixture
    def on_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )

    async def test_the_skeleton_send_leads_with_the_warning(
        self, music_bot: MusicBot, mock_ctx: MagicMock, on_default: None
    ) -> None:
        _ping_message(mock_ctx)
        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        embeds = mock_ctx.channel.send.await_args.kwargs["embeds"]
        assert len(embeds) == 2
        # First, above the health card — an advisory below the fold is one the
        # operator scrolls past.
        assert "Default database password" in (embeds[0].title or "")

    async def test_a_real_password_sends_only_the_health_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other side: without this, "always two embeds" would pass the test
        # above just as well as the correct behaviour.
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:9f3a1c@127.0.0.1:5432/musicbot"
        )
        _ping_message(mock_ctx)
        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        assert len(mock_ctx.channel.send.await_args.kwargs["embeds"]) == 1

    async def test_the_warning_survives_the_deadline_edit(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        on_default: None,
    ) -> None:
        """Every edit re-attaches it, or the advisory disappears the moment a
        probe resolves late — i.e. on exactly the unhealthy deployments most
        likely to be running unattended."""
        message = _ping_message(mock_ctx)
        monkeypatch.setattr(ping, "PING_TICK_SECS", 0.0)
        monkeypatch.setattr(ping, "PING_DEADLINE_SECS", 0.0)

        never = asyncio.Event()

        async def _hang(*a: Any, **k: Any) -> ProbeResult:
            await never.wait()
            raise AssertionError("unreachable")

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.probe_spotify", new=_hang):
                await command_callback(MusicBot.ping)(music_bot, mock_ctx)

        message.edit.assert_awaited()
        embeds = message.edit.await_args.kwargs["embeds"]
        assert len(embeds) == 2
        assert "Default database password" in (embeds[0].title or "")

    async def test_the_warning_survives_a_mid_loop_tick_edit(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        on_default: None,
    ) -> None:
        # The third write path. Send, tick-edit and deadline-edit are separate
        # call sites in run_health_dashboard, and dropping the warning from any
        # one of them survived the suite before these tests existed.
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
                gate.set()
                await _until(lambda: message.edit.await_count >= 1)
                await task

        embeds = message.edit.await_args.kwargs["embeds"]
        assert len(embeds) == 2
        assert "Default database password" in (embeds[0].title or "")


class TestTheAdvisoryIsForTheOperator:
    """Who sees the default-password warning. -ping has no permission check and
    answers to `-status`, `-health`, `-l`, so an ungated advisory confirms to every
    member of every guild — permanently, in Discord's history — that this host runs
    the public default. The leak is the confirmation, not the string."""

    @pytest.fixture(autouse=True)
    def on_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )

    async def test_everyone_else_gets_the_health_card_alone(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Same deployment, same misconfiguration — only the audience differs.
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)
        _ping_message(mock_ctx)
        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)
        embeds = mock_ctx.channel.send.await_args.kwargs["embeds"]
        assert len(embeds) == 1
        assert "Default database password" not in (embeds[0].title or "")

    async def test_a_non_owner_never_sees_it_on_an_edit_either(
        self, music_bot: MusicBot, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The gate is evaluated once, before the send — so it has to hold for
        # the edit paths too, or the advisory leaks on the first late probe.
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)
        message = _ping_message(mock_ctx)
        monkeypatch.setattr(ping, "PING_TICK_SECS", 0.0)
        monkeypatch.setattr(ping, "PING_DEADLINE_SECS", 0.0)

        never = asyncio.Event()

        async def _hang(*a: Any, **k: Any) -> ProbeResult:
            await never.wait()
            raise AssertionError("unreachable")

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.probe_spotify", new=_hang):
                await command_callback(MusicBot.ping)(music_bot, mock_ctx)

        message.edit.assert_awaited()
        assert len(message.edit.await_args.kwargs["embeds"]) == 1

    async def test_the_owner_check_is_never_paid_for_when_there_is_no_advisory(
        self, music_bot: MusicBot, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_owner() must be reached only when the advisory exists — it is not a
        local predicate: with no owner_id set, discord.py falls through to
        application_info(), a REST GET that retries ~25s on a 5xx and then RAISES,
        ahead of the skeleton send. is_owner answers True on purpose here, so a
        regression cannot pass by answering False."""
        # The ship default. Default_password_embed() returns None on its first
        # line, so there is nothing to gate and nothing to ask about.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        mock_ctx.bot.is_owner = AsyncMock(
            return_value=True, side_effect=RuntimeError("application_info() 500")
        )
        _ping_message(mock_ctx)

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            await command_callback(MusicBot.ping)(music_bot, mock_ctx)

        mock_ctx.bot.is_owner.assert_not_awaited()
        assert len(mock_ctx.channel.send.await_args.kwargs["embeds"]) == 1
