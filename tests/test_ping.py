"""Tests for src/ping.py — the -ping health dashboard: rendering and the
optimistic-send + live-edit message loop."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from src import diagnostics
from src.diagnostics import ProbeResult, ProbeState
from src.musicbot import MusicBot
from tests.helpers import command_callback, mocked


def _probe(state: ProbeState, ms: float | None = None) -> ProbeResult:
    return ProbeResult("x", state, latency_ms=ms)


def _patch_probes(**results: ProbeResult) -> Any:
    """Patch each diagnostics.probe_* to resolve to the given ProbeResult, and
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
        "src.ping.diagnostics",
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
        monkeypatch.setattr(diagnostics, "PING_TICK_SECS", 0.0)
        monkeypatch.setattr(diagnostics, "PING_DEADLINE_SECS", 0.0)

        never = asyncio.Event()  # never set → the probe hangs until cancelled

        async def _hang(*a: Any, **k: Any) -> ProbeResult:
            await never.wait()
            raise AssertionError("unreachable")

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.diagnostics.probe_spotify", new=_hang):
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
        monkeypatch.setattr(diagnostics, "PING_TICK_SECS", 0.01)
        monkeypatch.setattr(diagnostics, "PING_DEADLINE_SECS", 5.0)

        gate = asyncio.Event()

        async def _gated(*a: Any, **k: Any) -> ProbeResult:
            await gate.wait()
            return ProbeResult("Spotify API", ProbeState.OK, latency_ms=42.0)

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.diagnostics.probe_spotify", new=_gated):
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
        monkeypatch.setattr(diagnostics, "PING_TICK_SECS", 0.0)
        monkeypatch.setattr(diagnostics, "PING_DEADLINE_SECS", 0.0)

        never = asyncio.Event()

        async def _hang(*a: Any, **k: Any) -> ProbeResult:
            await never.wait()
            raise AssertionError("unreachable")

        with _patch_probes(redis=_probe(ProbeState.OK, 1.0)):
            with patch("src.ping.diagnostics.probe_spotify", new=_hang):
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
            patch("src.ping.diagnostics.probe_redis", new=redis_spy),
        ):
            await command_callback(MusicBot.join)(music_bot, mock_ctx)

        latency_line.assert_awaited_once()
        redis_spy.assert_not_awaited()  # no health dashboard on the join path
