"""Tests for the /healthz liveness endpoint (docs/K8S_DEPLOYMENT_PLAN.md §3.4)."""

from typing import Sequence

import aiohttp
import pytest

from src.healthz import start_healthz


class StubBot:
    """The three attributes /healthz reads, without a gateway connection."""

    def __init__(
        self,
        ready: bool = False,
        latency: float = float("nan"),
        guilds: Sequence[object] = (),
    ):
        self._ready = ready
        self.latency = latency
        self.guilds = list(guilds)

    def is_ready(self) -> bool:
        return self._ready


async def test_unset_port_is_a_noop(monkeypatch: pytest.MonkeyPatch):
    """No HEALTHZ_PORT → no server: the compose-pipeline no-op guarantee."""
    monkeypatch.delenv("HEALTHZ_PORT", raising=False)
    assert await start_healthz(StubBot()) is None  # type: ignore[arg-type]


async def test_empty_port_is_a_noop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEALTHZ_PORT", "")
    assert await start_healthz(StubBot()) is None  # type: ignore[arg-type]


async def _get_healthz(runner) -> tuple[int, dict]:
    port = runner.addresses[0][1]
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
            return resp.status, await resp.json()


async def test_healthz_before_ready(monkeypatch: pytest.MonkeyPatch):
    """Always 200 even while not ready (liveness, not readiness — §3.4), and
    the pre-heartbeat NaN latency serializes as null, not invalid-JSON NaN."""
    monkeypatch.setenv("HEALTHZ_PORT", "0")  # ephemeral port
    runner = await start_healthz(StubBot())  # type: ignore[arg-type]
    assert runner is not None
    try:
        status, body = await _get_healthz(runner)
        assert status == 200
        assert body == {"ready": False, "latency_s": None, "guilds": 0}
    finally:
        await runner.cleanup()


async def test_healthz_when_ready(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEALTHZ_PORT", "0")
    bot = StubBot(ready=True, latency=0.0625, guilds=[object(), object()])
    runner = await start_healthz(bot)  # type: ignore[arg-type]
    assert runner is not None
    try:
        status, body = await _get_healthz(runner)
        assert status == 200
        assert body == {"ready": True, "latency_s": 0.0625, "guilds": 2}
    finally:
        await runner.cleanup()
