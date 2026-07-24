"""Dependency health probes and version collection for the `-ping` dashboard.

Deliberately Discord-agnostic: this module owns the probe coroutines, the version
collectors, and the ProbeState/ProbeResult value types — but nothing about the
message or the live-edit loop, which are Discord-specific and live in
src/musicbot.py. That split is what lets a future healthz endpoint reuse the
probes verbatim (docs/PING_METADATA_PLAN.md §4/§11).

Each probe is an ``async def`` returning a ProbeResult and never raising out: a
dead dependency becomes a DOWN result, so the caller's loop can always render.
The one exception it lets through is CancelledError (the live-edit loop cancels
still-pending probes at its 3s deadline and flips them to FAILED itself).
"""

import asyncio
import os
import platform
import subprocess
import time
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import discord
import redis.asyncio as aioredis

# Import the value directly: `import yt_dlp` does not reliably pull in the
# `yt_dlp.version` submodule, so `yt_dlp.version.__version__` can AttributeError.
from yt_dlp.version import __version__ as _YTDLP_VERSION

from src import telemetry
from src.spotify import Spotify
from src.util import get_logger

log = get_logger(__name__)

# Live-edit loop tunables (read by src/musicbot.py's ping command). Env-overridable
# for slow/remote deployments. See docs/PING_METADATA_PLAN.md §5.2/§8.
PING_TICK_SECS: float = float(os.environ.get("PING_TICK_SECS", "1.0"))
PING_DEADLINE_SECS: float = float(os.environ.get("PING_DEADLINE_SECS", "3.0"))

_FFMPEG_PROBE_TIMEOUT_SECS = 2.0
_ffmpeg_version_cache: Optional[str] = None
_bot_version_cache: Optional[str] = None
# src/diagnostics.py → src/ → project root, where the Dockerfile copies pyproject.toml.
_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class ProbeState(Enum):
    PENDING = "pending"  # launched, not yet returned      (⏳) — transient
    OK = "ok"  # returned; colour by latency      (🟢/🟡/🟠)
    NA = "n/a"  # dependency not configured         (⚪)
    OFF = "off"  # deliberately disabled              (⚪)
    DOWN = "down"  # errored before the deadline        (🔴)
    FAILED = "failed"  # still pending at the deadline      (🔴)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    label: str
    state: ProbeState
    latency_ms: Optional[float] = None  # set only when state is OK
    detail: Optional[str] = None  # short error class, for the span/logs


async def _timed(label: str, body: Callable[[], Awaitable[object]]) -> ProbeResult:
    """Run a probe body, time it, and classify the outcome.

    Never raises except for CancelledError (which the deadline path relies on to
    flip a cancelled probe to FAILED). Any other failure becomes a DOWN result.
    """
    start = time.perf_counter()
    try:
        await body()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — a probe must never raise out
        log.warning(f"{label} probe failed: {type(e).__name__}: {e}")
        return ProbeResult(label, ProbeState.DOWN, detail=type(e).__name__)
    ms = (time.perf_counter() - start) * 1000
    return ProbeResult(label, ProbeState.OK, latency_ms=ms)


# ── Probes ─────────────────────────────────────────────────────────────────────


async def probe_redis(redis: Optional[aioredis.Redis]) -> ProbeResult:
    if redis is None:
        return ProbeResult("Redis", ProbeState.NA)
    return await _timed("Redis", lambda: redis.ping())


async def probe_spotify(spotify: Spotify) -> ProbeResult:
    if not (spotify.client_id and spotify.client_secret):
        return ProbeResult("Spotify API", ProbeState.NA)

    async def _do() -> None:
        # Reachability without spending quota: a tiny authenticated GET that also
        # exercises the token-refresh path. Confirms auth + data plane.
        await spotify.http_call(
            spotify.spotify_endpoint + "v1/browse/categories", params={"limit": 1}
        )

    return await _timed("Spotify API", _do)


async def probe_postgres(pg_pool: Optional[object]) -> ProbeResult:
    # Typed `object`: asyncpg is not a dependency on main. When the Postgres tier
    # lands, narrow to asyncpg.Pool (docs/PING_METADATA_PLAN.md §11).
    if pg_pool is None:
        return ProbeResult("Postgres", ProbeState.NA)

    async def _do() -> None:
        async with pg_pool.acquire() as conn:  # type: ignore[attr-defined]
            await conn.execute("SELECT 1")

    return await _timed("Postgres", _do)


async def probe_otel() -> ProbeResult:
    if telemetry._tracer_provider is None:
        return ProbeResult("OTEL collector", ProbeState.OFF)
    # urlparse only fills .hostname/.port when a scheme is present. Operators
    # often set OTEL_EXPORTER_OTLP_ENDPOINT scheme-less ("collector:4317"), which
    # would parse to hostname=None and silently probe localhost. Prepend "//" when
    # no scheme is present so we connect to the endpoint they actually configured.
    raw = telemetry._OTLP_ENDPOINT
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    host, port = parsed.hostname or "localhost", parsed.port or 4317

    async def _do() -> None:
        # gRPC OTLP has no cheap app-level ping; a TCP connect proves the port is
        # accepting connections. Liveness signal only (not a real OTLP handshake),
        # and — unlike the auto-instrumented Redis/aiohttp probes — it emits no
        # child span, so this row won't appear in the ping's fan-out trace.
        _, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()

    return await _timed("OTEL collector", _do)


# ── Versions ─────────────────────────────────────────────────────────────────


def bot_version() -> str:
    """The bot's own version, cached for process lifetime.

    Read from pyproject.toml (copied into the runtime image), not installed dist
    metadata: the container installs deps with `poetry install --no-root`, so the
    project itself is never a metadata-bearing distribution and
    importlib.metadata.version() would raise PackageNotFoundError. Falls back to
    dist metadata for a wheel install that ships no pyproject.toml, then "unknown".
    """
    global _bot_version_cache
    if _bot_version_cache is not None:
        return _bot_version_cache
    try:
        with _PYPROJECT.open("rb") as f:
            version: str = tomllib.load(f)["tool"]["poetry"]["version"]
        _bot_version_cache = version
        return version
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        log.warning(f"bot version read from pyproject failed: {type(e).__name__}: {e}")
    try:
        _bot_version_cache = metadata.version("discord-music-bot")
    except metadata.PackageNotFoundError:
        _bot_version_cache = "unknown"
    return _bot_version_cache


def ytdlp_version() -> str:
    return _YTDLP_VERSION


def ffmpeg_version() -> str:
    """`ffmpeg -version` first line → the bare version token. Cached for process
    lifetime (the binary can't change under a running container)."""
    global _ffmpeg_version_cache
    if _ffmpeg_version_cache is not None:
        return _ffmpeg_version_cache
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=_FFMPEG_PROBE_TIMEOUT_SECS,
            check=True,
        ).stdout
        _ffmpeg_version_cache = (
            out.split()[2] if out.startswith("ffmpeg version") else "unknown"
        )
    except Exception as e:  # noqa: BLE001 — a missing/broken ffmpeg must not break -ping
        log.warning(f"ffmpeg version probe failed: {type(e).__name__}: {e}")
        _ffmpeg_version_cache = "unknown"
    return _ffmpeg_version_cache


async def collect_versions() -> dict[str, str]:
    """All versions for the embed's Versions block. ffmpeg's first (uncached) call
    shells out, so it runs in the default executor to keep the loop unblocked;
    every other value is a dict lookup. The single await here also gives the
    already-scheduled immediate probes (NA/OFF) a chance to complete so the
    skeleton send can pre-drain them (docs/PING_METADATA_PLAN.md §6 step 2)."""
    loop = asyncio.get_running_loop()
    ffmpeg = await loop.run_in_executor(None, ffmpeg_version)
    return {
        "bot": bot_version(),
        "yt-dlp": ytdlp_version(),
        "ffmpeg": ffmpeg,
        "python": platform.python_version(),
        "discord.py": discord.__version__,
    }
