"""The `-ping` health dashboard: dependency probes and rendering.

The sequencing — send a skeleton, edit as probes land — is `src/dashboard.py`,
shared with `-debug`; this module supplies the probes, the rows and the callbacks
the driver calls. The probes are not shared with a healthz endpoint: healthz must
stay a dumb liveness probe, or a Redis blip becomes a pod restart loop.
"""

import asyncio
import math
import platform
import subprocess
import time
import tomllib
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import urlparse

import discord
import redis.asyncio as aioredis
from discord.ext import commands
from opentelemetry import trace
from opentelemetry.trace import Span

# `import yt_dlp` does not reliably pull in the `yt_dlp.version` submodule.
from yt_dlp.version import __version__ as _YTDLP_VERSION

from src import telemetry
from src.dashboard import run_live_dashboard
from src import config
from src.config import (
    DEFAULT_POSTGRES_PASSWORD,
    PING_DEADLINE_SECS,
    PING_TICK_SECS,
    SpotifyStatus,
    history_archive_enabled,
    using_default_postgres_password,
)
from src.spotify import Spotify
from src.util import (
    FOOTER_LIMIT,
    get_logger,
    join_footer,
    send_embed,
    trace_footer,
    truncate,
)

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · PROBES & VERSIONS — infrastructure, no Discord message concepts
# ════════════════════════════════════════════════════════════════════════════
# Each probe returns a ProbeResult and never raises out (a dead dependency becomes
# DOWN), except CancelledError: the deadline path cancels stragglers to FAILED.

# Throwaway key the Redis probe writes to prove the write path; self-expiring.
_REDIS_HEALTH_KEY = "health:ping"
_REDIS_HEALTH_TTL_SECS = 30

_FFMPEG_PROBE_TIMEOUT_SECS = 2.0
_ffmpeg_version_cache: Optional[str] = None
_bot_version_cache: Optional[str] = None
# src/ping.py → src/ → project root, where the Dockerfile copies pyproject.toml.
_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class ProbeState(Enum):
    PENDING = "pending"  # launched, not yet returned      (⏳) — transient
    OK = "ok"  # returned; colour by latency      (🟢/🟡/🟠)
    NA = "n/a"  # dependency not configured         (⚪)
    OFF = "off"  # deliberately disabled              (⚪)
    DOWN = "down"  # errored before the deadline        (🔴)
    FAILED = "failed"  # still pending at the deadline      (🔴)


class ArchiveHealth(Protocol):
    """What the Postgres row needs from the archive, declared structurally so this
    module stays out of asyncpg's import graph. Not the raw asyncpg.Pool: the pool
    is created lazily, so a just-started bot has none."""

    async def health_check(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProbeResult:
    label: str
    state: ProbeState
    latency_ms: Optional[float] = None  # set only when state is OK
    detail: Optional[str] = None  # short failure reason; rendered next to "down"


def _error_detail(e: Exception) -> str:
    """A short reason for a failed probe: a leading uppercase Redis code (MISCONF,
    OOM, READONLY) when there is one, else the exception class name."""
    head = str(e).split(maxsplit=1)[0] if str(e) else ""
    if head.isalpha() and head.isupper() and 2 < len(head) <= 12:
        return head
    return type(e).__name__


async def _timed(label: str, body: Callable[[], Awaitable[object]]) -> ProbeResult:
    """Run a probe body, time it, classify the outcome. Every failure becomes DOWN
    except CancelledError, which the deadline path uses to flip a probe to FAILED."""
    start = time.perf_counter()
    try:
        await body()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 — a probe must never raise out
        log.warning(f"{label} probe failed: {type(e).__name__}: {e}")
        return ProbeResult(label, ProbeState.DOWN, detail=_error_detail(e))
    ms = (time.perf_counter() - start) * 1000
    return ProbeResult(label, ProbeState.OK, latency_ms=ms)


# ── Probes ─────────────────────────────────────────────────────────────────────


async def probe_redis(redis: Optional[aioredis.Redis]) -> ProbeResult:
    """PING plus a throwaway write: Redis keeps serving reads while refusing writes
    (MISCONF after a failed bgsave, OOM, READONLY on a replica)."""
    if redis is None:
        return ProbeResult("Redis", ProbeState.NA)

    async def _do() -> None:
        await redis.ping()
        await redis.set(_REDIS_HEALTH_KEY, b"1", ex=_REDIS_HEALTH_TTL_SECS)

    return await _timed("Redis", _do)


async def probe_spotify(
    spotify: Optional[Spotify], status: SpotifyStatus = SpotifyStatus.ENABLED
) -> ProbeResult:
    """The source's usability, not just reachability: `status` (the startup probe's
    verdict) separates "configured but rejected" from "reachable but slow" without
    spending a doomed API call."""
    if (
        spotify is None
        or status is SpotifyStatus.DISABLED
        or not (spotify.client_id and spotify.client_secret)
    ):
        return ProbeResult("Spotify API", ProbeState.NA, detail="not configured")
    if status is SpotifyStatus.INVALID:
        return ProbeResult(
            "Spotify API", ProbeState.DOWN, detail="credentials rejected"
        )

    async def _do() -> None:
        # A tiny authenticated GET that also exercises the token-refresh path.
        await spotify.http_call(
            spotify.spotify_endpoint + "v1/browse/categories", params={"limit": 1}
        )

    return await _timed("Spotify API", _do)


async def probe_postgres(archive: Optional[ArchiveHealth]) -> ProbeResult:
    """A None archive splits on the flag: disabled means setup_hook built no archive
    (OFF, a choice); None with the flag on is a bot built without one some other
    way — tests, or a cog outside MusicBotApp — and stays NA."""
    if archive is None:
        if not history_archive_enabled():
            return ProbeResult("Postgres", ProbeState.OFF, detail="archive disabled")
        return ProbeResult("Postgres", ProbeState.NA)

    return await _timed("Postgres", archive.health_check)


async def probe_otel() -> ProbeResult:
    if telemetry._tracer_provider is None:
        return ProbeResult("OTEL collector", ProbeState.OFF)
    # urlparse fills .hostname/.port only with a scheme present, and operators set
    # the endpoint scheme-less ("collector:4317"); prepend "//" so it parses.
    raw = telemetry._OTLP_ENDPOINT
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    # .port is lazy: urlparse accepts "host:99999" and raises on dereference, so a
    # typo in .env must cost this row rather than the whole board.
    try:
        host, port = parsed.hostname or "localhost", parsed.port or 4317
    except ValueError:
        return ProbeResult("OTEL collector", ProbeState.FAILED)

    async def _do() -> None:
        # gRPC OTLP has no cheap app-level ping; a TCP connect proves the port
        # accepts connections and emits no child span.
        _, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()

    return await _timed("OTEL collector", _do)


# ── Versions ─────────────────────────────────────────────────────────────────


def bot_version() -> str:
    """The bot's version from pyproject.toml, cached for process lifetime. The
    container installs with `poetry install --no-root`, so dist metadata is only
    a fallback (a wheel install), then "unknown"."""
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
    """`ffmpeg -version`'s first line → the bare version token, cached for process
    lifetime."""
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
    """All versions for the Versions block. ffmpeg's first call shells out, so it
    goes to the default executor; that await also lets the immediate NA/OFF probes
    complete before the skeleton send."""
    loop = asyncio.get_running_loop()
    ffmpeg = await loop.run_in_executor(None, ffmpeg_version)
    return {
        "bot": bot_version(),
        "yt-dlp": ytdlp_version(),
        "ffmpeg": ffmpeg,
        "python": platform.python_version(),
        "discord.py": discord.__version__,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 · RENDERING — ProbeResults to a Discord embed
# ════════════════════════════════════════════════════════════════════════════
# Pure presentation: dots, colours and layout. No I/O.

# One band table drives both the status dot and the embed accent. These bands
# (≤100/≤200) differ from latency_color's (≤50/≤100/≤200) below — don't swap them.
_LATENCY_BANDS: tuple[tuple[float, str, int], ...] = (
    (100, "🟢", 0x44FF44),
    (200, "🟡", 0xFFD000),
    (float("inf"), "🟠", 0xFF6600),
)
_STATE_DOT = {
    ProbeState.PENDING: "⏳",
    ProbeState.NA: "⚪",
    ProbeState.OFF: "⚪",
    ProbeState.DOWN: "🔴",
    ProbeState.FAILED: "🔴",
}
_PING_RED = 0x990000
_PING_PROBING = 0x5865F2  # blurple: at least one row still pending
# Amber: the bot works, but the deployment needs attention. Distinct from
# _PING_RED so a standing advisory never reads as an outage.
_PING_WARN = 0xE67E22


def _latency_band(ms: float) -> tuple[str, int]:
    # nan (discord.py's latency while the gateway ws is down) fails every `<=`,
    # so it falls to the worst band rather than StopIteration.
    return next(
        ((dot, hue) for cap, dot, hue in _LATENCY_BANDS if ms <= cap),
        _LATENCY_BANDS[-1][1:],
    )


def _ping_value(r: ProbeResult) -> str:
    if r.state is ProbeState.OK:
        return f"{round(r.latency_ms or 0)} ms"
    word = {
        ProbeState.PENDING: "pending…",
        ProbeState.NA: "n/a",
        ProbeState.OFF: "off",
        ProbeState.DOWN: "down",
        ProbeState.FAILED: "failed",
    }[r.state]
    # The reason (MISCONF, OOM, "archive disabled") is the actionable half.
    if r.state in (ProbeState.DOWN, ProbeState.NA, ProbeState.OFF) and r.detail:
        return f"{word} ({r.detail})"
    return word


def _ping_line(r: ProbeResult) -> str:
    dot = (
        _latency_band(r.latency_ms or 0)[0]
        if r.state is ProbeState.OK
        else _STATE_DOT[r.state]
    )
    return f"{dot} {r.label:<16}{_ping_value(r)}"


def render_ping_embed(
    results: dict[str, ProbeResult],
    versions: dict[str, str],
    discord_ms: float,
    span: Span,
    *,
    debug_suffix: Optional[str] = None,
) -> discord.Embed:
    """The live health embed. Accent: any down/failed → red; else any pending →
    blurple; else the worst OK latency's band. `debug_suffix` is debug mode's
    footer, pre-rendered by the cog and constant for the invocation."""
    # nan latency means the gateway ws is reconnecting.
    disc = (
        ProbeResult("Discord gateway", ProbeState.DOWN, detail="reconnecting")
        if math.isnan(discord_ms)
        else ProbeResult("Discord gateway", ProbeState.OK, latency_ms=discord_ms)
    )
    rows = [disc, *results.values()]
    lat_lines = [_ping_line(r) for r in rows]

    if any(r.state in (ProbeState.DOWN, ProbeState.FAILED) for r in rows):
        color = _PING_RED
    elif any(r.state is ProbeState.PENDING for r in rows):
        color = _PING_PROBING
    else:
        worst = max((r.latency_ms or 0) for r in rows)
        color = _latency_band(worst)[1]

    ver_lines = [
        f"Bot        {versions['bot']}",
        f"yt-dlp     {versions['yt-dlp']}",
        f"ffmpeg     {versions['ffmpeg']}",
        f"Python     {versions['python']}  ·  discord.py  {versions['discord.py']}",
    ]
    embed = discord.Embed(title="🏓 Pong — service health", color=discord.Color(color))
    embed.add_field(
        name="Latency", value="```\n" + "\n".join(lat_lines) + "\n```", inline=False
    )
    embed.add_field(
        name="Versions", value="```\n" + "\n".join(ver_lines) + "\n```", inline=False
    )
    footer = f"environment: {config.ENVIRONMENT}"
    if any(r.state is ProbeState.PENDING for r in rows):
        footer += " · probing…"
    if (tf := trace_footer(span)) is not None:
        footer += f" · {tf}"
    embed.set_footer(text=join_footer(footer, debug_suffix or ""))
    return embed


def default_password_embed(
    *, debug_suffix: Optional[str] = None
) -> Optional[discord.Embed]:
    """A standing warning that the Postgres password is still the compose default,
    or None. Rendered on every -ping so it stays visible until fixed; gated on the
    archive flag, since with the archive off there is no deployed Postgres. It
    reads the DSN, not the server, which is why the remedy is ordered."""
    if not history_archive_enabled():
        return None
    if not using_default_postgres_password():
        return None
    embed = discord.Embed(
        title="⚠️ Default database password in use",
        description=(
            f"`POSTGRES_PASSWORD` is still `{DEFAULT_POSTGRES_PASSWORD}`, "
            "the fallback compose uses so the stack starts with nothing "
            "configured but a Discord token. Anything that can reach this host's "
            "published Postgres port can read every play this bot has recorded."
        ),
        color=discord.Color(_PING_WARN),
    )
    embed.add_field(
        name="Changing it — in this order",
        value=(
            "1. **The server first:** `docker compose exec postgres psql -U "
            "<user> -c \"ALTER USER <user> PASSWORD '<new>'\"`\n"
            "2. Then `.env`: `./setup_env.sh --force`, and set the same value\n"
            "3. Then `docker compose up -d` — the bot's DSN is baked at "
            "container-create time, so a restart alone keeps the old one\n"
            "To start clean instead: `docker compose down && docker volume rm "
            "discord-music-bot_postgres-data` — **not** `down -v`, which also "
            "drops the Redis volume holding plays not yet in Postgres.\n\n"
            "**The order is the point.** This warning reads the bot's DSN, not "
            "the server, so doing step 2 first makes it disappear while the "
            "database still accepts the old password — the one window where you "
            "are exposed and nothing says so.\n"
            "Editing `.env` alone never changes the server: Postgres reads the "
            "variable only when initializing an empty data directory, so an "
            "existing volume keeps its original password and the bot is simply "
            "locked out of its own database."
        ),
        inline=False,
    )
    if debug_suffix:
        embed.set_footer(text=truncate(debug_suffix, FOOTER_LIMIT))
    return embed


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 · COMMAND BODIES — what src/musicbot.py's cog delegates to
# ════════════════════════════════════════════════════════════════════════════


def latency_color(ms: float) -> discord.Color:
    """The accent for send_latency_line. Four bands, not _LATENCY_BANDS' three."""
    if ms <= 50:
        return discord.Color(0x44FF44)
    if ms <= 100:
        return discord.Color(0xFFD000)
    if ms <= 200:
        return discord.Color(0xFF6600)
    return discord.Color(0x990000)


async def send_latency_line(ctx: commands.Context, bot_latency: float) -> None:
    """The one-line WS-latency reply -join uses, so the common connect path never
    pays for the full dashboard. Through ctx.send, so it honours the NP host."""
    ms = bot_latency * 1000
    await send_embed(
        ctx,
        "Ping - latency in ms",
        f"Ping: **{round(ms)}** milliseconds!",
        latency_color(ms),
    )


async def run_health_dashboard(
    ctx: commands.Context,
    *,
    bot_latency: float,
    redis: Optional[aioredis.Redis],
    spotify: Optional[Spotify],
    spotify_status: SpotifyStatus = SpotifyStatus.ENABLED,
    # No default: a default of None would let a new caller silently render an
    # enabled archive's Postgres row as "n/a" by forgetting the argument.
    archive: Optional[ArchiveHealth],
    # Defaulted: None is a real state (debug mode off), and forgetting it costs a
    # footer, not a wrong health answer.
    debug_suffix: Optional[str] = None,
) -> None:
    """Send a skeleton embed immediately, edit it as probes return, fail any
    straggler at the deadline. Runs inside the caller's span and lets exceptions
    propagate — the command owns the user-facing error reply. `debug_suffix` is
    constant for the loop: the driver edits only when the render differs, so a
    suffix carrying elapsed-ms would edit the board every tick."""
    span = trace.get_current_span()
    discord_ms = bot_latency * 1000
    probes: dict[str, Callable[[], Coroutine[Any, Any, ProbeResult]]] = {
        "Redis": lambda: probe_redis(redis),
        "Spotify API": lambda: probe_spotify(spotify, spotify_status),
        "Postgres": lambda: probe_postgres(archive),
        "OTEL collector": lambda: probe_otel(),
    }
    results = {label: ProbeResult(label, ProbeState.PENDING) for label in probes}
    versions: dict[str, str] = {}
    warning: Optional[discord.Embed] = None

    async def _prepare() -> None:
        """The driver's pre-send step. collect_versions()'s executor hop also lets
        the immediate NA/OFF probes complete, so their rows never flash pending."""
        nonlocal versions, warning
        versions = await collect_versions()
        # Owner only: -ping has no permission gate, and this advisory confirms to
        # every member which hosts run on the default credential. The cheap local
        # check runs first — is_owner() is a REST GET (application_info, retried
        # up to ~25s on a 5xx) and would otherwise precede the skeleton send. It
        # RAISES on a 5xx, and _prepare runs before the send with exceptions
        # propagating, so an unguarded call would cost the whole board.
        warning = default_password_embed(debug_suffix=debug_suffix)
        if warning is not None:
            try:
                is_owner = await ctx.bot.is_owner(ctx.author)
            except Exception as e:  # noqa: BLE001 — an unknown owner is not an owner
                log.warning(f"ping owner check failed: {type(e).__name__}: {e}")
                is_owner = False
            if not is_owner:
                warning = None

    def _render() -> list[discord.Embed]:
        embed = render_ping_embed(
            results, versions, discord_ms, span, debug_suffix=debug_suffix
        )
        return [e for e in (warning, embed) if e is not None]

    def _settle(label: str, outcome: ProbeResult | Exception) -> None:
        # _timed guards the probe BODY only (probe_otel parses a URL before it);
        # the driver hands the failure over resolved so it cannot take the board
        # down before it is sent.
        if isinstance(outcome, Exception):
            log.warning(
                "ping probe raised outside its guard", probe=label, error=str(outcome)
            )
            results[label] = ProbeResult(label, ProbeState.FAILED)
        else:
            results[label] = outcome

    def _abandon(label: str) -> None:
        results[label] = ProbeResult(label, ProbeState.FAILED)

    await run_live_dashboard(
        ctx,
        probes=probes,
        settle=_settle,
        abandon=_abandon,
        render=_render,
        prepare=_prepare,
        tick_secs=PING_TICK_SECS,
        deadline_secs=PING_DEADLINE_SECS,
    )

    for r in results.values():  # self-documenting trace
        # PENDING survives only when the driver returned early (message deleted
        # mid-loop); recorded, it would read as a probe that never answered.
        if r.state is ProbeState.PENDING:
            continue
        span.set_attribute(f"ping.{r.label}.state", r.state.name.lower())
        if r.latency_ms is not None:
            span.set_attribute(f"ping.{r.label}.latency_ms", round(r.latency_ms, 2))
