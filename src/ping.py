"""The `-ping` health dashboard: dependency probes, rendering, and the live-edit loop.

One feature, one module. The three sections below are separated by rules rather
than by files: an earlier split put the probes in src/diagnostics.py so a healthz
endpoint could reuse them, but src/healthz.py is deliberately a dumb liveness probe
(it must NOT fail on dependency health, or a Redis blip becomes a pod restart loop),
so that second consumer never existed. Full design: docs/PING_METADATA_PLAN.md.

src/musicbot.py holds only the command registration and delegates in here.
"""

import asyncio
import math
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
from typing import Optional, Protocol
from urllib.parse import urlparse

import discord
import redis.asyncio as aioredis
from discord.ext import commands
from opentelemetry import trace
from opentelemetry.trace import Span

# Import the value directly: `import yt_dlp` does not reliably pull in the
# `yt_dlp.version` submodule, so `yt_dlp.version.__version__` can AttributeError.
from yt_dlp.version import __version__ as _YTDLP_VERSION

from src import telemetry
from src.config import ENVIRONMENT, SpotifyStatus
from src.spotify import Spotify
from src.util import get_logger, latency_color, send_embed, trace_footer

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · PROBES & VERSIONS — infrastructure, no Discord message concepts
# ════════════════════════════════════════════════════════════════════════════
# Nothing below this rule until SECTION 2 knows about embeds, channels or ctx. Each
# probe is an ``async def`` returning a ProbeResult and never raising out: a dead
# dependency becomes DOWN, so the caller's loop can always render. The one exception
# it lets through is CancelledError — the live-edit loop cancels still-pending probes
# at its deadline and flips them to FAILED itself.

# Live-edit loop tunables (see run_health_dashboard). Env-overridable
# for slow/remote deployments. See docs/PING_METADATA_PLAN.md §5.2/§8.
PING_TICK_SECS: float = float(os.environ.get("PING_TICK_SECS", "1.0"))
PING_DEADLINE_SECS: float = float(os.environ.get("PING_DEADLINE_SECS", "3.0"))

# Throwaway key the Redis probe writes to prove the write path is open (see
# probe_redis). Namespaced away from guild:* / spotify:* and self-expiring, so it
# never accumulates and can't collide with real state.
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
    """The only thing the Postgres row needs from the play-history archive: a
    way to prove its database answers.

    Declared structurally here rather than imported from history_archive.py so
    this module keeps its "no dependency on the tiers it reports on" shape —
    probe_redis takes a bare Redis handle for the same reason, and ping.py stays
    out of asyncpg's import graph. PostgresHistoryArchive satisfies it by
    having the method; nothing has to inherit anything.

    It is deliberately NOT the raw asyncpg.Pool this parameter used to be typed
    for. The pool is created lazily on first archive use, so a just-started bot
    has none, and a pool-shaped probe would have to report the required
    Postgres tier as "not configured" — or force the caller to open a
    connection before the dashboard's skeleton embed could send. Asking the
    archive instead moves both problems inside the probe task, where the
    timing already belongs.
    """

    async def health_check(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProbeResult:
    label: str
    state: ProbeState
    latency_ms: Optional[float] = None  # set only when state is OK
    detail: Optional[str] = None  # short failure reason; rendered next to "down"


def _error_detail(e: Exception) -> str:
    """A short, renderable reason for a failed probe.

    Server-side Redis errors lead with an uppercase code — MISCONF (persistence
    broken), OOM (maxmemory + noeviction), READONLY (replica) — which says far
    more to an operator than the redis-py exception class ("ResponseError").
    Falls back to the exception class name for everything else.
    """
    head = str(e).split(maxsplit=1)[0] if str(e) else ""
    if head.isalpha() and head.isupper() and 2 < len(head) <= 12:
        return head
    return type(e).__name__


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
        return ProbeResult(label, ProbeState.DOWN, detail=_error_detail(e))
    ms = (time.perf_counter() - start) * 1000
    return ProbeResult(label, ProbeState.OK, latency_ms=ms)


# ── Probes ─────────────────────────────────────────────────────────────────────


async def probe_redis(redis: Optional[aioredis.Redis]) -> ProbeResult:
    """PING *and* a throwaway write.

    PING alone is misleading: Redis keeps serving reads while refusing writes —
    MISCONF after a failed bgsave (observed live: full disk → every guild's state
    write failing while -ping still showed green), OOM under maxmemory+noeviction,
    READONLY on a replica. The bot writes guild state, queues and history
    constantly, so a probe that never writes isn't measuring what the bot needs.
    One short-TTL key, self-expiring, is enough to exercise the write path.
    """
    if redis is None:
        return ProbeResult("Redis", ProbeState.NA)

    async def _do() -> None:
        await redis.ping()
        await redis.set(_REDIS_HEALTH_KEY, b"1", ex=_REDIS_HEALTH_TTL_SECS)

    return await _timed("Redis", _do)


async def probe_spotify(
    spotify: Optional[Spotify], status: SpotifyStatus = SpotifyStatus.ENABLED
) -> ProbeResult:
    """Spotify's row, which reports the *source's* usability, not just reachability.

    `status` is the outcome of the startup credential probe (MusicBot._spotify_status):
    it is the only thing that can tell "configured but Spotify rejected the
    credentials" apart from "reachable but slow", and it does so without spending a
    doomed API call. It defaults to ENABLED so a caller that has no status to offer
    still gets the plain reachability probe.
    """
    # spotify is None when the bot was started without Spotify credentials (the
    # feature is off entirely); a non-None client with empty creds is the same story
    # from a probe's point of view. Both are "not configured" → N/A, not a failure.
    if (
        spotify is None
        or status is SpotifyStatus.DISABLED
        or not (spotify.client_id and spotify.client_secret)
    ):
        return ProbeResult("Spotify API", ProbeState.NA, detail="not configured")
    if status is SpotifyStatus.INVALID:
        # Credentials were present at startup and Spotify rejected them. Probing
        # would only re-earn the same 401 a second time, so report the known cause.
        return ProbeResult(
            "Spotify API", ProbeState.DOWN, detail="credentials rejected"
        )

    async def _do() -> None:
        # Reachability without spending quota: a tiny authenticated GET that also
        # exercises the token-refresh path. Confirms auth + data plane.
        await spotify.http_call(
            spotify.spotify_endpoint + "v1/browse/categories", params={"limit": 1}
        )

    return await _timed("Spotify API", _do)


async def probe_postgres(archive: Optional[ArchiveHealth]) -> ProbeResult:
    """The play-history archive's Postgres row.

    None means the bot was built without an archive — only reachable in tests
    and in a cog constructed outside MusicBotApp, since the tier is required
    (MusicBotApp.setup_hook refuses to start without POSTGRES_URL).
    """
    if archive is None:
        return ProbeResult("Postgres", ProbeState.NA)

    return await _timed("Postgres", archive.health_check)


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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 · RENDERING — ProbeResults to a Discord embed
# ════════════════════════════════════════════════════════════════════════════
# docs/PING_METADATA_PLAN.md §6.2. Pure presentation: takes the values SECTION 1
# produces and decides dots, colours and layout. No I/O.

# One band table drives BOTH the status dot and the embed accent so a probe can't
# show, say, a green dot under a yellow accent. These bands (≤100/≤200) differ on
# purpose from util.latency_color's (≤50/≤100/≤200), which backs the lighter
# send_latency_line reply — don't swap one for the other.
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


def _latency_band(ms: float) -> tuple[str, int]:
    # Total over every float, INCLUDING nan (nan <= cap is always False): fall back
    # to the worst band so an unknown latency can never StopIteration. discord.py's
    # Client.latency is nan whenever the gateway ws is down (reconnect window), which
    # is exactly when -ping is most likely to be run.
    return next(
        ((dot, hue) for cap, dot, hue in _LATENCY_BANDS if ms <= cap),
        _LATENCY_BANDS[-1][1:],
    )


def _ping_dot(r: ProbeResult) -> str:
    if r.state is not ProbeState.OK:
        return _STATE_DOT[r.state]
    return _latency_band(r.latency_ms or 0)[0]


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
    # A bare "down" makes an operator go read logs; the reason (MISCONF, OOM,
    # ConnectionError) is the actionable half and costs one short parenthetical.
    # An n/a row gets the same treatment when it has a reason to give — "not
    # configured" is why an optional dependency is dark, and it costs nothing.
    if r.state in (ProbeState.DOWN, ProbeState.NA) and r.detail:
        return f"{word} ({r.detail})"
    return word


def _ping_line(r: ProbeResult) -> str:
    return f"{_ping_dot(r)} {r.label:<16}{_ping_value(r)}"


def render_ping_embed(
    results: dict[str, ProbeResult],
    versions: dict[str, str],
    discord_ms: float,
    span: Span,
) -> discord.Embed:
    """Build the live health embed from the current probe results + versions.

    Accent: any down/failed → red; else any still-pending → blurple; else the
    worst OK latency's band colour (same bands as the dots)."""
    # discord.py reports nan latency while the gateway ws is reconnecting — show it
    # as down (red) rather than a bogus number, matching the old latency_color(nan).
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
    footer = f"environment: {ENVIRONMENT}"
    if any(r.state is ProbeState.PENDING for r in rows):
        footer += " · probing…"
    if (tf := trace_footer(span)) is not None:
        footer += f" · {tf}"
    embed.set_footer(text=footer)
    return embed


def _ping_embed_changed(new: discord.Embed, old: discord.Embed) -> bool:
    """True when a re-render actually differs — the only things that move are the
    two field values, the colour, and the footer, all captured by to_dict()."""
    return new.to_dict() != old.to_dict()


async def _safe_edit(
    message: discord.Message, embed: discord.Embed
) -> Optional[discord.Embed]:
    """Edit a message, tolerating a host the user deleted mid-loop (mirrors
    musicplayer._progress_updater). Returns the embed on success, None if gone."""
    try:
        await message.edit(embed=embed)
        return embed
    except discord.NotFound:
        return None


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 · COMMAND BODIES — what src/musicbot.py's cog delegates to
# ════════════════════════════════════════════════════════════════════════════


async def send_latency_line(ctx: commands.Context, bot_latency: float) -> None:
    """The lightweight one-line WS-latency reply. Used by -join (and cold -play,
    via join) so the common connect path never pays for the full health dashboard
    — see docs/PING_METADATA_PLAN.md §2.3. Sent through ctx.send so it still
    honours the Now Playing host machinery."""
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
    # No default, unlike spotify_status: `redis` and `spotify` are also
    # required-but-Optional, and for the same reason. A default of None would let
    # a new caller silently render the required Postgres tier as "n/a" by simply
    # forgetting the argument — which is exactly the bug this parameter had while
    # it was a pool nobody passed.
    archive: Optional[ArchiveHealth],
) -> None:
    """Optimistic-send + live-edit health dashboard (docs/PING_METADATA_PLAN.md §5).

    Fires a skeleton embed immediately, then edits it in place each tick as probes
    return; a hard deadline fails any straggler. Runs inside the caller's span.
    Exceptions propagate — the command owns the user-facing error reply.
    """
    span = trace.get_current_span()
    loop = asyncio.get_running_loop()
    tasks: dict[str, asyncio.Task[ProbeResult]] = {}

    def _drain() -> bool:
        """Fold every finished probe task into `results`; True if a row moved.
        Only genuinely-completed tasks are done() here (cancellation happens at
        the deadline/finally), so .result() never re-raises."""
        changed = False
        for label in [lbl for lbl in pending if tasks[lbl].done()]:
            results[label] = tasks[label].result()
            pending.discard(label)
            changed = True
        return changed

    try:
        # 1. launch probes INSIDE try so `finally` cancels them no matter where
        #    a later await raises. create_task copies the otel context, so child
        #    probe spans (auto-instrumented Redis/aiohttp) nest under bot.ping.
        tasks = {
            "Redis": asyncio.create_task(probe_redis(redis)),
            "Spotify API": asyncio.create_task(probe_spotify(spotify, spotify_status)),
            "Postgres": asyncio.create_task(probe_postgres(archive)),
            "OTEL collector": asyncio.create_task(probe_otel()),
        }
        results = {label: ProbeResult(label, ProbeState.PENDING) for label in tasks}
        pending = set(tasks)

        # 2. instant data + skeleton. collect_versions() awaits (executor hop),
        #    which lets the immediate NA/OFF probes complete; pre-drain them so
        #    those rows never flash "pending…" for a tick.
        versions = await collect_versions()
        _drain()
        discord_ms = bot_latency * 1000
        last = render_ping_embed(results, versions, discord_ms, span)
        message = await ctx.channel.send(embed=last)  # bypass NP host (§5.3)

        # 3. live-edit loop: tick, drain, edit-on-change; exit early when done.
        deadline = loop.time() + PING_DEADLINE_SECS
        while pending and (remaining := deadline - loop.time()) > 0:
            await asyncio.sleep(min(PING_TICK_SECS, remaining))
            if _drain():
                embed = render_ping_embed(results, versions, discord_ms, span)
                if _ping_embed_changed(embed, last):
                    last = await _safe_edit(message, embed) or last

        # 4. deadline: fail only what is STILL pending. Re-check done() first —
        #    a probe can finish during step 3's final edit await, and flipping
        #    it to FAILED unconditionally would report a healthy dep as red.
        if pending:
            for label in pending:
                if tasks[label].done():
                    results[label] = tasks[label].result()
                else:
                    tasks[label].cancel()
                    results[label] = ProbeResult(label, ProbeState.FAILED)
            await _safe_edit(
                message, render_ping_embed(results, versions, discord_ms, span)
            )

        for r in results.values():  # self-documenting trace
            span.set_attribute(f"ping.{r.label}.state", r.state.name.lower())
            if r.latency_ms is not None:
                span.set_attribute(f"ping.{r.label}.latency_ms", round(r.latency_ms, 2))
    finally:
        for t in tasks.values():  # never leak a probe task
            if not t.done():
                t.cancel()
