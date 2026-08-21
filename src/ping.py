"""The `-ping` health dashboard: dependency probes and rendering.

The SEQUENCING is not here. Sending a skeleton immediately and editing it as probes
land is `src/dashboard.py`, shared with `-debug`; this module supplies the probes,
the rows and the four callbacks that driver calls back into. What stays together
here is one feature's probes and its rendering, separated by rules rather than
files. The probes are deliberately not shared with a healthz endpoint: healthz must
stay a dumb liveness probe, or a Redis blip becomes a pod restart loop.
src/musicbot.py holds only the command registration and delegates in here.
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

# Import the value directly: `import yt_dlp` does not reliably pull in the
# `yt_dlp.version` submodule, so `yt_dlp.version.__version__` can AttributeError.
from yt_dlp.version import __version__ as _YTDLP_VERSION

from src import telemetry
from src.dashboard import run_live_dashboard

# Live-edit tunables for the shared driver (src/dashboard.py), env-overridable.
# TICK is a CEILING on how long it waits before re-rendering, not a cadence: the
# loop wakes on the first probe to finish. Defined in config.py so there is one
# answer to "what does this bot read from the environment?".
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
    send_embed,
    trace_footer,
    truncate,
)

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · PROBES & VERSIONS — infrastructure, no Discord message concepts
# ════════════════════════════════════════════════════════════════════════════
# Nothing below this rule until SECTION 2 knows about embeds, channels or ctx. Each
# probe returns a ProbeResult and never raises out (a dead dependency becomes DOWN),
# except CancelledError: the deadline path cancels stragglers and flips them to FAILED.

# Throwaway key the Redis probe writes to prove the write path is open. Namespaced away
# from guild:* / spotify:* and self-expiring, so it never accumulates or collides.
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
    """The only thing the Postgres row needs from the play-history archive: a way to
    prove its database answers. Declared structurally rather than imported from
    history_archive.py so ping.py stays out of asyncpg's import graph. Deliberately not
    the raw asyncpg.Pool: the pool is created lazily on first archive use, so a
    just-started bot has none and a pool-shaped probe would report the tier as
    "not configured".
    """

    async def health_check(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProbeResult:
    label: str
    state: ProbeState
    latency_ms: Optional[float] = None  # set only when state is OK
    detail: Optional[str] = None  # short failure reason; rendered next to "down"


def _error_detail(e: Exception) -> str:
    """A short, renderable reason for a failed probe. Server-side Redis errors lead with
    an uppercase code — MISCONF (persistence broken), OOM, READONLY (replica) — which
    tells an operator far more than the redis-py class name. Falls back to that name.
    """
    head = str(e).split(maxsplit=1)[0] if str(e) else ""
    if head.isalpha() and head.isupper() and 2 < len(head) <= 12:
        return head
    return type(e).__name__


async def _timed(label: str, body: Callable[[], Awaitable[object]]) -> ProbeResult:
    """Run a probe body, time it, classify the outcome. Never raises except
    CancelledError, which the deadline path relies on to flip a cancelled probe to
    FAILED; every other failure becomes DOWN.
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
    """PING *and* a throwaway write. PING alone is misleading: Redis keeps serving reads
    while refusing writes — MISCONF after a failed bgsave (observed live: a full disk
    failed every guild's state write while -ping stayed green), OOM under
    maxmemory+noeviction, READONLY on a replica. One short-TTL key exercises writes.
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
    """Spotify's row: the *source's* usability, not just reachability. `status` (from
    MusicBot._spotify_status' startup probe) separates "configured but rejected" from
    "reachable but slow" without spending a doomed API call; it defaults to enabled so a
    caller with no status still gets the plain reachability probe.
    """
    # None client (started without credentials) and a client with empty creds are
    # the same story here: "not configured" → N/A, not a failure.
    if (
        spotify is None
        or status is SpotifyStatus.DISABLED
        or not (spotify.client_id and spotify.client_secret)
    ):
        return ProbeResult("Spotify API", ProbeState.NA, detail="not configured")
    if status is SpotifyStatus.INVALID:
        # Rejected at startup — probing would only re-earn the same 401.
        return ProbeResult(
            "Spotify API", ProbeState.DOWN, detail="credentials rejected"
        )

    async def _do() -> None:
        # Reachability without spending quota: a tiny authenticated GET that also
        # exercises the token-refresh path, confirming auth + data plane.
        await spotify.http_call(
            spotify.spotify_endpoint + "v1/browse/categories", params={"limit": 1}
        )

    return await _timed("Spotify API", _do)


async def probe_postgres(archive: Optional[ArchiveHealth]) -> ProbeResult:
    """The play-history archive's Postgres row. A None archive splits on the flag:
    disabled (HISTORY_ARCHIVE_ENABLED off, the ship default) means setup_hook
    deliberately built no archive, so the row says OFF and reads as a choice rather than
    a fault. None with the flag ON is a bot built without an archive some other way —
    tests, or a cog constructed outside MusicBotApp — and stays NA.
    """
    if archive is None:
        if not history_archive_enabled():
            return ProbeResult("Postgres", ProbeState.OFF, detail="archive disabled")
        return ProbeResult("Postgres", ProbeState.NA)

    return await _timed("Postgres", archive.health_check)


async def probe_otel() -> ProbeResult:
    if telemetry._tracer_provider is None:
        return ProbeResult("OTEL collector", ProbeState.OFF)
    # urlparse fills .hostname/.port only with a scheme present, and operators often set
    # OTEL_EXPORTER_OTLP_ENDPOINT scheme-less ("collector:4317"), which parses to
    # hostname=None and silently probes localhost. Prepend "//" so we hit the real one.
    raw = telemetry._OTLP_ENDPOINT
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    # .port is a LAZY property: urlparse accepts "host:99999" and "host:4317 " and only
    # raises on dereference. Reading it outside a guard let a stray character in .env
    # raise out of the probe, through _settle, and out of the dashboard driver BEFORE
    # the skeleton send — so a typo cost the whole board rather than one row.
    try:
        host, port = parsed.hostname or "localhost", parsed.port or 4317
    except ValueError:
        return ProbeResult("OTEL collector", ProbeState.FAILED)

    async def _do() -> None:
        # gRPC OTLP has no cheap app-level ping, so a TCP connect proves only that the
        # port accepts connections. It emits no child span, so this row never appears
        # in the ping's fan-out trace.
        _, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()

    return await _timed("OTEL collector", _do)


# ── Versions ─────────────────────────────────────────────────────────────────


def bot_version() -> str:
    """The bot's own version, cached for process lifetime. From pyproject.toml (copied
    into the runtime image), not dist metadata: the container installs with `poetry
    install --no-root`, so metadata.version() would raise PackageNotFoundError. Falls
    back to dist metadata for a wheel install shipping no pyproject.toml, then "unknown".
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
    """All versions for the embed's Versions block. ffmpeg's first call shells out, so it
    goes to the default executor; the rest are dict lookups. This single await also lets
    the already-scheduled immediate probes (NA/OFF) complete for the skeleton send."""
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
# Pure presentation: takes the values SECTION 1 produces and decides dots, colours
# and layout. No I/O.

# One band table drives both the status dot and the embed accent, so a green dot can't
# appear under a yellow accent. These bands (≤100/≤200) differ on purpose from
# latency_color's (≤50/≤100/≤200) below, which backs send_latency_line — don't swap them.
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
# Amber: the bot works, but something about the deployment needs attention.
# Distinct from _PING_RED so a standing advisory never reads as an outage.
_PING_WARN = 0xE67E22


def _latency_band(ms: float) -> tuple[str, int]:
    # Total over every float INCLUDING nan (nan <= cap is always False): fall back to
    # the worst band so an unknown latency can never StopIteration. discord.py's
    # Client.latency is nan while the gateway ws is down — exactly when -ping runs.
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
    # A bare "down" sends an operator to the logs; the reason (MISCONF, OOM,
    # ConnectionError) is the actionable half. NA and OFF get the same treatment when
    # they have one — probe_postgres's "archive disabled" marks an opted-out row.
    if r.state in (ProbeState.DOWN, ProbeState.NA, ProbeState.OFF) and r.detail:
        return f"{word} ({r.detail})"
    return word


def _ping_line(r: ProbeResult) -> str:
    return f"{_ping_dot(r)} {r.label:<16}{_ping_value(r)}"


def render_ping_embed(
    results: dict[str, ProbeResult],
    versions: dict[str, str],
    discord_ms: float,
    span: Span,
    *,
    debug_suffix: Optional[str] = None,
) -> discord.Embed:
    """Build the live health embed from the current probe results + versions. Accent:
    any down/failed → red; else any still-pending → blurple; else the worst OK latency's
    band colour (same bands as the dots).

    `debug_suffix` is debug mode's footer, pre-rendered by the cog and constant for
    the invocation — see run_health_dashboard."""
    # nan latency means the gateway ws is reconnecting — show down (red) rather
    # than a bogus number.
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
    if debug_suffix:
        footer += f" · {debug_suffix}"
    embed.set_footer(text=truncate(footer, FOOTER_LIMIT))
    return embed


def default_password_embed(
    *, debug_suffix: Optional[str] = None
) -> Optional[discord.Embed]:
    """A standing warning that the Postgres password is still the compose default, or
    None when it is not.

    Rendered on every -ping rather than once at startup: this is the surface an operator
    returns to, so it keeps the problem visible until it is fixed. IT reads the DSN, not
    the SERVER, which is why the remedy below is ordered — `setup_env.sh --force` first
    would clear the warning while Postgres still accepts the old password. Gated on the
    archive flag, like setup_hook's startup ERROR: with the archive disabled there is no
    deployed Postgres to warn about, and an advisory for an absent database is noise.
    """
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
    """The accent for send_latency_line's one-line reply. Four bands, deliberately
    NOT _LATENCY_BANDS' three — see the comment there before reconciling them."""
    if ms <= 50:
        return discord.Color(0x44FF44)
    if ms <= 100:
        return discord.Color(0xFFD000)
    if ms <= 200:
        return discord.Color(0xFF6600)
    return discord.Color(0x990000)


async def send_latency_line(ctx: commands.Context, bot_latency: float) -> None:
    """The lightweight one-line WS-latency reply. Used by -join (and cold -play via
    join) so the common connect path never pays for the full dashboard. Sent through
    ctx.send, so it still honours the Now Playing host machinery."""
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
    # No default, unlike spotify_status: `redis` and `spotify` are required-but-Optional
    # for the same reason — a default of None would let a new caller silently render an
    # enabled archive's Postgres row as "n/a" by simply forgetting the argument.
    archive: Optional[ArchiveHealth],
    # Defaulted, unlike `archive`: None here is a real state (debug mode off, what
    # MusicBot._debug_suffix returns) rather than an unknown, and forgetting it
    # costs a footer, not a wrong health answer.
    debug_suffix: Optional[str] = None,
) -> None:
    """Optimistic-send + live-edit health dashboard: sends a skeleton embed immediately,
    then edits in place each tick as probes return, with a hard deadline failing any
    straggler. Runs inside the caller's span and lets exceptions propagate — the command
    owns the user-facing error reply.

    The sequencing lives in src/dashboard.py, shared with -debug; what stays here is
    what a row IS and how the board is drawn.

    `debug_suffix` is debug mode's footer, rendered once by the cog and constant for
    the loop. The driver only edits when the render differs, so a suffix carrying
    elapsed-ms would differ every tick and edit the board until the deadline — which
    is why the cog omits elapsed rather than measuring it.
    """
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
        """Instant data, on the driver's pre-send step. collect_versions()'s executor
        hop is also what lets the immediate NA/OFF probes complete before the skeleton
        renders, so their rows never flash "pending…"."""
        nonlocal versions, warning
        versions = await collect_versions()
        # Computed once: it is static, so it cannot make the driver's change-check see
        # a difference that is not there.
        #
        # OWNER only. -ping has no permission gate, so this advisory — which names the
        # credential and confirms this host runs on it — would otherwise reach every
        # member of every guild and stay in Discord's retained history. The value is a
        # public constant in a GPL repo, so the leak was never the string but the free
        # confirmation of which hosts are worth trying. The startup ERROR still fires on
        # every boot for anyone reading logs.
        #
        # Order matters here. is_owner() is not a local predicate:
        # MusicBotApp passes neither owner_id nor owner_ids, so discord.py falls through
        # to application_info() — a REST GET, retried up to 5 times over ~25s on a 5xx,
        # and it raises rather than returning False. Written as `embed() if await
        # is_owner() else None` the await runs first and unconditionally, putting a
        # network round trip ahead of the skeleton send this function promises is
        # immediate, and PING_DEADLINE_SECS does not bound it (that clock starts after
        # the send). Ask the cheap, local question first.
        #
        # Fails CLOSED. is_owner() RAISES on a 5xx rather than returning False, and
        # _prepare now runs inside the dashboard driver BEFORE the skeleton send with
        # exceptions propagating by design — so an unguarded call turns a transient
        # Discord API failure into "-ping produced no board at all". Suppressing the
        # advisory is the safe direction; the startup ERROR still names the problem.
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

    def _settle(label: str, outcome: "ProbeResult | Exception") -> None:
        # _timed guards the probe BODY, not everything a probe does: probe_otel parses
        # a URL before it. The driver hands the failure over already resolved rather
        # than as a task to unwrap, so this cannot re-raise into the driver's own task
        # and take the board down before it is even sent.
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
        # PENDING is documented as transient, and it survives to here only when the
        # driver returned early because the message was deleted mid-loop — it cancels
        # the stragglers rather than settling them. Recording it would put a state
        # that means "not yet" permanently in a trace, where it reads as a probe that
        # never answered. Skip those rows; the ones that did answer still land.
        if r.state is ProbeState.PENDING:
            continue
        span.set_attribute(f"ping.{r.label}.state", r.state.name.lower())
        if r.latency_ms is not None:
            span.set_attribute(f"ping.{r.label}.latency_ms", round(r.latency_ms, 2))
