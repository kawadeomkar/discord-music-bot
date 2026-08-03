"""The `-ping` health dashboard: dependency probes, rendering, live-edit loop.

One feature, one module — the sections below are separated by rules, not files.
An earlier split put the probes in src/diagnostics.py so a healthz endpoint could
reuse them, but healthz must stay a dumb liveness probe (failing it on dependency
health turns a Redis blip into a pod restart loop), so that consumer never
existed.

src/musicbot.py holds only the command registration and delegates in here.
"""

import asyncio
import math
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

# Live-edit loop tunables (see run_health_dashboard), env-overridable for
# slow/remote deployments. Defined in config.py rather than here so there is one
# answer to "what does this bot read from the environment?".
from src.config import (
    DEFAULT_POSTGRES_PASSWORD,
    ENVIRONMENT,
    PING_DEADLINE_SECS,
    PING_TICK_SECS,
    SpotifyStatus,
    history_archive_enabled,
    using_default_postgres_password,
)
from src.spotify import Spotify
from src.util import get_logger, latency_color, send_embed, trace_footer

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · PROBES & VERSIONS — infrastructure, no Discord message concepts
# ════════════════════════════════════════════════════════════════════════════
# Nothing below this rule until SECTION 2 knows about embeds, channels or ctx. Each
# probe returns a ProbeResult and never raises out — a dead dependency becomes DOWN so
# the caller can always render. CancelledError is the one exception let through: the
# live-edit loop cancels still-pending probes at its deadline and flips them to FAILED.

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
    broken), OOM, READONLY (replica) — which tells an operator far more than the
    redis-py class name ("ResponseError"). Falls back to the class name.
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
    """PING *and* a throwaway write.

    PING alone is misleading: Redis keeps serving reads while refusing writes —
    MISCONF after a failed bgsave (observed live: full disk failed every guild's
    state write while -ping stayed green), OOM under maxmemory+noeviction,
    READONLY on a replica. The bot writes constantly, so a probe that never writes
    isn't measuring what it needs. One short-TTL key exercises the write path.
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
    """Spotify's row: the *source's* usability, not just reachability.

    `status` (from MusicBot._spotify_status' startup probe) is the only thing that
    separates "configured but rejected" from "reachable but slow", without
    spending a doomed API call. Defaults to ENABLED so a caller with no status
    still gets the plain reachability probe.
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
    """The play-history archive's Postgres row.

    A None archive splits on the flag. Disabled (HISTORY_ARCHIVE_ENABLED off —
    the ship default), setup_hook deliberately built no archive and the row says
    OFF rather than shrugging, so the operator's choice is visible instead of
    reading like a fault. None with the flag ON means the bot was built without
    an archive some other way — reachable only in tests and in a cog constructed
    outside MusicBotApp — and stays NA.
    """
    if archive is None:
        if not history_archive_enabled():
            return ProbeResult("Postgres", ProbeState.OFF, detail="archive disabled")
        return ProbeResult("Postgres", ProbeState.NA)

    return await _timed("Postgres", archive.health_check)


async def probe_otel() -> ProbeResult:
    if telemetry._tracer_provider is None:
        return ProbeResult("OTEL collector", ProbeState.OFF)
    # urlparse fills .hostname/.port only with a scheme present, and operators often
    # set OTEL_EXPORTER_OTLP_ENDPOINT scheme-less ("collector:4317"), which parses to
    # hostname=None and silently probes localhost. Prepend "//" so we connect to the
    # endpoint they actually configured.
    raw = telemetry._OTLP_ENDPOINT
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    host, port = parsed.hostname or "localhost", parsed.port or 4317

    async def _do() -> None:
        # gRPC OTLP has no cheap app-level ping, so a TCP connect proves only that
        # the port accepts connections — not a real OTLP handshake. Unlike the
        # auto-instrumented Redis/aiohttp probes it emits no child span, so this
        # row never appears in the ping's fan-out trace.
        _, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()

    return await _timed("OTEL collector", _do)


# ── Versions ─────────────────────────────────────────────────────────────────


def bot_version() -> str:
    """The bot's own version, cached for process lifetime.

    From pyproject.toml (copied into the runtime image), not dist metadata: the
    container installs with `poetry install --no-root`, so the project is never a
    metadata-bearing distribution and metadata.version() would raise
    PackageNotFoundError. Falls back to dist metadata for a wheel install that ships
    no pyproject.toml, then "unknown".
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
    """All versions for the embed's Versions block. ffmpeg's first call shells out,
    so it goes to the default executor to keep the loop unblocked; the rest are dict
    lookups. This single await also lets the already-scheduled immediate probes
    (NA/OFF) complete so the skeleton send can pre-drain them."""
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

# One band table drives BOTH the status dot and the embed accent, so a green dot
# can't appear under a yellow accent. These bands (≤100/≤200) differ on purpose
# from util.latency_color's (≤50/≤100/≤200), which backs the lighter
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
    # ConnectionError) is the actionable half and costs one parenthetical. NA and
    # OFF get the same treatment when they have a reason to give — "not
    # configured" is why an optional dependency is dark, and probe_postgres's
    # "archive disabled" is what distinguishes an opted-out row from any other OFF.
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
) -> discord.Embed:
    """Build the live health embed from the current probe results + versions.

    Accent: any down/failed → red; else any still-pending → blurple; else the
    worst OK latency's band colour (same bands as the dots)."""
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
    footer = f"environment: {ENVIRONMENT}"
    if any(r.state is ProbeState.PENDING for r in rows):
        footer += " · probing…"
    if (tf := trace_footer(span)) is not None:
        footer += f" · {tf}"
    embed.set_footer(text=footer)
    return embed


def default_password_embed() -> Optional[discord.Embed]:
    """A standing warning that the Postgres password is still the compose
    default, or None when it is not.

    Rendered on EVERY -ping rather than once at startup, deliberately. The
    startup log is seen once and usually by whoever already knows; this is the
    surface an operator looks at repeatedly, so it is the one that keeps the
    problem visible until it is fixed.

    Static — nothing here changes between ticks — so the dashboard's
    edit-on-change loop is unaffected: it still diffs only the health embed.

    IT READS THE DSN, NOT THE SERVER, and the remedy below is ordered around
    that limit. `setup_env.sh --force` changes only the DSN, so performing it
    first clears this warning while Postgres still accepts the old password —
    the control switching itself off mid-remedy, in the window where it is most
    needed. Changing the server first means the warning survives until the fix
    is genuinely complete. A detector that could not be fooled would attempt a
    connection with DEFAULT_POSTGRES_PASSWORD instead of parsing a string.

    Gated on the archive flag, like setup_hook's startup ERROR: with the
    archive disabled there is no deployed Postgres for the bot to warn about,
    and a credential advisory for a database that isn't there is noise that
    trains operators to ignore warnings.
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
    return embed


def _ping_embed_changed(new: discord.Embed, old: discord.Embed) -> bool:
    """True when a re-render actually differs. Only the two field values, the
    colour and the footer move, and to_dict() captures all of them."""
    return new.to_dict() != old.to_dict()


async def _safe_edit(
    message: discord.Message, embeds: list[discord.Embed]
) -> Optional[discord.Embed]:
    """Edit a message, tolerating a host the user deleted mid-loop (mirrors
    musicplayer._progress_updater). Returns the HEALTH embed on success (the
    caller diffs against that one alone), None if the message is gone.

    Takes the full embed list because the warning above rides along on every
    edit; dropping it would make it flicker away on the first tick."""
    try:
        await message.edit(embeds=embeds)
        return embeds[-1]
    except discord.NotFound:
        return None


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 · COMMAND BODIES — what src/musicbot.py's cog delegates to
# ════════════════════════════════════════════════════════════════════════════


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
    # No default, unlike spotify_status: `redis` and `spotify` are also
    # required-but-Optional, and for the same reason. A default of None would let
    # a new caller silently render the required Postgres tier as "n/a" by simply
    # forgetting the argument — which is exactly the bug this parameter had while
    # it was a pool nobody passed.
    archive: Optional[ArchiveHealth],
) -> None:
    """Optimistic-send + live-edit health dashboard.

    Sends a skeleton embed immediately, then edits in place each tick as probes
    return; a hard deadline fails any straggler. Runs inside the caller's span and
    lets exceptions propagate — the command owns the user-facing error reply.
    """
    span = trace.get_current_span()
    loop = asyncio.get_running_loop()
    tasks: dict[str, asyncio.Task[ProbeResult]] = {}

    def _drain() -> bool:
        """Fold every finished probe into `results`; True if a row moved. Only
        genuinely-completed tasks are done() here (cancellation happens at the
        deadline/finally), so .result() never re-raises."""
        changed = False
        for label in [lbl for lbl in pending if tasks[lbl].done()]:
            results[label] = tasks[label].result()
            pending.discard(label)
            changed = True
        return changed

    try:
        # 1. launch INSIDE try so `finally` cancels them wherever a later await
        #    raises. create_task copies the otel context, so child probe spans
        #    nest under bot.ping.
        tasks = {
            "Redis": asyncio.create_task(probe_redis(redis)),
            "Spotify API": asyncio.create_task(probe_spotify(spotify, spotify_status)),
            "Postgres": asyncio.create_task(probe_postgres(archive)),
            "OTEL collector": asyncio.create_task(probe_otel()),
        }
        results = {label: ProbeResult(label, ProbeState.PENDING) for label in tasks}
        pending = set(tasks)

        # 2. instant data + skeleton. collect_versions()'s executor hop lets the
        #    immediate NA/OFF probes complete; pre-drain them so those rows never
        #    flash "pending…" for a tick.
        versions = await collect_versions()
        _drain()
        discord_ms = bot_latency * 1000
        last = render_ping_embed(results, versions, discord_ms, span)
        # Computed once: it is static, so it costs nothing per tick and cannot
        # make _ping_embed_changed see a difference that is not there.
        #
        # OWNER ONLY. -ping has no permission gate, so this advisory — which names
        # the credential and confirms THIS host runs on it — otherwise reaches
        # every member of every guild and stays in Discord's retained history. The
        # value is a public constant in a GPL repo, so the leak was never the
        # string; it was the free confirmation of which hosts are worth trying.
        # The startup ERROR still fires on every boot for anyone reading logs.
        #
        # ORDER IS LOAD-BEARING, not style. is_owner() is not a local predicate:
        # MusicBotApp passes neither owner_id nor owner_ids, so discord.py falls
        # through to application_info() — a REST GET, retried up to 5 times over
        # ~25s on a 5xx, and it raises rather than returning False. Written as
        # `embed() if await is_owner() else None` the await runs FIRST and
        # UNCONDITIONALLY, putting a network round trip ahead of the skeleton send
        # this function promises is immediate — so the dashboard broke exactly when
        # Discord was degraded, which is when it gets run. PING_DEADLINE_SECS does
        # not bound it; that clock starts after the send. Ask the cheap, local
        # question first.
        warning = default_password_embed()
        if warning is not None and not await ctx.bot.is_owner(ctx.author):
            warning = None
        message = await ctx.channel.send(  # bypass NP host
            embeds=[e for e in (warning, last) if e is not None]
        )

        # 3. live-edit loop: tick, drain, edit-on-change; exit early when done.
        deadline = loop.time() + PING_DEADLINE_SECS
        while pending and (remaining := deadline - loop.time()) > 0:
            await asyncio.sleep(min(PING_TICK_SECS, remaining))
            if _drain():
                embed = render_ping_embed(results, versions, discord_ms, span)
                if _ping_embed_changed(embed, last):
                    edited = await _safe_edit(
                        message, [e for e in (warning, embed) if e is not None]
                    )
                    last = edited or last

        # 4. deadline: fail only what is STILL pending. Re-check done() first — a
        #    probe can finish during step 3's final edit await, and flipping it to
        #    FAILED unconditionally would report a healthy dep as red.
        if pending:
            for label in pending:
                if tasks[label].done():
                    results[label] = tasks[label].result()
                else:
                    tasks[label].cancel()
                    results[label] = ProbeResult(label, ProbeState.FAILED)
            await _safe_edit(
                message,
                [
                    e
                    for e in (
                        warning,
                        render_ping_embed(results, versions, discord_ms, span),
                    )
                    if e is not None
                ],
            )

        for r in results.values():  # self-documenting trace
            span.set_attribute(f"ping.{r.label}.state", r.state.name.lower())
            if r.latency_ms is not None:
                span.set_attribute(f"ping.{r.label}.latency_ms", round(r.latency_ms, 2))
    finally:
        for t in tasks.values():  # never leak a probe task
            if not t.done():
                t.cancel()
