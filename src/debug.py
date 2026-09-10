"""Debug mode: the `-debug` diagnostic snapshot and the per-guild toggle behind it.

OBSERVATION-ONLY: nothing here changes playback, caching, queueing or persistence,
only what the bot shows, so nothing validated with debug on changes when it flips
off. Every collector degrades to a labeled `unknown`/`n/a` rather than raising
(see _safe_block). src/musicbot.py owns only the command registration and the
override state; parsing, collection and rendering live here, as -ping's do.
"""

import asyncio
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
import discord
import orjson
from discord.ext import commands
from opentelemetry import trace

from src import config
from src.config import (
    DEBUG_DEADLINE_SECS,
    DEBUG_TICK_SECS,
    debug_mode_default,
)
from src.dashboard import run_live_dashboard
from src.ping import bot_version, collect_versions
from src.redis_client import GuildRedisStore, outbox_depth, read_guild_configs
from src.util import (
    FOOTER_SUFFIX_SEP,
    cancel_task,
    fmt_duration,
    get_logger,
    join_footer,
    trace_footer,
    trace_id_of,
    truncate,
)

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.history_archive import ArchiveStats

    # A runtime import would close the cycle (musicbot imports this module).
    from src.musicplayer import MusicPlayer
    from src.ytdlp_pool import PoolState

log = get_logger(__name__)


class ArchiveStatsReader(Protocol):
    """What -debug's Postgres block needs from the archive, declared structurally
    (like ping's ArchiveHealth) so `import asyncpg` stays out of this module's
    runtime graph; the ArchiveStats annotation is a type, not an edge."""

    async def stats(self) -> ArchiveStats: ...


# The extension loads within seconds of process start.
_PROCESS_START = time.time()

# The CPU sampling window; also the loop-lag measurement, so the two share one wait.
_CPU_WINDOW_SECS = 0.5
_GIT_PROBE_TIMEOUT_SECS = 2.0
# Bound on the whole Redis probe, window included: the pool sets no
# socket_timeout, so a server that accepts the socket and never answers would
# otherwise run to the dashboard deadline.
_REDIS_PROBE_TIMEOUT_SECS = _CPU_WINDOW_SECS + 3.0
# Postgres samples over a longer window: pg_stat_database flushes a backend's
# stats at transaction end and at most about once a second, so a shorter delta
# reads flush quantization rather than load.
_PG_WINDOW_SECS = 2.0
# Bounds the two-sample Postgres probe. Defense in depth behind the dashboard
# deadline: covers a raised DEBUG_DEADLINE_SECS and the pre-loop send, and does
# not stretch with either.
_PG_PROBE_TIMEOUT_SECS = _PG_WINDOW_SECS + 6.0
_git_sha_cache: Optional[str] = None

# Block order in the embed, and which probe fills each deferred one — what lets
# the deadline mark exactly the blocks a straggler owed.
_BLOCK_ORDER = (
    "Build",
    "Versions",
    "Config",
    "Runtime",
    "Discord",
    "Redis",
    "Postgres",
    "This server",
    "Checks",
)
_PROBE_BLOCKS: dict[str, tuple[str, ...]] = {
    "runtime": ("Runtime",),
    "redis": ("Redis", "Checks"),
    "postgres": ("Postgres",),
    "build": ("Build",),
}
_PENDING_LINES = ["⏳ collecting…"]
# Named rather than blank: a vanished block would read as "this host has no Postgres".
_TIMEOUT_LINES = ["⚠️ timed out"]

# What a non-owner is told instead of the host blocks.
_OPERATOR_NOTICE = (
    "-# Host details (configuration, storage, runtime) are shown to the bot owner "
    "only. Run `-ping` for dependency health."
)

# Discord's cap on an embed field value; util.py's FOOTER_LIMIT is its footer sibling.
_FIELD_LIMIT = 1024

_DEBUG_COLOR = discord.Color(0xE67E22)  # amber: an operator surface, not an alert


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · TOGGLE PARSING
# ════════════════════════════════════════════════════════════════════════════


class DebugAction(Enum):
    """What a `-debug` invocation asks for."""

    STATUS = "status"
    ENABLE = "enable"
    DISABLE = "disable"


_ACTIONS: dict[str, DebugAction] = {
    "": DebugAction.STATUS,
    "--enable": DebugAction.ENABLE,
    "--disable": DebugAction.DISABLE,
}

DEBUG_USAGE = "`-debug`, `-debug --enable`, `-debug --disable`"


def parse_debug_arg(arg: str) -> Optional[DebugAction]:
    """The action `arg` names, or None. Hand-parsed: these are valueless switches
    a FlagConverter cannot express, and `-debug enable` gets a did-you-mean here
    instead of a FlagError before the body runs."""
    return _ACTIONS.get(arg.strip().lower())


def unknown_arg_message(arg: str) -> str:
    """The reply for an unparseable argument. Never echoes the argument: user
    text rendered into a bot embed is a mention-injection surface."""
    cleaned = arg.strip().lower()
    if f"--{cleaned}" in _ACTIONS:
        return f"Did you mean `-debug --{cleaned}`? Options take two dashes."
    return f"Unknown option. Usage: {DEBUG_USAGE}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 · FOOTER DECORATION
# ════════════════════════════════════════════════════════════════════════════
# With debug mode on, every embed the bot sends grows a footer identifying the
# request. See docs/ARCHITECTURE.md#debug-footer-seams.

_DEBUG_MARK = "🐞"


def debug_footer(
    *,
    span: Optional[trace.Span] = None,
    elapsed_ms: Optional[float] = None,
    shard_id: Optional[int] = None,
    runtime: Optional[RuntimeSnapshot] = None,
    skip_environment: bool = False,
    skip_trace: bool = False,
) -> str:
    """The debug suffix, or "" when nothing is worth showing. The environment
    leads (a property of the process); every other part has an absent case. Two
    lines — where the request ran, then what it counted — broken here rather than
    by Discord, which wraps mid-segment on the 32-hex trace id."""
    where: list[str] = []
    counts: list[str] = []
    if not skip_environment:
        where.append(config.ENVIRONMENT)
    if elapsed_ms is not None:
        where.append(f"{round(elapsed_ms)} ms")
    if shard_id is not None:
        where.append(f"shard {shard_id}")
    if runtime is not None:
        if runtime.cpu_percent is not None:
            where.append(f"cpu {runtime.cpu_percent:.0f}%")
        if runtime.mem_percent is not None:
            where.append(f"mem {runtime.mem_percent:.0f}%")
        where.append(f"lag {runtime.lag_ms:.1f} ms")
        counts.append(f"tasks {runtime.tasks}")
        counts.append(f"pool {runtime.pool_workers}")
    if not skip_trace and span is not None and (trace_id := trace_id_of(span)):
        counts.append(f"trace {trace_id}")
    lines = [" · ".join(line) for line in (where, counts) if line]
    if not lines:
        return ""
    return f"{_DEBUG_MARK} " + FOOTER_SUFFIX_SEP.join(lines)


def _strip_debug_suffix(text: str) -> str:
    """`text` with any previous debug suffix and the break before it removed. The
    first mark is the boundary — no bot-authored footer contains one (asserted by
    a test) — so a doubled suffix heals in one pass."""
    idx = text.find(_DEBUG_MARK)
    if idx == -1:
        return text
    return text[:idx].removesuffix(FOOTER_SUFFIX_SEP)


def strip_debug_footers(embeds: Sequence[discord.Embed]) -> None:
    """Remove a previous debug suffix — what the seams call while debug mode is
    off. Decoration is in place and `play_message` outlives the toggle, so a
    --disable has to strip rather than skip."""
    for embed in embeds:
        existing = embed.footer.text or ""
        if _DEBUG_MARK not in existing:
            continue
        _write_footer(embed, _strip_debug_suffix(existing), "")


def decorate_embeds(
    embeds: Sequence[discord.Embed],
    *,
    span: Optional[trace.Span] = None,
    elapsed_ms: Optional[float] = None,
    shard_id: Optional[int] = None,
    runtime: Optional[RuntimeSnapshot] = None,
) -> None:
    """Write the debug footer onto each embed in place, replacing a previous
    suffix so a cached embed sent more than once (`play_message`, re-served by
    -now) does not grow a footer per send. Every embed gets a suffix — the
    environment alone is always worth showing."""
    for embed in embeds:
        existing = embed.footer.text or ""
        base = _strip_debug_suffix(existing)
        suffix = debug_footer(
            span=span,
            elapsed_ms=elapsed_ms,
            shard_id=shard_id,
            runtime=runtime,
            # Read from the pre-suffix footer: error embeds carry their own trace,
            # but a trace in our own previous suffix must not suppress the fresh one.
            skip_trace="trace:" in base or "trace " in base,
        )
        _write_footer(embed, base, suffix)


def _write_footer(embed: discord.Embed, base: str, suffix: str) -> None:
    """Write `base` and `suffix` as the footer, the suffix on its own line."""
    text = join_footer(base, suffix)
    embed.set_footer(
        text=text or None,
        # Discord rejects an icon with no text.
        icon_url=embed.footer.icon_url if text else None,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 · CONFIG ALLOWLIST
# ════════════════════════════════════════════════════════════════════════════
# An env var absent from _CONFIG_ALLOWLIST does not render at all, so a knob added
# later opts in at review time and a future secret can never leak by default.


class _ConfigKind(Enum):
    VALUE = "value"  # rendered as configured
    SECRET = "secret"  # presence only — the value never renders
    URL = "url"  # userinfo and credential-bearing query params stripped


@dataclass(frozen=True, slots=True, kw_only=True)
class _ConfigVar:
    name: str
    kind: _ConfigKind
    # What the bot uses when unset, as text; read from config's resolved constants
    # where there is one, so it cannot drift from the real default.
    fallback: Optional[str] = None
    # For a default main() can still replace after this tuple is built.
    fallback_factory: Optional[Callable[[], str]] = None


_CONFIG_ALLOWLIST: tuple[_ConfigVar, ...] = (
    _ConfigVar(
        name="ENVIRONMENT",
        kind=_ConfigKind.VALUE,
        fallback_factory=lambda: config.ENVIRONMENT,
    ),
    _ConfigVar(name="DEBUG_MODE", kind=_ConfigKind.VALUE, fallback="false"),
    _ConfigVar(
        name="HISTORY_ARCHIVE_ENABLED", kind=_ConfigKind.VALUE, fallback="false"
    ),
    _ConfigVar(
        name="HISTORY_OUTBOX_MAX",
        kind=_ConfigKind.VALUE,
        fallback=str(config.HISTORY_OUTBOX_MAX),
    ),
    _ConfigVar(
        name="POSTGRES_STATEMENT_CACHE",
        kind=_ConfigKind.VALUE,
        fallback=str(config.POSTGRES_STATEMENT_CACHE),
    ),
    _ConfigVar(name="YTDLP_POOL_WORKERS", kind=_ConfigKind.VALUE, fallback="4"),
    _ConfigVar(
        name="PLAY_INFLIGHT_MAX",
        kind=_ConfigKind.VALUE,
        fallback=str(config.PLAY_INFLIGHT_MAX),
    ),
    _ConfigVar(
        name="PLAY_RESOLVE_CONCURRENCY",
        kind=_ConfigKind.VALUE,
        fallback=str(config.PLAY_RESOLVE_CONCURRENCY),
    ),
    _ConfigVar(
        name="NOW_PLAYING_UPDATE_INTERVAL_SECS",
        kind=_ConfigKind.VALUE,
        fallback=str(config.NOW_PLAYING_UPDATE_INTERVAL_SECS),
    ),
    _ConfigVar(
        name="HEARTBEAT_INTERVAL_SECS",
        kind=_ConfigKind.VALUE,
        fallback=str(config.HEARTBEAT_INTERVAL_SECS),
    ),
    _ConfigVar(
        name="PING_TICK_SECS",
        kind=_ConfigKind.VALUE,
        fallback=str(config.PING_TICK_SECS),
    ),
    _ConfigVar(
        name="PING_DEADLINE_SECS",
        kind=_ConfigKind.VALUE,
        fallback=str(config.PING_DEADLINE_SECS),
    ),
    _ConfigVar(
        name="DEBUG_TICK_SECS", kind=_ConfigKind.VALUE, fallback=str(DEBUG_TICK_SECS)
    ),
    _ConfigVar(
        name="DEBUG_DEADLINE_SECS",
        kind=_ConfigKind.VALUE,
        fallback=str(DEBUG_DEADLINE_SECS),
    ),
    _ConfigVar(
        name="POT_PROVIDER_URL", kind=_ConfigKind.URL, fallback="http://127.0.0.1:4416"
    ),
    _ConfigVar(name="OTEL_SDK_DISABLED", kind=_ConfigKind.VALUE, fallback="false"),
    _ConfigVar(
        name="OTEL_SERVICE_NAME", kind=_ConfigKind.VALUE, fallback="discord-music-bot"
    ),
    _ConfigVar(
        name="OTEL_EXPORTER_OTLP_ENDPOINT",
        kind=_ConfigKind.URL,
        fallback="http://localhost:4317",
    ),
    _ConfigVar(
        name="REDIS_URL", kind=_ConfigKind.URL, fallback="redis://localhost:6379"
    ),
    _ConfigVar(name="DEBUG_PROMETHEUS_URL", kind=_ConfigKind.URL),
    # Credential-bearing: presence only. POSTGRES_URL embeds the password in its
    # userinfo, so it is SECRET rather than URL.
    _ConfigVar(name="DISCORD_TOKEN", kind=_ConfigKind.SECRET),
    _ConfigVar(name="SPOTIFY_CLIENT_ID", kind=_ConfigKind.SECRET),
    _ConfigVar(name="SPOTIFY_CLIENT_SECRET", kind=_ConfigKind.SECRET),
    _ConfigVar(name="POSTGRES_URL", kind=_ConfigKind.SECRET),
)

# Substrings marking a query parameter whose value is a credential, matched
# against the lowercased key.
_CREDENTIAL_QUERY_KEYS = (
    "pass",  # also covers passwd / password
    "pwd",
    "secret",
    "token",
    "key",
    "auth",
    "sig",
    "cred",
)


def redact_url(raw: str, *, hide_host: bool = False) -> str:
    """A URL safe to render: userinfo and credential-bearing query values replaced
    with `***` — the two places a DSN hides a password."""
    # The whole body is guarded: `.port` is lazy, so urlsplit("redis://h:99999/0")
    # succeeds and the ValueError fires on dereference.
    try:
        # Scheme-less values go to an authority first; otherwise urlsplit puts the
        # host in .scheme/.path and every rewrite below misses it.
        parts = urlsplit(raw if "://" in raw else f"//{raw}")
        netloc = parts.netloc
        if parts.username is not None or parts.password is not None:
            host = parts.hostname or ""
            netloc = f"***@{host}:{parts.port}" if parts.port else f"***@{host}"
        if hide_host:
            # The snapshot is posted where the operator typed, so host:port is
            # topology to redact unconditionally — localhost included, or the
            # presence of `***` becomes the disclosure.
            if "://" not in raw:
                # Without a scheme the host can land in .path ("http:/host:9090/x"),
                # so no single component rewrite is safe.
                return "***"
            netloc = "***"
        query = parts.query
        if query:
            # safe="*" so the marker survives quoting as `***`, not %2A%2A%2A.
            query = urlencode(
                [
                    (
                        k,
                        "***"
                        if any(t in k.lower() for t in _CREDENTIAL_QUERY_KEYS)
                        else v,
                    )
                    for k, v in parse_qsl(query, keep_blank_values=True)
                ],
                safe="*",
            )
        # Everything after a stray `#` lands in the fragment, past both redactions
        # above. Marked, not dropped, so the row still says something was there.
        fragment = "***" if parts.fragment else ""
        return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
    except ValueError:
        return "unparseable"


def render_config_value(var: _ConfigVar) -> str:
    """One allowlist row's value, redacted per its kind."""
    raw = os.environ.get(var.name)
    if var.kind is _ConfigKind.SECRET:
        # Presence only: this block has to stay safe to screenshot into an issue.
        return "set" if (raw or "").strip() else "unset"
    if raw is None or not raw.strip():
        fallback = var.fallback_factory() if var.fallback_factory else var.fallback
        return "unset" if fallback is None else f"{fallback} (default)"
    if var.kind is _ConfigKind.URL:
        return redact_url(raw.strip(), hide_host=True)
    return raw.strip()


def config_lines() -> list[str]:
    width = max(len(var.name) for var in _CONFIG_ALLOWLIST) + 2
    return [
        f"{var.name:<{width}}{render_config_value(var)}" for var in _CONFIG_ALLOWLIST
    ]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 · COLLECTORS — one block of lines each, none of them raising
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, kw_only=True)
class DebugInputs:
    """What the cog hands the snapshot: state this module cannot reach without
    importing MusicBot. New fields take defaults, so the cog's call site stays
    one expression."""

    debug_enabled: bool
    debug_overridden: bool
    # False only when a toggle's Redis write failed, so the snapshot can say
    # "this session only" rather than claim a durability that did not happen.
    debug_persisted: bool = True
    players: int
    player: Optional[MusicPlayer] = None
    redis: Optional[aioredis.Redis] = None
    store: Optional[GuildRedisStore] = None
    archive: Optional[ArchiveStatsReader] = None
    archive_enabled: bool = False
    prometheus_url: Optional[str] = None
    # Gates every block describing the HOST rather than the caller's own server.
    # Defaults False so a call site that forgets to ask discloses nothing.
    operator: bool = False
    # None = not asked, and only an operator is asked. Set symmetrically with
    # `operator`, or the row's absence would be the answer.
    default_password: Optional[bool] = None
    # Debug mode's footer, rendered once by the cog and constant for the loop.
    # None when the guild has debug mode off.
    debug_suffix: Optional[str] = None


def _safe_block(label: str, fn: Callable[[], list[str]]) -> list[str]:
    """Run a collector, or render why it could not run."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — a debug collector must never raise out
        log.warning(f"debug block {label!r} failed: {type(e).__name__}: {e}")
        return [f"unavailable ({type(e).__name__})"]


async def _safe_block_async(
    label: str, fn: Callable[[], Awaitable[list[str]]]
) -> list[str]:
    try:
        return await fn()
    except Exception as e:  # noqa: BLE001 — same rule for the collectors that do IO
        log.warning(f"debug block {label!r} failed: {type(e).__name__}: {e}")
        return [f"unavailable ({type(e).__name__})"]


async def _dbsize(redis: Optional[aioredis.Redis]) -> Optional[int]:
    if redis is None:
        return None
    try:
        return int(await redis.dbsize())
    except Exception as e:  # noqa: BLE001 — a key count is a row, not a failure
        log.warning(f"debug DBSIZE failed: {type(e).__name__}: {e}")
        return None


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


def _fmt_ms(seconds: float) -> str:
    """Latency in ms; nan is discord.py's gateway-reconnecting value and inf the
    voice client's before-first-heartbeat value."""
    ms = seconds * 1000
    if math.isnan(ms):
        return "reconnecting"
    if math.isinf(ms):
        return "warming up"
    return f"{round(ms)} ms"


def git_sha() -> str:
    """The commit this build was made from, cached for process lifetime: the
    GIT_SHA baked into the image (a dirty tag passes through unchanged), else
    `git rev-parse` for `just run` from a checkout."""
    global _git_sha_cache
    if _git_sha_cache is not None:
        return _git_sha_cache
    baked = (os.environ.get("GIT_SHA") or "").strip()
    if baked:
        _git_sha_cache = baked
        return baked
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_PROBE_TIMEOUT_SECS,
            check=True,
        )
        _git_sha_cache = result.stdout.strip() or "unknown"
    except Exception as e:  # noqa: BLE001 — no checkout and no ENV is not an error
        log.debug(f"git sha probe failed: {type(e).__name__}: {e}")
        _git_sha_cache = "unknown"
    return _git_sha_cache


def build_lines(sha: Optional[str]) -> list[str]:
    """`sha` is required and resolved by the caller off the event loop: git_sha()
    shells out (~16ms), and a default would make that reachable from any caller
    that forgot. None means "looked, and there is none"."""
    return [
        f"version      {bot_version()}",
        f"commit       {sha if sha is not None else 'unknown'}",
        f"environment  {config.ENVIRONMENT}",
        f"container    {'yes' if _in_container() else 'no'}",
    ]


def version_lines(versions: dict[str, str]) -> list[str]:
    return [
        f"bot          {versions['bot']}",
        f"yt-dlp       {versions['yt-dlp']}",
        f"ffmpeg       {versions['ffmpeg']}",
        f"python       {versions['python']}  ·  discord.py {versions['discord.py']}",
    ]


# ── Process & event-loop readers ──────────────────────────────────────────────
# psutil is not cgroup-aware (psutil#2100): inside a container it reports the
# HOST, so the cgroup files are read by hand.

# Module constants so tests can point them at a fixture directory.
_CGROUP = Path("/sys/fs/cgroup")
_PROC_STATUS = Path("/proc/self/status")
_PROC_MEMINFO = Path("/proc/meminfo")


def _read_text(path: Path) -> str:
    """A pseudo-file's contents, or "" wherever it does not exist (macOS dev,
    cgroup v1, a restricted container). Never raises."""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _read_int(path: Path) -> Optional[int]:
    raw = _read_text(path)
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class CpuSample:
    """A CUMULATIVE cpu reading plus the clock it was taken at; CPU% exists only
    between two of these. `scope` says what was measured: cgroup covers the whole
    container (FFmpeg and pool workers included), os.times() only this process
    and its REAPED children."""

    seconds: float
    monotonic: float
    scope: str
    cores: float


def cpu_cores() -> float:
    """Cores this process may use: the cgroup quota when set, else the host
    count. A container capped at 1.5 cores must not report 12% of 12 while
    saturated."""
    raw = _read_text(_CGROUP / "cpu.max").split()
    if len(raw) == 2 and raw[0] != "max":
        try:
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                return quota / period
        except ValueError:
            pass
    return float(os.cpu_count() or 1)


def read_cpu_sample() -> CpuSample:
    # Gated on _in_container(): on bare metal /sys/fs/cgroup is the ROOT cgroup
    # and the counter covers every process on the machine.
    if _in_container():
        for line in _read_text(_CGROUP / "cpu.stat").splitlines():
            if line.startswith("usage_usec "):
                try:
                    usec = int(line.split()[1])
                except IndexError, ValueError:
                    break
                return CpuSample(
                    seconds=usec / 1_000_000,
                    monotonic=time.monotonic(),
                    scope="container",
                    cores=cpu_cores(),
                )
    times = os.times()
    return CpuSample(
        seconds=times.user + times.system + times.children_user + times.children_system,
        monotonic=time.monotonic(),
        scope="process",
        cores=float(os.cpu_count() or 1),
    )


def cpu_percent(first: CpuSample, second: CpuSample) -> Optional[float]:
    """Percent of available cores used between two samples, or None when the
    window is too short to divide by."""
    wall = second.monotonic - first.monotonic
    if wall <= 0 or second.cores <= 0:
        return None
    return max(0.0, (second.seconds - first.seconds) / (wall * second.cores) * 100)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryReading:
    used_bytes: int
    limit_bytes: Optional[int]
    scope: str  # "container" | "process"
    label: str  # what used_bytes measures: current / rss / peak

    @property
    def percent(self) -> Optional[float]:
        if not self.limit_bytes:
            return None
        return self.used_bytes / self.limit_bytes * 100


def read_memory() -> MemoryReading:
    """Memory used and the ceiling it counts against, best source first: the
    cgroup pair the OOM killer acts on, then /proc, then getrusage. Gated on
    _in_container() as read_cpu_sample is — off-container the cgroup path is the
    whole machine's usage, labelled `(container)`."""
    current = _read_int(_CGROUP / "memory.current") if _in_container() else None
    if current is not None:
        return MemoryReading(
            used_bytes=current,
            limit_bytes=_cgroup_memory_limit(),
            scope="container",
            label="current",
        )
    rss = _proc_vmrss_bytes()
    if rss is not None:
        return MemoryReading(
            used_bytes=rss, limit_bytes=_physical_memory(), scope="process", label="rss"
        )
    # macOS dev: no /proc. ru_maxrss is a PEAK, in bytes on darwin and KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return MemoryReading(
        used_bytes=peak if sys.platform == "darwin" else peak * 1024,
        limit_bytes=_physical_memory(),
        scope="process",
        label="peak",
    )


def _cgroup_memory_limit() -> Optional[int]:
    raw = _read_text(_CGROUP / "memory.max")
    if raw and raw != "max":
        try:
            return int(raw)
        except ValueError:
            return None
    # "max" means uncapped, so the host's total is the real ceiling.
    return _meminfo_total()


def _meminfo_total() -> Optional[int]:
    for line in _read_text(_PROC_MEMINFO).splitlines():
        if line.startswith("MemTotal:"):
            try:
                return int(line.split()[1]) * 1024
            except IndexError, ValueError:
                return None
    return None


def _proc_vmrss_bytes() -> Optional[int]:
    for line in _read_text(_PROC_STATUS).splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except IndexError, ValueError:
                return None
    return None


def _physical_memory() -> Optional[int]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except ValueError, OSError, AttributeError:
        return _meminfo_total()


async def measure_loop_lag(delay: float) -> float:
    """How late the loop ran a callback asked for `delay` seconds out, in ms.
    Doubles as the CPU window when the caller brackets its samples around it."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.sleep(delay)
    return max(0.0, (loop.time() - start - delay) * 1000)


def pool_state() -> PoolState:
    # Resolved per call: the test seam monkeypatches src.youtube.ytdlp_pool.
    from src.youtube import ytdlp_pool

    return ytdlp_pool.state


def chart_pool_state() -> PoolState:
    # Same per-call rule: conftest patches src.chart_pool.chart_pool.
    from src.chart_pool import chart_pool

    return chart_pool.state


def runtime_lines(
    *,
    cpu: Optional[float] = None,
    cpu_scope: str = "",
    cpu_total: Optional[float] = None,
    cores: Optional[float] = None,
    memory: Optional[MemoryReading] = None,
    lag_ms: Optional[float] = None,
    tasks: Optional[int] = None,
) -> list[str]:
    # A plain duration, not a `<t:…:R>` timestamp: inside a ``` fence Discord
    # prints markup as its literal source.
    lines = [f"uptime       {fmt_duration(int(time.time() - _PROCESS_START))}"]
    if cpu is None:
        lines.append("cpu          unknown")
    else:
        total = f" · {fmt_duration(int(cpu_total))} total" if cpu_total else ""
        lines.append(f"cpu          {cpu:.1f}% of {cores:g} cpus ({cpu_scope}){total}")
    if memory is None:
        lines.append("mem          unknown")
    else:
        pct = f" ({memory.percent:.0f}%)" if memory.percent is not None else ""
        limit = f" / {_mb(memory.limit_bytes)}" if memory.limit_bytes else ""
        lines.append(
            f"mem          {_mb(memory.used_bytes)}{limit}{pct} · "
            f"{memory.label} ({memory.scope})"
        )
    lag = f" · lag {lag_ms:.1f} ms" if lag_ms is not None else ""
    # "loop tasks": all_tasks() is loop-global and counts discord.py's internals.
    # `tasks` is counted by the caller BEFORE the dashboard launches its probes;
    # counting here would inflate by however many are still in flight.
    count = len(asyncio.all_tasks()) if tasks is None else tasks
    lines.append(f"loop         {count} loop tasks{lag}")
    state = pool_state()
    spawned = "spawned" if state.spawned else "not spawned"
    healed = f" · {state.generation} generations" if state.generation > 1 else ""
    lines.append(f"yt-dlp pool  {state.max_workers} workers · {spawned}{healed}")
    # Reported even when never spawned: "not spawned" answers "why is this bot
    # not holding 173 MB of matplotlib", and an absent row reads as a missing feature.
    chart = chart_pool_state()
    chart_spawned = "spawned" if chart.spawned else "not spawned"
    chart_healed = f" · {chart.generation} generations" if chart.generation > 1 else ""
    lines.append(
        f"chart pool   {chart.max_workers} worker · {chart_spawned}{chart_healed}"
    )
    return lines


def _mb(value: float) -> str:
    return f"{value / 1_048_576:.0f} MB"


def discord_lines(bot: commands.Bot, *, players: int) -> list[str]:
    # latencies is AutoShardedClient-only; the single-shard shape is the fallback
    # for a plain Bot (and the doubles in tests).
    latencies = getattr(bot, "latencies", None)
    if not isinstance(latencies, list):
        latencies = [(0, bot.latency)]
    shown = ", ".join(f"#{sid} {_fmt_ms(lat)}" for sid, lat in latencies[:4])
    if len(latencies) > 4:
        shown += ", …"
    return [
        f"shards       {len(latencies)}",
        f"gateway      {shown}",
        f"guilds       {len(bot.guilds)}",
        f"players      {players}",
        f"voice        {len(bot.voice_clients)} connected",
    ]


# ── The rolling sampler behind the footer ─────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeSnapshot:
    """The runtime metrics the debug footer prints. Optional fields because the
    first tick has no previous sample to rate against."""

    cpu_percent: Optional[float]
    mem_percent: Optional[float]
    lag_ms: float
    tasks: int
    pool_workers: int


# Delay before the sampler's first sample — see RuntimeSampler._run.
_FIRST_SAMPLE_SECS = 0.5


class RuntimeSampler:
    """A background sampler feeding the debug footer on the Now Playing tick's
    cadence. Background because CPU% needs a wall-clock window and a response
    must never wait on one; a send reads the cached snapshot, at most one tick
    old. The tick's own scheduling drift IS the loop-lag measurement. One
    instance, held on the cog: a module global would outlive a cog reload and
    leak the task."""

    # Tied to the NP tick, the fastest surface that renders a snapshot. Floored
    # so a tiny tick cannot spin /proc reads, capped so a long one cannot stale
    # command replies.
    INTERVAL_SECS = max(1.0, min(5.0, config.NOW_PLAYING_UPDATE_INTERVAL_SECS))

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task[None]] = None
        self._snapshot: Optional[RuntimeSnapshot] = None
        self._cpu: Optional[CpuSample] = None

    @property
    def snapshot(self) -> Optional[RuntimeSnapshot]:
        """The latest sample, or None before the first tick completes — the
        footer then omits the runtime segment rather than guessing."""
        return self._snapshot

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def apply(self, *, wanted: bool) -> None:
        """Start or stop to match whether ANY guild is effectively debug-enabled.
        Called at cog load as well as on every toggle: a DEBUG_MODE=true
        deployment starts enabled and no toggle ever fires."""
        if wanted:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self.running:
            return
        self._cpu = read_cpu_sample()
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        task, self._task = self._task, None
        self._snapshot = None
        self._cpu = None
        if task is not None and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        """Stop and await the task, so a cog reload cannot leave a sampler
        dripping /proc reads forever."""
        task = self._task
        self.stop()
        await cancel_task(task)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        # First sample on a short delay, so `-debug --enable` does not answer
        # with a footer carrying no runtime numbers; not instant, since cpu%
        # needs a window and start() took the baseline.
        delay = min(_FIRST_SAMPLE_SECS, self.INTERVAL_SECS)
        while True:
            expected = loop.time() + delay
            await asyncio.sleep(delay)
            try:
                self._snapshot = self._sample(max(0.0, (loop.time() - expected) * 1000))
            except Exception as e:  # noqa: BLE001 — one bad tick must not end the loop
                log.warning(f"runtime sample failed: {type(e).__name__}: {e}")
            delay = self.INTERVAL_SECS

    def _sample(self, lag_ms: float) -> RuntimeSnapshot:
        previous, self._cpu = self._cpu, read_cpu_sample()
        memory = read_memory()
        return RuntimeSnapshot(
            cpu_percent=cpu_percent(previous, self._cpu) if previous else None,
            mem_percent=memory.percent,
            lag_ms=lag_ms,
            tasks=len(asyncio.all_tasks()),
            pool_workers=pool_state().max_workers,
        )


def _voice_latency_text(vc: discord.VoiceClient) -> str:
    """Voice WS heartbeat latency and its 20-sample average. Both are inf (and
    average_latency raises ZeroDivisionError) until the first ACK, ~5s after
    joining — discord.py #6430."""
    try:
        latency = _fmt_ms(vc.latency)
        average = _fmt_ms(vc.average_latency)
    except ZeroDivisionError:
        return "warming up"
    if latency == "warming up":
        return latency
    return f"{latency} (avg {average})"


def _fence_safe(text: str) -> str:
    """User-controlled text, made safe to interpolate into a ``` fence. A voice-
    channel name is the one user-picked string here and the block is public: a
    backtick run closes the fence, and a masked link in a channel name renders
    clickable inside a card carrying the bot's own name."""
    return truncate(text.replace("`", "'"), 60)


def guild_lines(guild: discord.Guild, inputs: DebugInputs, *, source: str) -> list[str]:
    mp = inputs.player
    lines = [
        f"player       {'yes' if mp is not None else 'no'}",
        # display_size(), so a claimed-but-not-started song still reads as queued.
        f"queue        {mp.queue.display_size() if mp is not None else 0} queued",
        f"volume       {round(mp.volume * 100) if mp is not None else 100}%",
        f"debug        {'on' if inputs.debug_enabled else 'off'} ({source})",
    ]
    vc = guild.voice_client
    if not isinstance(vc, discord.VoiceClient) or vc.channel is None:
        lines.append("voice        not connected")
        return lines
    state = "playing" if vc.is_playing() else "paused" if vc.is_paused() else "idle"
    lines.append(f"voice        {_fence_safe(vc.channel.name)} · {state}")
    lines.append(f"voice ws     {_voice_latency_text(vc)}")
    # connect/speak: the two whose absence looks like a playback bug.
    perms = vc.channel.permissions_for(guild.me)
    lines.append(
        f"perms        connect {'✅' if perms.connect else '⚠️'} · "
        f"speak {'✅' if perms.speak else '⚠️'}"
    )
    return lines


# ── Dependencies: everything in-protocol ──────────────────────────────────────
# The bot cannot read another container's /proc or cgroup, so a dependency's
# metrics exist here only if its wire protocol reports them: Redis over INFO;
# Postgres answers the database's own questions over SQL and nothing about its
# container (see the Prometheus row for that half).


@dataclass(frozen=True, slots=True, kw_only=True)
class RedisSample:
    info: dict[str, Any]
    monotonic: float


async def read_redis_sample(redis: Optional[aioredis.Redis]) -> Optional[RedisSample]:
    """One INFO read, or None when Redis is absent or unreachable. Two of these
    bracketing a window give Redis's CPU%."""
    if redis is None:
        return None
    try:
        info = cast(dict[str, Any], await redis.info())
    except Exception as e:  # noqa: BLE001 — a down dependency is a row, not a failure
        log.warning(f"debug redis INFO failed: {type(e).__name__}: {e}")
        return None
    return RedisSample(info=info, monotonic=time.monotonic())


def redis_lines(
    first: Optional[RedisSample],
    second: Optional[RedisSample],
    *,
    dbsize: Optional[int],
) -> list[str]:
    if second is None:
        return ["unavailable (no INFO — Redis down or not configured)"]
    info = second.info
    lines: list[str] = []
    # No core normalization: Redis is single-threaded, so 100% is one saturated core.
    used_cpu = _redis_cpu(first, second)
    lines.append(
        "cpu          unknown" if used_cpu is None else f"cpu          {used_cpu:.1f}%"
    )
    used = info.get("used_memory")
    maxmemory = info.get("maxmemory") or 0
    frag = info.get("mem_fragmentation_ratio")
    if used is None:
        lines.append("mem          unknown")
    else:
        # Against maxmemory, the threshold volatile-lru acts at: the eviction runway.
        ceiling = (
            f" / {_mb(maxmemory)} ({used / maxmemory * 100:.0f}%)"
            if maxmemory
            else " (no maxmemory)"
        )
        frag_text = f" · frag {frag:.2f}" if isinstance(frag, (int, float)) else ""
        lines.append(f"mem          {_mb(used)}{ceiling}{frag_text}")
    expires = sum(
        db.get("expires", 0)
        for key, db in info.items()
        if key.startswith("db") and isinstance(db, dict)
    )
    if dbsize is None:
        lines.append("keys         unknown")
    else:
        # The persistent population is what golden rule 12 protects.
        lines.append(
            f"keys         {dbsize} total · {max(0, dbsize - expires)} persistent"
        )
    hits = info.get("keyspace_hits", 0)
    misses = info.get("keyspace_misses", 0)
    rate = f"{hits / (hits + misses) * 100:.0f}%" if hits + misses else "n/a"
    lines.append(
        f"clients      {info.get('connected_clients', '?')} · "
        f"{info.get('instantaneous_ops_per_sec', '?')} ops/sec"
    )
    lines.append(
        f"cache        hit rate {rate} · evicted {info.get('evicted_keys', '?')}"
    )
    return lines


def _redis_cpu(
    first: Optional[RedisSample], second: Optional[RedisSample]
) -> Optional[float]:
    if first is None or second is None:
        return None
    wall = second.monotonic - first.monotonic
    if wall <= 0:
        return None

    def total(sample: RedisSample) -> Optional[float]:
        sys_cpu = sample.info.get("used_cpu_sys")
        user_cpu = sample.info.get("used_cpu_user")
        if not isinstance(sys_cpu, (int, float)) or not isinstance(
            user_cpu, (int, float)
        ):
            return None
        return float(sys_cpu) + float(user_cpu)

    before, after = total(first), total(second)
    if before is None or after is None:
        return None
    return max(0.0, (after - before) / wall * 100)


# ── Container metrics, from the observability plane ───────────────────────────
# Stock Postgres exposes no OS metrics over SQL and its container's cgroup is
# invisible here, so cpu/mem come from the metrics stack. Only that pair: the
# direct SQL read proves THIS bot's pool reaches the database, is exact rather
# than scrape-interval stale, and survives an LGTM outage.

# The compose container_name, promoted to a Prometheus label from docker_stats'
# container.name attribute. A rename there silently empties this row.
_POSTGRES_CONTAINER = "discord-postgres"
# The OTel docker_stats receiver's names as Prometheus renames them — NOT
# cAdvisor's; utilization is already 0-100 despite the `_ratio` suffix.
_CONTAINER_METRICS = (
    "container_cpu_utilization_ratio",
    "container_memory_usage_total_bytes",
    "container_memory_usage_limit_bytes",
)
_PROMETHEUS_TIMEOUT_SECS = 2.0
# An instant query over three series answers in well under a kilobyte.
_PROMETHEUS_MAX_BYTES = 1 << 20

_prometheus_session_cache: Optional[aiohttp.ClientSession] = None


def _prometheus_session() -> aiohttp.ClientSession:
    """The process's Prometheus session, created on first use: a ClientSession
    binds to the running loop, and this module is imported before there is one.
    Closed by close_prometheus_session()."""
    global _prometheus_session_cache
    if _prometheus_session_cache is None or _prometheus_session_cache.closed:
        _prometheus_session_cache = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=_PROMETHEUS_TIMEOUT_SECS)
        )
    return _prometheus_session_cache


async def close_prometheus_session() -> None:
    """Release the shared session. Called from cog_unload; safe to call twice."""
    global _prometheus_session_cache
    session, _prometheus_session_cache = _prometheus_session_cache, None
    if session is not None and not session.closed:
        await session.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class ContainerMetrics:
    cpu_percent: Optional[float]
    used_bytes: Optional[int]
    limit_bytes: Optional[int]


async def read_container_metrics(
    base_url: Optional[str], container: str
) -> Optional[ContainerMetrics]:
    """CPU and memory for one container, or None when there is no source. Absent
    series are tolerated: an archive-disabled deployment runs no postgres
    container, so "no data" is the ordinary answer."""
    if not base_url:
        return None
    selector = (
        f'{{__name__=~"{"|".join(_CONTAINER_METRICS)}",container_name="{container}"}}'
    )
    try:
        session = _prometheus_session()
        async with session.get(
            f"{base_url.rstrip('/')}/api/v1/query", params={"query": selector}
        ) as response:
            response.raise_for_status()
            # ClientTimeout bounds TIME, not BYTES; the default target is a
            # loopback port on a host-networked bot, so anything that can bind
            # :9090 could otherwise feed .json() gigabytes.
            body = await response.content.read(_PROMETHEUS_MAX_BYTES + 1)
            if len(body) > _PROMETHEUS_MAX_BYTES:
                raise ValueError(f"response exceeded {_PROMETHEUS_MAX_BYTES} bytes")
            payload = orjson.loads(body)
    except Exception as e:  # noqa: BLE001 — an absent metrics stack is a row
        log.warning(f"debug prometheus query failed: {type(e).__name__}: {e}")
        return None
    values: dict[str, float] = {}
    for series in payload.get("data", {}).get("result", []):
        name = series.get("metric", {}).get("__name__")
        try:
            values[name] = float(series["value"][1])
        except KeyError, IndexError, TypeError, ValueError:
            continue
    if not values:
        return None
    used = values.get("container_memory_usage_total_bytes")
    limit = values.get("container_memory_usage_limit_bytes")
    return ContainerMetrics(
        cpu_percent=values.get("container_cpu_utilization_ratio"),
        used_bytes=int(used) if used is not None else None,
        limit_bytes=int(limit) if limit is not None else None,
    )


def _container_lines(metrics: Optional[ContainerMetrics]) -> list[str]:
    """The cpu/mem pair, CPU first, or one labeled absence."""
    if metrics is None:
        return [
            "cpu          n/a (no metrics source)",
            "mem          n/a (no metrics source)",
        ]
    cpu = "unknown" if metrics.cpu_percent is None else f"{metrics.cpu_percent:.1f}%"
    if metrics.used_bytes is None:
        mem = "unknown"
    else:
        pct = (
            f" ({metrics.used_bytes / metrics.limit_bytes * 100:.0f}%)"
            if metrics.limit_bytes
            else ""
        )
        limit = f" / {_mb(metrics.limit_bytes)}" if metrics.limit_bytes else ""
        mem = f"{_mb(metrics.used_bytes)}{limit}{pct}"
    return [
        f"cpu          {cpu} — prometheus",
        f"mem          {mem} — prometheus",
    ]


def _rate(value: float) -> str:
    """One decimal below 10, integer above, decided off the ROUNDED value so
    9.99 renders "10", never "10.0"; an exact zero stays a bare 0."""
    if value == 0:
        return "0"
    text = f"{value:.1f}"
    return text if float(text) < 10 else f"{value:.0f}"


def _count_rate(value: float) -> str:
    # Units promote off the ROUNDED mantissa, so "1000.0k" can never render.
    if round(value / 1_000, 1) >= 1_000:
        return f"{value / 1_000_000:.1f}M"
    if round(value, 1) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return _rate(value)


def _bytes_rate(value: float) -> str:
    # Same rounded-mantissa promotion ("1024 KB/s" never renders).
    if round(value / 1024) >= 1024:
        return f"{value / 1_048_576:.1f} MB/s"
    if round(value) >= 1024:
        return f"{value / 1024:.0f} KB/s"
    return f"{value:.0f} B/s"


def _wait_parenthetical(stats: ArchiveStats) -> str:
    """Only nonzero wait kinds; `(0 waiting)` when every active backend is on-CPU;
    nothing at 0 active. On-CPU is implied (active − waits), never printed."""
    if stats.active_backends == 0:
        return ""
    nonzero = [
        (count, kind)
        for count, kind in (
            (stats.active_io_wait, "io"),
            (stats.active_lock_wait, "lock"),
            (stats.active_other_wait, "other"),
        )
        if count > 0
    ]
    if not nonzero:
        return " (0 waiting)"
    if len(nonzero) == 1:
        count, kind = nonzero[0]
        return f" ({count} {kind}-wait)"
    return " (" + " · ".join(f"{count} {kind}" for count, kind in nonzero) + ")"


def postgres_lines(
    container: Optional[ContainerMetrics],
    before: Optional[ArchiveStats],
    after: ArchiveStats,
) -> list[str]:
    """Pure renderer over the two-sample bracket _postgres_probe takes. Instant
    fields come from `after` alone; the rates are windowed deltas clamped at 0 so
    a pg_stat_reset() between samples cannot render negative. The native rows are
    not a fallback for the Prometheus pair — they measure demand, not
    utilization — so they render whenever the SQL answers."""
    window = (after.monotonic - before.monotonic) if before is not None else 0.0
    if before is not None and window > 0:
        # busy averages when work was REPORTED (stats flush at transaction end),
        # so a spike above the active count is a long statement finishing inside
        # the window. Rendered uncapped.
        busy = max(0.0, after.active_time_ms - before.active_time_ms) / 1000 / window
        busy_text = f"busy {busy:.1f} bk-s/s"
        xacts = max(0, after.xacts_total - before.xacts_total) / window
        tuples = max(0, after.tuples_total - before.tuples_total) / window
        throughput = f"{_rate(xacts)} tx/s · {_count_rate(tuples)} tuples/s"
        delta_hit = max(0, after.blks_hit - before.blks_hit)
        delta_read = max(0, after.blks_read - before.blks_read)
        # A zero denominator is the common case: an idle database touched no blocks.
        window_hit = (
            f"window hit {delta_hit / (delta_hit + delta_read) * 100:.1f}%"
            if delta_hit + delta_read > 0
            else "window hit n/a (idle)"
        )
        spill = _bytes_rate(max(0, after.temp_bytes - before.temp_bytes) / window)
        deadlocks = max(0, after.deadlocks - before.deadlocks)
        mem_signal = f"{window_hit} · spill {spill} · deadlocks {deadlocks}"
    else:
        busy_text = "busy unknown"
        throughput = "unknown"
        mem_signal = "unknown"
    return _container_lines(container) + [
        f"load         {after.active_backends} active"
        f"{_wait_parenthetical(after)} · {busy_text}",
        f"throughput   {throughput}",
        f"mem signal   {mem_signal}",
        f"storage      db {_mb(after.database_bytes)} · "
        f"play_history {_mb(after.table_bytes)}",
        # Estimates, not COUNT(*): play_history is unbounded. A non-zero rejected
        # count means __post_init__'s clamp regressed — `just db-rejects`.
        f"rows         ~{after.rows_estimate} plays · "
        f"{after.rejected_estimate} rejected",
        f"conns        {after.connections} / {after.max_connections}",
        # (lifetime), since an unlabeled cumulative ratio beside the windowed one
        # above invites misreading.
        f"cache        buffers {after.shared_buffers} · "
        f"hit rate {after.cache_hit_ratio * 100:.1f}% (lifetime)",
    ]


# ── Deployment invariants, as live assertions ─────────────────────────────────


def _check(ok: Optional[bool], label: str, detail: str) -> str:
    mark = "✅" if ok else ("⚠️" if ok is False else "❔")
    return f"{mark} {label:<14}{detail}"


async def _none() -> None:
    """An awaitable None, so the gather below stays one shape with no store."""
    return None


async def checks_lines(
    sample: Optional[RedisSample],
    *,
    redis: Optional[aioredis.Redis],
    store: Optional[GuildRedisStore],
    archive_enabled: bool,
    default_password: Optional[bool],
) -> list[str]:
    lines: list[str] = []
    info = sample.info if sample is not None else {}
    if sample is None:
        lines.append(_check(None, "redis", "no INFO — cannot verify"))
    else:
        policy = info.get("maxmemory_policy")
        lines.append(
            _check(
                policy == "volatile-lru",
                "eviction",
                f"maxmemory-policy {policy} (must be volatile-lru)"
                if policy != "volatile-lru"
                else "volatile-lru",
            )
        )
        # Redis keeps serving READS after a failed bgsave while refusing writes,
        # so this goes red BEFORE state writes start failing.
        rdb = info.get("rdb_last_bgsave_status", "unknown")
        aof = info.get("aof_last_write_status", "unknown")
        healthy = rdb == "ok" and aof in ("ok", "unknown")
        lines.append(_check(healthy, "persistence", f"bgsave {rdb} · aof {aof}"))
    # Two independent round trips, so they ride together.
    outbox_line, ttl = await asyncio.gather(
        _outbox_check(redis, archive_enabled=archive_enabled),
        store.history_ttl() if store is not None else _none(),
    )
    lines.append(outbox_line)
    if store is not None:
        # A @_guild_op read, so None rather than raising. -1 is "no expiry" (the
        # PERSIST invariant); -2 is "no such key", a guild that has played nothing.
        if ttl is None:
            lines.append(_check(None, "history ttl", "unknown (Redis unreachable)"))
        elif ttl == -2:
            lines.append(_check(True, "history ttl", "no history yet"))
        else:
            lines.append(
                _check(
                    ttl == -1,
                    "history ttl",
                    "persistent (no expiry)" if ttl == -1 else f"expires in {ttl}s",
                )
            )
    if default_password:
        # Detail-free: DEFAULT_POSTGRES_PASSWORD is a literal in this public
        # repo, and this block is readable by anyone in the channel. -ping
        # carries the full wording for the operator.
        lines.append(_check(False, "db password", "see -ping"))
    elif default_password is False and archive_enabled:
        lines.append(_check(True, "db password", "not the default"))
    return lines


async def _outbox_check(
    redis: Optional[aioredis.Redis], *, archive_enabled: bool
) -> str:
    if redis is None:
        return _check(None, "outbox", "unknown (no Redis)")
    try:
        # The outbox helpers RAISE (golden rule 5's split), so every other
        # consumer owns its own try/except.
        depth = await outbox_depth(redis)
    except Exception as e:  # noqa: BLE001 — a debug row, not the drain path
        return _check(None, "outbox", f"unknown ({type(e).__name__})")
    if archive_enabled:
        return _check(True, "outbox", f"{depth} buffered, draining")
    # Mirrors _warn_if_outbox_left_over: with the archive off nothing drains
    # these, and they sit in a non-evictable key.
    return _check(
        depth == 0,
        "outbox",
        "empty" if depth == 0 else f"{depth} stranded (archive disabled)",
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 · RENDERING
# ════════════════════════════════════════════════════════════════════════════


def _codeblock_fields(name: str, lines: list[str]) -> list[tuple[str, str]]:
    """Lines as one or more codeblock fields within Discord's 1024-char field
    cap. Splits rather than truncates: a silently clipped config listing reads
    as a complete one."""
    fence = 8  # "```\n" + "\n```"
    fields: list[tuple[str, str]] = []
    chunk: list[str] = []
    size = 0
    for line in lines:
        line = truncate(line, _FIELD_LIMIT - fence)
        if chunk and size + len(line) + 1 + fence > _FIELD_LIMIT:
            fields.append((name if not fields else f"{name} (cont.)", _fence(chunk)))
            chunk, size = [], 0
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        fields.append((name if not fields else f"{name} (cont.)", _fence(chunk)))
    return fields


def _fence(lines: list[str]) -> str:
    return "```\n" + "\n".join(lines) + "\n```"


def mode_source(overridden: bool, *, persisted: bool = True) -> str:
    """Why debug mode is in its current state, rendered inside "Debug mode is
    **on** for this server (...)". "saved here" is a stored choice; "host default"
    means the guild never chose and follows DEBUG_MODE; "this session only" is a
    toggle whose Redis write failed, which the toggle already reported."""
    if not overridden:
        return "host default"
    return "saved here" if persisted else "this session only"


async def _runtime_blocks(tasks: int) -> dict[str, list[str]]:
    """Runtime alone, touching NO IO: everything it reads (os.times, /proc,
    pool_state) is local, so this block renders while every dependency is down.
    cpu_before/cpu_after bracket one window that IS the loop-lag measurement;
    the Redis probe runs its own window concurrently."""
    # Accepted skew: on the `process` scope os.times() counts REAPED children,
    # and the build probe may reap `git rev-parse` inside this window (~0.2-0.5%
    # on 12 cores, and only where GIT_SHA is not baked in).
    cpu_before = read_cpu_sample()
    lag_ms = await measure_loop_lag(_CPU_WINDOW_SECS)
    cpu_after = read_cpu_sample()
    return {
        "Runtime": _safe_block(
            "runtime",
            lambda: runtime_lines(
                cpu=cpu_percent(cpu_before, cpu_after),
                cpu_scope=cpu_after.scope,
                cpu_total=cpu_after.seconds,
                cores=cpu_after.cores,
                memory=read_memory(),
                lag_ms=lag_ms,
                tasks=tasks,
            ),
        ),
    }


async def _redis_blocks(inputs: DebugInputs) -> dict[str, list[str]]:
    """Redis and Checks from one probe: Checks reads the same post-window
    snapshot the Redis rates are computed from, so the two blocks cannot
    disagree about the same instant. Bounded explicitly — the pool sets no
    socket_timeout — and failing here costs these two blocks and nothing else."""
    async with asyncio.timeout(_REDIS_PROBE_TIMEOUT_SECS):
        redis_before = await read_redis_sample(inputs.redis)
        await asyncio.sleep(_CPU_WINDOW_SECS)
        # DBSIZE has no ordering relationship with the second INFO.
        redis_after, dbsize = await asyncio.gather(
            read_redis_sample(inputs.redis), _dbsize(inputs.redis)
        )
    return {
        "Redis": _safe_block(
            "redis", lambda: redis_lines(redis_before, redis_after, dbsize=dbsize)
        ),
        "Checks": await _safe_block_async(
            "checks",
            lambda: checks_lines(
                redis_after,
                redis=inputs.redis,
                store=inputs.store,
                archive_enabled=inputs.archive_enabled,
                default_password=inputs.default_password,
            ),
        ),
    }


async def _postgres_probe(inputs: DebugInputs) -> list[str]:
    """Two stats() samples bracketing _PG_WINDOW_SECS, so the counter rows render
    windowed rates. The failure paths are asymmetric: a failed first sample costs
    only the rates (`unknown`); the second is the sole source of the instant
    fields, so its failure is the degraded row."""
    archive = inputs.archive
    if archive is None:
        # Mirrors -ping's OFF row: a missing archive reads as the choice it is.
        return ["off (archive disabled)" if not inputs.archive_enabled else "n/a"]
    async with asyncio.timeout(_PG_PROBE_TIMEOUT_SECS):
        try:
            before = await archive.stats()
        except Exception as e:  # noqa: BLE001 — degrade rates, keep the block
            log.warning(f"first postgres sample failed: {type(e).__name__}: {e}")
            before = None
        # A dead first sample leaves nothing to rate over, so skip the window.
        if before is not None:
            await asyncio.sleep(_PG_WINDOW_SECS)
        # Concurrent: awaiting Prometheus first would spend up to its timeout of
        # this probe's budget before the query carrying the block's content starts.
        container, after = await asyncio.gather(
            read_container_metrics(inputs.prometheus_url, _POSTGRES_CONTAINER),
            archive.stats(),
        )
    return postgres_lines(container, before, after)


async def _postgres_blocks(inputs: DebugInputs) -> dict[str, list[str]]:
    return {
        "Postgres": await _safe_block_async("postgres", lambda: _postgres_probe(inputs))
    }


async def _build_blocks() -> dict[str, list[str]]:
    """git_sha() shells out, so it goes to the default executor like
    collect_versions()'s ffmpeg probe: a subprocess on the event loop stalls
    voice heartbeats. Not cancellable — a thread already started runs to
    _GIT_PROBE_TIMEOUT_SECS after the deadline gave up — which is bounded, once
    per process, and contends with nothing (the yt-dlp pool has its own)."""
    loop = asyncio.get_running_loop()
    sha = await loop.run_in_executor(None, git_sha)
    return {"Build": _safe_block("build", lambda: build_lines(sha))}


def instant_blocks(
    ctx: commands.Context, inputs: DebugInputs, *, source: str
) -> dict[str, list[str]]:
    """Everything answerable without IO, so it is on the skeleton send."""
    blocks: dict[str, list[str]] = {}
    guild = ctx.guild
    if inputs.operator:
        blocks["Config"] = _safe_block("config", config_lines)
        blocks["Discord"] = _safe_block(
            "discord", lambda: discord_lines(ctx.bot, players=inputs.players)
        )
    if guild is not None:
        blocks["This server"] = _safe_block(
            "guild", lambda: guild_lines(guild, inputs, source=source)
        )
    return blocks


def render_snapshot_embed(
    ctx: commands.Context,
    inputs: DebugInputs,
    *,
    blocks: dict[str, list[str]],
    source: str,
) -> discord.Embed:
    """One embed from whatever has been collected so far. Pure and cheap: the
    live loop calls it every tick and discards the result unless it differs."""
    embed = discord.Embed(
        title="\U0001f41e Debug snapshot",
        description=_snapshot_description(inputs, ctx.guild is not None, source),
        color=_DEBUG_COLOR,
    )
    for name in _BLOCK_ORDER:
        lines = blocks.get(name)
        if lines is None:
            continue
        for field_name, value in _codeblock_fields(name, lines):
            embed.add_field(name=field_name, value=value, inline=False)
    # Published to everyone while the same value is an operator-gated row above;
    # -ping prints this identical footer to every caller, so gating it hides nothing.
    footer = f"environment: {config.ENVIRONMENT}"
    if (tf := trace_footer(trace.get_current_span())) is not None:
        footer += f" · {tf}"
    embed.set_footer(text=join_footer(footer, inputs.debug_suffix or ""))
    return embed


def _snapshot_description(inputs: DebugInputs, in_guild: bool, source: str) -> str:
    scope = "this server" if in_guild else "direct messages"
    text = (
        f"Debug mode is **{'on' if inputs.debug_enabled else 'off'}** for "
        f"{scope} ({source})."
    )
    return text if inputs.operator else f"{text}\n\n{_OPERATOR_NOTICE}"


async def run_debug_dashboard(ctx: commands.Context, inputs: DebugInputs) -> None:
    """The `-debug` snapshot, sent immediately and filled in as its IO lands —
    -ping's shape: skeleton now, edits as blocks land, a deadline that marks
    stragglers rather than failing the card. A non-operator has no deferred
    blocks, so the driver degrades to a single send with no loop."""
    span = trace.get_current_span()
    source = mode_source(inputs.debug_overridden, persisted=inputs.debug_persisted)
    blocks = instant_blocks(ctx, inputs, source=source)

    # Counted before the driver creates its probe tasks — see runtime_lines.
    loop_tasks = len(asyncio.all_tasks())

    probes: dict[str, Callable[[], Coroutine[Any, Any, dict[str, list[str]]]]] = {}
    if inputs.operator:
        probes = {
            "runtime": lambda: _runtime_blocks(loop_tasks),
            "redis": lambda: _redis_blocks(inputs),
            "postgres": lambda: _postgres_blocks(inputs),
            "build": _build_blocks,
        }
        for names in _PROBE_BLOCKS.values():
            for name in names:
                blocks[name] = list(_PENDING_LINES)

    async def _prepare() -> None:
        """Versions is the one public block that needs a (cached, executor-
        hopped) call, so it rides the pre-send step as -ping's does."""
        versions = await collect_versions()
        blocks["Versions"] = _safe_block("versions", lambda: version_lines(versions))

    def _settle(key: str, outcome: dict[str, list[str]] | Exception) -> None:
        if isinstance(outcome, Exception):
            # _safe_block guards each collector, so reaching here means the probe
            # itself broke. Fail only its own blocks.
            e = outcome
            log.warning(f"debug probe {key!r} failed: {type(e).__name__}: {e}")
            for name in _PROBE_BLOCKS[key]:
                blocks[name] = [f"unavailable ({type(e).__name__})"]
            return
        blocks.update(outcome)

    def _abandon(key: str) -> None:
        for name in _PROBE_BLOCKS[key]:
            blocks[name] = list(_TIMEOUT_LINES)

    await run_live_dashboard(
        ctx,
        probes=probes,
        settle=_settle,
        abandon=_abandon,
        render=lambda: [
            render_snapshot_embed(ctx, inputs, blocks=blocks, source=source)
        ],
        prepare=_prepare,
        tick_secs=DEBUG_TICK_SECS,
        deadline_secs=DEBUG_DEADLINE_SECS,
    )

    span.set_attribute("debug.enabled", inputs.debug_enabled)
    span.set_attribute("debug.source", source)
    span.set_attribute("debug.players", inputs.players)
    span.set_attribute("debug.operator", inputs.operator)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 · PER-GUILD DEBUG SETTINGS — the mutable half of debug mode
# ════════════════════════════════════════════════════════════════════════════
# Everything above renders debug mode; this decides whether it is ON. The cog
# holds one instance, which MusicContext.send and the NP render read directly.


class DebugSettings:
    """Per-guild debug-mode state: the durable choice, its read cache, and the
    sampler feeding the footer. DEBUG_MODE is the default every guild starts
    from; the durable copy lives in guild:{id}:config and this holds the read
    cache. One instance per cog, built in MusicBot.__init__."""

    __slots__ = (
        "_default",
        "_overrides",
        "_toggle_seq",
        "_toggled_at",
        "_unpersisted",
        "_sampler",
    )

    def __init__(self) -> None:
        # Read ONCE, here, so a garbage value aborts startup inside load_extension.
        self._default: bool = debug_mode_default()
        # A guild that never chose is ABSENT and follows _default, including when
        # the env var later changes — absence is not False.
        self._overrides: dict[int, bool] = {}
        # A monotonic stamp bumped by every toggle, and its value at each guild's
        # last toggle: hydrate() applies a Redis read across an await and uses
        # these to leave a guild toggled in that window alone. _unpersisted holds
        # the guilds whose cached value did NOT reach Redis.
        self._toggle_seq: int = 0
        self._toggled_at: dict[int, int] = {}
        self._unpersisted: set[int] = set()
        self._sampler = RuntimeSampler()
        if self._default and config.ENVIRONMENT == "production":
            # Observation-only, so an advisory rather than a refusal.
            log.warning(
                "DEBUG_MODE is on in production: every command response will "
                "carry a debug footer (trace id, timings, runtime metrics). "
                "Nothing about playback changes. Unset DEBUG_MODE to turn it off."
            )

    # ── Queries ───────────────────────────────────────────────────────────────

    def enabled(self, guild_id: Optional[int]) -> bool:
        """Whether debug mode is on for this guild. A guild with no stored choice
        (and every DM) follows the host default. Synchronous and in-memory:
        MusicContext.send calls this on every reply."""
        if guild_id is None:
            return self._default
        return self._overrides.get(guild_id, self._default)

    @property
    def default(self) -> bool:
        """The host's DEBUG_MODE, which a guild with no stored choice follows."""
        return self._default

    @property
    def snapshot(self) -> Optional[RuntimeSnapshot]:
        """The rolling runtime metrics the footer prints, or None before the
        sampler's first tick. Read by MusicContext.send."""
        return self._sampler.snapshot

    def has_override(self, guild_id: Optional[int]) -> bool:
        """Has this guild made an explicit choice, rather than following the host?"""
        return guild_id is not None and guild_id in self._overrides

    def is_persisted(self, guild_id: Optional[int]) -> bool:
        """Did this guild's choice reach Redis? True for a guild that never chose."""
        return guild_id not in self._unpersisted

    def footer(
        self, guild: Optional[discord.Guild], *, host_metrics: bool = True
    ) -> Optional[str]:
        """Debug mode's footer for the two live dashboards, which neither
        decoration seam reaches. Rendered once per invocation with no elapsed-ms,
        since every segment must be constant across the loop. `host_metrics=False`
        drops the runtime segment for a non-owner. None while debug mode is off."""
        if not self.enabled(guild.id if guild else None):
            return None
        return (
            debug_footer(
                shard_id=guild.shard_id if guild else None,
                runtime=self.snapshot if host_metrics else None,
                # Both cards print `environment:` and `trace:` in their own footer.
                # The trace flag is inert while no span is passed; kept so adding
                # one cannot double it.
                skip_environment=True,
                skip_trace=True,
            )
            or None
        )

    def decorate(
        self,
        embeds: Sequence[discord.Embed],
        guild: Optional[discord.Guild],
        *,
        span: Optional[trace.Span] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """Put debug mode's footer on `embeds`, or take a stale one off — the
        single entry point for every embed the bot sends. Off is a STRIP, not a
        skip: `play_message` and a host's cached embeds outlive a mid-song
        `--disable`. See docs/ARCHITECTURE.md#debug-footer-seams."""
        if not self.enabled(guild.id if guild else None):
            strip_debug_footers(embeds)
            return
        decorate_embeds(
            embeds,
            span=span,
            elapsed_ms=elapsed_ms,
            shard_id=guild.shard_id if guild else None,
            runtime=self.snapshot,
        )

    # ── Mutations ─────────────────────────────────────────────────────────────

    async def hydrate(
        self, redis: Optional[aioredis.Redis], guilds: Sequence[discord.Guild]
    ) -> None:
        """Hydrate the in-memory cache from each guild's stored config. Runs at
        cog_load and again on every on_ready, so it must be safe to replay. Two
        skip rules: a guild whose config could not be READ is skipped
        (read_guild_configs omits it, so "Redis blinked" cannot revert a stored
        choice to the host default), and a guild toggled while this pass was
        reading is skipped (the read and the apply straddle an await)."""
        if redis is None:
            return
        guilds = list(guilds)
        if not guilds:
            return
        started = self._toggle_seq
        configs = await read_guild_configs(redis, [g.id for g in guilds])
        for guild in guilds:
            config = configs.get(guild.id)
            if config is None or self._toggled_at.get(guild.id, 0) > started:
                continue
            if config.debug_mode is None:
                # No stored choice: follow the host default, uncached, or a later
                # env change would not reach this guild.
                self._overrides.pop(guild.id, None)
            else:
                self._overrides[guild.id] = config.debug_mode
                self._unpersisted.discard(guild.id)
        self.sync_sampler()

    async def toggle(
        self, redis: Optional[aioredis.Redis], guild_id: int, enabled: bool
    ) -> bool:
        """Apply an explicit choice for one guild. Returns whether it reached
        Redis; the caller reports that."""
        # Redis FIRST, cache second: a failed write must not leave the cache
        # claiming a setting the next on_ready would silently undo.
        persisted = False
        if redis is not None:
            persisted = await GuildRedisStore(redis, guild_id).set_debug_mode(enabled)
        self._overrides[guild_id] = enabled
        # Stamp this guild so a hydration pass that read BEFORE this write cannot
        # apply its older value on top of it.
        self._toggle_seq += 1
        self._toggled_at[guild_id] = self._toggle_seq
        if persisted:
            self._unpersisted.discard(guild_id)
        else:
            self._unpersisted.add(guild_id)
        # The sampler runs only while some guild wants it.
        self.sync_sampler()
        log.info(
            f"debug mode {'enabled' if enabled else 'disabled'} by command",
            persisted=persisted,
        )
        return persisted

    async def forget(self, redis: Optional[aioredis.Redis], guild_id: int) -> None:
        """Drop a departed guild's override, cache and durable copy alike, or a
        guild that removes and re-adds the bot silently resumes a setting nobody
        there chose — and the config key carries no TTL, so it would sit in
        Redis forever."""
        self._toggled_at.pop(guild_id, None)
        self._unpersisted.discard(guild_id)
        if self._overrides.pop(guild_id, None) is not None:
            self.sync_sampler()
        if redis is not None:
            await GuildRedisStore(redis, guild_id).clear_config()

    # ── Sampler lifecycle ─────────────────────────────────────────────────────

    def sync_sampler(self) -> None:
        """Run the sampler exactly while some guild is effectively debug-enabled.
        Public because cog_load calls it before any toggle has happened."""
        self._sampler.apply(wanted=self._default or any(self._overrides.values()))

    async def aclose(self) -> None:
        """Stop the sampler, unconditionally."""
        await self._sampler.aclose()
