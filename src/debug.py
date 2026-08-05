"""Debug mode: the `-debug` diagnostic snapshot and the footer behind it.

OBSERVATION-ONLY, and that is the whole design constraint. Nothing here changes
playback, caching, queueing or persistence — only what the bot shows. It is what
keeps "test with debug on, ship with debug off" a valid methodology: nothing you
validated changes when the toggle flips.

Every collector degrades to a labeled `unknown`/`n/a` rather than raising (see
_safe_block) — a debug tool that crashes is worse than no debug tool. src/musicbot.py
owns only the command registration and the override state; parsing, collection and
rendering live here, the same split -ping uses.
"""

import asyncio
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import TYPE_CHECKING, Any, Optional

import discord
from discord.ext import commands
from opentelemetry import trace

from enum import Enum

from src import config
from src.config import DEBUG_DEADLINE_SECS, DEBUG_TICK_SECS, ENVIRONMENT
from src.dashboard import run_live_dashboard
from src.ping import bot_version, collect_versions
from src.redis_client import GuildRedisStore
from src.util import (
    cancel_task,
    get_logger,
    trace_footer,
    trace_id_of,
    truncate,
)

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.musicplayer import MusicPlayer
    from src.ytdlp_pool import PoolState

log = get_logger(__name__)

# Discord's hard cap on an embed's footer text.
FOOTER_LIMIT = 2048


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
    """The action `arg` names, or None when it names none.

    Hand-parsed rather than a FlagConverter: these are valueless switches, which the
    converter's `--flag value` grammar cannot express, and a missing-dashes `-debug
    enable` gets a helpful answer here (see unknown_arg_message) instead of a FlagError
    raised before the command body ever runs.
    """
    return _ACTIONS.get(arg.strip().lower())


def unknown_arg_message(arg: str) -> str:
    """The reply for an unparseable argument.

    Deliberately does not echo the argument back: rendering user text into an embed
    the bot sends is a mention-injection surface, and the did-you-mean branch already
    covers the realistic typo.
    """
    cleaned = arg.strip().lower()
    if f"--{cleaned}" in _ACTIONS:
        return f"Did you mean `-debug --{cleaned}`? Options take two dashes."
    return f"Unknown option. Usage: {DEBUG_USAGE}"


def mode_source(overridden: bool, *, persisted: bool = True) -> str:
    """Why debug mode is in its current state — the half of the answer a bare on/off
    does not give. Renders inside "Debug mode is **on** for this server (...)", so
    it stays short enough not to repeat the sentence around it.

    Three states, not two. "saved here" is a stored choice that outlives restarts;
    "host default" means this guild has never set one and follows DEBUG_MODE, so
    changing that variable moves it and setting it here would pin it. "this session
    only" is a toggle whose Redis write failed: the toggle already warned the user,
    and reporting it as "saved here" one message later would make the command whose
    job is to describe reality contradict the one that just changed it.
    """
    if not overridden:
        return "host default"
    return "saved here" if persisted else "this session only"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 · FOOTER DECORATION
# ════════════════════════════════════════════════════════════════════════════
# What debug mode actually does to ordinary traffic: every command response grows
# a footer identifying the request. The trace id is the point — it is already the
# join key for every log line and span, so pasting one out of Discord finds the
# exact request in Loki/Tempo. -ping is not decorated: it bypasses this seam.

_DEBUG_MARK = "🐞"


def debug_footer(
    *,
    span: Optional[trace.Span] = None,
    elapsed_ms: Optional[float] = None,
    shard_id: Optional[int] = None,
    runtime: Optional["RuntimeSnapshot"] = None,
    skip_trace: bool = False,
) -> str:
    """The debug suffix, or "" when nothing is known worth showing.

    Every part is optional because every part has an absent case: a send outside
    any command has no elapsed time, a DM has no shard, an unsampled span has no
    trace id, and the runtime segment is absent until the sampler's first tick.
    """
    parts: list[str] = []
    if elapsed_ms is not None:
        parts.append(f"{round(elapsed_ms)} ms")
    if shard_id is not None:
        parts.append(f"shard {shard_id}")
    if runtime is not None:
        if runtime.cpu_percent is not None:
            parts.append(f"cpu {runtime.cpu_percent:.0f}%")
        if runtime.mem_percent is not None:
            parts.append(f"mem {runtime.mem_percent:.0f}%")
        parts.append(f"lag {runtime.lag_ms:.1f} ms")
        parts.append(f"tasks {runtime.tasks}")
        parts.append(f"pool {runtime.pool_workers}")
    if not skip_trace and span is not None and (trace_id := trace_id_of(span)):
        parts.append(f"trace {trace_id}")
    if not parts:
        return ""
    return f"{_DEBUG_MARK} " + " · ".join(parts)


def decorate_embeds(
    embeds: Sequence[discord.Embed],
    *,
    span: Optional[trace.Span] = None,
    elapsed_ms: Optional[float] = None,
    shard_id: Optional[int] = None,
    runtime: Optional["RuntimeSnapshot"] = None,
) -> None:
    """Append the debug footer to each embed, IN PLACE.

    Mutating is safe — embeds are freshly constructed per response everywhere in
    this codebase — and it is what lets MusicContext.send decorate both of its send
    paths without either of them reshaping its kwargs.
    """
    for embed in embeds:
        existing = embed.footer.text or ""
        suffix = debug_footer(
            span=span,
            elapsed_ms=elapsed_ms,
            shard_id=shard_id,
            runtime=runtime,
            # Error embeds already carry one from _command_error. The same id twice
            # in one footer reads as two different traces.
            skip_trace="trace:" in existing or "trace " in existing,
        )
        if not suffix:
            continue
        text = f"{existing} · {suffix}" if existing else suffix
        embed.set_footer(
            text=truncate(text, FOOTER_LIMIT), icon_url=embed.footer.icon_url
        )


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


# ── Process & event-loop readers ──────────────────────────────────────────────
# psutil is not cgroup-aware (psutil#2100): inside a container virtual_memory()
# reports the HOST. Honest container numbers mean reading /sys/fs/cgroup by hand
# either way, so there is no dependency to add here — only files to read.

# Module constants, not literals inline: these are the seam tests point at a
# fixture directory, since the real files exist only on Linux.
_CGROUP = Path("/sys/fs/cgroup")
_PROC_STATUS = Path("/proc/self/status")
_PROC_MEMINFO = Path("/proc/meminfo")


def _read_text(path: Path) -> str:
    """A pseudo-file's contents, or "" wherever it does not exist (macOS dev, cgroup
    v1, a restricted container). Never raises: every caller is a debug row."""
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
    """A CUMULATIVE cpu reading plus the clock it was taken at.

    CPU% is a rate and exists only between two of these — psutil's
    cpu_percent(interval=None) returning 0.0 on its first call is the same lesson
    from the other side. `scope` says what was measured, because the two sources
    answer different questions: cgroup covers the whole container (FFmpeg
    subprocesses and pool workers included), os.times() only this process and its
    REAPED children (so FFmpeg lands at song end, workers at shutdown).
    """

    seconds: float
    monotonic: float
    scope: str
    cores: float


def cpu_cores() -> float:
    """Cores this process may actually use: the cgroup quota when one is set, else
    the host count. A container capped at 1.5 cores must not report 12% of 12 while
    it is saturated."""
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
    # Gated on _in_container(): cpu.stat is readable on a bare-metal host too, where
    # /sys/fs/cgroup is the ROOT cgroup and the counter covers EVERY process on the
    # machine — the bot would report the whole host's CPU as its own. A wrong number
    # is worse than a narrower one, so off-container this falls to the process scope.
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
    """Percent of available cores used between two samples, or None when the window
    is too short to divide by."""
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
    """Memory used and the ceiling it counts against, best source first.

    The cgroup pair is the one the OOM killer acts on, which is what makes its
    percentage meaningful rather than decorative.

    Gated on _in_container() for the same reason read_cpu_sample is: off-container
    that path reads the ROOT cgroup, so the row would report the whole machine's
    usage against the machine's total and label it `(container)`. Worse than the CPU
    twin, in fact — this is an absolute number rather than a rate, so nothing about
    it looks wrong. The /proc fallback below is strictly more correct off-container.
    """
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
    # macOS dev: no /proc at all. ru_maxrss is a PEAK, not a current reading, and
    # its units differ by platform — bytes on darwin, KiB on Linux (getrusage(2)).
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
    """How late the loop ran a callback it asked for `delay` seconds out, in ms.

    Doubles as the CPU window when the caller brackets its samples around this —
    the measurement and the window are the same wait, so neither costs the other.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.sleep(delay)
    return max(0.0, (loop.time() - start - delay) * 1000)


def pool_state() -> "PoolState":
    # Resolved per call rather than imported at module scope: the test seam
    # monkeypatches src.youtube.ytdlp_pool, and a captured reference would miss it.
    from src.youtube import ytdlp_pool

    return ytdlp_pool.state


# ── The rolling sampler behind the footer ─────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeSnapshot:
    """The runtime metrics the debug footer prints. Every field is optional because
    the first tick has no previous sample to rate against."""

    cpu_percent: Optional[float]
    mem_percent: Optional[float]
    lag_ms: float
    tasks: int
    pool_workers: int


class RuntimeSampler:
    """A ~5s background sampler feeding the debug footer.

    Sampled in the background rather than at send time because CPU% needs a
    wall-clock window and a response must never wait on one; a send reads the
    cached snapshot, at most one tick old. The tick's own scheduling drift IS the
    loop-lag measurement — a late tick measures exactly what it wanted to report.

    ONE instance, held on the cog. Deliberately not module state: a module global
    outlives a cog reload and would leak the task.
    """

    INTERVAL_SECS = 5.0

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task[None]] = None
        self._snapshot: Optional[RuntimeSnapshot] = None
        self._cpu: Optional[CpuSample] = None

    @property
    def snapshot(self) -> Optional[RuntimeSnapshot]:
        """The latest sample, or None before the first tick completes — in which
        case the footer simply omits the runtime segment rather than guessing."""
        return self._snapshot

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def apply(self, *, wanted: bool) -> None:
        """Start or stop to match whether ANY guild is effectively debug-enabled.

        Must be called at cog load as well as on every toggle: a DEBUG_MODE=true
        deployment starts already-enabled and no toggle ever fires, so a
        transition-only trigger would leave the footer permanently without runtime
        metrics.
        """
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
        """Stop and await the task. Cog teardown uses this so a reload cannot leave
        a sampler dripping /proc reads forever."""
        task = self._task
        self.stop()
        await cancel_task(task)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            expected = loop.time() + self.INTERVAL_SECS
            await asyncio.sleep(self.INTERVAL_SECS)
            try:
                self._snapshot = self._sample(max(0.0, (loop.time() - expected) * 1000))
            except Exception as e:  # noqa: BLE001 — one bad tick must not end the loop
                log.warning(f"runtime sample failed: {type(e).__name__}: {e}")

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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 · CONFIG ALLOWLIST
# ════════════════════════════════════════════════════════════════════════════
# An env var absent from _CONFIG_ALLOWLIST does not render at all, so a knob added
# later opts in at review time and a future secret can never leak by default.
# Django's debug-page scrub convention, decided by hand instead of by pattern-match.


class _ConfigKind(Enum):
    VALUE = "value"  # rendered as configured
    SECRET = "secret"  # presence only — the value never renders
    URL = "url"  # userinfo and credential-bearing query params stripped


@dataclass(frozen=True, slots=True, kw_only=True)
class _ConfigVar:
    name: str
    kind: _ConfigKind
    # What the bot uses when the variable is unset, as text. Read from config's own
    # resolved constants where there is one, so this can't drift from the real default.
    fallback: Optional[str] = None


_CONFIG_ALLOWLIST: tuple[_ConfigVar, ...] = (
    _ConfigVar(name="ENVIRONMENT", kind=_ConfigKind.VALUE, fallback=ENVIRONMENT),
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
        name="NOW_PLAYING_UPDATE_INTERVAL_SECS",
        kind=_ConfigKind.VALUE,
        fallback=str(config.NOW_PLAYING_UPDATE_INTERVAL_SECS),
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
    # Credential-bearing: presence only, never the value. POSTGRES_URL is here rather
    # than under URL because it embeds the password in its userinfo.
    _ConfigVar(name="DISCORD_TOKEN", kind=_ConfigKind.SECRET),
    _ConfigVar(name="SPOTIFY_CLIENT_ID", kind=_ConfigKind.SECRET),
    _ConfigVar(name="SPOTIFY_CLIENT_SECRET", kind=_ConfigKind.SECRET),
    _ConfigVar(name="POSTGRES_URL", kind=_ConfigKind.SECRET),
)

# Substrings marking a query parameter whose value is a credential. Matched against
# the lowercased key, so `api_key` and `X-Auth-Token` are both caught.
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
    """A URL safe to render: userinfo replaced with `***`, credential-bearing query
    values replaced with `***`. Those are the two places a DSN hides a password —
    asyncpg honours `?password=` as readily as userinfo."""
    # The whole body is guarded, not just urlsplit. `.port` is a LAZY property:
    # urlsplit("redis://h:99999/0") succeeds and the ValueError fires on dereference,
    # so a try around the parse alone never caught the one input that needed it — one
    # typo'd port in .env replaced all of Config with "unavailable (ValueError)",
    # precisely while someone was diagnosing that host.
    try:
        # Normalise a scheme-less value to an authority BEFORE splitting. Without
        # the "//", urlsplit puts the host in .scheme/.path and leaves .netloc
        # empty, so every rewrite below — including hide_host's — lands in a slot
        # that was never carrying the host, and the host survives verbatim next to
        # a `***` that reads as "this was redacted". Operators do write these
        # scheme-less: ping.probe_otel carries the same normalisation and says so.
        parts = urlsplit(raw if "://" in raw else f"//{raw}")
        netloc = parts.netloc
        if parts.username is not None or parts.password is not None:
            host = parts.hostname or ""
            netloc = f"***@{host}:{parts.port}" if parts.port else f"***@{host}"
        if hide_host:
            # The snapshot is posted in the channel the operator typed in, so this
            # block has to survive being read by everyone there. A host:port pair is
            # internal topology even with no credential on it. Redacted
            # UNCONDITIONALLY, including for localhost: redacting only remote hosts
            # would make the presence of `***` the disclosure instead.
            if "://" not in raw:
                # Without a scheme there is no authority to rewrite, and the host
                # can land in .path as well ("http:/host:9090/x"), so replacing one
                # component would leave it visible beside a `***` that claims it was
                # hidden. Nothing in a value this shape is safe to echo.
                return "***"
            netloc = "***"
        query = parts.query
        if query:
            # safe="*" so the redaction marker survives quoting as `***` rather than
            # arriving as %2A%2A%2A, which reads like data rather than a redaction.
            query = urlencode(
                [
                    (k, "***" if _is_credential_key(k) else v)
                    for k, v in parse_qsl(query, keep_blank_values=True)
                ],
                safe="*",
            )
        # A fragment is never meaningful in any allowlisted URL, and urlsplit puts
        # everything after a stray `#` there — so it is the one component that can
        # carry a credential past both redactions above. Marked, not dropped, so
        # the row still says something was there.
        fragment = "***" if parts.fragment else ""
        return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
    except ValueError:
        return "unparseable"


def _is_credential_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _CREDENTIAL_QUERY_KEYS)


def render_config_value(var: _ConfigVar) -> str:
    """One allowlist row's value, redacted per its kind."""
    raw = os.environ.get(var.name)
    if var.kind is _ConfigKind.SECRET:
        # Presence only, always. This block has to stay safe to screenshot into an
        # issue; a value here is one paste away from a leaked credential.
        return "set" if (raw or "").strip() else "unset"
    if raw is None or not raw.strip():
        return "unset" if var.fallback is None else f"{var.fallback} (default)"
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


# Stamped at import: the extension loads within seconds of process start, so this
# is the process's start time for every purpose this command has.
_PROCESS_START = time.time()

_GIT_PROBE_TIMEOUT_SECS = 2.0
_git_sha_cache: Optional[str] = None

# Block order in the embed, and which probe fills each deferred one. The mapping is
# what lets the deadline mark exactly the blocks a straggler owed and no others.
_BLOCK_ORDER = (
    "Build",
    "Versions",
    "Config",
    "Discord",
    "This server",
)
_PROBE_BLOCKS: dict[str, tuple[str, ...]] = {
    "build": ("Build",),
}
_PENDING_LINES = ["⏳ collecting…"]
# Named rather than blank: a block that silently vanished would read as "this host
# has no Build", which is a different and wrong answer.
_TIMEOUT_LINES = ["⚠️ timed out"]

_OPERATOR_NOTICE = (
    "-# Host details (configuration, storage, runtime) are shown to the bot owner "
    "only. Run `-ping` for dependency health."
)

# Discord's hard caps on an embed field value and its footer text.
_FIELD_LIMIT = 1024
FOOTER_LIMIT = 2048

_DEBUG_COLOR = discord.Color(0xE67E22)  # amber: an operator surface, not an alert


@dataclass(frozen=True, slots=True, kw_only=True)
class DebugInputs:
    """What the cog hands the snapshot: state this module cannot reach without
    importing MusicBot, which would be an import cycle. Later blocks add fields with
    defaults, so the cog's call site stays one expression."""

    debug_enabled: bool
    debug_overridden: bool
    # False only when a toggle's Redis write failed, so the snapshot can say
    # "this session only" instead of claiming a durability the user was just
    # warned did not happen. True for a guild that never chose, where the
    # question does not arise.
    debug_persisted: bool = True
    players: int
    player: Optional["MusicPlayer"] = None
    redis: Optional["aioredis.Redis"] = None
    store: Optional[GuildRedisStore] = None
    # Is the caller the bot owner? Gates every block that describes the HOST rather
    # than the caller's own server — see _OPERATOR_BLOCKS. Defaults False so a call
    # site that forgets to ask discloses nothing.
    operator: bool = False
    # None = not asked, and only an operator is ever asked. Set symmetrically with
    # `operator` rather than only when it is True: a row that appears for False and
    # vanishes for True would make its own absence the answer.
    default_password: Optional[bool] = None


def _safe_block(label: str, fn: Callable[[], list[str]]) -> list[str]:
    """Run a collector, or render why it could not run. The degrade principle made
    mechanical: no block may take the embed down with it."""
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


def _fmt_ms(seconds: float) -> str:
    """Latency in ms, or the word for a latency that is not a number yet.

    nan is discord.py's gateway-is-reconnecting value and inf is the voice client's
    before-first-heartbeat value; both would render as garbage arithmetic.
    """
    ms = seconds * 1000
    if math.isnan(ms):
        return "reconnecting"
    if math.isinf(ms):
        return "warming up"
    return f"{round(ms)} ms"


def git_sha() -> str:
    """The commit this build was made from, cached for process lifetime.

    GIT_SHA is baked into the runtime image (Dockerfile ARG→ENV); the git fallback
    covers `just run` from a checkout, where there is no image at all. A dirty
    build's tag is `<sha>-dirty.<digest>` and passes through unchanged, so the bot
    reports exactly the tag that was deployed.
    """
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
    """`sha` is REQUIRED, and resolved by the caller off the event loop.

    Not defaulted to None-then-git_sha(): git_sha() shells out (measured 16ms
    uncached, most of an audio frame's scheduling slack) and a default made that
    blocking call reachable from any future caller that forgot. None here means
    "the caller looked and there is no sha", which renders as unknown.
    """
    return [
        f"version      {bot_version()}",
        f"commit       {sha if sha is not None else 'unknown'}",
        f"environment  {ENVIRONMENT}",
        f"container    {'yes' if _in_container() else 'no'}",
    ]


def version_lines(versions: dict[str, str]) -> list[str]:
    return [
        f"bot          {versions['bot']}",
        f"yt-dlp       {versions['yt-dlp']}",
        f"ffmpeg       {versions['ffmpeg']}",
        f"python       {versions['python']}  ·  discord.py {versions['discord.py']}",
    ]


def _mb(value: float) -> str:
    return f"{value / 1_048_576:.0f} MB"


def discord_lines(bot: commands.Bot, *, players: int) -> list[str]:
    # latencies is AutoShardedClient-only; the single-shard shape is the fallback so
    # this block works against a plain Bot (and the doubles in tests).
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


def _voice_latency_text(vc: discord.VoiceClient) -> str:
    """Voice WS heartbeat latency and its 20-sample average.

    Both are inf (and average_latency raises ZeroDivisionError) until the first ACK,
    ~5s after joining — discord.py #6430. That window is exactly when a tester looks,
    since testing starts with a fresh join.
    """
    try:
        latency = _fmt_ms(vc.latency)
        average = _fmt_ms(vc.average_latency)
    except ZeroDivisionError:
        return "warming up"
    if latency == "warming up":
        return latency
    return f"{latency} (avg {average})"


def _fence_safe(text: str) -> str:
    """User-controlled text, made safe to interpolate into a ``` fence.

    A voice-channel name is the only string in this snapshot a user picks, and the
    "This server" block is public. A name containing a backtick run CLOSES the fence,
    and Discord renders masked links inside embed field values — so `[Verify your
    account](https://evil)` in a channel name becomes a clickable link inside a card
    carrying the bot's own name and avatar. Join-to-create bots hand ordinary members
    naming rights over their channel, so this needs no elevated permission.
    unknown_arg_message already refuses to echo user text for this reason; the rule
    is the same one function over.
    """
    return truncate(text.replace("`", "'"), 60)


def guild_lines(guild: discord.Guild, inputs: DebugInputs, *, source: str) -> list[str]:
    mp = inputs.player
    lines = [
        f"player       {'yes' if mp is not None else 'no'}",
        f"queue        {mp.queue.qsize() if mp is not None else 0} queued",
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
    # connect/speak only: they are the two whose absence looks exactly like a
    # playback bug rather than a permission problem.
    perms = vc.channel.permissions_for(guild.me)
    lines.append(
        f"perms        connect {'✅' if perms.connect else '⚠️'} · "
        f"speak {'✅' if perms.speak else '⚠️'}"
    )
    return lines


def _codeblock_fields(name: str, lines: list[str]) -> list[tuple[str, str]]:
    """Lines as one or more codeblock fields, each within Discord's 1024-char field
    cap. Splits rather than truncates: a silently clipped config listing is worse
    than none, since it reads as a complete one."""
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


async def _build_blocks() -> dict[str, list[str]]:
    """git_sha() shells out to git, so it goes to the default executor — the same
    hop collect_versions() makes for `ffmpeg -version`, and for the same reason:
    a subprocess on the event loop stalls voice heartbeats and every other guild.

    Not cancellable, and that is accepted rather than overlooked: cancelling this
    future does not stop a thread that has already started, so a `git rev-parse`
    blocked on an index.lock holds one default-executor thread for its full
    _GIT_PROBE_TIMEOUT_SECS after the deadline gave up on it. Bounded by that
    timeout, once per process (git_sha caches), and nothing else contends for that
    executor — the yt-dlp pool has its own.
    """
    loop = asyncio.get_running_loop()
    sha = await loop.run_in_executor(None, git_sha)
    return {"Build": _safe_block("build", lambda: build_lines(sha))}


def instant_blocks(
    ctx: commands.Context, inputs: DebugInputs, *, source: str
) -> dict[str, list[str]]:
    """Everything answerable without IO, so it is on the skeleton send.

    Versions is here rather than gated because -ping already publishes the same
    tuple to everyone; gating it would hide nothing.
    """
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
    """One embed from whatever has been collected so far. Pure and cheap: the live
    loop calls it every tick and throws the result away unless it differs."""
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
    # Published to everyone, while the same value is an operator-gated row in Config
    # and Build — and on a `just run` deployment it is the git branch name. Known and
    # inherited rather than decided here: -ping prints this identical footer to every
    # caller, so gating it would hide nothing the sibling command does not disclose.
    footer = f"environment: {ENVIRONMENT}"
    if (tf := trace_footer(trace.get_current_span())) is not None:
        footer += f" \u00b7 {tf}"
    embed.set_footer(text=truncate(footer, FOOTER_LIMIT))
    return embed


def _snapshot_description(inputs: DebugInputs, in_guild: bool, source: str) -> str:
    scope = "this server" if in_guild else "direct messages"
    text = (
        f"Debug mode is **{'on' if inputs.debug_enabled else 'off'}** for "
        f"{scope} ({source})."
    )
    return text if inputs.operator else f"{text}\n\n{_OPERATOR_NOTICE}"


async def run_debug_dashboard(ctx: commands.Context, inputs: DebugInputs) -> None:
    """The `-debug` snapshot, sent immediately and filled in as its IO lands.

    A host block costs a round trip somewhere — here, a git subprocess. Collecting
    it before the first send makes the command look hung on a healthy host and far
    longer on a sick one, which is exactly when it gets run.

    So the shape is -ping's: skeleton now, edits as blocks land, a deadline that
    marks stragglers rather than failing the whole card. That last part matters
    more here than in -ping, because this card grows more blocks and one slow
    dependency must not be able to take all of them down with it.

    A non-operator has no deferred blocks at all (nothing they see needs IO), so
    the driver degrades to a single send with no loop.
    """
    span = trace.get_current_span()
    source = mode_source(inputs.debug_overridden, persisted=inputs.debug_persisted)
    blocks = instant_blocks(ctx, inputs, source=source)

    probes: dict[str, Callable[[], Coroutine[Any, Any, dict[str, list[str]]]]] = {}
    if inputs.operator:
        probes = {"build": _build_blocks}
        for names in _PROBE_BLOCKS.values():
            for name in names:
                blocks[name] = list(_PENDING_LINES)

    async def _prepare() -> None:
        """Versions is the one public block that needs a (cached, executor-hopped)
        call, so it rides the pre-send step exactly as -ping's does."""
        versions = await collect_versions()
        blocks["Versions"] = _safe_block("versions", lambda: version_lines(versions))

    def _settle(key: str, outcome: "dict[str, list[str]] | Exception") -> None:
        if isinstance(outcome, Exception):
            # _safe_block guards each collector, so reaching here means the probe
            # itself broke rather than one block. Fail only its own blocks.
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
