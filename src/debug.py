"""Debug mode: the footer every reply grows while it is on.

OBSERVATION-ONLY, and that is the whole design constraint. Nothing here changes
playback, caching, queueing or persistence — only what the bot shows. It is what
keeps "test with debug on, ship with debug off" a valid methodology: nothing you
validated changes when the toggle flips.
"""

import asyncio
import os
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import discord
from opentelemetry import trace

from src.util import cancel_task, get_logger, trace_id_of, truncate

if TYPE_CHECKING:
    from src.ytdlp_pool import PoolState

log = get_logger(__name__)

# Discord's hard cap on an embed's footer text.
FOOTER_LIMIT = 2048


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · FOOTER DECORATION
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
