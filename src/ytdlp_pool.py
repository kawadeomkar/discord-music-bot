"""Lifecycle for the process pool that runs yt-dlp extraction.

Extraction is half GIL-bound (JSON parsing, signature decryption, format
selection), so processes rather than threads. Each worker costs a full CPython +
yt-dlp import (~80–120 MB RSS), hence the conservative YTDLP_POOL_WORKERS
default. Only lifecycle lives here: the callable is supplied per call (run()),
which is what lets tests swap in a thread-pool-backed instance.

Pickle contract for what crosses the boundary: docs/ARCHITECTURE.md#yt-dlp-process-boundary.
"""

import asyncio
import logging
import multiprocessing
import os
import pickle
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from concurrent.futures import BrokenExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from logging.handlers import QueueListener
from typing import Any, Optional, TypeVar

import structlog
from opentelemetry import trace

from src.config import YTDLP_WORKER_MEMORY_MB
from src.telemetry import configure_worker_logging
from src.util import get_logger

try:
    import resource
except ImportError:  # pragma: no cover - not a Unix platform
    resource = None

log = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_WORKERS = int(os.environ.get("YTDLP_POOL_WORKERS", "4"))
_DEFAULT_WORKER_MEMORY_BYTES = YTDLP_WORKER_MEMORY_MB * 1024 * 1024
# How long shutdown waits before abandoning the join: yt-dlp's socket_timeout=30
# with retries=10 can outlive any shutdown.
_SHUTDOWN_TIMEOUT_SECS = 10.0


def _warmup_noop() -> None:
    """Submitted by prewarm() to force a worker to spawn and import yt-dlp.
    Module-level so it is picklable."""
    return None


def _apply_memory_limit(limit_bytes: int) -> None:
    """Cap this worker's committed private memory so a runaway response body
    fails as a MemoryError in one job. RLIMIT_DATA, not RLIMIT_AS: yt-dlp's Deno
    child inherits the limit, and V8 reserves address space it never commits —
    Deno aborts under a 1 GiB RLIMIT_AS and runs under the same RLIMIT_DATA.
    See docs/ARCHITECTURE.md#extraction-bounds."""
    if limit_bytes <= 0 or resource is None:
        return
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit_bytes = min(limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, hard))


def _worker_init(log_queue: Optional[Any] = None, memory_limit_bytes: int = 0) -> None:
    """Per-worker setup that can never raise: an initializer that raises makes
    every pending and future submit raise BrokenProcessPool, and the heal-once
    retry runs the same initializer. Reported on stderr, because what just
    failed may be the logging configuration."""
    try:
        configure_worker_logging(log_queue)
    except Exception:
        print("worker logging setup failed:", file=sys.stderr)
        traceback.print_exc()
    try:
        _apply_memory_limit(memory_limit_bytes)
    except Exception:
        print("worker memory limit not applied:", file=sys.stderr)
        traceback.print_exc()


def _trace_carrier() -> dict[str, str]:
    """The parent's trace context as picklable strings. Workers have no
    TracerProvider, so correlation must be carried explicitly. Empty when no
    span is active."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }


def _call_with_context(carrier: dict[str, str], fn: Callable[..., T], *args: Any) -> T:
    """Bind the parent's trace context, then run fn through the picklable-error
    net. bound_contextvars resets on exit, so the worker's next job does not
    inherit a stale trace_id."""
    with structlog.contextvars.bound_contextvars(**carrier):
        return _picklable_call(fn, *args)


class RemoteCallError(Exception):
    """Picklable stand-in for a worker exception that cannot cross the boundary.
    Every field must have a default: BaseException.__reduce__ rebuilds as
    `cls(*args)`, and a required positional fails to unpickle in the parent's
    executor thread, bricking the pool. Same rule as ExtractionError."""

    def __init__(self, message: str = "", original_type: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.original_type = original_type


def _picklable_call(fn: Callable[..., T], *args: Any) -> T:
    """Run fn(*args) in the worker, guaranteeing whatever propagates survives
    pickling. BrokenExecutor is re-raised untouched: it is the parent's healing
    signal."""
    try:
        return fn(*args)
    except BrokenExecutor:
        raise
    except Exception as e:
        try:
            pickle.loads(
                pickle.dumps(e)
            )  # loads too: dumps alone passes the broken case
        except Exception:
            raise RemoteCallError(str(e), type(e).__name__) from e
        raise


class PoolClosedError(RuntimeError):
    """Raised when work is submitted after shutdown(): a submit then means a
    background task outlived close(), and fresh workers spawned for it would be
    orphaned. RuntimeError matches the stdlib's submit-after-shutdown contract."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PoolState:
    """The pool's lifecycle as -debug reports it. `spawned` is False until the
    first extraction; `generation` counts executors BUILT, so a value above 1
    means a worker died and the pool was healed."""

    max_workers: int
    spawned: bool
    generation: int
    closed: bool


class YtdlpPool:
    """A process pool with lazy creation, heal-once on BrokenProcessPool, an
    off-loop join and worker-log plumbing. One instance per consumer (yt-dlp in
    src.youtube, the chart renderer in src.chart_pool), distinguished by `name`
    in log lines and PoolClosedError. Lazy because under spawn/forkserver each
    worker re-imports the parent's modules, and an eager pool would have every
    worker construct a nested one."""

    def __init__(
        self,
        max_workers: int = _DEFAULT_WORKERS,
        executor_factory: Optional[Callable[[], Executor]] = None,
        name: str = "yt-dlp extraction",
        memory_limit_bytes: int = _DEFAULT_WORKER_MEMORY_BYTES,
    ) -> None:
        self._max_workers = max_workers
        self._name = name
        # RLIMIT_DATA applied by each worker's initializer; 0 applies none.
        self._memory_limit_bytes = memory_limit_bytes
        self._executor_factory = executor_factory or self._spawn_process_pool
        self._executor: Optional[Executor] = None
        self._closed = False
        # Per executor built, so break logs before and after a rebuild differ.
        self._generation = 0
        # Guards _executor, _closed and _generation; shutdown() may run from
        # atexit/signal, off the loop thread.
        self._lock = threading.Lock()
        # Worker-log plumbing, pool-scoped: built on the first real spawn and
        # reused across break-heal rebuilds. None under the test seam.
        self._log_queue: Optional[Any] = None
        self._log_listener: Optional[QueueListener] = None

    def _spawn_process_pool(self) -> Executor:
        # Cheap under the lock: ProcessPoolExecutor.__init__ does not spawn.
        if self._log_listener is None:
            # respect_handler_level=True so a worker DEBUG record is not
            # force-emitted by an INFO handler. Takes the root handlers live at
            # spawn time — post-setup_telemetry() in production.
            self._log_queue = multiprocessing.Queue()
            self._log_listener = QueueListener(
                self._log_queue, *logging.root.handlers, respect_handler_level=True
            )
            self._log_listener.start()
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_worker_init,
            initargs=(self._log_queue, self._memory_limit_bytes),
        )

    def _stop_log_listener(self) -> None:
        """Drain and stop the listener. Only after the workers are gone: stop()
        drains what is already queued, so stopping while workers still emit
        discards their final records."""
        listener = self._log_listener
        self._log_listener = None
        self._log_queue = None
        if listener is not None:
            listener.stop()

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def state(self) -> PoolState:
        """A read-only lifecycle snapshot for -debug."""
        with self._lock:
            return PoolState(
                max_workers=self._max_workers,
                spawned=self._executor is not None,
                generation=self._generation,
                closed=self._closed,
            )

    def _acquire(self) -> Executor:
        """The live executor, building it on first use. Raises once shut down."""
        with self._lock:
            if self._closed:
                raise PoolClosedError(f"{self._name} pool is shut down")
            if self._executor is None:
                self._executor = self._executor_factory()
                self._generation += 1
            return self._executor

    def _replace(self, broken: Executor) -> None:
        """Drop `broken` so the next _acquire() builds a fresh executor.
        Identity-checked: when two concurrent extractions both hit
        BrokenProcessPool, the second must not throw away the healthy
        replacement."""
        with self._lock:
            if self._executor is not broken:
                return
            self._executor = None
        try:
            broken.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            log.debug(f"discarding broken {self._name} pool raised: {e}")

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """Run `fn(*args)` in the pool, healing a broken pool once (a worker
        killed by OOM breaks the executor permanently); a second failure
        propagates. `fn` is never stored, so `patch(...)` on it keeps working."""
        loop = asyncio.get_running_loop()
        carrier = _trace_carrier()
        executor = self._acquire()
        try:
            return await loop.run_in_executor(
                executor, _call_with_context, carrier, fn, *args
            )
        except BrokenProcessPool:
            log.warning(
                f"{self._name} process pool #{self._generation} broke (a worker "
                "died) — rebuilding and retrying once"
            )
            self._replace(executor)
            return await loop.run_in_executor(
                self._acquire(), _call_with_context, carrier, fn, *args
            )

    def prewarm(self, warm: Callable[[], Any] = _warmup_noop) -> None:
        """Spawn the workers now so the first real call does not pay spawn and
        import latency. Fire-and-forget: submits `warm` once per worker. The
        default no-op warms only what a worker pays on the way up (all of
        yt-dlp's cost); the chart pool passes a callable that imports
        matplotlib. Must be picklable, hence module-level."""
        executor = self._acquire()
        if not isinstance(executor, ProcessPoolExecutor):
            return  # a thread pool (tests) has nothing to spawn
        for _ in range(self._max_workers):
            # concurrent.futures never reports an unretrieved exception, so
            # without the callback a warm that raises is silent.
            executor.submit(warm).add_done_callback(self._log_warm_failure)

    def _log_warm_failure(self, future: Future[Any]) -> None:
        error = future.exception()
        if error is not None:
            log.warning(f"{self._name} worker warm failed: {error}")

    def _close(self) -> Optional[Executor]:
        """Mark the pool closed and unpublish its executor, returning it to be
        joined by the caller: the join blocks, and holding the lock across it
        would stall every concurrent run()."""
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        return executor

    async def aclose(self, timeout: float = _SHUTDOWN_TIMEOUT_SECS) -> None:
        """Close from the event loop: flip the flag here, join off-thread. A
        join that outruns `timeout` is abandoned — nothing can cancel a thread
        mid-join, but the exiting process takes it along."""
        executor = self._close()
        if executor is not None:
            loop = asyncio.get_running_loop()
            join = partial(executor.shutdown, wait=True, cancel_futures=True)
            try:
                async with asyncio.timeout(timeout):
                    await loop.run_in_executor(None, join)
            except TimeoutError:
                log.warning(
                    f"{self._name} pool #{self._generation} did not finish joining "
                    f"within {timeout}s — terminating its workers"
                )
                # shutdown(wait=False) does not bound exit: the abandoned join
                # keeps the manager thread alive and _python_exit joins it at
                # interpreter exit (61s measured with an in-flight extraction).
                if isinstance(executor, ProcessPoolExecutor):
                    executor.terminate_workers()
                else:
                    executor.shutdown(wait=False, cancel_futures=True)
        # Unconditional: a concurrent _replace() during a break-heal can null
        # the executor while leaving the listener running, and an early return
        # would leak that thread for the life of the process.
        self._stop_log_listener()

    def shutdown(self, wait: bool = True) -> None:
        """Synchronous close for a caller with no event loop (tests; production
        flows through aclose()). Not an atexit or signal handler: discord.py
        already routes SIGTERM and KeyboardInterrupt through the bot's close().
        Idempotent; after this, submits raise PoolClosedError."""
        executor = self._close()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
        self._stop_log_listener()
