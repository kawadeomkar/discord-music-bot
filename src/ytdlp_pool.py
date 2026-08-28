"""Lifecycle for the process pool that runs yt-dlp extraction.

Extraction is only half I/O — JSON parsing, signature decryption and format selection
are GIL-bound, so processes (not threads) give concurrent extractions real parallelism
instead of contention that also starves the voice heartbeat. Each worker costs a full
CPython + yt-dlp import (~80–120 MB RSS), hence the conservative YTDLP_POOL_WORKERS
default. Lifecycle is all this module owns: the callable is supplied per call (run()),
which is what lets tests swap in a thread-pool-backed instance.

What crosses the process boundary, and the pickle contract each side owes:
docs/ARCHITECTURE.md#yt-dlp-process-boundary.
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
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures import BrokenExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from logging.handlers import QueueListener
from typing import Any, Optional, TypeVar

import structlog
from opentelemetry import trace

from src.telemetry import configure_worker_logging
from src.util import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_WORKERS = int(os.environ.get("YTDLP_POOL_WORKERS", "4"))
# How long shutdown waits before abandoning the join: yt-dlp's socket_timeout=30 with
# retries=10 can outlive any shutdown. Mirrors loop.shutdown_default_executor()'s.
_SHUTDOWN_TIMEOUT_SECS = 10.0


def _warmup_noop() -> None:
    """Submitted by prewarm() to force a worker to spawn and import yt-dlp before the
    first real extraction pays that cost. Module-level so it is picklable."""
    return None


def _worker_init(log_queue: Optional[Any] = None) -> None:
    """Per-worker setup, hardened so it can never raise: per the stdlib contract
    (verified on 3.14.6) an initializer that raises makes every pending AND future submit
    raise BrokenProcessPool, and run()'s heal-once retry cannot help since a rebuilt pool
    runs the same initializer. Unstructured worker logs beat bricking all extraction —
    reported on stderr, because what just failed is the logging configuration.
    """
    try:
        configure_worker_logging(log_queue)
    except Exception:
        print("yt-dlp worker logging setup failed:", file=sys.stderr)
        traceback.print_exc()


def _trace_carrier() -> dict[str, str]:
    """The parent's trace context as picklable strings for a worker to rebind. Workers
    have no TracerProvider, so get_current_span() is always invalid there and correlation
    must be carried explicitly. Empty when no span is active (prewarm, tests)."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }


def _call_with_context(carrier: dict[str, str], fn: Callable[..., T], *args: Any) -> T:
    """Bind the parent's trace context, then run fn through the picklable-error net.
    Runs in the worker (or the test thread). bound_contextvars resets on exit, so the
    worker's next job does not inherit a stale trace_id."""
    with structlog.contextvars.bound_contextvars(**carrier):
        return _picklable_call(fn, *args)


class RemoteCallError(Exception):
    """Generic picklable stand-in for a worker exception that can't cross the boundary.
    Every field must have a default: BaseException.__reduce__ rebuilds as `cls(*args)`,
    so a required positional *serialises* fine in the worker and then fails to *unpickle*
    in the parent's executor-manager thread, bricking the pool. Same rule as
    ExtractionError in src/youtube.py.
    """

    def __init__(self, message: str = "", original_type: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.original_type = original_type


def _picklable_call(fn: Callable[..., T], *args: Any) -> T:
    """Run fn(*args) in the worker, guaranteeing whatever propagates survives pickling.
    BrokenExecutor is re-raised untouched: it is the parent's healing signal, and a real
    one never originates in a worker anyway.
    """
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
    """Raised when work is submitted after shutdown(). An error rather than a silent
    rebuild: a submit during shutdown means a background task outlived close(), and fresh
    workers spawned to serve it would be orphaned (nothing joins a pool created after the
    join). Subclasses RuntimeError to match the stdlib's submit-after-shutdown contract.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class PoolState:
    """The pool's lifecycle as -debug reports it. `spawned` is False until the first
    extraction — the executor is lazy — and `generation` counts executors BUILT, so
    a value above 1 means a worker died abnormally and the pool was healed."""

    max_workers: int
    spawned: bool
    generation: int
    closed: bool


class YtdlpPool:
    """The process's yt-dlp extraction pool: lazy creation, break-healing, shutdown.
    One instance per process, held by src.youtube. Deliberately not a singleton — tests
    build their own with a thread-pool factory, since a ProcessPoolExecutor pickles the
    submitted callable and the MagicMock patched onto _ytdlp_extract is unpicklable. The
    executor is lazy because under spawn/forkserver each worker re-imports the parent's
    modules, and an eager pool would have every worker construct a nested one.
    """

    def __init__(
        self,
        max_workers: int = _DEFAULT_WORKERS,
        executor_factory: Optional[Callable[[], Executor]] = None,
    ) -> None:
        self._max_workers = max_workers
        self._executor_factory = executor_factory or self._spawn_process_pool
        self._executor: Optional[Executor] = None
        self._closed = False
        # Monotonic per executor built, so "the pool broke" logs from before and after a
        # rebuild are distinguishable and repeated breaks don't look like one break.
        self._generation = 0
        # Guards _executor, _closed and _generation. Defence in depth — aclose() keeps
        # every mutation on the loop thread — but shutdown() may run from atexit/signal.
        self._lock = threading.Lock()
        # Worker-log plumbing. Pool-scoped, not executor-scoped: built on the first real
        # spawn and reused across break-heal rebuilds, since the listener forwards to the
        # parent's root handlers regardless of generation. None under the test seam.
        self._log_queue: Optional[Any] = None
        self._log_listener: Optional[QueueListener] = None

    @property
    def max_workers(self) -> int:
        """Worker count this pool runs with, for callers sizing their own bounds
        against it — YTDLP_POOL_WORKERS stays read in one place."""
        return self._max_workers

    def _spawn_process_pool(self) -> Executor:
        # initializer runs _worker_init() per worker so yt-dlp's warnings reach the
        # parent structured. Cheap under the lock: __init__ does not spawn.
        if self._log_listener is None:
            # respect_handler_level=True so a worker DEBUG record is not force-emitted
            # by an INFO handler. Takes the root handlers live at spawn time — post-
            # setup_telemetry() in production, so worker records reach Loki.
            self._log_queue = multiprocessing.Queue()
            self._log_listener = QueueListener(
                self._log_queue, *logging.root.handlers, respect_handler_level=True
            )
            self._log_listener.start()
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_worker_init,
            initargs=(self._log_queue,),
        )

    def _stop_log_listener(self) -> None:
        """Drain and stop the listener. Must run only after the workers are gone
        (join or terminate): stop() enqueues a sentinel and drains what is already
        queued, so stopping while workers still emit discards their final records."""
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
        """A read-only lifecycle snapshot for -debug. Deliberately only what the
        pool already tracks: an in-flight or completed-extraction counter would be
        bookkeeping on the hot path for a diagnostic line."""
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
                raise PoolClosedError("yt-dlp extraction pool is shut down")
            if self._executor is None:
                self._executor = self._executor_factory()
                self._generation += 1
            return self._executor

    def _replace(self, broken: Executor) -> None:
        """Drop `broken` so the next _acquire() builds a fresh executor.
        Identity-checked: when two concurrent extractions both hit BrokenProcessPool,
        only the first discards — the second would throw away the healthy replacement.
        Shut down without waiting; a broken pool never accepts work anyway.
        """
        with self._lock:
            if self._executor is not broken:
                return
            self._executor = None
        try:
            broken.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            log.debug(f"discarding broken yt-dlp pool raised: {e}")

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """Run `fn(*args)` in the pool, healing a broken pool once. A ProcessPoolExecutor
        breaks permanently when a worker dies abnormally (most plausibly the OOM killer),
        after which every submit raises BrokenProcessPool for the life of the process; a
        second failure propagates. `fn` is a parameter, never stored — looked up in the
        caller's module at call time, which keeps `patch(...)` on it working.
        """
        loop = asyncio.get_running_loop()
        carrier = _trace_carrier()
        executor = self._acquire()
        try:
            return await loop.run_in_executor(
                executor, _call_with_context, carrier, fn, *args
            )
        except BrokenProcessPool:
            log.warning(
                f"yt-dlp process pool #{self._generation} broke (a worker died) — "
                "rebuilding and retrying once"
            )
            self._replace(executor)
            return await loop.run_in_executor(
                self._acquire(), _call_with_context, carrier, fn, *args
            )

    def prewarm(self, warm: Callable[[], object] = _warmup_noop) -> None:
        """Spawn the workers now (from setup_hook) so the first -play doesn't absorb
        process-spawn + yt-dlp-import latency. `warm` is what each worker runs, supplied
        by the caller like every run() callable, so a caller can also pay its own
        first-use costs here. Fire-and-forget: submits one call per worker and returns
        without awaiting them."""
        executor = self._acquire()
        if not isinstance(executor, ProcessPoolExecutor):
            return  # a thread pool (tests) has nothing to spawn
        for _ in range(self._max_workers):
            # Through _call_with_context like every run() call: it flattens a yt-dlp
            # exception, which otherwise fails to unpickle and bricks the pool.
            executor.submit(_call_with_context, {}, warm)

    def _close(self) -> Optional[Executor]:
        """Mark the pool closed and unpublish its executor, returning it to be joined.
        The join is the caller's business because it blocks: holding the lock across it
        would stall every concurrent run() for nothing.
        """
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        return executor

    async def aclose(self, timeout: float = _SHUTDOWN_TIMEOUT_SECS) -> None:
        """Close the pool from the event loop: flip the flag here, join off-thread, so
        cross-thread mutation is structurally impossible rather than merely locked
        (modelled on loop.shutdown_default_executor()). A join that outruns `timeout` is
        abandoned, not awaited — nothing can cancel a thread mid-join, but the exiting
        process takes it along.
        """
        executor = self._close()
        if executor is not None:
            loop = asyncio.get_running_loop()
            join = partial(executor.shutdown, wait=True, cancel_futures=True)
            try:
                async with asyncio.timeout(timeout):
                    await loop.run_in_executor(None, join)
            except TimeoutError:
                log.warning(
                    f"yt-dlp pool #{self._generation} did not finish joining within "
                    f"{timeout}s — terminating its workers"
                )
                # shutdown(wait=False) does not bound exit: the abandoned join keeps
                # the manager thread alive and _python_exit joins it at interpreter
                # exit. Measured: 61s with an in-flight extraction, 3.4s once SIGTERMed.
                if isinstance(executor, ProcessPoolExecutor):
                    executor.terminate_workers()
                else:
                    executor.shutdown(wait=False, cancel_futures=True)
        # Unconditional even when _close() returned None: a concurrent _replace() during
        # a break-heal can null the executor out while leaving the listener running, and
        # an early return would leak that thread for the life of the process. With an
        # executor this still runs only after its workers are gone (join or terminate).
        self._stop_log_listener()

    def shutdown(self, wait: bool = True) -> None:
        """Synchronous close, for a caller with no event loop to await. Used by tests;
        production flows through aclose(). Deliberately not an atexit or signal handler
        despite the shape inviting it: discord.py already routes SIGTERM and
        KeyboardInterrupt through the bot's close(). Blocking by default, idempotent,
        safe when no executor was created; after this, submits raise PoolClosedError.
        """
        executor = self._close()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
        self._stop_log_listener()
