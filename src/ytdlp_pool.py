"""Lifecycle for the process pool that runs yt-dlp extraction.

Extraction is half GIL-bound (JSON parsing, signature decryption, format
selection), so processes rather than threads. Each worker costs a full CPython +
yt-dlp import (~80–120 MB RSS), hence the conservative default for
config.YTDLP_POOL_WORKERS. Only lifecycle lives here: the callable is supplied per
call (run()), which is what lets tests swap in a thread-pool-backed instance.

Pickle contract for what crosses the boundary: docs/ARCHITECTURE.md#yt-dlp-process-boundary.
"""

import asyncio
import contextlib
import logging
import multiprocessing
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
from multiprocessing.context import BaseContext
from queue import Empty
from typing import Any, Optional, TypeVar

import structlog
from opentelemetry import trace

from src import config
from src.telemetry import configure_worker_logging
from src.util import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Extractions a worker serves before the pool replaces it. A worker's resident set
# grows with every extraction and never flattens, so 16 caps one near 300MB and keeps
# four inside the container. See docs/ARCHITECTURE.md#yt-dlp-process-boundary.
_MAX_TASKS_PER_CHILD = 16

# How long shutdown waits before abandoning the join: yt-dlp's socket_timeout=30
# with retries=10 can outlive any shutdown.
_SHUTDOWN_TIMEOUT_SECS = 10.0

# Explicit, because multiprocessing.Queue() defaults to maxsize=0 and sizes its
# semaphore at SEM_VALUE_MAX (32,767 here): 30,000 put_nowait calls were measured
# to raise no Full at all, so the drop-on-full a progress producer relies on would
# never happen and the worker would buffer without bound instead.
_PROGRESS_QUEUE_MAX = 256
# The drain polls rather than blocking, so stopping it never writes to a queue a
# SIGTERMed worker may have left holding _wlock.
_PROGRESS_POLL_SECS = 0.2
_PROGRESS_JOIN_SECS = 2.0

# Set in each worker by _worker_init. None in the parent, and None under the test
# seam, where an executor_factory is supplied and the initializer never runs.
_PROGRESS_Q: Optional[Any] = None


def worker_progress_queue() -> Optional[Any]:
    """The worker's progress transport, or None when the pool was built without
    one. Read per call rather than captured: a worker's module globals are set by
    the initializer, after import. This module owns the transport and knows
    nothing about what rides it."""
    return _PROGRESS_Q


def _pool_context() -> BaseContext:
    """The start method the pool spawns workers with: the platform default, unless
    that is `fork`, which a task budget rejects and which is unsafe to fork a
    multi-threaded asyncio process from.

    Passed EXPLICITLY. Supplying max_tasks_per_child without an mp_context makes
    ProcessPoolExecutor force spawn, replacing 3.14's forkserver default on Linux at
    ~23x the worker startup cost. See docs/ARCHITECTURE.md#yt-dlp-process-boundary.
    """
    method = multiprocessing.get_start_method()
    if method == "fork":
        available = multiprocessing.get_all_start_methods()
        method = "forkserver" if "forkserver" in available else "spawn"
    return multiprocessing.get_context(method)


def _warmup_noop() -> None:
    """Submitted by prewarm() to force a worker to spawn and import yt-dlp.
    Module-level so it is picklable."""
    return None


def _worker_init(
    log_queue: Optional[Any] = None, progress_queue: Optional[Any] = None
) -> None:
    """Per-worker setup that can never raise: an initializer that raises makes
    every pending and future submit raise BrokenProcessPool, and the heal-once
    retry runs the same initializer. Reported on stderr, because what just
    failed is the logging configuration."""
    global _PROGRESS_Q
    _PROGRESS_Q = progress_queue
    if progress_queue is not None:
        # Progress is advisory, so dropping the tail is correct — and it has to
        # be dropped: at worker exit Queue._finalize_join JOINS the feeder thread,
        # which is blocked in send_bytes once the parent stops reading, so the
        # worker cannot exit and shutdown burns its full timeout every time.
        progress_queue.cancel_join_thread()
    try:
        configure_worker_logging(log_queue)
    except Exception:
        print("worker logging setup failed:", file=sys.stderr)
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
        max_workers: int = config.YTDLP_POOL_WORKERS,
        executor_factory: Optional[Callable[[], Executor]] = None,
        name: str = "yt-dlp extraction",
        progress_sink: Optional[Callable[[Any], None]] = None,
        recycle_workers: bool = True,
    ) -> None:
        self._max_workers = max_workers
        self._name = name
        # Whether a worker is replaced after _MAX_TASKS_PER_CHILD calls. False for
        # the chart pool, whose one worker pays matplotlib's import again on every
        # replacement.
        self._recycle_workers = recycle_workers
        # What a worker's progress messages are handed to, on the drain THREAD.
        # It must not block and must not touch the event loop. None (the default,
        # and the chart pool's) builds no queue and no drain at all.
        self._progress_sink = progress_sink
        self._progress_queue: Optional[Any] = None
        self._progress_thread: Optional[threading.Thread] = None
        self._progress_stop = threading.Event()
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

    @property
    def max_workers(self) -> int:
        """Worker count this pool runs with, for callers sizing their own bounds
        against it — config.YTDLP_POOL_WORKERS stays read in one place."""
        return self._max_workers

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
        self._start_progress_drain()
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_worker_init,
            initargs=(self._log_queue, self._progress_queue),
            # A recycled worker re-runs the initializer, so worker logging and the
            # progress transport survive the turnover.
            max_tasks_per_child=(
                _MAX_TASKS_PER_CHILD if self._recycle_workers else None
            ),
            mp_context=_pool_context(),
        )

    def _start_progress_drain(self) -> None:
        """Build the progress queue and the thread that drains it.

        Rebuilt on a break-heal, unlike the log listener beside it, and for a
        reason the log queue does not share: a worker SIGKILLed between
        Queue._feed's wacquire() and wrelease() holds the write lock forever.
        _sem never drains, and after _PROGRESS_QUEUE_MAX puts every put_nowait
        raises Full — silently, permanently, for the life of the process. Reusing
        the queue across a heal is exactly the case where that has happened."""
        if self._progress_sink is None:
            return
        # Not joined: this runs under _lock, from _acquire on the event loop, and
        # the old drain leaves on its own within _PROGRESS_POLL_SECS of the flag.
        self._stop_progress_drain(join=False)
        self._progress_queue = multiprocessing.Queue(maxsize=_PROGRESS_QUEUE_MAX)
        # A fresh flag, not a cleared one: _stop_progress_drain set the old one,
        # and the thread below must not start already told to stop.
        self._progress_stop = threading.Event()
        self._progress_thread = threading.Thread(
            target=self._drain_progress,
            args=(self._progress_queue, self._progress_stop),
            name=f"{self._name}-progress",
            daemon=True,
        )
        self._progress_thread.start()

    def _drain_progress(self, queue: Any, stop: threading.Event) -> None:
        """Drain worker progress on a THREAD, like the QueueListener beside it.
        get_nowait measured at 38.4 µs/op and a playlist's messages arrive in a
        burst — on the event loop that is starvation, delaying every other guild's
        Now Playing edit and the playback loop's song handoff."""
        while not stop.is_set():
            try:
                message = queue.get(timeout=_PROGRESS_POLL_SECS)
            except Empty:
                continue
            except Exception as e:
                # A closed queue at shutdown is expected and says nothing; any
                # other cause ends progress for the life of this executor, so it
                # must not be the one failure that leaves no line anywhere.
                if not stop.is_set():
                    log.warning(f"{self._name} progress drain stopped: {e!r}")
                return
            sink = self._progress_sink
            if sink is None:
                continue
            try:
                sink(message)
            except Exception as e:
                log.warning(f"{self._name} progress sink failed: {e!r}")

    def _stop_progress_drain(self, *, join: bool = True) -> None:
        """Stop the drain and drop the queue. Stopping is a FLAG rather than a
        sentinel message, which makes the workers' state irrelevant: after
        terminate_workers() the queue's _wlock is a POSIX semaphore a worker
        SIGTERMed mid-write never released, so a parent write could block here
        forever. Idempotent, because a break-heal calls it before rebuilding.

        close() releases the parent's own handle and feeder thread. It does NOT
        stop the drain — this side reads, and closing a queue does not interrupt
        a blocked get — so it is cleanup rather than a backstop for the flag."""
        thread, queue = self._progress_thread, self._progress_queue
        self._progress_thread = self._progress_queue = None
        self._progress_stop.set()
        if thread is not None and join:
            thread.join(timeout=_PROGRESS_JOIN_SECS)
        if queue is not None:
            with contextlib.suppress(Exception):
                queue.close()

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
            # Through _call_with_context like every run() call: it flattens a yt-dlp
            # exception, which otherwise fails to unpickle and bricks the pool.
            # concurrent.futures never reports an unretrieved exception the way
            # asyncio does, so without the callback a warm that raises is silent.
            executor.submit(_call_with_context, {}, warm).add_done_callback(
                self._log_warm_failure
            )

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
        self._stop_progress_drain()

    def shutdown(self, wait: bool = True) -> None:
        """Synchronous close for a caller with no event loop (tests; production
        flows through aclose()). Not an atexit or signal handler: discord.py
        already routes SIGTERM and KeyboardInterrupt through the bot's close().
        Idempotent; after this, submits raise PoolClosedError."""
        executor = self._close()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
        self._stop_log_listener()
        self._stop_progress_drain()
