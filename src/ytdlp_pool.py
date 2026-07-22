"""Lifecycle for the process pool that runs yt-dlp extraction.

yt-dlp extraction is only half I/O — JSON parsing, signature decryption and format
selection are all GIL-bound Python. Running it in a ProcessPoolExecutor (not threads)
gives concurrent extractions across guilds true parallelism instead of GIL contention
that also steals time from the event loop serving voice heartbeats. Worker count is
env-tunable (YTDLP_POOL_WORKERS); each worker holds a full CPython + yt-dlp import
(~80–120 MB RSS), so the default is deliberately conservative — raise it if multi-guild
extraction bursts become the bottleneck. Design: docs/ARCHITECTURE_PLAN.md §3.1.

This module owns the pool's *lifecycle* only and knows nothing about yt-dlp: the
callable is supplied per call (see run()). That separation is load-bearing for tests —
see docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §2.5.
"""

import asyncio
import os
import sys
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Any, Optional, TypeVar

from src.telemetry import configure_worker_logging
from src.util import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_WORKERS = int(os.environ.get("YTDLP_POOL_WORKERS", "4"))
# How long shutdown waits for in-flight extractions before abandoning the join. yt-dlp
# carries socket_timeout=30 and retries=10, so an unlucky extraction can outlive any
# reasonable shutdown; waiting for it would hang the bot's exit. Mirrors the timeout
# asyncio added to loop.shutdown_default_executor() for the same reason.
_SHUTDOWN_TIMEOUT_SECS = 10.0


def _warmup_noop() -> None:
    """Submitted by prewarm() only to force a worker to spawn and import yt-dlp before
    the first real extraction has to pay that cost. Module-level so it is picklable to
    a worker."""
    return None


def _worker_init() -> None:
    """Per-worker setup, hardened so it can never raise.

    The stdlib contract is unforgiving: "Should initializer raise an exception, all
    currently pending jobs will raise a BrokenProcessPool, as well as any attempt to
    submit more jobs to the pool." Verified against 3.14.6 — and because a rebuilt pool
    runs the same initializer, run()'s heal-once retry is futile for this failure class:
    it would double the failure latency and log a misleading "a worker died".

    Degrading to unstructured worker logs is strictly better than bricking every
    extraction in the process. Reported via stderr rather than log.*, because the thing
    that just failed is the logging configuration.
    """
    try:
        configure_worker_logging()
    except Exception:
        print("yt-dlp worker logging setup failed:", file=sys.stderr)
        traceback.print_exc()


class PoolClosedError(RuntimeError):
    """Raised when work is submitted after shutdown().

    Deliberately an error rather than a silent rebuild: a submit during shutdown means a
    background task outlived close(), and spawning four fresh worker processes to serve it
    would leave them orphaned (nothing joins a pool created after the join). Callers on the
    extraction path already handle exceptions — prefetch_stream logs and swallows, the
    command paths surface an error embed.

    Subclasses RuntimeError to match the stdlib exactly: Executor.submit() after
    shutdown() "will raise RuntimeError", so any handler written against the underlying
    executor's contract keeps working.
    """


class YtdlpPool:
    """The process's yt-dlp extraction pool: lazy creation, break-healing, shutdown.

    One instance per process, held by src.youtube. Not a singleton by construction —
    tests build their own with a thread-pool factory, which is the whole point (a
    ProcessPoolExecutor pickles the submitted callable, and the MagicMock that tests
    patch onto _ytdlp_extract is unpicklable; a real patch would never reach a worker
    anyway).

    The executor is created lazily so importing this module never spawns children:
    under the 3.14 spawn/forkserver start method each worker re-imports the parent's
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
        # Monotonic, incremented per executor built. Without it, "the pool broke" logs
        # from before and after a rebuild are indistinguishable, and a pool breaking
        # repeatedly looks identical to one breaking once.
        self._generation = 0
        # Guards _executor, _closed and _generation. aclose() keeps every mutation on
        # the event-loop thread, so this is defence in depth rather than load-bearing —
        # but shutdown() is public and will eventually be called from an atexit hook or
        # a signal handler, and the class should be correct when it is. Never held
        # across a join.
        self._lock = threading.Lock()

    def _spawn_process_pool(self) -> Executor:
        # initializer runs _worker_init() once per worker so yt-dlp's warnings (emitted
        # from inside extract_info, now in a worker) stay structured.
        # ProcessPoolExecutor.__init__ does not spawn — workers start on first submit —
        # so constructing this under the lock is cheap.
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_worker_init,
        )

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

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

        Identity-checked: if two concurrent extractions both hit BrokenProcessPool, only
        the first discards — the second would otherwise throw away the healthy
        replacement the first just built. The broken executor is best-effort shut down
        without waiting, to release its manager thread (a broken pool never accepts new
        work, so waiting would be pointless).
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
        """Run `fn(*args)` in the pool, healing a broken pool once.

        A ProcessPoolExecutor becomes permanently broken if a worker dies abnormally —
        most plausibly the OOM killer reaping one under memory pressure — after which
        every submit raises BrokenProcessPool for the life of the process. Rebuild and
        retry once so a single worker death doesn't brick all extraction across every
        guild (the old ThreadPoolExecutor had no equivalent all-or-nothing failure mode).
        A second failure is a real problem and propagates to the caller's existing error
        handling.

        `fn` is a parameter, never stored: it is looked up in the caller's module at call
        time, which is what keeps `patch("src.youtube._ytdlp_extract")` working in the 29
        tests that rely on it.
        """
        loop = asyncio.get_running_loop()
        executor = self._acquire()
        try:
            return await loop.run_in_executor(executor, fn, *args)
        except BrokenProcessPool:
            log.warning(
                f"yt-dlp process pool #{self._generation} broke (a worker died) — "
                "rebuilding and retrying once"
            )
            self._replace(executor)
            return await loop.run_in_executor(self._acquire(), fn, *args)

    def prewarm(self) -> None:
        """Spawn the workers now (from setup_hook) so the first -play doesn't absorb
        process-spawn + yt-dlp-import latency. Fire-and-forget: submits one no-op per
        worker and returns without awaiting them."""
        executor = self._acquire()
        if not isinstance(executor, ProcessPoolExecutor):
            return  # a thread pool (tests) has nothing to spawn
        for _ in range(self._max_workers):
            executor.submit(_warmup_noop)

    def _close(self) -> Optional[Executor]:
        """Mark the pool closed and unpublish its executor, returning it to be joined.

        The join is the caller's business precisely because it blocks: the lock is
        released before it happens. Holding it across a join would stall every concurrent
        run() for no benefit — the executor is already unpublished and _closed is already
        set, so there is nothing left to race over.
        """
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        return executor

    async def aclose(self, timeout: float = _SHUTDOWN_TIMEOUT_SECS) -> None:
        """Close the pool from the event loop: flip the flag here, join off-thread.

        The production shutdown path. Modelled on asyncio's own
        loop.shutdown_default_executor(): the state change happens synchronously on the
        event-loop thread, and only the blocking join is handed to a worker thread. That
        is what makes cross-thread mutation of this object's state structurally
        impossible rather than merely locked.

        A join that outruns `timeout` is abandoned, not awaited: yt-dlp's own retry
        budget can keep an extraction alive far longer than a bot shutdown should take.
        The abandoned thread is still running — nothing can cancel a thread mid-join —
        but the process is exiting, and the workers are killed with it.
        """
        executor = self._close()
        if executor is None:
            return
        loop = asyncio.get_running_loop()
        join = partial(executor.shutdown, wait=True, cancel_futures=True)
        try:
            async with asyncio.timeout(timeout):
                await loop.run_in_executor(None, join)
        except TimeoutError:
            log.warning(
                f"yt-dlp pool #{self._generation} did not finish joining within "
                f"{timeout}s — abandoning the join"
            )
            executor.shutdown(wait=False, cancel_futures=True)

    def shutdown(self, wait: bool = True) -> None:
        """Synchronous close, for callers with no event loop (tests, atexit, signals).

        Blocking by default (joins worker processes). Idempotent, and safe when no
        executor was ever created. After this, submits raise PoolClosedError rather than
        silently spawning a fresh pool that nothing will ever join. Prefer aclose() from
        async code.
        """
        executor = self._close()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
