"""Tests for src/ytdlp_pool.py — the extraction pool's lifecycle.

Almost everything here drives a YtdlpPool built with an `executor_factory` seam, so no
worker process is spawned: the class owns lifecycle, not extraction, and its logic is
about *when* an executor is built, replaced and closed. The one exception is
TestRealWorkerProcess, which spawns for real — see its docstring.
"""

import asyncio
import os
import pickle
import threading
from collections.abc import Callable
from concurrent.futures import (
    BrokenExecutor,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from concurrent.futures.process import BrokenProcessPool
from logging.handlers import QueueListener
from unittest.mock import MagicMock, patch

import pytest

from src.ytdlp_pool import (
    PoolClosedError,
    RemoteCallError,
    YtdlpPool,
    _picklable_call,
    _warmup_noop,
    _worker_init,
)


def _double(value: int) -> int:
    """Module-level so it is picklable to a real worker process."""
    return value * 2


def _double_in_worker(value: int) -> tuple[int, int]:
    """Same, but reports the PID that ran it — so a caller can prove the work actually
    crossed a process boundary rather than quietly running on a thread."""
    return os.getpid(), value * 2


class _UnpicklableError(Exception):
    """A *required* positional and no default — serialises but fails to *unpickle*, exactly
    the trap that bricks the pool if it reaches the result queue (§12.1). Module-level so
    dumps() can resolve the class by reference and get far enough to fail on loads()."""

    def __init__(self, message: str, extra: object) -> None:
        super().__init__(message)
        self.extra = extra


class _UnpicklableBroken(BrokenExecutor):
    """A BrokenExecutor that itself cannot cross the boundary. The real BrokenProcessPool
    is picklable, so only a synthetic one like this can prove the exclusion in
    _picklable_call is load-bearing rather than incidental."""

    def __init__(self, message: str, extra: object) -> None:
        super().__init__(message)
        self.extra = extra


def _raise_classified_error_in_worker(_ignored: object) -> object:
    """Runs in a real worker: reproduce _ytdlp_extract's except path with a genuine
    yt-dlp error whose exc_info/cause are populated, without any network. Proves the flat
    ExtractionError — and its cause — survive a real pickle boundary with fields intact.
    """
    import sys

    from yt_dlp.utils import DownloadError, ExtractorError

    from src.youtube import _classify_ytdlp_error

    try:
        try:
            raise ExtractorError("Video unavailable", video_id="vid42", expected=True)
        except ExtractorError:
            raise DownloadError(
                "ERROR: [youtube] vid42: Video unavailable",
                sys.exc_info(),  # type: ignore[arg-type]
            )
    except DownloadError as e:
        raise _classify_ytdlp_error(e) from e


def _return_raw_info_from_worker(_ignored: object) -> object:
    """Runs in a real worker: return the un-slimmed info dict extract_info() hands back,
    live unpicklable object and all. The pool must pickle this into its result queue, so
    it proves — across a real boundary — that returning the raw dict fails the *call*
    (not silently succeeds) the way _slim_info's absence would in production."""
    from tests.test_youtube import _realistic_raw_info

    return _realistic_raw_info()


def _return_slimmed_info_from_worker(_ignored: object) -> object:
    """Runs in a real worker: return the same dict through _slim_info, exactly as
    _ytdlp_extract does. Proves the slimmed result crosses the boundary intact."""
    from tests.test_youtube import _realistic_raw_info

    from src.youtube import _slim_info

    return _slim_info(_realistic_raw_info())


def _log_warning_in_worker(message: str) -> int:
    """Runs in a real worker: emit one warning the way youtube._YtdlpLogger does, so the
    parent's listener can prove it arrives with worker_id and the propagated trace_id.
    """
    from src.util import get_logger

    get_logger("yt_dlp.worker").warning(message)
    return os.getpid()


def _thread_pool_factory(max_workers: int = 2) -> Callable[[], Executor]:
    """A factory the pool can call to get an in-process executor."""
    return lambda: ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="ytdlp-test"
    )


class TestLazyCreation:
    def test_executor_is_created_lazily_and_memoized(self) -> None:
        """Nothing is built at construction — importing src.youtube must never spawn."""
        factory = MagicMock(return_value=MagicMock(spec=Executor))
        pool = YtdlpPool(executor_factory=factory)

        factory.assert_not_called()

        first = pool._acquire()
        second = pool._acquire()

        assert first is second
        factory.assert_called_once_with()

    def test_default_factory_builds_a_sized_process_pool(self) -> None:
        """The default factory is the only place a real ProcessPoolExecutor is named.

        Patched rather than constructed: this asserts the wiring (worker count, the
        hardened initializer), not that multiprocessing works.
        """
        pool = YtdlpPool(max_workers=3)
        sentinel = MagicMock(name="ProcessPoolExecutor-instance")

        with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=sentinel) as ctor:
            executor = pool._acquire()

        try:
            assert executor is sentinel
            # Wiring, not multiprocessing: the hardened initializer plus the worker-log
            # queue handed to it via initargs (§12.2 Option B).
            ctor.assert_called_once_with(
                max_workers=3,
                initializer=_worker_init,
                initargs=(pool._log_queue,),
            )
            # the real spawn path starts a listener to drain that queue into the parent
            assert pool._log_listener is not None
        finally:
            pool.shutdown(wait=False)

    def test_generation_increments_per_executor_built(self) -> None:
        """Rebuild bumps the counter so logs from either side of a break are
        distinguishable — a pool breaking repeatedly must not read like one break."""
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        try:
            first = pool._acquire()
            assert pool._generation == 1

            pool._replace(first)
            pool._acquire()

            assert pool._generation == 2
        finally:
            pool.shutdown(wait=False)

    def test_concurrent_acquire_creates_exactly_one_executor(self) -> None:
        """N threads racing into _acquire() must build one executor, not N.

        The pool this class replaced justified skipping a lock by asserting all mutation
        happened on the event-loop thread; that was untrue of its shutdown path
        (docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §2.4 D2). The lock is the fix, and a
        barrier is how you prove it: every thread is released into _acquire() at once.
        """
        threads = 8
        barrier = threading.Barrier(threads)
        built: list[Executor] = []
        built_lock = threading.Lock()

        def counting_factory() -> Executor:
            executor = MagicMock(spec=Executor)
            with built_lock:
                built.append(executor)
            return executor

        pool = YtdlpPool(executor_factory=counting_factory)
        acquired: list[Executor] = []
        acquired_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            executor = pool._acquire()
            with acquired_lock:
                acquired.append(executor)

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=5)

        assert len(built) == 1, "the factory ran more than once under contention"
        assert len(acquired) == threads
        assert all(e is built[0] for e in acquired)
        assert pool._generation == 1


class TestPrewarm:
    def test_prewarm_is_noop_for_a_thread_pool(self) -> None:
        """A thread pool (what tests run on) has no spawn cost to pay up front."""
        executor = MagicMock(spec=ThreadPoolExecutor)
        pool = YtdlpPool(executor_factory=lambda: executor)

        pool.prewarm()

        executor.submit.assert_not_called()

    def test_prewarm_submits_one_noop_per_worker(self) -> None:
        from concurrent.futures import ProcessPoolExecutor

        executor = MagicMock(spec=ProcessPoolExecutor)
        pool = YtdlpPool(max_workers=3, executor_factory=lambda: executor)

        pool.prewarm()

        assert executor.submit.call_count == 3
        for call in executor.submit.call_args_list:
            assert call.args[0] is _warmup_noop

    def test_prewarm_after_shutdown_raises(self) -> None:
        """The closed gate covers every entry point, not just run()."""
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        pool.shutdown()

        with pytest.raises(PoolClosedError):
            pool.prewarm()


class TestRun:
    async def test_run_executes_the_callable_and_returns_its_result(self) -> None:
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        try:
            assert await pool.run(_double, 21) == 42
        finally:
            pool.shutdown(wait=False)

    async def test_run_resolves_the_callable_at_call_time(self) -> None:
        """The constraint the whole API shape exists to satisfy.

        29 tests in test_youtube.py patch src.youtube._ytdlp_extract with a MagicMock.
        That only works because run() takes the callable as a *parameter*, resolved from
        the caller's module at call time — capturing it in __init__ would bind the real
        function once at construction and every one of those patches would silently miss
        (docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §2.5).
        """
        import src.ytdlp_pool as module

        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        try:
            with patch.object(module, "_warmup_noop", return_value="patched"):
                assert await pool.run(module._warmup_noop) == "patched"
        finally:
            pool.shutdown(wait=False)

    async def test_run_heals_a_broken_pool_and_retries_once(self) -> None:
        """A worker death must not brick extraction across every guild: rebuild once
        and retry, succeeding on the second attempt."""
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        fn = MagicMock(side_effect=[BrokenProcessPool("worker died"), {"ok": 1}])
        try:
            first = pool._acquire()

            assert await pool.run(fn, "u") == {"ok": 1}

            assert fn.call_count == 2
            assert pool._acquire() is not first  # the broken executor was replaced
            assert pool._generation == 2
        finally:
            pool.shutdown(wait=False)

    async def test_run_propagates_a_second_broken_pool(self) -> None:
        """Heals exactly once — a pool that breaks again is a real problem, not a
        retry loop."""
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        fn = MagicMock(side_effect=BrokenProcessPool("worker died"))
        try:
            with pytest.raises(BrokenProcessPool):
                await pool.run(fn, "u")

            assert fn.call_count == 2
        finally:
            pool.shutdown(wait=False)

    async def test_run_after_shutdown_raises_instead_of_resurrecting(self) -> None:
        """D1 — the defect this refactor closes.

        The module global this replaced used None as "build one", so a late caller (an
        in-flight prefetch_stream, exactly what is likely still running during close())
        silently spawned a fresh 4-worker pool that nothing would ever join. Now it
        raises, matching Executor.submit()'s own documented post-shutdown contract.
        """
        factory = MagicMock(side_effect=_thread_pool_factory())
        pool = YtdlpPool(executor_factory=factory)
        pool._acquire()
        pool.shutdown(wait=False)

        with pytest.raises(PoolClosedError):
            await pool.run(_double, 21)

        assert factory.call_count == 1, "a post-shutdown caller resurrected the pool"

    def test_pool_closed_error_is_a_runtime_error(self) -> None:
        """Subclassed deliberately: the stdlib raises RuntimeError for a submit after
        shutdown, so a handler written against the executor's contract keeps working —
        including for the accepted race where the stdlib's own error surfaces instead.
        """
        assert issubclass(PoolClosedError, RuntimeError)


class TestReplace:
    def test_replace_ignores_a_stale_executor(self) -> None:
        """Two concurrent extractions can both hit BrokenProcessPool. Only the first
        discards — otherwise the second throws away the healthy replacement the first
        just built."""
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        try:
            broken = pool._acquire()
            pool._replace(broken)
            fresh = pool._acquire()

            pool._replace(broken)  # the straggler, arriving late

            assert pool._acquire() is fresh
        finally:
            pool.shutdown(wait=False)

    def test_replace_swallows_a_failing_shutdown(self) -> None:
        """Discarding a broken pool is best-effort — a pool that is already broken may
        well fail to shut down, and that must not mask the original failure."""
        executor = MagicMock(spec=Executor)
        executor.shutdown.side_effect = OSError("already gone")
        pool = YtdlpPool(executor_factory=lambda: executor)

        pool._replace(pool._acquire())  # must not raise

        assert pool._executor is None


class TestShutdown:
    def test_shutdown_joins_and_marks_closed(self) -> None:
        executor = MagicMock(spec=Executor)
        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()

        pool.shutdown()

        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        assert pool.is_closed
        assert pool._executor is None

    def test_shutdown_is_safe_when_never_used(self) -> None:
        """No executor was ever built — closing must not construct one to close it."""
        factory = MagicMock()
        pool = YtdlpPool(executor_factory=factory)

        pool.shutdown()  # must not raise

        factory.assert_not_called()
        assert pool.is_closed

    def test_shutdown_is_idempotent(self) -> None:
        executor = MagicMock(spec=Executor)
        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()

        pool.shutdown()
        pool.shutdown()

        executor.shutdown.assert_called_once()


class TestAclose:
    def _blocking_executor(self) -> tuple[MagicMock, threading.Event, threading.Event]:
        """An executor whose join blocks until released — but only when wait=True, so
        aclose()'s abandon path (wait=False) still returns immediately."""
        started = threading.Event()
        release = threading.Event()

        def shutdown(wait: bool = True, cancel_futures: bool = False) -> None:
            if wait:
                started.set()
                release.wait(timeout=5)

        executor = MagicMock(spec=Executor)
        executor.shutdown.side_effect = shutdown
        return executor, started, release

    async def test_aclose_joins_and_marks_closed(self) -> None:
        executor = MagicMock(spec=Executor)
        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()

        await pool.aclose()

        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        assert pool.is_closed

    async def test_aclose_is_safe_when_never_used(self) -> None:
        factory = MagicMock()
        pool = YtdlpPool(executor_factory=factory)

        await pool.aclose()

        factory.assert_not_called()
        assert pool.is_closed

    async def test_aclose_marks_closed_before_awaiting_the_join(self) -> None:
        """The shape borrowed from asyncio's loop.shutdown_default_executor(): the state
        change happens on the event-loop thread, and only the blocking join goes to a
        worker thread. That is what makes the cross-thread mutation the old module global
        performed (§2.4 D2) structurally impossible rather than merely locked."""
        executor, started, release = self._blocking_executor()
        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()

        task = asyncio.create_task(pool.aclose())
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)

            # The join is still running on its thread, yet submits are already refused.
            assert pool.is_closed
            with pytest.raises(PoolClosedError):
                await pool.run(_double, 1)
        finally:
            release.set()
            await task

    async def test_aclose_abandons_a_join_that_exceeds_the_timeout(self) -> None:
        """yt-dlp's socket_timeout=30 with retries=10 can keep an extraction alive far
        longer than a bot shutdown should take, so the join is bounded and then
        abandoned rather than allowed to hang the process's exit."""
        executor, _started, release = self._blocking_executor()
        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()

        try:
            await pool.aclose(timeout=0.01)

            assert pool.is_closed
            # A thread pool (this mock's spec) has no terminate_workers, so the abandon
            # path falls back to shutdown(wait=False).
            assert executor.shutdown.call_count == 2
            assert executor.shutdown.call_args_list[-1].kwargs == {
                "wait": False,
                "cancel_futures": True,
            }
        finally:
            release.set()

    async def test_aclose_terminates_workers_when_a_process_pool_join_times_out(
        self,
    ) -> None:
        """A ProcessPoolExecutor must be actively terminated on timeout: shutdown(wait=False)
        does NOT bound interpreter exit (the abandoned join is re-joined by _python_exit at
        exit; measured 61s → 3.4s once workers are SIGTERMed, §12.3). The isinstance guard
        picks terminate_workers() for a real pool and the shutdown(wait=False) fallback for
        the thread-pool seam."""
        started = threading.Event()
        release = threading.Event()

        def shutdown(wait: bool = True, cancel_futures: bool = False) -> None:
            if wait:
                started.set()
                release.wait(timeout=5)

        # spec=ProcessPoolExecutor: passes isinstance and exposes terminate_workers, which
        # a bare MagicMock(spec=Executor) does not — that is why the older abandon test
        # exercises only the else branch.
        executor = MagicMock(spec=ProcessPoolExecutor)
        executor.shutdown.side_effect = shutdown
        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()

        try:
            await pool.aclose(timeout=0.01)

            assert pool.is_closed
            executor.terminate_workers.assert_called_once_with()
            # only the wait=True join ran; the wait=False fallback must NOT be taken
            executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        finally:
            release.set()

    async def test_worker_termination_precedes_the_log_listener_stop(self) -> None:
        """The §12.2 ordering rule: terminate_workers() THEN listener.stop(). Reversed,
        stop() drains and closes the queue while the dying workers are still emitting, so
        their final records — the reason a shutdown-time extraction failed — are lost.
        """
        order: list[str] = []
        started = threading.Event()
        release = threading.Event()

        def shutdown(wait: bool = True, cancel_futures: bool = False) -> None:
            if wait:
                started.set()
                release.wait(timeout=5)

        executor = MagicMock(spec=ProcessPoolExecutor)
        executor.shutdown.side_effect = shutdown
        executor.terminate_workers.side_effect = lambda: order.append("terminate")
        listener = MagicMock(spec=QueueListener)
        listener.stop.side_effect = lambda: order.append("stop")

        pool = YtdlpPool(executor_factory=lambda: executor)
        pool._acquire()
        # Simulate the state a real _spawn_process_pool() would have left: a listener
        # draining the worker-log queue. The custom factory above bypasses that path.
        pool._log_listener = listener
        pool._log_queue = MagicMock()

        try:
            await pool.aclose(timeout=0.01)
        finally:
            release.set()

        assert order == ["terminate", "stop"]

    async def test_aclose_stops_the_listener_even_when_the_executor_was_already_nulled(
        self,
    ) -> None:
        """A BrokenProcessPool heal (_replace) can null the executor out from under a
        concurrent aclose(), so _close() returns None. The listener the original spawn
        started is still draining the worker-log queue — aclose() must stop it anyway,
        not early-return and leak the QueueListener thread (and its queue feeder) for the
        life of the process."""
        listener = MagicMock(spec=QueueListener)
        pool = YtdlpPool(executor_factory=lambda: MagicMock(spec=ProcessPoolExecutor))
        # The state a break-heal leaves behind: _replace() already dropped the executor,
        # but the log listener from the original spawn is still running.
        pool._log_listener = listener
        pool._log_queue = MagicMock()
        assert pool._executor is None

        await pool.aclose()

        assert pool.is_closed
        listener.stop.assert_called_once_with()
        assert pool._log_listener is None


class TestPicklableCall:
    """_picklable_call — the generic net that keeps an un-shippable worker exception from
    breaking the pool (§12.1)."""

    def test_a_picklable_exception_passes_through_unchanged(self) -> None:
        def boom() -> None:
            raise ValueError("plain and picklable")

        with pytest.raises(ValueError, match="plain and picklable"):
            _picklable_call(boom)

    def test_an_unpicklable_exception_becomes_a_remotecallerror(self) -> None:
        """The one that matters: without this the unpickle fails on the parent's
        executor-manager thread and the pool breaks permanently."""

        def boom() -> None:
            raise _UnpicklableError("cannot ship me", extra=object())

        with pytest.raises(RemoteCallError) as caught:
            _picklable_call(boom)

        assert caught.value.message == "cannot ship me"
        assert caught.value.original_type == "_UnpicklableError"
        # and the substitute must itself survive the boundary
        assert pickle.loads(pickle.dumps(caught.value)).message == "cannot ship me"

    def test_a_picklable_broken_executor_is_re_raised_untouched(self) -> None:
        """BrokenExecutor is run()'s healing signal — it must NOT be converted, or the
        heal-once retry never fires."""

        def boom() -> None:
            raise BrokenProcessPool("a worker died")

        with pytest.raises(BrokenProcessPool):
            _picklable_call(boom)

    def test_even_an_unpicklable_broken_executor_is_re_raised_not_converted(
        self,
    ) -> None:
        """Pins the exclusion as load-bearing: a picklable BrokenProcessPool round-trips
        and is re-raised either way, so removing `except BrokenExecutor: raise` is
        invisible unless the broken signal is itself unshippable. It must still reach the
        caller as a BrokenExecutor (so run() heals), never as a RemoteCallError."""

        def boom() -> None:
            raise _UnpicklableBroken("a worker died", extra=object())

        with pytest.raises(BrokenExecutor):
            _picklable_call(boom)

    def test_the_result_passes_through_on_success(self) -> None:
        assert _picklable_call(_double, 21) == 42


class TestRemoteCallError:
    def test_survives_pickle_round_trip_with_its_fields(self) -> None:
        """loads(dumps(...)): a multi-arg __init__ needs an explicit __reduce__ or it
        unpickles into a TypeError on the parent side (§12.1)."""
        err = RemoteCallError("boom", "SomeWorkerError")
        back = pickle.loads(pickle.dumps(err))

        assert isinstance(back, RemoteCallError)
        assert str(back) == "boom"
        assert back.message == "boom"
        assert back.original_type == "SomeWorkerError"


class TestWorkerInit:
    def test_worker_init_configures_worker_logging(self) -> None:
        sentinel_queue = MagicMock(name="log-queue")
        with patch("src.ytdlp_pool.configure_worker_logging") as configure:
            _worker_init(sentinel_queue)

        # the queue reaches the worker's logging setup so its records go to the parent
        configure.assert_called_once_with(sentinel_queue)

    def test_worker_init_swallows_a_failing_configure(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An initializer that raises breaks the pool *and every rebuild of it* (the
        stdlib contract, verified on 3.14.6), so heal-once cannot recover. Degrading to
        unstructured worker logs beats bricking all extraction in the process.

        Reported on stderr rather than through log.*, because what just failed is the
        logging configuration.
        """
        with patch(
            "src.ytdlp_pool.configure_worker_logging",
            side_effect=RuntimeError("structlog exploded"),
        ):
            _worker_init()  # must not raise

        assert "yt-dlp worker logging setup failed" in capsys.readouterr().err

    def test_warmup_noop_returns_nothing(self) -> None:
        assert _warmup_noop() is None


class TestRealWorkerProcess:
    """The few tests that spawn real worker processes.

    Everything else here uses a thread-pool seam, so nothing else asserts that the
    production path — spawn, run the initializer, pickle the arguments and the result (or
    the *exception*), ship them back, route the worker's logs to the parent, join —
    actually works. These do, end to end.

    Kept to the minimum, deliberately. Under `spawn` each worker re-imports the pytest
    entry point, conftest and this module (discord, fakeredis, structlog, OTel), which
    costs ~1 s per test here. Each of the three asserts something no cheap tier can, and
    that Option B (§12.2) made worth the spawn:
      * the callable runs across a real boundary and a late submit is refused (D1);
      * a worker's ExtractionError survives pickling with its fields and cause (§12.1) —
        the exact thing the autouse thread-pool seam can never exercise, because it never
        pickles an exception;
      * a worker's log record reaches a parent handler carrying worker_id and the
        propagated trace_id (§12.2).
    Picklability of *arguments* stays with TestProcessBoundaryContract; orphan reaping
    stays a manual gate (docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §7).
    """

    async def test_a_real_worker_process_runs_the_submitted_callable(self) -> None:
        pool = YtdlpPool(max_workers=1)
        try:
            pool.prewarm()

            worker_pid, result = await pool.run(_double_in_worker, 21)

            assert result == 42
            assert worker_pid != os.getpid(), "ran in-process — no worker was spawned"
        finally:
            await pool.aclose()

        assert pool.is_closed
        # D1, against real processes: a late submit is refused rather than served by a
        # freshly spawned pool that nothing would join.
        with pytest.raises(PoolClosedError):
            await pool.run(_double, 21)

    async def test_a_worker_extraction_error_survives_the_real_boundary(self) -> None:
        """The defect the branch shipped (§12.1): a worker's yt-dlp error reaches the
        parent as a flat ExtractionError with its fields, not an opaque pickling error,
        and the original is preserved as __cause__ (a _RemoteTraceback once it crosses).
        """
        from src.youtube import ExtractionError

        pool = YtdlpPool(max_workers=1)
        try:
            with pytest.raises(ExtractionError) as caught:
                await pool.run(_raise_classified_error_in_worker, None)
        finally:
            await pool.aclose()

        err = caught.value
        assert err.original_type == "DownloadError"
        assert err.video_id == "vid42"
        assert err.expected is True
        assert "Video unavailable" in err.message
        # the real worker traceback came back attached, stringified by the stdlib
        assert err.__cause__ is not None
        assert "DownloadError" in str(err.__cause__)

    async def test_the_returned_info_dict_survives_the_real_boundary_only_slimmed(
        self,
    ) -> None:
        """The success-path return value (§ Finding 1): a raw process=True info dict
        carries live/oversized values the pool cannot pickle back, so returning it fails
        the call with a pickling error — while _slim_info's result crosses intact. Both on
        one pool, which also proves a result-queue pickling failure does not brick it (the
        stdlib re-delivers the error on the future; the result thread survives)."""
        pool = YtdlpPool(max_workers=1)
        try:
            # Raw: the pool pickles the result synchronously; the unpicklable field comes
            # back as an exception on the future, not an opaque BrokenProcessPool.
            with pytest.raises((TypeError, pickle.PicklingError, RemoteCallError)):
                await pool.run(_return_raw_info_from_worker, None)

            # Same worker pool still serves the slimmed result — the fix, end to end.
            slim = await pool.run(_return_slimmed_info_from_worker, None)
        finally:
            await pool.aclose()

        assert isinstance(slim, dict)
        assert slim["webpage_url"] == "https://www.youtube.com/watch?v=test"
        assert slim["url"].startswith("https://r2.googlevideo.com/")
        assert "formats" not in slim and "thumbnails" not in slim

    async def test_worker_logs_reach_the_parent_with_worker_id_and_trace_id(
        self,
    ) -> None:
        """Option B (§12.2): worker records travel the queue to a parent handler carrying
        worker_id (bound in _worker_init) and the trace_id run() propagated from the active
        span — so a failed -play's worker diagnostics land in Loki, correlated."""
        import logging

        from opentelemetry import trace as ot_trace
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        captured: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        cap = _Capture()
        trace_id = 0x4BF92F3577B34DA6A3CE929D0E0E4736
        span_id = 0x00F067AA0BA902B7
        marker = "yt-dlp: SABR-only experiment detected [marker-6]"

        # Present on root BEFORE the spawn: the listener captures root.handlers at start.
        logging.root.addHandler(cap)
        pool = YtdlpPool(max_workers=1)
        try:
            pool.prewarm()
            span = NonRecordingSpan(
                SpanContext(
                    trace_id,
                    span_id,
                    is_remote=False,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            )
            with ot_trace.use_span(span, end_on_exit=False):
                await pool.run(_log_warning_in_worker, marker)
            # aclose terminates the worker, THEN stops the listener, draining the record.
            await pool.aclose()
        finally:
            logging.root.removeHandler(cap)

        matching = [r for r in captured if "marker-6" in r.getMessage()]
        assert matching, "worker log never reached the parent handler"
        body = matching[0].getMessage()
        assert "SpawnProcess" in body, f"worker_id missing from: {body}"
        assert format(trace_id, "032x") in body, f"trace_id missing from: {body}"


class TestDefaults:
    def test_worker_count_defaults_from_the_environment(self) -> None:
        """YTDLP_POOL_WORKERS is read once at import; the constructor default carries
        it, so a pool built with no arguments is the env-configured one."""
        import src.ytdlp_pool as module

        pool = YtdlpPool()

        assert pool._max_workers == module._DEFAULT_WORKERS

    def test_explicit_worker_count_overrides_the_default(self) -> None:
        assert YtdlpPool(max_workers=7)._max_workers == 7

    def test_a_fresh_pool_is_open_and_empty(self) -> None:
        pool = YtdlpPool()
        assert not pool.is_closed
        assert pool._executor is None
