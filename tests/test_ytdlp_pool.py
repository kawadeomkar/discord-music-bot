"""Tests for src/ytdlp_pool.py — the extraction pool's lifecycle.

Almost everything drives a YtdlpPool built with an `executor_factory` seam, spawning
no worker process: the class owns lifecycle, not extraction, and its logic is about
*when* an executor is built, replaced and closed. TestRealWorkerProcess is the one
exception — it spawns for real; see its docstring."""

import asyncio
import multiprocessing
import os
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import (
    BrokenExecutor,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from concurrent.futures.process import BrokenProcessPool
from logging.handlers import QueueListener
from queue import Empty
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.ytdlp_pool import (
    _PROGRESS_JOIN_SECS,
    _PROGRESS_QUEUE_MAX,
    PoolClosedError,
    RemoteCallError,
    YtdlpPool,
    _call_with_context,
    _picklable_call,
    _MAX_TASKS_PER_CHILD,
    _pool_context,
    _warmup_noop,
    _worker_init,
    worker_progress_queue,
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
    the trap that bricks the pool if it reaches the result queue. Module-level so
    dumps() can resolve the class by reference and get far enough to fail on loads()."""

    def __init__(self, message: str, extra: object) -> None:
        super().__init__(message)
        self.extra = extra


class _UnpicklableBroken(BrokenExecutor):
    """A BrokenExecutor that itself cannot cross the boundary. The real BrokenProcessPool
    is picklable, so only a synthetic one like this can prove the exclusion in
    _picklable_call is needed."""

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


def _emit_progress_in_worker(payload: tuple[str, int, int]) -> int:
    """Runs in a real worker: put one message on the progress queue the way
    youtube._count_entry does, so the parent's drain thread can prove it arrives.
    """
    queue = worker_progress_queue()
    assert queue is not None, "the initializer never handed the worker a queue"
    queue.put_nowait(payload)
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
        Patched rather than constructed: this asserts the wiring (worker count,
        hardened initializer), not that multiprocessing works."""
        pool = YtdlpPool(max_workers=3)
        sentinel = MagicMock(name="ProcessPoolExecutor-instance")

        with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=sentinel) as ctor:
            executor = pool._acquire()

        try:
            assert executor is sentinel
            # Wiring, not multiprocessing: the hardened initializer plus the worker-log
            # queue handed to it via initargs (Option B).
            ctor.assert_called_once_with(
                max_workers=3,
                initializer=_worker_init,
                initargs=(pool._log_queue, None),
                max_tasks_per_child=_MAX_TASKS_PER_CHILD,
                mp_context=ANY,
            )
            # The context is passed EXPLICITLY, and it is the platform's own default.
            # Omit it and CPython silently forces spawn wherever a task budget is set,
            # which on Linux replaces 3.14's forkserver default and measured 23-30x
            # slower worker startup — invisible on macOS, where spawn is the default
            # anyway, so only an assertion on the passed value can catch it.
            assert (
                ctor.call_args.kwargs["mp_context"].get_start_method()
                == multiprocessing.get_start_method()
            )
            # the real spawn path starts a listener to drain that queue into the parent
            assert pool._log_listener is not None
            # No progress_sink, so no second queue and no drain thread: the chart
            # pool and every test pool build exactly what they did before.
            assert pool._progress_queue is None
            assert pool._progress_thread is None
        finally:
            pool.shutdown(wait=False)

    def test_a_pool_that_opts_out_is_given_no_task_budget(self) -> None:
        """The chart pool's one worker pays matplotlib's import and a warm render on
        every replacement, so it is built without a budget. The context still goes
        in: it is the platform's own default either way."""
        pool = YtdlpPool(max_workers=1, name="chart render", recycle_workers=False)
        with patch("src.ytdlp_pool.ProcessPoolExecutor") as ctor:
            pool._acquire()
        try:
            assert ctor.call_args.kwargs["max_tasks_per_child"] is None
        finally:
            pool.shutdown(wait=False)

    def test_the_chart_pool_opts_out(self) -> None:
        """Reloaded, because conftest swaps the module's pool for a thread-backed
        one: what is asserted is the pool the module itself builds."""
        import importlib

        from src import chart_pool

        swapped = chart_pool.chart_pool
        try:
            built = importlib.reload(chart_pool).chart_pool
            assert built._recycle_workers is False
        finally:
            setattr(chart_pool, "chart_pool", swapped)

    def test_the_pool_context_never_uses_fork(self) -> None:
        """`fork` is rejected outright by max_tasks_per_child, and forking a
        multi-threaded asyncio process is unsafe regardless — so a host whose default
        is fork falls to forkserver rather than failing to build a pool at all."""
        with patch(
            "src.ytdlp_pool.multiprocessing.get_start_method", return_value="fork"
        ):
            assert _pool_context().get_start_method() in ("forkserver", "spawn")

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
        """N threads racing into _acquire() must build one executor, not N. The
        predecessor skipped the lock on the claim that all mutation happened on the
        event-loop thread, untrue of its shutdown path; a barrier is what proves the
        fix, releasing every thread into _acquire() at once."""
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
            # Through the picklable-error net, like every run() submission: a warm
            # that raises a yt-dlp error would otherwise fail to unpickle in the
            # parent's result thread and brick the pool.
            assert call.args[0] is _call_with_context
            assert call.args[2] is _warmup_noop

    def test_prewarm_submits_the_warm_up_the_caller_supplies(self) -> None:
        """Lifecycle is all this module owns — what a worker warms is the caller's,
        like every run() callable. src.youtube passes a YoutubeDL construction."""
        from concurrent.futures import ProcessPoolExecutor

        def warm() -> None:
            return None

        executor = MagicMock(spec=ProcessPoolExecutor)
        pool = YtdlpPool(max_workers=2, executor_factory=lambda: executor)

        pool.prewarm(warm)

        assert executor.submit.call_count == 2
        for call in executor.submit.call_args_list:
            assert call.args[0] is _call_with_context
            assert call.args[2] is warm

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
        """The constraint the whole API shape exists to satisfy: ~29 tests in
        test_youtube.py patch src.youtube._ytdlp_extract, which works only because
        run() takes the callable as a *parameter*, resolved from the caller's module
        per call. Capturing it in __init__ makes every one of those patches miss."""
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
        """The module global this replaced read None as "build one", so a late caller
        (an in-flight prefetch_stream, likely still running during close()) silently
        spawned a fresh 4-worker pool nothing would ever join. Now it raises, matching
        Executor.submit()'s documented post-shutdown contract."""
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
        performed structurally impossible rather than merely locked."""
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
        """A ProcessPoolExecutor must be actively terminated on timeout: shutdown(
        wait=False) does not bound interpreter exit — _python_exit re-joins the
        abandoned pool (measured 61s → 3.4s once workers are SIGTERMed). The
        isinstance guard picks terminate_workers() over the thread-pool fallback."""
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
            # Only the wait=True join ran; the wait=False fallback must not be taken
            executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        finally:
            release.set()

    async def test_worker_termination_precedes_the_log_listener_stop(self) -> None:
        """Ordering rule: terminate_workers() THEN listener.stop(). Reversed,
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
        concurrent aclose(), so _close() returns None while the original spawn's
        listener still drains the worker-log queue. aclose() must stop it anyway, not
        early-return and leak the QueueListener thread for the life of the process."""
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
    breaking the pool."""

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
        """BrokenExecutor is run()'s healing signal — it must not be converted, or the
        heal-once retry never fires."""

        def boom() -> None:
            raise BrokenProcessPool("a worker died")

        with pytest.raises(BrokenProcessPool):
            _picklable_call(boom)

    def test_even_an_unpicklable_broken_executor_is_re_raised_not_converted(
        self,
    ) -> None:
        """Pins the exclusion: a picklable BrokenProcessPool round-trips
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
        unpickles into a TypeError on the parent side."""
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
        """An initializer that raises breaks the pool *and every rebuild of it*
        (stdlib contract, verified on 3.14.6), so heal-once cannot recover:
        unstructured worker logs beat bricking all extraction. Reported on stderr
        because what just failed is the logging configuration."""
        with patch(
            "src.ytdlp_pool.configure_worker_logging",
            side_effect=RuntimeError("structlog exploded"),
        ):
            _worker_init()  # must not raise

        # Not "yt-dlp worker": this is the one message that bypasses the structured
        # pipeline (what failed IS the logging setup), so it is printed by both pools
        # and naming one of them there is how a chart failure reads as an extraction.
        assert "worker logging setup failed" in capsys.readouterr().err

    def test_warmup_noop_returns_nothing(self) -> None:
        assert _warmup_noop() is None


class TestRealWorkerProcess:
    """The only tests that spawn real worker processes.

    Everything else uses a thread-pool seam, so nothing else asserts the production
    path end to end (spawn, initializer, pickle arguments and result, ship back,
    route worker logs to the parent, join). Kept to three, ~1 s of re-imports each:
    a callable crossing a real boundary with a late submit refused; an
    ExtractionError surviving pickling with its fields and cause, which the seam can
    never exercise since it never pickles an exception; a worker log record reaching
    a parent handler with worker_id and trace_id; and worker recycling, which is a
    property of the real executor and invisible to the seam. Argument picklability
    stays with TestProcessBoundaryContract."""

    async def test_a_worker_is_recycled_after_its_task_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_tasks_per_child is what turns "a worker grew until the OS killed it"
        into routine turnover, and only a real executor honours it — the thread-pool
        seam ignores the argument entirely. Budget of 1 so the test costs one extra
        spawn rather than 64."""
        monkeypatch.setattr("src.ytdlp_pool._MAX_TASKS_PER_CHILD", 1)
        pool = YtdlpPool(max_workers=1)
        try:
            first, _ = await pool.run(_double_in_worker, 1)
            second, _ = await pool.run(_double_in_worker, 2)
        finally:
            await pool.aclose()

        assert first != second, "the worker served both tasks — it was never recycled"

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
        # Against real processes: a late submit is refused rather than served by a
        # freshly spawned pool that nothing would join.
        with pytest.raises(PoolClosedError):
            await pool.run(_double, 21)

    async def test_a_worker_extraction_error_survives_the_real_boundary(self) -> None:
        """The defect the branch shipped: a worker's yt-dlp error reaches the
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

    async def test_a_worker_progress_message_reaches_the_parent(self) -> None:
        """The one end-to-end assertion for the progress transport: the queue is
        handed to a real worker through initargs, written from inside it, and read
        by the parent's drain thread. The thread-pool seam never runs the
        initializer, so nothing else can see this."""
        received: list[object] = []
        pool = YtdlpPool(max_workers=1, progress_sink=received.append)
        try:
            worker_pid = await pool.run(_emit_progress_in_worker, ("rid", 25, 1671))
            assert worker_pid != os.getpid(), "ran in-process — no worker was spawned"
            async with asyncio.timeout(15):
                while not received:
                    await asyncio.sleep(0.02)
        finally:
            await pool.aclose()

        assert received == [("rid", 25, 1671)]

    async def test_the_returned_info_dict_survives_the_real_boundary_only_slimmed(
        self,
    ) -> None:
        """A raw process=True info dict carries live/oversized values the pool cannot
        pickle back, so returning it fails the call, while _slim_info's result crosses
        intact. Both run on one pool, which also proves a result-queue pickling
        failure does not brick it — the error is re-delivered on the future."""
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
        """Option B: worker records travel the queue to a parent handler carrying
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

        # Present on root before the spawn: the listener captures root.handlers at start.
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
        # worker_id is multiprocessing.current_process().name, whose prefix is
        # start-method dependent: SpawnProcess-N (spawn, macOS default), ForkProcess-N
        # (fork), ForkServerProcess-N (forkserver, the Linux/3.14 default under CI). Assert
        # the start-method-independent shape so the test passes wherever the pool runs.
        import json

        worker_id = json.loads(body).get("worker_id", "")
        assert "Process-" in worker_id, f"worker_id missing from: {body}"
        assert format(trace_id, "032x") in body, f"trace_id missing from: {body}"


class TestDefaults:
    def test_worker_count_defaults_from_the_environment(self) -> None:
        """YTDLP_POOL_WORKERS is read once, by config at import; the constructor
        default carries it, so a pool built with no arguments is the env-configured
        one."""
        from src import config

        pool = YtdlpPool()

        assert pool._max_workers == config.YTDLP_POOL_WORKERS

    def test_explicit_worker_count_overrides_the_default(self) -> None:
        assert YtdlpPool(max_workers=7)._max_workers == 7

    def test_a_fresh_pool_is_open_and_empty(self) -> None:
        pool = YtdlpPool()
        assert not pool.is_closed
        assert pool._executor is None


class TestPoolNaming:
    """The `name=` parameter. Only the strings are yt-dlp-specific; everything the
    class does is generic, which is what lets src.chart_pool reuse it rather than
    fork ~300 lines of lifecycle. Without the parameter a chart-render failure reads
    as an extraction failure in Loki, which is where an operator looks first."""

    def test_the_default_still_says_yt_dlp(self) -> None:
        pool = YtdlpPool(max_workers=1)
        pool.shutdown(wait=False)
        with pytest.raises(PoolClosedError, match="yt-dlp extraction"):
            pool._acquire()

    def test_a_named_pool_says_its_own_name_when_closed(self) -> None:
        pool = YtdlpPool(max_workers=1, name="chart render")
        pool.shutdown(wait=False)
        with pytest.raises(PoolClosedError, match="chart render"):
            pool._acquire()

    async def test_the_break_heal_warning_names_the_pool(self) -> None:
        """The log line an operator greps after a worker is OOM-killed. Two pools now
        emit it, and the message is the only thing distinguishing them."""
        pool = YtdlpPool(
            max_workers=1,
            executor_factory=lambda: ThreadPoolExecutor(max_workers=1),
            name="chart render",
        )
        calls = {"n": 0}

        def _break_once() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise BrokenProcessPool("worker died")
            return "ok"

        with patch("src.ytdlp_pool.log") as log:
            assert await pool.run(_break_once) == "ok"
        pool.shutdown(wait=False)
        warning = log.warning.call_args[0][0]
        assert "chart render" in warning
        assert "yt-dlp" not in warning


class TestPrewarmCallable:
    """prewarm() takes the callable because the default no-op only warms what a worker
    pays on the way UP. For yt-dlp that is the whole cost — the extractor rides the
    entry module the forkserver already holds. The chart pool's dominant cost is
    matplotlib, which nothing imports until a render runs."""

    def test_the_default_is_still_the_no_op(self) -> None:
        pool = YtdlpPool(max_workers=2, executor_factory=lambda: MagicMock())
        executor = MagicMock(spec=ProcessPoolExecutor)
        with patch.object(pool, "_acquire", return_value=executor):
            pool.prewarm()
        assert executor.submit.call_count == 2
        # Submitted through _call_with_context, so the warm callable is its third
        # argument: a yt-dlp exception raised in the worker has to be flattened on
        # the way back or it fails to unpickle and bricks the pool.
        assert executor.submit.call_args[0][0] is _call_with_context
        assert executor.submit.call_args[0][2] is _warmup_noop

    def test_a_supplied_callable_is_submitted_once_per_worker(self) -> None:
        def _warm() -> None: ...

        pool = YtdlpPool(max_workers=3, executor_factory=lambda: MagicMock())
        executor = MagicMock(spec=ProcessPoolExecutor)
        with patch.object(pool, "_acquire", return_value=executor):
            pool.prewarm(_warm)
        assert [c[0][2] for c in executor.submit.call_args_list] == [_warm] * 3

    def test_a_thread_pool_seam_submits_nothing(self) -> None:
        """The test seam is thread-backed, so there is no process to warm — and
        submitting a real matplotlib import there would pay ~1.3s inside the suite."""
        pool = YtdlpPool(
            max_workers=1, executor_factory=lambda: ThreadPoolExecutor(max_workers=1)
        )
        executor = pool._acquire()
        with patch.object(executor, "submit") as submit:
            pool.prewarm(lambda: None)
        submit.assert_not_called()
        pool.shutdown(wait=False)


class TestProgressTransport:
    """The second worker queue: created with the pool, drained on a thread, and
    stopped after the workers are gone. Only the LIFECYCLE lives here — what a
    message means is src/youtube.py's."""

    def test_no_sink_builds_no_queue_and_no_thread(self) -> None:
        """The chart pool, and every pool built without a card to feed, pays
        nothing for this."""
        pool = YtdlpPool(executor_factory=_thread_pool_factory())
        try:
            pool._acquire()
            assert pool._progress_queue is None
            assert pool._progress_thread is None
        finally:
            pool.shutdown(wait=False)

    def test_a_sink_gets_what_the_drain_reads(self) -> None:
        received: list[object] = []
        pool = YtdlpPool(max_workers=1, progress_sink=received.append)
        try:
            with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=MagicMock()):
                pool._acquire()
            queue = pool._progress_queue
            assert queue is not None
            queue.put(("rid", 25, 1671))
            deadline = time.monotonic() + 5.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            pool.shutdown(wait=False)

        assert received == [("rid", 25, 1671)]

    def test_a_sink_that_raises_does_not_kill_the_drain(self) -> None:
        seen: list[object] = []

        def _sink(message: object) -> None:
            seen.append(message)
            if len(seen) == 1:
                raise RuntimeError("subscriber blew up")

        pool = YtdlpPool(max_workers=1, progress_sink=_sink)
        try:
            with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=MagicMock()):
                pool._acquire()
            queue = pool._progress_queue
            assert queue is not None
            queue.put(("a", 1, None))
            queue.put(("b", 2, None))
            deadline = time.monotonic() + 5.0
            while len(seen) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            pool.shutdown(wait=False)

        assert seen == [("a", 1, None), ("b", 2, None)]

    def test_a_break_heal_hands_the_new_workers_a_new_queue(self) -> None:
        """A worker SIGKILLed between Queue._feed's wacquire() and wrelease() holds
        the write lock forever: _sem never drains and every put_nowait past
        _PROGRESS_QUEUE_MAX raises Full, silently, for the life of the process.
        Reusing the queue across a heal is exactly when that has happened. The
        parent drains the queue the new workers were handed, not the old one."""
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        try:
            with patch(
                "src.ytdlp_pool.ProcessPoolExecutor", return_value=MagicMock()
            ) as executor:
                first = pool._acquire()
                queue, thread = pool._progress_queue, pool._progress_thread
                pool._replace(first)
                pool._acquire()

            assert queue is not None and thread is not None
            assert pool._progress_queue is not queue
            assert executor.call_args.kwargs["initargs"][1] is pool._progress_queue
            assert pool._progress_thread is not None
            assert pool._progress_thread.is_alive()
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        finally:
            pool.shutdown(wait=False)

    def test_a_heal_does_not_wait_on_the_old_drain(self) -> None:
        """_acquire builds the executor under the pool lock, on the event loop. A
        drain stuck in its sink must not hold a heal for _PROGRESS_JOIN_SECS."""
        entered, release = threading.Event(), threading.Event()

        def _stuck(_message: object) -> None:
            entered.set()
            release.wait(timeout=10.0)

        pool = YtdlpPool(max_workers=1, progress_sink=_stuck)
        try:
            with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=MagicMock()):
                first = pool._acquire()
                assert pool._progress_queue is not None
                pool._progress_queue.put(("rid", 1, None))
                assert entered.wait(timeout=5.0)
                pool._replace(first)
                started = time.monotonic()
                pool._acquire()
                assert time.monotonic() - started < _PROGRESS_JOIN_SECS / 2
        finally:
            release.set()
            pool.shutdown(wait=False)

    def test_the_drain_stops_on_a_flag_rather_than_a_sentinel_message(self) -> None:
        """After terminate_workers() the queue's _wlock is a POSIX semaphore a
        worker SIGTERMed mid-write never released, so telling the drain to stop by
        writing to the queue could block the parent there forever."""
        queue = MagicMock()
        queue.get.side_effect = Empty
        stop = threading.Event()
        pool = YtdlpPool(progress_sink=lambda _m: None)
        thread = threading.Thread(
            target=pool._drain_progress, args=(queue, stop), daemon=True
        )
        thread.start()

        stop.set()
        thread.join(timeout=5.0)

        assert not thread.is_alive()
        queue.put.assert_not_called()
        queue.put_nowait.assert_not_called()

    def test_shutdown_stops_the_drain_thread(self) -> None:
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=MagicMock()):
            pool._acquire()
        thread = pool._progress_thread
        assert thread is not None and thread.is_alive()

        pool.shutdown(wait=False)

        thread.join(timeout=5.0)
        # The flag is what stops it, and nothing else can: close() releases the
        # parent's own handle and does not interrupt a blocked get, so a drain
        # told to stop only by its queue dying would outlive the pool. Measured —
        # this is not a backstop arrangement, it is the only mechanism.
        assert pool._progress_stop.is_set()
        assert not thread.is_alive()
        assert pool._progress_queue is None

    async def test_aclose_stops_the_drain_thread(self) -> None:
        """The production close path. shutdown() is the test seam's; a drain left
        running by aclose() outlives the pool for the life of the process."""
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        with patch("src.ytdlp_pool.ProcessPoolExecutor", return_value=MagicMock()):
            pool._acquire()
        thread = pool._progress_thread
        assert thread is not None and thread.is_alive()

        await pool.aclose(timeout=1.0)

        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert pool._progress_queue is None

    def test_the_worker_initializer_binds_the_queue_and_drops_its_join(self) -> None:
        """cancel_join_thread, because at worker exit Queue._finalize_join JOINS
        the feeder thread — blocked in send_bytes once the parent stops reading —
        so the worker cannot exit and shutdown burns its full 10s timeout."""
        queue = MagicMock()
        try:
            _worker_init(None, queue)

            assert worker_progress_queue() is queue
            queue.cancel_join_thread.assert_called_once()
        finally:
            _worker_init(None, None)

    def test_no_queue_leaves_the_worker_global_unset(self) -> None:
        _worker_init(None, None)
        assert worker_progress_queue() is None

    def test_the_progress_queue_is_bounded(self) -> None:
        """maxsize=0 is a SEM_VALUE_MAX semaphore, so put_nowait would never
        raise and a stalled drain would grow the queue without limit. The bound
        is what makes a dropped message the failure instead of memory."""
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        pool._start_progress_drain()
        try:
            assert pool._progress_queue is not None
            assert pool._progress_queue._maxsize == _PROGRESS_QUEUE_MAX
        finally:
            pool._stop_progress_drain()

    def test_a_restarted_drain_stops_the_one_before_it(self) -> None:
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        pool._start_progress_drain()
        first_queue, first_thread = pool._progress_queue, pool._progress_thread
        try:
            pool._start_progress_drain()
            assert pool._progress_queue is not first_queue
            assert pool._progress_thread is not first_thread
            assert first_thread is not None
            first_thread.join(timeout=5.0)
            assert not first_thread.is_alive()
            assert pool._progress_thread is not None
            assert pool._progress_thread.is_alive()
        finally:
            pool._stop_progress_drain()

    def test_a_drain_that_dies_unexpectedly_says_so(self) -> None:
        """This is the one failure that leaves no line anywhere: the thread
        returns, progress ends for the life of the executor, and every card after
        it shows an elapsed line instead of a bar with nothing to explain why."""
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        stop = threading.Event()

        class _Broken:
            def get(self, timeout: float) -> object:
                raise OSError("handle is gone")

        with patch("src.ytdlp_pool.log") as logger:
            pool._drain_progress(_Broken(), stop)
        assert logger.warning.called

    def test_a_drain_stopped_on_purpose_is_silent(self) -> None:
        """A closed queue at shutdown is the expected ending and says nothing —
        otherwise every restart logs a warning nobody should read."""
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        stop = threading.Event()

        class _Closing:
            def get(self, timeout: float) -> object:
                stop.set()
                raise OSError("closed at shutdown")

        with patch("src.ytdlp_pool.log") as logger:
            pool._drain_progress(_Closing(), stop)
        assert not logger.warning.called

    def test_stopping_twice_is_inert(self) -> None:
        """A heal calls it before rebuilding, so it runs on an already-stopped
        transport as a matter of course."""
        pool = YtdlpPool(max_workers=1, progress_sink=lambda _m: None)
        pool._start_progress_drain()
        pool._stop_progress_drain()
        pool._stop_progress_drain()
        assert pool._progress_queue is None
