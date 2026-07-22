"""Tests for src/ytdlp_pool.py — the extraction pool's lifecycle.

Almost everything here drives a YtdlpPool built with an `executor_factory` seam, so no
worker process is spawned: the class owns lifecycle, not extraction, and its logic is
about *when* an executor is built, replaced and closed. The one exception is
TestRealWorkerProcess, which spawns for real — see its docstring.
"""

import asyncio
import os
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock, patch

import pytest

from src.ytdlp_pool import (
    PoolClosedError,
    YtdlpPool,
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

        assert executor is sentinel
        ctor.assert_called_once_with(max_workers=3, initializer=_worker_init)

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
        including for the accepted race where the stdlib's own error surfaces instead."""
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
            # Abandoned: the second call gives up on the join rather than waiting.
            assert executor.shutdown.call_count == 2
            assert executor.shutdown.call_args_list[-1].kwargs == {
                "wait": False,
                "cancel_futures": True,
            }
        finally:
            release.set()


class TestWorkerInit:
    def test_worker_init_configures_worker_logging(self) -> None:
        with patch("src.ytdlp_pool.configure_worker_logging") as configure:
            _worker_init()

        configure.assert_called_once_with()

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
    """The one test that spawns a real worker process.

    Everything else here uses a thread-pool seam, so nothing else asserts that the
    production path — spawn, run the initializer, pickle the arguments, ship the result
    back, join — actually works. This does, end to end, in a single test.

    Exactly one, deliberately. Under `spawn` each worker re-imports the pytest entry
    point, conftest and this module (discord, fakeredis, structlog, OTel), which costs
    ~3 s per test here versus 0.3 s for the same work in a standalone script. A second
    such test would re-pay that to assert what the cheap tiers already cover:
    picklability by TestProcessBoundaryContract (tests/test_youtube.py), and orphan
    reaping by the manual gate (docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §7).
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
