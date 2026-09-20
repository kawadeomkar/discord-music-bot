"""The chart renderer's process pool.

Its coverage used to be scattered across test_ytdlp_pool, test_main, test_debug and
test_analytics_card, so no one file answered "what is asserted about the chart pool".
The lifecycle it owns is small but load-bearing: a worker is resident for the life of
any archive-enabled process, and the only thing standing between a broken renderer and
a silently chart-less deployment is what `warm()` submits.
"""

from unittest.mock import MagicMock, patch

import pytest

from src import chart_pool
from src.guild_state import AnalyticsMetrics
from src.ytdlp_pool import YtdlpPool


class TestWarmCallable:
    """`_warm_worker` is submitted by name and never called by the tests that assert
    the submission, so its BODY has to be pinned separately — emptying it leaves the
    startup warm forking a worker that warms nothing, and every assertion about
    prewarm still passes."""

    def test_the_warm_callable_actually_renders(self) -> None:
        with patch("src.analytics_render.render_dashboard") as render:
            chart_pool._warm_worker()
        render.assert_called_once()
        # An empty aggregate: the point is paying matplotlib's import and the first
        # savefig, not producing anything.
        assert isinstance(render.call_args.args[0], AnalyticsMetrics)

    def test_warm_submits_that_callable_to_the_pool(self) -> None:
        pool = MagicMock(spec=YtdlpPool)
        with patch.object(chart_pool, "chart_pool", pool):
            chart_pool.warm()
        pool.prewarm.assert_called_once_with(chart_pool._warm_worker)


class TestChartAvailable:
    def test_it_reports_matplotlib_without_importing_it(self) -> None:
        """find_spec is a finder lookup: for a dotless top-level name it resolves the
        module without executing it, which is what keeps ~1.3 s off the bot process."""
        chart_pool.chart_available.cache_clear()
        try:
            assert chart_pool.chart_available() is True
        finally:
            chart_pool.chart_available.cache_clear()

    def test_a_missing_matplotlib_reads_as_unavailable(self) -> None:
        chart_pool.chart_available.cache_clear()
        try:
            with patch("importlib.util.find_spec", return_value=None):
                assert chart_pool.chart_available() is False
        finally:
            chart_pool.chart_available.cache_clear()


class TestRealWorkerProcess:
    """The chart pool's production path, which the autouse thread seam hides
    everywhere else. conftest's docstring promised this test existed; it did not.

    One test, because it pays a real spawn plus a matplotlib import. It covers what
    the seam structurally cannot: that a spawned interpreter can import
    src.analytics_render, that Agg works there under the runtime MPLCONFIGDIR, that
    an AnalyticsMetrics pickles in and PNG bytes pickle back.
    """

    @pytest.mark.timeout(120)
    async def test_a_real_worker_renders_and_ships_the_png_back(self) -> None:
        from src.analytics_render import render_dashboard

        pool = YtdlpPool(max_workers=1, name="chart render")
        try:
            png = await pool.run(render_dashboard, AnalyticsMetrics())
        finally:
            await pool.aclose()
        assert png.startswith(b"\x89PNG"), "not a PNG — the boundary mangled it"
        assert len(png) > 10_000
