"""The process pool that renders `-analytics` charts. A module of its own so
main.py, debug.py and conftest.py have a name to reach the pool by while
analytics_render stays free of module-level state; it imports only ytdlp_pool,
so a re-importing worker drags in nothing. A process, not a thread: figure
construction holds the GIL, and so does discord.py's audio player thread. No
idle reaper. See docs/ARCHITECTURE.md#analytics-rendering."""

import importlib.util
from functools import cache
from typing import Final

from src.ytdlp_pool import YtdlpPool

# Resolve per call: a from-import captures the object and misses conftest's
# thread-backed replacement.
chart_pool: Final[YtdlpPool] = YtdlpPool(max_workers=1, name="chart render")


def _warm_worker() -> None:
    """Pay matplotlib's import and one throwaway rasterization IN THE WORKER.
    Module-level so it is picklable. Renders rather than merely importing: the
    first savefig warms the Agg backend and the font manager on top."""
    from src.analytics_render import render_dashboard
    from src.guild_state import AnalyticsMetrics

    render_dashboard(AnalyticsMetrics())


def warm() -> None:
    """Spawn the chart worker and warm it now, from setup_hook. Called only
    while the archive is enabled: -analytics is gated on it, and a default
    deployment must not pay for a resident worker it cannot use."""
    if not chart_available():
        return
    chart_pool.prewarm(_warm_worker)


@cache
def chart_available() -> bool:
    """Whether matplotlib is importable, WITHOUT importing it (find_spec is a
    finder lookup). Checked in the parent: deferring would spawn a worker only
    to fail on the import and stay resident. Cached: it sits on a command path."""
    return importlib.util.find_spec("matplotlib") is not None
