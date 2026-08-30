"""The process pool that renders `-analytics` charts.

A module of its own, holding one name: `analytics_render` is re-imported by every
worker and golden rule 10 keeps it free of module-level state, while main.py, debug.py
and conftest.py all need a name to reach the pool by. This module imports only
ytdlp_pool, so a re-importing worker drags in nothing.

A process, not a thread: matplotlib's figure construction holds the GIL, and so does
discord.py's audio player thread. Separate from the yt-dlp pool, so a render and an
extraction do not queue behind each other.

Lazily created, then warmed from `setup_hook` on any archive-enabled deployment, so
the worker is resident for the life of the process. There is deliberately no idle
reaper. Measurements and both decisions: docs/ARCHITECTURE.md#analytics-rendering.
"""

import importlib.util
from functools import cache
from typing import Final

from src.ytdlp_pool import YtdlpPool

# Resolve this per call — `from src.chart_pool import chart_pool` captures the object
# and would miss conftest's thread-backed replacement, exactly as debug.py already
# records for the yt-dlp pool.
chart_pool: Final[YtdlpPool] = YtdlpPool(max_workers=1, name="chart render")


def _warm_worker() -> None:
    """Pay matplotlib's import and one throwaway rasterization IN THE WORKER.

    Module-level so it is picklable. It renders rather than merely importing because
    the two costs are separate: the import is ~1.3s and the first savefig warms the
    Agg backend and the font manager on top of it. An empty AnalyticsMetrics draws six
    labelled-but-empty panels, which is enough to touch all of it.
    """
    from src.analytics_render import render_dashboard
    from src.guild_state import AnalyticsMetrics

    render_dashboard(AnalyticsMetrics())


def warm() -> None:
    """Spawn the chart worker and warm it now, from setup_hook.

    Called ONLY when the history archive is enabled, because -analytics is gated on it
    — a default deployment has no way to reach this pool and must not pay for it. That
    is the narrowing that makes warming affordable: it costs one resident worker
    (~141 MB Pss, measured) on deployments that can actually run the command.

    Without it the first -analytics in a process pays ~1.3s of matplotlib import plus
    its render, in front of a user. With it the parent blocks ~21ms here (the fork; the
    forkserver is already up because yt-dlp's prewarm ran first) and the import happens
    in the worker while startup continues.
    """
    if not chart_available():
        return
    chart_pool.prewarm(_warm_worker)


@cache
def chart_available() -> bool:
    """Whether matplotlib is importable, WITHOUT importing it.

    `find_spec` is a finder lookup, so the "~1.3 s must never be paid by the bot
    process" rule still holds. Checked in the parent rather than in the worker
    because deferring it would spawn a ~173 MB process only to have it fail on the
    import — and then stay resident.

    Cached: the answer cannot change within a process, and this sits on a command
    path. matplotlib is a plain main dependency, so False means a hand-built
    environment, not a supported configuration.
    """
    return importlib.util.find_spec("matplotlib") is not None
