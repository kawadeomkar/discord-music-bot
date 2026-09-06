"""The `-analytics` chart: a six-panel PNG, rendered in a worker process.

matplotlib is imported inside `build_figure`, never at module scope: the parent
imports this module only to name the callable it hands the pool. Nothing is
constructed at module scope either (golden rule 10); the pool lives in
src.chart_pool.

Only machine-minted labels reach the image: matplotlib's bundled font covers no
CJK, Thai or emoji, and a missing glyph renders as tofu silently in a worker.
`_ascii_safe()` guards every drawn string; human-authored text renders in the
embed. `query_source` is the one archive-derived string here, safe because
sources.py stores a hostname matching `[a-z0-9.-]{1,64}`; the dimension is open,
so it is capped at the top few plus a minted "other".

See docs/ARCHITECTURE.md#analytics-rendering.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

from src.guild_state import (
    WAIT_MEDIAN_INDEX,
    WAIT_PERCENTILES,
    WAIT_UNAVAILABLE,
    AnalyticsMetrics,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# 1100x792 at dpi 110 — 2x Discord's ~550px embed render, for HiDPI.
# See docs/ARCHITECTURE.md#analytics-rendering.
DPI: Final[int] = 110
FIGSIZE: Final[tuple[float, float]] = (10.0, 7.2)
PIXELS: Final[tuple[int, int]] = (
    int(FIGSIZE[0] * DPI),
    int(FIGSIZE[1] * DPI),
)

# Dark, committed to: a PNG cannot follow the viewer's theme, and dark matches the
# Discord embed surface these are pasted onto.
_BG: Final[str] = "#2b2d31"
_PANEL: Final[str] = "#232428"
_INK: Final[str] = "#e6e8eb"
_MUTED: Final[str] = "#9aa1ab"
_GRID: Final[str] = "#3a3d43"

# The reference palette's dark steps in validated slot order; the order carries
# the adjacent-pair CVD separation. See docs/ARCHITECTURE.md#analytics-rendering.
_SOURCE_COLORS: Final[tuple[str, ...]] = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
)
_OTHER_COLOR: Final[str] = "#9085e9"  # violet — the minted residual bucket
_ACCENT: Final[str] = "#3987e5"

# Sequential ramp for magnitude, darkest first so the near-zero end recedes.
_SEQUENTIAL_BLUE: Final[tuple[str, ...]] = (
    "#0d366b",
    "#104281",
    "#184f95",
    "#1c5cab",
    "#256abf",
    "#2a78d6",
    "#3987e5",
    "#5598e7",
    "#6da7ec",
    "#86b6ef",
    "#9ec5f4",
    "#b7d3f6",
    "#cde2fb",
)

# The residual bucket for every source past the cap — `query_source` is an open
# set. Five plus this is six.
OTHER_SOURCE: Final[str] = "other"
# Shown for the empty query_source every pre-archive backfill row carries.
UNKNOWN_SOURCE: Final[str] = "unknown"
MAX_SOURCES: Final[int] = len(_SOURCE_COLORS)

# The domain sources.py guarantees (normalize_query_host), plus our own bucket.
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9.-]{1,64}$")

# Room at the top for the suptitle; hspace for the per-panel titles plus panel
# 1's legend band.
_LAYOUT: Final[dict[str, float]] = {
    "left": 0.055,
    "right": 0.988,
    "top": 0.885,
    "bottom": 0.062,
    "wspace": 0.15,
    "hspace": 0.58,
}

# Fingerprints what the image looks like, so a palette or layout change
# invalidates the PNG cache, whose key otherwise digests the aggregate alone.
RENDER_VERSION: Final[str] = hashlib.blake2b(
    repr(
        (DPI, FIGSIZE, _LAYOUT, _SOURCE_COLORS, _SEQUENTIAL_BLUE, _OTHER_COLOR)
    ).encode(),
    digest_size=4,
).hexdigest()

_WEEKDAYS: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# Minted from the tuple the SQL asks for, so a label cannot name a percentile the
# query does not return.
_PCT_LABELS: Final[tuple[str, ...]] = tuple(
    f"p{int(pct * 100)}" for pct in WAIT_PERCENTILES
)


class UnsafeGlyphError(ValueError):
    """A string bound for the image that the bundled font cannot render. Raised,
    since `python -O` strips asserts and this is the only guard; the command
    routes it to the embed-only card."""


def _ascii_safe(text: str) -> str:
    """Every drawn string passes through here. ASCII is the subset of DejaVu Sans
    this module promises, checked on the character domain rather than on
    provenance, so a panel smuggling an uploader name in is caught."""
    if not text.isascii():
        raise UnsafeGlyphError(
            f"non-ASCII string bound for the chart image: {text!r} — "
            "human-authored text belongs in the embed"
        )
    return text


def fold_sources(metrics: AnalyticsMetrics) -> tuple[str, ...]:
    """The source dimension the panels draw, most plays first, capped. The tail
    folds into one minted bucket rather than being dropped, or the stacked bars
    would read as a play count lower than the topline."""
    totals: dict[str, int] = {}
    for row in metrics.daily_by_source:
        totals[row.source] = totals.get(row.source, 0) + row.plays
    # completion and source_completion index this result too, so their sources
    # are seeded here and every panel's lookup resolves inside `kept`.
    for bucket in metrics.completion:
        totals.setdefault(bucket.source, 0)
    for entry in metrics.source_completion:
        totals.setdefault(entry.source, 0)
    ranked = sorted(totals, key=lambda s: (-totals[s], s))
    kept = tuple(ranked[:MAX_SOURCES])
    # OTHER_SOURCE can also be a real query_source (normalize_query_host accepts a
    # dotless label); appending it blindly would draw that segment twice.
    if len(ranked) > MAX_SOURCES and OTHER_SOURCE not in kept:
        return kept + (OTHER_SOURCE,)
    return kept


def _source_label(source: str, kept: tuple[str, ...]) -> str:
    """Which legend entry a raw source belongs to. A value outside the domain
    folds into "other" rather than raising: a mis-stamped row must cost a legend
    entry, not the whole chart."""
    if source in kept:
        return source
    if _SLUG_RE.match(source):
        return OTHER_SOURCE if OTHER_SOURCE in kept else source
    return OTHER_SOURCE


def _legend_label(source: str) -> str:
    """`''` is a real archived value (every backfill row), and matplotlib reads an
    empty label as NO label, so the legend would render empty and warn."""
    return source or UNKNOWN_SOURCE


def _color_for(source: str, kept: tuple[str, ...]) -> str:
    if source == OTHER_SOURCE:
        return _OTHER_COLOR
    try:
        return _SOURCE_COLORS[kept.index(source)]
    except ValueError:
        return _OTHER_COLOR


def _style(ax: Any, title: str, *, ylabel: str = "", title_pad: float = 7) -> None:
    ax.set_facecolor(_PANEL)
    ax.set_title(_ascii_safe(title), color=_INK, fontsize=10, pad=title_pad, loc="left")
    if ylabel:
        ax.set_ylabel(_ascii_safe(ylabel), color=_MUTED, fontsize=8)
    ax.tick_params(colors=_MUTED, labelsize=7, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def dense_buckets(m: AnalyticsMetrics) -> tuple[list[str], list[int], list[float]]:
    """Every bucket in the window, zero-filled — labels, plays, hours. The SQL
    returns only buckets with plays, so this keeps the time axis even. Falls back
    to the aggregate's own buckets when the window is unstamped."""
    step = 7 * 86400 if m.bucket_unit == "week" else 86400
    span = m.window_end_epoch - m.window_start_epoch
    # A weekly window snaps its start back to a Monday, so it runs up to six days
    # longer than `days`.
    if m.window_start_epoch <= 0 or not 0 < span <= (m.days + 7) * 86400:
        return (
            [p.day for p in m.daily],
            [p.plays for p in m.daily],
            [p.listen_secs / 3600 for p in m.daily],
        )
    known = {p.day: p for p in m.daily}
    labels: list[str] = []
    plays: list[int] = []
    hours: list[float] = []
    edge = m.window_start_epoch
    while edge < m.window_end_epoch:
        day = datetime.fromtimestamp(edge, tz=timezone.utc).strftime("%Y-%m-%d")
        point = known.get(day)
        labels.append(day)
        plays.append(point.plays if point else 0)
        hours.append((point.listen_secs if point else 0) / 3600)
        edge += step
    return labels, plays, hours


def _tick_stride(count: int, target: int = 10) -> int:
    """About `target` x labels however long the series is."""
    return max(1, -(-count // target))


def _panel_daily_by_source(
    ax: Any, m: AnalyticsMetrics, kept: tuple[str, ...], days: list[str]
) -> None:
    index = {d: i for i, d in enumerate(days)}
    stacks: dict[str, list[float]] = {s: [0.0] * len(days) for s in kept}
    for row in m.daily_by_source:
        if row.day in index:
            stacks[_source_label(row.source, kept)][index[row.day]] += row.plays
    from matplotlib.collections import PolyCollection

    # One collection per source. Collections do not autoscale, hence the limits
    # below. See docs/ARCHITECTURE.md#analytics-rendering.
    half = 0.82 / 2
    bottom = [0.0] * len(days)
    for source in kept:
        heights = stacks[source]
        ax.add_collection(
            PolyCollection(
                [
                    [(x - half, b), (x - half, b + h), (x + half, b + h), (x + half, b)]
                    for x, (b, h) in enumerate(zip(bottom, heights))
                ],
                facecolors=_color_for(source, kept),
                # A surface-coloured hairline so two adjacent hues never share an edge.
                edgecolors=_PANEL,
                linewidths=0.4,
                label=_ascii_safe(_legend_label(source)),
            )
        )
        bottom = [b + v for b, v in zip(bottom, heights)]
    if days:
        # add_collection feeds the data limits without applying them, and a
        # collection has no sticky edge at y=0.
        ax.autoscale_view()
        ax.set_ylim(bottom=0)
    unit = "week" if m.bucket_unit == "week" else "day"
    # Extra title pad: the legend sits between the title and the axes.
    _style(
        ax,
        f"Plays per {unit}, by source",
        ylabel=f"plays / {unit}",
        title_pad=17 if kept else 7,
    )
    stride = _tick_stride(len(days))
    ax.set_xticks(range(0, len(days), stride))
    ax.set_xticklabels(
        [_ascii_safe(days[i][5:]) for i in range(0, len(days), stride)],
        rotation=0,
    )
    if kept:
        legend = ax.legend(
            fontsize=6.5,
            ncol=max(len(kept), 1),
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            handlelength=0.9,
            handletextpad=0.4,
            columnspacing=0.9,
        )
        for text in legend.get_texts():
            text.set_color(_MUTED)


def _panel_heatmap(ax: Any, m: AnalyticsMetrics, fig: Figure) -> None:
    grid = [[0 for _ in range(24)] for _ in range(7)]
    for cell in m.heat:
        if 1 <= cell.dow <= 7 and 0 <= cell.hour <= 23:
            grid[cell.dow - 1][cell.hour] += cell.plays
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    cmap = LinearSegmentedColormap.from_list(
        "analytics_seq", _SEQUENTIAL_BLUE
    ).with_extremes(under=_PANEL)
    # An empty hour is absence: vmin sits above zero and with_extremes catches
    # what falls below. Both bounds explicit, so the scale cannot degenerate.
    peak = max((max(row) for row in grid), default=0)
    image = ax.imshow(
        grid,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0.5,
        vmax=max(1.0, float(peak)),
    )
    _style(ax, "When this server listens (UTC)")
    ax.grid(False)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    ax.set_yticks(range(7))
    ax.set_yticklabels([_ascii_safe(d) for d in _WEEKDAYS])
    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045)
    # Integer ticks: the scale counts plays, and the sub-1 vmin would otherwise
    # put fractional labels on a count.
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    bar.ax.tick_params(colors=_MUTED, labelsize=6, length=0)
    bar.outline.set_visible(False)


def _panel_listening_time(
    ax: Any, m: AnalyticsMetrics, days: list[str], hours: list[float]
) -> None:
    ax.plot(range(len(days)), hours, color=_ACCENT, linewidth=2)
    ax.fill_between(range(len(days)), hours, color=_ACCENT, alpha=0.22)
    unit = "week" if m.bucket_unit == "week" else "day"
    _style(ax, f"Listening time per {unit}", ylabel="hours")
    stride = _tick_stride(len(days))
    ax.set_xticks(range(0, len(days), stride))
    ax.set_xticklabels([_ascii_safe(days[i][5:]) for i in range(0, len(days), stride)])
    ax.set_ylim(bottom=0)


def _panel_completion(ax: Any, m: AnalyticsMetrics, kept: tuple[str, ...]) -> None:
    """Per-play completion, grouped by source because the distributions differ in
    shape even where their means converge. The duration-weighted ratio answers a
    different question and lives on the song-length panel."""
    series: dict[str, list[float]] = {s: [0.0] * 10 for s in kept}
    for row in m.completion:
        if 1 <= row.bucket <= 10:
            series[_source_label(row.source, kept)][row.bucket - 1] += row.plays
    width = 0.8 / max(len(kept), 1)
    for i, source in enumerate(kept):
        ax.bar(
            [b + i * width for b in range(10)],
            series[source],
            width=width * 0.9,  # a real gap between adjacent fills
            label=_ascii_safe(_legend_label(source)),
            color=_color_for(source, kept),
        )
    excluded = (
        f" ({m.livestream_plays} livestream excluded)" if m.livestream_plays else ""
    )
    _style(ax, f"How much of each song plays{excluded}", ylabel="plays")
    ax.set_xticks([b + 0.4 - width / 2 for b in range(10)])
    ax.set_xticklabels([f"{(b + 1) * 10}" for b in range(10)])
    ax.set_xlabel(_ascii_safe("% of the song played"), color=_MUTED, fontsize=8)


def _panel_durations(ax: Any, m: AnalyticsMetrics, kept: tuple[str, ...]) -> None:
    """Song length, annotated with the duration-weighted completion per source —
    a handful of abandoned hour-long mixes dominates that number, which is why it
    belongs on the length axis."""
    counts = [0] * 21
    for row in m.durations:
        if 0 <= row.minutes <= 20:
            counts[row.minutes] += row.plays
    ax.bar(range(21), counts, width=0.85, color=_ACCENT)
    _style(ax, "Song length mix", ylabel="plays")
    ax.set_xticks(range(0, 21, 4))
    ax.set_xticklabels(["0", "4", "8", "12", "16", "20+"])
    ax.set_xlabel(_ascii_safe("minutes"), color=_MUTED, fontsize=8)
    weighted = [
        (s.source, 100.0 * s.played_secs / s.duration_secs)
        for s in m.source_completion
        if s.duration_secs > 0
    ]
    if weighted:
        # Headroom, so the note never lands on a bar.
        ax.set_ylim(top=max(max(counts), 1) * 1.28)
        # Summed by label before ranking, so a folded group reports once. Clamped
        # at 100: nothing constrains played_secs <= duration_secs.
        merged: dict[str, tuple[float, float]] = {}
        for entry in m.source_completion:
            if entry.duration_secs > 0:
                label = _legend_label(_source_label(entry.source, kept))
                played, total = merged.get(label, (0.0, 0.0))
                merged[label] = (
                    played + entry.played_secs,
                    total + entry.duration_secs,
                )
        ranked = sorted(
            ((k, min(100.0, 100.0 * p / d)) for k, (p, d) in merged.items()),
            key=lambda pair: -pair[1],
        )
        note = "  ".join(f"{k} {pct:.0f}%" for k, pct in ranked[:3])
        ax.text(
            0.99,
            0.94,
            _ascii_safe(f"of all queued time played: {note}"),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=_MUTED,
        )


def _fmt_secs(value: float) -> str:
    """A wait, in the largest unit that keeps it short."""
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.0f}m"
    return f"{value / 3600:.1f}h"


def _panel_wait(ax: Any, m: AnalyticsMetrics) -> None:
    """A percentile strip: box p25-p75, median at p50, whiskers to p10 and p90.
    Percentiles, never a mean: one song queued overnight skews a mean by orders
    of magnitude."""
    # No ylabel: one row, and the units belong on the value axis, which is x.
    _style(ax, "Queue wait")
    if len(m.wait_pcts) != len(WAIT_PERCENTILES) or m.wait_p50_secs <= WAIT_UNAVAILABLE:
        ax.text(
            0.5,
            0.5,
            _ascii_safe("no queue-wait data in this window"),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=_MUTED,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        return
    p10, p25, p50, p75, p90 = m.wait_pcts
    # bxp, not boxplot: the percentiles are computed server-side, and boxplot()
    # would want every raw wait shipped from Postgres.
    ax.bxp(
        [
            {
                "label": "",
                "whislo": p10,
                "q1": p25,
                "med": p50,
                "q3": p75,
                "whishi": p90,
                "fliers": [],
            }
        ],
        orientation="horizontal",
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": _ACCENT, "edgecolor": _ACCENT, "alpha": 0.55},
        medianprops={"color": _INK, "linewidth": 1.8},
        whiskerprops={"color": _MUTED, "linewidth": 1.2},
        capprops={"color": _MUTED, "linewidth": 1.2},
    )
    ax.set_yticks([])
    ax.grid(True, axis="x", color=_GRID, linewidth=0.6, alpha=0.7)
    ax.grid(False, axis="y")
    ax.set_xlabel(_ascii_safe("seconds"), color=_MUTED, fontsize=8)
    ax.set_xlim(left=0)
    # Only the median is labelled — the x-axis carries the rest, and on a skewed
    # distribution end labels overlap.
    ax.annotate(
        _ascii_safe(f"{_PCT_LABELS[WAIT_MEDIAN_INDEX]} {_fmt_secs(p50)}"),
        xy=(p50, 1.0),
        xytext=(0, 22),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=_INK,
    )
    # bxp's default whiskers are 1.5*IQR and these are not; say so.
    ax.text(
        0.99,
        0.93,
        _ascii_safe("box p25-p75  whiskers p10-p90"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=_MUTED,
    )
    ax.set_ylim(0.6, 1.8)


def build_figure(metrics: AnalyticsMetrics) -> Figure:
    """The six panels as a Figure, split from rasterization so tests can inspect
    one construction. `matplotlib.figure.Figure` directly, never pyplot: pyplot
    keeps every figure in a global registry until closed, a leak in a long-lived
    worker."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    kept = fold_sources(metrics)
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=_BG)
    axes = fig.subplots(3, 2)
    # A fixed layout, applied BEFORE the panels: fig.colorbar(ax=...) takes its
    # space from the parent axes at creation. The grid is always 3x2 at one
    # figsize with one font, so the solve has one answer and these numbers are it.
    # See docs/ARCHITECTURE.md#analytics-rendering.
    fig.subplots_adjust(**_LAYOUT)
    # Solved once for both panels, which must agree on the axis anyway.
    days, _, hours = dense_buckets(metrics)
    _panel_daily_by_source(axes[0][0], metrics, kept, days)
    _panel_heatmap(axes[0][1], metrics, fig)
    _panel_listening_time(axes[1][0], metrics, days, hours)
    _panel_completion(axes[1][1], metrics, kept)
    _panel_durations(axes[2][0], metrics, kept)
    _panel_wait(axes[2][1], metrics)
    fig.suptitle(
        _ascii_safe(f"{metrics.period_label.capitalize()} - times in UTC"),
        color=_MUTED,
        fontsize=8,
        x=0.012,
        ha="left",
        y=0.985,
    )
    return fig


def render_dashboard(metrics: AnalyticsMetrics) -> bytes:
    """Rasterize the dashboard to PNG bytes — the worker entry point. PNG, not
    JPEG: flat-colour panels with thin lines and text are the DCT pathological
    case (measured 65% larger and lower quality)."""
    import io

    figure = build_figure(metrics)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=_BG, dpi=DPI)
    return buffer.getvalue()
