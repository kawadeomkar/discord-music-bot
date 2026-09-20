"""Tests for src/analytics_render.py — the six-panel chart.

Almost everything here asserts over a FIGURE rather than over pixels, and exactly one
test pays savefig: a 460 ms render times N tests wrecks a 60 s suite, which is why
conftest's autouse use_thread_chart_pool exists as well.
"""

import pickle
import struct
import subprocess
import sys
from collections.abc import Sequence
from typing import Any, cast

import pytest
from matplotlib.patches import Rectangle

from src import analytics_render
from src.analytics_render import (
    MAX_SOURCES,
    OTHER_SOURCE,
    PIXELS,
    UnsafeGlyphError,
    _ascii_safe,
    build_figure,
    dense_buckets,
    fold_sources,
    render_dashboard,
)
from src.guild_state import (
    WAIT_UNAVAILABLE,
    AnalyticsMetrics,
    CompletionBucket,
    DailyPoint,
    DurationBucket,
    HeatCell,
    SourceCompletion,
    SourceDay,
    TopArtist,
    TopListener,
    TopSong,
)


def _ANALYTICS_PCTS() -> list[float]:
    """The percentile array the SQL actually asks Postgres for."""
    import re

    from src.history_archive import _ANALYTICS_SQL

    found = re.search(r"percentile_cont\(ARRAY\[([^\]]+)\]", _ANALYTICS_SQL)
    assert found is not None
    return [float(v) for v in found.group(1).split(",")]


_TODAY = 1_787_961_600.0
_DAY = 86400.0

# Every string this module is allowed to draw, minted here rather than found in the
# data. The domain check below allows these plus anything matching _SLUG_RE.
_MINTED_WORDS = frozenset(
    """Plays per day week by source When this server listens UTC Listening time
    How much of each song plays livestream excluded of the played Song length mix
    minutes seconds Queue wait no queue-wait data in this window plays hours
    Mon Tue Wed Thu Fri Sat Sun p10 p25 p50 p75 p90 Last days weeks times all
    queued time other unknown""".split()
)


def _metrics(**over: Any) -> AnalyticsMetrics:
    base: dict[str, Any] = dict(
        days=30,
        window_start_epoch=_TODAY - 30 * _DAY,
        window_end_epoch=_TODAY,
        today_start_epoch=_TODAY,
        bucket_unit="day",
        archived_days=30,
        plays=156,
        listen_secs=26_909,
        unique_songs=118,
        unique_listeners=4,
        unique_artists=82,
        wait_p50_secs=38.0,
        livestream_plays=1,
        daily=tuple(
            DailyPoint(day=f"2026-08-{d:02d}", plays=d, listen_secs=d * 200)
            for d in range(1, 15)
        ),
        daily_by_source=tuple(
            SourceDay(day=f"2026-08-{d:02d}", source=s, plays=d)
            for d in range(1, 15)
            for s in ("search", "spotify.com")
        ),
        heat=tuple(
            HeatCell(dow=d, hour=h, plays=d * h)
            for d in range(1, 8)
            for h in (2, 14, 22)
        ),
        completion=tuple(
            CompletionBucket(source=s, bucket=b, plays=b)
            for s in ("search", "spotify.com")
            for b in range(1, 11)
        ),
        durations=tuple(DurationBucket(minutes=m, plays=21 - m) for m in range(21)),
        source_completion=(
            SourceCompletion(source="search", played_secs=20_000, duration_secs=22_000),
            SourceCompletion(
                source="spotify.com", played_secs=9_000, duration_secs=40_000
            ),
        ),
        wait_pcts=(1.0, 3.2, 38.0, 192.0, 414.0),
        top_listeners=(TopListener(requester_id=7, requester_name="Ann", plays=133),),
        top_artists=(TopArtist(uploader="Lofi Girl", plays=11),),
        top_songs=(TopSong(title="Know My Name", webpage_url="https://y/1", plays=3),),
    )
    return AnalyticsMetrics(**(base | over))


@pytest.fixture(scope="module")
def figure() -> Any:
    """Module-scoped so a dozen structural assertions share one construction. This is
    exactly why build_figure() is split from render_dashboard()."""
    return build_figure(_metrics())


class TestGlyphGuard:
    """The image's only defence. matplotlib's own missing-glyph warning is swallowed
    in production: filters do not cross a process boundary, so the spawned worker
    starts with the defaults and prints one unstructured line to raw stderr, bypassing
    the QueueHandler -> Loki plumbing entirely — measured. The PNG ships full of tofu
    and nobody finds out."""

    @pytest.mark.parametrize(
        "text",
        ["米津玄師", "아이유", "周杰倫", "เพลง", "🎵", "Café", "Кино"],
        ids=["japanese", "korean", "chinese", "thai", "emoji", "accent", "cyrillic"],
    )
    def test_non_ascii_raises(self, text: str) -> None:
        with pytest.raises(UnsafeGlyphError):
            _ascii_safe(text)

    @pytest.mark.parametrize("text", ["Plays per day", "spotify.com", "p50", "20+"])
    def test_minted_labels_pass(self, text: str) -> None:
        assert _ascii_safe(text) == text

    def test_it_raises_rather_than_asserts(self) -> None:
        """`python -O` strips asserts, and this repo already records that hazard at
        musicbot.play's HACK. An assert here would disable the guard in exactly the
        build that ships."""
        source = __import__("pathlib").Path(analytics_render.__file__).read_text()
        body = source.split("def _ascii_safe")[1].split("\ndef ")[0]
        assert "raise UnsafeGlyphError" in body
        assert "assert " not in body

    def test_a_cjk_source_cannot_reach_the_legend(self) -> None:
        """query_source is a parsed hostname accepted against [a-z0-9.-], so this is
        not reachable today — but the panel would draw whatever it is handed, and the
        guard is what makes that safe rather than lucky."""
        with pytest.raises(UnsafeGlyphError):
            build_figure(
                _metrics(
                    daily_by_source=(
                        SourceDay(day="2026-08-01", source="音楽.com", plays=1),
                    )
                )
            )


class TestDrawnStringDomain:
    def test_every_drawn_string_is_minted_or_a_source_slug(self, figure: Any) -> None:
        """The domain, never membership in a fixed list: query_source is an OPEN set
        whose values legitimately reach the legend, so a closed-list assertion would
        fail on any real fixture. This catches a future panel smuggling `uploader` in
        EVEN WHEN the test data is ASCII, which a CJK test cannot.
        """
        drawn: list[str] = []
        for ax in figure.axes:
            drawn.append(ax.get_title(loc="left"))
            drawn.append(ax.get_xlabel())
            drawn.append(ax.get_ylabel())
            drawn += [t.get_text() for t in ax.get_xticklabels()]
            drawn += [t.get_text() for t in ax.get_yticklabels()]
            drawn += [t.get_text() for t in ax.texts]
            legend = ax.get_legend()
            if legend is not None:
                drawn += [t.get_text() for t in legend.get_texts()]
        assert drawn, "no strings collected — the walk is broken"
        for text in drawn:
            assert text.isascii(), f"non-ASCII on the figure: {text!r}"
            for word in text.replace("(", " ").replace(")", " ").split():
                stripped = word.strip("%,:-/+")
                if not stripped or stripped.replace(".", "").isdigit():
                    continue
                assert stripped in _MINTED_WORDS or analytics_render._SLUG_RE.match(
                    stripped
                ), f"unminted word on the figure: {stripped!r} (from {text!r})"

    def test_no_archive_authored_name_appears(self, figure: Any) -> None:
        """The three fields the payload carries for the EMBED, asserted absent from
        the image. They ride along in the same dataclass, so nothing but this stops a
        panel reaching for one."""
        blob = " ".join(
            t.get_text() for ax in figure.axes for t in ax.texts
        ) + " ".join(ax.get_title(loc="left") for ax in figure.axes)
        for forbidden in ("Ann", "Lofi Girl", "Know My Name"):
            assert forbidden not in blob


class TestFigureStructure:
    def test_it_draws_six_panels(self, figure: Any) -> None:
        # Seven axes: six panels plus the heatmap's colorbar.
        panels = [ax for ax in figure.axes if ax.get_title(loc="left")]
        assert len(panels) == 6

    def test_every_panel_is_titled(self, figure: Any) -> None:
        titles = [
            ax.get_title(loc="left") for ax in figure.axes if ax.get_title(loc="left")
        ]
        assert len({t for t in titles}) == 6

    def test_the_stacked_bars_carry_a_legend(self, figure: Any) -> None:
        legends = [ax for ax in figure.axes if ax.get_legend() is not None]
        assert len(legends) == 1

    def test_the_y_axis_unit_moves_with_the_bucket(self) -> None:
        """ "plays per week" is not "plays per day", and the switch is invisible
        otherwise — a 53-point weekly chart labelled "per day" is a plausible-looking
        wrong answer."""
        weekly = build_figure(_metrics(days=365, bucket_unit="week"))
        titles = " ".join(ax.get_title(loc="left") for ax in weekly.axes)
        labels = " ".join(ax.get_ylabel() for ax in weekly.axes)
        assert "per week" in titles
        assert "plays / week" in labels
        assert "per day" not in titles

    def test_the_excluded_livestream_count_is_named(self) -> None:
        """A guild with many livestreams would otherwise be silently reading a
        partial completion panel."""
        titles = " ".join(
            ax.get_title(loc="left")
            for ax in build_figure(_metrics(livestream_plays=7)).axes
        )
        assert "7 livestream excluded" in titles
        none = " ".join(
            ax.get_title(loc="left")
            for ax in build_figure(_metrics(livestream_plays=0)).axes
        )
        assert "excluded" not in none

    def test_an_empty_window_still_builds(self) -> None:
        """The command sends a notice instead, but a figure that raises on empty
        input would turn a cache-decode edge into a lost card."""
        assert build_figure(AnalyticsMetrics(days=30)) is not None

    def test_a_guild_with_no_wait_data_gets_a_labelled_blank(self) -> None:
        fig = build_figure(_metrics(wait_pcts=(), wait_p50_secs=WAIT_UNAVAILABLE))
        texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "no queue-wait data" in texts


class TestDenseBuckets:
    def test_days_with_no_plays_still_occupy_the_axis(self) -> None:
        """The SQL returns only buckets that HAVE plays. Drawing those alone rescales
        the time axis silently: a quiet week collapses to nothing, adjacent bars stop
        being adjacent days, and the area chart draws one straight segment across a
        six-day gap as though it were a single day."""
        m = _metrics(
            window_start_epoch=_TODAY - 10 * _DAY,
            window_end_epoch=_TODAY,
            daily=(
                DailyPoint(day="2026-08-19", plays=5, listen_secs=600),
                DailyPoint(day="2026-08-27", plays=9, listen_secs=900),
            ),
        )
        labels, plays, hours = dense_buckets(m)
        assert len(labels) == 10
        assert labels == sorted(labels)
        assert sum(plays) == 14
        assert plays.count(0) == 8
        assert hours[labels.index("2026-08-27")] == pytest.approx(0.25)

    def test_weekly_windows_step_by_week(self) -> None:
        m = _metrics(
            bucket_unit="week",
            window_start_epoch=_TODAY - 28 * _DAY,
            window_end_epoch=_TODAY,
            daily=(DailyPoint(day="2026-08-17", plays=3, listen_secs=60),),
        )
        labels, _, _ = dense_buckets(m)
        assert len(labels) == 4

    def test_an_unstamped_window_falls_back_rather_than_spinning(self) -> None:
        """A malformed cache entry must cost the even spacing, not the panel — and an
        epoch of 0 would otherwise be ~20,000 iterations."""
        labels, plays, _ = dense_buckets(
            _metrics(window_start_epoch=0.0, window_end_epoch=0.0)
        )
        assert len(labels) == 14
        assert plays[0] == 1


class TestSourceFolding:
    def test_sources_are_capped_and_the_remainder_is_minted(self) -> None:
        """query_source is an OPEN set — sources.py stores a parsed hostname — so
        without a cap every distinct host a guild has ever played becomes a legend
        entry and a stack segment, with no bound."""
        hosts = [f"h{i}.com" for i in range(9)]
        kept = fold_sources(
            _metrics(
                daily_by_source=tuple(
                    SourceDay(day="2026-08-01", source=h, plays=9 - i)
                    for i, h in enumerate(hosts)
                )
            )
        )
        assert len(kept) == MAX_SOURCES + 1
        assert kept[-1] == OTHER_SOURCE
        assert kept[:MAX_SOURCES] == tuple(hosts[:MAX_SOURCES])

    def test_sources_seen_only_by_the_other_panels_still_get_a_slot(self) -> None:
        """completion and source_completion index `kept` too. A payload where they
        carry a source daily_by_source does not — a skewed cache entry, since both
        come off the same slice — used to take the whole figure down on a KeyError."""
        m = _metrics(
            daily_by_source=(),
            completion=(CompletionBucket(source="youtube.com", bucket=10, plays=5),),
            source_completion=(
                SourceCompletion(source="zzz.com", played_secs=5, duration_secs=9),
            ),
        )
        kept = fold_sources(m)
        assert "youtube.com" in kept and "zzz.com" in kept
        assert render_dashboard(m).startswith(b"\x89PNG")

    def test_no_other_bucket_when_nothing_is_folded(self) -> None:
        kept = fold_sources(_metrics())
        assert OTHER_SOURCE not in kept
        assert set(kept) == {"search", "spotify.com"}

    def test_folded_plays_are_kept_not_dropped(self) -> None:
        """A guild whose long tail vanished would read the stacked bars as a play
        count lower than its own topline."""
        # The single day the window below covers; dense_buckets mints the axis from
        # the window, so a row on any other date would silently not be drawn.
        day = "2026-08-28"
        rows = tuple(
            SourceDay(day=day, source=f"h{i}.com", plays=10 - i) for i in range(8)
        )
        fig = build_figure(
            _metrics(
                daily_by_source=rows,
                daily=(DailyPoint(day=day, plays=52, listen_secs=1),),
                window_start_epoch=_TODAY - _DAY,
                window_end_epoch=_TODAY,
            )
        )
        panel = next(ax for ax in fig.axes if "by source" in ax.get_title(loc="left"))
        # The stack is drawn as one PolyCollection per source, so each segment's
        # height is its quad's y-extent rather than a Rectangle's get_height().
        # Path.vertices is typed ArrayLike; at runtime it is an (N, 2) float array.
        drawn = 0.0
        for coll in panel.collections:
            for path in coll.get_paths():
                ys = [float(v[1]) for v in cast(Any, path.vertices)]
                drawn += max(ys) - min(ys)
        assert drawn == sum(r.plays for r in rows)

    def test_the_stack_is_one_collection_per_source_not_one_patch_per_segment(
        self,
    ) -> None:
        """ax.bar mints a Rectangle per day per source -- 540 artists at 90 days,
        which measured 304ms of a 404ms figure and grew with the window. One
        collection per source is flat in day count."""
        day = "2026-08-28"
        rows = tuple(SourceDay(day=day, source=f"h{i}.com", plays=5) for i in range(3))
        fig = build_figure(
            _metrics(
                daily_by_source=rows,
                daily=(DailyPoint(day=day, plays=15, listen_secs=1),),
                window_start_epoch=_TODAY - _DAY,
                window_end_epoch=_TODAY,
            )
        )
        panel = next(ax for ax in fig.axes if "by source" in ax.get_title(loc="left"))
        assert not [p for p in panel.patches if isinstance(p, Rectangle)]
        assert len(panel.collections) == len(
            fold_sources(_metrics(daily_by_source=rows))
        )


class TestRasterization:
    def test_the_pixel_budget_is_the_one_that_was_chosen(self) -> None:
        """PIXELS is derived from FIGSIZE x DPI, so asserting the IHDR against it
        pins "savefig honours the figure" and not "the image is the size we picked".
        Halving DPI recomputes both and passes."""
        assert PIXELS == (1100, 792)

    def test_the_png_has_the_exact_pixel_dimensions(self) -> None:
        """From the IHDR chunk, never a byte-size band: PNG length moves with the
        FreeType build, the zlib build, the matplotlib patch version (including the
        length of its Software: tEXt chunk) and platform font resolution — and this
        suite runs on macOS, on python:3.14-slim and on ubuntu-latest. A band tight
        enough to catch a regression on one fails on another; one loose enough for
        all three also passes a figure missing four panels."""
        png = render_dashboard(_metrics())
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", png[16:24]) == PIXELS

    def test_it_is_a_png_not_a_jpeg(self) -> None:
        """JPEG measured 65% LARGER on this content (93.3 KiB against 56.6 KiB) AND
        lower quality — flat-colour panels with thin lines and text are the DCT
        pathological case."""
        assert render_dashboard(_metrics())[1:4] == b"PNG"


class TestProcessBoundaryContract:
    """Golden rule 9. A required positional field pickles fine in the worker and then
    fails to UNPICKLE in the parent's executor-manager thread, bricking the pool."""

    def test_the_payload_round_trips(self) -> None:
        m = _metrics()
        assert pickle.loads(pickle.dumps(m)) == m

    def test_an_empty_payload_round_trips(self) -> None:
        m = AnalyticsMetrics()
        assert pickle.loads(pickle.dumps(m)) == m

    def test_the_payload_stays_small(self) -> None:
        """It crosses on every cold render. A 90-day window is ~15 KB; anything near
        a megabyte means a panel started carrying rows instead of aggregates."""
        assert len(pickle.dumps(_metrics())) < 100_000

    def test_importing_the_renderer_drags_in_no_bot_graph(self) -> None:
        """Unpickling a dataclass imports its DEFINING module, which is why
        AnalyticsMetrics lives in guild_state (stdlib + orjson) rather than beside the
        archive or the command. A subprocess, because the parent test process has all
        four imported already and could never observe this.

        matplotlib is deliberately NOT asserted absent — the worker imports it, on
        purpose. What is asserted is that importing the module does not.
        """
        code = (
            "import sys; import src.analytics_render as r;"
            "print(','.join(m for m in "
            "('discord','asyncpg','redis','opentelemetry','matplotlib') "
            "if m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert out.stdout.strip() == "", f"renderer imported: {out.stdout.strip()}"


class TestLayoutIsFixed:
    """`fig.tight_layout()` measured every drawn string to re-solve a layout that has
    exactly one answer — 266 ms of a 476 ms build_figure, measured in the runtime
    image. The grid is always 3x2 at one figsize with one font, so the answer is a
    constant."""

    @staticmethod
    def _calls() -> list[str]:
        """Attribute calls in build_figure, in source order. AST rather than a text
        search: the comments name tight_layout deliberately, to say why it is gone."""
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path("src/analytics_render.py").read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "build_figure"
        )
        return [
            n.func.attr
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]

    def test_tight_layout_is_not_called(self) -> None:
        calls = self._calls()
        assert "tight_layout" not in calls
        assert "subplots_adjust" in calls

    def test_the_layout_is_applied_before_the_panels_are_drawn(self) -> None:
        """fig.colorbar(ax=…) steals its space from the parent axes' position at the
        moment it is created, so an adjustment made afterwards moves the axes out from
        under the colorbar. Ordering is what keeps the heatmap and its bar aligned."""
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path("src/analytics_render.py").read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "build_figure"
        )
        lines = {
            n.func.attr: n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        panels = min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id.startswith("_panel_")
        )
        assert lines["subplots_adjust"] < panels

    def test_every_panel_stays_inside_the_figure(self, figure: Any) -> None:
        """A fixed layout cannot adapt, so this is the guard tight_layout used to be:
        no panel may spill past the canvas."""
        for ax in figure.axes:
            box = ax.get_position()
            assert 0.0 <= box.x0 < box.x1 <= 1.0, ax.get_title(loc="left")
            assert 0.0 <= box.y0 < box.y1 <= 1.0, ax.get_title(loc="left")

    def test_panels_do_not_overlap_vertically(self, figure: Any) -> None:
        """hspace has to clear each panel's title, and panel 1's legend band on top of
        that. Rows overlapping is what a too-small value looks like."""
        panels = sorted(
            (ax for ax in figure.axes if ax.get_title(loc="left")),
            key=lambda a: -a.get_position().y0,
        )
        rows: list[float] = []
        for ax in panels:
            box = ax.get_position()
            if not rows or abs(rows[-1] - box.y0) > 0.01:
                rows.append(box.y0)
        assert len(rows) == 3
        heights = [ax.get_position().height for ax in panels]
        for upper, lower in zip(rows, rows[1:]):
            assert upper > lower + max(heights) * 0.99


class TestPanelForms:
    """Each panel's chart FORM, which is a spec decision rather than a detail: the
    plan names panel 6 a box/percentile strip and the heatmap a single-hue sequential
    ramp, and both are the kind of thing a later edit silently changes."""

    def test_the_queue_wait_panel_is_a_box_strip_not_bars(self) -> None:
        """Bars per percentile invite reading five independent quantities where there
        is one distribution, and put a baseline at zero that means nothing here."""
        fig = build_figure(_metrics())
        panel = next(
            ax for ax in fig.axes if ax.get_title(loc="left").startswith("Queue wait")
        )
        # bxp draws the box as a PathPatch and the whiskers/median as Line2Ds; a bar
        # chart draws Rectangles and no whiskers.
        assert not [p for p in panel.patches if isinstance(p, Rectangle)]
        assert len(panel.lines) >= 4  # 2 whiskers, 2 caps, median

    def test_the_box_spans_p25_to_p75_with_p10_p90_whiskers(self) -> None:
        m = _metrics(wait_pcts=(10.0, 20.0, 50.0, 80.0, 90.0), wait_p50_secs=50.0)
        panel = next(
            ax
            for ax in build_figure(m).axes
            if ax.get_title(loc="left").startswith("Queue wait")
        )
        # get_xdata() is typed as a union that includes non-iterables; at runtime
        # a Line2D always hands back a sequence.
        xs = {
            round(float(x))
            for line in panel.lines
            for x in cast(Sequence[float], line.get_xdata())
        }
        for edge in (10, 20, 80, 90):
            assert edge in xs, f"{edge} missing from the drawn strip"

    def test_the_strip_says_which_percentiles_it_draws(self) -> None:
        """bxp's default whiskers are 1.5*IQR and these are not, so a reader who
        assumes the default reads the tails wrong."""
        panel = next(
            ax
            for ax in build_figure(_metrics()).axes
            if ax.get_title(loc="left").startswith("Queue wait")
        )
        note = " ".join(t.get_text() for t in panel.texts)
        assert "p25-p75" in note
        assert "p10-p90" in note

    def test_the_completion_panel_draws_every_bucket_including_the_tenth(
        self,
    ) -> None:
        """Bucket 10 -- played to the end -- is the modal value in real data, so a
        bound that excludes it drops most of the panel while the axis, the labels
        and the title all still render."""
        m = _metrics(
            completion=tuple(
                CompletionBucket(source="search", bucket=b, plays=b)
                for b in range(1, 11)
            )
        )
        panel = next(
            ax
            for ax in build_figure(m).axes
            if ax.get_title(loc="left").startswith("How much")
        )
        # Non-zero only: every kept source gets ten bars, so a source with no
        # completion rows contributes a flat row of zeros.
        heights = sorted(
            h for p in panel.patches if (h := cast(Rectangle, p).get_height())
        )
        assert heights == [float(b) for b in range(1, 11)]

    def test_the_wait_labels_are_minted_from_the_percentiles_asked_for(self) -> None:
        """Four places agreed on "five, median at index 2" by hand. Adding p99 to the
        SQL alone flipped every length check to its SAFE branch, so the card printed
        "Median queue wait n/a" for a guild with a full window of data."""
        from src.guild_state import WAIT_MEDIAN_INDEX, WAIT_PERCENTILES

        assert len(analytics_render._PCT_LABELS) == len(WAIT_PERCENTILES)
        assert analytics_render._PCT_LABELS[WAIT_MEDIAN_INDEX] == "p50"
        assert _ANALYTICS_PCTS() == list(WAIT_PERCENTILES)

    @pytest.mark.parametrize("days", [7, 30, 90, 365, 730])
    def test_every_window_the_allowlist_could_hold_is_densified(
        self, days: int
    ) -> None:
        """The guard was a fixed 400 days, so a wider --days silently stopped
        zero-filling: a quiet fortnight collapses to nothing and the area chart draws
        one straight segment across the gap, which is what dense_buckets prevents."""
        unit = "week" if days >= 365 else "day"
        m = _metrics(
            days=days,
            bucket_unit=unit,
            window_start_epoch=_TODAY - days * _DAY,
            window_end_epoch=_TODAY,
            daily=(DailyPoint(day="2026-08-27", plays=1, listen_secs=1),),
        )
        labels, _, _ = dense_buckets(m)
        # ceil: the walk steps from the window start until it passes the end.
        expected = -(-days // 7) if unit == "week" else days
        assert len(labels) == expected

    def test_no_panel_title_collides_with_the_panel_above_it(self) -> None:
        """_LAYOUT is six hand-tuned floats and the grid is hardcoded 3x2, so a
        seventh panel or a new axis label means re-deriving them by eye. Nothing else
        in the suite can see the result, and a cached PNG hides a regression from the
        operator for a day on top of that."""
        panels = [
            ax for ax in build_figure(_metrics()).axes if ax.get_title(loc="left")
        ]
        boxes = [ax.get_position() for ax in panels]
        for box in boxes:
            assert 0.0 <= box.y0 < box.y1 <= 1.0, "panel left the figure"
            assert 0.0 <= box.x0 < box.x1 <= 1.0, "panel left the figure"
        rows = sorted({round(b.y0, 3) for b in boxes}, reverse=True)
        assert len(rows) == 3, f"expected a 3x2 grid, got rows {rows}"
        # Each row must clear the one above it, with room for the title drawn in the
        # gap. Too small an hspace is what puts a title onto the axes above it.
        for upper, lower in zip(rows, rows[1:]):
            top_of_lower = max(b.y1 for b in boxes if round(b.y0, 3) == lower)
            assert upper - top_of_lower > 0.02, "rows too close for their titles"

    def test_the_heatmap_ramp_is_one_hue_and_monotonic(self) -> None:
        """Magnitude takes a sequential ramp: one hue, light->dark. A multi-hue map
        (magma, viridis) reads as a rainbow, so the reader decodes hue instead of
        lightness — and its floor is pure black, unrelated to the panel."""
        import colorsys

        def _lum(hexcolor: str) -> float:
            channels = [int(hexcolor[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [
                c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                for c in channels
            ]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        lums = [_lum(c) for c in analytics_render._SEQUENTIAL_BLUE]
        assert all(a < b for a, b in zip(lums, lums[1:])), "not monotonic"
        hues = {
            round(
                colorsys.rgb_to_hls(*[int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)])[
                    0
                ]
                * 360
            )
            for c in analytics_render._SEQUENTIAL_BLUE
        }
        assert max(hues) - min(hues) <= 15, f"more than one hue: {sorted(hues)}"

    def test_an_empty_hour_recedes_to_the_panel_rather_than_the_ramp_floor(
        self,
    ) -> None:
        """Most of a weekday x hour grid is empty. Painted as the ramp's darkest step
        the panel reads as one solid block and the ramp spends its whole range on the
        few cells that carry data."""
        panel = next(
            ax
            for ax in build_figure(_metrics()).axes
            if ax.get_title(loc="left").startswith("When this server")
        )
        image = panel.images[0]
        # The resolved colour, not the vmin that produces it: matplotlib may
        # override an explicit vmin while autoscaling, so the constant and the
        # behaviour are different assertions.
        # Normalize.__call__ is stubbed for arrays and vmin/vmax as a union; at
        # runtime both take and return plain scalars here.
        assert tuple(image.cmap(cast(Any, image.norm)(0.0))) == tuple(
            image.cmap.get_under()
        )
        under = image.get_cmap().get_under()
        assert tuple(round(c, 4) for c in under[:3]) == tuple(
            round(int(analytics_render._PANEL[i : i + 2], 16) / 255, 4)
            for i in (1, 3, 5)
        )

    @pytest.mark.parametrize("peak", [0, 1, 2, 20])
    def test_the_heatmap_scale_never_collapses_onto_one_colour(self, peak: int) -> None:
        """An empty cell must resolve to the panel at every scale. Normalize rescues
        an equal vmin/vmax by expanding it, but an all-zero grid discards vmin
        outright and maps zero into the ramp, so all 168 cells read as active."""
        cells = (HeatCell(dow=1, hour=9, plays=peak),) if peak else ()
        panel = next(
            ax
            for ax in build_figure(_metrics(heat=cells)).axes
            if ax.get_title(loc="left").startswith("When this server")
        )
        image = panel.images[0]
        norm = cast(Any, image.norm)
        assert cast(float, norm.vmin) < cast(float, norm.vmax), "degenerate scale"
        empty = tuple(image.cmap(norm(0.0)))
        assert empty == tuple(image.cmap.get_under())
        if peak:
            assert tuple(image.cmap(norm(float(peak)))) != empty
