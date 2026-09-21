"""Tests for src/queue_rows.py — the row every queue listing shares."""

import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.queue_rows import (
    eta_at,
    fmt_total_duration,
    requester_mention,
    ROW_BYLINE_MAX,
    ROW_LIMIT,
    ROW_TITLE_MAX,
    ROWS_BUDGET,
    EtaWalk,
    advance_walk,
    fmt_eta,
    queue_row,
    queue_rows,
    queue_runtime,
    remaining_secs,
)
from src.sources import YTSource
from src.util import EMBED_DESCRIPTION_LIMIT
from src.youtube import QueueObject

_NOW = datetime.datetime(2026, 9, 20, 21, 38, tzinfo=ZoneInfo("US/Pacific"))
_START = EtaWalk(cumulative_secs=180, uncertain=False)


def _song(author: MagicMock, n: int = 1, **kwargs: Any) -> QueueObject:
    fields: dict[str, Any] = {"duration": 60, "uploader": f"Channel {n}", **kwargs}
    return QueueObject(f"https://yt.com/v={n}", f"Song {n}", author, **fields)


class TestRemainingSecs:
    def test_normal_item_full_duration(self, mock_author: MagicMock) -> None:
        assert remaining_secs(_song(mock_author, duration=210)) == 210

    def test_resume_entry_counts_only_tail(self, mock_author: MagicMock) -> None:
        item = _song(mock_author, ts=150, duration=210, is_resume=True)
        assert remaining_secs(item) == 60

    def test_unknown_duration_is_none(self, mock_author: MagicMock) -> None:
        assert remaining_secs(_song(mock_author, duration=None)) is None

    def test_non_resume_ts_does_not_shrink_duration(
        self, mock_author: MagicMock
    ) -> None:
        # A ?t= start offset is a playback preference, not a shorter song —
        # only resume entries are known to play just their tail.
        assert remaining_secs(_song(mock_author, ts=150, duration=210)) == 210


class TestTheWalk:
    def test_a_known_length_adds_its_seconds(self, mock_author: MagicMock) -> None:
        assert advance_walk(_START, _song(mock_author)) == EtaWalk(
            cumulative_secs=240, uncertain=False
        )

    def test_an_unknown_length_marks_everything_after_it(
        self, mock_author: MagicMock
    ) -> None:
        walk = advance_walk(_START, _song(mock_author, duration=None))
        assert walk == EtaWalk(cumulative_secs=180, uncertain=True)
        assert fmt_eta(_NOW, walk.uncertain).startswith("~")

    def test_an_unresolved_search_is_an_unknown_length(self) -> None:
        assert advance_walk(_START, YTSource(ytsearch="ytsearch:x")).uncertain

    def test_the_runtime_is_partial_over_an_unknown_length(
        self, mock_author: MagicMock
    ) -> None:
        items = [_song(mock_author), YTSource(ytsearch="ytsearch:x")]
        assert queue_runtime(items) == (60, True)


class TestQueueRow:
    def test_a_song_is_index_link_length_eta_then_channel_and_requester(
        self, mock_author: MagicMock
    ) -> None:
        row = queue_row(_song(mock_author), 4, now=_NOW, walk=_START)
        assert row == (
            "`4` [**Song 1**](https://yt.com/v=1) · `1:00` · "
            "Est. playing at **9:41 PM PDT**\n"
            f"Channel 1 · {mock_author.mention}"
        )

    def test_without_a_byline_it_is_the_first_line_alone(
        self, mock_author: MagicMock
    ) -> None:
        full = queue_row(_song(mock_author), 4, now=_NOW, walk=_START)
        assert (
            queue_row(_song(mock_author), 4, now=_NOW, walk=_START, byline=False)
            == full.split("\n")[0]
        )

    def test_a_resume_entry_says_where_it_resumes(self, mock_author: MagicMock) -> None:
        item = _song(mock_author, ts=95, duration=200, is_resume=True)
        assert "⏮ resumes at `1:35`" in queue_row(item, 1, now=_NOW, walk=_START)

    def test_a_hostile_title_cannot_close_the_label(
        self, mock_author: MagicMock
    ) -> None:
        """The title sits inside a masked link's LABEL and is uploader-chosen: an
        unbalanced `]` closes the label early and re-points the link."""
        item = _song(mock_author, duration=100)
        item.title = "Song](https://evil.example) [FREE NITRO"
        line = queue_row(item, 1, now=_NOW, walk=_START)
        assert "](https://evil.example)" not in line
        assert "[FREE NITRO" not in line
        assert "](https://yt.com/v=1)" in line

    def test_an_unresolved_search_is_sanitized_too(self) -> None:
        item = YTSource(ytsearch="ytsearch:[click](https://evil.example)", process=True)
        line = queue_row(item, 1, now=_NOW, walk=_START)
        assert "[" not in line and "](" not in line
        assert line.endswith("*resolving...*")


def _track(**kwargs: Any) -> YTSource:
    """An unresolved Spotify track, with the display fields it is queued with."""
    fields: dict[str, Any] = {
        "ytsearch": "ytsearch:DNA. Kendrick Lamar",
        "requester_id": 4242,
        "title": "DNA.",
        "uploader": "Kendrick Lamar",
        "duration": 185,
        "webpage_url": "https://open.spotify.com/track/abc",
        **kwargs,
    }
    return YTSource(**fields)


class TestAnUnresolvedTrackRow:
    """A Spotify track that has not reached YouTube yet renders the song row from
    the fields it was queued with."""

    def test_it_is_the_song_row_with_the_artists_where_the_channel_goes(self) -> None:
        assert queue_row(_track(), 3, now=_NOW, walk=_START) == (
            "`3` [**DNA.**](https://open.spotify.com/track/abc) · `3:05` · "
            "Est. playing at **9:41 PM PDT**\n"
            "Kendrick Lamar · <@4242>"
        )

    def test_without_a_link_the_title_is_bold_text(self) -> None:
        row = queue_row(
            _track(webpage_url=None), 3, now=_NOW, walk=_START, byline=False
        )
        assert row.startswith("`3` **DNA.** · `3:05`")

    def test_missing_pieces_have_placeholders(self) -> None:
        row = queue_row(
            _track(uploader=None, duration=None, requester_id=None),
            1,
            now=_NOW,
            walk=_START,
        )
        assert "`?:??`" in row
        assert row.endswith("Unknown artist · Unknown")

    def test_a_track_queued_without_display_fields_is_its_search_text(self) -> None:
        row = queue_row(_track(title=None), 1, now=_NOW, walk=_START)
        assert row == "`1` DNA. Kendrick Lamar · *resolving...*"

    def test_spotifys_text_cannot_style_the_row(self) -> None:
        hostile = "[x](https://evil.example)"
        row = queue_row(
            _track(title=hostile, uploader=hostile), 1, now=_NOW, walk=_START
        )
        assert "](https://evil.example)" not in row

    def test_its_length_counts_and_marks_what_follows_approximate(
        self, mock_author: MagicMock
    ) -> None:
        """Spotify's length is not the YouTube match's."""
        assert advance_walk(_START, _track()) == EtaWalk(
            cumulative_secs=180 + 185, uncertain=True
        )
        rows = queue_rows(
            [_track(), _song(mock_author)],
            first_index=1,
            now=_NOW,
            walk=_START,
            byline=False,
        ).split("\n")
        assert "Est. playing at **9:41 PM PDT**" in rows[0]
        assert "Est. playing at ~**9:44 PM PDT**" in rows[1]

    def test_the_runtime_counts_it_and_stays_approximate(
        self, mock_author: MagicMock
    ) -> None:
        assert queue_runtime([_song(mock_author), _track()]) == (60 + 185, True)


class TestQueueRows:
    def test_no_items_is_no_text(self) -> None:
        assert queue_rows([], first_index=1, now=_NOW, walk=_START) == ""

    def test_rows_are_numbered_from_the_first_index_and_chain_the_walk(
        self, mock_author: MagicMock
    ) -> None:
        text = queue_rows(
            [_song(mock_author, 1), _song(mock_author, 2)],
            first_index=7,
            now=_NOW,
            walk=_START,
            byline=False,
        )
        first, second = text.split("\n")
        assert first.startswith("`7` ") and "**9:41 PM PDT**" in first
        assert second.startswith("`8` ") and "**9:42 PM PDT**" in second

    @pytest.mark.parametrize("byline,gap", [(True, "\n\n"), (False, "\n")])
    def test_two_line_rows_are_set_apart_and_one_line_rows_are_not(
        self, mock_author: MagicMock, byline: bool, gap: str
    ) -> None:
        items = [_song(mock_author, n) for n in (1, 2, 3)]
        text = queue_rows(items, first_index=1, now=_NOW, walk=_START, byline=byline)
        assert len(text.split(gap)) == 3

    def test_past_the_limit_the_rest_is_counted(self, mock_author: MagicMock) -> None:
        items = [_song(mock_author, n) for n in range(ROW_LIMIT + 4)]
        text = queue_rows(items, first_index=1, now=_NOW, walk=_START, byline=False)
        rows = text.split("\n")
        assert len(rows) == ROW_LIMIT + 1
        assert rows[-1] == "*... and 4 more*"

    def test_exactly_the_limit_has_no_tail(self, mock_author: MagicMock) -> None:
        items = [_song(mock_author, n) for n in range(ROW_LIMIT)]
        text = queue_rows(items, first_index=1, now=_NOW, walk=_START, byline=False)
        assert "more" not in text

    def test_long_rows_end_the_listing_early_and_are_counted_as_more(
        self, mock_author: MagicMock
    ) -> None:
        """Ten rows at both caps overflow a 4096-character description."""
        items = [_song(mock_author, n, uploader="C" * 500) for n in range(ROW_LIMIT)]
        for item in items:
            item.title = "T" * 500
        text = queue_rows(items, first_index=1, now=_NOW, walk=_START, budget=1500)
        shown = text.count("Est. playing at")
        assert 1 <= shown < ROW_LIMIT
        assert len(text) <= 1500 + len("\n\n*... and 99 more*")
        assert text.endswith(f"*... and {ROW_LIMIT - shown} more*")

    def test_only_the_rows_shown_are_formatted(self, mock_author: MagicMock) -> None:
        """A 10,000-track playlist is listed by formatting ten of them."""
        items = [_song(mock_author, n) for n in range(500)]
        with patch("src.queue_rows.safe_label", side_effect=lambda t, _w: t) as escape:
            queue_rows(items, first_index=1, now=_NOW, walk=_START, byline=False)

        assert escape.call_count == ROW_LIMIT

    @pytest.mark.parametrize("byline,gap", [(True, 2), (False, 1)])
    def test_a_row_that_exactly_fills_the_budget_is_kept(
        self, mock_author: MagicMock, byline: bool, gap: int
    ) -> None:
        """The boundary the `>` sits on. A budget one character short drops the
        second row, so `>=` here would cost a row on every listing that fits."""
        items = [_song(mock_author, n) for n in (1, 2)]
        one = queue_row(items[0], 1, now=_NOW, walk=_START, byline=byline)
        two = queue_row(
            items[1], 2, now=_NOW, walk=advance_walk(_START, items[0]), byline=byline
        )
        exact = len(one) + gap + len(two)

        kept = queue_rows(
            items, first_index=1, now=_NOW, walk=_START, byline=byline, budget=exact
        )
        assert "... and" not in kept

        dropped = queue_rows(
            items, first_index=1, now=_NOW, walk=_START, byline=byline, budget=exact - 1
        )
        assert dropped.endswith("*... and 1 more*")

    def test_the_gap_between_rows_counts_against_the_budget(
        self, mock_author: MagicMock
    ) -> None:
        """Two-line rows are joined by a blank line and one-line rows by one
        newline, so the same rows fit a tighter budget in one-line mode. Dropping
        `len(gap)` from the running total would make the two modes agree."""
        items = [_song(mock_author, n) for n in (1, 2)]
        one = queue_row(items[0], 1, now=_NOW, walk=_START, byline=False)
        two = queue_row(
            items[1], 2, now=_NOW, walk=advance_walk(_START, items[0]), byline=False
        )
        budget = len(one) + 1 + len(two)

        assert "... and" not in queue_rows(
            items, first_index=1, now=_NOW, walk=_START, byline=False, budget=budget
        )
        # The same budget in two-line mode is one character short of its wider gap.
        assert queue_rows(
            items, first_index=1, now=_NOW, walk=_START, byline=True, budget=budget
        ).endswith("*... and 1 more*")

    def test_one_oversized_row_still_shows(self, mock_author: MagicMock) -> None:
        text = queue_rows(
            [_song(mock_author)], first_index=1, now=_NOW, walk=_START, budget=1
        )
        assert "Song 1" in text


class TestTheDefaultBudgetHoldsTheDescriptionUnder4096:
    """ROWS_BUDGET is the only thing between a listing of long rows and Discord's
    4096-character description limit, which 400s the whole send rather than
    truncating. Every other test here passes an explicit `budget=`, so without
    this one the default is exercised by nothing and can be raised silently."""

    @staticmethod
    def _maximal(author: MagicMock, n: int) -> QueueObject:
        """A row at both caps: yt-dlp bounds neither a title nor an uploader."""
        return QueueObject(
            "https://www.youtube.com/watch?v=" + "x" * 11,
            "T" * (ROW_TITLE_MAX + 50),
            author,
            duration=215,
            uploader="U" * (ROW_BYLINE_MAX + 50),
        )

    @pytest.mark.parametrize("byline", [True, False])
    def test_ten_maximal_rows_fit_an_embed_description(
        self, mock_author: MagicMock, byline: bool
    ) -> None:
        items = [self._maximal(mock_author, n) for n in range(ROW_LIMIT)]
        text = queue_rows(items, first_index=1, now=_NOW, walk=_START, byline=byline)
        assert len(text) <= EMBED_DESCRIPTION_LIMIT

    def test_the_budget_is_what_holds_them(self, mock_author: MagicMock) -> None:
        """Pins the mechanism, not just the outcome: with the budget lifted the
        same ten rows overflow, so the constant is load-bearing and not incidental
        to the ten-row limit."""
        items = [self._maximal(mock_author, n) for n in range(ROW_LIMIT)]
        unbounded = queue_rows(
            items, first_index=1, now=_NOW, walk=_START, budget=10**6
        )
        assert len(unbounded) > EMBED_DESCRIPTION_LIMIT
        assert ROWS_BUDGET < EMBED_DESCRIPTION_LIMIT


class TestFmtTotalDuration:
    def test_seconds_only(self) -> None:
        assert fmt_total_duration(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert fmt_total_duration(185) == "3m 5s"

    def test_hours_minutes_seconds(self) -> None:
        assert fmt_total_duration(3723) == "1h 2m 3s"

    def test_zero(self) -> None:
        assert fmt_total_duration(0) == "0s"

    def test_exactly_one_hour(self) -> None:
        assert fmt_total_duration(3600) == "1h"

    def test_hours_no_minutes_with_seconds(self) -> None:
        # Regression: 1h 0m 45s previously showed as "1h" (seconds dropped)
        assert fmt_total_duration(3645) == "1h 45s"

    def test_hours_and_minutes_no_seconds(self) -> None:
        assert fmt_total_duration(3780) == "1h 3m"


class TestRequesterMention:
    def test_returns_mention_when_present(self, mock_author: MagicMock) -> None:
        assert requester_mention(mock_author) == mock_author.mention

    def test_returns_unknown_when_none(self) -> None:
        assert requester_mention(None) == "Unknown"


class TestEtaAcrossADstTransition:
    """Adding a timedelta to an aware datetime is wall-clock arithmetic: the offset
    is not renormalized, so a queue spanning a transition lands an hour out and can
    render a local time that does not exist."""

    # US/Pacific skips 02:00-03:00 on this date.
    _BEFORE = datetime.datetime(2027, 3, 14, 0, 30, tzinfo=ZoneInfo("US/Pacific"))

    def test_two_hours_over_the_spring_forward_is_two_real_hours(self) -> None:
        assert fmt_eta(eta_at(self._BEFORE, 7200), False) == "**3:30 AM PDT**"

    def test_it_never_renders_an_hour_that_does_not_exist(self) -> None:
        """02:00-03:00 is skipped locally, so no span may land inside it."""
        for secs in range(0, 3 * 3600, 60):
            rendered = eta_at(self._BEFORE, secs)
            assert not (rendered.hour == 2 and rendered.tzname() == "PST")

    def test_an_ordinary_span_is_unchanged(self) -> None:
        plain = datetime.datetime(2026, 9, 20, 21, 38, tzinfo=ZoneInfo("US/Pacific"))
        assert eta_at(plain, 3600) == plain + datetime.timedelta(seconds=3600)
