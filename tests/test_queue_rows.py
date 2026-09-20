"""Tests for src/queue_rows.py — the row every queue listing shares."""

import datetime
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.queue_rows import (
    ROW_LIMIT,
    EtaWalk,
    advance_walk,
    fmt_eta,
    queue_row,
    queue_rows,
    queue_runtime,
    remaining_secs,
)
from src.sources import YTSource
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
        assert advance_walk(_START, _song(mock_author)) == EtaWalk(240, False)

    def test_an_unknown_length_marks_everything_after_it(
        self, mock_author: MagicMock
    ) -> None:
        walk = advance_walk(_START, _song(mock_author, duration=None))
        assert walk == EtaWalk(180, True)
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

    def test_one_oversized_row_still_shows(self, mock_author: MagicMock) -> None:
        text = queue_rows(
            [_song(mock_author)], first_index=1, now=_NOW, walk=_START, budget=1
        )
        assert "Song 1" in text
