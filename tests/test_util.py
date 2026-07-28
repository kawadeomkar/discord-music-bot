"""Tests for src/util.py — queue formatting and logging utilities."""

from typing import Any
import logging

import pytest

from src.guild_state import HistoryEntry
from src.util import (
    fmt_duration,
    get_logger,
    history_embeds,
    pluralize,
    queue_message,
)


class TestQueueMessage:
    def test_empty_list_returns_empty_string(self) -> None:
        assert queue_message([]) == ""

    def test_two_items_shows_both(self) -> None:
        result = queue_message(["song_a", "song_b"])
        assert "1: song_a" in result
        assert "2: song_b" in result

    def test_five_items_shows_all_five(self) -> None:
        songs = [f"song{i}" for i in range(5)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line]
        assert len(lines) == 5
        assert "1: song0" in result
        assert "5: song4" in result

    def test_exactly_ten_items_no_ellipsis(self) -> None:
        songs = [f"track{i}" for i in range(10)]
        result = queue_message(songs)
        assert "..." not in result

    def test_exactly_ten_items_shows_all_ten(self) -> None:
        songs = [f"track{i}" for i in range(10)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line]
        assert len(lines) == 10

    def test_more_than_ten_items_appends_ellipsis(self) -> None:
        songs = [f"track{i}" for i in range(15)]
        result = queue_message(songs)
        assert "..." in result

    def test_more_than_ten_items_caps_at_ten_shown(self) -> None:
        songs = [f"track{i}" for i in range(20)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line and line != "..."]
        assert len(lines) == 10

    def test_numbering_starts_at_one(self) -> None:
        result = queue_message(["first", "second", "third"])
        assert result.startswith("1:")

    def test_songs_sliced_to_ten_before_processing(self) -> None:
        songs = [f"song{i}" for i in range(25)]
        result = queue_message(songs)
        assert "song15" not in result
        assert "song20" not in result


class TestGetLogger:
    def test_returns_structlog_logger(self) -> None:
        logger = get_logger("test.module")
        # structlog returns a lazy proxy — not a stdlib Logger
        assert not isinstance(logger, logging.Logger)
        assert hasattr(logger, "info") and hasattr(logger, "warning")

    def test_logging_methods_are_callable(self) -> None:
        logger = get_logger("test.callable")
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.debug)

    def test_logging_does_not_raise(self) -> None:
        logger = get_logger("test.no_raise")
        logger.info("test message", key="value")

    def test_calling_twice_returns_functional_loggers(self) -> None:
        logger_a = get_logger("test.no_dup")
        logger_b = get_logger("test.no_dup")
        # Both proxies are usable; no errors on repeated calls
        logger_a.info("from a")
        logger_b.info("from b")

    def test_different_names_return_different_loggers(self) -> None:
        logger_a = get_logger("module.a")
        logger_b = get_logger("module.b")
        assert logger_a is not logger_b
        assert logger_a.name != logger_b.name


class TestPluralize:
    """The one noun-form helper. Every embed and command line that used to spell
    `f"song{'s' if n != 1 else ''}"` inline routes through this, so its
    boundaries are user-visible text in ~7 places."""

    @pytest.mark.parametrize(
        "count,expected",
        [
            (1, "song"),
            (2, "songs"),
            (0, "songs"),  # English pluralizes zero — "0 songs", not "0 song"
            (-1, "songs"),
            (100, "songs"),
        ],
    )
    def test_only_exactly_one_is_singular(self, count: int, expected: str) -> None:
        """The rule is `count == 1`, not `count <= 1`.

        0 and -1 are the cases that separate the two: relaxing the condition to
        `<= 1` would render "0 song" and "-1 song" and pass every other test.
        """
        assert pluralize(count, "song") == expected

    def test_plural_override_used_for_irregulars(self) -> None:
        assert pluralize(2, "person", "people") == "people"

    def test_plural_override_ignored_when_singular(self) -> None:
        assert pluralize(1, "person", "people") == "person"

    @pytest.mark.parametrize("count", [0, -1, 3])
    def test_plural_override_applies_to_every_non_one_count(self, count: int) -> None:
        """The override must not be reachable only via count > 1 — the zero and
        negative paths share the same branch and were previously unexercised."""
        assert pluralize(count, "person", "people") == "people"

    def test_explicit_none_plural_falls_back_to_s_suffix(self) -> None:
        """`plural=None` must mean "derive it", not "return None"."""
        assert pluralize(3, "song", None) == "songs"

    def test_empty_plural_override_is_honored_not_treated_as_missing(self) -> None:
        """`plural=""` is falsy but explicitly passed. The implementation tests
        `is not None`, so it must be respected; an `if plural:` regression would
        silently emit "s" instead."""
        assert pluralize(3, "song", "") == ""

    def test_singular_returned_verbatim(self) -> None:
        """No suffix logic on the singular branch — multi-word nouns survive."""
        assert pluralize(1, "queued song") == "queued song"


class TestFmtDuration:
    """The one clock formatter — progress bar, queue/pause/skip lines, history,
    and YTDL.duration all render through this."""

    def test_minutes_seconds(self) -> None:
        assert fmt_duration(225) == "3:45"

    def test_hours_zero_pads_minutes_and_seconds(self) -> None:
        assert fmt_duration(3725) == "1:02:05"

    def test_zero(self) -> None:
        assert fmt_duration(0) == "0:00"

    def test_negative_clamps_to_zero(self) -> None:
        assert fmt_duration(-5) == "0:00"

    def test_under_a_minute(self) -> None:
        assert fmt_duration(7) == "0:07"

    def test_exactly_one_hour(self) -> None:
        # Boundary: the hours branch must engage at exactly 3600, not above it.
        assert fmt_duration(3600) == "1:00:00"

    def test_minute_rollover_pads_seconds(self) -> None:
        assert fmt_duration(61) == "1:01"


def _rich_entry(**overrides: Any) -> HistoryEntry:
    fields: dict = dict(
        title="Rich Song",
        webpage_url="https://yt.com/v=rich",
        duration_secs=242,
        played_secs=225,
        requester_id=42,
        requester_name="Omkar",
        thumbnail="https://i.ytimg.com/t.jpg",
        uploader="Chan",
        played_at=1752530000.0,
    )
    fields.update(overrides)
    return HistoryEntry(**fields)


class TestHistoryEmbeds:
    def test_layout_title_url_then_info_line(self) -> None:
        # numbered title; webpage_url on its own line beneath it;
        # played/duration · requester · absolute timestamp on ONE line below.
        [embed] = history_embeds([_rich_entry()])
        assert embed.title == "1. Rich Song"
        assert embed.description is not None
        assert embed.description.splitlines() == [
            "https://yt.com/v=rich",
            "3:45 / 4:02 · requested by <@42> · <t:1752530000:f>",
        ]

    def test_numbering_follows_given_order(self) -> None:
        embeds = history_embeds([_rich_entry(), _rich_entry(title="Second")])
        assert embeds[0].title == "1. Rich Song"
        assert embeds[1].title == "2. Second"

    def test_thumbnail_set_when_present(self) -> None:
        [embed] = history_embeds([_rich_entry()])
        assert embed.thumbnail.url == "https://i.ytimg.com/t.jpg"

    def test_no_thumbnail_when_absent(self) -> None:
        [embed] = history_embeds([_rich_entry(thumbnail="")])
        assert embed.thumbnail.url is None

    def test_requester_mention_survives_member_departure(self) -> None:
        # The raw <@id> mention needs no member cache to render.
        [embed] = history_embeds([_rich_entry(requester_id=999)])
        assert embed.description is not None
        assert "<@999>" in embed.description

    def test_requester_name_fallback_when_id_unknown(self) -> None:
        [embed] = history_embeds(
            [_rich_entry(requester_id=0, requester_name="SomeUser")]
        )
        assert embed.description is not None
        assert "requested by SomeUser" in embed.description

    def test_timestamp_omitted_when_played_at_unknown(self) -> None:
        # played_at == 0 means unknown; <t:0:f> would render "1 January 1970".
        [embed] = history_embeds([_rich_entry(played_at=0.0)])
        assert embed.description is not None
        assert "<t:" not in embed.description
        assert embed.description.splitlines() == [
            "https://yt.com/v=rich",
            "3:45 / 4:02 · requested by <@42>",
        ]

    def test_over_length_title_truncated_to_discord_limit(self) -> None:
        # Discord rejects any embed title > 256 chars, failing the whole send.
        [embed] = history_embeds([_rich_entry(title="x" * 300)])
        assert embed.title is not None
        assert len(embed.title) == 256
        assert embed.title.endswith("…")

    def test_title_at_limit_not_truncated(self) -> None:
        # "1. " (3) + 253 = 256 exactly — must pass through untouched.
        [embed] = history_embeds([_rich_entry(title="y" * 253)])
        assert embed.title == "1. " + "y" * 253
        assert embed.title is not None
        assert "…" not in embed.title

    def test_empty_input(self) -> None:
        assert history_embeds([]) == []
