"""Tests for src/util.py — queue formatting and logging utilities."""

import logging

from src.guild_state import HistoryEntry
from src.util import fmt_duration, get_logger, history_embeds, queue_message


class TestQueueMessage:
    def test_empty_list_returns_empty_string(self):
        assert queue_message([]) == ""

    def test_two_items_shows_both(self):
        result = queue_message(["song_a", "song_b"])
        assert "1: song_a" in result
        assert "2: song_b" in result

    def test_five_items_shows_all_five(self):
        songs = [f"song{i}" for i in range(5)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line]
        assert len(lines) == 5
        assert "1: song0" in result
        assert "5: song4" in result

    def test_exactly_ten_items_no_ellipsis(self):
        songs = [f"track{i}" for i in range(10)]
        result = queue_message(songs)
        assert "..." not in result

    def test_exactly_ten_items_shows_all_ten(self):
        songs = [f"track{i}" for i in range(10)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line]
        assert len(lines) == 10

    def test_more_than_ten_items_appends_ellipsis(self):
        songs = [f"track{i}" for i in range(15)]
        result = queue_message(songs)
        assert "..." in result

    def test_more_than_ten_items_caps_at_ten_shown(self):
        songs = [f"track{i}" for i in range(20)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line and line != "..."]
        assert len(lines) == 10

    def test_numbering_starts_at_one(self):
        result = queue_message(["first", "second", "third"])
        assert result.startswith("1:")

    def test_songs_sliced_to_ten_before_processing(self):
        songs = [f"song{i}" for i in range(25)]
        result = queue_message(songs)
        assert "song15" not in result
        assert "song20" not in result


class TestGetLogger:
    def test_returns_structlog_logger(self):
        logger = get_logger("test.module")
        # structlog returns a lazy proxy — not a stdlib Logger
        assert not isinstance(logger, logging.Logger)
        assert hasattr(logger, "info") and hasattr(logger, "warning")

    def test_logging_methods_are_callable(self):
        logger = get_logger("test.callable")
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.debug)

    def test_logging_does_not_raise(self):
        logger = get_logger("test.no_raise")
        logger.info("test message", key="value")

    def test_calling_twice_returns_functional_loggers(self):
        logger_a = get_logger("test.no_dup")
        logger_b = get_logger("test.no_dup")
        # Both proxies are usable; no errors on repeated calls
        logger_a.info("from a")
        logger_b.info("from b")

    def test_different_names_return_different_loggers(self):
        logger_a = get_logger("module.a")
        logger_b = get_logger("module.b")
        assert logger_a is not logger_b
        assert logger_a.name != logger_b.name


class TestFmtDuration:
    def test_minutes_seconds(self):
        assert fmt_duration(225) == "3:45"

    def test_hours_zero_pads_minutes_and_seconds(self):
        assert fmt_duration(3725) == "1:02:05"

    def test_zero(self):
        assert fmt_duration(0) == "0:00"

    def test_negative_clamps_to_zero(self):
        assert fmt_duration(-5) == "0:00"

    def test_under_a_minute(self):
        assert fmt_duration(7) == "0:07"


def _rich_entry(**overrides) -> HistoryEntry:
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
    def test_layout_title_url_then_info_line(self):
        # Plan §6: numbered title; webpage_url on its own line beneath it;
        # played/duration · requester · absolute timestamp on ONE line below.
        [embed] = history_embeds([_rich_entry()])
        assert embed.title == "1. Rich Song"
        assert embed.description.splitlines() == [
            "https://yt.com/v=rich",
            "3:45 / 4:02 · requested by <@42> · <t:1752530000:f>",
        ]

    def test_numbering_follows_given_order(self):
        embeds = history_embeds([_rich_entry(), _rich_entry(title="Second")])
        assert embeds[0].title == "1. Rich Song"
        assert embeds[1].title == "2. Second"

    def test_thumbnail_set_when_present(self):
        [embed] = history_embeds([_rich_entry()])
        assert embed.thumbnail.url == "https://i.ytimg.com/t.jpg"

    def test_no_thumbnail_when_absent(self):
        [embed] = history_embeds([_rich_entry(thumbnail="")])
        assert embed.thumbnail.url is None

    def test_requester_mention_survives_member_departure(self):
        # The raw <@id> mention needs no member cache to render.
        [embed] = history_embeds([_rich_entry(requester_id=999)])
        assert "<@999>" in embed.description

    def test_requester_name_fallback_when_id_unknown(self):
        [embed] = history_embeds(
            [_rich_entry(requester_id=0, requester_name="SomeUser")]
        )
        assert "requested by SomeUser" in embed.description

    def test_timestamp_omitted_when_played_at_unknown(self):
        # played_at == 0 means unknown; <t:0:f> would render "1 January 1970".
        [embed] = history_embeds([_rich_entry(played_at=0.0)])
        assert "<t:" not in embed.description
        assert embed.description.splitlines() == [
            "https://yt.com/v=rich",
            "3:45 / 4:02 · requested by <@42>",
        ]

    def test_over_length_title_truncated_to_discord_limit(self):
        # Discord rejects any embed title > 256 chars, failing the whole send.
        [embed] = history_embeds([_rich_entry(title="x" * 300)])
        assert len(embed.title) == 256
        assert embed.title.endswith("…")

    def test_title_at_limit_not_truncated(self):
        # "1. " (3) + 253 = 256 exactly — must pass through untouched.
        [embed] = history_embeds([_rich_entry(title="y" * 253)])
        assert embed.title == "1. " + "y" * 253
        assert "…" not in embed.title

    def test_empty_input(self):
        assert history_embeds([]) == []
