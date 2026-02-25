"""Tests for src/util.py — queue formatting and logging utilities."""
import logging

import pytest

from src.util import get_logger, queue_message


class TestQueueMessage:
    def test_empty_list_returns_empty_string(self):
        assert queue_message([]) == ""

    def test_two_items_shows_first_item(self):
        # range(1, len(songs[:10])) with 2 items = range(1, 2) = [1]
        # Only index 0 is shown; the last song is always omitted (off-by-one)
        result = queue_message(["song_a", "song_b"])
        assert "1: song_a" in result
        assert "song_b" not in result

    def test_five_items_shows_four(self):
        # range(1, 5) = [1, 2, 3, 4] — shows songs[0..3], not songs[4]
        songs = [f"song{i}" for i in range(5)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line]
        assert len(lines) == 4
        assert "1: song0" in result
        assert "4: song3" in result
        assert "song4" not in result

    def test_exactly_ten_items_no_ellipsis(self):
        songs = [f"track{i}" for i in range(10)]
        result = queue_message(songs)
        assert "..." not in result

    def test_more_than_ten_items_appends_ellipsis(self):
        songs = [f"track{i}" for i in range(15)]
        result = queue_message(songs)
        assert "..." in result

    def test_more_than_ten_items_caps_at_nine_shown(self):
        # songs[:10] has 10 entries; range(1, 10) = 9 entries shown
        songs = [f"track{i}" for i in range(20)]
        result = queue_message(songs)
        lines = [line for line in result.split("\n") if line and line != "..."]
        assert len(lines) == 9

    def test_numbering_starts_at_one(self):
        result = queue_message(["first", "second", "third"])
        assert result.startswith("1:")

    def test_songs_sliced_to_ten_before_processing(self):
        # Ensures songs beyond index 9 are never in the output
        songs = [f"song{i}" for i in range(25)]
        result = queue_message(songs)
        assert "song15" not in result
        assert "song20" not in result


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        logger = get_logger("my.custom.module")
        assert logger.name == "my.custom.module"

    def test_logger_level_is_info(self):
        logger = get_logger("test.level_check")
        assert logger.level == logging.INFO

    def test_logger_has_exactly_one_handler(self):
        logger = get_logger("test.single_handler")
        assert len(logger.handlers) == 1

    def test_calling_twice_does_not_add_duplicate_handlers(self):
        logger_a = get_logger("test.no_dup")
        logger_b = get_logger("test.no_dup")
        assert logger_a is logger_b
        assert len(logger_b.handlers) == 1

    def test_handler_is_stream_handler(self):
        logger = get_logger("test.stream_handler")
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_handler_formatter_includes_name_and_level(self):
        logger = get_logger("test.formatter_check")
        formatter = logger.handlers[0].formatter
        fmt = formatter._fmt
        assert "%(name)s" in fmt
        assert "%(levelname)s" in fmt
        assert "%(message)s" in fmt

    def test_different_names_return_different_loggers(self):
        logger_a = get_logger("module.a")
        logger_b = get_logger("module.b")
        assert logger_a is not logger_b
        assert logger_a.name != logger_b.name
