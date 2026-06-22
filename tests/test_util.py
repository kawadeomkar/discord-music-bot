"""Tests for src/util.py — queue formatting and logging utilities."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from src.util import get_logger, queue_message, send_queue_phrases


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


class TestSendQueuePhrases:
    async def test_pineapplecat_receives_special_phrase(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        ctx.message.author.name = "pineapplecat"
        await send_queue_phrases(ctx)
        ctx.send.assert_awaited_once()

    async def test_bryan_receives_insult(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        ctx.message.author.name = "Bryan"
        await send_queue_phrases(ctx)
        ctx.send.assert_awaited_once()
        sent_msg = ctx.send.call_args[0][0]
        assert "bryan" in sent_msg.lower()

    async def test_other_user_receives_nothing(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        ctx.message.author.name = "regularuser"
        await send_queue_phrases(ctx)
        ctx.send.assert_not_awaited()
