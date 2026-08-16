"""Tests for src/util.py — queue formatting, the typing indicator, and logging
utilities."""

import asyncio
import contextlib
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.util import (
    _typing_keepalive,
    background_typing,
    fmt_duration,
    get_logger,
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
        """The rule is `count == 1`, not `count <= 1`: 0 and -1 separate the two,
        since `<= 1` renders "0 song" / "-1 song" and passes every other test."""
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


class TestBackgroundTyping:
    """Typing indicator must never delay the command body."""

    async def test_body_runs_while_first_typing_post_is_in_flight(
        self, mock_ctx: MagicMock
    ) -> None:
        post_started = asyncio.Event()
        release_post = asyncio.Event()

        async def slow_post() -> None:
            post_started.set()
            await release_post.wait()

        mock_ctx.typing.return_value.__aenter__ = AsyncMock(side_effect=slow_post)

        async with background_typing(mock_ctx):
            # The body is executing while the POST is still blocked — the ~500ms
            # first POST no longer serializes ahead of the work.
            await asyncio.wait_for(post_started.wait(), timeout=1)
            assert not release_post.is_set()

    async def test_cancel_during_first_post_never_enters_typing_cm(
        self, mock_ctx: MagicMock
    ) -> None:
        """Exiting while the first POST is in flight must not leak the keepalive:
        the CM never entered, so __aexit__ is never called (the AttributeError
        hazard of driving __aenter__/__aexit__ manually cannot occur)."""
        post_started = asyncio.Event()

        async def hung_post() -> None:
            post_started.set()
            await asyncio.sleep(3600)

        mock_ctx.typing.return_value.__aenter__ = AsyncMock(side_effect=hung_post)

        async with background_typing(mock_ctx):
            await asyncio.wait_for(post_started.wait(), timeout=1)
        await asyncio.sleep(0)  # let the cancelled keepalive task unwind
        mock_ctx.typing.return_value.__aexit__.assert_not_awaited()

    async def test_typing_cm_exited_after_body_completes(
        self, mock_ctx: MagicMock
    ) -> None:
        exited = asyncio.Event()
        mock_ctx.typing.return_value.__aexit__ = AsyncMock(
            side_effect=lambda *a: exited.set()
        )
        entered = asyncio.Event()
        mock_ctx.typing.return_value.__aenter__ = AsyncMock(
            side_effect=lambda: entered.set()
        )

        async with background_typing(mock_ctx):
            await asyncio.wait_for(entered.wait(), timeout=1)
        # Cancellation unwinds through the async with → indicator dropped promptly.
        await asyncio.wait_for(exited.wait(), timeout=1)
        mock_ctx.typing.return_value.__aexit__.assert_awaited_once()

    async def test_typing_failure_never_surfaces_into_command_body(
        self, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.typing.side_effect = RuntimeError("typing endpoint down")

        async with background_typing(mock_ctx):
            await asyncio.sleep(0)  # let the keepalive task hit the failure
            await asyncio.sleep(0)
        # No exception propagates; the command body is unaffected.

    async def test_body_exception_still_cancels_keepalive(
        self, mock_ctx: MagicMock
    ) -> None:
        entered = asyncio.Event()
        exited = asyncio.Event()
        mock_ctx.typing.return_value.__aenter__ = AsyncMock(
            side_effect=lambda: entered.set()
        )
        mock_ctx.typing.return_value.__aexit__ = AsyncMock(
            side_effect=lambda *a: exited.set()
        )

        with pytest.raises(ValueError):
            async with background_typing(mock_ctx):
                await asyncio.wait_for(entered.wait(), timeout=1)
                raise ValueError("command body blew up")
        await asyncio.wait_for(exited.wait(), timeout=1)


class TestTypingKeepaliveCancellation:
    """_typing_keepalive must catch Exception only and let CancelledError propagate.

    Statement coverage cannot see the distinction — one `except` line serves both
    arms, so a handler that also swallows CancelledError reports as covered while
    completing the task *normally*. These assert `task.cancelled()`, the only
    observable that separates the two; `done()` is true either way."""

    async def test_cancelled_keepalive_ends_cancelled_not_completed(
        self, mock_ctx: MagicMock
    ) -> None:
        """cancel() must leave the task cancelled, not just done(). Swallowing
        CancelledError makes cooperative cancellation a lie: task.cancelled() is
        False, and a cancellation aimed at an enclosing scope stops here."""
        task = asyncio.create_task(_typing_keepalive(mock_ctx))
        await asyncio.sleep(0)  # let it reach the sleep(3600)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert task.cancelled(), (
            "keepalive completed normally instead of ending cancelled — "
            "the handler is swallowing CancelledError"
        )
        # done() alone cannot tell the two apart; that is why it is not the assert.
        assert task.done()

    async def test_cancellation_still_exits_the_typing_cm(
        self, mock_ctx: MagicMock
    ) -> None:
        """Letting CancelledError propagate must not skip Typing.__aexit__ — the
        `async with` unwind is what drops the indicator, and skipping it leaves the
        bot apparently typing forever after every command."""
        exited = asyncio.Event()
        entered = asyncio.Event()
        mock_ctx.typing.return_value.__aenter__ = AsyncMock(
            return_value=None, side_effect=lambda: entered.set()
        )
        # return_value=None is required for any cancellation assert through this
        # CM: an __aexit__ returning a truthy value suppresses the exception being
        # unwound, so a bare AsyncMock() eats the CancelledError and fails the assert
        # below for an unrelated reason. Real Typing.__aexit__ returns None.
        mock_ctx.typing.return_value.__aexit__ = AsyncMock(
            return_value=None, side_effect=lambda *a: exited.set()
        )

        task = asyncio.create_task(_typing_keepalive(mock_ctx))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        await asyncio.wait_for(exited.wait(), timeout=1)
        mock_ctx.typing.return_value.__aexit__.assert_awaited_once()
        assert task.cancelled()

    async def test_cosmetic_typing_failure_is_still_swallowed(
        self, mock_ctx: MagicMock
    ) -> None:
        """The Exception arm must survive: typing failures stay invisible."""
        mock_ctx.typing.side_effect = RuntimeError("typing endpoint down")

        task = asyncio.create_task(_typing_keepalive(mock_ctx))
        await task  # must not raise

        assert task.done() and not task.cancelled()
        assert task.exception() is None

    async def test_base_exception_is_not_swallowed(self, mock_ctx: MagicMock) -> None:
        """Only Exception is cosmetic; a BaseException must propagate — CancelledError
        is one, so this pins the general rule. A custom subclass rather than
        SystemExit: asyncio re-raises SystemExit/KeyboardInterrupt into the loop,
        escaping pytest.raises for reasons unrelated to this handler."""

        class Shutdown(BaseException):
            pass

        mock_ctx.typing.side_effect = Shutdown("shutting down")

        task = asyncio.create_task(_typing_keepalive(mock_ctx))
        with pytest.raises(Shutdown):
            await task

    async def test_background_typing_leaves_its_keepalive_cancelled(
        self, mock_ctx: MagicMock
    ) -> None:
        """End-to-end: the task background_typing() spawns and cancels on exit
        must settle as cancelled. background_typing does not await it, so this
        is the only place the contract is observable from outside."""
        spawned: list[asyncio.Task[Any]] = []
        real_create_task = asyncio.create_task

        def capture(coro: Any, **kw: Any) -> asyncio.Task[Any]:
            task = real_create_task(coro, **kw)
            spawned.append(task)
            return task

        with patch("src.util.asyncio.create_task", side_effect=capture):
            async with background_typing(mock_ctx):
                await asyncio.sleep(0)

        assert len(spawned) == 1
        keepalive = spawned[0]
        with contextlib.suppress(asyncio.CancelledError):
            await keepalive
        assert keepalive.cancelled()
