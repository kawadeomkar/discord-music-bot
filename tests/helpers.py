"""Shared test helper functions.

Plain module-level functions (not fixtures) so they can be imported directly
from any test file or conftest without routing through pytest's plugin machinery.
"""

import asyncio
from typing import Any, Coroutine, Optional
from unittest.mock import MagicMock

from discord.utils import MISSING as _DISCORD_MISSING


def noop_ffmpeg_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """Replacement for FFmpegOpusAudio.__init__ that stubs all pre-spawn attributes.

    When a test patches FFmpegOpusAudio.__init__, the instance is created without
    running the real __init__, so the pre-spawn sentinels are never assigned.  GC
    then calls __del__ → cleanup() → _kill_process() / _check_process_returncode(),
    each of which reads these attributes and raises AttributeError.  Setting them
    here mirrors the pre-spawn state discord.py itself establishes before the
    subprocess is started, causing all guard methods to return early.
    """
    self._process = _DISCORD_MISSING
    self._stopped = False
    self._stdout = None
    self._stdin = None
    self._stderr = None


def stub_create_task(return_value: Optional[Any] = None) -> MagicMock:
    """Return a mock that replaces loop.create_task or asyncio.create_task.

    The real create_task schedules and owns the coroutine.  A plain
    MagicMock(return_value=...) just stores the coroutine in call_args without
    closing it, producing RuntimeWarning: coroutine '...' was never awaited on GC.
    This stub closes each coroutine immediately and returns a configurable mock
    Task so callers' return-value assertions still pass.
    """

    def _impl(coro: Coroutine[Any, Any, Any]) -> Any:
        coro.close()
        return return_value if return_value is not None else MagicMock()

    return MagicMock(side_effect=_impl)


def make_mock_task() -> MagicMock:
    """Return a MagicMock resembling a running asyncio.Task.

    Shorthand for the three-line boilerplate used wherever a test needs to
    verify that a running timer or background task is cancelled.
    """
    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False
    task.cancel = MagicMock()
    return task
