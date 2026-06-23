"""Shared test helper functions.

Plain module-level functions (not fixtures) so they can be imported directly
from any test file or conftest without routing through pytest's plugin machinery.
"""

from unittest.mock import MagicMock

from discord.utils import MISSING as _DISCORD_MISSING


def _noop_ffmpeg_init(self, *args, **kwargs):
    """Replacement for FFmpegOpusAudio.__init__ that sets _process = MISSING.

    When a test patches FFmpegOpusAudio.__init__, the instance is created without
    running the real __init__, so _process is never assigned.  GC then calls
    __del__ → cleanup() → _kill_process() → _check_process_returncode(), each of
    which reads self._process and raises AttributeError.  Setting _process to MISSING
    replicates the sentinel that discord.py itself writes before spawning the
    subprocess, which causes all three guard methods to return early.
    """
    self._process = _DISCORD_MISSING


def stub_create_task(return_value=None):
    """Return a mock that replaces loop.create_task or asyncio.create_task.

    The real create_task schedules and owns the coroutine.  A plain
    MagicMock(return_value=...) just stores the coroutine in call_args without
    closing it, producing RuntimeWarning: coroutine '...' was never awaited on GC.
    This stub closes each coroutine immediately and returns a configurable mock
    Task so callers' return-value assertions still pass.
    """
    rv = return_value if return_value is not None else MagicMock()

    def _impl(coro):
        coro.close()
        return rv

    return MagicMock(side_effect=_impl)
