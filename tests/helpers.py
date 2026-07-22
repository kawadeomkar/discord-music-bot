"""Shared test helper functions.

Plain module-level functions (not fixtures) so they can be imported directly
from any test file or conftest without routing through pytest's plugin machinery.
"""

import asyncio
from typing import Any, Optional, cast
from collections.abc import Callable, Coroutine
from unittest.mock import MagicMock

import discord
from discord.ext import commands
from discord.utils import MISSING as _DISCORD_MISSING

from src.youtube import QueueObject


def command_callback(
    command: "commands.Command[Any, ..., Any]",
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Return a command's raw callback, invocable as ``callback(cog, ctx, ...)``.

    Tests drive commands by calling the undecorated function directly, bypassing
    discord.py's invoke machinery.  ``Command.callback`` is typed as a union of
    the cog-bound and unbound signatures, and a call has to satisfy *both* union
    members to type-check — so passing the cog explicitly (correct at runtime for
    the class-level Command these tests hold) can never resolve statically.  The
    cast collapses the union to the shape every call site actually uses.
    """
    return cast(Callable[..., Coroutine[Any, Any, Any]], command.callback)


def mocked(obj: object) -> MagicMock:
    """The MagicMock behind an attribute that production types as the real thing.

    The fixtures hand MusicPlayer/MusicBot a MagicMock guild, bot and cog, but
    those attributes are *declared* as `discord.Guild`, `commands.Bot`, and so
    on — so reading `.side_effect` off one, or assigning to a read-only property
    like `Guild.voice_client`, is correct at runtime and rejected statically.

    This names that gap at the point of use instead of hiding it: the argument
    is `object` rather than `MagicMock` precisely so it accepts the
    production-typed expression the caller actually has.
    """
    return cast(MagicMock, obj)


def described(embed: "discord.Embed") -> str:
    """An embed's description, asserted non-empty.

    `Embed.description` is `Optional[str]`, and every assertion against it in
    this suite is really "the embed has a description, and it says X". Failing
    on the first half separately says which of the two actually broke.
    """
    assert embed.description is not None
    return embed.description


def queue_object(item: object) -> QueueObject:
    """A queue entry narrowed to `QueueObject`.

    `display_items()` and friends yield `QueueObject | YTSource`, because an
    unresolved Spotify track has no title, requester or duration yet. A test
    reading those fields is asserting the entry is resolved — this says so, and
    fails on that half separately when it isn't.
    """
    assert isinstance(item, QueueObject)
    return item


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
