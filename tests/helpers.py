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


def tier_enabled(*env_vars: str) -> bool:
    """Is an opt-in integration tier turned on by its environment?

    One definition for all three readers — conftest's gate hook and the two
    tier modules' own skipif — because a gate that disagrees with the thing it
    gates is worse than no gate: it either refuses a run that would have worked
    or waves through one that skips everything.

    "0", "false" and "" are DISABLED. `bool(os.getenv(...))` reads all three as
    enabled, so `RUN_PG_TESTS=0` started a testcontainer — the opposite of what
    an operator writing 0 asked for.
    """
    import os

    return any(
        os.getenv(name, "").strip().lower() not in ("", "0", "false", "no")
        for name in env_vars
    )


def bind_loopback_only(container: Any, port: int) -> None:
    """Publish `port` on 127.0.0.1 instead of every interface.

    Docker's default binding is 0.0.0.0, so a plain testcontainer puts its
    throwaway database on the LAN for the length of a run — `just test-pg` is
    ~45s of `test`/`test` reachable from any machine on the same coffee-shop
    wifi, and the redis tier the same with no auth at all. The contents are
    synthesized fixtures, so the exposure is the surface (an unauthenticated
    Redis, a Postgres 18 with a two-letter password), not the data.

    This matches what docker-compose.yml already does for every service it
    publishes — the whole `network_mode: host` / loopback analysis the compose
    file rests on — so the test tiers agreeing with it is the point.

    Mechanism: testcontainers passes `.ports` straight through to docker-py's
    `containers.run(ports=...)`, which accepts `(host_ip, host_port)` where a
    bare port would go. `None` keeps the random-high-port assignment, so
    `get_exposed_port()` resolves exactly as before. Must be called BEFORE the
    container starts, since that dict is read once at `run`.
    """
    container.ports[port] = ("127.0.0.1", None)
