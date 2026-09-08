"""Shared test helper functions.

Plain module-level functions (not fixtures) so any test file or conftest can
import them directly, without routing through pytest's plugin machinery.
"""

import asyncio
import contextlib
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Optional, cast
from collections.abc import Callable, Coroutine, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from discord.ext import commands
from discord.utils import MISSING as _DISCORD_MISSING

from src.guild_queue import GuildQueue, QueueItem
from src.play_placement import PlayMode, PlayRequest
from src.youtube import QueueObject

if TYPE_CHECKING:
    from src.musicbot import MusicBot


def admit(
    music_bot: MusicBot,
    ctx: MagicMock,
    mp: Any,
    *,
    mode: PlayMode = PlayMode.NORMAL,
    query: str = "test",
) -> PlayRequest:
    """Register a request as -play would, so a helper that inserts under the
    place lock can be called directly. A bare MagicMock answers `retired` with a
    truthy mock, which place() reads as torn down; pin it as mock_mp() does."""
    if isinstance(mp, MagicMock) and isinstance(mp.retired, MagicMock):
        mp.retired = False
    return music_bot._plays.register(ctx, query=query, mp=mp, mode=mode)


async def settle(ticks: int = 12) -> None:
    """Let every runnable task reach its next suspension point."""
    for _ in range(ticks):
        await asyncio.sleep(0)


def song(n: int, ctx: MagicMock) -> QueueObject:
    return QueueObject(f"https://yt.com/v={n}", f"Song {n}", ctx.author)


@contextlib.contextmanager
def recording_span() -> Iterator[MagicMock]:
    """A span double in place of the current span. Its context is invalid so
    structlog's OTel processor, which reads the same global, skips the trace-id
    format it would otherwise apply to a MagicMock."""
    with patch("src.musicbot.trace.get_current_span") as current:
        span = current.return_value
        span.get_span_context.return_value.is_valid = False
        yield span


def command_callback(
    command: commands.Command[Any, ..., Any],
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Return a command's raw callback, invocable as ``callback(cog, ctx, ...)``.

    ``Command.callback`` is a union of the cog-bound and unbound signatures and a
    call must satisfy *both*, so passing the cog explicitly (correct at runtime)
    can never type-check. The cast collapses it to the shape callers use."""
    return cast(Callable[..., Coroutine[Any, Any, Any]], command.callback)


def mocked(obj: object) -> MagicMock:
    """The MagicMock behind an attribute that production types as the real thing.

    Fixtures hand MusicPlayer/MusicBot mocks declared as `discord.Guild` /
    `commands.Bot`, so reading `.side_effect` or assigning a read-only property is
    right at runtime and rejected statically. The parameter is `object`, not
    `MagicMock`, precisely so it accepts the production-typed expression."""
    return cast(MagicMock, obj)


def described(embed: discord.Embed) -> str:
    """An embed's description, asserted non-empty.

    `.description` is `Optional[str]` and every assertion means "present, and it
    says X"; failing the first half separately says which of the two broke."""
    assert embed.description is not None
    return embed.description


def queue_object(item: object) -> QueueObject:
    """A queue entry narrowed to `QueueObject`.

    `display_items()` yields `QueueObject | YTSource` (an unresolved Spotify track
    has no title/requester/duration), so reading those fields asserts resolved."""
    assert isinstance(item, QueueObject)
    return item


def noop_ffmpeg_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """Replacement for FFmpegOpusAudio.__init__ that stubs all pre-spawn attributes.

    Patching out the real __init__ leaves the sentinels unassigned, so GC's
    __del__ → cleanup() → _kill_process() reads them and raises AttributeError.
    These mirror discord.py's pre-spawn state, so every guard returns early."""
    self._process = _DISCORD_MISSING
    self._stopped = False
    self._stdout = None
    self._stdin = None
    self._stderr = None


def stub_create_task(return_value: Optional[Any] = None) -> MagicMock:
    """Return a mock that replaces loop.create_task or asyncio.create_task.

    A plain MagicMock(return_value=...) parks the coroutine in call_args unclosed,
    raising "coroutine was never awaited" on GC. This closes each one immediately
    and returns a configurable mock Task so return-value assertions pass."""

    def _impl(coro: Coroutine[Any, Any, Any]) -> Any:
        coro.close()
        return return_value if return_value is not None else MagicMock()

    return MagicMock(side_effect=_impl)


def make_mock_task() -> MagicMock:
    """A MagicMock resembling a running asyncio.Task, for cancellation asserts."""
    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False
    task.cancel = MagicMock()
    return task


def tier_enabled(*env_vars: str) -> bool:
    """Is an opt-in integration tier turned on by its environment?

    One definition for all three readers (conftest's gate hook, both tiers' own
    skipif) — a gate disagreeing with what it gates is worse than no gate. "0",
    "false" and "" are disabled; `bool(os.getenv(...))` reads all three as on."""
    import os

    return any(
        os.getenv(name, "").strip().lower() not in ("", "0", "false", "no")
        for name in env_vars
    )


def bind_loopback_only(container: Any, port: int) -> None:
    """Publish `port` on 127.0.0.1 instead of every interface.

    Docker binds 0.0.0.0, so a plain testcontainer exposes an unauthenticated
    Redis / weak-password Postgres to the LAN for a whole run. testcontainers
    hands `.ports` to docker-py, which accepts `(host_ip, host_port)`; `None`
    keeps the random high port. Call before start — that dict is read once."""
    container.ports[port] = ("127.0.0.1", None)


def seed_queue(gq: GuildQueue, *items: QueueItem) -> None:
    """Queue items without touching Redis — `put()` minus the mirror.

    Synchronous, so the sync tests (embeds, ETA) can use it too. Nothing here
    claims: a test wanting an in-flight head calls `get()`, as production does.
    """
    gq._items.extend(items)
    gq._sync_wake()


def no_typing(target: str) -> AbstractContextManager[MagicMock]:
    """Stub a module's background_typing with an inert async CM.

    `target` is the MODULE the command under test resolves the name in, not where
    it is defined, and it has no default: every command body owns its own reference
    now, so a wrong module leaves the real keepalive running instead of failing.

    TestPlayCommand needs this because it patches asyncio.create_task as a join-task
    spy, and the typing keepalive would otherwise hit the same patch, polluting call
    counts and taking the fake join future. The wrapper itself is covered by
    TestBackgroundTyping."""
    return patch(target, MagicMock(return_value=contextlib.nullcontext()))


def in_authors_channel(vc: MagicMock, ctx: Optional[MagicMock]) -> MagicMock:
    """Seat a voice-client double in the author's channel, or somewhere else. Queue
    control is gated on the bot being in the author's channel at dispatch AND at
    the insert, and a double with no channel reads as "somewhere else"."""
    vc.channel = (
        ctx.author.voice.channel
        if ctx is not None
        else MagicMock(spec=discord.VoiceChannel)
    )
    return vc


def connected_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client, nothing playing — what a successful cold join leaves
    behind. is_connected is explicit: the cold path checks it, because discord.py
    registers the client on the guild before the handshake completes."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return in_authors_channel(vc, ctx)


def playing_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client, actively playing. Both flags must be set explicitly:
    an unstubbed is_paused() returns a truthy Mock, silently sending -play down the
    interjection branch instead of the append path."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return in_authors_channel(vc, ctx)


def paused_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client with a song parked paused. is_connected is explicit:
    -resume's rejoin checks it, and an auto-vivified one answers True by accident
    rather than by choice."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = True
    vc.is_connected.return_value = True
    return in_authors_channel(vc, ctx)


def mock_mp(qsize: int = 0) -> MagicMock:
    """MusicPlayer stand-in for the -play cold path, with the playback-gate
    hooks awaitable: play() takes defer_playback() as an async context manager
    and awaits wait_for_restore() before front-inserting."""
    mp = MagicMock()
    mp.defer_playback = MagicMock(return_value=contextlib.nullcontext())
    mp.wait_for_restore = AsyncMock(return_value=True)
    # Numeric, not auto-vivified: _abandon_cold_start COMPARES this, and a Mock
    # raises TypeError there rather than answering.
    mp.playback_holds = 1  # the hold this command itself takes
    # Explicit, not auto-vivified: place() reads it, and a MagicMock is truthy.
    mp.retired = False
    mp.queue.generation = 0
    mp.repark_crashed_head = AsyncMock()
    # Awaitable, not auto-vivified: interject_flow settles the prefetch BEFORE it
    # takes the place lock, and a bare Mock is not awaitable there.
    mp.settle_prefetch = AsyncMock()
    mp.queue_put_front = AsyncMock()
    mp.queue_put = AsyncMock()
    # `--next` inserts through its own wrapper, which neutralizes the loop's
    # prefetch first — a plain front insert lands behind that claim.
    mp.queue_put_next = AsyncMock()
    mp.queue.claim_outstanding = MagicMock(return_value=False)
    mp.queue.qsize = MagicMock(return_value=qsize)
    # Numeric for the same reason as playback_holds: this lands in
    # Analytics.queue_position and rides to Postgres through HistoryEntry's
    # integer clamp, which a Mock raises on rather than answering.
    mp.enqueue_depth = MagicMock(return_value=qsize)
    # Numeric for that reason too: _cold_start_left_something_playable compares it
    # to decide whether a late refusal may disconnect the session, and an
    # auto-vivified Mock is truthy — which spares every teardown these tests pin.
    mp.queue.display_size = MagicMock(return_value=qsize)
    # Mirrors the real builder's contract: a notice only when the restore
    # actually left something in the queue (see build_resume_notice_embed).
    mp.build_resume_notice_embed = MagicMock(
        return_value=discord.Embed(title="❗ Resumed from queue") if qsize else None
    )
    return mp
