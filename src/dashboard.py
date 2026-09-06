"""Optimistic-send + live-edit driver, shared by `-ping` and `-debug`: launch
the probes concurrently, send what is known immediately, edit that one message
as results land, and stop at a deadline. What a "result" is stays with the
callers; sequencing, edit-only-on-change and never-leak-a-task live here."""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, Optional, TypeVar

import discord
from discord.ext import commands

from src.util import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Floor on a single wait: a mis-set tick degrades to a slow board, not a hot loop.
_MIN_WAIT_SECS = 0.01


def _retrieve_exception(task: asyncio.Task[Any]) -> None:
    """Mark a settled probe's exception as retrieved. Attached at creation: a
    probe cancelled at the deadline can raise while unwinding and settle AFTER
    the driver returned, where its `finally` cannot see it."""
    if not task.cancelled():
        task.exception()


def embeds_changed(new: list[discord.Embed], old: list[discord.Embed]) -> bool:
    """True when a re-render differs; Discord rate-limits edits, and to_dict()
    captures every field a live dashboard moves."""
    return [e.to_dict() for e in new] != [e.to_dict() for e in old]


async def safe_edit(message: discord.Message, embeds: list[discord.Embed]) -> bool:
    """Edit a message; False when the user deleted it mid-loop. Every other
    HTTPException is logged and swallowed: raising would cancel every running
    probe and strand the skeleton, so a stale card beats it. NotFound first —
    it is an HTTPException subclass."""
    try:
        await message.edit(embeds=embeds)
        return True
    except discord.NotFound:
        return False
    except discord.HTTPException as e:
        log.warning("dashboard edit failed", error=str(e))
        return True


def outcome_of(task: asyncio.Task[T]) -> T | Exception:
    """A finished task's value, or the exception it died of. CancelledError is
    not caught: the driver reads only tasks it did not cancel, so one means the
    command is being torn down."""
    try:
        return task.result()
    except Exception as e:  # noqa: BLE001 — the caller renders it as a dead row
        return e


async def run_live_dashboard(
    ctx: commands.Context,
    *,
    probes: Mapping[str, Callable[[], Coroutine[Any, Any, T]]],
    settle: Callable[[str, T | Exception], None],
    abandon: Callable[[str], None],
    render: Callable[[], list[discord.Embed]],
    tick_secs: float,
    deadline_secs: float,
    prepare: Optional[Callable[[], Awaitable[None]]] = None,
) -> Optional[discord.Message]:
    """Send a skeleton immediately, then edit it as `probes` return.

    `settle(key, outcome)` folds one finished probe (its value OR the exception
    it died of, already resolved) into state `render()` closes over; `abandon(key)`
    marks a straggler the deadline gave up on. `prepare()` runs after launch and
    before the first send. Returns the message, or None if it was deleted
    mid-loop. Exceptions propagate: the command owns the error reply."""
    loop = asyncio.get_running_loop()
    tasks: dict[str, asyncio.Task[T]] = {}
    pending: set[str] = set()

    def _drain() -> bool:
        """Fold every finished task into caller state; True if anything moved."""
        changed = False
        for key in [k for k in pending if tasks[k].done()]:
            pending.discard(key)
            settle(key, outcome_of(tasks[key]))
            changed = True
        return changed

    try:
        # Inside the try so `finally` cancels them wherever a later await raises,
        # and one at a time: a comprehension binds `tasks` only once complete, so
        # a raising fn() would leak the tasks already created.
        for key, fn in probes.items():
            task = asyncio.create_task(fn())
            task.add_done_callback(_retrieve_exception)
            tasks[key] = task
        pending = set(tasks)

        if prepare is not None:
            await prepare()
        # create_task only schedules; one explicit yield lets a probe that needs
        # no IO settle before the pre-drain, so its row never flashes "pending".
        await asyncio.sleep(0)
        _drain()
        last = render()
        # channel.send, NOT ctx.send: an edit loop must not become the Now Playing
        # host. This also bypasses debug-mode decoration, so callers pre-render a
        # footer and thread it through `render`.
        # See docs/ARCHITECTURE.md#now-playing-host-model.
        message = await ctx.channel.send(embeds=last)

        deadline = loop.time() + deadline_secs
        last_edit_at = loop.time()
        while pending and (remaining := deadline - loop.time()) > 0:
            # Wake on the first probe to finish, not on a fixed cadence. min():
            # the deadline caps the last wait; max(): a non-positive tick would
            # make wait() return instantly forever.
            await asyncio.wait(
                [tasks[k] for k in pending],
                timeout=max(min(tick_secs, remaining), _MIN_WAIT_SECS),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not _drain():
                continue
            # Discord buckets edits per CHANNEL, shared with the NP progress bar,
            # so tick_secs is the floor between edits; the flush below carries
            # whatever a skipped edit would have shown.
            if loop.time() - last_edit_at < tick_secs:
                continue
            embeds = render()
            if embeds_changed(embeds, last):
                if not await safe_edit(message, embeds):
                    return None  # deleted mid-loop; nothing left to edit into
                last = embeds
                last_edit_at = loop.time()

        # Re-check done() first: a probe can finish during the last edit's await.
        for key in pending:
            if tasks[key].done():
                settle(key, outcome_of(tasks[key]))
            else:
                tasks[key].cancel()
                abandon(key)
        # Unconditional flush: coalescing above can hold a settled result.
        embeds = render()
        if embeds_changed(embeds, last):
            if not await safe_edit(message, embeds):
                return None
        return message
    finally:
        # Never leak a probe task; retrieval is _retrieve_exception's job.
        for t in tasks.values():
            if not t.done():
                t.cancel()
