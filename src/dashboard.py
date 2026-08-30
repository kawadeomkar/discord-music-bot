"""Optimistic-send + live-edit driver, shared by `-ping` and `-debug`.

Both commands answer a question that needs real IO across several services, and
both would otherwise leave the user watching nothing while it happens. The shape
is identical in each: launch the slow work concurrently, send what is already
known immediately, edit that one message as results land, and stop at a deadline
so a dead dependency cannot hold the reply open forever.

What differs is only what a "result" IS — a ProbeResult row for -ping, a block of
rendered lines for -debug — so that stays with the callers. The sequencing, the
edit-only-on-change rule and the never-leak-a-task guarantee live here, because
those are the parts that are subtle and were worth getting right once.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, Optional, TypeVar

import discord
from discord.ext import commands

from src.util import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Floor on a single wait, so a mis-set tick degrades to a slow board rather than a
# hot loop. Well below any sane tick, so it never shapes normal behaviour.
_MIN_WAIT_SECS = 0.01


def _retrieve_exception(task: asyncio.Task[Any]) -> None:
    """Mark a settled probe's exception as retrieved, wherever it settles.

    Attached at creation rather than handled in the driver's `finally`, because a
    probe cancelled at the deadline can still RAISE while unwinding — an aiohttp or
    asyncpg `__aexit__` doing cleanup — and it settles AFTER the driver has already
    returned. `finally` cannot see that one, so asyncio logs "Task exception was
    never retrieved" against a command that handled the failure fine.
    """
    if not task.cancelled():
        task.exception()


def embeds_changed(new: list[discord.Embed], old: list[discord.Embed]) -> bool:
    """True when a re-render actually differs. Discord rate-limits edits, so an
    edit that changes nothing spends budget for nothing; to_dict() captures every
    field a live dashboard moves (values, colour, footer)."""
    return [e.to_dict() for e in new] != [e.to_dict() for e in old]


async def safe_edit(message: discord.Message, embeds: list[discord.Embed]) -> bool:
    """Edit a message, tolerating one the user deleted mid-loop (mirrors
    musicplayer._push_np_edit, including its second arm). False when it is gone.

    Every other HTTPException is logged and swallowed, because the card is built
    INCREMENTALLY: raising here cancels every still-running probe and strands the
    skeleton in the channel next to a separate error embed, so a 400 on one edit
    costs the whole snapshot. A stale card beats that. NotFound must be caught
    first — it is an HTTPException subclass.
    """
    try:
        await message.edit(embeds=embeds)
        return True
    except discord.NotFound:
        return False
    except discord.HTTPException as e:
        log.warning("dashboard edit failed", error=str(e))
        return True


def outcome_of(task: asyncio.Task[T]) -> T | Exception:
    """A finished task's value, or the exception it died of.

    CancelledError is deliberately not caught: the driver only reads tasks it has
    not cancelled, so seeing one means the surrounding command is being torn down
    and it must keep unwinding.
    """
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

    `settle(key, outcome)` folds one finished probe into the caller's state and
    `abandon(key)` marks a straggler the deadline gave up on; both mutate state
    that `render()` closes over, which is what keeps this driver ignorant of what
    the caller is actually displaying. `outcome` is the probe's value OR the
    exception it died of — handed over already resolved, so a settle callback
    cannot accidentally re-raise into the driver's own task and take the whole
    board down. `prepare()` runs after launch and before the first send — the place
    for instant data whose own await gives the probes a head start.

    Returns the message, or None if it was deleted mid-loop. Runs inside the
    caller's span and lets exceptions propagate: the command owns the error reply.
    """
    loop = asyncio.get_running_loop()
    tasks: dict[str, asyncio.Task[T]] = {}
    pending: set[str] = set()

    def _drain() -> bool:
        """Fold every finished task into caller state; True if anything moved.
        Only genuinely-done tasks are settled here, so `.result()` cannot block."""
        changed = False
        for key in [k for k in pending if tasks[k].done()]:
            pending.discard(key)
            settle(key, outcome_of(tasks[key]))
            changed = True
        return changed

    try:
        # Launched inside the try so `finally` cancels them wherever a later await
        # raises. create_task copies the otel context, so probe spans nest under
        # the command's span rather than floating as roots.
        #
        # Filled one at a time rather than by comprehension: a comprehension binds
        # `tasks` only once it completes, so a raising fn() would leave the tasks it
        # had already created invisible to `finally` — leaked, and unretrieved.
        for key, fn in probes.items():
            task = asyncio.create_task(fn())
            task.add_done_callback(_retrieve_exception)
            tasks[key] = task
        pending = set(tasks)

        # Instant data, then the skeleton. prepare()'s await is also what lets an
        # already-resolvable probe finish, and the pre-drain is what stops its row
        # flashing "pending" for one tick before the first edit.
        if prepare is not None:
            await prepare()
        # One explicit yield before the pre-drain. create_task only SCHEDULES a
        # coroutine, so a probe that needs no IO has not run yet; whether it got a
        # turn otherwise depends on prepare() happening to await something real,
        # which is too subtle to rely on. Without this, an already-answerable row
        # renders its placeholder and needs a whole tick to correct itself.
        await asyncio.sleep(0)
        _drain()
        last = render()
        # channel.send, NOT ctx.send: an edit loop must not become the Now Playing
        # host, whose progress updater re-renders that same message every few
        # seconds. Two costs, and the second is the bigger one — these replies carry
        # no NP block, AND the existing host is not retired, so it stays above this
        # card instead of staying glued to the bottom of the channel until the next
        # ctx.send adopts a new host. -debug's card is nine blocks tall and lives for
        # the whole deadline, so the displacement is visible. A third, cheaper cost:
        # this also bypasses debug-mode decoration, so callers pre-render a footer
        # and thread it through `render`.
        # See docs/ARCHITECTURE.md#now-playing-host-model.
        message = await ctx.channel.send(embeds=last)

        deadline = loop.time() + deadline_secs
        last_edit_at = loop.time()
        while pending and (remaining := deadline - loop.time()) > 0:
            # Wake on the first probe to finish, not on a fixed cadence. A plain
            # sleep(tick) makes tick the RESOLUTION of the board: a probe landing
            # just after one tick sits invisible until the next, so a 0.5s answer
            # shows up at 1.0s.
            # min(): the deadline caps the last wait, so a tick larger than the time
            # left cannot overrun it. max(): a non-positive tick would make wait()
            # return instantly forever — config refuses that from the environment,
            # this covers a direct caller.
            await asyncio.wait(
                [tasks[k] for k in pending],
                timeout=max(min(tick_secs, remaining), _MIN_WAIT_SECS),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not _drain():
                continue
            # Waking per probe is not the same as EDITING per probe. Discord buckets
            # message edits per CHANNEL — message_id is not in the rate-limit key — so
            # a burst here competes with the Now Playing progress bar editing that same
            # channel every few seconds, and four probes answering together would spend
            # four edits inside a few ms. tick_secs is the floor between edits; the
            # flush below carries whatever the last skipped edit would have shown.
            if loop.time() - last_edit_at < tick_secs:
                continue
            embeds = render()
            if embeds_changed(embeds, last):
                if not await safe_edit(message, embeds):
                    return None  # deleted mid-loop; nothing left to edit into
                last = embeds
                last_edit_at = loop.time()

        # Give up on what is STILL pending. Re-check done() first: a probe can
        # finish during the last edit's await, and abandoning it unconditionally
        # would report a healthy dependency as failed.
        for key in pending:
            if tasks[key].done():
                settle(key, outcome_of(tasks[key]))
            else:
                tasks[key].cancel()
                abandon(key)
        # Unconditional flush. Coalescing above can be holding a settled result that
        # never earned an edit, so this is not reachable only via the abandon path.
        embeds = render()
        if embeds_changed(embeds, last):
            if not await safe_edit(message, embeds):
                return None
        return message
    finally:
        # Never leak a probe task. Retrieval is _retrieve_exception's job, attached
        # at creation because it also has to cover a probe that settles after this
        # returns; this loop only has to make sure every task is asked to stop,
        # including on an early-return or a raise partway through the launch loop.
        for t in tasks.values():
            if not t.done():
                t.cancel()
