"""The optimistic-send + live-edit driver shared by -ping and -debug.

These assert the SEQUENCING — send before the slow work finishes, edit only on a
real change, give up at the deadline, never leak a task. What each command puts in
those embeds is tested in test_ping.py and test_debug.py.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from src import dashboard
from src.dashboard import embeds_changed, outcome_of, run_live_dashboard


@pytest.fixture
def dash_ctx(mock_ctx: MagicMock) -> MagicMock:
    """A ctx whose channel.send returns an editable message double."""
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    mock_ctx.channel.send = AsyncMock(return_value=message)
    return mock_ctx


def _embed(text: str) -> list[discord.Embed]:
    return [discord.Embed(title=text)]


async def _until(predicate: Any, timeout: float = 2.0) -> None:
    """Spin until a predicate holds. pytest-timeout is the backstop, but a bounded
    wait fails with a readable assertion instead of a cancellation."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


class TestEmbedsChanged:
    def test_identical_renders_do_not_count_as_a_change(self) -> None:
        assert not embeds_changed(_embed("a"), _embed("a"))

    def test_a_moved_field_counts(self) -> None:
        assert embeds_changed(_embed("a"), _embed("b"))

    def test_a_added_or_removed_embed_counts(self) -> None:
        """The list, not just the last embed: -ping carries a static advisory
        alongside its card, and -debug's block count varies with the caller."""
        assert embeds_changed(_embed("a") + _embed("b"), _embed("a"))


class TestSafeEdit:
    async def test_returns_true_on_a_successful_edit(self) -> None:
        message = MagicMock(spec=discord.Message)
        message.edit = AsyncMock()
        assert await dashboard.safe_edit(message, _embed("x")) is True

    async def test_a_message_deleted_mid_loop_does_not_raise(self) -> None:
        """A user deleting the dashboard message must not take the command down."""
        message = MagicMock(spec=discord.Message)
        message.edit = AsyncMock(side_effect=discord.NotFound(MagicMock(), "gone"))
        assert await dashboard.safe_edit(message, _embed("x")) is False


class TestOutcomeOf:
    async def test_a_value_comes_back_as_itself(self) -> None:
        task = asyncio.create_task(asyncio.sleep(0, result="v"))
        await task
        assert outcome_of(task) == "v"

    async def test_an_exception_is_returned_not_raised(self) -> None:
        """settle() runs inside the driver's loop, so a raising probe would
        otherwise take down the whole dashboard — the thing it exists to render."""

        async def boom() -> None:
            raise RuntimeError("boom")

        task = asyncio.create_task(boom())
        with pytest.raises(RuntimeError):
            await task
        assert isinstance(outcome_of(task), RuntimeError)


class TestTheSendDoesNotWaitForTheProbes:
    """The whole point: the user gets a message before the IO finishes."""

    async def test_the_skeleton_is_sent_while_a_probe_is_still_running(
        self, dash_ctx: MagicMock
    ) -> None:
        gate = asyncio.Event()
        settled: dict[str, Any] = {}

        async def slow() -> str:
            await gate.wait()
            return "done"

        task = asyncio.create_task(
            run_live_dashboard(
                dash_ctx,
                probes={"slow": slow},
                settle=lambda k, v: settled.__setitem__(k, v),
                abandon=lambda k: settled.__setitem__(k, "failed"),
                render=lambda: _embed(str(settled)),
                tick_secs=0.01,
                deadline_secs=2.0,
            )
        )
        # Sent before the probe is allowed to finish — this is the contract.
        await _until(lambda: dash_ctx.channel.send.await_count == 1)
        assert settled == {}
        gate.set()
        await task
        assert settled == {"slow": "done"}

    async def test_prepare_runs_before_the_send_and_after_launch(
        self, dash_ctx: MagicMock
    ) -> None:
        order: list[str] = []

        async def probe() -> str:
            order.append("probe-started")
            return "v"

        async def prepare() -> None:
            order.append("prepare")

        dash_ctx.channel.send = AsyncMock(
            side_effect=lambda **_kw: (
                order.append("send")
                or MagicMock(spec=discord.Message, edit=AsyncMock())
            )
        )
        await run_live_dashboard(
            dash_ctx,
            probes={"p": probe},
            settle=lambda k, v: None,
            abandon=lambda k: None,
            render=lambda: _embed("x"),
            prepare=prepare,
            tick_secs=0.01,
            deadline_secs=1.0,
        )
        assert order.index("probe-started") < order.index("send")
        assert order.index("prepare") < order.index("send")

    async def test_a_probe_finished_before_the_send_never_renders_as_pending(
        self, dash_ctx: MagicMock
    ) -> None:
        """The pre-drain. Without it a probe that resolved during prepare() still
        shows its placeholder for one full tick."""
        state: dict[str, Any] = {"p": "pending"}

        async def instant() -> str:
            return "ok"

        await run_live_dashboard(
            dash_ctx,
            probes={"p": instant},
            settle=lambda k, v: state.__setitem__(k, v),
            abandon=lambda k: None,
            render=lambda: _embed(state["p"]),
            prepare=AsyncMock(),
            tick_secs=0.01,
            deadline_secs=1.0,
        )
        sent = dash_ctx.channel.send.await_args.kwargs["embeds"]
        assert sent[0].title == "ok"


class TestEditing:
    async def test_an_unchanged_render_is_not_edited(self, dash_ctx: MagicMock) -> None:
        """Discord rate-limits edits; a no-op edit spends budget for nothing.

        The probe is DELAYED on purpose. An instant probe settles in the pre-drain,
        `pending` empties, and the loop body never runs — so a version of this test
        with an instant probe passes whatever the change-check does.
        """
        settled: list[Any] = []

        async def probe() -> str:
            await asyncio.sleep(0.02)
            return "v"

        await run_live_dashboard(
            dash_ctx,
            probes={"p": probe},
            settle=lambda k, v: settled.append(v),
            abandon=lambda k: None,
            render=lambda: _embed("static"),
            tick_secs=0.001,
            deadline_secs=1.0,
        )
        assert settled == ["v"]  # the loop really ran
        dash_ctx.channel.send.return_value.edit.assert_not_awaited()

    async def test_the_baseline_advances_so_a_settled_board_is_not_re_edited(
        self, dash_ctx: MagicMock
    ) -> None:
        """Two probes, only one of which changes what renders. The second must cost
        no edit — true only if the change-check runs AND the baseline advanced to
        what was last sent. Leaving the baseline at the skeleton makes every later
        tick diff against it and re-edit an unchanged board.
        """
        state: dict[str, Any] = {}

        async def shown() -> str:
            await asyncio.sleep(0.02)
            return "shown"

        async def unshown() -> str:
            await asyncio.sleep(0.08)
            return "ignored"

        await run_live_dashboard(
            dash_ctx,
            probes={"a": shown, "b": unshown},
            settle=lambda k, v: state.__setitem__(k, v),
            abandon=lambda k: None,
            render=lambda: _embed(state.get("a", "pending")),  # only "a" is rendered
            tick_secs=0.001,
            deadline_secs=1.0,
        )
        assert state == {"a": "shown", "b": "ignored"}
        assert dash_ctx.channel.send.return_value.edit.await_count == 1

    async def test_a_burst_of_probes_does_not_spend_an_edit_each(
        self, dash_ctx: MagicMock
    ) -> None:
        """Waking per probe is not EDITING per probe. Discord buckets message edits
        per channel — message_id is not in the rate-limit key — so four probes
        answering inside one tick must not spend four edits in the same bucket the
        Now Playing progress bar is using.
        """
        state: dict[str, Any] = {}

        def after(name: str, turns: int) -> Any:
            async def _probe() -> str:
                # Staggered by event-loop TURNS, not by wall clock: the property
                # under test is "several probes answering inside one tick spend one
                # edit", and pinning that to real sleeps makes a loaded runner
                # stretch one past the tick and report a false failure.
                for _ in range(turns):
                    await asyncio.sleep(0)
                return name

            return _probe

        await run_live_dashboard(
            dash_ctx,
            probes={
                n: after(n, t) for n, t in (("a", 1), ("b", 2), ("c", 3), ("d", 4))
            },
            settle=lambda k, v: state.__setitem__(k, v),
            abandon=lambda k: None,
            render=lambda: _embed(",".join(sorted(state))),
            tick_secs=30.0,
            deadline_secs=60.0,
        )
        assert set(state) == {"a", "b", "c", "d"}
        # One flush carrying every result, not one edit per probe.
        assert dash_ctx.channel.send.return_value.edit.await_count == 1

    async def test_a_changed_render_is_edited(self, dash_ctx: MagicMock) -> None:
        gate = asyncio.Event()
        state: dict[str, Any] = {"v": "pending"}

        async def probe() -> str:
            await gate.wait()
            return "resolved"

        task = asyncio.create_task(
            run_live_dashboard(
                dash_ctx,
                probes={"p": probe},
                settle=lambda k, v: state.__setitem__("v", v),
                abandon=lambda k: None,
                render=lambda: _embed(state["v"]),
                tick_secs=0.01,
                deadline_secs=2.0,
            )
        )
        await _until(lambda: dash_ctx.channel.send.await_count == 1)
        gate.set()
        await task
        message = dash_ctx.channel.send.return_value
        assert message.edit.await_args.kwargs["embeds"][0].title == "resolved"

    async def test_the_edit_lands_on_completion_not_on_the_next_tick(
        self, dash_ctx: MagicMock
    ) -> None:
        """tick_secs is a ceiling, not a cadence. Sleeping the full tick makes it
        the board's RESOLUTION: a probe answering at 0.5s would sit invisible until
        1.0s. Here a probe that resolves at ~50ms is rendered at ~50ms, well inside
        a 5s tick."""
        state: dict[str, Any] = {"v": "pending"}

        async def probe() -> str:
            await asyncio.sleep(0.05)
            return "resolved"

        start = asyncio.get_running_loop().time()
        await run_live_dashboard(
            dash_ctx,
            probes={"p": probe},
            settle=lambda k, v: state.__setitem__("v", v),
            abandon=lambda k: None,
            render=lambda: _embed(state["v"]),
            tick_secs=5.0,
            deadline_secs=5.0,
        )
        elapsed = asyncio.get_running_loop().time() - start
        assert state["v"] == "resolved"
        assert elapsed < 1.0, f"waited {elapsed:.2f}s for a 0.05s probe"

    async def test_a_deleted_message_stops_the_loop_rather_than_retrying(
        self, dash_ctx: MagicMock
    ) -> None:
        gate = asyncio.Event()
        state: dict[str, Any] = {"v": "pending"}
        message = dash_ctx.channel.send.return_value
        message.edit = AsyncMock(side_effect=discord.NotFound(MagicMock(), "gone"))

        async def probe() -> str:
            await gate.wait()
            return "resolved"

        task = asyncio.create_task(
            run_live_dashboard(
                dash_ctx,
                probes={"p": probe},
                settle=lambda k, v: state.__setitem__("v", v),
                abandon=lambda k: None,
                render=lambda: _embed(state["v"]),
                tick_secs=0.01,
                deadline_secs=2.0,
            )
        )
        await _until(lambda: dash_ctx.channel.send.await_count == 1)
        gate.set()
        assert await task is None
        assert message.edit.await_count == 1  # gave up, did not retry every tick


class TestTheDeadline:
    async def test_a_straggler_is_abandoned_and_cancelled(
        self, dash_ctx: MagicMock
    ) -> None:
        state: dict[str, Any] = {}
        started = asyncio.Event()

        async def never() -> str:
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

        await run_live_dashboard(
            dash_ctx,
            probes={"p": never},
            settle=lambda k, v: state.__setitem__(k, v),
            abandon=lambda k: state.__setitem__(k, "timed out"),
            render=lambda: _embed(state.get("p", "pending")),
            tick_secs=0.01,
            deadline_secs=0.05,
        )
        assert state == {"p": "timed out"}
        message = dash_ctx.channel.send.return_value
        assert message.edit.await_args.kwargs["embeds"][0].title == "timed out"

    async def test_a_probe_that_lands_during_the_final_edit_is_not_abandoned(
        self, dash_ctx: MagicMock
    ) -> None:
        """The re-check. Flipping a probe to FAILED unconditionally at the deadline
        reports a healthy dependency as dead when it finished microseconds earlier.

        Reaching that branch needs a probe still in `pending` when the loop exits AND
        already done — which only happens if it finishes while the loop is awaiting an
        edit. Hence the slow edit: without it the probe settles on the normal drain
        path and this never exercises the re-check at all.
        """
        state: dict[str, Any] = {}
        editing = asyncio.Event()

        async def slow_edit(**_kw: Any) -> None:
            editing.set()
            # Only the "deadline has passed" half is left to the clock, and it is
            # one-sided: a slower runner makes this MORE true, never less.
            await asyncio.sleep(0.2)

        message = dash_ctx.channel.send.return_value
        message.edit = AsyncMock(side_effect=slow_edit)

        async def early() -> str:
            # Real time, and deliberately: it has to settle INSIDE the loop (not in
            # the pre-drain) and outside the coalescing floor, or no edit is spent
            # and the branch under test is never reached. 20x the tick, and a slow
            # runner only widens it — one-sided, unlike the race below.
            await asyncio.sleep(0.02)
            return "early"

        async def during_the_edit() -> str:
            # The interleaving is an event, not a race: this cannot settle until an
            # edit is genuinely in flight, which is the only way to reach the
            # re-check branch at all.
            await editing.wait()
            return "late"

        await run_live_dashboard(
            dash_ctx,
            probes={"a": early, "b": during_the_edit},
            settle=lambda k, v: state.__setitem__(k, v),
            abandon=lambda k: state.__setitem__(k, "timed out"),
            render=lambda: _embed(",".join(v for _, v in sorted(state.items()))),
            tick_secs=0.001,
            deadline_secs=0.1,
        )
        assert state == {"a": "early", "b": "late"}

    async def test_the_deadline_caps_a_wait_longer_than_the_tick(
        self, dash_ctx: MagicMock
    ) -> None:
        """A tick larger than the time remaining must not outlive the deadline."""

        async def never() -> str:
            await asyncio.Event().wait()
            return "unreachable"

        loop = asyncio.get_running_loop()
        start = loop.time()
        await run_live_dashboard(
            dash_ctx,
            probes={"p": never},
            settle=lambda k, v: None,
            abandon=lambda k: None,
            render=lambda: _embed("x"),
            tick_secs=5.0,
            deadline_secs=0.05,
        )
        assert loop.time() - start < 1.0


class TestNoTaskIsEverLeaked:
    async def test_probes_are_cancelled_when_the_render_raises(
        self, dash_ctx: MagicMock
    ) -> None:
        """Launched inside the try so `finally` reaches them wherever a later await
        raises — otherwise a broken renderer leaves probes running forever."""
        holder: dict[str, asyncio.Task[Any]] = {}

        async def never() -> str:
            asyncio.current_task().set_name("probe")  # pyright: ignore[reportOptionalMemberAccess]
            await asyncio.Event().wait()
            return "unreachable"

        def exploding_render() -> list[discord.Embed]:
            holder.update(
                {
                    t.get_name(): t
                    for t in asyncio.all_tasks()
                    if t.get_name() == "probe"
                }
            )
            raise RuntimeError("render is broken")

        with pytest.raises(RuntimeError):
            await run_live_dashboard(
                dash_ctx,
                probes={"p": never},
                settle=lambda k, v: None,
                abandon=lambda k: None,
                render=exploding_render,
                tick_secs=0.01,
                deadline_secs=1.0,
            )
        # One turn is enough to deliver the cancellation, but assert the outcome
        # rather than merely done(): `cancelled() or done()` is just done(), which a
        # probe that RETURNED would satisfy too.
        await asyncio.sleep(0)
        assert holder, "probe task was never observed"
        for task in holder.values():
            await asyncio.wait([task], timeout=1.0)
            assert task.cancelled()

    async def test_the_send_failing_still_cancels_the_probes(
        self, dash_ctx: MagicMock
    ) -> None:
        dash_ctx.channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=500), "boom")
        )
        seen: list[asyncio.Task[Any]] = []

        async def never() -> str:
            asyncio.current_task().set_name("probe")  # pyright: ignore[reportOptionalMemberAccess]
            await asyncio.Event().wait()
            return "unreachable"

        def render() -> list[discord.Embed]:
            seen.extend(t for t in asyncio.all_tasks() if t.get_name() == "probe")
            return _embed("x")

        with pytest.raises(discord.HTTPException):
            await run_live_dashboard(
                dash_ctx,
                probes={"p": never},
                settle=lambda k, v: None,
                abandon=lambda k: None,
                render=render,
                tick_secs=0.01,
                deadline_secs=1.0,
            )
        assert seen, "probe task was never observed"
        for task in seen:
            await asyncio.wait([task], timeout=1.0)
            assert task.cancelled()


async def _run_and_get_message(
    ctx: MagicMock,
    probe: Any,
    *,
    render: Any,
    tick_secs: float = 0.01,
    deadline_secs: float = 1.0,
) -> MagicMock:
    await run_live_dashboard(
        ctx,
        probes={"p": probe},
        settle=lambda k, v: None,
        abandon=lambda k: None,
        render=render,
        tick_secs=tick_secs,
        deadline_secs=deadline_secs,
    )
    return ctx.channel.send.return_value


class TestNothingToWaitFor:
    async def test_no_probes_sends_once_and_never_edits(
        self, dash_ctx: MagicMock
    ) -> None:
        """-debug's non-operator path: nothing deferred, so the driver degrades to
        a plain send with no loop and no deadline wait."""
        returned: Optional[discord.Message] = await run_live_dashboard(
            dash_ctx,
            probes={},
            settle=lambda k, v: None,
            abandon=lambda k: None,
            render=lambda: _embed("complete"),
            tick_secs=0.01,
            deadline_secs=5.0,
        )
        assert returned is dash_ctx.channel.send.return_value
        dash_ctx.channel.send.assert_awaited_once()
        dash_ctx.channel.send.return_value.edit.assert_not_awaited()


class TestLateAndPartialFailures:
    """Two holes in the never-leak guarantee that only open on paths the two
    current callers do not take — reported because this is a shared driver whose
    docstring sells that guarantee as the reason it exists."""

    async def test_a_probe_that_raises_while_unwinding_is_still_retrieved(
        self, dash_ctx: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A probe cancelled at the deadline can raise from its own cleanup — an
        aiohttp or asyncpg `__aexit__` — and it settles AFTER the driver returned.
        `finally` cannot see that one, so asyncio logged "Task exception was never
        retrieved" against a command that handled the failure fine."""
        started = asyncio.Event()

        async def raises_during_cleanup() -> str:
            try:
                started.set()
                await asyncio.Event().wait()
                return "unreachable"
            except asyncio.CancelledError:
                # Cleanup that itself fails: the task settles with THIS, not with
                # CancelledError, so it is a task holding an unretrieved exception.
                raise ValueError("cleanup blew up during teardown") from None

        errors: list[str] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: errors.append(str(context.get("message", "")))
        )

        await run_live_dashboard(
            dash_ctx,
            probes={"p": raises_during_cleanup},
            settle=lambda k, v: None,
            abandon=lambda k: None,
            render=lambda: _embed("done"),
            tick_secs=0.01,
            deadline_secs=0.05,
        )
        assert started.is_set()
        # Let the task settle and the loop run any handler it would have queued.
        for _ in range(5):
            await asyncio.sleep(0)
        assert not any("never retrieved" in e for e in errors), errors

    async def test_a_probe_factory_that_raises_does_not_orphan_its_siblings(
        self, dash_ctx: MagicMock
    ) -> None:
        """`tasks` used to be bound only once the whole comprehension succeeded, so
        a raising fn() left every task already created invisible to `finally`.

        Asserted on task IDENTITY rather than through the probe body: create_task
        only schedules, and the factory raises before anything gets a turn, so the
        orphan never runs a line of its own code — which is exactly why it is easy
        to leak and hard to notice.
        """

        async def never() -> str:
            await asyncio.Event().wait()
            return "unreachable"

        def explodes() -> Coroutine[Any, Any, str]:
            raise KeyError("probe factory exploded")

        before = asyncio.all_tasks()
        with pytest.raises(KeyError):
            await run_live_dashboard(
                dash_ctx,
                probes={"first": never, "second": explodes},
                settle=lambda k, v: None,
                abandon=lambda k: None,
                render=lambda: _embed("unused"),
                tick_secs=0.01,
                deadline_secs=1.0,
            )

        # all_tasks() only reports tasks that are not yet done, so a few turns to
        # deliver the cancellation is the difference between "stopped" and "leaked".
        for _ in range(5):
            await asyncio.sleep(0)
        assert not (asyncio.all_tasks() - before)
