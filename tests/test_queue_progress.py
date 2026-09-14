"""The live card a slow collection enqueue shows (src/queue_progress.py).

The renderer is pure and asserted as goldens; the driver owns a delay, a send, N
edits and a delete, and is asserted through a ctx double.

The delay, tick and ceiling are bot settings each card reads once on entry, so a
test that wants a fast card sets them with `config.set_override` before entering it.
"""

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import queue_progress, util
from src.dashboard import LiveMessage
from src.queue_progress import (
    _CARD_CLAIM,
    _ELAPSED_STEP_SECS,
    CardDetails,
    Source,
    EnqueuePhase,
    EnqueueProgress,
    card_ceiling,
    enqueue_progress,
    is_collection,
    render_progress_card,
)
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_url,
)
from src.util import BAR_WIDTH, progress_line
from src import config
from src.settings import SETTINGS, SettingScope
from tests.helpers import settle

_DONE, _REMAINING = "\U0001f7e6", "⬜"

# The drivers loop on set_within, which does not suspend once its event is set, so
# a regression that drops a `return` busy-spins, and asyncio.timeout cannot
# interrupt a spin. Ten seconds rather than the suite's 120.
pytestmark = pytest.mark.timeout(10)


@pytest.fixture
def card_ctx(mock_ctx: MagicMock) -> MagicMock:
    """A ctx whose channel.send returns an editable, deletable message double.

    tests/commands/test_play.py has never touched channel.send, and an
    auto-vivified return is a plain MagicMock whose `await message.edit(...)`
    raises TypeError. Mirrors tests/test_dashboard.py's dash_ctx.
    """
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    message.delete = AsyncMock()
    mock_ctx.channel.id = 4242
    mock_ctx.channel.send = AsyncMock(return_value=message)
    return mock_ctx


def _yt_playlist() -> Source:
    return parse_url("https://www.youtube.com/playlist?list=PLtest")


def _details(**kwargs: Any) -> CardDetails:
    return CardDetails(requester="<@1>", **kwargs)


# The delay a fast card is entered with; _fast() makes its tick as short.
_FAST_DELAY = 0.01


def _fast() -> None:
    config.set_override("QUEUE_PROGRESS_TICK_SECS", 0.01)


def _text(embeds: list[discord.Embed]) -> str:
    return embeds[0].description or ""


# ── the renderer ──────────────────────────────────────────────────────────────


class TestRenderProgressCard:
    def test_without_a_total_there_is_no_bar_only_an_elapsed_line(self) -> None:
        """Indeterminate is the STEADY state for a YouTube Mix — YouTube marks it
        infinite, so no total exists until the walk ends — and the card says how
        long it has been rather than faking a bar."""
        body = _text(
            render_progress_card(
                EnqueueProgress(done=40), elapsed_secs=37.0, details=_details()
            )
        )
        assert _DONE not in body and _REMAINING not in body
        # Both halves: `done` is real here (a playlist_index, a Spotify item
        # count) and is the only thing on this card that moves with the WORK
        # rather than the clock.
        assert "`40` checked" in body
        assert "`0:35` elapsed" in body

    def test_the_elapsed_line_moves_in_five_second_steps(self) -> None:
        """Per second is 0.5 edits/s against a 1.0/s per-channel budget the Now
        Playing bar already spends a third of. Asserted as the count of distinct
        renders over a span, because that count IS the edit budget: the value at
        any one instant is not what costs anything."""
        span_secs = 60
        renders = {
            _text(
                render_progress_card(
                    EnqueueProgress(), elapsed_secs=e / 10, details=_details()
                )
            )
            for e in range(0, span_secs * 10)
        }
        assert len(renders) == span_secs // _ELAPSED_STEP_SECS + 1

    def test_the_first_render_does_not_claim_no_time_has_passed(self) -> None:
        """The card is its delay old the first time it renders.
        Flooring made a message that took 2.5s to appear open with `0:00`, which
        reads as a card that is not working."""
        body = _text(
            render_progress_card(
                EnqueueProgress(),
                elapsed_secs=config.queue_progress_delay_secs(),
                details=_details(),
            )
        )
        assert "`0:00`" not in body

    def test_a_total_of_zero_renders_rather_than_dividing(self) -> None:
        """queue_source raises for an empty collection, but a Spotify page-1
        report of (0, 0) reaches the renderer first, and it must not depend on a
        caller's invariant."""
        body = _text(
            render_progress_card(
                EnqueueProgress(total=0), elapsed_secs=1.0, details=_details()
            )
        )
        assert "`0:00` elapsed" in body

    def test_the_bar_is_the_now_playing_bar_labelled_in_songs(self) -> None:
        """One bar in the product: the card renders through the same progress_line
        the Now Playing card does, with counts where the clock times go."""
        body = _text(
            render_progress_card(
                EnqueueProgress(done=400, total=1671),
                elapsed_secs=1.0,
                details=_details(),
            )
        )
        assert f"`335` {_DONE * 2}🔘{_REMAINING * 7} `1671` songs" in body
        assert progress_line(335, 1671, label=lambda n: str(int(n))) in body

    def test_a_one_song_total_is_singular(self) -> None:
        body = _text(
            render_progress_card(
                EnqueueProgress(done=0, total=1), elapsed_secs=1.0, details=_details()
            )
        )
        assert body.endswith("`1` song")

    @pytest.mark.parametrize("total", [1, 3, 9, 10, 11, 1671])
    def test_the_bar_is_always_the_shared_width(self, total: int) -> None:
        """A 4-song collection still draws BAR_WIDTH cells, as the Now Playing bar
        does for a 4-second song; the count label carries the exact number."""
        body = _text(
            render_progress_card(
                EnqueueProgress(done=total // 2, total=total),
                elapsed_secs=1.0,
                details=_details(),
            )
        )
        assert body.count(_DONE) + body.count("🔘") + body.count(_REMAINING) == (
            BAR_WIDTH
        )

    def test_a_finished_walk_puts_the_head_at_the_end(self) -> None:
        body = _text(
            render_progress_card(
                EnqueueProgress(done=1671, total=1671),
                elapsed_secs=1.0,
                details=_details(),
            )
        )
        assert f"`1671` {_DONE * (BAR_WIDTH - 1)}🔘 `1671` songs" in body

    def test_the_count_is_the_bar_s_own_cell_boundary(self) -> None:
        """The number and the picture beside it must agree: 400/1671 next to two
        filled cells of ten reads as a rendering bug. 335 is the smallest count
        that fills two."""
        body = _text(
            render_progress_card(
                EnqueueProgress(done=400, total=1671),
                elapsed_secs=1.0,
                details=_details(),
            )
        )
        assert body.count(_DONE) == 2
        assert "`335`" in body

    def test_a_render_only_moves_when_a_cell_does(self) -> None:
        """What makes embeds_changed do real work: a Spotify page landing every
        300ms moves `done` six times a tick, and none of them may spend an edit."""
        renders = {
            _text(
                render_progress_card(
                    EnqueueProgress(done=done, total=1671),
                    elapsed_secs=1.0,
                    details=_details(),
                )
            )
            for done in range(335, 501)
        }
        assert len(renders) == 1

    def test_a_replayed_count_past_the_total_does_not_overflow(self) -> None:
        """A BrokenProcessPool heal re-runs the extraction from entry 0, so a
        parent taking max() can be handed a count above its own total."""
        body = _text(
            render_progress_card(
                EnqueueProgress(done=5000, total=1671),
                elapsed_secs=1.0,
                details=_details(),
            )
        )
        assert f"`1671` {_DONE * (BAR_WIDTH - 1)}🔘 `1671` songs" in body

    def test_the_number_and_the_picture_never_disagree(self) -> None:
        """The count is integer maths and the bar is a float ratio, so they can
        floor to different cells and show `7` beside six filled boxes. Driven
        through the renderer over every `done` for a spread of totals, because the
        arithmetic identity on its own holds whatever the renderer does with it."""
        for total in (1, 2, 3, 7, 10, 11, 99, 100, 183, 1671, 10_000):
            dones = range(total + 1) if total <= 200 else range(0, total + 1, 7)
            for done in dones:
                body = _text(
                    render_progress_card(
                        EnqueueProgress(done=done, total=total),
                        elapsed_secs=1.0,
                        details=_details(),
                    )
                )
                shown = int(body.split("`")[1])
                cells = shown * BAR_WIDTH // total
                # The bar draws the cells its own count fills (the head takes the
                # last one), the count is the smallest that fills them, and it
                # never claims more than actually happened.
                assert body.count(_DONE) == min(cells, BAR_WIDTH - 1), (total, done)
                assert shown == 0 or (shown - 1) * BAR_WIDTH // total < cells
                assert shown <= done, (total, done, body)

    def test_it_shows_what_exists_before_the_resolve_returns(self) -> None:
        body = _text(
            render_progress_card(
                EnqueueProgress(),
                elapsed_secs=1.0,
                details=CardDetails(
                    requester="<@7>",
                    playlist_label="Playlist `PL1`",
                    warning="⚠️ bad timestamp",
                    placement_note="Plays next once it's queued.",
                ),
            )
        )
        assert "<@7>" in body
        assert "Playlist `PL1`" in body
        assert "Plays next once it's queued." in body
        assert "⚠️ bad timestamp" in body

    def test_the_debug_footer_is_threaded_through(self) -> None:
        """channel.send bypasses MusicContext.send, and with it debug-mode
        decoration, so the caller pre-renders the suffix once."""
        embeds = render_progress_card(
            EnqueueProgress(),
            elapsed_secs=1.0,
            details=_details(debug_suffix="shard: 0"),
        )
        assert embeds[0].footer.text == "shard: 0"

    def test_past_the_ceiling_the_title_says_so(self) -> None:
        titles = [
            render_progress_card(
                EnqueueProgress(phase=phase), elapsed_secs=1.0, details=_details()
            )[0].title
            for phase in EnqueuePhase
        ]
        assert titles[0] != titles[1]
        assert all("playlist" in (t or "") for t in titles)


class TestEnqueueProgressState:
    def test_done_is_absolute_so_a_dropped_report_self_corrects(self) -> None:
        progress = EnqueueProgress()
        progress.update(100, 250)
        progress.update(300, None)  # the 200 report never arrived
        assert (progress.done, progress.total) == (300, 250)

    def test_a_lower_report_never_moves_the_count_back(self) -> None:
        """A superseded extraction or a heal's retry reports from its own start,
        below where the bar already is."""
        progress = EnqueueProgress()
        progress.update(300, 400)
        progress.update(25, None)
        assert (progress.done, progress.total) == (300, 400)

    def test_a_replay_from_zero_is_idempotent(self) -> None:
        """A BrokenProcessPool heal re-runs the extraction from entry 0; with
        deltas the count would finish at 2N."""
        progress = EnqueueProgress()
        for done in (1, 2, 3, 1, 2, 3):
            progress.update(done, None)
        assert progress.done == 3


class TestIsCollection:
    def test_only_playlists_get_a_card(self) -> None:
        assert is_collection(_yt_playlist())
        assert is_collection(SpotifySource(SpotifyType.PLAYLIST, "pid"))
        assert not is_collection(SpotifySource(SpotifyType.TRACK, "tid"))
        assert not is_collection(YTSource("https://yt.com/v=1", type=YTType.TRACK))
        assert not is_collection(SoundcloudSource("https://soundcloud.com/a/b"))


# ── the driver ────────────────────────────────────────────────────────────────


class TestTheDelayThreshold:
    async def test_a_fast_enqueue_sends_nothing(self, card_ctx: MagicMock) -> None:
        """A cache-hit playlist is one Redis GET; the channel must see exactly
        what it sees today."""

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=30.0):
            pass

        card_ctx.channel.send.assert_not_called()

    async def test_a_slow_enqueue_sends_exactly_one_card(
        self, card_ctx: MagicMock
    ) -> None:
        """Without this, setting the delay to infinity — the card never appears
        for anyone — would make the suite greener."""
        _fast()

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            await asyncio.sleep(0.08)

        card_ctx.channel.send.assert_awaited_once()

    async def test_the_card_never_goes_through_ctx_send(
        self, card_ctx: MagicMock
    ) -> None:
        """MusicContext.send would adopt it as the Now Playing host, and the
        progress updater would then rewrite it every 3s."""
        _fast()

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            await asyncio.sleep(0.08)

        card_ctx.send.assert_not_called()


class TestTheCardMoves:
    async def test_an_indeterminate_card_still_edits(self, card_ctx: MagicMock) -> None:
        """The test that dies if the elapsed line is made static — which would
        leave a card `embeds_changed` never edits, i.e. slow_resolve_notice with
        extra machinery."""
        _fast()

        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=_FAST_DELAY
        ) as progress:
            await asyncio.sleep(0.05)
            # Ten seconds of resolve without waiting them out: the elapsed line
            # is what has to move, not the wall clock.
            progress.started_at -= 10.0
            await asyncio.sleep(0.05)

        assert card_ctx.channel.send.return_value.edit.await_count >= 1

    async def test_a_determinate_card_edits_when_a_cell_moves(
        self, card_ctx: MagicMock
    ) -> None:
        _fast()

        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=_FAST_DELAY
        ) as progress:
            progress.update(0, 100)
            await asyncio.sleep(0.05)
            before = card_ctx.channel.send.return_value.edit.await_count
            progress.update(50, None)
            await asyncio.sleep(0.05)

        assert card_ctx.channel.send.return_value.edit.await_count > before


class TestTeardown:
    async def test_the_card_is_deleted_when_the_enqueue_lands(
        self, card_ctx: MagicMock
    ) -> None:
        """Every exit path deletes it: the confirmation still goes out through
        ctx.send and still re-hosts the Now Playing block."""
        _fast()
        message = card_ctx.channel.send.return_value

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            await asyncio.sleep(0.05)

        message.delete.assert_awaited_once()

    async def test_nothing_is_edited_after_the_delete(
        self, card_ctx: MagicMock
    ) -> None:
        _fast()
        message = card_ctx.channel.send.return_value

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            await asyncio.sleep(0.05)
        after_exit = message.edit.await_count
        await asyncio.sleep(0.05)

        assert message.edit.await_count == after_exit

    async def test_a_card_the_user_deletes_stops_the_loop(
        self, card_ctx: MagicMock
    ) -> None:
        """Otherwise the driver spends the rest of a 99-second enqueue editing into
        404s, in the bucket the Now Playing bar shares."""
        _fast()
        message = card_ctx.channel.send.return_value
        message.edit = AsyncMock(side_effect=discord.NotFound(MagicMock(), "gone"))

        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=_FAST_DELAY
        ) as progress:
            await asyncio.sleep(0.05)
            progress.started_at -= 10.0
            await asyncio.sleep(0.1)

        assert message.edit.await_count == 1
        # Nothing to take back, so the teardown does not 404 on the delete either.
        message.delete.assert_not_awaited()

    async def test_a_raise_inside_the_block_still_takes_the_card_back(
        self, card_ctx: MagicMock
    ) -> None:
        _fast()
        message = card_ctx.channel.send.return_value

        with pytest.raises(RuntimeError):
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(0.05)
                raise RuntimeError("extraction failed")

        message.delete.assert_awaited_once()

    async def test_a_404_on_the_delete_does_not_replace_the_real_error(
        self, card_ctx: MagicMock
    ) -> None:
        """The delete runs inside AsyncExitStack.__aexit__ while an
        ExtractionError may be propagating, and the user's own error embed must
        not become a 404 about a card they deleted themselves."""
        _fast()
        message = card_ctx.channel.send.return_value
        message.delete = AsyncMock(side_effect=discord.NotFound(MagicMock(), "gone"))

        with pytest.raises(RuntimeError, match="extraction failed"):
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(0.05)
                raise RuntimeError("extraction failed")

    async def test_a_send_discord_refuses_leaves_the_enqueue_alone(
        self, card_ctx: MagicMock
    ) -> None:
        """Missing Send Messages, or a 429 the client gave up on: the card is
        advisory and must never take the command down."""
        _fast()
        message = card_ctx.channel.send.return_value
        card_ctx.channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "forbidden")
        )

        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=_FAST_DELAY
        ) as progress:
            await asyncio.sleep(0.05)
            progress.update(1, 2)

        # The body ran to completion, nothing was edited or deleted, and the
        # channel claim went back — a refused send must leave no trace at all.
        assert progress.done == 1
        message.edit.assert_not_awaited()
        message.delete.assert_not_awaited()
        assert not util._CLAIMED_CHANNELS.get(_CARD_CLAIM)

    async def test_a_cancel_during_the_send_still_takes_the_card_back(
        self, card_ctx: MagicMock
    ) -> None:
        """The driver is stopped by a signal and AWAITED, never cancelled: there
        is a window where the message exists on Discord and the local handle does
        not, and a cancel inside it would leave the card standing forever."""
        _fast()
        message = card_ctx.channel.send.return_value
        sending, release = asyncio.Event(), asyncio.Event()

        async def _send(**_: Any) -> MagicMock:
            sending.set()
            await release.wait()
            return message

        card_ctx.channel.send = AsyncMock(side_effect=_send)

        async def _body() -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(3600)

        task = asyncio.create_task(_body())
        async with asyncio.timeout(2):
            await sending.wait()
        task.cancel()
        await settle()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        message.delete.assert_awaited_once()

    async def test_a_cancel_while_joining_the_driver_still_takes_the_card_back(
        self, card_ctx: MagicMock
    ) -> None:
        """The cancel that matters lands INSIDE the teardown, at the join —
        shutdown cancels a command already unwinding, and join_task re-raises
        that rather than swallowing it. The driver deletes what it sent in its
        own finally, so the card still comes back from a join that never
        returns."""
        _fast()
        message = card_ctx.channel.send.return_value
        sending, release = asyncio.Event(), asyncio.Event()

        async def _send(**_: Any) -> MagicMock:
            sending.set()
            await release.wait()
            return message

        card_ctx.channel.send = AsyncMock(side_effect=_send)

        async def _body() -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                async with asyncio.timeout(2):
                    await sending.wait()

        task = asyncio.create_task(_body())
        async with asyncio.timeout(2):
            await sending.wait()
        # The body has returned; the exit stack is parked on the join.
        await settle()
        task.cancel()
        await settle()
        release.set()
        await settle()

        with pytest.raises(asyncio.CancelledError):
            await task
        message.delete.assert_awaited_once()


class TestTheCardsOwnBounds:
    @pytest.mark.parametrize(
        ("max_secs", "expected"),
        [(1.0, 20.0), (15.0, 20.0), (20.0, 20.0), (300.0, 300.0)],
        ids=["under-the-delay", "under-two-ticks", "at-the-rule", "above"],
    )
    def test_the_ceiling_is_the_delay_and_two_ticks_at_least(
        self, max_secs: float, expected: float
    ) -> None:
        assert card_ceiling(10.0, 5.0, max_secs) == expected

    async def test_a_ceiling_the_delay_outlasts_still_gets_an_ordinary_edit(
        self, card_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The delay and the ceiling are set separately, so a ceiling can end
        before the card appears. The card still makes one ordinary edit first:
        a card that only ever says "still working" is the feature doing nothing.

        The card's clock is held past the max and short of delay + 2 ticks, so
        only the ceiling decides whether its first check stalls, however late
        that check runs."""
        clock = [1000.0]
        monkeypatch.setattr(
            queue_progress, "time", SimpleNamespace(monotonic=lambda: clock[0])
        )
        _fast()
        config.set_override("QUEUE_PROGRESS_MAX_SECS", 0.001)
        message = card_ctx.channel.send.return_value
        titles: list[str] = []

        async def _edit(**kwargs: Any) -> None:
            titles.append(kwargs["embeds"][0].title or "")

        message.edit = AsyncMock(side_effect=_edit)

        # Ceiling: max(0.001, 0.01 + 2 * 0.01) = 0.03 past the card's start.
        async with enqueue_progress(card_ctx, _yt_playlist(), delay=0.01) as progress:
            clock[0] = 1000.02
            async with asyncio.timeout(2):
                while not titles:
                    progress.update(progress.done + 1, 10)
                    await asyncio.sleep(0.005)
            clock[0] = 1000.05
            async with asyncio.timeout(2):
                while not any("Still working" in t for t in titles):
                    await asyncio.sleep(0.005)

        assert titles[0] == queue_progress._TITLE

    async def test_a_teardown_does_not_wait_on_a_wedged_driver(
        self, card_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A -play unwinds through this join, so an unbounded wait on a Discord
        round trip is the command hanging. Past the bound the retraction is left
        to the driver's own finally."""
        _fast()
        monkeypatch.setattr(queue_progress, "QUEUE_PROGRESS_JOIN_SECS", 0.05)
        wedged = asyncio.Event()

        async def _never(**_: Any) -> MagicMock:
            await wedged.wait()
            return card_ctx.channel.send.return_value

        card_ctx.channel.send = AsyncMock(side_effect=_never)

        async with asyncio.timeout(2):
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(0.05)
        wedged.set()
        await settle()

    @pytest.mark.parametrize(
        "url,label",
        [
            (
                "https://www.youtube.com/playlist?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG",
                "Playlist `PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG`",
            ),
            (
                "https://www.youtube.com/watch?v=IhCDK_pSjnk&list=RDIhCDK_pSjnk&start_radio=1",
                "YouTube Mix",
            ),
            (
                "https://www.youtube.com/playlist?list=RDCLAK5uy_kmPRjHDECIcuVwnKsx2Ng",
                "Playlist `RDCLAK5uy_kmPRjHDECIcuVwnKsx2Ng`",
            ),
        ],
    )
    def test_the_card_names_the_list_whole_and_unescaped(
        self, url: str, label: str
    ) -> None:
        """Built from the id: a canonical playlist link is 72 characters, past the
        row, and a Mix link carries underscores that render as `\\_`."""
        assert queue_progress._playlist_label(parse_url(url)) == label

    @pytest.mark.parametrize(
        "list_id", ["x](https://evil.example)", "a`b", "", "x" * 65]
    )
    def test_a_list_id_that_is_not_a_youtube_id_is_not_rendered(
        self, list_id: str
    ) -> None:
        """The id comes from the pasted link unvalidated, and an embed description
        renders [text](url) as a masked link."""
        source = YTSource(
            url=f"https://www.youtube.com/playlist?list={list_id}",
            type=YTType.PLAYLIST,
            list_id=list_id,
        )
        assert queue_progress._playlist_label(source) == ""


class TestOneCardPerChannel:
    async def test_two_concurrent_enqueues_share_one_edit_loop(
        self, card_ctx: MagicMock
    ) -> None:
        """PLAY_INFLIGHT_MAX is 16 and requests resolve concurrently, so a pasted
        burst would otherwise be sixteen edit loops on one channel's bucket —
        eight times Discord's budget. PLAY_RESOLVE_CONCURRENCY cannot throttle
        them: it is taken inside the extraction, below where the card is entered.
        """
        _fast()
        second_settled = asyncio.Event()

        # The first card stays up until the second request settles. A loser re-asks
        # every tick, so a card that came down first would rightly hand it the claim.
        async def _first() -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await second_settled.wait()

        async def _second() -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(0.08)
            second_settled.set()

        async with asyncio.timeout(2):
            await asyncio.gather(_first(), _second())

        card_ctx.channel.send.assert_awaited_once()

    async def test_a_fast_enqueue_does_not_hold_the_slot(
        self, card_ctx: MagicMock
    ) -> None:
        """The claim is taken when the card is about to be sent, not at entry: a
        cache hit that finished inside the delay must not deny the slot to the
        99-second sibling beside it."""
        _fast()

        async def _cache_hit() -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                pass

        async def _slow() -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(0.08)

        await asyncio.gather(_cache_hit(), _slow())

        card_ctx.channel.send.assert_awaited_once()


class TestAClaimLostToAnotherCard:
    async def test_the_card_appears_once_the_other_is_taken_back(
        self, card_ctx: MagicMock
    ) -> None:
        """One card per channel, not one card per burst: a 5,547-track playlist
        pasted a second after a Mix otherwise runs ~70s with nothing on screen
        once the Mix's card goes."""
        _fast()
        shown: list[int] = []

        async def _one(secs: float) -> None:
            async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
                await asyncio.sleep(secs)
                shown.append(card_ctx.channel.send.await_count)

        await asyncio.gather(_one(0.05), _one(0.3))

        # The first card was up alone, and the second came after it went.
        assert shown == [1, 2]
        assert card_ctx.channel.send.return_value.delete.await_count == 2


def _recorded() -> tuple[InMemorySpanExporter, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


class TestCardTelemetry:
    """Recorded on the command's span from the driver task, which inherits it: the
    four card knobs are otherwise tuned blind."""

    @pytest.mark.parametrize(
        "outcome", ["shown", "stalled", "send_failed", "claimed_elsewhere"]
    )
    async def test_the_span_says_what_the_card_did(
        self,
        card_ctx: MagicMock,
        caplog: pytest.LogCaptureFixture,
        outcome: str,
    ) -> None:
        _fast()
        if outcome == "stalled":
            config.set_override("QUEUE_PROGRESS_MAX_SECS", 0.03)
        if outcome == "send_failed":
            card_ctx.channel.send.side_effect = discord.HTTPException(
                MagicMock(status=403), "Missing Permissions"
            )
        exporter, tracer = _recorded()

        with caplog.at_level(logging.WARNING), contextlib.ExitStack() as held:
            if outcome == "claimed_elsewhere":
                held.enter_context(util.channel_claim(_CARD_CLAIM, card_ctx.channel.id))
            with tracer.start_as_current_span("bot.play"):
                async with enqueue_progress(
                    card_ctx, _yt_playlist(), delay=_FAST_DELAY
                ):
                    await asyncio.sleep(0.1)

        (span,) = exporter.get_finished_spans()
        assert (span.attributes or {})["play.progress_card"] == outcome
        if outcome in ("stalled", "send_failed"):
            assert "queue progress card" in caplog.text


class TestTheCeiling:
    async def test_past_the_ceiling_it_says_so_once_and_stops_editing(
        self, card_ctx: MagicMock
    ) -> None:
        """Nothing else bounds the work: PLAY_RESOLVE_WAIT_SECS bounds the wait
        for a slot and not the extraction, and yt-dlp's own socket_timeout x
        retries lets one page hold 300s with no aggregate bound across 56 of them.
        """
        _fast()
        config.set_override("QUEUE_PROGRESS_MAX_SECS", 0.03)
        message = card_ctx.channel.send.return_value

        def _stalled() -> bool:
            call = message.edit.await_args
            return call is not None and "Still working" in (
                call.kwargs["embeds"][0].title or ""
            )

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            # Waited for rather than slept past: a slow loop would otherwise land
            # the STALLED edit after the count below is read.
            async with asyncio.timeout(2):
                while not _stalled():
                    await asyncio.sleep(0.01)
            settled = message.edit.await_count
            await asyncio.sleep(0.08)
            assert message.edit.await_count == settled
            # Stalled is not settled: the card stays on screen, and the claim
            # with it, so a sibling cannot post a second card beside it.
            message.delete.assert_not_awaited()
            assert util._CLAIMED_CHANNELS.get(_CARD_CLAIM)

        last: Optional[list[discord.Embed]] = message.edit.await_args.kwargs["embeds"]
        assert last is not None
        assert "Still working" in (last[0].title or "")
        message.delete.assert_awaited_once()
        assert not util._CLAIMED_CHANNELS.get(_CARD_CLAIM)

    async def test_a_stalled_card_the_user_deleted_releases_the_claim(
        self, card_ctx: MagicMock
    ) -> None:
        _fast()
        config.set_override("QUEUE_PROGRESS_MAX_SECS", 0.03)
        message = card_ctx.channel.send.return_value
        message.edit.side_effect = discord.NotFound(MagicMock(status=404), "gone")

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            await asyncio.sleep(0.12)
            assert not util._CLAIMED_CHANNELS.get(_CARD_CLAIM)
        message.delete.assert_not_awaited()


class TestAFailedEdit:
    async def test_a_transient_edit_failure_still_takes_the_card_back(
        self, card_ctx: MagicMock
    ) -> None:
        """Only a 404 means the user deleted the card. Anything else read as gone
        drops the handle, the finally deletes nothing, and "Working on that
        playlist…" stays in the channel permanently."""
        _fast()
        message = card_ctx.channel.send.return_value
        message.id = 5150
        message.edit.side_effect = [
            discord.HTTPException(MagicMock(status=503), "unavailable"),
            None,
            None,
        ]

        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=_FAST_DELAY
        ) as progress:
            await asyncio.sleep(0.03)
            progress.update(5, 10)
            await asyncio.sleep(0.05)

        assert message.edit.await_count >= 1
        message.delete.assert_awaited_once()


class TestADroppedRequest:
    """-stop, -clear and -remove stamp a request mid-resolve, and place() reads the
    stamp only once the resolve returns — ~90 s later for a 5,547-track playlist."""

    async def test_the_card_is_taken_back_before_the_resolve_returns(
        self, card_ctx: MagicMock
    ) -> None:
        _fast()
        message = card_ctx.channel.send.return_value
        dropped = asyncio.Event()

        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=_FAST_DELAY, dropped=dropped
        ):
            await asyncio.sleep(0.05)
            card_ctx.channel.send.assert_awaited_once()
            dropped.set()
            await asyncio.sleep(0.05)
            message.delete.assert_awaited_once()
            assert not util._CLAIMED_CHANNELS.get(_CARD_CLAIM)
        message.delete.assert_awaited_once()

    async def test_a_request_dropped_inside_the_delay_sends_nothing(
        self, card_ctx: MagicMock
    ) -> None:
        _fast()
        dropped = asyncio.Event()
        async with enqueue_progress(
            card_ctx, _yt_playlist(), delay=0.05, dropped=dropped
        ):
            dropped.set()
            await asyncio.sleep(0.1)
        card_ctx.channel.send.assert_not_awaited()


class TestTheStalledRender:
    async def test_it_is_not_lost_to_the_edit_floor(
        self, card_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The STALLED render is the card's last edit, so it goes through finish(),
        which ignores the floor between edits. Through tick() it is suppressed
        whenever the previous edit was recent, and the card never says it stalled.
        """
        _fast()
        config.set_override("QUEUE_PROGRESS_MAX_SECS", 0.03)
        monkeypatch.setattr(
            queue_progress, "LiveMessage", lambda _tick: LiveMessage(3600.0)
        )
        message = card_ctx.channel.send.return_value

        async with enqueue_progress(card_ctx, _yt_playlist(), delay=_FAST_DELAY):
            await asyncio.sleep(0.1)

        message.edit.assert_awaited_once()
        assert "Still working" in (
            message.edit.await_args.kwargs["embeds"][0].title or ""
        )


class TestTheEditBudget:
    def test_the_card_and_the_now_playing_bar_fit_one_channel(self) -> None:
        """Discord allows 5 edits / 5s per CHANNEL and every PATCH in a channel
        shares one bucket. A 429 never reaches safe_edit — discord.py sleeps it
        internally — so the symptom is the NP bar silently freezing, invisible in
        our logs."""
        card = 1.0 / config.queue_progress_tick_secs()
        now_playing = 1.0 / config.now_playing_update_interval_secs()
        assert card + now_playing < 1.0

    def test_the_fastest_cadences_chat_allows_still_fit(self) -> None:
        """The bot settings' chat minimums, both at once."""
        tick, bar = (
            next(s for s in SETTINGS if s.scope is SettingScope.BOT and s.key == key)
            for key in ("queue-progress-tick", "np-refresh")
        )
        assert isinstance(tick.minimum, float) and isinstance(bar.minimum, float)
        assert 1.0 / tick.minimum + 1.0 / bar.minimum < 1.0

    def test_the_tick_floor_is_higher_than_the_dashboards(self) -> None:
        """-ping and -debug can share a 0.05s floor because their deadlines cap
        the damage at ~8 edits; this card has no such cap."""
        from src.config import _MIN_DASHBOARD_SECS, _MIN_QUEUE_TICK_SECS

        assert _MIN_QUEUE_TICK_SECS > _MIN_DASHBOARD_SECS
