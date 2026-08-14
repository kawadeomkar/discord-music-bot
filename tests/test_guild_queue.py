"""Tests for src/guild_queue.py — the queue domain class.

Central property: the triad-sync invariant — after every operation the asyncio
queue, the display deque and the Redis mirror agree (persisted=False items live
on the in-memory legs only, by design)."""

import redis.asyncio as aioredis
from collections import deque
from dataclasses import replace
from typing import Any
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.guild_queue import (
    _LREM_MAX_ENTRIES,
    GuildQueue,
    RemoveMode,
    ShuffleOutcome,
    is_persisted,
    remove_matcher,
)
from src.guild_state import SearchQueueEntry, SongQueueEntry, parse_queue_entry
from src.redis_client import GuildRedisStore
from src.sources import YTSource
from src.youtube import QueueObject
from tests.helpers import queue_object, seed_queue


@pytest.fixture
def store(fake_redis: aioredis.Redis, mock_guild: MagicMock) -> GuildRedisStore:
    return GuildRedisStore(fake_redis, guild_id=mock_guild.id)


@pytest.fixture
def gq(mock_guild: MagicMock, store: GuildRedisStore) -> GuildQueue:
    return GuildQueue(mock_guild, store)


@pytest.fixture
def gq_no_redis(mock_guild: MagicMock) -> GuildQueue:
    return GuildQueue(mock_guild, None)


def _qobj(n: int, requester: Any, *, persisted: bool = True) -> QueueObject:
    return QueueObject(
        f"https://yt.com/v={n}", f"Song {n}", requester, persisted=persisted
    )


async def _assert_triad_sync(
    gq: GuildQueue, fake_redis: aioredis.Redis, store: GuildRedisStore
) -> None:
    """The invariant: all three legs agree (Redis holds persisted items only)."""
    items = gq.display_items()
    assert gq.qsize() == len(items)
    redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
    persisted = [i for i in items if is_persisted(i)]
    assert len(redis_items) == len(persisted)


# ── is_persisted ──────────────────────────────────────────────────────────────


class TestIsPersisted:
    def test_queue_object_reflects_flag(self, mock_author: MagicMock) -> None:
        assert is_persisted(_qobj(1, mock_author)) is True
        assert is_persisted(_qobj(1, mock_author, persisted=False)) is False

    def test_ytsource_always_persisted(self) -> None:
        assert is_persisted(YTSource(ytsearch="artist song")) is True

    def test_none_is_persisted(self) -> None:
        # The prefetch path's dequeues are always of real, Redis-mirrored
        # entries — redis_pop_for(None) must pop.
        assert is_persisted(None) is True


# ── put ───────────────────────────────────────────────────────────────────────


class TestPut:
    async def test_single_syncs_all_three_legs(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        item = _qobj(1, mock_author)
        await gq.put([item])
        assert gq.qsize() == 1
        assert gq.display_items() == [item]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [SongQueueEntry.from_queue_object(item).to_redis()]
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_batch_pushes_in_one_round_trip(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        recorded: list[str] = []
        original_batch = store.push_queue_batch
        original_single = store.push_queue

        async def spy_batch(entries: Any) -> None:
            recorded.append(f"batch:{len(entries)}")
            await original_batch(entries)

        async def spy_single(entry: Any) -> None:
            recorded.append("single")
            await original_single(entry)

        store.push_queue_batch = spy_batch
        store.push_queue = spy_single
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)], batch=True)
        assert recorded == ["batch:2"]

    async def test_non_batch_pushes_per_item(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 2
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_ytsource_items_persist_as_search_entries(
        self, gq: GuildQueue, fake_redis: aioredis.Redis, store: GuildRedisStore
    ) -> None:
        src = YTSource(ytsearch="ytsearch:some song", process=True)
        await gq.put([src], batch=True)
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert parse_queue_entry(redis_items[0]) == SearchQueueEntry.from_ytsource(src)

    async def test_in_memory_before_redis(
        self, mock_guild: MagicMock, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """The in-memory legs are populated for all items before the first
        Redis push (matching the original queue_put ordering)."""
        gq = GuildQueue(mock_guild, store)
        sizes_at_push: list[int] = []
        original = store.push_queue

        async def spy(entry: Any) -> None:
            sizes_at_push.append(gq.qsize())
            await original(entry)

        store.push_queue = spy
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        assert sizes_at_push == [2, 2]

    async def test_works_without_redis(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author)])
        assert gq_no_redis.qsize() == 1
        assert len(gq_no_redis.display_items()) == 1


class TestTheTwoCounters:
    """`qsize()` and `display_size()` mean different sets, and after the collapse
    they are one term apart over the same two fields — `len(_items) - _cursor`
    against `len(_items)` — so a swap compiles and type-checks.

    Nothing in the suite could tell them apart before: the split legs put them on
    two different objects, which made confusing them impossible rather than
    merely unlikely. These pin the gap.

    `display_size()` is the sole input to `play_history.queue_position`
    (MusicPlayer.enqueue_depth), so getting it wrong writes a plausible wrong
    number to Postgres permanently, with no error to notice."""

    async def test_they_differ_by_exactly_the_in_flight_head(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        items = [_qobj(n, mock_author) for n in range(1, 4)]
        await gq_no_redis.put(items)
        assert (gq_no_redis.qsize(), gq_no_redis.display_size()) == (3, 3)

        await gq_no_redis.get()  # claimed, not yet committed

        # The claimed song is gone from pending and still ahead of an arrival.
        assert gq_no_redis.qsize() == 2
        assert gq_no_redis.display_size() == 3

    async def test_a_claim_moves_only_qsize(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Each half of the swap, separately: give qsize() display_size()'s body
        and this fails; give display_size() qsize()'s body and the next one does."""
        await gq_no_redis.put([_qobj(1, mock_author)])
        before = gq_no_redis.qsize()
        await gq_no_redis.get()
        assert gq_no_redis.qsize() == before - 1

    async def test_a_claim_leaves_display_size_alone(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author)])
        before = gq_no_redis.display_size()
        await gq_no_redis.get()
        assert gq_no_redis.display_size() == before

    async def test_committing_the_claim_settles_both(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Only the commit retires the item from both counts — that is the window
        enqueue_depth's documented over-by-one lives in."""
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.get()
        assert (gq.qsize(), gq.display_size()) == (1, 2)

        assert await gq.try_commit_dequeue(gq.generation) is True

        assert (gq.qsize(), gq.display_size()) == (1, 1)

    async def test_empty_tracks_qsize_not_display_size(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """`empty()` is qsize()'s partner and answers about PENDING: a queue whose
        only item is claimed has nothing left to hand out, and the 300s idle
        disconnect keys off exactly that."""
        await gq_no_redis.put([_qobj(1, mock_author)])
        await gq_no_redis.get()

        assert gq_no_redis.empty() is True
        assert gq_no_redis.display_size() == 1


class TestBlockingWait:
    """`get()` parks on `_wake` instead of `asyncio.Queue`. Both I3 directions are
    tested with a BOUND rather than an assertion, because neither one raises: a
    stale-set Event spins `get()` with no suspension point until pytest's 120 s
    deadline, and a stale-clear one parks forever. A bounded wait_for turns both
    into a fast, legible failure."""

    async def test_a_removal_that_empties_the_queue_leaves_get_parked(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The wedge. `put()` sets `_wake`; `remove()` takes the only pending item
        and needs no cursor fix-up, which is exactly where a per-method rule says
        "nothing to do here". Leave `_wake` set and `get()`'s `while` loop has no
        suspension point — `Event.wait()` returns immediately when already set, so
        the whole event loop stops, not just this task."""
        item = _qobj(1, mock_author)
        await gq_no_redis.put([item])
        assert gq_no_redis._wake.is_set()

        outcome = await gq_no_redis.remove(remove_matcher(item.webpage_url))
        assert outcome.positions == [1]

        assert not gq_no_redis._wake.is_set()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gq_no_redis.get(), 0.5)

    async def test_a_restore_wakes_a_parked_getter(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The other direction: restore_* appends to an empty queue while the loop
        is parked. Today an ordering masks it — _restore_complete is set after the
        appends and the loop blocks on that first — but nothing states or tests
        that, so the wake has to be real."""
        getter = asyncio.create_task(gq_no_redis.get())
        await asyncio.sleep(0)
        assert not getter.done()

        entry = SongQueueEntry(
            webpage_url="https://yt.com/v=1",
            title="Song 1",
            requester_id=mock_author.id,
        )
        assert await gq_no_redis.restore_entries([entry]) == 1

        item = await asyncio.wait_for(getter, 0.5)
        assert isinstance(item, QueueObject)
        assert item.webpage_url == "https://yt.com/v=1"

    async def test_a_cancelled_getter_leaves_the_cursor_alone(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Cancellation lands in the wait, never mid-claim: there is no await
        between reading _items[_cursor] and returning it. A cancelled parked getter
        must claim nothing, or the item it was woken for is lost."""
        getter = asyncio.create_task(gq_no_redis.get())
        await asyncio.sleep(0)
        getter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await getter

        assert gq_no_redis._cursor == 0
        await gq_no_redis.put([_qobj(1, mock_author)])
        assert (await asyncio.wait_for(gq_no_redis.get(), 0.5)) is not None

    async def test_a_woken_then_cancelled_getter_does_not_strand_the_item(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The case asyncio.Queue.get() carries a recovery block for. Dropping the
        Queue drops that block, and the `while` re-test is what replaces it: the
        next getter re-reads the condition rather than trusting a wakeup handed to
        someone who never claimed."""
        getter = asyncio.create_task(gq_no_redis.get())
        await asyncio.sleep(0)
        await gq_no_redis.put([_qobj(1, mock_author)])  # wakes it, no chance to run
        getter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await getter

        assert gq_no_redis._cursor == 0
        assert (await asyncio.wait_for(gq_no_redis.get(), 0.5)) is not None

    async def test_a_second_waiter_re_parks_instead_of_claiming_thin_air(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Why the wait is a `while` and not an `if`. Event.wait() wakes EVERY
        waiter, so one item can wake two getters; the first claims it and
        _sync_wake() clears, and the second must re-test rather than walk off the
        end of the deque. With an `if` this raises IndexError instead of parking —
        the mutation that survives every other test here."""
        first = asyncio.create_task(gq_no_redis.get())
        second = asyncio.create_task(gq_no_redis.get())
        await asyncio.sleep(0)

        item = _qobj(1, mock_author)
        await gq_no_redis.put([item])

        assert (await asyncio.wait_for(first, 0.5)) is item
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(second), 0.3)
        assert gq_no_redis._cursor == 1

        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

    async def test_a_prefetch_can_take_the_item_a_getter_was_woken_for(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The same race with the real second consumer: the prefetch task takes
        items through get_nowait(), which can land between a getter's wakeup and
        its claim."""
        getter = asyncio.create_task(gq_no_redis.get())
        await asyncio.sleep(0)

        await gq_no_redis.put([_qobj(1, mock_author)])
        gq_no_redis.get_nowait()  # the prefetch beats the woken getter to it

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(getter), 0.3)

        getter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await getter

    async def test_randomised_park_and_wake_never_loses_an_item(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """A producer that outpaces and underruns a consumer by turns, so the
        getter both parks and finds work waiting. Every item comes out exactly
        once, in order, and the queue settles empty with _wake clear."""
        import random

        rng = random.Random(20260813)
        total = 60
        items = [_qobj(n, mock_author) for n in range(total)]
        got: list[QueueObject] = []

        async def consume() -> None:
            for _ in range(total):
                item = await gq_no_redis.get()
                assert isinstance(item, QueueObject)
                got.append(item)
                gq_no_redis.task_done()
                if rng.random() < 0.3:
                    await asyncio.sleep(0)

        consumer = asyncio.create_task(consume())
        sent = 0
        while sent < total:
            n = min(rng.randint(1, 5), total - sent)
            await gq_no_redis.put(items[sent : sent + n])
            sent += n
            for _ in range(rng.randint(0, 3)):
                await asyncio.sleep(0)

        await asyncio.wait_for(consumer, 2.0)
        assert got == items
        assert gq_no_redis.qsize() == 0
        assert not gq_no_redis._wake.is_set()


class TestDrainPending:
    """`_drain_pending` — the shared first step of clear(), shuffle() and remove(),
    so a bug here corrupts three commands at once. Its non-obvious contract is the
    task_done() balance: an unbalanced drain drifts the unfinished counter and hangs
    join(), which is what the bulk mutations depend on."""

    async def test_returns_every_item_in_queue_order(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        items = [_qobj(n, mock_author) for n in range(1, 4)]
        await gq_no_redis.put(items)

        drained = gq_no_redis._drain_pending()

        assert drained == items  # FIFO, not reversed or arbitrary

    async def test_leaves_the_pending_leg_empty(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(n, mock_author) for n in range(1, 4)])
        gq_no_redis._drain_pending()
        # The pending LEG, not qsize(): the public counter reads the deque now,
        # and _drain_pending touches only the leg it is named for. Both go in
        # phase 6.
        assert gq_no_redis._pending.qsize() == 0
        assert gq_no_redis._pending.empty()

    async def test_empty_queue_returns_empty_list(
        self, gq_no_redis: GuildQueue
    ) -> None:
        assert gq_no_redis._drain_pending() == []

    async def test_every_drained_item_is_balanced_with_task_done(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The counter must return to zero so join() completes. A missing
        task_done() hangs the next join(), surfacing as a frozen bulk
        mutation rather than an exception here."""
        await gq_no_redis.put([_qobj(n, mock_author) for n in range(1, 6)])
        assert gq_no_redis._pending._unfinished_tasks == 5

        gq_no_redis._drain_pending()

        assert gq_no_redis._pending._unfinished_tasks == 0
        await asyncio.wait_for(gq_no_redis._pending.join(), timeout=1)

    async def test_drain_is_idempotent(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """A second drain must be a no-op, not an over-balanced task_done():
        task_done() raises ValueError past the item count, so an unguarded
        re-drain crashes the caller."""
        await gq_no_redis.put([_qobj(1, mock_author)])
        assert len(gq_no_redis._drain_pending()) == 1
        assert gq_no_redis._drain_pending() == []
        await asyncio.wait_for(gq_no_redis._pending.join(), timeout=1)

    async def test_drained_items_can_be_put_back(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The drain/rebuild round-trip all four callers perform: re-putting the
        drained items must restore order and leave the counter consistent.

        Refills every leg, as clear()/shuffle()/remove() do — a _pending-only
        refill is a half-queue no caller produces, and get() reads the deque."""
        items = [_qobj(n, mock_author) for n in range(1, 4)]
        await gq_no_redis.put(items)

        drained = gq_no_redis._drain_pending()
        assert drained == items  # FIFO out
        for item in reversed(drained):
            gq_no_redis._pending.put_nowait(item)
        gq_no_redis._display = deque(reversed(drained))
        gq_no_redis._items = deque(reversed(drained))

        assert [gq_no_redis.get_nowait() for _ in range(3)] == list(reversed(items))

    async def test_mixed_item_types_survive_the_drain(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The pending leg holds unresolved YTSource entries alongside
        QueueObjects. The drain is type-agnostic and must not filter or coerce."""
        qobj = _qobj(1, mock_author)
        ytsrc = YTSource(url="https://yt.com/watch?v=lazy")
        await gq_no_redis.put([qobj, ytsrc])

        assert gq_no_redis._drain_pending() == [qobj, ytsrc]


# ── _FrontQueue ───────────────────────────────────────────────────────────────


class TestFrontQueue:
    """`put_front_nowait` — put_nowait's bookkeeping against appendleft. It reaches
    past the sanctioned _init/_get/_put hooks into CPython internals, so parity with
    put_nowait is asserted rather than assumed: a counter that drifts here hangs
    join(). Nothing in src/ calls join() today — the tests are its only consumer —
    but the same counter is what _drain_pending balances, so drift here surfaces
    as a corrupted bulk mutation rather than as a hang."""

    async def test_unfinished_task_parity_with_put_nowait(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        pending = gq_no_redis._pending
        pending.put_nowait(_qobj(1, mock_author))
        assert pending._unfinished_tasks == 1

        pending.put_front_nowait(_qobj(2, mock_author))

        assert pending._unfinished_tasks == 2

    async def test_join_completes_after_front_insert_and_task_done(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """put_nowait clears the finished-event; the head-insert twin must too."""
        pending = gq_no_redis._pending
        pending.put_front_nowait(_qobj(1, mock_author))
        pending.get_nowait()
        pending.task_done()
        await asyncio.wait_for(pending.join(), timeout=1)

    async def test_join_blocks_while_a_front_inserted_item_is_unconsumed(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The half the companion above cannot see. It consumes the item first, so
        the finished-event is set again either way and dropping _finished.clear()
        left it green — join() would then return while an unconsumed item is still
        queued."""
        pending = gq_no_redis._pending
        pending.put_nowait(_qobj(1, mock_author))
        pending.get_nowait()
        pending.task_done()  # queue drained: the finished-event is now SET

        pending.put_front_nowait(_qobj(2, mock_author))  # must clear it again

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(pending.join(), timeout=0.05)

    async def test_successive_inserts_stack_at_the_head(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        a, x, y = (_qobj(n, mock_author) for n in (1, 10, 11))
        pending = gq_no_redis._pending
        pending.put_nowait(a)

        pending.put_front_nowait(x)
        pending.put_front_nowait(y)

        assert [pending.get_nowait() for _ in range(3)] == [y, x, a]

    async def test_wakes_a_waiting_getter(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The playback loop parks in `await get()` on an empty queue. Without the
        _getters wakeup an interjection would sit at the head until some later put
        happened to release it."""
        pending = gq_no_redis._pending
        getter = asyncio.ensure_future(pending.get())
        await asyncio.sleep(0)  # let it park in the getter list

        item = _qobj(1, mock_author)
        pending.put_front_nowait(item)

        assert await asyncio.wait_for(getter, timeout=1) is item


# ── put_front (-playnow interjection) ─────────────────────────────────────────


class TestPutFront:
    async def test_front_inserts_on_all_three_legs(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        b, c = _qobj(2, mock_author), _qobj(3, mock_author)
        await gq.put([b, c])
        x, r = _qobj(10, mock_author), _qobj(11, mock_author)
        await gq.put_front([x, r])

        assert gq.display_items() == [x, r, b, c]
        # Pending leg dequeues in the same order.
        assert [gq.get_nowait() for _ in range(4)] == [x, r, b, c]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [
            SongQueueEntry.from_queue_object(i).to_redis() for i in (x, r, b, c)
        ]

    async def test_empty_items_is_noop(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.put_front([])
        assert gq.qsize() == 1
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_in_flight_head_stays_ahead_and_redis_rebuilt(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """A dequeued-but-uncommitted head (completed prefetch) must keep its
        place AHEAD of the inserted items on display and Redis — its
        commit-time LPOP retires ITS entry, not the new front item."""
        a, b = _qobj(1, mock_author), _qobj(2, mock_author)
        await gq.put([a, b])
        assert gq.get_nowait() is a  # prefetch-style dequeue; display untouched

        x = _qobj(10, mock_author)
        await gq.put_front([x])

        assert gq.display_items() == [a, x, b]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [
            SongQueueEntry.from_queue_object(i).to_redis() for i in (a, x, b)
        ]
        # Pending resumes at the inserted item (a is still held by the "prefetch").
        assert [gq.get_nowait() for _ in range(2)] == [x, b]

    async def test_multiple_items_keep_order_behind_an_in_flight_head(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """The interjection pair (song + resume tail) landing behind an in-flight
        head — both head-insert legs reverse what they insert, so this is where an
        inverted pair would show up."""
        a, b = _qobj(1, mock_author), _qobj(2, mock_author)
        await gq.put([a, b])
        assert gq.get_nowait() is a  # prefetch-style dequeue; display untouched

        x, r = _qobj(10, mock_author), _qobj(11, mock_author)
        await gq.put_front([x, r])

        assert gq.display_items() == [a, x, r, b]
        assert [gq.get_nowait() for _ in range(3)] == [x, r, b]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [
            SongQueueEntry.from_queue_object(i).to_redis() for i in (a, x, r, b)
        ]

    async def test_unpersisted_head_excluded_from_redis(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """A crash-recovered head (persisted=False) sits in front on the
        in-memory legs; LPUSHed items must land at the REDIS head without it."""
        crashed = _qobj(1, mock_author, persisted=False)
        b = _qobj(2, mock_author)
        await gq.put([crashed, b])
        # restore_crashed() puts the crashed head on the in-memory legs only;
        # rebuild the Redis leg to mirror that state (put() above wrote both).
        await fake_redis.delete(store.queue_key())
        await store.push_queue(SongQueueEntry.from_queue_object(b))

        x = _qobj(10, mock_author)
        await gq.put_front([x])

        assert gq.display_items() == [x, crashed, b]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [
            SongQueueEntry.from_queue_object(i).to_redis() for i in (x, b)
        ]

    async def test_task_accounting_balanced(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.put_front([_qobj(2, mock_author)])
        # Every pending item can be consumed and task_done'd without the
        # counter over- or under-flowing.
        while gq.qsize():
            gq.get_nowait()
            gq.task_done()
        with pytest.raises(ValueError):
            gq.task_done()  # one extra would mean the counter drifted

    async def test_works_without_redis(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author)])
        await gq_no_redis.put_front([_qobj(2, mock_author)])
        assert [queue_object(i).title for i in gq_no_redis.display_items()] == [
            "Song 2",
            "Song 1",
        ]


# ── clear ─────────────────────────────────────────────────────────────────────


class TestClear:
    async def test_drains_all_three_legs(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        items = [_qobj(1, mock_author), _qobj(2, mock_author)]
        await gq.put(items)
        cleared = await gq.clear()
        assert cleared == items
        assert gq.qsize() == 0
        assert gq.display_items() == []
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_sets_cleared_flag_consumed_once(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.clear()
        assert gq.consume_cleared_flag() is True
        assert gq.consume_cleared_flag() is False  # read-and-reset

    async def test_drain_balances_task_accounting(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Every get_nowait() in the drain is matched by task_done() — the
        unfinished-task counter returns to zero."""
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.clear()
        assert gq._pending._unfinished_tasks == 0

    async def test_empty_queue_clear_returns_empty(self, gq: GuildQueue) -> None:
        assert await gq.clear() == []

    async def test_in_flight_head_is_among_the_returned_items(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The return value is the DISPLAY leg, which still holds a dequeued-but-
        uncommitted head. Callers flushing played songs out of it depend on that:
        the loop discards such a head silently when its commit fails, so if clear()
        omitted it the play would be recorded by nobody."""
        a, b = _qobj(1, mock_author), _qobj(2, mock_author)
        await gq.put([a, b])
        assert gq.get_nowait() is a  # pending popped, display keeps it

        assert await gq.clear() == [a, b]

    async def test_works_without_redis(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author)])
        cleared = await gq_no_redis.clear()
        assert len(cleared) == 1
        assert gq_no_redis.qsize() == 0


# ── shuffle ───────────────────────────────────────────────────────────────────


class TestShuffle:
    async def test_too_few_songs_leaves_everything_untouched(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        items = [_qobj(1, mock_author), _qobj(2, mock_author), _qobj(3, mock_author)]
        await gq.put(items)
        before = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert await gq.shuffle() is ShuffleOutcome.TOO_FEW_SONGS
        assert gq.display_items() == items
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == before

    async def test_shuffle_preserves_item_set(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        items = [_qobj(n, mock_author) for n in range(1, 6)]
        await gq.put(items)
        assert await gq.shuffle() is ShuffleOutcome.SHUFFLED
        # QueueObject is unhashable — compare identity multisets, not sets.
        assert sorted(id(i) for i in gq.display_items()) == sorted(id(i) for i in items)
        assert gq.qsize() == 5
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_persisted_false_item_excluded_from_redis_rebuild(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        crashed = _qobj(99, mock_author, persisted=False)
        # Inject the crashed item the way restore does: in-memory only.
        seed_queue(gq, crashed)
        await gq.put([_qobj(n, mock_author) for n in range(1, 5)])

        assert await gq.shuffle() is ShuffleOutcome.SHUFFLED

        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        urls = {
            e.webpage_url
            for e in (parse_queue_entry(i) for i in redis_items)
            if isinstance(e, SongQueueEntry)
        }
        assert "https://yt.com/v=99" not in urls
        assert len(redis_items) == 4
        # ...but it is still in the in-memory legs.
        assert crashed in gq.display_items()

    async def test_shuffle_balances_task_accounting(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(n, mock_author) for n in range(1, 6)])
        await gq.shuffle()
        # 5 unfinished puts remain (the refilled items), not 10.
        assert gq._pending._unfinished_tasks == 5

    async def test_works_without_redis(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(n, mock_author) for n in range(1, 6)])
        assert await gq_no_redis.shuffle() is ShuffleOutcome.SHUFFLED
        assert gq_no_redis.qsize() == 5


# ── remove ────────────────────────────────────────────────────────────────────


class TestRemoveMatcher:
    """`remove_matcher` — the union policy, tested without a queue. RESOLVED is
    tried first so every removal that worked before works identically."""

    def _song(self, url: str, origin: str | None) -> QueueObject:
        return QueueObject(url, "Song", MagicMock(), user_input=origin)

    def test_resolved_url_still_matches(self) -> None:
        item = self._song("https://yt.com/v=1", "some search")
        assert remove_matcher("https://yt.com/v=1")(item) is RemoveMode.RESOLVED

    def test_the_search_text_matches(self) -> None:
        item = self._song("https://yt.com/v=1", "never gonna give you up")
        match = remove_matcher("never gonna give you up")
        assert match(item) is RemoveMode.ORIGIN

    def test_search_text_is_case_and_space_insensitive(self) -> None:
        """Retyping a search rather than pasting it is the normal way to use this,
        so folding is what makes it usable at all."""
        item = self._song("https://yt.com/v=1", "Never  Gonna Give You Up")
        assert remove_matcher(" never gonna give you up ")(item) is RemoveMode.ORIGIN

    def test_links_keep_their_case(self) -> None:
        """A Spotify id is case-sensitive base62, so folding a link would let one
        album's id match another's. Text still folds; links must not."""
        item = self._song("https://yt.com/v=1", "https://open.spotify.com/album/AbC")
        assert remove_matcher("https://open.spotify.com/album/abc")(item) is None
        assert (
            remove_matcher("https://open.spotify.com/album/AbC")(item)
            is RemoveMode.ORIGIN
        )

    def test_unresolved_search_entry_matches_on_its_origin(self) -> None:
        """A Spotify-playlist track has no resolved URL yet — the origin is the
        only thing it can be matched by, and the only place the album link is."""
        album = "https://open.spotify.com/album/xyz"
        item = YTSource(ytsearch="ytsearch:Track One", user_input=album)
        assert remove_matcher(album)(item) is RemoveMode.ORIGIN

    def test_no_origin_recorded_never_matches_by_origin(self) -> None:
        """A pre-feature queue entry rehydrates with user_input=None, which must
        not collide with anything — least of all an empty argument."""
        item = self._song("https://yt.com/v=1", None)
        assert remove_matcher("")(item) is None

    def test_unrelated_needle_matches_nothing(self) -> None:
        item = self._song("https://yt.com/v=1", "some search")
        assert remove_matcher("something else")(item) is None


class TestRemove:
    async def test_removes_matching_and_returns_positions(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        target = _qobj(2, mock_author)
        other = _qobj(1, mock_author)
        duplicate = QueueObject("https://yt.com/v=2", "Song 2 again", mock_author)
        await gq.put([other, target, duplicate])

        outcome = await gq.remove(remove_matcher("https://yt.com/v=2"))

        assert outcome.positions == [2, 3]
        assert outcome.removed == [target, duplicate]  # the items, not just where
        assert gq.display_items() == [other]
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_no_match_returns_empty_and_mutates_nothing(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        before = await fake_redis.lrange(store.queue_key(), 0, -1)
        outcome = await gq.remove(remove_matcher("https://yt.com/v=none"))
        assert (outcome.positions, outcome.removed) == ([], [])
        assert len(gq.display_items()) == 1
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == before

    async def test_removing_everything_deletes_redis_key(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        outcome = await gq.remove(remove_matcher("https://yt.com/v=1"))
        assert outcome.positions == [1]
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_matches_ytsource_by_url(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        src = YTSource(url="https://yt.com/v=7", process=False)
        await gq.put([src, _qobj(1, mock_author)])
        outcome = await gq.remove(remove_matcher("https://yt.com/v=7"))
        assert outcome.positions == [1]
        assert outcome.removed == [src]  # YTSources come back too; the caller filters
        assert len(gq.display_items()) == 1


class TestResumeTailDepth:
    """How deep the -playnow stack is: the run of parked plays behind the head.
    Index 0 is skipped because it is the song that just cut the line, not
    something waiting to resume."""

    def _tail(self, n: int, requester: Any) -> QueueObject:
        return QueueObject(
            f"https://yt.com/v={n}", f"Song {n}", requester, ts=30 * n, is_resume=True
        )

    async def test_empty_queue_is_zero(self, gq_no_redis: GuildQueue) -> None:
        assert gq_no_redis.resume_tail_depth() == 0

    async def test_head_alone_is_zero(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author)])
        assert gq_no_redis.resume_tail_depth() == 0

    async def test_plain_interjection_is_one(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(9, mock_author), self._tail(1, mock_author)])
        assert gq_no_redis.resume_tail_depth() == 1

    async def test_an_in_flight_head_does_not_hide_the_interjection(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The interjected song is not always at display index 0. put_front's
        in-flight-head branch keeps a dequeued-but-uncommitted item ahead of what
        it inserts — reachable when the loop awaits a still-running prefetch while
        interject() runs inside that await. Counting from a fixed index 1 started
        the run ON the interjected song, so a real interjection reported 0."""
        await gq_no_redis.put([_qobj(7, mock_author)])
        held = gq_no_redis.get_nowait()  # dequeued, display still holds it
        assert held is not None

        await gq_no_redis.put_front([_qobj(9, mock_author), self._tail(1, mock_author)])

        assert gq_no_redis.resume_tail_depth() == 1

    async def test_counts_the_consecutive_run(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put(
            [_qobj(9, mock_author)] + [self._tail(n, mock_author) for n in (3, 2, 1)]
        )
        assert gq_no_redis.resume_tail_depth() == 3

    async def test_stops_at_the_first_ordinary_song(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Songs past the tails were queued normally and were never interrupted.
        Counting them would report depth as "queue length" on any guild that had
        ever used -playnow."""
        await gq_no_redis.put(
            [
                _qobj(9, mock_author),
                self._tail(1, mock_author),
                _qobj(5, mock_author),
                self._tail(2, mock_author),  # not contiguous — not part of the stack
            ]
        )
        assert gq_no_redis.resume_tail_depth() == 1

    async def test_a_head_that_is_itself_a_tail_is_not_counted(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The head is about to play, so it is a live fragment rather than a
        parked one — the same one-play-one-count rule has_resume_tail encodes."""
        await gq_no_redis.put([self._tail(1, mock_author), self._tail(2, mock_author)])
        assert gq_no_redis.resume_tail_depth() == 1

    async def test_a_search_entry_breaks_the_run(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put(
            [
                _qobj(9, mock_author),
                self._tail(1, mock_author),
                YTSource(ytsearch="ytsearch:something"),
                self._tail(2, mock_author),
            ]
        )
        assert gq_no_redis.resume_tail_depth() == 1


# ── crash recovery ────────────────────────────────────────────────────────────


class TestRestoreEntries:
    def _entry(self, n: int, requester_id: int) -> SongQueueEntry:
        return SongQueueEntry(
            webpage_url=f"https://yt.com/v={n}",
            title=f"Song {n}",
            requester_id=requester_id,
            duration=100 + n,
        )

    async def test_restores_in_order_in_memory_only(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_guild: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        mock_guild.get_member = MagicMock(return_value=mock_author)
        count = await gq.restore_entries(
            [self._entry(1, mock_author.id), self._entry(2, mock_author.id)]
        )
        assert count == 2
        items = gq.display_items()
        assert [queue_object(i).webpage_url for i in items] == [
            "https://yt.com/v=1",
            "https://yt.com/v=2",
        ]
        assert queue_object(items[0]).requester is mock_author
        assert queue_object(items[0]).duration == 101
        # In-memory only: the entries were already on the Redis list.
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_departed_member_falls_back_to_owner(
        self, gq: GuildQueue, mock_guild: MagicMock
    ) -> None:
        mock_guild.get_member = MagicMock(return_value=None)
        count = await gq.restore_entries([self._entry(1, 12345)])
        assert count == 1
        assert queue_object(gq.display_items()[0]).requester is mock_guild.owner

    async def test_unresolvable_requester_drops_entry(
        self, gq: GuildQueue, mock_guild: MagicMock
    ) -> None:
        mock_guild.get_member = MagicMock(return_value=None)
        mock_guild.owner = None
        count = await gq.restore_entries([self._entry(1, 12345)])
        assert count == 0
        assert gq.qsize() == 0

    async def test_playnow_flags_rehydrate(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        mock_guild.get_member.return_value = mock_author
        entry = SongQueueEntry(
            webpage_url="https://yt.com/v=1",
            title="Resume Tail",
            requester_id=mock_author.id,
            ts=151,
            is_resume=True,
            start_paused=True,
        )
        assert await gq.restore_entries([entry]) == 1
        item = gq.display_items()[0]
        assert isinstance(item, QueueObject)
        assert item.is_resume is True
        assert item.start_paused is True
        assert item.interjected is False
        assert item.ts == 151

    async def test_origin_rehydrates_on_the_search_branch(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        """The song branch already carried user_input; the search branch is the new
        leg, and it is the one that matters — a Spotify-album track holds the album
        link nowhere else, so losing it here breaks -remove after a restart."""
        mock_guild.get_member.return_value = mock_author
        album = "https://open.spotify.com/album/abc123"
        assert (
            await gq.restore_entries(
                [SearchQueueEntry(ytsearch="ytsearch:Track One", user_input=album)]
            )
            == 1
        )
        (restored,) = gq.display_items()
        assert isinstance(restored, YTSource)
        assert restored.user_input == album

    async def test_enqueue_stamps_rehydrate_on_both_entry_types(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        """_rehydrate is the entry → item hop for the whole restored queue, and
        the stamps must survive it: guild_state's CURRENT_SONG_QUEUED_AT records
        that they are carried and never restamped, so a crash-recovered song
        archives the position it was originally queued at. Zeroing either leg
        left the whole suite green."""
        mock_guild.get_member.return_value = mock_author
        song = SongQueueEntry(
            webpage_url="https://yt.com/v=1",
            title="Song 1",
            requester_id=mock_author.id,
            queued_at=1752529000.5,
            queue_position=3,
        )
        search = SearchQueueEntry(
            ytsearch="ytsearch:abc",
            process=True,
            queued_at=1752529111.5,
            queue_position=7,
        )
        assert await gq.restore_entries([song, search]) == 2
        restored_song, restored_search = gq.display_items()
        assert (
            restored_song.analytics.queued_at,
            restored_song.analytics.queue_position,
        ) == (1752529000.5, 3)
        assert (
            restored_search.analytics.queued_at,
            restored_search.analytics.queue_position,
        ) == (1752529111.5, 7)

    async def test_search_entries_rehydrate_to_ytsource(
        self, gq: GuildQueue, mock_guild: MagicMock
    ) -> None:
        entry = SearchQueueEntry(ytsearch="ytsearch:abc", process=True)
        count = await gq.restore_entries([entry])
        assert count == 1
        item = gq.display_items()[0]
        assert isinstance(item, YTSource)
        assert item.ytsearch == "ytsearch:abc"


class TestRestoreCrashed:
    def _crashed_entry(self, requester_id: int | None) -> SongQueueEntry:
        return SongQueueEntry(
            webpage_url="https://yt.com/v=crash",
            title="Crashed",
            requester_id=requester_id,
            ts=95,
            persisted=False,
        )

    async def test_requeues_with_position_and_persisted_false(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        mock_guild.get_member = MagicMock(return_value=mock_author)
        assert await gq.restore_crashed(
            self._crashed_entry(mock_author.id), requester_fallback=mock_guild.me
        )
        item = gq.display_items()[0]
        assert item.ts == 95
        assert queue_object(item).persisted is False
        assert queue_object(item).requester is mock_author

    async def test_carries_the_enqueue_stamps_of_the_original_enqueue(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        # The crashed song archives where it was queued, not where the restart
        # put it — it goes to the front on recovery, which is not position 0.
        mock_guild.get_member = MagicMock(return_value=mock_author)
        entry = replace(
            self._crashed_entry(mock_author.id),
            queued_at=1752529000.5,
            queue_position=6,
        )
        assert await gq.restore_crashed(entry, requester_fallback=mock_guild.me)
        item = queue_object(gq.display_items()[0])
        assert (item.analytics.queued_at, item.analytics.queue_position) == (
            1752529000.5,
            6,
        )

    async def test_fallback_used_when_member_gone(
        self, gq: GuildQueue, mock_guild: MagicMock
    ) -> None:
        mock_guild.get_member = MagicMock(return_value=None)
        fallback = mock_guild.me
        assert await gq.restore_crashed(
            self._crashed_entry(12345), requester_fallback=fallback
        )
        assert queue_object(gq.display_items()[0]).requester is fallback

    async def test_no_requester_id_goes_straight_to_fallback(
        self, gq: GuildQueue, mock_guild: MagicMock
    ) -> None:
        mock_guild.get_member = MagicMock(return_value=None)
        assert await gq.restore_crashed(
            self._crashed_entry(None), requester_fallback=mock_guild.me
        )
        mock_guild.get_member.assert_not_called()

    async def test_unresolvable_returns_false_and_enqueues_nothing(
        self, gq: GuildQueue, mock_guild: MagicMock
    ) -> None:
        # Member gone AND no fallback resolvable (guild.me and guild.owner
        # both None — the caller passes `me or owner`, and _rehydrate's own
        # owner default must also come up empty).
        mock_guild.get_member = MagicMock(return_value=None)
        mock_guild.owner = None
        assert not await gq.restore_crashed(
            self._crashed_entry(12345), requester_fallback=None
        )
        assert gq.qsize() == 0
        assert gq.display_items() == []


# ── display access ────────────────────────────────────────────────────────────


class TestDisplayAccess:
    async def test_display_items_returns_copy(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        items = gq.display_items()
        items.clear()  # mutating the copy must not touch the queue
        assert len(gq.display_items()) == 1

    async def test_peek_next(self, gq: GuildQueue, mock_author: MagicMock) -> None:
        assert gq.peek_next() is None
        first = _qobj(1, mock_author)
        await gq.put([first, _qobj(2, mock_author)])
        assert gq.peek_next() is first


# ── loop dequeue bookkeeping ──────────────────────────────────────────────────


class TestDequeueBookkeeping:
    async def test_redis_pop_for_persisted_item(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.redis_pop_for(_qobj(1, mock_author))
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 1

    async def test_redis_pop_skipped_for_unpersisted_item(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.redis_pop_for(_qobj(99, mock_author, persisted=False))
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 1  # untouched

    async def test_pop_display_head_warns_on_empty(
        self, gq: GuildQueue, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="src.guild_queue"):
            gq.pop_display_head("failed-song pop")
        assert "failed-song pop" in caplog.text

    async def test_pop_display_head_pops(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        # Claim first, as every caller does — this runs only from
        # finish_failed_dequeue, which is reached after a get().
        await gq.get()
        gq.pop_display_head()
        assert gq.display_items() == []

    async def test_try_pop_display_head(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        assert gq.try_pop_display_head() is False
        await gq.put([_qobj(1, mock_author)])
        await gq.get()  # the claim this settles
        assert gq.try_pop_display_head() is True
        assert gq.display_items() == []

    async def test_commit_dequeue_shares_the_bulk_mutation_lock(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """try_commit_dequeue() and the bulk ops really do serialize on one
        lock — a held lock blocks clear() until released. (Whitebox: the lock
        is deliberately not part of the public API since Phase 5.)"""
        await gq.put([_qobj(1, mock_author)])
        async with gq._mutex:
            clear_task = asyncio.create_task(gq.clear())
            await asyncio.sleep(0)
            assert not clear_task.done()  # blocked on the lock we hold
        cleared = await clear_task
        assert len(cleared) == 1

    async def test_finish_failed_dequeue_triplet(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """One call retires a failed dequeue on all three legs: display head
        popped, Redis LPOPed, task_done balanced."""
        item = _qobj(1, mock_author)
        await gq.put([item])
        _ = await gq.get()  # the loop dequeued it
        await gq.finish_failed_dequeue(item)
        assert gq.display_items() == []
        assert await fake_redis.exists(store.queue_key()) == 0
        assert gq._pending._unfinished_tasks == 0

    async def test_finish_failed_dequeue_skips_redis_for_unpersisted(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])  # the real, persisted entry
        crashed = _qobj(99, mock_author, persisted=False)
        seed_queue(gq, crashed)
        _ = await gq.get()
        await gq.finish_failed_dequeue(crashed)
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 1  # persisted entry untouched

    async def test_try_commit_dequeue_true_then_false_after_clear(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.get()  # the claim the first commit settles
        generation = gq.generation
        assert await gq.try_commit_dequeue(generation) is True
        await gq.clear()
        assert await gq.try_commit_dequeue(generation) is False


# ── requeue_front ─────────────────────────────────────────────────────────────


class TestRequeueFront:
    async def test_restores_item_to_front_in_order(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        a, b, c = (_qobj(n, mock_author) for n in (1, 2, 3))
        await gq.put([a, b, c])
        got = gq.get_nowait()
        assert got is a
        gq.requeue_front(got)
        assert gq.qsize() == 3
        assert gq.display_items() == [a, b, c]
        await _assert_triad_sync(gq, fake_redis, store)
        assert gq.get_nowait() is a

    async def test_task_slot_transfers_to_future_consumer(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        a = _qobj(1, mock_author)
        await gq.put([a])
        gq.requeue_front(gq.get_nowait())
        assert gq.get_nowait() is a
        gq.task_done()
        await asyncio.wait_for(gq._pending.join(), timeout=1)

    async def test_accepts_resolved_substitute(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        # A YTSource dequeued by the prefetch may come back in resolved form.
        src = YTSource(ytsearch="artist song")
        await gq_no_redis.put([src])
        gq_no_redis.get_nowait()
        resolved = _qobj(9, mock_author)
        gq_no_redis.requeue_front(resolved)
        assert gq_no_redis.qsize() == 1
        assert gq_no_redis.get_nowait() is resolved


# ── bulk mutations vs in-flight dequeue ───────────────────────────────────────


class TestShuffleWithInFlightDequeue:
    async def test_in_flight_head_keeps_display_and_redis_position(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """The loop is resolving a dequeued item (display/Redis heads uncommitted);
        -shuffle must reorder only the pending items and carry the in-flight head
        through on both legs, or the commit retires someone else's entry."""
        items = [_qobj(n, mock_author) for n in range(1, 6)]
        await gq.put(items)
        in_flight = await gq.get()  # the loop's dequeue; commit comes later

        assert await gq.shuffle() is ShuffleOutcome.SHUFFLED

        display = gq.display_items()
        assert display[0] is in_flight
        assert len(display) == 5
        assert gq.qsize() == 4
        assert sorted(id(i) for i in display[1:]) == sorted(id(i) for i in items[1:])
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 5
        assert parse_queue_entry(redis_items[0]) == SongQueueEntry.from_queue_object(
            queue_object(in_flight)
        )

        # The loop finishes resolving and commits, exactly as musicplayer
        # does: display pop + the start transaction's LPOP.
        assert await gq.try_commit_dequeue(gq.generation) is True
        await store.pop_queue()
        gq.task_done()
        await _assert_triad_sync(gq, fake_redis, store)
        redis_after = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert [parse_queue_entry(r) for r in redis_after] == [
            SongQueueEntry.from_queue_object(queue_object(i))
            for i in gq.display_items()
        ]

    async def test_unpersisted_in_flight_head_kept_on_display_not_redis(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_guild: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        """The crash-recovered head (persisted=False) mid-resolve: shuffle
        must keep its display-head position but never write it to Redis."""
        mock_guild.get_member = MagicMock(return_value=mock_author)
        crashed = SongQueueEntry(
            webpage_url="https://yt.com/v=crash",
            title="Crashed",
            requester_id=mock_author.id,
            persisted=False,
        )
        assert await gq.restore_crashed(crashed, requester_fallback=mock_guild.me)
        await gq.put([_qobj(n, mock_author) for n in range(1, 5)])
        in_flight = await gq.get()
        assert queue_object(in_flight).persisted is False

        assert await gq.shuffle() is ShuffleOutcome.SHUFFLED

        assert gq.display_items()[0] is in_flight
        assert len(gq.display_items()) == 5
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 4  # the crashed head is never persisted


class TestMirrorWriteChoice:
    """Which Redis write a mutation picks. A rebuild costs the same whether it
    drops one entry or fifty, so a small -remove takes the LREM path instead —
    but only a removal may, because LREM says the survivors kept their order."""

    def _spy(self, store: GuildRedisStore) -> list[str]:
        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries", "delete_queue"):
            original = getattr(store, name)

            def spy(*args: Any, _n: str = name, _o: Any = original, **kw: Any) -> Any:
                calls.append(_n)
                return _o(*args, **kw)

            setattr(store, name, spy)
        return calls

    async def test_a_small_removal_lrems(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(n, mock_author) for n in range(1, 6)])
        calls = self._spy(store)

        await gq.remove(remove_matcher("https://yt.com/v=3"))

        assert calls == ["remove_queue_entries"]

    async def test_a_large_removal_rebuilds(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """Past the threshold the per-LREM scans overtake one rewrite."""
        album = "https://open.spotify.com/album/big"
        await gq.put(
            [
                QueueObject(
                    f"https://yt.com/v={n}", f"T{n}", mock_author, user_input=album
                )
                for n in range(_LREM_MAX_ENTRIES + 1)
            ]
            # A survivor, or the queue empties and DELETE wins before the
            # threshold is ever consulted.
            + [_qobj(999, mock_author)]
        )
        calls = self._spy(store)

        outcome = await gq.remove(remove_matcher(album))

        assert len(outcome.positions) == _LREM_MAX_ENTRIES + 1
        assert calls == ["rebuild_queue"]

    async def test_removing_everything_deletes_the_key(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """Not an LREM of each: an empty list must not be left behind for the next
        restore to find, and DELETE is one round trip rather than n."""
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        calls = self._spy(store)

        await gq.remove(lambda _item: RemoveMode.RESOLVED)

        assert calls == ["delete_queue"]

    async def test_shuffle_always_rebuilds(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """LREM cannot express a reorder — it removes, and shuffle removes nothing.
        Passing it a `removed` set here would silently leave Redis in the old
        order while memory held the new one."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 6)])
        calls = self._spy(store)

        await gq.shuffle()

        assert calls == ["rebuild_queue"]


class TestRemoveWithInFlightDequeue:
    async def test_in_flight_head_survives_and_positions_match_embed(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        items = [_qobj(n, mock_author) for n in range(1, 6)]
        await gq.put(items)
        in_flight = await gq.get()  # items[0]

        outcome = await gq.remove(remove_matcher(items[2].webpage_url))

        # The queue embed numbers display items from 1 with the in-flight
        # head included, so items[2] shows as #3.
        assert outcome.positions == [3]
        assert outcome.removed == [items[2]]
        display = gq.display_items()
        assert display[0] is in_flight
        assert display == [in_flight, items[1], items[3], items[4]]
        assert gq.qsize() == 3
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert parse_queue_entry(redis_items[0]) == SongQueueEntry.from_queue_object(
            queue_object(in_flight)
        )

        assert await gq.try_commit_dequeue(gq.generation) is True
        await store.pop_queue()
        gq.task_done()
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_in_flight_head_never_removed_even_on_url_match(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """Removing the resolving/starting song is -skip's job: a URL match
        against the in-flight head removes only pending duplicates."""
        a1 = _qobj(1, mock_author)
        a_dup = _qobj(1, mock_author)  # same URL, still pending
        b = _qobj(2, mock_author)
        await gq.put([a1, a_dup, b])
        in_flight = await gq.get()
        assert in_flight is a1

        outcome = await gq.remove(remove_matcher(a1.webpage_url))

        assert outcome.positions == [2]  # only the pending duplicate
        # The head is committed to play, so it is not among the removed items
        # either — a flush over `removed` must never record a song still playing.
        assert outcome.removed == [a_dup]
        assert gq.display_items() == [a1, b]
        assert gq.qsize() == 1


# ── put vs clear mutual exclusion ─────────────────────────────────────────────


class TestPutClearMutualExclusion:
    async def test_clear_cannot_interleave_between_puts_memory_and_redis_writes(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """put() suspends at its Redis push; a concurrent clear() must block
        on the mutex instead of draining at that point — otherwise the push
        lands after clear's DEL and resurrects the entry as a ghost that the
        next dequeue would LPOP instead of its own."""
        release = asyncio.Event()
        original_push = store.push_queue

        async def gated_push(entry: Any) -> None:
            await release.wait()
            await original_push(entry)

        with patch.object(store, "push_queue", new=gated_push):
            put_task = asyncio.create_task(gq.put([_qobj(1, mock_author)]))
            await asyncio.sleep(0)  # put reaches the gated push, holding the mutex
            clear_task = asyncio.create_task(gq.clear())
            await asyncio.sleep(0)
            assert not clear_task.done()  # blocked on the mutex, not interleaved
            release.set()
            await put_task
            cleared = await clear_task

        assert [queue_object(i).title for i in cleared] == ["Song 1"]
        assert gq.qsize() == 0
        assert gq.display_items() == []
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == []
