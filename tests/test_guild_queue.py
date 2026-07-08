"""Tests for src/guild_queue.py — the queue domain class.

The central property under test is the triad-sync invariant: after every
operation, the asyncio queue, the display deque, and the Redis mirror agree
(persisted=False items exist only on the in-memory legs by design).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.guild_queue import GuildQueue, ShuffleOutcome
from src.guild_state import SearchQueueEntry, SongQueueEntry, parse_queue_entry
from src.redis_client import GuildRedisStore
from src.sources import YTSource
from src.youtube import QueueObject


@pytest.fixture
def store(fake_redis, mock_guild):
    return GuildRedisStore(fake_redis, guild_id=mock_guild.id)


@pytest.fixture
def gq(mock_guild, store):
    return GuildQueue(mock_guild, store)


@pytest.fixture
def gq_no_redis(mock_guild):
    return GuildQueue(mock_guild, None)


def _qobj(n: int, requester, *, persisted: bool = True) -> QueueObject:
    return QueueObject(
        f"https://yt.com/v={n}", f"Song {n}", requester, persisted=persisted
    )


async def _assert_triad_sync(gq: GuildQueue, fake_redis, store) -> None:
    """The invariant: all three legs agree (Redis holds persisted items only)."""
    items = gq.display_items()
    assert gq.qsize() == len(items)
    redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
    persisted = [i for i in items if getattr(i, "persisted", True)]
    assert len(redis_items) == len(persisted)


# ── put ───────────────────────────────────────────────────────────────────────


class TestPut:
    async def test_single_syncs_all_three_legs(
        self, gq, fake_redis, store, mock_author
    ):
        item = _qobj(1, mock_author)
        await gq.put([item])
        assert gq.qsize() == 1
        assert gq.display_items() == [item]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [SongQueueEntry.from_queue_object(item).to_redis()]
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_batch_pushes_in_one_round_trip(self, gq, store, mock_author):
        recorded: list[str] = []
        original_batch = store.push_queue_batch
        original_single = store.push_queue

        async def spy_batch(entries):
            recorded.append(f"batch:{len(entries)}")
            await original_batch(entries)

        async def spy_single(entry):
            recorded.append("single")
            await original_single(entry)

        store.push_queue_batch = spy_batch
        store.push_queue = spy_single
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)], batch=True)
        assert recorded == ["batch:2"]

    async def test_non_batch_pushes_per_item(self, gq, fake_redis, store, mock_author):
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 2
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_ytsource_items_persist_as_search_entries(
        self, gq, fake_redis, store
    ):
        src = YTSource(ytsearch="ytsearch:some song", process=True)
        await gq.put([src], batch=True)
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert parse_queue_entry(redis_items[0]) == SearchQueueEntry.from_ytsource(src)

    async def test_in_memory_before_redis(self, mock_guild, store, mock_author):
        """The in-memory legs are populated for ALL items before the first
        Redis push (matching the original queue_put ordering)."""
        gq = GuildQueue(mock_guild, store)
        sizes_at_push: list[int] = []
        original = store.push_queue

        async def spy(entry):
            sizes_at_push.append(gq.qsize())
            await original(entry)

        store.push_queue = spy
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        assert sizes_at_push == [2, 2]

    async def test_works_without_redis(self, gq_no_redis, mock_author):
        await gq_no_redis.put([_qobj(1, mock_author)])
        assert gq_no_redis.qsize() == 1
        assert len(gq_no_redis.display_items()) == 1


# ── clear ─────────────────────────────────────────────────────────────────────


class TestClear:
    async def test_drains_all_three_legs(self, gq, fake_redis, store, mock_author):
        items = [_qobj(1, mock_author), _qobj(2, mock_author)]
        await gq.put(items)
        cleared = await gq.clear()
        assert cleared == items
        assert gq.qsize() == 0
        assert gq.display_items() == []
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_sets_cleared_flag_consumed_once(self, gq, mock_author):
        await gq.put([_qobj(1, mock_author)])
        await gq.clear()
        assert gq.consume_cleared_flag() is True
        assert gq.consume_cleared_flag() is False  # read-and-reset

    async def test_drain_balances_task_accounting(self, gq, mock_author):
        """Every get_nowait() in the drain is matched by task_done() — the
        unfinished-task counter returns to zero."""
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.clear()
        assert gq._pending._unfinished_tasks == 0

    async def test_empty_queue_clear_returns_empty(self, gq):
        assert await gq.clear() == []

    async def test_works_without_redis(self, gq_no_redis, mock_author):
        await gq_no_redis.put([_qobj(1, mock_author)])
        cleared = await gq_no_redis.clear()
        assert len(cleared) == 1
        assert gq_no_redis.qsize() == 0


# ── shuffle ───────────────────────────────────────────────────────────────────


class TestShuffle:
    async def test_too_few_songs_leaves_everything_untouched(
        self, gq, fake_redis, store, mock_author
    ):
        items = [_qobj(1, mock_author), _qobj(2, mock_author), _qobj(3, mock_author)]
        await gq.put(items)
        before = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert await gq.shuffle() is ShuffleOutcome.TOO_FEW_SONGS
        assert gq.display_items() == items
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == before

    async def test_shuffle_preserves_item_set(self, gq, fake_redis, store, mock_author):
        items = [_qobj(n, mock_author) for n in range(1, 6)]
        await gq.put(items)
        assert await gq.shuffle() is ShuffleOutcome.SHUFFLED
        # QueueObject is unhashable — compare identity multisets, not sets.
        assert sorted(id(i) for i in gq.display_items()) == sorted(id(i) for i in items)
        assert gq.qsize() == 5
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_persisted_false_item_excluded_from_redis_rebuild(
        self, gq, fake_redis, store, mock_author
    ):
        crashed = _qobj(99, mock_author, persisted=False)
        # Inject the crashed item the way restore does: in-memory only.
        await gq._pending.put(crashed)
        gq._display.append(crashed)
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

    async def test_shuffle_balances_task_accounting(self, gq, mock_author):
        await gq.put([_qobj(n, mock_author) for n in range(1, 6)])
        await gq.shuffle()
        # 5 unfinished puts remain (the refilled items), not 10.
        assert gq._pending._unfinished_tasks == 5

    async def test_works_without_redis(self, gq_no_redis, mock_author):
        await gq_no_redis.put([_qobj(n, mock_author) for n in range(1, 6)])
        assert await gq_no_redis.shuffle() is ShuffleOutcome.SHUFFLED
        assert gq_no_redis.qsize() == 5


# ── remove ────────────────────────────────────────────────────────────────────


class TestRemove:
    async def test_removes_matching_and_returns_positions(
        self, gq, fake_redis, store, mock_author
    ):
        target = _qobj(2, mock_author)
        other = _qobj(1, mock_author)
        duplicate = QueueObject("https://yt.com/v=2", "Song 2 again", mock_author)
        await gq.put([other, target, duplicate])

        positions = await gq.remove("https://yt.com/v=2")

        assert positions == [2, 3]
        assert gq.display_items() == [other]
        await _assert_triad_sync(gq, fake_redis, store)

    async def test_no_match_returns_empty_and_mutates_nothing(
        self, gq, fake_redis, store, mock_author
    ):
        await gq.put([_qobj(1, mock_author)])
        before = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert await gq.remove("https://yt.com/v=none") == []
        assert len(gq.display_items()) == 1
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == before

    async def test_removing_everything_deletes_redis_key(
        self, gq, fake_redis, store, mock_author
    ):
        await gq.put([_qobj(1, mock_author)])
        positions = await gq.remove("https://yt.com/v=1")
        assert positions == [1]
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_matches_ytsource_by_url(self, gq, fake_redis, store, mock_author):
        src = YTSource(url="https://yt.com/v=7", process=False)
        await gq.put([src, _qobj(1, mock_author)])
        positions = await gq.remove("https://yt.com/v=7")
        assert positions == [1]
        assert len(gq.display_items()) == 1


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
        self, gq, fake_redis, store, mock_guild, mock_author
    ):
        mock_guild.get_member = MagicMock(return_value=mock_author)
        count = await gq.restore_entries(
            [self._entry(1, mock_author.id), self._entry(2, mock_author.id)]
        )
        assert count == 2
        items = gq.display_items()
        assert [i.webpage_url for i in items] == [
            "https://yt.com/v=1",
            "https://yt.com/v=2",
        ]
        assert items[0].requester is mock_author
        assert items[0].duration == 101
        # In-memory only: the entries were already on the Redis list.
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_departed_member_falls_back_to_owner(self, gq, mock_guild):
        mock_guild.get_member = MagicMock(return_value=None)
        count = await gq.restore_entries([self._entry(1, 12345)])
        assert count == 1
        assert gq.display_items()[0].requester is mock_guild.owner

    async def test_unresolvable_requester_drops_entry(self, gq, mock_guild):
        mock_guild.get_member = MagicMock(return_value=None)
        mock_guild.owner = None
        count = await gq.restore_entries([self._entry(1, 12345)])
        assert count == 0
        assert gq.qsize() == 0

    async def test_search_entries_rehydrate_to_ytsource(self, gq, mock_guild):
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
        self, gq, mock_guild, mock_author
    ):
        mock_guild.get_member = MagicMock(return_value=mock_author)
        assert await gq.restore_crashed(
            self._crashed_entry(mock_author.id), requester_fallback=mock_guild.me
        )
        item = gq.display_items()[0]
        assert item.ts == 95
        assert item.persisted is False
        assert item.requester is mock_author

    async def test_fallback_used_when_member_gone(self, gq, mock_guild):
        mock_guild.get_member = MagicMock(return_value=None)
        fallback = mock_guild.me
        assert await gq.restore_crashed(
            self._crashed_entry(12345), requester_fallback=fallback
        )
        assert gq.display_items()[0].requester is fallback

    async def test_no_requester_id_goes_straight_to_fallback(self, gq, mock_guild):
        mock_guild.get_member = MagicMock(return_value=None)
        assert await gq.restore_crashed(
            self._crashed_entry(None), requester_fallback=mock_guild.me
        )
        mock_guild.get_member.assert_not_called()

    async def test_unresolvable_returns_false_and_enqueues_nothing(
        self, gq, mock_guild
    ):
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
    async def test_display_items_returns_copy(self, gq, mock_author):
        await gq.put([_qobj(1, mock_author)])
        items = gq.display_items()
        items.clear()  # mutating the copy must not touch the queue
        assert len(gq.display_items()) == 1

    async def test_peek_next(self, gq, mock_author):
        assert gq.peek_next() is None
        first = _qobj(1, mock_author)
        await gq.put([first, _qobj(2, mock_author)])
        assert gq.peek_next() is first


# ── loop dequeue bookkeeping ──────────────────────────────────────────────────


class TestDequeueBookkeeping:
    async def test_redis_pop_for_persisted_item(
        self, gq, fake_redis, store, mock_author
    ):
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.redis_pop_for(_qobj(1, mock_author))
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 1

    async def test_redis_pop_skipped_for_unpersisted_item(
        self, gq, fake_redis, store, mock_author
    ):
        await gq.put([_qobj(1, mock_author)])
        await gq.redis_pop_for(_qobj(99, mock_author, persisted=False))
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 1  # untouched

    async def test_pop_display_head_warns_on_empty(self, gq, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="src.guild_queue"):
            gq.pop_display_head("failed-song pop")
        assert "failed-song pop" in caplog.text

    async def test_pop_display_head_pops(self, gq, mock_author):
        await gq.put([_qobj(1, mock_author)])
        gq.pop_display_head()
        assert gq.display_items() == []

    async def test_try_pop_display_head(self, gq, mock_author):
        assert gq.try_pop_display_head() is False
        await gq.put([_qobj(1, mock_author)])
        assert gq.try_pop_display_head() is True
        assert gq.display_items() == []

    async def test_commit_dequeue_shares_the_bulk_mutation_lock(self, gq, mock_author):
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
        self, gq, fake_redis, store, mock_author
    ):
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
        self, gq, fake_redis, store, mock_author
    ):
        await gq.put([_qobj(1, mock_author)])  # the real, persisted entry
        crashed = _qobj(99, mock_author, persisted=False)
        await gq._pending.put(crashed)
        gq._display.append(crashed)
        _ = await gq.get()
        await gq.finish_failed_dequeue(crashed)
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert len(redis_items) == 1  # persisted entry untouched

    async def test_try_commit_dequeue_true_then_false_after_clear(
        self, gq, mock_author
    ):
        await gq.put([_qobj(1, mock_author)])
        assert await gq.try_commit_dequeue() is True
        await gq.clear()
        assert await gq.try_commit_dequeue() is False
