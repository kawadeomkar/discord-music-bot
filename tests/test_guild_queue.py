"""Tests for src/guild_queue.py — the queue domain class.

Central property: after every operation the deque, the cursor and the Redis
mirror agree — I4, checked by _assert_mirror_matches. persisted=False items live
in memory only, by design, so the mirror is a subset rather than a copy."""

import redis.asyncio as aioredis
from dataclasses import replace
from typing import Any
import asyncio
import contextlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.guild_queue import (
    _to_entry,
    _LREM_MAX_ENTRIES,
    _LREM_MAX_SHARE,
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


async def _assert_mirror_matches(
    gq: GuildQueue, fake_redis: aioredis.Redis, store: GuildRedisStore
) -> None:
    """I4: the Redis list equals the persisted subset of the deque, IN ORDER.

    Contents, not counts. A length-only check passes an ordering inversion, which
    is exactly what a wrong LREM produces — and the next commit-time LPOP then
    retires the entry at the head rather than its own."""
    items = gq.display_items()
    assert gq.qsize() == len(items) - gq._cursor
    stored = await fake_redis.lrange(store.queue_key(), 0, -1)
    assert stored == [_to_entry(i).to_redis() for i in items if is_persisted(i)]


# ── is_persisted ──────────────────────────────────────────────────────────────


class TestIsPersisted:
    def test_queue_object_reflects_flag(self, mock_author: MagicMock) -> None:
        assert is_persisted(_qobj(1, mock_author)) is True
        assert is_persisted(_qobj(1, mock_author, persisted=False)) is False

    def test_ytsource_always_persisted(self) -> None:
        assert is_persisted(YTSource(ytsearch="artist song")) is True

    def test_none_is_persisted(self) -> None:
        # The default for "no item to ask", which is what the ordinary dequeue
        # paths want. NOT because a caller passing None always holds a persisted
        # claim — the playback loop's prefetched branch passes None and can hold
        # an unpersisted one, which is why redis_pop_for takes an explicit
        # override rather than trusting this.
        assert is_persisted(None) is True


# ── put ───────────────────────────────────────────────────────────────────────


class TestPut:
    async def test_an_unpersisted_item_never_reaches_the_mirror(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """persisted=False means "no entry of mine is on that list" and redis_pop_for
        honours it at the dequeue, so writing one here would leave an entry nothing
        ever LPOPs — the mirror one ahead of memory forever."""
        await gq.put([_qobj(1, mock_author, persisted=False)])

        assert gq.display_size() == 1
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == []

    async def test_a_mixed_batch_mirrors_only_the_persisted_half(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        persisted = _qobj(2, mock_author)
        await gq.put([_qobj(1, mock_author, persisted=False), persisted])

        assert gq.display_size() == 2
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == [
            SongQueueEntry.from_queue_object(persisted).to_redis()
        ]

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
        await _assert_mirror_matches(gq, fake_redis, store)

    async def test_an_unpersisted_item_stays_out_of_the_mirror(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """persisted=False means the entry was already retired — the crash-recovered
        head, whose LPOP committed in the run that crashed. RPUSHing one writes an
        entry no dequeue will ever LPOP, leaving the mirror a permanent entry ahead
        of the deque."""
        keep = _qobj(1, mock_author)
        recovered = QueueObject(
            "https://yt.com/v=crashed", "Crashed", mock_author, persisted=False
        )
        await gq.put([keep, recovered])

        assert gq.display_items() == [keep, recovered]
        redis_items = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert redis_items == [SongQueueEntry.from_queue_object(keep).to_redis()]

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
        await _assert_mirror_matches(gq, fake_redis, store)

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


class TestRemoveByOriginSurvivesARestart:
    """A Spotify album expands to N unresolved searches whose `ytsearch` is a
    title this code generated, so the album link survives only on the wire: every
    leg of enqueue → Redis → restart → rehydrate → match has to carry it. Driven
    through real store calls, since each leg passing alone would not prove the
    removal still matches after the trip."""

    async def test_an_album_link_still_removes_its_tracks_after_a_restart(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        mock_guild: MagicMock,
        mock_author: MagicMock,
    ) -> None:
        mock_guild.get_member.return_value = mock_author
        album = "https://open.spotify.com/album/abc123"
        await gq.put(
            [
                YTSource(ytsearch=f"ytsearch:Track {n}", process=True, user_input=album)
                for n in range(1, 4)
            ]
            + [_qobj(99, mock_author)],  # queued separately, must survive
            batch=True,
        )

        # The restart: nothing in memory, everything from the mirror, read the
        # way _restore_state reads it.
        restored = GuildQueue(mock_guild, store)
        snapshot = await store.get_playback_snapshot()
        assert snapshot is not None
        assert await restored.restore_entries(snapshot.queue) == 4

        outcome = await restored.remove(remove_matcher(album))

        assert outcome.positions == [1, 2, 3]
        assert outcome.mode is RemoveMode.ORIGIN
        survivors = restored.display_items()
        assert [queue_object(i).webpage_url for i in survivors] == [
            "https://yt.com/v=99"
        ]


class TestSmallSurfacesThatNothingElseCovers:
    """Four behaviours, each one line of production code that fails silently."""

    async def test_get_nowait_raises_queue_empty_on_a_drained_queue(
        self, gq_no_redis: GuildQueue
    ) -> None:
        """The prefetch catches asyncio.QueueEmpty specifically. Raise anything
        else — an IndexError from walking off the deque — and it escapes into the
        background task instead of returning None."""
        with pytest.raises(asyncio.QueueEmpty):
            gq_no_redis.get_nowait()

    async def test_get_nowait_raises_when_everything_is_claimed(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """Not just an empty deque: a queue whose every item is in flight has
        nothing left to hand out either."""
        await gq_no_redis.put([_qobj(1, mock_author)])
        await gq_no_redis.get()

        with pytest.raises(asyncio.QueueEmpty):
            gq_no_redis.get_nowait()

    async def test_requeue_front_with_nothing_claimed_leaves_the_tail_alone(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The guard's BEHAVIOUR, not its spelling: a cursor driven negative
        writes the substitute to _items[-1], the TAIL, losing what was there."""
        items = [_qobj(1, mock_author), _qobj(2, mock_author)]
        await gq_no_redis.put(items)

        gq_no_redis.requeue_front(_qobj(99, mock_author))

        assert gq_no_redis._cursor == 0
        assert gq_no_redis.display_items() == items  # tail intact

    async def test_put_front_never_mirrors_an_unpersisted_item(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """A crash-recovered item is on no Redis list, so writing one there gives
        it an entry its dequeue will never LPOP — the mirror ends a permanent
        entry ahead."""
        await gq.put_front([_qobj(1, mock_author, persisted=False)])

        assert gq.display_size() == 1
        assert await fake_redis.lrange(store.queue_key(), 0, -1) == []

    async def test_has_resume_tail_ignores_a_plain_copy_of_the_same_url(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """It answers about resume TAILS. Drop the is_resume test and an ordinary
        second copy of the live song's URL stops the live song being counted,
        which writes a queue_position one too low to play_history."""
        url = "https://yt.com/v=same"
        await gq_no_redis.put([QueueObject(url, "Plain", mock_author)])
        assert gq_no_redis.has_resume_tail(url) is False

        await gq_no_redis.put([QueueObject(url, "Tail", mock_author, is_resume=True)])
        assert gq_no_redis.has_resume_tail(url) is True

    async def test_a_large_removal_from_a_short_queue_rebuilds(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """The ratio gate. A rebuild scales with what SURVIVES, so dropping most
        of a short queue is cheaper to rewrite than to LREM one entry at a time."""
        album = "https://open.spotify.com/playlist/big"
        await gq.put(
            [
                QueueObject(
                    f"https://yt.com/v={n}", f"T{n}", mock_author, user_input=album
                )
                for n in range(8)
            ]
            + [_qobj(99, mock_author)]
        )
        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries"):
            original = getattr(store, name)

            def spy(*a: Any, _n: str = name, _o: Any = original, **k: Any) -> Any:
                calls.append(_n)
                return _o(*a, **k)

            setattr(store, name, spy)

        outcome = await gq.remove(remove_matcher(album))

        assert len(outcome.positions) == 8  # 8 dropped, 1 survivor
        assert calls == ["rebuild_queue"]


class TestReleaseVsFinishFailedDequeue:
    """The distinction the playback loop's error handler turns on. `release()`
    settles the claim in memory alone; only `finish_failed_dequeue` also mirrors
    the dequeue to Redis. A handler using the former drops the item from memory
    and leaves its entry for the next LPOP to retire in place of its own."""

    async def test_release_leaves_the_mirror_alone(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.get()

        assert gq.try_release() is True

        assert gq.display_size() == 1
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 2

    async def test_finish_failed_dequeue_settles_both(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        first = _qobj(1, mock_author)
        await gq.put([first, _qobj(2, mock_author)])
        await gq.get()

        await gq.finish_failed_dequeue(first, context="test")

        assert gq.display_size() == 1
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 1

    async def test_the_settle_and_the_pop_share_one_mutex_hold(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """Both legs under ONE hold, or a bulk mutation lands between them and
        rebuilds the mirror — and the LPOP then retires the head of the NEW list.
        The mutex is what makes the pair atomic; nothing else does."""
        held: list[bool] = []
        original = store.pop_queue

        async def spy() -> None:
            held.append(gq._mutex.locked())
            await original()

        first = _qobj(1, mock_author)
        await gq.put([first, _qobj(2, mock_author)])
        await gq.get()

        with patch.object(store, "pop_queue", spy):
            await gq.finish_failed_dequeue(first, context="test")

        assert held == [True]

    async def test_a_cleared_claim_does_not_pop_the_song_that_replaced_it(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """-clear during a resolve leaves nothing claimed, and the queue has since
        been refilled. The settle correctly no-ops; an LPOP alongside it would
        retire the NEW head, which this dequeue never owned — losing it from Redis
        alone, so it survives in memory until a restart drops it."""
        claimed = _qobj(1, mock_author)
        await gq.put([claimed])
        await gq.get()

        await gq.clear()
        replacement = _qobj(2, mock_author)
        await gq.put([replacement])

        await gq.finish_failed_dequeue(claimed, context="test")

        assert gq.display_items() == [replacement]
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 1


def test_the_lrem_cap_stays_under_the_measured_crossover() -> None:
    """The value, not just the mechanism: every other test here sizes its input
    from the constant, so they move with it and cannot notice it being raised.
    18 is the low end of the measured crossover (see
    docs/ARCHITECTURE.md#queue-operations); past it the pipelined LREMs cost more
    than the rebuild they replace while holding one MULTI/EXEC, which stalls every
    guild rather than the one running -remove."""
    assert _LREM_MAX_ENTRIES <= 18


async def test_a_rebuilding_removal_serializes_only_the_survivors(
    gq: GuildQueue, mock_author: MagicMock
) -> None:
    """Nothing is built for the LREM shortcut until the shortcut is taken.

    The gate needs only a count, so a removal too big for it — one -remove of a
    collection link is routinely hundreds — serializes nothing but the survivors
    the rebuild writes. Building 500 entries measured 2.5ms on the single event
    loop."""
    collection = "https://open.spotify.com/playlist/big"
    drops = [
        QueueObject(
            f"https://yt.com/v=drop{n}", f"Drop {n}", mock_author, user_input=collection
        )
        for n in range(_LREM_MAX_ENTRIES * 4)  # far past the count cap
    ]
    keeps = [_qobj(1000 + n, mock_author) for n in range(3)]
    await gq.put([*keeps, *drops])

    import src.guild_queue as gq_module

    real = gq_module._to_entry
    built: list[Any] = []

    def counting(item: Any) -> Any:
        built.append(item)
        return real(item)

    with patch.object(gq_module, "_to_entry", counting):
        outcome = await gq.remove(remove_matcher(collection))

    assert len(outcome.removed) == len(drops)
    assert len(built) == len(keeps), (
        f"serialized {len(built)} entries to rebuild {len(keeps)} survivors"
    )


def test_the_lrem_share_keeps_shallow_queues_on_the_rebuild() -> None:
    """The sibling constant, pinned in the other direction.

    `_LREM_MAX_SHARE` keeps a shallow queue rebuilding: below roughly 80 survivors
    a full rewrite is under a millisecond, so the shortcut has nothing to win.
    Lowering it inverts that policy — at 1, a 3-song queue removing 1 takes the
    LREM path — and the tests around it size their input from the constant. A
    floor, since raising it only sends more removals to the rebuild."""
    assert _LREM_MAX_SHARE >= 4


class TestLremFallsBackWhenItCannotBeTrusted:
    """LREM matches on exact serialized bytes, which the rest of the codebase does
    not promise: a resume tail gaining np_* ids, an enriched duration and a
    substituted requester all mutate a queued object after its mirror entry was
    written, so the recomputed blob misses and the mirror leads memory forever."""

    async def test_a_mutated_item_falls_back_to_a_rebuild(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """The C2 case, as production reaches it: musicplayer stamps np_message_id
        onto a queued resume tail in memory only."""
        tail = _qobj(1, mock_author)
        keep = _qobj(2, mock_author)
        await gq.put([tail, keep])
        tail.np_message_id = 1234  # mirrored entry now says 0

        outcome = await gq.remove(remove_matcher(tail.webpage_url))

        assert outcome.positions == [1]
        # The mirror agrees with memory — the LREM missed and the rebuild ran.
        stored = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert stored == [SongQueueEntry.from_queue_object(keep).to_redis()]

    async def test_a_claimed_items_twin_is_never_lremed(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """LREM takes the HEAD-most equal element, so with a byte-identical twin it
        would eat the entry awaiting a commit-time LPOP. The queue is long enough
        that both other gates pass and `_claimed_blobs` is the clause deciding.

        Asserted on the store CALLS: an LREM here removes one entry and reports one
        removed, so the short-count fallback cannot catch it and the divergence
        only surfaces at the next commit-time LPOP."""
        first, second = _qobj(1, mock_author), _qobj(1, mock_author)
        middle = [_qobj(n, mock_author) for n in range(2, 7)]
        await gq.put([first, *middle, second])
        await gq.get()  # `first` is claimed and un-removable
        assert SongQueueEntry.from_queue_object(first).to_redis() == (
            SongQueueEntry.from_queue_object(second).to_redis()
        )
        # Both gates pass, so nothing but _claimed_blobs can refuse the shortcut.
        assert 1 <= _LREM_MAX_ENTRIES and 1 * _LREM_MAX_SHARE <= len(middle) + 1

        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries", "delete_queue"):
            original = getattr(store, name)

            def spy(*a: Any, _n: str = name, _o: Any = original, **k: Any) -> Any:
                calls.append(_n)
                return _o(*a, **k)

            setattr(store, name, spy)

        outcome = await gq.remove(remove_matcher(second.webpage_url))

        assert calls == ["rebuild_queue"], "the LREM would have eaten the claimed twin"
        assert outcome.positions == [7]  # the pending twin, not the claimed one
        stored = await fake_redis.lrange(store.queue_key(), 0, -1)
        assert stored == [
            SongQueueEntry.from_queue_object(item).to_redis()
            for item in [first, *middle]
        ]

    async def test_a_clean_removal_still_takes_the_lrem_path(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """The fallback must not swallow the optimisation it guards. Enough
        items that the ratio gate is satisfied: one dropped, the rest survive."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 8)])
        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries", "delete_queue"):
            original = getattr(store, name)

            def spy(*a: Any, _n: str = name, _o: Any = original, **k: Any) -> Any:
                calls.append(_n)
                return _o(*a, **k)

            setattr(store, name, spy)

        await gq.remove(remove_matcher("https://yt.com/v=2"))

        assert calls == ["remove_queue_entries"]

    async def test_the_absolute_cap_alone_can_force_the_rebuild(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """The count cap keeps one guild's -remove from stalling the whole server:
        the LREMs run inside one MULTI/EXEC, and past this many they cost more than
        the rebuild they replace. Sized so the RATIO gate passes and the cap is the
        only clause that can refuse."""
        collection = "https://open.spotify.com/playlist/cap"
        drops = [
            QueueObject(
                f"https://yt.com/v=drop{n}",
                f"Drop {n}",
                mock_author,
                user_input=collection,
            )
            for n in range(_LREM_MAX_ENTRIES + 1)
        ]
        # Sized off the drop count, not the cap: the ratio gate is
        # `len(dropped) * _LREM_MAX_SHARE <= survivors`, so the survivors have to
        # cover the drops that exist, or that gate refuses too and the assertion
        # below passes for the wrong clause.
        keeps = [
            _qobj(1000 + n, mock_author) for n in range(len(drops) * _LREM_MAX_SHARE)
        ]
        await gq.put([*keeps, *drops])
        # The ratio gate is satisfied, so the cap is the only clause left to
        # fail. The production expression, character for character.
        assert len(drops) * _LREM_MAX_SHARE <= len(keeps)
        assert len(drops) > _LREM_MAX_ENTRIES

        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries", "delete_queue"):
            original = getattr(store, name)

            def spy(*a: Any, _n: str = name, _o: Any = original, **k: Any) -> Any:
                calls.append(_n)
                return _o(*a, **k)

            setattr(store, name, spy)

        outcome = await gq.remove(remove_matcher(collection))

        assert len(outcome.removed) == len(drops)
        assert calls == ["rebuild_queue"], (
            f"{len(drops)} LREMs were admitted; the cap is meant to stop that"
        )

    async def test_a_redis_failure_during_lrem_rebuilds(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """@_guild_op turns a Redis error into 0 removed, which is short of what was
        asked for — so it lands on the rebuild rather than reporting success."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 8)])
        calls: list[str] = []

        async def boom(*_a: Any, **_k: Any) -> int:
            calls.append("remove_queue_entries")
            return 0

        original_rebuild = store.rebuild_queue

        async def spy_rebuild(*a: Any, **k: Any) -> bool:
            calls.append("rebuild_queue")
            return await original_rebuild(*a, **k)

        store.remove_queue_entries = boom
        store.rebuild_queue = spy_rebuild

        await gq.remove(remove_matcher("https://yt.com/v=2"))

        assert calls == ["remove_queue_entries", "rebuild_queue"]

    async def test_the_rebuild_path_does_not_serialize_what_it_drops(
        self, gq: GuildQueue, mock_author: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A removal too big for the shortcut must not pay to encode the entries it
        is about to throw away: the blobs serve the _claimed_blobs guard alone, so
        they belong inside the count/share gate. A collection link routinely drops
        hundreds, all on the single event loop."""
        collection = "https://open.spotify.com/playlist/hoist"
        drops = [
            QueueObject(
                f"https://yt.com/v=drop{n}",
                f"Drop {n}",
                mock_author,
                user_input=collection,
            )
            for n in range(_LREM_MAX_ENTRIES + 1)  # one past the cap: rebuild path
        ]
        keeps = [_qobj(2000 + n, mock_author) for n in range(4)]
        await gq.put([*keeps, *drops])

        encoded: list[str] = []
        original = SongQueueEntry.to_redis

        def counting(self: SongQueueEntry) -> bytes:
            encoded.append(self.webpage_url)
            return original(self)

        monkeypatch.setattr(SongQueueEntry, "to_redis", counting)
        outcome = await gq.remove(remove_matcher(collection))

        assert len(outcome.removed) == len(drops)
        # Only the survivors are written, and each exactly once — the rebuild's
        # own RPUSH encoding and nothing else.
        assert sorted(encoded) == sorted(k.webpage_url for k in keeps), (
            f"{len(encoded)} entries encoded for a rebuild of {len(keeps)}"
        )


class TestClearWhileAClaimIsOutstanding:
    """`clear()` resets the cursor as well as the deque, and the reset is not
    bookkeeping — it is what keeps the two consistent when the loop is mid-song.

    Without it `_cursor` outlives the items it indexed: `qsize()` goes NEGATIVE,
    `empty()` lies, and the next `try_release()` pops an empty deque."""

    async def test_the_cursor_is_reset_with_the_items(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq_no_redis.get()  # the loop is mid-song
        assert gq_no_redis._cursor == 1

        await gq_no_redis.clear()

        assert gq_no_redis._cursor == 0

    async def test_the_counters_stay_sane_after_clearing_mid_song(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq_no_redis.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq_no_redis.get()

        await gq_no_redis.clear()

        assert gq_no_redis.qsize() == 0  # not -1
        assert gq_no_redis.display_size() == 0
        assert gq_no_redis.empty() is True

    async def test_the_loops_failure_path_no_ops_instead_of_raising(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The loop claims, a -clear lands, and the loop's failure path releases
        into an empty queue. The guard makes that a no-op."""
        await gq_no_redis.put([_qobj(1, mock_author)])
        await gq_no_redis.get()
        await gq_no_redis.clear()

        assert gq_no_redis.try_release() is False

    async def test_a_refused_commit_leaves_nothing_to_settle(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """And the generation check gets there first: the commit refuses across a
        clear(), so the guard is the second line of defence, not the only one."""
        await gq.put([_qobj(1, mock_author)])
        await gq.get()
        generation = gq.generation

        await gq.clear()

        async with gq.commit_dequeue(generation) as committed:
            assert committed is False
        assert gq._cursor == 0


class TestCursorAndWakeDiscipline:
    """Structural rules the type checker cannot express, asserted against the
    source. Both failures are silent at runtime, which is why they are pinned
    here rather than left to review."""

    @staticmethod
    def _source() -> list[str]:
        """Module source with COMMENTS STRIPPED: the guards these tests match on
        are also described in the prose above them, so an unstripped scan could be
        satisfied by a comment alone."""
        import inspect
        import re as _re

        import src.guild_queue as module

        return [
            _re.sub(r"#.*$", "", ln) for ln in inspect.getsource(module).split("\n")
        ]

    def test_only_sync_wake_writes_the_event(self) -> None:
        """`_sync_wake()` derives `_wake` from `_cursor` and `len(_items)`. A
        hand-written set does not degrade when wrong: Event.wait() returns without
        yielding when already set, so get()'s wait loses its suspension point and
        the whole event loop stops."""
        lines = self._source()
        writers = [
            (i + 1, ln.strip())
            for i, ln in enumerate(lines)
            if "_wake.set()" in ln or "_wake.clear()" in ln
        ]
        assert len(writers) == 2, writers

        start = next(i for i, ln in enumerate(lines) if "def _sync_wake" in ln)
        end = next(
            i
            for i, ln in enumerate(lines[start + 1 :], start + 1)
            if ln.startswith("    def ")
        )
        assert all(start < line <= end for line, _ in writers), writers

    def test_every_cursor_decrement_is_guarded(self) -> None:
        """Unguarded, `_cursor -= 1` goes negative (I1) and the write that follows
        lands at `_items[-1]` — the TAIL — instead of the head."""
        lines = self._source()
        for i, ln in enumerate(lines):
            if "_cursor -= 1" not in ln:
                continue
            preceding = "\n".join(lines[max(0, i - 4) : i])
            assert (
                "if self._cursor == 0" in preceding
                or "if self._cursor > 0" in preceding
            ), f"unguarded cursor decrement at line {i + 1}: {ln.strip()}"

    def test_only_the_claim_paths_advance_the_cursor(self) -> None:
        """`_cursor += 1` is a claim. Anywhere else it would hand out an item the
        queue never gave, which reads as a silently skipped song."""
        lines = self._source()
        advances = [i for i, ln in enumerate(lines) if "_cursor += 1" in ln]
        owners = set()
        for i in advances:
            owners.add(
                next(
                    lines[j].strip()
                    for j in range(i, -1, -1)
                    if lines[j].startswith("    def ")
                    or lines[j].startswith("    async def ")
                )
            )
        assert owners == {
            "async def get(self) -> QueueItem:",
            "def get_nowait(self) -> QueueItem:",
        }


class TestRemovePositionContract:
    """`RemoveOutcome.positions` is 1-indexed as the queue embed numbers items, and
    the `-remove` reply prints it. The discriminating case is a removal with an
    item IN FLIGHT: it occupies a position and is never itself removable, so every
    plausible wrong formula gives a different answer for it."""

    async def test_the_second_of_three_with_one_in_flight_is_position_three(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        items = [_qobj(n, mock_author) for n in range(1, 4)]
        await gq_no_redis.put(items)
        await gq_no_redis.get()  # items[0] is now in flight, still position 1

        outcome = await gq_no_redis.remove(remove_matcher(items[2].webpage_url))

        assert outcome.positions == [3]
        assert outcome.removed == [items[2]]

    async def test_positions_match_what_the_queue_embed_shows(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The contract stated as the equality it is: position N is
        display_items()[N-1], in flight or not."""
        items = [_qobj(n, mock_author) for n in range(1, 6)]
        await gq_no_redis.put(items)
        await gq_no_redis.get()
        shown = gq_no_redis.display_items()

        outcome = await gq_no_redis.remove(remove_matcher(items[3].webpage_url))

        (pos,) = outcome.positions
        assert shown[pos - 1] is items[3]

    async def test_an_in_flight_item_is_never_removed_but_still_counts(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """It is committed to play — stopping it is -skip's job — but it holds
        position 1, which is what shifts every pending position up by one."""
        items = [_qobj(1, mock_author), _qobj(1, mock_author)]
        await gq_no_redis.put(items)
        await gq_no_redis.get()

        outcome = await gq_no_redis.remove(remove_matcher(items[0].webpage_url))

        assert outcome.positions == [2]  # not [1, 2]
        assert gq_no_redis.display_items() == [items[0]]

    async def test_nothing_matched_leaves_every_leg_untouched(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """No match means no mutation, so no mirror write either — a rebuild that
        changes nothing still costs a round trip."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 4)])
        before = gq.display_items()
        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries", "delete_queue"):
            setattr(store, name, lambda *a, _n=name, **k: calls.append(_n))

        outcome = await gq.remove(remove_matcher("https://yt.com/v=nope"))

        assert (outcome.positions, outcome.removed, outcome.mode) == ([], [], None)
        assert gq.display_items() == before
        assert calls == []


class TestTheTwoCounters:
    """`qsize()` and `display_size()` mean different sets and are one term apart
    over the same two fields — `len(_items) - _cursor` against `len(_items)` — so
    a swap compiles and type-checks. `display_size()` is the sole input to
    `play_history.queue_position` (MusicPlayer.enqueue_depth), so a swap writes a
    plausible wrong number to Postgres permanently, with no error to notice."""

    async def test_claim_outstanding_follows_the_cursor(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The third view of the same two fields, and the one that survives the
        handoff: between loop() taking a prefetch result out of its slot and
        committing it, this is the only thing that says a song is on its way —
        `_prefetch_task` is already None and `current_song` is not yet set."""
        a = _qobj(1, mock_author)
        await gq.put([a])
        assert gq.claim_outstanding() is False

        claimed = gq.get_nowait()
        assert gq.claim_outstanding() is True

        gq.requeue_front(claimed)
        assert gq.claim_outstanding() is False

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
        """Each half of the swap, separately: a claim moves qsize() and only
        qsize(), and the next test pins the other direction."""
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

        async with gq.commit_dequeue(gq.generation) as committed:
            assert committed is True

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
    """`get()` parks on `_wake`. Both I3 directions are tested with a BOUND rather
    than an assertion, because neither one raises: a stale-set Event spins `get()`
    with no suspension point until pytest's 120 s deadline, and a stale-clear one
    parks forever. A bounded wait_for turns both into a fast, legible failure."""

    async def test_a_removal_that_empties_the_queue_leaves_get_parked(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """`remove()` takes the only pending item and needs no cursor fix-up, which
        is where a per-method rule would say "nothing to do here". Leave `_wake` set
        and `get()`'s `while` loop has no suspension point — `Event.wait()` returns
        immediately when already set, so the whole event loop stops."""
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
        is parked, so the wake has to be real."""
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

    async def test_put_front_wakes_a_parked_getter(
        self, gq_no_redis: GuildQueue, mock_author: MagicMock
    ) -> None:
        """`-play --now` into an idle player: the loop is parked on an empty queue, and
        put_front has to wake it like put() and restore_entries() do. An unwoken
        getter does not fail — it waits forever on a queue that has an item in
        it."""
        getter = asyncio.create_task(gq_no_redis.get())
        await asyncio.sleep(0)
        assert not getter.done()

        qobj = _qobj(1, mock_author)
        assert await gq_no_redis.put_front([qobj]) == [qobj]

        assert await asyncio.wait_for(getter, 0.5) is qobj

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
        """A wakeup handed to a getter that is cancelled before claiming: the
        `while` re-test is what covers it, since the next getter re-reads the
        condition rather than trusting the wakeup."""
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
        end of the deque — with an `if` it raises IndexError instead of parking."""
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


# ── put_front (interjection) ─────────────────────────────────────────────────


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
        await _assert_mirror_matches(gq, fake_redis, store)

    async def test_in_flight_head_stays_ahead_and_redis_rebuilt(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """A dequeued-but-uncommitted head (completed prefetch) must keep its
        place ahead of the inserted items on display and Redis — its
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

    async def test_claims_balance_and_over_release_is_refused(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.put_front([_qobj(2, mock_author)])

        while gq.qsize():
            gq.get_nowait()

        # Two claims outstanding, and settling exactly two settles them all.
        assert gq._cursor == 2
        assert gq.try_release() is True
        assert gq.try_release() is True
        # A third release would drive _cursor negative AND eat a pending item,
        # so it no-ops and says so.
        assert gq.try_release() is False
        assert gq._cursor == 0

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
        # Claim one FIRST. Without this the cursor is already 0 and the assertion
        # below holds whatever clear() does to it — which is what made the older
        # version of this test pass against a clear() that reset nothing.
        await gq.get()
        cleared = await gq.clear()
        assert cleared == items  # the claimed prefix comes back too
        assert gq.qsize() == 0
        assert gq._cursor == 0
        assert gq.display_items() == []
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_bumps_the_generation_so_an_in_flight_dequeue_is_refused(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """A captured generation names the queue an item came from, so a claim
        taken before a clear() is refused even when a refill has been claimed
        since and the cursor alone would let the commit through.

        The refill is claimed on purpose: with nothing claimed the cursor reset
        refuses by itself, and this test would pass with the generation check
        deleted. A read-and-reset flag beside it cannot do this job: a consumer
        that reads it before the clear lands reads it a whole song late."""
        await gq.put([_qobj(1, mock_author)])
        captured = gq.generation
        await gq.get()
        await gq.clear()
        assert gq.generation != captured
        # -play refills and a prefetch claims the refill before the stale commit.
        await gq.put([_qobj(2, mock_author)])
        fresh = gq.generation
        await gq.get()
        assert gq._cursor == 1
        async with gq.commit_dequeue(captured) as committed:
            assert committed is False
        assert gq._cursor == 1  # the refused commit released nothing
        # The claim taken AFTER the clear carries the new value and commits.
        async with gq.commit_dequeue(fresh) as committed:
            assert committed is True
        assert gq._cursor == 0

    async def test_clear_leaves_nothing_pending_to_wake_on(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """I3 after a clear(): _wake is set iff something is pending, and nothing
        is. Left set, get()'s wait loop returns without yielding, and the event
        loop stops ticking — so the flag is asserted first, since a bounded get()
        under that defect never reaches its deadline."""
        await gq.put([_qobj(1, mock_author)])
        assert gq._wake.is_set()
        await gq.clear()
        assert not gq._wake.is_set()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await gq.get()

    async def test_clear_settles_an_outstanding_claim(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """clear() resets the cursor alongside the deque, which is what settles a
        claim the loop is still holding — the loop's commit then refuses on the
        generation and releases nothing."""
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.get()
        assert gq._cursor == 1

        await gq.clear()

        assert gq._cursor == 0  # the outstanding claim went with the deque

    async def test_mixed_item_types_all_come_back(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """clear() returns BOTH kinds, and its caller depends on that: the loop
        flushes the returned items to history (_flush_played) and renders their
        names, so an unresolved Spotify-playlist track dropped here loses a
        play_history row with no error."""
        song = _qobj(1, mock_author)
        search = YTSource(ytsearch="ytsearch:Artist - Song", process=True)
        await gq.put([song, search])

        cleared = await gq.clear()

        assert cleared == [song, search]
        assert [type(i) for i in cleared] == [QueueObject, YTSource]

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

    async def test_a_claimed_head_counts_toward_the_threshold(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """-queue and -debug both render display_size(), so counting pending alone
        refuses "at least 4" to a user looking at exactly four songs — which is
        every mid-song -shuffle, since the loop or a prefetch holds one claim."""
        items = [_qobj(n, mock_author) for n in range(1, 5)]
        await gq.put(items)
        await gq.get()

        assert gq.display_size() == 4
        assert gq.qsize() == 3
        assert await gq.shuffle() is ShuffleOutcome.SHUFFLED

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
        await _assert_mirror_matches(gq, fake_redis, store)

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

    async def test_shuffle_replaces_rather_than_appends(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(n, mock_author) for n in range(1, 6)])
        await gq.shuffle()
        # The rebuild replaces, never appends: five items still, not ten, and
        # nothing was claimed along the way.
        assert gq.display_size() == 5
        assert gq.qsize() == 5
        assert gq._cursor == 0

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

    def test_an_empty_needle_matches_nothing(self) -> None:
        """An unresolved search carries url=None, which the resolved leg reads as
        "" — so an empty needle compares equal to it and `-remove` takes out every
        lazily-queued Spotify track in the guild."""
        match = remove_matcher("")
        assert match(YTSource(ytsearch="ytsearch:a song")) is None
        assert match(self._song("https://yt.com/v=1", "")) is None
        assert match(self._song("", None)) is None

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


class TestAngleBracketedLinks:
    """Discord wraps a link in <> when a user suppresses its embed, so
    `-remove <https://youtu.be/x>` is an ordinary thing to paste back. Both legs
    have to see the stripping: an entry queued through a playlist expansion
    carries the collection link as its origin, not the track URL."""

    def test_the_resolved_leg_sees_through_the_brackets(
        self, mock_author: MagicMock
    ) -> None:
        item = _qobj(1, mock_author)  # user_input is None: origin cannot match
        assert item.user_input is None
        match = remove_matcher(f"<{item.webpage_url}>")
        assert match(item) is RemoveMode.RESOLVED

    def test_a_bare_link_still_matches(self, mock_author: MagicMock) -> None:
        item = _qobj(1, mock_author)
        assert remove_matcher(item.webpage_url)(item) is RemoveMode.RESOLVED


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
        await _assert_mirror_matches(gq, fake_redis, store)

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
    """How deep the interjection stack is: the run of parked plays behind the head.
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
        """The interjected song is not always at display index 0: put_front keeps a
        dequeued-but-uncommitted item ahead of what it inserts, so the run starts
        after the claimed prefix rather than at a fixed index."""
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
        ever interjected."""
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

    async def test_interjection_flags_rehydrate(
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

    async def test_origin_rehydrates_on_the_song_branch(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        """The hop a plain `-play <search text>` takes. The restart end-to-end
        covers only the search branch, so dropping this carry left `-remove <search
        text>` broken after a restart with nothing red."""
        mock_guild.get_member.return_value = mock_author
        typed = "never gonna give you up"
        entry = SongQueueEntry(
            webpage_url="https://yt.com/v=1",
            title="Song 1",
            requester_id=mock_author.id,
            user_input=typed,
        )
        assert await gq.restore_entries([entry]) == 1

        (restored,) = gq.display_items()
        assert isinstance(restored, QueueObject)
        assert restored.user_input == typed
        assert remove_matcher(typed)(restored) is RemoveMode.ORIGIN

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
        """_rehydrate is the entry → item hop for the whole restored queue, and the
        stamps must survive it: they are carried and never restamped, so a
        crash-recovered song archives the position it was originally queued at."""
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

    async def test_it_goes_to_the_front_even_with_a_queue_behind_it(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        """ "At the front of the line" is the contract, and its one caller runs on an
        empty deque, where an append satisfies it by accident. The crashed song was
        PLAYING, so behind the restored queue it would return after songs queued
        while it was mid-play."""
        mock_guild.get_member = MagicMock(return_value=mock_author)
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])

        assert await gq.restore_crashed(
            self._crashed_entry(mock_author.id), requester_fallback=mock_guild.me
        )

        assert [queue_object(i).title for i in gq.display_items()] == [
            "Crashed",
            "Song 1",
            "Song 2",
        ]

    async def test_it_goes_ahead_of_pending_but_behind_a_claim(
        self, gq: GuildQueue, mock_guild: MagicMock, mock_author: MagicMock
    ) -> None:
        """Front means the PENDING front. Ahead of a claimed item it would break the
        claimed-prefix invariant Redis's LPOP retirement depends on."""
        mock_guild.get_member = MagicMock(return_value=mock_author)
        await gq.put([_qobj(1, mock_author), _qobj(2, mock_author)])
        await gq.get()  # Song 1 claimed

        assert await gq.restore_crashed(
            self._crashed_entry(mock_author.id), requester_fallback=mock_guild.me
        )

        assert [queue_object(i).title for i in gq.display_items()] == [
            "Song 1",
            "Crashed",
            "Song 2",
        ]
        assert gq._cursor == 1

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

    async def test_explicit_persisted_false_beats_the_none_default(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """item=None defaults to popping; the override is the only way a caller
        whose claim is not a QueueItem can say otherwise. The playback loop's
        prefetched branch is that caller."""
        await gq.put([_qobj(1, mock_author)])
        await gq.redis_pop_for(None, persisted=False)
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 1

        # And the default still pops, so the override is doing the work rather
        # than the call shape.
        await gq.redis_pop_for(None)
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 0

    async def test_finish_failed_dequeue_forwards_the_override(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """The loop's outer handler settles a prefetched claim with item=None and
        persisted=False. Dropping the forward LPOPs a real entry for a head that
        never had one."""
        await gq.put([_qobj(1, mock_author)])  # the real, persisted entry
        seed_queue(gq, _qobj(99, mock_author, persisted=False))
        _ = await gq.get()  # the prefetch's claim, as the loop takes it
        await gq.finish_failed_dequeue(
            None, context="unhandled loop error", persisted=False
        )
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 1
        assert gq._cursor == 0  # settled in memory either way

    async def test_a_settled_claim_does_not_take_the_mirror_with_it(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """A clear() during a resolve retires the claim and resets the cursor, so
        the release below is a no-op — and the LPOP must be too. The mirror by then
        holds only what was queued AFTER the clear, and an LPOP is at-most-once, so
        it would take an unrelated song with no error and no second copy."""
        await gq.put([_qobj(1, mock_author)])
        claimed = await gq.get()  # the loop, mid-resolve

        await gq.clear()  # -clear lands
        await gq.put(  # -play refills, memory and mirror agree
            [_qobj(2, mock_author), _qobj(3, mock_author), _qobj(4, mock_author)]
        )

        await gq.finish_failed_dequeue(claimed, context="failed-song pop")

        # Contents, not counts: an LPOP here takes the HEAD, so the survivors are
        # still a valid-looking list one song short.
        assert len(gq.display_items()) == 3
        await _assert_mirror_matches(gq, fake_redis, store)

    async def test_finishing_a_dequeue_with_nothing_claimed_says_so(
        self,
        gq: GuildQueue,
        mock_author: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The settle is gated on the claim (asserted above); the warning is the
        only trace that a caller tried to retire a claim a clear() already took,
        and it has to name the caller's context to be worth reading."""
        await gq.put([_qobj(1, mock_author)])
        claimed = await gq.get()
        await gq.clear()
        with caplog.at_level(logging.WARNING, logger="src.guild_queue"):
            await gq.finish_failed_dequeue(claimed, context="resolve failure")
        assert "resolve failure" in caplog.text
        assert "leaving the mirror alone" in caplog.text

    async def test_pop_display_head_pops(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        # Claim first, as every caller does — this runs only from
        # finish_failed_dequeue, which is reached after a get().
        await gq.get()
        assert gq.try_release() is True
        assert gq.display_items() == []

    async def test_try_pop_display_head(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        assert gq.try_release() is False
        await gq.put([_qobj(1, mock_author)])
        await gq.get()  # the claim this settles
        assert gq.try_release() is True
        assert gq.display_items() == []

    async def test_commit_dequeue_shares_the_bulk_mutation_lock(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """commit_dequeue() and the bulk ops really do serialize on one
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
        """One call settles a failed claim on both legs: the deque head
        popped, Redis LPOPed, the claim settled."""
        item = _qobj(1, mock_author)
        await gq.put([item])
        _ = await gq.get()  # the loop dequeued it
        await gq.finish_failed_dequeue(item)
        assert gq.display_items() == []
        assert await fake_redis.exists(store.queue_key()) == 0
        assert gq._cursor == 0  # every claim settled

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

    async def test_commit_dequeue_true_then_false_after_clear(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.get()  # the claim the first commit settles
        generation = gq.generation
        async with gq.commit_dequeue(generation) as committed:
            assert committed is True
        await gq.clear()
        async with gq.commit_dequeue(generation) as committed:
            assert committed is False


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
        await _assert_mirror_matches(gq, fake_redis, store)
        assert gq.get_nowait() is a

    async def test_the_claim_goes_back_with_the_item(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        a = _qobj(1, mock_author)
        await gq.put([a])

        gq.requeue_front(gq.get_nowait())

        # The claim went back with the item: it is pending again, and claimable.
        assert gq._cursor == 0
        assert gq.qsize() == 1
        assert gq.get_nowait() is a
        assert gq._cursor == 1  # and the new consumer holds it

    async def test_a_returned_claim_does_not_destroy_a_later_one(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """Two claims are live — the prefetch's, and the loop's, taken while the
        prefetch's cancel awaited. The returned item belongs in its own slot, index
        0; written at the cursor it overwrote the entry loop() had just claimed,
        which the mirror kept, and the commit pops both by position."""
        prefetched, claimed_by_loop, pending = (
            _qobj(n, mock_author) for n in (1, 2, 3)
        )
        await gq.put([prefetched, claimed_by_loop, pending])
        assert gq.get_nowait() is prefetched
        assert gq.get_nowait() is claimed_by_loop

        gq.requeue_front(prefetched)

        # Nothing destroyed, nothing duplicated, order untouched — and one claim
        # given back, so the newest entry is pending again.
        assert gq.display_items() == [prefetched, claimed_by_loop, pending]
        assert gq._cursor == 1
        assert gq.qsize() == 2
        await _assert_mirror_matches(gq, fake_redis, store)

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
        async with gq.commit_dequeue(gq.generation) as committed:
            assert committed is True
        await store.pop_queue()
        await _assert_mirror_matches(gq, fake_redis, store)
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


class TestStaleMirror:
    """The queue's record of a start transaction whose list leg did not land:
    the list is then one entry ahead of memory, and only a write that REPLACES
    it can repair that. note_mirror_write is the loop's report; a bulk rebuild
    clears the flag in passing; the LREM shortcut is refused over a stale list,
    because LREM keeps whatever it does not name."""

    def test_only_a_missed_retirement_makes_the_mirror_stale(
        self, gq: GuildQueue
    ) -> None:
        assert not gq.mirror_dirty
        gq.note_mirror_write(landed=False, retired=False)  # state-only write
        assert not gq.mirror_dirty
        gq.note_mirror_write(landed=False, retired=True)  # the LPOP did not land
        assert gq.mirror_dirty
        gq.note_mirror_write(landed=False, retired=False)  # still stale
        assert gq.mirror_dirty
        gq.note_mirror_write(landed=True, retired=True)  # a rebuild landed
        assert not gq.mirror_dirty

    def test_mirror_entries_are_what_a_rebuild_writes(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The persisted subset, claimed prefix included: a claimed entry is still
        on the list until its own LPOP, and a persisted=False head never was."""
        seed_queue(
            gq,
            _qobj(1, mock_author, persisted=False),
            _qobj(2, mock_author),
            _qobj(3, mock_author),
        )
        gq.get_nowait()  # claim the unpersisted head
        gq.get_nowait()  # and the first persisted entry
        blobs = [e.to_redis() for e in gq.mirror_entries()]
        assert [b"v=2" in b for b in blobs] == [True, False]
        assert [b"v=3" in b for b in blobs] == [False, True]

    async def test_a_bulk_rebuild_repairs_a_stale_mirror(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(n, mock_author) for n in range(1, 5)])
        await fake_redis.lpush(store.queue_key(), b"stale-head")  # a missed LPOP
        gq.note_mirror_write(landed=False, retired=True)

        await gq.shuffle()

        assert not gq.mirror_dirty
        await _assert_mirror_matches(gq, fake_redis, store)

    async def test_a_clear_repairs_a_stale_mirror(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        gq.note_mirror_write(landed=False, retired=True)
        await gq.clear()
        assert not gq.mirror_dirty
        assert await fake_redis.exists(store.queue_key()) == 0

    async def test_a_failed_rebuild_leaves_the_mirror_stale(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """@_guild_op turns a Redis failure into False; the flag must outlive it
        so the next song start still replaces the list rather than LPOPing it."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 5)])
        gq.note_mirror_write(landed=False, retired=True)

        async def _down(*_a: Any, **_k: Any) -> bool:
            return False

        with patch.object(store, "rebuild_queue", new=_down):
            await gq.shuffle()
        assert gq.mirror_dirty

    async def test_a_removal_over_a_stale_mirror_rebuilds_instead_of_lremming(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        """One LREM out of a list that is already one entry ahead leaves the
        stale head in place, so the shortcut's precondition — the survivors are
        exactly the list minus the removed — is false."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 21)])
        await fake_redis.lpush(store.queue_key(), b"stale-head")
        gq.note_mirror_write(landed=False, retired=True)
        calls: list[str] = []
        for name in ("rebuild_queue", "remove_queue_entries"):
            original = getattr(store, name)

            def spy(*args: Any, _n: str = name, _o: Any = original, **kw: Any) -> Any:
                calls.append(_n)
                return _o(*args, **kw)

            setattr(store, name, spy)

        await gq.remove(remove_matcher("https://yt.com/v=7"))

        assert calls == ["rebuild_queue"]
        assert not gq.mirror_dirty
        await _assert_mirror_matches(gq, fake_redis, store)


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
        """Small against the SURVIVORS, not in absolute terms — the gate is a
        ratio, because a rebuild's cost scales with what it rewrites."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 8)])
        calls = self._spy(store)

        await gq.remove(remove_matcher("https://yt.com/v=3"))

        assert calls == ["remove_queue_entries"]

    async def test_exactly_the_cap_still_takes_the_shortcut(
        self, gq: GuildQueue, store: GuildRedisStore, mock_author: MagicMock
    ) -> None:
        """The boundary the constant names. Tests either side of it leave `<=` and
        `<` indistinguishable, and this is a measured crossover a maintainer is
        expected to move — an off-by-one sends a qualifying removal to the rebuild
        and no test notices."""
        album = "https://open.spotify.com/album/exact"
        survivors = [_qobj(900 + n, mock_author) for n in range(_LREM_MAX_ENTRIES * 5)]
        await gq.put(
            [
                QueueObject(
                    f"https://yt.com/v={n}", f"T{n}", mock_author, user_input=album
                )
                for n in range(_LREM_MAX_ENTRIES)
            ]
            + survivors
        )
        calls = self._spy(store)

        outcome = await gq.remove(remove_matcher(album))

        assert len(outcome.positions) == _LREM_MAX_ENTRIES
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

    async def test_a_short_lrem_falls_through_to_the_rebuild(
        self,
        gq: GuildQueue,
        store: GuildRedisStore,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """The mirror can lose entries with nothing raising — an evicted key (it
        is TTL'd, so volatile-lru may take it), a swallowed write. Editing it in
        place would leave it short forever, since every later removal takes the
        same path; the rebuild restates the whole list from memory."""
        await gq.put([_qobj(n, mock_author) for n in range(1, 8)])
        await fake_redis.delete(store.queue_key())
        calls = self._spy(store)

        await gq.remove(remove_matcher("https://yt.com/v=3"))

        assert calls == ["remove_queue_entries", "rebuild_queue"]
        assert len(await fake_redis.lrange(store.queue_key(), 0, -1)) == 6

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

        async with gq.commit_dequeue(gq.generation) as committed:
            assert committed is True
        await store.pop_queue()
        await _assert_mirror_matches(gq, fake_redis, store)

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


# ── Generation counter (stream preemption) ────────────────────────────────────


class TestCommitDequeueHoldsTheMutexAcrossTheWrite:
    """The commit and the LPOP that retires its entry settle under one mutex hold.

    Without that, a put_front() reads `_cursor == 0` between them, takes the LPUSH
    branch, and prepends ahead of the entry the pending LPOP removes — so the LPOP
    eats the inserted song, and memory and Redis drift by one until the next
    rebuild."""

    async def test_a_put_front_cannot_land_between_the_commit_and_the_lpop(
        self,
        gq: GuildQueue,
        fake_redis: aioredis.Redis,
        store: GuildRedisStore,
        mock_author: MagicMock,
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.get()  # the loop claims the head and resolves it
        generation = gq.generation
        interjected = _qobj(2, mock_author)

        async with gq.commit_dequeue(generation) as committed:
            assert committed
            # -playnow, scheduled into the window the hold has to close.
            racer = asyncio.create_task(gq.put_front([interjected]))
            await asyncio.sleep(0)
            assert not racer.done(), (
                "put_front got in before the LPOP — the mutex is not held "
                "across the body"
            )
            await store.pop_queue()  # the start transaction's server-side LPOP

        await racer

        assert [queue_object(i).webpage_url for i in gq.display_items()] == [
            "https://yt.com/v=2"
        ]
        await _assert_mirror_matches(gq, fake_redis, store)

    async def test_a_refused_commit_still_releases_the_mutex(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        """The refusal path exits through the same `async with`, so a queue cleared
        mid-resolve leaves the mutex free — every later enqueue and the teardown's
        own writes park on it otherwise."""
        await gq.put([_qobj(1, mock_author)])
        await gq.get()
        stale = gq.generation
        await gq.clear()

        async with gq.commit_dequeue(stale) as committed:
            assert committed is False

        assert not gq._mutex.locked()
        assert await gq.put([_qobj(2, mock_author)]) is not None


class TestCommitDequeueUnwinds:
    """The body is no longer trivial — it carries vc.play(), an assert and the
    store dispatch — and a stranded asyncio.Lock has no timeout, so every later
    put/put_front/clear/shuffle/remove for that guild would park forever."""

    async def test_a_raising_body_releases_the_mutex(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.get()

        with pytest.raises(RuntimeError):
            async with gq.commit_dequeue(gq.generation) as committed:
                assert committed is True
                raise RuntimeError("vc.play blew up inside the hold")

        assert not gq._mutex.locked()
        await gq.put([_qobj(2, mock_author)])

    async def test_a_cancelled_body_releases_the_mutex(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        await gq.get()
        inside = asyncio.Event()

        async def _hold() -> None:
            async with gq.commit_dequeue(gq.generation):
                inside.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(_hold())
        await inside.wait()
        assert gq._mutex.locked()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert task.cancelled()
        assert not gq._mutex.locked()
        await gq.put([_qobj(2, mock_author)])


class TestGenerationCounter:
    async def test_starts_at_zero(self, gq: GuildQueue) -> None:
        assert gq.generation == 0

    async def test_clear_bumps_generation(
        self, gq: GuildQueue, mock_author: MagicMock
    ) -> None:
        await gq.put([_qobj(1, mock_author)])
        gen = gq.generation
        await gq.clear()
        assert gq.generation == gen + 1
