"""Tests for src/history_archive.py — the outbox drainer and the archive's
no-connection paths.

The asyncpg implementation is exercised against a real Postgres only by the
opt-in integration tier (docs/POSTGRES_HISTORY_PLAN.md §9); here the drainer
runs against fakeredis + an in-memory HistoryArchive fake, and
PostgresHistoryArchive is covered exactly as far as it can go without a
server (early-outs, row mapping, close-before-connect).
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collections.abc import Callable, Sequence
from typing import Any

from redis.asyncio import Redis

from src.guild_state import HistoryEntry
from src.history_archive import (
    HistoryArchive,
    HistoryOutboxDrainer,
    PostgresHistoryArchive,
    _entry_to_row,
    _row_to_entry,
)
from src.redis_client import HISTORY_OUTBOX_KEY, GuildRedisStore


def _entry(n: int, guild_id: int = 42) -> HistoryEntry:
    return HistoryEntry(
        guild_id=guild_id,
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=200,
        requester_id=n,
        requester_name=f"user{n}",
        played_at=1000.0 + n,
    )


class FakeArchive:
    """In-memory HistoryArchive: records insert batches, fails on demand."""

    def __init__(self) -> None:
        self.batches: list[list[HistoryEntry]] = []
        self.fail = False

    @property
    def inserted(self) -> list[HistoryEntry]:
        return [e for batch in self.batches for e in batch]

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        if self.fail:
            raise RuntimeError("pg down")
        self.batches.append(list(entries))

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        # Newest-first, matching the protocol + PostgresHistoryArchive.recent();
        # inserted is oldest-first, so reverse. (limit<=0 → empty, as the real one.)
        if limit <= 0:
            return []
        mine = [e for e in self.inserted if e.guild_id == guild_id]
        return list(reversed(mine))[:limit]


# Static conformance check: if the HistoryArchive protocol signatures drift,
# this assignment stops type-checking and the fake can't silently rot.
_: HistoryArchive = FakeArchive()


@pytest.fixture
def archive() -> FakeArchive:
    return FakeArchive()


@pytest.fixture
def drainer(fake_redis: Redis, archive: Any) -> HistoryOutboxDrainer:
    return HistoryOutboxDrainer(fake_redis, archive)


async def _push(fake_redis: Redis, *ns: int) -> None:
    store = GuildRedisStore(fake_redis, guild_id=42)
    for n in ns:
        await store.push_history(_entry(n))


async def _eventually(cond: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


class TestDrainOnce:
    async def test_moves_entries_oldest_first(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        await _push(fake_redis, 1, 2, 3)
        assert await drainer._drain_once() == 3
        assert archive.batches == [[_entry(1), _entry(2), _entry(3)]]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_display_list_untouched(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        store = GuildRedisStore(fake_redis, guild_id=42)
        await _push(fake_redis, 1)
        await drainer._drain_once()
        assert await store.get_history() == [_entry(1)]

    async def test_empty_outbox_is_noop(
        self, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        assert await drainer._drain_once() == 0
        assert archive.batches == []

    async def test_batch_capped(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        await _push(fake_redis, *range(drainer.BATCH_SIZE + 7))
        assert await drainer._drain_once() == drainer.BATCH_SIZE
        assert len(archive.inserted) == drainer.BATCH_SIZE
        # The oldest BATCH_SIZE went first; the newest 7 remain.
        assert archive.inserted[0] == _entry(0)
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 7

    async def test_corrupt_entry_retired_not_inserted(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Corrupt bytes must be consumed (or they'd wedge the queue head
        # forever) while the surviving entries still make it to the archive.
        await _push(fake_redis, 1)
        await fake_redis.lpush(HISTORY_OUTBOX_KEY, b"not json")  # newer than entry 1
        assert await drainer._drain_once() == 2
        assert archive.inserted == [_entry(1)]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_archive_failure_leaves_outbox_intact(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Retire happens strictly after a successful insert — a failed insert
        # must leave every entry in place for the retry.
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 2

    async def test_redelivery_after_failure(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        archive.fail = False
        assert await drainer._drain_once() == 2
        assert archive.inserted == [_entry(1), _entry(2)]


class TestDrainerLoop:
    async def test_notify_triggers_drain(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: archive.inserted == [_entry(1)])
            assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0
        finally:
            await drainer.stop()

    async def test_backlog_drained_across_batches_without_renotify(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # More than one batch waiting: the loop keeps draining until empty
        # instead of stalling one-batch-per-wakeup.
        await _push(fake_redis, *range(drainer.BATCH_SIZE + 5))
        drainer.start()
        try:
            drainer.notify()
            await _eventually(lambda: len(archive.inserted) == drainer.BATCH_SIZE + 5)
        finally:
            await drainer.stop()

    async def test_failure_backs_off_then_recovers(
        self,
        fake_redis: Redis,
        archive: Any,
        drainer: HistoryOutboxDrainer,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_START", 0.01)
        archive.fail = True
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: "outbox drain failed" in caplog.text)
            assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1  # nothing lost
            archive.fail = False
            await _eventually(lambda: archive.inserted == [_entry(1)])
        finally:
            await drainer.stop()

    async def test_depth_alarm_escalates_to_error(
        self,
        fake_redis: Redis,
        archive: Any,
        drainer: HistoryOutboxDrainer,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(HistoryOutboxDrainer, "DEPTH_ALARM", 2)
        await _push(fake_redis, 1, 2, 3)
        archive.fail = True
        await drainer._log_retry(RuntimeError("pg down"), backoff=1.0)
        assert any(r.levelname == "ERROR" for r in caplog.records)

    async def test_stop_makes_final_drain_attempt(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Entries pushed but never notify()ed (e.g. the notify was lost to a
        # crash) still ship on clean shutdown.
        drainer.start()
        await _push(fake_redis, 1)
        await drainer.stop()
        assert archive.inserted == [_entry(1)]

    async def test_stop_without_start_is_safe(
        self, drainer: HistoryOutboxDrainer
    ) -> None:
        await drainer.stop()

    async def test_stop_swallows_final_drain_failure(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Shutdown must never raise — undrained entries stay in the outbox
        # for the next start.
        await _push(fake_redis, 1)
        archive.fail = True
        drainer.start()
        await drainer.stop()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1


class TestRowMapping:
    def test_round_trip(self) -> None:
        entry = _entry(1, guild_id=222222222222222222)
        row = _entry_to_row(entry)
        keys = (
            "guild_id",
            "title",
            "webpage_url",
            "duration_secs",
            "played_secs",
            "requester_id",
            "requester_name",
            "thumbnail",
            "uploader",
            "played_at",
        )
        assert _row_to_entry(dict(zip(keys, row))) == entry

    def test_played_at_maps_to_utc_datetime(self) -> None:
        row = _entry_to_row(_entry(1))
        assert row[-1] == datetime.fromtimestamp(1001.0, tz=timezone.utc)

    def test_epoch_zero_unknown_sentinel_survives(self) -> None:
        # played_at 0.0 = "unknown" — carried into Postgres as to_timestamp(0),
        # not NULL (docs/POSTGRES_HISTORY_PLAN.md §4).
        entry = HistoryEntry(guild_id=1, title="x")
        row = _entry_to_row(entry)
        assert row[-1] == datetime.fromtimestamp(0, tz=timezone.utc)
        keys = (
            "guild_id",
            "title",
            "webpage_url",
            "duration_secs",
            "played_secs",
            "requester_id",
            "requester_name",
            "thumbnail",
            "uploader",
            "played_at",
        )
        assert _row_to_entry(dict(zip(keys, row))).played_at == 0.0


class TestPostgresArchiveWithoutServer:
    async def test_empty_insert_never_connects(self) -> None:
        # insert_batch([]) early-outs before _ensure() — a bogus DSN proves
        # no connection was attempted.
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive.insert_batch([])

    async def test_nonpositive_recent_never_connects(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        assert await archive.recent(42, 0) == []
        assert await archive.recent(42, -1) == []

    async def test_close_before_connect_is_safe(self) -> None:
        await PostgresHistoryArchive("postgresql://nope:1/nope").close()

    async def test_health_check_runs_select_1_on_the_pool(self) -> None:
        # Stubs the lazy pool so the query path is covered without a server; the
        # real thing is exercised by the pg tier (test_pg_integration.py).
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        conn = MagicMock()
        conn.execute = AsyncMock()
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        # return_value=None, not a bare AsyncMock: a truthy __aexit__ would
        # swallow any exception raised inside the block.
        acquire_cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire_cm)
        with patch.object(archive, "_ensure", AsyncMock(return_value=pool)):
            await archive.health_check()
        conn.execute.assert_awaited_once_with("SELECT 1")

    async def test_health_check_propagates_failure(self) -> None:
        # probe_postgres relies on this: it turns the exception into a red row,
        # so a swallowed error here would render an unreachable database green.
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        with patch.object(
            archive, "_ensure", AsyncMock(side_effect=OSError("unreachable"))
        ):
            with pytest.raises(OSError):
                await archive.health_check()


class TestPostgresArchiveClosedGuard:
    """close() nulls _pool without holding the init lock, so a later _ensure()
    would build a pool that nothing is left to close. Only reachable now that
    health_check() gives -ping a path into _ensure() that shutdown does not
    sequence — hence the guard rather than the ordering assumption alone.
    """

    async def test_ensure_refuses_after_close(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive.close()
        with pytest.raises(RuntimeError, match="closed"):
            await archive._ensure()

    async def test_health_check_after_close_fails_instead_of_reconnecting(
        self,
    ) -> None:
        # A -ping probe in flight while the bot shuts down: it must fail (the row
        # goes red on a bot that is going away anyway), never resurrect a pool.
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive.close()
        with patch("src.history_archive.asyncpg.create_pool", AsyncMock()) as create:
            with pytest.raises(RuntimeError, match="closed"):
                await archive.health_check()
        create.assert_not_awaited()

    async def test_insert_and_recent_also_refuse_after_close(self) -> None:
        # The guard lives in _ensure, so every connecting path inherits it —
        # asserted here so a future refactor that inlines _ensure cannot quietly
        # reopen the hole for the drainer's paths too.
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive.close()
        with pytest.raises(RuntimeError, match="closed"):
            await archive.insert_batch([_entry(1)])
        with pytest.raises(RuntimeError, match="closed"):
            await archive.recent(42, 10)

    async def test_close_is_idempotent(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive.close()
        await archive.close()


class TestDrainerSupervision:
    """The done-callback is the only thing that makes a drainer that dies
    outside its inner try loud instead of silent (a dead drainer lets the
    non-evictable outbox grow unbounded)."""

    async def test_logs_error_when_task_dies_unexpectedly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def boom() -> None:
            raise RuntimeError("kaboom")

        task = asyncio.create_task(boom())
        with pytest.raises(RuntimeError):
            await task
        HistoryOutboxDrainer._on_task_done(task)
        assert "died unexpectedly" in caplog.text
        assert "kaboom" in caplog.text
        assert any(r.levelname == "ERROR" for r in caplog.records)

    async def test_no_log_on_normal_cancellation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def forever() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(forever())
        await asyncio.sleep(0)  # let it start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        HistoryOutboxDrainer._on_task_done(task)
        assert "died unexpectedly" not in caplog.text
