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

import pytest

from src.db import Database
from src.guild_state import HistoryEntry
from src.history_archive import (
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

    def __init__(self):
        self.batches: list[list[HistoryEntry]] = []
        self.fail = False

    @property
    def inserted(self) -> list[HistoryEntry]:
        return [e for batch in self.batches for e in batch]

    async def insert_batch(self, entries):
        if self.fail:
            raise RuntimeError("pg down")
        self.batches.append(list(entries))

    async def recent(self, guild_id, limit):
        return [e for e in self.inserted if e.guild_id == guild_id][:limit]


@pytest.fixture
def archive():
    return FakeArchive()


@pytest.fixture
def drainer(fake_redis, archive):
    return HistoryOutboxDrainer(fake_redis, archive)


async def _push(fake_redis, *ns: int) -> None:
    store = GuildRedisStore(fake_redis, guild_id=42)
    for n in ns:
        await store.push_history(_entry(n), outbox=True)


async def _eventually(cond, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


class TestDrainOnce:
    async def test_moves_entries_oldest_first(self, fake_redis, archive, drainer):
        await _push(fake_redis, 1, 2, 3)
        assert await drainer._drain_once() == 3
        assert archive.batches == [[_entry(1), _entry(2), _entry(3)]]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_display_list_untouched(self, fake_redis, drainer):
        store = GuildRedisStore(fake_redis, guild_id=42)
        await _push(fake_redis, 1)
        await drainer._drain_once()
        assert await store.get_history() == [_entry(1)]

    async def test_empty_outbox_is_noop(self, archive, drainer):
        assert await drainer._drain_once() == 0
        assert archive.batches == []

    async def test_batch_capped(self, fake_redis, archive, drainer):
        await _push(fake_redis, *range(drainer.BATCH_SIZE + 7))
        assert await drainer._drain_once() == drainer.BATCH_SIZE
        assert len(archive.inserted) == drainer.BATCH_SIZE
        # The oldest BATCH_SIZE went first; the newest 7 remain.
        assert archive.inserted[0] == _entry(0)
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 7

    async def test_corrupt_entry_retired_not_inserted(
        self, fake_redis, archive, drainer
    ):
        # Corrupt bytes must be consumed (or they'd wedge the queue head
        # forever) while the surviving entries still make it to the archive.
        await _push(fake_redis, 1)
        await fake_redis.lpush(HISTORY_OUTBOX_KEY, b"not json")  # newer than entry 1
        assert await drainer._drain_once() == 2
        assert archive.inserted == [_entry(1)]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_archive_failure_leaves_outbox_intact(
        self, fake_redis, archive, drainer
    ):
        # Retire happens strictly after a successful insert — a failed insert
        # must leave every entry in place for the retry.
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 2

    async def test_redelivery_after_failure(self, fake_redis, archive, drainer):
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        archive.fail = False
        assert await drainer._drain_once() == 2
        assert archive.inserted == [_entry(1), _entry(2)]


class TestDrainerLoop:
    async def test_notify_triggers_drain(self, fake_redis, archive, drainer):
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: archive.inserted == [_entry(1)])
            assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0
        finally:
            await drainer.stop()

    async def test_backlog_drained_across_batches_without_renotify(
        self, fake_redis, archive, drainer
    ):
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
        self, fake_redis, archive, drainer, monkeypatch, caplog
    ):
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
        self, fake_redis, archive, drainer, monkeypatch, caplog
    ):
        monkeypatch.setattr(HistoryOutboxDrainer, "DEPTH_ALARM", 2)
        await _push(fake_redis, 1, 2, 3)
        archive.fail = True
        await drainer._log_retry(RuntimeError("pg down"), backoff=1.0)
        assert any(r.levelname == "ERROR" for r in caplog.records)

    async def test_stop_makes_final_drain_attempt(self, fake_redis, archive, drainer):
        # Entries pushed but never notify()ed (e.g. the notify was lost to a
        # crash) still ship on clean shutdown.
        drainer.start()
        await _push(fake_redis, 1)
        await drainer.stop()
        assert archive.inserted == [_entry(1)]

    async def test_stop_without_start_is_safe(self, drainer):
        await drainer.stop()

    async def test_stop_swallows_final_drain_failure(
        self, fake_redis, archive, drainer
    ):
        # Shutdown must never raise — undrained entries stay in the outbox
        # for the next start.
        await _push(fake_redis, 1)
        archive.fail = True
        drainer.start()
        await drainer.stop()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1


class TestRowMapping:
    def test_round_trip(self):
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

    def test_played_at_maps_to_utc_datetime(self):
        row = _entry_to_row(_entry(1))
        assert row[-1] == datetime.fromtimestamp(1001.0, tz=timezone.utc)

    def test_epoch_zero_unknown_sentinel_survives(self):
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
    async def test_empty_insert_never_connects(self):
        # insert_batch([]) early-outs before any acquire — a bogus DSN proves
        # no connection was attempted.
        archive = PostgresHistoryArchive(Database("postgresql://nope:1/nope"))
        await archive.insert_batch([])

    async def test_nonpositive_recent_never_connects(self):
        archive = PostgresHistoryArchive(Database("postgresql://nope:1/nope"))
        assert await archive.recent(42, 0) == []
        assert await archive.recent(42, -1) == []
