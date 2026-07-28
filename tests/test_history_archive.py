"""Tests for src/history_archive.py — the outbox drainer and the archive's
no-connection paths.

The asyncpg implementation is exercised against a real Postgres only by the
opt-in integration tier (docs/POSTGRES_HISTORY_PLAN.md §9); here the drainer
runs against fakeredis + an in-memory HistoryArchive fake, and
PostgresHistoryArchive is covered exactly as far as it can go without a
server (early-outs, row mapping, close-before-connect).
"""

import asyncio
import math
from dataclasses import replace as dataclasses_replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from collections.abc import Callable, Sequence
from typing import Any, cast

from redis.asyncio import Redis

from src.guild_state import HistoryEntry, serialize_history_entry
from src.history_archive import (
    _INT4_MAX,
    _POISON,
    _TS_MAX,
    HistoryArchive,
    HistoryOutboxDrainer,
    PostgresHistoryArchive,
    RowConversionError,
    SchemaVersionError,
    _entry_to_row,
    _row_to_entry,
    _sanitize_entry,
)
from src.redis_client import (
    DRAINER_LEASE_KEY,
    DRAINER_LEASE_MS,
    HISTORY_DLQ_KEY,
    HISTORY_OUTBOX_KEY,
    GuildRedisStore,
    hold_drainer_lease,
)


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


def _entry_at(played_at: float) -> HistoryEntry:
    return HistoryEntry(guild_id=42, title="poison", played_at=played_at)


def _as_record(mapping: dict[str, Any]) -> asyncpg.Record:
    """asyncpg.Record cannot be constructed from Python, and _row_to_entry only
    ever does __getitem__ — so a dict is a faithful stand-in. The cast is the
    honest way to say that rather than widening the production signature."""
    return cast(asyncpg.Record, mapping)


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
        assert _row_to_entry(_as_record(dict(zip(keys, row)))) == entry

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
        assert _row_to_entry(_as_record(dict(zip(keys, row)))).played_at == 0.0


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
    """The done-callback is what makes a drainer that dies outside its inner
    try loud instead of silent, AND what brings it back — a dead drainer lets
    the non-evictable outbox grow unbounded until someone bounces the bot."""

    async def test_logs_error_when_task_dies_unexpectedly(
        self, drainer: HistoryOutboxDrainer, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def boom() -> None:
            raise RuntimeError("kaboom")

        task = asyncio.create_task(boom())
        with pytest.raises(RuntimeError):
            await task
        drainer._on_task_done(task)
        try:
            assert "died unexpectedly" in caplog.text
            assert "kaboom" in caplog.text
            assert any(r.levelname == "ERROR" for r in caplog.records)
        finally:
            await drainer.stop()  # cancels the scheduled respawn

    async def test_no_log_on_normal_cancellation(
        self, drainer: HistoryOutboxDrainer, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def forever() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(forever())
        await asyncio.sleep(0)  # let it start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        drainer._on_task_done(task)
        assert "died unexpectedly" not in caplog.text

    async def test_dead_task_is_respawned_and_keeps_draining(
        self,
        fake_redis: Redis,
        archive: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The point of M5: a drainer that dies must come back on its own. The
        # first _run raises immediately; the respawn must produce a task that
        # actually drains.
        monkeypatch.setattr(HistoryOutboxDrainer, "RESTART_BASE", 0.01)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        calls = 0
        real_run = drainer._run

        async def flaky_run() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("first run explodes")
            await real_run()

        monkeypatch.setattr(drainer, "_run", flaky_run)
        drainer.start()
        try:
            await _push(fake_redis, 1)
            await _eventually(lambda: archive.inserted == [_entry(1)])
            assert calls >= 2  # it came back, it did not merely log
        finally:
            await drainer.stop()

    async def test_stop_cancels_a_pending_respawn(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A respawn scheduled during teardown would resurrect the drainer after
        # the final drain — and after the archive is closed.
        monkeypatch.setattr(HistoryOutboxDrainer, "RESTART_BASE", 0.01)
        drainer = HistoryOutboxDrainer(fake_redis, archive)

        async def boom() -> None:
            raise RuntimeError("kaboom")

        task = asyncio.create_task(boom())
        with pytest.raises(RuntimeError):
            await task
        drainer._on_task_done(task)
        assert drainer._respawn_handle is not None
        await drainer.stop()
        assert drainer._respawn_handle is None
        await asyncio.sleep(0.05)  # past the respawn delay
        assert drainer._task is None

    async def test_restart_delay_backs_off_then_resets_on_a_productive_cycle(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        async def boom() -> None:
            raise RuntimeError("kaboom")

        # Two deaths in a row double the delay...
        for _ in range(2):
            task = asyncio.create_task(boom())
            with pytest.raises(RuntimeError):
                await task
            drainer._on_task_done(task)
            if drainer._respawn_handle is not None:
                drainer._respawn_handle.cancel()
                drainer._respawn_handle = None
        assert drainer._restart_delay > HistoryOutboxDrainer.RESTART_BASE
        # ...and a cycle that actually delivers proves health, resetting it.
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: archive.inserted == [_entry(1)])
            await _eventually(
                lambda: drainer._restart_delay == HistoryOutboxDrainer.RESTART_BASE
            )
        finally:
            await drainer.stop()


class TestSanitizeEntry:
    """C1 layer 2. Every entry that PARSES must also be INSERTABLE, or one bad
    row fails the executemany carrying 99 good ones and the outbox wedges."""

    def test_clean_entry_is_returned_unchanged(self) -> None:
        # Identity, not just equality: the clean path is every real entry and
        # must not allocate a copy per row.
        entry = _entry(1)
        assert _sanitize_entry(entry) is entry

    def test_nul_stripped_from_every_text_field(self) -> None:
        # Postgres text cannot hold NUL; the server raises
        # CharacterNotInRepertoireError and takes the whole batch with it.
        entry = HistoryEntry(
            guild_id=42,
            title="ab\x00cd",
            webpage_url="https://yt\x00.com",
            requester_name="u\x00ser",
            thumbnail="\x00",
            uploader="chan\x00",
        )
        clean = _sanitize_entry(entry)
        assert clean.title == "abcd"
        assert clean.webpage_url == "https://yt.com"
        assert clean.requester_name == "user"
        assert clean.thumbnail == ""
        assert clean.uploader == "chan"

    def test_huge_epoch_collapses_to_unknown_sentinel(self) -> None:
        # datetime.fromtimestamp would raise OverflowError in _entry_to_row,
        # before any SQL is even sent.
        assert _sanitize_entry(_entry_at(1e18)).played_at == 0.0

    def test_negative_epoch_collapses_to_unknown_sentinel(self) -> None:
        assert _sanitize_entry(_entry_at(-1e18)).played_at == 0.0

    def test_nan_epoch_collapses_to_unknown_sentinel(self) -> None:
        # The chained comparison is False for NaN — which is the reason it is
        # written as a range test rather than two explicit bounds checks.
        assert _sanitize_entry(_entry_at(math.nan)).played_at == 0.0

    def test_timestamptz_ceiling_is_accepted_unchanged(self) -> None:
        assert _sanitize_entry(_entry_at(_TS_MAX)).played_at == _TS_MAX

    def test_int4_columns_are_clamped(self) -> None:
        entry = dataclasses_replace(_entry(1), duration_secs=_INT4_MAX + 1)
        assert _sanitize_entry(entry).duration_secs == _INT4_MAX

    def test_negative_ints_clamp_to_zero(self) -> None:
        entry = dataclasses_replace(_entry(1), played_secs=-5)
        assert _sanitize_entry(entry).played_secs == 0

    def test_snowflake_ids_survive(self) -> None:
        # int8 columns: a Discord snowflake must not be clamped into nonsense.
        entry = dataclasses_replace(_entry(1), requester_id=222222222222222222)
        assert _sanitize_entry(entry).requester_id == 222222222222222222

    def test_all_four_vectors_become_insertable(self) -> None:
        # The end-to-end property: _entry_to_row must not raise on anything
        # that came out of _sanitize_entry.
        for entry in (
            HistoryEntry(guild_id=1, title="x\x00y"),
            _entry_at(1e18),
            _entry_at(-1e18),
            _entry_at(math.nan),
        ):
            _entry_to_row(_sanitize_entry(entry))


class PoisonArchive:
    """Archive that rejects specific titles the way Postgres would: a batch
    containing one poison row fails entirely, exactly like executemany."""

    def __init__(self, poison_titles: set[str]) -> None:
        self._poison = poison_titles
        self.inserted: list[HistoryEntry] = []
        self.transient_failures = 0

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise OSError("connection reset")
        if any(e.title in self._poison for e in entries):
            raise asyncpg.exceptions.DataError("invalid byte sequence")
        self.inserted.extend(entries)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        return []


class TestPoisonQuarantine:
    """C1 layer 1. One entry Postgres will never accept must cost one entry —
    not the batch behind it, and not every guild's archiving."""

    async def test_poison_entry_quarantined_others_delivered(
        self, fake_redis: Redis
    ) -> None:
        archive = PoisonArchive({"Song 2", "Song 4"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2, 3, 4, 5)

        assert await drainer._drain_once() == 5

        assert [e.title for e in archive.inserted] == ["Song 1", "Song 3", "Song 5"]
        assert await fake_redis.llen(HISTORY_DLQ_KEY) == 2
        # The head is retired either way — that is the whole point.
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_dlq_keeps_the_original_wire_bytes(self, fake_redis: Redis) -> None:
        # Parked verbatim so a quarantined play can be replayed once the cause
        # is understood.
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 2)
        await drainer._drain_once()
        assert await fake_redis.lrange(HISTORY_DLQ_KEY, 0, -1) == [
            serialize_history_entry(_entry(2))
        ]

    async def test_quarantine_logs_the_offending_entry(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 2)
        await drainer._drain_once()
        assert "quarantined to DLQ" in caplog.text
        assert any(r.levelname == "ERROR" for r in caplog.records)

    async def test_transient_failure_inside_quarantine_redelivers_the_batch(
        self, fake_redis: Redis
    ) -> None:
        # A network blip while isolating must NOT dead-letter and must NOT
        # retire: the batch comes back and the dedup index absorbs the rows
        # that already landed.
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2, 3)
        archive.transient_failures = 2  # batch insert, then the first isolate
        with pytest.raises(OSError):
            await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 3
        assert await fake_redis.llen(HISTORY_DLQ_KEY) == 0

    async def test_transient_errors_do_not_dead_letter(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # A Postgres restart is not a poison entry. Regression guard on the
        # _POISON tuple: widening it to bare Exception would silently start
        # deleting history on every outage.
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_DLQ_KEY) == 0
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 2

    async def test_catch_all_quarantines_after_repeated_identical_failures(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        # The exotic-poison catch-all: an error type _POISON does not name,
        # raised forever on the same head, must eventually be isolated rather
        # than wedging the outbox for good.
        class AlwaysFails:
            def __init__(self) -> None:
                self.singles: list[HistoryEntry] = []

            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                if len(entries) == 1:
                    # Isolated retry succeeds — the batch, not the row, was bad.
                    self.singles.extend(entries)
                    return
                raise RuntimeError("something exotic")

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                return []

        archive = AlwaysFails()
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2)
        for _ in range(HistoryOutboxDrainer.QUARANTINE_AFTER):
            with pytest.raises(RuntimeError):
                await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 2  # nothing lost yet
        # The next cycle skips the doomed batch insert and isolates instead.
        assert await drainer._drain_once() == 2
        assert [e.title for e in archive.singles] == ["Song 1", "Song 2"]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_head_failure_counter_resets_when_the_head_changes(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        await _push(fake_redis, 1)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert drainer._head_fails == 1
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert drainer._head_fails == 2
        # A successful drain clears the count entirely.
        archive.fail = False
        await drainer._drain_once()
        assert drainer._head_fails == 0
        assert drainer._head_sig is None


class TestDrainDeadline:
    """H1. A connected-but-unresponsive Postgres used to hang the drainer with
    no exception — so no backoff, no DEPTH_ALARM, and not one log line."""

    async def test_hung_archive_becomes_a_timeout_error(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class WedgedArchive:
            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                await asyncio.Event().wait()  # never returns

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                return []

        monkeypatch.setattr(HistoryOutboxDrainer, "DRAIN_DEADLINE_SECS", 0.05)
        drainer = HistoryOutboxDrainer(fake_redis, WedgedArchive())
        await _push(fake_redis, 1)
        with pytest.raises(TimeoutError):
            await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1  # nothing retired

    async def test_hang_reaches_the_retry_log_and_reports_depth(
        self,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The observable the original repro showed was ABSENT: a hang must
        # produce the same warning + backlog depth an error does.
        class WedgedArchive:
            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                await asyncio.Event().wait()

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                return []

        monkeypatch.setattr(HistoryOutboxDrainer, "DRAIN_DEADLINE_SECS", 0.05)
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_START", 0.01)
        drainer = HistoryOutboxDrainer(fake_redis, WedgedArchive())
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: "outbox drain failed" in caplog.text)
            assert "backlog=1" in caplog.text
            assert "TimeoutError" in caplog.text
        finally:
            await drainer.stop(timeout=0.2)

    def test_deadline_stays_below_the_lease_ttl(self) -> None:
        # INVARIANT: a batch must never outlive the ownership it started under.
        assert HistoryOutboxDrainer.DRAIN_DEADLINE_SECS * 1000 < DRAINER_LEASE_MS


class TestDrainerLease:
    """H2, cross-process half. peek → insert → retire is single-consumer; a
    second drainer retires entries the first never inserted."""

    async def test_a_second_drainer_cannot_touch_the_holders_batch(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REGRESSION: this test used to start two drainers and assert a single
        batch, which the scheduler satisfied for free — with 3 entries against
        fakeredis the first drainer always finished before the second peeked, so
        the assertion held whether or not the lease was consulted. Weakening the
        gate in _run to `... or True` passed the entire suite, and so did making
        _lease_id a shared constant, while a forced overlap loses entries
        permanently: the second drainer retires the batch the first is still
        inserting, and those plays end up neither archived nor queued.

        So park the holder *inside* insert_batch and drive the second drainer's
        full cycle, lease check included, while it is parked.
        """
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedArchive(FakeArchive):
            """Blocks the FIRST insert only. Later calls run straight through,
            so a drainer that wrongly ignores the lease actually completes its
            retire and the damage becomes observable — a gate that blocked
            everyone would hide the very bug this test is for."""

            def __init__(self) -> None:
                super().__init__()
                self._gate_armed = True

            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                if self._gate_armed:
                    self._gate_armed = False
                    entered.set()
                    await release.wait()
                await super().insert_batch(entries)

        archive = GatedArchive()
        # Short tick so the second drainer takes many laps inside the window
        # below rather than one lucky one.
        monkeypatch.setattr(HistoryOutboxDrainer, "TICK_SECS", 0.01)
        await _push(fake_redis, 1, 2, 3)
        first = HistoryOutboxDrainer(fake_redis, archive)
        second = HistoryOutboxDrainer(fake_redis, archive)
        first.start()
        try:
            async with asyncio.timeout(2):
                await entered.wait()
            # The holder is now parked mid-insert with all 3 entries still on
            # the outbox, its lease held.
            assert not await hold_drainer_lease(fake_redis, second._lease_id)
            second.start()
            await asyncio.sleep(0.1)  # ~10 laps of second's loop
            assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 3
            assert archive.batches == []  # second inserted nothing
        finally:
            release.set()
            await second.stop()
            await first.stop()
        # And the holder still delivers: nothing was lost to the standoff.
        assert archive.inserted == [_entry(1), _entry(2), _entry(3)]
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_each_drainer_gets_its_own_lease_id(self, fake_redis: Redis) -> None:
        # A restarted process must not inherit the lease of the process it
        # replaced: _if_still_owner compares ids, so a shared/derived id would
        # let a successor RENEW a dead predecessor's lease and drain alongside
        # a still-live one. Cheap to assert, and the failure is invisible until
        # it costs history.
        a = HistoryOutboxDrainer(fake_redis, FakeArchive())
        b = HistoryOutboxDrainer(fake_redis, FakeArchive())
        assert a._lease_id != b._lease_id

    async def test_lease_is_taken_over_after_it_lapses(
        self, fake_redis: Redis, archive: Any
    ) -> None:
        # A drainer that died without releasing must not block its replacement
        # forever — the TTL is what makes a dead holder recoverable.
        dead = HistoryOutboxDrainer(fake_redis, archive)
        await fake_redis.set(DRAINER_LEASE_KEY, dead._lease_id, px=DRAINER_LEASE_MS)
        successor = HistoryOutboxDrainer(fake_redis, archive)
        assert await successor._drain_once() == 0  # drain itself is unblocked
        await fake_redis.delete(DRAINER_LEASE_KEY)  # simulate the lapse
        successor.start()
        try:
            await _push(fake_redis, 1)
            successor.notify()
            await _eventually(lambda: archive.inserted == [_entry(1)])
        finally:
            await successor.stop()

    async def test_stop_releases_the_lease(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Released rather than left to expire, so the next instance in a
        # rolling deploy starts draining immediately instead of idling out the
        # 90s TTL.
        drainer.start()
        async with asyncio.timeout(2.0):  # wait for the loop to take it
            while await fake_redis.get(DRAINER_LEASE_KEY) is None:
                await asyncio.sleep(0.01)
        await drainer.stop()
        assert await fake_redis.get(DRAINER_LEASE_KEY) is None

    async def test_a_foreign_lease_is_never_released(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Compare-and-delete: stopping must not hand away a lease that lapsed
        # and was retaken by someone else in the meantime.
        await fake_redis.set(DRAINER_LEASE_KEY, "someone-else", px=DRAINER_LEASE_MS)
        await drainer.stop()
        assert await fake_redis.get(DRAINER_LEASE_KEY) == b"someone-else"

    async def test_final_drain_is_skipped_without_the_lease(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # stop()'s final flush is a drain like any other, so it needs the same
        # gate — otherwise shutdown IS the concurrent second drainer.
        await fake_redis.set(DRAINER_LEASE_KEY, "someone-else", px=DRAINER_LEASE_MS)
        await _push(fake_redis, 1)
        await drainer.stop()
        assert archive.inserted == []
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 1


class TestStopIdempotence:
    """H2, in-process half. Two concurrent stop()s each ran their own
    peek → insert → retire; the second retired entries the first had not yet
    inserted (reproduced: 6 pushed, 3 archived, outbox empty)."""

    async def test_concurrent_stops_drain_exactly_once(self, fake_redis: Redis) -> None:
        gate = asyncio.Event()

        class GatedArchive:
            def __init__(self) -> None:
                self.calls = 0
                self.inserted: list[HistoryEntry] = []

            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                self.calls += 1
                await gate.wait()  # hold the first drain open
                self.inserted.extend(entries)

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                return []

        archive = GatedArchive()
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, *range(6))
        # Deliberately NOT started: this isolates the two stop() paths from the
        # run loop's own drain, so `calls` counts only what shutdown did.

        stops = [asyncio.create_task(drainer.stop()) for _ in range(2)]
        await asyncio.sleep(0.05)  # both stops in flight, first one blocked
        gate.set()
        await asyncio.gather(*stops)

        # calls == 2 is the loud failure signal for a second concurrent drain.
        assert archive.calls == 1
        assert len(archive.inserted) == 6
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

    async def test_second_stop_is_a_noop(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        drainer.start()
        await _push(fake_redis, 1)
        await drainer.stop()
        await _push(fake_redis, 2)
        await drainer.stop()
        # The second stop must not run a drain of its own.
        assert archive.inserted == [_entry(1)]


class TestOutboxCap:
    """M3. Opt-in only — dropping un-archived plays is data loss, so the
    default must stay unbounded."""

    async def test_cap_off_by_default(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        assert HistoryOutboxDrainer.OUTBOX_MAX == 0
        await _push(fake_redis, *range(5))
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 5

    async def test_cap_drops_oldest_and_logs_an_error(
        self,
        fake_redis: Redis,
        archive: Any,
        drainer: HistoryOutboxDrainer,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(HistoryOutboxDrainer, "BATCH_SIZE", 2)
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 3)
        await _push(fake_redis, *range(8))  # 8 queued, batch of 2 drains first
        await drainer._drain_once()
        # 6 left, cap 3 → the 3 oldest remaining are dropped.
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 3
        assert "over cap" in caplog.text
        assert any(r.levelname == "ERROR" for r in caplog.records)
        # The dropped ones are the OLDEST: entries 2,3,4 go, 5,6,7 stay.
        remaining = await fake_redis.lrange(HISTORY_OUTBOX_KEY, 0, -1)
        assert remaining[-1] == serialize_history_entry(_entry(5))

    async def test_cap_is_enforced_while_postgres_is_down(
        self,
        fake_redis: Redis,
        archive: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REGRESSION: _enforce_cap used to run only on the success tail of
        _drain_batch, so it was unreachable for the whole duration of an
        outage — the one scenario HISTORY_OUTBOX_MAX exists for. Measured
        before the fix: cap 10, 50 entries, ~100 failed cycles, depth still 50.
        """
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 3)
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_START", 0.01)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        archive.fail = True  # Postgres unreachable for every cycle below
        await _push(fake_redis, *range(9))
        drainer.start()
        try:
            # Depth converges to the cap purely on failing cycles — nothing is
            # ever archived, so before the fix this stayed at 9 forever.
            async with asyncio.timeout(2):
                while await fake_redis.llen(HISTORY_OUTBOX_KEY) != 3:
                    await asyncio.sleep(0.01)
            assert archive.batches == []
        finally:
            await drainer.stop(timeout=0.2)

    async def test_cap_failure_does_not_break_the_backoff(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The failure-path cap check runs while Redis may itself be what broke;
        # it must never replace the error the drainer is already backing off on.
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 1)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        with patch(
            "src.history_archive.outbox_depth", side_effect=RuntimeError("redis down")
        ):
            await drainer._enforce_cap_quietly()  # must not raise


class TestSchemaVersionGuard:
    """M2. The archive no longer creates its own schema — it verifies it."""

    async def test_unmigrated_database_raises_actionably(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        conn = MagicMock()
        conn.fetchval = AsyncMock(side_effect=asyncpg.exceptions.UndefinedTableError())
        with pytest.raises(SchemaVersionError, match="db-migrate"):
            await archive._assert_schema_version(conn)

    async def test_old_schema_raises(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value=0)
        with pytest.raises(SchemaVersionError):
            await archive._assert_schema_version(conn)

    async def test_expected_schema_passes(self) -> None:
        from src.db_migrate import EXPECTED_SCHEMA_VERSION

        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value=EXPECTED_SCHEMA_VERSION)
        await archive._assert_schema_version(conn)

    async def test_newer_schema_warns_but_proceeds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A rollback must not become an outage: migrations are additive, so an
        # older bot against a newer schema keeps working.
        from src.db_migrate import EXPECTED_SCHEMA_VERSION

        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        conn = MagicMock()
        conn.fetchval = AsyncMock(return_value=EXPECTED_SCHEMA_VERSION + 1)
        await archive._assert_schema_version(conn)
        assert "ahead of this build" in caplog.text


class TestDrainerLoopInvariants:
    """Two properties whose docstrings state them plainly and whose removal
    every existing test survived."""

    async def test_second_stop_waits_rather_than_returning_early(
        self, fake_redis: Redis
    ) -> None:
        """_stop_lock exists so the second caller WAITS for the first to
        finish. Only the _stopped flag was covered, and _stopped alone satisfies
        both existing tests because nothing awaits between its check and its
        set — so dropping the lock entirely passed all 73.

        If the second stop returns early, MusicBotApp.close() proceeds to
        archive.close() while the first stop's final drain is still running, and
        that drain dies on a closed pool — losing the shutdown flush.
        """
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedArchive(FakeArchive):
            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                entered.set()
                await release.wait()
                await super().insert_batch(entries)

        drainer = HistoryOutboxDrainer(fake_redis, GatedArchive())
        await _push(fake_redis, 1)
        first = asyncio.ensure_future(drainer.stop(timeout=5))
        async with asyncio.timeout(2):
            await entered.wait()  # first stop is inside the final drain
        second = asyncio.ensure_future(drainer.stop(timeout=5))
        await asyncio.sleep(0.05)

        assert not second.done()  # it is WAITING, not short-circuiting

        release.set()
        await asyncio.gather(first, second)

    async def test_idle_loop_does_not_hot_spin(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_wake.clear() after the idle wait. Without it the event stays set
        once notify() has fired, wait() returns instantly every iteration, and
        the loop spins at full speed issuing a lease round-trip plus an LRANGE
        per pass — forever, on an empty outbox. One pegged core and a permanent
        Redis load floor, on the Redis that also serves playback."""
        monkeypatch.setattr(HistoryOutboxDrainer, "TICK_SECS", 5.0)
        peeks = {"n": 0}
        real_lrange = fake_redis.lrange

        async def counting_lrange(*args: Any, **kwargs: Any) -> Any:
            peeks["n"] += 1
            return await real_lrange(*args, **kwargs)

        monkeypatch.setattr(fake_redis, "lrange", counting_lrange)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: archive.inserted == [_entry(1)])
            # A productive cycle `continue`s and peeks again to check for more
            # backlog; let that settle before sampling.
            await asyncio.sleep(0.1)
            settled = peeks["n"]
            # TICK_SECS is 5s and the outbox is empty, so a correctly-idling
            # loop peeks ZERO times in this window. A hot spin does thousands.
            await asyncio.sleep(0.2)
            assert peeks["n"] == settled
        finally:
            await drainer.stop(timeout=0.5)


class TestCloseNeverEscalates:
    """H2/F5. close() runs on the shutdown path, ahead of the Redis pool,
    discord.py's own close and the span flush. It must not be able to take any
    of those down with it."""

    async def test_a_hung_graceful_close_is_forced_and_swallowed(self) -> None:
        # Measured against a connected-but-unresponsive Postgres: pool.close()
        # waits for in-flight queries the server never finishes, then raised
        # TimeoutError 30s later — which aborted the rest of MusicBotApp.close().
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        pool = MagicMock()
        pool.close = AsyncMock(side_effect=TimeoutError("server stopped answering"))
        pool.terminate = MagicMock()
        archive._pool = pool

        await archive.close()  # must not raise

        pool.terminate.assert_called_once()  # sockets not left behind
        assert archive._pool is None

    async def test_slow_graceful_close_is_bounded_then_forced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A close() that never returns must still hand control back to teardown.
        import src.history_archive as history_archive

        monkeypatch.setattr(history_archive, "_POOL_CLOSE_TIMEOUT_SECS", 0.05)
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")

        async def never_returns() -> None:
            # An async side_effect, not a lambda returning a coroutine: AsyncMock
            # awaits the former and hands the latter back un-awaited, which would
            # make close() return instantly and test nothing.
            await asyncio.Event().wait()

        pool = MagicMock()
        pool.close = AsyncMock(side_effect=never_returns)
        pool.terminate = MagicMock()
        archive._pool = pool

        await archive.close()

        pool.terminate.assert_called_once()

    async def test_stop_survives_a_task_that_died_with_an_exception(
        self, fake_redis: Redis, archive: Any
    ) -> None:
        """_on_task_done leaves _task pointing at the FAILED task while a
        respawn is pending, so awaiting it re-raises whatever killed it.
        Catching only CancelledError let that escape stop(), which skipped both
        the final drain and the lease release — the successor in a rolling
        deploy then idled out the full 90s TTL instead of taking over."""
        drainer = HistoryOutboxDrainer(fake_redis, archive)

        async def boom() -> None:
            raise RuntimeError("drainer blew up")

        task = asyncio.ensure_future(boom())
        await asyncio.sleep(0)  # let it run and fail
        assert task.done() and not task.cancelled()
        drainer._task = task
        await hold_drainer_lease(fake_redis, drainer._lease_id)

        await drainer.stop()  # must not raise

        # The lease was still released, so a successor starts draining at once.
        assert await fake_redis.get(DRAINER_LEASE_KEY) is None


class TestEnsureCloseRace:
    """M1. _closed was checked before the lock only, so a -ping health_check
    racing shutdown built a pool nothing was left to close."""

    async def test_close_during_in_flight_ensure_leaves_no_pool(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        release = asyncio.Event()
        pool = MagicMock()
        pool.close = AsyncMock()
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=MagicMock(fetchval=AsyncMock()))
        acquire_cm.__aexit__ = AsyncMock(return_value=None)
        pool.acquire = MagicMock(return_value=acquire_cm)

        async def gated_create_pool(*args: Any, **kwargs: Any) -> Any:
            await release.wait()
            return pool

        with (
            patch("src.history_archive.asyncpg.create_pool", gated_create_pool),
            patch.object(PostgresHistoryArchive, "_assert_schema_version", AsyncMock()),
        ):
            ensuring = asyncio.create_task(archive._ensure())
            await asyncio.sleep(0)  # let _ensure reach the gated create_pool
            closing = asyncio.create_task(archive.close())
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(RuntimeError, match="closed"):
                await ensuring
            await closing

        # The pool that _ensure built during the close must be closed by
        # _ensure itself — nothing else can still see it.
        pool.close.assert_awaited()
        assert archive._pool is None

    async def test_close_waits_for_an_in_flight_ensure(self) -> None:
        # close() takes the init lock, so it cannot null _pool underneath an
        # _ensure that is about to assign it.
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        assert archive._init_lock is not None
        async with archive._init_lock:
            closing = asyncio.create_task(archive.close())
            await asyncio.sleep(0.01)
            assert not closing.done()  # blocked on the lock
        await closing

    async def test_concurrent_first_ensures_create_one_pool(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        pool = MagicMock()
        pool.close = AsyncMock()
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=MagicMock())
        acquire_cm.__aexit__ = AsyncMock(return_value=None)
        pool.acquire = MagicMock(return_value=acquire_cm)
        with (
            patch(
                "src.history_archive.asyncpg.create_pool",
                AsyncMock(return_value=pool),
            ) as create,
            patch.object(PostgresHistoryArchive, "_assert_schema_version", AsyncMock()),
        ):
            a, b = await asyncio.gather(archive._ensure(), archive._ensure())
        assert a is b is pool
        assert create.await_count == 1


class TestPoolConfiguration:
    """H1/L7: the kwargs are the fix, so they are asserted rather than trusted."""

    async def test_create_pool_kwargs(self) -> None:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        with patch(
            "src.history_archive.asyncpg.create_pool", AsyncMock()
        ) as create_pool:
            await archive._create_pool()
        assert create_pool.await_args is not None
        kwargs = create_pool.await_args.kwargs
        assert kwargs["timeout"] == 10  # connect bound
        assert kwargs["command_timeout"] == 30  # statement bound (H1)
        assert kwargs["statement_cache_size"] == 100  # PgBouncer knob (L7)
        assert kwargs["server_settings"]["application_name"] == "musicbot-history"


class TestRowConversionClassification:
    """The poison taxonomy has to be a property of the CODE PATH, not of which
    exception the platform happened to raise. `datetime.fromtimestamp(1e18)`
    raises OverflowError on Linux and `OSError: [Errno 84]` on macOS — and
    OSError is what a Postgres restart looks like, so a taxonomy naming it would
    delete healthy history on every outage."""

    async def test_unconvertible_entry_raises_row_conversion_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.history_archive as history_archive

        monkeypatch.setattr(history_archive, "_sanitize_entry", lambda e: e)
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        with pytest.raises(RowConversionError):
            await archive.insert_batch([_entry_at(1e18)])

    async def test_row_conversion_error_is_poison(self) -> None:
        assert isinstance(RowConversionError("x"), _POISON)

    async def test_transport_errors_are_not_poison(self) -> None:
        # The regression this guards: OSError and its subclasses are how a
        # restart, failover or timeout presents.
        for exc in (
            OSError(84, "Value too large"),
            ConnectionResetError("reset"),
            TimeoutError("deadline"),
            asyncpg.exceptions.PostgresConnectionError("down"),
        ):
            assert not isinstance(exc, _POISON), exc

    async def test_conversion_happens_before_any_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What makes the wrapping unambiguous: nothing has touched a socket
        # yet, so a failure here cannot be a disguised network error.
        import src.history_archive as history_archive

        monkeypatch.setattr(history_archive, "_sanitize_entry", lambda e: e)
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        with patch("src.history_archive.asyncpg.create_pool", AsyncMock()) as create:
            with pytest.raises(RowConversionError):
                await archive.insert_batch([_entry_at(1e18)])
        create.assert_not_awaited()

    async def test_unconvertible_entry_is_dead_lettered_not_retried_forever(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End of the chain: an entry the sanitizer somehow missed reaches the
        # DLQ on the FIRST cycle rather than wedging the outbox until the
        # QUARANTINE_AFTER catch-all eventually notices.
        import src.history_archive as history_archive

        monkeypatch.setattr(history_archive, "_sanitize_entry", lambda e: e)

        class RealConversionArchive(FakeArchive):
            """Runs the REAL row mapping, then records what survived.

            It used to hand-roll insert_batch's try/except-RowConversionError
            wrapper as well — a duplicate of production logic that would keep
            testing the old taxonomy, and keep passing, after the real one
            changed. (That is the same shape as the `_Unsanitized` double an
            earlier review removed.) Now it only calls _entry_to_row and lets
            the failure out; the classification under test is the drainer's.
            """

            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                try:
                    [_entry_to_row(e) for e in entries]
                except Exception as e:
                    raise RowConversionError(str(e)) from e
                await super().insert_batch(entries)

        store = GuildRedisStore(fake_redis, guild_id=42)
        await store.push_history(_entry(1))
        await store.push_history(_entry_at(1e18))
        archive = RealConversionArchive()
        drainer = HistoryOutboxDrainer(fake_redis, archive)

        assert await drainer._drain_once() == 2
        assert [e.title for e in archive.inserted] == ["Song 1"]
        assert await fake_redis.llen(HISTORY_DLQ_KEY) == 1
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0
