"""Tests for src/history_archive.py — the outbox drainer and the archive's
no-connection paths.

The asyncpg implementation is exercised against a real Postgres only by the
opt-in integration tier; here the drainer
runs against fakeredis + an in-memory HistoryArchive fake, and
PostgresHistoryArchive is covered exactly as far as it can go without a
server (early-outs, row mapping, close-before-connect).
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
import redis.asyncio as aioredis

from collections.abc import Callable, Sequence
from typing import Any, Optional, cast

from redis.asyncio import Redis

from src.guild_state import HistoryEntry, serialize_history_entry
from src.history_archive import (
    _INSERT_SQL,
    _POISON,
    _RECENT_SQL,
    HistoryArchive,
    HistoryOutboxDrainer,
    PostgresHistoryArchive,
    SchemaVersionError,
    _entry_to_row,
    _row_to_entry,
)
from src.redis_client import (
    HISTORY_OUTBOX_GROUP,
    HISTORY_OUTBOX_KEY,
    OUTBOX_FIELD,
    GuildRedisStore,
    ensure_outbox_group,
    outbox_pending_count,
    read_outbox_new,
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


def _as_record(mapping: dict[str, Any]) -> asyncpg.Record:
    """asyncpg.Record cannot be constructed from Python, and _row_to_entry only
    ever does __getitem__ — so a dict is a faithful stand-in. The cast is the
    honest way to say that rather than widening the production signature."""
    return cast(asyncpg.Record, mapping)


class RecordsRejections:
    """record_rejection for archive test doubles.

    The HistoryArchive protocol requires it and every double wants the identical
    trivial implementation, so it lives here once rather than eleven times. No
    __init__ — the list is created on first use — which is what lets it mix into
    doubles that have their own constructors without any MRO cooperation.
    """

    @property
    def rejections(self) -> list[tuple[HistoryEntry, BaseException]]:
        if not hasattr(self, "_rejections"):
            self._rejections: list[tuple[HistoryEntry, BaseException]] = []
        return self._rejections

    @property
    def rejected_wires(self) -> list[Optional[bytes]]:
        if not hasattr(self, "_rejected_wires"):
            self._rejected_wires: list[Optional[bytes]] = []
        return self._rejected_wires

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None:
        # wire is captured, not dropped: it is the DELIVERED bytes, and the
        # whole reason the parameter exists is that they can differ from a
        # re-serialization of `entry`.
        self.rejections.append((entry, error))
        self.rejected_wires.append(wire)


class FakeArchive(RecordsRejections):
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
async def drainer(fake_redis: Redis, archive: Any) -> HistoryOutboxDrainer:
    # Bootstrapping here mirrors production: setup_hook calls
    # ensure_outbox_group before the drainer ever runs. Without it every test
    # would silently exercise the NOGROUP heal path instead of the ordinary
    # one, and that path has its own test.
    await ensure_outbox_group(fake_redis)
    return HistoryOutboxDrainer(fake_redis, archive)


async def _push(fake_redis: Redis, *ns: int) -> None:
    store = GuildRedisStore(fake_redis, guild_id=42)
    for n in ns:
        await store.push_history(_entry(n))


async def _outbox_wires(fake_redis: Redis) -> list[bytes]:
    """Every payload still in the outbox, oldest first.

    The list-era assertions used `lrange(key, 0, -1)`, which returned
    NEWEST-first because the producer LPUSHed. XRANGE is oldest-first, so
    expectations that were reversed under the list read forward here.
    """
    return [fields[OUTBOX_FIELD] for _id, fields in await _stream_entries(fake_redis)]


async def _stream_entries(fake_redis: Redis) -> list[tuple[bytes, dict[bytes, bytes]]]:
    """The outbox stream, oldest first, narrowed to the ordinary-entry shape.

    redis-py types XRANGE's reply as a union wide enough to cover XAUTOCLAIM's
    4-tuple rows and RESP3 dict forms, so every unpack would otherwise need its
    own ignore. One cast here says "one stream, ordinary entries" once.
    """
    return cast(
        list[tuple[bytes, dict[bytes, bytes]]],
        await fake_redis.xrange(HISTORY_OUTBOX_KEY),
    )


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
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

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
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 7

    async def test_corrupt_entry_retired_not_inserted(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Corrupt bytes must be consumed (or they'd wedge the queue head
        # forever) while the surviving entries still make it to the archive.
        await _push(fake_redis, 1)
        await fake_redis.xadd(
            HISTORY_OUTBOX_KEY, {OUTBOX_FIELD: b"not json"}
        )  # newer than entry 1
        assert await drainer._drain_once() == 2
        assert archive.inserted == [_entry(1)]
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_archive_failure_leaves_outbox_intact(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Retire happens strictly after a successful insert — a failed insert
        # must leave every entry in place for the retry.
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 2

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
            assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0
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
            assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 1  # nothing lost
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
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 1


# The play_history column order, once. _entry_to_row emits a positional tuple
# against _INSERT_SQL and _row_to_entry reads a Record by name, so a column
# added to one and not the other is exactly the drift these tests catch — and a
# per-test literal list is how that drift stays invisible.
_COLUMNS = (
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
    "message_id",
)


class TestRowMapping:
    def test_row_width_matches_the_insert_statement(self) -> None:
        """A column added to _entry_to_row but not to _INSERT_SQL (or vice
        versa) is a runtime asyncpg error on the first real insert, and the
        fakeredis-backed drainer tests cannot see it. message_id landed exactly
        this way: the wire carried it for two branches while every INSERT still
        listed ten columns, so the value was silently discarded."""
        assert len(_entry_to_row(_entry(1))) == len(_COLUMNS)
        assert _INSERT_SQL.count("$") == len(_COLUMNS)
        for column in _COLUMNS:
            assert column in _INSERT_SQL
            assert column in _RECENT_SQL

    def test_message_id_round_trips(self) -> None:
        # The NP host id is a snowflake; it must survive as a native int, and
        # it must actually reach the row rather than defaulting to 0.
        entry = HistoryEntry(guild_id=1, title="x", message_id=1234567890123456789)
        row = _entry_to_row(entry)
        assert row[_COLUMNS.index("message_id")] == 1234567890123456789
        assert _row_to_entry(_as_record(dict(zip(_COLUMNS, row)))).message_id == (
            1234567890123456789
        )

    def test_round_trip(self) -> None:
        entry = _entry(1, guild_id=222222222222222222)
        row = _entry_to_row(entry)
        keys = _COLUMNS
        assert _row_to_entry(_as_record(dict(zip(keys, row)))) == entry

    def test_played_at_maps_to_utc_datetime(self) -> None:
        row = _entry_to_row(_entry(1))
        assert row[_COLUMNS.index("played_at")] == datetime.fromtimestamp(
            1001.0, tz=timezone.utc
        )

    def test_epoch_zero_unknown_sentinel_survives(self) -> None:
        # played_at 0.0 = "unknown" — carried into Postgres as to_timestamp(0),
        # not NULL.
        entry = HistoryEntry(guild_id=1, title="x")
        row = _entry_to_row(entry)
        assert row[_COLUMNS.index("played_at")] == datetime.fromtimestamp(
            0, tz=timezone.utc
        )
        keys = _COLUMNS
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

    async def test_close_winning_the_race_to_the_lock_still_refuses(self) -> None:
        """The re-check INSIDE _init_lock, which the pre-lock one cannot cover.

        _ensure reads _closed, then awaits the lock. A close() landing in that
        window passes the first check and would otherwise go on to build a pool
        with nothing left to close it — the exact leak the guard exists for, and
        the only arm of it no test reached.
        """
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive._init_lock.acquire()  # _ensure will block here
        ensure = asyncio.ensure_future(archive._ensure())
        await asyncio.sleep(0)  # let it get past the pre-lock check
        archive._closed = True  # close() wins while _ensure waits
        archive._init_lock.release()
        with (
            patch("src.history_archive.asyncpg.create_pool", AsyncMock()) as create,
            pytest.raises(RuntimeError, match="closed"),
        ):
            await ensure
        create.assert_not_awaited()


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


class PoisonArchive(RecordsRejections):
    """Archive that rejects specific titles the way Postgres would: a batch
    containing one poison row fails entirely, exactly like executemany."""

    def __init__(
        self,
        poison_titles: set[str],
        transient_titles: Optional[set[str]] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        self._poison = poison_titles
        # Which refusal to raise. DataError by default (SQLSTATE 22xxx, how a
        # NUL or an out-of-range value used to present); CheckViolationError is
        # the 23514 arm the play_history CHECK constraints make reachable, and the
        # two live on
        # different branches of asyncpg's hierarchy — which is the entire reason
        # _POISON names both.
        self._exc = exc or asyncpg.exceptions.DataError("invalid byte sequence")
        # Fires ONCE per title, then forgets it: a blip on one entry, not an
        # outage. Checked AFTER the poison set so a batch containing both still
        # fails the way executemany does (poison first), leaving the transient
        # to land during per-entry isolation.
        self._transient = set(transient_titles or ())
        self.inserted: list[HistoryEntry] = []
        self.transient_failures = 0

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise OSError("connection reset")
        if any(e.title in self._poison for e in entries):
            raise self._exc
        for e in entries:
            if e.title in self._transient:
                self._transient.discard(e.title)
                raise OSError("connection reset")
        self.inserted.extend(entries)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        return []


class PushDuringInsert(RecordsRejections):
    """Wraps an archive and LPUSHes a fresh play onto the outbox as each insert
    runs — a song finishing in some guild while the drainer is mid-batch.

    That window is the whole reason retire counts have to come from the peek:
    an insert takes real time (measured 15.7ms batched, 343ms isolating), and
    anything arriving inside it sits at the HEAD of the list while retire pops
    from the TAIL. A retire sized to anything other than what was peeked eats
    plays that were never inserted.
    """

    def __init__(self, inner: Any, fake_redis: Redis, arrivals: list[int]) -> None:
        self._inner = inner
        self._redis = fake_redis
        self._arrivals = list(arrivals)

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        if self._arrivals:
            await _push(self._redis, self._arrivals.pop(0))
        await self._inner.insert_batch(entries)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        return await self._inner.recent(guild_id, limit)

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None:
        # Forwarded, not inherited from RecordsRejections: a wrapper that
        # swallowed rejections into its own list would make the inner archive
        # look like nothing was ever refused, which is precisely what these
        # tests assert on.
        await self._inner.record_rejection(entry, error, trace_id, wire)


class TestRejectionIsolation:
    """One row Postgres refuses must cost one row — not the 99 batched with it,
    and not every guild's archiving.

    Since the schema lock this whole path is expected to be dead code: entries
    are clamped into the column domain at construction, so a refusal means the
    validator regressed or the schema drifted. It is kept, and tested, because
    dropping the batch on a regression would turn a one-row bug into a
    hundred-play loss — a regression is exactly when you most want the other 99.
    """

    async def test_refused_entry_is_recorded_and_the_rest_are_delivered(
        self, fake_redis: Redis
    ) -> None:
        archive = PoisonArchive({"Song 2", "Song 4"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2, 3, 4, 5)

        assert await drainer._drain_once() == 5

        assert [e.title for e in archive.inserted] == ["Song 1", "Song 3", "Song 5"]
        assert [e.title for e, _ in archive.rejections] == ["Song 2", "Song 4"]
        # The head is retired either way — that is the whole point.
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_a_check_violation_routes_to_record_rejection(
        self, fake_redis: Redis
    ) -> None:
        # The arm the play_history CHECK constraints make reachable.
        # CheckViolationError is
        # SQLSTATE 23514 and NOT a DataError, so before _POISON was widened this
        # propagated instead of being isolated — a permanent drain wedge.
        archive = PoisonArchive(
            {"Song 2"}, exc=asyncpg.exceptions.CheckViolationError("guild_id > 0")
        )
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2, 3)

        assert await drainer._drain_once() == 3

        assert [e.title for e in archive.inserted] == ["Song 1", "Song 3"]
        [(entry, error)] = archive.rejections
        assert entry.title == "Song 2"
        assert isinstance(error, asyncpg.exceptions.CheckViolationError)
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_the_recorded_entry_is_the_whole_play(
        self, fake_redis: Redis
    ) -> None:
        # Replayability: record_rejection receives the entry itself, which it
        # serializes verbatim into play_history_rejected.payload. The DLQ this
        # replaces parked raw wire bytes for the same reason.
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 2)
        await drainer._drain_once()
        assert [e for e, _ in archive.rejections] == [_entry(2)]

    async def test_the_delivered_wire_bytes_are_what_get_parked(
        self, fake_redis: Redis
    ) -> None:
        """payload must be the bytes that were REFUSED, not a re-encoding.

        The distinction is invisible while every build agrees on the schema and
        decisive during a rollout: an entry written by a newer build carries
        fields this parser drops, so re-serializing would park a lossy copy of
        the row Postgres rejected. It also splits the md5 dedup identity across
        build versions, and this table's contract is that one row means one
        failure.
        """
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 2)
        delivered = await _outbox_wires(fake_redis)
        await drainer._drain_once()
        assert archive.rejected_wires == delivered

    async def test_a_direct_call_without_wire_bytes_still_records(self) -> None:
        # The fallback arm: a caller holding only an entry (a test, or any
        # future producer that never went through Redis) must still be able to
        # park it rather than lose it to a TypeError.
        archive = PostgresHistoryArchive("postgresql://unused")
        entry = _entry(1)
        # No server, so the insert fails and the terminal handler logs — which
        # is the point: it reaches the insert at all, with a payload.
        await archive.record_rejection(entry, RuntimeError("refused"))

    async def test_rejection_logs_the_offending_entry(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 2)
        await drainer._drain_once()
        # The message names the CAUSE, not the symptom: a row here means a bug
        # in the validator or a schema this build was not written for.
        assert "refused a row" in caplog.text
        assert "validator regressed" in caplog.text
        assert any(r.levelname == "ERROR" for r in caplog.records)

    async def test_a_transient_failure_mid_isolation_redelivers_the_rest(
        self, fake_redis: Redis
    ) -> None:
        # A network blip while isolating must NOT record a rejection and must
        # NOT retire past its progress: the remainder comes back and the dedup
        # index absorbs the rows that already landed.
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2, 3)
        archive.transient_failures = 2  # batch insert, then the first isolate
        with pytest.raises(OSError):
            await drainer._drain_once()
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 3
        assert archive.rejections == []

    async def test_transient_errors_are_never_recorded_as_rejections(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # A Postgres restart is not a refused row. Regression guard on the
        # _POISON tuple: widening it to bare Exception would start dropping
        # history on every outage.
        await _push(fake_redis, 1, 2)
        archive.fail = True
        with pytest.raises(RuntimeError):
            await drainer._drain_once()
        assert archive.rejections == []
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 2

    async def test_an_unforeseen_error_type_still_wedges_rather_than_drops(
        self, fake_redis: Redis
    ) -> None:
        """The deliberate trade the QUARANTINE_AFTER catch-all used to make
        differently.

        That counter isolated a repeatedly-failing head after 8 tries even when
        the exception said "transient", to cover poison shapes _POISON did not
        name. With the schema lock there is no such shape left: a data-caused
        refusal is a CHECK or a DataError, both named. Anything else genuinely
        IS transient, so retrying forever is correct and dropping the batch
        would be data loss. The outbox growing is the visible symptom, and
        HISTORY_OUTBOX_MAX plus the depth gauge are what page on it.
        """

        class AlwaysFails(RecordsRejections):
            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                raise RuntimeError("something exotic")

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                return []

        archive = AlwaysFails()
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2)
        for _ in range(10):
            with pytest.raises(RuntimeError):
                await drainer._drain_once()
        # Nothing lost, nothing recorded as rejected: it redelivers forever.
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 2
        assert archive.rejections == []


class TestRetireAccounting:
    """The outbox's one hard rule: never retire what was not inserted.

    All three guards below shipped with no assertion over them — each was found
    by mutation (delete the guard, run the suite, watch it all pass). They matter
    because retire is a *positional* RPOP: it pops N from the tail with no idea
    which entries those are, so any N that did not come from this cycle's peek
    destroys plays silently — nothing in play_history, nothing in
    play_history_rejected, no log line, nothing for an operator to find.
    """

    async def test_retire_pops_exactly_the_peeked_entries(
        self, fake_redis: Redis
    ) -> None:
        # retire_outbox(len(raw)), never retire_outbox(BATCH_SIZE): the peek is
        # bounded by BATCH_SIZE but is usually far short of it.
        archive = FakeArchive()
        drainer = HistoryOutboxDrainer(
            fake_redis, PushDuringInsert(archive, fake_redis, [99])
        )
        await _push(fake_redis, 1, 2, 3)

        assert await drainer._drain_once() == 3

        assert [e.title for e in archive.inserted] == ["Song 1", "Song 2", "Song 3"]
        # Song 99 arrived after the peek and was never inserted, so it must
        # still be queued.
        assert await _outbox_wires(fake_redis) == [serialize_history_entry(_entry(99))]

    async def test_isolation_does_not_retire_the_batch_a_second_time(
        self, fake_redis: Redis
    ) -> None:
        # _isolate settles and retires entry by entry, so _drain_batch must NOT
        # also retire len(raw) afterwards. Isolating 100 entries takes ~343ms,
        # which is plenty of time for arrivals to stack up behind them.
        poison = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(
            fake_redis, PushDuringInsert(poison, fake_redis, [97, 98])
        )
        await _push(fake_redis, 1, 2, 3)

        assert await drainer._drain_once() == 3

        assert [e.title for e in poison.inserted] == ["Song 1", "Song 3"]
        assert [e.title for e, _ in poison.rejections] == ["Song 2"]
        # Both arrivals survive. A second retire here pops from the tail and
        # takes them instead — inserted nowhere, recorded nowhere.
        assert await _outbox_wires(fake_redis) == [
            serialize_history_entry(_entry(97)),
            serialize_history_entry(_entry(98)),
        ]

    async def test_isolation_retires_corrupt_elements_too(
        self, fake_redis: Redis
    ) -> None:
        """_isolate iterates RAW BYTES, never the parsed entries.

        A corrupt element parses to None and is absent from `entries`, so an
        isolation pass that walked `entries` would never retire it — and since
        it sits at the head of a non-evictable list, the drain would re-peek the
        same batch forever. That is the exact wedge this path exists to prevent,
        reached from the other direction.

        Found by mutation: rewriting the loop to skip unparseable elements
        passed the whole suite before this test existed.
        """
        await _push(fake_redis, 1)
        await fake_redis.xadd(HISTORY_OUTBOX_KEY, {OUTBOX_FIELD: b"not json"})
        await _push(fake_redis, 2)  # the refused row, newest
        archive = PoisonArchive({"Song 2"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)

        assert await drainer._drain_once() == 3

        assert [e.title for e in archive.inserted] == ["Song 1"]
        assert [e.title for e, _ in archive.rejections] == ["Song 2"]
        # All three settled: inserted, dropped-as-corrupt, recorded. Nothing is
        # left to wedge the head.
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_a_blip_mid_isolation_records_the_refused_row_once(
        self, fake_redis: Redis
    ) -> None:
        """Per-entry retire is what makes an interrupted isolation RESUMABLE.

        Retiring the batch once at the end instead means a transient error
        partway through raises before any retire, so the whole batch redelivers
        and every row already recorded as rejected is recorded AGAIN. The
        destination is now a Postgres table rather than a Redis list, which
        changes where the duplicates land, not whether they happen (reproduced
        against the DLQ: one poison entry, one blip, two copies).
        """
        archive = PoisonArchive({"Song 2"}, transient_titles={"Song 3"})
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, 1, 2, 3)

        # Batch fails on the refused row → isolate. Song 1 lands, Song 2 is
        # recorded, Song 3 hits the blip and raises out of the cycle.
        with pytest.raises(OSError):
            await drainer._drain_once()

        assert [e.title for e in archive.inserted] == ["Song 1"]
        assert [e.title for e, _ in archive.rejections] == ["Song 2"]
        # Only the unsettled tail is left behind — the progress is kept.
        assert await _outbox_wires(fake_redis) == [serialize_history_entry(_entry(3))]

        # Redelivery settles Song 3 without replaying Songs 1 and 2.
        assert await drainer._drain_once() == 1
        assert [e.title for e in archive.inserted] == ["Song 1", "Song 3"]
        assert [e.title for e, _ in archive.rejections] == ["Song 2"]


class TestDrainDeadline:
    """H1. A connected-but-unresponsive Postgres used to hang the drainer with
    no exception — so no backoff, no DEPTH_ALARM, and not one log line."""

    async def test_hung_archive_becomes_a_timeout_error(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class WedgedArchive(RecordsRejections):
            async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
                await asyncio.Event().wait()  # never returns

            async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
                return []

        monkeypatch.setattr(HistoryOutboxDrainer, "DRAIN_DEADLINE_SECS", 0.05)
        drainer = HistoryOutboxDrainer(fake_redis, WedgedArchive())
        await _push(fake_redis, 1)
        with pytest.raises(TimeoutError):
            await drainer._drain_once()
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 1  # nothing retired

    async def test_hang_reaches_the_retry_log_and_reports_depth(
        self,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The observable the original repro showed was ABSENT: a hang must
        # produce the same warning + backlog depth an error does.
        class WedgedArchive(RecordsRejections):
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


class TestConcurrentDrainers:
    """H2, cross-process half — now enforced by the SERVER, not by a lease.

    Under the list transport this needed `history:drainer`: two drainers both
    peeked the same tail and the second's positional retire dropped entries the
    first had not inserted. The stream consumer group removes the mechanism —
    `>` delivers disjoint sets, `0` replays a shared set that ON CONFLICT
    collapses, and settling is by ID — so concurrency costs duplicated work
    instead of destroyed plays.

    Read the limits of this honestly. These tests assert a property of Redis,
    not of our code: disjointness survives deleting the XACK, deleting the XDEL,
    reversing the 0/> order, or switching to a per-process consumer name. They
    are regression guards against a transport change, and the actual evidence
    that the drainer is correct is in the mutation-guard tests elsewhere in this
    file and in test_redis_client.py.
    """

    async def test_a_second_drainer_duplicates_work_but_destroys_nothing(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The C1 scenario, forced — and note what the outcome actually is.

        A short-tick second drainer runs many full cycles while the first is
        parked mid-insert holding all three entries. Under the list transport
        its positional RPOP destroyed them: inserted nowhere, queued nowhere.

        Here it REPLAYS them. Sharing the consumer name means the pending read
        hands the peer the same in-flight set, so it inserts them a second time
        and settles them by ID. That is the designed trade and it is worth
        stating rather than discovering: concurrency costs duplicated work, and
        `ON CONFLICT DO NOTHING` is what makes the duplicate free in production.
        The FakeArchive here has no unique index, so the duplicate is visible —
        hence the assertions are on the set of titles, not the sequence.

        The property under test is that nothing is LOST and that the parked
        drainer's own settle, arriving late against IDs a peer already
        destroyed, is a harmless no-op rather than an error.
        """
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedArchive(FakeArchive):
            """Blocks the FIRST insert only. Later calls run straight through,
            so a drainer that wrongly proceeds actually completes its settle and
            the damage becomes observable — a gate that blocked everyone would
            hide the very bug this test is for."""

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
        monkeypatch.setattr(HistoryOutboxDrainer, "TICK_SECS", 0.01)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3)
        first = HistoryOutboxDrainer(fake_redis, archive)
        second = HistoryOutboxDrainer(fake_redis, archive)
        first.start()
        try:
            async with asyncio.timeout(2):
                await entered.wait()
            # The holder is parked mid-insert; all three are delivered to the
            # shared consumer name and none are acked.
            assert await outbox_pending_count(fake_redis) == 3
            second.start()
            await _eventually(lambda: len(archive.inserted) == 3)
        finally:
            release.set()
            await second.stop()
            await first.stop()
        # Every entry reached the archive exactly once as a DISTINCT play, and
        # the parked drainer's late settle of already-destroyed IDs neither
        # raised nor resurrected anything.
        assert {e.title for e in archive.inserted} == {"Song 1", "Song 2", "Song 3"}
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0
        assert await outbox_pending_count(fake_redis) == 0

    async def test_new_entries_are_split_between_live_drainers(
        self, fake_redis: Redis
    ) -> None:
        # The property that replaced the lease: `>` never hands the same entry
        # to two consumers, so two drainers complement rather than collide.
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2, 3, 4)
        a = FakeArchive()
        b = FakeArchive()
        da = HistoryOutboxDrainer(fake_redis, a)
        db = HistoryOutboxDrainer(fake_redis, b)
        monkeypatch_batch = 2
        da.BATCH_SIZE = monkeypatch_batch  # instance attr, not the class
        db.BATCH_SIZE = monkeypatch_batch
        assert await da._drain_once() == 2
        assert await db._drain_once() == 2
        titles = [e.title for e in a.inserted] + [e.title for e in b.inserted]
        assert sorted(titles) == ["Song 1", "Song 2", "Song 3", "Song 4"]
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_a_dead_drainers_batch_is_inherited_not_stranded(
        self, fake_redis: Redis
    ) -> None:
        """The stable-consumer-name payoff, at the drainer level.

        A process that dies mid-batch leaves entries delivered and unacked. Its
        successor recovers them on its very next cycle by reading `0` under the
        same name — no lease TTL to wait out, no XAUTOCLAIM needed. With a
        per-process name the successor's PEL is empty and those plays wait for
        the periodic sweep instead.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        dead_archive = FakeArchive()
        dead_archive.fail = True
        dead = HistoryOutboxDrainer(fake_redis, dead_archive)
        with pytest.raises(RuntimeError):
            await dead._drain_once()  # delivered, then "crashed"
        assert await outbox_pending_count(fake_redis) == 2

        archive = FakeArchive()
        successor = HistoryOutboxDrainer(fake_redis, archive)
        assert await successor._drain_once() == 2
        assert archive.inserted == [_entry(1), _entry(2)]
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_stop_drains_without_any_gate(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        """stop()'s final flush used to be gated on holding the lease, because
        without it shutdown WAS the concurrent second drainer. That gate is gone
        and nothing replaces it: a concurrent drain is safe by design now, so
        the final drain is an ordinary drain that happens to be the last one.

        The behaviour change worth naming: under a shared consumer name the
        final drain's pending read returns a live peer's in-flight batch, so a
        departing process duplicates the survivor's work rather than
        complementing it. Harmless — dedup plus an idempotent ack — but it means
        shutdown does more work than before, not less.
        """
        await _push(fake_redis, 1)
        await drainer.stop()
        assert archive.inserted == [_entry(1)]
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0


class TestPendingBeforeNew:
    async def test_a_cycle_never_reads_new_while_the_pel_is_dirty(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEMORY INVARIANT, not a FIFO preference.

        The PEL is bounded by BATCH_SIZE x concurrent readers only while a cycle
        never reads `>` with a non-empty PEL. The natural shape of a future
        head-of-line-blocking fix — "keep draining new entries while the stuck
        batch retries" — breaks that bound, so the ordering is asserted by
        command sequence rather than left to the happy path.
        """
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1, 2)
        archive = FakeArchive()
        archive.fail = True
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        with pytest.raises(RuntimeError):
            await drainer._drain_once()  # 2 entries now pending

        await _push(fake_redis, 3)  # new arrival, must NOT be read this cycle
        seen: list[str] = []
        real_xreadgroup = fake_redis.xreadgroup

        async def recording(*args: Any, **kwargs: Any) -> Any:
            seen.append(args[2][HISTORY_OUTBOX_KEY])
            return await real_xreadgroup(*args, **kwargs)

        monkeypatch.setattr(fake_redis, "xreadgroup", recording)
        archive.fail = False
        assert await drainer._drain_once() == 2  # the pending pair only
        assert seen == ["0"]  # `>` was never issued
        assert [e.title for e in archive.inserted] == ["Song 1", "Song 2"]

    async def test_new_entries_are_read_once_the_pel_is_clean(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        await _push(fake_redis, 1)
        assert await drainer._drain_once() == 1
        await _push(fake_redis, 2)
        assert await drainer._drain_once() == 1
        assert [e.title for e in archive.inserted] == ["Song 1", "Song 2"]


class TestTombstones:
    """P1. An entry delivered, then deleted while still pending.

    XTRIM and any operator XDEL both produce this, because neither consults the
    PEL. The ID replays with an empty field map, and a literal fields[b"e"] is a
    KeyError — neither a ResponseError nor a parse failure, so it escapes every
    handler the drain path has, reaches the generic backoff, and replays the
    same entry every cycle. Forever: the drainer never dies, so the respawn
    supervision never fires and nothing escalates past the 60s ceiling, while
    the outbox grows unbounded on a key that cannot be evicted.
    """

    async def _tombstone(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        archive = FakeArchive()
        archive.fail = True
        stuck = HistoryOutboxDrainer(fake_redis, archive)
        with pytest.raises(RuntimeError):
            await stuck._drain_once()  # deliver, do not ack
        ids = [i for i, _ in await _stream_entries(fake_redis)]
        await fake_redis.xdel(HISTORY_OUTBOX_KEY, ids[0])  # body gone, ID pending

    async def test_a_tombstone_is_acked_and_does_not_stall_the_drain(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        await _push(fake_redis, 1, 2)
        await self._tombstone(fake_redis, drainer)
        assert await drainer._drain_once() == 2  # tombstone + the live entry
        assert [e.title for e in archive.inserted] == ["Song 2"]
        assert await outbox_pending_count(fake_redis) == 0
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0
        # And the next cycle is genuinely idle rather than replaying it.
        assert await drainer._drain_once() == 0

    async def test_a_tombstone_is_logged_as_a_lost_play(
        self,
        fake_redis: Redis,
        drainer: HistoryOutboxDrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # ERROR, and worded as loss: the bytes are gone from Redis and nothing
        # can reproduce them. Counted separately from a parse failure, which is
        # a different and recoverable-in-principle fault.
        await _push(fake_redis, 1)
        await self._tombstone(fake_redis, drainer)
        await drainer._drain_once()
        assert any(
            r.levelname == "ERROR" and "payload had already been deleted" in r.message
            for r in caplog.records
        )

    async def test_an_all_tombstone_batch_still_makes_progress(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        # The loop condition is forward progress, not "batch was empty". A
        # cycle that reads only tombstones must still count as progress, or the
        # drain loop stops with entries left to settle.
        await _push(fake_redis, 1)
        await self._tombstone(fake_redis, drainer)
        assert await drainer._drain_once() == 1
        assert await outbox_pending_count(fake_redis) == 0

    async def test_a_tombstone_does_not_reach_the_parser(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        """MUTATION GUARD: routing a tombstone through parse_history_entry.

        A tombstone is not corrupt data — there is no data. Treating it as a
        parse failure would log the wrong thing (recoverable corruption rather
        than an unrecoverable lost play) and, worse, would mean the empty-payload
        case had to survive orjson, which it does only by accident.
        """
        await _push(fake_redis, 1)
        await self._tombstone(fake_redis, drainer)
        with patch(
            "src.history_archive.parse_history_entry", side_effect=AssertionError
        ) as parser:
            await drainer._drain_once()
        parser.assert_not_called()


class TestNogroupHealing:
    async def test_a_deleted_key_is_healed_in_the_drain_cycle(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        """The upgrade note tells operators to `DEL history:outbox`, and DEL
        destroys the consumer group along with the key. XADD then recreates the
        key with NO group, after which every XREADGROUP returns NOGROUP —
        identically, forever, if the bootstrap only ran at startup.

        Under the list transport a DEL merely emptied the outbox and the drainer
        kept working, so this is a foot-gun the stream introduces in the one
        operation we ask operators to perform.
        """
        await fake_redis.delete(HISTORY_OUTBOX_KEY)
        await _push(fake_redis, 1)  # recreates the key, still no group
        assert await drainer._drain_once() == 1
        assert archive.inserted == [_entry(1)]

    async def test_the_heal_is_attempted_once_not_in_a_loop(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Rebuild-and-retry-ONCE, the same shape ytdlp_pool uses for a
        # BrokenProcessPool. A persistent NOGROUP must reach the backoff loop
        # rather than spinning inside one cycle.
        with patch(
            "src.history_archive.ensure_outbox_group", new=AsyncMock()
        ) as ensure:
            with pytest.raises(aioredis.ResponseError, match="NOGROUP"):
                with patch.object(
                    fake_redis,
                    "xreadgroup",
                    AsyncMock(side_effect=aioredis.ResponseError("NOGROUP nope")),
                ):
                    await drainer._drain_once()
        assert ensure.await_count == 1

    async def test_other_response_errors_are_not_healed(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        # Only NOGROUP is recoverable. Healing the class would paper over a
        # WRONGTYPE — a pre-R1 list at the key — by pointlessly recreating a
        # group that cannot exist.
        with patch(
            "src.history_archive.ensure_outbox_group", new=AsyncMock()
        ) as ensure:
            with pytest.raises(aioredis.ResponseError, match="WRONGTYPE"):
                with patch.object(
                    fake_redis,
                    "xreadgroup",
                    AsyncMock(side_effect=aioredis.ResponseError("WRONGTYPE nope")),
                ):
                    await drainer._drain_once()
        ensure.assert_not_awaited()


class TestPelSweep:
    async def test_first_cycle_sweeps(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        with patch(
            "src.history_archive.reclaim_outbox_stale",
            new=AsyncMock(return_value=(0, 0)),
        ) as sweep:
            await drainer._sweep_if_due()
        sweep.assert_awaited_once()

    async def test_second_cycle_within_the_interval_does_not(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        # Everything the sweep catches is rare by construction and it costs an
        # XAUTOCLAIM scan, so it must not ride every TICK_SECS.
        with patch(
            "src.history_archive.reclaim_outbox_stale",
            new=AsyncMock(return_value=(0, 0)),
        ) as sweep:
            await drainer._sweep_if_due()
            await drainer._sweep_if_due()
        assert sweep.await_count == 1

    async def test_a_sweep_failure_does_not_stop_the_drain(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # Housekeeping must never cost the drain. The entries it moves are the
        # exception; the entries the drain moves are the whole point.
        await _push(fake_redis, 1)
        with patch(
            "src.history_archive.reclaim_outbox_stale",
            new=AsyncMock(side_effect=RuntimeError("xautoclaim exploded")),
        ):
            await drainer._sweep_if_due()
        assert await drainer._drain_once() == 1
        assert archive.inserted == [_entry(1)]

    async def test_reclaimed_work_is_logged(
        self,
        fake_redis: Redis,
        drainer: HistoryOutboxDrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # _log_retry runs only on the failure path, so without this a healthy
        # drainer reclaiming a stranded PEL leaves no trace at all. INFO, so
        # caplog's default WARNING threshold has to be lowered.
        caplog.set_level(logging.INFO)
        with patch(
            "src.history_archive.reclaim_outbox_stale",
            new=AsyncMock(return_value=(3, 1)),
        ):
            await drainer._sweep_if_due()
        assert "reclaimed 3" in caplog.text and "purged 1" in caplog.text

    async def test_min_idle_exceeds_the_drain_deadline(self) -> None:
        """INVARIANT, and the reason it moved rather than disappeared.

        Under a shared consumer name "idle" is measured from last DELIVERY, so a
        sweep whose min_idle_time is below the drain deadline reclaims a live
        sibling's batch while it is still inserting. This is the same cross-file
        coupling the deleted `DRAIN_DEADLINE_SECS < DRAINER_LEASE_MS` invariant
        had: removing the lease did not remove the constraint.
        """
        assert (
            HistoryOutboxDrainer.SWEEP_MIN_IDLE_MS
            > HistoryOutboxDrainer.DRAIN_DEADLINE_SECS * 1000
        )

    async def test_sweep_reclaims_a_stranded_foreign_pel_end_to_end(
        self, fake_redis: Redis, archive: Any, drainer: HistoryOutboxDrainer
    ) -> None:
        # The case the shared-name pending read cannot reach: entries left under
        # a consumer name nothing live reads, which a deploy that renamed the
        # consumer produces.
        await _push(fake_redis, 1)
        await fake_redis.xreadgroup(
            HISTORY_OUTBOX_GROUP, "renamed-in-a-past-deploy", {HISTORY_OUTBOX_KEY: ">"}
        )
        assert await drainer._drain_once() == 0  # invisible to us
        drainer.SWEEP_MIN_IDLE_MS = 0
        await drainer._sweep_if_due()
        assert await drainer._drain_once() == 1
        assert archive.inserted == [_entry(1)]


class TestSuccessPathDepthSampling:
    async def test_a_healthy_but_deep_backlog_is_reported(
        self,
        fake_redis: Redis,
        drainer: HistoryOutboxDrainer,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # _log_retry is the only other reader of depth and it runs exclusively
        # in the except arm, so a drainer that is succeeding while falling
        # behind used to report nothing at all.
        monkeypatch.setattr(HistoryOutboxDrainer, "DEPTH_ALARM", 2)
        await _push(fake_redis, 1, 2, 3)
        await drainer._sample_depth()
        assert any(
            r.levelname == "ERROR" and "while the drain is SUCCEEDING" in r.message
            for r in caplog.records
        )

    async def test_a_shallow_backlog_is_silent(
        self,
        fake_redis: Redis,
        drainer: HistoryOutboxDrainer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _push(fake_redis, 1)
        await drainer._sample_depth()
        assert caplog.records == []

    async def test_a_redis_failure_here_is_swallowed(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        # A watchdog, not a metric: it must never be the thing that breaks the
        # loop it is watching.
        with patch.object(
            fake_redis, "xlen", AsyncMock(side_effect=aioredis.ConnectionError("down"))
        ):
            await drainer._sample_depth()  # must not raise

    async def test_the_alarm_fires_from_the_loop_while_the_drain_succeeds(
        self,
        fake_redis: Redis,
        archive: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """REGRESSION: the sample sat below `if drained: continue`.

        Every test above calls _sample_depth directly, so all of them passed
        while the loop could not reach it. "Postgres is keeping up but not
        catching up" IS a succeeding drain, and a succeeding drain takes the
        productive branch every cycle — so the one condition the alarm exists to
        report was the one condition that produced no log line, forever.
        """
        monkeypatch.setattr(HistoryOutboxDrainer, "DEPTH_ALARM", 2)
        monkeypatch.setattr(HistoryOutboxDrainer, "BATCH_SIZE", 1)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await _push(fake_redis, *range(6))  # 6 deep, drained one per cycle
        drainer.start()
        try:
            async with asyncio.timeout(5):
                while not any(
                    "while the drain is SUCCEEDING" in r.message for r in caplog.records
                ):
                    await asyncio.sleep(0.01)
            # …and it really was succeeding, not erroring into _log_retry.
            assert archive.batches
            assert not any("drain failed" in r.message for r in caplog.records)
        finally:
            await drainer.stop(timeout=0.5)

    async def test_the_productive_path_sample_is_rate_limited(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        """A backlogged drain `continue`s straight into the next batch, so an
        ungated sample would add two Redis round trips per BATCH_SIZE entries
        for the whole length of a catch-up — on the Redis serving playback."""
        calls = 0

        async def counting(self: HistoryOutboxDrainer) -> None:
            nonlocal calls
            calls += 1

        with patch.object(HistoryOutboxDrainer, "_sample_depth", counting):
            await drainer._sample_depth_if_due()  # deadline unset → due now
            assert calls == 1
            # The real _sample_depth stamps the deadline; the stub above did
            # not, so do it here and prove the gate then holds.
            drainer._next_depth_sample = (
                asyncio.get_running_loop().time() + drainer.DEPTH_SAMPLE_INTERVAL_SECS
            )
            await drainer._sample_depth_if_due()
            await drainer._sample_depth_if_due()
            assert calls == 1

    async def test_sampling_stamps_its_own_deadline(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer
    ) -> None:
        # _sample_depth stamps rather than leaving it to the _if_due wrapper, so
        # an idle sample and a productive one share one clock. Without this a
        # loop alternating idle and busy cycles samples on both paths and
        # doubles the round trips the rate limit exists to avoid.
        assert drainer._next_depth_sample is None
        await drainer._sample_depth()
        assert drainer._next_depth_sample is not None


class TestStopIdempotence:
    """H2, in-process half. Two concurrent stop()s each ran their own
    peek → insert → retire; the second retired entries the first had not yet
    inserted (reproduced: 6 pushed, 3 archived, outbox empty)."""

    async def test_concurrent_stops_drain_exactly_once(self, fake_redis: Redis) -> None:
        gate = asyncio.Event()

        class GatedArchive(RecordsRejections):
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
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

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
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 5

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
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 3
        assert "over cap" in caplog.text
        assert any(r.levelname == "ERROR" for r in caplog.records)
        # The dropped ones are the OLDEST: entries 2,3,4 go, 5,6,7 stay.
        assert await _outbox_wires(fake_redis) == [
            serialize_history_entry(_entry(n)) for n in (5, 6, 7)
        ]

    async def test_cap_acks_the_doomed_in_flight_set_before_the_trim(
        self,
        fake_redis: Redis,
        archive: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MUTATION GUARD: XACK must precede XTRIM inside _enforce_cap.

        Deleting the ack passes every state-based assertion in this class,
        because the end state self-heals later through the sweep and
        _settle_tombstones — so, exactly as with
        test_retire_acks_before_it_deletes, the order has to be asserted by
        command sequence. Trim-first (or trim-only) leaves the doomed delivered
        entries as tombstones: IDs pending with no body, replaying every cycle
        forever on a non-evictable key.
        """
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 2)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, *range(5))
        # Deliver the oldest three so the PEL holds exactly what the cap will
        # destroy (depth 5, cap 2 → the trim wants the oldest 3 gone).
        delivered = await read_outbox_new(fake_redis, 3)
        doomed = {e.id for e in delivered}
        assert len(doomed) == 3
        seen: list[tuple[str, tuple[Any, ...]]] = []
        real_exec = type(fake_redis).execute_command

        async def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append((str(args[0]), args[1:]))
            return await real_exec(self, *args, **kwargs)

        monkeypatch.setattr(type(fake_redis), "execute_command", recording)
        await drainer._enforce_cap()
        names = [name for name, _ in seen]
        assert "XACK" in names, "the doomed pending set was never acked"
        assert "XTRIM" in names
        assert names.index("XACK") < names.index("XTRIM")
        # The ack named exactly the delivered-and-doomed IDs…
        acked = next(args for name, args in seen if name == "XACK")
        assert set(acked[2:]) == doomed
        # …so no pending record outlived its body.
        assert await outbox_pending_count(fake_redis) == 0
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 2

    async def test_cap_minid_discovery_is_paged_and_converges(
        self,
        fake_redis: Redis,
        archive: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REGRESSION: the minid discovery used to be one XRANGE COUNT=over+1.

        XRANGE replies carry bodies, so that single fetch scaled with the
        backlog — enabling the cap against a 500k-entry outbox would haul
        ~240 MB over the socket in one reply, the stream re-creation of the
        206 MB `RPOP key 490000` incident trim_outbox_below documents as
        closed. The fetch must stay bounded by CAP_PAGE per pass and the cap
        must converge across passes instead.
        """
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 3)
        monkeypatch.setattr(HistoryOutboxDrainer, "CAP_PAGE", 2)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, *range(10))  # over = 7, pages of 2
        counts: list[int] = []
        real_exec = type(fake_redis).execute_command

        async def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Guard on the keyword (bytes on the wire): the assertion helpers
            # below also XRANGE, count-less, and their args[-1] is the "+"
            # max bound.
            if str(args[0]) == "XRANGE" and args[-2] in (b"COUNT", "COUNT"):
                counts.append(int(args[-1]))
            return await real_exec(self, *args, **kwargs)

        monkeypatch.setattr(type(fake_redis), "execute_command", recording)
        await drainer._enforce_cap()
        assert counts, "no XRANGE ran"
        assert max(counts) <= drainer.CAP_PAGE + 1
        assert len(counts) >= 2  # it actually paged, not one overage-sized fetch
        # And still converged fully in one invocation, dropping the OLDEST.
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 3
        assert await _outbox_wires(fake_redis) == [
            serialize_history_entry(_entry(n)) for n in (7, 8, 9)
        ]

    async def test_cap_does_nothing_while_shutting_down(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A departing process has the least information about what a live peer
        is doing and no reason to be destroying plays at all. The check is per
        PASS, not per call, so it also halts a convergence loop already running
        — which is what keeps stop()'s final drain from racing a long trim."""
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 2)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, *range(6))
        drainer._stopping = True
        await drainer._enforce_cap()
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 6

    async def test_cap_returns_when_the_page_comes_back_short(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Depth and the XRANGE are two round trips, so a concurrent drain can
        settle the stream shorter in between. The page then cannot name a
        (page+1)-th ID, and inventing a minid from the entries it DID get would
        trim live entries the depth check never authorised."""
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 2)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, *range(3))
        # Depth lies high (as a drain that lands mid-pass makes it): the cap
        # believes 9 entries exist and asks for a page the stream cannot fill.
        monkeypatch.setattr(
            "src.history_archive.outbox_depth", AsyncMock(return_value=9)
        )
        await drainer._enforce_cap()
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 3

    async def test_cap_gives_up_when_a_trim_removes_nothing(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The convergence loop's termination guard. Depth is XLEN, which
        over-counts acked-but-undeleted entries, so a depth permanently above
        the cap with a trim that can no longer remove anything is reachable —
        and without the zero-dropped exit the loop would spin on it forever,
        inside the drain cycle, logging an ERROR per pass."""
        monkeypatch.setattr(HistoryOutboxDrainer, "OUTBOX_MAX", 2)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, *range(6))
        trim = AsyncMock(return_value=0)
        monkeypatch.setattr("src.history_archive.trim_outbox_below", trim)
        async with asyncio.timeout(5):  # a spin here HANGS rather than fails
            await drainer._enforce_cap()
        trim.assert_awaited_once()

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
                while await fake_redis.xlen(HISTORY_OUTBOX_KEY) != 3:
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
        the loop spins at full speed issuing an XREADGROUP per pass — forever,
        on an empty outbox. One pegged core and a permanent Redis load floor, on
        the Redis that also serves playback.

        Two things about the measurement, both learned by mutating _wake.clear()
        away and watching what the test did:

        It counts XREADGROUP. The test was written against the LIST outbox and
        kept its LRANGE instrument across the switch to a stream, where the
        drain path never issues one — so its assertion had become 0 == 0,
        proving nothing about the loop it names.

        It samples event-loop TURNS against an absolute ceiling, not two timed
        windows compared to each other. Both of those matter. Wall-clock windows
        get starved by the very spin they are measuring, so the old shape never
        reached its assertion and reported the regression as a 120s
        pytest-timeout — a cancellation that burns a CI job's deadline instead
        of naming the defect. And a window-to-window comparison is defeated by
        the spin's own consequences: measured against the real mutant, the loop
        issued 57,600 reads in the first window and exactly 0 in the second,
        because by then it had died and entered the supervisor's 5s respawn
        damping. Both windows agreed, and the test passed.

        The absolute form has neither weakness: an idle loop parked in
        _wake.wait() issues at most the one follow-up backlog check, whatever
        else is happening on the loop."""
        monkeypatch.setattr(HistoryOutboxDrainer, "TICK_SECS", 5.0)
        peeks = {"n": 0}
        real_exec = type(fake_redis).execute_command

        async def counting_exec(self: Any, *args: Any, **kwargs: Any) -> Any:
            if str(args[0]) == "XREADGROUP":
                peeks["n"] += 1
            return await real_exec(self, *args, **kwargs)

        async def turns(n: int) -> None:
            for _ in range(n):
                await asyncio.sleep(0)

        monkeypatch.setattr(type(fake_redis), "execute_command", counting_exec)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        drainer.start()
        try:
            await _push(fake_redis, 1)
            drainer.notify()
            await _eventually(lambda: archive.inserted == [_entry(1)])
            peeks["n"] = 0
            await turns(400)
            # TICK_SECS is 5s and the outbox is now empty, so an idling loop
            # spends this window parked in _wake.wait(). The allowance covers
            # the follow-up cycle a productive one runs to check for more
            # backlog, at two reads each (pending replay, then new). Measured:
            # 4 idle against 111,614 spinning, so the exact ceiling is not
            # load-bearing — only its order of magnitude.
            assert peeks["n"] <= 8
        finally:
            # Bounded, and cancelled outright if it overruns: a drainer that is
            # actually spinning cannot be joined by a graceful stop, so an
            # unbounded teardown here would swallow the assertion above and
            # re-report the defect as a 120s deadline expiry.
            task = drainer._task
            try:
                async with asyncio.timeout(1):
                    await drainer.stop(timeout=0.5)
            except TimeoutError:
                if task is not None:
                    task.cancel()

    async def test_the_tick_timeout_wakes_a_loop_nobody_notified(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the idle wait: TICK_SECS expiring, not notify().

        Everything else here pushes AND notifies, so the timeout arm was never
        taken. It is the only thing that moves an entry pushed by a process
        whose notify never landed — a second bot instance on the same Redis, or
        a notify dropped in the window the clear() comment describes.
        """
        monkeypatch.setattr(HistoryOutboxDrainer, "TICK_SECS", 0.05)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        drainer.start()
        try:
            await _eventually(lambda: drainer._task is not None)
            await asyncio.sleep(0.1)  # let the loop reach its idle wait
            await _push(fake_redis, 1)  # NO notify() — the timeout must do it
            await _eventually(lambda: archive.inserted == [_entry(1)])
        finally:
            await drainer.stop(timeout=0.5)

    async def test_a_respawn_that_lands_after_stop_does_not_restart(
        self, fake_redis: Redis, archive: Any
    ) -> None:
        # _respawn is a call_later callback, so it can fire between stop()
        # cancelling the handle and the loop actually running it. Restarting
        # there resurrects a drainer nothing will ever stop again.
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        drainer._stopping = True
        drainer._respawn()
        assert drainer._task is None

    async def test_retry_logs_an_unknowable_depth_when_redis_is_down(
        self, fake_redis: Redis, drainer: HistoryOutboxDrainer, caplog: Any
    ) -> None:
        # Redis being the thing that broke is the common case for this arm —
        # the drain failed because Redis failed — so the retry line must still
        # be emitted, with the backlog reported as unknown rather than as 0.
        # A 0 there reads as "nothing is waiting", the opposite of the truth.
        with patch.object(
            fake_redis, "xlen", AsyncMock(side_effect=aioredis.ConnectionError("down"))
        ):
            await drainer._log_retry(RuntimeError("pg down"), backoff=1.0)
        assert "backlog=-1" in caplog.text

    async def test_retry_backoff_doubles_then_holds_at_the_ceiling(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ladder itself, which no test read. Every existing failure-path
        test asserts only THAT a retry happened, so both halves of
        `min(backoff * 2, _BACKOFF_MAX)` were free to regress silently: a
        constant backoff turns a Postgres outage into a hot retry loop against
        the Redis that also serves playback, and a missing ceiling walks the
        delay past any useful recovery time (2^20 s is 12 days).

        Reads the delay off _log_retry, which receives the same value the sleep
        gets and is the number the operator sees in the log line."""
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_START", 0.01)
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_MAX", 0.04)
        delays: list[float] = []
        real_log_retry = HistoryOutboxDrainer._log_retry

        async def recording(
            self: HistoryOutboxDrainer, error: Exception, backoff: float
        ) -> None:
            delays.append(backoff)
            await real_log_retry(self, error, backoff)

        monkeypatch.setattr(HistoryOutboxDrainer, "_log_retry", recording)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        archive.fail = True  # every cycle raises, forever
        await _push(fake_redis, 1)
        drainer.start()
        try:
            async with asyncio.timeout(5):
                while len(delays) < 5:
                    await asyncio.sleep(0.01)
        finally:
            await drainer.stop(timeout=0.5)
        # Doubling from the base, then pinned at the ceiling — not merely
        # non-decreasing, which a constant ladder also satisfies.
        assert delays[:5] == [0.01, 0.02, 0.04, 0.04, 0.04]

    async def test_a_successful_cycle_resets_the_backoff(
        self, fake_redis: Redis, archive: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half: the ladder must fall back to the base once the
        archive recovers, or the first blip of a long-lived process permanently
        slows every later retry."""
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_START", 0.01)
        monkeypatch.setattr(HistoryOutboxDrainer, "_BACKOFF_MAX", 0.08)
        delays: list[float] = []
        real_log_retry = HistoryOutboxDrainer._log_retry

        async def recording(
            self: HistoryOutboxDrainer, error: Exception, backoff: float
        ) -> None:
            delays.append(backoff)
            if len(delays) == 3:
                archive.fail = False  # Postgres comes back
            await real_log_retry(self, error, backoff)

        monkeypatch.setattr(HistoryOutboxDrainer, "_log_retry", recording)
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        archive.fail = True
        await _push(fake_redis, 1)
        drainer.start()
        try:
            await _eventually(lambda: archive.inserted == [_entry(1)])
            assert delays == [0.01, 0.02, 0.04]
            # Fail again after the recovery: the ladder must restart at the
            # base, not resume from 0.08.
            archive.fail = True
            await _push(fake_redis, 2)
            drainer.notify()
            async with asyncio.timeout(5):
                while len(delays) < 4:
                    await asyncio.sleep(0.01)
        finally:
            await drainer.stop(timeout=0.5)
        assert delays[3] == 0.01


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
        Catching only CancelledError let that escape stop(), which skipped the
        final drain — every entry the departing process was holding then waited
        for whenever something started again."""
        drainer = HistoryOutboxDrainer(fake_redis, archive)
        await ensure_outbox_group(fake_redis)
        await _push(fake_redis, 1)

        async def boom() -> None:
            raise RuntimeError("drainer blew up")

        task = asyncio.ensure_future(boom())
        await asyncio.sleep(0)  # let it run and fail
        assert task.done() and not task.cancelled()
        drainer._task = task

        await drainer.stop()  # must not raise

        # And the final drain still ran: catching only CancelledError let the
        # task's own exception escape stop(), which skipped the flush entirely.
        assert archive.inserted == [_entry(1)]


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


class TestRecordRejection:
    """play_history_rejected (migrations/0001_play_history.sql). Expected to stay
    empty forever,
    which is exactly why the path has to be tested — nothing in production will
    exercise it before the day it matters."""

    @staticmethod
    def _archive_with_conn(conn: Any) -> PostgresHistoryArchive:
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        pool = MagicMock()
        acquired = MagicMock()
        acquired.__aenter__ = AsyncMock(return_value=conn)
        # return_value=None, not a bare AsyncMock: a truthy __aexit__ SUPPRESSES
        # the exception under test and silently turns a failing assertion green.
        acquired.__aexit__ = AsyncMock(return_value=None)
        pool.acquire = MagicMock(return_value=acquired)
        archive._pool = pool
        return archive

    async def test_writes_the_row_with_the_wire_payload(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        archive = self._archive_with_conn(conn)
        entry = _entry(1)

        await archive.record_rejection(
            entry, asyncpg.exceptions.CheckViolationError("boom"), "abc123"
        )

        assert conn.execute.await_args is not None
        sql, guild_id, error_type, detail, trace_id, payload = (
            conn.execute.await_args.args
        )
        assert "play_history_rejected" in sql
        assert guild_id == 42
        assert error_type == "CheckViolationError"
        assert "boom" in detail
        assert trace_id == "abc123"
        # Verbatim wire bytes — that is what makes a replay exact, and why the
        # column is bytea rather than jsonb or text.
        assert payload == serialize_history_entry(entry)

    async def test_nul_in_the_error_message_is_scrubbed(self) -> None:
        # asyncpg echoes the offending value, so the very poison this table
        # records would otherwise fail the insert recording it. error_detail is
        # text; only payload is bytea.
        conn = MagicMock()
        conn.execute = AsyncMock()
        archive = self._archive_with_conn(conn)

        await archive.record_rejection(_entry(1), ValueError("bad\x00value"))

        detail = conn.execute.await_args.args[3]
        assert "\x00" not in detail
        assert detail == "badvalue"

    async def test_long_error_message_is_capped(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        archive = self._archive_with_conn(conn)

        await archive.record_rejection(_entry(1), ValueError("x" * 10_000))

        assert len(conn.execute.await_args.args[3]) == 2000

    async def test_a_failing_reject_insert_logs_and_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # TERMINAL by design: a rejects table that can fail into a retry loop is
        # worse than no rejects table. The caller is already handling one error.
        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=OSError("connection reset"))
        archive = self._archive_with_conn(conn)

        await archive.record_rejection(_entry(1), ValueError("original"))

        assert "unrecordable" in caplog.text
        assert any(r.levelname == "ERROR" for r in caplog.records)
        # The payload still reaches the operator, in the log, so nothing is lost
        # silently.
        assert "Song 1" in caplog.text

    async def test_a_closed_archive_does_not_raise_either(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # close() is terminal, and -ping's health probe can race shutdown, so
        # _ensure() raising must not escape onto an error path.
        archive = PostgresHistoryArchive("postgresql://nope:1/nope")
        await archive.close()
        await archive.record_rejection(_entry(1), ValueError("original"))
        assert "unrecordable" in caplog.text


class TestPoisonClassification:
    """What _POISON must and must not catch.

    Since the schema lock this tuple is expected to be unreachable, but it is
    still the difference between one bad row costing one row and a permanent
    wedge — so the membership is asserted directly rather than inferred from
    drainer behaviour.
    """

    def test_check_violation_is_poison(self) -> None:
        # SQLSTATE 23514, and NOT a DataError: it inherits from
        # IntegrityConstraintViolationError. The play_history CHECK
        # constraints make this reachable, and without the arm a violation would propagate past the
        # quarantine path and wedge the drain head on a non-evictable list.
        assert not issubclass(
            asyncpg.exceptions.CheckViolationError, asyncpg.exceptions.DataError
        )
        assert isinstance(asyncpg.exceptions.CheckViolationError("x"), _POISON)

    def test_not_null_violation_is_poison(self) -> None:
        assert isinstance(asyncpg.exceptions.NotNullViolationError("x"), _POISON)

    def test_data_error_is_poison(self) -> None:
        assert isinstance(asyncpg.exceptions.DataError("x"), _POISON)

    def test_unique_violation_is_not_poison(self) -> None:
        # play_history_dedup is the ON CONFLICT target, so this cannot surface.
        # Catching it would dead-letter rows on a genuine index bug instead of
        # surfacing it.
        assert not isinstance(asyncpg.exceptions.UniqueViolationError("x"), _POISON)

    def test_transport_errors_are_not_poison(self) -> None:
        # The regression this guards: OSError and its subclasses are how a
        # restart, failover or timeout presents. Dead-lettering on those would
        # delete healthy history on every outage.
        for exc in (
            OSError(84, "Value too large"),
            ConnectionResetError("reset"),
            TimeoutError("deadline"),
            asyncpg.exceptions.PostgresConnectionError("down"),
        ):
            assert not isinstance(exc, _POISON), exc
