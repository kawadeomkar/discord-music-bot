"""Tests for src/backfill_history.py — moving pre-archive Redis history into
Postgres.

The two properties that matter and that no other test covers: legacy entries
(written before HistoryEntry carried a guild_id) must be stamped from their
KEY, or every guild's old rows collide with every other guild's on the
(guild_id, played_at, webpage_url) dedup index; and the run must be resumable,
because a backfill over a real deployment's history will be interrupted.
"""

import sys
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from redis.asyncio import Redis

from src import backfill_history
from src.backfill_history import backfill
from src.guild_state import HistoryEntry, serialize_history_entry
from src.redis_client import GUILD_HISTORY_KEY, HISTORY_OUTBOX_KEY


def _entry(n: int, guild_id: int = 0) -> HistoryEntry:
    return HistoryEntry(
        guild_id=guild_id,
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=200,
        played_at=1000.0 + n,
    )


class CollectingArchive:
    def __init__(self) -> None:
        self.rows: list[HistoryEntry] = []
        self.batches = 0
        self.health_checks = 0
        # Not part of the HistoryArchive protocol, but _run() owns the archive's
        # lifecycle and must close it — asserted by TestCli.
        self.closed = AsyncMock(return_value=None)

    async def insert_batch(self, entries: Any) -> None:
        self.batches += 1
        self.rows.extend(entries)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        return []

    async def health_check(self) -> None:
        # Not on the HistoryArchive protocol — _run holds the concrete
        # PostgresHistoryArchive and preflights it before the walk, so that the
        # dry run learns whether the destination is reachable at all. A double
        # without this reports the preflight failing, which is the correct
        # signal: it means the fake is not modelling what the CLI uses.
        self.health_checks += 1

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None:
        # The backfill never calls this — it is here only because backfill()
        # takes the whole HistoryArchive protocol, which is the union of three
        # consumers' needs (GuildHistory reads, the backfill writes, the drainer
        # does both plus rejections). A rejection recorded from here would be a
        # bug, so it raises rather than collecting.
        #
        # `wire` arrived with the archive's dead-letter fix: the drainer parks
        # the DELIVERED bytes rather than a re-serialization. Irrelevant to this
        # double's behaviour, but the signature has to match or the protocol
        # check fails — which is the point of keeping the full shape here.
        raise AssertionError("the backfill must not record rejections")

    async def close(self) -> None:
        await self.closed()


async def _seed(redis: Redis, guild_id: int, *entries: HistoryEntry) -> None:
    key = GUILD_HISTORY_KEY.format(guild_id=guild_id)
    for entry in entries:
        await redis.lpush(key, serialize_history_entry(entry))


class TestBackfill:
    async def test_moves_every_guilds_history(self, fake_redis: Redis) -> None:
        await _seed(fake_redis, 1, _entry(1, guild_id=1), _entry(2, guild_id=1))
        await _seed(fake_redis, 2, _entry(3, guild_id=2))
        archive = CollectingArchive()

        report = await backfill(fake_redis, archive)

        assert report.guilds == 2
        assert report.scanned == 3
        assert report.attempted == 3
        assert {e.title for e in archive.rows} == {"Song 1", "Song 2", "Song 3"}

    async def test_concurrent_plays_cannot_slide_entries_out_of_the_window(
        self, fake_redis: Redis
    ) -> None:
        """REGRESSION: paging used head-relative indices (`lrange(key, start,
        start + page - 1)`) while push_history LPUSHes at the HEAD, so every
        song that finished mid-run shifted the window and the oldest entries
        fell out unread — reproduced as 3 songs during a 10-entry backfill
        skipping OLD0/OLD1/OLD2 while the report still said 10. Cutover then
        LTRIMs the list, so those plays were gone for good.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=7)
        await _seed(fake_redis, 7, *(_entry(n, guild_id=7) for n in range(10)))

        # Interleave live plays with the scan exactly as the bot would: one new
        # entry LPUSHed at the head between pages.
        real_lrange = fake_redis.lrange

        async def pushing_lrange(name: str, start: int, end: int) -> Any:
            out = await real_lrange(name, start, end)
            await fake_redis.lpush(key, serialize_history_entry(_entry(99, guild_id=7)))
            return out

        archive = CollectingArchive()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fake_redis, "lrange", pushing_lrange)
            await backfill(fake_redis, archive, page=5)

        # Every pre-existing entry was read. Newly pushed ones may or may not
        # appear (they reach Postgres via the outbox either way) — what must
        # never happen is an original going missing.
        archived = {e.title for e in archive.rows}
        assert {f"Song {n}" for n in range(10)} <= archived

    async def test_legacy_entries_are_stamped_from_their_key(
        self, fake_redis: Redis
    ) -> None:
        # guild_id=0 is the pre-migration wire shape. Left as 0, guild 1's and
        # guild 2's rows would dedup against each other.
        await _seed(fake_redis, 111, _entry(1))
        await _seed(fake_redis, 222, _entry(1))
        archive = CollectingArchive()

        await backfill(fake_redis, archive)

        assert sorted(e.guild_id for e in archive.rows) == [111, 222]

    async def test_existing_guild_id_is_preserved(self, fake_redis: Redis) -> None:
        # Only the zero sentinel is stamped — an entry that already knows its
        # guild must not be rewritten from the key.
        await _seed(fake_redis, 111, _entry(1, guild_id=999))
        archive = CollectingArchive()
        await backfill(fake_redis, archive)
        assert [e.guild_id for e in archive.rows] == [999]

    async def test_dry_run_writes_nothing_but_still_counts(
        self, fake_redis: Redis
    ) -> None:
        await _seed(fake_redis, 1, _entry(1), _entry(2))
        archive = CollectingArchive()
        report = await backfill(fake_redis, archive, dry_run=True)
        assert report.attempted == 2
        assert archive.rows == []

    async def test_corrupt_entries_are_counted_and_skipped(
        self, fake_redis: Redis
    ) -> None:
        await _seed(fake_redis, 1, _entry(1))
        await fake_redis.lpush(GUILD_HISTORY_KEY.format(guild_id=1), b"not json")
        archive = CollectingArchive()
        report = await backfill(fake_redis, archive)
        assert report.corrupt == 1
        assert [e.title for e in archive.rows] == ["Song 1"]

    async def test_pages_large_guilds(self, fake_redis: Redis) -> None:
        await _seed(fake_redis, 1, *[_entry(n) for n in range(7)])
        archive = CollectingArchive()
        await backfill(fake_redis, archive, page=3)
        assert archive.batches == 3  # 3 + 3 + 1
        assert len(archive.rows) == 7

    async def test_never_touches_the_outbox(self, fake_redis: Redis) -> None:
        # Routing a historical backlog through the outbox would bury the live
        # drain behind it — the whole reason this writes to the archive direct.
        #
        # XLEN, not LLEN: the outbox became a stream, and LLEN on the key this
        # test cares about answers 0 while it is absent (passing for the wrong
        # reason) and raises WRONGTYPE the moment it is not (failing for the
        # wrong reason). Neither reports the thing the test is named after.
        await _seed(fake_redis, 1, _entry(1))
        await backfill(fake_redis, CollectingArchive())
        assert await fake_redis.xlen(HISTORY_OUTBOX_KEY) == 0

    async def test_leaves_the_redis_lists_intact(self, fake_redis: Redis) -> None:
        # Non-destructive: Phase C's trim is a separate, later, gated step.
        await _seed(fake_redis, 1, _entry(1), _entry(2))
        await backfill(fake_redis, CollectingArchive())
        assert await fake_redis.llen(GUILD_HISTORY_KEY.format(guild_id=1)) == 2

    async def test_rerun_is_idempotent_at_the_archive(self, fake_redis: Redis) -> None:
        # The tool itself re-sends everything; ON CONFLICT DO NOTHING is what
        # makes that safe, so what is asserted here is that a rerun produces
        # the same rows rather than erroring.
        await _seed(fake_redis, 1, _entry(1))
        archive = CollectingArchive()
        first = await backfill(fake_redis, archive)
        second = await backfill(fake_redis, archive)
        assert first.attempted == second.attempted == 1

    async def test_legacy_poison_arrives_already_clamped(
        self, fake_redis: Redis
    ) -> None:
        """The PREMISE this tool relies on, characterised at its boundary.

        Renamed from test_sanitizes_poison_entries, whose comment said "the
        backfill has to apply it too" — the opposite of what the source says
        ("No sanitize call"), left over from a revision where `_sanitize_entry`
        was called here. Nothing in this module sanitizes anything; every
        assertion below holds before a single line of backfill_history runs,
        because parse_history_entry constructs a HistoryEntry and
        __post_init__ clamps.

        Kept rather than deleted, and kept HERE rather than moved to
        test_guild_state, because it is the assumption that lets this module
        hand legacy bytes straight to executemany. If the clamp ever moves or
        weakens, this is the test that should make the backfill's reviewer
        look — the drainer has _isolate as a backstop and this tool does not.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await fake_redis.lpush(
            key, orjson.dumps({"title": "bad\x00title", "played_at": 1e18})
        )
        archive = CollectingArchive()
        await backfill(fake_redis, archive)
        assert archive.rows[0].title == "badtitle"  # NUL scrubbed
        assert archive.rows[0].played_at == 0.0  # absurd epoch → sentinel

    async def test_an_all_corrupt_page_writes_nothing(self, fake_redis: Redis) -> None:
        # The `entries and` half of the write guard: no test produced a page
        # whose entries were ALL corrupt, so dropping it survived. Harmless
        # today (insert_batch early-returns on empty), but branch coverage
        # reports the line as fully branched, so the gap is invisible.
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        for _ in range(3):
            await fake_redis.lpush(key, b"not json")
        archive = CollectingArchive()

        report = await backfill(fake_redis, archive, page=3)

        assert report.corrupt == 3
        assert report.attempted == 0
        assert archive.batches == 0  # not called with an empty list

    async def test_corrupt_entries_log_their_bytes(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`corrupt: N` was the entire forensic record of entries the archive
        build is about to trim away.

        This is where the backfill has to diverge from the drainer, which drops
        corrupt entries just as quietly: an unparseable OUTBOX entry still has
        its twin on the history list, so nothing is lost. Here that list IS the
        copy. Logging is the only durable record left.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await fake_redis.lpush(key, b'{"deliberately": "unparseable"')
        await backfill(fake_redis, CollectingArchive())

        assert "corrupt history entry" in caplog.text
        assert "deliberately" in caplog.text  # the BYTES, not just a count
        assert "guild:1:history" in caplog.text  # and where they were
        assert any(r.levelname == "ERROR" for r in caplog.records)

    async def test_a_list_that_shrinks_from_the_tail_is_flagged(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reconciliation in the direction that can mean loss.

        `attempted` exceeding `total` is normal — the last page's clamped
        window picks up concurrent pushes, which dedup absorbs. Coming up SHORT
        means entries the initial LLEN counted were never read, which only a
        tail-side shrink causes. Nothing shrinks these lists from the tail
        today; the bot's own LTRIM does, and the entries it eats are the
        OLDEST — exactly the ones this tool exists to save. Simulated here with
        RPOP between pages.

        The VERDICT is the assertion that matters. This test used to check the
        log line alone, and that is precisely how the defect shipped: the
        shrink was detected, warned about, and then folded into `guilds` as a
        guild that had moved in full, so `ok` stayed True and the tool exited 0
        over plays it had just watched disappear — while the README tells the
        operator to gate an irreversible deploy on that exit code.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await _seed(fake_redis, 1, *(_entry(n, guild_id=1) for n in range(6)))
        real_lrange = fake_redis.lrange

        async def shrinking_lrange(name: str, start: int, end: int) -> Any:
            out = await real_lrange(name, start, end)
            await fake_redis.rpop(key)  # a trim eats the oldest, mid-run
            return out

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fake_redis, "lrange", shrinking_lrange)
            report = await backfill(fake_redis, CollectingArchive(), page=2)

        assert "trimmed from the tail" in caplog.text
        assert report.short_guilds == 1
        # NOT counted as backfilled: that number is what the summary reports.
        assert report.guilds == 0
        assert not report.ok

    async def test_a_trim_that_keeps_the_length_is_still_caught(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The steady-state trim, which no count can see.

        push_history LPUSHes and LTRIMs in ONE transaction, so a list already at
        HISTORY_CACHE_LIMIT keeps its length exactly while every song end
        destroys one unread tail entry. `attempted + corrupt < total` is
        therefore always false here — the old reconciliation was structurally
        blind to it — yet pre-archive plays are being lost the whole time.

        This is not an exotic race: it is the shape of every active guild on a
        deployment already running this build with the archive off, i.e. exactly
        the population the README's "enable the archive later" flow sends to
        this tool. Caught by comparing the OLDEST entry's bytes across the walk
        rather than counting.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await _seed(fake_redis, 1, *(_entry(n, guild_id=1) for n in range(6)))
        real_lrange = fake_redis.lrange
        pushed = 100

        async def churning_lrange(name: str, start: int, end: int) -> Any:
            out = await real_lrange(name, start, end)
            # One song end: LPUSH a new play, LTRIM the oldest away. Net length
            # unchanged — which is the entire point.
            nonlocal pushed
            await fake_redis.lpush(
                key, serialize_history_entry(_entry(pushed, guild_id=1))
            )
            await fake_redis.rpop(key)
            pushed += 1
            return out

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fake_redis, "lrange", churning_lrange)
            report = await backfill(fake_redis, CollectingArchive(), page=2)

        # The count check alone would have said everything was fine.
        assert report.scanned == report.attempted
        assert report.short_guilds == 1
        assert not report.ok
        assert "trimmed from the tail" in caplog.text

    async def test_ignores_keys_with_an_unparseable_guild_id(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        await fake_redis.lpush(
            "guild:notanid:history", serialize_history_entry(_entry(1))
        )
        archive = CollectingArchive()
        report = await backfill(fake_redis, archive)
        assert report.guilds == 0
        assert archive.rows == []
        assert "unparseable guild id" in caplog.text
        # The key still holds real plays that did not move, so the run is not
        # clean — skipping it silently let the verdict claim otherwise.
        assert report.skipped_keys == 1
        assert not report.ok

    async def test_the_corrupt_entry_offset_points_at_the_corrupt_entry(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The offset is the forensic half of the only record these plays get.

        A page runs newest→oldest internally while `start` advances
        oldest→newest, so the distance from the oldest end is
        `start + (len-1-i)`. The old expression walked the page the wrong way
        and mixed a guild-cumulative `corrupt` with a page-local
        `len(entries)`: on this list — 5 entries, the OLDEST corrupt — it
        reported tail offset 4, pointing the operator at a healthy entry.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        # LPUSH order: the FIRST pushed ends up oldest (tail offset 0).
        await fake_redis.lpush(key, b"not json")
        for n in range(1, 5):
            await fake_redis.lpush(key, serialize_history_entry(_entry(n, guild_id=1)))

        await backfill(fake_redis, CollectingArchive(), page=5)

        assert "at tail offset 0," in caplog.text

    async def test_the_corrupt_entry_dump_is_not_truncated_below_a_real_entry(
        self, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        # config.py sizes an entry at ~420 bytes and a realistic one measures
        # 545, so the old 400-byte cap stopped mid-thumbnail and never reached
        # uploader / played_at / message_id — the fields needed to attribute
        # the play. Build something comfortably past the old cap and assert the
        # tail of it survives.
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        marker = "END-OF-PAYLOAD-MARKER"
        wire = b'{"title": "' + b"x" * 600 + marker.encode() + b'"'  # unparseable
        await fake_redis.lpush(key, wire)

        await backfill(fake_redis, CollectingArchive())

        assert marker in caplog.text

    async def test_a_key_returned_twice_by_scan_is_counted_once(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SCAN is at-LEAST-once: a rehash can hand back the same key again.

        The rows are idempotent so nothing durable breaks, but `guilds`,
        `scanned` and `attempted` double — and README tells the operator to
        verify those counts against `SELECT count(*) FROM play_history` before
        an irreversible deploy, while the module docstring calls them "the only
        signal".
        """
        await _seed(fake_redis, 1, _entry(1, guild_id=1), _entry(2, guild_id=1))
        key = GUILD_HISTORY_KEY.format(guild_id=1).encode()

        async def duplicating_scan(*args: Any, **kwargs: Any) -> Any:
            for k in (key, key):  # the same key, twice
                yield k

        monkeypatch.setattr(fake_redis, "scan_iter", duplicating_scan)
        archive = CollectingArchive()

        report = await backfill(fake_redis, archive)

        assert report.guilds == 1
        assert report.scanned == 2
        assert report.attempted == 2
        assert len(archive.rows) == 2

    async def test_each_page_is_inserted_oldest_first(self, fake_redis: Redis) -> None:
        # insert_batch documents "Insert oldest-first". A page arrives
        # newest→oldest while pages advance oldest→newest, so unreversed the
        # global order was neither — and this tool and the drainer could touch
        # overlapping keys in opposite orders, which is how a deadlock costs a
        # whole guild here rather than one row.
        await _seed(fake_redis, 1, *(_entry(n, guild_id=1) for n in range(4)))
        archive = CollectingArchive()

        await backfill(fake_redis, archive, page=2)

        assert [e.title for e in archive.rows] == [
            "Song 0",
            "Song 1",
            "Song 2",
            "Song 3",
        ]

    async def test_empty_redis_is_a_noop(self, fake_redis: Redis) -> None:
        report = await backfill(fake_redis, CollectingArchive())
        assert report.guilds == 0
        assert report.attempted == 0

    async def test_only_history_keys_are_read(self, fake_redis: Redis) -> None:
        """The key pattern is a safety predicate, and nothing pinned it — the
        mutation `_HISTORY_KEY_MATCH = "*"` passed every test, because no test
        seeded a non-history key.

        It matters more than "it would crash on WRONGTYPE" suggests, because
        parse_history_entry accepts ANY json object and defaults every field:
        `guild:{id}:queue` is also a LIST, so a pattern drift in that direction
        inserts never-played songs, counted as `attempted` rather than
        `corrupt` — no crash, no signal, wrong rows in the durable record.
        """
        await _seed(fake_redis, 1, _entry(1, guild_id=1))
        # One of every other key family in the Redis schema.
        await fake_redis.hset("guild:1:state", "volume", "50")
        await fake_redis.rpush(
            "guild:1:queue", orjson.dumps({"webpage_url": "https://yt.com/v=q"})
        )
        await fake_redis.hset("guild:1:now_playing", "title", "x")
        await fake_redis.set("ytdl:source:whatever", b"{}")
        await fake_redis.xadd(HISTORY_OUTBOX_KEY, {b"e": b"{}"})
        archive = CollectingArchive()

        report = await backfill(fake_redis, archive)

        assert report.guilds == 1
        assert [e.webpage_url for e in archive.rows] == ["https://yt.com/v=1"]

    @pytest.mark.parametrize("bad_id", ["0", "-1", str(2**63)])
    async def test_keys_outside_the_guild_id_domain_are_skipped(
        self, fake_redis: Redis, bad_id: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """This tool is the only place that can manufacture an out-of-domain
        guild_id, and HistoryEntry.__post_init__ CLAMPS rather than validates:
        `-1` floors to 0 and 2**63 clamps to int8max. play_history's CHECK is
        strictly `> 0`, so a stamped 0 is constructible and NOT insertable —
        and executemany is atomic, so it costs the whole batch.

        The clamp is why parseability was not enough. Skipping beats clamping:
        a key that cannot be attributed to a guild has no correct destination,
        and inventing one files real plays under the wrong guild silently.
        """
        await fake_redis.lpush(
            f"guild:{bad_id}:history", serialize_history_entry(_entry(1))
        )
        await _seed(fake_redis, 7, _entry(2, guild_id=7))  # a good one alongside
        archive = CollectingArchive()

        report = await backfill(fake_redis, archive)

        assert report.guilds == 1  # only the good key
        assert [e.guild_id for e in archive.rows] == [7]
        assert "outside the play_history domain" in caplog.text
        # …and the run says so. A skipped key is data that still exists and did
        # not move, so it has to reach the verdict the deploy is gated on.
        assert report.skipped_keys == 1
        assert not report.ok

    async def test_every_guild_is_enumerated_past_one_scan_page(
        self, fake_redis: Redis
    ) -> None:
        """SCAN is a CURSOR. Every other test here seeds one or two guilds, so
        a single non-cursor `scan(...)` page passes all of them — measured: the
        mutation survived all 18.

        Against 250 guilds one `SCAN COUNT 100` page returns 100 keys and a
        non-zero cursor, so that regression silently skips 150 guilds' entire
        pre-archive history. Enumeration is the one primitive with no
        downstream safety net: a guild the scan never yields produces no
        corrupt count, no log line, and no discrepancy in the report — and the
        bot then trims the only copy. fakeredis models the cursor
        faithfully, so nothing about this needs a real server.

        Seeded past `count=100` on purpose; a smaller number proves nothing.
        """
        seeded = 250
        for gid in range(1, seeded + 1):
            await _seed(fake_redis, gid, _entry(1, guild_id=gid))
        archive = CollectingArchive()

        report = await backfill(fake_redis, archive)

        assert report.guilds == seeded
        assert len(archive.rows) == seeded
        assert {e.guild_id for e in archive.rows} == set(range(1, seeded + 1))


class FailingArchive(CollectingArchive):
    """An archive that refuses one guild's rows.

    The real PostgresHistoryArchive raises on everything ("callers own the
    error policy") and uses executemany, so one bad row costs the whole batch.
    CollectingArchive cannot fail at all, which left every failure path here
    unpinned — a mutation swallowing insert_batch errors passed all 18 tests.
    """

    def __init__(self, fail_guild: int) -> None:
        super().__init__()
        self.fail_guild = fail_guild

    async def insert_batch(self, entries: Any) -> None:
        if any(e.guild_id == self.fail_guild for e in entries):
            raise RuntimeError(f"insert refused for guild {self.fail_guild}")
        await super().insert_batch(entries)


class TestFailureContainment:
    """What happens when a guild does not move.

    The tool runs once, by hand, immediately before the step that trims the
    lists it reads — so "did all of it move?" is the only question, and these
    pin that it can always be answered.
    """

    async def test_one_failing_guild_does_not_abort_the_others(
        self, fake_redis: Redis
    ) -> None:
        """REGRESSION: every error propagated out of backfill(), so the first
        failure ended the run — and because _run's summary print sits after its
        try/finally, the operator got a bare traceback with no report at all.
        Which guilds were skipped depended on Redis' hash order, so it differed
        between runs.
        """
        for gid in (1, 2, 3):
            await _seed(fake_redis, gid, _entry(1, guild_id=gid))
        archive = FailingArchive(fail_guild=2)

        report = await backfill(fake_redis, archive)

        assert report.failed_guilds == 1
        assert report.guilds == 2  # the two that moved IN FULL, not all three
        assert {e.guild_id for e in archive.rows} == {1, 3}
        assert not report.ok

    async def test_a_failed_guild_is_not_counted_as_backfilled(
        self, fake_redis: Redis
    ) -> None:
        # `guilds` gates the operator's confidence, so a guild whose rows did
        # not land must never inflate it — even though it was visited.
        await _seed(fake_redis, 1, _entry(1, guild_id=1))
        report = await backfill(fake_redis, FailingArchive(fail_guild=1))
        assert report.guilds == 0
        assert report.failed_guilds == 1

    async def test_an_aborted_scan_is_reported_not_raised(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Enumeration dying is different in kind from a guild failing: the
        # population itself is unknown, so the counts become a floor rather
        # than a total and need their own flag.
        def exploding_scan_iter(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("redis went away mid-scan")

        monkeypatch.setattr(fake_redis, "scan_iter", exploding_scan_iter)
        report = await backfill(fake_redis, CollectingArchive())
        assert report.scan_aborted
        assert not report.ok

    async def test_a_clean_run_is_ok(self, fake_redis: Redis) -> None:
        # The other side of the verdict — `ok` must not be vacuously false.
        await _seed(fake_redis, 1, _entry(1, guild_id=1))
        report = await backfill(fake_redis, CollectingArchive())
        assert report.ok
        assert report.failed_guilds == 0


class TestCli:
    """`_run`/`main` were 0% covered, including --dry-run — a flag that runs by
    hand against production Redis at the most irreversible moment of the
    migration, and whose whole promise is that it touches nothing."""

    @pytest.fixture
    def wired(
        self, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> CollectingArchive:
        archive = CollectingArchive()
        monkeypatch.setenv("POSTGRES_URL", "postgresql://stub")
        monkeypatch.setattr(backfill_history, "create_redis_pool", lambda: MagicMock())
        monkeypatch.setattr(backfill_history, "get_redis", lambda _pool: fake_redis)
        monkeypatch.setattr(
            backfill_history, "close_redis_pool", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            backfill_history,
            "PostgresHistoryArchive",
            lambda _url: cast(Any, archive),
        )
        return archive

    async def test_missing_postgres_url_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert await backfill_history._run(dry_run=False) == 1
        assert "POSTGRES_URL is not set" in capsys.readouterr().err

    async def test_run_writes_and_closes_both_pools(
        self, fake_redis: Redis, wired: CollectingArchive
    ) -> None:
        await _seed(fake_redis, 5, _entry(1, guild_id=5))
        assert await backfill_history._run(dry_run=False) == 0
        assert len(wired.rows) == 1
        wired.closed.assert_awaited_once()
        cast(Any, backfill_history.close_redis_pool).assert_awaited_once()

    async def test_a_failed_guild_exits_nonzero_and_says_not_to_deploy(
        self,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The operator-facing half of the containment fix.

        A partially-applied run must be impossible to mistake for a complete
        one — by exit code for anything scripted, and in words for the human
        reading the terminal. Previously an insert failure produced a traceback
        and NO summary, because the print sits after the try/finally.
        """
        archive = FailingArchive(fail_guild=2)
        monkeypatch.setenv("POSTGRES_URL", "postgresql://stub")
        monkeypatch.setattr(backfill_history, "create_redis_pool", lambda: MagicMock())
        monkeypatch.setattr(backfill_history, "get_redis", lambda _pool: fake_redis)
        monkeypatch.setattr(
            backfill_history, "close_redis_pool", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            backfill_history, "PostgresHistoryArchive", lambda _url: cast(Any, archive)
        )
        for gid in (1, 2):
            await _seed(fake_redis, gid, _entry(1, guild_id=gid))

        assert await backfill_history._run(dry_run=False) == 1

        out = capsys.readouterr()
        # The summary still printed — the guild that DID move is accounted for.
        assert "1 guild(s) backfilled" in out.out
        # …and the verdict names the consequence rather than leaving the
        # operator to infer it from a count.
        assert "INCOMPLETE" in out.err
        assert "1 guild(s) FAILED" in out.err
        assert "Do NOT deploy" in out.err

    async def test_it_says_what_it_connected_to_with_the_password_redacted(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """ "0 entries scanned" and "wrong Redis" are otherwise the same output.

        REDIS_URL defaults silently to localhost, so forgetting it produces a
        clean run, exit 0, and a report that reads as "nothing to migrate" —
        immediately before the step that destroys the source. Naming both
        endpoints is the only thing that distinguishes them.

        Redacted because operators paste this output into tickets.
        """
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://bot:hunter2@db.internal/musicbot"
        )
        monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379")

        assert await backfill_history._run(dry_run=True) == 0

        out = capsys.readouterr().out
        assert "cache.internal" in out
        assert "db.internal" in out
        assert "hunter2" not in out  # password never printed
        assert "bot:***@" in out  # …but the user is, so the DSN is identifiable

    async def test_every_run_states_the_ordering_it_cannot_verify(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The ordering violation the docs call unrecoverable, which nothing can
        detect.

        There is nothing to assert a gate on, because there is no gate: the cap
        is unconditional, and the state it produces is unobservable after the
        fact. A guild's list sitting at HISTORY_CACHE_LIMIT is what a trimmed
        guild AND a healthy migrated guild both look like, so no check here
        could tell an operator they ran this too late. The notice is therefore
        printed on every run, including the successful path — which is the one
        place an operator is still reading.
        """
        await _seed(fake_redis, 1, _entry(1, guild_id=1))
        assert await backfill_history._run(dry_run=False) == 0
        out = capsys.readouterr().out
        assert "BEFORE deploying" in out
        assert str(backfill_history.HISTORY_CACHE_LIMIT) in out

    async def test_an_unreachable_database_fails_before_the_walk(
        self,
        fake_redis: Redis,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """REGRESSION: --dry-run validated nothing about Postgres.

        The archive pool is lazy — connect and the schema-version check live in
        _ensure(), reached only from insert_batch, which a dry run never calls.
        So a rehearsal against a dead host, wrong credentials or an unmigrated
        schema walked the whole keyspace and reported success with exit 0,
        which is precisely the thing the rehearsal exists to catch.

        Also asserts it fails BEFORE reading anything: learning this after
        scanning a million entries teaches the same lesson minutes later.
        """
        archive = CollectingArchive()
        monkeypatch.setattr(
            archive,
            "health_check",
            AsyncMock(side_effect=ConnectionRefusedError("no route to host")),
        )
        monkeypatch.setenv("POSTGRES_URL", "postgresql://stub")
        monkeypatch.setattr(backfill_history, "create_redis_pool", lambda: MagicMock())
        monkeypatch.setattr(backfill_history, "get_redis", lambda _pool: fake_redis)
        monkeypatch.setattr(
            backfill_history, "close_redis_pool", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            backfill_history, "PostgresHistoryArchive", lambda _url: cast(Any, archive)
        )
        await _seed(fake_redis, 1, _entry(1, guild_id=1))

        assert await backfill_history._run(dry_run=True) == 1

        assert "cannot reach the play-history database" in capsys.readouterr().err
        assert archive.batches == 0  # nothing was read or written

    async def test_the_dry_run_preflights_the_database(
        self, fake_redis: Redis, wired: CollectingArchive
    ) -> None:
        # The other half: a healthy dry run really does contact Postgres, so
        # the flag's promise ("this is the rehearsal") is now backed by
        # something. Without the probe this counter stays 0.
        await _seed(fake_redis, 1, _entry(1, guild_id=1))
        assert await backfill_history._run(dry_run=True) == 0
        assert wired.health_checks == 1
        assert wired.rows == []

    async def test_an_aborted_scan_also_exits_nonzero_with_its_own_message(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The other INCOMPLETE verdict, and it has to read differently: with a
        # failed guild the population is known, whereas here an unknown number
        # were never looked at, which makes the printed counts a floor.
        def exploding_scan_iter(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("redis went away mid-scan")

        monkeypatch.setattr(fake_redis, "scan_iter", exploding_scan_iter)
        assert await backfill_history._run(dry_run=False) == 1
        err = capsys.readouterr().err
        assert "INCOMPLETE" in err
        assert "floor" in err
        assert "Do NOT deploy" in err

    async def test_a_mid_run_trim_exits_nonzero_and_says_to_stop_the_bot(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The exit code is the contract the README gates the deploy on.

        This is the case that used to print a clean summary and exit 0 having
        just watched plays be destroyed, so anything scripting
        `just db-backfill && ./build_docker.sh` proceeded straight into the
        irreversible step. It is also the only verdict that must NOT promise a
        re-run fixes it — what has been trimmed is gone.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await _seed(fake_redis, 1, *(_entry(n, guild_id=1) for n in range(4)))
        real_lrange = fake_redis.lrange

        async def shrinking_lrange(name: str, start: int, end: int) -> Any:
            out = await real_lrange(name, start, end)
            await fake_redis.rpop(key)
            return out

        monkeypatch.setattr(fake_redis, "lrange", shrinking_lrange)

        assert await backfill_history._run(dry_run=False) == 1

        err = capsys.readouterr().err
        assert "INCOMPLETE" in err
        assert "STOP THE BOT" in err
        assert "nothing recovers" in err

    async def test_an_unattributable_key_exits_nonzero(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A key holding real plays that this tool will never move. Reporting
        # success over it is the same class of lie as the trim above.
        await fake_redis.lpush("guild:0:history", serialize_history_entry(_entry(1)))
        assert await backfill_history._run(dry_run=False) == 1
        err = capsys.readouterr().err
        assert "INCOMPLETE" in err
        assert "could not be attributed" in err

    async def test_every_applicable_verdict_prints_not_just_the_first(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A run can fail in more than one way at once.

        These were an if/elif chain, so the first reason won and the rest were
        invisible — and an operator who fixes only the reported one re-runs
        straight into the next, on a tool whose next step is irreversible.
        """
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await _seed(fake_redis, 1, *(_entry(n, guild_id=1) for n in range(4)))
        await fake_redis.lpush("guild:0:history", serialize_history_entry(_entry(9)))
        real_lrange = fake_redis.lrange

        async def shrinking_lrange(name: Any, start: int, end: int) -> Any:
            out = await real_lrange(name, start, end)
            # scan_iter yields BYTES keys, so this has to normalise before
            # comparing — matching str against bytes silently never fires.
            if (name.decode() if isinstance(name, bytes) else name) == key:
                await fake_redis.rpop(key)
            return out

        monkeypatch.setattr(fake_redis, "lrange", shrinking_lrange)

        assert await backfill_history._run(dry_run=False) == 1

        err = capsys.readouterr().err
        assert "STOP THE BOT" in err  # the trim
        assert "could not be attributed" in err  # …and the skipped key
        assert err.count("INCOMPLETE") == 2

    async def test_a_failing_close_does_not_hide_the_run(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Both closes are guarded individually so a failing archive close
        # cannot skip the Redis pool close and chain over the real outcome —
        # the shape of the documented MusicBotApp.close() incident. The run's
        # own result must survive teardown noise.
        wired.closed = AsyncMock(side_effect=RuntimeError("pg close blew up"))
        monkeypatch.setattr(
            backfill_history,
            "close_redis_pool",
            AsyncMock(side_effect=RuntimeError("redis close blew up")),
        )
        await _seed(fake_redis, 5, _entry(1, guild_id=5))

        assert await backfill_history._run(dry_run=False) == 0
        assert "1 guild(s) backfilled" in capsys.readouterr().out
        cast(Any, backfill_history.close_redis_pool).assert_awaited_once()

    async def test_the_summary_reports_what_a_dry_run_actually_did(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The message, not just the behaviour. Mutations that make the summary
        lie — swapping a counter, or hardcoding the verb so a dry run claims
        "submitted" — survived every behavioural test, and this line is what the
        operator reads before deciding the migration is done.

        EVERY COUNT IS DISTINCT here on purpose: 3 scanned, 2 submitted, 1
        corrupt, 1 guild. An earlier version of this test seeded two clean
        entries, so `attempted == scanned` and swapping those two fields in the
        f-string was undetectable — the test asserted a lying summary was fine.
        """
        await _seed(fake_redis, 5, _entry(1, guild_id=5), _entry(2, guild_id=5))
        await fake_redis.lpush(GUILD_HISTORY_KEY.format(guild_id=5), b"not json")

        assert await backfill_history._run(dry_run=True) == 0

        out = capsys.readouterr().out
        assert "1 guild(s) backfilled" in out
        assert "3 entries scanned" in out
        assert "would submit 2" in out
        assert "1 corrupt" in out
        # The verb is the dry-run one, and the writing verb appears nowhere.
        assert "submitted 2" not in out.replace("would submit 2", "")
        assert wired.rows == []

    def test_dry_run_flag_reaches_the_backfill(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The mutation this kills: `asyncio.run(_run(False))`. It leaves every
        test green while `just db-backfill --dry-run` WRITES every historical
        entry instead of counting them."""
        seen: list[bool] = []
        real_run = backfill_history._run

        async def spy(dry_run: bool) -> int:
            seen.append(dry_run)
            return await real_run(dry_run)

        monkeypatch.setattr(backfill_history, "_run", spy)
        monkeypatch.setattr(sys, "argv", ["backfill_history", "--dry-run"])
        assert backfill_history.main() == 0
        assert seen == [True]

    def test_default_is_not_a_dry_run(
        self,
        fake_redis: Redis,
        wired: CollectingArchive,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[bool] = []
        real_run = backfill_history._run

        async def spy(dry_run: bool) -> int:
            seen.append(dry_run)
            return await real_run(dry_run)

        monkeypatch.setattr(backfill_history, "_run", spy)
        monkeypatch.setattr(sys, "argv", ["backfill_history"])
        assert backfill_history.main() == 0
        assert seen == [False]

    async def test_dry_run_writes_nothing(
        self, fake_redis: Redis, wired: CollectingArchive
    ) -> None:
        await _seed(fake_redis, 5, _entry(1, guild_id=5), _entry(2, guild_id=5))
        assert await backfill_history._run(dry_run=True) == 0
        assert wired.rows == []  # counted, not written
        assert await fake_redis.llen(GUILD_HISTORY_KEY.format(guild_id=5)) == 2
