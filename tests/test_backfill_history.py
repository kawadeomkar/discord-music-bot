"""Tests for src/backfill_history.py — the §5.6 migration CLI's testable core
(`run`) against fakeredis + a dedup-faithful in-memory archive."""

import dataclasses
from datetime import datetime, timezone

import orjson
import pytest

from src.backfill_history import run
from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT


def _entry(n: int, guild_id: int = 0) -> HistoryEntry:
    # guild_id deliberately defaults to 0: the script must stamp it from the
    # key, treating the wire value as untrusted.
    return HistoryEntry(
        guild_id=guild_id,
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=190,
        requester_id=n,
        requester_name=f"user{n}",
        played_at=1000.0 + n,
    )


def _pre_phase_a_bytes(n: int) -> bytes:
    """Wire bytes from the pre-guild_id writer — no guild_id key at all."""
    return orjson.dumps(
        {
            "title": f"Song {n}",
            "webpage_url": f"https://yt.com/v={n}",
            "duration_secs": 200,
            "played_secs": 190,
            "requester_id": n,
            "requester_name": f"user{n}",
            "thumbnail": "",
            "uploader": "",
            "played_at": 1000.0 + n,
        }
    )


class FakeArchive:
    """Dedup-faithful archive fake: unique on (guild_id, played_at,
    webpage_url) like the real index, and played_at is µs-quantized on insert
    like timestamptz — reads return the round-tripped value, not the raw
    float. lossy=True silently drops writes — the failure --verify exists to
    catch."""

    def __init__(self, lossy: bool = False):
        self.rows: dict[tuple, HistoryEntry] = {}
        self.lossy = lossy

    async def insert_batch(self, entries):
        if self.lossy:
            return
        for e in entries:
            e = dataclasses.replace(
                e,
                played_at=datetime.fromtimestamp(
                    e.played_at, tz=timezone.utc
                ).timestamp(),
            )
            self.rows.setdefault((e.guild_id, e.played_at, e.webpage_url), e)

    async def count(self, guild_id):
        return sum(1 for k in self.rows if k[0] == guild_id)

    async def recent(self, guild_id, limit):
        mine = [e for e in self.rows.values() if e.guild_id == guild_id]
        mine.sort(key=lambda e: e.played_at, reverse=True)
        return mine[:limit]


@pytest.fixture
def archive():
    return FakeArchive()


async def _seed(fake_redis, guild_id: int, count: int) -> None:
    """LPUSH `count` entries (oldest first ⇒ list ends newest-first, like
    the real writer)."""
    for n in range(count):
        await fake_redis.lpush(
            f"guild:{guild_id}:history", _entry(n, guild_id=guild_id).to_redis()
        )


class TestBackfill:
    async def test_moves_all_guilds_and_stamps_guild_id_from_key(
        self, fake_redis, archive
    ):
        # Entries whose wire guild_id is 0 (pre-Phase-A) land with the key's.
        await fake_redis.lpush("guild:7:history", _pre_phase_a_bytes(1))
        await fake_redis.lpush("guild:8:history", _entry(1).to_redis())  # wire gid 0
        assert await run(fake_redis, archive) == 0
        assert await archive.count(7) == 1
        assert await archive.count(8) == 1
        assert (await archive.recent(7, 1))[0].guild_id == 7
        assert (await archive.recent(8, 1))[0].guild_id == 8

    async def test_rerun_inserts_nothing(self, fake_redis, archive):
        await _seed(fake_redis, 1, 3)
        lines: list[str] = []
        assert await run(fake_redis, archive, out=lines.append) == 0
        assert "guild 1: redis=3 inserted=3 dup=0 corrupt=0" in lines
        lines.clear()
        assert await run(fake_redis, archive, out=lines.append) == 0
        assert "guild 1: redis=3 inserted=0 dup=3 corrupt=0" in lines

    async def test_corrupt_entries_counted_and_skipped(self, fake_redis, archive):
        await _seed(fake_redis, 1, 2)
        await fake_redis.lpush("guild:1:history", b"not json")
        lines: list[str] = []
        assert await run(fake_redis, archive, out=lines.append) == 0
        assert "guild 1: redis=2 inserted=2 dup=0 corrupt=1" in lines

    async def test_ignores_non_history_keys(self, fake_redis, archive):
        await fake_redis.lpush("guild:1:queue", b"{}")
        await fake_redis.set("history:outbox", b"x")  # wrong shape for pattern
        lines: list[str] = []
        assert await run(fake_redis, archive, out=lines.append) == 0
        assert "no guild history lists found" in lines

    async def test_reads_lists_larger_than_one_chunk(
        self, fake_redis, archive, monkeypatch
    ):
        # Pre-cutover lists are unbounded, so the read pages in _BATCH-sized
        # LRANGEs instead of one giant reply — nothing may be dropped at the
        # chunk seams.
        import src.backfill_history as backfill

        monkeypatch.setattr(backfill, "_BATCH", 2)
        await _seed(fake_redis, 1, 5)
        lines: list[str] = []
        assert await run(fake_redis, archive, out=lines.append) == 0
        assert "guild 1: redis=5 inserted=5 dup=0 corrupt=0" in lines

    async def test_oldest_first_insert_order(self, fake_redis, archive):
        recorded: list[list[HistoryEntry]] = []
        real_insert = archive.insert_batch

        async def spy(entries):
            recorded.append(list(entries))
            await real_insert(entries)

        archive.insert_batch = spy
        await _seed(fake_redis, 1, 3)
        await run(fake_redis, archive)
        flat = [e for batch in recorded for e in batch]
        assert [e.title for e in flat] == ["Song 0", "Song 1", "Song 2"]


class TestVerify:
    async def test_passes_after_clean_backfill(self, fake_redis, archive):
        await _seed(fake_redis, 1, 3)
        lines: list[str] = []
        assert await run(fake_redis, archive, verify=True, out=lines.append) == 0
        assert "verification passed" in lines

    async def test_sub_microsecond_played_at_verifies(self, fake_redis, archive):
        # time.time() carries sub-µs precision (essentially every entry on
        # Linux) while the archive returns the µs-quantized round trip — the
        # newest-entry identity check must quantize its Redis side or verify
        # fails forever, and re-running can never fix it (H1).
        e = dataclasses.replace(_entry(1, guild_id=1), played_at=1752969600.1234567)
        await fake_redis.lpush("guild:1:history", e.to_redis())
        lines: list[str] = []
        assert await run(fake_redis, archive, verify=True, out=lines.append) == 0
        assert "verification passed" in lines

    async def test_fails_when_archive_drops_writes(self, fake_redis):
        await _seed(fake_redis, 1, 3)
        lines: list[str] = []
        assert (
            await run(
                fake_redis, FakeArchive(lossy=True), verify=True, out=lines.append
            )
            == 1
        )
        assert any("VERIFY FAIL guild 1" in line for line in lines)


class TestDemote:
    async def test_demote_without_verify_refused(self, fake_redis, archive):
        assert await run(fake_redis, archive, demote=True) == 2

    async def test_demote_trims_and_arms_ttl(self, fake_redis, archive):
        await _seed(fake_redis, 1, HISTORY_CACHE_LIMIT + 10)
        await fake_redis.persist("guild:1:history")
        assert await run(fake_redis, archive, verify=True, demote=True) == 0
        assert await fake_redis.llen("guild:1:history") == HISTORY_CACHE_LIMIT
        assert await fake_redis.ttl("guild:1:history") > 0

    async def test_demote_keeps_newest_window(self, fake_redis, archive):
        await _seed(fake_redis, 1, HISTORY_CACHE_LIMIT + 10)
        await run(fake_redis, archive, verify=True, demote=True)
        head = await fake_redis.lrange("guild:1:history", 0, 0)
        assert head == [
            _entry(HISTORY_CACHE_LIMIT + 9, guild_id=1).to_redis()
        ]  # newest survives

    async def test_failed_verify_blocks_demote(self, fake_redis):
        await _seed(fake_redis, 1, HISTORY_CACHE_LIMIT + 10)
        assert (
            await run(fake_redis, FakeArchive(lossy=True), verify=True, demote=True)
            == 1
        )
        # List untouched — demote never ran.
        assert await fake_redis.llen("guild:1:history") == HISTORY_CACHE_LIMIT + 10
