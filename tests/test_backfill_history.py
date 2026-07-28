"""Tests for src/backfill_history.py — moving pre-archive Redis history into
Postgres.

The two properties that matter and that no other test covers: legacy entries
(written before HistoryEntry carried a guild_id) must be stamped from their
KEY, or every guild's old rows collide with every other guild's on the
(guild_id, played_at, webpage_url) dedup index; and the run must be resumable,
because a backfill over a real deployment's history will be interrupted.
"""

import sys
from typing import Any, cast
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
        # Not part of the HistoryArchive protocol, but _run() owns the archive's
        # lifecycle and must close it — asserted by TestCli.
        self.closed = AsyncMock(return_value=None)

    async def insert_batch(self, entries: Any) -> None:
        self.batches += 1
        self.rows.extend(entries)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        return []

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
        await _seed(fake_redis, 1, _entry(1))
        await backfill(fake_redis, CollectingArchive())
        assert await fake_redis.llen(HISTORY_OUTBOX_KEY) == 0

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

    async def test_sanitizes_poison_entries(self, fake_redis: Redis) -> None:
        # Legacy rows have never been through the drainer's sanitizer, so the
        # backfill has to apply it too — otherwise the first bad historical
        # entry fails a 500-row batch.
        key = GUILD_HISTORY_KEY.format(guild_id=1)
        await fake_redis.lpush(
            key, orjson.dumps({"title": "bad\x00title", "played_at": 1e18})
        )
        archive = CollectingArchive()
        await backfill(fake_redis, archive)
        assert archive.rows[0].title == "badtitle"
        assert archive.rows[0].played_at == 0.0

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

    async def test_empty_redis_is_a_noop(self, fake_redis: Redis) -> None:
        report = await backfill(fake_redis, CollectingArchive())
        assert report.guilds == 0
        assert report.attempted == 0


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
