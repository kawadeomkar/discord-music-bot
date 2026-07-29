"""One-shot backfill of pre-archive play history from Redis into Postgres.

    just db-backfill --dry-run     # count what would move, touch nothing
    just db-backfill               # do it

The Postgres tier only archives songs played *after* it was deployed: entries
already sitting on the guild:{id}:history lists were written before anything
was pushing to the outbox. This walks those lists and inserts them directly.

Directly, not through the outbox, for two reasons. Pushing a large historical
backlog onto the outbox would bury the live drain behind it — the outbox is
supposed to be near-empty, and the whole tier's latency story depends on that.
And entries written before HistoryEntry carried a guild_id parse as guild_id=0,
so every guild's legacy rows would collide with every other guild's on the
(guild_id, played_at, webpage_url) dedup index. This stamps the real guild id
from the key instead, which is the only place that information still exists
(docs/POSTGRES_HISTORY_PLAN.md §5.6).

Safe to re-run and safe to interrupt: ON CONFLICT DO NOTHING makes every insert
idempotent, so a run that dies half-way is resumed simply by running it again.
Order within a guild does not matter — reads sort on played_at.

This must run BEFORE HISTORY_REDIS_CUTOVER=1 (Phase C). Trimming the Redis
lists first destroys the only copy of exactly the entries this exists to move.
"""

import argparse
import asyncio
import dataclasses
import sys
from typing import Optional

import redis.asyncio as aioredis

from src import config
from src.guild_state import parse_history_entry
from src.history_archive import HistoryArchive, PostgresHistoryArchive
from src.redis_client import (
    GUILD_HISTORY_KEY,
    close_redis_pool,
    create_redis_pool,
    get_redis,
)
from src.util import get_logger

log = get_logger(__name__)

# Entries per INSERT round-trip. Large enough that a 10k-entry guild is 20
# statements, small enough that one batch stays well inside command_timeout.
_PAGE = 500

# guild:{id}:history → the {id} part. Derived from the key template rather than
# hardcoded as index 1, so a change to the template can't silently start
# parsing the wrong segment.
_GUILD_ID_INDEX = GUILD_HISTORY_KEY.split(":").index("{guild_id}")
_HISTORY_KEY_MATCH = GUILD_HISTORY_KEY.format(guild_id="*")


# kw_only: four adjacent int fields is exactly the shape where a positional
# call can transpose two of them and still type-check.
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BackfillReport:
    guilds: int = 0
    scanned: int = 0
    # Rows handed to insert_batch, NOT rows that landed. The archive inserts
    # ON CONFLICT DO NOTHING, so entries colliding on
    # (guild_id, played_at, webpage_url) — two plays of one URL inside the same
    # clock second, or several epoch-0 "unknown time" legacy entries for one
    # URL — collapse into a single row and are counted here anyway. Named for
    # what it can actually promise: reporting these as "inserted" would make
    # the operator's pre-cutover verification step attest to a durability
    # claim this number cannot support.
    attempted: int = 0
    corrupt: int = 0


def _guild_id_from_key(key: bytes) -> Optional[int]:
    parts = key.decode(errors="replace").split(":")
    try:
        return int(parts[_GUILD_ID_INDEX])
    except IndexError, ValueError:
        log.warning(f"skipping history key with unparseable guild id: {key!r}")
        return None


async def backfill(
    redis: aioredis.Redis,
    archive: HistoryArchive,
    *,
    page: int = _PAGE,
    dry_run: bool = False,
) -> BackfillReport:
    """Copy every guild's Redis history list into the archive."""
    report = BackfillReport()
    async for key in redis.scan_iter(match=_HISTORY_KEY_MATCH, count=100):
        guild_id = _guild_id_from_key(key)
        if guild_id is None:
            continue
        total = await redis.llen(key)
        attempted = corrupt = 0
        for start in range(0, total, page):
            # Paged from the TAIL (oldest), NOT by head-relative index.
            # push_history LPUSHes at the head, so with `lrange(key, start,
            # start + page - 1)` every play that finishes mid-run shifts every
            # index right by one and entries slide out of the window unread —
            # measured: 3 songs during a 10-entry backfill silently skipped the
            # 3 OLDEST, while the report still said 10. That is unrecoverable,
            # because cutover then LTRIMs the list this was the only copy of.
            # Tail-relative indices are stable under head pushes; the worst a
            # concurrent play can now do is make us re-read a page, which the
            # dedup index absorbs. Entries pushed during the run are skipped on
            # purpose — they reach Postgres through the outbox.
            raw = await redis.lrange(key, -(start + page), -(start + 1))
            entries = []
            for wire in raw:
                entry = parse_history_entry(wire)
                if entry is None:
                    corrupt += 1
                    continue
                if entry.guild_id == 0:
                    # Pre-guild_id wire format: the key is the only remaining
                    # record of which guild this play belongs to.
                    # dataclasses.replace re-runs __post_init__, so the entry is
                    # re-validated at the one place this tool modifies it.
                    entry = dataclasses.replace(entry, guild_id=guild_id)
                # No sanitize call. The guarantee moved from the archive
                # IMPLEMENTATION to the TYPE: parse_history_entry constructs a
                # HistoryEntry, whose __post_init__ clamps into the play_history
                # column domain. Unlike the old _sanitize_entry call this holds
                # for the HistoryArchive protocol rather than for one conformer,
                # which is what the previous comment here wanted and could not get.
                entries.append(entry)
            if entries and not dry_run:
                await archive.insert_batch(entries)
            attempted += len(entries)
        report = dataclasses.replace(
            report,
            guilds=report.guilds + 1,
            scanned=report.scanned + total,
            attempted=report.attempted + attempted,
            corrupt=report.corrupt + corrupt,
        )
        log.info(
            f"{'would backfill' if dry_run else 'backfilled'} guild {guild_id}: "
            f"{attempted} entries ({total} scanned, {corrupt} corrupt)"
        )
    return report


async def _run(dry_run: bool) -> int:
    url = config.postgres_url()
    if not url:
        print("Error: POSTGRES_URL is not set.", file=sys.stderr)
        return 1
    pool = create_redis_pool()
    redis = get_redis(pool)
    archive = PostgresHistoryArchive(url)
    try:
        report = await backfill(redis, archive, dry_run=dry_run)
    finally:
        await archive.close()
        await close_redis_pool(pool)
    verb = "would submit" if dry_run else "submitted"
    print(
        f"{report.guilds} guild(s), {report.scanned} entries scanned, "
        f"{verb} {report.attempted}, {report.corrupt} corrupt entries skipped "
        f"(submitted counts rows sent, not rows stored — duplicates collapse)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be inserted without writing anything",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
