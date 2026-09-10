"""One-shot backfill of pre-archive play history from Redis into Postgres.

    just db-backfill --dry-run     # count what would move, touch nothing
    just db-backfill               # do it

The archive only captures songs played after it was deployed; this walks the
guild:{id}:history lists and inserts them directly, not through the outbox
(a historical backlog would bury the live drain). Entries written before
HistoryEntry carried a guild_id parse as guild_id=0 and would collide across
guilds on the (guild_id, played_at, webpage_url) dedup index, so the real id is
stamped from the key, the only place it still exists.

Safe to re-run and to interrupt: ON CONFLICT DO NOTHING makes every insert
idempotent, and reads sort on played_at, so order within a guild is irrelevant.

Must RUN before this BUILD is DEPLOYED: push_history LTRIMs each list to
HISTORY_CACHE_LIMIT on every write, in both archive modes, so a guild's first
song under it destroys the only copy of what this exists to move — per guild,
with no flag to check and nothing to undo. Nothing here can detect that it
already ran (a list at exactly HISTORY_CACHE_LIMIT is also a healthy migrated
guild); the printed counts are the only signal, and only when the run precedes
the deploy. See docs/ARCHITECTURE.md#history-backfill.
"""

import argparse
import asyncio
import dataclasses
import os
import sys
from typing import Optional

import redis.asyncio as aioredis

from src import config
from src.guild_state import parse_history_entry
from src.history_archive import HistoryArchive, PostgresHistoryArchive
from src.redis_client import (
    GUILD_HISTORY_KEY,
    HISTORY_CACHE_LIMIT,
    close_redis_pool,
    create_redis_pool,
    get_redis,
)
from src.telemetry import setup_cli_logging
from src.util import get_logger

log = get_logger(__name__)

# Entries per INSERT round-trip, and the LRANGE cadence: Redis walks a list
# from the nearer end, so paging an n-entry list costs O(n^2/page) element
# steps on the server the live bot shares — run this in a quiet window.
_PAGE = 500

# guild:{id}:history → the {id} part, derived from the key template so a
# template change cannot silently parse the wrong segment.
_GUILD_ID_INDEX = GUILD_HISTORY_KEY.split(":").index("{guild_id}")
_HISTORY_KEY_MATCH = GUILD_HISTORY_KEY.format(guild_id="*")
# Upper bound of play_history.guild_id (bigint). A validation bound, kept apart
# from guild_state's clamp constants: the clamp is not the validation.
_INT8_MAX = 2**63 - 1
# Mirrors create_redis_pool's default; only REPORTS what the run connected to.
_DEFAULT_REDIS = "redis://localhost:6379"
# How much of a corrupt entry's wire bytes to log. A realistic entry measures
# 545 bytes, so anything under ~600 truncates the fields an operator needs.
_WIRE_DUMP_MAX = 2000


# kw_only: four adjacent int fields is the shape where a positional call
# transposes two and still type-checks.
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BackfillReport:
    guilds: int = 0
    scanned: int = 0
    # Rows handed to insert_batch, not rows that landed: ON CONFLICT DO NOTHING
    # collapses collisions on the dedup index, and "inserted" would make the
    # operator's pre-deploy check attest to a durability claim this cannot support.
    attempted: int = 0
    corrupt: int = 0
    # Guilds whose backfill raised: an unknown amount of that guild's history
    # did not move (a corrupt entry is data this tool understood and rejected).
    failed_guilds: int = 0
    # Guilds whose list was trimmed from the tail mid-run, destroying their
    # oldest entries before they could be read. Fatal to `ok`: anything
    # scripting `just db-backfill && ./build_docker.sh` gates on the exit code.
    short_guilds: int = 0
    # History keys that could not be attributed to a guild (_guild_id_from_key);
    # their entries were not migrated, so they must reach `ok` too.
    skipped_keys: int = 0
    # Enumeration itself died: an unknown number of guilds were never looked at,
    # unlike failed_guilds, where the population is known.
    scan_aborted: bool = False

    @property
    def ok(self) -> bool:
        """Did every guild this run could see move in full? The one thing that
        may gate the deploy; every not-moved outcome belongs here, or this
        returns True over destroyed data."""
        return not (
            self.failed_guilds
            or self.short_guilds
            or self.skipped_keys
            or self.scan_aborted
        )


def _guild_id_from_key(key: bytes) -> Optional[int]:
    """The guild id a history key belongs to, or None if it is not usable.
    play_history's CHECK is `guild_id > 0` while __post_init__ clamps to
    `0 <= v <= int8max`, so `guild:0:history` constructs fine and Postgres
    refuses it, and a 20-digit key id clamps to 2**63-1 and files that history
    under a fabricated guild with no error. Hand-edited keys are skipped, not
    clamped: an unattributable key has no correct destination."""
    parts = key.decode(errors="replace").split(":")
    try:
        guild_id = int(parts[_GUILD_ID_INDEX])
    except IndexError, ValueError:
        log.warning(f"skipping history key with unparseable guild id: {key!r}")
        return None
    if not 0 < guild_id <= _INT8_MAX:
        log.error(
            f"skipping history key whose guild id is outside the play_history "
            f"domain (0 < id <= 2**63-1): {key!r}. Its entries were NOT "
            f"migrated — a key like this cannot be attributed to a guild."
        )
        return None
    return guild_id


async def backfill(
    redis: aioredis.Redis,
    archive: HistoryArchive,
    *,
    page: int = _PAGE,
    dry_run: bool = False,
) -> BackfillReport:
    """Copy every guild's Redis history list into the archive. Never RAISES:
    the operator's only question is "did all of it move?", and an exception
    (one WRONGTYPE from a stray key) would skip every guild after it in SCAN
    order and _run's summary. Failures are contained per guild and counted, so
    a systemic fault fails every guild rather than the first, giving the SCOPE.
    Ctrl-C still works: KeyboardInterrupt and CancelledError are BaseException."""
    report = BackfillReport()
    # SCAN is at-least-once (a rehash can return a key twice); a repeat costs
    # nothing durable but would double-count the printed totals the operator
    # verifies against `SELECT count(*)` before an irreversible deploy.
    seen_keys: set[bytes] = set()
    try:
        keys = redis.scan_iter(match=_HISTORY_KEY_MATCH, count=100)
        async for key in keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            report = await _backfill_one(
                redis, archive, key, report, page=page, dry_run=dry_run
            )
    except Exception as e:
        # What was counted before the failure is still returned: a floor, not
        # a total.
        log.error(f"history key scan aborted: {type(e).__name__}: {e}", exc_info=e)
        report = dataclasses.replace(report, scan_aborted=True)
    return report


async def _backfill_one(
    redis: aioredis.Redis,
    archive: HistoryArchive,
    key: bytes,
    report: BackfillReport,
    *,
    page: int,
    dry_run: bool,
) -> BackfillReport:
    """One guild's list, folded into `report`. Never raises — see backfill()."""
    guild_id = _guild_id_from_key(key)
    if guild_id is None:
        # Counted so the run cannot report a clean migration while a key
        # holding real plays sits unread.
        return dataclasses.replace(report, skipped_keys=report.skipped_keys + 1)
    try:
        total = await redis.llen(key)
        # The oldest entry's bytes, the identity anchor for the reconciliation
        # below. push_history LPUSHes, so the tail is the oldest entry and the
        # first thing a trim destroys.
        tail_before = await redis.lindex(key, -1)
        attempted = corrupt = 0
        for start in range(0, total, page):
            # Paged from the TAIL (oldest): head-relative indices shift right
            # by one on every play that finishes mid-run, sliding entries out
            # of the window unread. Tail-relative indices are stable under
            # head pushes; the worst a concurrent play can do is a re-read the
            # dedup index absorbs. Entries pushed during the run are skipped —
            # they reach Postgres via the outbox.
            raw = await redis.lrange(key, -(start + page), -(start + 1))
            entries = []
            for i, wire in enumerate(raw):
                entry = parse_history_entry(wire)
                if entry is None:
                    corrupt += 1
                    # The page runs newest→oldest while `start` advances
                    # oldest→newest, so the distance from the oldest end is
                    # start + (len-1-i) — never derived from how many parsed.
                    offset = start + len(raw) - 1 - i
                    # The bytes, not just a count: parse_history_entry logs only
                    # its exception, and this is the whole forensic record of a
                    # play the build is about to trim away. Not
                    # play_history_rejected — Postgres was never offered it.
                    log.error(
                        f"corrupt history entry in {key!r} at tail offset "
                        f"{offset}, NOT migrated and about to be "
                        f"unrecoverable: {wire[:_WIRE_DUMP_MAX]!r}"
                    )
                    continue
                if entry.guild_id == 0:
                    # Pre-guild_id wire format: the key is the only remaining
                    # record of the guild. replace() re-runs __post_init__,
                    # which clamps rather than refuses; the validation is
                    # _guild_id_from_key upstream.
                    entry = dataclasses.replace(entry, guild_id=guild_id)
                # No sanitize call: __post_init__ clamped every field into the
                # column domain except guild_id, whose CHECK is strictly > 0 and
                # is closed upstream. If a refusal still happens, executemany is
                # atomic: the batch is lost and the guild reports incomplete.
                entries.append(entry)
            if entries and not dry_run:
                # Oldest-first, as insert_batch documents. The page arrives
                # newest→oldest, and letting this tool and the drainer touch
                # overlapping keys in opposite orders risks a deadlock that
                # costs a whole guild.
                entries.reverse()
                await archive.insert_batch(entries)
            attempted += len(entries)
    except Exception as e:
        # Counted, not raised, and not folded into `guilds` ("moved in full").
        # Already-inserted rows collapse on the dedup index on re-run.
        log.error(
            f"backfill FAILED for guild {guild_id}, its history did not move: "
            f"{type(e).__name__}: {e}",
            exc_info=e,
        )
        return dataclasses.replace(report, failed_guilds=report.failed_guilds + 1)
    # RECONCILIATION — did the list lose tail entries while we walked it?
    # By identity, not count alone: push_history LPUSHes and LTRIMs in one
    # transaction, so a list already at HISTORY_CACHE_LIMIT keeps its length
    # while each song end destroys one unread tail entry. attempted may exceed
    # total (the last page clamps at index 0 and picks up entries pushed
    # during the run) — not loss, not flagged.
    tail_after = await redis.lindex(key, -1)
    shrank = attempted + corrupt < total or (
        tail_before is not None and tail_after != tail_before
    )
    if shrank:
        log.warning(
            f"guild {guild_id}: the list was trimmed from the tail DURING the "
            f"run (read {attempted + corrupt} of the {total} entries the "
            f"initial LLEN counted; oldest entry changed: "
            f"{tail_before != tail_after}). Its oldest plays were destroyed "
            f"before they could be read. Stop the bot and re-run — re-running "
            f"is safe, but it cannot recover what has already been trimmed."
        )
    log.info(
        f"{'would backfill' if dry_run else 'backfilled'} guild {guild_id}: "
        f"{attempted} entries ({total} scanned, {corrupt} corrupt)"
        f"{' — INCOMPLETE, list trimmed mid-run' if shrank else ''}"
    )
    # A guild that lost entries mid-run did not move in full, so it must not
    # inflate `guilds`, the count the summary reports as backfilled.
    return dataclasses.replace(
        report,
        guilds=report.guilds + (0 if shrank else 1),
        short_guilds=report.short_guilds + (1 if shrank else 0),
        scanned=report.scanned + total,
        attempted=report.attempted + attempted,
        corrupt=report.corrupt + corrupt,
    )


def _redacted(dsn: str) -> str:
    """A DSN safe to print: everything but the password."""
    scheme, _, rest = dsn.partition("://")
    creds, at, hostpart = rest.rpartition("@")
    if not at:
        return dsn
    user, _, _password = creds.partition(":")
    return f"{scheme}://{user}:***@{hostpart}"


async def _run(dry_run: bool) -> int:
    url = config.postgres_url()
    if not url:
        print("Error: POSTGRES_URL is not set.", file=sys.stderr)
        return 1
    pool = create_redis_pool()
    redis = get_redis(pool)
    archive = PostgresHistoryArchive(url)
    # Say what we connected to first: a run against a flushed or wrong Redis
    # prints "0 guild(s), 0 entries scanned" and exits 0, byte-identical to a
    # completed migration, and REDIS_URL silently defaults to localhost.
    print(f"source (Redis):     {_redacted(os.getenv('REDIS_URL', _DEFAULT_REDIS))}")
    print(f"destination (PG):   {_redacted(url)}")
    # Unconditional because it is unverifiable: a trimmed list is
    # indistinguishable from a short one.
    print(
        f"note:               this build caps these lists at "
        f"{HISTORY_CACHE_LIMIT} entries per guild, from each\n"
        f"                    guild's next song end. Run this BEFORE deploying "
        f"it; anything older\n"
        f"                    than that window is unrecoverable once it has."
    )
    try:
        # Preflight, and what makes --dry-run mean anything: the archive pool
        # is lazy and a dry run never calls insert_batch, so without this a
        # rehearsal against an unreachable host or unmigrated schema would walk
        # the whole keyspace and exit 0.
        try:
            await archive.health_check()
        except Exception as e:
            print(
                f"Error: cannot reach the play-history database: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 1
        report = await backfill(redis, archive, dry_run=dry_run)
    finally:
        # Guarded individually: a failing archive close must not skip the Redis
        # pool close and bury the error the operator needs.
        try:
            await archive.close()
        except Exception as e:
            log.warning(f"archive close failed: {e}")
        try:
            await close_redis_pool(pool)
        except Exception as e:
            log.warning(f"redis pool close failed: {e}")
    verb = "would submit" if dry_run else "submitted"
    print(
        f"{report.guilds} guild(s) backfilled, {report.scanned} entries scanned, "
        f"{verb} {report.attempted}, {report.corrupt} corrupt entries skipped "
        f"(submitted counts rows sent, not rows stored — duplicates collapse)"
    )
    # The completeness verdict gates the deploy, so it says what to do, and the
    # exit code carries the same answer for scripts. Every applicable reason
    # prints, not the first of an if/elif chain: a run can fail several ways at
    # once, and `ok` is the single source of the exit code.
    if report.scan_aborted:
        print(
            "INCOMPLETE: the scan for history keys aborted, so an unknown "
            "number of guilds were never read. The counts above are a floor, "
            "not a total. Fix the cause and re-run — re-running is safe. Do "
            "NOT deploy until this reports a clean run.",
            file=sys.stderr,
        )
    if report.failed_guilds:
        print(
            f"INCOMPLETE: {report.failed_guilds} guild(s) FAILED and their "
            "history did not move (see the errors above). Re-running is safe "
            "and retries them. Do NOT deploy until this reports 0 failures.",
            file=sys.stderr,
        )
    if report.short_guilds:
        # The only verdict reporting damage already done, so the only one that
        # does not promise a re-run fixes it.
        print(
            f"INCOMPLETE: {report.short_guilds} guild(s) had their history "
            "list trimmed WHILE this ran, destroying their oldest plays before "
            "they could be read. This bot is already running a build that caps "
            f"the lists at {HISTORY_CACHE_LIMIT} entries. STOP THE BOT, then "
            "re-run: re-running picks up what is left, but nothing recovers "
            "what has already been trimmed.",
            file=sys.stderr,
        )
    if report.skipped_keys:
        print(
            f"INCOMPLETE: {report.skipped_keys} history key(s) could not be "
            "attributed to a guild and were NOT migrated (see the errors "
            "above). Fix or remove those keys and re-run.",
            file=sys.stderr,
        )
    return 0 if report.ok else 1


# Operator-facing help, separate from the module docstring: argparse's default
# formatter collapses whitespace and would fuse the two example commands.
_HELP = """\
Copy pre-archive play history from Redis into Postgres.

The archive only records songs played AFTER it was deployed. This walks the
guild:{id}:history lists and inserts everything already sitting there.

  just db-backfill --dry-run   # count what would move, write nothing
  just db-backfill             # do it

Safe to re-run and safe to interrupt: every insert is idempotent, so a run that
dies part-way is resumed by running it again. Exits non-zero if any guild
failed — re-run until it reports 0 failures.

MUST complete before this build is deployed: it caps the Redis lists this
reads from, at each guild's next song end. Anything missed by then is
unrecoverable.
"""


def main() -> int:
    # Configure logging before anything can emit: unconfigured, structlog
    # prints to STDOUT outside the logging module, so `2> errors.log` captures
    # none of it, and the corrupt-entry ERROR is the only durable record of a
    # play about to become unrecoverable.
    setup_cli_logging()
    parser = argparse.ArgumentParser(
        description=_HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be inserted without writing anything",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
