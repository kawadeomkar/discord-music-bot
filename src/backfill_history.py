"""One-shot backfill of pre-archive play history from Redis into Postgres.

    just db-backfill --dry-run     # count what would move, touch nothing
    just db-backfill               # do it

The Postgres tier only archives songs played *after* it was deployed; entries
already on the guild:{id}:history lists predate anything pushing to the outbox.
This walks those lists and inserts them directly — not through the outbox, which
would bury the live drain behind a historical backlog, and because entries
written before HistoryEntry carried a guild_id parse as guild_id=0, colliding
every guild's legacy rows with every other guild's on the (guild_id, played_at,
webpage_url) dedup index. This stamps the real guild id from the key, the only
place that information still exists.

Safe to re-run and safe to interrupt: ON CONFLICT DO NOTHING makes every insert
idempotent. Order within a guild does not matter — reads sort on played_at.

Must RUN before this BUILD is DEPLOYED. push_history LTRIMs each guild's list to
HISTORY_CACHE_LIMIT on every write, so a guild's first song under it destroys the
only copy of exactly what this exists to move. The window opens per guild at that
guild's next song end, with no flag to check and nothing to undo. "This build",
not "the archive build": the cap is not part of the archive tier (push_history
trims in both modes), so an operator who never opts in is on the same clock.

Nothing here can detect that it already happened: a list sitting at exactly
HISTORY_CACHE_LIMIT is also what a healthy migrated guild looks like. The counts
printed are the only signal, and only when the run precedes the deploy.
See docs/ARCHITECTURE.md#history-backfill.
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

# Entries per INSERT round-trip. Large enough that a 10k-entry guild is 20
# statements, small enough that one batch stays well inside command_timeout.
#
# It also sets the LRANGE cadence: Redis walks a list from the nearer end, so
# paging an n-entry list costs O(n^2/page) element steps on the single-threaded
# server the live bot shares (~2000 calls at ~1M entries). Head-relative paging
# has the same shape — it is a reason to run this in a quiet window, not a
# reason to page differently.
_PAGE = 500

# guild:{id}:history → the {id} part. Derived from the key template rather than
# hardcoded as index 1, so a change to the template can't silently start
# parsing the wrong segment.
_GUILD_ID_INDEX = GUILD_HISTORY_KEY.split(":").index("{guild_id}")
_HISTORY_KEY_MATCH = GUILD_HISTORY_KEY.format(guild_id="*")
# Upper bound of play_history.guild_id (bigint). Local, not imported from
# guild_state's clamp constants: this is a validation bound, and the point of
# _guild_id_from_key's check is that the clamp is not the validation.
_INT8_MAX = 2**63 - 1
# Mirrors create_redis_pool's default, duplicated because it only REPORTS what
# the run connected to; the pool stays the single place that decides it.
_DEFAULT_REDIS = "redis://localhost:6379"
# How much of a corrupt entry's wire bytes to log. Matches history_archive's
# _REJECT_DETAIL_MAX: a realistic history entry measures 545 bytes, so anything
# under ~600 truncates the fields an operator would use.
_WIRE_DUMP_MAX = 2000


# kw_only: four adjacent int fields is exactly the shape where a positional
# call can transpose two of them and still type-check.
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BackfillReport:
    guilds: int = 0
    scanned: int = 0
    # Rows handed to insert_batch, not rows that landed: ON CONFLICT DO NOTHING
    # collapses entries colliding on (guild_id, played_at, webpage_url) — two
    # plays of one URL in the same clock second, several epoch-0 legacy entries
    # for one URL — and they are counted here anyway. Named for what it can
    # promise: "inserted" would make the operator's pre-deploy verification
    # attest to a durability claim this number cannot support.
    attempted: int = 0
    corrupt: int = 0
    # Guilds whose backfill raised. Not `corrupt`: a corrupt entry is data this
    # tool understood and rejected, while a failure here means an unknown amount
    # of that guild's history did not move. Only this one makes the run
    # incomplete, and completeness gates the deploy that destroys the source.
    failed_guilds: int = 0
    # Guilds whose list was trimmed from the tail while this run walked it, so
    # some of their oldest entries were destroyed before they could be read.
    # Detected, reported, and — the point of this counter — fatal to `ok`:
    # anything scripting `just db-backfill && ./build_docker.sh` gates on the
    # exit code, so a clean summary here is a green light issued after the loss.
    short_guilds: int = 0
    # History keys that could not be attributed to a guild (see
    # _guild_id_from_key). Their entries were not migrated and never will be by
    # this tool, so — like short_guilds — they have to reach `ok`, or the
    # headline count excludes data that still exists.
    skipped_keys: int = 0
    # True when enumeration itself died, i.e. an unknown number of guilds were
    # never even looked at. Distinct from failed_guilds, where the population is
    # known and the failures are counted within it.
    scan_aborted: bool = False

    @property
    def ok(self) -> bool:
        """Did every guild this run could see move in full?

        The one thing that may gate the deploy. A property rather than a
        caller-side expression so the check changes in one place when a future
        failure mode is added. Every not-moved outcome belongs here, not just the
        loud ones — each omission lets this return True over destroyed data.
        """
        return not (
            self.failed_guilds
            or self.short_guilds
            or self.skipped_keys
            or self.scan_aborted
        )


def _guild_id_from_key(key: bytes) -> Optional[int]:
    """The guild id a history key belongs to, or None if it is not usable.

    Parseability is not enough, and this is the one place in the system that can
    manufacture an out-of-domain guild_id: play_history's CHECK is `guild_id > 0`
    (strict) while HistoryEntry.__post_init__ clamps to `0 <= v <= int8max`, so
    `guild:0:history` yields an entry that constructs fine and Postgres refuses,
    and a 20-digit key id clamps to 2**63-1 and files that whole history under a
    fabricated guild with no error at all. Neither shape comes from the bot —
    they are hand-edited keys, which a one-shot migration tool meets. Skip rather
    than clamp: an unattributable key has no correct destination.
    """
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
    """Copy every guild's Redis history list into the archive.

    Never RAISES, and that is a contract: this runs once, by hand, immediately
    before the step that trims the lists it reads, so the operator's only
    question is "did all of it move?" — an exception cannot answer it honestly.
    One WRONGTYPE from a stray key would kill the run, skip every guild after it
    in SCAN order and skip _run's summary (which sits after the try/finally).

    Failures are contained per guild and counted instead, so `failed_guilds` names
    how much did not move and a systemic fault (schema drift, a dead database)
    fails every guild rather than the first, giving the operator the SCOPE.

    Ctrl-C still works: KeyboardInterrupt and CancelledError are BaseException,
    which `except Exception` does not catch.
    """
    report = BackfillReport()
    # SCAN is at-LEAST-once: a rehash can return the same key twice. The rows are
    # idempotent, so a repeat costs nothing durable — but it double-counts
    # `guilds`, `scanned` and `attempted`, which the operator verifies against a
    # `SELECT count(*) FROM play_history` before an irreversible deploy, and the
    # module docstring calls those counts "the only signal". Bounded by guild
    # count, which this tool already holds per-guild state for.
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
        # Enumeration itself died, so an UNKNOWN number of guilds were never
        # looked at — hence its own flag rather than a failed_guilds increment.
        # What was counted before the failure is still returned: a floor, not a
        # total.
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
        # Counted, not silently dropped: _guild_id_from_key logged WHY, and this
        # carries it into the verdict so the run cannot report a clean migration
        # while a key holding real plays sits unread.
        return dataclasses.replace(report, skipped_keys=report.skipped_keys + 1)
    try:
        total = await redis.llen(key)
        # The oldest entry's bytes, an identity anchor for the reconciliation
        # below. Index -1 because push_history LPUSHes: the tail is the oldest
        # entry and the first thing a trim destroys.
        tail_before = await redis.lindex(key, -1)
        attempted = corrupt = 0
        for start in range(0, total, page):
            # Paged from the TAIL (oldest), not by head-relative index.
            # push_history LPUSHes at the head, so with `lrange(key, start,
            # start + page - 1)` every play that finishes mid-run shifts every
            # index right by one and entries slide out of the window unread —
            # measured: 3 songs during a 10-entry backfill silently skipped the
            # 3 oldest while the report still said 10, unrecoverable once this
            # build LTRIMs the only copy. Tail-relative indices are stable under
            # head pushes; the worst a concurrent play can do is make us re-read
            # a page, which the dedup index absorbs. Entries pushed during the
            # run are skipped on purpose — they reach Postgres via the outbox.
            raw = await redis.lrange(key, -(start + page), -(start + 1))
            entries = []
            for i, wire in enumerate(raw):
                entry = parse_history_entry(wire)
                if entry is None:
                    corrupt += 1
                    # The page runs newest→oldest internally (LPUSHed list, and a
                    # tail-relative LRANGE still returns list order) while
                    # `start` advances oldest→newest, so the distance from the
                    # oldest end is start + (len-1-i) — never anything derived
                    # from how many entries parsed, which mixes a
                    # guild-cumulative count with a page-local one and points the
                    # operator at a healthy entry.
                    offset = start + len(raw) - 1 - i
                    # The bytes, not just a count: parse_history_entry logs only
                    # its exception, so `corrupt: 3` would be the whole forensic
                    # record of three plays the build is about to trim away. Not
                    # play_history_rejected, which means "Postgres refused this
                    # row" — a row this tool could not parse was never offered to
                    # Postgres. _WIRE_DUMP_MAX because a realistic entry measures
                    # 545 bytes; a 400-byte cap truncated mid-thumbnail, never
                    # reaching uploader, played_at or message_id.
                    log.error(
                        f"corrupt history entry in {key!r} at tail offset "
                        f"{offset}, NOT migrated and about to be "
                        f"unrecoverable: {wire[:_WIRE_DUMP_MAX]!r}"
                    )
                    continue
                if entry.guild_id == 0:
                    # Pre-guild_id wire format: the key is the only remaining
                    # record of which guild this play belongs to. replace()
                    # (not object.__setattr__) is the correct way to modify a
                    # frozen dataclass; it re-runs __post_init__, which is
                    # belt-and-braces — the validation is _guild_id_from_key
                    # upstream, because __post_init__ clamps an out-of-domain
                    # key id into a plausible wrong one rather than refusing it.
                    entry = dataclasses.replace(entry, guild_id=guild_id)
                # No sanitize call: parse_history_entry constructs a HistoryEntry,
                # whose __post_init__ clamps into the play_history column domain.
                # One field is exempt, and it is the one this tool writes:
                # __post_init__ clamps to 0 <= v <= int8max while play_history's
                # CHECK on guild_id is strictly > 0, so the type does not prove a
                # stamped entry insertable. The hole is closed upstream by
                # _guild_id_from_key; if a refusal still happens, executemany is
                # atomic — the batch is lost and the guild reports incomplete.
                entries.append(entry)
            if entries and not dry_run:
                # Oldest-first, which is what insert_batch documents. The page
                # arrives newest→oldest while pages advance oldest→newest, so
                # unreversed the global order is neither — harmless for an
                # ON CONFLICT DO NOTHING index, but it lets this tool and the
                # drainer touch overlapping keys in OPPOSITE orders, and a
                # deadlock costs a whole guild rather than one row.
                entries.reverse()
                await archive.insert_batch(entries)
            attempted += len(entries)
    except Exception as e:
        # Counted, not raised, and not folded into `guilds` — that counter means
        # "guilds moved in full". Already-inserted rows stay in Postgres and
        # collapse on the dedup index when the operator re-runs.
        log.error(
            f"backfill FAILED for guild {guild_id}, its history did not move: "
            f"{type(e).__name__}: {e}",
            exc_info=e,
        )
        return dataclasses.replace(report, failed_guilds=report.failed_guilds + 1)
    # RECONCILIATION — did the list lose entries from the tail while we walked it?
    # push_history LTRIMs on every song end, from the tail, eating the oldest
    # entries: exactly the ones this tool exists to save.
    #
    # BY identity, not by count. `attempted + corrupt < total` only sees a list
    # that got shorter, but push_history LPUSHes and LTRIMs in one transaction, so
    # a list already at HISTORY_CACHE_LIMIT keeps its length exactly while each
    # song end destroys one unread tail entry. Measured on a 50-entry list with
    # one push per page: 4 pre-archive plays gone, scanned == attempted, no
    # warning, exit 0. Two LINDEX round trips answer what the count cannot.
    #
    # attempted may legitimately EXCEED total — the last page's window clamps at
    # index 0 and picks up entries pushed during the run. Not loss, not flagged.
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
    # inflate `guilds` — that counter is what the summary reports as backfilled.
    return dataclasses.replace(
        report,
        guilds=report.guilds + (0 if shrank else 1),
        short_guilds=report.short_guilds + (1 if shrank else 0),
        scanned=report.scanned + total,
        attempted=report.attempted + attempted,
        corrupt=report.corrupt + corrupt,
    )


def _redacted(dsn: str) -> str:
    """A DSN safe to print: everything but the password. Printed rather than
    assumed because "0 entries scanned" and "this database is empty" are the same
    output, and the operator is about to act on it."""
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
    # SAY WHAT WE CONNECTED TO, before doing anything. A run against a flushed or
    # wrong Redis prints "0 guild(s), 0 entries scanned" and exits 0 —
    # byte-identical to a completed migration, immediately before the step that
    # destroys the source. REDIS_URL silently defaults to localhost, so this is
    # the only line separating "I forgot to set it" from "nothing to migrate".
    print(f"source (Redis):     {_redacted(os.getenv('REDIS_URL', _DEFAULT_REDIS))}")
    print(f"destination (PG):   {_redacted(url)}")
    # the ordering notice, unconditional because it is unverifiable: a trimmed
    # list is indistinguishable from a short one, so an operator who runs this
    # after deploying sees a clean-looking run over whatever the trim left.
    # Saying so every time is the only warning that can be given.
    print(
        f"note:               this build caps these lists at "
        f"{HISTORY_CACHE_LIMIT} entries per guild, from each\n"
        f"                    guild's next song end. Run this BEFORE deploying "
        f"it; anything older\n"
        f"                    than that window is unrecoverable once it has."
    )
    try:
        # PREFLIGHT, and it is what makes --dry-run mean anything. The archive
        # pool is lazy — _ensure() (connect + schema-version check) is reached
        # only from insert_batch, which a dry run never calls, so a rehearsal
        # against an unreachable host, wrong credentials or an unmigrated schema
        # would walk the whole keyspace and report success, exit 0. Before the
        # walk: failing after a million entries teaches the same thing later.
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
        # pool close and bury the error the operator actually needs.
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
    # The completeness verdict, stated rather than inferred from a count: this
    # line gates the deploy, so it says what to do, and the exit code carries the
    # same answer for anything driving this from a script. Every applicable
    # reason prints, not the first to win an if/elif chain — a run can fail
    # several ways at once, and fixing only the reported one means re-running
    # straight into the next, on a tool whose next step is irreversible. `ok` is
    # the single source of the exit code, so these branches cannot disagree.
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
        # The only verdict reporting damage already done rather than work still
        # outstanding, so the only one that does not promise a re-run fixes it.
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


# Operator-facing help. Not the module docstring: argparse's default formatter
# collapses whitespace, so passing __doc__ fuses the two example commands into a
# copy-pasteable trap. The rationale stays in the module docstring where
# maintainers look; this is what someone running it needs.
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
    # CONFIGURE LOGGING first, before anything can emit. Unconfigured, structlog
    # falls back to PrintLoggerFactory — readable text on STDOUT that never
    # touches the logging module, so `just db-backfill 2> errors.log` captures
    # none of it and `docker compose run --rm` deletes the container that held
    # it. This module's corrupt-entry ERROR is the only durable record of a play
    # about to become unrecoverable; terminal scrollback is not a record.
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
