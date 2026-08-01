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

This must run BEFORE the cutover that demotes the Redis lists to a capped
cache (HISTORY_REDIS_CUTOVER=1 — the flag arrives with that change and does not
exist yet). Trimming those lists first destroys the only copy of exactly the
entries this exists to move.
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
    close_redis_pool,
    create_redis_pool,
    get_redis,
)
from src.util import get_logger

log = get_logger(__name__)

# Entries per INSERT round-trip. Large enough that a 10k-entry guild is 20
# statements, small enough that one batch stays well inside command_timeout.
#
# It also sets the LRANGE cadence, and that side has a cost worth knowing before
# anyone runs this against a very large list: Redis walks a list from the nearer
# end, so paging an n-entry list is ~n/page calls averaging n/2 element steps —
# O(n^2/page) overall, on the single-threaded server the live bot shares. At the
# ~1M entries the README's sizing note contemplates that is ~2000 calls. Not
# introduced by the tail-relative paging (head-relative has the same shape) and
# not worth trading for a cursor this tool would use exactly once, but a reason
# to run the migration in a quiet window rather than mid-evening.
_PAGE = 500

# guild:{id}:history → the {id} part. Derived from the key template rather than
# hardcoded as index 1, so a change to the template can't silently start
# parsing the wrong segment.
_GUILD_ID_INDEX = GUILD_HISTORY_KEY.split(":").index("{guild_id}")
_HISTORY_KEY_MATCH = GUILD_HISTORY_KEY.format(guild_id="*")
# Upper bound of play_history.guild_id (bigint). Local rather than imported from
# guild_state's clamp constants on purpose: this is a validation bound, and the
# whole point of _guild_id_from_key's check is that the clamp is NOT the
# validation. Sharing the constant would invite someone to "reuse the clamp".
_INT8_MAX = 2**63 - 1
# Mirrors create_redis_pool's default. Duplicated rather than imported because
# it is only used to REPORT what the run connected to; the pool remains the
# single place that decides it.
_DEFAULT_REDIS = "redis://localhost:6379"


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
    # Guilds whose backfill raised. NOT the same as `corrupt`: a corrupt entry
    # is data this tool understood and rejected, while a failure here means an
    # unknown amount of that guild's history did not move. The two must not be
    # summed into one "problems" number, because only this one makes the run
    # INCOMPLETE — and completeness is what gates the cutover that destroys the
    # source. `ok` below is the single question the operator is actually asking.
    failed_guilds: int = 0
    # True when enumeration itself died, i.e. an unknown number of guilds were
    # never even looked at. Distinct from failed_guilds, where the population is
    # known and the failures are counted within it.
    scan_aborted: bool = False

    @property
    def ok(self) -> bool:
        """Did every guild this run could see move in full?

        The one thing that may gate `HISTORY_REDIS_CUTOVER`. Deliberately a
        property rather than a caller-side expression: the check has to change
        in one place when a future failure mode is added to this report.
        """
        return not self.failed_guilds and not self.scan_aborted


def _guild_id_from_key(key: bytes) -> Optional[int]:
    """The guild id a history key belongs to, or None if it is not usable.

    Parseability is not enough, and this is the one place in the system that
    can manufacture an out-of-domain guild_id. play_history's CHECK is
    `guild_id > 0` (strict), while HistoryEntry.__post_init__ CLAMPS to
    `0 <= v <= int8max` — so a key of `guild:0:history` or `guild:-1:history`
    yields an entry that constructs fine and Postgres then refuses, and a
    20-digit key id clamps to 2**63-1 and files that guild's whole history
    under a fabricated id with no error at all.

    Neither shape can come from the bot: GuildRedisStore is only ever built
    with `guild.id`, a positive snowflake. They come from a hand-made or
    hand-edited key, which is exactly the situation a one-shot migration tool
    meets. Skipping is right rather than clamping: a key we cannot attribute
    has no correct destination, and inventing one would file real plays under
    the wrong guild silently.
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

    NEVER RAISES, and that is a contract rather than defensiveness. This tool
    runs once, by hand, immediately before the step that trims the lists it
    reads — so the operator's only question is "did all of it move?", and an
    exception is the one answer that cannot be given honestly. An early version
    let everything propagate: one WRONGTYPE from a stray key killed the run,
    every guild after it in SCAN order was never attempted, and _run's summary
    never printed because it sits after the try/finally. The result was a
    partially-applied migration reported as a bare traceback, with WHICH guilds
    were missed depending on Redis' hash order and so differing between runs.

    Failures are contained per guild and counted instead. Three consequences,
    all deliberate:

    - A transient fault on one guild no longer discards the walk. Against a
      real deployment this is hours of work.
    - The report stays truthful: `failed_guilds` names how much did not move,
      and `ok` is what may gate the cutover.
    - A systemic fault (schema drift, a dead database) fails every guild rather
      than the first one, which tells the operator the SCOPE instead of just
      the first symptom. It is noisier by design.

    Ctrl-C still works: KeyboardInterrupt and CancelledError are BaseException,
    which `except Exception` does not catch.
    """
    report = BackfillReport()
    try:
        keys = redis.scan_iter(match=_HISTORY_KEY_MATCH, count=100)
        async for key in keys:
            report = await _backfill_one(
                redis, archive, key, report, page=page, dry_run=dry_run
            )
    except Exception as e:
        # Enumeration itself died, so an UNKNOWN number of guilds were never
        # looked at — which is why this is its own flag and not a failed_guilds
        # increment. Whatever was counted before the failure is still returned
        # and still printed; it is a floor, not a total.
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
        return report
    try:
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
                    # The BYTES, not just a count. parse_history_entry logs
                    # only its exception, so `corrupt: 3` in the summary was
                    # otherwise the entire forensic record of three plays that
                    # cutover is about to delete.
                    #
                    # This is where the backfill diverges from the drainer,
                    # which drops corrupt entries just as quietly: an outbox
                    # entry that fails to parse still has its twin on the
                    # guild:{id}:history list, so nothing is lost. Here that
                    # list IS the copy, and the next migration step trims it.
                    # Logging is the only durable record left — deliberately
                    # not play_history_rejected, which means "Postgres refused
                    # this row" and whose emptiness `just db-rejects` reports
                    # as a code-defect signal. A row this tool could not even
                    # parse was never offered to Postgres.
                    log.error(
                        f"corrupt history entry in {key!r} at tail offset "
                        f"{start + len(entries) + corrupt - 1}, NOT migrated "
                        f"and about to be unrecoverable: {wire[:400]!r}"
                    )
                    continue
                if entry.guild_id == 0:
                    # Pre-guild_id wire format: the key is the only remaining
                    # record of which guild this play belongs to.
                    #
                    # replace() rather than object.__setattr__ because it is the
                    # correct way to modify a frozen dataclass; it also re-runs
                    # __post_init__, which is belt-and-braces rather than the
                    # validation. The validation is _guild_id_from_key, upstream
                    # — __post_init__ CLAMPS, so it would turn an out-of-domain
                    # key id into a plausible wrong one rather than refusing it.
                    entry = dataclasses.replace(entry, guild_id=guild_id)
                # No sanitize call. The guarantee moved from the archive
                # IMPLEMENTATION to the TYPE: parse_history_entry constructs a
                # HistoryEntry, whose __post_init__ clamps into the play_history
                # column domain. Unlike the old _sanitize_entry call this holds
                # for the HistoryArchive protocol rather than for one conformer,
                # which is what the previous comment here wanted and could not get.
                #
                # ONE FIELD IS EXEMPT, and it is the one this tool writes.
                # __post_init__ clamps integers to 0 <= v <= int8max, while
                # play_history's CHECK on guild_id is strictly > 0 — so the type
                # does NOT prove a stamped entry is insertable. (The drainer's
                # docstring names the same exception, and handles it with
                # _isolate + a dead-letter row; this tool does not have that.)
                # The hole is closed upstream instead: _guild_id_from_key
                # refuses any key outside 0 < id <= int8max, so nothing that
                # reaches here can carry an out-of-domain stamp. If a refusal
                # somehow still happens, executemany is atomic — the batch is
                # lost, the guild is counted in failed_guilds, and the run
                # reports INCOMPLETE rather than pretending otherwise.
                entries.append(entry)
            if entries and not dry_run:
                await archive.insert_batch(entries)
            attempted += len(entries)
    except Exception as e:
        # Counted, not raised, and the guild is NOT folded into `guilds` — that
        # counter means "guilds moved in full", so a partially-inserted guild
        # must not inflate it. Its already-inserted rows stay in Postgres and
        # collapse on the dedup index when the operator re-runs.
        log.error(
            f"backfill FAILED for guild {guild_id}, its history did not move: "
            f"{type(e).__name__}: {e}",
            exc_info=e,
        )
        return dataclasses.replace(report, failed_guilds=report.failed_guilds + 1)
    # RECONCILIATION, in the one direction that can mean loss. attempted may
    # legitimately EXCEED total: the last page's tail window clamps at index 0
    # and picks up entries pushed during the run, which the dedup index absorbs.
    # Coming up SHORT is the opposite — it means entries the initial LLEN
    # counted were not read back, which only a tail-side shrink can cause.
    # Nothing shrinks these lists from the tail today, so this is latent; it
    # goes live the moment the cutover's LTRIM ships, and by then the entries it
    # would eat are the oldest, i.e. exactly the ones this tool exists to save.
    if attempted + corrupt < total:
        log.warning(
            f"guild {guild_id}: read {attempted + corrupt} of {total} entries "
            f"the initial LLEN counted — the list shrank from the tail during "
            f"the run. Re-run to pick up anything missed."
        )
    log.info(
        f"{'would backfill' if dry_run else 'backfilled'} guild {guild_id}: "
        f"{attempted} entries ({total} scanned, {corrupt} corrupt)"
    )
    return dataclasses.replace(
        report,
        guilds=report.guilds + 1,
        scanned=report.scanned + total,
        attempted=report.attempted + attempted,
        corrupt=report.corrupt + corrupt,
    )


def _redacted(dsn: str) -> str:
    """A DSN safe to print: everything but the password.

    Printed rather than assumed because "0 entries scanned" and "this database
    is empty" are the same output, and the operator is about to act on it.
    """
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
    # SAY WHAT WE CONNECTED TO, before doing anything. A run against a flushed
    # or simply wrong Redis prints "0 guild(s), 0 entries scanned" and exits 0 —
    # byte-identical to a completed migration, immediately before the step that
    # destroys the source. REDIS_URL in particular defaults silently to
    # localhost, so "I forgot to set it" and "there is nothing to migrate" look
    # the same. This is the only line that tells them apart.
    print(f"source (Redis):     {_redacted(os.getenv('REDIS_URL', _DEFAULT_REDIS))}")
    print(f"destination (PG):   {_redacted(url)}")
    try:
        # PREFLIGHT, and it is what makes --dry-run mean anything. The archive
        # pool is lazy — _ensure() (connect + schema-version check) is reached
        # only from insert_batch, which a dry run never calls. So a rehearsal
        # against an unreachable host, wrong credentials or an unmigrated
        # schema previously walked the entire keyspace and reported success,
        # exit 0. That is the one thing the rehearsal exists to catch, on the
        # tool whose next step destroys the source data.
        #
        # Deliberately BEFORE the walk: failing after scanning a million
        # entries teaches the operator the same thing several minutes later.
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
        # pool close and bury the error the operator actually needs, which is
        # the shape of the documented MusicBotApp.close() incident.
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
    # The completeness verdict, stated rather than left for the operator to
    # infer from a count. This is the line that gates the cutover, so it says
    # what to do rather than only what happened — and the exit code carries the
    # same answer for anything driving this from a script.
    if report.scan_aborted:
        print(
            "INCOMPLETE: the scan for history keys aborted, so an unknown "
            "number of guilds were never read. The counts above are a floor, "
            "not a total. Fix the cause and re-run — re-running is safe. Do "
            "NOT enable HISTORY_REDIS_CUTOVER.",
            file=sys.stderr,
        )
    elif report.failed_guilds:
        print(
            f"INCOMPLETE: {report.failed_guilds} guild(s) FAILED and their "
            "history did not move (see the errors above). Re-running is safe "
            "and retries them. Do NOT enable HISTORY_REDIS_CUTOVER until this "
            "reports 0 failures.",
            file=sys.stderr,
        )
    return 0 if report.ok else 1


# Operator-facing help. NOT the module docstring: argparse's default formatter
# collapses whitespace, so passing __doc__ re-wrapped 40 lines of design
# rationale into one blob — fusing the two example commands into a
# copy-pasteable trap ("just db-backfill --dry-run # count what would move,
# touch nothing just db-backfill # do it") and surfacing a gitignored docs/ path
# in help text an operator reads. The rationale stays in the module docstring
# where maintainers look; this is what someone running it needs.
_HELP = """\
Copy pre-archive play history from Redis into Postgres.

The archive only records songs played AFTER it was deployed. This walks the
guild:{id}:history lists and inserts everything already sitting there.

  just db-backfill --dry-run   # count what would move, write nothing
  just db-backfill             # do it

Safe to re-run and safe to interrupt: every insert is idempotent, so a run that
dies part-way is resumed by running it again. Exits non-zero if any guild
failed — re-run until it reports 0 failures.

MUST complete before HISTORY_REDIS_CUTOVER=1, which trims the Redis lists this
reads from. Anything missed before that point is unrecoverable.
"""


def main() -> int:
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
