"""
Postgres play-history archive — the durable long-term home for every played song.

- HistoryArchive — the protocol the drainer and the backfill tool program against.
- ArchiveReader — the read side MusicBot holds (-ping's probe, -leaderboard,
  -analytics), kept apart so write-surface fakes never grow a read method.
- PostgresHistoryArchive — asyncpg. Connects lazily so startup never blocks on
  Postgres; applies no DDL (migrations/ owns the schema, _ensure verifies it).
- HistoryOutboxDrainer — Redis outbox STREAM → Postgres: replay this consumer's
  pending IDs → read new → INSERT ... ON CONFLICT DO NOTHING → XACK+XDEL by ID.
  At-least-once: a crash between insert and ack redelivers and play_history_dedup
  collapses the replay. The playback loop never awaits Postgres. Concurrent
  drainers are safe without a lease: `>` gives them disjoint entries and the
  shared pending replay collapses on the index.

Row mapping (HistoryEntry ↔ play_history row) lives here, not in guild_state.py,
whose contract is pure wire schema with no runtime imports.

The outbox is non-evictable, so anything that stalls the drain grows a Redis key
that eventually refuses every write in the process. Each guard closes one door:

  poison entry  blocks the batch behind it forever → HistoryEntry.__post_init__
                clamps into the column domain; _isolate parks what still will
                not insert in play_history_rejected
  hung server   a connected-but-unresponsive Postgres never returns, so there is
                no exception and no alarm → command_timeout + DRAIN_DEADLINE_SECS
  two drainers  structural — XACK settles only the IDs this process archived
  tombstone     body deleted while pending: the ID replays with no payload →
                _settle_tombstones acks and logs it
  stranded PEL  entries under a consumer name nothing reads → XAUTOCLAIM sweep
  dead drainer  the task dies unnoticed → _on_task_done respawn with damping

See docs/ARCHITECTURE.md#history-archive-tier.
"""

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Optional, Protocol, cast

import asyncpg
import orjson
import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import Span
from redis.exceptions import ResponseError

from src import config
from src.db_migrate import EXPECTED_SCHEMA_VERSION
from src.guild_state import (
    TS_MAX,
    WAIT_MEDIAN_INDEX,
    WAIT_PERCENTILES,
    WAIT_UNAVAILABLE,
    AnalyticsMetrics,
    CompletionBucket,
    DailyPoint,
    DurationBucket,
    HeatCell,
    HistoryEntry,
    SourceCompletion,
    SourceDay,
    TopArtist,
    TopListener,
    TopSong,
    parse_history_entry,
    serialize_history_entry,
)
from src.redis_client import (
    HISTORY_OUTBOX_KEY,
    OutboxEntry,
    ensure_outbox_group,
    outbox_depth,
    ack_outbox,
    outbox_pending_below,
    outbox_pending_count,
    read_outbox_new,
    read_outbox_pending,
    reclaim_outbox_stale,
    retire_outbox,
    trim_outbox_below,
)
from src.telemetry import get_tracer
from src.util import get_logger, trace_id_of

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_INSERT_SQL = """
INSERT INTO play_history (guild_id, title, webpage_url, duration_secs,
                          played_secs, requester_id, requester_name,
                          thumbnail, uploader, played_at, message_id, channel_id,
                          queued_at, queue_position, query_source)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
ON CONFLICT (guild_id, played_at, webpage_url) DO NOTHING
"""

_RECENT_SQL = """
SELECT guild_id, title, webpage_url, duration_secs, played_secs,
       requester_id, requester_name, thumbnail, uploader, played_at, message_id,
       channel_id, queued_at, queue_position, query_source
FROM play_history
WHERE guild_id = $1
ORDER BY played_at DESC, id DESC
LIMIT $2
"""

# ON CONFLICT against play_history_rejected_dedup makes this exactly-once: two
# drainers can replay the same pending entry, and `just db-rejects` must report
# three rows as three failures, not one seen three times.
_REJECT_SQL = """
INSERT INTO play_history_rejected (guild_id, error_type, error_detail, trace_id, payload)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT ON CONSTRAINT play_history_rejected_dedup DO NOTHING
"""

# The -leaderboard aggregates. Sentinel groups (requester_id 0, webpage_url '')
# are excluded: each would merge unrelated plays into one top-10 row. $3 is the
# period cutoff, to_timestamp(0) for all-time.
#
# Two passes: aggregate first, then resolve the winners' display values through
# LATERAL. Taking them inline as an ordered `array_agg(...)[1]` removes hash
# aggregation from the planner's options (880ms against 53ms at 300k rows).
# See docs/ARCHITECTURE.md#history-archive-tier.
_TOP_REQUESTERS_SQL = """
WITH top AS (
    SELECT requester_id,
           count(*)         AS plays,
           sum(played_secs) AS played_secs
    FROM play_history
    WHERE guild_id = $1 AND requester_id > 0 AND played_at >= $3
    GROUP BY requester_id
    ORDER BY played_secs DESC, plays DESC, requester_id
    LIMIT $2
)
SELECT t.requester_id, l.requester_name, t.plays, t.played_secs
FROM top t
CROSS JOIN LATERAL (
    -- The cutoff belongs here too, so play_history_recent serves the lookup
    -- instead of filtering over the guild's whole history.
    SELECT p.requester_name
    FROM play_history p
    WHERE p.guild_id = $1 AND p.requester_id = t.requester_id
      AND p.played_at >= $3
    ORDER BY p.played_at DESC, p.id DESC
    LIMIT 1
) l
ORDER BY t.played_secs DESC, t.plays DESC, t.requester_id
"""

_TOP_SONGS_SQL = """
WITH top AS (
    SELECT webpage_url,
           count(*)         AS plays,
           sum(played_secs) AS played_secs
    FROM play_history
    WHERE guild_id = $1 AND webpage_url <> '' AND played_at >= $3
    GROUP BY webpage_url
    ORDER BY played_secs DESC, plays DESC, webpage_url
    LIMIT $2
)
SELECT t.webpage_url, l.title, l.duration_secs, l.query_source,
       t.plays, t.played_secs
FROM top t
CROSS JOIN LATERAL (
    -- Cutoff-bound for the same reason as the requesters lateral above.
    SELECT p.title, p.duration_secs, p.query_source
    FROM play_history p
    WHERE p.guild_id = $1 AND p.webpage_url = t.webpage_url
      AND p.played_at >= $3
    ORDER BY p.played_at DESC, p.id DESC
    LIMIT 1
) l
ORDER BY t.played_secs DESC, t.plays DESC, t.webpage_url
"""

# ── -analytics ────────────────────────────────────────────────────────────────
# Seven aggregates over one guild-and-window slice, one statement, one JSON row.
# $1 guild_id, $2 window in days, $3 bucket unit ('day' or 'week'). The window
# is N COMPLETE UTC units, [end - N, end), so every bar covers a whole bucket
# and the result is immutable until the boundary turns, which is what lets the
# cache TTL run to end of day. See docs/ARCHITECTURE.md#analytics-rendering.
_WAIT_PCT_SQL: Final[str] = ",".join(str(pct) for pct in WAIT_PERCENTILES)

_ANALYTICS_SQL = f"""
WITH bounds AS (
    SELECT ($3::text = 'week')                            AS weekly,
           date_trunc('day',  now() AT TIME ZONE 'UTC')   AS day_end,
           date_trunc('week', now() AT TIME ZONE 'UTC')   AS week_end
), win AS (
    SELECT (CASE WHEN weekly THEN week_end ELSE day_end END) AT TIME ZONE 'UTC'
               AS w_end,
           (CASE WHEN weekly
                 THEN date_trunc('week',
                        (CASE WHEN weekly THEN week_end ELSE day_end END)
                        - ($2::int * interval '1 day'))
                 ELSE day_end - ($2::int * interval '1 day')
            END) AT TIME ZONE 'UTC'                        AS w_start
    FROM bounds
), slice AS MATERIALIZED (
    -- webpage_url and uploader are projected for their DISTINCT counts alone.
    -- They widen the tuplestore; see docs/ARCHITECTURE.md#analytics-rendering.
    SELECT (played_at AT TIME ZONE 'UTC')                  AS lt,
           played_secs, duration_secs, query_source,
           requester_id, webpage_url, uploader,
           -- Sentinel queued_at (the epoch-0 backfill default) is EXCLUDED, not
           -- clamped: it would read as a ~1.79e9-second wait. Real cross-clock
           -- drift goes slightly negative and clamps to 0, which is the benign case.
           CASE WHEN queued_at > to_timestamp(0)
                THEN greatest(extract(epoch FROM played_at - queued_at), 0)
           END                                              AS wait_secs
    FROM play_history, win
    WHERE guild_id = $1
      AND played_at >= win.w_start
      AND played_at <  win.w_end
)
SELECT
  (SELECT extract(epoch FROM w_start) FROM win)             AS window_start,
  (SELECT extract(epoch FROM w_end)   FROM win)             AS window_end,
  -- Today's UTC midnight, whatever the bucket unit. The cache's expiry is derived
  -- from THIS rather than from a second clock read, so an aggregate computed a
  -- millisecond before midnight cannot be cached as if it covered the new day.
  (SELECT extract(epoch FROM (day_end AT TIME ZONE 'UTC')) FROM bounds)
                                                            AS today_start,
  -- The ONE explicit emptiness signal. Never inferred from a branch's shape:
  -- every aggregate below coalesces to '[]', so an empty series also describes a
  -- guild whose rows all failed that branch's own filter.
  (SELECT count(*) FROM slice)                              AS n_rows,
  (SELECT coalesce(extract(epoch FROM min(lt)), 0) FROM slice)
                                                            AS first_play,
  (SELECT coalesce(sum(played_secs), 0) FROM slice)         AS listen_secs,
  -- count(*) over a DISTINCT subquery: Postgres has no hashed count(DISTINCT), and
  -- this form plans as a HashAggregate over a tuplestore each scan rescans.
  (SELECT count(*) FROM (
      SELECT DISTINCT webpage_url FROM slice WHERE webpage_url <> '') u)
                                                            AS unique_songs,
  (SELECT count(*) FROM (
      SELECT DISTINCT requester_id FROM slice WHERE requester_id > 0) u)
                                                            AS unique_listeners,
  (SELECT count(*) FROM (
      SELECT DISTINCT uploader FROM slice WHERE uploader <> '') u)
                                                            AS unique_artists,
  (SELECT count(*) FROM slice WHERE duration_secs <= 0)     AS livestream_plays,
  coalesce((SELECT jsonb_agg(x) FROM (
      SELECT to_char(date_trunc($3, lt), 'YYYY-MM-DD') AS d,
             count(*) AS plays, sum(played_secs) AS secs
      FROM slice GROUP BY 1 ORDER BY 1) x), '[]'::jsonb)     AS daily,
  coalesce((SELECT jsonb_agg(x) FROM (
      SELECT to_char(date_trunc($3, lt), 'YYYY-MM-DD') AS d,
             query_source AS src, count(*) AS plays
      FROM slice GROUP BY 1,2) x), '[]'::jsonb)              AS daily_by_source,
  coalesce((SELECT jsonb_agg(x) FROM (
      SELECT extract(isodow FROM lt)::int AS dow,
             extract(hour   FROM lt)::int AS hr, count(*) AS plays
      FROM slice GROUP BY 1,2) x), '[]'::jsonb)              AS heat,
  -- Three guards, all reachable live. nullif: duration_secs = 0 is a legal
  -- livestream. ::float8: both columns are integer. The ratio is clamped at 1.0 and
  -- the BUCKET folded separately, because width_bucket's upper bound is exclusive
  -- and a ratio of exactly 1.0 returns eleven. CASE keeps NULL a livestream.
  coalesce((SELECT jsonb_agg(x) FROM (
      SELECT src, CASE WHEN b > 10 THEN 10 ELSE b END AS b, count(*) AS plays
      FROM (SELECT query_source AS src,
                   width_bucket(least(played_secs::float8
                                / nullif(duration_secs, 0), 1.0), 0, 1, 10) AS b
            FROM slice WHERE duration_secs > 0) r
      GROUP BY 1,2) x), '[]'::jsonb)                         AS completion,
  coalesce((SELECT jsonb_agg(x) FROM (
      SELECT least(duration_secs / 60, 20) AS b, count(*) AS plays
      FROM slice WHERE duration_secs > 0 GROUP BY 1 ORDER BY 1) x), '[]'::jsonb)
                                                            AS durations,
  coalesce((SELECT jsonb_agg(x) FROM (
      SELECT query_source AS src, sum(played_secs) AS played,
             sum(duration_secs) AS dur
      FROM slice WHERE duration_secs > 0 GROUP BY 1) x), '[]'::jsonb)
                                                            AS source_completion,
  coalesce((SELECT to_jsonb(percentile_cont(ARRAY[{_WAIT_PCT_SQL}])
                            WITHIN GROUP (ORDER BY wait_secs))
            FROM slice WHERE wait_secs IS NOT NULL), '[]'::jsonb)
                                                            AS wait_pcts
"""

# The three text-keyed top-N branches stay out of the CTE so it can project
# narrowly. $3/$4 are the window the main query resolved, so the clock is read
# once.
_ANALYTICS_TOP_LISTENERS_SQL = """
SELECT t.requester_id, l.requester_name, t.plays, t.played_secs
FROM (
    SELECT requester_id, count(*) AS plays, sum(played_secs) AS played_secs
    FROM play_history
    WHERE guild_id = $1 AND requester_id > 0
      AND played_at >= $3 AND played_at < $4
    GROUP BY requester_id
    ORDER BY played_secs DESC, plays DESC, requester_id
    LIMIT $2
) t
CROSS JOIN LATERAL (
    -- Window-bound, so the index serves the lookup and the name comes from a play
    -- inside the window it describes.
    SELECT p.requester_name FROM play_history p
    WHERE p.guild_id = $1 AND p.requester_id = t.requester_id
      AND p.played_at >= $3 AND p.played_at < $4
    ORDER BY p.played_at DESC, p.id DESC LIMIT 1
) l
ORDER BY t.played_secs DESC, t.plays DESC, t.requester_id
"""

# No LATERAL: uploader IS the display name.
_ANALYTICS_TOP_ARTISTS_SQL = """
SELECT uploader, count(*) AS plays, sum(played_secs) AS played_secs
FROM play_history
WHERE guild_id = $1 AND uploader <> ''
  AND played_at >= $3 AND played_at < $4
GROUP BY uploader
ORDER BY played_secs DESC, plays DESC, uploader
LIMIT $2
"""

_ANALYTICS_TOP_SONGS_SQL = """
SELECT t.webpage_url, l.title, l.query_source, t.plays, t.played_secs
FROM (
    SELECT webpage_url, count(*) AS plays, sum(played_secs) AS played_secs
    FROM play_history
    WHERE guild_id = $1 AND webpage_url <> ''
      AND played_at >= $3 AND played_at < $4
    GROUP BY webpage_url
    ORDER BY played_secs DESC, plays DESC, webpage_url
    LIMIT $2
) t
CROSS JOIN LATERAL (
    -- Window-bound, as the listeners lateral above.
    SELECT p.title, p.query_source FROM play_history p
    WHERE p.guild_id = $1 AND p.webpage_url = t.webpage_url
      AND p.played_at >= $3 AND p.played_at < $4
    ORDER BY p.played_at DESC, p.id DESC LIMIT 1
) l
ORDER BY t.played_secs DESC, t.plays DESC, t.webpage_url
"""

# Past this window the daily series downsamples to weeks (371 days is 53 bars).
WEEKLY_BUCKET_MIN_DAYS: Final[int] = 365

_SCHEMA_VERSION_SQL = "SELECT max(version) FROM schema_migrations"

# Cap on the asyncpg message in play_history_rejected.error_detail: enough for
# the SQLSTATE and the offending value.
_REJECT_DETAIL_MAX = 2000

# Per-statement bound, covering a server that accepted the connection and then
# stopped answering. A liveness bound, not a latency target.
_COMMAND_TIMEOUT_SECS = 30.0
# Wait for a free connection: the drainer, -ping and the readers share
# max_size=4, so one stuck consumer must not block the rest unboundedly.
_ACQUIRE_TIMEOUT_SECS = 10.0
# Concurrent user-triggered reads against that max_size=4. Unbounded across
# guilds, a burst takes every connection and the drainer's acquire times out;
# two leaves one for the drainer and one for -ping.
_READ_CONCURRENCY = 2
# Whole-operation bound for a read, including the wait for a slot. Well under
# command_timeout: a read that cannot answer in this long is better failed than
# left holding a connection the drain needs.
_READ_DEADLINE_SECS = 15.0
# Graceful pool shutdown before terminate(). Short: it runs on the shutdown path
# ahead of the Redis pool, discord.py's close and the span flush.
_POOL_CLOSE_TIMEOUT_SECS = 5.0


def _entry_to_row(entry: HistoryEntry) -> tuple:
    return (
        entry.guild_id,
        entry.title,
        entry.webpage_url,
        entry.duration_secs,
        entry.played_secs,
        entry.requester_id,
        entry.requester_name,
        entry.thumbnail,
        entry.uploader,
        datetime.fromtimestamp(entry.played_at, tz=timezone.utc),
        entry.message_id,
        entry.channel_id,
        datetime.fromtimestamp(entry.queued_at, tz=timezone.utc),
        entry.queue_position,
        entry.query_source,
    )


def _json(row: asyncpg.Record, key: str) -> list:
    """One jsonb branch of the analytics row, as a list. The pool sets no jsonb
    codec, so the value arrives as `str`. Every branch coalesces to '[]' in SQL
    (`jsonb_agg` over zero rows is NULL); the None arm is a backstop."""
    raw = row[key]
    if raw is None:
        return []
    parsed = orjson.loads(raw if isinstance(raw, (bytes, str)) else str(raw))
    return parsed if isinstance(parsed, list) else []


def _row_to_metrics(
    row: asyncpg.Record,
    listener_rows: Sequence[asyncpg.Record],
    artist_rows: Sequence[asyncpg.Record],
    song_rows: Sequence[asyncpg.Record],
    days: int,
    unit: str,
) -> AnalyticsMetrics:
    """Decode one _ANALYTICS_SQL row plus its three top-N result sets. Free of
    asyncpg types on the way out: the result is pickled to a chart worker."""
    window_end = float(row["window_end"])
    first_play = float(row["first_play"])
    # The requested window clamped to what the archive covers; the title names
    # both, so the requested number stays visible.
    archived = days
    if first_play > 0:
        archived = max(1, min(days, math.ceil((window_end - first_play) / 86400)))
    pcts = tuple(float(v) for v in _json(row, "wait_pcts"))
    return AnalyticsMetrics(
        days=days,
        bucket_unit=unit,
        window_start_epoch=float(row["window_start"]),
        window_end_epoch=window_end,
        today_start_epoch=float(row["today_start"]),
        archived_days=archived,
        plays=int(row["n_rows"]),
        listen_secs=int(row["listen_secs"]),
        unique_songs=int(row["unique_songs"]),
        unique_listeners=int(row["unique_listeners"]),
        unique_artists=int(row["unique_artists"]),
        # Empty means every queued_at in the window was the epoch-0 sentinel,
        # which renders "unavailable" rather than 0s.
        wait_p50_secs=(
            pcts[WAIT_MEDIAN_INDEX]
            if len(pcts) == len(WAIT_PERCENTILES)
            else WAIT_UNAVAILABLE
        ),
        livestream_plays=int(row["livestream_plays"]),
        daily=tuple(
            DailyPoint(day=e["d"], plays=int(e["plays"]), listen_secs=int(e["secs"]))
            for e in _json(row, "daily")
        ),
        daily_by_source=tuple(
            SourceDay(day=e["d"], source=e["src"], plays=int(e["plays"]))
            for e in _json(row, "daily_by_source")
        ),
        heat=tuple(
            HeatCell(dow=int(e["dow"]), hour=int(e["hr"]), plays=int(e["plays"]))
            for e in _json(row, "heat")
        ),
        completion=tuple(
            CompletionBucket(source=e["src"], bucket=int(e["b"]), plays=int(e["plays"]))
            for e in _json(row, "completion")
        ),
        durations=tuple(
            DurationBucket(minutes=int(e["b"]), plays=int(e["plays"]))
            for e in _json(row, "durations")
        ),
        source_completion=tuple(
            SourceCompletion(
                source=e["src"],
                played_secs=int(e["played"]),
                duration_secs=int(e["dur"]),
            )
            for e in _json(row, "source_completion")
        ),
        wait_pcts=pcts,
        top_listeners=tuple(
            TopListener(
                requester_id=r["requester_id"],
                requester_name=r["requester_name"],
                plays=r["plays"],
                played_secs=r["played_secs"],
            )
            for r in listener_rows
        ),
        top_artists=tuple(
            TopArtist(
                uploader=r["uploader"], plays=r["plays"], played_secs=r["played_secs"]
            )
            for r in artist_rows
        ),
        top_songs=tuple(
            TopSong(
                title=r["title"],
                webpage_url=r["webpage_url"],
                query_source=r["query_source"],
                plays=r["plays"],
                played_secs=r["played_secs"],
            )
            for r in song_rows
        ),
    )


def _row_to_entry(row: asyncpg.Record) -> HistoryEntry:
    return HistoryEntry(
        guild_id=row["guild_id"],
        title=row["title"],
        webpage_url=row["webpage_url"],
        duration_secs=row["duration_secs"],
        played_secs=row["played_secs"],
        requester_id=row["requester_id"],
        requester_name=row["requester_name"],
        thumbnail=row["thumbnail"],
        uploader=row["uploader"],
        played_at=row["played_at"].timestamp(),
        message_id=row["message_id"],
        channel_id=row["channel_id"],
        queued_at=row["queued_at"].timestamp(),
        queue_position=row["queue_position"],
        query_source=row["query_source"],
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RequesterLeader:
    """One row of the -leaderboard listeners board. requester_name is the most
    recent one recorded for that id, so a rename shows the current name."""

    requester_id: int
    requester_name: str
    plays: int
    played_secs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SongLeader:
    """One row of the -leaderboard songs board, grouped by webpage_url. title,
    duration_secs and query_source are the values of that URL's newest play."""

    title: str
    webpage_url: str
    duration_secs: int
    query_source: str = ""
    plays: int
    played_secs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Leaderboard:
    # Tuples: frozen=True over a list would leave the rows mutable.
    requesters: tuple[RequesterLeader, ...]
    songs: tuple[SongLeader, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArchiveStats:
    """What -debug's Postgres block shows. Sizes in bytes. Row counts are
    planner estimates from pg_stat_user_tables, never COUNT(*) — an exact count
    is a full scan of the largest table for one line of a diagnostic embed.
    The counter fields are cumulative since pg_stat_reset; -debug samples twice
    and renders their deltas over `monotonic`."""

    database_bytes: int
    table_bytes: int
    rows_estimate: int
    rejected_estimate: int
    connections: int
    max_connections: int
    shared_buffers: str
    cache_hit_ratio: float
    active_backends: int = 0
    active_io_wait: int = 0
    active_lock_wait: int = 0
    active_other_wait: int = 0
    active_time_ms: float = 0.0
    xacts_total: int = 0
    tuples_total: int = 0
    blks_hit: int = 0
    blks_read: int = 0
    temp_bytes: int = 0
    deadlocks: int = 0
    monotonic: float = 0.0


# One row for one embed field. to_regclass + COALESCE so a database without
# these tables degrades to zeros instead of UndefinedTable while an operator is
# diagnosing it. The active_* FILTERs take client backends only, matching
# pg_stat_database's session counters (the other half of -debug's load row),
# and exclude pg_backend_pid() or the probe reads itself as permanent load.
# pg_stat_activity masks other roles' state as NULL, so the FILTERs undercount
# rather than error; every connection today is the bot's own role.
_STATS_SQL = """
SELECT
    pg_database_size(current_database())                        AS database_bytes,
    COALESCE(
        pg_total_relation_size(to_regclass('public.play_history')), 0
    )                                                           AS table_bytes,
    COALESCE((
        SELECT n_live_tup FROM pg_stat_user_tables
        WHERE relname = 'play_history'
    ), 0)                                                       AS rows_estimate,
    COALESCE((
        SELECT n_live_tup FROM pg_stat_user_tables
        WHERE relname = 'play_history_rejected'
    ), 0)                                                       AS rejected_estimate,
    act.connections                                             AS connections,
    current_setting('max_connections')::int                     AS max_connections,
    current_setting('shared_buffers')                           AS shared_buffers,
    CASE WHEN COALESCE(db.blks_hit, 0) + COALESCE(db.blks_read, 0) > 0
         THEN db.blks_hit::float8 / (db.blks_hit + db.blks_read)
         ELSE 0 END                                             AS cache_hit_ratio,
    act.active_backends                                         AS active_backends,
    act.active_io_wait                                          AS active_io_wait,
    act.active_lock_wait                                        AS active_lock_wait,
    act.active_other_wait                                       AS active_other_wait,
    COALESCE(db.active_time, 0)                                 AS active_time_ms,
    COALESCE(db.xact_commit + db.xact_rollback, 0)              AS xacts_total,
    COALESCE(db.tup_returned + db.tup_fetched, 0)               AS tuples_total,
    COALESCE(db.blks_hit, 0)                                    AS blks_hit,
    COALESCE(db.blks_read, 0)                                   AS blks_read,
    COALESCE(db.temp_bytes, 0)                                  AS temp_bytes,
    COALESCE(db.deadlocks, 0)                                   AS deadlocks
FROM (
    SELECT
        count(*)                                                AS connections,
        count(*) FILTER (
            WHERE state = 'active'
              AND backend_type = 'client backend'
              AND pid <> pg_backend_pid()
        )                                                       AS active_backends,
        count(*) FILTER (
            WHERE state = 'active'
              AND backend_type = 'client backend'
              AND pid <> pg_backend_pid()
              AND wait_event_type = 'IO'
        )                                                       AS active_io_wait,
        count(*) FILTER (
            WHERE state = 'active'
              AND backend_type = 'client backend'
              AND pid <> pg_backend_pid()
              AND wait_event_type = 'Lock'
        )                                                       AS active_lock_wait,
        count(*) FILTER (
            WHERE state = 'active'
              AND backend_type = 'client backend'
              AND pid <> pg_backend_pid()
              AND wait_event_type IS NOT NULL
              AND wait_event_type NOT IN ('IO', 'Lock')
        )                                                       AS active_other_wait
    FROM pg_stat_activity
    WHERE datname = current_database()
) AS act
LEFT JOIN pg_stat_database AS db ON db.datname = current_database()
"""


class ArchiveReader(Protocol):
    """What MusicBot needs from the archive: liveness for -ping's Postgres row
    and the aggregates behind -leaderboard and -analytics. Structural, like
    ping's ArchiveHealth (which it satisfies), so the cog stays fake-able."""

    async def health_check(self) -> None: ...

    async def leaderboard(
        self, guild_id: int, limit: int, *, since_epoch: float = 0.0
    ) -> Leaderboard: ...

    async def analytics(
        self, guild_id: int, *, days: int, top_n: int
    ) -> AnalyticsMetrics: ...


class HistoryArchive(Protocol):
    """The write surface the drainer and the backfill tool program against.
    recent() reads the durable record; -history does not use it (see
    docs/ARCHITECTURE.md#history-read-path)."""

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None: ...

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]: ...

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None: ...


class SchemaVersionError(RuntimeError):
    """The database's schema is older than the code expects: "run the
    migrations" (an operator action) as distinct from "Postgres is down"."""


class PostgresHistoryArchive:
    """asyncpg-backed archive. All methods raise on failure — callers own the
    error policy (the drainer backs off, the backfill counts the guild failed)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()
        self._closed = False
        # Per instance: one archive owns one pool.
        self._read_slots = asyncio.Semaphore(_READ_CONCURRENCY)
        # Taken in addition to _read_slots, which keeps the reader ceiling at
        # the budget; this one serializes analytics, the heaviest reader,
        # inside it.
        self._analytics_slot = asyncio.Semaphore(1)

    async def _create_pool(self) -> asyncpg.Pool:
        return await asyncpg.create_pool(
            self._url,
            min_size=1,
            max_size=4,
            # Connect bound: a fast failure keeps the drainer's backoff loop
            # responsive (asyncpg's default is 60s).
            timeout=10,
            # Statement bound: a server that accepts the connection then stops
            # answering would otherwise hang executemany with no exception, no
            # backoff and no log line while the outbox grows.
            command_timeout=_COMMAND_TIMEOUT_SECS,
            # Prepared statements are per-connection; POSTGRES_STATEMENT_CACHE=0
            # for a transaction-pooling PgBouncer.
            statement_cache_size=config.POSTGRES_STATEMENT_CACHE,
            # Identifies this bot's connections in pg_stat_activity.
            server_settings={"application_name": "musicbot-history"},
        )

    async def _ensure(self) -> asyncpg.Pool:
        """Lazy pool + schema-version check, double-checked under the lock.
        Refuses after close(); _closed is re-checked under the lock and after
        the awaits because close() can win any of those suspension points, and
        every escape closes the pool it built."""
        if self._closed:
            raise RuntimeError("PostgresHistoryArchive is closed")
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._closed:  # close() may have won the race to the lock
                raise RuntimeError("PostgresHistoryArchive is closed")
            if self._pool is None:
                pool = await self._create_pool()
                try:
                    async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                        await self._assert_schema_version(conn)
                    if self._closed:  # close() ran during our awaits
                        raise RuntimeError("PostgresHistoryArchive is closed")
                except BaseException:
                    # A pool self._pool never received is one close() cannot see.
                    await pool.close()
                    raise
                self._pool = pool
        return self._pool

    @staticmethod
    async def _assert_schema_version(conn: Any) -> None:
        """Verify the database carries the schema this build was written for.
        A newer database is tolerated with a warning: migrations are additive,
        and refusing would turn a rolled-back deploy into an outage. `conn` is
        Any because the pool hands out a PoolConnectionProxy."""
        try:
            version = await conn.fetchval(_SCHEMA_VERSION_SQL)
        except asyncpg.exceptions.UndefinedTableError:
            version = None  # never migrated at all
        if version is None or version < EXPECTED_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"play_history schema is at version {version if version else 'none'}, "
                f"this build needs {EXPECTED_SCHEMA_VERSION}. "
                f"Run `just db-migrate` (or `python -m src.db_migrate`) against "
                f"the archive database."
            )
        if version > EXPECTED_SCHEMA_VERSION:
            log.warning(
                f"play_history schema is at version {version}, ahead of this "
                f"build's {EXPECTED_SCHEMA_VERSION}; continuing (migrations are "
                f"additive), but this build is older than the database."
            )

    async def insert_batch(self, entries: Sequence[HistoryEntry]) -> None:
        """Insert oldest-first; replays and backfill overlap dedup via the
        play_history_dedup unique index. No conversion guard: __post_init__
        already clamped every field into this table's column domain."""
        if not entries:
            return
        rows = [_entry_to_row(e) for e in entries]
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            await conn.executemany(_INSERT_SQL, rows)

    async def recent(self, guild_id: int, limit: int) -> list[HistoryEntry]:
        """The `limit` most recent entries for one guild, newest first. id is
        the tie-break so epoch-0 (unknown-time) entries order stably."""
        if limit <= 0:
            return []
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            rows = await conn.fetch(_RECENT_SQL, guild_id, limit)
        return [_row_to_entry(r) for r in rows]

    async def leaderboard(
        self, guild_id: int, limit: int, *, since_epoch: float = 0.0
    ) -> Leaderboard:
        """Top requesters and songs for one guild, ranked by total played_secs.
        since_epoch 0.0 = all-time, the only window epoch-0 rows appear in.
        Bounded twice: _read_slots keeps a burst of commands off the drainer's
        connections, and the deadline covers the wait for a slot as well as the
        two statements (otherwise bounded only by 2 x command_timeout)."""
        if limit <= 0:
            return Leaderboard(requesters=(), songs=())
        # `> 0.0` is false for NaN, folding it into the all-time cutoff rather
        # than past fromtimestamp's range.
        clamped = min(since_epoch, TS_MAX) if since_epoch > 0.0 else 0.0
        cutoff = datetime.fromtimestamp(clamped, tz=timezone.utc)
        async with asyncio.timeout(_READ_DEADLINE_SECS), self._read_slots:
            pool = await self._ensure()
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                requester_rows = await conn.fetch(
                    _TOP_REQUESTERS_SQL, guild_id, limit, cutoff
                )
                song_rows = await conn.fetch(_TOP_SONGS_SQL, guild_id, limit, cutoff)
        return Leaderboard(
            requesters=tuple(
                RequesterLeader(
                    requester_id=r["requester_id"],
                    requester_name=r["requester_name"],
                    plays=r["plays"],
                    played_secs=r["played_secs"],
                )
                for r in requester_rows
            ),
            songs=tuple(
                SongLeader(
                    title=r["title"],
                    webpage_url=r["webpage_url"],
                    duration_secs=r["duration_secs"],
                    query_source=r["query_source"],
                    plays=r["plays"],
                    played_secs=r["played_secs"],
                )
                for r in song_rows
            ),
        )

    async def analytics(
        self, guild_id: int, *, days: int, top_n: int
    ) -> AnalyticsMetrics:
        """Every -analytics aggregate for one guild and one complete-days window.
        Four statements on one connection: the CTE, then the three top-N
        branches over the window the CTE resolved, so the buckets, the footer
        and the cache TTL agree on when the day turned. _analytics_slot
        serializes analytics, then _read_slots counts it against the reader
        budget. The caller must release before rendering."""
        unit = "week" if days >= WEEKLY_BUCKET_MIN_DAYS else "day"
        async with (
            asyncio.timeout(_READ_DEADLINE_SECS),
            self._analytics_slot,
            self._read_slots,
        ):
            pool = await self._ensure()
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                row = await conn.fetchrow(_ANALYTICS_SQL, guild_id, days, unit)
                if row is None or not row["n_rows"]:
                    # Nothing in the window: the top-N queries could only
                    # return nothing.
                    return AnalyticsMetrics(
                        days=days,
                        bucket_unit=unit,
                        window_start_epoch=float(row["window_start"]) if row else 0.0,
                        window_end_epoch=float(row["window_end"]) if row else 0.0,
                        today_start_epoch=float(row["today_start"]) if row else 0.0,
                    )
                start = datetime.fromtimestamp(
                    float(row["window_start"]), tz=timezone.utc
                )
                end = datetime.fromtimestamp(float(row["window_end"]), tz=timezone.utc)
                listener_rows = await conn.fetch(
                    _ANALYTICS_TOP_LISTENERS_SQL, guild_id, top_n, start, end
                )
                artist_rows = await conn.fetch(
                    _ANALYTICS_TOP_ARTISTS_SQL, guild_id, top_n, start, end
                )
                song_rows = await conn.fetch(
                    _ANALYTICS_TOP_SONGS_SQL, guild_id, top_n, start, end
                )
        return _row_to_metrics(row, listener_rows, artist_rows, song_rows, days, unit)

    async def record_rejection(
        self,
        entry: HistoryEntry,
        error: BaseException,
        trace_id: str = "",
        wire: Optional[bytes] = None,
    ) -> None:
        """Park one refused play in play_history_rejected. Best-effort and
        terminal: never retries, a failed insert goes to the log. Reachable
        only on a rejection, so the server is up. error_detail is NUL-scrubbed
        and capped, or the poison being recorded fails the record. payload is
        the delivered `wire` verbatim: a newer build's entry re-serialized by
        this parser would be lossy, and the dedup identity is an md5 of payload,
        so one entry seen by two builds would count as two failures."""
        detail = str(error).replace("\x00", "")[:_REJECT_DETAIL_MAX]
        # serialize_history_entry cannot raise: __post_init__ already proved the
        # entry orjson-encodable.
        payload = wire if wire is not None else serialize_history_entry(entry)
        try:
            pool = await self._ensure()
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                await conn.execute(
                    _REJECT_SQL,
                    entry.guild_id,
                    type(error).__name__,
                    detail,
                    trace_id,
                    payload,
                )
        except Exception as e:
            log.error(
                f"play rejected AND unrecordable ({type(e).__name__}: {e}); "
                f"payload={payload!r}"
            )

    async def stats(self) -> ArchiveStats:
        """Size, row estimates, connection/cache state and cumulative counters
        for -debug's Postgres block, which calls this twice per snapshot and
        rates the counters over `monotonic`. Raises on failure. Bounded like
        leaderboard(): a diagnostic must never starve the archive."""
        async with asyncio.timeout(_READ_DEADLINE_SECS), self._read_slots:
            pool = await self._ensure()
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
                row = await conn.fetchrow(_STATS_SQL)
        return ArchiveStats(
            database_bytes=row["database_bytes"],
            table_bytes=row["table_bytes"],
            rows_estimate=row["rows_estimate"],
            rejected_estimate=row["rejected_estimate"],
            connections=row["connections"],
            max_connections=row["max_connections"],
            shared_buffers=row["shared_buffers"],
            cache_hit_ratio=row["cache_hit_ratio"],
            active_backends=row["active_backends"],
            active_io_wait=row["active_io_wait"],
            active_lock_wait=row["active_lock_wait"],
            active_other_wait=row["active_other_wait"],
            active_time_ms=row["active_time_ms"],
            xacts_total=row["xacts_total"],
            tuples_total=row["tuples_total"],
            blks_hit=row["blks_hit"],
            blks_read=row["blks_read"],
            temp_bytes=row["temp_bytes"],
            deadlocks=row["deadlocks"],
            monotonic=time.monotonic(),
        )

    async def health_check(self) -> None:
        """Is the server answering? Raises on failure — -ping's probe times it
        into a red row. Connects if nothing has yet: before the first song end
        the lazy pool is absent, and "not configured" for an enabled tier would
        be a lie. SELECT 1, not a table read: _ensure settled the schema."""
        pool = await self._ensure()
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT_SECS) as conn:
            await conn.execute("SELECT 1")

    async def close(self) -> None:
        """Close the pool. Terminal: _ensure() refuses afterwards. _closed is set
        first so new callers are turned away, then the init lock is taken so an
        _ensure() mid-connect cannot hand a pool to a field nobody reads again
        (up to the 10s connect timeout). MusicBotApp.close() runs
        drainer.stop() before this so the final drain can reach Postgres."""
        self._closed = True
        async with self._init_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            try:
                async with asyncio.timeout(_POOL_CLOSE_TIMEOUT_SECS):
                    await pool.close()
            except asyncio.CancelledError:
                pool.terminate()  # don't leave sockets behind on the way out
                raise
            except Exception as e:
                # A graceful close waits on in-flight queries a hung server
                # never releases; raising here would abort every later step of
                # MusicBotApp.close(). terminate() is synchronous.
                pool.terminate()
                log.warning(
                    f"history archive pool close forced: {type(e).__name__}: {e}"
                )


# Errors meaning "this data will never be accepted", not "try again later".
# Expected unreachable since HistoryEntry.__post_init__ clamps into the column
# domain; the backstop for a validator regression or an unexpected schema.
#
#   DataError             SQLSTATE 22xxx server-side data rejections
#   CheckViolationError   23514 — not a DataError, and without it a CHECK
#   NotNullViolationError violation wedges the drain head permanently
#   UndefinedColumnError  42703 — the database is older than this build, which
#                         migrate() cannot see. Treated as transient it would
#                         redeliver forever onto the non-evictable outbox
#
# Not here, each would break the drain: UndefinedTableError (the rejection
# insert cannot land either), UniqueViolationError (the ON CONFLICT target;
# catching it hides an index bug), bare ValueError/TypeError (an ordinary bug
# in insert_batch would dead-letter a healthy batch), and anything
# OSError-shaped (a restart or failover would delete healthy history).
_POISON = (
    asyncpg.exceptions.DataError,
    asyncpg.exceptions.CheckViolationError,
    asyncpg.exceptions.NotNullViolationError,
    asyncpg.exceptions.UndefinedColumnError,
)


class HistoryOutboxDrainer:
    """The one task per process that drains the Redis outbox into the archive.
    Wakes on notify() with a periodic fallback tick, drains in batches until the
    outbox is empty, and on failure backs off exponentially while entries
    accumulate in the outbox (persistent, non-evictable — see HISTORY_OUTBOX_KEY).
    Not single-consumer: a second instance on the same consumer group reads
    disjoint new entries and shares the pending set, which the archive's unique
    index collapses; _REJECT_SQL's ON CONFLICT does the same for rejections."""

    BATCH_SIZE: int = 100
    TICK_SECS: float = 30.0
    DEPTH_ALARM: int = 10_000  # backlog that escalates the retry warning to ERROR
    # Whole-cycle bound over command_timeout: covers connection acquisition and
    # anything else asyncpg does not bound, so a hang becomes a TimeoutError on
    # _run's error path and DEPTH_ALARM fires for hangs as well as errors.
    DRAIN_DEADLINE_SECS: float = 60.0
    # Rate limit for the depth watchdog on the productive path; matched to
    # TICK_SECS so a busy drain reports a growing backlog on an idle one's cadence.
    DEPTH_SAMPLE_INTERVAL_SECS: float = 30.0
    # PEL sweep (reclaim_outbox_stale). Slow: what it catches is rare, and it
    # costs an XAUTOCLAIM scan.
    SWEEP_INTERVAL_SECS: float = 300.0
    # INVARIANT: must exceed DRAIN_DEADLINE_SECS * 1000. "idle" is measured from
    # last delivery, so a smaller value lets the sweep reclaim a live sibling's
    # batch while it is still inserting.
    SWEEP_MIN_IDLE_MS: int = 300_000
    # Bounds one sweep's work, and terminates the cursor loop under fakeredis,
    # which returns the last-scanned ID where real Redis returns "0-0".
    SWEEP_MAX_PASSES: int = 20
    # Respawn damping for a drainer task that dies outside its own error
    # handling; the cap keeps a hard-broken drainer from flooding the log.
    RESTART_BASE: float = 5.0
    RESTART_MAX: float = 300.0
    # 0 = unbounded, the durability default. See config.HISTORY_OUTBOX_MAX.
    OUTBOX_MAX: int = config.HISTORY_OUTBOX_MAX
    # Bounds one _enforce_cap pass's XRANGE, which carries bodies (there is no
    # ID-only form): uncapped, a 500k backlog would haul the whole overage over
    # the socket in one reply. 10k entries ≈ 5 MB on the wire; resident cost per
    # entry steps with the allocator bin (~548 B, ~626 B from 500 B of wire).
    # See docs/ARCHITECTURE.md#why-query_source-is-stored-rather-than-derived.
    CAP_PAGE: int = 10_000
    _BACKOFF_START: float = 1.0
    _BACKOFF_MAX: float = 60.0

    def __init__(self, redis: aioredis.Redis, archive: HistoryArchive) -> None:
        self._redis = redis
        self._archive = archive
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        # Shutdown/supervision state.
        self._stop_lock = asyncio.Lock()
        self._stopped = False
        self._stopping = False
        self._restart_delay = self.RESTART_BASE
        self._respawn_handle: Optional[asyncio.TimerHandle] = None
        # Monotonic deadlines; None = due now (start()).
        self._next_sweep: Optional[float] = None
        self._next_depth_sample: Optional[float] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stopping = False
        # stop() latches _stopped; without this reset a start-after-stop would
        # spawn a _run the next stop() returns early from without cancelling.
        self._stopped = False
        self._restart_delay = self.RESTART_BASE
        self._spawn()

    def _spawn(self) -> None:
        self._task = asyncio.create_task(self._run(), name="history-outbox-drainer")
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Supervision. _run only ever exits via cancellation, so any exception
        here is a bug that leaves the non-evictable outbox growing: log loudly
        and restart with exponential damping."""
        if self._stopping or task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return  # _run never returns normally; defensive
        log.error(
            f"history outbox drainer died unexpectedly "
            f"({type(exc).__name__}: {exc}); restarting in "
            f"{self._restart_delay:.0f}s",
            exc_info=exc,
        )
        self._respawn_handle = asyncio.get_running_loop().call_later(
            self._restart_delay, self._respawn
        )
        self._restart_delay = min(self._restart_delay * 2, self.RESTART_MAX)

    def _respawn(self) -> None:
        self._respawn_handle = None
        if not self._stopping:
            self._spawn()

    def notify(self) -> None:
        """Signal a fresh outbox push — cheap, sync, callable from anywhere."""
        self._wake.set()

    async def stop(self, timeout: float = 5.0) -> None:
        """Cancel the loop, then one bounded final-drain attempt. Never raises;
        anything left stays in the outbox for the next start. Reentrant:
        discord.py calls close() from run()'s finally as well as on demand, and
        the lock makes a second caller wait for the first rather than return
        early into a still-draining shutdown."""
        async with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stopping = True
            if self._respawn_handle is not None:
                self._respawn_handle.cancel()
                self._respawn_handle = None
            task = self._task
            self._task = None
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    # A task that already finished with an exception (respawn
                    # pending) re-raises it here; letting that escape would
                    # skip the final drain below.
                    log.warning(f"history drainer task ended in error: {e}")
            try:
                async with asyncio.timeout(timeout):
                    while await self._drain_once():
                        pass
            except Exception as e:
                log.warning(f"history outbox final drain incomplete: {e}")

    # ── Drain loop ───────────────────────────────────────────────────────────

    async def _run(self) -> None:
        backoff = self._BACKOFF_START
        while True:
            try:
                await self._sweep_if_due()
                drained = await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The cap is enforced here as well as on _drain_batch's success
                # tail, which is unreachable for the whole of a Postgres outage
                # — the one scenario the cap exists for. Depth is an O(1) XLEN.
                # The batch in flight is delivered and unacked here, and
                # _enforce_cap trims across the PEL; safe because everything it
                # destroys is XACKed first, so a trimmed in-flight batch is
                # dropped-and-logged, never a tombstone.
                await self._enforce_cap_quietly()
                await self._log_retry(e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._BACKOFF_MAX)
                continue
            backoff = self._BACKOFF_START
            if drained:
                # A cycle that delivered proves the drainer healthy.
                self._restart_delay = self.RESTART_BASE
                # "Keeping up but not catching up" is a succeeding drain that
                # takes this branch every time, so it must sample here too.
                await self._sample_depth_if_due()
                continue  # backlog: keep draining without waiting
            # Idle. _log_retry only fires in the except arm, so without this a
            # healthy drainer on a growing outbox (or a stranded PEL) says nothing.
            await self._sample_depth()
            # A notify() landing between wait() and clear() is dropped, but the
            # next iteration drains unconditionally, so it costs at most one
            # TICK_SECS delay.
            try:
                async with asyncio.timeout(self.TICK_SECS):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()

    async def _sweep_if_due(self) -> None:
        """PEL housekeeping every SWEEP_INTERVAL_SECS and once at the first
        cycle after start() — see reclaim_outbox_stale(). Non-fatal, but
        besides XACK it is the only thing that clears a tombstone on Redis 7."""
        loop = asyncio.get_running_loop()
        if self._next_sweep is not None and loop.time() < self._next_sweep:
            return
        self._next_sweep = loop.time() + self.SWEEP_INTERVAL_SECS
        try:
            reclaimed, purged = await reclaim_outbox_stale(
                self._redis,
                min_idle_ms=self.SWEEP_MIN_IDLE_MS,
                count=self.BATCH_SIZE,
                max_passes=self.SWEEP_MAX_PASSES,
            )
        except Exception as e:
            log.warning(f"history outbox PEL sweep failed: {e}")
            return
        if reclaimed or purged:
            # INFO: a healthy drainer's own work is invisible in the log, so a
            # stranded PEL being reclaimed would otherwise leave no trace.
            log.info(
                f"history outbox PEL sweep reclaimed {reclaimed} stale "
                f"entries and purged {purged} tombstones"
            )

    async def _drain_once(self) -> int:
        """One batch, under the whole-cycle deadline. Returns entries SETTLED —
        callers loop on it, and "batch was empty" is not a sufficient stop
        condition: a tombstone the cycle could not ack redelivers forever."""
        async with asyncio.timeout(self.DRAIN_DEADLINE_SECS):
            return await self._drain_batch()

    async def _read_batch(self) -> list[OutboxEntry]:
        """Pending first, then new — never both in one cycle: the PEL stays
        within BATCH_SIZE x concurrent readers only while a cycle never reads
        `>` with a non-empty PEL, and the drainer's memory bound is sized on
        that. NOGROUP is healed here, not only at startup: deleting the key
        destroys its groups and XADD recreates it groupless, after which every
        read fails identically forever. Rebuild and retry once."""
        for attempt in range(2):
            try:
                pending = await read_outbox_pending(self._redis, self.BATCH_SIZE)
                if pending:
                    return pending
                return await read_outbox_new(self._redis, self.BATCH_SIZE)
            except ResponseError as e:
                if attempt or not str(e).startswith("NOGROUP"):
                    raise
                log.warning(
                    "history outbox consumer group is missing (the key was "
                    "deleted); recreating it and retrying"
                )
                await ensure_outbox_group(self._redis)
        return []  # unreachable: the loop either returns or raises

    async def _drain_batch(self) -> int:
        """Read a batch, insert it, settle it by ID. Every entry was produced by
        HistoryEntry.__post_init__, so a refusal is a validator regression,
        schema drift or an unstamped guild_id (0, which __post_init__ declines
        to fix up): recorded to play_history_rejected, dropped, batch continues.
        Corrupt bytes are dropped and still settled, or they wedge the queue
        head forever. Tombstones are separated out first: no bytes to parse."""
        batch = await self._read_batch()
        if not batch:
            return 0
        with _tracer.start_as_current_span("history.drain") as span:
            span.set_attribute("drain.batch", len(batch))
            settled = await self._settle_tombstones(batch, span)
            live = [e for e in batch if e.wire is not None]
            raw = [e.wire for e in live if e.wire is not None]
            entries = [e for e in map(parse_history_entry, raw) if e is not None]
            span.set_attribute("drain.parsed", len(entries))
            try:
                if entries:
                    await self._archive.insert_batch(entries)
            except _POISON:
                # _isolate settles each entry itself; try/except/else keeps the
                # two settle paths exclusive.
                await self._isolate(live, span)
            else:
                await retire_outbox(self._redis, [e.id for e in live])
            settled += len(live)
            await self._enforce_cap()
            return settled

    async def _settle_tombstones(self, batch: list[OutboxEntry], span: Span) -> int:
        """Ack and log entries whose body is gone. Returns how many. XTRIM and
        an operator XDEL remove bodies without consulting the PEL, so the ID
        replays with an empty field map — a lost PLAY, logged at ERROR. The ack
        is unconditional: left pending, a tombstone is re-read every cycle
        forever with no error to escalate, a permanent silent stall."""
        tombstones = [e.id for e in batch if e.wire is None]
        if not tombstones:
            return 0
        await retire_outbox(self._redis, tombstones)
        span.set_attribute("drain.tombstones", len(tombstones))
        log.error(
            f"history outbox delivered {len(tombstones)} entries whose payload "
            f"had already been deleted (ids "
            f"{b', '.join(tombstones[:5]).decode()}"
            f"{'…' if len(tombstones) > 5 else ''}) — those plays are lost and "
            f"cannot reach Postgres; acked so the drain can make progress"
        )
        return len(tombstones)

    async def _isolate(self, batch: list[OutboxEntry], span: Span) -> None:
        """One batch refused: retry it row by row so one bad row costs one row.
        Settles PER ENTRY, never once at the end — a transient error partway
        through would redeliver rows already recorded as rejected, and singles
        cost ~22x a batch, so on a degraded server this pass can exceed
        DRAIN_DEADLINE_SECS and must be resumable. Iterates the delivered batch,
        not what parsed: a corrupt element settled by neither path is
        delivered-and-unacked forever. A transient error raises out, leaving the
        rest to redeliver; _REJECT_SQL's ON CONFLICT absorbs the replay."""
        rejected = 0
        trace_id = trace_id_of(trace.get_current_span())
        for item in batch:
            entry = parse_history_entry(item.wire) if item.wire is not None else None
            if entry is not None:
                try:
                    await self._archive.insert_batch([entry])
                except _POISON as e:
                    # item.wire, not the re-serialized entry — see
                    # record_rejection's docstring on mixed-version rollouts.
                    await self._archive.record_rejection(entry, e, trace_id, item.wire)
                    rejected += 1
                    log.error(
                        f"play_history refused a row ({type(e).__name__}) — the "
                        f"HistoryEntry validator regressed or the schema "
                        f"drifted: {entry.title[:60]!r} / "
                        f"{entry.webpage_url[:80]}"
                    )
            # else: corrupt, dropped — but still settled below.
            await retire_outbox(self._redis, [item.id])
        span.set_attribute("drain.rejected", rejected)

    async def _enforce_cap(self) -> None:
        """Opt-in outbox ceiling (config.HISTORY_OUTBOX_MAX, default off). Every
        drop is data loss and logs at ERROR. Never runs while shutting down (a
        departing process knows least about a live peer) and re-checks each
        pass, so stop() halts a convergence loop in progress.

        ACK BEFORE TRIM: XTRIM does not consult the PEL, and trimming first
        leaves an ID pending with no body — a tombstone, which replays forever.
        A crash between the two leaves entries acked but not trimmed, reclaimed
        next pass. The cap crosses the PEL on purpose: during an outage the
        oldest entries are permanently in flight, so a clamp below the oldest
        pending ID would trim nothing while the backlog grows.

        Paged (CAP_PAGE) until depth is back at the cap: the trim is one MINID
        command however large the tranche, but minid discovery is an XRANGE
        that carries bodies, and one COUNT=overage fetch would run unbounded."""
        if not self.OUTBOX_MAX:
            return
        while not self._stopping:
            depth = await outbox_depth(self._redis)
            if depth <= self.OUTBOX_MAX:
                return
            over = depth - self.OUTBOX_MAX
            page = min(over, self.CAP_PAGE)
            # The ID of the (page+1)-th oldest entry is this pass's exclusive
            # upper bound: everything strictly below it is the tranche.
            oldest = cast(
                list[tuple[bytes, dict[bytes, bytes]]],
                await self._redis.xrange(HISTORY_OUTBOX_KEY, count=page + 1),
            )
            if len(oldest) <= page:
                # Raced shorter than the page by a concurrent drain; the next
                # pass's depth check settles whether anything is left over.
                return
            minid = cast(bytes, oldest[-1][0])
            in_flight = await outbox_pending_below(self._redis, minid)
            await ack_outbox(self._redis, in_flight)
            dropped = await trim_outbox_below(self._redis, minid)
            if not dropped:
                return
            # XTRIM's returned count, never depth - OUTBOX_MAX: XLEN over-counts
            # acked-but-undeleted entries.
            log.error(
                f"history outbox over cap (depth={depth}, HISTORY_OUTBOX_MAX="
                f"{self.OUTBOX_MAX}); dropped {dropped} oldest entries — those "
                f"plays are lost and will not reach Postgres"
            )
            if in_flight:
                log.error(
                    f"{len(in_flight)} of those entries were already delivered "
                    f"to a drainer; their pending records were cleared so the "
                    f"trim could not leave them replaying forever"
                )

    async def _enforce_cap_quietly(self) -> None:
        """_enforce_cap for the failure path, where Redis may itself be what
        broke. Never raises: a second error would cost the caller its backoff
        and turn a Redis blip into a hot retry loop."""
        try:
            await self._enforce_cap()
        except Exception as e:
            log.warning(f"history outbox cap check failed: {e}")

    async def _sample_depth_if_due(self) -> None:
        """_sample_depth for the productive path, rate-limited: a cycle that
        drained `continue`s straight into the next batch, and sampling there
        unconditionally adds two round trips per BATCH_SIZE entries on the
        Redis that also serves playback."""
        now = asyncio.get_running_loop().time()
        if self._next_depth_sample is not None and now < self._next_depth_sample:
            return
        await self._sample_depth()

    async def _sample_depth(self) -> None:
        """Report a backlog on the success path. Best-effort and silent when
        shallow: a watchdog, not a metric. Stamps the rate-limit deadline itself
        so the idle and productive paths share one clock."""
        self._next_depth_sample = (
            asyncio.get_running_loop().time() + self.DEPTH_SAMPLE_INTERVAL_SECS
        )
        try:
            depth = await outbox_depth(self._redis)
            pending = await outbox_pending_count(self._redis)
        except Exception:
            return
        if depth >= self.DEPTH_ALARM:
            log.error(
                f"history outbox backlog is {depth} entries ({pending} "
                f"delivered and unacked) while the drain is SUCCEEDING — "
                f"Postgres is keeping up but not catching up"
            )

    async def _log_retry(self, error: Exception, backoff: float) -> None:
        try:
            depth = await outbox_depth(self._redis)
        except Exception:
            depth = -1  # Redis itself is down; depth unknowable
        emit = log.error if depth >= self.DEPTH_ALARM else log.warning
        emit(
            f"history outbox drain failed (backlog={depth}): "
            f"{type(error).__name__}: {error}; retrying in {backoff:.0f}s"
        )
