-- Play-history archive: the durable long-term home for every played song.
-- Extracted verbatim from PostgresHistoryArchive._SCHEMA_DDL, which used to run
-- this on first connection (docs/POSTGRES_HISTORY_PLAN.md §4).
--
-- The IF NOT EXISTS clauses are load-bearing on the upgrade path, not habit:
-- deployments that predate the migration runner already have this table from
-- the old first-use DDL, and this file must record itself as applied there
-- without failing.
--
-- The zero-value convention ("0 / empty string = unknown") carries over from
-- the wire format — no NULLs. Deliberate: standard unique indexes treat NULLs
-- as distinct, which would break dedup exactly on the unknown-played_at rows
-- that need it most. played_at epoch 0 = unknown, same sentinel as the wire.
CREATE TABLE IF NOT EXISTS play_history (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    guild_id       bigint      NOT NULL,
    title          text        NOT NULL DEFAULT '',
    webpage_url    text        NOT NULL DEFAULT '',
    duration_secs  integer     NOT NULL DEFAULT 0,
    played_secs    integer     NOT NULL DEFAULT 0,
    requester_id   bigint      NOT NULL DEFAULT 0,
    requester_name text        NOT NULL DEFAULT '',
    thumbnail      text        NOT NULL DEFAULT '',
    uploader       text        NOT NULL DEFAULT '',
    played_at      timestamptz NOT NULL DEFAULT to_timestamp(0)
);

-- Dedup for at-least-once delivery (drainer redelivery) and backfill overlap.
-- Uniqueness only — it does NOT serve the -history read. It was once assumed to,
-- on the grounds that equal-timestamp groups are tiny; backfilled guilds can
-- carry tens of thousands of rows at the epoch-0 sentinel, which made that read
-- 37x slower. See migrations/0003_play_history_recent_idx.sql.
--
-- Known edge: two genuinely distinct plays of the same URL in one guild collapse
-- into one row when both carry the epoch-0 "unknown time" sentinel (legacy
-- entries imported by backfill_history, or values _sanitize_entry clamped).
-- Accepted: a silent merge of indistinguishable pre-archive rows is better than
-- a wider key that stops deduping the drainer's redeliveries, which is what this
-- index exists for.
CREATE UNIQUE INDEX IF NOT EXISTS play_history_dedup
    ON play_history (guild_id, played_at, webpage_url);
