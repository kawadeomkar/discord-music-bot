-- Baseline: the play-history table (docs/POSTGRES_HISTORY_PLAN.md §4.1).
--
-- Zero-value convention carried over from the Redis wire format — no NULLs
-- (standard unique indexes treat NULLs as distinct, which would break dedup
-- exactly on the unknown-played_at rows that need it most). played_at epoch 0
-- means "unknown", same sentinel as the wire.
--
-- IF NOT EXISTS guards are politeness for databases that predate the ledger
-- (Phase A applied this DDL from code); the schema_migrations ledger, not the
-- guards, is what prevents re-runs.

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

-- Dedup key (idempotent backfill + at-least-once drain) AND the -history read
-- index: leading (guild_id, played_at) serves ORDER BY played_at DESC via
-- backward scan.
CREATE UNIQUE INDEX IF NOT EXISTS play_history_dedup
    ON play_history (guild_id, played_at, webpage_url);
