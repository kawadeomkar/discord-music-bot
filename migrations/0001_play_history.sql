-- The play-history schema: the durable long-term home for every played song,
-- plus the reject table that catches anything the server refuses.
--
-- Deliberately ONE migration. Nothing has shipped yet, so there is no database
-- anywhere whose schema this has to evolve — and a pre-release sequence of
-- ALTER TABLE / ADD CONSTRAINT steps would be pure fiction, describing upgrades
-- that never happened to anyone. Until the first production release this file is
-- edited IN PLACE: add the column here, re-run `just db-migrate` against a
-- scratch database, done. After that release the rule inverts and every change
-- becomes a new numbered migration, because from then on the ALTERs are real.
--
-- Consequence worth knowing: the IF NOT EXISTS clauses make this whole file a
-- no-op against a database that already holds these tables, so a dev box that
-- applied an earlier shape does NOT get updated — drop the tables (or the
-- database) and re-run. That is the right trade only for as long as the "no
-- deployment anywhere" premise above holds; it is the first thing that stops
-- being true at release.
--
-- The zero-value convention ("0 / empty string = unknown") carries over from
-- the wire format — no NULLs. Deliberate: standard unique indexes treat NULLs
-- as distinct, which would break dedup exactly on the unknown-played_at rows
-- that need it most. played_at epoch 0 = unknown, same sentinel as the wire.
--
-- The CHECK constraints are the schema lock, and they are NOT input validation:
-- the wire side (HistoryEntry in src/guild_state.py) is what clamps values into
-- this domain before an insert is attempted (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md
-- §5.1). A violation here is therefore never bad user data — it is that validator
-- regressing, or a build talking to a schema it was not written for. Being loud is
-- the entire point, and play_history_rejected below is where the evidence lands.
-- Declared inline rather than as NOT VALID ADD CONSTRAINT steps: on a table
-- created empty they validate for free. A constraint added AFTER release wants
-- NOT VALID instead, to stay off the full-scan, ACCESS EXCLUSIVE path on a table
-- that by then holds real rows.
--
-- webpage_url is deliberately unconstrained. The validator stores '' rather than
-- rejecting it, because the read path runs that same validator over rows that
-- already contain it (see the dedup index below, which documents that merge as
-- accepted). See findings §5.3 and open decision §11.1.
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
    played_at      timestamptz NOT NULL DEFAULT to_timestamp(0),

    -- When the row reached Postgres, as opposed to when the song was played.
    -- played_at is a client clock captured at song end and can be arbitrarily
    -- far in the past for backfilled or long-buffered entries; nothing else
    -- records how far behind the archive actually was. This gives outage
    -- forensics ("everything from the 14th landed on the 16th") and backfill
    -- auditing (`WHERE inserted_at > <backfill start>`) a column to ask about.
    -- Not in _INSERT_SQL / _RECENT_SQL on purpose: the default fills it, and
    -- HistoryEntry stays exactly the wire schema.
    inserted_at    timestamptz NOT NULL DEFAULT now(),

    -- The Discord message id of the Now Playing embed that hosted this song, so
    -- a history row can be joined back to the message a listener actually saw.
    --
    -- 0 = unknown, and for now that is the only value it ever holds: nothing
    -- writes this table yet (see README, "Operating the play-history archive"),
    -- and HistoryEntry carries no message_id, so the column reads 0 until the
    -- wire schema grows the field and the archive's row mapping passes it. It
    -- stays the right sentinel afterwards — a stream that never produced audio
    -- has its NP block retired rather than finalized, and backfilled rows
    -- predate the field entirely.
    --
    -- Not a foreign key and not unique, deliberately. This points at Discord
    -- state the bot does not own: the NP host migrates across messages during
    -- one song (MusicContext.send re-hosts the block on every command response
    -- in the home channel), a dedicated NP message is DELETED when it is
    -- retired, and users can delete messages themselves. The id is a forensic
    -- breadcrumb, not a reference that can be kept valid.
    message_id     bigint      NOT NULL DEFAULT 0,

    CONSTRAINT play_history_guild_id_valid    CHECK (guild_id > 0),
    CONSTRAINT play_history_requester_valid   CHECK (requester_id >= 0),
    CONSTRAINT play_history_played_secs_valid CHECK (played_secs >= 0),
    CONSTRAINT play_history_duration_valid    CHECK (duration_secs >= 0),
    CONSTRAINT play_history_message_id_valid  CHECK (message_id >= 0)
);

-- Dedup for at-least-once delivery (drainer redelivery) and backfill overlap.
-- Uniqueness only — it does NOT serve the -history read. It was once assumed to,
-- on the grounds that equal-timestamp groups are tiny; backfilled guilds can
-- carry tens of thousands of rows at the epoch-0 sentinel, which made that read
-- 37x slower. See play_history_recent below.
--
-- message_id is emphatically NOT part of this key. A wider key stops deduping
-- the drainer redeliveries the index exists for, and that column would defeat it
-- outright: the NP host id is not stable across a song's life, so two deliveries
-- of the same play can legitimately disagree on it and would land as two rows.
--
-- Known edge: two genuinely distinct plays of the same URL in one guild collapse
-- into one row when both carry the epoch-0 "unknown time" sentinel (legacy
-- entries imported by backfill_history, or values the validator clamped).
-- Accepted: a silent merge of indistinguishable pre-archive rows is better than
-- a wider key that stops deduping the drainer's redeliveries, which is what this
-- index exists for.
CREATE UNIQUE INDEX IF NOT EXISTS play_history_dedup
    ON play_history (guild_id, played_at, webpage_url);

-- The -history read index.
--
-- play_history_dedup (guild_id, played_at, webpage_url) only PARTIALLY serves
-- `WHERE guild_id = $1 ORDER BY played_at DESC, id DESC LIMIT $2`: played_at is
-- presorted but id is not, so Postgres plans an Incremental Sort and must
-- consume an entire equal-played_at group before it can emit the first row.
--
-- Equal-played_at groups are not rare and not small. parse_history_entry
-- defaults a missing played_at to 0.0, the validator collapses NaN and
-- out-of-range values to 0.0, and backfill_history imports legacy entries that
-- predate the field entirely — so a backfilled guild can carry tens of
-- thousands of rows at to_timestamp(0). Measured on postgres:18 with 1M rows,
-- a guild with 50k epoch-0 rows:
--
--   Limit (actual time=78.580..78.602 rows=50)
--     -> Incremental Sort  Presorted Key: played_at
--        -> Index Scan Backward using play_history_dedup  rows=50000
--                                                   Buffers: shared hit=9964
--
--   p50 49.98ms / max 136.5ms, vs p50 1.34ms for an ordinary guild.
--
-- This index matches the ORDER BY exactly, so the plan becomes a plain backward
-- index scan with no sort node and a bounded buffer count.
--
-- Plain CREATE INDEX, not CONCURRENTLY: db_migrate runs each migration inside a
-- transaction and CONCURRENTLY cannot. It takes a brief write lock on
-- play_history; the only writer is the drainer, which retries.
CREATE INDEX IF NOT EXISTS play_history_recent
    ON play_history (guild_id, played_at DESC, id DESC);

-- Expected to stay empty forever. Every entry reaching the drainer is insertable by
-- construction (the wire-side validator clamps into the play_history column domain),
-- so a row here means exactly one of two things: that validator regressed, or this
-- build is talking to a schema it was not written for. Both are bugs, and both want
-- the payload preserved verbatim for replay. Treat any row as page-worthy — it is a
-- code defect, not a data problem.
--
-- payload is bytea, NOT jsonb and NOT text, and the obvious choice is wrong. A NUL byte
-- is the first poison vector (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md §3.2) and both jsonb
-- ("unsupported Unicode escape sequence") and text ("null character not permitted")
-- REFUSE it — so either would be unable to store the exact class of row this table
-- exists to capture, and the reject insert would itself fail. bytea takes the raw orjson
-- bytes verbatim, which is also what makes replay exact. Inspect with
-- encode(payload, 'escape'), never convert_from(payload, 'UTF8') — that raises on the
-- invalid UTF-8 a corrupt payload may carry.
--
-- No index. The table is expected to hold zero rows and the only query is
-- ORDER BY rejected_at DESC LIMIT n; a sequential scan over an empty table is free, and
-- an index now would be speculative.
CREATE TABLE IF NOT EXISTS play_history_rejected (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rejected_at  timestamptz NOT NULL DEFAULT now(),
    guild_id     bigint      NOT NULL DEFAULT 0,
    error_type   text        NOT NULL DEFAULT '',
    error_detail text        NOT NULL DEFAULT '',
    trace_id     text        NOT NULL DEFAULT '',
    payload      bytea       NOT NULL
);
