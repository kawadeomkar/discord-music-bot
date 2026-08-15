-- The play-history schema: the durable long-term home for every played song,
-- plus the reject table that catches anything the server refuses.
--
-- Deliberately ONE migration. No deployment holds this schema, so there is no
-- database whose shape this has to evolve, and a sequence of ALTER steps would
-- describe upgrades that never happened to anyone. While that holds, this file
-- is edited IN PLACE.
--
-- The trigger for freezing it is a DEPLOYED database, not a tagged release:
-- v1.3.0 onward all ship this file and v2.4.0 onward all connect, yet editing
-- it stays safe for exactly as long as no instance has run one. Once one has,
-- every change becomes a new numbered migration, because from then on the
-- ALTERs are real.
--
-- Consequence while the window is open: migrate() skips a version already in
-- schema_migrations WITHOUT reading the file, and IF NOT EXISTS makes the DDL a
-- no-op anyway, so a database on an earlier shape does NOT get updated and
-- still reports version 1. Dropping the tables is not enough either — the
-- ledger row survives and the re-run applies nothing. Drop the database (or the
-- compose volume) and re-run.
--
-- The zero-value convention ("0 / empty string = unknown") carries over from
-- the wire format — no NULLs. Deliberate: unique indexes treat NULLs as
-- distinct, which would break dedup exactly on the unknown-played_at rows that
-- need it most. played_at epoch 0 = unknown, same sentinel as the wire.
--
-- The CHECK constraints are the schema lock, and they are NOT input validation:
-- the wire side (HistoryEntry in src/guild_state.py) clamps values into this
-- domain before an insert is attempted (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md
-- §5.1). A violation is therefore never bad user data — it is that validator
-- regressing, or a build talking to a schema it was not written for. Being loud
-- is the point, and play_history_rejected below is where the evidence lands.
-- Inline because on a table created empty they validate for free; a constraint
-- added after release wants NOT VALID instead, to stay off the full-scan,
-- ACCESS EXCLUSIVE path.
--
-- webpage_url is deliberately unconstrained: the validator stores '' rather than
-- rejecting it, because the read path runs that same validator over rows that
-- already contain it (findings §5.3, open decision §11.1).
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
    -- When the audio STARTED, one value per PLAY rather than per fragment: every
    -- An interjection's resume tail inherits the played_at of what it continues.
    -- So rows are not in recording order, and an interrupted play's [played_at,
    -- played_at + played_secs] interval overlaps the songs that interrupted it —
    -- "what was playing at time T" has more than one answer here.
    played_at      timestamptz NOT NULL DEFAULT to_timestamp(0),

    -- When the user ASKED for the song, and how many songs were ahead of it at
    -- that moment, counting the one playing. Epoch 0 = unknown (the same
    -- sentinel as played_at), and 0 = played immediately; both are also what an
    -- entry written before these fields existed parses as.
    --
    -- queued_at is the command message's snowflake time, on DISCORD's clock
    -- while played_at is on the host's, so their difference is cross-clock and
    -- goes slightly negative under drift. A `played_at >= queued_at` filter
    -- drops those real plays.
    --
    -- queue_position is depth at ASK, left approximate against where the song
    -- landed by the loop's continuous dequeuing, so it disagrees with played_at
    -- ordering. Rows imported by `just db-backfill` predate the switch and hold
    -- the older meaning of both columns: insert time, and position at insert.
    queued_at      timestamptz NOT NULL DEFAULT to_timestamp(0),
    queue_position integer     NOT NULL DEFAULT 0,

    -- How the song was asked for: the literal 'search' for a plaintext term, or
    -- the host of the link that was pasted ('spotify.com', 'youtube.com',
    -- 'tiktok.com', …). '' = unknown, which is what every row written before this
    -- column parses as.
    --
    -- NOT derivable from webpage_url, which is why it is stored: a Spotify link
    -- resolves to a YouTube title search and archives with a youtube.com URL, as
    -- does a plaintext search, so those two and a pasted YouTube link are
    -- indistinguishable there. Classified at parse time in src/sources.py and
    -- carried on the queue entry, so a Spotify PLAYLIST track — resolved to a
    -- YouTube URL only at dequeue — still records where it came from.
    query_source   text        NOT NULL DEFAULT '',

    -- When the row reached Postgres, as opposed to when the song was played.
    -- played_at is a client clock captured when the audio started and can be
    -- arbitrarily far in the past for backfilled or long-buffered entries, so
    -- this is what outage forensics and backfill auditing (`WHERE inserted_at >
    -- <start>`) can ask about. Not in _INSERT_SQL / _RECENT_SQL on purpose: the
    -- default fills it, and HistoryEntry stays exactly the wire schema.
    inserted_at    timestamptz NOT NULL DEFAULT now(),

    -- The Now Playing embed that hosted this song, resolvable only as a PAIR:
    -- discord.py has no guild.fetch_message(id), so resolve via
    -- `channel.get_partial_message(message_id)`. The persisted text-channel id
    -- cannot stand in, which is why channel_id exists — set_context reassigns the
    -- home channel on every command, so it records where the last command ran
    -- rather than where any card was posted. Both are read off the same message at
    -- song end, so they are both real or both 0. 0 also covers a failed NP send, a
    -- host deleted mid-song, and rows predating the columns.
    --
    -- Kept out of play_history_dedup, and not foreign keys: the host migrates
    -- across messages during one song and a dedicated NP message is DELETED when
    -- retired, so two deliveries of one play can disagree on them and the message
    -- they point at may no longer exist.
    message_id     bigint      NOT NULL DEFAULT 0,
    channel_id     bigint      NOT NULL DEFAULT 0,

    CONSTRAINT play_history_guild_id_valid    CHECK (guild_id > 0),
    CONSTRAINT play_history_requester_valid   CHECK (requester_id >= 0),
    CONSTRAINT play_history_played_secs_valid CHECK (played_secs >= 0),
    CONSTRAINT play_history_duration_valid    CHECK (duration_secs >= 0),
    CONSTRAINT play_history_message_id_valid  CHECK (message_id >= 0),
    CONSTRAINT play_history_channel_id_valid  CHECK (channel_id >= 0),
    CONSTRAINT play_history_queue_position_valid CHECK (queue_position >= 0),

    -- Both timestamps carry the same domain the wire validator clamps them to
    -- (_EPOCH_FIELDS against _TS_MAX in src/guild_state.py). Without these the
    -- epoch-0 floor that -leaderboard's all-time cutoff relies on is asserted
    -- only in Python, so a pre-epoch played_at would sort ahead of every real
    -- play and no constraint would catch the validator regressing. The bounds
    -- are to_timestamp() of the same epoch seconds the validator uses —
    -- to_timestamp(double precision) is IMMUTABLE, so a CHECK may call it.
    CONSTRAINT play_history_played_at_valid CHECK (
        played_at BETWEEN to_timestamp(0) AND to_timestamp(253402300799)
    ),
    CONSTRAINT play_history_queued_at_valid CHECK (
        queued_at BETWEEN to_timestamp(0) AND to_timestamp(253402300799)
    ),

    -- Unlike title and uploader, this column never holds third-party text: every
    -- value is minted by src/sources.py's normalizer, so anything outside the
    -- domain is a producer defect rather than an unusual song. HistoryEntry
    -- clamps to '' on the same pattern (_SLUG_FIELDS), so this cannot fire.
    CONSTRAINT play_history_query_source_valid CHECK (
        query_source ~ '^[a-z0-9.-]{0,64}$'
    )
);

-- CREATE TABLE IF NOT EXISTS is a no-op against an existing table, so on its own
-- it records this migration as applied over whatever shape it found — a drift the
-- schema-version machinery cannot see: the ledger says migrated, verify_schema()
-- agrees, and every insert raises UndefinedColumnError. These make the CREATE
-- idempotent per column and are no-ops on a correct table, so a new column needs a
-- line here as well as above. Columns only — Postgres has no ADD CONSTRAINT IF
-- NOT EXISTS, and the CHECKs are inline in the CREATE.
ALTER TABLE play_history ADD COLUMN IF NOT EXISTS message_id     bigint      NOT NULL DEFAULT 0;
ALTER TABLE play_history ADD COLUMN IF NOT EXISTS channel_id     bigint      NOT NULL DEFAULT 0;
ALTER TABLE play_history ADD COLUMN IF NOT EXISTS queued_at      timestamptz NOT NULL DEFAULT to_timestamp(0);
ALTER TABLE play_history ADD COLUMN IF NOT EXISTS queue_position integer     NOT NULL DEFAULT 0;
ALTER TABLE play_history ADD COLUMN IF NOT EXISTS query_source   text        NOT NULL DEFAULT '';
ALTER TABLE play_history ADD COLUMN IF NOT EXISTS inserted_at    timestamptz NOT NULL DEFAULT now();

-- Dedup for at-least-once delivery (drainer redelivery) and backfill overlap.
-- Uniqueness only — it does NOT serve the -history read; play_history_recent
-- below does.
--
-- Known edge: two genuinely distinct plays of the same URL in one guild collapse
-- into one row when both carry the epoch-0 "unknown time" sentinel (legacy
-- entries imported by backfill_history, or values the validator clamped).
-- Accepted: a silent merge of indistinguishable pre-archive rows beats a wider
-- key that stops deduping the redeliveries this index exists for.
--
-- Second edge, cutting the other way: every fragment of one interrupted play
-- INHERITS played_at (see the column comment), so the key also masks a genuine
-- double-record. Two writers recording one play land as one row here while the
-- Redis list keeps both — -history shows the duplicate, the archive does not, and
-- nothing rejects or logs it.
CREATE UNIQUE INDEX IF NOT EXISTS play_history_dedup
    ON play_history (guild_id, played_at, webpage_url);

-- The -history read index: `WHERE guild_id = $1 ORDER BY played_at DESC, id DESC
-- LIMIT $2`.
--
-- play_history_dedup only PARTIALLY serves that — played_at is presorted but id
-- is not, so Postgres plans an Incremental Sort and must consume an entire
-- equal-played_at group before it can emit the first row. Those groups are
-- neither rare nor small: a missing played_at parses to 0.0, the validator
-- collapses NaN and out-of-range values to 0.0, and backfill imports legacy
-- entries that predate the field, so a backfilled guild can carry tens of
-- thousands of rows at to_timestamp(0). Measured on postgres:18 with 1M rows, a
-- guild with 50k epoch-0 rows: p50 49.98ms / max 136.5ms, against p50 1.34ms
-- for an ordinary guild. This index matches the ORDER BY exactly, so the plan
-- becomes a plain backward index scan with no sort node.
--
-- Plain CREATE INDEX, not CONCURRENTLY: db_migrate runs each migration inside a
-- transaction and CONCURRENTLY cannot. It takes a brief write lock on
-- play_history; the only writer is the drainer, which retries.
CREATE INDEX IF NOT EXISTS play_history_recent
    ON play_history (guild_id, played_at DESC, id DESC);

-- Expected to stay empty forever: every entry reaching the drainer is insertable
-- by construction, so a row here means the wire-side validator regressed or this
-- build is talking to a schema it was not written for. Both are code defects,
-- and both want the payload preserved verbatim for replay. Treat any row as
-- page-worthy.
--
-- payload is bytea, and both obvious choices are wrong: a NUL byte is the first
-- poison vector (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md §3.2) and jsonb
-- ("unsupported Unicode escape sequence") and text ("null character not
-- permitted") both REFUSE it — so either would fail to store the exact class of
-- row this table exists to capture. bytea takes the raw orjson bytes, which is
-- also what makes replay exact. Inspect with encode(payload, 'escape'), never
-- convert_from(payload, 'UTF8') — that raises on invalid UTF-8.
--
-- One index, and it is not for querying: the table is expected to hold zero
-- rows and the only read is ORDER BY rejected_at DESC LIMIT n. It exists to
-- make the INSERT idempotent.
--
-- Why that is needed: the outbox is a stream consumer group under a stable
-- consumer name, so two drainers can replay the same pending entry, both fail
-- it, and both land here. Under the drainer lease this path was exactly-once
-- for free and nothing noticed the dependency. Duplicates matter here more than
-- ordinary duplicated work do, because the entire diagnostic value of this
-- table is that it is normally empty — `just db-rejects` printing three rows
-- must mean three distinct failures, never one failure seen three times.
--
-- The key is "the same entry, failing the same way, in the same guild".
-- payload IS the entry verbatim, so its digest identifies it exactly; a
-- STORED generated column rather than a bare expression index so the constraint
-- has a NAME for _REJECT_SQL's ON CONFLICT ON CONSTRAINT to reference, instead
-- of the app restating the index expression and drifting from it. md5 is
-- immutable over bytea, which is what a generated column requires — and it is
-- being used as a content digest, not as a security primitive.
--
-- rejected_at is deliberately NOT in the key: the same entry failing again an
-- hour later is the same defect, and a timestamp in the key would defeat the
-- whole clause.
CREATE TABLE IF NOT EXISTS play_history_rejected (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rejected_at  timestamptz NOT NULL DEFAULT now(),
    guild_id     bigint      NOT NULL DEFAULT 0,
    error_type   text        NOT NULL DEFAULT '',
    error_detail text        NOT NULL DEFAULT '',
    trace_id     text        NOT NULL DEFAULT '',
    payload      bytea       NOT NULL,
    payload_md5  text        GENERATED ALWAYS AS (md5(payload)) STORED,

    CONSTRAINT play_history_rejected_dedup
        UNIQUE (guild_id, error_type, payload_md5)
);
