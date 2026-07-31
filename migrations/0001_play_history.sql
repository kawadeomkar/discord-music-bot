-- The play-history schema: the durable long-term home for every played song,
-- plus the reject table that catches anything the server refuses.
--
-- Deliberately ONE migration. No bot has ever opened a Postgres connection, so
-- there is no database whose schema this has to evolve, and a pre-release
-- sequence of ALTER steps would describe upgrades that never happened to
-- anyone. While that holds, this file is edited IN PLACE.
--
-- The window closes when the first release whose bot actually reads the archive
-- ships (the stack/4 durable-tier work — setup_hook requiring POSTGRES_URL and
-- constructing PostgresHistoryArchive). From then on every change becomes a new
-- numbered migration, because from then on the ALTERs are real. Note this is
-- NOT "the first tagged release": v1.3.0 and v1.4.0 already ship this file, and
-- editing it stayed safe only because their bots never connected to Postgres.
--
-- Consequence: IF NOT EXISTS makes this whole file a no-op against a database
-- that already holds these tables, so a dev box on an earlier shape does NOT
-- get updated — drop the tables (or the volume) and re-run.
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
    played_at      timestamptz NOT NULL DEFAULT to_timestamp(0),

    -- When the row reached Postgres, as opposed to when the song was played.
    -- played_at is a client clock captured at song end and can be arbitrarily
    -- far in the past for backfilled or long-buffered entries, so this is what
    -- outage forensics and backfill auditing (`WHERE inserted_at > <start>`) can
    -- ask about. Not in _INSERT_SQL / _RECENT_SQL on purpose: the default fills
    -- it, and HistoryEntry stays exactly the wire schema.
    inserted_at    timestamptz NOT NULL DEFAULT now(),

    -- The Discord message id of the Now Playing embed that hosted this song.
    --
    -- A CORRELATION TOKEN, not a resolvable pointer. discord.py has no
    -- guild.fetch_message(id) — resolving a message needs channel.fetch_message,
    -- and no channel id is stored here. The persisted text-channel id cannot
    -- stand in either: MusicPlayer.set_context reassigns the home channel on
    -- every command, so the NP host migrates across text channels within one
    -- guild and that id only records wherever the last command ran. Match this
    -- against a log line or a span (song.np_host_id), not against the API. If a
    -- resolvable pointer is ever wanted, it needs a companion channel_id column
    -- stamped from the same host — and that should land WITH the archive wiring
    -- below, not before it, so the wire format does not grow a second field
    -- nothing reads.
    --
    -- NOT POPULATED YET. HistoryEntry carries the value on the Redis wire —
    -- HistoryEntry.from_song stamps it from the host the playback loop captured
    -- at song end — but the archive's INSERT does not list this column, so the
    -- default fills every row and the stamped value is dropped when the outbox
    -- entry is retired. Until _INSERT_SQL / _RECENT_SQL / _entry_to_row /
    -- _row_to_entry in history_archive.py carry it, a 0 here means "the archive
    -- does not write this column", NOT "this song had no host" — do not read a
    -- table of zeroes as a fleet-wide Now Playing failure.
    --
    -- Once that lands, 0 becomes a real value rather than only a default, but
    -- an ambiguous one: it also covers a song whose NP send failed, a host a
    -- listener deleted mid-song (released on discord.NotFound), an entry from a
    -- build older than the field, and anything the backfill inserts.
    --
    -- Kept out of play_history_dedup, and not a foreign key, for the same
    -- reason: the NP host is not stable or permanent. It migrates across
    -- messages during one song (MusicContext.send re-hosts the block on every
    -- command response in the home channel), and a dedicated NP message is
    -- DELETED when retired. So two deliveries of one play can disagree on it —
    -- in the key that would land them as two rows — and the id it holds may
    -- point at a message that no longer exists.
    message_id     bigint      NOT NULL DEFAULT 0,

    CONSTRAINT play_history_guild_id_valid    CHECK (guild_id > 0),
    CONSTRAINT play_history_requester_valid   CHECK (requester_id >= 0),
    CONSTRAINT play_history_played_secs_valid CHECK (played_secs >= 0),
    CONSTRAINT play_history_duration_valid    CHECK (duration_secs >= 0),
    CONSTRAINT play_history_message_id_valid  CHECK (message_id >= 0)
);

-- Dedup for at-least-once delivery (drainer redelivery) and backfill overlap.
-- Uniqueness only — it does NOT serve the -history read; play_history_recent
-- below does.
--
-- Known edge: two genuinely distinct plays of the same URL in one guild collapse
-- into one row when both carry the epoch-0 "unknown time" sentinel (legacy
-- entries imported by backfill_history, or values the validator clamped).
-- Accepted: a silent merge of indistinguishable pre-archive rows beats a wider
-- key that stops deduping the redeliveries this index exists for.
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
-- No index: the table is expected to hold zero rows and the only query is
-- ORDER BY rejected_at DESC LIMIT n.
CREATE TABLE IF NOT EXISTS play_history_rejected (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rejected_at  timestamptz NOT NULL DEFAULT now(),
    guild_id     bigint      NOT NULL DEFAULT 0,
    error_type   text        NOT NULL DEFAULT '',
    error_detail text        NOT NULL DEFAULT '',
    trace_id     text        NOT NULL DEFAULT '',
    payload      bytea       NOT NULL
);
