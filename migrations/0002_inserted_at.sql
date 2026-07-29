-- When the row reached Postgres, as opposed to when the song was played.
--
-- played_at is a client clock captured at song end and can be arbitrarily far
-- in the past for backfilled or long-buffered entries; nothing recorded how
-- far behind the archive actually was. This gives outage forensics
-- ("everything from the 14th landed on the 16th") and backfill auditing
-- (`WHERE inserted_at > <backfill start>`) a column to ask about.
--
-- Not in _INSERT_SQL / _RECENT_SQL on purpose: the default fills it, and
-- HistoryEntry stays exactly the wire schema.
ALTER TABLE play_history
    ADD COLUMN IF NOT EXISTS inserted_at timestamptz NOT NULL DEFAULT now();
