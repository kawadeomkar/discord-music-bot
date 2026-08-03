-- Where a song sat in the queue when it was requested: when it was first
-- enqueued, and how many songs were ahead of it then (counting the one
-- playing). Both carry the wire format's zero-value convention — epoch 0 and
-- position 0 mean "unknown", which is also what an entry written before these
-- fields existed parses as, and what every pre-existing row defaults to.
--
-- Position 0 is therefore ambiguous by construction: it reads as both "played
-- immediately" and "not recorded". That is deliberate — a nullable column would
-- distinguish them at the cost of NULL handling in every aggregate, and no
-- consumer needs the distinction.
--
-- The first migration after 0001 was frozen (see its header). Additive and
-- idempotent, so a retry after a failed transaction re-runs cleanly, and
-- ADD COLUMN with a non-volatile DEFAULT is metadata-only on PG11+ — no table
-- rewrite, no ACCESS EXCLUSIVE held for a scan.
ALTER TABLE play_history
    ADD COLUMN IF NOT EXISTS queued_at      timestamptz NOT NULL DEFAULT to_timestamp(0);

ALTER TABLE play_history
    ADD COLUMN IF NOT EXISTS queue_position integer     NOT NULL DEFAULT 0;

-- NOT VALID, unlike 0001's inline CHECKs: those validated for free on a table
-- created empty, this one would scan every existing row under ACCESS EXCLUSIVE.
-- The constraint is enforced on all new rows regardless — it is the schema lock
-- against a regressed HistoryEntry validator (0001's header), not input
-- validation, so the rows it would scan are ones this app already clamped.
--
-- Wrapped because ADD CONSTRAINT has no IF NOT EXISTS and this file must stay
-- retry-safe; 42710 is duplicate_object.
DO $$
BEGIN
    ALTER TABLE play_history
        ADD CONSTRAINT play_history_queue_position_valid CHECK (queue_position >= 0) NOT VALID;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;
