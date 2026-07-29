-- The schema lock. HistoryEntry.__post_init__ already guarantees every one of these
-- (docs/HISTORY_SCHEMA_FIRST_FINDINGS.md §5.1), so a violation is never bad user
-- data — it is a regression in that validator, or a build talking to a schema it was
-- not written for. Being loud is the entire point.
--
-- NOT VALID is load-bearing twice over. It keeps this off the exclusive-lock path on
-- a table that may already hold millions of rows, and it means EXISTING rows are not
-- checked — legacy backfilled rows can carry values written before the validator
-- existed, and a validating constraint would fail the migration on real data. A
-- follow-up migration can VALIDATE once production is known clean; that takes only
-- SHARE UPDATE EXCLUSIVE and does not block the drainer.
--
-- webpage_url is deliberately NOT constrained here. __post_init__ stores '' rather
-- than rejecting it, because _row_to_entry runs the same validator when READING rows
-- that already contain it (migrations/0001 documents that merge as accepted). See
-- findings §5.3 and open decision §11.1.
--
-- ADD CONSTRAINT has no IF NOT EXISTS in Postgres 18, and each migration runs in its
-- own transaction under pg_advisory_xact_lock, so a retry after a partial failure
-- must not trip over a constraint that already landed. The DO block makes this
-- idempotent-safe on retry, which src/db_migrate.py requires.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'play_history_guild_id_valid') THEN
        ALTER TABLE play_history
            ADD CONSTRAINT play_history_guild_id_valid
            CHECK (guild_id > 0) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'play_history_requester_valid') THEN
        ALTER TABLE play_history
            ADD CONSTRAINT play_history_requester_valid
            CHECK (requester_id >= 0) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'play_history_played_secs_valid') THEN
        ALTER TABLE play_history
            ADD CONSTRAINT play_history_played_secs_valid
            CHECK (played_secs >= 0) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'play_history_duration_valid') THEN
        ALTER TABLE play_history
            ADD CONSTRAINT play_history_duration_valid
            CHECK (duration_secs >= 0) NOT VALID;
    END IF;
END $$;
