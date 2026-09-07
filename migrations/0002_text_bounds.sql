-- Byte bounds on the five text columns, mirroring HistoryEntry's
-- _TEXT_BYTE_LIMITS (src/guild_state.py). The validator cuts every value to
-- these budgets before an insert is attempted, so a violation here is a
-- validator regression, and it lands in play_history_rejected like the other
-- CHECKs. See docs/ARCHITECTURE.md#text-column-bounds.
--
-- The webpage_url bound is what keeps play_history_dedup insertable: a btree
-- refuses an index tuple past 2704 bytes with SQLSTATE 54000
-- (ProgramLimitExceededError), which no CHECK can express. 2048 bytes of URL
-- plus the two 8-byte key columns and the tuple header stays under it.
--
-- NOT VALID: deployed tables hold rows that predate these bounds, and a
-- validating ADD CONSTRAINT scans the table under ACCESS EXCLUSIVE. Enforcement
-- is immediate for new rows either way. Each ADD is wrapped so the migration is
-- safe to re-run — Postgres has no ADD CONSTRAINT IF NOT EXISTS.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'play_history_title_len') THEN
        ALTER TABLE play_history ADD CONSTRAINT play_history_title_len
            CHECK (octet_length(title) <= 1024) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'play_history_webpage_url_len') THEN
        ALTER TABLE play_history ADD CONSTRAINT play_history_webpage_url_len
            CHECK (octet_length(webpage_url) <= 2048) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'play_history_requester_name_len') THEN
        ALTER TABLE play_history ADD CONSTRAINT play_history_requester_name_len
            CHECK (octet_length(requester_name) <= 1024) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'play_history_thumbnail_len') THEN
        ALTER TABLE play_history ADD CONSTRAINT play_history_thumbnail_len
            CHECK (octet_length(thumbnail) <= 1024) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'play_history_uploader_len') THEN
        ALTER TABLE play_history ADD CONSTRAINT play_history_uploader_len
            CHECK (octet_length(uploader) <= 1024) NOT VALID;
    END IF;
END
$$;
