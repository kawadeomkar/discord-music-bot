-- -history --user per-requester reads (docs/POSTGRES_HISTORY_PLAN.md §7.2):
-- equality on (guild_id, requester_id), newest-first scan on played_at.
CREATE INDEX IF NOT EXISTS play_history_by_requester
    ON play_history (guild_id, requester_id, played_at DESC);
