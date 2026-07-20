-- -stats most-played aggregation (docs/POSTGRES_HISTORY_PLAN.md §7.1):
-- GROUP BY webpage_url within one guild. Ships with the feature that needs
-- it, per §4.3 — indexes land with their queries, not speculatively.
CREATE INDEX IF NOT EXISTS play_history_by_song
    ON play_history (guild_id, webpage_url);
