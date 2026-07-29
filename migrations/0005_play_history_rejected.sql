-- Expected to stay empty forever. Every entry reaching the drainer is insertable by
-- construction (HistoryEntry.__post_init__ clamps into the play_history column domain),
-- so a row here means exactly one of two things: that validator regressed, or this build
-- is talking to a schema it was not written for. Both are bugs, and both want the
-- payload preserved verbatim for replay. Treat any row as page-worthy — it is a code
-- defect, not a data problem.
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
