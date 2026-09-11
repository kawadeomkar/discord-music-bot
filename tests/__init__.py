"""Test package.

Exists so `tests/commands/` cannot collide with a same-named module above it —
and, because it is imported before conftest and therefore before conftest imports
`src`, it is the only place the environment can be cleaned early enough.
"""

import os

# Every environment variable src/ reads except HISTORY_ARCHIVE_ENABLED, which
# conftest's autouse fixture pins instead. Deleted, not defaulted: the suite
# asserts the SHIPPED defaults in dozens of places, and `just test` does not load
# .env, so the exposure is whatever the developer happens to have exported.
# Fourteen of these were measured turning a clean tree red — DISCORD_TOKEN and
# REDIS_URL among them. tests/test_config.py asserts the tuple stays level with src/.
SCRUBBED_ENV: tuple[str, ...] = (
    "ANALYTICS_RENDER_DEADLINE_SECS",
    "DEBUG_DEADLINE_SECS",
    "DEBUG_MODE",
    "DEBUG_PROMETHEUS_URL",
    "DEBUG_TICK_SECS",
    "DISCORD_TOKEN",
    "ENVIRONMENT",
    "GIT_SHA",
    "HEARTBEAT_INTERVAL_SECS",
    "HISTORY_OUTBOX_MAX",
    "LIVENESS_FILE",
    "LIVENESS_INTERVAL_SECS",
    "NOW_PLAYING_UPDATE_INTERVAL_SECS",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SDK_DISABLED",
    "OTEL_SERVICE_NAME",
    "PING_DEADLINE_SECS",
    "PING_TICK_SECS",
    "PLAY_INFLIGHT_MAX",
    "PLAY_RESOLVE_CONCURRENCY",
    "POSTGRES_MIGRATE_URL",
    "POSTGRES_STATEMENT_CACHE",
    "POSTGRES_URL",
    "POT_PROVIDER_URL",
    "REDIS_URL",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "STREAM_PROBE_TIMEOUT_SECS",
    "YTDLP_POOL_WORKERS",
)

# NOT scrubbed: the tier switches are read by tests/helpers.py rather than src/,
# and they are the one thing an operator is meant to export into a run.
PRESERVED_ENV: tuple[str, ...] = ("POSTGRES_TEST_URL", "REDIS_TEST_URL")

# Here rather than only in the autouse fixture, for the same reason MPLCONFIGDIR
# sits at conftest's module scope: config.py and debug.py's _CONFIG_ALLOWLIST
# resolve these at IMPORT, which happens while conftest is still being imported —
# long before any fixture runs. The fixture repeats the delete so that a test
# setting one of these through monkeypatch still gets it undone afterwards.
for _name in SCRUBBED_ENV:
    os.environ.pop(_name, None)
del _name
