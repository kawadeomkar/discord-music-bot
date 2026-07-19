#!/bin/sh
# Nightly play-history backup — the "produce" half of the split backup design
# (docs/POSTGRES_HISTORY_PLAN.md §8.1). Dumps INTO the postgres container's
# /backups volume; a separate external process pulls completed dumps to
# long-term storage and owns retention there.
#
# Install (host crontab; 04:10 local, after the day's listening):
#   10 4 * * * /Users/Omkar/discord-music-bot/pg_backup.sh || logger -t pg_backup "FAILED"
#
# Contract with the puller (do not break):
#   - completed dumps match  /backups/musicbot-YYYYMMDD-HHMMSS.dump  (UTC,
#     sortable, pg_dump custom format) — any file matching the glob is
#     complete, because partials only ever exist as .musicbot.dump.tmp
#     (rename within one volume is atomic)
#   - this script's prune is the ONLY deleter in /backups (keeps newest 7 as
#     a local buffer); the puller must never delete
set -eu

CONTAINER="${CONTAINER:-discord-postgres}"
DB_USER="${DB_USER:-musicbot}"
DB_NAME="${DB_NAME:-musicbot}"
KEEP="${KEEP:-7}"

# Dump: write to a dot-tmp name, fsync-equivalent via pg_dump completing,
# then atomically rename to the pullable name.
docker exec "$CONTAINER" sh -c "
  set -eu
  pg_dump -Fc -U '$DB_USER' '$DB_NAME' -f /backups/.musicbot.dump.tmp
  mv /backups/.musicbot.dump.tmp \"/backups/musicbot-\$(date -u +%Y%m%d-%H%M%S).dump\"
"

# Prune: keep the newest $KEEP completed dumps. Local retention is only a
# buffer — long-term retention lives wherever the puller archives.
docker exec "$CONTAINER" sh -c "
  set -eu
  ls -1t /backups/musicbot-*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | while IFS= read -r f; do
    rm -- \"\$f\"
  done
"

echo "pg_backup: ok ($(docker exec "$CONTAINER" sh -c 'ls -1 /backups/musicbot-*.dump 2>/dev/null | wc -l' | tr -d ' ') dumps buffered)"
