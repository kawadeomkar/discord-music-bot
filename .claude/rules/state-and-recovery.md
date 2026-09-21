---
paths:
  - "src/guild_state.py"
  - "src/redis_client.py"
  - "src/recovery.py"
  - "src/guild_history.py"
  - "src/history_archive.py"
  - "src/backfill_history.py"
  - "src/db_migrate.py"
  - "src/musicplayer.py"
  - "src/leaderboard.py"
  - "src/analytics_card.py"
  - "src/analytics_render.py"
  - "src/chart_pool.py"
  - "src/commands/{_common,analytics,history,leaderboard,play,resume}.py"
  - "migrations/**"
  - "tests/test_{guild_state,redis_client,recovery,guild_history,history_archive,backfill_history,db_migrate,leaderboard,analytics_card,analytics_render,chart_pool}.py"
  - "tests/test_{pg,redis}_integration.py"
---

# State: the Redis schema, the history backfill and crash recovery

The path-scoped half of CLAUDE.md: its golden rules apply here, and cite by number.

## Architecture

### Redis schema and persistence model

All schema lives in `src/guild_state.py` (frozen `slots` dataclasses + field-name
constant classes; wire tables are spelled out explicitly so renaming a Python attribute
can never silently rename a Redis field). `GuildRedisStore` (redis_client.py) is the only
IO surface. The pool is created with `decode_responses=False` — readers `cast()` HGETALL
to `dict[bytes, bytes]` and decode in `from_redis()`; do not "simplify" this.

| Key | Type | TTL | Contents |
|---|---|---|---|
| `guild:{id}:state` | hash | 24h | voice/text channel IDs, `current_song_*` (a parked queue entry), `last_position_secs` + `last_heartbeat_epoch` (the recorded playback position — what recovery reads), and the legacy `play_start_epoch`, `total_pause_seconds`, `pause_start_epoch` it replaced, **dual-written for one release** so a rollback still recovers. Still *parses* a legacy `volume` field — see `:config` |
| `guild:{id}:queue` | list | 24h, re-armed by every queue write and every song start | JSON entries, `type` discriminator: `"qobj"` (SongQueueEntry) / `"ytsource"` (SearchQueueEntry — e.g. unresolved Spotify-playlist tracks). Both carry `user_input`, what the user typed, and the requester (`requester_id`, on a search only when known); a search also carries, each only when known, the display fields a listing shows until it resolves (`title`, `uploader`, `duration`, `webpage_url`); on a search entry it is the ONLY surviving record of the collection link, since its `ytsearch` is a generated title. Mirror writes all go through `GuildQueue._write_mirror` — rebuild, DELETE, or LREM |
| `guild:{id}:now_playing` | hash | 24h | display snapshot for `-now` / recovered embed (deleted wholesale on song end: empty == no song) |
| `guild:{id}:history` | list | **none, ever (PERSISTed)** | HistoryEntry JSON, most recently RECORDED first (~625 B/entry), LTRIMmed to `HISTORY_CACHE_LIMIT` (50) on every write. The ONLY source `-history` reads — bounded by length so it can be retained forever. Postgres is the durable record behind it |
| `history:outbox` | **stream** | **none, ever** | global write-ahead buffer, written only while the archive is enabled (disabled — the default — the key is never created): every play, all guilds interleaved, one `serialize_history_entry` blob per entry under field `e`, drained oldest-first into Postgres by the `drainers` consumer group. Non-evictable — an evicted entry is a silently lost play |
| `guild:{id}:config` | hash | **none, ever (PERSISTed)** | durable per-guild preferences (`GuildConfig`). Eight fields today: `debug_mode` (`"1"`/`"0"`), `volume`, `timezone` (an IANA name, resolved by `GuildConfig.tzinfo()` at read time so a name the host's tz database cannot resolve degrades to the default instead of raising on a render path), `idle_timeout_secs`, `alone_timeout_secs`, `np_refresh_secs`, `slow_notice_secs` (whose `0.0`, `OFF_SECS`, is a set value meaning off: compared, never tested for truthiness) and `queue_progress_delay_secs`. **Absent always means "no choice made"** — for debug that is "follow the host `DEBUG_MODE`", for volume it is "use the default", and keeping it distinct from an explicit `0`/`false` is why every field is Optional. `volume` MOVED here from `:state`, and the legacy field is **dual-written for one release rather than deleted** — deleting it made `just up <older-sha>` silently reset every migrated guild to 100%, since the older build reads only `:state`. Restore reads config-then-legacy and SEEDS config from what it finds, through `GuildSettings.write(mode=SEED)` (`migrate_volume`, `HSETNX` — never an overwrite, or a snapshot read before a concurrent `-volume` would durably clobber it), and the seed is refused when a volume write or reset committed after the snapshot read began. Every writer of this hash goes through `GuildSettings` (`docs/ARCHITECTURE.md#settings-resolution`). Drop the legacy write, `StateField.VOLUME` and `GuildStateData.volume` together after one release. Deliberately not fields on `:state`, which expires in 24h — a durable choice must not evaporate on an idle guild. A numeric value outside `CONFIG_DOMAIN` reads as unset. `writer_app_id` is a wire field, not a setting: the application id of the bot that last wrote a setting there (`CONFIG_WRITER_FIELD`), stamped by every `GuildSettings` write and never by a reset. Excluded from every TTL path; deleted on `on_guild_remove`, and once per process start by `GuildSettings.sweep_orphans` for a guild no longer in `bot.guilds` whose key carries this application's stamp — an unstamped key, or another application's sharing the Redis, is never deleted |
| `bot:{application_id}:config` | hash | **none, ever (PERSISTed)** | the operator's bot-wide overrides: one field per `-settings bot` knob (`Knob.field`), named as its env var in lower case, absent meaning "run on the environment value". Written only by `BotSettings.write`/`write_reset` (`-settings bot <setting> <value>`/`reset`), applied at startup by `BotSettings.hydrate`, never read while `BOT_SETTINGS_OVERRIDES=ignore`. Keyed by application so a dev and a prod bot sharing one Redis cannot retune each other. `debug-default` is never stored. `just bot-settings` lists and deletes it without chat |
| `ytdl:source:{query}` | string | 24h | search → {webpage_url, title, duration, uploader, thumbnail, cached_at}. The query is case-folded, EXCEPT for a URL — YouTube video ids are case-sensitive and two would share an entry. Past `_YT_SOURCE_FRESH_SECS` (1h) a hit is served as-is and a SEARCH is refreshed behind the reply (`_revalidate_source`, one flat POST); a link is not, since what ages is the ranking a search resolved through. An entry with no `cached_at` is from the build before the stamp and reads as fresh — it carries that build's 1h TTL |
| `ytdl:stream:{webpage_url}` | string | ≤30m (expire-capped) | probed-playable stream URL + `_STREAM_CACHE_FIELDS` metadata, plus a `traceparent` naming the extraction that minted the URL — the only record of where a serving URL came from, and what the playback span links back to — and `probed_at` when the verdict was PLAYABLE: inside `_PROBE_REUSE_SECS` (10s) the playback loop reuses that verdict instead of re-probing a URL the resolve just confirmed. An UNCONFIRMED entry is never stamped |
| `ytdl:playlist:v3:{list id}` | string | 15m | a YouTube playlist's kept entries, in order, as the five identity fields a queue entry needs, beside the playlist's `title` and its `unavailable` count. Keyed on the `list=` id, so the playlist page, a watch link carrying `&index=`, and a `&t=` copy share one entry. Entries yt-dlp could not describe — null, id-less, or id-only (a private or deleted video: no title and no duration) — are dropped and counted BEFORE the write, so a hit cannot resurrect them; an empty result is not written at all. TTL'd, so eviction-safe |
| `leaderboard:v{n}:{guild_id}:{days}:{top_n}` | string | 60s | orjson aggregate cache for `-leaderboard`, one entry per requested window (`:0` = all-time). Keyed by row limit and codec version too, so neither can decode stale. TTL'd, so eviction-safe |
| `spotify:auth:token` | string | expires_in − 30s | raw bearer token (NOT orjson — deliberate) |
| `spotify:{track,artist,album}:{id}`, `spotify:playlist:v4:{id}`, `spotify:album_tracks:v2:{id}` | string | 24h/24h/24h, 1h, 24h | cached lookups. A playlist entry is a dict of the kept titles, one `[name, artists, duration_secs, url]` display row per title beside them under `tracks`, and the playlist's name, total length and unavailable-item count; only a complete walk writes one. An album entry is the same dict plus its artists and cover. The key is versioned by that VALUE's shape: under the 1h TTL an unversioned key would answer a deploy that changes it with the previous build's entries |
| `lock:guild:{id}:recovery` | string | 60s | SET NX EX recovery lock, one restore per guild at a time. The value is a per-acquisition random token, and release is a WATCH/MULTI compare-and-delete — an unconditional DEL would let a holder whose lock expired mid-recovery delete its successor's |

Postgres holds two tables — `play_history`, and `play_history_rejected` (rows the server
refused; expected to stay empty forever, since `HistoryEntry.__post_init__` clamps every
entry into the column domain, so a row there means that validator regressed or the build
is talking to a schema it was not written for — inspect with `just db-rejects`) — plus
the `schema_migrations` ledger. The app
NEVER applies DDL — it verifies `max(version) >= EXPECTED_SCHEMA_VERSION` and raises
`SchemaVersionError` naming `just db-migrate` otherwise. A database NEWER than the build
warns and proceeds (migrations are additive, so a rollback must not be an outage).

Wire-format compatibility: parsers default missing fields and drop corrupt entries with
a warning (the rest of the list survives). When adding a queue/history field, add it to
the Field-constant class, the dataclass (with a default), `to_redis`, and the parse path
with `.get(..., default)` so pre-migration entries stay readable.

Contract subtleties the store encodes: readers distinguish "empty" (zero-value snapshot)
from "Redis unavailable" (None); TTL refresh EXPIREs ride the same pipeline AFTER writes
(EXPIRE on a missing key is a no-op); history is excluded from every SHARED TTL path, because `push_history` owns that
key's retention alone — LTRIM + PERSIST, unconditionally (the PERSIST also self-heals
keys carrying an older build's 24h expiry); `on_resume` is a documented
non-atomic read-modify-write (single writer per guild today — must become Lua/WATCH under
multi-process sharding).

### History backfill (one-shot, and it has a deadline)

`src/backfill_history.py` (`just db-backfill`, or `just db-backfill-docker` on a host
with no venv; a raw `docker compose run --rm db-backfill` fails unless the `archive`
profile is active) walks every `guild:{id}:history` list and inserts it straight
into `play_history`, bypassing the outbox — routing a historical backlog through the
outbox would bury the live drain behind it. It exists because the archive only captures
songs played *after* it was deployed.

**It must run before this build is deployed, and the window closes silently.**
`push_history` LTRIMs each guild's list to `HISTORY_CACHE_LIMIT` on every write, so a
guild's first song under this build destroys the oldest entries — exactly what the
backfill exists to move. The clock is per guild (it starts at that guild's next song
end), there is no flag to check, and nothing can be undone. Note the trim happens in
**both** archive modes, so an operator who never opts in is on the same deadline.

Nothing in the script can detect that it already ran: a list sitting at exactly
`HISTORY_CACHE_LIMIT` is also what a healthy migrated guild looks like. The printed
counts are the only signal, and only when the run precedes the deploy. Re-running is
safe regardless — `ON CONFLICT DO NOTHING` makes every insert idempotent, and order
within a guild is irrelevant because reads sort on `played_at`. Entries written before
`HistoryEntry` carried a `guild_id` parse as `guild_id=0`, which would collide every
guild's legacy rows on the `(guild_id, played_at, webpage_url)` dedup index; the script
stamps the real id from the Redis key, the only place that information still exists.
`--dry-run` counts what would move and touches nothing.

### Crash recovery

```
on_ready (cold start / session loss; NOT WebSocket resume; skipped when redis is None)
  └─ _recover_after_ready: ONE GuildSettings.hydrate pass (bounded per batch), then
  └─ per guild: _restore_guild (background task)
       ├─ skip if guild already in mps
       ├─ acquire lock:guild:{id}:recovery (SET NX EX 60, random token) — one restore per guild
       ├─ get_recovery_gate(): ONE pipeline = state hash + queue LLEN (contents stay
       │    off the wire on the common nothing-to-do path; a -stopped guild keeps a
       │    possibly-long persisted queue by design)
       ├─ gate is None (read failed) → skip, lock expires, next on_ready retries
       ├─ no persisted channel pair → return; channels deleted → clear_connection(),
       │    best-effort user notification in a reachable channel
       ├─ nothing restorable (empty queue, no crashed song) → return
       ├─ voice_channel.connect(timeout=30) + self-deafen
       └─ MusicPlayer(...).start() → mps[guild.id]
            start(): voice client exists → open_playback_gate(); spawn _restore_state()
            _restore_state():
              • get_playback_snapshot(): ONE pipeline = state + full queue + now_playing
                + newest-50 history (all-or-nothing on failure)
              • GuildSettings.seed(snapshot.config): volume and timezone are restored,
                and a legacy volume migrated, only for the fields it accepted — a
                settings write that committed after the read began keeps its value
              • crashed song: crashed_position_at() returns the RECORDED
                last_position_secs (no clock read), capped at the snapshot's own
                current_song_duration − 10s (EOF guard, no IO), rebuilt via
                SongQueueEntry.from_crashed_state (persisted=False, interjected flag
                preserved) → queue.restore_crashed at the FRONT; state cleared
                unconditionally so a failed re-queue can't loop every restart
              • restore_entries() for the pending queue, history.restore(), refresh_ttl()
              • finally: _restore_complete.set() — loop() blocks on this before its
                first dequeue (otherwise it could LPOP Redis for the crashed head,
                silently deleting an unrelated still-queued song)
  └─ once per process, after every _restore_guild above returned, and only when this
     process runs every shard: GuildSettings.sweep_orphans — forget the config of
     each guild not in bot.guilds whose writer_app_id is this application's
     (≤500 per start, stops at the first unconfirmed DELETE)
```

The closed loop that makes this work: `SongQueueEntry.from_song → HSET state (start
transaction) → crash → from_crashed_state → re-queue`. The `current_song_*` state fields
ARE a parked queue entry; `_now_playing_state_mapping` is the single signature enforcing
that identity.

**Clearing that state hands the only copy to memory**, which is why
`MusicPlayer.repark_crashed_head()` exists. `_restore_state` HDELs `current_song_*` the
moment it re-queues the song (unconditionally — a re-queue that failed must not re-enter
that block every restart), so from then until the song plays, the player's queue is the
only place it exists. Any teardown before that loses it silently: no error, no log line,
and nothing left for a later restore to find. `repark_crashed_head` writes a
`persisted=False` display head back into the hash, backdating `play_start_epoch` by its
resume offset because the hash carries no `ts`. It must run **after** `cleanup()`, whose
`clear_connection()` HDELs exactly those fields. Its one caller is
`MusicBot._abandon_cold_start`, the shared teardown for a cold-start command (`-play`,
`-resume`) whose join never produced a connected voice client.

That teardown is not optional in the other direction either: `defer_playback` opens the
gate as it unwinds whether or not the join worked, and a `loop()` released with no voice
client fails its `vc` assertion once per restored song — draining the in-memory legs
while Redis keeps every entry (the LPOP lives past the assertion), so the queue
resurrects on the next restore and does it again. Tearing the player down first makes
that gate-open land on a cancelled loop, which is inert. `_abandon_cold_start` no-ops
while `mp.playback_holds > 1`: the other holder is mid-join on the same player and owns
the decision. Symmetrically, the 300s `_PLAYBACK_GATE_TIMEOUT` re-waits instead of
tearing down while any hold is outstanding — every hold is released by an `async with`,
raise or not, so it cannot park forever.

**Five** call sites wait on the restore before touching the queue, bounded by
`RESTORE_WAIT_SECS` (musicplayer.py): `-play` (one site, covering warm and cold),
`-resume`, `-shuffle`, `-clear` and `-remove`. The pool sets `socket_connect_timeout` but no `socket_timeout`,
so a Redis that accepts the connection and then stalls would hang the command outright.
The two cold-start paths must not **insert** against an unread snapshot — `-play`
front-inserting there double-queues the song — and the other three must not **rebuild**
the mirror from a deque the restore has not filled, which deletes the saved queue
outright. All five abandon and say so. `MusicPlayer.restore_read_failed` separates
"nothing was saved" from "the store could not be read"; only the first may be reported
to a guild as an empty queue.

Recovery reads a position the loop **recorded**, never one it infers from a clock:
`_heartbeat_updater` writes `last_position_secs` every `HEARTBEAT_INTERVAL_SECS` while a
song plays, the start transaction seeds it from the `-ss` offset, and `MusicPlayer.pause`
writes one final exact value (the ticker skips paused songs). The task is created inside
that transaction's `store is not None` block — the store is its only writer, so a
Redis-less guild would otherwise tick for the whole song to reach a no-op. Downtime is
therefore never credited and clock skew between restarts stops mattering; the worst case
is replaying one interval, which is the deliberate bias — replaying 3s is imperceptible,
skipping 3s is not. The legacy wall-clock fields are still written so a rollback still
recovers, and `crashed_position_at` falls back to them for a hash written by the previous
build — and for a recorded position whose `last_heartbeat_epoch` PREDATES the current
song's start, which is what an older image leaves behind when it cannot clear fields it
does not know about. Drop the legacy write, `_legacy_wall_clock_position_at`,
`StateField.PLAY_START_EPOCH`, `TOTAL_PAUSE_SECONDS`, `PAUSE_START_EPOCH` and
`on_pause`/`on_resume` together one release after this ships.

## Concurrency primitives

| Primitive | Protects |
|---|---|
| `_restore_complete` | loop dequeuing before restore has injected the crashed head |
| `PostgresHistoryArchive._analytics_slot` | one -analytics aggregate in flight per process. Deliberately NOT `_read_slots`: that budget is two against a max_size=4 pool, sized when leaderboard was its only taker, and this is the heaviest of the three readers |
| `chart_pool` (1 worker) | matplotlib off the event loop AND off the GIL. A thread is no better than no thread — figure construction is pure Python and contends with discord.py's audio player thread; measured loop lag spikes to 108ms threaded, against 4.23ms frame lateness in a process. Warmed at `setup_hook` **only when the archive is enabled**, after the yt-dlp prewarm that brings the forkserver up — the cold first render is 688ms warmed against 2,976ms not |
| `lock:guild:{id}:recovery` (Redis) | two instances recovering the same guild |
| `history:outbox` consumer group (Redis) | replaced the `history:drainer` lease. Not mutual exclusion — `XREADGROUP >` gives two drainers **disjoint** entries and `XACK` settles by ID, so a second drainer duplicates work instead of destroying plays it never inserted |
| `PostgresHistoryArchive._init_lock` | pool creation racing `close()` |
| `HistoryOutboxDrainer._stop_lock` | concurrent `stop()`s each running their own final drain |

## Recipes

**Add a persisted per-guild state field**: constant in `StateField` → field with default
on `GuildStateData` + `from_redis` → the write-path method on `GuildRedisStore` (or
`_now_playing_state_mapping` if it's per-song) → decide whether it belongs in
`_TRANSIENT_SONG_FIELDS` / `clear_connection` → tests in test_guild_state.py and
test_redis_client.py.

**Add a schema migration**: **while no deployment holds the schema, don't** — edit
`migrations/0001_play_history.sql` in place (its header explains why: nothing is deployed,
so an ALTER sequence would describe upgrades that never happened), then drop and re-create
the scratch **database** — not just the tables, since the `schema_migrations` row survives
them and the re-run applies nothing. The trigger for freezing `0001` is a deployed
database, not a tagged release. Once one exists, editing a migration fails silently and in
the worst direction: `migrate()` skips a version already in the ledger without reading the
file, so the change reaches fresh databases only, the deployed one keeps the old shape and
still passes the version check, and every insert then raises `UndefinedColumnError` —
which is not in `_POISON`, so the drainer treats it as transient and redelivers onto the
non-evictable outbox forever. From that point on: new
`migrations/NNNN_short_name.sql` (numeric prefix, next
free number — `discover()` rejects duplicates and orders numerically, so `0010` follows
`0009`) → bump `EXPECTED_SCHEMA_VERSION` in `src/db_migrate.py` (a test asserts the two
agree) → `just db-migrate` locally → `just test-pg`. Each migration runs in its own
transaction under `pg_advisory_xact_lock`, so it must be idempotent-safe on retry
(`IF NOT EXISTS`). `CREATE INDEX CONCURRENTLY` cannot be used — it is illegal inside a
transaction. After adding one, `DOCKER=1` recipes need no rebuild (`migrations/` is
bind-mounted) but the runtime image does.

**Add a history-entry field**: `HistoryEntry` in guild_state.py (with a default) →
**add it to exactly one domain tuple in guild_state.py — `_TEXT_FIELDS`, `_INT4_FIELDS`
(`integer` columns), `_INT8_FIELDS` (`bigint`), `_EPOCH_FIELDS` (`timestamptz`) or
`_SLUG_FIELDS` (a machine-minted token, clamped to `^[a-z0-9.-]{0,64}$`) — or
`__post_init__` silently does not clamp it and the schema lock has a hole** (a test asserts every field is covered, so
forgetting fails the suite rather than shipping) → check where the added bytes land
against the outbox's allocator-bin cliff (`docs/ARCHITECTURE.md#why-query_source-is-stored-rather-than-derived`:
18 bytes once cost 11% of the OOM runway and the next 32 cost another 14%, and the
curve is NOT monotonic — an unstamped entry measured worse than a larger stamped
one, so measure every shape a field takes and never just the populated one) → `to_redis`/`parse_history_entry`
(`.get(..., default)`, so pre-migration wire entries still parse) → the column in
`migrations/0001_play_history.sql`, plus a named `CHECK` for its domain — inline in the
table definition pre-release (free to validate on an empty table); a separate `NOT VALID`
`ADD CONSTRAINT` once the table holds real rows, so the migration neither scans it nor
takes ACCESS EXCLUSIVE → `_INSERT_SQL`/`_RECENT_SQL`/`_entry_to_row`/`_row_to_entry` in
history_archive.py.

**Touch the history outbox**: it is a Redis **stream** with the `drainers` consumer
group, and four rules are load-bearing rather than stylistic. (1) Settle by ID —
`retire_outbox`'s transactional `XACK`-then-`XDEL`, in that order; the reverse leaves a
tombstone, which is unrecoverable. (2) Never `XTRIM MAXLEN`: it means "keep the newest
n", so a re-send after concurrent `XADD`s destroys a second tranche. `MINID` names an
absolute ID and is inert on re-send. (3) Always pass `approximate=False` — redis-py's
default trims nothing on a small real stream while fakeredis models it as exact, so a
green unit test proves nothing here. (4) `XTRIM` is blind to the PEL, so anything that
destroys entries must `XACK` them first or they replay forever. Read the outbox section
of `redis_client.py` and `HistoryOutboxDrainer._enforce_cap` before changing any of it,
and run `just test-redis` — the unit tier cannot see three of these.

## Golden rules 5 and 12, in full

CLAUDE.md carries the rule; these are the enumerations behind them.

### 5 — which Redis helpers raise, and which swallow

5. **Redis IO never raises out of `GuildRedisStore` or `BotConfigStore`.** Their methods are
   wrapped by `@_guild_op` and `@_bot_op` (log a warning prefixed `[guild:{id}]` or
   `[bot:{application_id}]`, return the default). Everything must degrade gracefully
   when Redis is down or `store is None` — the in-memory bot keeps working. Keep new
   store methods on this pattern; never pass a **mutable** `default=` to either
   decorator (use `default_factory`; `TestStoreOpDefaults` enforces this for both).
   **The scope is the class, not the module.** The outbox-stream helpers in the same
   file (`ensure_outbox_group`, `read_outbox_pending`, `read_outbox_new`,
   `retire_outbox`, `ack_outbox`, `outbox_depth`, `outbox_pending_count`,
   `outbox_pending_below`, `trim_outbox_below`, `reclaim_outbox_stale`)
   deliberately DO raise: the drainer's backoff loop is their error handler, and a
   swallowed error there would look like an empty outbox and silently stall the drain.
   Do not "fix" them onto the `@_guild_op` pattern. The split is asserted, not assumed
   (`TestOutboxDrainHelpers::test_helpers_raise_on_redis_error`).
   `push_history`'s `XADD` leg is on the other side and must stay there — the playback
   loop cannot die because Redis blinked. The consequence is that the producer can never
   report a mis-shaped outbox, which is why a `WRONGTYPE` at `history:outbox` aborts
   **startup** in `setup_hook` instead: that is the only place the signal can be loud.
   (Enabled mode. With the archive disabled the XADD leg is gated off, `setup_hook`
   never creates the group, and a mis-shaped key is inert — downgraded to a startup
   warning by the leftover-outbox probe.)

### 12 — the four non-evictable keys

12. **Redis eviction policy is `volatile-lru` on purpose.** Four keys carry no TTL
    and none may become an eviction candidate. `history:outbox` holds plays that
    are not durable in Postgres yet (written only while the archive is enabled — when
    it is off the key is never created, but the policy must still protect a leftover
    from an earlier enabled run); evicting one loses that play with no error, no
    `play_history_rejected` row and no log line. `guild:{id}:history` is PERSISTed
    and capped at `HISTORY_CACHE_LIMIT` — it is the ONLY source `-history` reads, in
    both archive modes, so evicting or expiring it answers a guild with silence.
    `guild:{id}:config` holds a guild's DURABLE choices (debug mode, volume, timezone, idle and alone timeouts, progress-bar refresh, lookup notice, playlist card delay) — evicting it silently reverts a setting the guild chose, with no log
    line and no error, which is exactly the failure the in-memory version had. It is
    a fixed handful of fields per guild, written only by an explicit command and
    deleted on guild removal — or, for a guild removed while the bot was offline, by
    the startup orphan sweep, which the `writer_app_id` stamp keeps away from the
    guilds only another bot sharing the Redis is in (a guild both serve shares this key,
    as it shares every `guild:{id}:*` key) — so it scales with guild count and not with
    runtime.
    `bot:{application_id}:config` holds the operator's bot-wide overrides — evicting
    it silently returns every overridden knob to its environment value at the next
    start. One hash per application, bounded by the number of bot settings, and
    written only by `-settings bot`.
    Never switch the compose Redis to `allkeys-lru`, and never put a TTL on the
    history or config keys: history is bounded by LENGTH, and config is bounded by
    the number of settings that exist.

## The two-tier boundary

**discord-music-bot** (v2.39.0, GPL-3.0) is a self-hosted Discord music bot that streams
audio from YouTube, Spotify, SoundCloud, and any other yt-dlp-supported site into voice
channels. It is a **single-process Python asyncio application** built on discord.py
(`AutoShardedBot`), yt-dlp, and FFmpeg, with a **two-tier data layer**: Redis for all
runtime state (queue, caching, playback position, crash recovery) and — **opt-in,
default OFF** — **Postgres for durable play history**, fed asynchronously through a
Redis outbox so the playback loop never awaits the database. The archive is a consent
gate, not an infrastructure default: `HISTORY_ARCHIVE_ENABLED=true` turns the app side
on, the `archive` compose profile deploys the database, and a default deployment
collects nothing long-term (`docs/ARCHITECTURE.md#history-archive-tier`). Playback
survives bot restarts: on startup the bot rejoins voice and resumes the interrupted
song from the position it left off.

The boundary is a rule, not a preference: durable records go to Postgres (when the
operator opted in), runtime and cache state stays in Redis forever. Reads follow the
same rule in BOTH modes — `-history` is served from the capped Redis list alone (50
entries per guild, exactly the command's ceiling, written ahead of the archive), and
Postgres backs the commands that need the permanent record (`-leaderboard`).
