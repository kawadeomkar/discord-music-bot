---
paths:
  - "src/config.py"
  - "src/settings.py"
  - "src/settings_card.py"
  - "src/commands/settings.py"
  - ".env.example"
  - "docker-compose.yml"
  - "Dockerfile"
  - "justfile"
  - "build_common.sh"
  - "setup_env.sh"
  - "tests/test_{config,settings,settings_card}.py"
  - "tests/commands/test_settings.py"
---

# Configuration: every environment variable and the settings primitives

The path-scoped half of CLAUDE.md: its golden rules apply here, and cite by number.

## Configuration reference (all env vars; `.env` for compose)

The operator can override the following at runtime with `-settings bot <setting> <value>`
(`NOW_PLAYING_UPDATE_INTERVAL_SECS`, `HEARTBEAT_INTERVAL_SECS`, `PLAY_SLOW_NOTICE_SECS`,
`PLAY_INFLIGHT_MAX`, `PLAY_RESOLVE_CONCURRENCY`, `PLAY_RESOLVE_WAIT_SECS`,
`STREAM_PROBE_TIMEOUT_SECS`, `PING_TICK_SECS`/`PING_DEADLINE_SECS`,
`DEBUG_TICK_SECS`/`DEBUG_DEADLINE_SECS`, `ANALYTICS_RENDER_DEADLINE_SECS`, and
`QUEUE_PROGRESS_DELAY_SECS`/`QUEUE_PROGRESS_TICK_SECS`/`QUEUE_PROGRESS_MAX_SECS`), within a chat
range narrower than the environment's: its minimum sits above the variable's floor for a
time, and at or above it for a count. A stored override wins over the variable
until it is reset, startup logs a WARNING for each one that shadows a set variable, and
`-debug` and `-settings bot` label every value `default`, `env` or `bot owner; env 3s`. A
variable set outside the chat range is honoured and labelled `env, outside chat range`; chat
can only move it back inside.

| Variable | Default | Notes |
|---|---|---|
| `DISCORD_TOKEN` | — | required; startup fails without it |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | — | both or neither; validated live at startup |
| `REDIS_URL` | `redis://localhost:6379` | bot runs degraded (no persistence/recovery) without Redis |
| `HISTORY_ARCHIVE_ENABLED` | `false` | **the consent gate for long-term storage** — `true` enables the Postgres archive tier (outbox writes, drainer, `POSTGRES_URL` requirement). Strict parse (`true/1/yes` / `false/0/no`, case-insensitive; unset/empty → false; garbage aborts startup, and `setup_hook` reads it FIRST so the ValueError cannot be swallowed by `@_guild_op`). Set together with `COMPOSE_PROFILES=archive` — the pair is documented in `.env.example` |
| `BOT_SETTINGS_OVERRIDES` | `apply` | `ignore` runs the process on env and code values: the bot hash `bot:{application_id}:config` is never read (one WARNING says so), and `BotSettings` refuses to change a stored bot setting (`debug-default` is never stored, so it stays settable). The key is left in place, so removing the variable and restarting brings its values back. Unset, empty and `apply` apply what is stored; anything else aborts startup, and `setup_hook` reads it right after `HISTORY_ARCHIVE_ENABLED`. The other way back from a harmful override is `just bot-settings reset <application_id>` |
| `COMPOSE_PROFILES` | — | read by Docker Compose from `.env`, not by the bot: `archive` deploys `postgres` + `db-migrate`. `just down` names every profile (`--profile archive --profile metrics`) — a `down` with a profile inactive leaves its containers running; explicitly naming profiled services (`up -d redis postgres db-migrate`) auto-activates the profile. Three profiles exist: `archive` (postgres + db-migrate), `ops` (`db-backfill`, kept out of `up` entirely because it is run by hand — `just db-backfill-docker`), and `metrics` (`otelcol-metrics`, the docker_stats → Prometheus sidecar `-debug`'s Postgres cpu/mem row reads; it mounts the Docker socket, so it is opt-in) |
| `POSTGRES_URL` | — | **required while the archive is enabled**; `setup_hook` raises without it. Ignored (with an INFO) when disabled — the flag, never URL presence, is what enables archiving. Compose supplies it; `just run` derives it from the parts below |
| `POSTGRES_PASSWORD` | `password` | compose defaults it so a token-only archive-enabled `docker compose up` works; the bot warns loudly (startup ERROR + owner-only `-ping` row) while the default is in use AND the archive is enabled (`build_common.sh`'s preflight warns more widely: flag truthy OR the profile in `COMPOSE_PROFILES`, covering the profile-on/flag-off drift case where an idle default-credential postgres runs with the bot's warnings silenced), and `./setup_env.sh` generates a real one. Changing it after the volume is initialized needs `ALTER USER` — Postgres reads it on first init only. **`.env` is the only supported place it is set**; a per-install `POSTGRES_PASSWORD_FILE` was proposed and declined (see the comment above `DEFAULT_POSTGRES_PASSWORD` in config.py), and `using_default_postgres_password()` is scoped to the DSN shape that decision produces — do not re-add asyncpg's full resolution ladder |
| `POSTGRES_USER` / `POSTGRES_DB` | `musicbot` / `musicbot` | compose only; also the parts `just run`/`db-*` build a host DSN from |
| `POSTGRES_HOST_PORT` | `5432` | host-side published port, to dodge a local Postgres |
| `POSTGRES_MIGRATE_URL` | falls back to `POSTGRES_URL` | lets migrations run as a different (higher-privilege) role |
| `POSTGRES_STATEMENT_CACHE` | `100` | asyncpg `statement_cache_size`; set `0` behind a statement-rewriting pooler |
| `HISTORY_OUTBOX_MAX` | `0` (unbounded) | opt-in outbox ceiling, meaningful only while the archive is enabled. Dropping entries is real data loss; every drop logs ERROR |
| `ENVIRONMENT` | `development` | `main()` infers `production`/the branch slug from git when unset and a repo is present; set explicitly in CI/Docker |
| `DEBUG_MODE` | `false` | process-wide default for debug mode, which decorates every embed the bot sends with a trace/timing/runtime footer. Four seams apply it, because "every embed" is sent from four places: `MusicContext.send` (command responses), `MusicPlayer._decorate_for_debug` (the NP block at every render site — refreshed on each progress tick — plus the player's own notices), `restore_guild` (the channels-deleted notice, which has no player), and a pre-rendered `debug_suffix` threaded into the four sends that bypass `MusicContext.send` (`-ping`, `-debug`, the queue-progress card and the slow-resolve notice). Every seam routes through `DebugSettings.decorate()`, which owns the enabled check, the strip fallback, the shard and the sampler's runtime figures; the environment leading the suffix is read by `debug_footer` itself, since it is a property of the process rather than of the request. A seam passes only the span it names and, for command responses, elapsed-ms. Same strict parse as `HISTORY_ARCHIVE_ENABLED`, read ONCE by `MusicBot.__init__` so garbage aborts startup inside `load_extension`. `-debug --enable`/`--disable` and `-settings debug on`/`off`/`reset` override it **per guild, persisted to `guild:{id}:config`**, and require **Manage Server** (or the bot's operator); both write through `GuildSettings`, so they are one choice with two spellings. The stored choice survives restarts and WINS over this variable, so a guild that opted out stays out when the host default flips on; a guild that never chose follows this value and keeps following it. The operator's `-settings bot debug-default on`/`off` replaces this default for every guild that never chose, until the process restarts: it is never stored, so the variable is back in force at every start. Redis unavailable → the toggle applies in memory only and says so. The per-guild scope is scoping, not a trust boundary — it exists so enabling debug in one guild does not enable it everywhere. Observation-only — it changes what is shown, never what the bot does |
| `DEBUG_PROMETHEUS_URL` | — | Prometheus query API `-debug` reads the **postgres container's** CPU/memory from (the bot cannot see another container's cgroup, and Postgres reports no OS metrics over SQL). Compose sets `http://localhost:9090`; the series come from the `otelcol-metrics` `docker_stats` receiver, selected by `container_name="discord-postgres"`. **That collector is behind the `metrics` compose profile**, so on a default `up` it does not run and the cpu/mem rows render `n/a (no metrics source)` even though the URL is set and Prometheus answers — set `COMPOSE_PROFILES=metrics` (or `docker compose --profile metrics up -d`) as well. Unset URL → the same `n/a`. Only those two rows depend on it: the block's load/throughput/mem-signal rows are native SQL over the archive's own pool and render regardless. The container name is a hand-checked cross-file pin (see golden rule 6) |
| `PROMETHEUS_HOST_PORT` | `9090` | host-side published port for the metrics stack's Prometheus, loopback-bound. Also the port `DEBUG_PROMETHEUS_URL` defaults to — the two are written separately in compose (golden rule 6c) |
| `GIT_SHA` | — | the deploy tag, baked into the runtime image as an `ENV` (and a label). The ENV is the one the process can read, which is what lets `-debug` report the commit it is running; outside a container `-debug` shells out to `git rev-parse` instead |
| `LIVENESS_FILE` | — (`/tmp/bot-alive` in the Dockerfile) | path a loop-resident task touches every `LIVENESS_INTERVAL_SECS`, read by the runtime image's `HEALTHCHECK`. A stale mtime (>90s) means the event loop wedged while the process stayed up, which `restart: always` cannot see — it observes only the process exiting. The healthcheck **reports**, it does not act: the engine takes no action on an unhealthy container (Swarm, k8s or an autoheal sidecar do), so under compose this is a status, not a restart. NOT a dependency probe: a Redis blip must not mark the bot dead. Unset outside Docker, where the task never starts |
| `LIVENESS_INTERVAL_SECS` | `15.0` | touch cadence. Must stay well under the healthcheck's 90s staleness window; the two are written separately (Dockerfile ↔ config.py) and are not enforced by `just pins` (golden rule 6e). Between 1 and 60: a value outside refuses startup, naming the variable |
| `POT_PROVIDER_URL` | `http://127.0.0.1:4416` | bgutil PO-token sidecar base URL |
| `YTDLP_POOL_WORKERS` | `4` | extraction worker processes (~80–120 MB RSS each). Floored at 1: a value below refuses startup, naming the variable |
| `PLAY_INFLIGHT_MAX` | `16` | per-guild ceiling on `-play` requests ADMITTED at once; past it a request is declined with the existing notice. Its unit is one coroutine, one open span and one typing keepalive — memory, not pool time, which `PLAY_RESOLVE_CONCURRENCY` bounds instead. Requests resolve concurrently and serialize only at the insert. `play.inflight` on the `bot.play` span is the number that says whether 16 is right. Floored at 1 |
| `PLAY_RESOLVE_CONCURRENCY` | `2` | per-guild ceiling on admitted requests holding a yt-dlp worker to RESOLVE (`_GuildPlays.resolves`). The pool is process-wide and FIFO, so admission alone bounds nothing on it: sixteen links is sixteen jobs against four workers, and what queues behind them includes the playback loop's own in-band extractions in OTHER guilds — dead air between their songs. Half the default pool, so one guild can never hold all of it; requests wait here rather than being refused, bounded by `PLAY_RESOLVE_WAIT_SECS`. It does NOT cover the enqueue-time stream warm, which is spawned per song and bounded by `prefetch_warm_slot()` instead. Raise it with `YTDLP_POOL_WORKERS` — and raise it to `YTDLP_POOL_WORKERS` on a single-guild install, where the fairness it buys has no other guild to protect and a 3-link burst serializes its third request behind two slots while two workers idle. Floored at 1 |
| `PLAY_RESOLVE_WAIT_SECS` | `120.0` | Bound on the WAIT for one of `PLAY_RESOLVE_CONCURRENCY`'s slots, never on the extraction holding it — a 5,547-track playlist legitimately runs 99s, and a bound covering it would cancel exactly the resolve the slot was sized for. What it bounds is the stretch that produces no output at all, which cannot be told from a hung bot. Expiry raises `ResolveWaitExpired`, which the command renders: the request is declined having searched and queued nothing, so unlike a stalled place, "try again" cannot duplicate a song. It is a `PoolSlotUnavailable`, so `_extract_once` re-elects a leader on it rather than failing joiners that hold no worker. See `docs/ARCHITECTURE.md#a-resolve-that-has-to-wait`. Floored at 1.0 |
| `PLAY_SLOW_NOTICE_SECS` | `6.0` | How long a NON-collection `-play` resolves before `slow_resolve_notice` posts, covering the whole wait rather than the slot queue alone (a full pool and a slow extraction are the same silence). Above the 1–4s a warm resolve takes, so it marks the unusual rather than narrating every request. Posts through `ctx.channel.send` so it never becomes the NP host; one notice per channel (`_CLAIMED_CHANNELS`), retracted by the task that sent it when the song lands. Collections take the card instead — two messages for one `-play` is worse than either. Floored at 0.5. A server's `-settings slow-notice` overrides it for that server (4–60s, or `off`); both `-play` entry points read the server's value synchronously as they enter the notice and pass it as `slow_resolve_notice`'s required `delay=`, and `off` arms no poster at all |
| `STREAM_PROBE_TIMEOUT_SECS` | `2.0` | Cap on the pre-playback stream-URL probe. Short because a single resolve can pay it twice and exceeding it now costs a **cache entry**, not just a verdict — an unconfirmed URL still plays, so firing early is cheap. Raise it only if `stream URL probe did not complete` warnings correlate with songs that then play fine. Floored at 0.1 and refused non-finite |
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | `3.0` | NP progress-bar edit cadence. Floored at 1.0 and refused non-finite: every playing tick is a real edit on the channel's 5-edits-per-5s bucket. A server's `-settings np-refresh` can only slow its own bar: the updater sleeps the larger of the two, read each tick (`GuildSettings.np_refresh_secs`) |
| `HEARTBEAT_INTERVAL_SECS` | `3.0` | How often a playing guild records its playback position for crash recovery. Bounds the worst-case recovery error — a crash resumes at the last heartbeat, so at most this many seconds replay. Same default as the progress bar because the same reasoning applies, but a separate knob: one is display cadence, the other durability. Floored at 0.5s and refused non-finite — each tick is a Redis write per PLAYING guild, so `0` would be an unbounded HSET loop and `inf` would silently disable recovery |
| `QUEUE_PROGRESS_DELAY_SECS` | `2.5` | How long a COLLECTION enqueue resolves before the live progress card appears. Marks the unusual rather than narrating every `-play`: a cache-hit playlist is one Redis GET and lands first. Above 2.0 because a measured ten-track enqueue ran 2.03s end to end; the real trigger is a cliff at ~101 tracks (a second continuation page ≈ 3.1s), so any value between them behaves the same. The clock starts when the card's context manager is entered, at the top of `_resolve_and_place`, so it is effectively the whole command. A server's `-settings queue-progress-delay` overrides it for that server (2–60s); both entry points read the server's value synchronously as they enter the card and pass it as `enqueue_progress`'s required `delay=` |
| `QUEUE_PROGRESS_TICK_SECS` | `5.0` | Card edit cadence, floored at **2.0s** rather than the dashboards' 0.05s. Discord allows 5 edits / 5s per CHANNEL, one bucket shared with the NP bar's 3.0s cadence; a 429 never reaches `safe_edit` (discord.py sleeps it internally), so the symptom is the NP bar silently freezing. `-ping`/`-debug` can share the lower floor because their deadlines cap the damage at ~8 edits |
| `QUEUE_PROGRESS_MAX_SECS` | `300.0` | The card's own ceiling. Nothing else bounds the resolve it watches — `PLAY_RESOLVE_WAIT_SECS` bounds the wait for a slot and not the extraction, `PLACE_TIMEOUT_SECS` bounds 0.01s of a 29s command. Past it the card says so once and stops editing, and it stays up, holding the channel, until the enqueue settles. Must be at least `QUEUE_PROGRESS_DELAY_SECS` + `QUEUE_PROGRESS_TICK_SECS`, its default included, or startup is refused. `-settings bot` moves the three separately, so each card reads them once on entry and runs to `queue_progress.card_ceiling`: this value, or its delay plus two ticks if that is longer. The card sends after its delay and checks the ceiling only after a tick, so anything shorter stalls it without one ordinary edit |
| `PING_TICK_SECS` / `PING_DEADLINE_SECS` | `1.0` / `3.0` | -ping live-edit loop |
| `DEBUG_TICK_SECS` / `DEBUG_DEADLINE_SECS` | `1.0` / `8.0` | -debug live-edit loop. Longer deadline than -ping's: each block does more work (the Postgres probe brackets a 2s sampling window between two stats queries, plus a Prometheus round trip) and a straggler renders `⚠️ timed out` rather than being retried — keep the deadline comfortably above that ~2.2s floor. The tick is a CEILING, not a cadence — the loop wakes on the first probe to finish |
| `ANALYTICS_RENDER_DEADLINE_SECS` | `20.0` | How long `-analytics` waits for its chart before sending the card without one. Sized for the COLD path, which dominates: measured end to end in the deployed image at **5.9s**, of which the render is ~1.0s — the rest is what a spawned worker pays on the way up (`import src.main` 3.6s under forkserver, matplotlib 2.4s). Twenty rather than ten because expiring is SILENT: the card still sends, just without its chart. It bounds the CALLER, not the pool — a `ProcessPoolExecutor` cannot cancel a running call, so the worker finishes its render regardless. Same `_float_env` floor as the dashboard knobs |
| `OTEL_SDK_DISABLED` | `false` | `true` disables tracing/log export (stdout logs remain) |
| `OTEL_SERVICE_NAME` / `OTEL_EXPORTER_OTLP_ENDPOINT` | `discord-music-bot` / `http://localhost:4317` | |

## Concurrency primitives

| Primitive | Protects |
|---|---|
| `GuildSettings` per-guild write lock (src/settings.py) | `guild:{id}:config`: it is the ONLY writer (`-settings`, `-volume`, `-debug --enable/--disable`, restore's volume migration, guild removal), each write a bounded store call then a synchronous commit that stamps (guild, field) from one sequence counter. A read (restore's `seed`, the startup `hydrate`, `load`) never locks: it skips any field stamped after it began, so a read straddling a write never undoes it. The locks are refcounted and dropped when idle, and `reading()` registrations bound how long stamps and forget marks live. `DebugSettings` is its projection of `debug_mode`. `docs/ARCHITECTURE.md#settings-resolution` |
| `config`'s knob maps (`_OVERRIDES`, `_BASELINES`) | each bot knob's `-settings bot` override and environment value, by variable. One writer of an override, `BotSettings` in src/settings.py (guard G3); read synchronously by calling the knob's handle, `config.<knob in lower case>()`, when the value applies — never its `.baseline`, never at import (G1, G2), and never in a pool worker (G4). A reload of `config` drops them. `docs/ARCHITECTURE.md#settings-resolution` |

## Recipes

**Add a per-guild SETTING** (a durable choice, not runtime state): constant in
`ConfigField`, and its name in the `ConfigFieldName` Literal and, unless it has its own
reset as `volume` does, `ResettableConfigField` → `Optional` field on `GuildConfig` (Optional is not optional — absent
must keep meaning "follow the host default", or "never chose" collapses into "chose
the default") → a numeric field gets a `CONFIG_DOMAIN` entry in `guild_state.py`, and that
entry is its `to_redis` and `from_redis` too: both iterate the domain. The `-settings`
registry's static bounds must equal it (a test compares them), and outside it
`GuildConfig.__post_init__` reads a value as unset. A field that is not a number (as
`debug_mode` and `timezone` are not) gets its own line in both: `to_redis` writes it only
when set, `from_redis` reads an unrecognised value as unset →
the store writes it with `GuildRedisStore.update_config`, which PERSISTs and takes
`writer=`; a field that needs a write-boundary check or a second copy (as `timezone`
and `volume` do) gets a dedicated writer instead, added to the set `update_config`
refuses, that PERSISTs, takes `writer=` and
**encodes through `GuildConfig(field=value).to_redis()` rather than by hand** (a
single-field config serializes to exactly that field, so the wire format has one
definition and a setter cannot drift from what `from_redis` expects), and that
writer's row in `GuildSettings._dispatch` (src/settings.py), where every other field
falls through to `update_config`. `_dispatch` is the ONLY caller of the store's config
writers (`TestGuildConfigHasOneWriter` fails any other), and everything writes through
`GuildSettings.write`/`reset`, never the store → **validate at the write boundary if
the value is user-typed** (see `valid_timezone`: a bad value stored here fails
silently — the write succeeds, the command reports success, and the guild keeps the
default forever) → a hot path reads it synchronously from `GuildSettings`' cache,
never by awaiting Redis on a send → a registry `SettingSpec` in the commit that wires
the code reading it, whose `ConfigField` value is the key with `-` → `_`, plus `_secs`
for a time value (registry invariants 2 and 10) → tests in test_guild_state.py,
test_redis_client.py and test_settings.py. It goes in `guild:{id}:config`, NOT
`guild:{id}:state`: that hash carries a 24h TTL and a setting stored there reverts on
any guild idle for a day.

**Add a bot setting** (an owner-tunable, process-wide knob): declare it once in `config.py`,
`<name in lowercase> = _secs("<NAME>", default, minimum=<a named floor>)` (`_count` for an
int). That statement is the env parse, the accessor and the `bot:{id}:config` field → every
consumer calls `config.<name>()` when the value applies. Never call it at
import, class-body or default-argument time, and never read its `.baseline`
(`TestBotKnobsAreReadAtCallTime` G1/G2) → a registry `SettingSpec` with `knob=config.<name>`
and a chat range whose minimum is strictly above `knob.floor` for a time-valued knob, at least
equal for a count (invariant 3, which also fails a knob with no spec); the spec is also its
`-debug` row → tests set it with
`config.<name>.set_override(value)`. A value built into
a long-lived object (a semaphore, a session) applies only when that object is rebuilt, and the
spec's `applies` string says so. No pool worker may read it (G4).
