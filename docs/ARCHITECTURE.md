# Architecture

_Last full audit: 2026-07-14, verified line-by-line against `src/` on branch `task/help-command-2` (adds the `-help`/`--help` man-page help surface and the `-playnow` interjection subsystem)._

_Durable-tier update: 2026-08-02 — history, Redis eviction, deployment topology, modules and shutdown were re-verified against `src/`. The Postgres archive is **opt-in** (`HISTORY_ARCHIVE_ENABLED`, default off) and is **not** on the `-history` read path; see [History read path](#history-read-path) and [History archive tier](#history-archive-tier)._

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Deployment](#deployment)
4. [Module Reference](#module-reference)
5. [Commands Reference](#commands-reference)
6. [Configuration](#configuration)
7. [System Flows](#system-flows)
   - [Startup and Initialization](#startup-and-initialization)
   - [Voice Channel Join](#voice-channel-join)
   - [-play Command Pipeline](#-play-command-pipeline)
   - [-playnow Interjection](#-playnow-interjection)
   - [Source Resolution](#source-resolution)
   - [yt-dlp Pipeline](#yt-dlp-pipeline)
   - [Playback Loop](#playback-loop)
   - [Queue Operations](#queue-operations)
   - [Now Playing Host Model](#now-playing-host-model)
   - [Pause / Resume](#pause--resume)
   - [Auto-Disconnect](#auto-disconnect)
   - [Crash Recovery](#crash-recovery)
   - [Graceful Shutdown](#graceful-shutdown)
8. [Data Model](#data-model)
9. [Concurrency Model](#concurrency-model)
10. [Spotify Integration](#spotify-integration)
11. [Observability](#observability)
12. [Audio Pipeline](#audio-pipeline)
13. [Per-Guild State Machine](#per-guild-state-machine)
14. [History Archive Tier](#history-archive-tier)
    - [History read path](#history-read-path)
    - [Redis memory bounds](#redis-memory-bounds)
    - [Postgres credential handling](#postgres-credential-handling)
    - [History backfill](#history-backfill)
15. [Subsystem Invariants](#subsystem-invariants)
    - [Redis connection retry](#redis-connection-retry)
    - [yt-dlp client strategy](#yt-dlp-client-strategy)
    - [yt-dlp process boundary](#yt-dlp-process-boundary)
    - [Queue invariant](#queue-invariant)
    - [Now Playing host model](#now-playing-host-model)
    - [Debug footer seams](#debug-footer-seams)
16. [Design Decisions](#design-decisions)

---

## System Overview

The discord-music-bot is a single-process Python asyncio application. It connects to Discord's gateway, resolves audio metadata from YouTube, Spotify, or SoundCloud, and streams Opus-encoded audio to Discord voice channels via FFmpeg.

The data layer is two-tier: **Redis** holds runtime state (queue, playback state, volume — so the bot can recover after a container restart and rejoin the voice channel it was in), evictable caches, the capped `guild:{id}:history` display window, and the `history:outbox` write-ahead buffer; **Postgres** is the **opt-in** durable tier (`HISTORY_ARCHIVE_ENABLED`, default off) — the `play_history` table holds every played song, fed asynchronously from the outbox so the playback loop never awaits Postgres. The rule: data a user would miss a week later goes to Postgres; data that only matters to the running player stays in Redis, permanently. An OpenTelemetry pipeline exports traces, structured logs, and metrics to a local Grafana LGTM stack.

```mermaid
graph TD
    User["Discord User"]
    GW["Discord Gateway (WebSocket)"]
    VC["Discord Voice (UDP/Opus)"]
    Bot["discord-music-bot\n(Python 3.14 asyncio)"]
    Redis["Redis 7\n(state + cache + outbox)"]
    PG["Postgres 18\n(play_history — durable tier)"]
    LGTM["grafana/otel-lgtm\n(traces + logs + metrics)"]
    YT["YouTube CDN\n(signed stream URLs)"]
    SP["Spotify API\n(track/playlist metadata)"]
    SC["SoundCloud\n(via yt-dlp)"]

    User -->|text command| GW
    GW -->|gateway events| Bot
    Bot -->|Opus audio| VC
    VC -->|heard by| User
    Bot <-->|queue / state / cache / outbox| Redis
    Bot -->|"outbox drain (asyncpg)"| PG
    Bot -->|OTLP gRPC| LGTM
    Bot -->|yt-dlp HTTP| YT
    Bot -->|REST API| SP
    Bot -->|yt-dlp HTTP| SC
```

---

## Technology Stack

| Component | Package / Version | Role |
|---|---|---|
| Runtime | Python 3.14 (`requires-python >=3.14,<4.0`) | Asyncio event loop |
| Discord client | `discord.py` 2.7.1 | Gateway, voice, commands framework |
| Audio extraction | `yt-dlp` 2026.8.18.122307.dev0 (pinned to a **nightly**; `[default, deno]` extras) | YouTube / SoundCloud metadata and stream URLs; extras ship `yt-dlp-ejs` (JS challenge solver) + the Deno runtime so yt-dlp's fallback client stays available |
| PO token provider | `bgutil-ytdlp-pot-provider` 1.3.1 (pip plugin, pinned to the sidecar image tag) | Mints GVS Proof-of-Origin tokens via the `discord-pot-provider` sidecar so the fallback client's formats are served at all |
| Codec | FFmpeg (system, installed in the runtime image) | Decode + re-encode to Opus for Discord |
| State / cache | `redis` 8.x (`redis.asyncio` client; constraint `>=5.0.0`) | Runtime queue/state, yt-dlp URL cache, Spotify cache, `history:outbox` buffer |
| Durable tier | `asyncpg` (Postgres 18) | `play_history` archive: outbox drain writes, `-leaderboard` reads, in-app SQL migration runner (`src/db_migrate.py`) |
| Serialization | `orjson` | Fast JSON serialization for Redis payloads |
| HTTP client | `aiohttp` | Spotify REST API calls |
| JSON (Spotify) | `ujson` | Spotify response deserialization |
| Voice crypto | `PyNaCl` 1.6.2 | Opus packet encryption required by Discord |
| Tracing | OpenTelemetry SDK ≥1.27 + OTLP gRPC exporter | Spans for commands, playback, yt-dlp, Spotify |
| Auto-instrumentation | `opentelemetry-instrumentation-redis` / `-aiohttp-client` / `-asyncpg` | Client spans for Redis, Spotify HTTP, and Postgres calls |
| Logging | `structlog` (JSON) → OTLP log exporter | Structured logs correlated with traces |
| Type checking | `pyright` (`pythonVersion 3.14`) | Static analysis (`python -m pyright src/`, 0-error convention) |
| Testing | `pytest` + `pytest-asyncio` + `fakeredis`; `testcontainers[postgres]` for the opt-in pg tier (`RUN_PG_TESTS=1 poetry run pytest -m pg --no-cov`) | Async test suite (coverage floor 80%); real-Postgres integration tests |

---

## Deployment

Five containers are defined in [docker-compose.yml](../docker-compose.yml):

| Container | Image | Key config |
|---|---|---|
| `discord-music-bot` | Local build (tagged with git SHA) | `restart: always`, `network_mode: host`, `depends_on: service_healthy` on Redis, named volume for yt-dlp's disk cache (player JS + solved challenges survive restarts) |
| `discord-redis` | `redis:7-alpine` | AOF persistence, 256 MB `volatile-lru` eviction, healthcheck |
| `discord-postgres` | `postgres:18-alpine` | The durable tier (`play_history`). On the `archive` profile, which the deploy tooling activates from `HISTORY_ARCHIVE_ENABLED` — Compose never reads that flag itself, so a raw `docker compose up` deploys no Postgres; NOT in `depends_on` — the outbox buffers in Redis until it comes up. Volume target is `/var/lib/postgresql` (the 18+ image layout — the old `.../data` path would store the cluster outside the volume) plus a `postgres-backups` volume at `/backups` for `pg_backup.sh`'s nightly dumps |
| `discord-pot-provider` | `brainicism/bgutil-ytdlp-pot-provider:1.3.1` (tag locked to the pip plugin pin) | GVS Proof-of-Origin token minting for yt-dlp's fallback client, `127.0.0.1:4416`; NOT in `depends_on` — the bot degrades gracefully without it (see [PO_TOKEN_SIDECAR_PLAN.md](PO_TOKEN_SIDECAR_PLAN.md)) |
| `discord-otel-lgtm` | `grafana/otel-lgtm` | All-in-one Grafana (UI `:3000`) + Tempo + Loki + Mimir, OTLP gRPC on `:4317` |

`network_mode: host` is used because the bot makes only outbound connections — no inbound ports are needed.

Redis is configured with:
- `appendonly yes` + `appendfsync everysec` — data survives container restarts with at most 1 second of loss
- `maxmemory 256mb` + `maxmemory-policy volatile-lru` — **only TTL-carrying keys are eviction candidates**. Caches (`ytdl:*`, `spotify:*`) and the `guild:{id}:{state,queue,now_playing}` keys carry TTLs and are reconstructible or re-creatable. **Three kinds of key deliberately carry none** and must never become candidates: `history:outbox` (plays not yet durable in Postgres), `guild:{id}:history` (the capped, PERSISTed window `-history` reads) and `guild:{id}:config` (a guild's durable choices — evicting one silently reverts a setting the guild made). Never switch to `allkeys-*` — see [Redis memory bounds](#redis-memory-bounds)

```mermaid
graph LR
    subgraph Host["Host (network_mode: host)"]
        subgraph compose["docker-compose"]
            Bot["discord-music-bot\n(Python 3.14)"]
            Redis["discord-redis\n(Redis 7-alpine)\n127.0.0.1:6379"]
            PG["discord-postgres\n(Postgres 18-alpine)\n127.0.0.1:5432"]
            POT["discord-pot-provider\n(bgutil)\n127.0.0.1:4416"]
            LGTM["discord-otel-lgtm\nGrafana :3000\nOTLP gRPC :4317"]
        end
        Bot -->|"localhost:6379"| Redis
        Bot -->|"localhost:5432 (asyncpg)"| PG
        Bot -->|"localhost:4416 (PO tokens)"| POT
        Bot -->|"localhost:4317"| LGTM
    end
    Bot -->|"HTTPS / WSS"| Internet["Discord / YouTube / Spotify"]
```

**Dockerfile** ([Dockerfile](../Dockerfile)) — multi-stage on `python:3.14-slim`:

| Stage | Contents |
|---|---|
| `base` | Env defaults; Poetry 2.1.3 pinned via `POETRY_VERSION`; in-project venvs |
| `builder` | `build-essential`, Poetry, `poetry install --only=main` with BuildKit cache mounts (pip + poetry caches persist across builds — matters for yt-dlp's frequent updates and pynacl's C compile) |
| `test` | Builder venv + `--only=main,test` deps + `tests/`; used by the container-test CI job, never pushed |
| `runtime` | `ffmpeg` via apt, venv copied from builder (no Poetry in the final image), `src/` copied last for layer caching, `CMD python -m src.main`, `ENVIRONMENT=production` build-arg default |

**Build & deploy**: three pipelines share one image (the `Dockerfile` runtime stage), one test gate (`just check` — ruff, pyright, pytest), and one secret set — they diverge only at the "run it" step. (1) **Compose** — [build_docker.sh](../build_docker.sh): gate, build the image tagged `:latest` + `:$GIT_SHA`, then hand off to [deploy_docker.sh](../deploy_docker.sh), which resolves the `archive` profile from `HISTORY_ARCHIVE_ENABLED`, applies pending migrations through the `db-migrate` one-shot (gating the deploy on the result, so a failed migration leaves the running bot untouched), and then `docker compose up -d`. Build and deploy are separate entry points on purpose (build once, deploy many): `./deploy_docker.sh <git-sha>` rolls back to any already-built image without a rebuild or a re-run of the gate, and refuses tags absent from the local store rather than letting Compose build one from the working tree and label it with that SHA. Developer-facing verbs (`just lint|types|test|check|build|up`) are indexed in the [justfile](../justfile), which CI's lint and test jobs also call rather than reimplement. (2) **Kubernetes dev** — `build_k8s_dev.sh`: same local gate, then an atomic Kustomize apply to Docker Desktop's built-in cluster (shared image store, no registry). (3) **Kubernetes prod** — `build_k8s_prod.sh`: deploy-only and provenance-gated (HEAD on `origin/main`, CI-built GHCR image must exist) against the k3s server. The two k8s entry points share their cluster guards and the atomic apply via `k8s_common.sh`; the compose and dev-cluster paths share the test gate and image build via [build_common.sh](../build_common.sh) — both are sourced libraries, not runnable. Teardown (either cluster): `k8s_down.sh`, the `docker compose down` peer. Manifests: `deploy/k8s/` (base + dev/production overlays); ops: `deploy/k8s/README.md`. Deploys are seamless by design — the pod is killed without cleanup and the next boot's crash recovery resumes playback (never "fix" the missing SIGTERM handler). At most one pipeline runs the bot at a time (single Discord token). CI (`.github/workflows/`) runs lint/type/test **and** the GHCR image build in one workflow (`ci.yml` — the build job is `needs`-gated on the three check jobs and `if`-gated to pushes on `main`; it was split out into a `build.yml` triggered by `workflow_run` until that turned out to tag images with the default branch's tip rather than the commit that passed), plus security scans (`security.yml`). Python setup is shared by `.github/actions/setup-python-env`.

> **Note (2026-07-21):** the two Kubernetes pipelines and `k8s_common.sh` / `k8s_down.sh` above are written and dev-validated but **unmerged** (`task/k8s-deployment*`); only the Compose pipeline exists on `main` today. `build_common.sh` on `main` already exposes the exact API those branches source (`resolve_environment` / `run_test_gate` / `build_runtime_image`) so the merge stays a one-file resolution.

---

## Module Reference

```mermaid
graph TD
    main["src/main.py\nMusicBotApp + MusicContext"]
    musicbot["src/musicbot.py\nMusicBot (Cog)"]
    musicplayer["src/musicplayer.py\nMusicPlayer"]
    guild_queue["src/guild_queue.py\nGuildQueue"]
    guild_history["src/guild_history.py\nGuildHistory"]
    guild_state["src/guild_state.py\nschema / value objects"]
    history_archive["src/history_archive.py\nPostgresHistoryArchive + drainer"]
    db["src/db_migrate.py\nSQL migration runner"]
    youtube["src/youtube.py\nYTDL + QueueObject"]
    ytdlp_pool["src/ytdlp_pool.py\nYtdlpPool"]
    sources["src/sources.py\nparse_input + source types"]
    spotify["src/spotify.py\nSpotify client"]
    redis_client["src/redis_client.py\nGuildRedisStore + cache helpers"]
    telemetry["src/telemetry.py\nOTel + structlog setup"]
    config["src/config.py\nENVIRONMENT + tunables"]
    help_cmd["src/help.py\nMusicHelpCommand"]
    dashboard["src/dashboard.py\noptimistic-send + live-edit driver"]
    ping["src/ping.py\n-ping probes + rendering"]
    debug_mod["src/debug.py\n-debug snapshot + debug mode"]
    util["src/util.py\nlogging + embed helpers"]

    main --> musicbot
    main --> redis_client
    main --> telemetry
    main --> config
    main --> help_cmd
    main --> db
    main --> history_archive
    musicbot --> musicplayer
    musicbot --> ping
    musicbot --> debug_mod
    ping --> dashboard
    debug_mod --> dashboard
    debug_mod --> ping
    main --> debug_mod
    musicbot --> sources
    musicbot --> spotify
    musicbot --> youtube
    musicplayer --> guild_queue
    musicplayer --> guild_history
    musicplayer --> youtube
    musicplayer --> redis_client
    guild_queue --> guild_state
    guild_queue --> redis_client
    guild_history --> redis_client
    guild_history --> history_archive
    history_archive --> db
    history_archive --> guild_state
    history_archive --> redis_client
    redis_client --> guild_state
    youtube --> redis_client
    youtube --> ytdlp_pool
    spotify --> redis_client
```

| Module | Responsibility |
|---|---|
| `main.py` | Entry point. `MusicBotApp` (extends `AutoShardedBot`): `setup_hook` creates the Redis pool, wires the durable tier when **`HISTORY_ARCHIVE_ENABLED` is true** (`PostgresHistoryArchive` → `HistoryOutboxDrainer.start()`) — the flag is the consent gate, never URL presence: enabled without `POSTGRES_URL` **raises**, and disabled ignores a set one with an INFO, leaving bit-identical pre-Postgres behavior, and loads extensions; `close()` tears down drainer → database → Redis pool and flushes telemetry off-loop; `invoke()` is overridden so that `--help` anywhere in a command message short-circuits to that command's help embed *before* any check or argument parsing runs; `help_command=MusicHelpCommand()` replaces discord.py's plaintext default. `MusicContext` (custom `commands.Context`, installed via `get_context` override): its `send()` glues the Now Playing embed block to the bottom of the player's channel (see [Now Playing Host Model](#now-playing-host-model)). `main()` calls `setup_telemetry()` before anything else. |
| `musicbot.py` | `MusicBot` Cog. All Discord commands (including `-playnow`, which resolves a source and calls `MusicPlayer.interject()`). Owns `mps: dict[guild_id → MusicPlayer]`, the per-guild alone-disconnect timers, and per-command OTel spans + structlog contextvars (`cog_before_invoke`/`cog_after_invoke`). Handles voice-state events (auto-disconnect) and crash recovery via `on_ready`. |
| `musicplayer.py` | Per-guild playback orchestration: `loop()` task, prefetch task, progress-bar task, Now-Playing host management, embeds/ETA, presence updates, pause/resume accounting, and `-playnow` interjection (`interject()` → `InterjectOutcome`, resume-entry bookkeeping via `_skip_history_for`). Delegates every queue operation to `self.queue: GuildQueue` and history to `self.history: GuildHistory`. |
| `guild_queue.py` | `GuildQueue` — the queue domain class. Privately owns **one deque plus a cursor into it** (`_items[:_cursor]` claimed, `_items[_cursor:]` pending) and the Redis mirror, along with the bulk-mutation mutex, the cleared-flag, the generation counter, and the `_wake` Event whose sole writer is `_sync_wake()`. Every queue operation (put/clear/shuffle/remove/restore/dequeue bookkeeping) lives here. This replaced an `asyncio.Queue` plus a parallel display `deque` whose agreement had to be maintained by hand — see [Queue invariant](#queue-invariant). |
| `guild_history.py` | `GuildHistory` — played-song history domain class. Two legs, both bounded at `HISTORY_CACHE_LIMIT` (50): the PERSISTed `guild:{id}:history` Redis list and an in-memory deque of the same window. `recent()` merges those two and **never reads Postgres** — see [History read path](#history-read-path). Writes additionally XADD the outbox while the archive is enabled. |
| `guild_state.py` | Schema module: **every byte persisted to Redis is defined here**. Field-name constants (`StateField`, `NowPlayingField`, `QueueEntryField`, `ConfigField`) + frozen value objects (`GuildStateData`, `NowPlayingData`, `SongQueueEntry`/`SearchQueueEntry`, `GuildPlaybackSnapshot`, `HistoryEntry`, `GuildConfig`) with `from_redis`/`to_redis` converters. `GuildConfig` is the durable-settings object behind `guild:{id}:config`, and every one of its fields is `Optional` on purpose: absent means "follow the host default", which an explicit `False`/`0.0` does not (`tzinfo()` resolves the stored IANA name at read time, falling back to `DEFAULT_TIMEZONE` rather than raising on a render path). Pure data — no domain logic, no project runtime imports. Wire formats are pinned by golden-fixture tests. |
| `db_migrate.py` | The SQL migration runner (`python -m src.db_migrate`, also `just db-migrate`). Forward-only `NNNN_description.sql` files in `migrations/`, ordered numerically, recorded in the `schema_migrations` ledger, each applied in its own transaction under `pg_advisory_xact_lock` (so a migration must be idempotent-safe on retry). Holds `EXPECTED_SCHEMA_VERSION`; the app verifies that version and never applies DDL itself. Every deploy runs it before recreating the bot and aborts on failure; a database ahead of the build exits 0 with a note, matching the archive's own tolerance, so rollbacks deploy. `POSTGRES_MIGRATE_URL` lets migrations run as a higher-privilege role. |
| `history_archive.py` | Postgres archive + drainer: `HistoryArchive` protocol (writes), `ArchiveReader` protocol (the read surface MusicBot holds: `-ping`'s liveness probe and `-leaderboard`'s aggregate), `PostgresHistoryArchive` (lazy asyncpg pool, `HistoryEntry`↔row mapping, schema-version check, `leaderboard()`), `HistoryOutboxDrainer` (one supervised task per process: replay this consumer's pending IDs → read new → `INSERT … ON CONFLICT DO NOTHING` → `XACK`+`XDEL` by ID; at-least-once, deduped by `play_history_dedup`). The outbox is a **stream with a `drainers` consumer group**, so two live drainers are safe by construction. Present only when `HISTORY_ARCHIVE_ENABLED` is true. |
| `backfill_history.py` | One-shot CLI (`just db-backfill [--dry-run]`): copies pre-archive `guild:{id}:history` entries into `play_history`, stamping the real guild id from the key (legacy entries parse as `guild_id=0`). Inserts directly rather than through the outbox. Idempotent (dedup index + ON CONFLICT), so it is safe to re-run and safe to interrupt. Must run **before** this build is deployed — `push_history` LTRIMs each list on the guild's next song end. |
| `youtube.py` | yt-dlp integration. `QueueObject` dataclass. `YTDL(FFmpegOpusAudio)` with frame-counted position tracking. `yt_source`, `yt_stream`, `prefetch_stream`, `yt_playlist` classmethods. Holds the process's one `YtdlpPool` instance. |
| `ytdlp_pool.py` | `YtdlpPool` — lifecycle for the process pool that runs yt-dlp extraction: lazy creation, prewarm, heal-a-broken-pool-once, bounded shutdown, `PoolClosedError` after close. Knows nothing about yt-dlp (the callable is supplied per call). |
| `sources.py` | Input parsing. `parse_input`/`parse_url` classify a string into `YTSource` (track or playlist), `SpotifySource` (track or playlist), or `SoundcloudSource`. |
| `spotify.py` | Spotify Client Credentials API over aiohttp. Double-checked locking for token refresh; token itself is Redis-cached across restarts. `track`, `playlist`, `artists`, `albums` methods with per-type Redis cache TTLs. |
| `redis_client.py` | Connection-pool lifecycle + `GuildRedisStore` (per-guild Redis ops: queue/state/now-playing/history/config keys, pause epochs, recovery gate + lock, atomic start-song transaction). Every write to `guild:{id}:config` `PERSIST`s it and no path `EXPIRE`s it — that key is a guild's durable settings and is excluded from every shared TTL pipeline. Module-level `cache_get`/`cache_set`, the outbox-stream helpers, `read_guild_configs` (pipelined, chunked — the per-guild fan-out it replaced exhausted the connection pool above `max_connections` guilds and reported the failures as "never chose") and Spotify-token helpers. Every store method catches and logs Redis errors — Redis being down degrades persistence, never playback. |
| `leaderboard.py` | `-leaderboard`'s tunables (`TOP_N`, `MAX_DAYS`, `CACHE_TTL_SECS`), `LeaderboardFlags`, the Redis result-cache codec (`cache_key`/`to_cache`/`from_cache`, versioned so a shape change cannot decode stale) and the embed renderer (`build_embed`). Pure — takes a `Leaderboard` and returns strings, dicts or an embed. The command stays on the cog, where dispatch, the archive handle and the error-embed policy are. Cannot live in `util.py`: that module is in the yt-dlp worker import graph and this one reads `history_archive`'s row types. |
| `telemetry.py` | `setup_telemetry()` (tracer + logger + **meter** providers, OTLP gRPC exporters, structlog config, asyncpg/redis/aiohttp auto-instrumentation; no-op when `OTEL_SDK_DISABLED=true`), `get_tracer()`, `get_meter()` (API-level proxy — instruments created before setup are no-ops that upgrade when the provider lands), `shutdown_telemetry()` (force-flush incl. metrics). |
| `config.py` | The one module that answers "what does the bot read from the environment?". `ENVIRONMENT` (from `$ENVIRONMENT`, else derived from the git branch: `main` → `production`), `NOW_PLAYING_UPDATE_INTERVAL_SECS` (3.0), and the four live-dashboard knobs `PING_TICK_SECS`/`PING_DEADLINE_SECS` (1.0/3.0) and `DEBUG_TICK_SECS`/`DEBUG_DEADLINE_SECS` (1.0/8.0) — read through `_float_env`, which refuses non-finite values separately from its floor because `inf` makes a deadline never expire (the command holds its `max_concurrency` slot forever) and a tick of `0` turns the driver's timed wait into a hot spin. Plus the call-time accessors: `history_archive_enabled()`, `postgres_url()`, `using_default_postgres_password()`, `debug_mode_default()` (the host default for guilds that have never chosen — a stored `guild:{id}:config` choice wins over it) and `debug_prometheus_url()`. Every boolean goes through one strict parse table: unset and empty are False, a typo raises rather than silently reading as off. |
| `help.py` | `MusicHelpCommand` — a `commands.HelpCommand` subclass rendering the command list and per-command help as man(1)-styled embeds (NAME / SYNOPSIS / DESCRIPTION / EXAMPLES / NOTES). Per-command copy (`brief`/`help`/`usage`/`extras`) lives on the command declarations in `musicbot.py`; categories/order come from `CATEGORY_COMMANDS`. `get_destination()` returns the `MusicContext` (not the bare channel) so help output routes through the NP-block attach path. |
| `dashboard.py` | `run_live_dashboard` — the optimistic-send + live-edit driver `-ping` and `-debug` share. Launch the probes concurrently, send what is already known immediately, edit that **one** message as results land, and stop at a deadline so a dead dependency cannot hold the reply open forever. Only the sequencing lives here: what a "result" *is* (a `ProbeResult` row, a block of rendered lines) stays with the caller, which supplies `settle`/`abandon`/`render` callbacks over its own state. Edits only when the render actually changed, so the common case is one edit rather than one per tick. Every probe's exception is retrieved wherever it settles — one cancelled at the deadline can still raise while unwinding, *after* the driver has returned. Both callers reply through `ctx.channel.send`, never `MusicContext.send`: a message an edit loop owns must not also be the NP host — see [Now Playing Host Model](#now-playing-host-model). |
| `ping.py` | `-ping`'s probes and rows: Discord, Redis, Spotify, the Postgres archive and the OTLP endpoint, plus the bot / yt-dlp / FFmpeg version tuple (`collect_versions`, cached and executor-hopped). Sequencing is `dashboard.py`; `musicbot.py` holds only the command registration. The probes are deliberately **not** shared with a healthz endpoint — healthz must stay a dumb liveness probe, or a Redis blip becomes a pod restart loop. |
| `debug.py` | `-debug`: the snapshot's collectors and rendering, plus `--enable`/`--disable` argument parsing. **Observation-only by rule** — nothing here changes playback, caching, queueing or persistence, which is what keeps "test with debug on, ship with debug off" a valid methodology. Every collector degrades to a labeled `unknown`/`n/a` rather than raising (`_safe_block`): a debug tool that crashes is worse than no debug tool. The host blocks are gated on bot ownership at **collection**, not at render, so a non-owner's `-debug` launches no probe at all — the public surface is versions plus this server's own player/voice state. The `Config` block renders a deny-by-default allowlist: `SECRET` variables show `set`/`unset` and never a value, `URL` variables lose userinfo, credential-bearing query params and the host itself. `musicbot.py` owns the command registration and the per-guild override cache. |
| `util.py` | `get_logger` (structlog), `queue_message` (numbered list, capped at 10), `notice_embed`/`send_embed` (every command response is an embed — see design note), `cancel_task`, `latency_color`, `trace_footer`, `record_span_error`. |

**Key types:**

| Type | Module | Description |
|---|---|---|
| `QueueObject` | `youtube.py` | Dataclass: `webpage_url`, `title`, `requester`, `ts` (seek secs), `user_input`, `duration`, `uploader`, `thumbnail`, `persisted` (False only for the crash-recovered current song) |
| `YTDL` | `youtube.py` | `FFmpegOpusAudio` subclass with full song metadata; counts its own `read()` calls → `elapsed_secs`/`position_secs`; the object passed to `voice_client.play()` |
| `YTSource` | `sources.py` | Frozen dataclass: `url`, `ytsearch`, `ts`, `process`, `type` (`YTType.TRACK`/`PLAYLIST`), `list_id`, `index` (the playlist's 1-based start position) and `video_id` (the link's `v=`, kept only to tell whether `ts` belongs to the queued head) — an unresolved YouTube item |
| `SpotifySource` | `sources.py` | Frozen dataclass: `type` (`SpotifyType.TRACK`/`PLAYLIST`), `id` |
| `SoundcloudSource` | `sources.py` | Frozen dataclass: `url` |
| `GuildQueue` | `guild_queue.py` | Queue domain class; `QueueItem = Union[QueueObject, YTSource]` is the live-item type |
| `SongQueueEntry` / `SearchQueueEntry` | `guild_state.py` | At-rest queue entries (`"qobj"` / `"ytsource"` wire discriminator). `SongQueueEntry` also carries the `-playnow` fields `interjected` / `is_resume` / `start_paused`, the play's start `played_at`, and the `np_message_id` / `np_channel_id` / `np_dedicated` pointer a resume tail disposes its fragment's card by; both carry the ask-time analytics `queued_at` / `queue_position` (flat on the wire; grouped as `Analytics` in memory), the parse-time `query_source`, and `user_input` — what the user typed, which `-remove` matches on. For an unresolved Spotify-playlist track that field is the **only** surviving record of the collection link: its `ytsearch` is a title the expansion generated, and the YouTube URL it resolves to names neither |
| `Analytics` | `guild_state.py` | The pure-analytics values a live queue object carries — `queued_at` / `queue_position`, zero reads outside serialize/carry. Frozen, so carry sites can alias one instance. **In-memory only**: every wire shape and Postgres column stays flat, exploded and rebuilt at this module's serialization boundary. Its membership *is* the pure-analytics class, and the admission rule (nothing may branch on or render a member) lives on its docstring — `query_source`, `played_at` and the `np_*` trio all look eligible and are not |
| `GuildStateData` / `NowPlayingData` / `GuildPlaybackSnapshot` | `guild_state.py` | Typed snapshots of the state hash, now-playing hash, and the full restore read |
| `HistoryEntry` | `guild_state.py` | One played song (title, url, durations, requester, `guild_id`, `played_at`, `message_id`, `channel_id`, `queued_at`, `queue_position`, `query_source`) — the wire format shared by the Redis display list, the outbox, and the Postgres row mapping |
| `GuildRedisStore` | `redis_client.py` | Per-guild Redis operations namespace |

---

## Commands Reference

All commands use the prefix `-`. `strip_after_prefix=True` so `-play`, `- play`, and `-  play` all work.

Every command also accepts a `--help` flag anywhere in its message: `MusicBotApp.invoke()` short-circuits to that command's help embed **before** any check, the `validate_commands` voice-channel gate, or argument parsing runs — so `-play --help` answers from outside a voice channel instead of searching YouTube for the string `"--help"`. It is equivalent to `-help <command>`.

| Command | Aliases | Arguments | Description |
|---|---|---|---|
| `-play` | `p`, `sing` | `url` | Enqueue a YouTube URL / search / **YouTube playlist**, Spotify track/playlist, or SoundCloud URL. Joins voice first if not connected. |
| `-playnow` | `pn` | `url` | Interject a song **immediately**, parking the current one to resume from its exact position afterward. Falls back to `-play` when nothing is live. Playlists interject only their first track. See [-playnow Interjection](#-playnow-interjection). |
| `-skip` | `sk` | — | Stop the current song and advance to the next. |
| `-stop` | `st` | — | Stop playback, disconnect from voice, and clean up the player. |
| `-pause` | `po` | — | Pause playback. Adds ⏸️ and sends a confirmation embed showing the frozen position. |
| `-resume` | `r` | — | Resume paused playback; re-hosts the Now Playing block so the pause confirmation becomes plain history. |
| `-join` | `summon` | — | Join the user's voice channel (`connect(timeout=10.0)`). Saves channel IDs to Redis. |
| `-shuffle` | — | — | Shuffle all songs currently in the queue (requires 3+ songs). |
| `-clear` | `c` | — | Empty the queue and its mirror, reporting the removed songs (or "already empty"). |
| `-remove` | `rm` | `<link or search text>` | Remove **all** queued songs matching, by the resolved yt-dlp URL **or** by what the user originally typed — a search term or a source link, so one playlist link takes back out every track it added. Reports the removed positions and names which of the two matched. Consume-rest, so a multi-word search works. Without an argument, prints usage. |
| `-now` | `np`, `rn`, `nowplaying` | — | Display the now-playing embed, rebuilt live for the current song. |
| `-queue` | `q` | — | Display the next 10 songs with per-song ETA. |
| `-history` | `h` | `[--limit N]` | Display the last N played songs (default 10, max 50). Served from the capped Redis list alone, in both archive modes — see [History read path](#history-read-path). |
| `-leaderboard` | `lb`, `top` | `[--days N]` | Top 10 listeners and top 10 songs for this server, ranked by total listening time; `--days` scopes both boards to a rolling window. Aggregated from the Postgres archive (the first production reader of it) behind a 60 s Redis cache; replies with a notice when the archive is disabled. |
| `-volume` | `v`, `vol`, `sound` | `0–100` | Set playback volume (takes effect on next song). Persisted to Redis. |
| `-ping` | `latency`, `l`, `delay`, `health`, `status` | — | Live-editing service-health dashboard: probes Discord, Redis, Spotify, the Postgres archive and the OTLP endpoint, and reports the bot / yt-dlp / FFmpeg versions. One in flight per guild. |
| `-debug` | `dbg` | `[--enable \| --disable]` | Live-editing diagnostic snapshot: what is running and how it is configured, against `-ping`'s "are my dependencies up?". Public blocks are versions and this server's player/voice state; build, configuration, runtime, storage and health checks are **bot-owner only**. `--enable`/`--disable` toggle per-guild debug mode (adds a trace/timing/runtime footer to every embed the bot sends in that guild, the live Now Playing card included) and require **Manage Server**. The choice persists to `guild:{id}:config` and outlives restarts; a guild that has never set one follows the host's `DEBUG_MODE`. Observation-only, and exempt from `cog_before_invoke`'s `get_mp()` for that reason. One in flight per guild. |
| `-jump` | `j` | — | Stub; replies "currently in development". |
| `-help` | `commands` | `[command]` | Man-page-styled embed help: the full command list, or detailed help for one command (`-help play`). Aliases resolve too (`-help np`). Rendered by `MusicHelpCommand` (`help.py`). |

**Permission model:**

Every command that touches playback is gated by `@commands.before_invoke(validate_commands)`. The read-only ones are not: `-ping` and `-leaderboard` answer without the author being in voice (and `-help` is a `HelpCommand`, not a cog command at all). `cog_before_invoke` runs first for all of them: it binds structlog contextvars (`guild_id`, `user_id`, `command`), opens a `command.{name}` OTel span (closed in `cog_after_invoke`), creates the guild's `MusicPlayer` if needed, and refreshes the persisted `(voice_channel_id, text_channel_id)` pair when the command channel changed. A command carrying `extras={"observation_only": True}` — today only `-debug` — returns after the span and the contextvars and skips both the `get_mp()` and the channel-persistence steps: a command that reports on a guild's player must not manufacture one to look at, and creating it would also start a restore and a 300 s gate timeout on an idle guild. `validate_commands` then checks:
1. The author is a `discord.Member` (not a `discord.User`)
2. The author is in a voice channel
3. For non-`play` commands: the bot is in the same voice channel as the author

`-help` (and the `--help` flag on any command) is exempt from the voice-channel gate: the help command carries no `before_invoke(validate_commands)`, and `--help` short-circuits in `invoke()` ahead of it — so help is always reachable, even from outside a voice channel. `-playnow` **is** gated like `-play` (it may join voice first).

**Supported `-play` inputs:**

| Input format | Example | Resolution path |
|---|---|---|
| YouTube watch URL | `https://youtube.com/watch?v=...` | `YTSource(process=False)` → `yt_source` (unified full extraction — the `process` field is parse metadata only) |
| YouTube short URL | `https://youtu.be/...` | `YTSource(process=False)` |
| YouTube URL with timestamp | `?t=120` | `YTSource(ts=120)` → seeks via FFmpeg `-ss` |
| YouTube playlist URL | `.../playlist?list=...`, or any `watch?v=…&list=…` | `YTSource(type=PLAYLIST, list_id=...)` → `YTDL.yt_playlist` (flat extraction) → N `QueueObject`s. `_YTDL_PLAYLIST_OPTS` uses `extract_flat="in_playlist"`, not `True`: a watch URL resolves to a `url_result` pointing at the playlist, and `True` stops at it with no entries |
| …carrying `&index=N` | `watch?v=…&list=…&index=4` | 1-based start position — `_apply_playlist_index` drops the N−1 tracks ahead of it. N past the end raises `PlaylistIndexError`, whose `user_message` names both the requested index and the real length (rendered by `_command_error`, like the yt-dlp user-facing errors) rather than enqueueing nothing. `-playnow` interjects that track instead of the first. A `t=` on the same link applies to the queued head only when it is the `v=` video (`_apply_playlist_timestamp`), since one offset cannot belong to N tracks |
| YouTube search string | `never gonna give you up` | `YTSource(ytsearch="ytsearch:...", process=True)` |
| Spotify track URL | `https://open.spotify.com/track/...` | `SpotifySource(TRACK)` → `Spotify.track()` → YouTube search |
| Spotify playlist URL | `https://open.spotify.com/playlist/...` | `SpotifySource(PLAYLIST)` → `Spotify.playlist()` → N `YTSource` search items |
| SoundCloud URL | `https://soundcloud.com/...` | `SoundcloudSource` → yt-dlp directly |

---

## Configuration

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `DISCORD_TOKEN` | Yes | Bot token from Discord Developer Portal |
| `SPOTIFY_CLIENT_ID` | Yes | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | Yes | Spotify app client secret |
| `REDIS_URL` | No | Redis connection URL (defaults to `redis://localhost:6379`) |
| `POSTGRES_URL` | No | Durable-tier DSN (e.g. `postgresql://musicbot:musicbot@127.0.0.1:5432/musicbot`). Unset → the entire Postgres tier is off (no outbox writes, no drainer, pre-Postgres read behavior) |
| `ENVIRONMENT` | No | Deployment environment label; defaults to git-branch-derived (`main` → `production`, else the branch slug). Stamped on the OTel resource. |
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | No | Progress-bar edit cadence (default `3.0`; sized against Discord's ~5 edits/5 s per-channel bucket) |
| `OTEL_SERVICE_NAME` | No | OTel service name (default `discord-music-bot`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTLP gRPC endpoint (default `http://localhost:4317`) |
| `OTEL_SDK_DISABLED` | No | `true` disables telemetry entirely (tests set this) |

**Bot configuration** (`main.py`):

| Setting | Value | Notes |
|---|---|---|
| `command_prefix` | `"-"` | Single-hyphen prefix |
| `strip_after_prefix` | `True` | Ignores spaces after prefix |
| `intents` | `discord.Intents.all()` | All intents enabled |
| Bot class | `AutoShardedBot` | Multi-shard within one process; Discord requires sharding at 2500 guilds |
| Context class | `MusicContext` | Installed via `get_context` override — the NP-block attach point |

**yt-dlp option profiles** (`youtube.py`) — three profiles share `_YTDL_BASE_OPTS` (`quiet`, `no_warnings`, `noplaylist`, `nocheckcertificate`, `source_address 0.0.0.0`, `socket_timeout 30`, `extractor_args: player_client ["default", "-tv_simply"]`):

| Profile | Used by | Deltas from base |
|---|---|---|
| `_YTDL_STREAM_OPTS` | `yt_stream` / `prefetch_stream` | `format: bestaudio/best[height<=360]/best`, `check_formats: False` (skips HEAD probes), `retries: 10` |
| `_YTDL_STREAM_SEARCH_OPTS` | `yt_source` (unified single extraction) | stream opts + `default_search: auto` — one call yields identity **and** stream URL, populating both caches |
| `_YTDL_PLAYLIST_OPTS` | `yt_playlist` | `noplaylist: False`, `extract_flat: True` — enumerates entries without per-video extraction |

`rm_cachedir` is deliberately **absent**: yt-dlp's JS-player cache is kept across calls so the signature-decryption JS is only re-fetched when YouTube publishes a new player version. A fresh `YoutubeDL` instance is constructed per extraction call (`_ytdlp_extract`), and it is handed a **shallow copy** of the opts profile — `YoutubeDL.__init__` keeps the params dict by reference and writes into it (`js_runtimes`, `http_headers`, ...), so passing the module-level dicts directly would make them shared mutable state across repeated extractions within a worker process.

**Extraction pool** (`ytdlp_pool.py`, instantiated in `youtube.py`):

```python
ytdlp_pool = YtdlpPool()   # max_workers from YTDLP_POOL_WORKERS, default 4
```

Extraction runs in **worker processes**, not threads: JSON parsing, signature decryption and format selection are GIL-bound Python, so threads would contend for the GIL and steal time from the event loop serving voice heartbeats.

`YtdlpPool` owns the lifecycle and nothing else — the callable is passed per call (`ytdlp_pool.run(_ytdlp_extract, ...)`), which is what keeps it free of yt-dlp knowledge and keeps `patch("src.youtube._ytdlp_extract")` working in tests. Lifecycle: created lazily (so importing the module never spawns children), prewarmed from `setup_hook`, rebuilt once if a worker dies (`BrokenProcessPool`), and closed from `close()` via `aclose()`, which bounds the join at 10 s. A submit after close raises `PoolClosedError` rather than silently spawning a pool nothing would join.

Each worker holds a full CPython + yt-dlp import (~80–120 MB RSS), so the default worker count is deliberately conservative.

---

## System Flows

### Startup and Initialization

```mermaid
sequenceDiagram
    participant OS as OS / Docker
    participant Main as main.py
    participant Tel as telemetry.py
    participant Bot as MusicBotApp
    participant Redis as Redis
    participant Discord as Discord Gateway

    OS->>Main: python -m src.main
    Main->>Tel: setup_telemetry() — must run before any get_logger()
    Main->>Main: assert DISCORD_TOKEN / SPOTIFY_* present
    Main->>Bot: bot.run(token) → connect WebSocket
    Discord-->>Bot: setup_hook (before READY)
    Bot->>Redis: create_redis_pool() + get_redis()
    Bot->>Bot: HISTORY_ARCHIVE_ENABLED? → archive →<br/>HistoryOutboxDrainer.start() (lazy — no PG connection yet)
    Bot->>Bot: load_extension("src.musicbot")
    Discord-->>Bot: on_ready (all guilds cached)
    Bot->>Bot: presence = "Playing music"
    Bot->>Bot: MusicBot.on_ready → create_task(_restore_guild) × N guilds
```

`setup_hook` runs after the WebSocket connection is established but before READY is dispatched — Redis is guaranteed available before any command is processed. `setup_telemetry()` runs first in `main()` because it configures structlog before the first `get_logger()` call resolves.

---

### Voice Channel Join

```mermaid
sequenceDiagram
    participant User
    participant Bot as MusicBot (Cog)
    participant MP as MusicPlayer
    participant Redis as GuildRedisStore
    participant VC as Discord VoiceClient

    User->>Bot: -join
    Bot->>Bot: validate_commands()
    Bot->>VC: channel.connect(timeout=10.0)
    VC-->>Bot: VoiceClient connected
    Bot->>VC: guild.change_voice_state(self_deaf=True)
    Bot->>MP: get_mp(ctx) → MusicPlayer.from_context()
    MP->>MP: start() → create_task(_restore_state()), create_task(loop())
    Bot->>Redis: store.set_connection(voice_channel_id, text_channel_id)
    Bot->>User: 👋 reaction + ping embed
```

`self_deaf=True` is always set — the bot does not listen to voice, only transmit. `MusicPlayer.start()` sets `_restore_complete` immediately when there is no Redis store; otherwise `_restore_state()` sets it when done, and `loop()` blocks on it before its first dequeue.

---

### -play Command Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Bot as MusicBot
    participant Sources as sources.py
    participant SP as Spotify
    participant YTDL as YTDL
    participant MP as MusicPlayer
    participant Redis as GuildRedisStore
    participant PF as prefetch_stream (bg task)

    User->>Bot: -play <url or search>
    Bot->>Sources: parse_input(url)
    Sources-->>Bot: YTSource | SpotifySource | SoundcloudSource

    alt Spotify playlist
        Bot->>SP: spotify.playlist(id) → List[str] titles
        Bot->>Sources: spotify_playlist_to_ytsearch(titles) → List[YTSource]
        Bot->>MP: queue_put(items, prefetch=False)
    else YouTube playlist
        Bot->>YTDL: yt_playlist(playlist_url, author) → List[QueueObject]
        Bot->>MP: queue_put(tracks, prefetch=False)
    else Single track (YouTube / Spotify track / SoundCloud)
        Bot->>SP: spotify.track(id)  [if Spotify track]
        Bot->>YTDL: yt_source(requester, search, ts, redis)
        Note over YTDL: checks ytdl:source cache first (1h TTL);<br/>on miss, one unified extraction<br/>writes BOTH ytdl:source and ytdl:stream
        YTDL-->>Redis: cache_set("ytdl:stream:<url>", data, ≤1800s)
        YTDL-->>Bot: QueueObject(webpage_url, title, ...)
        Bot->>MP: queue_put([qobj])
        MP->>PF: create_task(YTDL.prefetch_stream(qobj, redis))
        Note over PF: cache-hit no-op for unified-path songs;<br/>full extraction only for sparse items<br/>(playlist entries, requeues)
    end

    MP->>Redis: RPUSH typed queue entries (SongQueueEntry / SearchQueueEntry)
    Bot->>User: 👍 reaction + queued embed (via MusicContext — NP block leads)
```

Playlists (both kinds) enqueue with `prefetch=False` so a 100-track playlist doesn't saturate the extraction pool at enqueue time.

---

### -playnow Interjection

`-playnow` (`pn`) plays a song **immediately**, parking the current one to resume from its exact position afterward. Full design: [PLAYNOW_PROPOSAL.md](PLAYNOW_PROPOSAL.md).

```mermaid
flowchart TD
    Start(["-playnow url"])
    Live{"a song live?\n(current_song + vc playing/paused)"}
    Fallback["fall back to -play\n(joins if needed; playlists enqueue in full)"]
    Resolve["parse_input + _resolve_playnow_source\n(playlist → first track only)\nqobj.interjected = True"]
    Warm["await YTDL.prefetch_stream(qobj)\nwarm stream cache BEFORE interrupting\n(avoids dead air; back-fills embed metadata)"]
    Interject["mp.interject(qobj, vc)"]
    Ended{"song ended\nmid-resolve?"}
    FrontOnly["queue.put_front([qobj])\n(interjected reset)\n'Playing next' notice"]
    Resume["build resume SongQueueEntry\n(is_resume=True, ts=position_secs,\nstart_paused=was_paused)\ninherits played_at + query_source"]
    PutFront["queue.put_front([qobj, resume?])\n_skip_history_for = stopped song\nvc.stop() → loop picks up qobj"]

    Start --> Live
    Live -->|No| Fallback
    Live -->|Yes| Resolve --> Warm --> Interject
    Interject --> Ended
    Ended -->|Yes| FrontOnly
    Ended -->|No| Resume --> PutFront
```

Key properties:

- **Resume fidelity**: the parked song's position comes from `position_secs` (the frame counter), stored as `ts` on an `is_resume` `SongQueueEntry`. When the loop dequeues that entry it seeks via FFmpeg `-ss ts` and, if `start_paused`, comes back paused — the interruption is invisible to playback position.
- **Warm before interrupt**: `prefetch_stream` is **awaited** (not fire-and-forget like `queue_put`'s warm-up) so the current song keeps playing through a possible yt-dlp miss rather than cutting to silence before the playnow song is ready.
- **Nearly-finished guard**: a song with almost no time left gets no resume entry (`resume_position is None`) — it just ends.
- **Stacking**: interjecting on top of another `-playnow` song parks it like any other, in front of the tails already waiting, so the queue unwinds LIFO and every parked song returns. Depth is unbounded and recorded on the span as `interject.depth` (the run of consecutive `is_resume` entries from display index 1, i.e. parked *plays*, via `GuildQueue.resume_tail_depth`). `ts` is absolute at every level, so a tail of a tail resumes at the position actually reached rather than at its own fragment's start.
- **History once**: `_skip_history_for` holds the parked song's identity so the stop-transition's history step skips it — it is recorded exactly once, when its resume tail finishes. It holds the song object (not a bare flag) because the song can end naturally during `interject()`'s awaits. The same marker is what lets a *teardown* record safely: `cog.cleanup` claims the mid-play song through `MusicPlayer.claim_current_song_for_history()`, which declines when the marker already names it (a parked tail will record the play on `-resume`) and otherwise takes the marker so the loop cannot record it twice.
- **Crash-safe**: resume entries are ordinary persisted `SongQueueEntry`s (LPUSHed to the front of the Redis list), so a crash mid-interjection recovers the parked song from the queue like any other.

---

### Source Resolution

`parse_input` / `parse_url` in `sources.py` classify the raw input string:

```mermaid
flowchart TD
    Input["Raw input string"]
    IsYT{"youtube.com\nor youtu.be?"}
    IsPL{"playlist\n(list= param)?"}
    IsSP{"open.spotify.com\nor spotify.com?"}
    IsSC{"soundcloud.com?"}

    Input --> IsYT
    IsYT -->|Yes| IsPL
    IsPL -->|Yes| YTPL["YTSource(type=PLAYLIST, list_id=...)"]
    IsPL -->|"No (?t= honored)"| YTS["YTSource(url, ts?, process=False)"]

    IsYT -->|No| IsSP
    IsSP -->|track| SPS_T["SpotifySource(TRACK, id)"]
    IsSP -->|playlist| SPS_P["SpotifySource(PLAYLIST, id)"]
    IsSP -->|No| IsSC
    IsSC -->|Yes| SC["SoundcloudSource(url)"]
    IsSC -->|No| YTS_S["YTSource(ytsearch='ytsearch:...', process=True)"]
```

Spotify sources are converted to YouTube searches before any audio work:
- **Track**: `Spotify.track(id)` → `"Title Artist"` → `YTSource(ytsearch=..., process=True)`
- **Playlist**: `Spotify.playlist(id)` → `List[str]` titles → `spotify_playlist_to_ytsearch()` wraps each as a `YTSource`

---

### yt-dlp Pipeline

Every song passes through up to three phases:

```mermaid
flowchart LR
    subgraph Phase1["Phase 1 — Enqueue (unified single extraction)"]
        P1["yt_source(search)\nytdl:source cache (1h) or ONE\nyt-dlp full extraction (process=True)\npopulating ytdl:source + ytdl:stream\nReturns: QueueObject"]
    end

    subgraph Phase1b["Phase 1b — Eager Prefetch (background)"]
        P1b["prefetch_stream(QueueObject)\ncache-hit no-op after Phase 1;\nfull extraction only for bare\nQueueObjects (playlists, requeues)\n+ enriches QueueObject metadata"]
    end

    subgraph Phase2["Phase 2 — Play (low latency)"]
        P2["yt_stream(QueueObject)\nRedis cache hit → no yt-dlp call\nConstructs YTDL(FFmpegOpusAudio)\nFFmpeg starts reading stream"]
    end

    Phase1 -->|"asyncio.create_task"| Phase1b
    Phase1 -->|"queue.put(QueueObject)"| Phase2
    Phase1b -->|"Redis cache populated"| Phase2
```

**Phase 1** (`YTDL.yt_source`): Checks the `ytdl:source:{normalized query}` Redis cache (TTL 1 h) before running yt-dlp — repeat plays of the same input skip the 3–4 s lookup. On a miss, **one** full extraction with `_YTDL_STREAM_SEARCH_OPTS` and hardcoded `process=True` (searches *and* direct URLs — unprocessed extraction would do no format selection and leave nothing to cache) yields identity plus a selected stream URL, and `_probe_and_cache` writes the `ytdl:stream` entry alongside the `ytdl:source` one. A failed probe skips only the stream write — the song still enqueues on identity. Returns a `QueueObject`.

**Phase 1b** (`YTDL.prefetch_stream`): Fire-and-forget task spawned by `queue_put` (single tracks only). For songs Phase 1 just resolved it is a cache-hit no-op (one Redis GET); it runs a full extraction only for bare `QueueObject`s that skipped the unified path (playlist entries, requeues). On extraction it strips the yt-dlp payload to `_STREAM_CACHE_FIELDS` (16 fields) before caching (via the shared `_probe_and_cache`), and back-fills the live `QueueObject`'s `duration`/`uploader`/`thumbnail` via `_enrich_queueobject` so queue embeds/ETA improve as prefetches land. Errors are logged and swallowed — Phase 2 recovers by extracting fresh.

**Phase 2** (`YTDL.yt_stream`): Called just before playback. Cache hit → construct `YTDL` with no yt-dlp call; miss → extract and cache.

**Stream URL properties:**
- YouTube CDN URLs have a 6-hour expiry window and are IP-bound (the `ip` field is inside the HMAC-signed `sparams`) — they cannot be reused from a different host
- Cache TTL formula: `min(expire − now − 1800s, _STREAM_URL_MAX_TTL=1800s)`; not written if the result is under 60 s. YouTube revokes URLs well before their advertised `expire`, so the 1800 s cap — not `expire` — is what bounds a fresh extraction's TTL in practice

---

### Playback Loop

`MusicPlayer.loop()` runs as a long-lived asyncio task. It first awaits `bot.wait_until_ready()` and `_restore_complete` (so crash-restore finishes populating the queue before the first dequeue). Each iteration is wrapped in a `player.loop.iteration` span and carries a `claim_outstanding` flag so the exception handler can release a claim no other path settled.

```mermaid
flowchart TD
    Start(["iteration start\nplay_next.clear()"])
    Cleared{"queue cleared while\nprefetch ran?"}
    DropPF["cursor already reset\nprefetched_song.cleanup()"]
    HavePF{"prefetched_song\navailable?"}
    UsePF["current_song = prefetched_song\n(its claim becomes ours)"]
    GetQueue["queue_get() — 300s timeout"]
    Timeout["TimeoutError → stop()"]
    Resolve["_resolve_source()\nYTSource → QueueObject"]
    Stream["_stream_source() → YTDL"]
    Failed{"YTDL is None?"}
    FailPop["queue.finish_failed_dequeue()\nsend 'Failed…' via send_with_np"]
    Commit{"queue.try_commit_dequeue()?"}
    Discard["cleared mid-resolve:\ncommit refused + song.cleanup()"]
    Play["vc.play(song, after=play_next.set\nvia call_soon_threadsafe)"]
    Persist["Redis MULTI/EXEC:\npop_queue_and_start_song(entry,\nbackdated play_start, now_playing)\n(or set_current_song_state for\ncrash-recovered song)"]
    NP["update_activity(song)\n_send_now_playing(song)\n→ progress task starts"]
    Prefetch["create_task(_prefetch_next_song())"]
    Wait["await play_next.wait()"]
    Retire["cancel progress + pause-debounce\nrelease NP host, fire final\n'bar complete' edit"]
    Collect["prefetched_song = await _prefetch_task"]
    History["history.add(HistoryEntry)\n→ deque + Redis list + outbox (one pipeline)\n+ drainer notify; store.clear_song_end_state()"]
    Done["claim settled at start\ncurrent_song = None\nupdate_activity(None)"]

    Start --> Cleared
    Cleared -->|Yes| DropPF --> HavePF
    Cleared -->|No| HavePF
    HavePF -->|Yes| UsePF --> Failed
    HavePF -->|No| GetQueue
    GetQueue -->|timeout| Timeout
    GetQueue --> Resolve --> Stream --> Failed
    Failed -->|Yes| FailPop --> Start
    Failed -->|No| Commit
    Commit -->|No| Discard --> Start
    Commit -->|Yes| Play --> Persist --> NP --> Prefetch --> Wait
    Wait --> Retire --> Collect --> History --> Done --> Start
```

Key details:

- **Atomic start transaction**: for a real queue item, `pop_queue_and_start_song` LPOPs the Redis queue and writes all current-song state fields plus the `now_playing` display snapshot in one `MULTI/EXEC` — there is no window where the song is neither on the queue list nor in the state hash. A crash-recovered song (`persisted=False`) was never on the Redis list, so only the state fields are written (an LPOP would drop an unrelated queued song).
- **Backdated epoch**: `play_start_epoch` is stored as `play_start − song.start_offset` so recovery position math (`now − play_start_epoch − pauses`) yields the true audio position for `?t=` songs and double-crash recoveries.
- **`_prefetch_next_song`** dequeues via `queue.get_nowait()`, resolves + streams the next item while the current song plays. If cancelled (clear/shuffle/remove), it returns the item to the front via `queue.requeue_front()` — the claim goes back with it. If resolve/stream fails, it settles the claim and mirrors it via `queue.finish_failed_dequeue()`.
- **Every claim is settled exactly once** — by `try_commit_dequeue()` (the song starts), `finish_failed_dequeue()` (failure), or `requeue_front()` (cancellation, which returns the claim with the item). The loop's exception handler releases a claim no other path settled, tracked by `claim_outstanding`; a claim left standing would keep its item counted as in flight forever and the next release would settle a different song.
- **Resume entries**: an `is_resume` `SongQueueEntry` (from `-playnow`) replays through the same FFmpeg `-ss ts` seek path as a `?t=` song and honours `start_paused`. The parked song's history add is deferred via `_skip_history_for` so it is recorded once, at its resume tail — see [-playnow Interjection](#-playnow-interjection).

---

### Queue Operations

All queue state lives behind `GuildQueue` (`guild_queue.py`). One deque and an index into it, both **private to the class** — the invariant is structural, not call-site discipline:

| | Holds | |
|---|---|---|
| `_items[:_cursor]` | claimed by a consumer, not yet settled | the "in-flight head" |
| `_items[_cursor:]` | pending | what `get()` hands out |
| `_wake` | `asyncio.Event`, set iff something is pending | maintained only by `_sync_wake()` |
| Redis `guild:{id}:queue` | JSON `SongQueueEntry`/`SearchQueueEntry` | the `is_persisted()` subset, in order |

**The cursor is the boundary, not a per-item flag** — because Redis retires entries by LPOP, so in-flight items are necessarily a *prefix*. This replaced an `asyncio.Queue` plus a parallel `deque` (`_pending` / `_display`) whose agreement was maintained by hand across thirteen mutating methods; the second representation was derivable from the first, and the migration proved it with a runtime assertion before deleting it.

**`get()` waits on a `while`, never an `if`.** Dropping `asyncio.Queue` dropped its cancellation-recovery block — the one that hands a wakeup on when a woken getter is cancelled before claiming. Re-testing the condition replaces it, and covers what that block never did: `Event` wakes *every* waiter, and the prefetch's `get_nowait()` is a real second consumer, so one item can wake two claimants and the loser must find the queue empty again rather than index past the end.

**`_sync_wake()` is the sole writer of `_wake`.** A stale set does not degrade — `Event.wait()` returns without yielding when already set, so `get()`'s wait loop loses its suspension point and the single event loop serving playback, the gateway and every guild stops. Measured at 2 000 001 iterations with 0 other loop ticks.

Every mutation that touches the Redis mirror (put, clear, shuffle, remove, `finish_failed_dequeue`) runs under one bulk-mutation mutex. They rebuild from `_items` and leave `_cursor` alone, so the claimed prefix survives a shuffle/remove during a multi-second resolve and the commit retires the entry it claimed.

**Settling a claim asks the claim, not the item.** `redis_pop_for(item, *, persisted=None)` defaults to deriving the answer from `item`, and `item=None` defaults to popping — right for every ordinary dequeue and wrong for exactly one caller. The playback loop's prefetched branch holds a claim whose item is a `YTDL`, which is not a `QueueItem` and cannot be passed; a prefetch really can hold an *unpersisted* claim (a cold-start `-play` front-inserts at cursor 0, ahead of the crash-recovered head, so the prefetch behind it takes that head). So the loop carries `claim_persisted` from the moment it takes the claim through to its outer error handler, and both settle paths — the start transaction's LPOP and `finish_failed_dequeue`'s — read that one flag rather than re-deriving it. Re-deriving is what LPOPed a real entry for a head that never had one, deleting the next queued song with no error and no log line.

`put`/`put_front` return the list they enqueued, which the caller uses to spawn per-item prefetch. They no longer patch what passes through: queue objects arrive complete. `queue_position` is depth **at ask** — `MusicPlayer.enqueue_depth()` read once at command dispatch, alongside the `queued_at` taken from the command message — rather than depth at insert computed under this mutex. It is approximate against the insert by design: the playback loop dequeues continuously, so the two differ routinely with no user involvement, and the quantity the field proxies for is stored exactly beside it as `played_at − queued_at`. `enqueue_depth()` reads `display_size()`, never `qsize()`: a claimed song is gone from the pending count and still ahead of a new arrival, so `qsize()` would undercount by one exactly when a `-play` lands during another song's resolve. The two are now `len(_items)` and `len(_items) - _cursor` — one term apart over the same fields, which is why five tests pin them apart and each half of the swap fails a different subset.

`MusicPlayer`'s thin wrappers (`queue_clear`/`queue_shuffle`/`queue_remove`) call `_cancel_prefetch()` **before** delegating — a still-running prefetch holds an item from `get_nowait()`, and cancellation returns it via `requeue_front()` so the bulk mutation processes it with everything else.

- **Shuffle**: islices the pending tail under the mutex, `random.shuffle`, re-enqueues, rebuilds the mirror. Returns a `ShuffleOutcome` enum.
- **Clear**: empties the deque, resets the cursor, sets the cleared-flag the loop consumes (`consume_cleared_flag()`), returns the removed titles for the report embed.
- **Remove**: takes a **predicate** (`RemoveMatcher`), not a URL, and returns a `RemoveOutcome` — the removed items, their 1-based positions, and the `RemoveMode` that matched. The items are in it because a removed entry can be the last record of a song that already played (`MusicPlayer._flush_played`). The policy lives in `remove_matcher()` beside the class rather than inside it, so it is testable without a queue.

**One writer for the mirror.** `_write_mirror(items, *, removed=())` owns the rebuild-vs-delete-vs-LREM choice for all three bulk mutations; before, each of them inlined its own copy and they disagreed about the empty case. Empty means `DELETE`, not skip — a queue whose every persisted entry just went would otherwise leave the old list behind for the next restore to find. The rebuild is `MULTI/EXEC` (`DEL` + `RPUSH` atomically, so a concurrent LPOP never sees an empty-key window).

`removed` is the LREM shortcut and **only a removal may pass it**: LREM asserts the survivors kept their order, which is false for a shuffle and for any insert.

**The crossover is a COUNT, and `_LREM_MAX_ENTRIES` (16) is the bound that matters.** `LREM key 1 <blob>` scans from the head and stops at its first match, so it is `O(position)` — not `O(1)`. N of them cost `O(N × depth)`, and a rebuild costs `O(survivors)`, which is also `O(depth)`. **The depth term cancels**, so at a fixed N the two curves keep the same order at every depth, and no ratio enters the comparison.

An earlier revision of this section read that backwards — "a ratio, not a count", gating on `drop ≤ survivors / 5`, which has no depth term at all. That admitted N=200 whenever the queue held 1000 and kept admitting it as the queue grew: at depth 1200 dropping 200 it picks LREM at 10.0 ms against a 6.3 ms rebuild. Two independent measurements put the true crossover at 18–50 (`redis:7-alpine`) and 50–150 (native `redis 8.10`), so the cap sits below both rather than splitting them. At 16 the LREM path costs **less than the rebuild it replaces at every depth measured** — 250: 1.1 ms vs 1.8; 1000: 1.7 vs 6.2; 5000: 4.4 vs 31.7; 20000: 15.4 vs 131.2.

That bound is the point rather than a micro-optimisation: the LREMs run inside **one `MULTI/EXEC`**, so single-threaded Redis serves *nobody* for their duration — every other guild's `pop_queue_and_start_song`, every `-history` read, the outbox drain. Staying under the rebuild caps that stall at what the alternative already costs. `_LREM_MAX_SHARE` remains as a shallow-queue guard: below ~80 survivors a full rewrite is under a millisecond, so there is nothing for the shortcut to win. A test pins the cap's *value* against the measurement (`_LREM_MAX_ENTRIES ≤ 18`), because the tests around it size their input from the constant and so move with it.

The shortcut is guarded twice more, because LREM matches on **exact serialized bytes** and nothing else in the codebase promises them. It is skipped outright when a removed blob is byte-identical to a **claimed** item's — LREM takes the head-most equal element, which would be the entry awaiting its commit-time LPOP — and it falls back to the rebuild whenever `remove_queue_entries` returns fewer than it was asked for. That short count is what a queued object mutated after its entry was written looks like (a resume tail gaining `np_*` ids, an enriched duration, a substituted requester), and it is also what a swallowed Redis error or an evicted key returns. The rebuild cannot be wrong by construction: it restates the whole list from memory.

Counted per distinct serialization, never `LREM … 0`: two enqueues of one song usually differ on the wire (`queue_position`, `queued_at`), but when they do not, removing "all matching" would take out a copy still queued.

**Known residual window (by design)**: the loop's `try_commit_dequeue()` → `pop_queue_and_start_song()` handoff releases the mutex before the store's atomic transaction dispatches; a bulk mutation scheduled in that single event-loop tick can race the LPOP server-side. The start transaction is a store-level atomicity boundary — see the `guild_queue.py` module docstring.

---

### Now Playing Host Model

While a song is live, the **Now Playing embed block** (`[now_playing, next_up?]`, built by `np_embed_block()`) lives in exactly one **host message** — the newest bot message in the player's channel — so the live progress bar is always at the bottom. Full design: [NOW_PLAYING_EMBED_ATTACH_PLAN.md](NOW_PLAYING_EMBED_ATTACH_PLAN.md).

Mechanics:

- **`MusicContext.send()`** (installed bot-wide via `get_context`): every command response in the player's channel, while a song is live, is sent as `NP block + response's own embeds` in **one message** (atomic — the bar is never even momentarily buried). After the send, `_adopt_np_host_if_current(message, own, song)` makes that message the new host and retires the old one. The adopt is gated on the song still being current — the send's `await` can cross a song boundary, and adopting a stale block would delete the next song's fresh host (the gate sheds the stale block from the just-sent message instead).
- **Retiring the old host**: a *dedicated* NP message (sent by `_send_now_playing` with nothing else) is deleted; a *command-response* host is strip-edited back to its own embeds. All mutations of an old host (progress-tick edits, retires) go through `_np_edit_lock` so a strip/delete is always the final write.
- **Pointer-first, synchronous adoption** (`_adopt_np_host`): the host pointer swap happens atomically on the event loop before any awaits, so no progress tick can edit a message that is about to be retired.
- **`send_with_np()`**: for bot-initiated messages (loop errors, alone-countdown notice) — same attach behavior outside a command context. **Never** send to the player's channel with a bare `channel.send()` while a song is live.
- **Song end**: the loop releases the host (the finished bar stays behind as a historical record) and fires one final edit so the bar renders fully complete instead of frozen at the last tick.
- **Stop/cleanup**: `retire_np_host_on_stop()` disposes of the host after all tasks are cancelled.
- Discord's 10-embed cap is checked defensively at attach time (worst case here is 3).

**Progress bar**: `_progress_updater` edits the host's NP embed every `NOW_PLAYING_UPDATE_INTERVAL_SECS` (default 3 s). Position comes from the audio itself: `YTDL.read()` counts frames (`elapsed_secs = frames × 20 ms`), and `position_secs = start_offset + elapsed_secs`. Because discord.py's `AudioPlayer` simply doesn't call `read()` while paused, the counter freezes automatically for explicit pauses **and** involuntary stalls (voice reconnects) with zero bookkeeping. `position_secs` is the single source of truth for every position surface (bar, presence tooltip, pause confirmation).

**Presence**: `update_activity(song)` sets a "Listening to *title · uploader*" activity with `timestamps` derived from `position_secs` (backdated `start`, computed `end`). While paused, `timestamps` is empty — Discord's Activity schema has no "frozen" representation. On song end it resets to "Playing music", but only when **no other guild** is still playing.

---

### Pause / Resume

`-pause` / `-resume` funnel through single entry points on `MusicPlayer` so no call site can forget a side effect:

```
pause(vc):  vc.pause()  → store.on_pause(now)        → mark_paused()
resume(vc): vc.resume() → store.on_resume(now)       → mark_resumed()
```

- **Redis epoch accounting** (crash-recovery correctness): `on_pause` writes `pause_start_epoch`; `on_resume` folds the pause interval into `total_pause_seconds` and clears `pause_start_epoch`. Recovery position = `now − play_start_epoch − total_pause_seconds` (still-open pauses handled at read time).
- **`mark_paused`/`mark_resumed`** both fire `_fire_pause_state_updates()` — a debounced one-off NP-embed edit + presence refresh, so the bar and tooltip freeze/unfreeze promptly rather than waiting for the next 3 s tick.
- `-pause` replies with a **confirmation embed** (`build_pause_confirmation_embed`) showing the frozen position; `-resume` calls `rehost_np_after_resume()` so a pause confirmation hosting the block becomes plain history rather than sitting beneath a live, advancing bar.

---

### Auto-Disconnect

Two independent triggers, both handled in `musicbot.py`:

1. **Queue idle timeout**: `loop()`'s `queue_get()` times out after 300 s with nothing queued → `stop()` → cleanup + disconnect.
2. **Alone in channel** (`on_voice_state_update`):
   - Bot ejected (`before.channel` set, `after.channel` None) → full `cleanup()`.
   - Bot *moved* between channels → cancel any stale alone-timer from the old channel.
   - Last human leaves the bot's channel → start a **10-second countdown** (`_alone_countdown`, tracked in `_alone_timers`): sends a notice via `send_with_np`, sleeps 10 s, re-checks channel membership, and cleans up if still alone. A human rejoining (or an explicit stop) cancels the timer. Mute/deafen events (channel unchanged) are ignored.

`cleanup()` also cancels any pending alone-timer first, so the timer can't fire after cleanup and attempt a second teardown.

---

### Crash Recovery

On `on_ready` (cold start or session loss — **not** WebSocket resume, which fires `on_resumed`), `MusicBot` spawns a `_restore_guild` task per guild (skipped if the guild already has a player):

```mermaid
sequenceDiagram
    participant MusicBot
    participant Redis as GuildRedisStore
    participant VC as Discord Voice
    participant MP as MusicPlayer

    MusicBot->>Redis: acquire_recovery_lock() [SET NX EX 60, random token]
    Note over Redis: lock:guild:{id}:recovery — prevents two instances racing
    MusicBot->>Redis: get_recovery_gate() [pipelined: state hash + LLEN queue]
    Note over MusicBot: gate is None (Redis read failed) → skip with warning,<br/>retried on next on_ready. Queue *contents* deliberately not read here.
    MusicBot->>MusicBot: gate: voice/text channel IDs present?
    MusicBot->>MusicBot: channels still exist? if deleted → clear_connection() +<br/>notice to a usable channel
    MusicBot->>MusicBot: gate: anything restorable? (queue length or crashed song)
    MusicBot->>VC: voice_channel.connect(timeout=30.0, reconnect=True)
    MusicBot->>MP: MusicPlayer(...) + start() → loop() + _restore_state()

    MP->>Redis: get_playback_snapshot() — one round-trip for state +<br/>queue + now_playing + history
    MP->>MP: restore volume (only if stored)
    MP->>MP: crashed song → SongQueueEntry.from_crashed_state()<br/>→ queue.restore_crashed() (persisted=False)
    MP->>Redis: clear current_song_url immediately (at-most-once)
    MP->>MP: queue.restore_entries(pending) + history.restore()
    MP->>Redis: refresh_ttl()
    MP->>MP: _restore_complete.set() → loop() may dequeue

    MusicBot->>Redis: release_recovery_lock()
```

Key properties:

- **Lightweight gate**: `get_recovery_gate()` reads only the state hash and the queue **length** (LLEN). A `-stop`ped guild keeps its (possibly long) queue list, so gating on LLEN keeps that payload off the wire on every `on_ready`. The full payload is read once by `_restore_state` after a successful connect.
- **At-most-once crashed song**: `current_song_url` is written when a song starts and cleared on normal end. On recovery the crashed song is rebuilt, injected in-memory only (`persisted=False` — it was never on the Redis queue list), and `current_song_url` is cleared immediately, even when the requester is unresolvable.
- **Failure isolation**: a failed snapshot read aborts the whole restore rather than fabricating partial state; the lock's 60 s TTL auto-expires if the holder crashes, and release compare-and-deletes so an expired holder cannot delete its successor's lock.
- **Intentional stop vs crash**: `cleanup()` calls `clear_connection()`, which empties the channel-ID fields — `on_ready` then skips that guild.

---

### Graceful Shutdown

Triggered by `-stop`, the alone-timer, bot ejection, or container shutdown:

```mermaid
sequenceDiagram
    participant Trigger as Trigger (-stop / alone timer / eject)
    participant MusicBot
    participant MP as MusicPlayer
    participant Redis as GuildRedisStore
    participant VC as Discord VoiceClient

    Trigger->>MusicBot: cleanup(guild)
    MusicBot->>MusicBot: cancel alone-timer; atomic mps.pop(guild.id)
    Note over MusicBot: pop-first gate — a concurrent cleanup call gets None and returns
    MusicBot->>MP: gather-cancel: _prefetch_task, _progress_task,<br/>_pause_debounce_task, _player, _restore_task
    MusicBot->>MP: retire_np_host_on_stop()
    Note over MP: no task can race this; dedicated NP msg deleted,<br/>command-response host strip-edited
    MusicBot->>VC: voice_client.disconnect(force=False)
    MusicBot->>Redis: store.clear_connection() + refresh_ttl()
```

`clear_connection()` distinguishes intentional stop from a crash (the Redis queue list is intentionally left intact — only the channel IDs and now-playing state are cleared). On process shutdown, `MusicBotApp.close()` tears down in order: drainer `stop()` (cancel, then one bounded final-drain attempt so a healthy Postgres receives whatever is buffered; never raises, even for an already-crashed task) → `PostgresHistoryArchive.close()` → Redis pool drain → `shutdown_telemetry()` (a blocking force-flush, run in an executor). Anything left in the outbox simply drains on next start.

---

## Data Model

### In-Memory Structures (per MusicPlayer instance)

`MusicPlayer` uses `__slots__`. Key attributes:

| Attribute | Type | Description |
|---|---|---|
| `current_song` | `Optional[YTDL]` | The `FFmpegOpusAudio` object currently playing |
| `play_next` | `asyncio.Event` | Set by the `after=` callback (thread-safe via `call_soon_threadsafe`); cleared at the start of each loop iteration |
| `queue` | `GuildQueue` | All queue state and operations (one deque + cursor, private to the class) |
| `history` | `GuildHistory` | Played songs: the in-memory ring (maxlen 50) and the Redis list are what `recent()` reads — it never touches Postgres. That list carries **no TTL, ever** (PERSISTed, capped by LENGTH); Postgres `play_history` is the durable record behind it, fed by the outbox drain and read only by `-leaderboard` |
| `play_message` | `Optional[discord.Embed]` | Cached NP embed for `-now`; cleared on song end |
| `volume` | `float` | 0.0–1.0; applied via FFmpeg `-filter:a volume=` on next song |
| `store` | `Optional[GuildRedisStore]` | `None` if no Redis configured |
| `_player` | `Optional[asyncio.Task]` | Long-lived `loop()` task |
| `_prefetch_task` | `Optional[asyncio.Task]` | Active `_prefetch_next_song()` task |
| `_restore_task` / `_restore_complete` | `Optional[asyncio.Task]` / `asyncio.Event` | One-shot `_restore_state()`; the event gates `loop()`'s first dequeue |
| `_progress_task` | `Optional[asyncio.Task]` | Per-song progress-bar updater |
| `_pause_debounce_task` | `Optional[asyncio.Task]` | Debounced pause/resume embed+presence refresh |
| `_skip_history_for` | `Optional[YTDL]` | Set by `interject()` to the parked song whose history add is deferred to its resume tail (holds identity, not a flag — the song can end during interject's awaits) |
| `_np_host_message` / `_np_host_own_embeds` / `_np_host_dedicated` | host pointer + its own embeds + kind | The one message currently carrying the NP block |
| `_np_edit_lock` | `asyncio.Lock` | Serializes old-host mutations (ticks vs retire) |
| `_background_tasks` | `set` | Keeps fire-and-forget tasks alive (GC guard) |

### Redis Schema

**The schema is defined in one place: `src/guild_state.py`** — field constants + frozen value objects with `from_redis`/`to_redis` converters; no other module touches raw wire bytes. Wire formats are pinned by golden-fixture tests (rolling restarts mix writers, so serializer changes must keep old entries readable).

All guild keys are prefixed `guild:{guild_id}:`. `GUILD_TTL = 86400` (24 h idle expiry), refreshed on writes, restore, and clean shutdown.

| Key | Type | Schema | TTL |
|---|---|---|---|
| `guild:{id}:state` | Hash | 18 fields → `GuildStateData`: `volume`, `voice_channel_id`, `text_channel_id`, `current_song_url/_title/_duration/_uploader/_requester_id/_interjected/_is_resume/_start_paused/_queued_at/_queue_position/_query_source/_played_at` (a parked `SongQueueEntry`), `play_start_epoch`, `total_pause_seconds`, `pause_start_epoch` | 24 h |
| `guild:{id}:now_playing` | Hash | 12 display fields → `NowPlayingData`: `title`, `webpage_url`, `uploader`, `duration`, `thumbnail`, `view_count`, `like_count`, `abr`, `asr`, `acodec`, `requester_id`, `requester_mention` | 24 h |
| `guild:{id}:queue` | List | JSON entries discriminated by `"type"`: `"qobj"` → `SongQueueEntry` (`webpage_url`, `title`, `requester_id`, `ts`, `user_input`, `duration`, `uploader`, `thumbnail`, `persisted`, `interjected`, `is_resume`, `start_paused`, `queued_at`, `queue_position`, `query_source`, `played_at`, `np_message_id`, `np_channel_id`, `np_dedicated`), `"ytsource"` → `SearchQueueEntry` (`ytsearch`, `url`, `ts`, `process`, `user_input`, `queued_at`, `queue_position`, `query_source`). RPUSH on enqueue (LPUSH to the front for `-playnow` resume entries); LPOP inside the atomic start transaction | 24 h |
| `guild:{id}:history` | List | JSON `HistoryEntry` objects (most recently RECORDED first), LTRIMmed to `HISTORY_CACHE_LIMIT` (50) and PERSISTed on every push. The **only** source `-history` reads, in both archive modes | **none, ever** |
| `guild:{id}:config` | Hash | 3 fields → `GuildConfig`, a guild's DURABLE choices: `debug_mode` (`"1"`/`"0"`), `volume`, `timezone` (an IANA name, resolved by `GuildConfig.tzinfo()` at read time). Every field is `Optional` and **absent means "no choice made"** — distinct from an explicit `0`/`false`, which is why it cannot be a plain `bool`. Deliberately NOT fields on `:state`: that hash expires in 24 h, so a choice stored there reverts on any guild idle for a day. Written only by an explicit command, PERSISTed, deleted on `on_guild_remove` | **none, ever** |
| `history:outbox` | Stream | Global (all guilds) write-ahead buffer for the Postgres archive, drained by the `drainers` consumer group — same `HistoryEntry` wire bytes under field `e`, each carrying `guild_id`. Near-empty in steady state; grows only while Postgres is down. Written only while `HISTORY_ARCHIVE_ENABLED` is true | **None — deliberately persistent** (holds not-yet-durable entries; never an eviction candidate under `volatile-lru`) |
| `leaderboard:v{n}:{guild_id}:{days}:{top_n}` | String | orjson aggregate cache for `-leaderboard` — one entry per requested window (`:0` = all-time). TTL'd, so it is a legitimate `volatile-lru` eviction candidate: losing it costs one re-query | 60 s |
| `lock:guild:{id}:recovery` | String | random token (SET NX EX — one restore per guild) | 60 s |
| `ytdl:stream:{webpage_url}` | String | JSON dict stripped to 16 fields (`url`, `webpage_url`, `title`, `uploader`, `uploader_url`, `upload_date`, `thumbnail`, `description`, `duration`, `tags`, `view_count`, `like_count`, `dislike_count`, `abr`, `asr`, `acodec`) | `expire − now − 1800s`; not written if < 60 s |
| `ytdl:source:{normalized search}` | String | `(webpage_url, title)` resolution of a search query | 1 h |
| `spotify:track:{id}` | String | `"Title Artist"` search string | 24 h |
| `spotify:playlist:{id}` | String | JSON array of track titles | 1 h (user-editable) |
| `spotify:artist:{ids}` / `spotify:album:{ids}` | String | JSON (ids comma-joined, sorted) | 24 h |
| Spotify token | String | Access token cached with its remaining TTL | token expiry |

### Postgres Schema

Applied by the in-app migration runner (`src/db_migrate.py`; files in `migrations/`, `schema_migrations` ledger, `pg_advisory_xact_lock` around each run):

- **`play_history`** — one row per played song: `id` (identity PK), `guild_id`, `title`, `webpage_url`, `duration_secs`, `played_secs`, `requester_id`, `requester_name`, `thumbnail`, `uploader`, `played_at timestamptz` (when the audio started — stamped once per play, so a `-playnow`-interrupted song files under the moment it first began, not the moment its resume tail ended), `inserted_at timestamptz` (server default; not on the wire), `message_id` and `channel_id` (the NP host at song end and the channel it was in — resolvable only as a pair, via `channel.get_partial_message(message_id)`, and both captured off the same message so they are both real or both `0`), `queued_at timestamptz` and `queue_position` (both read at **ask** time — when the user's command message was sent, and how many songs were ahead of it at that moment, counting the one playing; 0 = played immediately, and also what a row predating the fields carries. `queued_at` comes from the message snowflake, so it counts the 1–4s yt-dlp resolve and gateway delivery as the wait they are — but it is Discord's clock while `played_at` is the host's, so under host drift `played_at − queued_at` can come out slightly negative: that is skew, not corruption. `queue_position` is approximate against the insert by design, since the loop dequeues while a command resolves), and `query_source` (how the song was asked for: the literal `search`, or the host of the pasted link — `''` = unknown). **No NULLs** — the wire format's zero-value convention carries over (epoch-0 `played_at`/`queued_at` = unknown), because NULLs would break dedup-index semantics. Named `CHECK` constraints are the schema lock, held up by `HistoryEntry.__post_init__` clamping every value into the column domain before an insert is attempted.
- **`play_history_dedup`** — unique on `(guild_id, played_at, webpage_url)`: the at-least-once drain's dedup key. Uniqueness only; `play_history_recent` `(guild_id, played_at DESC, id DESC)` serves the reads. It bounds row *selection* for both `-leaderboard` aggregates via its `guild_id` prefix, and their `LATERAL` legs seek on it directly; it cannot bound the aggregation itself, which visits every matching row for that guild by definition.
- **`play_history_rejected`** — rows the server refused, payload preserved verbatim as `bytea`. Expected to stay empty forever; inspect with `just db-rejects`.

#### Why `query_source` is stored rather than derived

`webpage_url` looks like it already encodes where a request came from, and for link-shaped input it does — a SoundCloud link archives with a `soundcloud.com` URL, and any host `sources.py` does not special-case goes straight to yt-dlp and keeps its own. It collapses **three** kinds of request, though, and they are the three an operator most wants to tell apart:

- A **Spotify** link resolves to a *title string* (`MusicBot._resolve_source` calls `Spotify.track`), which `yt_source` runs as a `ytsearch:`. The archived URL is yt-dlp's YouTube result.
- A **plaintext search** reaches `yt_source` the same way and is byte-identical once archived.
- A **pasted YouTube link** produces the same shape again.

Measured against a real archive after a smoke test: 11 Spotify `-play` invocations, 10 rows, every one hosted at `www.youtube.com`. So the classification is captured at parse time in `sources.py` and carried on the queue entry — which is also why a Spotify **playlist** track, which sits in Redis as an unresolved search and becomes a YouTube URL only at dequeue, still records Spotify.

One token, no dataclass per site: a constant for each special-cased service (`youtu.be` collapses onto `youtube.com` — a shortener is not a different service) and the bare host for everything else, which is what distinguishes `tiktok.com` from `vimeo.com` without a type apiece. `HistoryEntry.__post_init__` clamps anything outside `^[a-z0-9.-]{0,64}$` to the unknown sentinel, so `play_history`'s `CHECK` on the same domain can never fire — unlike `title` and `uploader`, this column holds only machine-minted values, so an out-of-domain one is a producer defect rather than an unusual song.

`-leaderboard` resolves it through the same `CROSS JOIN LATERAL` that picks each winner's title, so it means "how it was most recently asked for" and costs no extra planner work.

**Wire cost is a step, not a size.** Outbox entries pack into listpack nodes bounded by `stream-node-max-bytes` (4096), so resident memory per entry jumps whenever a node loses one entry rather than tracking the payload. Measured on `redis:7-alpine` over 50 000 entries, that cliff sits at ~440 B of payload: 486.8 B/entry below it, 547.4 B above. Adding this field crossed it, so the 256 MB compose Redis now holds ~491k un-drained plays instead of ~552k — an 11% cut in Postgres-outage runway bought with 18 wire bytes, and the empty token pays it just as fully as a populated one.

**The next boundary was the allocator's, not the node's.** Adding `channel_id` (~32 B populated) took a 484 B entry to 516 B and cost another 14%: 547.7 → 625.3 B/entry, 490k un-drained plays down to 429k. The node cap explains that one — 8 entries per listpack became 7 — but not the shape of the curve, which is **not monotonic in payload size**. An *unstamped* entry (`channel_id: 0`, 499 B) is the worst of the three at **678.3 B/entry**, worse than the larger stamped one, because 499 B still packs ~8 per node and pushes the listpack from 3872 B to 3977 B — over 4096 with its header, into jemalloc's next size class. So the real step is the allocator bin the *node* lands in, and the reachable pathology is a payload that keeps its entry count while overflowing the node. Measure with 50 000 identical XADDs and `MEMORY USAGE`; `XINFO STREAM`'s `radix-tree-keys` gives entries-per-listpack, which is what makes a result explicable rather than merely observed. Never predict this from `len(to_redis())` — the arithmetic here would have said `+6%` and ranked the three cases in the wrong order.

**The spike belongs to the size, not to the entry.** A later 1-byte-resolution sweep (redis 7.4, 20 000 XADDs per point, same method) mapped the curve either side of that result: **~548 B/entry up to 497 B of wire, ~676 B/entry at 498–499 B, ~626 B/entry from 500 B on.** So the 678.3 above is that spike, and reading it as "unstamped entries are expensive" is the wrong lesson — an entry is expensive when its *wire size* lands on 498–499 B, whatever put it there. Two consequences. Any field whose populated and empty forms straddle that band must be measured in **both** shapes, which is the rule the paragraph above already states for a different reason. And a change of a few bytes anywhere else on the curve is close to free: the same sweep put every non-spike delta inside ±1 B/entry, which is how the ask-time `queued_at` switch (a 3 B shrink, µs → ms resolution) was cleared — it does not move the curve, it only shifts which titles land on the spike.

---

## Concurrency Model

Single asyncio event loop. All I/O is async. yt-dlp extraction is offloaded to the `YtdlpPool`'s `ProcessPoolExecutor` (`YTDLP_POOL_WORKERS`, default 4).

```mermaid
flowchart TD
    EL["asyncio Event Loop (main thread)"]

    subgraph PerGuild["Per-guild tasks"]
        PLAYER["_player — MusicPlayer.loop()"]
        RESTORE["_restore_task — _restore_state()"]
        PREF["_prefetch_task — _prefetch_next_song()"]
        PROG["_progress_task — _progress_updater()"]
        DEB["_pause_debounce_task"]
        ALONE["alone-timer — _alone_countdown()"]
    end

    subgraph BGTasks["Fire-and-forget tasks"]
        EPREF["prefetch_stream (one per enqueued single)"]
        REC["_restore_guild (one per guild on on_ready)"]
        FIN["NP finalize / one-off edits\n(held in _background_tasks)"]
    end

    subgraph ProcTasks["Process-wide tasks"]
        DRAIN["history-outbox-drainer\n(one per process; wake-on-notify + 30s tick;\nsupervised via done-callback)"]
    end

    subgraph ProcPool["ProcessPoolExecutor (YTDLP_POOL_WORKERS, default 4)"]
        YTDLP["yt-dlp extractions — _ytdlp_extract()"]
    end

    EL --> PerGuild
    EL --> BGTasks
    EL --> ProcTasks
    EPREF -->|run_in_executor| YTDLP
    PREF -->|via yt_stream| YTDLP
    PLAYER -->|via yt_stream| YTDLP
```

**Synchronization primitives:**

| Primitive | Owner | Guards |
|---|---|---|
| `play_next: asyncio.Event` | `MusicPlayer` | Song-completion signal from discord.py's audio thread (`call_soon_threadsafe`) |
| `_restore_complete: asyncio.Event` | `MusicPlayer` | `loop()` must not dequeue before restore populates the queue |
| `GuildQueue._mutex: asyncio.Lock` | `GuildQueue` (private) | Bulk queue mutations + the loop's `try_commit_dequeue()` |
| `GuildQueue._wake: asyncio.Event` | `GuildQueue` (private) | The pending-item signal a parked `get()` waits on. Set iff `_cursor < len(_items)`; `_sync_wake()` is its ONLY writer, because a stale set leaves the wait loop with no suspension point and stalls the whole event loop (I3) |
| `_np_edit_lock: asyncio.Lock` | `MusicPlayer` | Old-host edits vs retire (strip/delete is always the final write) |
| `Spotify._auth_lock: asyncio.Lock` | `Spotify` | Double-checked locking for token refresh |
| `mps.pop()` atomic gate | `MusicBot.cleanup` | Concurrent cleanup calls (stop racing voice-state event) |
| `HistoryOutboxDrainer._wake: asyncio.Event` | drainer | Outbox-push notify → drain wakeup (clear-after-wait ordering makes a racing push never lost) |
| `PostgresHistoryArchive._init_lock: asyncio.Lock` | `PostgresHistoryArchive` | Double-checked lazy pool creation + migration run (first successful `acquire()` wins) |

---

## Spotify Integration

Spotify URLs are resolved to YouTube search strings before any audio work begins — no audio is ever fetched from Spotify.

**Authentication**: Client Credentials flow (app-level, no user OAuth), over `aiohttp`. The token is cached in memory **and** mirrored to Redis with its remaining TTL, so a restart doesn't force a token round-trip. Refresh uses double-checked locking on `_auth_lock`.

**Cached API methods** (all trace-instrumented, all via the shared `_cached_call` helper):

| Method | Cache key | TTL | Returns |
|---|---|---|---|
| `track(id)` | `spotify:track:{id}` | 24 h | `"Title Artist"` search string |
| `playlist(id)` | `spotify:playlist:{id}` | 1 h (playlists are user-editable) | `List[str]` of track titles |
| `artists(ids)` | `spotify:artist:{sorted,ids}` | 24 h | Artist JSON |
| `albums(ids)` | `spotify:album:{sorted,ids}` | 24 h | Album JSON |

---

## Observability

Configured in `src/telemetry.py`; `setup_telemetry()` is the first call in `main()`. Disabled entirely when `OTEL_SDK_DISABLED=true` (the test suite sets this).

| Signal | Mechanism | Backend |
|---|---|---|
| Traces | Manual spans (`@_tracer.start_as_current_span(...)` decorators + context managers) on commands, playback-loop iterations, yt-dlp phases, Spotify calls, recovery; auto-instrumentation for Redis, aiohttp, and asyncpg | Tempo (via OTLP gRPC `:4317`) |
| Logs | `structlog` JSON → OTLP log exporter; `cog_before_invoke` binds `guild_id`/`user_id`/`command` contextvars to every log line in a command's scope | Loki |
| Metrics | `MeterProvider` + `PeriodicExportingMetricReader` (60 s) → same OTLP endpoint. First instruments: `musicbot.history.outbox.depth` (gauge — 0 on every drain-to-empty, refreshed from LLEN on each failed-drain retry; the alert line is depth growing across two scrapes = Postgres down longer than the drainer's backoff ceiling) and `musicbot.history.outbox.drained` (counter of entries retired, corrupt-dropped included). Instruments are API-proxy no-ops until `setup_telemetry()` runs. | Mimir |
| Resource | `service.name` (default `discord-music-bot`), `deployment.environment` = `ENVIRONMENT` | — |

Span conventions worth knowing:
- Every command invocation gets a `command.{name}` span opened in `cog_before_invoke` and closed in `cog_after_invoke` (error path ends it early).
- `player.loop.iteration` spans deliberately stay open for the full song duration (3–5 min typically) — this is expected, not a leak.
- The alone-countdown span covers only the post-sleep decision so it doesn't sit open for 10 s.
- Error embeds include a `trace: {trace_id}` footer (`util.trace_footer`) for cross-referencing user reports with Tempo.
- `shutdown_telemetry()` force-flushes on close, run in an executor because it can block up to 30 s.


---

## Audio Pipeline

```mermaid
flowchart LR
    YT["YouTube CDN\n(signed HTTPS stream)"]
    FFmpeg["FFmpeg process\n- Input: HTTP stream\n- Reconnect flags\n- Output: Opus frames\n- Volume filter (if ≠ 1.0)\n- Seek offset (-ss N, if ts set)"]
    Reader["discord.py reader thread\n(reads Opus frames from FFmpeg stdout\n→ YTDL.read() counts frames)"]
    VC["Discord Voice UDP\n(Opus + NaCl encryption)"]
    User["Discord Client\n(decodes Opus)"]

    YT -->|HTTP| FFmpeg
    FFmpeg -->|pipe| Reader
    Reader -->|encrypted UDP| VC
    VC --> User
```

**FFmpeg flags:**
- `before_options`: `-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5` — reconnects to the stream URL on drop
- `options`: `-vn` (audio only); extended with `-ss {ts}` for timestamp seeks and `-filter:a volume={v}` for non-unity volume

**Volume** takes effect on the **next** song — the FFmpeg process for the current song is already running.

**Position tracking**: the reader thread calls `YTDL.read()` once per 20 ms Opus frame; the subclass counts frames, giving `elapsed_secs` (frozen automatically during any pause or stall) and `position_secs = start_offset + elapsed_secs` — the single source of truth for the progress bar, presence timestamps, and pause confirmation.

**Timestamp seek**: a `?t=N` URL parameter is carried on `QueueObject.ts` → FFmpeg `-ss N` → recorded as `YTDL.start_offset` so position surfaces and the backdated `play_start_epoch` agree.

---

## Per-Guild State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle : MusicPlayer created\n(cog_before_invoke)

    Idle --> WaitingForSong : loop() started\n(_restore_complete set)

    WaitingForSong --> ResolvingSource : song enqueued\n(queue.get() returns)

    ResolvingSource --> Streaming : _resolve_source() complete\n(YTSource → QueueObject)

    Streaming --> Playing : try_commit_dequeue() ok\nvc.play(YTDL)

    Playing --> Prefetching : after= callback wired\nprogress task ticking

    Prefetching --> WaitingForEnd : _prefetch_next_song() running

    WaitingForEnd --> Playing : play_next set\n(prefetched_song available)
    WaitingForEnd --> ResolvingSource : play_next set\n(no prefetch available)
    WaitingForEnd --> WaitingForSong : play_next set\n(queue empty after song)

    Playing --> WaitingForSong : stream error (None YTDL)\nskip to next

    WaitingForSong --> [*] : 300s queue timeout\nor cleanup()
    Playing --> [*] : cleanup() / alone timer / eject
```

---

## History Archive Tier

Long-form context for the durable play-history tier. Source comments in
`guild_history.py`, `redis_client.py`, `history_archive.py` and `config.py` carry
the invariant and link here for the reasoning behind it.

The archive is **opt-in and default off**, and that flag is the only switch.
`HISTORY_ARCHIVE_ENABLED=true` turns the application side on; `resolve_archive_profile`
(`build_common.sh`, called by every deploy path) turns it into the `archive` compose
profile, which deploys `postgres` and the `db-migrate` one-shot. Compose cannot derive
it — profiles activate only from `COMPOSE_PROFILES` or `--profile`, and interpolation
has no value-conditional form — so a raw `docker compose up` deploys the default stack
whatever the flag says. A default deployment collects nothing long-term: no outbox
writes, no drainer, no `POSTGRES_URL` requirement. The flag — never the presence of
a URL — is what constitutes consent, which is why `setup_hook` reads it first, and
why a set `POSTGRES_URL` is explicitly ignored while it is off.

Enabled, the write path is: `GuildHistory.add()` → one Redis transaction that
LPUSHes the display list, LTRIMs and PERSISTs it, and XADDs the same wire bytes
onto the `history:outbox` stream → `notify()` the drainer. Song end therefore costs
exactly one Redis round trip and never awaits Postgres. The drainer then replays
its pending IDs, reads new ones, `INSERT … ON CONFLICT DO NOTHING`, and settles by
`XACK`+`XDEL`.

Four rules on the outbox are load-bearing rather than stylistic:

1. **Settle by ID.** `XACK` then `XDEL`, in that order. The reverse leaves a
   tombstone — an ID pending with no body — which replays forever.
2. **Never `XTRIM MAXLEN`.** MAXLEN means "keep the newest n", so a re-send after
   concurrent `XADD`s destroys a second tranche. `MINID` names an absolute ID and
   is inert on re-send.
3. **Always `approximate=False`.** redis-py's default trims to node boundaries,
   which on a small real stream trims nothing while reporting success — and
   fakeredis models it as exact, so a green unit test proves nothing here.
4. **`XTRIM` is blind to the PEL.** Anything that destroys entries must `XACK`
   them first or they replay forever.

An enabled archive that cannot durably record refuses to serve: `ensure_outbox_group`
aborts **startup** on a `WRONGTYPE` at `history:outbox`, the same policy as `setup_hook`
refusing an enabled archive without `POSTGRES_URL`. That check has to live at startup
because `push_history` is `@_guild_op`-wrapped, so the same error there would be
swallowed into one warning per song while both legs of its transaction failed. With the
archive disabled the key is inert — `setup_hook` never creates the group (that would
MKSTREAM the non-evictable key into existence) and a leftover is downgraded to a
warning.

The read side is `-leaderboard`, the only production reader of `play_history` rows
(`-ping`'s liveness probe is the other user of the pool). Two `GROUP BY` aggregates
— requesters and songs, both ranked by `sum(played_secs)` — run on **one** pooled
connection out of the `max_size=4` pool the drainer and `-ping` also draw from.
Songs group by `webpage_url` rather than title (titles drift, the URL is stable
identity) and the displayed title is simply the most recent one. The unknown
sentinels — `requester_id` 0 and `webpage_url` `''` — are excluded, since each would
merge unrelated plays into one top-10 row. A single `$3` cutoff parameter serves both
all-time and windowed boards: all-time passes `to_timestamp(0)`, the wire format's
floor, so the inclusive compare excludes nothing, while any real cutoff also excludes
epoch-0 unknown-time rows by definition. Entries still in the outbox are ignored: the
boards lag song end by the drain, which the help copy says rather than promising
real-time numbers.

**Each aggregate is two passes, and that is load-bearing.** Picking the display
name/title inline with `(array_agg(x ORDER BY played_at DESC, id DESC))[1]` makes it
an *ordered* aggregate, and an ordered aggregate removes hash aggregation from the
planner's options entirely: both boards then plan as `GroupAggregate` over a full sort
of every matching row, and `array_agg`'s state is not `work_mem`-bounded and cannot
spill. Measured at 3M rows: 6.8 s, 140 MB of external merge, and 431 MB RSS in one
backend for a single large group — enough to OOM a co-resident Postgres. Aggregating
first and resolving the ten winners through `LATERAL` keeps it a `HashAggregate` with
no temp files and no per-group state (300k rows: 880 ms → 53 ms, identical output).

The `LATERAL` leg rides `play_history_recent` and then filters, so it walks back to each
winner's newest play: cheap for a song still in rotation, proportional to the guild's
history for one that ranks on old plays alone. That worst case measured 111 ms at 300k
rows — still 8× better than the ordered-aggregate form. A
`(guild_id, webpage_url, played_at DESC, id DESC)` index would turn the walk into an
exact seek, at write amplification on an append-only table the drainer writes to
continuously; not worth it at these numbers. The leg deliberately carries no cutoff:
the totals are about the window, the title is the song's current name, so a play
outside the window still supplies it (pinned by `test_windowed_board_names_a_song_by_its_newest_title`).

**Three bounds, because this is the pool's only user-triggered traffic.** `max_concurrency(1, guild)`
serializes per guild; a 60 s Redis cache (`leaderboard:v{n}:{guild_id}:{days}:{top_n}`) collapses
repeats of the *same* window, though `--days` is a 0–3650 axis so it is not a rate
limit; and `_READ_CONCURRENCY = 2` against `max_size=4` keeps reads off the last
connections so a burst cannot starve the drainer — measured, 64 concurrent boards
pushed `insert_batch` from 11.8 ms to a 10 s acquire timeout and into backoff. A
`_READ_DEADLINE_SECS` bound covers the whole operation including the wait for a slot,
since two statements on one connection are otherwise bounded only by
2 × `command_timeout` — longer than the drainer's entire `DRAIN_DEADLINE_SECS`.

There is **no quarantine counter**. With the schema lock, a data-caused refusal is a
CHECK violation or a `DataError`, both named in `_POISON`; anything else is genuinely
transient and redelivers forever rather than being dropped. A growing outbox is the
visible symptom, and `HISTORY_OUTBOX_MAX` plus the depth gauge are what page on it.

Two operational hazards the stream transport introduces. `DEL history:outbox` — the
remedy the upgrade note asks operators to perform — also destroys the consumer group,
after which every read fails identically forever; `_read_batch` therefore heals
`NOGROUP` itself rather than relying on startup. And `SWEEP_MIN_IDLE_MS` must exceed
`DRAIN_DEADLINE_SECS * 1000`, because under a shared consumer name "idle" is measured
from last delivery, so a shorter value reclaims a live sibling's in-flight batch
mid-insert.

Exactly-once `play_history_rejected` recording used to come free from the drainer
lease, since only one process could hold a poisoned batch. Under a stable consumer name
two drainers can replay the same pending entry and both fail it, so exactness moved into
`_REJECT_SQL`'s `ON CONFLICT ON CONSTRAINT play_history_rejected_dedup` — it matters
because that table is expected to stay empty forever and `just db-rejects` reports its
contents, so two rows must mean two poisoned entries, never one seen twice.

**Disabling the archive does not erase anything.** `push_history` still PERSISTs
`guild:{id}:history` in both modes, so an operator who reads "nothing is written to
long-term storage", is later asked to erase a user's data, and removes the
`postgres-data` volume deletes nothing at all — the only copy is the Redis list.

The consumer name is **stable, not per-process**. The PEL belongs to the name, so a
starting process inherits whatever its predecessor left in flight and recovery
needs no lease, TTL or housekeeping. Two live drainers are safe by construction:
`>` hands them disjoint entries, and the pending replay hands them a shared set
that `ON CONFLICT DO NOTHING` collapses. Concurrency costs duplicated work, never
destroyed plays — which is the property the earlier `history:drainer` lease could
not offer, since a positional `RPOP` retire could delete entries the popping
process had never archived.

### History read path

**`-history` is served from the capped Redis list alone, in both archive modes.
Postgres is deliberately not on this path.**

The arithmetic that makes reading one leg complete: `musicbot.HISTORY_MAX_LIMIT`
(the command's ceiling) is pinned to `HISTORY_CACHE_LIMIT` (50), `push_history`
LTRIMs the list to exactly that many entries on every write, and `get_history()`
reads that whole window. So the Redis leg carries a slot for every play the command
can be asked for, and an archive read could only add rows *older* than the window
or duplicates of rows already in it. The list is also written synchronously at song
end, ahead of the drain, so it **leads** the archive rather than lagging it.

An earlier three-tier merge (Postgres-primary, freshness-merged against Redis and
the deque) was removed. Its complexity was entirely reconciliation the single-leg
read does not need: µs-truncated `timestamptz` against raw `time.time()` floats,
and an archive leg that looked full while the newest plays were still in the
outbox.

Two details that look like flaws and are not:

- **"A slot for every play", not "every play".** The ceiling equals the window, so
  there is no headroom: anything that occupies a slot without yielding a renderable
  play shortens the page by one. Two things can — an entry `get_history` drops as
  corrupt, and a duplicate `recent()` dedups. The second is reachable, because
  `create_redis_pool` sets `retry_on_error` and the LPUSH is not idempotent, so a
  timeout after the server applied `EXEC` re-sends the pipeline. This is stated
  rather than hidden: trimming to a margin above the ceiling would buy a one-row
  edge case at the cost of the "retention cap == display cap" identity the
  command's own help copy states to users.
- **The deque is merged, not a fallback.** As a second leg it can only *add* depth.
  Reaching it only when the leg above came back empty let a single Redis row
  suppress the whole cache — reproduced at Redis-erroring / cache-holding-9, where
  `recent(10)` returned 1 where the pre-archive code returned 9.

Dedup is on the **whole entry**, not `(played_at, url)`. Both legs carry the same
object's values (orjson round-trips a double without loss), so a full-value
comparison is exact. The narrower key collapsed genuinely distinct plays: `played_at`
defaults to `0.0` for entries predating the timestamped wire format, so two
different plays of one song keyed `(0.0, url)` became one.

`-history` is additionally capped at one render per guild
(`max_concurrency(1, guild, wait=False)`). It is the heaviest send in the bot — up to
8 song embeds plus the prepended Now Playing block — so unbounded concurrent renders
rate-limit a guild out of its own channel. `wait=False` is load-bearing: queueing the
extra invocations behind the first still issues every send, so they must be declined
outright, and `cog_command_error` renders the resulting `MaxConcurrencyReached` as a
notice rather than an error embed.

The single-leg rule is specifically about **`-history`'s recent-window read**, not
about the archive being unreadable. `-leaderboard` (see
[the archive tier](#history-archive-tier) above) is a production reader
of `play_history` and is the first: it aggregates over unbounded history, which the
50-entry window cannot answer, and the read-path rule sanctions exactly that —
Postgres backs the commands that need the permanent record.

`PostgresHistoryArchive.recent()` still has **no production caller**. It is the durable
record's row-level read side, exercised by the `pg` tier and available to tooling; the
archive's own query is served by the `play_history_recent` index with no sort node. Planning it
off `play_history_dedup` instead adds an Incremental Sort that must consume an entire
equal-`played_at` group before emitting `LIMIT 50` — and with backfilled rows all
landing on the epoch-0 sentinel those groups are large. Measured 37x slower: p50
49.98 ms vs 1.34 ms, ~9,900 buffers per call.

The Redis read is bounded at `_READ_TIMEOUT_SECS` even though `GuildRedisStore`
swallows its own errors — swallowing turns a *failure* into `[]`, but a
connected-and-unresponsive server produces no error to swallow and the pool sets no
socket read timeout. A slow tier must cost depth, never an error embed.

### Redis memory bounds

Three kinds of key carry no TTL and are therefore never eviction candidates under
`volatile-lru`: `guild:{id}:history`, `guild:{id}:config` and `history:outbox`. Once they fill
`maxmemory` with no TTL-bearing key left to evict, Redis rejects **every** write
with OOM — state, queue and cache alike — and each store method swallows it and
logs, so persistence degrades silently rather than crashing.

Only the **outbox** can reach that state by growing. The history lists are bounded
at `HISTORY_CACHE_LIMIT` per guild, so their total scales with guild count
(~24 KB each, ~24 MB across a thousand), not with runtime. The config hashes are
bounded by the number of settings that exist — three fields, ~160 bytes measured,
per guild that has chosen one — so they scale with guild count even more weakly
(~1.5 MB across ten thousand) and cannot grow on their own at all. The outbox is near-empty
whenever the drainer keeps up and grows for the whole duration of a Postgres
outage, at ~625 bytes per play — so the bundled 256 MB budget holds roughly 429k
un-archived plays. `HISTORY_OUTBOX_MAX` is the opt-in bound on it; it defaults to
unbounded because dropping entries there is real data loss, and with the lists
capped a dropped outbox entry has no second copy anywhere.

**The caveat the "function of guild count" phrasing hides:** the trim is *lazy*. It
runs inside `push_history` and nowhere else, so a guild that stops playing keeps
whatever oversized list it already had, forever, and no TTL path touches it. On a
deployment upgrading from a build that never trimmed, a dormant guild holding 100k
entries is ~49 MB, permanently, in a key `volatile-lru` can never evict. Only a
further play — or a manual `DEL` — reclaims it.

`outbox_depth` under-reports whenever entries were destroyed while still pending —
which is exactly when plays are being lost. The cap's ack-before-trim rule is what
prevents that; without it the depth alarm goes quiet during the incident it exists to
catch. The cap's MINID discovery also pages at `CAP_PAGE` and converges across passes:
a single `XRANGE COUNT=<overage>` would scale with the backlog (~240 MB in one reply
against a 500k-entry outbox), which is the stream re-creation of the 206 MB
`RPOP key 490000` incident.

A Redis memory/eviction alarm is still owed. Never switch to `allkeys-lru` as a
workaround: it makes the outbox evictable, and an evicted entry is a play that
vanishes with no error, no `play_history_rejected` row and no log line.

### Postgres credential handling

`docker-compose.yml` falls back to `POSTGRES_PASSWORD=password` so that
`docker compose up` works with nothing configured but `DISCORD_TOKEN`. That is a
first-run convenience and a liability everywhere else, so the bot detects it and
complains loudly: a startup ERROR, plus an owner-only row on `-ping`. It is only
defensible because the compose `postgres` service publishes on `127.0.0.1` — an
unauthenticated-in-practice database reachable from the network would not be an
acceptable default at any level of warning.

`.env` is the **one** supported place the real password is set: written by
`./setup_env.sh`, read by compose and by `just run`'s DSN derivation.

A per-install secret — a generated file behind `POSTGRES_PASSWORD_FILE`, or having
the `db-migrate` one-shot run `setup_env.sh` — was proposed and **declined**. It
adds a second secret store for a value that is set once at install and effectively
never rotated (Postgres reads `POSTGRES_PASSWORD` only when initializing an empty
data directory, so changing it later means `ALTER USER` either way). The exposure
it targets is instead carried by the loopback binding and the two warnings.

`using_default_postgres_password()` is scoped to the DSN shape that decision
produces — a password in the DSN userinfo, assembled from `.env` — and reads it out
of `POSTGRES_URL` rather than `POSTGRES_PASSWORD`, because the bot only ever sees
the assembled DSN. **Do not re-add asyncpg's full resolution ladder.** The cost of
the narrow scope is fail-open (the advisory goes quiet) for three hand-written DSN
shapes asyncpg accepts and this misses: `?password=` in the query string, a password
containing an unescaped `@`, and `PGPASSWORD` in the environment. None is reachable
from compose or `just run`.

The advisory is **owner-gated on `-ping`**. That command carries no permission check
and answers to `-status`, `-health` and `-l`, so an ungated advisory would confirm to
every member of every guild — permanently, in Discord's retained history — that this
host runs the public default. The value is a public constant in a GPL repo; the leak
is the confirmation, not the string. The `is_owner()` await must also be reached only
when the advisory exists: `MusicBotApp` sets neither `owner_id` nor `owner_ids`, so
discord.py falls through to `application_info()`, a REST GET that retries ~25 s on a
5xx and then raises — ahead of the skeleton send the command promises is immediate.

The default lives in six places with nothing linking them: `src/config.py`,
`build_common.sh`'s preflight, three `docker-compose.yml` service interpolations, and
the justfile's DSN derivation. Every drift fails **open** — rotate one and the
detector goes quiet while the deployment still runs on a known credential, or the
`just` recipes build a DSN the database rejects, surfacing much later as a drainer
backoff loop because the archive connects lazily. `tests/test_config.py` is the only
thing holding the six together.

The detector reads the bot's DSN; it cannot observe the server. A detector that could
not be fooled would attempt a connection using `DEFAULT_POSTGRES_PASSWORD`, at the
cost of a login attempt per render.

### History backfill

`src/backfill_history.py` copies pre-archive `guild:{id}:history` entries into
`play_history`, stamping the real guild id from the key (legacy entries parse as
`guild_id=0`, so every guild's rows would otherwise collide on the dedup index). It
inserts directly rather than through the outbox, which would bury the live drain
behind a historical backlog.

It must run **before** this build is deployed, and that window is unforgiving:
`push_history` LTRIMs each list on that guild's next song end, destroying the only
copy of exactly what the tool exists to move. The window is per guild, has no flag to
check and nothing to undo. "This build", not "the archive build" — the cap is not
part of the archive tier, so an operator who never opts in is on the same clock and
simply has no backfill to run.

Three operator-safety properties were learned the expensive way and are easy to
regress:

- **Exceptions are counted, not propagated.** An early version let them escape: one
  `WRONGTYPE` from a stray key killed the run, every guild later in SCAN order went
  unattempted, and the summary never printed (it sits after the `try/finally`), so a
  partially-applied migration surfaced as a bare traceback missing a different set of
  guilds each run.
- **`short_guilds` and `skipped_keys` are fatal to `ok`.** They were once warnings
  that still folded the guild into the success count, so a run that had just watched
  plays disappear printed a clean summary and exited 0 — and
  `just db-backfill && ./build_docker.sh` gates on that exit code.
- **Reconciliation is by identity, not by count.** `push_history` LPUSHes and LTRIMs
  in one transaction, so a list already at the cap keeps its length exactly while each
  song end destroys one unread tail entry. A count check sees nothing; re-reading the
  tail bytes is what catches it.

## Subsystem Invariants

Long-form context for invariants outside the archive tier. Source comments state the
rule and link here for the reasoning.

### Redis connection retry

The pool's `retry_on_error` must name **redis-py's** `ConnectionError`/`TimeoutError`,
not the builtins of the same name — redis-py's derive from `RedisError`, so the
builtins match nothing it ever raises. The retry object must also be the **asyncio**
`Retry` (`redis.asyncio.retry`), not `redis.retry`: the two classes share a name, a
constructor and their attributes, but only the async one awaits. The sync class's
`call_with_retry` returns an un-awaited coroutine, so nothing raises inside its `try`
and the error escapes outside the retry loop entirely.

Every assertion on the *configuration* passes under both classes. This pool was
configured for three retries and performed none for a year, with a green suite; only
counting attempts against a real connect distinguishes them. Without an explicit
`Retry`, redis-py also synthesises `Retry(NoBackoff(), 1)` for a non-empty
`retry_on_error` — one immediate reattempt and no backoff.

### yt-dlp client strategy

The bot names **no client**: `_EXTRACTOR_ARGS` passes `default`, which is yt-dlp's own
list. That is the strategy, not an omission — upstream moves its default when YouTube
breaks a client, and a hardcoded name pins us to one nobody is defending. Today
`default` is `visionos,web` (yt-dlp README: "By default, `visionos,web` is used"); it
was `android_vr`-led in an earlier release, so **treat every client name in this section
as a snapshot to re-verify on each yt-dlp bump.**

`visionos` carries playback: no PO token, no JS player, audio-only https (251/opus).
`web` is the fallback, and it only exists because the image ships Deno plus yt-dlp-ejs —
yt-dlp drops `web` from `default` outright when no JS runtime is available — with the
bgutil sidecar minting the GVS PO token its formats require. The plugin pin in
`pyproject.toml` and the sidecar image tag in `docker-compose.yml` **move in lockstep,
and nothing enforces it**: `just pins` does not cover this pair, so it is a hand check
(see CLAUDE.md rule 6a).

The degradation ladder is designed so every rung lands on a previously-working
configuration: `visionos` healthy → audio-only 251/opus; `visionos` out → `web` muxed or
SABR, warned once per format by `_record_serving_format` (yt-dlp's `tv_downgraded` /
`web_embedded` clients are **not** a rung here: they live in `_DEFAULT_AUTHED_CLIENTS`,
selected only `if self.is_authenticated`, and this bot sends no cookies or
credentials); sidecar down → `web`'s formats are withheld without a GVS
token, and measurement shows `web` alone frequently resolves no usable format at all, so
this rung is thinner than it reads; Deno broken → `web` leaves `default` entirely and
`visionos` is all that remains. Those warnings are the early-warning system for
YouTube-side changes; watch them after any yt-dlp bump.

Two facts that constrain deployment rather than extraction: YouTube signs `ip` inside
the `sparams` HMAC of every stream URL, so a URL is bound to the host that extracted
it and can never be replayed from another machine — relevant to any multi-host or
sharded deployment. And `fetch_pot=auto` consults the sidecar only when a selected
format requires a token, so it costs nothing while `visionos` is healthy; YouTube's
PO-Token guide lists HLS as exempt "currently", which is why the sidecar is
provisioned ahead of enforcement rather than after it.

### yt-dlp process boundary

Four things cross into the worker processes, each with its own contract:

- **The request** — frozen, slotted and `kw_only`, so adjacent same-type parameters
  (`download`/`process`) cannot transpose at a call site.
- **The callable** — pickled by qualified name, so it must stay module-level and be
  resolved per call. Capturing it would silently defeat every
  `patch("src.youtube._ytdlp_extract")` in the suite.
- **The result** — `_slim_info` is what makes it picklable at all; a raw
  `process=True` info dict carries live objects and commonly 100 KB–1 MB nobody reads.
- **The exception** — flattened in the worker by `_classify_ytdlp_error`, where the
  structure still exists (yt-dlp's own exceptions carry live tracebacks and cannot
  cross). **Every field of the flattened error needs a default**: a required
  positional pickles fine and fails on *unpickling* in the parent's result thread,
  which bricks the pool permanently.

### Queue invariant

The invariants the code cites by number:

| | |
|---|---|
| **I1** | `0 ≤ _cursor ≤ len(_items)` |
| **I2** | `_items[:_cursor]` are claimed-but-unsettled; `_items[_cursor:]` are pending |
| **I3** | `_wake.is_set()` iff `_cursor < len(_items)` — `_sync_wake()` is its only writer |
| **I4** | the Redis list equals the `is_persisted()` subset of `_items`, in order |
| **I5** | settles are FIFO — the released item is always index 0 |
| **I6** | in-flight items are a **prefix**, because Redis retires them by LPOP |

**`clear()` returns the claimed prefix too.** It feeds `MusicPlayer._flush_played`, so a parked
`-playnow` tail earns its `play_history` row only because the return covers `_items` entire and
not just what was pending. Return the pending slice instead and that row goes with no error and
no log line.

**`clear()` resets `_cursor` as well as the deque, and that is not bookkeeping.** A cursor
outliving the items it indexed makes `qsize()` return negative, `empty()` lie, and the next
`try_release()` pop an empty deque. The loop's own path there is safe twice over: the bumped
`_generation` makes `try_commit_dequeue()` refuse first, and the guard on `try_release()` makes
the failure path a no-op second. See [Queue Operations](#queue-operations) for the structure.

**Why `put_front`'s in-flight branch is not dead code.** `MusicPlayer.interject()`
neutralizes the prefetch before calling `GuildQueue.put_front()`, which normally means
no dequeued-but-uncommitted head exists. `MusicBot._interject_flow` reaches it anyway:
when `interject()` returns no outcome it falls back to `queue_put_front()`, and the
prefetch's claim is still open there. `put_front` must then rebuild the Redis mirror
rather than LPUSH, because the in-flight item's entry is still at the list head
awaiting a commit-time LPOP. Delete the branch as "unreachable" and that path silently
eats the new head.

Note that `-shuffle` requires **4** queued songs while `MusicPlayer.queue_shuffle()`
and `-help` both say 3 (tracked by an in-code FIXME).

### Now Playing host model

The NP block lives in exactly one host message. `_adopt_np_host` is pointer-first: the
pointer swap is synchronous, retirement is fire-and-forget under `_np_edit_lock`.
Overlapping sends can complete out of order — channel position is send-*start* order
while adopts run in send-*return* order — so an adopt for an older message id sheds its
own block instead of becoming host.

Song end *releases* the host, leaving a completed bar as truthful history. `-stop`
*retires* it, because a bar frozen mid-song on a stopped player is misleading. A stream
that never produced audio has its block disposed of rather than finalized, since a
completed bar would be a false record.

**Interrupted fragments clean up after themselves.** Releasing rather than retiring is
right for a song that ended, but a `-playnow`-interrupted fragment leaves a bar frozen
at its interrupt position — and a stack leaves one per interjection. So the resume tail
inherits a pointer to that card (`np_message_id` / `np_channel_id` / `np_dedicated` on
the wire, plus a runtime-only `np_host_ref`) and disposes of it when the tail starts,
strictly *after* its own card is up. Three constraints shape this:

- **Never a re-adopt.** `_adopt_np_host` refuses a message older than the current host
  by design — the live bar belongs at the channel bottom. The stored ids exist to
  *dispose* of the old card, never to move back to it.
- **The channel id comes from the host message**, not from the persisted text-channel id,
  which `set_context` reassigns on every command and which therefore records where the
  last command ran rather than where any card was posted.
- **Capture is late-bound**, at the interrupted fragment's iteration end. Anything read
  inside `interject()` can name a message the `-playnow` confirmation's own adopt has
  already retired.

The runtime ref is what allows full fidelity: a card hosted by a *command response* is
strip-edited back to its own embeds, which ids alone cannot reconstruct. After a restart
only the ids survive, so by-id cleanup is gated to **dedicated** cards — deleting a
non-dedicated host would destroy a user's reply. The ids are also not rewritten into the
already-serialized Redis entry, so post-crash cleanup is best-effort; both gaps leave the
pre-feature behaviour (the card simply stays) and are accepted as cosmetic.

While a guild has debug mode on, the block carries the debug footer like every other
embed — see [Debug footer seams](#debug-footer-seams).

### Debug footer seams

With debug mode on, every embed the bot sends grows a `🐞 …` footer identifying the
request (`debug_footer()`). The trace id is what makes it useful: it is already the
join key for every log line and span, so pasting one out of Discord finds the exact
request in Loki/Tempo.

"Every embed" is sent from three places, so three seams apply it:

| Seam | Covers |
|---|---|
| `MusicContext.send` (main.py) | command responses — their own `embed=`/`embeds=` kwargs |
| `MusicPlayer._decorate_for_debug` (musicplayer.py) | the NP block, applied inside `np_embed_block()`, plus the player's own notices |
| `MusicBot._debug_suffix` (musicbot.py) | `-ping` and `-debug`, which reply via `channel.send` and then edit, so neither seam above reaches them |

Rules each seam encodes:

- **The block decorates at build time, not at the attach site**, so every render —
  command attach, dedicated host, periodic tick, pause debounce, song-end finalize —
  produces one, and the tick refreshes the metrics alongside the bar.
- **The block carries no trace id.** It re-renders under the command span when a
  response attaches it and under the playback span on the next tick, so a trace id
  there would alternate on a single message. One-shot notices do carry theirs.
- **A host's cached own embeds are never re-decorated.** Their elapsed-ms records the
  request that sent them, so a command response that became the host before a toggle
  keeps the footer it was sent with until a new host replaces it.
- **The dashboard suffix is constant for the life of the invocation**, and omits
  elapsed-ms for that reason: the live-dashboard driver only edits when the render
  differs, so a per-tick-varying footer would edit the board until its deadline.
- **`-debug` gates the runtime segment on `operator`.** That card withholds its
  Runtime block from a non-owner and says so in the same embed.
- **Decoration replaces rather than appends**, and removes a stale suffix when there
  is nothing to show. `play_message` is built once per song, decorated in place, and
  re-sent by `-now`, so it outlives both a re-send and a mid-song toggle.

`RuntimeSampler` feeds the runtime segments on the NP tick's cadence
(`INTERVAL_SECS`, floored at 1 s and capped at 5 s), running only while some guild is
effectively debug-enabled.

## Design Decisions

### yt-dlp three-phase pipeline

The queue stores lightweight `QueueObject`s rather than fully resolved `YTDL` objects. Full stream extraction is deferred to just before playback so signed YouTube CDN URLs are as fresh as possible when FFmpeg starts. Phase 1's unified single extraction warms both the `ytdl:source` (1 h) and `ytdl:stream` caches in one yt-dlp call at enqueue time — halving YouTube request volume versus the previous search-then-prefetch double extraction — so Phase 2 is almost always a cache read; the Phase 1b eager prefetch covers the entries that skip the unified path (playlist items, requeues).

### Keeping `_prefetch_next_song`

Even with enqueue-time prefetch warming the cache, `_prefetch_next_song` still constructs the `YTDL` object (starting the FFmpeg process) while the current song plays — the next song's FFmpeg process has already buffered data when `play_next` fires, achieving zero inter-song gap.

### `GuildQueue` owns the structure

Earlier revisions kept parallel queue collections on `MusicPlayer`, synced by call-site discipline; they became two private legs, and are now one deque plus a cursor. Nothing outside `GuildQueue` can move either, every Redis-touching mutation runs under the class's one mutex, and the claimed prefix survives a bulk mutation because nothing before the cursor moves. The one accepted residual race (commit → LPOP handoff) is documented in the module docstring rather than papered over.

### Schema module with golden fixtures

Every persisted byte is defined in `guild_state.py` as frozen value objects with explicit converters, and the wire formats are pinned by golden-fixture tests. Rolling restarts mix old and new writers, so serializer changes must keep old entries readable. The `"qobj"`/`"ytsource"` discriminator strings are kept verbatim from the original serializer for exactly this reason.

### At-most-once delivery for crash recovery

`current_song_url` is written when a song begins and deleted when it ends normally; a non-empty value at startup means a mid-song crash. Recovery re-enqueues the song in-memory only (`persisted=False` — it never touches the Redis list, and `redis_pop_for()` skips it) and clears the key immediately, so repeated crash-restart cycles cannot accumulate duplicates. The start transaction (`pop_queue_and_start_song`) makes the LPOP and the state write atomic, closing the historical at-most-once window for normal dequeues too.

### `-playnow` interjection via front-inserted resume entries

`-playnow` interrupts the current song and hands it back afterward without any new task, timer, or side channel: the parked song becomes an ordinary `is_resume` `SongQueueEntry` LPUSHed to the front of the queue (`put_front`), carrying its `position_secs` as `ts` and its paused state as `start_paused`. The loop replays it through the same `-ss`/seek path any `?t=` song uses, so resume fidelity and crash recovery come for free. The only extra state is `_skip_history_for` (so a parked song is logged to history once, at its resume tail, not twice). Interjections **stack**: a `-playnow` on top of a `-playnow` parks that song too, and the queue unwinds LIFO. One `_skip_history_for` slot is still enough at any depth — each interjection stops exactly one song, and that song's loop iteration consumes the marker before the next `-playnow` can finish resolving. Full design: [PLAYNOW_PROPOSAL.md](PLAYNOW_PROPOSAL.md).

### Persisted `YTSource` entries

Spotify playlist tracks are enqueued as unresolved `YTSource(ytsearch=...)` items to keep `-play` fast for large playlists (metadata comes from Spotify's cached API, not N yt-dlp calls). They are **not** prefetched at enqueue time (no stable `webpage_url`; would saturate the extraction pool), but they **are** persisted to the Redis queue as `"ytsource"` wire entries and survive restarts.

### Now Playing block attached at send time

The NP block is prepended to every outgoing message in the player's channel (via `MusicContext.send`) rather than re-sent or edited-in afterwards: response + block form one atomic message, so the live bar is never buried and never flickers. Host adoption is pointer-first and gated on the song still being current, because a send's `await` can cross a song boundary.

### Frame-counted playback position

Position is derived from `YTDL.read()` call counts rather than wall-clock timestamps. discord.py's `AudioPlayer` skips `read()` entirely while paused, so the counter freezes for explicit pauses *and* involuntary stalls (voice-gateway reconnects) with zero manual bookkeeping — an accumulator approach would only see explicit pauses.

### Errors degrade persistence, never playback

Every `GuildRedisStore` method catches and logs Redis exceptions internally. Redis being down means no crash recovery and cold caches — playback, queueing, and embeds keep working from in-memory state.

### Per-guild isolation

`MusicBot.mps: dict[guild_id → MusicPlayer]` gives each guild an independent player. `cleanup()` pops the entry atomically (first caller wins — a concurrent voice-state event can't double-clean), cancels all five per-guild tasks, retires the NP host, disconnects, and clears the connection info.

### Distributed recovery lock

`lock:guild:{id}:recovery` (`SET NX EX 60`) admits one restore per guild at a time. Its contention is mostly *not* cross-instance — `AutoShardedBot` puts every shard in one process, and Discord routes a guild to exactly one shard, so even multi-process sharding gives a guild exactly one restorer. What it does contend with is repetition inside one process: `on_ready` re-dispatches on any reconnect that fails to RESUME, and `restore_guild`'s `guild.id in cog.mps` early-out does not cover the window, because `mps[guild.id]` is assigned only *after* the voice connect. Two processes overlap only when their shard assignments do — a surge-style rolling deploy, or a second bot started by hand against the same Redis.

The lock is not the last line of defence there. `abc.Connectable.connect` registers the voice client via `state._add_voice_client(...)` **before** its first await, so a second concurrent `restore_guild` in the same process raises `ClientException` and never reaches `MusicPlayer(...)`. The lock's job is to stop the duplicated work and the duplicated user-facing notice ahead of that.

The 60 s TTL auto-expires if the holder crashes before releasing. The value is a per-acquisition random token so release can compare-and-delete under WATCH/MULTI: a holder whose lock expired mid-recovery must not delete the lock its successor now owns, which would admit a third restore and produce exactly the double-restore the lock prevents.

### `AutoShardedBot`

Discord requires sharding at 2500+ guilds. `AutoShardedBot` negotiates shards automatically within the single process; migration to multi-process sharding (with Redis Streams for cross-process queues) is planned at far larger scale.

### `volatile-lru` eviction policy

With 256 MB `maxmemory` and `volatile-lru`, only TTL-carrying keys are eviction candidates — all caches and all `guild:*` runtime keys, every one reconstructible (caches) or re-creatable (runtime state). Three kinds of key are deliberately TTL-less and must never be evicted: `history:outbox`, which holds played-song entries not yet drained to Postgres; `guild:{id}:history`, the capped window `-history` reads and the only source it has; and `guild:{id}:config`, which holds each guild's durable choices, where an eviction is a setting silently reverting with no log line. An `allkeys-*` policy would let memory pressure destroy not-yet-durable history or a guild's settings, which is why the compose file pins the policy with a do-not-change comment.

### Two-tier data architecture (Redis + Postgres)

The durable/runtime boundary is drawn once: data a user would miss a week later lives in Postgres (`play_history` now; future stats/preferences); data that only matters to the running player stays in Redis, permanently — the runtime tier is *correctly placed*, not "not yet migrated". Writes cross the boundary through the `history:outbox` Redis **stream**, drained by one background task (replay pending → read new → `INSERT … ON CONFLICT DO NOTHING` → `XACK`+`XDEL`), so the playback loop keeps Redis-only latency, Postgres downtime buffers instead of losing entries, and the dedup unique index makes at-least-once delivery and backfill idempotent by the same mechanism. **Reads follow the same rule**: `-history` shows the recent window and is served from the capped Redis list alone — see [History read path](#history-read-path) — while `-leaderboard` needs the permanent record and is the archive's one reader. The archive is opt-in and a default deployment collects nothing long-term.
