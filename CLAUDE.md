# CLAUDE.md

Guidance for Claude Code when working in this repository. Everything here was derived
from the code itself — module docstrings and comments in this codebase are unusually
detailed and are the authoritative record of design decisions and past incidents.

## Project overview

**discord-music-bot** (v2.11.0, GPL-3.0) is a self-hosted Discord music bot that streams
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

| | |
|---|---|
| Language / runtime | Python **3.14+** (`requires-python = ">=3.14,<4.0"`) |
| Package manager | Poetry 2.x (`poetry.toml`, in-project venv) |
| Task runner | `just` (justfile is the index of every dev command) |
| Discord | discord.py 2.7.1 (exact pin), prefix commands (`-`), voice via PyNaCl/FFmpeg |
| Extraction | yt-dlp 2026.7.4 (exact pin, extras `[default, deno]`) in a **ProcessPoolExecutor** |
| Runtime state | Redis 7 (redis-py asyncio), orjson as the project-wide wire codec |
| Durable history | Postgres 18 + asyncpg (no ORM); migrations in `migrations/`, applied by `src/db_migrate.py` |
| Observability | OpenTelemetry (OTLP gRPC) + structlog JSON; Grafana LGTM stack in compose |
| Tests | pytest + pytest-asyncio (`asyncio_mode = "auto"`) + fakeredis + pytest-timeout; ~2,330 tests plus two opt-in integration tiers (testcontainers): a 74-test `pg` tier and a 41-test `redis` tier; coverage gate `fail_under = 80` (actual ~94%) |
| Lint/types | ruff 0.15.21 (format + lint) and pyright 1.1.411 (exact pins) |

Entry point: `just run` (loads `.env`) or `poetry run bot` → `src.main:main`.
**`POSTGRES_URL` is required while the archive is enabled** — `setup_hook` refuses to
start an enabled archive without it. Disabled (the default), no Postgres is needed.

## Golden rules — read before editing anything

1. **The comments are load-bearing.** Docstrings and comments in this repo document
   invariants, race windows, incident postmortems, and "do not simplify this" traps
   (e.g. `decode_responses=False` casts, the in-flight-head branch in
   `GuildQueue.put_front`). Never delete or contradict a comment without updating the
   behavior it describes; when you change behavior, update the comment in the same edit.
2. **`docs/ARCHITECTURE.md` is tracked; the rest of `docs/` is not.** `.gitignore` is
   `docs/*` plus `!docs/ARCHITECTURE.md` (the negation needs `docs/*` — git cannot
   re-include a file whose parent directory is excluded). Comments carry the invariant
   inline and link to an anchor for the long-form context they no longer repeat
   (`See docs/ARCHITECTURE.md#queue-invariant`) — **read the anchor, it resolves.**
   Renaming one of its headings orphans those pointers silently; nothing checks them yet.
   Any other `docs/*.md` path is local-only working material (plans, reviews,
   proposals): do not try to read it and do not treat its absence as an error. No such
   reference survives in `src/` or `tests/` today, and new ones do not belong there —
   if a comment needs more context than it carries, add the section to ARCHITECTURE.md
   and link that.
3. **`except A, B:` is intentional.** Bare unparenthesized multi-exception catches
   (`except ValueError, TypeError:`) are **PEP 758 (Python 3.14+) tuple-catch syntax**,
   normalized by ruff at `target-version = "py314"`. This is NOT the Python-2 form. Do
   not re-parenthesize (ruff strips it back) and do not "fix" it.
4. **Every user-visible reply is an embed.** `MusicContext.send` prepends the Now Playing
   block to responses; a bare `content` string would render as loose text above the
   block. Use `notice_embed()` / `send_embed()` from `src/util.py`.
5. **Redis IO never raises out of `GuildRedisStore`.** Its methods are wrapped by the
   `@_guild_op` decorator (log warning, return default). Everything must degrade
   gracefully when Redis is down or `store is None` — the in-memory bot keeps working.
   Keep new store methods on this pattern; never pass a **mutable** `default=` to
   `_guild_op` (use `default_factory`; a test enforces this).
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
6. **Version pins move in lockstep.** Bump both halves in the same commit. `just pins`
   enforces seven pairs — it is a dep of `check` and CI also runs it as its own step,
   deliberately: Dependabot's `pip` and `pre-commit` ecosystems open SEPARATE PRs that
   each move one half, and those PRs are validated by CI and never by a local `check`.
   The seven: the ruff pin (pyproject) ↔ the ruff hook `rev` in
   `.pre-commit-config.yaml`; the image name (justfile `IMAGE` ↔ `build_common.sh`
   `IMAGE_NAME`); and `postgres:18-alpine` / `redis:7-alpine` each across three files —
   the integration tier's `_PG_IMAGE`/`_REDIS_IMAGE`, `ci.yml`'s service container, and
   `docker-compose.yml` (compared tier↔ci and compose↔ci, so all three agree); and
   `_POSTGRES_CONTAINER` (`src/debug.py`) ↔ the postgres service's `container_name`,
   which is a Prometheus label selector, so a rename there would otherwise leave
   `-debug`'s cpu/mem row reading `n/a (no metrics source)` forever rather than
   failing. The compose legs are anchored to the named service, not `head -1`, so a
   second postgres or redis service cannot silently shift what is compared.
   **Three pairs are NOT enforced — this list is what a maintainer checks by hand,
   so keep it complete:**
   (a) `bgutil-ytdlp-pot-provider` (pyproject) ↔ the
   `brainicism/bgutil-ytdlp-pot-provider` image tag in `docker-compose.yml`. The plugin
   and the sidecar are released in lockstep; drift breaks PO-token minting, which
   surfaces as YouTube playback failures, not as a red build.
   (b) The published Prometheus port `9090`, in **four** places that move together: the
   `PROMETHEUS_HOST_PORT` defaults inside the bot service's `DEBUG_PROMETHEUS_URL` and
   inside the otel-lgtm service's `ports:` entry (both `docker-compose.yml`), and the
   commented-out `DEBUG_PROMETHEUS_URL` and `PROMETHEUS_HOST_PORT` assignments in
   `.env.example`. Change one and `-debug` queries a port nothing publishes. A **fifth**
   literal — the container side of that same `ports:` entry — is Prometheus's own listen
   port inside `grafana/otel-lgtm` and must NOT move with them; both files also name the
   number in prose, which drifts just as silently.
   (c) `otel/opentelemetry-collector-contrib` (the `otelcol-metrics` service) ↔ the
   otelcol-contrib build inside `grafana/otel-lgtm` (the `otel-lgtm` service), both in
   `docker-compose.yml`. The comment above the collector's `image:` line states the rule
   — bump either image and check the other by hand. Like (a), drift is invisible to
   every build: the symptom lands on the metrics path, where a missing `docker_stats`
   series leaves `-debug`'s cpu/mem row reading `n/a (no metrics source)`, which is also
   exactly what "the `metrics` profile is not running" looks like.
7. **Do not create `pyrightconfig.json`.** `[tool.pyright]` in `pyproject.toml` is the
   single source of truth; a `pyrightconfig.json` would silently override it for editors
   only. Do not re-add `venvPath`/`venv` there either — `just types` passes
   `--pythonpath` explicitly (see the long comment in pyproject for why).
8. **Suppressions name their rule.** Use `# pyright: ignore[reportSomeRule]`, never bare
   `# type: ignore` (`reportUnnecessaryTypeIgnoreComment = "error"` will flag stale ones).
   New ruff rules must exist (`ruff rule <CODE>`) — this repo once carried a pyright
   setting that was never a real rule and silently checked nothing.
9. **Pickle contracts on process-boundary exceptions.** Every field of
   `ExtractionError` (src/youtube.py) and `RemoteCallError` (src/ytdlp_pool.py) MUST have
   a default. A required positional breaks unpickling in the executor's result thread and
   permanently bricks the pool. A round-trip test guards this — keep it passing.
10. **Never construct `MusicBotApp` at module scope.** yt-dlp pool workers re-import
    modules under spawn/forkserver; the bot is built inside `main()` only.
11. **`pytest` filterwarnings is `error`.** Any new `DeprecationWarning` fails the suite.
    Add a targeted `ignore:` entry in `[tool.pytest.ini_options]` only with a comment
    explaining what upstream fix removes it (see the existing audioop entry).
12. **Redis eviction policy is `volatile-lru` on purpose.** Three keys carry no TTL
    and none may become an eviction candidate. `history:outbox` holds plays that
    are not durable in Postgres yet (written only while the archive is enabled — when
    it is off the key is never created, but the policy must still protect a leftover
    from an earlier enabled run); evicting one loses that play with no error, no
    `play_history_rejected` row and no log line. `guild:{id}:history` is PERSISTed
    and capped at `HISTORY_CACHE_LIMIT` — it is the ONLY source `-history` reads, in
    both archive modes, so evicting or expiring it answers a guild with silence.
    `guild:{id}:config` holds a guild's DURABLE choices (debug mode, volume, timezone) — evicting it silently reverts a setting the guild chose, with no log
    line and no error, which is exactly the failure the in-memory version had. It is
    a fixed handful of fields per guild, written only by an explicit command and
    deleted on guild removal, so it scales with guild count and not with runtime.
    Never switch the compose Redis to `allkeys-lru`, and never put a TTL on the
    history or config keys: history is bounded by LENGTH, and config is bounded by
    the number of settings that exist.

## Commands

All dev commands go through `just` (must be installed system-wide, not only in the venv —
the pre-push hook depends on it). Run `just` alone to list recipes.

```bash
just install        # venv with main + test + lint + dev groups (contributors)
just hooks          # install pre-commit (fast) + pre-push (just check) git hooks
just hooks-run      # run every hook over every file, not just staged ones
just hooks-update   # bump pinned hook revs in .pre-commit-config.yaml

# Inner loop — fastest first
just fmt            # ruff format + autofix (REWRITES files)     ~0.1s
just fmt-justfile   # `just --fmt --check` on the justfile        ~0.01s
just fmt-check      # format check only                          ~0.05s
just lint           # ruff check                                  ~0.05s
just pins           # assert the six duplicated version/name pins ~0.02s
just types          # pyright over src/ AND tests/                ~6s
just test           # pytest with coverage (fail_under=80)        ~27s
just test-report    # `test` + the coverage/JUnit artifacts CI's PR comment consumes
just check          # fmt-justfile + pins + fmt-check + lint + types + test  ~35s
just test-pg        # opt-in real-Postgres tier (testcontainers, needs Docker) ~45s
just test-redis     # opt-in real-Redis tier (testcontainers, needs Docker)     ~15s
just container-test # build test image, run suite inside it       ~1min
just ci             # check + container-test + test-pg + test-redis — local mirror of CI

# Test selection (args forward to pytest)
just test tests/test_youtube.py
just test -k spotify
just test --maxfail=1

# Database (operator tools. db-migrate/db-backfill run the LOCAL venv against
# POSTGRES_URL; setup/backup/restore are shell around pg_dump/psql, no venv needed)
just setup                 # bootstrap .env with a generated POSTGRES_PASSWORD
just db-migrate            # apply pending migrations — REQUIRED before the bot serves
just db-backfill [--dry-run] # move pre-archive Redis history into Postgres — see below
just db-rejects [n]        # list play_history rows Postgres refused (expected: nothing)
just outbox [idle_ms]      # outbox health: depth, in-flight, stranded, TOMBSTONES (lost plays)
just db-backup             # dump to backups/
just db-restore FILE [DB]  # restore into a SCRATCH db (live needs CONFIRM=1 + a name)

# Build & deploy
just image                 # build runtime image :latest and :<git-sha> (no test gate)
./build_docker.sh          # full pipeline: just check → just image → deploy
just up [sha]              # deploy an already-built image (never builds; refuses unknown tags)
just down / restart / logs / ps
just test-image-rebuild    # required after changing pyproject.toml/poetry.lock
```

`DOCKER=1 just check` (prefix must come BEFORE the recipe) runs any of
fmt/fmt-check/lint/types/test/check inside the test image — no local Python/Poetry/Node
needed. `src/`, `tests/`, `pyproject.toml` are bind-mounted; formatting runs as your uid.

Run the bot locally: `./setup_env.sh` (or `just setup`), then
`docker compose up -d redis` — or, to run with the archive,
`docker compose up -d redis postgres db-migrate` (explicitly naming the profiled
services auto-activates their `archive` profile, so this works without
`COMPOSE_PROFILES`; archiving also needs `HISTORY_ARCHIVE_ENABLED=true` in `.env`) —
then **`just run`**. Use `just run` rather than `poetry run bot`: the bot reads only
the environment and has no `.env` support — `just run` loads `.env` and derives
`POSTGRES_URL` from the same `POSTGRES_*` parts compose uses (the disabled bot
ignores it). Needs FFmpeg on PATH and a `DISCORD_TOKEN`. Full stack:
`docker compose up` (bot + Redis + PO-token sidecar + Grafana LGTM; ~1 GB first
pull; + Postgres and the migration one-shot when the `archive` profile is active).
Compose requires `.env`.

## Repository layout

```
src/
├── main.py           # entrypoint: MusicBotApp (AutoShardedBot), MusicContext, Redis pool wiring
├── musicbot.py       # MusicBot cog — every command, per-guild player registry (mps), crash-recovery entry
├── musicplayer.py    # MusicPlayer — per-guild playback loop, prefetch, gate, NP host, ETA, interject
├── guild_queue.py    # GuildQueue — one deque + cursor, the mirror writer, bulk-mutation mutex
├── guild_history.py  # GuildHistory — played-song history (capped Redis list + in-memory cache; writes feed the outbox while the archive is enabled, reads never touch Postgres)
├── history_archive.py# Postgres archive (asyncpg) + HistoryOutboxDrainer (outbox → play_history)
├── leaderboard.py    # -leaderboard tunables, Redis result-cache codec, embed renderer (pure;
│                     # the command itself stays on the cog)
├── db_migrate.py     # SQL migration runner (`python -m src.db_migrate`, EXPECTED_SCHEMA_VERSION)
├── backfill_history.py # ONE-SHOT operator script: pre-archive Redis history → Postgres, direct
│                     # (not via the outbox). Run BEFORE deploying this build — see below
├── guild_state.py    # Pure Redis schema: frozen value objects, field constants, orjson wire formats
├── redis_client.py   # Connection pool, GuildRedisStore (@_guild_op), cache helpers, recovery lock
├── youtube.py        # yt-dlp integration: caches, stream probe/heal, YTDL audio source, worker fn
├── ytdlp_pool.py     # ProcessPoolExecutor lifecycle: lazy spawn, break-healing, worker log plumbing
├── sources.py        # Input parsing → YTSource / SpotifySource / SoundcloudSource; mints query_source
├── spotify.py        # Spotify Web API client (client-credentials, Redis-cached)
├── help.py           # man(1)-styled embed -help command (copy lives on the commands themselves)
├── dashboard.py      # optimistic-send + live-edit driver shared by -ping and -debug
├── ping.py           # -ping health dashboard: probes + render (sequencing is dashboard.py)
├── debug.py          # -debug snapshot + debug-mode toggle parsing; OBSERVATION-ONLY by rule
│                     # collectors are live-edit probes (dashboard.py); host blocks are owner-only
├── telemetry.py      # OTel traces+logs, structlog config, worker logging, gateway span filter
├── config.py         # ENVIRONMENT detection (env var or git branch), SpotifyStatus, tunables
└── util.py           # logger factory, embed helpers, fmt_duration, task helpers

migrations/           # NNNN_*.sql, applied in numeric order; the ONLY source of schema
docs/ARCHITECTURE.md  # the only tracked file under docs/ — anchor target for comments (rule 2)
tests/                # one test_<module>.py per src module + conftest.py (seams) + helpers.py
                      # test_pg_integration.py / test_redis_integration.py are the opt-in tiers
justfile              # every dev command; build_common.sh / build_docker.sh / deploy_docker.sh compose them
Dockerfile            # 3 stages: builder (deps) → test (adds test+lint groups) → runtime (ffmpeg, no poetry)
docker-compose.yml    # bot (host network) + redis + postgres + db-migrate (one-shot,
                      # `archive` profile) + db-backfill (one-shot, `ops` profile, run by
                      # hand) + bgutil-pot-provider + otel-lgtm
.github/workflows/    # ci.yml, security.yml (pip-audit), todo-to-issue.yml
```

## Architecture

### System overview

```
Discord gateway/voice                    YouTube / Spotify / SoundCloud CDNs
      ▲    ▲                                        ▲
      │    │ Opus/UDP                               │ HTTPS (extraction + stream)
      │    │                                        │
┌─────┴────┴──────────────────────────────────┐     │
│ MusicBotApp (AutoShardedBot, one process)   │     │
│                                             │     │
│  MusicContext.send ──► NP-block attach      │     │
│  MusicBot cog ─► mps: {guild_id: player}    │     │
│       │                                     │     │
│  ┌────▼──────────── per guild ───────────┐  │     │
│  │ MusicPlayer                           │  │  ┌──┴───────────────────┐
│  │  • loop() playback task               │◄─┼──┤ YtdlpPool            │
│  │  • prefetch task (one ahead)          │  │  │ ProcessPoolExecutor  │
│  │  • playback gate + restore task       │  │  │ (yt-dlp workers, 4)  │
│  │  • NP host / progress / presence      │  │  └──────────────────────┘
│  │  • GuildQueue  • GuildHistory         │  │
│  │  • GuildRedisStore                    │  │   FFmpeg subprocess per song
│  └────────────────┬──────────────────────┘  │   (spawned by YTDL/FFmpegOpusAudio)
└───────────────────┼─────────────────────────┘
                    ▼
   Redis 7 (AOF) ── guild:{id}:{state,queue,now_playing,history}
                    ytdl:source:* / ytdl:stream:* / spotify:* caches
                    lock:guild:{id}:recovery
                    history:outbox  (STREAM + "drainers" consumer group, no TTL)
                         │
                         │ ARCHIVE TIER — opt-in (HISTORY_ARCHIVE_ENABLED +
                         │ the `archive` compose profile); default OFF: no
                         │ outbox writes, no drainer, no Postgres deployed.
                         │ HistoryOutboxDrainer (one task per process, no lease):
                         │   XREADGROUP pending(0) else new(>) 100
                         │     → INSERT ON CONFLICT DO NOTHING
                         │     → MULTI: XACK + XDEL by ID
                         ▼
   Postgres 18 ───── play_history (durable, unbounded); schema owned by migrations/
                     the durable record. NOT on the -history read path: that command
                     is served from the capped Redis list, which leads the archive
   Sidecars: bgutil-pot-provider (:4416, PO tokens), otel-lgtm (:4317 OTLP, :3014 Grafana)
```

### Startup and shutdown

`main()` order matters: `setup_telemetry()` first (configures structlog before any
`get_logger()` resolves), then `DISCORD_TOKEN` check, then `MusicBotApp()` construction
(inside `main()` only — see golden rule 10). `setup_hook` reads
`HISTORY_ARCHIVE_ENABLED` **first** (before anything else can consume it — the parser
raises on garbage, and the next reader is `@_guild_op`-swallowed `push_history`, so
startup is the only loud place), creates the Redis pool, then branches. Enabled: it
**requires `POSTGRES_URL`** (it raises otherwise — an enabled bot running without the
archive would XADD onto an outbox nobody drains), constructs `PostgresHistoryArchive`
(lazy: no connection is made here, so startup never blocks on Postgres) and starts the
`HistoryOutboxDrainer`. Disabled (the default): `history_archive`/`history_drainer`
stay `None`, one INFO says so, a set `POSTGRES_URL` is explicitly ignored (the flag,
never URL presence, is consent), and a leftover outbox from an earlier enabled run
draws a WARNING naming the un-drained depth (never auto-deleted). Either way it then
loads the `src.musicbot` extension and fire-and-forgets `ytdlp_pool.prewarm()` so the
first `-play` doesn't pay worker-spawn + yt-dlp-import latency. `MusicBotApp.invoke` also
short-circuits `--help` anywhere in a command message straight to that command's help
embed, before voice checks or argument parsing.

`close()` order: `history_drainer.stop()` (final drain, needs Redis AND the archive) →
`history_archive.close()` — both skipped when the archive tier is off (the attrs are
`None`) → close Redis pool → `super().close()` → `ytdlp_pool.aclose()`
(10s join timeout, then `terminate_workers()` — an unbounded join measured 61s to exit)
→ `shutdown_telemetry()` via executor (blocking span flush, up to 30s). `close()` is
one-shot (`_teardown_started`) and **every step is individually guarded**: a hung
Postgres once made `archive.close()` raise, which skipped every later step permanently.

### The life of `-play` — the three-phase yt-dlp pipeline

The core performance design: metadata resolution, stream extraction, and playback are
three separate phases so queueing is instant and songs start with near-zero latency.

```
-play <input>
  │ cog_before_invoke: bind structlog ctx, open command span, get_mp() (creates+starts
  │                    MusicPlayer if absent), persist voice/text channel IDs to Redis
  │ validate_commands: author must be in a usable voice channel
  ▼
play():
  ├─ voice client paused + song live? ──► _interject_flow (resume_paused=False):
  │                                        "-play means play" — see interjection section
  ├─ parse_input (sources.py): single word → parse_url (youtube/spotify/soundcloud/
  │        any dotted domain → URLSource.OTHER, handed raw to yt-dlp); else ytsearch
  ├─ bot disconnected? front=True:
  │      • defer_playback() hold (gate stays shut so a Redis-restored queue head
  │        can't start while this input resolves)
  │      • join launched CONCURRENTLY with queue_source (no data dependency);
  │        any failure cancels join and runs full cleanup() (zombie-loop prevention)
  │      • wait_for_restore() BEFORE put_front — ordering is load-bearing: put_front
  │        LPUSHes Redis, restore_entries replays entries already on that list
  │        in-memory-only; inserting first would double-queue this song
  ▼
PHASE 1 — RESOLVE (enqueue time, instant on repeats):
  queue_source → YTDL.yt_source: check ytdl:source:{normalized query} (TTL 1h).
  Miss → ONE unified stream-opts extraction in the process pool returns identity
  AND a selected playable stream URL, so both the source cache and the stream
  cache are written from a single network round (probe first — see phase 2).
  Spotify track → title search; Spotify playlist → titles → YTSource ytsearch
  entries (resolved lazily at dequeue); YouTube playlist → flat extraction to
  QueueObjects. Enqueue via GuildQueue.put (batch=one round-trip for playlists).
  ▼
PHASE 2 — PREFETCH (background):
  • per-song prefetch_stream task at enqueue (skipped for bulk playlists — N
    concurrent extractions would mint URLs that expire before playback)
  • _prefetch_next_song: while song N plays, song N+1 is fully resolved AND its
    YTDL/FFmpeg source constructed, cached in ytdl:stream:{webpage_url}
    (TTL = min(URL expire − 30min, 30min) — YouTube revokes well before `expire`)
  • every candidate URL is PROBED with a plain no-Range GET (exactly how ffmpeg
    opens it; HEAD and ranged GETs lie about revoked URLs); only proven-playable
    URLs are cached
  ▼
PHASE 3 — STREAM (playback loop, usually zero extraction):
  loop(): gate open → dequeue → resolve (if YTSource) → yt_stream (cache hit →
  no yt-dlp call) → vc.play(YTDL) → atomic Redis start transaction →
  NP embed + 3s progress updater → spawn prefetch for next → play_next.wait()
  → history add, clear transient state, next iteration
```

The playback loop (`MusicPlayer.loop`, bottom of musicplayer.py) is the most delicate
code in the repo. Its bookkeeping invariants:

- `claim_outstanding` tracks an unsettled `queue.get()` so the outer exception handler can
  settle the claim, so `_cursor` never drifts.
- `try_commit_dequeue()` (under the queue mutex) detects "queue cleared while this song
  resolved" — the song is discarded and its FFmpeg subprocess `cleanup()`ed (leak
  otherwise).
- The Redis start write is `pop_queue_and_start_song` (MULTI/EXEC: LPOP + state HSET +
  now_playing HSET) so a crash can never observe the song absent from both the queue and
  `current_song_url`. Crash-recovered songs use `set_current_song_state` (no LPOP —
  they were never on the Redis list; `persisted=False` on the QueueObject encodes this,
  read only via `guild_queue.is_persisted()`).
- `play_start_epoch` is **backdated by the FFmpeg `-ss` start offset** so recovery math
  yields true audio position for `?t=` starts and double-crash recoveries.
- Stream-never-opened detection: `stream_failed = not song.produced_audio and
  play_error[0] is not None` — zero frames alone also describes a paused-parked song;
  an error alone also describes a mid-song death that earned its history entry. A dead
  stream drops the cached URL (`_handle_dead_stream`) and notifies the channel.
- Idle disconnect: `queue_get` times out at 300s; the playback gate itself times out at
  300s (a player built by a command that never connects must not leak forever) — unless
  a `defer_playback` hold is outstanding, which means a command is mid-join.

**`-resume` is the second cold-start path.** With the bot out of voice there is nothing
to un-pause — the paused song went with the voice client — but the queue outlives it in
Redis under a 24h TTL, so `-resume` joins the author's channel and lets that queue play.
It differs from `-play`'s cold path in two ways that are not stylistic: it inserts
**nothing** (so the `wait_for_restore`-before-`put_front` rule is moot, and the head it
describes is the song that plays), and it restores **before** joining rather than
concurrently, because there is no 1–4s extraction to hide the handshake behind and
joining first would park the bot in a channel for an empty queue. It refuses to reuse a
player failing `can_rejoin_cold()` (a song still held, or a gate already open, with no
voice client — an eject that never reached `on_voice_state_update`), rebuilding instead.
`max_concurrency(1, guild)` is load-bearing: two racing invocations both read
`voice_client is None`, so `validate_commands`' "already being used in channel X" check
cannot fire for either, and the second would move the bot to its own author's channel.

### Per-guild object graph

One `MusicPlayer` per guild, registered in `MusicBot.mps: dict[int, MusicPlayer]`
(`get_mp` creates + `start()`s lazily; `cleanup()` atomically pops — first caller wins,
concurrent callers no-op). Each player owns:

- `queue: GuildQueue`, `history: GuildHistory`, `store: Optional[GuildRedisStore]`
- tasks: `_player` (loop), `_prefetch_task`, `_restore_task`, `_progress_task`,
  `_pause_debounce_task`, plus `_background_tasks` (fire-and-forget via
  `spawn_background`)
- events: `play_next`, `_restore_complete`, `_playback_gate` (+ `_playback_holds`
  refcount for `defer_playback()`)
- NP host state: `_np_host_message` / `_np_host_own_embeds` / `_np_host_dedicated` /
  `_np_edit_lock`

`cleanup(guild)` cancels all five tasks BEFORE disconnecting (so the loop can't start
the next song mid-teardown), retires the NP host, disconnects voice, resets presence,
and — for an intentional stop — `clear_connection()` so `on_ready` skips recovery.

### GuildQueue: one deque and a cursor

A guild's queue is **one deque plus an index into it**, privately owned by `GuildQueue`,
mirrored to Redis:

| | | |
|---|---|---|
| `_items[:_cursor]` | claimed by a consumer, not yet settled | the "in-flight head" |
| `_items[_cursor:]` | pending | what `get()` hands out |
| `_wake` | `asyncio.Event`, set iff something is pending | I3 |
| Redis mirror | `guild:{id}:queue` list | the `is_persisted()` subset, in order |

The cursor is the boundary and NOT a per-item flag, because Redis retires entries by
LPOP — so in-flight items are necessarily a **prefix** (I6). This replaced an
`asyncio.Queue` + a parallel `deque` whose agreement had to be maintained by hand.

Rules encoded in the class (violating any of these corrupts the queue or Redis):

- Every multi-leg mutation (`put`, `put_front`, `clear`, `shuffle`, `remove`,
  `finish_failed_dequeue`) runs under one bulk-mutation mutex.
- A dequeue is **two-phase**: `get()` advances `_cursor`; the item and the Redis LPOP
  settle later via `try_commit_dequeue()` / `redis_pop_for()` (or are undone via
  `requeue_front()` / retired via `finish_failed_dequeue()`). `put_front` inserts at
  `_cursor`, which IS inserting behind the in-flight head.
- **`_sync_wake()` is the only writer of `_wake`**, and that is not style. A stale set
  does not degrade: `Event.wait()` returns without yielding when already set, so `get()`'s
  wait loop loses its suspension point and the whole event loop stops — measured at
  2,000,001 iterations with 0 other loop ticks. The wait is a `while`, never an `if`:
  `Event` wakes every waiter, and the prefetch's `get_nowait()` is a second consumer.
- **Every cursor decrement is guarded** (`try_release`, `requeue_front`). Unguarded it goes
  negative and the write that follows lands at `_items[-1]` — the TAIL. `clear()` resets it
  to 0 alongside the deque; without that, `qsize()` returns negative and the next release
  pops an empty deque. Tests assert all of this against the module source.
- **`qsize()` is PENDING, `display_size()` is pending PLUS in-flight.** One term apart over
  the same two fields, so a swap compiles and type-checks; `display_size()` is the sole
  input to `play_history.queue_position`, so a swap writes a plausible wrong number to
  Postgres forever.
- Callers with a prefetch task must `_cancel_prefetch()` BEFORE clear/shuffle/remove so
  the prefetch's `CancelledError` handler `requeue_front()`s its item into the drain.
- `clear()` sets a cleared-flag the loop consumes (`consume_cleared_flag`) to discard a
  prefetched song it is holding.
- `restore_crashed` / `restore_entries` write the deque ONLY (entries are already on /
  never were on the Redis list, respectively).
- Redis rebuilds (`rebuild_queue`) are MULTI DELETE+RPUSH so a concurrent LPOP never
  observes an empty-window queue.
- Every mirror write goes through `_write_mirror(items, *, removed=())`, which owns the
  rebuild / DELETE / LREM choice. Empty means DELETE, never skip. **Only a removal may
  pass `removed`** — LREM asserts the survivors kept their order, which is false for a
  shuffle or an insert — and only below `_LREM_MAX_ENTRIES` (200; the measured crossover
  against a real server is ~270, and the ~40 an earlier fakeredis estimate gave is wrong
  because a pipeline there costs what its commands cost).
- `remove()` takes a **predicate**, and `remove_matcher()` beside the class owns the
  policy: resolved yt-dlp URL first, then `user_input`. Links compare literally, text
  casefolds — folding a link would let one Spotify album's base62 id match another's.

### Redis schema and persistence model

All schema lives in `src/guild_state.py` (frozen `slots` dataclasses + field-name
constant classes; wire tables are spelled out explicitly so renaming a Python attribute
can never silently rename a Redis field). `GuildRedisStore` (redis_client.py) is the only
IO surface. The pool is created with `decode_responses=False` — readers `cast()` HGETALL
to `dict[bytes, bytes]` and decode in `from_redis()`; do not "simplify" this.

| Key | Type | TTL | Contents |
|---|---|---|---|
| `guild:{id}:state` | hash | 24h | voice/text channel IDs, `current_song_*` (a parked queue entry), `play_start_epoch`, `total_pause_seconds`, `pause_start_epoch`. Still *parses* a legacy `volume` field — see `:config` |
| `guild:{id}:queue` | list | 24h | JSON entries, `type` discriminator: `"qobj"` (SongQueueEntry) / `"ytsource"` (SearchQueueEntry — e.g. unresolved Spotify-playlist tracks). Both carry `user_input`, what the user typed; on a search entry it is the ONLY surviving record of the collection link, since its `ytsearch` is a generated title. Mirror writes all go through `GuildQueue._write_mirror` — rebuild, DELETE, or LREM |
| `guild:{id}:now_playing` | hash | 24h | display snapshot for `-now` / recovered embed (deleted wholesale on song end: empty == no song) |
| `guild:{id}:history` | list | **none, ever (PERSISTed)** | HistoryEntry JSON, most recently RECORDED first (~625 B/entry), LTRIMmed to `HISTORY_CACHE_LIMIT` (50) on every write. The ONLY source `-history` reads — bounded by length so it can be retained forever. Postgres is the durable record behind it |
| `history:outbox` | **stream** | **none, ever** | global write-ahead buffer, written only while the archive is enabled (disabled — the default — the key is never created): every play, all guilds interleaved, one `serialize_history_entry` blob per entry under field `e`, drained oldest-first into Postgres by the `drainers` consumer group. Non-evictable — an evicted entry is a silently lost play |
| `guild:{id}:config` | hash | **none, ever (PERSISTed)** | durable per-guild preferences (`GuildConfig`). Three fields today: `debug_mode` (`"1"`/`"0"`), `volume`, and `timezone` (an IANA name, resolved by `GuildConfig.tzinfo()` at read time so a name the host's tz database cannot resolve degrades to the default instead of raising on a render path). **Absent always means "no choice made"** — for debug that is "follow the host `DEBUG_MODE`", for volume it is "use the default", and keeping it distinct from an explicit `0`/`false` is why every field is Optional. `volume` MOVED here from `:state`, and the legacy field is **dual-written for one release rather than deleted** — deleting it made `just up <older-sha>` silently reset every migrated guild to 100%, since the older build reads only `:state`. Restore reads config-then-legacy and SEEDS config from what it finds (`migrate_volume`, `HSETNX` — never an overwrite, or a snapshot read before a concurrent `-volume` would durably clobber it). Drop the legacy write, `StateField.VOLUME` and `GuildStateData.volume` together after one release. Deliberately not fields on `:state`, which expires in 24h — a durable choice must not evaporate on an idle guild. Excluded from every TTL path; deleted on `on_guild_remove` |
| `ytdl:source:{query, lowercased}` | string | 1h | search → {webpage_url, title, duration, uploader, thumbnail} |
| `ytdl:stream:{webpage_url}` | string | ≤30m (expire-capped) | probed-playable stream URL + `_STREAM_CACHE_FIELDS` metadata |
| `leaderboard:v{n}:{guild_id}:{days}:{top_n}` | string | 60s | orjson aggregate cache for `-leaderboard`, one entry per requested window (`:0` = all-time). Keyed by row limit and codec version too, so neither can decode stale. TTL'd, so eviction-safe |
| `spotify:auth:token` | string | expires_in − 30s | raw bearer token (NOT orjson — deliberate) |
| `spotify:{track,playlist,artist,album}:{id}` | string | 24h/1h/24h/24h | cached lookups |
| `lock:guild:{id}:recovery` | string | 60s | SET NX EX distributed recovery lock |

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

`src/backfill_history.py` (`just db-backfill`, or `docker compose run --rm db-backfill`
on a host with no venv) walks every `guild:{id}:history` list and inserts it straight
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
  └─ per guild: _restore_guild (background task)
       ├─ skip if guild already in mps
       ├─ acquire lock:guild:{id}:recovery (SET NX EX 60) — rolling-restart safety
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
              • volume restored only if a value was stored (never clobber a concurrent
                -volume with a fabricated default)
              • crashed song: crashed_position_at(now) = elapsed − pauses, capped at
                cached duration − 10s (EOF guard), rebuilt via
                SongQueueEntry.from_crashed_state (persisted=False, interjected flag
                preserved) → queue.restore_crashed at the FRONT; state cleared
                unconditionally so a failed re-queue can't loop every restart
              • restore_entries() for the pending queue, history.restore(), refresh_ttl()
              • finally: _restore_complete.set() — loop() blocks on this before its
                first dequeue (otherwise it could LPOP Redis for the crashed head,
                silently deleting an unrelated still-queued song)
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

Both cold-start commands wait on the restore before touching the queue, bounded by
`RESTORE_WAIT_SECS` (musicbot.py): the pool sets `socket_connect_timeout` but no
`socket_timeout`, so a Redis that accepts the connection and then stalls would hang the
command outright. A restore that does not land is a reason **not** to insert — `-play`
front-inserting against an unread snapshot double-queues the song — so both abandon and
say so. `MusicPlayer.restore_read_failed` separates "nothing was saved" from "the store
could not be read"; only the first may be reported to a guild as an empty queue.

Known limitation (FIXME in guild_state.py): recovery counts **bot downtime** as playback
position — a song 30s in that stays down 10min resumes near its end (duration−10s cap).
The designed fix is a periodic position heartbeat, not yet implemented.

### `-playnow` interjection and resume entries

`MusicPlayer.interject(qobj, vc, resume_paused)` implements "play this now, then put the
interrupted song back where it was":

1. `_neutralize_prefetch()` first — a completed prefetch bypasses the queue and would
   play INSTEAD of the interjection. Claim-then-settle: `_prefetch_task` is nulled
   synchronously on both sides (interject and the loop) so exactly one consumer sees any
   given prefetch result. A completed prefetch is rebuilt into an equivalent QueueObject
   (carrying `-ss` offset and every flag), `requeue_front()`ed, and its FFmpeg subprocess
   killed.
2. Capture the current song's frame-counted `position_secs` (frozen during pause), build
   a resume `QueueObject(ts=position, is_resume=True, start_paused=was_paused &&
   resume_paused)` — skipped when < 5s remain (`_MIN_RESUME_REMAINING_SECS`), position
   EOF-capped at duration − 10s.
3. `queue.put_front([qobj, resume])` (both persisted → LPUSHed, so crash recovery
   mid-interjection works unchanged), then `vc.stop()` — only if the measured song is
   still current.
4. **Stacking**: interjecting over an interjection parks that song too, in front of the
   tails already waiting, so the queue unwinds LIFO and every parked song returns.
   Unbounded by design (each `-playnow` pays a 1–4s resolve first); `ts` is absolute at
   every level, so a tail of a tail resumes where it actually stopped. Depth rides the
   span as `interject.depth` (`GuildQueue.resume_tail_depth`), and `interjected` is now
   attribution only — its one behavioural read was the replace gate.
5. History: the interrupted song is recorded ONCE, when its resume tail finishes
   (`_skip_history_for` holds the song's identity, not a boolean — a stale flag would eat
   the next song's entry). One slot suffices at any depth: each interjection stops
   exactly one song, whose iteration consumes the marker before the next can land. A
   parked tail destroyed by `-clear`/`-remove` before it can play is recorded there
   instead (`MusicPlayer._flush_played`) — a queue object is recorded exactly once, when
   it leaves the queue for good. A song abandoned *mid-play* has no queue object at all
   (its entry was LPOPed at start), so `cog.cleanup` claims it synchronously before any
   await — `claim_current_song_for_history()` — and writes it alongside the teardown.
   That claim takes the same `_skip_history_for` marker, which is what keeps the two
   writers from both recording; it declines when the marker already names the song,
   because then a parked tail survives in Redis and records the play on `-resume`.

`-play` while paused routes through the same flow with `resume_paused=False` (the
interrupted song comes back PLAYING — "-play means play"); `-playnow` restores the exact
paused state (`start_paused` re-pauses the player thread synchronously at `vc.play`,
before any await, leaking at most a frame or two). Playlists collapse to their first
track for `-playnow`; plain `-play` front-inserts a playlist in full (nothing is playing
to keep waiting).

### The Now Playing host system

The NP card (embed with a 10-segment live progress bar, edited every
`NOW_PLAYING_UPDATE_INTERVAL_SECS` = 3.0s) stays glued to the bottom of the channel.
**The two live dashboards are the documented exception**: `-ping` and `-debug` reply
through `ctx.channel.send`, not `MusicContext.send`, because a message an edit loop
owns must not also be the NP host — the progress updater would re-render it every
3s. So those replies carry no NP block AND do not retire the current host, which
stays above them until the next ordinary `ctx.send` adopts a new one. Bypassing
`MusicContext.send` also bypasses debug-mode decoration, so the cog hands each
dashboard a pre-rendered `debug_suffix` instead — computed ONCE per invocation and
held constant, because the driver only edits when the render changes and a
per-tick-varying footer would edit the board until its deadline (which is why that
suffix omits elapsed-ms).
Mechanism: `MusicContext.send` (main.py) asks the guild's player for `np_embed_block()`
and **prepends it to every command response in the player's home channel** (≤ Discord's
10-embed cap; worst case here is 3), then `_adopt_np_host_if_current` makes that message
the new host and retires the previous one (dedicated NP message → deleted; command
response → strip-edited back to its own embeds). Attaching at send time makes response +
block one atomic message, so the bar is never momentarily buried. Song end: host is
released, one final edit completes the bar — only if the song truly reached its end
(`_reached_end`, 5s margin); skipped/interjected songs finalize at their true position;
a stream that never produced audio gets its block retired instead (a completed bar would
be a false record). Pause updates are debounced 0.5s.
An interjected fragment's frozen bar is the one case release-don't-retire leaves behind,
and a stack leaves one per interjection — so its resume tail carries a pointer to that
card (`np_message_id`/`np_channel_id`/`np_dedicated` on the wire, plus a runtime-only
`np_host_ref`) and disposes of it when the tail starts, **after** its own card is up.
Never a re-adopt (`_adopt_np_host` refuses older ids by design — the bar belongs at the
channel bottom); the channel id comes from `message.channel.id`, never the persisted
home channel; and capture is late-bound to the fragment's iteration end, because an id
read inside `interject()` can name a message the confirmation's own adopt already
retired. Only the runtime ref can strip-edit a response host, so the by-id path (post-
restart) is gated to dedicated cards — deleting a response would destroy a user's reply.

### yt-dlp: process pool, client strategy, caching, healing

**Process pool** (`ytdlp_pool.py`): extraction is half GIL-bound (JSON, signature
decryption, format selection), so it runs on a `ProcessPoolExecutor`
(`YTDLP_POOL_WORKERS`, default 4; ~80–120 MB RSS each). Lifecycle only — the callable is
supplied per call, which is the seam tests use. Lazy creation (workers re-import parent
modules under spawn); `prewarm()` from setup_hook; a `BrokenProcessPool` (e.g. OOM-killed
worker) is healed by rebuild-and-retry ONCE; worker logs travel a
multiprocessing Queue → parent `QueueListener` → the parent's handlers (so yt-dlp's
SABR/PO-token/signature warnings — the early-warning system for YouTube rule changes —
reach Loki structured, with `worker_id` and propagated `trace_id`). Results are made
picklable and small in the worker: exceptions flattened to `ExtractionError` (yt-dlp's
own exceptions carry live tracebacks and can't cross), successes `_slim_info`'d
(sanitize + drop `formats`/`thumbnails`/etc., commonly 100 KB–1 MB nobody reads).

**Client strategy** (comment block above `_EXTRACTOR_ARGS` in youtube.py):
`android_vr` primary (no PO token needed, audio-only formats), `web_safari` as a
*working* fallback — enabled by shipping Deno (`deno` extra) + yt-dlp-ejs (`default`
extra) for JS challenges, and the **bgutil PO-token sidecar** (compose service on :4416,
plugin pin in lockstep) for when YouTube enforces tokens on muxed formats. Format ladder
`bestaudio/best[height<=360]/best` — the 360p cap matters: on the muxed fallback rung,
plain `best` would stream ~120 MB of 1080p video per song just for ffmpeg's `-vn` to
discard. `_record_serving_format` warns once per format_id when serves degrade to
muxed/HLS (the observable symptom of the primary path being down). Degradation ladder is
designed so every rung lands on a previously-working configuration.

**Revoked-URL healing** (`_resolve_playable_stream`): a revoked URL fails in the worst
way — ffmpeg 403s and exits, discord.py reports "song finished", silence. So every URL
is probed pre-play; a revoked cached URL is dropped and re-extracted once; a URL revoked
in the seconds between probe and first read is caught post-hoc by `produced_audio` and
its cache entry invalidated. `_stream_url_ttl` reads `expire` from both query-string
(https formats) and path-segment (`/expire/<epoch>/`, HLS) forms, then caps at 30min.

**FFmpeg**: `YTDL(discord.FFmpegOpusAudio)` with
`-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5` and `-vn`; `?t=`/interject
seeks via `-ss`; volume via `-filter:a volume=` (which is why `-volume` applies from the
next song). `read()` counts frames → `elapsed_secs`/`position_secs` is the single source
of truth for every position surface (bar, presence, pause confirmation, history,
interject resume point) and freezes during any pause automatically.

### Spotify

Optional feature: both `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET` present →
`spotify_enabled()`. Status is a three-state `SpotifyStatus` (DISABLED / ENABLED /
INVALID): `cog_load` fire-and-forgets a startup probe (`validate()` — fresh
cache-bypassing token grant + fetch of a known track, 10s cap) and only a genuine
`SpotifyAuthError` (non-2xx grant, or 401/403 on a call) downgrades to INVALID; network
failures are inconclusive and leave it ENABLED. `_require_spotify()` at every dispatch
raises `SpotifyDisabledError` with a status-specific user-facing message. Client caches
the bearer token in Redis (TTL = expires_in − 30s, skipped if that margin would exceed
the token's life) and track/playlist lookups. Spotify content resolves to **YouTube
searches** (`"<name> <artist1> <artist2>"`); playlist tracks enqueue as lazy
`SearchQueueEntry`s resolved per-song at dequeue.

### Observability

structlog JSON to stdout always; when `OTEL_SDK_DISABLED` ≠ true, OTLP gRPC traces +
logs to `OTEL_EXPORTER_OTLP_ENDPOINT` (compose: Grafana LGTM — Tempo/Loki/Grafana at
localhost:3014, admin/admin). Every log line carries `environment`, `trace_id`/`span_id`
when in a span, and command context (`guild_id`, `user_id`, `command`) bound in
`cog_before_invoke` (which also opens a `command.<name>` span; `cog_after_invoke`
closes it, `cog_command_error` records onto it). `_DiscordGatewayFilter` drops
discord.py-internal HTTP spans. Redis and aiohttp are auto-instrumented. Spans embed
their `trace_id` in error-embed footers (`trace_footer`) so a user report can be joined
to a trace. `-ping` is a live-editing dashboard (1s tick, 3s deadline, env-tunable)
probing Discord/Redis/Spotify/Postgres/OTEL and reporting bot/yt-dlp/ffmpeg
versions; `max_concurrency(1, guild)`.

## Concurrency model — quick reference

Single asyncio event loop + these off-loop resources: the yt-dlp process pool,
per-song FFmpeg subprocesses, and discord.py's audio player **thread** (which is why
`_after_play` uses `call_soon_threadsafe` and why `start_paused` pauses synchronously).

Per-guild synchronization primitives and what they protect:

| Primitive | Protects |
|---|---|
| `GuildQueue._mutex` | the deque and its Redis mirror during bulk mutations; dequeue commits |
| `_playback_gate` (+ holds) | loop consuming the queue before a real voice connection / while `-play` resolves or `-resume` rejoins |
| `_restore_complete` | loop dequeuing before restore has injected the crashed head |
| `play_next` (Event) | song-end handoff from the audio thread |
| `_np_edit_lock` | concurrent NP message edits |
| `Spotify._auth_lock` | token refresh double-fire |
| `lock:guild:{id}:recovery` (Redis) | two instances recovering the same guild |
| `history:outbox` consumer group (Redis) | replaced the `history:drainer` lease. Not mutual exclusion — `XREADGROUP >` gives two drainers **disjoint** entries and `XACK` settles by ID, so a second drainer duplicates work instead of destroying plays it never inserted |
| `PostgresHistoryArchive._init_lock` | pool creation racing `close()` |
| `HistoryOutboxDrainer._stop_lock` | concurrent `stop()`s each running their own final drain |
| claim-then-null on `_prefetch_task` | exactly-one-consumer of a prefetch result (loop vs interject) |

One known, documented, accepted race remains open (ISSUE header in guild_queue.py): a
bulk mutation can land between `try_commit_dequeue()` releasing the mutex and the start
transaction's server-side LPOP, drifting memory and Redis by one entry. The sketched fix
(hold the mutex across the store dispatch) is described there — if you touch this code,
read that header first.

## Code conventions

- **Typing**: pyright `basic` + `reportMissingParameterType`/`reportUnnecessaryTypeIgnoreComment`
  as errors; ruff `ANN` rules enabled except `ANN401` (the `Any`s at the yt-dlp and
  discord.py boundaries are load-bearing and documented). `cast()` (not bare
  annotations) for assertions the checker can't verify — `grep cast(` is the audit trail.
- **Dataclasses**: schema/value objects are `frozen=True, slots=True, kw_only=True`;
  `kw_only` is deliberately load-bearing where adjacent same-type params could transpose
  (see `ExtractRequest`).
- **Serialization**: orjson everywhere on the wire (snowflake IDs must stay native ints —
  never route them through float); ujson only as aiohttp's `json_serialize` in spotify.py.
- **Errors to users**: commands wrap their body in try/except → `self._command_error(ctx,
  e, title=...)`, which logs with traceback, records the span, and renders an embed with
  a trace-id footer. `ExtractionError.user_message` is the only yt-dlp text safe to show
  (raw messages can carry yt-dlp's bug-report boilerplate).
- **Tasks**: fire-and-forget via `spawn_background(coro, tracked_set)` (auto-discard);
  cancel via `cancel_task()` (awaits, suppresses CancelledError). Never swallow your own
  coroutine's CancelledError (see `_typing_keepalive`'s comment for the pattern).
- **Command definitions** carry their own help copy: `brief`, `usage`, `help`, and
  `extras={"category", "examples", "note"}` — help.py renders from these, so a new
  command documents itself. Add it to `CATEGORY_COMMANDS` in help.py for ordering
  (unlisted commands land under "Other").
- **Durations** render via `fmt_duration` (`3:45`, `1:02:05`) everywhere — mixed clock
  formats between the bar, presence, and embeds was a real bug. Embed titles through
  `truncate_embed_title` (Discord 400s the whole send at >256 chars).
- ETA timestamps render in `America/Los_Angeles` (`_PST` in musicplayer.py).

## Testing

- Layout: one `tests/test_<module>.py` per src module (`telemetry.py` is the sole
  exception — it has no test file; `test_leaderboard.py` also owns the cog command
  that drives it, since splitting the renderer's tests from the command's would make
  a reader check two files to learn what one board looks like; `test_debug.py`
  likewise owns `MusicBot._debug_suffix` and the `-debug` card's end-to-end
  assertions, for the same reason — what the footer says and what puts it there are
  one behavior), plus `conftest.py` (shared fixtures/seams),
  `helpers.py` (builders), `test_context.py` (Discord context doubles). `config.py` and
  `telemetry.py` are the two intentionally-least-covered modules.
- **The yt-dlp seam** (autouse fixture `use_thread_ytdlp_pool`): every test runs
  extraction on an in-process ThreadPoolExecutor-backed `YtdlpPool`, because tests patch
  `src.youtube._ytdlp_extract` with MagicMocks that could never be pickled to a real
  worker. Both module-level names (`ytdlp_pool`, `_ytdlp_extract`) are resolved per call
  in `_run_extract` specifically to keep those patches working — don't capture them.
  Consequence: no test spawns worker processes; the pickle contract is asserted directly
  (`TestProcessBoundaryContract`), and one dedicated test spawns a real worker.
- **The suite runs archive-ENABLED, inverting the ship default**: a conftest autouse
  fixture pins `HISTORY_ARCHIVE_ENABLED=true` (next to the `POSTGRES_URL` scrub),
  because the enabled configuration exercises strictly more code and hundreds of
  existing assertions encode it. Disabled-mode behavior is covered by explicit tests
  that monkeypatch the flag per case — which wins over the fixture (same MonkeyPatch
  instance, later call). Don't "fix" the fixture to match the ship default.
- Redis in tests is `fakeredis`; Discord objects are `MagicMock(spec=...)` doubles.
  **fakeredis executes every stream command the outbox uses and gets five of them
  wrong**, all in the safe-looking direction (green tests, broken production): the
  `xtrim(approximate=True)` default trims exactly here and nothing on a real small
  stream; `XAUTOCLAIM`'s completion cursor is the last-scanned id rather than `0-0`;
  `XINFO GROUPS` `lag` is off by one and can go negative; `XADD` against a list raises
  `AttributeError` rather than `ResponseError`; `ref_policy` is unsupported. They are
  enumerated in `tests/test_redis_integration.py`'s docstring because they have to be
  known rather than discovered. What fakeredis *does* model faithfully is the tombstone
  shape `(id, {})`, so the P1-critical drain path is unit-testable.
- **The `pg` tier** (`tests/test_pg_integration.py`, marker `pg`) runs against a real
  `postgres:18-alpine` via testcontainers (`just test-pg`, needs Docker) or against
  `POSTGRES_TEST_URL` in CI. Excluded from the default run. Several invariants live ONLY
  there (ON CONFLICT dedup, the `-history` tie-break, the schema lock in both directions,
  `NOT VALID`'s treatment of legacy rows, and `play_history_rejected.payload` holding a
  NUL byte that `jsonb` and `text` both refuse),
  so a conftest hook fails `-m pg` outright if the tier is selected but disabled — an
  all-skipped tier used to exit 0 and look green.
- **The `redis` tier** (`tests/test_redis_integration.py`, marker `redis`) is the same
  shape against a real `redis:7-alpine` (`just test-redis`, or `REDIS_TEST_URL` in CI),
  and the conftest hook gates it identically. It exists because of the divergence list
  above: that an exact trim actually trims, that `WRONGTYPE` is a `ResponseError`, and
  that `XAUTOCLAIM`'s cursor is `0-0` are all things fakeredis answers **wrongly**
  rather than not at all. It also fails deliberately if the server reaches Redis 8.2,
  where `XTRIM ... ACKED` collapses the cap's hand-rolled ack-before-trim rule into one
  keyword.
- `pytest-timeout` sets a 120s per-test deadline. Several guards here are
  `asyncio.timeout()` calls whose removal makes a test HANG rather than fail; without
  the deadline that burns a CI job's full timeout and reports a cancellation.
- structlog is reconfigured per-session for readable output, and contextvars are cleared
  between tests (autouse).
- Run `just check` before pushing (the pre-push hook runs it). It is the contract for
  CI's lint and test jobs but NOT the whole pipeline: `just ci` adds the container job
  and both integration tiers; the runtime-image build and pip-audit run only in CI.
  `check` is a plain dependency list of six — `fmt-justfile pins fmt-check lint types
  test` — run in order, so it stops at the first failure. The four cheap ones cost
  ~1.3s combined, which is what lets the pre-push hook give them a status line each
  (pre-commit renders one line per hook, runs hooks sequentially, and buffers a hook's
  output until it exits, so line count is hook count). CI invokes `lint`/`types`/`test`
  individually rather than calling `check`, so its jobs fail independently of this
  ordering.
- Warnings are errors (see golden rule 11). Also note: running pytest from a **detached
  worktree** without `ENVIRONMENT` set dies at collection (config.py's git-branch
  RuntimeWarning is promoted to an error — documented TODO); `export
  ENVIRONMENT=development` first.

## CI/CD and deployment

`ci.yml` jobs: **resolve-env** (environment name + semver-validated version from
pyproject — single source for image and release tags) → **lint** (justfile fmt/parse,
pin agreement, ruff, pyright) and **test** (coverage + PR comment) and **container-test**
(suite inside the test image; deliberately runs with a read-only token — it executes PR
code) and **pg-integration** (the `pg` tier against a postgres service container) and
**redis-integration** (the `redis` tier against a redis service container) — both real
merge gates, `build` needs them → **build** (runtime stage; on branches it only validates the build; on main it
pushes three GHCR tags: immutable `sha-<commit>`, `latest`, and the bare pyproject
version) → **release** (tag + GitHub release on main). Concurrency: PR pushes supersede each
other; main commits each get their own group so no build is ever dropped.
`security.yml` runs pip-audit against `poetry.lock` (push + schedule).
`todo-to-issue.yml` converts TODO comments to issues — write new `TODO:`/`FIXME:`
markers with that in mind (multi-line context is picked up).

Docker: three-stage build (builder → test → runtime; runtime has ffmpeg, no Poetry).
Deploys are separate from builds — `just up <sha>` deploys any locally-present image tag
and refuses to build; dirty trees produce `<sha>-dirty.<digest>` tags so a tag never lies
about its commit. `just restart` restarts the existing container and does NOT pick up a
new image. Compose runs the bot with **host networking**; a named `ytdlp-cache` volume
persists yt-dlp's player-JS/challenge cache across restarts.

`GIT_SHA` is both the deploy tag and a build-arg baked into the runtime image, as an
`ENV` **and** an `org.opencontainers.image.revision` label — the ENV is the one the
process can read (labels are invisible from inside the container), which is what lets
`-debug` report the commit it is running. `build_runtime_image()` is the single
`--build-arg` seam; every caller must **export** `GIT_SHA` before calling it, and CI
passes `github.sha`. Not a seventh `just pins` pair: the value is derived, not
duplicated.

## Configuration reference (all env vars; `.env` for compose)

| Variable | Default | Notes |
|---|---|---|
| `DISCORD_TOKEN` | — | required; startup fails without it |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | — | both or neither; validated live at startup |
| `REDIS_URL` | `redis://localhost:6379` | bot runs degraded (no persistence/recovery) without Redis |
| `HISTORY_ARCHIVE_ENABLED` | `false` | **the consent gate for long-term storage** — `true` enables the Postgres archive tier (outbox writes, drainer, `POSTGRES_URL` requirement). Strict parse (`true/1/yes` / `false/0/no`, case-insensitive; unset/empty → false; garbage aborts startup, and `setup_hook` reads it FIRST so the ValueError cannot be swallowed by `@_guild_op`). Set together with `COMPOSE_PROFILES=archive` — the pair is documented in `.env.example` |
| `COMPOSE_PROFILES` | — | read by Docker Compose from `.env`, not by the bot: `archive` deploys `postgres` + `db-migrate`. `just down` names every profile (`--profile archive --profile metrics`) — a `down` with a profile inactive leaves its containers running; explicitly naming profiled services (`up -d redis postgres db-migrate`) auto-activates the profile. Three profiles exist: `archive` (postgres + db-migrate), `ops` (`db-backfill`, kept out of `up` entirely because it is run by hand — `docker compose run --rm db-backfill`), and `metrics` (`otelcol-metrics`, the docker_stats → Prometheus sidecar `-debug`'s Postgres cpu/mem row reads; it mounts the Docker socket, so it is opt-in) |
| `POSTGRES_URL` | — | **required while the archive is enabled**; `setup_hook` raises without it. Ignored (with an INFO) when disabled — the flag, never URL presence, is what enables archiving. Compose supplies it; `just run` derives it from the parts below |
| `POSTGRES_PASSWORD` | `password` | compose defaults it so a token-only archive-enabled `docker compose up` works; the bot warns loudly (startup ERROR + owner-only `-ping` row) while the default is in use AND the archive is enabled (`build_common.sh`'s preflight warns more widely: flag truthy OR the profile in `COMPOSE_PROFILES`, covering the profile-on/flag-off drift case where an idle default-credential postgres runs with the bot's warnings silenced), and `./setup_env.sh` generates a real one. Changing it after the volume is initialized needs `ALTER USER` — Postgres reads it on first init only. **`.env` is the only supported place it is set**; a per-install `POSTGRES_PASSWORD_FILE` was proposed and declined (see the comment above `DEFAULT_POSTGRES_PASSWORD` in config.py), and `using_default_postgres_password()` is scoped to the DSN shape that decision produces — do not re-add asyncpg's full resolution ladder |
| `POSTGRES_USER` / `POSTGRES_DB` | `musicbot` / `musicbot` | compose only; also the parts `just run`/`db-*` build a host DSN from |
| `POSTGRES_HOST_PORT` | `5432` | host-side published port, to dodge a local Postgres |
| `POSTGRES_MIGRATE_URL` | falls back to `POSTGRES_URL` | lets migrations run as a different (higher-privilege) role |
| `POSTGRES_STATEMENT_CACHE` | `100` | asyncpg `statement_cache_size`; set `0` behind a statement-rewriting pooler |
| `HISTORY_OUTBOX_MAX` | `0` (unbounded) | opt-in outbox ceiling, meaningful only while the archive is enabled. Dropping entries is real data loss; every drop logs ERROR |
| `ENVIRONMENT` | git branch (`main`→`production`) | set explicitly in CI/Docker/worktrees |
| `DEBUG_MODE` | `false` | process-wide default for debug mode, which decorates every embed the bot sends with a trace/timing/runtime footer. Three seams apply it, because "every embed" is sent from three places: `MusicContext.send` (command responses), `MusicPlayer._decorate_for_debug` (the NP block at every render site — refreshed on each progress tick — plus the player's own notices), and a pre-rendered `debug_suffix` threaded into the two live dashboards. Same strict parse as `HISTORY_ARCHIVE_ENABLED`, read ONCE by `MusicBot.__init__` so garbage aborts startup inside `load_extension`. `-debug --enable`/`--disable` override it **per guild, persisted to `guild:{id}:config`**, and require **Manage Server** (or bot ownership). The stored choice survives restarts and WINS over this variable, so a guild that opted out stays out when the host default flips on; a guild that never chose follows this value and keeps following it. Redis unavailable → the toggle applies in memory only and says so. The per-guild scope is scoping, not a trust boundary — it exists so enabling debug in one guild does not enable it everywhere. Observation-only — it changes what is shown, never what the bot does |
| `DEBUG_PROMETHEUS_URL` | — | Prometheus query API `-debug` reads the **postgres container's** CPU/memory from (the bot cannot see another container's cgroup, and Postgres reports no OS metrics over SQL). Compose sets `http://localhost:9090`; the series come from the `otelcol-metrics` `docker_stats` receiver, selected by `container_name="discord-postgres"`. **That collector is behind the `metrics` compose profile**, so on a default `up` it does not run and the cpu/mem rows render `n/a (no metrics source)` even though the URL is set and Prometheus answers — set `COMPOSE_PROFILES=metrics` (or `docker compose --profile metrics up -d`) as well. Unset URL → the same `n/a`. Only those two rows depend on it: the block's load/throughput/mem-signal rows are native SQL over the archive's own pool and render regardless. The container name is a hand-checked cross-file pin (see golden rule 6) |
| `PROMETHEUS_HOST_PORT` | `9090` | host-side published port for the metrics stack's Prometheus, loopback-bound. Also the port `DEBUG_PROMETHEUS_URL` defaults to — the two are written separately in compose (golden rule 6c) |
| `GIT_SHA` | — | the deploy tag, baked into the runtime image as an `ENV` (and a label). The ENV is the one the process can read, which is what lets `-debug` report the commit it is running; outside a container `-debug` shells out to `git rev-parse` instead |
| `POT_PROVIDER_URL` | `http://127.0.0.1:4416` | bgutil PO-token sidecar base URL |
| `YTDLP_POOL_WORKERS` | `4` | extraction worker processes (~80–120 MB RSS each) |
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | `3.0` | NP progress-bar edit cadence |
| `PING_TICK_SECS` / `PING_DEADLINE_SECS` | `1.0` / `3.0` | -ping live-edit loop |
| `DEBUG_TICK_SECS` / `DEBUG_DEADLINE_SECS` | `1.0` / `8.0` | -debug live-edit loop. Longer deadline than -ping's: each block does more work (the Postgres probe brackets a 2s sampling window between two stats queries, plus a Prometheus round trip) and a straggler renders `⚠️ timed out` rather than being retried — keep the deadline comfortably above that ~2.2s floor. The tick is a CEILING, not a cadence — the loop wakes on the first probe to finish |
| `OTEL_SDK_DISABLED` | `false` | `true` disables tracing/log export (stdout logs remain) |
| `OTEL_SERVICE_NAME` / `OTEL_EXPORTER_OTLP_ENDPOINT` | `discord-music-bot` / `http://localhost:4317` | |

## Known issues (tracked as in-code markers — read the marker before touching)

| Where | Marker | Summary |
|---|---|---|
| guild_queue.py (module header) | ISSUE | dequeue-commit ↔ Redis-LPOP race window; accepted, fix sketched |
| guild_state.py `crashed_position_at` | FIXME | bot downtime counted as playback position; heartbeat fix designed |
| redis_client.py `push_history` | ISSUE | non-evictable keys can OOM Redis and stall ALL writes. Only the OUTBOX can still get there — the history lists are capped per guild (~24 KB each), so their total scales with guild count, not runtime. `HISTORY_OUTBOX_MAX` is the opt-in bound on the outbox (and a disabled archive removes the outbox entirely); a memory alarm is still owed |
| guild_queue.py `shuffle` | FIXME | requires 4 songs but the user-facing message and help say 3 |
| spotify.py `playlist` | FIXME | playlists >100 tracks silently truncated (first page only, `next` cursor never followed) |
| sources.py `SoundcloudSource` | TODO | SoundCloud timestamp params ignored (YouTube-only `t`/`ts` parsing) |
| config.py `_git_branch` | TODO | detached-worktree pytest runs die at collection (warning→error) |
| youtube.py `yt_source` | TODOs | untyped `Exception("Could not find song")`; dead `download=True` param; no format validation on search results |
| musicbot.py `__init__` | HACK | `getattr(bot, "redis")` hides the MusicBotApp dependency from the type checker |
| musicbot.py `play` (playlist branch) | HACK | an `assert isinstance(source, YTSource)` stands in for a correlation the signature can't express — a `ResolvedYoutubePlaylist` always arrives with a `YTSource`, but they are separate parameters. `python -O` strips the assert and leaves the attribute reads unguarded; the fix is to have the `Resolved*Playlist` dataclasses carry their own source |
| musicplayer.py ETA zone | TODO | **Only the plumbing landed — the user-visible defect is open.** `queue_embed`'s "Est. playing at" and the NP "Estimated finish" read `GuildConfig.timezone`, but nothing WRITES it: `set_timezone` has no caller in `src/` and the `-options` command it was built for does not exist, so `ConfigField.TIMEZONE` is always absent and every guild still renders `DEFAULT_TIMEZONE` (US/Pacific), quoting users elsewhere a clock time that is not theirs. The `%Z` suffix is real and fixed a *different* bug — a hardcoded "PST" that was wrong the ~8 months a year US/Pacific spends in PDT. Two things owed: a write path, and per-VIEWER rendering (a guild-wide zone is still one clock for everyone in the guild). Fix for the second: Discord relative timestamps (`<t:epoch:R>`) |
| main.py `on_ready` | FIXME | "Bot commands:" log line actually logs an intent flag |
| redis_client.py `clear_connection` | HACK | dead `last_author_id` field still scrubbed; safe to delete after one release |
| musicbot.py `jump` | TODO | `-jump` is a stub ("in development") — implement or drop it from the command list |

## Recipes for common changes

**Add a command**: method on `MusicBot` with `@commands.command(name=..., aliases=...,
brief=..., usage=..., help=..., extras={"category": ..., "examples": [...], "note": ...})`;
add `@commands.before_invoke(validate_commands)` if it needs the author in voice; open a
span with `@_tracer.start_as_current_span("bot.<name>")`; body in try/except →
`_command_error`; every reply an embed; list it in help.py's `CATEGORY_COMMANDS`; tests
in tests/test_musicbot.py.

**Add a persisted per-guild state field**: constant in `StateField` → field with default
on `GuildStateData` + `from_redis` → the write-path method on `GuildRedisStore` (or
`_now_playing_state_mapping` if it's per-song) → decide whether it belongs in
`_TRANSIENT_SONG_FIELDS` / `clear_connection` → tests in test_guild_state.py and
test_redis_client.py.

**Add a per-guild SETTING** (a durable choice, not runtime state): constant in
`ConfigField` → `Optional` field on `GuildConfig` (Optional is not optional — absent
must keep meaning "follow the host default", or "never chose" collapses into "chose
the default") → `to_redis` writes it only when set → `from_redis` reads an
unrecognised value as unset → write method on `GuildRedisStore` that PERSISTs and
**encodes through `GuildConfig(field=value).to_redis()` rather than by hand** (a
single-field config serializes to exactly that field, so the wire format has one
definition and a setter cannot drift from what `from_redis` expects) → **validate at
the write boundary if the value is user-typed** (see `valid_timezone`: a bad value
stored here fails silently — the write succeeds, the command reports success, and the
guild keeps the default forever) → if a hot path reads it, cache it in memory and
hydrate in `_load_debug_overrides`'s shape rather than adding Redis IO to every send;
a multi-guild hydration must read through `read_guild_configs`, whose omission-on-
failure contract is what stops a Redis blink from deleting stored choices → tests in
test_guild_state.py and test_redis_client.py. It goes in `guild:{id}:config`, NOT
`guild:{id}:state`: that hash carries a 24h TTL and a setting stored there reverts on
any guild idle for a day.

**Add a queue-entry field**: `QueueEntryField` constant → `SongQueueEntry` field with
default → `from_queue_object`/`from_song`/`from_crashed_state` as applicable →
`to_redis` table → `parse_queue_entry` with `.get(..., default)` (old wire entries must
parse) → `QueueObject` + `GuildQueue._rehydrate` → **`YTDL.__init__`'s keyword, its
instance assignment, and `YTDL.from_queue_object` in `src/youtube.py`** — miss these three
and the field is silently dropped the moment the queue object becomes a playing song,
which is where every read of it happens → carry it through `_neutralize_prefetch`'s
rebuild if playback-relevant. If it is a DURABLE property of the play rather than of the
queue slot, it also needs `StateField` + `GuildStateData` + `_now_playing_state_mapping` +
`_TRANSIENT_SONG_FIELDS`, or a crash silently resets it (see `is_resume`/`start_paused`).

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

**Bump yt-dlp**: it is exact-pinned; if `bgutil-ytdlp-pot-provider` moves too, bump the
compose image tag in the same commit. After any dependency change, `just
test-image-rebuild` before `DOCKER=1` recipes. Watch `_record_serving_format` warnings
and the `_YtdlpLogger` warnings after deploy — they are the early-warning system for
YouTube-side changes.

**Touch the playback loop / queue**: re-read the module docstrings of guild_queue.py and
the loop() bookkeeping comments first; every claim, release, and Redis
LPOP is accounted for exactly once on every path (success, cleared, resolve-failure,
stream-failure, cancellation). test_musicplayer.py (6.5k lines) and test_guild_queue.py
encode these paths — run `just test tests/test_musicplayer.py tests/test_guild_queue.py`
early and often.
