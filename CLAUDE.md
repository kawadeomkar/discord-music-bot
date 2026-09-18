# CLAUDE.md

Guidance for Claude Code when working in this repository. Everything here was derived
from the code itself — module docstrings and comments in this codebase are unusually
detailed and are the authoritative record of design decisions and past incidents.

## Project overview

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

| | |
|---|---|
| Language / runtime | Python **3.14+** (`requires-python = ">=3.14,<4.0"`) |
| Package manager | Poetry 2.x (`poetry.toml`, in-project venv) |
| Task runner | `just` (justfile is the index of every dev command) |
| Discord | discord.py 2.7.1 (exact pin), prefix commands (`-`), voice via PyNaCl/FFmpeg |
| Extraction | yt-dlp 2026.8.18.122307.dev0 (exact pin, a **nightly** — see the note on the pin; extras `[default, deno]`) in a **ProcessPoolExecutor** |
| Runtime state | Redis 7 (redis-py asyncio), orjson as the project-wide wire codec |
| Durable history | Postgres 18 + asyncpg (no ORM); migrations in `migrations/`, applied by `src/db_migrate.py` |
| Observability | OpenTelemetry (OTLP gRPC) + structlog JSON; Grafana LGTM stack in compose |
| Tests | pytest + pytest-asyncio (`asyncio_mode = "auto"`) + fakeredis + pytest-timeout; ~4,900 passing tests (this figure is always the PASSING count, not the collected one) plus two opt-in integration tiers (testcontainers): a 99-test `pg` tier and a 58-test `redis` tier; coverage gate `fail_under = 80` (actual ~96%) |
| Lint/types | ruff 0.15.21 (format + lint) and pyright 1.1.411 (exact pins) |

Entry point: `just run` (loads `.env`) or `poetry run bot` → `src.main:main`.
**`POSTGRES_URL` is required while the archive is enabled** — `setup_hook` refuses to
start an enabled archive without it. Disabled (the default), no Postgres is needed.

## Golden rules — read before editing anything

1. **The comments are load-bearing, and short.** Docstrings and comments here document
   invariants, race windows, measurements, and "do not simplify this" traps (e.g.
   `decode_responses=False` casts, the in-flight-head branch in `GuildQueue.put_front`).
   Never delete or contradict a comment without updating the behavior it describes; when
   you change behavior, update the comment in the same edit.
   **Describe only what is implemented — never the road not taken.** No rejected
   alternatives, no "this used to…", no comparison against deleted code or against
   `main`: git carries that, and a reader forced to hold two models is worse off than
   one told nothing. Scope a comment to the lines under it, target 3-4 lines, and state
   the fact without editorializing about its importance. A comment that needs more is
   the signal `docs/ARCHITECTURE.md` is missing a section — write it there and link the
   anchor (rule 2). Where a rejected alternative must be recorded, it goes in the commit
   message or ARCHITECTURE.md, never at the call site.
   The test: delete every clause naming something the code does not do; if what is left
   still serves the next few lines, the clauses were noise.
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
6. **Version pins move in lockstep.** Bump both halves in the same commit. `just pins`
   enforces eight pairs, one name and one list — it is a dep of `check` and CI also runs it as
   its own step, deliberately: Dependabot's `pip` and `pre-commit` ecosystems open
   SEPARATE PRs that each move one half, and those PRs are validated by CI and never
   by a local `check`.
   The eight: the ruff pin (pyproject) ↔ the ruff hook `rev` in
   `.pre-commit-config.yaml`; the image name (justfile `IMAGE` ↔ `build_common.sh`
   `IMAGE_NAME`); and `postgres:18-alpine` / `redis:7-alpine` each across three files —
   the integration tier's `_PG_IMAGE`/`_REDIS_IMAGE`, `ci.yml`'s service container, and
   `docker-compose.yml` (compared tier↔ci and compose↔ci, so all three agree); and
   `_POSTGRES_CONTAINER` (`src/debug.py`) ↔ the postgres service's `container_name`,
   which is a Prometheus label selector, so a rename there would otherwise leave
   `-debug`'s cpu/mem row reading `n/a (no metrics source)` forever rather than
   failing. The compose legs are anchored to the named service, not `head -1`, so a
   second postgres or redis service cannot silently shift what is compared.
   The eighth is the **yt-dlp version** (pyproject) ↔ the copies quoted in prose by
   `CLAUDE.md` and `docs/ARCHITECTURE.md`, which describe the client strategy for a
   specific version: Dependabot moves pyproject + `poetry.lock` without touching either,
   and main has carried a stale copy for exactly that reason.
   The **name** is the `charts` extra: `[tool.poetry.extras]` defines it, and three
   sites select it (the `CHART_EXTRAS` ARG default and the test stage in `Dockerfile`,
   and `just install`). Each site is asserted separately rather than counted — the
   Dockerfile names it twice, so a count lets a typo in either hide behind the other.
   Poetry IGNORES an unknown extra, so drift builds green and ships an image whose
   charts are silently absent. See `docs/ARCHITECTURE.md#the-charts-extra`.
   The **list** is the pre-push gate: `check`'s dependency list ↔ the five
   `entry: just <recipe>` pre-push hooks in `.pre-commit-config.yaml`, the same
   recipes in the same order. Drift runs one way and reports green — a step added to
   `check` alone simply stops running on push.
   **Five pairs are NOT enforced — this list is what a maintainer checks by hand,
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
   (d) `MPLCONFIGDIR`, written in **three** places that must agree on a WRITABLE path:
   the Dockerfile's test stage (`/tmp/mplcache`, beside `RUFF_CACHE_DIR`), its runtime
   stage (`/home/app/.cache/matplotlib`, created and chowned in the same `RUN` as
   `useradd`), and `tests/conftest.py` at module scope. The three deliberately hold
   DIFFERENT paths — what must agree is that each is writable by the uid that runs
   there, which is why this cannot be a `just pins` string comparison. Unwritable is
   the failure, and in the two Dockerfile copies it is quiet: matplotlib falls back to
   a temp directory and warns once per process, so the symptom is a stderr line nobody
   reads. The conftest copy is the exception — the suite renders in-process, so rule
   11 turns that warning into a red build.
   (e) `LIVENESS_INTERVAL_SECS`'s `maximum` (`_MAX_LIVENESS_SECS`, 60s, `config.py`) ↔
   the `HEALTHCHECK`'s 90s staleness window (`Dockerfile`). The cap exists so a touch
   cadence can never outlast the window; raise the window and the cap may follow. Lower
   it below 60s and lower the cap with it, or a cadence the cap still accepts outlasts
   the window and that container reports unhealthy. No build compares them.
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

13. **The chat-command surface is not a SemVer API.** Adding, renaming or REMOVING a
    command is a minor bump, not a major one: the version is a deploy tag for a
    self-hosted bot, nothing links against these names, and CI validates only that the
    string is semver-shaped. `-playnow` and `-playnext` survive as spellings of
    `-p --now` / `-p --next`, and renaming them would be minor too. A removal still owes users an
    `## Upgrading to <version>` section in README.md — nothing else records the
    migration, and the release notes are minted from the tag.

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
just pins           # assert the eight duplicated version/name pins ~0.02s
just types          # pyright over src/ AND tests/                ~6s
just test           # full suite, PARALLEL (-n auto), coverage gated (fail_under=80) ~35s
just test-report    # `test` + the coverage/JUnit artifacts CI's PR comment consumes
just check          # fmt-justfile + pins + fmt-check + lint + types + test  ~38s
just test-pg        # opt-in real-Postgres tier (testcontainers, needs Docker) ~45s
just test-redis     # opt-in real-Redis tier (testcontainers, needs Docker)     ~15s
just container-test # build test image, run suite inside it (spec cache OFF) ~1min
just ci             # check + container-test + test-pg + test-redis — local mirror of CI

# Test selection (args forward to pytest). ANY argument means a subset run, so it runs
# SERIALLY and coverage is skipped — fail_under is a PROJECT floor and one file measures
# ~26%, which used to fail a green run with exit 1. The gate rides the no-args form —
# what `just check` and the pre-push hook invoke — and `test-report`, whose arguments are
# reporting flags rather than a selection, keeps it with COVERAGE_GATE=1.
#
# The no-args form is also the ONLY parallel one (`-n auto`), and that is deliberate:
# the gate is the only way the whole suite runs, so a test that is not parallel-safe
# fails the pre-push hook and CI instead of rotting a separate "fast" recipe. A subset
# stays serial because worker startup (~4s flat) cannot amortize over a narrow
# selection, and because execnet does not forward worker stdout — `-s` is silently
# swallowed under `-n` and `--pdb` disables it. `just test tests/` is the escape hatch:
# the whole suite, serially, to reproduce a parallel-only failure.
just test tests/test_youtube.py
just test -k spotify
just test --maxfail=1

# Database (operator tools. db-migrate/db-backfill run the LOCAL venv against
# POSTGRES_URL; setup/backup/restore are shell around pg_dump/psql, no venv needed)
just setup                 # bootstrap .env with a generated POSTGRES_PASSWORD
just db-migrate            # apply pending migrations — REQUIRED before the bot serves
just db-backfill [--dry-run] # move pre-archive Redis history into Postgres — deadline: rules/state-and-recovery.md
just db-backfill-docker    # same, via the compose one-shot — no local venv
just db-rejects [n]        # list play_history rows Postgres refused (expected: nothing)
just outbox [idle_ms]      # outbox health: depth, in-flight, stranded, TOMBSTONES (lost plays)
just bot-settings [reset <application_id>] # list stored bot overrides, or delete one bot's
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

Run the bot locally: `just setup`, then `just services` (Redis always; Postgres +
`db-migrate` only when `.env` sets `HISTORY_ARCHIVE_ENABLED=true`), then **`just run`**.
`just compose <args>` is raw compose with the archive profile derived from that flag. Use `just run` rather than `poetry run bot`: the bot reads only
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
├── musicbot.py       # MusicBot cog — command REGISTRATION and one try/except each;
│                     # per-guild player registry (mps), the discord.py hooks, crash-recovery entry
├── musicplayer.py    # MusicPlayer — per-guild playback loop, prefetch (ensure_prefetch), gate,
│                     # NP host, ETA, interject, the hooks -replay drives (hand_over_to_replay),
│                     # and still_live, the one liveness test both interrupts use
├── play_placement.py # -play's flag grammar, its voice gate, the two bounds on the resolve
│                     # (ResolveSlot's deadline on the WAIT for a slot, and slow_resolve_notice
│                     # saying so past it), and PlayRegistry: the per-guild
│                     # in-flight set and the place lock its inserts serialize on (the cog
│                     # keeps the commands; the grammar and the registry are tested in
│                     # test_play_placement.py, the placement itself in commands/test_play.py)
├── guild_queue.py    # GuildQueue — one deque + cursor, the mirror writer, bulk-mutation mutex
├── guild_history.py  # GuildHistory — played-song history (capped Redis list + in-memory cache; writes feed the outbox while the archive is enabled, reads never touch Postgres) and its embeds; the command body is commands/history.py
├── history_archive.py# Postgres archive (asyncpg) + HistoryOutboxDrainer (outbox → play_history)
├── recovery.py       # Voice-session lifecycle: rejoin after restart (crash recovery), the alone-in-channel leave watchdog, and the two cold-start helpers -play and -resume share
├── commands/         # ONE MODULE PER COMMAND — commands/<command>.py, each exposing
│                     # run(). The cog holds registration and one try/except; every
│                     # body is here. _common.py holds the restore guard three of the
│                     # queue commands share. A command with a domain module (history,
│                     # leaderboard, debug, ping, analytics) keeps only its entry point
│                     # here and imports the machinery
├── play_pipeline.py  # the machinery behind -play/-playnow: resolve, place, interject
├── leaderboard.py    # -leaderboard tunables, Redis result-cache codec, embed renderer (pure;
│                     # the command body is commands/leaderboard.py)
├── analytics_card.py # -analytics, everything but the command and the figure. Pure half:
│                     # --days allowlist, both cache codecs, the embed — every
│                     # human-authored string renders HERE, never in the image. IO half:
│                     # the chart pool, the PNG cache, the send
├── analytics_render.py # the six-panel figure; imports matplotlib INSIDE build_figure and
│                     # constructs nothing at module scope (a worker re-imports it)
├── chart_pool.py     # the chart pool's only home: one YtdlpPool(max_workers=1). Exists so
│                     # main.py/debug.py/conftest.py have a name to resolve per call
├── db_migrate.py     # SQL migration runner (`python -m src.db_migrate`, EXPECTED_SCHEMA_VERSION)
├── backfill_history.py # ONE-SHOT operator script: pre-archive Redis history → Postgres, direct
│                     # (not via the outbox). Run BEFORE deploying this build — see rules/state-and-recovery.md
├── guild_state.py    # Pure Redis schema: frozen value objects, field constants, orjson wire formats
├── redis_client.py   # Connection pool, GuildRedisStore (@_guild_op), cache helpers, recovery lock
├── youtube.py        # yt-dlp integration: caches, stream probe/heal, YTDL audio source, worker fn
├── ytdlp_pool.py     # ProcessPoolExecutor lifecycle: lazy spawn, break-healing, worker log plumbing
├── sources.py        # Input parsing → YTSource / SpotifySource / SoundcloudSource; mints query_source;
│                     # is_mix (a YouTube Mix, which yt-dlp walks window by window)
├── spotify.py        # Spotify Web API client (client-credentials, Redis-cached); the playlist
│                     # pager, its walk slot and single flight
├── help.py           # man(1)-styled embed -help command (copy lives on the commands themselves)
├── dashboard.py      # optimistic-send + live-edit driver shared by -ping and -debug;
│                     # LiveMessage (send/edit-on-change/floor) is also the card's
├── queue_progress.py # the live card a slow COLLECTION enqueue shows while it resolves:
│                     # the phase model, the renderer, and the context manager that
│                     # owns the delay, the driver task, the send, N edits and the
│                     # delete. Two entry points (play.py and play_pipeline)
├── ping.py           # -ping health dashboard: probes + render (sequencing is dashboard.py)
├── debug.py          # -debug snapshot machinery and DebugSettings; OBSERVATION-ONLY by rule
│                     # collectors are live-edit probes (dashboard.py); host blocks are owner-only
├── telemetry.py      # OTel traces+logs, structlog config, worker logging, gateway span filter
├── config.py         # ENVIRONMENT (env var; main() may infer it from the git branch), SpotifyStatus,
│                     # tunables and their override accessors
├── settings.py       # the -settings machinery: the registry of what chat may set (bounds,
│                     # rendering) and the grammar that parses a request, both pure;
│                     # GuildSettings, the cache and ONLY writer of guild:{id}:config; and
│                     # BotSettings, which applies the operator's stored bot overrides. What the
│                     # environment holds is config.py; this module decides what chat may change
├── settings_card.py  # -settings' embeds (pure): server card, bot card, detail, every reply;
│                     # the command body is commands/settings.py
└── util.py           # logger factory, embed helpers (safe_label, verbatim_code),
                      # fmt_duration/fmt_seconds, progress_bar/progress_line (the NP bar and
                      # the card's), task helpers (spawn_background, cancel_task, join_task,
                      # set_within), channel_claim, ProgressFn, PoolSlotUnavailable,
                      # is_operator (the owner check -debug and -ping share)

migrations/           # NNNN_*.sql, applied in numeric order; the ONLY source of schema
docs/ARCHITECTURE.md  # the only tracked file under docs/ — anchor target for comments (rule 2)
tests/                # one test_<module>.py per src module, commands/ mirroring src/commands/,
                      # + conftest.py (seams) + helpers.py + mock_spec_cache.py
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
startup is the only loud place) and `BOT_SETTINGS_OVERRIDES` second, for the same
reason; creates the Redis pool and `MusicBotApp.bot_settings`, then branches. Enabled: it
**requires `POSTGRES_URL`** (it raises otherwise — an enabled bot running without the
archive would XADD onto an outbox nobody drains), constructs `PostgresHistoryArchive`
(lazy: no connection is made here, so startup never blocks on Postgres) and starts the
`HistoryOutboxDrainer`. Disabled (the default): `history_archive`/`history_drainer`
stay `None`, one INFO says so, a set `POSTGRES_URL` is explicitly ignored (the flag,
never URL presence, is consent), and a leftover outbox from an earlier enabled run
draws a WARNING naming the un-drained depth (never auto-deleted). Either way it then
loads the `src.musicbot` extension, spawns `BotSettings.hydrate_until_read()` without
awaiting it (every knob runs on its env value until it lands; a failed read retries with
backoff, and the bot card and `-debug` say while it has not landed —
`docs/ARCHITECTURE.md#settings-resolution`), and fire-and-forgets
`ytdlp_pool.prewarm(warm_worker)` — then `chart_pool.warm()`, but only while the
archive is enabled, so a default deployment never spawns the matplotlib worker — so
the first `-play` doesn't pay worker-spawn, yt-dlp-import or first-`YoutubeDL`
latency (the pool stays lifecycle-only: the warm-up callable comes from
`src.youtube`, like every other callable it runs). `MusicBotApp.invoke` also
short-circuits `--help` anywhere in a command message straight to that command's help
embed, before voice checks or argument parsing.

`close()` order: `history_drainer.stop()` (final drain, needs Redis AND the archive) →
`history_archive.close()` — both skipped when the archive tier is off (the attrs are
`None`) → close Redis pool → `super().close()` → `ytdlp_pool.aclose()`
→ `chart_pool.aclose()` (inert when the worker was never spawned)
(10s join timeout, then `terminate_workers()` — an unbounded join measured 61s to exit)
→ `close_probe_session()` (latches the module closed, so a player loop still running
during the flush below cannot rebuild a session nothing will close)
→ `shutdown_telemetry()` via executor (blocking span flush, up to 30s). `close()` is
one-shot (`_teardown_started`) and **every step is individually guarded**: a hung
Postgres once made `archive.close()` raise, which skipped every later step permanently.

### Subsystem detail — `.claude/rules/`

The rest of the architecture, the concurrency primitives and the configuration
reference live in path-scoped rule files, each loaded when Claude reads a file its
`paths:` frontmatter names. Read one directly to reason about a subsystem without
opening its files. `.gitignore` excludes `.claude/*` except `rules/`.

| Rule file | Covers |
|---|---|
| `.claude/rules/playback.md` | the life of `-play` and the loop's bookkeeping, the per-guild object graph, `GuildQueue`, `--now`/`--next` and resume entries, the Now Playing host; the queue, placement and playback primitives |
| `.claude/rules/state-and-recovery.md` | the Redis schema, the history backfill, crash recovery; the restore, archive and outbox primitives |
| `.claude/rules/extraction.md` | the yt-dlp pool, client strategy and stream healing, Spotify; the extraction and playlist primitives |
| `.claude/rules/config.md` | every environment variable, its default and its bounds; the settings primitives |

### Observability

structlog JSON to stdout always; when `OTEL_SDK_DISABLED` ≠ true, OTLP gRPC traces +
logs to `OTEL_EXPORTER_OTLP_ENDPOINT` (compose: Grafana LGTM — Tempo/Loki/Grafana at
localhost:3014, admin/admin). Every log line carries `environment`, `trace_id`/`span_id`
when in a span, and command context (`guild_id`, `user_id`, `command`) bound in
`cog_before_invoke` (which also opens a `command.<name>` span; `cog_after_invoke`
closes it, `cog_command_error` records onto it). `_DiscordGatewayFilter` drops
discord.py-internal HTTP spans. Redis and aiohttp are auto-instrumented. Spans embed
their `trace_id` in error-embed footers (`trace_footer`) so a user report can be joined
to a trace. **`player.loop.iteration` is a ROOT span**, so one song is one trace — the
loop task inherits the context that created the player, and an inherited parent files
every song a guild ever plays under one `-play`. Its id is captured into
`MusicPlayer._playback_span` at the song's start and printed on the Now Playing card
and the playback-error notice, which is why both name the same trace. `-ping` is a live-editing dashboard (1s tick, 3s deadline, env- or owner-tunable)
probing Discord/Redis/Spotify/Postgres/OTEL and reporting bot/yt-dlp/ffmpeg
versions; `max_concurrency(1, guild)`.

## Concurrency model — quick reference

Single asyncio event loop + these off-loop resources: the yt-dlp process pool,
per-song FFmpeg subprocesses, and discord.py's audio player **thread** (which is why
`_after_play` uses `call_soon_threadsafe` and why `start_paused` pauses synchronously).

Per-guild synchronization primitives and what they protect are tabled in the
`.claude/rules/` file for their subsystem (see Subsystem detail above).

## Code conventions

- **Typing**: pyright `basic` + `reportMissingParameterType`/`reportUnnecessaryTypeIgnoreComment`
  as errors; ruff `ANN` rules enabled except `ANN401` (the `Any`s at the yt-dlp and
  discord.py boundaries are load-bearing and documented). `cast()` (not bare
  annotations) for assertions the checker can't verify — `grep cast(` is the audit trail.
- **Annotations are never quoted** (ruff `UP037`, autofixed by `just fmt`). Python 3.14
  evaluates them lazily (PEP 649), so a `TYPE_CHECKING`-only name is legal unquoted,
  and a quoted one is just a string to an IDE — no go-to-definition, no rename. The
  rule holds because nothing here RESOLVES annotations at runtime: adding
  `get_type_hints`, pydantic or attrs would reintroduce the constraint the quotes used
  to satisfy. discord.py DOES evaluate command-callback parameters to pick converters,
  but those name runtime imports either way.
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
  cancel via `cancel_task()` (awaits, suppresses CancelledError); **join** a task told to
  stop by a SIGNAL via `join_task()`, which shields it — `await task` makes the joined
  task the canceller's `_fut_waiter`, so an unshielded join cancels the very task it is
  waiting out, mid-cleanup. Never swallow your own
  coroutine's CancelledError (see `_typing_keepalive`'s comment for the pattern).
- **Command definitions** carry their own help copy: `brief`, `usage`, `help`, and
  `extras={"category", "examples", "note"}` — help.py renders from these, so a new
  command documents itself. Add it to `CATEGORY_COMMANDS` in help.py for ordering
  (unlisted commands land under "Other").
- **Durations** render via `fmt_duration` (`3:45`, `1:02:05`) everywhere — mixed clock
  formats between the bar, presence, and embeds was a real bug. The exception is a
  `-settings` seconds-kind value, which renders through `util.fmt_seconds` (`3s`); every
  setting value renders through the registry's `format_value` and parses back through
  its one grammar. Embed titles through `truncate_embed_title` (Discord 400s the whole
  send at >256 chars).

## Testing

- Layout: one `tests/test_<module>.py` per src module. A command's tests live with its
  BODY, and drive it through the cog's wrapper — the wrapper resolves the player and
  owns the try/except, so a body that raises and a body that reports are different
  behaviours and only the pair is the command. `tests/commands/test_<command>.py`
  mirrors `src/commands/`, and needs its `__init__.py`: `tests/` is a package, so a
  subdirectory without one collides with a same-named file above it.
  (`test_leaderboard.py` also owns
  the cog command that drives it, since splitting the renderer's tests from the
  command's would make a reader check two files to learn what one board looks like;
  `test_debug.py` likewise owns `MusicBot.debug_suffix` and the `-debug` card's
  end-to-end assertions, for the same reason — what the footer says and what puts it
  there are one behavior; `play_placement.py`'s grammar and registry are tested in
  `test_play_placement.py`, the placement itself in `commands/test_play.py`),
  plus `conftest.py` (shared fixtures/seams),
  `helpers.py` (builders), `test_context.py` (Discord context doubles). `config.py` is
  the intentionally-least-covered module.
  `test_telemetry.py` restores structlog's PROCESS-wide configuration itself, because
  conftest's `configure_structlog_for_tests` is session-scoped and `setup_telemetry()`
  reconfigures structlog for real — without that restore the production JSON chain
  would stand for every test that runs after it.
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
- **Bot knobs in tests** are set with `config.set_override(<knob>, value)` in the test body.
  Don't patch a consumer module's copy (there is none), and don't use `monkeypatch.setitem` on
  the private maps (pyright does not check that value). The autouse `clear_bot_knob_overrides`
  clears every override after each test. `monkeypatch.setattr(config, "<KNOB>", v)` patches the
  env **baseline**, which an override shadows. Use it only for tests about the baseline.
- Redis in tests is `fakeredis`; Discord objects are `MagicMock(spec=...)` doubles,
  built through the spec cache `tests/conftest.py` installs at import — so **a spec
  class must not be mutated once it has been used as a spec** (`functools.wraps` on
  the replacement keeps a class-level patch payload-neutral). It is the one file
  outside `src/` the coverage gate measures. **The container tier runs with the cache
  OFF** (`MOCK_SPEC_CACHE_DISABLE=1` in the Dockerfile's test stage), so
  `container-test` is a reference run against stock `unittest.mock` and the two tiers
  disagree if the cache ever answers what upstream would not; the cache's own tests
  skip themselves there and run in the venv tier. See
  `docs/ARCHITECTURE.md#the-mock-spec-cache`.
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
  `check` is a plain dependency list of five — `fmt-justfile pins fmt-check lint
  check-heavy` — whose first four run in order and stop at the first failure.
  `check-heavy` is the exception: it runs `types` and `test` concurrently and reports
  both outcomes, so a pyright failure no longer hides what pytest would have said.
  The four cheap ones cost ~1.3s combined, which is what lets the pre-push hook give
  them a status line each (pre-commit renders one line per hook, runs hooks
  sequentially, and buffers a hook's output until it exits, so line count is hook
  count); fusing exactly the two slow ones is what keeps that affordable. The five
  pre-push hooks mirror those five dependencies in order and `just pins` asserts it.
  CI invokes `lint`/`types`/`test` individually rather than calling `check`, so its
  jobs fail independently of this ordering.
- Warnings are errors (see golden rule 11). `ENVIRONMENT` is read from the environment
  alone at import (default `development`), so collection runs no git subprocess and a
  detached worktree needs nothing set.

## CI/CD and deployment

`ci.yml` jobs: **resolve-env** (environment name + semver-validated version from
pyproject — single source for image and release tags) → **version-bump** (pull
requests only: that version must be strictly above the base branch TIP's, compared as
major.minor.patch, so every merge moves it. A PR opened by `dependabot[bot]` is exempt
and the job reports success anyway — Dependabot does not set the project version, so the
rule could only ever fail it. Nothing else enforces the per-PR bump —
`release` treats an unchanged version as the ordinary no-op. It is deliberately absent
from `build`'s `needs`: a job `if`-skipped on push would skip `build` with it, and it
blocks a merge only once branch protection lists it as required) → **lint**
(justfile fmt/parse, pin agreement, ruff, pyright) and **test** (coverage + PR comment) and **container-test**
(suite inside the test image, with the mock spec cache OFF so it is the reference run
against stock `unittest.mock`; deliberately runs with a read-only token — it executes PR
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

## Configuration reference

Every environment variable, its default and its bounds: `.claude/rules/config.md`.

## Known issues (tracked as in-code markers — read the marker before touching)

| Where | Marker | Summary |
|---|---|---|
| redis_client.py `push_history` | ISSUE | non-evictable keys can OOM Redis and stall ALL writes. Only the OUTBOX can still get there — the history lists are capped per guild (~24 KB each), so their total scales with guild count, not runtime. `HISTORY_OUTBOX_MAX` is the opt-in bound on the outbox (and a disabled archive removes the outbox entirely); a memory alarm is still owed |
| sources.py `SoundcloudSource` | TODO | SoundCloud timestamp params ignored (YouTube-only `t`/`ts` parsing) |
| youtube.py `yt_source` / `_first_video_entry` | TODOs | untyped `Exception("Could not find song")`; dead `download=True` param; no format validation on search results (the marker moved to `_first_video_entry` with the loop it describes) |
| musicbot.py `__init__` | HACK | `getattr(bot, "redis")` hides the MusicBotApp dependency from the type checker |
| play_pipeline.py `enqueue_playlist` | HACK | an `assert isinstance(source, YTSource)` stands in for a correlation the signature can't express — a `ResolvedYoutubePlaylist` always arrives with a `YTSource`, but they are separate parameters. `python -O` strips the assert and leaves the attribute reads unguarded; the fix is to have the `Resolved*Playlist` dataclasses carry their own source |
| musicplayer.py ETA zone | TODO | `queue_embed`'s "Est. playing at" and the NP "Estimated finish" read `GuildConfig.timezone`, which `-settings timezone` writes; a guild that never set one renders `DEFAULT_TIMEZONE` (US/Pacific). The `%Z` suffix fixed a *different* bug — a hardcoded "PST" that was wrong the ~8 months a year US/Pacific spends in PDT. Still owed: per-VIEWER rendering, since a guild-wide zone is one clock for everyone in the guild. Fix: Discord relative timestamps (`<t:epoch:R>`) |
| main.py `on_ready` | FIXME | "Bot commands:" log line actually logs an intent flag |
| redis_client.py `clear_connection` | HACK | dead `last_author_id` field still scrubbed; safe to delete after one release |
| commands/jump.py `run` | TODO | `-jump` is a stub ("in development") — implement or drop it from the command list |
| guild_state.py `from_crashed_state` | FIXME | A crash-recovered song is a resume in everything but the flag. A song that WAS a `-play --now` tail now round-trips `is_resume` correctly (`from_song` carries it), but a song merely interrupted mid-play comes back with `ts` set and `is_resume` false, so it announces "Starting song at N seconds" rather than resuming. Synthesizing the flag from `ts > 0` would also move the queue display and the interjection wording, so it wants its own change |

## Recipes for common changes

**Add a command**: method on `MusicBot` with `@commands.command(name=..., aliases=...,
brief=..., usage=..., help=..., extras={"category": ..., "examples": [...], "note": ...})`;
add `@commands.before_invoke(validate_commands)` if it needs the author in voice; open a
span with `@_tracer.start_as_current_span("bot.<name>")`; every reply an embed; list it
in help.py's `CATEGORY_COMMANDS`; tests in `tests/commands/test_<command>.py`.

**The body belongs in the command's own module, not on the cog.** The cog keeps only
what discord.py owns — registration, converters, checks, cooldowns — and one
`try: await <module>.run(...) except Exception as e: await self._command_error(...)`.
`run()` takes `ctx`, the flags, and whatever the cog RESOLVES for it (`redis`,
`archive`, a `MusicPlayer` from `get_mp`), so the module never reaches back into
`MusicBot` and has no import edge to musicbot.py — or the COG itself, under a
`TYPE_CHECKING` guard, for the two things only it can do: reach the player registry,
and run another command through discord.py. **`run()` must not swallow**: the
`except` has to be the caller's, because `_command_error` logs with `exc_info=True`
and that only captures the live traceback from inside the handler. **Every command is
on this pattern**; musicbot.py holds no command logic at all. musicbot.py imports each
module as `<command>_cmd` — the bare name would be the module inside the like-named
cog method, which reads as the method and is not.

**Add a persisted per-guild state field**: constant in `StateField` → field with default
on `GuildStateData` + `from_redis` → the write-path method on `GuildRedisStore` (or
`_now_playing_state_mapping` if it's per-song) → decide whether it belongs in
`_TRANSIENT_SONG_FIELDS` / `clear_connection` → tests in test_guild_state.py and
test_redis_client.py.

**Add a per-guild SETTING** (a durable choice, not runtime state): constant in
`ConfigField`, and its name in the `ConfigFieldName` Literal and, unless it has its own
reset as `volume` does, `ResettableConfigField` → `Optional` field on `GuildConfig` (Optional is not optional — absent
must keep meaning "follow the host default", or "never chose" collapses into "chose
the default") → `to_redis` writes it only when set → `from_redis` reads an
unrecognised value as unset → a numeric field also gets a `CONFIG_DOMAIN` entry in
`guild_state.py`, which the `-settings` registry's static bounds must equal (a test
compares them) and outside which `GuildConfig.__post_init__` reads a value as unset →
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

**Add a bot setting** (an owner-tunable, process-wide knob): parse its env baseline in
`config.py` with `_float_env`/`_int_env` and a named floor → add the name to `FloatKnob` or
`IntKnob` → add the accessor, `def <name in lowercase>() -> float` returning
`_FLOAT_OVERRIDES.get("<NAME>", <NAME>)` (the accessor sweep in `test_config.py` fails a
missing or miswired one) → every consumer calls `config.<name>()` when the value applies. Never
call it at import, class-body or default-argument time, and never read `<NAME>` itself
(`TestBotKnobsAreReadAtCallTime` G1/G2) → its stored field in `guild_state.py`: a
`BotConfigField` constant whose value is `"<name in lowercase>"`, that name in the
`BotConfigFieldName` Literal, an Optional field on `BotConfig`, and its rows in
`BotConfig.to_redis` and `from_redis` (`_b_float`, or `_b_count` for a count) → a registry
`SettingSpec` with `attr` and `env` both `"<NAME>"` and that `field` (invariant 10), and a
chat range whose minimum is strictly above `config.env_floor("<NAME>")` for a time-valued knob, at
least equal for a count (invariant 3) → a `-debug` allowlist row with
`knob="<NAME>"` and no `fallback` → tests set it with `config.set_override`. A value built into
a long-lived object (a semaphore, a session) applies only when that object is rebuilt, and the
spec's `applies` string says so. No pool worker may read it (G4).

**Add a queue-entry field**: `QueueEntryField` constant → `SongQueueEntry` field with
default → `from_queue_object`/`from_song`/`from_crashed_state` as applicable →
`to_redis` table → `parse_queue_entry` with `.get(..., default)` (old wire entries must
parse) → `QueueObject` + `GuildQueue._rehydrate` → **`YTDL.__init__`'s keyword, its
instance assignment, and the `cls(...)` call in `yt_stream` in `src/youtube.py`** —
miss these three and the field is
silently dropped the moment the queue object becomes a playing song, which is where every
read of it happens → then **BOTH places a playing song is turned back into a
QueueObject**: `MusicPlayer._queue_object_of` (the rebuild `_neutralize_prefetch` and
the volume rebuild in `loop()` share) and `MusicPlayer.interject()`'s resume tail.
`YTDL.volume` is the one keyword that is never carried: it is the level baked into that
source, and a requeued song is rebuilt at the level current then. **Not gated on "playback-relevant"** —
`user_input` and `persisted` are neither, and both were lost through exactly that gap.
They fail differently: a `YTDL` missing the attribute outright *raises* there and strands
the prefetch's claim (which is what `persisted` did to every `--now`/`--next` over a
completed prefetch), while one that merely defaults reappears wrong. Both rebuild sites
are invisible to pyright unless `_prefetch_task` stays parameterized as
`asyncio.Task[Optional[YTDL]]`, and invisible to the tests while their song fixtures are
bare `MagicMock()` — drive the rebuild off a real `YTDL` (the `ytdl_instance` fixture
takes carried fields as kwargs) so a missing attribute raises in the suite rather than in
a guild. If it is a DURABLE property of the play rather than of the queue slot, it also
needs `StateField` + `GuildStateData` + `_now_playing_state_mapping` +
`_TRANSIENT_SONG_FIELDS` **and `SongQueueEntry.from_song` / `from_crashed_state`**, or a
crash silently resets it (see `is_resume`/`start_paused`, and `user_input`, which came
back `None` on the one song that was playing).

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
compose image tag in the same commit. The pin is currently a **nightly** (`.dev0`,
`allow-prereleases = true`) because the newest stable, 2026.7.4, 403s on the media fetch
for nearly every video under YouTube's current GVS enforcement — extraction succeeds, so
the client ladder never degrades and the song dies at ffmpeg with the stream refused.
**Check for a stable newer than 2026.7.4 before assuming a nightly
is still required**, and move back to one when it ships — `security.yml`'s weekly
`ytdlp-stable-watch` job warns when PyPI has one, since nothing else notices a nightly
quietly becoming permanent. **Rolling the image back reinstates the broken stable**: the
change is data-safe (nothing new is persisted; both caches are TTL'd and self-heal within
the hour) but rolling back restores the outage this pin exists to fix, so never do it to
chase an unrelated symptom. After any dependency change, `just
test-image-rebuild` before `DOCKER=1` recipes. Watch `_record_serving_format` warnings
and the `_YtdlpLogger` warnings after deploy — they are the early-warning system for
YouTube-side changes.

**Touch the playback loop / queue**: re-read the module docstrings of guild_queue.py and
the loop() bookkeeping comments first; every claim, release, and Redis
LPOP is accounted for exactly once on every path (success, cleared, resolve-failure,
stream-failure, cancellation). test_musicplayer.py (13.6k lines) and test_guild_queue.py
encode these paths — run `just test tests/test_musicplayer.py tests/test_guild_queue.py`
early and often.
