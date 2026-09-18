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
   enforces eight duplicated name/version pairs — it is a dep of `check` AND its own CI
   step, deliberately: Dependabot opens SEPARATE PRs that each move one half, and those
   are validated by CI, never by a local `check`. **Five more pairs are enforced by
   nothing** (the PO-token sidecar, the published Prometheus port, the otel collector
   images, `MPLCONFIGDIR`, and the liveness cap against the HEALTHCHECK window), and
   every one of them fails green: the build passes and the symptom lands at runtime.
   Both lists, with what each drift looks like: `.claude/rules/ci-and-build.md`.
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
src/               one module per concern; main.py builds the bot, musicbot.py
                   registers commands, commands/<name>.py holds each body
src/commands/      ONE MODULE PER COMMAND, each exposing run()
migrations/        NNNN_*.sql, applied in numeric order; the ONLY source of schema
docs/ARCHITECTURE.md  the only tracked file under docs/ (golden rule 2)
tests/             one test_<module>.py per src module, commands/ mirroring src/commands/
justfile           every dev command; build_*.sh / deploy_docker.sh compose them
Dockerfile         3 stages: builder (deps) → test (+ test/lint groups) → runtime
docker-compose.yml bot + redis + postgres (archive profile) + pot-provider + otel-lgtm
.github/workflows/ ci.yml, security.yml (pip-audit), todo-to-issue.yml
```

What each module holds and where a change belongs: `.claude/rules/layout.md`.

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
| `.claude/rules/layout.md` | the annotated module map: what each file holds and where a change belongs |
| `.claude/rules/ci-and-build.md` | the CI job graph, the image build and deploy, and every duplicated version pin — enforced and unenforced |
| `.claude/rules/testing.md` | the test layout, the yt-dlp and Discord seams, fakeredis's divergences, the `pg` and `redis` tiers |
| `.claude/rules/commands.md` | command registration, the one-module-per-command rule and the help copy each command carries |

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

One `tests/test_<module>.py` per src module, `tests/commands/` mirroring
`src/commands/`; a command's tests live with its BODY and drive it through the
cog's wrapper. Redis is fakeredis, Discord objects are spec'd mocks, and
**warnings are errors** (golden rule 11). Run `just check` before pushing — the
pre-push hook does. Two opt-in tiers, `just test-pg` and `just test-redis`, cover
what fakeredis and an in-process double get wrong; both are real merge gates.

The layout rules, every seam the suite installs, fakeredis's five stream
divergences and the tier gating: `.claude/rules/testing.md`.

## CI/CD and deployment

`ci.yml`: resolve-env → version-bump (PRs) → lint, test, container-test,
pg-integration, redis-integration → build → release. Every test job is a real
merge gate. Deploys are separate from builds: `just up <sha>` deploys a
locally-present tag and refuses to build, and a dirty tree tags `<sha>-dirty`
so a tag never lies about its commit.

The job graph, the three-stage image, host networking and the GIT_SHA seam:
`.claude/rules/ci-and-build.md`.

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

Each recipe lives in the rule file for the subsystem it changes, and loads with it:

- `.claude/rules/config.md` — Add a per-guild SETTING, Add a bot setting
- `.claude/rules/extraction.md` — Bump yt-dlp
- `.claude/rules/commands.md` — Add a command
- `.claude/rules/playback.md` — Add a queue-entry field, Touch the playback loop / queue
- `.claude/rules/state-and-recovery.md` — Add a persisted per-guild state field, Add a schema migration, Add a history-entry field, Touch the history outbox
