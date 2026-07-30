# discord-music-bot

[![CI](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/ci.yml)
[![Security](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/security.yml/badge.svg)](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/security.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

<!-- Pytest Coverage Comment:Begin -->
<!-- Pytest Coverage Comment:End -->

A self-hosted Discord music bot that streams audio from YouTube, Spotify, SoundCloud, and any other yt-dlp-supported site
into voice channels. Built as a single-process Python asyncio application on
[discord.py](https://github.com/Rapptz/discord.py), [yt-dlp](https://github.com/yt-dlp/yt-dlp),
and FFmpeg, with Redis for playback state, caching, and crash recovery.

## Features

- **Multi-source playback** — YouTube URLs and playlists, plain-text YouTube search,
  Spotify tracks and playlists (expanded to YouTube searches), SoundCloud links, and
  any other site yt-dlp supports (TikTok, Vimeo, Bandcamp, Twitch clips, …)
- **Near-zero inter-song latency** — a three-phase yt-dlp pipeline resolves metadata
  instantly at enqueue time, prefetches stream URLs in the background while the current
  song plays, and caches them in Redis
- **Live Now Playing card** — an embed with a live-updating progress bar that stays
  pinned to the bottom of the channel, re-attaching itself beneath every bot response
- **`-playnow` interjection** — interrupt the current song with another one; the
  interrupted song resumes afterward from the exact position it left off
- **Crash recovery** — queue, current song (with playback position), volume, and
  history persist in Redis; on restart the bot rejoins voice and resumes from the
  saved position
- **Per-guild isolation** — every server gets its own player, queue, history, and volume
- **Queue management** — shuffle, clear, remove-by-URL, per-song ETA estimates,
  persistent play history
- **Timestamp seeks** — a YouTube link with `?t=90` starts playback at 1:30
- **Rich `-help`** — a custom man-page-style help command with aliases, examples,
  and per-command notes
- **Resilient YouTube extraction** — PO-token sidecar support makes `web_safari` a
  working fallback client when the primary client is throttled or blocked
- **Observability** — OpenTelemetry tracing and structured logging (structlog), with a
  bundled Grafana LGTM stack in Docker Compose
- **Sharding-ready** — built on `AutoShardedBot`; FFmpeg streaming auto-reconnects on
  network drops

## Commands

The command prefix is `-`. Run `-help` for the full manual or `-help <command>` for
details, aliases, and examples.

### Playback

| Command | Aliases | Description |
|---|---|---|
| `-play <url\|search>` | `p`, `sing` | Queue a song and start playing |
| `-playnow <url\|search>` | `pn` | Play immediately; the interrupted song resumes after |
| `-skip` | `sk` | Skip to the next song in the queue |
| `-pause` | `po` | Pause the current song (reports the exact position) |
| `-resume` | `r` | Resume from where the song was paused |
| `-stop` | `st` | Stop playback, drop the queue, and disconnect |
| `-volume <0–100>` | `v`, `vol`, `sound` | Set playback volume (applies from the next song; saved per server) |

### Queue

| Command | Aliases | Description |
|---|---|---|
| `-queue` | `q` | List the songs waiting to play (up to 10) |
| `-now` | `np`, `rn`, `nowplaying` | Show the currently playing song |
| `-history` | `h` | Show recently played songs (persists across restarts) |
| `-shuffle` | — | Randomly reorder the queue (needs 3+ queued songs) |
| `-clear` | `c` | Empty the queue (the current song keeps playing) |
| `-remove <url>` | `rm` | Remove every queued song matching a YouTube URL |
| `-jump <position>` | `j` | Jump to a queue position *(in development)* |

### Utility

| Command | Aliases | Description |
|---|---|---|
| `-join` | `summon` | Connect the bot to your voice channel (`-play` does this automatically) |
| `-ping` | `latency`, `l`, `delay`, `health`, `status` | Live health check: Discord/Redis/Spotify/Postgres/OTEL latency + bot/yt-dlp/ffmpeg versions |
| `-help [command]` | — | Full command manual |

### Supported inputs

```
https://www.youtube.com/watch?v=VIDEO_ID
https://www.youtube.com/watch?v=VIDEO_ID&t=90    # start at timestamp
https://youtu.be/VIDEO_ID?t=90
https://www.youtube.com/playlist?list=LIST_ID    # whole playlist
https://open.spotify.com/track/TRACK_ID
https://open.spotify.com/playlist/PLAYLIST_ID
https://soundcloud.com/artist/track
https://www.tiktok.com/@user/video/VIDEO_ID      # any other yt-dlp-supported site
never gonna give you up                          # plain text searches YouTube
```

YouTube, Spotify, and SoundCloud get first-class handling (timestamps, playlist
expansion, Spotify→YouTube matching). Any other link is handed straight to
[yt-dlp](https://github.com/yt-dlp/yt-dlp) — if it's one of the ~1800 sites yt-dlp
supports (TikTok, Vimeo, Bandcamp, Twitch clips, …) it just plays; if not, the bot
replies that the link isn't from a site it can play.

## Quick start

### Requirements
<a id="requirements"></a>

**To run the bot** — Docker, plus credentials:

- A [Discord bot token](https://discord.com/developers/applications)
- _Optional:_ a [Spotify app](https://developer.spotify.com/dashboard) (client ID + secret) — only needed to play Spotify links. Without it the bot starts normally and Spotify links are declined; YouTube, SoundCloud, other yt-dlp sites, and search all still work. When credentials are provided, the bot validates them against the Spotify API on startup — invalid credentials are logged as an error and Spotify links are declined (everything else keeps working). Run `-ping` to see the current Spotify status.

The Docker Compose stack contains its own Redis to enable persistence, caching, and crash recovery.
Credentials *must* be set in a `.env` file at the project root before starting anything:
`docker-compose.yml` declares `env_file: .env`, and Compose treats a missing one as
an error rather than a warning. The format is under [step 2](#install-and-configure).

`docker compose up` starts the whole stack, not just the bot: Redis, Postgres, the
bgutil POT provider, and `grafana/otel-lgtm` (a ~1 GB pull the first time).

Postgres is the future durable home of play history. The server, its schema and the
operator tooling land ahead of the code that uses them — **no bot code reads or writes
Postgres yet**, so the service is inert and the bot starts with or without it.
`POSTGRES_PASSWORD` is still mandatory and has no fallback, because Compose refuses to
initialize a database with a credential nobody chose; run
[`./setup_env.sh`](#install-and-configure) to generate one. If port 5432 is already
taken on your machine, set `POSTGRES_HOST_PORT`.

The schema is applied by a migration runner, never by the bot. `docker compose up`
runs it for you as a one-shot `db-migrate` service; against an external Postgres, run
it yourself:

```bash
just db-migrate            # apply pending migrations
```

See [Operating the play-history archive](#operating-the-play-history-archive).

**To contribute**, add:

- [`just`](https://just.systems) — the task runner every command below goes
  through (`brew install just`)
- [Poetry](https://python-poetry.org/) 2.x
- Python 3.14+ (`pyproject.toml` pins `requires-python = '>=3.14,<4.0'`)
- [FFmpeg](https://ffmpeg.org/) on `PATH`
- [Redis](https://redis.io/) 7+ if you run the bot outside Compose — strongly
  recommended; the bot runs degraded without it

`just` must be installed system-wide, not only in the virtualenv. `just install`
places a copy at `.venv/bin/just`, but a virtualenv's `bin/` is on `PATH` only
while the environment is activated, and the pre-push git hook does not activate it.

With `just` and Docker, Poetry, Python and FFmpeg are not required: every check
runs in a container via `DOCKER=1` — see [Just recipes](#just-recipes).

### 1. Create the Discord application

1. Create an application in the [Discord Developer Portal](https://discord.com/developers/applications)
2. Under **Bot**, enable the **Message Content Intent** and **Server Members Intent**
3. Under **OAuth2 → URL Generator**, select the `bot` scope with these permissions:
   - **Voice**: Connect, Speak
   - **Text**: Send Messages, Embed Links, Add Reactions, Read Message History
4. Invite the bot to your server with the generated URL

### 2. Install and configure
<a id="install-and-configure"></a>

```bash
git clone https://github.com/kawadeomkar/discord-music-bot.git
cd discord-music-bot
poetry install
```

Create a `.env` file in the project root:

```env
DISCORD_TOKEN=your_discord_bot_token
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

Or start from the template and let the helper fill in the generated values:

```bash
./setup_env.sh     # copies .env.example -> .env, generates POSTGRES_PASSWORD
```

`POSTGRES_PASSWORD` is a fresh 128-bit random value. Re-running is safe: an existing
password is left alone, because Postgres reads it only when it first initializes its
data volume, and a silently-changed value would lock the bot out of its own database.
`--force` regenerates anyway and warns about exactly that.

`poetry install` installs the bot's runtime dependencies only. The `test`, `lint`
and `dev` groups are optional, so running the bot does not pull in pyright and its
bundled Node runtime.

Contributors should use `just install`, which adds those three groups. `just check`
requires them; the error "ruff not found … run 'just install' first" means they are
missing.

Every recipe below uses the project's virtualenv. With pyenv-virtualenv (this
project ships a `.python-version`), `poetry install` installs into that environment
rather than `./.venv`, so every recipe follows `$VIRTUAL_ENV` when it is set and
falls back to `./.venv` otherwise. Recipes report which interpreter they resolved
to when something is missing; `just --evaluate` prints it directly.

### 3. Run

```bash
# Start Redis if you don't have one running
docker compose up -d redis

# Optional for now: Postgres plus the one-shot that applies the schema. No bot
# code touches them yet, but `just db-migrate` and the backup recipes do.
docker compose up -d postgres db-migrate

poetry run bot
```

## Just recipes
<a id="just-recipes"></a>

[`just`](https://just.systems) is the task index: one recipe per entry point. Run
`just` with no arguments to list every recipe with its description, grouped by
purpose.

Multi-step pipelines live in the shell scripts (`./build_docker.sh`,
`./deploy_docker.sh`); the justfile indexes the primitives those scripts compose.

**With only Docker and `just`**, prefix `DOCKER=1` to `fmt`, `fmt-check`, `lint`,
`types`, `test` or `check` to run it inside the test image instead of a local
virtualenv. No Python, Poetry or Node is required on the host:

```bash
DOCKER=1 just check    # the full gate, container-only  (~31s)
DOCKER=1 just fmt      # ruff rewrites your files, not the image's
```

The prefix must come **before** the recipe name. `just check DOCKER=1` is an error
(`just` reads it as a second recipe to run), unlike `make check DOCKER=1`.

`src/`, `tests/` and `pyproject.toml` are bind-mounted, so the container reads and
writes your working tree. Formatting runs as your uid, so rewritten files are owned
by you rather than by root. The image is built automatically the first time; after
changing `pyproject.toml` or `poetry.lock`, run `just test-image-rebuild` so the
container picks up the new dependencies.

The native path is the default because it is faster (~24s vs ~31s, and ~0.05s vs
~0.6s for a bare `just lint` — the difference is container startup).

**Setup**

| Recipe | Does |
|---|---|
| `just install` | Create the venv with main + test + lint + dev dependencies |
| `just hooks` | Install the git hooks (see [Git hooks](#git-hooks)) |
| `just hooks-run` | Run every hook against every file, not just staged ones |
| `just hooks-update` | Bump the pinned hook revisions in `.pre-commit-config.yaml` |
| `just test-image-rebuild` | Rebuild the image `DOCKER=1` uses — needed after a dependency change |

**Develop** — ordered fastest first

| Recipe | Does | Cost |
|---|---|---|
| `just fmt` | Format and auto-fix `src/` and `tests/` — **rewrites files** | ~0.1s |
| `just fmt-check` | `ruff format --check`, no rewrites | ~0.05s |
| `just lint` | `ruff check`, no rewrites | ~0.05s |
| `just types` | pyright over `src/` **and** `tests/` | ~6s |
| `just test` | pytest with coverage | ~13s |
| `just check` | `fmt-check` + `lint` + `types` + `test` — **run this before pushing** | ~24s |
| `just container-test` | Build the test image and run the suite inside it | ~1min |
| `just ci` | `check` + `container-test` — full local mirror of CI | ~1.5min |

`just test` forwards extra arguments to pytest:

```bash
just test tests/test_youtube.py    # one file
just test -k spotify               # one pattern
just test --maxfail=1              # stop at the first failure
```

**Build**

| Recipe | Does |
|---|---|
| `just image` | Build the runtime image as `:latest` and `:<git-sha>` — no test gate |

`just image` has no test gate; the gate lives in the pipeline
(`./build_docker.sh`). Use `just image` when you want the artifact and have already
run `just check`.

**Database** — see [Operating the play-history archive](#operating-the-play-history-archive)

| Recipe | Does |
|---|---|
| `just db-migrate` | Apply pending play-history schema migrations |
| `just db-backup` | Dump the play-history database to `backups/` |
| `just db-restore <file> [db]` | Restore a dump into a scratch DB (or a named one) |

These resolve `POSTGRES_URL` from the environment first, then `.env`, and finally by
building a host DSN from `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`/
`POSTGRES_HOST_PORT` — Compose builds the one-shot's URL internally, so it never
appears in `.env`. They therefore work the same against the bundled Compose Postgres
and an external one. A value already exported in your shell wins over `.env`, so
`POSTGRES_URL=…staging just db-migrate` targets staging.

**Deploy**

| Recipe | Does |
|---|---|
| `just up [sha]` | Deploy an already-built image — HEAD's by default, or the given SHA |
| `just down` | Stop the compose stack (volumes are kept) |
| `just restart` | Restart the running bot in place — does **not** pick up a new image |
| `just logs [args]` | Follow the bot's logs (`just logs --tail 50`) |
| `just ps` | Show compose service status |

`just up` never builds. If no image exists for the current commit it fails rather
than letting Compose build one and label it with that SHA — see
[Rolling back](#rolling-back).

Shell completions ship in the binary: `just --completions zsh` (or `bash`/`fish`).

**Typical flows**

```bash
# Inner loop while writing code
just fmt && just check

# Gate, build and deploy in one step
./build_docker.sh

# The same steps individually
just check && just image && just up

# Inspect a running deployment, then roll back
just logs
just up <last-good-sha>
```

## Docker

The Compose stack runs the bot plus its supporting services:

| Service | Purpose |
|---|---|
| `discord-music-bot` | The bot itself (host networking) |
| `redis` | Redis 7 with AOF persistence — queue/state/cache storage |
| `postgres` | Postgres 18 — the play-history database (no bot code reads it yet) |
| `db-migrate` | One-shot schema migration; exits 0 once the database is up to date |
| `bgutil-pot-provider` | Mints YouTube Proof-of-Origin tokens so the `web_safari` fallback client works ([details](docs/PO_TOKEN_SIDECAR_PLAN.md)); optional — the bot degrades gracefully without it |
| `otel-lgtm` | Grafana LGTM observability stack — UI at [localhost:3014](http://localhost:3014) (admin/admin); optional |

```bash
# Full pipeline: test gate → build image → deploy
./build_docker.sh

# Or the individual steps
just check            # lint + type-check + tests (the gate)
just image            # build the runtime image, no gate
./deploy_docker.sh    # deploy the image already built for HEAD

# Just the essentials (bot + Redis, no observability/PO-token sidecar)
docker compose up -d discord-music-bot redis
```

`build_docker.sh` composes those three steps rather than reimplementing them. Its
gate is `just check`, so there is one definition of "will CI pass".
`build_common.sh` is a sourced library, not a runnable script; running it directly
exits 64.

Compose reads credentials from the same `.env` file and uses host networking. A named
volume persists yt-dlp's disk cache across container restarts so the first song after
a restart stays fast.

**Rolling back**
<a id="rolling-back"></a>

Deploys are separate from builds, so a rollback never requires a rebuild:

```bash
just up <git-sha>              # any SHA whose image is still in the local store
docker images discord-music-bot --format '{{.Tag}}\t{{.CreatedSince}}'
```

The script refuses to deploy a tag it cannot find locally rather than letting
Compose build one from your working tree and label it with that SHA.

A tag identifies exactly what was built. Building from anything other than a clean
checkout produces `<git-sha>-dirty.<digest>`, so a tag never identifies a commit it
was not built from. A clean tree produces the bare SHA, which is what you roll back
to.

Untracked files also count as unclean: they are not in the commit, but `COPY src/`
adds them to the image. The digest is a hash of the tree that was built, so two
different sets of local edits never share a tag. Rebuilding after an edit produces
a new tag, which is how `just up` detects there is a new image to deploy.

`just restart` is not a deploy — it restarts the existing container with the image
it already has. To run a newly built image, use `just up` (or `./deploy_docker.sh`).

## Configuration

All configuration is via environment variables (a `.env` file is loaded by Docker
Compose; for local runs, export them or use your shell's dotenv tooling).

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Discord bot token |
| `SPOTIFY_CLIENT_ID` | | — | Spotify app client ID (Client Credentials flow). Enables Spotify links; omit both Spotify vars to run without Spotify support |
| `SPOTIFY_CLIENT_SECRET` | | — | Spotify app client secret. Required alongside `SPOTIFY_CLIENT_ID` to enable Spotify links |
| `REDIS_URL` | | `redis://localhost:6379` | Redis connection URL |
| `POSTGRES_PASSWORD` | ✅ | — | Password for the bundled Postgres service. Required by the Compose stack, which fails the command outright rather than initializing a database with a credential nobody chose. Generated once by `./setup_env.sh`, independent of every other secret |
| `POSTGRES_URL` | | built by Compose for the `db-migrate` one-shot | Play-history database URL. Read by `just db-migrate` and the other `db-*` recipes; **no bot code reads it yet**. Set it in `.env` to point the recipes at a Postgres you already run |
| `POSTGRES_USER` | | `musicbot` | Role owning the bundled Postgres service's database |
| `POSTGRES_DB` | | `musicbot` | Database name on the bundled Postgres service |
| `POSTGRES_HOST_PORT` | | `5432` | Host port the bundled Postgres publishes on (loopback only). Change it when something else already owns 5432 — the `db-*` recipes run on the host and connect through this port |
| `POSTGRES_MIGRATE_URL` | | falls back to `POSTGRES_URL` | DSN used by `just db-migrate`, so the migrating role can be one with DDL rights while the bot's role has only `SELECT`/`INSERT` |
| `ENVIRONMENT` | | derived from git branch (`main` → `production`) | Environment name reported in logs/telemetry |
| `POT_PROVIDER_URL` | | `http://127.0.0.1:4416` | bgutil PO-token sidecar base URL |
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | | `3.0` | Progress-bar edit interval for the Now Playing card |
| `OTEL_SERVICE_NAME` | | `discord-music-bot` | OpenTelemetry service name |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | | `http://localhost:4317` | OTLP gRPC endpoint for traces |
| `OTEL_SDK_DISABLED` | | `false` | Set `true` to disable tracing entirely |

## Operating the play-history archive

Postgres is the durable home play history is moving to. This is the infrastructure
half: the server, the `play_history` schema and the operator tooling. **Nothing writes
to it yet** — play history still lives entirely on the Redis lists, and the bot never
opens a Postgres connection. Setting the tier up first means the schema is already
applied, backed up and restorable by the time the code that fills it arrives.

### Schema

```bash
just db-migrate            # apply pending migrations (idempotent, concurrency-safe)
```

Migrations live in `migrations/NNNN_name.sql` and are applied in numeric order by
`src/db_migrate.py`; the bot never runs DDL. Each one runs in its own transaction
holding an advisory lock, so two runners racing during a deploy is safe and a
half-applied migration is not a state the database can be left in. The runner records
what it applied in a `schema_migrations` ledger and re-running is a no-op.

### Backups

```bash
just db-backup                              # -> backups/play_history_<timestamp>.dump
just db-restore backups/play_history_...    # -> <db>_restore_check, live untouched
```

`db-restore` targets a **scratch** database by default. Overwriting the live one is
deliberately awkward, because it drops and replaces every row — including every play
since the dump:

```bash
CONFIRM=1 just db-restore backups/play_history_... musicbot
```

A nightly dump gives an **RPO of ≤ 24 h** for whatever the database holds. Schedule it
with cron or a systemd timer and prune old files:

```cron
0 4 * * *  cd /srv/discord-music-bot && /usr/local/bin/just db-backup && ls -1t backups/*.dump | tail -n +15 | xargs rm -f
```

Two details that bite in cron specifically: it runs with a minimal `PATH`, so `just`
needs its absolute path (`command -v just` to find yours), and `xargs -r` is a GNU
extension — `xargs rm -f` is the portable way to tolerate an empty list.

Test the restore quarterly — an untested backup is a hypothesis. `just db-restore
<file>` already restores into `<db>_restore_check` rather than the live database, so
the drill is just that command followed by `SELECT count(*) FROM play_history` against
that scratch database. WAL archiving / PITR is
out of scope for the bundled Compose stack; on a managed Postgres, use the platform's
own PITR.

## Architecture

One `MusicPlayer` per guild orchestrates a playback loop that streams Opus audio to
Discord over UDP via FFmpeg. Every `-play` goes through a three-phase yt-dlp pipeline:

1. **Resolve** — the input is classified (URL / search / playlist / Spotify /
   SoundCloud) and resolved to lightweight metadata, hitting a Redis search cache
   before ever invoking yt-dlp, so queueing is instant.
2. **Prefetch** — immediately after enqueue (and again while each song plays), a
   background task runs the full yt-dlp extraction and caches the stream URL in Redis.
3. **Stream** — when the song reaches the front, the loop usually finds a warm cache
   entry and starts FFmpeg with no extraction call at all.

Queue state lives in three synchronized representations (an `asyncio.Queue` for the
playback loop, a deque for display, and a Redis list for persistence), all privately
owned by a `GuildQueue` domain class. Redis also stores the current song and playback
position, which is how the bot survives crashes: on startup it detects interrupted
sessions, rejoins voice, and resumes the queue.

The full reference lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the
[docs index](docs/README.md) tracks design documents and plans.

### Project structure

```
src/
├── main.py            # entrypoint: MusicBotApp (AutoShardedBot), MusicContext, Redis pool
├── musicbot.py        # MusicBot cog — all Discord commands, per-guild player registry
├── musicplayer.py     # per-guild playback loop, prefetch, embeds/ETA, presence
├── guild_queue.py     # GuildQueue — owns the three queue representations
├── guild_history.py   # GuildHistory — played-song history (Redis + display cache)
├── guild_state.py     # Redis schema: frozen value objects + field constants
├── redis_client.py    # connection pool, GuildRedisStore, cache helpers
├── youtube.py         # yt-dlp integration, YTDL audio source, prefetch pipeline
├── sources.py         # input parsing → YTSource / SpotifySource / SoundcloudSource
├── spotify.py         # Spotify Client Credentials API client with Redis caching
├── help.py            # custom man-page-style -help command
├── telemetry.py       # OpenTelemetry + structlog setup
├── config.py          # ENVIRONMENT detection, tunables
└── util.py            # logging factory, queue message formatting

tests/                 # one test_*.py per src/ module, plus:
├── conftest.py        # shared fixtures
├── helpers.py         # test-only builders
└── test_context.py    # Discord context doubles

docs/                  # architecture reference + design docs (see docs/README.md)
```

Most modules have a matching `tests/test_<name>.py`. `config.py` and `telemetry.py`
do not, and are the two lowest-covered files in the report.
The coverage gate (`fail_under = 80`, project-wide) is enforced by `just test`.

## Development

Every command lives in the justfile — see [Just recipes](#just-recipes) for the
full list. This section covers behavior beyond the recipe list itself.

**`just check` is the contract for CI's lint and test jobs:** if it passes, those
two pass. Those jobs call the same recipes — `just fmt-justfile`, `just fmt-check`,
`just lint`, `just types`, `just test-report` — so there is one definition of each
check and both callers use it.

`just check` does not cover the whole pipeline:

| CI job | Covered locally by |
|---|---|
| Lint & Type Check | `just check` |
| Test Suite | `just check` |
| Container Test | `just ci` (adds `just container-test`) |
| Build Image | nothing — it builds the `runtime` stage, which no local recipe exercises |
| Security / pip-audit | nothing — it audits `poetry.lock` against advisories |

A green `just check` is therefore a strong signal, not a guarantee of a green PR: a
dependency that breaks only the runtime image, or a CVE published against a locked
package, turns the PR red with no local warning. `just ci` closes the container gap;
the other two run only remotely. Green CI on `main` publishes the runtime image to
GHCR.

`just types` passes `--pythonpath` explicitly for the same reason: pyright resolves
imports from the interpreter it is given, and pointing it at a path that `just
install` does not populate produces "green locally, red in CI". Every recipe uses
the same venv — `$VIRTUAL_ENV` when one is active, otherwise `./.venv`, which is
what CI and the Dockerfile use.

**Git hooks**
<a id="git-hooks"></a>

`just hooks` installs two stages, split by how long they take:

| Stage | Runs | Cost |
|---|---|---|
| pre-commit | `ruff check --fix`, `ruff format`, `just --fmt --check`, whitespace/YAML/TOML checks | ~0.1s |
| pre-push | `just check` | ~24s |

The hooks are a convenience, not the gate — CI runs every one of these checks, and
`--no-verify` is available. The formatting hooks **rewrite files**: a commit that
trips one fails and leaves the fixes unstaged, so `git add` them and commit again.
This is intended behavior.

The pre-push hook needs `just` on the `PATH` git provides, which the previous
`make`-based hook did not require: `/usr/bin/make` was always present, and a `just`
that exists only inside your virtualenv is not. A push failing with `just: command
not found` means `just` is not installed system-wide (see
[Requirements](#requirements)).

## License

[GPL-3.0](LICENSE)
