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
  Spotify tracks, albums and playlists (expanded to YouTube searches; collections
  start playing after their first page and keep queueing in the background),
  SoundCloud links, and any other site yt-dlp supports (TikTok, Vimeo, Bandcamp,
  Twitch clips, …)
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
- **Opt-in play-history archive** — off by default, and a default deployment keeps
  nothing long-term: the newest 50 plays per guild live in Redis and no Postgres is
  deployed at all. One flag (`HISTORY_ARCHIVE_ENABLED=true`) makes a deploy bring up
  Postgres, apply the schema, and record every play permanently
  ([details](#operating-the-play-history-archive))
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
| `-history` | `h` | Show recently played songs (up to 50, persists across restarts) |
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
https://open.spotify.com/album/ALBUM_ID
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

`docker compose up` starts the whole default stack, not just the bot: Redis, the
bgutil POT provider, and `grafana/otel-lgtm` (a ~1 GB pull the first time).

**Long-term storage is opt-in.** By default the bot archives nothing: plays live
only in a capped per-guild Redis list (the newest 50, serving `-history`), and no
Postgres is deployed at all. Opting in — `HISTORY_ARCHIVE_ENABLED=true` in
`.env`, then `just up` — permanently records every play (guild id, user id,
title, timestamp) in Postgres until you erase the volume. That is a decision the
deployer makes explicitly, never a side effect of `docker compose up`: the deploy
tooling derives Compose's `archive` profile from that one flag, and a raw
`docker compose up` deploys the default stack whatever the flag says. The
enable/disable/erase procedures are under
[Operating the play-history archive](#operating-the-play-history-archive).

While the archive is enabled, Postgres is required — the bot refuses to start
without `POSTGRES_URL`, rather than quietly buffering history into a Redis outbox
nothing would ever drain. `POSTGRES_PASSWORD` falls back to `password` so an
archive-enabled `docker compose up` still works with nothing else configured; the
bot logs an ERROR at startup and shows a warning on every `-ping` until you
replace it with [`./setup_env.sh`](#install-and-configure). To use a
Postgres you already run instead of the bundled service, set `POSTGRES_URL` in `.env`
and that is all — Compose builds its DSN as `${POSTGRES_URL:-<parts>}`, so yours wins
without editing any tracked file. If port 5432 is
already taken on your machine, set `POSTGRES_HOST_PORT` — the bot uses host
networking, so it reaches the database through the published port.

The schema is applied by a migration runner, not by the bot, and **every deploy
applies pending migrations before the new bot starts** — so a `git pull` that brings
schema changes needs no extra step, and re-running applies nothing. Against an
external Postgres, or to migrate without deploying:

```bash
just db-migrate            # apply pending migrations
```

The bot verifies the schema version on its first connection and refuses to archive
into an unmigrated database, with a message naming that command. See
[Operating the play-history archive](#operating-the-play-history-archive).

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
# Backing services: Redis, plus Postgres and the schema one-shot when
# HISTORY_ARCHIVE_ENABLED=true in .env. The recipe reads the flag, so there is
# nothing to remember. (db-migrate exits 0 when the database is up to date —
# the bot refuses to archive against an unmigrated one.)
just services

just run
```

`just run` loads `.env` and derives `POSTGRES_URL` from the same
`POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`/`POSTGRES_HOST_PORT` values
Compose uses. A local run archives only if `.env` also sets
`HISTORY_ARCHIVE_ENABLED=true` — the flag, not the presence of a database, is
what turns archiving on. The bot process itself reads only the environment — it
has no `.env` support — so `poetry run bot` works too, but only once you have
exported `POSTGRES_URL` (and `DISCORD_TOKEN`) yourself.

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
| `just services` | Start the backing services `just run` needs — Postgres included only when the archive is enabled |
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
| `just ci` | `check` + `container-test` + `test-pg` + `test-redis` — full local mirror of CI | ~2min |

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
| `just db-migrate` | Apply pending play-history schema migrations — every deploy does this too, so this is for external databases and out-of-band runs |
| `just db-backfill [--dry-run]` | Copy pre-archive Redis history into Postgres — **run once, before deploying this build** (the 50-entry cap applies in both archive modes — see [Upgrading to 2.5.0](#upgrading-to-250)) |
| `just db-backfill-docker [--dry-run]` | The same, through the Compose one-shot — for hosts with Docker and no Python toolchain |
| `just db-rejects [count]` | List play_history rows Postgres refused (expected: nothing) |
| `just db-backup` | Dump the play-history database to `backups/` |
| `just db-restore <file> [db]` | Restore a dump into a scratch DB (or a named one) |

These resolve `POSTGRES_URL` from the environment first, then `.env`, and finally by
building a host DSN from `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`/
`POSTGRES_HOST_PORT` — Compose builds the bot's URL internally, so it never appears in
`.env`. They therefore work the same against the bundled Compose Postgres and an
external one. A value already exported in your shell wins over `.env`, so
`POSTGRES_URL=…staging just db-migrate` targets staging.

**Deploy**

| Recipe | Does |
|---|---|
| `just up [sha]` | Deploy an already-built image — HEAD's by default, or the given SHA. With `HISTORY_ARCHIVE_ENABLED=true` it also deploys Postgres and applies pending migrations first, aborting the deploy if they fail |
| `just down` | Stop the compose stack (volumes are kept) |
| `just restart` | Restart the running bot in place — does **not** pick up a new image |
| `just logs [args]` | Follow the bot's logs (`just logs --tail 50`) |
| `just ps` | Show compose service status |
| `just compose <args>` | Any `docker compose` command, with the `archive` profile derived from the flag |

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
| `postgres` | Postgres 18 — the durable play-history archive. **Opt-in**: on the `archive` profile, which the deploy tooling activates when `HISTORY_ARCHIVE_ENABLED=true` |
| `db-migrate` | One-shot schema migration for the archive — same `archive` profile. Every deploy runs it before recreating the bot, and `docker compose up` runs it too; re-running applies nothing |
| `db-backfill` | One-shot copy of pre-archive Redis history into Postgres, run by hand ([procedure](#backfilling-history-that-predates-the-archive)). On the `ops` profile, **not** `archive`, so it is never started by `up` — only by `just db-backfill-docker` |
| `bgutil-pot-provider` | Mints YouTube Proof-of-Origin tokens so the `web_safari` fallback client works; optional — the bot degrades gracefully without it |
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

A rollback re-runs the *older* image's migrations, which is a no-op: every version it
knows about is already applied, and a database ahead of that image is accepted with a
note rather than refused. Rolling back the code does not roll back the schema — nothing
here drops columns, which is what makes an older build safe against a newer database.

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
| `HISTORY_ARCHIVE_ENABLED` | | `false` | **The consent switch for long-term storage.** `true` turns on the Postgres archive: every play (guild id, user id, title, timestamp) is recorded permanently. Unset or `false` — the default — nothing is written to long-term storage and no Postgres is required. Strictly parsed (`true/1/yes` or `false/0/no`, case-insensitive); anything else refuses startup rather than silently picking a side — and refuses the deploy too, for the same value. It is also the **only** switch: the deploy tooling derives Compose's `archive` profile from it, so nothing else needs setting. See [Operating the play-history archive](#operating-the-play-history-archive) |
| `COMPOSE_PROFILES` | | derived | Read by Docker Compose itself, never by the bot. `build_common.sh`'s `resolve_archive_profile` sets it from the flag above on every deploy path, adding or removing `archive` and preserving any other profile you set. Set it by hand only to drive `docker compose` directly |
| `POSTGRES_URL` | when the archive is enabled | built by Compose from the three vars below | Play-history archive connection URL. With `HISTORY_ARCHIVE_ENABLED=true` the bot refuses to start without it, rather than quietly buffering history into an outbox nothing drains; with the archive disabled it is ignored (an INFO line says so — the flag, never URL presence, is what enables archiving). To point at a Postgres you already run, set this in `.env`; Compose falls back to the parts-built DSN only when it is unset, so no tracked file needs editing |
| `POSTGRES_PASSWORD` | — | `password` | Password for the bundled Postgres service. The default exists so a first `docker compose up` needs no setup; while it is in use the bot logs an ERROR at startup and renders a warning embed on every `-ping`. Replace it with `./setup_env.sh` (a fresh 128-bit value, independent of every other secret). **Changing it later needs three, in order** — Postgres reads this only when initializing an empty data directory, so an existing volume keeps its old password: (1) `ALTER USER <user> PASSWORD '<new>'`, (2) edit `.env`, (3) `docker compose up -d`. Doing (2) first silences the warning while the server still accepts the old password. To start clean instead, drop only the database volume (`docker compose down && docker volume rm discord-music-bot_postgres-data`) — not `down -v`, which also removes the Redis volume holding un-drained plays |
| `POSTGRES_USER` | | `musicbot` | Role owning the bundled Postgres service's database |
| `POSTGRES_DB` | | `musicbot` | Database name on the bundled Postgres service |
| `POSTGRES_HOST_PORT` | | `5432` | Host port the bundled Postgres publishes on (loopback only). Change it when something else already owns 5432 — the bot runs on host networking and connects through this port |
| `POSTGRES_MIGRATE_URL` | | falls back to `POSTGRES_URL` | DSN used by `just db-migrate`, so the migrating role can be one with DDL rights while the bot's role has only `SELECT`/`INSERT` |
| `POSTGRES_STATEMENT_CACHE` | | `100` | asyncpg prepared-statement cache size per connection. **Set to `0` behind PgBouncer in transaction-pooling mode** — prepared statements are per-connection state, and transaction pooling hands each transaction a different backend |
| `HISTORY_OUTBOX_MAX` | | `0` (unbounded) | Opt-in ceiling on the un-archived history outbox — meaningful only while the archive is enabled (disabled, the outbox is never written). `0` keeps the durability contract: entries leave only once Postgres has them. A non-zero value drops the oldest entries above the cap — data loss, logged at ERROR — for operators who would rather bound Redis memory. A drop here is unrecoverable: the Redis history list is capped at 50 entries per guild, so anything older that the cap discards existed only in the outbox. See [Operating the play-history archive](#operating-the-play-history-archive) |
| `ENVIRONMENT` | | derived from git branch (`main` → `production`) | Environment name reported in logs/telemetry |
| `POT_PROVIDER_URL` | | `http://127.0.0.1:4416` | bgutil PO-token sidecar base URL |
| `YTDLP_POOL_WORKERS` | | `4` | Worker processes in the yt-dlp extraction pool. Each holds a full CPython + yt-dlp import (~80–120 MB RSS), so the default is deliberately conservative — raise it if multi-guild extraction bursts become the bottleneck |
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | | `3.0` | Progress-bar edit interval for the Now Playing card |
| `PING_TICK_SECS` | | `1.0` | `-ping` health dashboard: how often the embed is re-edited as probes return |
| `PING_DEADLINE_SECS` | | `3.0` | `-ping` health dashboard: how long a probe may run before the row is marked failed |
| `OTEL_SERVICE_NAME` | | `discord-music-bot` | OpenTelemetry service name |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | | `http://localhost:4317` | OTLP gRPC endpoint for traces |
| `OTEL_SDK_DISABLED` | | `false` | Set `true` to disable tracing entirely |

## Upgrading to 2.5.0

**Read this before deploying 2.5.0 or any later build over an install that predates it,
whether or not you use the archive.** One change here destroys data on upgrade, and it is
not opt-in. (The heading names the release that introduced the cap; every build since
carries it.)

This build caps every guild's Redis history list at **50 entries** — the same number
`-history` can display. Earlier builds never trimmed that list, so an established
deployment may be holding thousands of plays per guild. The cap is applied by
`push_history`, which runs on **every song end in both archive modes**, so each guild
loses everything beyond its newest 50 at its next song end. There is no flag, no warning
and nothing to undo.

Whether that matters depends on what else holds a copy:

| Upgrading from | Where your history lives | Before deploying |
|---|---|---|
| 2.4.x with the archive (Postgres was mandatory) | Postgres has every play | Nothing. The Redis list is a display cache; the durable copy is untouched. `-history` will show 50 rather than more |
| Any build with **no Postgres** | The Redis list is the **only** copy | **Back it up, or opt in and backfill** — see below |

### If you are not enabling the archive

`just db-backfill` cannot help you: it moves history *into* Postgres, and refuses to run
without a reachable, migrated database. Snapshot Redis instead, with the bot stopped so
nothing is mid-write:

```bash
docker compose stop discord-music-bot
docker compose exec redis redis-cli SAVE
docker run --rm -v discord-music-bot_redis-data:/data -v "$PWD:/backup" alpine \
    tar czf /backup/redis-history-backup.tar.gz -C /data .
docker compose up -d
```

That captures the whole keyspace (AOF and RDB); the part that matters is the
`guild:*:history` lists. Keep the tarball somewhere off the host — restoring it is a
manual job, but it is the difference between "recoverable" and "gone".

### If you are enabling the archive

Run the backfill **before** deploying this build, and verify it reports a clean run:
[Backfilling history that predates the archive](#backfilling-history-that-predates-the-archive).
The ordering is load-bearing and the tool cannot detect that you got it wrong — a list
already trimmed to 50 looks exactly like a small one.

## Operating the play-history archive

The archive is **off by default**. Everything in this section applies to a deployment that
has opted in — with one exception: the 50-entry cap described in
[Upgrading to 2.5.0](#upgrading-to-250) applies in **both** modes, and the backfill runbook
below is the only tool that preserves data against it.

### Enabling the archive

One line in `.env`:

```env
HISTORY_ARCHIVE_ENABLED=true
```

then `just up` (or `./build_docker.sh`, or `./deploy_docker.sh`). Postgres and the
`db-migrate` one-shot come up, the bot creates its outbox consumer group at the
next start, and every play from then on is archived. Plays from before enabling
were never collected — that was the point of the default — but the newest ≤50 per
guild still sit in the Redis display lists, and the
[backfill](#backfilling-history-that-predates-the-archive) can move exactly that
window into the new archive.

**It has to be one of those commands, not a bare `docker compose up`.** Compose
activates a profile only from `COMPOSE_PROFILES` or `--profile` and never reads
this flag, so the deploy scripts translate one into the other
(`resolve_archive_profile` in `build_common.sh`). A raw `docker compose up`
deploys the default stack with no Postgres behind an archive-enabled bot. If you
want to drive Compose directly, name the services
(`docker compose up -d postgres db-migrate`) or export `COMPOSE_PROFILES=archive`
yourself — both still work, and `just compose <args>` does the derivation for
arbitrary compose commands.

### Disabling — and erasing

Order matters, because `docker compose down` only removes services in the active
model:

```bash
just down                                   # FIRST — always passes --profile archive
# then set HISTORY_ARCHIVE_ENABLED=false in .env
just up
```

Doing it in the other order leaves Postgres running: deactivating a profile never
stops what it already started, and a `down` without the profile cannot see it.
`just up` warns when it finds that container with the archive disabled.
On the next start the bot logs a warning if the outbox still holds plays that
were buffered but never archived — they will not drain while the archive is off.
Re-enable to drain them, or discard them with `DEL history:outbox` (inspect
first: `just outbox`).

**Disabling stops collection; it does not erase.** Already-archived rows stay in
the `postgres-data` volume until you remove it:

```bash
docker volume rm discord-music-bot_postgres-data   # the erasure step
```

**There is a second copy, and it is not in Postgres.** Every deployment — including
one that never enabled the archive — keeps the newest 50 plays per guild in
`guild:{id}:history`, holding requester id, title and timestamp. That list is
deliberately `PERSIST`ed: it has no TTL, ever, because it is the only thing serving
`-history` and length is what bounds it. So it survives the volume removal above, and
a request to erase a user's data is not satisfied by that command alone:

```bash
# with the bot stopped, so nothing rewrites the keys mid-delete
docker compose stop discord-music-bot
docker compose exec redis sh -c \
    "redis-cli --scan --pattern 'guild:*:history' | xargs -r redis-cli del"
docker compose up -d
```

Erasing one user rather than everything means editing the lists rather than deleting
them — there is no tooling for that, and the 50-entry cap means an active guild ages a
given play out on its own within 50 further plays.

### The write path

While enabled, play history is written twice on every song end, in one Redis
pipeline: to the guild's display list, and to a global **outbox** the background
drainer moves into Postgres. The playback loop never waits on Postgres — an
unreachable database just means the outbox grows and drains later.

Reads go the other way. `-history` is served from the Redis list alone: it is capped at
50 entries per guild, which is exactly the command's own `--limit` ceiling, and it is
written synchronously at song end, so it always holds every play the command can be
asked for — including plays the drainer has not archived yet. Postgres is the permanent
record behind it, queried by the commands built on that record rather than by `-history`.

The outbox is a Redis **stream** with a consumer group, not a list. That is what makes
running two bot processes against one Redis safe: the server hands each drainer a
disjoint set of entries, and each settles only the IDs it actually archived. Overlap
costs duplicated work, which the archive's unique index collapses; it cannot destroy a
play that was never inserted.

### Schema

**Every deploy applies pending migrations before the new bot starts**, so a `git pull`
that brings schema changes needs no extra step — `just up`, `./deploy_docker.sh` and
`./build_docker.sh` all run them and abort the deploy if they fail, leaving the running
bot untouched. Out-of-band runs (an external database, or applying a migration without
deploying) go through the recipe:

```bash
just db-migrate            # apply pending migrations (idempotent, concurrency-safe)
```

Migrations live in `migrations/NNNN_name.sql`; the bot never runs DDL. Each version is
recorded in `schema_migrations` inside the same transaction as its own DDL, under an
advisory lock, so re-running applies nothing and two runners racing is fine — that is
what makes it safe on every deploy. On its first connection the bot reads that table and
refuses to serve a database older than the version it was built for, naming this command.
A *newer* database is accepted with a warning by both the bot and the migration runner —
migrations are additive, so rolling the bot back must not become an outage.

### Backfilling history that predates the archive

The archive only records songs played *after* it was deployed. Everything already on the
`guild:{id}:history` lists needs moving across once:

```bash
just db-backfill --dry-run   # count what would move, write nothing
just db-backfill             # do it

# Docker-only host (no local venv). Build FIRST — see below:
just image
just db-backfill-docker --dry-run
just db-backfill-docker

# Without `just`, activate the archive profile explicitly — db-backfill depends
# on postgres, which is undefined without it:
COMPOSE_PROFILES=archive docker compose run --rm db-backfill --dry-run
```

**The Docker path needs `just image` first, and the order is build → backfill → deploy.**
`docker compose run` uses a locally-present tag and will not rebuild a stale one, but
`db-backfill` is pinned to `discord-music-bot:${GIT_SHA:-latest}` — the tag your *running*
deployment already has. On a host that has not built this commit yet, that image predates
the backfill and the run ends at:

```
/app/.venv/bin/python: No module named src.backfill_history
```

It fails loudly rather than silently, but the obvious reaction ("deploy the new image
first, then backfill") is the unrecoverable direction. `just image` builds and tags
without deploying anything, which is why it is a separate step from `./build_docker.sh`.

Rehearse with `--dry-run` first: it checks the database is reachable and migrated, then
walks the same keys and reports the same counts without writing.

**Safe to re-run and safe to interrupt.** Every insert is idempotent, so a run that dies
part-way is resumed by running it again — there is no "figure out where it stopped" step.
It exits non-zero and prints `INCOMPLETE` if any guild failed, so **re-run until it
reports zero failures**.

**Order matters, and one direction is unrecoverable.** This must complete *before* this
build is deployed. It caps each guild's Redis history list at 50 entries,
trimming it at that guild's next song end — exactly the entries this command exists to
move, with Postgres as the only other copy.

Nothing can detect that you got the order wrong. A list sitting at 50 entries is what a
trimmed guild and a healthy migrated guild both look like, so `db-backfill` prints the
ordering notice on every run, including successful ones, and its counts are meaningful
only when the run preceded the deploy. Verify with the reported guild count and a
`SELECT count(*) FROM play_history` before you deploy.

#### After the deploy

The trim is **lazy**: a guild's list is capped at that guild's *next song end*, not at
startup. Dormant guilds — often the ones holding the most history — keep their full lists
until they play again, or until you `DEL` those keys by hand once the backfill has them.

`docker stats` will not move when they are reclaimed. The saving shows up in Redis'
`used_memory` (measured: 262 MB → 1.6 MB for a 500k-entry list), which is what `maxmemory`
and eviction actually gate on. But jemalloc does not return the pages to the OS: RSS stays
flat and `mem_fragmentation_ratio` climbs (1.27 → 1.49 over three cycles in testing). Check
`redis-cli info memory | grep used_memory:`, not the container's RSS. Add `--activedefrag
yes` to the redis command in `docker-compose.yml` if you want the pages back, at some CPU
cost.

### Backlog and sizing

The outbox is deliberately **not** evictable (Redis runs `volatile-lru`, and the
outbox key carries no TTL): an entry in it is a play that is not durable yet. Normally
it is empty within seconds. It only grows while Postgres is unreachable, at a measured
**≈ 487 B per entry** — `MEMORY USAGE ... SAMPLES 0` against the bundled
`redis:7-alpine` (7.4.9) at 100 000 entries, for a 420-byte wire payload with a full
title, a YouTube URL, a thumbnail URL and snowflake ids:

| Backlog | Redis memory |
|---|---|
| 10 000 entries | ≈ 5 MB |
| 100 000 entries | ≈ 49 MB |
| 1 000 000 entries | ≈ 487 MB |

Measure on the server you actually run. These figures were previously quoted as
≈ 410 B/entry, which was correct — for Redis 8, where the same payload costs 424 B. The
stream encoding is ~12% heavier on the 7.x line the bundled image tracks (433 B/entry as
a plain list there), so a horizon derived from the wrong major is optimistic by about a
fifth.

**The bundled Compose Redis runs `--maxmemory 256mb`**, which is ≈ 525 000 outbox
entries on its own, and that budget is shared with every guild's state, queue mirror and
the `ytdl:*`/`spotify:*` caches — so the practical ceiling is a few hundred thousand,
not a million. Past it, Redis has nothing evictable left (the outbox carries no TTL by
design) and starts refusing
**every** write in the process, not just history's. Either raise `maxmemory` for the
backlog you want to survive, or set `HISTORY_OUTBOX_MAX` and accept the loss. A backlog
past 10 000 escalates the drainer's
retry log from WARNING to ERROR; each drain cycle is also an OTEL span
(`history.drain`) carrying batch size, so drain latency and throughput are visible in
Grafana. If you would rather bound the memory than keep every play, set
`HISTORY_OUTBOX_MAX` — the drainer then drops the oldest entries above the cap and
logs each drop at ERROR.

Entries Postgres will never accept (the drainer isolates them one at a time rather
than letting one bad row block the batch) are parked, never deleted:

```bash
just db-rejects            # list rows Postgres refused; expected to print nothing
```

Each row is page-worthy: `HistoryEntry` clamps every field into the column domain at
construction, so a refusal means that validator regressed or the build is talking to a
schema it was not written for. Counts are exact — the insert dedups on
`(guild_id, error_type, digest of payload)`, so three rows means three distinct
failures, not one seen three times.

### Inspecting or resetting the outbox

```bash
just outbox              # depth, in flight, stranded entries, and lost plays
just outbox 5000         # same, with a tighter idle threshold for "stranded"
```

That is the one to reach for. It answers the four states `XLEN` alone cannot tell
apart — undelivered, in flight, acked-but-undeleted, and **tombstoned** — and the last
of those is a set of plays that reached no table at all, visible nowhere else. It also
names the two failure modes with remedies: a `WRONGTYPE` left by a pre-R1 build, and a
missing consumer group.

By hand, if you are on a host without the checkout:

```bash
redis-cli XLEN history:outbox                        # entries not yet in Postgres
redis-cli XPENDING history:outbox drainers           # of those, how many are in flight
redis-cli XINFO GROUPS history:outbox                # consumers, last-delivered id
```

`XLEN` is the backlog gauge. Prefer it to `XINFO GROUPS`' `lag`, which Redis returns as
nil whenever a deletion leaves a gap it cannot reconcile — including after a manual
`XDEL` — so it can go blank exactly when you are looking at it.

**Resetting it destroys plays, and there is one safe way to do it:**

```bash
docker compose stop discord-music-bot
redis-cli DEL history:outbox
docker compose start discord-music-bot
```

Stop the bot first. `DEL` removes the consumer group along with the key, and the next
`XADD` recreates the key *without* one — every read then fails `NOGROUP`. The drainer
heals that on its next tick, so doing it live is recoverable rather than fatal, but the
window costs whatever the drainer was mid-batch on. Never `DEL` this key to "clear a
backlog": every entry in it is a play that is not in Postgres yet.

### Growth and retention

Nothing prunes `play_history` — keeping every play is the point of the tier. Measured on
`postgres:18`, 1 000 000 rows is **≈ 373 MB of table plus ≈ 146 MB of indexes**, so a
busy server accumulates on the order of a gigabyte per few million plays. There is no
action to take at current scale; the thresholds worth revisiting are **10 M rows or
5 GB**, at which point monthly range partitions on `played_at`, or swapping the b-tree
for BRIN, are the obvious next steps. Migration `0002` added an `inserted_at` column
that a future retention job can key off; nothing reads it today.

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

A nightly dump gives an **RPO of ≤ 24 h** for archived history. Plays newer than the
last dump survive only in the Redis history list (`guild:{id}:history`) — *not* in the
outbox, which is emptied within seconds of each batch committing. That list holds only the
newest 50 plays per guild, so for anything older the nightly dump is the only copy and the
24 h window is real exposure. Schedule it with
cron or a systemd timer and prune old files:

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

### Backfill before deploying this build

Order is load-bearing: **backfill → verify zero failures → deploy**. Deploying first caps
the Redis lists at 50 entries per guild, trimming away exactly the entries the backfill
exists to preserve, and Postgres is the only other copy.

"This build", not "the archive build": the cap ships in **both** modes, so an operator who
stays opted out is on the same clock — they just have no backfill to run. See
[Upgrading to 2.5.0](#upgrading-to-250) for what to do instead.

The full procedure — dry-run rehearsal, the Docker-only path, and what to check before
deploying — is under
[Backfilling history that predates the archive](#backfilling-history-that-predates-the-archive).
Deliberately not repeated here: two copies of an irreversible runbook is how one of them
ends up missing the step that matters.

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

The full reference lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Project structure

```
src/
├── main.py            # entrypoint: MusicBotApp (AutoShardedBot), MusicContext, Redis pool
├── musicbot.py        # MusicBot cog — all Discord commands, per-guild player registry
├── musicplayer.py     # per-guild playback loop, prefetch, embeds/ETA, presence
├── guild_queue.py     # GuildQueue — owns the three queue representations
├── guild_history.py   # GuildHistory — play history: capped Redis list + cache
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

docs/                  # architecture reference + design docs
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
