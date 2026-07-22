# discord-music-bot

[![CI](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/ci.yml)
[![Security](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/security.yml/badge.svg)](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/security.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

<!-- Pytest Coverage Comment:Begin -->
<!-- Pytest Coverage Comment:End -->

A self-hosted Discord music bot that streams audio from YouTube, Spotify, and SoundCloud
into voice channels. Built as a single-process Python asyncio application on
[discord.py](https://github.com/Rapptz/discord.py), [yt-dlp](https://github.com/yt-dlp/yt-dlp),
and FFmpeg, with Redis for playback state, caching, and crash recovery.

## Features

- **Multi-source playback** — YouTube URLs and playlists, plain-text YouTube search,
  Spotify tracks and playlists (expanded to YouTube searches), and SoundCloud links
- **Near-zero inter-song latency** — a three-phase yt-dlp pipeline resolves metadata
  instantly at enqueue time, prefetches stream URLs in the background while the current
  song plays, and caches them in Redis
- **Live Now Playing card** — an embed with a live-updating progress bar that stays
  pinned to the bottom of the channel, re-attaching itself beneath every bot response
- **`-playnow` interjection** — interrupt the current song with another one; the
  interrupted song resumes afterward from the exact position it left off
- **Crash recovery** — queue, current song (with playback position), volume, and
  history persist in Redis; on restart the bot rejoins voice and picks up where it died
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
| `-ping` | `latency`, `l`, `delay` | Check the bot's WebSocket latency |
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
never gonna give you up                          # plain text searches YouTube
```

## Quick start

### Requirements
<a id="requirements"></a>

Two audiences, two answers.

**To run the bot** — Docker, plus credentials:

- A [Discord bot token](https://discord.com/developers/applications)
- A [Spotify app](https://developer.spotify.com/dashboard) (client ID + secret)

That is the whole toolchain — no Python, no Poetry. Compose brings its own Redis, so
persistence, caching, and crash recovery work out of the box. You do still need to put
those credentials in a `.env` file at the project root before starting anything:
`docker-compose.yml` declares `env_file: .env`, and Compose treats a missing one as
an error rather than a warning. The format is under [step 2](#install-and-configure).

Note `docker compose up` starts the whole stack, not just the bot: Redis, the
bgutil POT provider, and `grafana/otel-lgtm` (a ~1 GB pull the first time).

**To contribute**, add:

- [`just`](https://just.systems) — the task runner every command below goes
  through (`brew install just`)
- [Poetry](https://python-poetry.org/) 2.x
- Python 3.14+ (`pyproject.toml` pins `requires-python = '>=3.14,<4.0'`)
- [FFmpeg](https://ffmpeg.org/) on `PATH`
- [Redis](https://redis.io/) 7+ if you run the bot outside Compose — strongly
  recommended; the bot runs degraded without it

`just` has to be installed properly, not just present in the virtualenv. `just
install` does put a copy at `.venv/bin/just`, but a virtualenv's `bin/` is only on
your `PATH` while the environment is activated — and the pre-push git hook runs
`just check` in whatever environment git hands it. Install it system-wide.

With `just` and Docker you can skip Poetry, Python and FFmpeg entirely: every
check runs in a container via `DOCKER=1` — see [Just recipes](#just-recipes).

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

`poetry install` gives you the bot and nothing else. The `test`, `lint` and `dev`
groups are optional, so running the bot does not drag in pyright and its bundled
Node runtime.

Contributors want `just install` instead — it adds those three groups, which is
what `just check` needs. If you get "ruff not found … run 'just install' first",
this is why.

Every recipe below assumes the project's virtualenv is the active one. If you use
pyenv-virtualenv (this project ships a `.python-version`), that happens
automatically and `poetry install` lands there rather than in `./.venv` — which is
why every recipe follows `$VIRTUAL_ENV` when it is set and falls back to `./.venv`
otherwise. Recipes tell you which interpreter they resolved to if anything is
missing; `just --evaluate` prints it outright.

### 3. Run

```bash
# Start Redis if you don't have one running
docker compose up -d redis

poetry run bot
```

## Just recipes
<a id="just-recipes"></a>

[`just`](https://just.systems) is the task index: one verb per entry point, so you
can run only the thing you need. Run `just` on its own to list every recipe with
its description, grouped by what it is for.

Multi-step *pipelines* stay in the shell scripts (`./build_docker.sh`,
`./deploy_docker.sh`); the justfile is the index over the primitives they compose.

**Only have Docker and `just`?** Prefix `DOCKER=1` to `fmt`, `fmt-check`, `lint`,
`types`, `test` or `check` and it runs inside the test image instead of a local
virtualenv — no Python, no Poetry, no Node needed on your machine:

```bash
DOCKER=1 just check    # the full gate, container-only  (~31s)
DOCKER=1 just fmt      # ruff rewrites YOUR files, not the image's
```

The prefix has to come **before** the recipe name. `just check DOCKER=1` is an
error (`just` reads it as a second recipe to run), unlike `make check DOCKER=1`.

`src/`, `tests/` and `pyproject.toml` are bind-mounted, so the container reads and
writes your working tree. Formatting runs as your uid, so rewritten files stay
yours rather than turning up root-owned. The image is built automatically the
first time; after changing `pyproject.toml` or `poetry.lock`, run
`just test-image-rebuild` so the container picks up the new dependencies.

The native path stays the default because it is faster (~24s vs ~31s, and ~0.05s
vs ~0.6s for a bare `just lint` — the difference is container startup).

**Setup**

| Recipe | Does |
|---|---|
| `just install` | Create the venv with main + test + lint + dev dependencies |
| `just hooks` | Install the git hooks (see [Git hooks](#git-hooks)) |
| `just hooks-run` | Run every hook against every file, not just staged ones |
| `just hooks-update` | Bump the pinned hook revisions in `.pre-commit-config.yaml` |
| `just test-image-rebuild` | Rebuild the image `DOCKER=1` uses — needed after a dependency change |

**Develop** — the inner loop, fastest first

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

`just test` forwards extra arguments to pytest, which `make` could not do:

```bash
just test tests/test_youtube.py    # one file
just test -k spotify               # one pattern
just test --maxfail=1              # stop at the first failure
```

**Build**

| Recipe | Does |
|---|---|
| `just image` | Build the runtime image as `:latest` and `:<git-sha>` — no test gate |

`just image` deliberately has no gate. A gate you cannot skip is a gate people
route around, so it lives in the *pipeline* (`./build_docker.sh`) instead. Use
`just image` when you want the artifact and have already run `just check`.

**Deploy**

| Recipe | Does |
|---|---|
| `just up [sha]` | Deploy an already-built image — HEAD's by default, or the given SHA |
| `just down` | Stop the compose stack (volumes are kept) |
| `just restart` | Restart the running bot in place — does **not** pick up a new image |
| `just logs [args]` | Follow the bot's logs (`just logs --tail 50`) |
| `just ps` | Show compose service status |

`just up` never builds. If no image exists for the current commit it refuses
rather than letting Compose build one and label it with that SHA — see
[Rolling back](#rolling-back).

Shell completions ship in the binary: `just --completions zsh` (or `bash`/`fish`).

**Typical flows**

```bash
# Inner loop while writing code
just fmt && just check

# Ship it: gate → build → deploy, in one step
./build_docker.sh

# Same thing, one step at a time
just check && just image && just up

# Something's wrong in production
just logs
just up <last-good-sha>
```

## Docker

The Compose stack runs the bot plus its supporting services:

| Service | Purpose |
|---|---|
| `discord-music-bot` | The bot itself (host networking) |
| `redis` | Redis 7 with AOF persistence — queue/state/cache storage |
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

`build_docker.sh` is a composition of those three — it does not reimplement any of
them. Its gate *is* `just check`, so there is exactly one definition of "will CI
pass". `build_common.sh` is a sourced library, not a runnable script; running it
directly exits 64.

Compose reads credentials from the same `.env` file and uses host networking. A named
volume persists yt-dlp's disk cache across container restarts so the first song after
a restart stays fast.

**Rolling back**
<a id="rolling-back"></a>

Deploys are separate from builds precisely so this never requires a rebuild:

```bash
just up <git-sha>              # any SHA whose image is still in the local store
docker images discord-music-bot --format '{{.Tag}}\t{{.CreatedSince}}'
```

The script refuses to deploy a tag it cannot find locally rather than letting
Compose build one from your working tree and label it with that SHA.

Tags are honest about what went into them: building from anything other than a
clean checkout produces `<git-sha>-dirty.<digest>`, so a tag never claims to be a
commit it isn't. A clean tree gives the bare SHA, which is what you roll back to.

"Anything other than clean" includes untracked files — they are not in the commit,
but `COPY src/` puts them in the image just the same. The digest is a hash of the
actual tree that gets built, so two different sets of local edits never share a
tag: rebuild after an edit and you get a new tag, which is what makes `just up`
notice there is something new to deploy.

`just restart` is not a deploy — it restarts the existing container with the image
it already has. To run a newly built image, use `just up` (or `./deploy_docker.sh`).

## Configuration

All configuration is via environment variables (a `.env` file is loaded by Docker
Compose; for local runs, export them or use your shell's dotenv tooling).

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Discord bot token |
| `SPOTIFY_CLIENT_ID` | ✅ | — | Spotify app client ID (Client Credentials flow) |
| `SPOTIFY_CLIENT_SECRET` | ✅ | — | Spotify app client secret |
| `REDIS_URL` | | `redis://localhost:6379` | Redis connection URL |
| `ENVIRONMENT` | | derived from git branch (`main` → `production`) | Environment name reported in logs/telemetry |
| `POT_PROVIDER_URL` | | `http://127.0.0.1:4416` | bgutil PO-token sidecar base URL |
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | | `3.0` | Progress-bar edit interval for the Now Playing card |
| `OTEL_SERVICE_NAME` | | `discord-music-bot` | OpenTelemetry service name |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | | `http://localhost:4317` | OTLP gRPC endpoint for traces |
| `OTEL_SDK_DISABLED` | | `false` | Set `true` to disable tracing entirely |

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

Most modules have a matching `tests/test_<name>.py`; `config.py` and `telemetry.py`
currently do not, which is why they are the two lowest-covered files in the report.
The coverage gate (`fail_under = 80`, project-wide) is enforced by `just test`.

## Development

Every command lives in the justfile — see [Just recipes](#just-recipes) for the
full list. This section covers the two things worth knowing beyond "what runs".

**`just check` is the contract for CI's lint and test jobs:** if it passes, those
two pass. Not because the two were written to match, but because those jobs *call
these recipes* — `just fmt-justfile`, `just fmt-check`, `just lint`, `just types`,
`just test-report`. There is one definition of each check and both callers use it.

It is not the whole pipeline, and the gap is worth knowing before you push:

| CI job | Covered locally by |
|---|---|
| Lint & Type Check | `just check` |
| Test Suite | `just check` |
| Container Test | `just ci` (adds `just container-test`) |
| Build Image | nothing — it builds the `runtime` stage, which no local recipe exercises |
| Security / pip-audit | nothing — it audits `poetry.lock` against advisories |

So a green `just check` is a strong signal, not a guarantee of a green PR: a
dependency that breaks only the runtime image, or a CVE published against a locked
package, turns the PR red with no local warning. `just ci` closes the container gap;
the other two are remote by nature. Green CI on `main` publishes the runtime image
to GHCR.

That is also why `just types` passes `--pythonpath` explicitly: pyright resolves
imports from the interpreter it is *told* about, and pinning it to a path that
`just install` does not populate is how "green locally, red in CI" gets built in.
Every recipe points at the same venv — `$VIRTUAL_ENV` when one is active,
otherwise `./.venv`, which is what CI and the Dockerfile use.

**Git hooks**
<a id="git-hooks"></a>

`just hooks` installs two stages, deliberately split by how long they take:

| Stage | Runs | Cost |
|---|---|---|
| pre-commit | `ruff check --fix`, `ruff format`, `just --fmt --check`, whitespace/YAML/TOML checks | ~0.1s |
| pre-push | `just check` | ~24s |

The hooks are a convenience, not the gate — CI still runs every one of these
checks, and `--no-verify` is always available when you need it. Note that the
formatting hooks **rewrite files**: a commit that trips one fails and leaves the
fixes unstaged, so `git add` them and commit again. That is intended behavior.

One caveat that did not apply under `make`: the pre-push hook needs `just` on the
`PATH` git gives it. `/usr/bin/make` was always there; a `just` that only exists
inside your virtualenv is not. If a push fails with `just: command not found`,
that is why — install it system-wide (see [Requirements](#requirements)).

## License

[GPL-3.0](LICENSE)
