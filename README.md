# discord-music-bot

<!-- Pytest Coverage Comment:Begin -->
<!-- Pytest Coverage Comment:End -->

A self-hosted Discord music bot that streams audio from YouTube, Spotify, and SoundCloud directly into voice channels.

## Features

- Stream audio from YouTube URLs, YouTube search queries, Spotify tracks/playlists, and SoundCloud
- Queue management with shuffle, skip, pause, and resume
- Timestamp support — start a YouTube video at a specific time (`?t=90`)
- Spotify playlist expansion — automatically converts Spotify playlist tracks to YouTube searches
- Per-guild isolated playback with independent queues and history
- FFmpeg-backed streaming with automatic reconnection on network drops
- Structured logging throughout

## Commands

All commands use the `-` prefix.

| Command | Aliases | Description |
|---|---|---|
| `-play <url\|query>` | `p`, `pl`, `pla`, `sing` | Play a song or add it to the queue |
| `-skip` | `sk` | Skip the current song |
| `-stop` | `st` | Stop playback and disconnect |
| `-pause` | `po` | Pause the current song |
| `-resume` | `r` | Resume a paused song |
| `-queue` | `q` | Show the current queue (up to 10 songs) |
| `-now` | `np`, `rn`, `nowplaying` | Display the currently playing song |
| `-history` | `h` | Show recently played songs |
| `-shuffle` | — | Shuffle the queue (requires 3+ songs) |
| `-volume <0–100>` | `v`, `vol`, `sound` | Set the playback volume |
| `-join` | `summon` | Join your voice channel |
| `-ping` | `latency`, `l`, `delay` | Check bot latency |

### Supported URL formats

```
https://www.youtube.com/watch?v=VIDEO_ID
https://www.youtube.com/watch?v=VIDEO_ID&t=90   # start at timestamp
https://youtu.be/VIDEO_ID?t=90
https://open.spotify.com/track/TRACK_ID
https://open.spotify.com/playlist/PLAYLIST_ID
https://soundcloud.com/artist/track
plain text search query                          # searches YouTube automatically
```

## Requirements

Two audiences, two answers.

**To run the bot** — Docker, plus credentials:

- A [Discord bot token](https://discord.com/developers/applications)
- A [Spotify app](https://developer.spotify.com/dashboard) (client ID + secret)

That is the whole list. `docker compose up` needs nothing else.

**To contribute**, add:

- [`just`](https://just.systems) — the task runner every command below goes
  through (`brew install just`)
- [Poetry](https://python-poetry.org/) 2.x
- Python 3.14+ (`pyproject.toml` pins `requires-python = '>=3.14,<4.0'`)
- [FFmpeg](https://ffmpeg.org/)

`just` has to be installed properly, not just present in the virtualenv. `just
install` does put a copy at `.venv/bin/just`, but a virtualenv's `bin/` is only on
your `PATH` while the environment is activated — and the pre-push git hook runs
`just check` in whatever environment git hands it. Install it system-wide.

With `just` and Docker you can skip Poetry, Python and FFmpeg entirely: every
check runs in a container via `DOCKER=1` — see [Just recipes](#just-recipes).

## Local setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/kawadeomkar/discord-music-bot.git
cd discord-music-bot
poetry install
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
DISCORD_TOKEN=your_discord_bot_token
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

**3. Run the bot**

```bash
poetry run bot
```

Contributors should use `just install` instead of `poetry install` — it adds the
`test`, `lint` and `dev` dependency groups that `just check` needs.

Every recipe below assumes an activated virtualenv (or `poetry run just <recipe>`,
which works without one).

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

**Build and run with Docker Compose**

```bash
# Full pipeline: test gate → build image → deploy
./build_docker.sh

# Or the individual steps
just check            # lint + type-check + tests (the gate)
just image            # build the runtime image, no gate
./deploy_docker.sh    # deploy the image already built for HEAD
```

`build_docker.sh` is a composition of those three — it does not reimplement any of
them. Its gate *is* `just check`, so there is exactly one definition of "will CI
pass". `build_common.sh` is a sourced library, not a runnable script; running it
directly exits 64.

**Rolling back**
<a id="rolling-back"></a>

Deploys are separate from builds precisely so this never requires a rebuild:

```bash
just up <git-sha>              # any SHA whose image is still in the local store
docker images discord-music-bot --format '{{.Tag}}\t{{.CreatedSince}}'
```

The script refuses to deploy a tag it cannot find locally rather than letting
Compose build one from your working tree and label it with that SHA.

Tags are honest about what went into them: building with uncommitted changes to
tracked files produces `<git-sha>-dirty`, so a tag never claims to be a commit it
isn't. A clean tree gives the bare SHA, which is what you roll back to.

`just restart` is not a deploy — it restarts the existing container with the image
it already has. To run a newly built image, use `just up` (or `./deploy_docker.sh`).

The `docker-compose.yml` reads credentials from a `.env` file in the project root and uses host networking.

**Manual Docker build**

```bash
docker build -t discord-music-bot .
docker run --env-file .env --network host discord-music-bot
```

## Development

Every command lives in the justfile — see [Just recipes](#just-recipes) for the
full list. This section covers the two things worth knowing beyond "what runs".

**`just check` is the contract:** if it passes, CI passes. Not because the two
were written to match, but because CI's lint and test jobs *call these recipes* —
`just fmt-check`, `just lint`, `just types`, `just test-report`. There is one
definition of each check and both callers use it. Nothing else in the repo makes
that promise; in particular, a successful `./build_docker.sh` used not to, because
it applied formatting instead of checking it and never ran pyright at all.

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

## Project structure

```
src/
├── main.py          # bot entrypoint, intents, extension loading
├── musicbot.py      # Discord Cog with all slash commands
├── musicplayer.py   # per-guild queue management and playback loop
├── youtube.py       # yt-dlp integration, YTDL audio source class
├── sources.py       # URL parsing, source type detection
├── spotify.py       # Spotify API client (track/playlist lookup)
└── util.py          # logging factory, queue formatting utilities
tests/
├── conftest.py      # shared fixtures
├── test_sources.py
├── test_util.py
├── test_musicplayer.py
├── test_spotify.py
└── test_youtube.py
```

## Discord bot setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application
2. Under **Bot**, enable the **Message Content Intent** and **Server Members Intent**
3. Under **OAuth2 → URL Generator**, select the `bot` scope and the following permissions:
   - Connect, Speak (voice)
   - Send Messages, Embed Links, Add Reactions, Read Message History (text)
4. Use the generated URL to invite the bot to your server
