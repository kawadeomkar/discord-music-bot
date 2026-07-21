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

- Python 3.11+
- [Poetry](https://python-poetry.org/) 2.x
- [FFmpeg](https://ffmpeg.org/)
- A [Discord bot token](https://discord.com/developers/applications)
- A [Spotify app](https://developer.spotify.com/dashboard) (client ID + secret)

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

## Docker

**Build and run with Docker Compose**

```bash
# Full pipeline: test gate → build image → deploy
./build_docker.sh

# Or the individual steps
make check            # lint + type-check + tests (the gate)
make image            # build the runtime image, no gate
./deploy_docker.sh    # deploy the image already built for HEAD
```

`build_docker.sh` is a composition of those three — it does not reimplement any of
them. `build_common.sh` is a sourced library, not a runnable script.

**Rolling back**

Deploys are separate from builds precisely so this never requires a rebuild:

```bash
./deploy_docker.sh <git-sha>   # any SHA whose image is still in the local store
docker images discord-music-bot --format '{{.Tag}}\t{{.CreatedSince}}'
```

The script refuses to deploy a tag it cannot find locally rather than letting
Compose build one from your working tree and label it with that SHA.

The `docker-compose.yml` reads credentials from a `.env` file in the project root and uses host networking.

**Manual Docker build**

```bash
docker build -t discord-music-bot .
docker run --env-file .env --network host discord-music-bot
```

## Development

`make` is the task index — run `make` on its own to list every target.

```bash
make install   # create the venv with main + test + lint dependencies
make hooks     # install the git hooks (optional, recommended — see below)

make fmt       # format and auto-fix src/ and tests/ (rewrites files)
make lint      # ruff format --check + ruff check, no rewrites   (~0.1s)
make types     # pyright over src/ AND tests/                    (~5.5s)
make test      # pytest with coverage                            (~15s)
make check     # all three — run this before pushing
make ci        # check + the containerized test run, mirroring CI
```

**`make check` is the contract:** it runs exactly what CI's lint and test jobs
run, so if it passes, CI passes. Nothing else in the repo makes that promise —
in particular, a successful `./build_docker.sh` used not to, because it applied
formatting instead of checking it and never ran pyright at all.

**Git hooks**

`make hooks` installs two stages, deliberately split by how long they take:

| Stage | Runs | Cost |
|---|---|---|
| pre-commit | `ruff check --fix`, `ruff format`, whitespace/YAML/TOML checks | ~0.1s |
| pre-push | `make check` | ~25s |

The hooks are a convenience, not the gate — CI still runs every one of these
checks, and `--no-verify` is always available when you need it. Note that the
formatting hooks **rewrite files**: a commit that trips one fails and leaves the
fixes unstaged, so `git add` them and commit again. That is intended behavior.

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
