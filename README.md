# discord-music-bot

[![CI](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/ci.yml)
[![Build](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/build.yml/badge.svg)](https://github.com/kawadeomkar/discord-music-bot/actions/workflows/build.yml)
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

- [Python](https://www.python.org/downloads/) **3.14+**
- [Poetry](https://python-poetry.org/) **2.x**
- [FFmpeg](https://ffmpeg.org/) on `PATH`
- [Redis](https://redis.io/) 7+ *(strongly recommended — powers persistence, caching,
  and crash recovery; the bot runs degraded without it)*
- A [Discord bot token](https://discord.com/developers/applications)
- A [Spotify app](https://developer.spotify.com/dashboard) (client ID + secret)

### 1. Create the Discord application

1. Create an application in the [Discord Developer Portal](https://discord.com/developers/applications)
2. Under **Bot**, enable the **Message Content Intent** and **Server Members Intent**
3. Under **OAuth2 → URL Generator**, select the `bot` scope with these permissions:
   - **Voice**: Connect, Speak
   - **Text**: Send Messages, Embed Links, Add Reactions, Read Message History
4. Invite the bot to your server with the generated URL

### 2. Install and configure

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

### 3. Run

```bash
# Start Redis if you don't have one running
docker compose up -d redis

poetry run bot
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
# One-shot: format, test in a container, build, and start the stack detached
./build.sh

# Or manually
docker compose up --build -d

# Just the essentials (bot + Redis, no observability/PO-token sidecar)
docker compose up -d discord-music-bot redis
```

Compose reads credentials from the same `.env` file. A named volume persists yt-dlp's
disk cache across container restarts so the first song after a restart stays fast.

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
tests/                 # pytest suite (~950 tests, coverage gate ≥80%)
docs/                  # architecture reference + design docs (see docs/README.md)
```

## Development

```bash
# Install with dev dependencies
poetry install --with test,lint

# Run the test suite (asyncio auto mode, coverage report included)
poetry run pytest

# Run a single test
poetry run pytest tests/test_sources.py::TestParseUrlYouTube::test_youtube_watch_url

# Type-check
poetry run pyright src/

# Format
poetry run black src/ tests/
```

`./build.sh` reproduces CI locally: it builds the test image, formats with black,
runs the tests in a container, then builds the runtime image and starts the Compose
stack. GitHub Actions runs black/pyright, the test suite with coverage, a containerized
test pass, and a dependency security audit on every push; green CI on `main` publishes
the runtime image to GHCR.

## License

[GPL-3.0](LICENSE)
