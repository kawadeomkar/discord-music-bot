# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the bot (requires .env)
poetry run bot

# Run all tests
poetry run pytest

# Run a single test
poetry run pytest tests/test_sources.py::TestParseUrlYouTube::test_youtube_watch_url

# Type-check
python -m pyright src/

# Format + lint
poetry run ruff format src/ tests/
poetry run ruff check src/ tests/
```

The bot requires a `.env` file in the project root with `DISCORD_TOKEN`, `SPOTIFY_CLIENT_ID`, and `SPOTIFY_CLIENT_SECRET`.

## Environment

- **pyenv virtualenv**: `discord-music-bot-3.14` (Python 3.14.6) — set via `pyenv local` (`.python-version`); what bare `python` resolves to inside the repo. Has deps + an editable project install.
- **In-project `.venv`**: a second Python 3.14.6 env (Homebrew-based, gitignored, also has deps) — used by `poetry run …` (`poetry.toml` sets `virtualenvs.create = true`, in-project) and by Pylance/pyright (`pyrightconfig.json`: `venvPath "."` / `venv ".venv"`; gitignored)
- **Global pyenv**: `3.13.1` — a shell without the repo's pyenv env active cannot `poetry run` (requires-python is `>=3.14`; Poetry refuses under 3.13)
- Stale 3.13-era envs may still exist (`~/.pyenv/versions/discord-music-bot`, Poetry's cached `py3.11`/`py3.13` envs) — ignore them
- **Docker base image**: `python:3.14-slim` (multi-stage: base/builder/test/runtime)

---

## Marking follow-up work (GitHub issue triggers)

Comments starting with `TODO`, `FIXME`, `ISSUE`, or `HACK` become GitHub issues
**automatically**: `.github/workflows/todo-to-issue.yml` runs `alstr/todo-to-issue-action`
on every push to `main` and opens a labelled issue for each such comment added in the
diff. (The GitHub Pull Requests VS Code extension also offers a manual "Create issue from
comment" code action on the same identifiers — same triggers, pinned in `.vscode/settings.json`.)

**Whenever you notice code that should eventually be fixed, revisited, or cleaned up —
but that is out of scope for the change you are making — leave one of these comments
instead of silently moving on or burying the observation in chat.**

Pick the trigger by severity:

| Trigger | Use when | Example |
|---|---|---|
| `FIXME` | It is **wrong**. A real bug, race, or incorrect behavior that users can hit. Highest priority. | `# FIXME: play_start_epoch counts downtime as playback — restart resumes at the wrong position` |
| `HACK` | It **works, but it shouldn't have to**. A deliberate workaround, a magic constant, something that papers over a bug elsewhere or in a dependency. | `# HACK: 1800s TTL margin compensates for YouTube CDN clock skew; drop once yt-dlp exposes real expiry` |
| `ISSUE` | A **design/architecture concern** too big for a local fix — needs its own plan or discussion. | `# ISSUE: yt-dlp extraction blocks the event loop under load; needs ProcessPoolExecutor (see docs/ARCHITECTURE_PLAN.md)` |
| `TODO` | **Incomplete or planned work.** Not broken, just not finished. Lowest priority. | `# TODO: handle SoundCloud playlists — only single tracks are parsed today` |

Labels are applied automatically from the trigger: `FIXME` → `bug`, `ISSUE` → `enhancement`,
`HACK` → `technical-debt`, `TODO` → none.

### Shape (this is not stylistic — the parser depends on it)

The action splits the comment into **title = the first line, body = every line after it**.
So the first line must be a complete, self-contained sentence that works as an issue title
on its own; wrapped prose produces a title truncated mid-clause. Aim for ≤ ~70 characters
and state the *action* or the *defect*, not a fragment.

```python
# FIXME: Crash recovery counts bot downtime as playback position.
# `now` is read at RESTART time while play_start_epoch was written when the song
# started, so a bot that was down for 10 minutes adds those 10 minutes to the
# computed position — a song that crashed 30 seconds in comes back near its end.
# Fix is a periodic playback heartbeat written to Redis.
# Design: docs/CRASH_RECOVERY_HEARTBEAT_PLAN.md.
```

Body convention, in order: **symptom → consequence → fix/pointer.** End with a
`Design:` or `See:` line when a spec exists.

### Rules

- **The comment must stand alone.** Someone reading the issue on GitHub with no
  surrounding code must understand what is wrong and why it matters. `# TODO: fix this`
  is useless; state the symptom and the consequence.
- **One trigger per comment**, at the start, uppercase, followed by `:` and a space.
  Place it on the line directly above the offending code (or inside the function it
  describes) so the generated issue's permalink points at the right place.
- **Use a real `#` comment, never a docstring.** Docstrings are string literals, not
  comments — neither the action nor the VS Code code action will see a trigger inside one.
- **Don't use a trigger for work you're doing right now** — only for work deliberately
  deferred. If you're about to fix it in this same change, just fix it.
- **Don't duplicate an existing tracked item.** Grep for the trigger words first; if the
  concern is already captured in a comment, in `docs/ARCHITECTURE_PLAN.md`, or in one of
  the `docs/*_PLAN.md` specs, reference that document from the comment rather than adding
  a competing note.
- **Adding a trigger to `main` is what files the issue.** The action is diff-based: it
  only sees trigger comments on *added* lines. A comment that is already on `main` is
  invisible to it — to file an issue for one, edit the comment line itself (touching the
  file is not enough). Removing a comment does **not** close its issue; close it by hand.

## Architecture

The bot is a single-process Python asyncio application. One `MusicPlayer` instance per Discord guild. Audio is streamed as Opus over UDP via FFmpeg. Redis provides durable state and a yt-dlp URL cache.

The full architecture document is at `docs/ARCHITECTURE.md`.

### Module responsibilities

| Module | Role |
|---|---|
| `src/main.py` | `MusicBotApp` (extends `AutoShardedBot`). `setup_hook` creates Redis pool and loads extensions. |
| `src/musicbot.py` | `MusicBot` Cog. All Discord commands. Owns `mps: dict[guild_id → MusicPlayer]`. `on_ready` triggers crash recovery per guild. |
| `src/musicplayer.py` | Per-guild playback orchestration. `loop()` task, prefetch task, embeds/ETA, presence. Delegates every queue operation to `self.queue: GuildQueue`. |
| `src/guild_queue.py` | `GuildQueue` — the queue domain class. Privately owns all three queue representations plus the bulk-mutation mutex and cleared-flag; every queue operation (put/clear/shuffle/remove/restore/dequeue bookkeeping) lives here. |
| `src/guild_history.py` | `GuildHistory` — played-song history domain class. Privately owns the pair of legs: the unbounded Redis list (source of truth for all played songs) and a HISTORY_CACHE_LIMIT-capped in-memory display cache. |
| `src/guild_state.py` | Schema module: every byte persisted to Redis is defined here. Frozen value objects (`GuildStateData`, `NowPlayingData`, `SongQueueEntry`/`SearchQueueEntry`, `GuildPlaybackSnapshot`) + field-name constants. Pure data — no domain logic, no project runtime imports. |
| `src/youtube.py` | yt-dlp integration. `QueueObject` dataclass. `YTDL(FFmpegOpusAudio)`. `yt_source`, `yt_stream`, `prefetch_stream` classmethods. |
| `src/sources.py` | Input parsing. `parse_input` classifies a string into `YTSource`, `SpotifySource`, or `SoundcloudSource`. |
| `src/spotify.py` | Spotify Client Credentials API. Double-checked locking for token refresh. `track`, `playlist`, `artists`, `albums` methods with Redis caching. |
| `src/redis_client.py` | `GuildRedisStore` (per-guild Redis ops). Module-level `cache_get`/`cache_set` helpers. |
| `src/util.py` | `get_logger`, `queue_message`. |

### yt-dlp three-phase pipeline

Every `-play` command goes through three phases:

**Phase 1 — resolve to `QueueObject`** (`YTDL.yt_source`, called from `musicbot.py`):
- Search strings check the `ytdl:source:{normalized query}` Redis cache (TTL 1h) before running yt-dlp — repeat plays skip the 3–4s search.
- Runs yt-dlp with `process=False` for direct YouTube URLs (fast — no stream extraction), `process=True` for search strings.
- Returns `QueueObject(webpage_url, title, requester, ts, duration?, uploader?, thumbnail?)`. No audio data yet. Missing metadata fields are back-filled by prefetch (`_enrich_queueobject`) when full extraction lands.
- Spotify tracks are resolved to `"Title Artist"` search strings by `Spotify.track()` before this call.
- YouTube **playlist** URLs bypass this: `YTDL.yt_playlist` (flat extraction) returns `List[QueueObject]` in one call, enqueued with `prefetch=False`.

**Phase 1b — eager prefetch** (`YTDL.prefetch_stream`, spawned in `MusicPlayer.queue_put`):
- Immediately after enqueue, `asyncio.create_task(YTDL.prefetch_stream(item, redis=...))` is spawned as a fire-and-forget background task.
- Runs yt-dlp with full extraction and writes the result to Redis under key `ytdl:stream:{webpage_url}` with TTL ~19,800s (~5h30m).
- Only runs for `QueueObject` items — `YTSource` items (Spotify playlist tracks) have no stable `webpage_url` at enqueue time.
- Errors are logged and swallowed; Phase 2 recovers by extracting fresh.

**Phase 2 — stream before playback** (`YTDL.yt_stream`, called inside `MusicPlayer.loop()`):
- Checks Redis cache first. On hit, builds `YTDL(FFmpegOpusAudio)` with no yt-dlp call.
- On cache miss, runs yt-dlp extraction and caches the result.
- YouTube CDN URLs are IP-bound (HMAC-signed `sparams`) and have a 6-hour expiry window. The 1800s margin ensures cache hits are valid for songs up to 30 minutes long.

The split exists so queue operations are instant and inter-song latency is near zero (Phase 1b warms the cache while the current song plays).

### Three queue representations — owned by `GuildQueue`

A guild's queue exists in three representations, all **private to `GuildQueue`**
(`src/guild_queue.py`) — nothing outside the class can mutate one leg without the
others, so the sync invariant is structural rather than a call-site discipline:

| Structure | Type | Purpose |
|---|---|---|
| `_pending` | `asyncio.Queue[QueueObject \| YTSource]` | Consumed by the playback loop (via `get`/`get_nowait`/`task_done` pass-throughs) |
| `_display` | `deque[QueueObject \| YTSource]` | Ordered view for embeds/ETA (`display_items()`, `peek_next()`) |
| Redis list `guild:{id}:queue` | JSON `SongQueueEntry`/`SearchQueueEntry` wire entries | Persistence across restarts |

**Boundaries (deliberate):** stream prefetch stays in `MusicPlayer` — its wrappers
(`queue_clear`/`queue_shuffle`/`queue_remove`) cancel the prefetch task *before*
delegating (it may hold an item from `get_nowait()`). Embed/ETA building stays in
`MusicPlayer` (needs `current_song`). Live item ↔ at-rest entry conversion and
requester rehydration happen inside `GuildQueue`.

### MusicPlayer key attributes

| Attribute | Type | Notes |
|---|---|---|
| `current_song` | `Optional[YTDL]` | Currently playing `FFmpegOpusAudio` object |
| `play_next` | `asyncio.Event` | Set by `after=` callback (thread-safe via `call_soon_threadsafe`); cleared at start of each loop iteration |
| `queue` | `GuildQueue` | All queue state and operations (see above) |
| `history` | `GuildHistory` | All played songs (unbounded Redis list + 50-entry in-memory display cache — see `guild_history.py`) |
| `volume` | `float` | 0.0–1.0; applied via FFmpeg `-filter:a volume=` on next song |
| `_player` | `asyncio.Task` | Long-lived `loop()` task |
| `_prefetch_task` | `asyncio.Task` | Active `_prefetch_next_song()` task |
| `store` | `Optional[GuildRedisStore]` | `None` if no Redis URL configured |

### Playback loop flow

```
loop() start
  └─ prefetched_song available?
       ├─ Yes → use it directly (skip queue.get + yt_stream)
       └─ No  → queue.get (300s timeout → stop()) → _resolve_source → _stream_source
  └─ current_song is None? → queue.finish_failed_dequeue(), send error, continue
  └─ queue.try_commit_dequeue()  (False → cleared mid-resolve, discard song)
  └─ vc.play(current_song, after=λ: play_next.set())
  └─ atomic Redis MULTI/EXEC via SongQueueEntry.from_song(song) carrier:
     LPOP queue + current_song_* state fields
     (play_start_epoch backdated by -ss offset) + now_playing snapshot
  └─ _send_now_playing()
  └─ create_task(_prefetch_next_song())    ← dequeues next item, resolves + streams while current plays
  └─ await play_next.wait()
  └─ await _prefetch_task → prefetched_song
  └─ append history + clear Redis current_song_url
  └─ queue.task_done() → repeat
```

`_prefetch_next_song` calls `queue.get_nowait()` — it dequeues one item. If cancelled (clear/shuffle/remove), it returns the item to the front of the pending queue via `queue.requeue_front()` before re-raising `CancelledError`, so the bulk mutation processes it with everything else (no `task_done` — the task slot transfers with the item). If resolve/stream fails, it retires the dequeue on all three legs via `queue.finish_failed_dequeue()`.

### Source types

`parse_input` in `sources.py` returns one of three frozen dataclasses:
- `YTSource` — direct YouTube URL (`process=False`), search (`process=True`, `ytsearch="ytsearch:..."`), or playlist (`type=YTType.PLAYLIST`, `list_id=...` → `YTDL.yt_playlist` flat extraction)
- `SpotifySource` — `type=SpotifyType.TRACK` or `.PLAYLIST`; resolved via `Spotify.track()` / `Spotify.playlist()` to YouTube search strings
- `SoundcloudSource` — `url` passed directly to yt-dlp

Spotify playlists become a `List[YTSource]` (via `spotify_playlist_to_ytsearch`) and are enqueued as unresolved search items. They are **not** prefetched at enqueue time, but they **are** persisted to the Redis queue as `"ytsource"` wire entries (`SearchQueueEntry`) and survive restarts.

### Command gating

All commands use `@commands.before_invoke(validate_commands)`. `cog_before_invoke` runs first (creates MusicPlayer if needed). `validate_commands` checks: author is a guild Member, author is in a voice channel, and (for non-`play`) the bot is in the same channel.

### Per-guild isolation

`MusicBot.mps: dict[guild_id → MusicPlayer]` — one player per guild. `cleanup()` cancels all tasks (`_prefetch_task`, `_restore_task`, `_player`), calls `store.clear_connection()` (prevents spurious recovery on next `on_ready`), and deletes the entry from `mps`.

### Redis schema overview

All guild keys are prefixed `guild:{guild_id}:`. TTL is 24h idle expiry (`GUILD_TTL`),
refreshed on writes, restore, and clean shutdown — **except `guild:{id}:history`,
which is persistent** (PERSISTed on every push, excluded from every TTL refresh) so
full play history is never lost to idle expiry. **The schema is defined in one place:
`src/guild_state.py`** — field constants + frozen value objects with `from_redis`/
`to_redis` converters; no other module touches raw wire bytes.

| Key | Type | Contents (value object) |
|---|---|---|
| `guild:{id}:state` | Hash | 12 fields: `volume`, `voice_channel_id`, `text_channel_id`, `current_song_*` (url/title/duration/uploader/requester_id/interjected — a parked `SongQueueEntry`), `play_start_epoch`, `total_pause_seconds`, `pause_start_epoch` → `GuildStateData` |
| `guild:{id}:now_playing` | Hash | 12 display fields for the recovered Now Playing embed → `NowPlayingData` |
| `guild:{id}:queue` | List | JSON entries, `"type"`-discriminated → `SongQueueEntry` / `SearchQueueEntry` (RPUSH on enqueue, LPOP inside the atomic start transaction) |
| `guild:{id}:history` | List | JSON `HistoryEntry` objects (newest first; legacy `"title - url"` strings still parse) → `serialize_history_entry`/`parse_history_entry`. **No TTL, no trim** — unbounded full-history retention (Postgres eventually; see `docs/HISTORY_OVERHAUL_PLAN.md`) |
| `lock:guild:{id}:recovery` | String | `"1"`, TTL 60s (SET NX — distributed lock) |
| `ytdl:stream:{webpage_url}` | String | JSON stream data; TTL = `expire − now − 1800s` |
| `ytdl:source:{normalized search}` | String | Search→`(webpage_url, title)` resolution; TTL 1h |
| `spotify:track:{id}` | String | `"Title Artist"` search string; TTL 24h |
| `spotify:playlist:{id}` | String | JSON array of track titles; TTL 1h (user-editable) |

Redis uses `maxmemory-policy volatile-lru` at 256 MB: only TTL-carrying keys (yt-dlp/Spotify caches, the TTL-managed guild keys) are eviction candidates — the persistent history keys never are. Do not switch back to `allkeys-*`; that would let memory pressure silently destroy full play history.

### Crash recovery

On `on_ready` (cold start or session loss — not WebSocket resume), `_restore_guild` runs per guild:
1. Acquires the recovery lock (SET NX) — prevents two bot instances from racing.
2. Reads one `get_recovery_gate()` — a lightweight pipeline of the state hash + queue **length** (LLEN, never the contents). `None` means the Redis read failed → skip with a warning (retried on next `on_ready`); an empty gate means nothing stored → skip silently. The queue/now-playing/history *payload* is deliberately NOT read here: a `-stop`ped guild keeps its (possibly long) queue list, so gating on LLEN keeps that payload off the wire on every `on_ready`. The contents are re-read once by `_restore_state` after a successful connect.
3. Gates: `gate.state.has_active_connection` (voice/text channel pair persisted), then `gate.has_restorable_playback` (pending entries or a crashed song).
4. Connects to voice. Creates and starts a new `MusicPlayer`.
5. `_restore_state()` task: reads its own snapshot (the single round-trip covers everything — a failure aborts the whole restore rather than fabricating partial state); restores volume (only if stored); rebuilds the crashed song via `SongQueueEntry.from_crashed_state()` → `queue.restore_crashed()` (at-most-once: clears `current_song_url` immediately after, even when the requester is unresolvable); restores pending entries via `queue.restore_entries()`; restores history via `history.restore()`; refreshes TTLs.
6. Releases lock.

**Intentional stop vs crash**: `cleanup()` calls `store.clear_connection()` which clears `voice_channel_id` and `text_channel_id`. `on_ready` skips recovery for any guild where these fields are empty.

### Concurrency model

Single asyncio event loop. All I/O is async. yt-dlp extraction is offloaded to:

```python
_YTDLP_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ytdlp")
```

All `loop.run_in_executor(_YTDLP_POOL, _ytdlp_extract, ...)` calls share this pool. Max 8 concurrent extractions across all guilds.

Synchronization primitives:
- `asyncio.Event play_next` — song completion signal from discord.py audio thread via `call_soon_threadsafe`
- `GuildQueue._mutex` — serializes bulk queue mutations (clear/shuffle/remove) and the loop's `try_commit_dequeue()`; private to the queue class
- `Spotify._auth_lock` — double-checked locking for token refresh

### Audio pipeline

`vc.play(YTDL)` → discord.py reader thread reads Opus frames from FFmpeg stdout → encrypts with NaCl → UDP to Discord voice servers.

FFmpeg input flags (`before_options`): `-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5`
FFmpeg output flags (`options`): `-vn` (+ `-ss {ts}` for timestamp seeks, + `-filter:a volume={v}` for non-unity volume)

Volume changes take effect on the **next song** (the FFmpeg process for the current song is already running).

### Key invariants to preserve when making changes

1. **All queue mutations go through `GuildQueue`.** The three representations are private to the class, which keeps them in sync structurally — never add a parallel queue collection or reach into `queue._pending`/`queue._display` from production code. Every mutation that touches the Redis mirror runs under the class's one mutex, and bulk mutations carry a dequeued-but-uncommitted head through untouched (`_in_flight_head`). One residual window is accepted by design: a bulk mutation scheduled in the single event-loop tick between `try_commit_dequeue()` and the store's start-transaction LPOP can race it server-side (the start transaction is a store-level atomicity boundary — see the `guild_queue.py` module docstring).
2. **Cancel prefetch before bulk queue mutations.** The `MusicPlayer` wrappers (`queue_clear`/`queue_shuffle`/`queue_remove`) call `_cancel_prefetch()` before delegating: a still-running prefetch holds an item from `get_nowait()`, and cancellation returns it to the front of the pending queue (`requeue_front`) so the mutation processes it too. (A prefetch that already completed is fine either way — its item is a live in-flight head that `shuffle`/`remove` carry through.)
3. **Every `queue.get()`/`get_nowait()` is retired exactly once** — by `task_done()` (loop end / discard paths), by `finish_failed_dequeue()` (failure paths, including prefetch failures), or by `requeue_front()` (prefetch cancellation — transfers the slot instead of closing it). The loop's exception handler balances a committed dequeue via the `dequeue_owed` flag. `GuildQueue`'s own drains balance internally.
4. **`current_song_url` is cleared in Redis on normal song end.** Only a non-empty value at startup means a crash. Do not leave it populated after normal completion.
5. **`clear_connection()` is called on clean shutdown.** This is what prevents `on_ready` from re-joining voice after an intentional `-stop`. (The Redis queue list is intentionally left intact.)
6. **`persisted=False` items (the crash-recovered "current song") never touch the Redis list.** They are injected in-memory only, filtered out of rebuilds, and skipped by `redis_pop_for()`.
7. **The wire formats in `src/guild_state.py` are pinned by golden-fixture tests.** Changing a serializer must keep old entries readable (rolling restarts mix writers).
8. **The Now Playing embed block lives in exactly one host message at a time** (the newest bot message in the player's channel — see `docs/NOW_PLAYING_EMBED_ATTACH_PLAN.md`). Host swaps are pointer-first and synchronous (`_adopt_np_host`); all mutations of an *old* host (progress-tick edits, retires) go through `_np_edit_lock` so a strip/delete is always the final write. Every attach site must adopt through `_adopt_np_host_if_current(message, own, song)` — the send's await can cross a song boundary, and adopting a stale block would delete the next song's fresh host (the gate sheds the stale block instead). Never send to the player's channel with a bare `channel.send()` while a song is live — go through `ctx.send` (MusicContext attaches the block) or `mp.send_with_np()`.
