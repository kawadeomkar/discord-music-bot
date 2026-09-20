---
paths:
  - "src/youtube.py"
  - "src/ytdlp_pool.py"
  - "src/sources.py"
  - "src/spotify.py"
  - "tests/test_{youtube,ytdlp_pool,sources,spotify}.py"
---

# Extraction: the yt-dlp pool, client strategy, stream healing and Spotify

The path-scoped half of CLAUDE.md: its golden rules apply here, and cite by number.

## Architecture

### yt-dlp: process pool, client strategy, caching, healing

**Process pool** (`ytdlp_pool.py`): extraction is half GIL-bound (JSON, signature
decryption, format selection), so it runs on a `ProcessPoolExecutor`
(`YTDLP_POOL_WORKERS`, default 4; ~80–120 MB RSS each). Lifecycle only — the callable is
supplied per call, which is the seam tests use. Lazy creation (workers re-import parent
modules under spawn); `prewarm()` from setup_hook; a `BrokenProcessPool` (e.g. OOM-killed
worker) is healed by rebuild-and-retry ONCE, and `max_tasks_per_child=16`
(`_MAX_TASKS_PER_CHILD`) replaces a worker before it grows enough to get there — RSS
climbs ~5 MB per extraction and never flattens, so 16 caps a worker near 300 MB. The
start method is passed EXPLICITLY (`_pool_context`): a task budget with no `mp_context`
makes CPython force `spawn`, which on Linux replaces 3.14's forkserver default and
measured 23–30× slower worker startup. The chart pool opts out
(`recycle_workers=False`): its one worker would re-import matplotlib on every
replacement. Worker logs travel a
multiprocessing Queue → parent `QueueListener` → the parent's handlers (so yt-dlp's
SABR/PO-token/signature warnings — the early-warning system for YouTube rule changes —
reach Loki structured, with `worker_id` and propagated `trace_id`). A second, OPT-IN queue
(`progress_sink=`) carries a playlist resolve's per-entry progress out to the
queue-progress card: bounded, drained on a THREAD, and **rebuilt on a break-heal**, unlike
the log queue beside it — a worker SIGKILLed mid-write holds the queue's `_wlock` forever,
so reusing it across a heal is exactly the case where every later `put_nowait` raises
`Full`, silently and for the life of the process. Two separate rules govern its shutdown:
`cancel_join_thread()` in the worker, without which every restart during a playlist
resolve costs the full 10s shutdown timeout; and stopping by a FLAG after the join rather
than a sentinel, which makes the workers' state irrelevant — after `terminate_workers()`
that `_wlock` is a POSIX semaphore nothing will release, so a parent write could block
forever. See docs/ARCHITECTURE.md#progress-out-of-a-yt-dlp-worker. Results are made
picklable and small in the worker: exceptions flattened to `ExtractionError` (yt-dlp's
own exceptions carry live tracebacks and can't cross), successes `_slim_info`'d
(sanitize + drop `formats`/`thumbnails`/etc., commonly 100 KB–1 MB nobody reads). The
one thing kept out of `formats` is the **fallback audio ladder**: `_slim_info` mines
the top `_STREAM_CANDIDATES` (3) audio-only formats into `audio_candidates` first,
because after the drop those URLs exist nowhere in the process.

**Client strategy** (comment block above `_EXTRACTOR_ARGS` in youtube.py):
the config names **no client** — it passes `default`, yt-dlp's own list, which is
`visionos,web` today and was `android_vr`-led before. Tracking upstream's default IS the strategy: yt-dlp moves it
when YouTube breaks a client. Any client name in these docs is a record of what
`default` resolved to at the time — **re-verify on every yt-dlp bump**. `visionos`
carries playback (no PO token, no JS player, audio-only 251/opus over https); `web` is
the fallback, and both extras exist to keep it usable — yt-dlp DROPS `web` from
`default` when no JS runtime is present, so Deno (`deno` extra) + yt-dlp-ejs (`default`
extra) is what keeps a fallback at all, and the **bgutil PO-token sidecar** (compose
service on :4416, plugin pin hand-checked in lockstep — NOT covered by `just pins`, see
rule 6a) mints the GVS token `web`'s formats need. Format ladder
`bestaudio/best[height<=360]/best` — the 360p cap matters: on the muxed fallback rung,
plain `best` would stream ~120 MB of 1080p video per song just for ffmpeg's `-vn` to
discard. `_record_serving_format` warns once per format_id when serves degrade to
muxed/HLS (the observable symptom of the primary path being down). `youtubetab:skip=
webpage` drops the 878 KB homepage every search and playlist extraction used to open
with — read by youtube:search and youtube:tab alike, and inert without cookies. Degradation ladder is
designed so every rung lands on a previously-working configuration.

**Revoked-URL healing** (`_resolve_playable_stream`): a revoked URL fails in the worst
way — ffmpeg 403s and exits, discord.py reports "song finished", silence. So every URL
is probed pre-play, and the probe walks the mined ladder: **sideways before in place.**
The next format is a ~100 ms probe against a 3–5 s re-extraction that would re-select
the *same* format — curing a stale URL but not a format YouTube has stopped serving. A
winning fallback is promoted onto the info-dict wholesale (URL plus format shape, so
`_record_serving_format` and the source's abr/asr/acodec describe what is really
playing) and the entry is **rewritten**, or a cached ladder re-probes its own dead head
every play until the TTL lapses. A freshly extracted dict is copied before the walk,
since `_extract_once` shares it with every joiner. Only when every rung is dead is the
entry dropped and re-extracted once. Prefetch and the stream warm probe only the head.
A URL revoked in the seconds between probe and first read is caught post-hoc by
`produced_audio`, and the loop then retries the song (see playback.md). The probe is **tri-state** (`StreamProbe`), and the third
value is load-bearing: a probe that never completed is `UNCONFIRMED`, not `DEAD` and not
`PLAYABLE`. Read as `DEAD` it would fail songs over a blocked probe; read as `PLAYABLE`
the URL gets **cached**, which is how one unreachable CDN edge made a single song
unplayable for a full 30-minute TTL. So an unconfirmed URL still plays (ffmpeg judges
it) and is cached for `_UNCONFIRMED_STREAM_TTL` (120s) — probe failures are
process-wide, so declining the write would stop anything repopulating the cache.
**UNCONFIRMED also ends the ladder walk** where it stands and promotes that rung: every
candidate shares one host and one `expire`, so a probe that could not complete for one
rung will not complete for the next. An
unconfirmed **cached** URL is dropped and re-extracted for a freshly signed one, which
lands on the same edge and format and so cures an early revocation; that drop is FREE
(never charged against `_MAX_STREAM_EXTRACTIONS`, which is **1**), is suppressed once
`probe_path_looks_broken()` says the probe rather than the URL is at fault, and is
declined by the background prefetch (`allow_reextract=False`), whose cancellation every
bulk mutation waits on. HTTP 429/5xx are UNCONFIRMED, not DEAD.
See `docs/ARCHITECTURE.md#yt-dlp-client-strategy` for the measurements behind both.
`_stream_url_ttl` reads `expire` from both query-string
(https formats) and path-segment (`/expire/<epoch>/`, HLS) forms, then caps at 30min.

**FFmpeg**: `YTDL(discord.FFmpegOpusAudio)` with
`-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5` and `-vn`; `?t=`/interject
seeks are a **two-pass `-ss`** — `-ss N` before `-i` for the HTTP range request, `-ss 0`
after it to drop the pre-roll that lands in. Output-side alone downloads and decodes
from 0:00, which YouTube throttles to a stall on a deep offset; input-side alone lands
on the nearest webm cluster, measured **5–10s early**, which `position_secs` would then
overstate everywhere. Volume via `-filter:a volume=` (which is why `-volume` applies from the
song after next — the prefetch has already built the next one at the old level, and
rebuilding it would re-request a signed URL that may since have been revoked).
**Opus passthrough**: `codec="copy"` remuxes instead of re-encoding. `_passthrough_codec`
is the gate and all four clauses are required, because `-c:a copy` also discards the
`-ac 2 -ar 48000 -b:a 128k` discord.py always emits: `acodec` opus; no filter (ffmpeg
refuses copy alongside a filtergraph — exit 234, zero bytes — so `yt_stream` asks
`_audio_filters` what it produced rather than re-testing volume); `audio_channels` in
(1, 2) (a 5.1 serve reaches Discord as multistream and clients decode only the front
pair); and `format_id` in `{249, 250, 251}`, which stands in for the 20 ms frame
duration the info-dict does not report. Absent fields mean re-encode.
`read()` counts AUDIO frames (the first two packets are OpusHead and OpusTags, which
discord.py yields like any other) → `elapsed_secs`/`position_secs` is the single source
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
`SearchQueueEntry`s resolved per-song at dequeue, each carrying the requester's ID
(`spotify_playlist_to_ytsearch` requires it): the resolve runs long after the command,
when `_last_author` is whoever typed most recently.

## Concurrency primitives

| Primitive | Protects |
|---|---|
| `youtube.prefetch_warm_slot()` (semaphore, process-wide) | how many enqueue-time stream warms may hold a worker. A search resolves flat and leaves the stream to `prefetch_stream`, which `queue_put` spawns per song and nobody awaits — so those never pass through `resolves` and would otherwise be bounded only by `PLAY_INFLIGHT_MAX`. Half the pool, and NOT per guild: the harm is a warm queued ahead of another guild's in-band resolve. The loop's own one-ahead prefetch takes `_stream_source` instead and never waits here |
| `_playlist_slot()` (semaphore, src/spotify.py, process-wide) | how many Spotify playlist walks run at once (2) — Spotify's rate limiter is per application, and a walk is up to 100 requests. The wait is bounded by `PLAY_RESOLVE_WAIT_SECS` (`SpotifyBusyError`), and the semaphore is rebuilt when the running loop changes |
| `_INFLIGHT_PLAYLISTS` / `_PLAYLIST_SUBSCRIBERS` (src/spotify.py, process-wide) | one walk per playlist, awaited through `asyncio.shield` by every caller, each of whose cards receives the walk's page reports; the job writes the cache itself, so a cancelled caller costs nothing |
| `_PROGRESS_SUBSCRIBERS` (src/youtube.py, process-wide) | the cards watching a YouTube playlist extraction. Mutated on the event loop, READ on the yt-dlp pool's progress drain thread (`_publish_progress` copies the list before iterating), so a report can never touch the loop |
| `Spotify._auth_lock` | token refresh double-fire |

## Recipes

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
