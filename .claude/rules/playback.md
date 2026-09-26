---
paths:
  - "src/musicplayer.py"
  - "src/guild_queue.py"
  - "src/play_placement.py"
  - "src/play_pipeline.py"
  - "src/queue_progress.py"
  - "src/queue_rows.py"
  - "src/musicbot.py"
  - "src/main.py"
  - "src/util.py"
  - "src/commands/{clear,join,jump,now,pause,play,queue,remove,replay,resume,shuffle,skip,stop,volume}.py"
  - "src/commands/_common.py"
  - "tests/test_{musicplayer,guild_queue,play_placement,play_pipeline,queue_progress,queue_rows,musicbot,main,util}.py"
  - "tests/commands/test_{clear,join,now,pause,play,queue,remove,replay,resume,shuffle,skip,stop,volume}.py"
---

# Playback: `-play`, the queue, interjection and the Now Playing host

The path-scoped half of CLAUDE.md: its golden rules apply here, and cite by number.

## Architecture

### The life of `-play` — the three-phase yt-dlp pipeline

The core performance design: metadata resolution, stream extraction, and playback are
three separate phases so queueing is instant and songs start with near-zero latency.

```
-play <input>
  │ cog_before_invoke: bind structlog ctx, open command span, get_mp() (creates+starts
  │                    MusicPlayer if absent), persist voice/text channel IDs to Redis
  │ validate_commands: author must be in a usable voice channel
  ▼
play():
  ├─ split_play_args: strips a LEADING RUN of options off the argument, one per
  │        _PLAY_OPTIONS field (--now/--next share `mode`, so they exclude each
  │        other; --timestamp <time> sets its own), in any order. The parser names
  │        no option — the registry drives parsing, refusals and the
  │        did-you-mean. A near-miss like -now becomes a hint, not a search for
  │        "now <url>"; a repeat, a conflict or an unreadable value queues nothing
  ├─ PlayRegistry.register: admit to the guild's in-flight set (PLAY_INFLIGHT_MAX,
  │        default 16, declined past it), snapshot the queue generation. Requests
  │        resolve CONCURRENTLY; only the insert is serialized — see .place()
  ├─ song live? ──► _interject_flow:
  │      • --now                     → interrupt, park a resume tail (resume_paused=True)
  │      • plain -play + PAUSED song → same flow, resume_paused=False ("-play means play")
  │      • --next                    → NOT here: it never interrupts, paused or not
  ├─ parse_input (sources.py): one token once Discord's wrappers are off → parse_url, a
  │        linear, anchored link test (youtube/spotify/soundcloud; any other http(s)
  │        host → URLSource.OTHER for yt-dlp; a Spotify link that names nothing
  │        playable raises UnsupportedSpotifyLinkError before any join); else ytsearch.
  │        is_link is the one link-or-text verdict. docs/ARCHITECTURE.md#source-resolution
  ├─ --timestamp given? start_offset_refusal reads the PARSED source, so a
  │        collection naming no track is answered before the join. Applied in
  │        queue_source (beating the link's own t=), then past_end_refusal on the
  │        resolved duration — after the join, so the cold start is torn back down
  ├─ placement (Placement enum — the insert position, decided separately from
  │        cold_start, which also drives the analytics shortcut and the join dance):
  │      • disconnected              → COLD_FRONT
  │      • --now / --next with nothing live → NEXT (queue_put_next: it neutralizes
  │        the prefetch first, or the loop's open claim makes a front insert land SECOND)
  │      • otherwise                 → TAIL
  ├─ cold start:
  │      • defer_playback() hold (gate stays shut so a Redis-restored queue head
  │        can't start while this input resolves), RELEASED at the put — the
  │        confirmation embed is presentation, not something the first note waits on
  │      • join launched CONCURRENTLY with queue_source (no data dependency);
  │        any failure cancels join and runs full cleanup() (zombie-loop prevention);
  │        -join's own 👋 + latency line are SPAWNED, so waiters get the handshake
  ├─ EVERY placement, TAIL included: wait_for_restore() BEFORE the insert —
  │        ordering is load-bearing: put_front LPUSHes Redis, restore_entries replays
  │        entries already on that list in-memory-only, so inserting first
  │        double-queues this song; and a put() before the replay lands this song
  │        ahead of entries Redis already lists behind it
  ▼
PHASE 1 — RESOLVE (enqueue time, instant on repeats):
  queue_source → YTDL.yt_source: check ytdl:source:{normalized query} (TTL 24h,
  refreshed behind the reply past 1h).
  Miss, and the caller passed ResolveMode.FLAT_OK for a SEARCH → one flat search
  POST: identity only, ~0.6s, ytdl:source alone (the stream URL comes from phase
  2's prefetch). A COLD START is not FLAT_OK: its song plays immediately, so the
  stream extraction is on the path to audio either way and flat would only move the
  failure past the join. What the flat path gives up on the placements that take it
  is the enqueue-time playability check — an age-gated, region-blocked or
  members-only video has an id, a title and a duration, so it queues and fails at
  its turn instead of failing the command. A live/duration-less first result
  declines — after paying the flat POST — and falls through to:
  Miss on a LINK, an interjection head, or a declined flat entry → ONE unified
  stream-opts extraction returns identity AND a selected playable stream URL, so
  both caches are written from a single network round (probe first — see phase 2).
  The stream write is STARTED, not awaited: the reply needs identity, and the probe
  behind it is a network round trip. Everything that reads that cache joins the
  same job through `_stream_cache_get`, so nothing extracts the URL twice.
  See docs/ARCHITECTURE.md#resolve-mode and #warming-the-stream-cache.
  Spotify track → title search; Spotify album/playlist → titles → YTSource ytsearch
  entries (resolved lazily at dequeue); YouTube playlist → flat extraction to
  QueueObjects. Enqueue via GuildQueue.put (batch = one round trip per `_PUT_CHUNK` entries for
  playlists, so a 10,000-track paste yields to the event loop between chunks).
  A COLLECTION that outlives its server's queue-progress-delay gets a live card here
  (src/queue_progress.py), deleted when the enqueue lands. TWO entry points, since
  `-playnow <playlist>` returns above _resolve_and_place: that one and
  interject_flow. It is a SECOND message through ctx.channel.send — merging it
  into the confirmation would stop that message re-hosting the NP block.
  See docs/ARCHITECTURE.md#queue-progress-card.
  ▼
PHASE 2 — PREFETCH (background):
  • per-song prefetch_stream task at enqueue (skipped for bulk playlists — N
    concurrent extractions would mint URLs that expire before playback)
  • _prefetch_next_song: while song N plays, song N+1 is fully resolved AND its
    YTDL/FFmpeg source constructed, cached in ytdl:stream:{webpage_url}
    (TTL = min(URL expire − 30min, 30min) — YouTube revokes well before `expire`)
  • every candidate URL is PROBED with a plain no-Range GET (HEAD and ranged GETs
    lie about revoked URLs), handed to aiohttp PRE-ENCODED — yarl requotes a plain
    string and an HLS manifest signs its own path; only proven-playable URLs are
    cached, stamped `probed_at` so the play seconds later skips a second probe
  ▼
PHASE 3 — STREAM (playback loop, usually zero extraction):
  loop(): gate open → dequeue → resolve (if YTSource) → yt_stream (cache hit →
  no yt-dlp call) → rebuild if the volume changed → vc.play(YTDL) → atomic Redis
  start transaction →
  NP embed + 3s progress updater → spawn prefetch for next → play_next.wait()
  → history add, clear transient state, next iteration
```

The playback loop (`MusicPlayer.loop`, bottom of musicplayer.py) is the most delicate
code in the repo. Its bookkeeping invariants:

- `claim_outstanding` tracks an unsettled `queue.get()` so the outer exception handler can
  settle the claim, so `_cursor` never drifts.
- `commit_dequeue()` (under the queue mutex) detects "queue cleared while this song
  resolved" — the song is discarded and its FFmpeg subprocess `cleanup()`ed (leak
  otherwise).
- The Redis start write is `pop_queue_and_start_song` (MULTI/EXEC: LPOP + state HSET +
  now_playing HSET) so a crash can never observe the song absent from both the queue and
  `current_song_url`. Crash-recovered songs use `set_current_song_state` (no LPOP —
  they were never on the Redis list; `persisted=False` on the QueueObject encodes this,
  read only via `guild_queue.is_persisted()`).
- `play_start_epoch` is **backdated by the FFmpeg `-ss` start offset** so recovery math
  yields true audio position for `?t=` starts and double-crash recoveries.
- Stream-never-opened detection: `stream_failed = not song.produced_audio and
  play_error[0] is not None` — zero frames alone also describes a paused-parked song;
  an error alone also describes a mid-song death that earned its history entry. A dead
  stream drops the cached URL and gets **`_STREAM_PLAY_ATTEMPTS` (3) plays total**:
  `_retry_failed_stream` re-queues it next (`queue_put_next`) carrying
  `stream_attempts + 1`, and from the second retry the format that failed, so the fresh
  resolve walks the rest of the ladder first; the terminal failure falls through to
  `_handle_dead_stream`'s red embed, and `_STREAM_FAILURE_CIRCUIT` (2) consecutive dead
  songs withdraw the retries until one produces audio. Two placements are load-bearing:
  the retry DECISION and `_neutralize_prefetch()` run where `stream_failed` is computed,
  because by the iteration tail the loop has claimed `_prefetch_task` into a local and
  the already-resolved next song would play instead of the retry; and the history write
  carries `and not retrying`, so a retried fragment never records and the terminal
  attempt records exactly once. A `--now` that parked a tail inside the death window IS
  the requeue (`folded_into_tail`), stamped with the spent budget. Mid-song death is not
  retried: it produced audio and earned its entry.
  discord.py **does** report a failing ffmpeg — `FFmpegOpusAudio.read()` calls
  `_check_process_returncode()` on an empty packet, which reaches `after` as
  `FFmpegProcessError`. So `stream_failed` is the main path and the retry flow above
  owns it. The one window that check declines to judge is `poll()` returning None — a
  child that closed stdout but has not been reaped — and `_drop_unplayable_stream_cache`
  is the backstop for it, guarded by `note_deliberate_stop()` (a stop we initiate ends
  the player thread without another `read()`, so it also arrives as `error=None`) and by
  `start_paused`. Cache only: a false positive costs one re-extraction, while widening
  `stream_failed` would eat a real history entry.
- Idle disconnect: `queue_get` times out at the server's `idle-timeout` (5:00 by
  default, at most 30:00), read from `GuildSettings`' cache once per wait. Resolving the
  entry it dequeued keeps its own 300s bound (`_IN_BAND_RESOLVE_TIMEOUT_SECS`), so a
  wedged resolve is never held for the length of a long wait. The playback gate itself
  times out at `DEFAULT_IDLE_TIMEOUT_SECS`, 300s (a player built by a command that never
  connects must not leak forever) — unless a `defer_playback` hold is outstanding, which
  means a command is mid-join.

**`-resume` is the second cold-start path.** With the bot out of voice there is nothing
to un-pause — the paused song went with the voice client — but the queue outlives it in
Redis under a 24h TTL, so `-resume` joins the author's channel and lets that queue play.
It differs from `-play`'s cold path in two ways that are not stylistic: it inserts
**nothing** (so the `wait_for_restore`-before-`put_front` rule is moot, and the head it
describes is the song that plays), and it restores **before** joining rather than
concurrently, because there is no 1–4s extraction to hide the handshake behind and
joining first would park the bot in a channel for an empty queue. It refuses to reuse a
player failing `can_rejoin_cold()` (a song still held, or a gate already open, with no
voice client — an eject that never reached `on_voice_state_update`), rebuilding instead.
`max_concurrency(1, guild)` is load-bearing: two racing invocations both read
`voice_client is None`, so `validate_commands`' "already being used in channel X" check
cannot fire for either, and the second would move the bot to its own author's channel.

### Per-guild object graph

One `MusicPlayer` per guild, registered in `MusicBot.mps: dict[int, MusicPlayer]`
(`get_mp` creates + `start()`s lazily; `cleanup()` atomically pops — first caller wins,
concurrent callers no-op). Each player owns:

- `queue: GuildQueue`, `history: GuildHistory`, `store: Optional[GuildRedisStore]`
- tasks: `_player` (loop), `_prefetch_task`, `_restore_task`, `_progress_task`,
  `_heartbeat_task`, `_pause_debounce_task`, plus `_background_tasks`
  (fire-and-forget via `spawn_background`)
- events: `play_next`, `_restore_complete`, `_playback_gate` (+ `_playback_holds`
  refcount for `defer_playback()`)
- NP host state: `_np_host_message` / `_np_host_own_embeds` / `_np_host_dedicated` /
  `_np_edit_lock`

The gate hold lives on a nested `AsyncExitStack` inside `_resolve_and_place`, and the
enqueue helpers take a `release_hold` callable they invoke the moment the put lands:
the put is the whole of what the hold waits for, and behind it sat one or two Discord
round trips. `AsyncExitStack.aclose()` is idempotent, so every path that does NOT place
still releases through the outer stack unchanged — which is what `_abandon_cold_start`'s
hold-count read depends on.

`cleanup(guild)` cancels all six tasks BEFORE disconnecting (so the loop can't start
the next song mid-teardown), retires the NP host, disconnects voice, resets presence,
and — for an intentional stop — `clear_connection()` so `on_ready` skips recovery.

### GuildQueue: one deque and a cursor

A guild's queue is **one deque plus an index into it**, privately owned by `GuildQueue`,
mirrored to Redis:

| | | |
|---|---|---|
| `_items[:_cursor]` | claimed by a consumer, not yet settled | the "in-flight head" |
| `_items[_cursor:]` | pending | what `get()` hands out |
| `_wake` | `asyncio.Event`, set iff something is pending | I3 |
| Redis mirror | `guild:{id}:queue` list | the `is_persisted()` subset, in order |

The cursor is the boundary and NOT a per-item flag, because Redis retires entries by
LPOP — so in-flight items are necessarily a **prefix** (I6). This replaced an
`asyncio.Queue` + a parallel `deque` whose agreement had to be maintained by hand.

Rules encoded in the class (violating any of these corrupts the queue or Redis):

- Every multi-leg mutation (`put`, `put_front`, `clear`, `shuffle`, `remove`,
  `finish_failed_dequeue`) runs under one bulk-mutation mutex.
- A dequeue is **two-phase**: `get()` advances `_cursor`; the item and the Redis LPOP
  settle later via `commit_dequeue()` / `redis_pop_for()` (or are undone via
  `requeue_front()` / retired via `finish_failed_dequeue()`). `put_front` inserts at
  `_cursor`, which IS inserting behind the in-flight head.
- **`_sync_wake()` is the only writer of `_wake`.** A stale set
  does not degrade: `Event.wait()` returns without yielding when already set, so `get()`'s
  wait loop loses its suspension point and the whole event loop stops — measured at
  2,000,001 iterations with 0 other loop ticks. The wait is a `while`, never an `if`:
  `Event` wakes every waiter, and the prefetch's `get_nowait()` is a second consumer.
- **Every cursor decrement is guarded** (`try_release`, `requeue_front`). Unguarded it goes
  negative and the write that follows lands at `_items[-1]` — the TAIL. `clear()` resets it
  to 0 alongside the deque; without that, `qsize()` returns negative and the next release
  pops an empty deque. Tests assert all of this against the module source.
- **`qsize()` is PENDING, `display_size()` is pending PLUS in-flight.** One term apart over
  the same two fields, so a swap compiles and type-checks; `display_size()` is the sole
  input to `play_history.queue_position`, so a swap writes a plausible wrong number to
  Postgres forever.
- Callers with a prefetch task must settle it BEFORE clear/shuffle/remove so the
  prefetch's `CancelledError` handler `requeue_front()`s its item into the drain.
  `-clear` uses `_cancel_prefetch()`; **`-shuffle` and `-remove` use
  `_neutralize_prefetch()`**, because `cancel_task()` no-ops on a COMPLETED prefetch
  and its surviving claim would pin that song to the front of the reorder, or leave
  the next song unremovable for the whole current song. `-remove` settles only when
  the claimed head matches (`_claimed_head_matches`, which also tries the head's
  requeued form), and both refill the slot in a `finally` unless the player is
  `retired`: an orphan prefetch on a torn-down player can LPOP the saved queue.
- `clear()` invalidates in-flight work through the generation counter and the cursor
  reset ALONE — a prefetched song the loop is holding is discarded because
  `commit_dequeue` refuses (nothing is claimed once the cursor is 0). There was once
  a cleared-flag beside them; it was read once per loop iteration, so a `clear()` landing
  after that read survived an entire song and destroyed a song claimed long after it,
  leaking the claim. Do not reintroduce a level flag here.
- `restore_crashed` / `restore_entries` write the deque ONLY (entries are already on /
  never were on the Redis list, respectively).
- Redis rebuilds (`rebuild_queue`) are MULTI DELETE+RPUSH so a concurrent LPOP never
  observes an empty-window queue.
- Every mirror write **from a bulk mutation** — `clear`, `shuffle`, `remove`,
  `finish_failed_dequeue` — goes through `_write_mirror(items, *, removed=())`, which
  owns the rebuild / DELETE / LREM choice. The APPEND paths deliberately do not:
  `put`/`put_front` call `push_queue`/`push_queue_batch`/`push_queue_front` directly,
  because routing an append through `_write_mirror` turns an O(1) RPUSH into a full
  rebuild under the mutex on every `-play`. They join it only while `mirror_dirty`
  says the list is the wrong shape, where the rebuild IS the repair. Empty means
  DELETE, never skip. **Only a removal may
  pass `removed`** — LREM asserts the survivors kept their order, which is false for a
  shuffle, for an insert, and for a stale list. Four clauses gate the shortcut: `_LREM_MAX_ENTRIES` (16),
  `_LREM_MAX_SHARE` (one in five), `_claimed_blobs()`, and `mirror_dirty`, which
  refuses it over a list whose order is already unknown. **The count is the bound that
  matters**: LREM is `O(position)`, so N of them cost `O(N × depth)` against a rebuild's
  `O(depth)` — the depth cancels and the crossover is a COUNT, near 18 at the low end of
  two measurements. It is not a ratio; an earlier revision said it was and admitted
  200-entry LREMs that cost 1.6× the rebuild while holding one MULTI/EXEC, which stalls
  every guild, not just the one removing. A test pins the value (`≤ 18`) because the
  other tests size their input from the constant and move with it.
- **A swapped-in item keeps the entry the list holds.** `requeue_front` may hand back a
  claimed item's resolved or rebuilt form, which serializes differently from its entry.
  `_listed` records the replaced entry, and `_mirror_entry()` serializes the item as it
  for every byte-exact write: the rebuild, the LREM and `_claimed_blobs()`. Without it a
  claimed swap hides from `_claimed_blobs()`, an LREM takes its entry instead of a
  byte-identical twin's, and the song start's LPOP retires the next song's.
- `remove()` takes a **predicate**, and `remove_matcher()` beside the class owns the
  policy: resolved yt-dlp URL first, then `user_input`. Links compare literally, text
  casefolds — folding a link would let one Spotify playlist's base62 id match another's.

### `-play --now` / `--next` / `--timestamp` placement, interjection and resume entries

`MusicPlayer.interject(qobj, vc, resume_paused)` implements "play this now, then put the
interrupted song back where it was":

1. `_neutralize_prefetch()` first — a completed prefetch bypasses the queue and would
   play INSTEAD of the interjection. Claim-then-settle: `_prefetch_task` is nulled
   synchronously on both sides (interject and the loop) so exactly one consumer sees any
   given prefetch result. A completed prefetch is rebuilt into an equivalent QueueObject
   (carrying `-ss` offset and every flag), `requeue_front()`ed, and its FFmpeg subprocess
   killed.
2. Capture the current song's frame-counted `position_secs` (frozen during pause), build
   a resume `QueueObject(ts=position, is_resume=True, start_paused=was_paused &&
   resume_paused)` — skipped when < 5s remain (`_MIN_RESUME_REMAINING_SECS`), position
   EOF-capped at duration − 10s.
3. `queue.put_front([qobj, resume])` (both persisted → LPUSHed, so crash recovery
   mid-interjection works unchanged), then `vc.stop()` — only if the measured song is
   still current.
4. **Stacking**: interjecting over an interjection parks that song too, in front of the
   tails already waiting, so the queue unwinds LIFO and every parked song returns.
   Unbounded by design (each interjection pays a 1–4s resolve first); `ts` is absolute at
   every level, so a tail of a tail resumes where it actually stopped. Depth rides the
   span as `interject.depth` (`GuildQueue.resume_tail_depth`), and `interjected` is now
   attribution only — its one behavioural read was the replace gate.
5. History: the interrupted song is recorded ONCE, when its resume tail finishes
   (`_skip_history_for` holds the song's identity, not a boolean — a stale flag would eat
   the next song's entry). One slot suffices at any depth: each interjection stops
   exactly one song, whose iteration consumes the marker before the next can land. A
   parked tail destroyed by `-clear`/`-remove` before it can play is recorded there
   instead (`MusicPlayer._flush_played`) — a queue object is recorded exactly once, when
   it leaves the queue for good. A song abandoned *mid-play* has no queue object at all
   (its entry was LPOPed at start), so `cog.cleanup` claims it synchronously before any
   await — `claim_current_song_for_history()` — and writes it alongside the teardown.
   That claim takes the same `_skip_history_for` marker, which is what keeps the two
   writers from both recording; it declines when the marker already names the song,
   because then a parked tail survives in Redis and records the play on `-resume`.

`-play` while paused routes through the same flow with `resume_paused=False` (the
interrupted song comes back PLAYING — "-play means play"); `--now` restores the exact
paused state (`start_paused` re-pauses the player thread synchronously at `vc.play`,
before any await, leaking at most a frame or two). **`--next` is carved out of the
paused branch** — the request is not buried behind the paused song, it IS next, so
interjecting would stop the song the user chose to keep.

**Every placement takes a playlist in full.** `--now` interjects the head and puts the
rest between it and the resume entry, so the interrupted song returns after the WHOLE
playlist — deliberate, stated in the confirmation, and undone by one `-remove <the
link>` (which matches on `user_input`, carried by every track). Only the head is
resolved and stream-warmed; a Spotify collection's tail stays lazy `YTSource`s.

`--next` front-inserts without interrupting, via `MusicPlayer.queue_put_next` —
`_neutralize_prefetch()` then `put_front`, because `loop()`'s prefetch holds a claim
for the whole current song and a bare `put_front` lands BEHIND it (the song would play
second). It deliberately does not re-spawn the prefetch: `_prefetch_task` is one slot
under a claim-then-null protocol with `loop()`, and a re-spawn racing the loop's own
would strand a claim and drift `_cursor` permanently. Both flags are gated by the
same-channel rule (`play_takes_the_queue`): line-jumping is queue control, like
`-skip`/`-shuffle`/`-remove`.

### The Now Playing host system

The NP card (embed with a 10-segment live progress bar, edited every
`NOW_PLAYING_UPDATE_INTERVAL_SECS` = 3.0s) stays glued to the bottom of the channel.
**Five sends are documented exceptions**, and they bypass for two distinct reasons.
`-ping`, `-debug` and the queue-progress card (`dashboard.LiveMessage.start`), plus
the alone-disconnect countdown card (`VoiceWatchdog._send_card`), bypass because a
message an edit loop OWNS must not also be the NP host — the progress updater
rebuilds a host from its CACHED send-time own embeds every 3s, which would undo the
other writer's frames. `play_placement.slow_resolve_notice` is not an edit loop at
all; it bypasses because a message we DELETE must not be the host, or the retraction
drags the live bar onto a message that is about to vanish. All five reply through
`channel.send`, not `MusicContext.send`, so they carry no NP block AND do not retire
the current host, which stays above them until the next ordinary `ctx.send` adopts a
new one — except the countdown, which re-hosts the block itself
(`repin_now_playing()`) on the one path where a song is still live when it ends. See
`docs/ARCHITECTURE.md#now-playing-host-invariants`, which lists them.
Bypassing `MusicContext.send` also bypasses debug-mode decoration, so the cog hands
the four dashboard sends a pre-rendered `debug_suffix` instead — computed ONCE per
invocation and held constant, because the driver only edits when the render changes
and a per-tick-varying footer would edit the board until its deadline (which is why
that suffix omits elapsed-ms). The countdown card holds its SPAN constant across
frames for the same reason, decorating through `_decorate` in recovery.py.
Mechanism: `MusicContext.send` (main.py) asks the guild's player for `np_embed_block()`
and **prepends it to every command response in the player's home channel** (≤ Discord's
10-embed cap; worst case here is 6, a three-card block above a collection card and its
unavailable-songs and short-walk notices), then `_adopt_np_host_if_current` makes that message
the new host and retires the previous one (dedicated NP message → deleted; command
response → strip-edited back to its own embeds). Attaching at send time makes response +
block one atomic message, so the bar is never momentarily buried. Song end: host is
released, one final edit completes the bar — only if the song truly reached its end
(`_reached_end`, 5s margin); skipped/interjected songs finalize at their true position;
a stream that never produced audio gets its block retired instead (a completed bar would
be a false record). Pause updates are debounced 0.5s.
An interjected fragment's frozen bar is the one case release-don't-retire leaves behind,
and a stack leaves one per interjection — so its resume tail carries a pointer to that
card (`np_message_id`/`np_channel_id`/`np_dedicated` on the wire, plus a runtime-only
`np_host_ref`) and disposes of it when the tail starts, **after** its own card is up.
Never a re-adopt (`_adopt_np_host` refuses older ids by design — the bar belongs at the
channel bottom); the channel id comes from `message.channel.id`, never the persisted
home channel; and capture is late-bound to the fragment's iteration end, because an id
read inside `interject()` can name a message the confirmation's own adopt already
retired. Only the runtime ref can strip-edit a response host, so the by-id path (post-
restart) is gated to dedicated cards — deleting a response would destroy a user's reply.

## Concurrency primitives

| Primitive | Protects |
|---|---|
| `GuildQueue._mutex` | the deque and its Redis mirror during bulk mutations; dequeue commits |
| `GuildQueue._wake` (Event) | the pending-item signal a parked `get()` waits on; set iff `_cursor < len(_items)`, and `_sync_wake()` is its ONLY writer — a stale set turns the wait loop into a loop with no suspension point and stops the event loop (measured at 2,000,001 iterations with 0 other loop ticks) |
| `_GuildPlays.resolves` (semaphore, src/play_placement.py) | how many of a guild's admitted `-play`s may hold one of the shared, process-wide yt-dlp pool's workers to RESOLVE (`PLAY_RESOLVE_CONCURRENCY`, default 2 against 4 workers). Admission is a memory bound; this is the pool bound on the resolve, and without it one guild's paste burst delays every other guild's extractions — including the playback loop's own in-band ones. Taken **inside `_extract_once`**, around the job it starts: a source- or playlist-cache hit and a Spotify playlist hold no worker, and a caller joining an in-flight job holds none either. Threaded down as `pool_slot` because `src/youtube.py` knows nothing about guilds; a resolve reached outside a command passes None. `docs/ARCHITECTURE.md#where-the-resolve-bound-is-taken` |
| `_GuildPlays.lock` (`PlayRegistry`, src/play_placement.py) | the insert alone — one Redis round trip, bounded by `PLACE_TIMEOUT_SECS` (7s, deliberately outliving the start write's own 5s hold on the queue mutex). `-play` resolves with no lock and enters `PlayRegistry.place()` for the put, where four checks replace the re-read a serialized body relied on: the player is not `retired` (`-stop`/kick/watchdog since dispatch), `queue.generation` did not move (`-clear`), no command stamped it (`dropped_by` — a `-stop` landing before the join has no player to retire and no queue to bump, so the stamp has to invalidate on its own), the author is still in voice. Under the hold: the put — ONE put per placement, so a `-remove` waiting on the queue mutex takes a head and its `follow_on` together — and the `queue_position` on it. A request is `placed` once the checks pass, BEFORE the put, so a command arriving during it neither stamps nor reports it; a put that raises clears the flag. The playlist embeds and the front-insert notices are built BEFORE the lock; the tail confirmation is built AFTER the put and off the lock, so the slot it names is the slot the song took. `_GuildPlays.join` is the cold-start singleflight: one `-join` per guild, awaited through `asyncio.shield` by every request that found no voice client. `docs/ARCHITECTURE.md#play-placement` |
| `_CLAIMED_CHANNELS` (src/util.py, process-wide) | exclusive use of a CHANNEL for one KIND of transient message — the queue-progress card and the slow-resolve notice. Not refcounted, unlike typing: the two are different kinds and may coexist, but sixteen cards may not. `PLAY_INFLIGHT_MAX` is 16 and requests resolve concurrently, so a pasted burst would otherwise put sixteen send/edit/delete cycles on one channel's rate-limit bucket, and because discord.py sleeps that bucket internally the throttle lands on the confirmations the user asked for. `PLAY_RESOLVE_CONCURRENCY` cannot stand in: it is taken inside the extraction, below where either message is entered, so requests 3..16 park there and are guaranteed to cross the display threshold. Claimed at the SEND, never at entry — a request that settles inside its own delay never sends and must not hold the slot a slow sibling needs. A loser asks again until it settles, so each request still gets its message in turn. A conftest autouse fixture asserts every claim was released |
| `_TYPING_HOLDS` / `_TYPING_TASKS` (src/util.py, process-wide) | one typing keepalive per CHANNEL, refcounted across the concurrent commands sharing it — per channel, not per guild, because that is what Discord's indicator is scoped to |
| `_playback_gate` (+ holds) | loop consuming the queue before a real voice connection / while `-play` resolves or `-resume` rejoins |
| `play_next` (Event) | song-end handoff from the audio thread |
| `_np_edit_lock` | concurrent NP message edits |
| claim-then-null on `_prefetch_task` | exactly-one-consumer of a prefetch result (loop vs interject vs `-replay`). Every write of a new task goes through `ensure_prefetch()`, which never starts one over a task already in the slot: loop() settles only the task it reads, so a second one's claim drifts `_cursor` for good |

The dequeue commit and the start transaction's server-side LPOP share ONE mutex hold,
via `GuildQueue.commit_dequeue()` — the async context manager the playback loop wraps
around `vc.play()` and the store dispatch. This closes the race guild_queue.py used to
carry as an accepted ISSUE: with the lock released between them, a `put_front` scheduled
in that tick read a cursor of 0, LPUSHed ahead of the entry the pending LPOP was about to
retire, and the LPOP ate the new song. Cost is one Redis round trip under the mutex per
song start (p50 ~2.4ms, p99 ~5.4ms, measured against `redis:7-alpine` through Docker
Desktop's published port), **bounded by `_START_WRITE_TIMEOUT` (5s)** — the pool sets no
`socket_timeout`, so an unbounded write parks `-play`/`-clear`/`-shuffle`/`-remove` for
that guild for as long as Redis stalls, measured past 20s against one that accepts and
then stops answering. **It is the only write under the hold.** A start transaction that
does not land — timed out, swallowed by `@_guild_op`, or never dispatched because
`vc.play()` raised after the settle — leaves the list one entry ahead of memory, and the
loop reports that through `GuildQueue.note_mirror_write()` rather than repairing it in
place: a repair under the same mutex through the same stalled pool would park the guild
exactly as the bound exists to prevent. While `mirror_dirty` is set, the next song start
REPLACES the list (`rebuild_queue_and_start_song`: DEL + RPUSH + the state HSETs in one
MULTI) instead of LPOPing it, and any `-clear`/`-shuffle`/`-remove` rebuild clears the
flag in passing; the LREM shortcut is refused over a stale list. A crash inside the
window restores the song from its stale entry and replays it — the cost is a duplicate
play, never a lost one. The body of that `async with` must stay short and must never
touch Discord; a caller with no Redis write to make passes an empty body.

`mirror_dirty` has a second source and a second repair point. `put`/`put_front`/`clear`
mutate the deque and then await the mirror, and the place lock's bound
(`PLACE_TIMEOUT_SECS`) against a Redis that accepts and then stalls cancels between the
two, leaving the list short of what memory holds; `GuildQueue._mirror_write` records that
as the same flag. While it is set, the next enqueue REBUILDS the list instead of appending
to it — an RPUSH onto a list of the wrong shape preserves the difference — so whichever
comes first, the next enqueue or the next song start, is the repair, and only a replace
that landed clears the flag. The two bounds stack: the placement lock's insert parks on
the queue mutex the loop holds for up to `_START_WRITE_TIMEOUT`, so a stalled start write
can spend the whole placement budget before the insert begins.

## Recipes

**The listing row**: every LISTING of queue items goes through `src/queue_rows.py` —
`queue_embed` and `queued_rows` (the queued album/playlist cards). The single-entry "Up
next" and "Queued song" cards are a separate renderer reading the same fields, so a
change to the row format is not automatically a change to them. A new surface that lists
queue items calls `queue_rows`, never its own format; an unresolved search renders from
`YTSource`'s display fields, so a new kind of lazy entry sets those rather than teaching
the formatter a new type. See `docs/ARCHITECTURE.md#queue-rows`.

**Add a queue-entry field**: `QueueEntryField` constant → `SongQueueEntry` field with
default → `from_queue_object`/`from_song`/`from_crashed_state` as applicable →
`to_redis` table → `parse_queue_entry` with `.get(..., default)` (old wire entries must
parse) → `QueueObject` + `GuildQueue._rehydrate` → **`YTDL.__init__`'s keyword, its
instance assignment, and the `cls(...)` call in `yt_stream` in `src/youtube.py`** —
miss these three and the field is
silently dropped the moment the queue object becomes a playing song, which is where every
read of it happens → then **BOTH places a playing song is turned back into a
QueueObject**: `MusicPlayer._requeued_form` (the rebuild `_neutralize_prefetch` and
the stream retry share — carry it) and `MusicPlayer.interject()`'s resume tail (a
different entry by construction — decide, don't copy: it resets `stream_attempts` and
`failed_format_ids`, which `_requeued_form` inherits, and both are runtime-only).
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

**Add a SEARCH-entry field** (a field on an unresolved `YTSource`, e.g. a listing's display
fields) — a different checklist, and the step that differs is the one upgrades depend on:
`YTSource` field with an `Optional` default → `SearchQueueEntry` field with the same
default → `SearchQueueEntry.from_ytsource` → `to_redis`, written **only when the value is
not None**, never as a flat table entry → `parse_queue_entry` with `.get` →
`GuildQueue._rehydrate` → a golden-bytes test beside `_GOLDEN_YTSOURCE`. The when-known
write is what keeps an entry queued by the previous build byte-identical, and LREM matches
these entries by their exact bytes: write the key unconditionally and every `-remove` and
`-clear` misses on every entry already in Redis, each one then rewriting the whole list
under the bulk mutex. Nothing here goes near `YTDL`: a search has no playing-song form
until it resolves, and resolution builds a fresh `QueueObject`.

**Touch the playback loop / queue**: re-read the module docstrings of guild_queue.py and
the loop() bookkeeping comments first; every claim, release, and Redis
LPOP is accounted for exactly once on every path (success, cleared, resolve-failure,
stream-failure, cancellation). test_musicplayer.py (13.6k lines) and test_guild_queue.py
encode these paths — run `just test tests/test_musicplayer.py tests/test_guild_queue.py`
early and often.
