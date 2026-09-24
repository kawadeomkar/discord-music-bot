# Changelog

What changed in each release, and what a deployment has to do about it. Newest first.

Versions are [SemVer](https://semver.org) and every PR bumps one, so the number says
how big the change was: a PATCH is a fix or an internal edit, a MINOR adds a feature or
a backward-compatible behaviour change, a MAJOR breaks something. The chat-command
surface is not part of that contract — adding, renaming or removing a command is a
MINOR, because nothing links against those names.

Not every version appears here. A release earns a section when a deployment could
NOTICE it: new or changed behaviour, a new setting, a migration to run, an ordering
that matters. Refactors, test work and dependency bumps ship silently, and the
per-release [GitHub releases](https://github.com/kawadeomkar/discord-music-bot/releases)
page lists every merged PR if you want the full record.

Entries are written for whoever runs the bot, not whoever wrote it: what you will see
differently, what you have to do, and whether you can roll it back.

## 2.52.0 — 2026-09-23

**The bot comes back about 1.5 seconds sooner, and container logs can no longer
fill the disk.** Nothing to configure and no data touched; rolling back is only a
redeploy. Both changes are deployment-level — no command behaves differently.

- **Restarts are ~1.5s faster.** Every start used to spend a flat two seconds
  waiting to see whether another server would arrive, whether or not one ever did:
  the library waits that long after the last server for one more, and the wait only
  ever ends by expiring. It is now half a second, which measured 1.9× more headroom
  than the whole server list needed here. A server that arrives late is still picked
  up, so nothing is lost by not waiting. If a server is ever missing from the bot's
  list at startup, raise `GUILD_READY_TIMEOUT_SECS` — it is the new setting behind
  this, accepts 0.1 to 10 seconds, and refuses startup outside that.
- **Every container caps its own log at 30 MB.** Previously they were unbounded. A
  container that is crash-looping writes to the same disk that is usually the reason
  it is crashing — Postgres in particular PANICs when it cannot write, restarts, and
  logs the cycle — so the log could fill the host it was reporting from. Each service
  now keeps at most 3 files of 10 MB. **Existing logs are not truncated:** the cap
  applies to containers created after the redeploy, so run `docker compose up -d` (or
  `just up <sha>`) to recreate them, and delete any oversized log left behind.

## 2.51.0 — 2026-09-23

**The alone-disconnect warning counts down, and says which way it went.** Nothing to
configure and no data touched; rolling back is only a redeploy. The countdown's length
is still `-settings alone-timeout` (10s by default, up to 2 minutes) — only what you
see while it runs has changed.

- **The warning is now a live card.** It used to post once, quote a fixed number of
  seconds, and then sit there — so the only way to know how long was left was to watch
  the voice channel and guess. It now re-renders about once a second, with a bar that
  drains, and it counts from *your server's* timeout rather than a hardcoded 10.
- **It closes with a final frame instead of going stale.** When someone comes back the
  card says `Someone rejoined` and playback carries on; when the timer runs out it says
  `Disconnected from voice channel` and reminds you the queue is kept for 24 hours and
  `-resume` picks it back up. Previously the last thing in the channel was a warning
  about a disconnect that may or may not have happened.
- **A rejoin now ends the countdown gracefully.** It used to cancel the countdown
  outright, which is why it could never report the outcome. Whether the bot actually
  leaves is still decided by who is in the channel when the clock runs out, not by the
  card — a rejoin the gateway never told us about still keeps the bot in place.
- **The Now Playing block goes back to the bottom of the channel.** The card sits on
  top of the block while it ticks, so when a song is still playing and someone rejoins,
  the block is re-pinned underneath it.

## 2.50.0 — 2026-09-23

**`-pause` always answers, and the paused card lives with the song it describes.**
Nothing to configure and no data touched; rolling back is only a redeploy.

- **`-pause` with nothing to pause now says so.** It used to do nothing at all — no
  reaction, no reply — so there was no way to tell "the bot ignored me" from "the bot
  is not running". Out of voice, or in the seconds between two songs, it answers `No
  songs are currently playing.` An **already-paused** song gets its own line instead,
  pointing at `-resume`: a paused song is not "nothing playing", it is loaded,
  positioned and resumable.
- **The Now Playing card says it is paused.** Title, colour and layout used to be
  identical playing or paused, and the only tell was a progress bar that had stopped
  moving — so scrolling past the confirmation, or `-now` re-pinning a fresh card, left
  a paused song reading as a playing one. The title now reads `⏸️ Paused: <song>` for
  as long as the song is held.
- **The paused card joins the Now Playing block, and leaves when the pause does.** It
  used to be a one-shot reply that landed below the queue head and then stayed in the
  channel forever, describing a pause that had long since ended. It is now the block's
  second card: directly under the song it describes, and gone the moment `-resume`
  rebuilds the block — nothing deletes it, the rebuild simply does not include it.
- **It names who paused and when.** The clock is a Discord timestamp, so each viewer
  reads it in their own zone and the relative half ("3 minutes ago") keeps counting on
  a card nothing will edit again. A song the bot parked paused on its own — a
  `-play --now` tail coming back, or a crash-recovered song — carries no byline rather
  than crediting whoever happened to queue it.
- **`-pause` re-pins the block** instead of posting its own embed, so the card lands at
  the bottom of the channel where the bar belongs.

## 2.49.0 — 2026-09-23

**Songs that used to fail quietly now play.** Nothing to configure and no data touched;
rolling back is only a redeploy.

- **A song with a start offset actually starts.** The seek went *after* ffmpeg's input,
  so it downloaded and decoded from `0:00` and threw the audio away until it reached the
  offset — and because that open carries no Range header, YouTube served it at a trickle
  and then stopped. A link at `48:10` of a 58-minute song read 408 KB, produced no audio
  and never started, **with nothing in the logs, because nothing had failed**. The seek
  is now a range request straight to the offset. This is reached by a `?t=` link,
  `-play --timestamp`, the song `-play --now` interrupted coming back, and a restart
  resuming into a long song.
- **A stream that will not open is retried rather than abandoned.** One dead URL used to
  be one red embed. A song now gets three attempts: the second with a fresh URL, which
  cures a link revoked between the check and playback; the third on a different audio
  format. Two dead songs in a row stop the retries until one plays.
- **Less CPU per song.** Most of what YouTube serves is already in the format Discord
  wants, and it was being decoded and re-encoded to the same thing. It is now passed
  through untouched where that is safe.
- **The lookup workers stop being killed.** Each one grew about 5 MB per lookup and never
  gave it back, so a long-running bot eventually lost a worker to the kernel and had to
  rebuild the pool. Workers are now retired and replaced before they get that large.
- **A search picks a result it can actually play**, instead of accepting one with no
  playable format and failing later, where the error looks unrelated to the search.

**For operators:** `just ytdl-formats <url-or-search>` prints the format yt-dlp selects,
the ladder it chose from, and the fallback ladder the retry would walk. It calls the
bot's own picker rather than a copy, so what it prints is what the bot does. Run it after
every yt-dlp bump: the format choices in the code are empirical, and both YouTube and
yt-dlp move under them.

## 2.48.0 — 2026-09-22

**`-play` takes a start offset, and three things it already did read differently.**
Nothing to configure and no data touched; rolling back is only a redeploy.

- **New:** `-play --timestamp 1:32 <song>` (or `-ts`) starts any song partway in —
  a search, a Spotify track and a SoundCloud link included, none of which could carry
  a start offset before. It takes `1:32`, `2:04:30`, `90`, `90s` or `2h30m15s`, goes
  among the leading options in any order, and works on `-playnow` and `-playnext` too.
  It beats a `?t=` on the same link. A time at or past the end of the song queues
  nothing and says so; a link that queues a playlist is refused, unless it names a
  `v=` video, where it starts that track exactly as a `&t=` on it already did.
- **Queue ETAs shrink for a song with a start offset.** A `?t=` song used to be billed
  its full length, so every song behind it was estimated that much too late. It is now
  billed what actually plays. Nothing about playback changes — only the estimates, and
  only for a queue holding such a song.
- **A start offset renders as a clock.** `starts at 1:30` in the queue and the Now
  Playing card, and `Starting song at 1:30` when it begins, where all three read
  `90s` / `90 seconds` before.
- **For operators:** a `-play` span now carries `play.start_offset` when the flag set
  one (absent otherwise, so a filter for offset plays does not match every `-play`), and
  `play.refused` naming why a request queued nothing. A refusal sends an embed and logs
  nothing, so without that attribute it left no record it had run.
- **A repeated or conflicting option on `-play` is now answered, not searched for.**
  `-play --now --next <song>` and `-play --now --now <song>` used to search YouTube for
  the leftover flag as part of the text; they now reply and queue nothing. An option
  after the song is unaffected — `-play <song> --now` is still a search for all of it.

## 2.45.0 — 2026-09-22

**Some links now play that failed, and a few route differently.** Nothing to configure and
no data touched; rolling back is only a redeploy.

- **Now plays:** a link in `<…>`, `||…||`, a masked `[text](link)`, a code span or
  parentheses, or with a full stop or comma on the end. Spotify URIs (`spotify:track:…`),
  localized and embed Spotify links (`/intl-de/track/…`, `/embed/track/…`),
  `play.spotify.com`, and the old `/user/<name>/playlist/…` path. A YouTube share link
  (`attribution_link?u=…`) plays what it points at.
- **Routes like the lowercase link:** a link with any uppercase in its host
  (`WWW.YOUTUBE.COM/…`, `YOUTU.BE/…`). A `watch?v=…&list=…` spelled that way now queues
  the playlist, like its lowercase twin, instead of one song.
- **Starts at its timestamp:** `youtu.be/<video>?list=…&t=30`, when the video is the
  playlist's first queued track.
- **Refused with a message:** Spotify artist, show and episode links, and any other
  Spotify link that names nothing playable. These used to fail with an internal error.
- **Now a search:** a token whose link does not start it, such as `ftp://…`,
  `//host/…`, `user@host/…` or `listen:https://…`. yt-dlp failed all of these before.

In the archive, `query_source` moves for two of these: `YOUTU.BE/…` records
`youtube.com`, and `play.spotify.com` records `spotify.com`. A scheme-less link
(`youtu.be/…`) gets its own source-cache entry on its first play after the upgrade,
since its key no longer folds case.

A single long word in `-play` used to stall audio in every guild for up to a few hundred
milliseconds; the link test now runs in linear time.

## 2.44.0 — 2026-09-21

**A queued album or playlist is listed the way `-queue` lists songs.** The "Queued album"
and "Queued playlist" replies used to print their own numbered list of search strings
("DNA. Kendrick Lamar"). They now show `-queue`'s row for each track: its queue position,
its name as a link, its length and when it is expected to start. `-queue` itself shows the
same for album and Spotify-playlist tracks, which used to read `resolving...` with no
length until just before they played; the queue's total now counts them.

Two things to know. A Spotify track has no YouTube page until it is about to play, so its
title links to the track on Spotify until then. And its length is Spotify's, not the
YouTube match's, so start times behind it carry a `~`.

Nothing to configure. Each album and Spotify playlist is read from Spotify once more after
the upgrade, because the cached copies do not hold the new per-track details. Tracks
already in a queue when you upgrade keep the old `resolving...` row until they play.
Rolling back is only a redeploy.

**It costs Redis memory.** Holding a name, artists, length and link for every queued track
makes a saved queue entry about 400 bytes instead of about 270, and roughly triples the
cached copy of a collection. A 10,000-track playlist that is both queued and cached now
holds around 6 MB rather than around 3 MB. That matters only if you run enormous
collections: the bundled Redis is capped at 256 MB, and when it fills it evicts the keys
that carry a TTL — which includes saved queues, and an evicted queue is not restored after
a restart. `-queue`, `-remove` and `-clear` are unaffected either way.

## 2.43.0 — 2026-09-21

**Spotify album links now queue.** `-play https://open.spotify.com/album/…` takes the
whole album, the way a playlist link already did: under `--now` and `--next` too, with
the same live card while a long one is read, and one `-remove <the link>` takes it back
out. The confirmation names the album, its artists and its cover. An album is read once
and kept for 24 hours; Spotify is asked again after that.

Three smaller changes to what a pasted Spotify link does:

- **A share link from a non-English client works.** Spotify's own share sheet produces
  `open.spotify.com/intl-de/album/…`; the locale segment used to make the link fail.
- **A link the bot cannot queue says so.** An `/artist/` or `/show/` link, or one with no
  id, used to answer with a Python exception. It now names the three kinds it takes.
- **A link pasted as `<link>`** (how Discord sends one whose preview you suppressed) is
  read as the link inside.

If Spotify stops sending an album or playlist before its own count of it, the
confirmation is now preceded by a line saying some songs may be missing, instead of
reporting the partial count as the whole. Nothing to configure, and no data moves.
Rolling back is only a redeploy; the album cache entries an older build never reads
expire on their own.

## 2.40.0 — 2026-09-19

**Three settings now refuse startup when they are out of range**, the way the other
tunables already did. Each used to be read with no range check, so an out-of-range value
reached the loop that uses it:

| Variable | Accepted | What an out-of-range value used to do |
|---|---|---|
| `NOW_PLAYING_UPDATE_INTERVAL_SECS` | 1.0 or more | `0` edited the Now Playing card back to back, spending the rate limit the channel's other messages share |
| `STREAM_PROBE_TIMEOUT_SECS` | 0.1 or more | `0` or a negative value removed the timeout entirely, so a stream host that never answered held a song's start |
| `YTDLP_POOL_WORKERS` | 1 or more | `0` failed every lookup, each time the pool tried to start |

A value outside its range, `nan` or `inf` now stops the bot at startup instead of
starting. Something that is not a number already stopped it; the error now names the
variable. If yours start as before, nothing changes: every default is inside its range.
Rolling back is only a redeploy.

## 2.39.0 — 2026-09-18

**`LIVENESS_INTERVAL_SECS` is now bounded to 1-60 seconds.** It was read with no range
check, so a value above the container HEALTHCHECK's 90s staleness window made a healthy
bot report unhealthy between touches, and `0` turned the touch loop into a spin. Either
now stops the bot at startup, naming the variable, instead of starting. If yours is unset
or inside that range, nothing changes. Rolling back is only a redeploy.
## 2.37.0 — 2026-09-13

**A Spotify playlist over 100 tracks now queues in full.** Only the first page was ever
read — the `next` cursor was never followed — so a 300-track playlist queued 100 and
reported success, with nothing to say the other 200 had been dropped. If you have been
working around that by splitting playlists up, stop: `-play <playlist link>` takes the
whole thing, up to Spotify's own 10,000-item ceiling.

Two consequences worth knowing before you paste a big one. The lookup takes longer,
because it is one HTTPS round trip per 100 tracks: a 1,000-track playlist is ten
sequential requests, typically a second or two, against the ~150 ms a truncated one used
to take. And it is bounded twice — 20 seconds for any single request, 120 seconds for the
whole walk — past which the command fails and queues **nothing**, rather than queueing a
part of a playlist you would have to work out the shape of yourself. The two are reported
differently, because only one is worth retrying: a stalled request says so and invites a
retry, while a playlist that used the whole budget tells you to queue it in parts. Items a playlist
can hold with no title of their own — removed or region-dropped tracks, and podcast
episodes that carry none — are skipped, so "Queued N songs" can be lower than the
playlist's own item count. A local file is kept: its name is exactly what a YouTube
search wants.

Already-cached playlists are not stale for an hour after the upgrade: the cache key moved,
so the first `-play` after deploy re-reads from Spotify. Nothing to configure, no data
touched, and rolling back is only a redeploy.

## 2.35.1 — 2026-09-08

**`-play <search>` answers in about half a second instead of two and a half.** A search —
typed words, or a Spotify track link, which resolves to one — is now answered from the
search response itself: title, length, uploader and artwork, with no stream URL. The
stream is extracted by the background prefetch that already ran for every queued song, so
nothing new happens on the network per play; measured against the same queries, the
**lookup** went from ~2.5s to ~0.6s. That is the lookup, not the whole reply: a `-play`
that has to join a voice channel first still waits for the handshake, and the card lands
after it. Pasted links, `-play --now`, playlists, and a `-play` that finds the bot
disconnected are unchanged — the last of those still resolves in full, because its song
plays immediately and a failure there would leave the bot sitting in an empty channel.

**One behaviour genuinely changes.** A search no longer selects a format at enqueue, so
a video that cannot actually be played — private, geo-blocked, age-gated, members-only,
or with no usable audio — is no longer caught by the command. It queues successfully and
fails when its turn comes, with a red "Failed to load the next song, skipping." card and
a gap in playback, the way playlist tracks always have; the queue carries on to the next
song. A bad *link* still fails the command with nothing queued.

**Several `-play`s sent at an idle bot play in reverse order.** Each one finds no voice
client, so each takes the front of the queue: paste three and they play third, second,
first. Send them one at a time, or use `-play --next` once the first is playing.

Nothing to configure, no data touched, and no migration: rolling back is only a redeploy.

## 2.35.0 — 2026-09-02

**`-play` takes a `--now` flag**, which interrupts what is playing:

```
-p --now never gonna give you up
```

The behaviour is the one `-playnow` always had — the interrupted song returns from the
exact position it left off at, and interjections still stack. `-playnow` and its `pn`
alias stay exactly as they were; the flag is a second spelling, not a replacement. It
must be the **first** word, so a `--now` inside a search term stays part of the search.

Two things changed alongside it:

- An interjection is no longer exempt from the "bot is already being used in channel X"
  rule. Queueing into a session running elsewhere still works; **stopping** what that
  channel is hearing now requires being in it.
- `-play` requests sent while another is still being looked up are looked up alongside
  it and land as each one is ready, so a `--now` sent behind a long playlist interrupts as
  soon as its own song resolves. `-clear`, `-stop` and `-remove` drop requests still being
  looked up and say so. Two ceilings apply, and they bound different things: a server may
  have 16 requests waiting at once (`PLAY_INFLIGHT_MAX`) and past that one is declined,
  while only 2 of them hold a yt-dlp worker (`PLAY_RESOLVE_CONCURRENCY`) — the rest wait
  their turn rather than being refused, so one server's paste burst cannot delay the
  extractions another server's playback is waiting on. That wait is bounded
  (`PLAY_RESOLVE_WAIT_SECS`, 2 minutes): a request that never gets a slot is declined
  having queued nothing, so sending it again cannot double-queue the song. A lookup
  still running after `PLAY_SLOW_NOTICE_SECS` says so, and takes the message back once
  the song is queued.

**`-play` takes a `--next` flag**, which queues a song at the front without
interrupting what is playing:

```
-p --next never gonna give you up
```

Like `--now`, it must be the **first** word, and it is subject to the "bot is already
being used in channel X" rule — cutting to the front of a queue is queue control, the
same as `-skip` or `-shuffle`. It also gained a command spelling, `-playnext` (`pnx`),
so both placements are reachable the same two ways.

**A playlist is no longer collapsed to its first track.** `-p --now <playlist>` used to
play track 1 and discard the rest; it now plays track 1 immediately and queues the whole
playlist behind it. The song it interrupted therefore does not return until the last
track — on a long playlist, in practice, never. If that was not what you wanted,
`-remove <the same link>` takes the queued tracks back out in one command; the one already
playing is not queued any more, so it needs `-skip`.

The same is true of plain `-play <playlist>` while a song is **paused** — that has always
interrupted the paused song, and now brings the whole playlist with it rather than one
track.

## 2.5.0 — 2026-08-02

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
[Backfilling history that predates the archive](README.md#backfilling-history-that-predates-the-archive).
The ordering is load-bearing and the tool cannot detect that you got it wrong — a list
already trimmed to 50 looks exactly like a small one.
