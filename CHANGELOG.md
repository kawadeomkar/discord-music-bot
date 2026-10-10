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

## 2.58.1 — 2026-10-03

**Spotify links resolve more reliably by recording.** A Spotify album longer than 50
tracks no longer loses its tracks' ISRCs to an oversized request, a malformed ISRC is no
longer searched verbatim, and a Spotify track whose ISRC YouTube does not index resolves
one search faster.

Nothing to do on deploy, and rolling back is safe.

## 2.58.0 — 2026-10-03

**New setting: `-settings loudness`.** Off by default, and off behaves exactly as the bot
always has. The other two even out how loud songs play:

- `peak` caps the loudest peaks, so a hot track stops jumping out of the mix. Anything
  below the ceiling is left alone.
- `normalize` brings every song to one loudness, raising quiet ones as well as lowering
  loud ones. A YouTube song uses the loudness YouTube itself reports, so it starts as fast
  as any other. Anything else — SoundCloud, a direct file — is measured once and remembered
  for a month, and a song queued behind another is measured while it waits its turn, so
  it starts without a pause too. Only a song that reaches its turn unmeasured, typically
  the first of a session, waits: `-settings bot loudness-scan-timeout` (default 8s,
  `LOUDNESS_SCAN_TIMEOUT_SECS`) bounds that wait. Past it the song plays at its own level
  and the measurement finishes in the background for its next play. A song longer than 20
  minutes — a DJ set, a full album — is measured from its first twenty. A recording with
  very sharp peaks lands short of the rest rather than being squashed into line.

Measuring costs CPU and bandwidth only in a `normalize` server, and only for songs YouTube
did not measure: at most two measurements run at once, each reads at most twenty minutes
of the song (the smallest YouTube format when one is needed), and `off` and `peak` measure
nothing. Either mode re-encodes the audio instead of copying YouTube's bitstream through
untouched, which spends one lossy generation and about 3% of one CPU core per playing server;
`off` remains the only bit-exact setting. If a future
yt-dlp update stops reporting YouTube's figure, nothing breaks — YouTube songs are measured
before their first play again — and `just ytdl-formats <url>` prints a `LOUDNESS` line that
says which is happening.

The README has a troubleshooting section for a listener who hears the bot muffled when
nobody else does.

Nothing to do on deploy: no guild has the loudness setting until someone sets it. Rolling
back is safe; an older build ignores the setting and the cached measurements.

## 2.57.0 — 2026-10-03

**Songs the bot re-encodes sound right.** Anything not bit-copied from YouTube — every
SoundCloud track, every song played at a volume other than 100 %, every video whose only
audio is AAC — was leaving the encoder in Opus's speech mode. It now stays in the music
mode, which is also cheaper to encode. Budget for roughly a third more voice traffic per
such song (119 to 153 kbps measured on a real YouTube stream), though how much depends
entirely on the music.

**A lossless link encodes at the voice channel's bitrate.** A direct FLAC, WAV or AIFF URL
now uses the channel's own ceiling, up to 384k, instead of 128k: +2 dB measured on a 256k
channel. Apple Lossless is not included — it shares `.m4a` with AAC. Everything else stays
at 128k, because a lossy source gains little or nothing from more. Those songs send up to
3x the voice traffic they did; like `-volume`, it applies from the song after next.

**A song no longer ends halfway when the connection hiccups.** A connection that died
mid-play and was answered with a server error used to stop the song there, with nothing
retried and nothing logged. It is now retried and resumes exactly where it left off. A
failure that cannot be recovered ends the song about four seconds later than before, on
the image's ffmpeg 7.1. A `just run` on an older ffmpeg, such as Ubuntu 24.04's 6.1,
recovers the same songs but takes about 12 seconds to give up on one it cannot.

**Videos with dubbed audio tracks are copied, not re-encoded.** They were decoded and
re-encoded like a non-Opus source, with nothing in the logs.

**A new WARNING if YouTube serves an Opus format the bot does not copy.** It reads
`YouTube served Opus format <id>, which is not in the passthrough allowlist`, fires at most
once per format per restart, and means those songs are being re-encoded — worth reporting.

Nothing to do on deploy, and rolling back is safe: nothing here is stored.

## 2.56.0 — 2026-10-02

**Searches that YouTube refused to answer now play.** Some searches came back as
"Couldn't find anything playable for that." for songs that exist and play fine. The cause
was not the bot and not yt-dlp: when the top of a result set holds age-restricted content,
YouTube serves a signed-out client an empty page — a "Confirm your age" card where the
videos would be — and reports zero results. `xvi akiaura` is one; `akiaura xvi`, the same
two words reordered, returns tens of thousands. Nothing in what yt-dlp hands back
distinguishes that from a query with no matches, which is why the bot believed it.

Such a search now gets one more try through YouTube Music, which answers it normally, and
plays the first track it names.

- **Nothing to do.** No setting, no key, no account, no new service. It uses the
  music.youtube.com search yt-dlp already supports.
- **Only a search that found NOTHING reaches it**, so no search that works today changes
  its answer. A link never reaches it at all.
- **It costs about 1.5 seconds** on top of the failed search, about 2 seconds in all,
  and only on a request that previously failed outright. The answer is remembered
  against the words you typed for a day, so a repeat of the same wording starts as fast
  as any other cached search.
- **A search that genuinely has no matches still says so**, with the same wording as
  before, about half a second later than it used to — except a search queued while the
  bot is already in voice, which skips a search it used to send twice and answers about
  as fast as it always did. YouTube Music always offers *something*, so a track is only accepted when
  its title carries at least half of the meaningful words you asked for, is not a
  karaoke, nightcore, live or similar version you did not ask for, and — for a Spotify
  link — runs within ten seconds of the length Spotify gives. A typo gets the same
  "couldn't find anything" it always got, not a stranger's song.
- **Roll back freely**, to any 2.55.x. The one thing written to Redis is an ordinary
  day-long search cache entry, which any 2.55.x reads as it reads its own.

## 2.55.3 — 2026-10-02

**Two ways a half-connected bot used to go unnoticed, both closed.** discord.py registers
a voice client before the handshake lands, so an abandoned one stays registered and
answers as if the bot had joined. Nothing checked.

- **The bot no longer rejoins a channel after you stopped it.** `-stop`, the alone timer,
  an eject and `-resume` all tore the session down without cancelling a join that was
  still running, so that join could land afterwards and put the bot back in the channel
  — silent, with no queue and nothing playing, after the saved connection had already
  been cleared. Worse, when that stale join finally timed out it dropped whatever voice
  client the guild held by then, which could be a healthy one a later `-play` had just
  made. That one presented as the bot being in the channel while every command insisted
  it was not. A stop that lands in the instant the bot is asking Discord to join now
  tells Discord it left as well, and a stop never waits more than 3 seconds on Discord
  confirming it.
- **`-join` repairs a half-connected bot instead of half-working.** Handed an abandoned
  client it skipped its own connect, reported nothing, saved the channel and started
  playback — and every song then failed one at a time, emptying the queue in memory while
  Redis still held it. It now drops the abandoned client and connects properly. If that
  connection does not complete either, it says "Couldn't finish connecting to your voice
  channel" and asks for another `-join`, saves nothing, and leaves the queue alone. A
  `-play` that joins for you reports the same way it always did.
- **`-join` brings the bot to you when nobody is listening where it is.** It used to
  answer "Bot is already being used in channel X" even from an empty channel — the
  bot sat alone until the alone timer disconnected it. It now moves, and the song
  playing carries on in your channel. A channel with someone still in it keeps the bot,
  as before.
- **Muting the bot no longer keeps it in an empty channel for good.** Server-muting or
  deafening the bot while it counted down to leaving an empty channel stopped the
  countdown, so it never left. The countdown now runs on.
- **Nothing to do**, and no state to clean up: both windows were in memory only. If you
  have a server where the bot shows as connected but answers "I'm not in a voice
  channel", `-join` once after deploying this.
- **Roll back freely**, to any earlier 2.55.x.

## 2.55.2 — 2026-10-02

**A `-play` whose song lookup fails no longer breaks the next one.** When the lookup
raised on a bot that was not yet in a voice channel — nothing found for the search, a
link it could not read, a lookup that ran out of time — the bot abandoned the
half-finished voice handshake without telling Discord it was leaving. Discord went on
listing it in the channel, so the NEXT `-play` in that server could not join at all: it
asked for the channel it was already in, got no answer, and sat out a 10-second
handshake timeout plus another 10 waiting for a confirmation that could no longer
arrive. Twenty seconds, then a red **Command failed** embed naming a bare
`TimeoutError` — for a request that was perfectly fine.

- **Nothing to do.** No setting, no migration, no state to clear.
- **The old symptom read as contagion**: one bad song appeared to poison the song after
  it, and only that one. The failed join's own teardown forced the clear on its way
  out, so a third `-play` connected normally.
- **Roll back freely**, to 2.55.1 or earlier. Nothing here changes what is written to Redis
  or Postgres.

## 2.55.1 — 2026-10-06

**A security update to one of the bot's libraries.** `multidict`, which the bot's HTTP
client (aiohttp) uses for headers, moves from 6.7.1 to 6.9.1 for CVE-2026-104874. No
behaviour changes.

- **Nothing to do** beyond deploying the new image.
- **Roll back freely**: only the library version changes.

## 2.55.0 — 2026-10-02

**Spotify links now play the album recording, not the music video.** A Spotify track
carries the recording's ISRC, and YouTube indexes that on the label's own upload, so the
bot searches it before falling back to the title. "Shape of You" used to queue the
4:24 official video — dialogue, then the song — and now queues the 3:54 album master
Spotify names. Album tracks get the same treatment through one extra Spotify request per
50 tracks; a track with no ISRC, or whose ISRC YouTube does not know, resolves by title
exactly as before.

Two things to know. Spotify links resolved in the last 24 hours keep the video they
already resolved to until that entry ages out. And the Spotify caches are re-keyed by
this release: the first play of a playlist or album after deploying walks it again.
Rolling back is safe — the old build ignores the new keys and re-walks once itself.

## 2.54.1 — 2026-09-27

**The reader for the queue entry builds before 2.54.0 wrote is gone.** Nothing changes in
chat, and nothing this build writes to Redis changes. What goes with the reader is the
old-shape tally 2.54.0 asked you to watch: the restart line reports the restored count
and, when a queue held entries it could not read, how many.

- **Deploy this only once that tally has read zero across restarts**, which is what
  2.54.0 put it there for. If a queue does still hold an old-shape track, that track is
  dropped and does not play. Nothing else in that queue is affected: the list is
  rewritten once, so no other song shifts position or replays, and the restart logs one
  warning per server naming how many entries it dropped and the shape it met
  (`entry type 'ytsource'`).
- **Roll back freely, to 2.53.7 or newer.** No entry this build writes is new to the
  build before it. A track already dropped does not come back, though — the rollback
  restores the reader, not the queue.

## 2.54.0 — 2026-09-26

**A collection track that has not resolved yet is written to Redis as an ordinary queued
song.** Nothing changes in chat: the same tracks queue, show and play the same way. What
moves is the shape of the saved entry, so this is the release the 2.53.2 note pointed
forward at.

- **Do not roll back past 2.53.7 once this build has run.** 2.53.7 is the first build
  that can read the new entry; an older one restores such a track as a song it believes
  is playable, pointing at the Spotify track page — at nothing at all when the collection
  gave no link for it — and neither plays. Each one posts an error as its turn comes and
  the queue moves on to the next. Rolling back TO 2.53.7 is safe.
- **The first `-remove` touching a track an earlier build queued rewrites the whole queue
  list once.** Removal matches an entry by its exact saved bytes, and those tracks were
  saved in the old shape, so the first attempt misses and the list is rebuilt instead.
  That rebuild logs one `queue mirror diverged from memory` warning for the guild, which
  is expected here and not a sign of damage. Nothing is lost and nothing is duplicated;
  afterwards every entry is in the new shape and removals are one-shot again. `-clear`
  never has to match bytes — it deletes the list — so it costs nothing extra.
- **Each restart reports, per guild, how many restored entries were still in the old
  shape** — on the line that gives the restored count, and as `restore.old_shape_entries`
  on that guild's restore span. A guild whose saved queue was empty gets the count with
  no tally after it, so every `N of M` in the log is a list that was really read. A zero
  covers only the guilds this start actually put a player back into. A server the bot had
  been told to leave, one whose saved voice or text channel is gone, and one whose
  reconnect failed are all skipped — the last two say so in their own warning — and each
  keeps its saved queue for the 24 hours that list lives, counted by nobody. So read the
  zeros alongside the `Recovery skipped` and `Could not rejoin voice` warnings, and give
  the last of those a full day before you believe them. To settle it outright rather than
  infer it, scan the `guild:*:queue` lists for entries whose `"type"` is `"ytsource"`.
- **A queued collection costs a little more Redis**: ~540 bytes per unresolved track
  against ~400 before, so a 10,000-track playlist holds ~5 MB of queue mirror rather than
  ~4 MB, against the 256 MB the bundled Redis is given. Nothing to do; noted so the number
  is not a surprise.

## 2.53.7 — 2026-09-26

**This build can read a queue entry the next one writes.** Nothing changes in chat, and
nothing this build writes to Redis changes. A queued song entry may now carry the search
term of a collection track that has not resolved yet; this build reads that entry back and
restores the track as still-to-resolve, while it keeps writing such tracks in the shape
earlier builds read.

- **Deploy this before the release that stops writing the old shape.** That release says
  so in its own section; with this one running underneath it, rolling it back lands on a
  build that can still play every queued track.
- **Coming back down to this build rewrites a queue list once.** Every track the newer
  build queued still plays, but this build writes those tracks in the old shape again,
  so the first `-remove` in a server holding one rewrites that server's whole queue
  list instead of deleting out of it, and logs `queue mirror diverged from memory in
  guild …` at WARNING when it does — expected here, not damage. (`-clear` deletes the
  list outright, so it never matches bytes and never pays this.) Once per server,
  nothing is lost, and there is nothing to do about it.
- **Roll back freely.** No entry this build writes is new to the build before it.

## 2.53.4 — 2026-09-26

**Crash recovery keeps the flags of the song that was actually playing.** The parked entry's
own fields now win over the stored blob whenever both describe the same play; the blob
contributes only the thumbnail and the card ids the fields never carried. Before this, a
rollback to 2.53.2 and back could bring a paused song back playing.

- **A restart that cannot place every saved entry rewrites the queue.** When a saved entry
  names a requester nobody can resolve, it is dropped with a warning and the queue in Redis
  is rebuilt to match, so the next song played is the next song shown.
- **`-replay` gets its full retry budget.** A replayed song no longer inherits the stream
  attempts the original spent.
- **A crash-recovered song shows the cover it was playing with**, not the one it was queued
  with.
- **The state hash costs a few hundred bytes more per guild that has ever played**, and the
  encoding it moves to never comes back on its own — measured, not predicted, in
  `docs/ARCHITECTURE.md#the-parked-song`. Nothing to do; noted so the number is not a surprise.

## 2.53.3 — 2026-09-26

**The playing song is parked in Redis as one queue entry.** Nothing changes in chat. The
state hash `guild:{id}:state` gains a `current_song` field holding the whole entry, written
beside the thirteen `current_song_*` fields it stands in for. Both are written for one
release, so a crash recovers the playing song on this build or on the one before it.

- **Deploy and roll back freely.** This build reads the prefixed fields when no blob is
  there, and a build before it reads the fields this one still writes. A blob is trusted
  only when it describes the same play those fields do.
- **A crash-recovered song keeps its thumbnail again**, which the thirteen fields never
  carried.
- **The thirteen fields stay for one more release.** Dropping them earlier would leave a
  rollback with no song to recover.

## 2.53.2 — 2026-09-26

**A queue holds one kind of item.** Nothing changes in chat but the one case below. A
collection's tracks used to wait as a different type from an ordinary queued song, and the
two carried the same fields side by side; now there is one, resolved or not.

- **A saved song nobody can be matched to is credited to the bot.** A restart resolves
  every saved song's requester up front now, the same way for a collection track still
  waiting to resolve as for an ordinary queued song. One whose requester has left the
  server, and who the bot has not seen anywhere else, keeps its place in the queue and is
  shown and archived under the bot itself — or under whoever ran the command that brought
  the bot back, if one did. It used to name the server owner. Nothing to do.
- **Deploying and rolling back are both just a redeploy.** A track still waiting to
  resolve is written to Redis in the shape earlier builds already read, so no build reads
  the queue any differently for this change. That shape goes in a later release, which
  will carry its own note about what a rollback past it costs.

## 2.53.0 — 2026-09-24

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

## 2.52.0 — 2026-09-23

**An enabled history archive that cannot reach Postgres now says so at startup.** Only
affects deployments running with `HISTORY_ARCHIVE_ENABLED=true`; the default (archive
off) is untouched, nothing is configured and no data moves. Rolling back is only a
redeploy.

- **A new ERROR about a minute after startup, when it applies.** If the flag is true and
  Postgres has not answered by then, the log names what is accumulating and how to
  deploy the database. Previously this combination started clean and stayed quiet: the
  bot played music, answered commands, and wrote nothing durable.
- **Why it could go unnoticed.** A bare `docker compose up` does not activate the
  `archive` profile, so no Postgres is deployed — but the connection string is handed to
  the bot either way, so its existing "the archive needs a database" check passes. The
  connection is only opened at the first song end, and a failure there is a warning
  among the playback logs. Meanwhile every play is appended to a Redis key that has no
  expiry and is exempt from eviction, so it grows until Redis runs out of memory and
  stops accepting writes — at which point the bot stops working for reasons that look
  nothing like this.
- **Nothing to do if your archive is healthy.** One INFO line says the probe got an
  answer. If you see the ERROR, `just up` deploys Postgres alongside the bot — it
  derives the profile from the flag, which a raw compose invocation cannot.
- **Nothing is written or deleted by this.** The probe runs `SELECT 1`, and the entries
  already queued in Redis drain on their own once the database is reachable.

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
