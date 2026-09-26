---
paths:
  - "src/**"
  - "tests/**"
  - "migrations/**"
---

# Repository layout — what each module holds, annotated

The map, with the one-line reason each module exists and the boundaries between
them. Read it to find where a change belongs before opening files.

```
src/
├── main.py           # entrypoint: MusicBotApp (AutoShardedBot), MusicContext, Redis pool wiring
├── archive_tier.py   # starting and stopping the history archive: the enabled arm's DSN
│                     # requirement and password advisory, the outbox consumer group, the
│                     # archive + drainer + reachability probe, the disabled arm's reporting,
│                     # and the order aclose() unwinds the three in. main.py decides WHETHER
│                     # to run it (it reads the flag first); this owns what running it means
├── musicbot.py       # MusicBot cog — command REGISTRATION and one try/except each;
│                     # per-guild player registry (mps), the discord.py hooks, crash-recovery entry
├── musicplayer.py    # MusicPlayer — per-guild playback loop, prefetch (ensure_prefetch), gate,
│                     # NP host, ETA, interject, the hooks -replay drives (hand_over_to_replay),
│                     # and still_live, the one liveness test both interrupts use
├── play_placement.py # -play's flag grammar, its voice gate, the two bounds on the resolve
│                     # (ResolveSlot's deadline on the WAIT for a slot, and slow_resolve_notice
│                     # saying so past it), and PlayRegistry: the per-guild
│                     # in-flight set and the place lock its inserts serialize on (the cog
│                     # keeps the commands; the grammar and the registry are tested in
│                     # test_play_placement.py, the placement itself in commands/test_play.py)
├── guild_queue.py    # GuildQueue — one deque + cursor, the mirror writer, bulk-mutation mutex
├── queue_rows.py     # one queued item as a row of text, and the ETA walk down a listing (pure);
│                     # -queue and the queued-collection cards list through it
├── guild_history.py  # GuildHistory — played-song history (capped Redis list + in-memory cache; writes feed the outbox while the archive is enabled, reads never touch Postgres) and its embeds; the command body is commands/history.py
├── history_archive.py# Postgres archive (asyncpg) + HistoryOutboxDrainer (outbox → play_history)
├── recovery.py       # Voice-session lifecycle: rejoin after restart (crash recovery), the alone-in-channel leave watchdog (a countdown card ticked down to a final frame saying which way it went), and the two cold-start helpers -play and -resume share
├── commands/         # ONE MODULE PER COMMAND — commands/<command>.py, each exposing
│                     # run(). The cog holds registration and one try/except; every
│                     # body is here. _common.py holds the restore guard three of the
│                     # queue commands share. A command with a domain module (history,
│                     # leaderboard, debug, ping, analytics) keeps only its entry point
│                     # here and imports the machinery
├── play_pipeline.py  # the machinery behind -play/-playnow: resolve, place, interject
├── leaderboard.py    # -leaderboard tunables, Redis result-cache codec, embed renderer (pure;
│                     # the command body is commands/leaderboard.py)
├── analytics_card.py # -analytics, everything but the command and the figure. Pure half:
│                     # --days allowlist, both cache codecs, the embed — every
│                     # human-authored string renders HERE, never in the image. IO half:
│                     # the chart pool, the PNG cache, the send
├── analytics_render.py # the six-panel figure; imports matplotlib INSIDE build_figure and
│                     # constructs nothing at module scope (a worker re-imports it)
├── chart_pool.py     # the chart pool's only home: one YtdlpPool(max_workers=1). Exists so
│                     # main.py/debug.py/conftest.py have a name to resolve per call
├── db_migrate.py     # SQL migration runner (`python -m src.db_migrate`, EXPECTED_SCHEMA_VERSION)
├── backfill_history.py # ONE-SHOT operator script: pre-archive Redis history → Postgres, direct
│                     # (not via the outbox). Run BEFORE deploying this build — see rules/state-and-recovery.md
├── guild_state.py    # Pure Redis schema: frozen value objects, field constants, orjson wire formats
├── redis_client.py   # Connection pool, GuildRedisStore (@_guild_op), cache helpers, recovery lock
├── youtube.py        # yt-dlp integration: caches, stream probe/heal, YTDL audio source, worker fn
├── ytdlp_pool.py     # ProcessPoolExecutor lifecycle: lazy spawn, break-healing, worker recycling,
│                     # worker log plumbing
├── sources.py        # Input parsing → YTSource / SpotifySource / SoundcloudSource; mints query_source;
│                     # is_mix (a YouTube Mix, which yt-dlp walks window by window)
├── spotify.py        # Spotify Web API client (client-credentials, Redis-cached); the playlist
│                     # pager, its walk slot and single flight, and the album walk
├── help.py           # man(1)-styled embed -help command (copy lives on the commands themselves)
├── dashboard.py      # optimistic-send + live-edit driver shared by -ping and -debug;
│                     # LiveMessage (send/edit-on-change/floor) is also the card's
├── queue_progress.py # the live card a slow COLLECTION enqueue shows while it resolves:
│                     # the phase model, the renderer, and the context manager that
│                     # owns the delay, the driver task, the send, N edits and the
│                     # delete. Two entry points (play.py and play_pipeline)
├── ping.py           # -ping health dashboard: probes + render (sequencing is dashboard.py)
├── debug.py          # -debug snapshot machinery and DebugSettings; OBSERVATION-ONLY by rule
│                     # collectors are live-edit probes (dashboard.py); host blocks are owner-only
├── telemetry.py      # OTel traces+logs, structlog config, worker logging, gateway span filter
├── config.py         # ENVIRONMENT (env var; main() may infer it from the git branch), SpotifyStatus,
│                     # tunables and their override accessors
├── settings.py       # the -settings machinery: the registry of what chat may set (bounds,
│                     # rendering) and the grammar that parses a request, both pure;
│                     # GuildSettings, the cache and ONLY writer of guild:{id}:config; and
│                     # BotSettings, which applies the operator's stored bot overrides. What the
│                     # environment holds is config.py; this module decides what chat may change
├── settings_card.py  # -settings' embeds (pure): server card, bot card, detail, every reply;
│                     # the command body is commands/settings.py
└── util.py           # logger factory, embed helpers (safe_label, verbatim_code),
                      # fmt_duration/fmt_seconds, progress_bar/progress_line (the NP bar and
                      # the card's), task helpers (spawn_background, cancel_task, join_task,
                      # set_within), channel_claim, ProgressFn, PoolSlotUnavailable,
                      # is_operator (the owner check -debug and -ping share)

scripts/              # run by hand, never imported by the bot — so the runtime image
                      # does not carry them. ytdl_formats.py is `just ytdl-formats
                      # <url>` (the format yt-dlp selects and the fallback ladder the
                      # retry would walk; run at every yt-dlp bump), deploy.sh is
                      # `just deploy`'s no-`just` twin. The TEST image copies this
                      # directory, because their tests are a merge gate like any other
migrations/           # NNNN_*.sql, applied in numeric order; the ONLY source of schema
docs/ARCHITECTURE.md  # the only tracked file under docs/ — anchor target for comments (rule 2)
tests/                # one test_<module>.py per src module, commands/ mirroring src/commands/,
                      # + conftest.py (seams) + helpers.py + mock_spec_cache.py
                      # test_pg_integration.py / test_redis_integration.py are the opt-in tiers
justfile              # every dev command; build_common.sh / build_docker.sh / deploy_docker.sh
                      # compose them, and scripts/deploy.sh is `just deploy`'s no-`just` twin
Dockerfile            # 3 stages: builder (deps) → test (adds test+lint groups) → runtime (ffmpeg, no poetry)
docker-compose.yml    # bot (host network) + redis + postgres + db-migrate (one-shot,
                      # `archive` profile) + db-backfill (one-shot, `ops` profile, run by
                      # hand) + bgutil-pot-provider + otel-lgtm
.github/workflows/    # ci.yml, security.yml (pip-audit), todo-to-issue.yml
```
