---
paths:
  - "src/main.py"
  - "src/musicbot.py"
  - "src/telemetry.py"
  - "src/debug.py"
  - "src/ping.py"
  - "src/dashboard.py"
  - "src/commands/debug.py"
  - "src/commands/ping.py"
  - "tests/test_{main,musicbot,telemetry,debug,ping,dashboard}.py"
---

# Process lifecycle and observability

What `main()` and `setup_hook` do in what order, what `close()` unwinds, and the
telemetry every log line and span carries. Order is load-bearing in both
directions: a step in the wrong place either reads a value nothing has parsed yet
or skips teardown that a later step depends on.

### Startup and shutdown

`main()` order matters: `setup_telemetry()` first (configures structlog before any
`get_logger()` resolves), then `DISCORD_TOKEN` check, then `MusicBotApp()` construction
(inside `main()` only — see golden rule 10). `setup_hook` reads
`HISTORY_ARCHIVE_ENABLED` **first** (before anything else can consume it — the parser
raises on garbage, and the next reader is `@_guild_op`-swallowed `push_history`, so
startup is the only loud place) and `BOT_SETTINGS_OVERRIDES` second, for the same
reason; creates the Redis pool and `MusicBotApp.bot_settings`, then branches. Enabled: it
**requires `POSTGRES_URL`** (it raises otherwise — an enabled bot running without the
archive would XADD onto an outbox nobody drains), constructs `PostgresHistoryArchive`
(lazy: no connection is made here, so startup never blocks on Postgres), starts the
`HistoryOutboxDrainer`, and spawns `_verify_archive_reachable` — a background probe
that retries `health_check` for ~a minute and then logs ONE error. It exists because
the required-`POSTGRES_URL` check above cannot see this failure: compose interpolates
the DSN before profile filtering, so a bare `docker compose up` (which never activates
the `archive` profile) gives an enabled bot a URL with no database behind it, and the
lazy pool would not discover that until the first song end. Disabled (the default): `history_archive`/`history_drainer`
stay `None`, one INFO says so, a set `POSTGRES_URL` is explicitly ignored (the flag,
never URL presence, is consent), and a leftover outbox from an earlier enabled run
draws a WARNING naming the un-drained depth (never auto-deleted). Either way it then
loads the `src.musicbot` extension, spawns `BotSettings.hydrate_until_read()` without
awaiting it (every knob runs on its env value until it lands; a failed read retries with
backoff, and the bot card and `-debug` say while it has not landed —
`docs/ARCHITECTURE.md#settings-resolution`), and fire-and-forgets
`ytdlp_pool.prewarm(warm_worker)` — then `chart_pool.warm()`, but only while the
archive is enabled, so a default deployment never spawns the matplotlib worker — so
the first `-play` doesn't pay worker-spawn, yt-dlp-import or first-`YoutubeDL`
latency (the pool stays lifecycle-only: the warm-up callable comes from
`src.youtube`, like every other callable it runs). `MusicBotApp.invoke` also
short-circuits `--help` anywhere in a command message straight to that command's help
embed, before voice checks or argument parsing.

`close()` order: cancel `_archive_probe_task` (before the archive: the probe reads its
pool, and `_ensure()` refuses once `close()` has latched it shut) →
`history_drainer.stop()` (final drain, needs Redis AND the archive) →
`history_archive.close()` — all three skipped when the archive tier is off (the attrs
are `None`) → close Redis pool → `super().close()` → `ytdlp_pool.aclose()`
→ `chart_pool.aclose()` (inert when the worker was never spawned)
(10s join timeout, then `terminate_workers()` — an unbounded join measured 61s to exit)
→ `close_probe_session()` (latches the module closed, so a player loop still running
during the flush below cannot rebuild a session nothing will close)
→ `shutdown_telemetry()` via executor (blocking span flush, up to 30s). `close()` is
one-shot (`_teardown_started`) and **every step is individually guarded**: a hung
Postgres once made `archive.close()` raise, which skipped every later step permanently.

### Observability

structlog JSON to stdout always; when `OTEL_SDK_DISABLED` ≠ true, OTLP gRPC traces +
logs to `OTEL_EXPORTER_OTLP_ENDPOINT` (compose: Grafana LGTM — Tempo/Loki/Grafana at
localhost:3014, admin/admin). Every log line carries `environment`, `trace_id`/`span_id`
when in a span, and command context (`guild_id`, `user_id`, `command`) bound in
`cog_before_invoke` (which also opens a `command.<name>` span; `cog_after_invoke`
closes it, `cog_command_error` records onto it). `_DiscordGatewayFilter` drops
discord.py-internal HTTP spans. Redis and aiohttp are auto-instrumented. Spans embed
their `trace_id` in error-embed footers (`trace_footer`) so a user report can be joined
to a trace. **`player.loop.iteration` is a ROOT span**, so one song is one trace — the
loop task inherits the context that created the player, and an inherited parent files
every song a guild ever plays under one `-play`. Its id is captured into
`MusicPlayer._playback_span` at the song's start and printed on the Now Playing card
and the playback-error notice, which is why both name the same trace. `-ping` is a live-editing dashboard (1s tick, 3s deadline, env- or owner-tunable)
probing Discord/Redis/Spotify/Postgres/OTEL and reporting bot/yt-dlp/ffmpeg
versions; `max_concurrency(1, guild)`.
