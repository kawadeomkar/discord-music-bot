# Adversarial Review — `task/async-pg-impl` (Postgres foundation)

**Branch:** `task/async-pg-impl` &nbsp;·&nbsp; **Baseline:** `main` (`b745475`) &nbsp;·&nbsp; **Diff:** 15 files, +1124 / −29 (src: +456)
**Method:** three parallel adversarial reviewers (correctness/concurrency · architecture/conventions/security · performance/test-coverage), de-duplicated and independently verified against the code and git history.
**Date:** 2026-07-23

---

## Verdict

**No Critical or High findings.** This is the *foundation* branch of the Postgres-history stack — it adds the durable substrate (`HistoryArchive` protocol, `PostgresHistoryArchive` with an embedded lazy asyncpg pool + inline schema DDL, the `HistoryOutboxDrainer`), the Redis outbox write path, the `guild_id` wire field, main wiring, and the compose pg service + env-var inference. It has **no** `db.py`, migrations dir, backfill CLI, or `GuildHistory._merge_fresh` — those land in later stack branches, so the pg-18 quantization defects **do not apply here.**

Both of my merge resolutions were independently verified **correct and race-free**: keeping the outbox LPUSH + dropping the `# type: ignore` (the pipeline is transactional, so the dual write is atomic), and the auto-merged `close()` teardown order. The `Any`/`cast` type-sweep masks no real interface mismatch.

**The one finding with teeth is a merge regression I introduced**, plus a coverage gap inherent to a foundation branch and a cluster of low-severity hygiene items.

---

## Findings by priority

| # | Sev | Location | Issue |
|---|-----|----------|-------|
| M1 | **MED** | build system (merge) | `2827fcd`'s `POSTGRES_PASSWORD` fail-fast **preflight was dropped** when the merge deleted `build.sh`; not ported → only compose's late `:?` guard remains |
| M2 | **MED** | `history_archive.py` (coverage) | Entire asyncpg surface (`_ensure` DDL, `insert_batch`, `recent`, pool lifecycle) is **0%-covered**; file at 75%. SQL constants first execute on a downstream branch |
| L1 | LOW | `history_archive.py:~200` | Drainer task has **no supervision / done-callback** → an abnormal exit stops all draining silently, outbox grows unbounded, no ERROR log |
| L2 | LOW | `redis_client.py:461,464` | `push_history` **double-serializes** the entry (display + outbox LPUSH) when `outbox=True` |
| L3 | LOW | `history_archive.py:45-61` | Inline DDL is `CREATE … IF NOT EXISTS` — **creates but cannot evolve** a stale schema; concurrent bootstrap has a theoretical cross-process race |
| L4 | LOW | `history_archive.py:71,96,159` | `recent()` / `_RECENT_SQL` / `_row_to_entry` are **dead + untested** on this branch (Phase-B scaffolding); a column-name mismatch wouldn't surface until Phase B |
| L5 | LOW | `tests/test_history_archive.py:70` | `archive: Any` fixtures **defeat `FakeArchive` ↔ `HistoryArchive` conformance checking** |
| L6 | LOW | `.env.example:12` | Ships `POSTGRES_PASSWORD=change_me` — a concrete value that **satisfies the `:?` guard**, so a verbatim copy runs with `change_me` |
| L7 | LOW | `tests/test_history_archive.py:60` | `FakeArchive.recent()` returns **oldest-first**, but the protocol/impl specify newest-first — would encode wrong ordering into Phase-B tests |
| L8 | LOW | `docker-compose.yml:20` | Comment hardcodes `see main.py:113` — a **fragile line reference** that rots on the next edit |
| I1 | INFO | `history_archive.py:169-173` | `close()` sets `_pool = None` **before** awaiting `pool.close()` — safe only because teardown order guarantees no concurrent `_ensure()`; pin with a comment |
| I2 | INFO | merge | `CLAUDE.md` deletion is staged (kept on disk, gitignored on main) — a conscious call before committing |

---

## MEDIUM

### M1 — Fail-fast PG-password preflight lost in the merge  *(my resolution — actionable)*
**build system** · *verified directly*

The branch's headline commit `2827fcd` ("Update inferring of PG env vars") added a preflight to `build.sh` that greps `.env` for `POSTGRES_PASSWORD` and **exits 1 before building the image** if it's unset — its stated purpose: *"so we don't build an image just to trip over a missing secret."* The merge deleted `build.sh` (main replaced it with `build_docker.sh`/`build_common.sh` in the just-migration), and I accepted that deletion — but **the preflight was not carried into the new build scripts** (`grep POSTGRES build_common.sh build_docker.sh deploy_docker.sh justfile` → nothing). The only remaining guard is compose's `${POSTGRES_PASSWORD:?…}` at `docker-compose.yml:21,76`, which fires at `up` time — the exact *late* failure the preflight existed to prevent.

**Fix:** re-port the preflight into `build_common.sh` (where the compose build is invoked), or make a conscious decision to rely on the `:?` guard and document it. *I can apply this on request.*

### M2 — The asyncpg implementation is untested on this branch
**`src/history_archive.py` — 75% (26/122 stmts missed); total 91.22%, 1224 passed**

The 0%-covered lines are exactly the server-touching code: `_ensure` (132-148: `create_pool`, DDL execution, the `except BaseException: await pool.close()` rollback), `insert_batch`'s real body (acquire + `executemany`), `recent`'s real body (acquire + `fetch` + row map), and `close()` with a live pool. The only PG tests use a bogus DSN to hit early-outs. Consequences: `_SCHEMA_DDL` never runs against a real server (a SQL typo ships silently); `_INSERT_SQL`/`_RECENT_SQL` 10-param binding, the `play_history_dedup` `ON CONFLICT`, `executemany` atomicity, and timestamptz round-tripping are validated only by inspection. **"All tests green" does not validate the server-touching code.** Acceptable for a foundation branch (the Docker-gated `pg` tier lands later), but the SQL constants defined here first execute on a downstream branch — land the integration tier before anything relies on the real path in production.

---

## LOW

- **L1 — Drainer has no crash supervision** (`history_archive.py`, `start()` → bare `create_task`). `_run` defends the drain path (`except Exception` → backoff), but an exception in the idle branch or a `BaseException` kills the task; nothing retrieves the result, so it surfaces only as a late GC "Task exception was never retrieved" warning. Draining stops while `add()` keeps LPUSHing → the non-evictable outbox grows unbounded, no ERROR. Low reachability (real failures live on the covered drain path). *Memory notes "M2 drainer crash supervision" landed on a later branch; it's absent here.* Fix: `add_done_callback` that logs and restarts.
- **L2 — Double serialization** (`redis_client.py:461,464`): `serialize_history_entry(entry)` runs twice per song-end when `outbox=True`. The pipeline is transactional so it's *correctness*-safe — pure redundant `orjson.dumps`. Fix: `wire = serialize_history_entry(entry)` once, reuse in both LPUSHes (also guarantees byte-identity).
- **L3 — Inline DDL can't evolve; bootstrap race** (`history_archive.py:45-61`, `_ensure`). `CREATE … IF NOT EXISTS` is idempotent and injection-free, but if `play_history` already exists with an older column set, `IF NOT EXISTS` no-ops and a later `_INSERT_SQL` referencing a new column fails at runtime, not DDL time. Two processes racing first-use can also collide on `CREATE INDEX`. The one-process-per-PG deployment + the advisory-locked runner arriving with `db.py` later both mitigate; noting the "creates but cannot ALTER" trap for the migration cutover.
- **L4 — Dead, untested Phase-B read path** (`history_archive.py:71,96,159`). Production `-history` reads go through `GuildHistory.recent()` → Redis; `PostgresHistoryArchive.recent()`/`_RECENT_SQL`/`_row_to_entry(row: Any)` have zero callers and zero server coverage here. Intentional scaffolding (docstrings say so). When Phase B wires it, add a `pg`-marked test and type `row` as `asyncpg.Record` rather than `Any`.
- **L5 — `Any` fakes defeat conformance checking** (`tests/test_history_archive.py`). `HistoryOutboxDrainer(fake_redis, archive)` is built through an `archive: Any`, so pyright can't verify `FakeArchive` still satisfies `HistoryArchive`. Fix: one `_: HistoryArchive = FakeArchive()` (or type the fixture `-> FakeArchive` and let the drainer's `HistoryArchive` param check it). *(This is a residue of my `Any` type-sweep — the fakes are faithful today, but nothing enforces it.)*
- **L6 — Placeholder password satisfies its own guard** (`.env.example:12`). `POSTGRES_PASSWORD=change_me` is a concrete value, so `${POSTGRES_PASSWORD:?…}` passes on a verbatim `.env` copy → the stack runs authenticated with `change_me`. Mitigated by `127.0.0.1`-only binding. Fix: leave it empty in the example, or reject `change_me`.
- **L7 — Fake `recent()` ordering wrong** (`tests/test_history_archive.py:60`). Returns oldest-first; protocol says newest-first. Not load-bearing yet (no production caller), but Phase-B tests written against it would encode the wrong order. Fix: reverse it, or add a `# TODO` noting the divergence.
- **L8 — Fragile line-number comment** (`docker-compose.yml:20`, "see main.py:113"). Accurate today; rots on the next `main.py` edit (this merge already shifted those lines). Prefer "see `MusicBotApp.setup_hook`".

## INFO

- **I1 — `close()` nulls the pool before awaiting close** (`history_archive.py:169-173`). Safe *only* because `main.close()` runs `drainer.stop()` (finishes/cancels draining) strictly before `archive.close()`, so no `_ensure()` runs concurrently. If that order were reversed, a concurrent `_ensure()` would see `_pool is None` and build a second pool mid-shutdown (leak). Add a one-line comment pinning the ordering dependency.
- **I2 — `CLAUDE.md` deletion staged.** main doesn't track it; the file survives on disk (25 KB, gitignored) but drops from the tree on a clean checkout. Conscious call before committing (I kept it locally, matching the pg-18 decision).

---

## Verified clean / positives

- **Merge resolution (a): outbox LPUSH kept + `# type: ignore` dropped — CORRECT.** `redis.pipeline()` defaults to `transaction=True` (verified: `Redis.pipeline(transaction: bool = True)`), so display-LPUSH + PERSIST + outbox-LPUSH run as one MULTI/EXEC — atomic, no partial-write window. The dropped ignore was genuinely unnecessary (a pipeline `lpush` returns synchronously; unlike the still-needed cast on the *awaited* `lrange`).
- **Merge resolution (b): `close()` order `drainer.stop() → archive.close() → redis pool → ytdlp_pool` — CORRECT.** The bounded final drain needs Redis + archive alive, so it precedes both closes; `stop()` cancels-and-awaits before the final drain (no `_run`/drain overlap) and never raises.
- **At-least-once (peek → insert → retire): correct.** `insert_batch` strictly precedes `retire_outbox`; a crash/error between them leaves entries in place; redelivery collapses via the `play_history_dedup` unique index + `ON CONFLICT DO NOTHING`. Test-pinned.
- **Oldest-first ordering & single-consumer:** LPUSH-head / LRANGE-tail-slice+reverse / RPOP-tail traced index-by-index; concurrent head pushes untouched. `RPOP key count` needs Redis ≥ 6.2 — image is `redis:7-alpine` ✓.
- **Corrupt-entry handling:** retired via `len(raw)` (not `len(entries)`) — a poison entry can't wedge the head; surviving batch entries still insert.
- **notify/wake:** no lost-wakeup (draining peeks the outbox each iteration; `clear()` after `wait()`; missed notify bounded by `TICK_SECS=30`).
- **SQL & secrets:** every query `$n`-parameterized; DDL fully static; `POSTGRES_URL` never logged (`_log_retry` logs only `type(error).__name__` + message).
- **Outbox non-evictable invariant:** `HISTORY_OUTBOX_KEY` never gets EXPIRE/persist, absent from `_pipe_expire_all`/`refresh_ttl`; under `volatile-lru` never an eviction candidate. Preserved end-to-end.
- **Wire back-compat:** `guild_id` added as leading field, default `0`; golden bytes updated + `test_pre_postgres_entry_parses_with_guild_id_zero` + snowflake precision test. Tolerant both directions across rolling restarts.
- **Postgres off the playback hot path:** `add()` awaits only Redis + sync `notify()`; `recent()` reads Redis; drainer is decoupled; lazy pool → startup never blocks on PG. Verified.
- **`Any`/`cast` masks nothing** (both arch + correctness reviewers): `cast(Any, object())` is a pure non-None sentinel (`add()` only checks `is not None`); `FakeArchive` faithfully implements the protocol signatures (aside from the L7 ordering nit).

---

## Suggested order of work

1. **Re-port the `POSTGRES_PASSWORD` preflight** (M1) into `build_common.sh` — the one real regression, and it's from this merge.
2. Land the **Docker-gated PG integration tier** (M2) before any consumer relies on the real asyncpg path — the SQL constants are currently unvalidated against a server.
3. Opportunistic LOWs: drainer `add_done_callback` (L1), single serialize (L2), typed `FakeArchive` binding (L5), empty/`change_me`-reject `.env.example` (L6). L1/L5 also arrive naturally when later stack branches merge down.
