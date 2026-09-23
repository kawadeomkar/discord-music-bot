---
paths:
  - "tests/**"
  - "pyproject.toml"
---

# Testing: layout, the seams the suite installs, and the two opt-in tiers

How the suite is organized, what it fakes and where it deliberately does not,
plus the traps that make a green run mean nothing if they are broken.

- Layout: one `tests/test_<module>.py` per src module. A command's tests live with its
  BODY, and drive it through the cog's wrapper — the wrapper resolves the player and
  owns the try/except, so a body that raises and a body that reports are different
  behaviours and only the pair is the command. `tests/commands/test_<command>.py`
  mirrors `src/commands/`, and needs its `__init__.py`: `tests/` is a package, so a
  subdirectory without one collides with a same-named file above it.
  (`test_leaderboard.py` also owns
  the cog command that drives it, since splitting the renderer's tests from the
  command's would make a reader check two files to learn what one board looks like;
  `test_debug.py` likewise owns `MusicBot.debug_suffix` and the `-debug` card's
  end-to-end assertions, for the same reason — what the footer says and what puts it
  there are one behavior; `play_placement.py`'s grammar and registry are tested in
  `test_play_placement.py`, the placement itself in `commands/test_play.py`),
  plus `conftest.py` (shared fixtures/seams),
  `helpers.py` (builders), `test_context.py` (Discord context doubles). `config.py` is
  the intentionally-least-covered module.
  `test_telemetry.py` restores structlog's PROCESS-wide configuration itself, because
  conftest's `configure_structlog_for_tests` is session-scoped and `setup_telemetry()`
  reconfigures structlog for real — without that restore the production JSON chain
  would stand for every test that runs after it.
- **The yt-dlp seam** (autouse fixture `use_thread_ytdlp_pool`): every test runs
  extraction on an in-process ThreadPoolExecutor-backed `YtdlpPool`, because tests patch
  `src.youtube._ytdlp_extract` with MagicMocks that could never be pickled to a real
  worker. Both module-level names (`ytdlp_pool`, `_ytdlp_extract`) are resolved per call
  in `_run_extract` specifically to keep those patches working — don't capture them.
  Consequence: no test spawns worker processes; the pickle contract is asserted directly
  (`TestProcessBoundaryContract`), and one dedicated test spawns a real worker.
- **The suite runs archive-ENABLED, inverting the ship default**: a conftest autouse
  fixture pins `HISTORY_ARCHIVE_ENABLED=true` (next to the `POSTGRES_URL` scrub),
  because the enabled configuration exercises strictly more code and hundreds of
  existing assertions encode it. Disabled-mode behavior is covered by explicit tests
  that monkeypatch the flag per case — which wins over the fixture (same MonkeyPatch
  instance, later call). Don't "fix" the fixture to match the ship default.
- **Bot knobs in tests** are set with `config.<knob>.set_override(value)` in the test body.
  Don't patch a consumer module's copy (there is none), and don't use `monkeypatch.setitem` on
  `_OVERRIDES` (pyright does not check that value). To stand in for an ENVIRONMENT value,
  which has no setter, `monkeypatch.setitem(config._BASELINES, "<NAME>", value)`. The autouse `clear_bot_knob_overrides`
  clears every override after each test. `monkeypatch.setattr(config, "<KNOB>", v)` patches the
  env **baseline**, which an override shadows. Use it only for tests about the baseline.
- Redis in tests is `fakeredis`; Discord objects are `MagicMock(spec=...)` doubles,
  built through the spec cache `tests/conftest.py` installs at import — so **a spec
  class must not be mutated once it has been used as a spec** (`functools.wraps` on
  the replacement keeps a class-level patch payload-neutral). It is the one file
  outside `src/` the coverage gate measures. **The container tier runs with the cache
  OFF** (`MOCK_SPEC_CACHE_DISABLE=1` in the Dockerfile's test stage), so
  `container-test` is a reference run against stock `unittest.mock` and the two tiers
  disagree if the cache ever answers what upstream would not; the cache's own tests
  skip themselves there and run in the venv tier. See
  `docs/ARCHITECTURE.md#the-mock-spec-cache`.
  **fakeredis executes every stream command the outbox uses and gets five of them
  wrong**, all in the safe-looking direction (green tests, broken production): the
  `xtrim(approximate=True)` default trims exactly here and nothing on a real small
  stream; `XAUTOCLAIM`'s completion cursor is the last-scanned id rather than `0-0`;
  `XINFO GROUPS` `lag` is off by one and can go negative; `XADD` against a list raises
  `AttributeError` rather than `ResponseError`; `ref_policy` is unsupported. They are
  enumerated in `tests/test_redis_integration.py`'s docstring because they have to be
  known rather than discovered. What fakeredis *does* model faithfully is the tombstone
  shape `(id, {})`, so the P1-critical drain path is unit-testable.
- **The `pg` tier** (`tests/test_pg_integration.py`, marker `pg`) runs against a real
  `postgres:18-alpine` via testcontainers (`just test-pg`, needs Docker) or against
  `POSTGRES_TEST_URL` in CI. Excluded from the default run. Several invariants live ONLY
  there (ON CONFLICT dedup, the `-history` tie-break, the schema lock in both directions,
  `NOT VALID`'s treatment of legacy rows, and `play_history_rejected.payload` holding a
  NUL byte that `jsonb` and `text` both refuse),
  so a conftest hook fails `-m pg` outright if the tier is selected but disabled — an
  all-skipped tier used to exit 0 and look green.
- **The `redis` tier** (`tests/test_redis_integration.py`, marker `redis`) is the same
  shape against a real `redis:7-alpine` (`just test-redis`, or `REDIS_TEST_URL` in CI),
  and the conftest hook gates it identically. It exists because of the divergence list
  above: that an exact trim actually trims, that `WRONGTYPE` is a `ResponseError`, and
  that `XAUTOCLAIM`'s cursor is `0-0` are all things fakeredis answers **wrongly**
  rather than not at all. It also fails deliberately if the server reaches Redis 8.2,
  where `XTRIM ... ACKED` collapses the cap's hand-rolled ack-before-trim rule into one
  keyword.
- `pytest-timeout` sets a 120s per-test deadline. Several guards here are
  `asyncio.timeout()` calls whose removal makes a test HANG rather than fail; without
  the deadline that burns a CI job's full timeout and reports a cancellation.
- structlog is reconfigured per-session for readable output, and contextvars are cleared
  between tests (autouse).
- Run `just check` before pushing (the pre-push hook runs it). It is the contract for
  CI's lint and test jobs but NOT the whole pipeline: `just ci` adds the container job
  and both integration tiers; the runtime-image build and pip-audit run only in CI.
  `check` is a plain dependency list of five — `fmt-justfile pins fmt-check lint
  check-heavy` — whose first four run in order and stop at the first failure.
  `check-heavy` is the exception: it runs `types` and `test` concurrently and reports
  both outcomes, so a pyright failure no longer hides what pytest would have said.
  The four cheap ones cost ~1.3s combined, which is what lets the pre-push hook give
  them a status line each (pre-commit renders one line per hook, runs hooks
  sequentially, and buffers a hook's output until it exits, so line count is hook
  count); fusing exactly the two slow ones is what keeps that affordable. The five
  pre-push hooks mirror those five dependencies in order and `just pins` asserts it.
  CI invokes `lint`/`types`/`test` individually rather than calling `check`, so its
  jobs fail independently of this ordering.
- Warnings are errors (see golden rule 11). `ENVIRONMENT` is read from the environment
  alone at import (default `development`), so collection runs no git subprocess and a
  detached worktree needs nothing set.

## Why a subset run is serial and ungated

Test selection (args forward to pytest). ANY argument means a subset run, so it runs
SERIALLY and coverage is skipped — fail_under is a PROJECT floor and one file measures
~26%, which used to fail a green run with exit 1. The gate rides the no-args form —
what `just check` and the pre-push hook invoke — and `test-report`, whose arguments are
reporting flags rather than a selection, keeps it with COVERAGE_GATE=1.

The no-args form is also the ONLY parallel one (`-n auto`), and that is deliberate:
the gate is the only way the whole suite runs, so a test that is not parallel-safe
fails the pre-push hook and CI instead of rotting a separate "fast" recipe. A subset
stays serial because worker startup (~4s flat) cannot amortize over a narrow
selection, and because execnet does not forward worker stdout — `-s` is silently
swallowed under `-n` and `--pdb` disables it. `just test tests/` is the escape hatch:
the whole suite, serially, to reproduce a parallel-only failure.
