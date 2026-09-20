---
paths:
  - ".github/**"
  - "Dockerfile"
  - "docker-compose.yml"
  - "justfile"
  - ".pre-commit-config.yaml"
  - "build_common.sh"
  - "build_docker.sh"
  - "deploy_docker.sh"
  - "pyproject.toml"
  - "poetry.lock"
---

# CI, the image build, deployment, and every duplicated version pin

## The duplicated pins

Golden rule 6 is the rule; this is the enumeration it stands on. `just pins`
enforces the first group and nothing checks the second, so the unenforced list is
what a maintainer reads before touching a version.

6. **Version pins move in lockstep.** Bump both halves in the same commit. `just pins`
   enforces eight pairs, one name and one list — it is a dep of `check` and CI also runs it as
   its own step, deliberately: Dependabot's `pip` and `pre-commit` ecosystems open
   SEPARATE PRs that each move one half, and those PRs are validated by CI and never
   by a local `check`.
   The eight: the ruff pin (pyproject) ↔ the ruff hook `rev` in
   `.pre-commit-config.yaml`; the image name (justfile `IMAGE` ↔ `build_common.sh`
   `IMAGE_NAME`); and `postgres:18-alpine` / `redis:7-alpine` each across three files —
   the integration tier's `_PG_IMAGE`/`_REDIS_IMAGE`, `ci.yml`'s service container, and
   `docker-compose.yml` (compared tier↔ci and compose↔ci, so all three agree); and
   `_POSTGRES_CONTAINER` (`src/debug.py`) ↔ the postgres service's `container_name`,
   which is a Prometheus label selector, so a rename there would otherwise leave
   `-debug`'s cpu/mem row reading `n/a (no metrics source)` forever rather than
   failing. The compose legs are anchored to the named service, not `head -1`, so a
   second postgres or redis service cannot silently shift what is compared.
   The eighth is the **yt-dlp version** (pyproject) ↔ the copies quoted in prose by
   `CLAUDE.md` and `docs/ARCHITECTURE.md`, which describe the client strategy for a
   specific version: Dependabot moves pyproject + `poetry.lock` without touching either,
   and main has carried a stale copy for exactly that reason.
   The **name** is the `charts` extra: `[tool.poetry.extras]` defines it, and three
   sites select it (the `CHART_EXTRAS` ARG default and the test stage in `Dockerfile`,
   and `just install`). Each site is asserted separately rather than counted — the
   Dockerfile names it twice, so a count lets a typo in either hide behind the other.
   Poetry IGNORES an unknown extra, so drift builds green and ships an image whose
   charts are silently absent. See `docs/ARCHITECTURE.md#the-charts-extra`.
   The **list** is the pre-push gate: `check`'s dependency list ↔ the five
   `entry: env DOCKER=0 just <recipe>` pre-push hooks in `.pre-commit-config.yaml`, the
   same recipes in the same order. Drift runs one way and reports green — a step added
   to `check` alone simply stops running on push. The `env DOCKER=0` prefix is matched,
   not skipped, so a misspelled pin fails here rather than quietly containerizing a
   hook.
   **Five pairs are NOT enforced — this list is what a maintainer checks by hand,
   so keep it complete:**
   (a) `bgutil-ytdlp-pot-provider` (pyproject) ↔ the
   `brainicism/bgutil-ytdlp-pot-provider` image tag in `docker-compose.yml`. The plugin
   and the sidecar are released in lockstep; drift breaks PO-token minting, which
   surfaces as YouTube playback failures, not as a red build.
   (b) The published Prometheus port `9090`, in **four** places that move together: the
   `PROMETHEUS_HOST_PORT` defaults inside the bot service's `DEBUG_PROMETHEUS_URL` and
   inside the otel-lgtm service's `ports:` entry (both `docker-compose.yml`), and the
   commented-out `DEBUG_PROMETHEUS_URL` and `PROMETHEUS_HOST_PORT` assignments in
   `.env.example`. Change one and `-debug` queries a port nothing publishes. A **fifth**
   literal — the container side of that same `ports:` entry — is Prometheus's own listen
   port inside `grafana/otel-lgtm` and must NOT move with them; both files also name the
   number in prose, which drifts just as silently.
   (c) `otel/opentelemetry-collector-contrib` (the `otelcol-metrics` service) ↔ the
   otelcol-contrib build inside `grafana/otel-lgtm` (the `otel-lgtm` service), both in
   `docker-compose.yml`. The comment above the collector's `image:` line states the rule
   — bump either image and check the other by hand. Like (a), drift is invisible to
   every build: the symptom lands on the metrics path, where a missing `docker_stats`
   series leaves `-debug`'s cpu/mem row reading `n/a (no metrics source)`, which is also
   exactly what "the `metrics` profile is not running" looks like.
   (d) `MPLCONFIGDIR`, written in **three** places that must agree on a WRITABLE path:
   the Dockerfile's test stage (`/tmp/mplcache`, beside `RUFF_CACHE_DIR`), its runtime
   stage (`/home/app/.cache/matplotlib`, created and chowned in the same `RUN` as
   `useradd`), and `tests/conftest.py` at module scope. The three deliberately hold
   DIFFERENT paths — what must agree is that each is writable by the uid that runs
   there, which is why this cannot be a `just pins` string comparison. Unwritable is
   the failure, and in the two Dockerfile copies it is quiet: matplotlib falls back to
   a temp directory and warns once per process, so the symptom is a stderr line nobody
   reads. The conftest copy is the exception — the suite renders in-process, so rule
   11 turns that warning into a red build.
   (e) `LIVENESS_INTERVAL_SECS`'s `maximum` (`_MAX_LIVENESS_SECS`, 60s, `config.py`) ↔
   the `HEALTHCHECK`'s 90s staleness window (`Dockerfile`). The cap exists so a touch
   cadence can never outlast the window; raise the window and the cap may follow. Lower
   it below 60s and lower the cap with it, or a cadence the cap still accepts outlasts
   the window and that container reports unhealthy. No build compares them.

## CI/CD and deployment

`ci.yml` jobs: **resolve-env** (environment name + semver-validated version from
pyproject — single source for image and release tags) → **version-bump** (pull
requests only: that version must be strictly above the base branch TIP's, compared as
major.minor.patch, so every merge moves it. A PR opened by `dependabot[bot]` is exempt
and the job reports success anyway — Dependabot does not set the project version, so the
rule could only ever fail it. Nothing else enforces the per-PR bump —
`release` treats an unchanged version as the ordinary no-op. It is deliberately absent
from `build`'s `needs`: a job `if`-skipped on push would skip `build` with it, and it
blocks a merge only once branch protection lists it as required) → **lint**
(justfile fmt/parse, pin agreement, ruff, pyright) and **test** (coverage + PR comment) and **container-test**
(suite inside the test image, with the mock spec cache OFF so it is the reference run
against stock `unittest.mock`; deliberately runs with a read-only token — it executes PR
code) and **pg-integration** (the `pg` tier against a postgres service container) and
**redis-integration** (the `redis` tier against a redis service container) — both real
merge gates, `build` needs them → **build** (runtime stage; on branches it only validates the build; on main it
pushes three GHCR tags: immutable `sha-<commit>`, `latest`, and the bare pyproject
version) → **release** (tag + GitHub release on main). Concurrency: PR pushes supersede each
other; main commits each get their own group so no build is ever dropped.
`security.yml` runs pip-audit against `poetry.lock` (push + schedule).
`todo-to-issue.yml` converts TODO comments to issues — write new `TODO:`/`FIXME:`
markers with that in mind (multi-line context is picked up).

Docker: three-stage build (builder → test → runtime; runtime has ffmpeg, no Poetry).
Deploys are separate from builds — `just up <sha>` deploys any locally-present image tag
and refuses to build; dirty trees produce `<sha>-dirty.<digest>` tags so a tag never lies
about its commit. `just restart` restarts the existing container and does NOT pick up a
new image. Compose runs the bot with **host networking**; a named `ytdlp-cache` volume
persists yt-dlp's player-JS/challenge cache across restarts.

`GIT_SHA` is both the deploy tag and a build-arg baked into the runtime image, as an
`ENV` **and** an `org.opencontainers.image.revision` label — the ENV is the one the
process can read (labels are invisible from inside the container), which is what lets
`-debug` report the commit it is running. `build_runtime_image()` is the single
`--build-arg` seam; every caller must **export** `GIT_SHA` before calling it, and CI
passes `github.sha`. Not a seventh `just pins` pair: the value is derived, not
duplicated.

## Operator and deploy recipes

```bash
# Database (operator tools. db-migrate/db-backfill run the LOCAL venv against
# POSTGRES_URL; setup/backup/restore are shell around pg_dump/psql, no venv needed)
just setup                 # bootstrap .env with a generated POSTGRES_PASSWORD
just db-migrate            # apply pending migrations — REQUIRED before the bot serves
just db-backfill [--dry-run] # move pre-archive Redis history into Postgres — see the state rule
just db-backfill-docker    # same, via the compose one-shot — no local venv
just db-rejects [n]        # list play_history rows Postgres refused (expected: nothing)
just outbox [idle_ms]      # outbox health: depth, in-flight, stranded, TOMBSTONES (lost plays)
just bot-settings [reset <application_id>] # list stored bot overrides, or delete one bot's
just db-backup             # dump to backups/
just db-restore FILE [DB]  # restore into a SCRATCH db (live needs CONFIRM=1 + a name)

# Build & deploy
just image                 # build runtime image :latest and :<git-sha> (no test gate)
./build_docker.sh          # full pipeline: just check → just image → deploy
just up [sha]              # deploy an already-built image (never builds; refuses unknown tags)
just down / restart / logs / ps
just test-image-rebuild    # required after changing pyproject.toml/poetry.lock
```
