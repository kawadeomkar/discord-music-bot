# syntax=docker/dockerfile:1.7
FROM python:3.14-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# ── Builder stage ─────────────────────────────────────────────────────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

# Copy lockfiles first — dep install layer only invalidates when deps change, not source.
COPY pyproject.toml poetry.lock ./

# BuildKit cache mounts: reuse pip/poetry download cache across builds.
# Critical for yt-dlp (large, frequent updates) and pynacl (C extension compile).
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only=main --no-root

# Bytecode-compile the venv. Poetry leaves site-packages as .py only, and the runtime
# stage runs as uid 10001 against a root-owned /app/.venv — so the interpreter cannot
# write __pycache__ and RE-COMPILES every module on every import, in every process, for
# the life of the container. Measured in this image (median of 7, page cache warm):
#
#     import src.main    3747ms -> 1670ms
#     import matplotlib  2485ms -> 1300ms
#
# Paid on bot startup, by the forkserver and by every worker. `|| true` for the few
# dependencies shipping unparseable vendored sources; `test -d` guards the GLOB, which
# `|| true` would otherwise swallow if a base-image bump moved the path.
RUN test -d /app/.venv/lib/python*/site-packages \
 && (python -m compileall -q -j 0 /app/.venv/lib/python*/site-packages || true)

# ── Test stage ────────────────────────────────────────────────────────────────
# Inherits the builder venv and adds test deps (pytest, fakeredis, pytest-cov)
# plus lint deps (ruff, pyright). Runs pytest by default; `DOCKER=1 just <recipe>`
# overrides the command to run ruff or pyright against a bind-mounted src/ and
# tests/ instead. Deliberately does NOT include the `dev` group, so the task
# runner itself stays out of this image -- it is invoked from the host.
# Used by the container-test CI job. Never pushed to GHCR.
FROM builder AS test

# The rm shares this RUN deliberately. nodejs-wheel-binaries (pulled by pyright's
# `nodejs` extra — see pyproject.toml) ships 65MB of C headers for building native
# Node addons, and pyright is pure JavaScript that compiles nothing. Deleting them
# in a LATER layer would reclaim nothing: layers are additive, so the files would
# still sit in this one and only be masked. Same layer, or it is pure theatre.
# The `test -d` is not decoration: `rm -rf <glob>` exits 0 whether or not the glob
# matched, so if pyright ever drops the nodejs extra or the wheel relocates its
# headers, 62MB would quietly return to this image with nothing to notice it. Failing
# the build is the point — the assumption is then re-examined rather than silently lost.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only=main,test,lint --no-root \
 && test -d /app/.venv/lib/python*/site-packages/nodejs_wheel/include \
 && rm -rf /app/.venv/lib/python*/site-packages/nodejs_wheel/include

# The test group's own packages, for the same reason as the builder stage above: the
# container tier imports pytest, fakeredis and matplotlib on every run.
RUN test -d /app/.venv/lib/python*/site-packages \
 && (python -m compileall -q -j 0 /app/.venv/lib/python*/site-packages || true)

COPY src/ ./src/
COPY tests/ ./tests/
# The migration runner discovers .sql files at run time, and a test asserts the
# directory's contents agree with EXPECTED_SCHEMA_VERSION — so the suite needs
# them present, not just the module.
COPY migrations/ ./migrations/
# Same reason, different file: `just outbox` hardcodes the outbox key and consumer
# group because a shell recipe cannot import them, and a test reads this file back
# to prove the two have not drifted. Without it that guard fails in the container
# tier with FileNotFoundError while passing everywhere else — which is how a check
# ends up skipped instead of fixed.
COPY justfile ./
# And the same again for the default-password coupling. The literal lives in four
# places no Python import can reach — compose, the build preflight, the env
# template and the generator — so the tests read those files to prove they have
# not drifted, and setup_env.sh is executed against a temp copy to prove it
# tightens .env's mode. All four have to be in the image or those guards are
# container-tier failures rather than assertions.
COPY docker-compose.yml build_common.sh setup_env.sh .env.example ./
# The two deploy entry points, for the same reason once more: the archive's
# opt-in is a shell parser (resolve_archive_profile) and an ordering — migrate,
# then `up` — that only these files record, so the tests read them back. Absent,
# those seven guards fail with FileNotFoundError in the container tier alone.
COPY deploy_docker.sh build_docker.sh ./

ARG ENVIRONMENT=development
# RUFF_CACHE_DIR is under /tmp so it stays writable when the container runs as
# the host uid (needed so ruff's rewrites come out host-owned, not root-owned).
# MPLCONFIGDIR is there for the same reason: an unwritable one makes matplotlib
# fall back to a temp dir and warn on every import. Its three sites — here, the
# runtime stage and tests/conftest.py — hold DIFFERENT paths and are hand-checked on
# the property that each is writable by the uid running there (CLAUDE.md rule 6d).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="." \
    ENVIRONMENT="${ENVIRONMENT}" \
    RUFF_CACHE_DIR=/tmp/ruff-cache \
    MPLCONFIGDIR=/tmp/mplcache

CMD ["python", "-m", "pytest", "--tb=short", "-q"]

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtualenv from builder — no Poetry in the runtime image.
COPY --from=builder /app/.venv /app/.venv

# Copy source last — most frequently changed, should be the last layer.
COPY src/ ./src/
# The app's own modules. Unlike the venv above this is NOT a measured win --
# `import src.main` is the same before and after, within noise, because third-party
# module execution dominates and src is 27 files. It is here so the image is
# self-contained: /app/src stays root-owned while the process runs as uid 10001, so
# src/__pycache__ can never be written at runtime, and without this step the layout
# would depend on whether a build host happened to leak one.
RUN python -m compileall -q -j 0 /app/src
COPY pyproject.toml ./
# Required by the compose `db-migrate` one-shot, which runs `python -m
# src.db_migrate` out of THIS image so the runner and the schema it applies can
# never be different versions.
COPY migrations/ ./migrations/

# Non-root: ffmpeg and the venv need no privilege. The uid/gid are fixed rather
# than distro-assigned so volume ownership survives rebuilds and base-image bumps.
# HOME is set below because yt-dlp derives its cache directory from it, and
# docker-compose.yml mounts ytdlp-cache at that path.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --home-dir /home/app --create-home app \
 && mkdir -p /home/app/.cache/yt-dlp /home/app/.cache/matplotlib \
 && chown -R app:app /home/app
# Numeric, not `app`: kubelet checks runAsNonRoot against the image's USER and
# cannot resolve a name, failing the pod with "image has non-numeric user".
USER 10001:10001

ARG ENVIRONMENT=production
# The commit this image was built from. GIT_SHA existed only as an image TAG, so a
# running bot could not report its own commit — `just up <sha>` deploys by tag and
# the process never saw it. Both forms are needed: the LABEL is the OCI-standard
# annotation external tooling reads, but labels are invisible from inside the
# container, so -debug reads the ENV. Dirty builds pass `<sha>-dirty.<digest>`
# through unchanged, so the bot reports exactly the tag that was deployed.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="." \
    HOME="/home/app" \
    ENVIRONMENT="${ENVIRONMENT}" \
    GIT_SHA="${GIT_SHA}" \
    MPLCONFIGDIR="/home/app/.cache/matplotlib" \
    LIVENESS_FILE="/tmp/bot-alive"

# `restart: always` only covers the process exiting; a wedged event loop leaves
# the container up while it answers nothing. The bot touches LIVENESS_FILE from a
# loop-resident task, so a stale mtime means the loop stopped turning. Not a
# dependency probe: a Redis blip must not mark the bot dead.
#
# This REPORTS, it does not act. The engine takes no action on an unhealthy
# container — only Swarm, Kubernetes or an autoheal sidecar restarts one — so
# under plain compose the effect is a status `docker ps` and monitoring can see.
#
# start-period covers login and extension load. interval x retries marks it
# unhealthy after ~90s, above the 15s touch cadence.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,sys,time; f=os.environ['LIVENESS_FILE']; sys.exit(0 if os.path.exists(f) and time.time()-os.path.getmtime(f) < 90 else 1)"

CMD ["python", "-m", "src.main"]
