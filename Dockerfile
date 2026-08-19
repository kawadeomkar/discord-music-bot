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
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="." \
    ENVIRONMENT="${ENVIRONMENT}" \
    RUFF_CACHE_DIR=/tmp/ruff-cache

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
COPY pyproject.toml ./
# Required by the compose `db-migrate` one-shot, which runs `python -m
# src.db_migrate` out of THIS image so the runner and the schema it applies can
# never be different versions.
COPY migrations/ ./migrations/

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
    ENVIRONMENT="${ENVIRONMENT}" \
    GIT_SHA="${GIT_SHA}"

CMD ["python", "-m", "src.main"]
