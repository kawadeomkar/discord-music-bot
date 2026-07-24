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

# Migrations ship with the code: src/db.py applies db/migrations/*.sql on
# first pool use, and the test suite pins the baseline file's presence.
COPY db/ ./db/
COPY src/ ./src/
COPY tests/ ./tests/

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

# Migrations before source: db/ changes rarely, so its layer stays cached.
# Without it the runtime bot would apply zero migrations and every history
# drain would fail (src/db.py resolves db/migrations relative to the repo root).
COPY db/ ./db/

# Copy source last — most frequently changed, should be the last layer.
COPY src/ ./src/
COPY pyproject.toml ./

ARG ENVIRONMENT=production
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="." \
    ENVIRONMENT="${ENVIRONMENT}"

CMD ["python", "-m", "src.main"]
