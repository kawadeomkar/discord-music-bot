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
# plus lint deps (ruff, pyright). Runs pytest by default; build.sh overrides the
# command to run ruff against a bind-mounted src/ and tests/.
# Used by the container-test CI job. Never pushed to GHCR.
FROM builder AS test

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only=main,test,lint --no-root

# Migrations ship with the code: src/db.py applies db/migrations/*.sql on
# first pool use, and the test suite pins the baseline file's presence.
COPY db/ ./db/
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml ./

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
