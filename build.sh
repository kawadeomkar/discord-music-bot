#!/usr/bin/env bash

set -euo pipefail

# Preflight: compose interpolates POSTGRES_PASSWORD from .env into both the
# postgres service and the bot's POSTGRES_URL (docker-compose.yml). Its `:?`
# guard would only fail at `docker compose up` — after the whole build. Fail
# here instead so we don't build an image just to trip over a missing secret.
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found — copy .env.example and fill it in." >&2
    exit 1
fi
# Take the last uncommented assignment (compose uses last-wins), strip an
# optional `export`, surrounding quotes, and trailing whitespace, then require
# a non-empty result.
pg_password=$(
    grep -E '^[[:space:]]*(export[[:space:]]+)?POSTGRES_PASSWORD=' "$ENV_FILE" \
        | tail -n1 \
        | sed -E 's/^[[:space:]]*(export[[:space:]]+)?POSTGRES_PASSWORD=//; s/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/' \
        || true
)
if [ -z "$pg_password" ]; then
    echo "Error: POSTGRES_PASSWORD is not set in $ENV_FILE." >&2
    echo "       Set it (see .env.example) — the postgres service and the bot's" >&2
    echo "       POSTGRES_URL are both built from it." >&2
    exit 1
fi

if [ -z "${ENVIRONMENT:-}" ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "development")
    [ "$BRANCH" = "HEAD" ] && BRANCH="development"
    [ "$BRANCH" = "main" ] && ENVIRONMENT="production" || ENVIRONMENT="$BRANCH"
fi
export ENVIRONMENT

# One image carries both ruff and pytest. Both runs bind-mount src/ and tests/
# over the image's baked-in copy, so pytest sees exactly what ruff just wrote.
echo "Building test image"
docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "discord-music-bot:test" --target test -f Dockerfile .

echo "Formatting and linting src/ and tests/ with ruff"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/src:/app/src" \
    -v "$PWD/tests:/app/tests" \
    "discord-music-bot:test" \
    sh -c "python -m compileall src/ && ruff format src/ tests/ && ruff check src/ tests/"

echo "Running tests"
docker run --rm \
    -v "$PWD/src:/app/src" \
    -v "$PWD/tests:/app/tests" \
    "discord-music-bot:test"

export GIT_SHA="$(git rev-parse HEAD)"
BUILD_TAG="discord-music-bot:$GIT_SHA"

echo "Building docker image"
docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "discord-music-bot:latest" -t "$BUILD_TAG" -f Dockerfile .

echo "Running docker with build tag $BUILD_TAG"
docker compose up -d
