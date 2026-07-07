#!/usr/bin/env bash

set -euo pipefail

if [ -z "${ENVIRONMENT:-}" ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "development")
    [ "$BRANCH" = "HEAD" ] && BRANCH="development"
    [ "$BRANCH" = "main" ] && ENVIRONMENT="production" || ENVIRONMENT="$BRANCH"
fi
export ENVIRONMENT

poetry install --only=main,lint --no-root
poetry run python -m compileall src/
poetry run ruff format src/ tests/
poetry run ruff check src/ tests/

export GIT_SHA="$(git rev-parse HEAD)"
BUILD_TAG="discord-music-bot:$GIT_SHA"

echo "Building docker image"
docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "discord-music-bot:latest" -t "$BUILD_TAG" -f Dockerfile .

echo "Running docker with build tag $BUILD_TAG"
docker compose up
