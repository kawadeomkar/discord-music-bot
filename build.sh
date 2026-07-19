#!/usr/bin/env bash

set -euo pipefail

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
