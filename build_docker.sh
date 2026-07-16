#!/usr/bin/env bash

set -euo pipefail

if [ -z "${ENVIRONMENT:-}" ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "development")
    [ "$BRANCH" = "HEAD" ] && BRANCH="development"
    [ "$BRANCH" = "main" ] && ENVIRONMENT="production" || ENVIRONMENT="$BRANCH"
fi
export ENVIRONMENT

# One image carries both black and pytest. Both runs bind-mount src/ and tests/
# over the image's baked-in copy, so pytest sees exactly what black just wrote.
echo "Building test image"
docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "discord-music-bot:test" --target test -f Dockerfile .

echo "Formatting src/ and tests/ with black"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/src:/app/src" \
    -v "$PWD/tests:/app/tests" \
    "discord-music-bot:test" \
    black src/ tests/

echo "Running tests"
docker run --rm \
    -v "$PWD/src:/app/src" \
    -v "$PWD/tests:/app/tests" \
    "discord-music-bot:test"

export GIT_SHA="$(git rev-parse HEAD)"
BUILD_TAG="discord-music-bot:$GIT_SHA"

echo "Building docker image"
docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "discord-music-bot:latest" -t "$BUILD_TAG" -f Dockerfile .

# Refuse to double-run the bot: one token, one live process (see deploy/k8s/README.md).
# Check BOTH clusters explicitly — the current kubectl context is whichever was used
# last and says nothing about where a bot pod might be running.
if command -v kubectl >/dev/null 2>&1; then
    for CTX in docker-desktop k3s-production; do
        if [ "$(kubectl --context "$CTX" -n discord-music-bot get deploy discord-music-bot \
                -o jsonpath='{.status.availableReplicas}' 2>/dev/null)" = "1" ]; then
            echo "WARNING: bot is running in k8s ($CTX) — two processes on one token means double audio." >&2
            echo "Scale it down first: kubectl --context $CTX -n discord-music-bot scale deploy discord-music-bot --replicas=0" >&2
        fi
    done
fi

echo "Running docker with build tag $BUILD_TAG"
docker compose up -d
