#!/usr/bin/env bash

set -euo pipefail

poetry install --only=main,lint --no-root
poetry run python -m black src/ tests/ --target-version py313

export GIT_SHA="$(git rev-parse HEAD)"
BUILD_TAG="discord-music-bot:$GIT_SHA"

echo "Building docker image"
docker build -t "discord-music-bot:latest" -t "$BUILD_TAG" -f Dockerfile .

echo "Running docker with build tag $BUILD_TAG"
docker compose up
