#!/usr/bin/env bash

set -aeuxETo pipefail

GIT_SHA=$(git rev-parse HEAD)
export GIT_SHA=$GIT_SHA

BUILD_TAG="discord-music-bot:$GIT_SHA"

echo "Building docker image"
docker build --rm  -t "discord-music-bot:latest" -t "$BUILD_TAG" -f Dockerfile .
