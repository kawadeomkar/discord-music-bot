#!/usr/bin/env bash
# Docker Compose pipeline — peer of build_k8s_dev.sh (dev cluster).
# Shared test gate + image build: build_common.sh.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./build_common.sh

resolve_environment
run_test_gate

export GIT_SHA="$(git rev-parse HEAD)"
BUILD_TAG="discord-music-bot:$GIT_SHA"

# `latest` too: docker-compose.yml pins it, so compose runs what was just built.
echo "Building docker image"
build_runtime_image "discord-music-bot:latest" "$BUILD_TAG"

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
