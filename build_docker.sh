#!/usr/bin/env bash
# Compose pipeline: gate → image → deploy.
#
# This is a composition, not a fourth implementation. The gate is `make check`,
# the image build is build_common.sh, the deploy is deploy_docker.sh. Nothing
# here reimplements any of them — keeping its own copy of the gate is exactly how
# the old build.sh drifted out of sync with CI while still reporting success.
#
# Need only one step? `make check`, `make image`, `./deploy_docker.sh`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./build_common.sh

resolve_environment
run_test_gate

# Split from the export deliberately: `export FOO="$(cmd)"` returns export's own
# status, so a failing command inside it does NOT trip set -e.
#
# git_sha_tag, not `git rev-parse HEAD`: a build from a dirty tree gets a
# `-dirty` suffix so the tag never claims to be a commit it isn't.
GIT_SHA="$(git_sha_tag)"
export GIT_SHA

# `:latest` too — docker-compose.yml falls back to it when GIT_SHA is unset, so a
# bare `docker compose up` still runs whatever was last built here.
echo "Building runtime image"
build_runtime_image "$IMAGE_NAME:latest" "$IMAGE_NAME:$GIT_SHA"

exec ./deploy_docker.sh "$GIT_SHA"
