#!/usr/bin/env bash
# Compose pipeline: gate → image → deploy.
#
# This is a composition, not a fourth implementation. The gate is `just check`,
# the image build is build_common.sh, the deploy is deploy_docker.sh. Nothing
# here reimplements any of them — keeping its own copy of the gate is exactly how
# the old build.sh drifted out of sync with CI while still reporting success.
#
# Need only one step? `just check`, `just image`, `./deploy_docker.sh`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./build_common.sh

resolve_environment
# Resolved here too, not only in deploy_docker.sh: a garbage
# HISTORY_ARCHIVE_ENABLED must fail before ~24s of checks and a full image
# build, not after. The exported list survives the exec below and the deploy leg
# re-resolves it anyway — the rebuild is idempotent, `archive` is stripped and
# re-added rather than appended.
resolve_archive_profile
# Warn before the test gate + image build, so an operator on the default
# credential sees it now rather than after ~24s of checks and a full image
# build. It only WARNS: compose has a fallback for the password, so refusing to
# build would put back the first-run cliff that default exists to remove.
warn_default_postgres_password
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
