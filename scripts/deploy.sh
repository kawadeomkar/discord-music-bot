#!/usr/bin/env bash
# One-command deploy for running the bot WITHOUT `just` installed: pull → build → deploy.
#
# This is the no-`just` twin of `just deploy`, and that recipe now calls THIS script, so
# the two are one code path and cannot drift. It is the only `just` command a regular
# user (as opposed to a contributor) needs, because it covers all three "I just want the
# containers running" cases:
#
#   1. First run after cloning — no image, no containers: it builds the image and
#      `docker compose up -d` creates the whole stack from scratch.
#   2. Refresh a running stack after local changes — rebuilds and recreates the bot
#      container (Redis + the sidecars keep running).
#   3. Behind origin — fast-forwards to upstream first, then rebuilds and redeploys.
#
# NO test gate. The gated build → check → deploy is ./build_docker.sh, which needs the
# Python / test-image toolchain a regular user may not have; they just want the current
# code running. Contributors who want the gate still have `just check` / ./build_docker.sh.

set -euo pipefail
# This file lives in scripts/, so repo root is one level up. cd there so `git`,
# `build_common.sh` and `deploy_docker.sh` all resolve regardless of where it was invoked.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./build_common.sh

# Needed by build_runtime_image's --build-arg (deploy_docker.sh resolves it again itself).
resolve_environment

# 1. Best-effort fast-forward to upstream so we build the latest code.
#    Name the remote+branch explicitly (`<remote> <branch>`) rather than a bare
#    `git pull`: a branch with no upstream tracking ref (never pushed with `-u`) makes a
#    bare pull fail with "no tracking information" instead of fast-forwarding. Pulling
#    `<remote> <current-branch>` needs no tracking config at all. The remote defaults to
#    origin and is overridable via DMB_REMOTE (forks / non-standard remote names).
#    --ff-only never creates a surprise merge commit or opens an editor mid-build; `|| true`
#    keeps it non-fatal, so offline / a diverged branch / a branch absent on the remote / a
#    detached HEAD all fall through to shipping the current tree. git still prints why it
#    could not fast-forward, so a skip is visible.
echo "Pulling latest changes (best-effort)"
remote="${DMB_REMOTE:-origin}"
branch="$(git branch --show-current 2>/dev/null || true)"
if [ -n "$branch" ]; then
    git pull --ff-only "$remote" "$branch" || true
else
    echo "Detached HEAD or no current branch — skipping pull"
fi

# 2. Build :latest and :<git-sha>. Assigned first, not inlined: `export FOO="$(cmd)"`
#    returns export's own status, so a failing git_sha_tag would not trip set -e.
GIT_SHA="$(git_sha_tag)"
export GIT_SHA
echo "Building runtime image"
build_runtime_image "$IMAGE_NAME:latest" "$IMAGE_NAME:$GIT_SHA"

# 3. Deploy exactly the SHA just built. deploy_docker.sh guards that the image exists,
#    refuses to let compose build-and-mislabel one, and recreates only the bot container.
exec ./deploy_docker.sh "$GIT_SHA"
