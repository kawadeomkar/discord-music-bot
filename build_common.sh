#!/usr/bin/env bash
# Shared environment + image logic for build_docker.sh and deploy_docker.sh —
# sourced, never run.
#
# ── Fixed API ────────────────────────────────────────────────────────────────
# resolve_environment / run_test_gate / build_runtime_image are a CONTRACT, not
# an implementation detail. build_k8s_dev.sh (written and dev-validated on
# task/k8s-deployment-3-manifests, unmerged) sources this file and calls exactly
# those three names, with build_runtime_image variadic — it passes one SHA tag
# where the compose path passes two. Renaming or re-signaturing one of them turns
# that branch's merge into a rewrite under conflict markers, months from now.
# Rationale: docs/CICD_PIPELINE_RESTRUCTURE_PLAN.md §4.1.
#
# git_sha_tag is ADDITIVE — new callers may use it, the three names above keep
# working untouched, so the k8s merge is unaffected by its existence.
#
# Cluster-side helpers live in k8s_common.sh (also unmerged). This file knows
# nothing about Kubernetes, which is exactly why build_docker.sh can source it.
#
# Contract for callers: source this, then call resolve_environment before
# anything that reads $ENVIRONMENT.

# Sourced-only: running this directly would silently do nothing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "$0 is a library — run ./build_docker.sh or ./deploy_docker.sh." >&2
    exit 64
fi

IMAGE_NAME="discord-music-bot"

# The tag that identifies what is actually in the image.
#
# deploy_docker.sh refuses to deploy a SHA tag it cannot find, precisely so a
# rollback can never get "today's source wearing last week's tag". Building had
# the same hole from the other side: `docker build -t $IMAGE:$(git rev-parse HEAD)`
# with uncommitted changes stamps HEAD's SHA onto bytes that are not HEAD, and the
# deploy guard then waves it through because the tag does exist.
#
# The fix is honesty rather than a gate. Refusing to build a dirty tree would be a
# gate people route around (iterating and deploying from the same working tree is
# the normal case here); a `-dirty` suffix instead makes the tag true, keeps the
# deploy guard meaningful, and leaves clean-SHA rollbacks untouched.
# `git diff --quiet HEAD` covers staged AND unstaged changes to TRACKED files —
# the case that actually matters, since those are the bytes the Dockerfile copies.
# A brand-new untracked file is not flagged; it would be over-flagging, as scratch
# files sit in the tree constantly and .dockerignore keeps most of them out.
git_sha_tag() {
    local sha
    sha=$(git rev-parse HEAD)
    if ! git diff --quiet HEAD 2>/dev/null; then
        sha="$sha-dirty"
    fi
    echo "$sha"
}

# ENVIRONMENT: an explicit env var wins, else derive it from the branch.
# Exported for `docker build --build-arg` and for docker-compose.yml.
resolve_environment() {
    if [ -z "${ENVIRONMENT:-}" ]; then
        local branch
        branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "development")
        [ "$branch" = "HEAD" ] && branch="development"
        [ "$branch" = "main" ] && ENVIRONMENT="production" || ENVIRONMENT="$branch"
    fi
    export ENVIRONMENT
}

# The gate every deploy passes, delegated to `make check` so there is exactly ONE
# definition of "will CI pass".
#
# The old build.sh kept its own copy of the gate and ran it in a container over
# bind-mounted source. Two things went wrong with that, and both are why this is
# now one line: the copy drifted (build.sh ran `ruff format`, which rewrites,
# where CI runs `ruff format --check`, which fails — and build.sh never ran
# pyright at all), and wrapping 0.13s of ruff in an image build plus two
# container starts made the fast checks slow enough to skip.
#
# The image is still tested end-to-end — by `make container-test`, mirroring CI's
# container-test job. That is a different question (does the IMAGE run?) and it
# belongs in `make ci`, not in front of every deploy.
run_test_gate() {
    echo "Running gate: make check"
    make check
}

# build_runtime_image <tag> [extra tags...] — the runtime image every pipeline
# deploys. Lives here so the --build-arg can never differ between the compose
# path and the k8s path.
build_runtime_image() {
    local tag_args=()
    local tag
    for tag in "$@"; do
        tag_args+=(-t "$tag")
    done
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" "${tag_args[@]}" --target runtime -f Dockerfile .
}
