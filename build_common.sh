#!/usr/bin/env bash
# Shared image + test-gate logic for build_docker.sh (compose) and
# build_k8s_dev.sh (dev cluster) — sourced, never run.
#
# The two pipelines promise "same image, same test gate" (deploy/k8s/README.md);
# they diverge only at the "run it" step. That promise is what this file makes
# structural: two copies of the gate is two chances for one pipeline to ship on
# a stale one, and the copy that drifts is the one you notice last.
#
# Cluster-side helpers (guards, deploy) live in k8s_common.sh — this file knows
# nothing about Kubernetes, which is why build_docker.sh can source it.
#
# Contract: callers source this, then call resolve_environment before anything
# that reads $ENVIRONMENT.

# Sourced-only: running this directly would silently do nothing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "$0 is a library — run ./build_docker.sh or ./build_k8s_dev.sh." >&2
    exit 64
fi

TEST_IMAGE="discord-music-bot:test"

# ENVIRONMENT: explicit env var wins, else derive from the branch. Exported for
# the docker build --build-arg and for anything downstream that reads it.
resolve_environment() {
    if [ -z "${ENVIRONMENT:-}" ]; then
        local branch
        branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "development")
        [ "$branch" = "HEAD" ] && branch="development"
        [ "$branch" = "main" ] && ENVIRONMENT="production" || ENVIRONMENT="$branch"
    fi
    export ENVIRONMENT
}

# The gate both pipelines must pass before anything is deployed anywhere.
# One image carries both black and pytest. Both runs bind-mount src/ and tests/
# over the image's baked-in copy, so pytest sees exactly what black just wrote.
run_test_gate() {
    echo "Building test image"
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "$TEST_IMAGE" --target test -f Dockerfile .

    # --user: black writes to the bind-mounted source, so the files must come
    # back owned by the caller, not root.
    echo "Formatting src/ and tests/ with black"
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -v "$PWD/src:/app/src" \
        -v "$PWD/tests:/app/tests" \
        "$TEST_IMAGE" \
        black src/ tests/

    echo "Running tests"
    docker run --rm \
        -v "$PWD/src:/app/src" \
        -v "$PWD/tests:/app/tests" \
        "$TEST_IMAGE"
}

# build_runtime_image <tag> [extra tags...] — the runtime image both pipelines
# deploy. Kept here so the --build-arg can never differ from the gate's.
build_runtime_image() {
    local tag_args=()
    local tag
    for tag in "$@"; do
        tag_args+=(-t "$tag")
    done
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" "${tag_args[@]}" -f Dockerfile .
}
