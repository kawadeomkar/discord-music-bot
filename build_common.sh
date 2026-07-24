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
#
# ${BASH_SOURCE[0]:-} with the default, not a bare ${BASH_SOURCE[0]}: the array does
# not exist in every shell, and under `set -u` referencing it dies right here — on the
# line whose whole job is to produce a friendly message. The contract above invites
# arbitrary callers to source this file, so it must not assume they are bash.
if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    echo "$0 is a library — run ./build_docker.sh or ./deploy_docker.sh." >&2
    exit 64
fi

# The image name, and the one definition of it. The justfile carries its own `IMAGE`
# because just variables cannot be sourced from shell; `just pins` asserts the two
# agree, so a rename here fails the gate rather than silently splitting the tag space.
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
# the normal case here); a `-dirty.<digest>` suffix instead makes the tag true,
# keeps the deploy guard meaningful, and leaves clean-SHA rollbacks untouched.
#
# The digest is a real git tree hash of what the Dockerfile would copy, computed in
# a THROWAWAY index so the caller's staging area is never touched. Two properties
# matter here, and the earlier bare `-dirty` marker had neither:
#
#   1. Untracked files count. `git diff --quiet HEAD` does not see them, but
#      `COPY src/ ./src/` does, and .dockerignore excludes nothing under src/. A
#      new untracked module therefore produced a CLEAN sha tag on an image that
#      contained it — precisely the "source wearing someone else's tag" failure
#      this function exists to prevent, arrived at from the untracked side.
#   2. The suffix is content-addressed. A bare `-dirty` collapses every distinct
#      working tree at a commit onto ONE tag, so deploy_docker.sh's guard ("refuse
#      a tag I cannot find") could never fire for a dirty build: after the first
#      one that tag always exists. Worse, rebuilding after an edit reproduced the
#      same tag, and compose — seeing no change in image or config — would not even
#      recreate the container, reporting a deploy that changed nothing.
#
# `git add -A` honours .gitignore, so .venv/, docs/ and other local-only trees stay
# out of the digest. It also records deletions, so removing a tracked file moves the
# tag too.
# Every step is `|| return 1` rather than left to `set -e`, and that is not belt and
# braces — set -e does NOT reach into this function from its main caller. Callers use
# it as `TAG="$(git_sha_tag)"`, and in a command substitution a failing step here is
# followed by the trailing `echo`, which succeeds and becomes the function's status.
# Verified: with a bare `sha=$(git rev-parse HEAD)` the caller sails past a "not a git
# repository" error with TAG="" and builds `discord-music-bot:`. Explicit returns are
# the only thing that propagates.
git_sha_tag() {
    # Not `status`: that is a read-only parameter in zsh (an alias for $?), so the
    # name alone made this function fail for anyone sourcing the file from a zsh
    # shell — which the header invites, since the contract is "callers source this".
    local sha wt_status index tree
    sha=$(git rev-parse HEAD) || return 1

    # --porcelain covers staged, unstaged AND untracked; `git diff HEAD` does not.
    wt_status=$(git status --porcelain) || return 1

    if [ -n "$wt_status" ]; then
        index="$(mktemp "${TMPDIR:-/tmp}/dmb-index.XXXXXX")" || return 1
        # mktemp leaves an empty file behind and an empty file is not a valid git
        # index — git has to create its own here.
        rm -f "$index"
        GIT_INDEX_FILE="$index" git read-tree HEAD || { rm -f "$index"; return 1; }
        GIT_INDEX_FILE="$index" git add -A || { rm -f "$index"; return 1; }
        tree=$(GIT_INDEX_FILE="$index" git write-tree) || { rm -f "$index"; return 1; }
        rm -f "$index"
        sha="$sha-dirty.${tree:0:8}"
    fi

    echo "$sha"
}

# Identifies the dependency set baked into the test image.
#
# The image is rebuilt when ABSENT, which is right for source changes (src/ and tests/
# are bind-mounted) but was wrong for dependency changes: pyproject.toml is mounted
# read-only into a container whose venv was installed at build time, so after editing
# poetry.lock, `DOCKER=1 just check` ran ruff/pyright/pytest against the NEW config and
# the OLD packages and reported green. build_docker.sh then built the runtime image
# with the new dependencies, gated by a check that never saw them. The old build.sh
# rebuilt unconditionally every run, so this could not happen there.
#
# Stamped as a label by `just test-image-rebuild` and compared by `_tools`.
# Explicit propagation, for the reason spelled out above git_sha_tag.
dep_hash() {
    local root
    root=$(git rev-parse --show-toplevel) || return 1
    cat "$root/poetry.lock" "$root/pyproject.toml" | git hash-object --stdin
}

# ENVIRONMENT: an explicit env var wins, else derive it from the branch.
# Exported for `docker build --build-arg` and for docker-compose.yml.
#
# The resolved value is echoed because an ambient ENVIRONMENT wins SILENTLY and
# shells commonly export one — this machine exports ENVIRONMENT=development from its
# login profile, which makes the branch derivation below dead code and stamps a build
# from main as `development`. deploy_docker.sh echoes it, but build_docker.sh and
# `just image` did not, so the value reached `docker build --build-arg` unseen.
resolve_environment() {
    if [ -z "${ENVIRONMENT:-}" ]; then
        local branch
        branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "development")
        [ "$branch" = "HEAD" ] && branch="development"
        # if/else rather than `a && b || c` (SC2015): that idiom silently takes the
        # `||` branch if the middle command ever fails, and "assign a string" only
        # looks infallible until someone edits it.
        if [ "$branch" = "main" ]; then
            ENVIRONMENT="production"
        else
            ENVIRONMENT="$branch"
        fi
        echo "ENVIRONMENT=$ENVIRONMENT (derived from branch '$branch')" >&2
    else
        echo "ENVIRONMENT=$ENVIRONMENT (inherited from the environment)" >&2
    fi
    export ENVIRONMENT
}

# The gate every deploy passes, delegated to `just check` so there is exactly ONE
# definition of "will CI pass".
#
# The old build.sh kept its own copy of the gate and ran it in a container over
# bind-mounted source. Two things went wrong with that, and both are why this is
# now one line: the copy drifted (build.sh ran `ruff format`, which rewrites,
# where CI runs `ruff format --check`, which fails — and build.sh never ran
# pyright at all), and wrapping 0.13s of ruff in an image build plus two
# container starts made the fast checks slow enough to skip.
#
# The image is still tested end-to-end — by `just container-test`, mirroring CI's
# container-test job. That is a different question (does the IMAGE run?) and it
# belongs in `just ci`, not in front of every deploy.
run_test_gate() {
    echo "Running gate: just check"
    just check
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
    # ${a[@]+"${a[@]}"} rather than a bare "${tag_args[@]}". Expanding an EMPTY array
    # under `set -u` is an error before bash 4.4, and macOS still ships bash 3.2 —
    # verified: 3.2 aborts with "tag_args[@]: unbound variable", 5.3 is fine. No
    # current caller passes zero tags, but the header advertises this as variadic for
    # the unmerged k8s branch, so a zero-arg call is a supported shape.
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" \
        ${tag_args[@]+"${tag_args[@]}"} --target runtime -f Dockerfile .
}

# require_postgres_password — compose-path preflight. NOT part of the fixed API
# the k8s build contract depends on (that path gets its secret from a Secret, not
# .env), so it is called only from build_docker.sh, never build_common's callers.
#
# compose interpolates POSTGRES_PASSWORD from .env into both the postgres service
# and the bot's POSTGRES_URL (docker-compose.yml). Its `:?` guard only fails at
# `docker compose up` — after the whole image build. Fail here instead so we
# don't build an image just to trip over a missing secret. (Ported from the
# preflight the pre-restructure build.sh carried; see task/async-pg-impl 2827fcd.)
require_postgres_password() {
    local env_file=".env" pg_password
    if [ ! -f "$env_file" ]; then
        echo "Error: $env_file not found — copy .env.example and fill it in." >&2
        exit 1
    fi
    # Last uncommented assignment wins (compose semantics); strip an optional
    # `export`, surrounding quotes, and trailing whitespace, then require non-empty.
    pg_password=$(
        grep -E '^[[:space:]]*(export[[:space:]]+)?POSTGRES_PASSWORD=' "$env_file" \
            | tail -n1 \
            | sed -E 's/^[[:space:]]*(export[[:space:]]+)?POSTGRES_PASSWORD=//; s/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/' \
            || true
    )
    if [ -z "$pg_password" ]; then
        echo "Error: POSTGRES_PASSWORD is not set in $env_file." >&2
        echo "       Set it (see .env.example) — the postgres service and the bot's" >&2
        echo "       POSTGRES_URL are both built from it." >&2
        exit 1
    fi
}
