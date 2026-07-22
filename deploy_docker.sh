#!/usr/bin/env bash
# Deploy an ALREADY-BUILT image to the compose stack. Never builds, never tests.
#
# That restraint is the whole point (build once, deploy many): it makes rollback
# a five-second operation instead of a rebuild, and it means the bytes you deploy
# are the bytes that passed the gate — not a fresh build that merely came from
# the same commit.
#
#   ./deploy_docker.sh              # deploy the image built for HEAD
#   ./deploy_docker.sh <git-sha>    # roll back (or forward) to any built image
#
# Building + gating + deploying in one step is ./build_docker.sh.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./build_common.sh

resolve_environment

# Default matches what build_docker.sh / `just image` actually tagged, `-dirty`
# suffix included — otherwise `just up` after a dirty build looks for a clean-SHA
# tag that was never created and the guard below rejects it.
GIT_SHA="${1:-$(git_sha_tag)}"
export GIT_SHA
TAG="$IMAGE_NAME:$GIT_SHA"

# docker-compose.yml pins `image: discord-music-bot:${GIT_SHA:-latest}` but also
# carries a `build:` section — so `compose up` will happily BUILD a missing tag
# from the current working tree and then label it with the SHA you asked for.
# On a rollback that is silently, dangerously wrong: you get today's source
# wearing last week's tag, and the image store now lies about it. Refuse instead.
if ! docker image inspect "$TAG" >/dev/null 2>&1; then
    echo "No local image $TAG — refusing to let compose build one and label it with that SHA." >&2
    echo "Build the current commit with ./build_docker.sh, or pick a tag that exists:" >&2
    # `|| true` guards a SIGPIPE race. `head` closes the pipe after 20 lines; if
    # docker is still writing at that point it dies of SIGPIPE (141), and under
    # `set -o pipefail` that becomes the script's status — aborting HERE, before
    # the `exit 1` below, so the user would see a bare 141 instead of this
    # message. It needs output larger than the ~64KB pipe buffer to fire (docker
    # exits cleanly below that, measured), i.e. on the order of a thousand tags —
    # latent, not imminent, but free to rule out.
    docker images "$IMAGE_NAME" --format '  {{.Tag}}\t{{.CreatedSince}}' | head -20 >&2 || true
    exit 1
fi

# NOTE (merge-time, docs/CICD_PIPELINE_RESTRUCTURE_PLAN.md §8.3): when the k8s
# stack lands, its "is a bot pod already live in a cluster?" guard belongs HERE,
# immediately below — one Discord token means one live process, and this is the
# line that starts one. It currently sits in that branch's build_docker.sh, which
# is resolved in main's favour, so it must be carried over by hand or it is lost.
# Putting it here rather than in build_docker.sh also covers rollbacks, which the
# branch's version does not.

echo "Deploying $TAG (ENVIRONMENT=$ENVIRONMENT)"
# Only the bot's own container is recreated — Redis, the POT sidecar and
# otel-lgtm are unchanged by a new bot tag, so compose leaves them running.
docker compose up -d
