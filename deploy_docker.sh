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
# Turns HISTORY_ARCHIVE_ENABLED into the `archive` compose profile, so the flag
# in .env is the only thing an operator sets and no tracked file is ever edited.
# Before the preflight below, which reports on the resolved state.
resolve_archive_profile
# This script is the one that STARTS containers, so it is the one that has to
# carry the credential preflight. build_docker.sh also calls it, but only
# build_docker.sh — and `just up`, a rollback, and the README's own enable and
# disable procedures (`docker compose up -d`) all reach a running Postgres
# without ever passing through it. That left the drift case — `archive` profile
# active, HISTORY_ARCHIVE_ENABLED off — with no warning anywhere at all: the
# bot's startup ERROR and its -ping row are both gated on the flag, so a
# default-credential database holding the full play archive ran silently, which
# is precisely the case build_common.sh's wider OR-gate was written to catch.
#
# Deliberately NOT deduplicated against build_docker.sh's earlier call. That one
# fires before ~24s of checks and a full image build so the operator learns
# early; by the time those have scrolled past, this one puts the warning back on
# screen immediately before the containers start. Same text, two moments that
# both matter. It only warns — see build_common.sh for why refusing would be
# wrong — so repeating it costs nothing but a line.
warn_default_postgres_password

# Default matches what build_docker.sh / `just image` actually tagged, `-dirty`
# suffix included — otherwise `just up` after a dirty build looks for a clean-SHA
# tag that was never created and the guard below rejects it.
#
# Written as an if rather than "${1:-$(git_sha_tag)}": a command substitution inside
# a ${:-} default does not set the assignment's exit status, so `set -e` did not fire
# when git failed and GIT_SHA silently became "". The guard below then reported "No
# local image discord-music-bot:" — a confusing symptom for "this is not a git
# repository". build_docker.sh already splits its own assignment for this reason.
# `-n "${1:-}"` and not just `$# -ge 1`: `just up` passes its empty TAG default as a
# quoted argument, so this is called with one EMPTY argument in the common case.
# Counting arguments alone would take that empty string as the requested tag and try
# to deploy `discord-music-bot:`.
if [ -n "${1:-}" ]; then
    GIT_SHA="$1"
else
    GIT_SHA="$(git_sha_tag)"
fi
export GIT_SHA
TAG="$IMAGE_NAME:$GIT_SHA"

# docker-compose.yml pins `image: discord-music-bot:${GIT_SHA:-latest}` but also
# carries a `build:` section — so `compose up` will happily BUILD a missing tag
# from the current working tree and then label it with the SHA you asked for.
# On a rollback that is silently, dangerously wrong: you get today's source
# wearing last week's tag, and the image store now lies about it. Refuse instead.
# Probed first so the guard below can only mean one thing. `docker image inspect`
# fails identically when the daemon is unreachable, and the guard then told you to
# build an image you already had — while its "here are the tags that exist" hint came
# back empty for the same reason, muted by the `|| true`.
if ! docker info >/dev/null 2>&1; then
    echo "Cannot reach the Docker daemon — is Docker running?" >&2
    exit 1
fi

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

# NOTE (merge-time): when the k8s
# stack lands, its "is a bot pod already live in a cluster?" guard belongs HERE,
# immediately below — one Discord token means one live process, and this is the
# line that starts one. It currently sits in that branch's build_docker.sh, which
# is resolved in main's favour, so it must be carried over by hand or it is lost.
# Putting it here rather than in build_docker.sh also covers rollbacks, which the
# branch's version does not.

# Deactivating a profile does not stop what it already started: compose leaves a
# running postgres alone and `down` without the profile cannot see it either. So
# say so — the operator who just turned the archive off expects the database to
# go away, and silence here looks like it did.
if [ "$ARCHIVE_ENABLED" -eq 0 ] \
    && [ -n "$(docker ps -q --filter 'name=^discord-postgres$' 2>/dev/null)" ]; then
    echo "WARNING: the archive is disabled but discord-postgres is still running" >&2
    echo "         from an earlier enabled deploy. This deploy leaves it up." >&2
    echo "         Stop it with: just down   (then re-run this deploy)" >&2
    echo "         Archived rows survive in the postgres-data volume either way." >&2
fi

# Migrate BEFORE the bot is recreated, and gate the deploy on it: a `git pull`
# brings migrations with the code, and the bot refuses to archive against a
# schema older than its build, so starting it first archives nothing until
# someone reads the logs. Idempotent — versions are recorded with their DDL and
# skipped thereafter, and a database newer than this image exits 0 (rollbacks).
#
# `run --rm`, not `up`: only `run` reports the runner's exit status, which is
# what makes this a gate. depends_on waits for postgres to be healthy first.
if [ "$ARCHIVE_ENABLED" -eq 1 ]; then
    echo "Applying play-history migrations from $TAG"
    # Empty unless POSTGRES_URL names a database off this host — the compose
    # service's own DSN addresses postgres by service name, and .env's is the
    # host form, which is unreachable from inside the container.
    resolve_external_postgres_env
    if ! docker compose run --rm -T ${EXTERNAL_PG_ENV[@]+"${EXTERNAL_PG_ENV[@]}"} db-migrate; then
        echo "Migration failed — NOT deploying $TAG." >&2
        echo "The running bot is untouched. Fix the database, then re-run this." >&2
        echo "Inspect with: just compose logs postgres" >&2
        exit 1
    fi
fi

echo "Deploying $TAG (ENVIRONMENT=$ENVIRONMENT)"
# Only the bot's own container is recreated — Redis, the POT sidecar and
# otel-lgtm are unchanged by a new bot tag, so compose leaves them running.
# With the archive enabled this also starts postgres and the db-migrate service;
# the migration above has already run, so that one is a no-op second pass. It is
# left in the model deliberately, for operators who drive compose directly.
docker compose up -d
