#!/usr/bin/env bash
# Deploy to the dev cluster (Docker Desktop): local test gate + local image.
# Peer: build_k8s_prod.sh. Shared logic: k8s_common.sh.
# Ops: deploy/k8s/README.md; teardown: ./k8s_down.sh dev.
#
#   ./build_k8s_dev.sh
#   ./build_k8s_dev.sh --rotate-secrets   overwrite the cluster's Secrets from .env
#
# No registry round-trip: Docker Desktop's cluster reads the local daemon's
# image store, so the image never leaves this machine.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./build_common.sh   # test gate + image build (shared with build_docker.sh)
source ./k8s_common.sh     # cluster guards + deploy (shared with build_k8s_prod.sh)

EXPECTED_CONTEXT="${K8S_CONTEXT:-docker-desktop}"
OVERLAY_NAME="dev"
TARGET_LABEL="dev"

ROTATE_SECRETS=0
for arg in "$@"; do
  case "$arg" in
    --rotate-secrets) ROTATE_SECRETS=1 ;;
    *) echo "usage: $0 [--rotate-secrets]" >&2; exit 64 ;;
  esac
done

k8s_guard

# ── Secrets: create from .env if absent ───────────────────────────────────────
# Secrets are deliberately NOT in the kustomization (they'd be committed), so a
# re-provisioned cluster has manifests but no credentials and the pods sit in
# CreateContainerConfigError — which reads like a manifest bug and isn't. The
# dev cluster is disposable (a Docker Desktop restart wipes it), so bootstrap
# it here rather than by hand.
#
# Create-if-missing, NOT upsert: once the separate dev Discord application
# lands, this Secret is meant to diverge from .env (deploy/k8s/README.md) and a
# routine deploy must not silently drag it back to the shared token.
# --rotate-secrets is for when overwriting IS the intent.
k8s_ensure_namespace
have_bot=0; k8s_secret_exists discord-music-bot-secrets && have_bot=1
have_grafana=0; k8s_secret_exists grafana-admin && have_grafana=1

if [ "$have_bot" = "0" ] || [ "$have_grafana" = "0" ] || [ "$ROTATE_SECRETS" = "1" ]; then
    if [ ! -f .env ]; then
        echo "ERROR: .env not found — needed to bootstrap the dev cluster's Secrets." >&2
        exit 1
    fi

    # .env is NEVER sourced. It is read as data: the raw `KEY=value` line is
    # handed to kubectl's --from-env-file parser, which (verified against this
    # kubectl and this docker) agrees byte-for-byte with the `env_file:` parser
    # compose uses on the same file. That agreement is the point — both
    # pipelines must derive the same secret from the same line.
    #
    # `set -a; . ./.env` did not: the shell is a THIRD parser and a live one.
    # It strips quotes ("a b" → a b), expands $VARS, and executes `backticks` —
    # so any token containing $, ", or ` would silently reach k8s different
    # from the one compose uses, or run code. It also clobbered ENVIRONMENT
    # (.env sets it), which needed a subshell to contain; reading data instead
    # removes the whole class, subshell included.
    #
    # tail -1: on a duplicate key, shell and docker both take last-wins, so we
    # do too. (kubectl itself hard-errors on dupes, hence one line per key.)
    env_line() { grep -E "^$1=" .env | tail -1; }

    missing=""
    for var in DISCORD_TOKEN SPOTIFY_CLIENT_ID SPOTIFY_CLIENT_SECRET; do
        line="$(env_line "$var")"
        # Non-empty line AND non-empty value after the '='.
        { [ -n "$line" ] && [ -n "${line#*=}" ]; } || missing="$missing $var"
    done
    if [ -n "$missing" ]; then
        echo "ERROR: .env is missing (or has empty):$missing" >&2
        exit 1
    fi

    # Piped, never --from-literal: a literal puts the Discord token in this
    # process's argv, i.e. in `ps` output for every user on the box.
    if [ "$have_bot" = "0" ] || [ "$ROTATE_SECRETS" = "1" ]; then
        echo "Bootstrapping Secret discord-music-bot-secrets from .env"
        {
            env_line DISCORD_TOKEN
            env_line SPOTIFY_CLIENT_ID
            env_line SPOTIFY_CLIENT_SECRET
        } | kubectl create secret generic discord-music-bot-secrets \
                --from-env-file=/dev/stdin --dry-run=client -o yaml \
          | $KUBECTL apply -f - >/dev/null
    fi

    if [ "$have_grafana" = "0" ] || [ "$ROTATE_SECRETS" = "1" ]; then
        echo "Bootstrapping Secret grafana-admin"
        # Optional in .env; dev default is the same inline 'admin' compose uses.
        gf_line="$(env_line GF_SECURITY_ADMIN_PASSWORD)"
        [ -n "$gf_line" ] || gf_line="GF_SECURITY_ADMIN_PASSWORD=admin"
        printf '%s\n' "$gf_line" \
          | kubectl create secret generic grafana-admin \
                --from-env-file=/dev/stdin --dry-run=client -o yaml \
          | $KUBECTL apply -f - >/dev/null
    fi
fi

# ── Test gate + image, shared with build_docker.sh ───────────────────────────
# NOTE: in-cluster, the ConfigMap's ENVIRONMENT always wins over the image's
# baked build-arg — this resolution matters for the test-stage run (and keeps
# parity with build_docker.sh), not for the deployed pod.
resolve_environment
run_test_gate

# Docker Desktop's cluster shares the daemon's image store: no registry push.
# Only the SHA tag — `latest` is compose's contract, and a k8s deploy must never
# be able to resolve to a floating tag.
#
# Why the tag carries a worktree hash: kubectl diffs the pod spec's image tag as
# a STRING, not the image bytes. Rebuilding the same tag from edited source
# leaves the Deployment "unchanged" — nothing rolls out, the pod keeps running
# the old code, and the script still prints "successfully rolled out". (compose
# is immune: it compares resolved image IDs and recreates the container, so
# build_docker.sh needs no equivalent.) Hashing the diff keeps re-runs a genuine
# no-op: the same edit hashes the same, so only a real change mints a new tag
# and triggers the Recreate rollout. A clean tree tags the bare SHA, exactly as
# prod does.
GIT_SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    # Tracked edits (git diff) plus untracked, non-ignored files by name AND
    # content — .env and friends are gitignored, so --exclude-standard drops
    # them. git hash-object: no dependency on shasum/sha1sum portability.
    WORKTREE_HASH="$(
        {
            git diff HEAD
            git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
                printf '%s\n' "$f"
                cat "$f" 2>/dev/null
            done
        } | git hash-object --stdin | cut -c1-8
    )"
    IMAGE="discord-music-bot:$GIT_SHA-dirty.$WORKTREE_HASH"
    echo "NOTE: worktree is dirty — deploying $IMAGE (not a committed SHA)." >&2
else
    IMAGE="discord-music-bot:$GIT_SHA"
fi
echo "Building runtime image $IMAGE"
build_runtime_image "$IMAGE"

# ── Coexistence warning (one token, one live bot process per machine) ─────────
# Advisory, not a guard: it warns and deploys anyway. Deliberate — the operator
# may be mid-migration, and only they know. Peer of the k8s check in
# build_docker.sh, pointing the other way.
if docker compose ps --status running discord-music-bot 2>/dev/null | grep -q discord-music-bot; then
    echo "WARNING: compose bot is running — stop it before deploying to k8s:" >&2
    echo "  docker compose stop discord-music-bot" >&2
fi

k8s_deploy_image "$IMAGE"
