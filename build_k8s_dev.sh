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
    # Subshell, deliberately: .env also defines ENVIRONMENT, and sourcing it
    # into this shell would clobber the branch-derived value resolve_environment
    # computes below — baking "development" into the image on exactly those runs
    # that happened to bootstrap Secrets, and differing from what
    # build_docker.sh bakes from the same commit. Nothing from .env escapes here.
    (
        set -a; . ./.env; set +a
        missing=""
        for var in DISCORD_TOKEN SPOTIFY_CLIENT_ID SPOTIFY_CLIENT_SECRET; do
            [ -n "${!var:-}" ] || missing="$missing $var"
        done
        if [ -n "$missing" ]; then
            echo "ERROR: .env is missing (or has empty):$missing" >&2
            exit 1
        fi
        if [ "$have_bot" = "0" ] || [ "$ROTATE_SECRETS" = "1" ]; then
            echo "Bootstrapping Secret discord-music-bot-secrets from .env"
            kubectl create secret generic discord-music-bot-secrets \
                --from-literal=DISCORD_TOKEN="$DISCORD_TOKEN" \
                --from-literal=SPOTIFY_CLIENT_ID="$SPOTIFY_CLIENT_ID" \
                --from-literal=SPOTIFY_CLIENT_SECRET="$SPOTIFY_CLIENT_SECRET" \
                --dry-run=client -o yaml | $KUBECTL apply -f - >/dev/null
        fi
        if [ "$have_grafana" = "0" ] || [ "$ROTATE_SECRETS" = "1" ]; then
            echo "Bootstrapping Secret grafana-admin"
            kubectl create secret generic grafana-admin \
                --from-literal=GF_SECURITY_ADMIN_PASSWORD="${GF_SECURITY_ADMIN_PASSWORD:-admin}" \
                --dry-run=client -o yaml | $KUBECTL apply -f - >/dev/null
        fi
    )
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

# ── Coexistence guard (one token, one live bot process per machine) ───────────
if docker compose ps --status running discord-music-bot 2>/dev/null | grep -q discord-music-bot; then
    echo "WARNING: compose bot is running — stop it before deploying to k8s:" >&2
    echo "  docker compose stop discord-music-bot" >&2
fi

k8s_deploy_image "$IMAGE"
