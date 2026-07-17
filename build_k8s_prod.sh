#!/usr/bin/env bash
# Deploy to the production cluster (k3s server). Deploy-only: no build, no test
# gate — production runs exactly the bytes CI built from main, never local ones.
# Peer: build_k8s_dev.sh. Shared logic: k8s_common.sh.
# Ops: deploy/k8s/README.md; teardown: ./k8s_down.sh prod.
#
#   ./build_k8s_prod.sh
#
# HEAD must be an ancestor of origin/main and its CI-built GHCR image must
# already exist; both are checked below.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./k8s_common.sh

EXPECTED_CONTEXT="${K8S_CONTEXT:-k3s-production}"
OVERLAY_NAME="production"
TARGET_LABEL="prod"
GHCR_REPO="ghcr.io/kawadeomkar/discord-music-bot"

[ $# -eq 0 ] || { echo "usage: $0" >&2; exit 64; }

k8s_guard

# ── Secrets: verify only, never write ────────────────────────────────────────
# Production credentials are managed out-of-band and .env holds the dev ones,
# so bootstrapping from it would overwrite a rotated prod secret with a stale
# local value, or push the dev token to production. Fail loud instead — the
# alternative is pods stuck in CreateContainerConfigError, which reads like a
# manifest bug and isn't.
k8s_ensure_namespace
for secret in discord-music-bot-secrets grafana-admin; do
    if ! k8s_secret_exists "$secret"; then
        echo "ERROR: Secret '$secret' is missing on '$EXPECTED_CONTEXT'." >&2
        echo "Bootstrap it by hand — prod creds are never read from .env:" >&2
        echo "  see 'Secret bootstrap' in deploy/k8s/README.md" >&2
        exit 1
    fi
done

# ── Provenance gate: prod runs only main-merged, CI-built images ──────────────
GIT_SHA="$(git rev-parse HEAD)"
git fetch origin main --quiet
if ! git merge-base --is-ancestor "$GIT_SHA" origin/main; then
    echo "ERROR: HEAD ($GIT_SHA) is not on origin/main." >&2
    echo "Production deploys only main-merged, CI-built images." >&2
    exit 1
fi

IMAGE="$GHCR_REPO:sha-$GIT_SHA"
if ! docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: $IMAGE not found in GHCR." >&2
    echo "Has the Build workflow finished for this SHA? (gh run list --workflow Build)" >&2
    exit 1
fi

k8s_deploy_image "$IMAGE"
