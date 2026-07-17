#!/usr/bin/env bash
# Shared logic for build_k8s_dev.sh, build_k8s_prod.sh and k8s_down.sh —
# sourced, never run.
# Ops: deploy/k8s/README.md; design: docs/K8S_DEPLOYMENT_PLAN.md §6.2.
#
# Everything here is target-agnostic on purpose. The build entry scripts differ
# only in how they obtain an image (build it vs. verify CI built it) and in
# their secret policy; the guards and the deploy itself are identical, and both
# are code where a subtle bug is expensive — they must never exist in two
# copies that can drift.
#
# Contract: the caller sets EXPECTED_CONTEXT and TARGET_LABEL, then calls
# k8s_guard, which exports $KUBECTL for everything after it. k8s_deploy_image
# additionally needs OVERLAY_NAME; teardown callers use neither it nor
# k8s_ensure_namespace — an unused function is the point of a library.

# Sourced-only: running this directly would silently do nothing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "$0 is a library — run ./build_k8s_dev.sh, ./build_k8s_prod.sh or ./k8s_down.sh." >&2
    exit 64
fi

NAMESPACE="discord-music-bot"

# ── Guards: right cluster, and one that actually answers ──────────────────────
k8s_guard() {
    local current
    current="$(kubectl config current-context 2>/dev/null || echo '<none>')"
    if [ "$current" != "$EXPECTED_CONTEXT" ]; then
        echo "ERROR: kubectl context is '$current', expected '$EXPECTED_CONTEXT' for '$TARGET_LABEL'." >&2
        echo "Switch with: kubectl config use-context $EXPECTED_CONTEXT (or set K8S_CONTEXT=)" >&2
        exit 1
    fi
    # Fail fast: a dead cluster must not surface as a failed apply minutes into
    # a test gate. Docker Desktop's kind cluster reports "running" while gone.
    if ! kubectl --context "$EXPECTED_CONTEXT" get --raw /readyz >/dev/null 2>&1; then
        echo "ERROR: cluster '$EXPECTED_CONTEXT' is not reachable." >&2
        echo "Docker Desktop: is Kubernetes green? (docker desktop kubernetes status; a" >&2
        echo "Docker Desktop restart re-provisions the kind cluster — see deploy/k8s/README.md)" >&2
        echo "k3s: is the server up and the kubeconfig current?" >&2
        exit 1
    fi
    KUBECTL="kubectl --context $EXPECTED_CONTEXT -n $NAMESPACE"
}

# apply -k creates the namespace too, but Secrets must precede it on a fresh
# cluster — so both entry scripts ensure it before touching Secrets.
k8s_ensure_namespace() {
    kubectl --context "$EXPECTED_CONTEXT" apply -f deploy/k8s/base/namespace.yaml >/dev/null
}

k8s_secret_exists() {
    $KUBECTL get secret "$1" >/dev/null 2>&1
}

# ── Deploy: ONE apply, image injected via a transient kustomization ───────────
# An `apply -k overlay` + `kubectl set image` two-step is a correctness bug
# here: apply resets the Deployment to the manifest's image, so with strategy
# Recreate every deploy would kill the pod, start it on wrong/stale bytes
# (live on the shared token!), then kill it again for the real SHA. Instead, a
# throwaway kustomization wraps the overlay and sets the image, so manifests
# and image land in one atomic apply. The base's sentinel tag guarantees
# nothing else ever supplies a runnable image.
k8s_deploy_image() {
    local image="$1"
    local deploy_dir="deploy/k8s/.deploy"   # gitignored; recreated every run
    trap 'rm -rf "deploy/k8s/.deploy"' EXIT # never leave a stale copy behind
    rm -rf "$deploy_dir" && mkdir -p "$deploy_dir"
    cat > "$deploy_dir/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../overlays/$OVERLAY_NAME
images:
  - name: discord-music-bot
    # Quoted: a hex SHA that happens to be all digits (or <digits>e<digits>)
    # would otherwise parse as a YAML number and fail kustomize's string check.
    newName: "${image%:*}"
    newTag: "${image##*:}"
EOF

    echo "Deploying $image to $TARGET_LABEL (overlay: $OVERLAY_NAME)"
    kubectl --context "$EXPECTED_CONTEXT" apply -k "$deploy_dir"
    # Timeout must exceed the startupProbe budget (§3.4: 60 × 10s = 600s) — a
    # slow many-shard boot is legitimate and must not fail the deploy as if it
    # were stuck.
    $KUBECTL rollout status deployment/discord-music-bot --timeout=660s
    rm -rf "$deploy_dir"
}
