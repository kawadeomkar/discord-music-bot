#!/usr/bin/env bash
# Kubernetes teardown — peer of build_k8s_dev.sh / build_k8s_prod.sh; the k8s
# answer to `docker compose down`. Shared guards: k8s_common.sh.
# Ops: deploy/k8s/README.md; design: docs/K8S_DEPLOYMENT_PLAN.md §6.3.
#
#   ./k8s_down.sh dev             delete workloads, KEEP data + secrets   (~ compose down)
#   ./k8s_down.sh dev --stop      scale to zero, keep everything          (~ compose stop)
#   ./k8s_down.sh dev --volumes   ALSO delete PVCs — DESTROYS history     (~ compose down -v)
#   ./k8s_down.sh dev --all       delete the whole namespace: PVCs + Secrets too
#
# WHY NOT `kubectl delete -k deploy/k8s/overlays/<t>`: the base includes
# namespace.yaml, so that command deletes the Namespace and cascades into the
# Secrets and every PVC — i.e. a `compose down` that silently wipes your volumes
# and forces a secret re-bootstrap. compose keeps volumes on `down`; this script
# does the same, and never touches data unless you ask for it by name.
#
# WHY dev|prod IS AN ARGUMENT HERE, while the build scripts are split per target:
# split where behaviour diverges, parameterise where it doesn't. Building dev vs
# prod are genuinely different procedures (build an image vs. verify CI built
# one; create Secrets vs. refuse to). Tearing down is the same procedure on both
# — the only target-dependent line is the prod confirmation below. This file's
# complexity runs along MODE (stop/down/volumes/all), not target; splitting it
# per target would copy all four mode blocks, delete logic included, to remove
# one `if`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./k8s_common.sh

TARGET="${1:-}"
[ $# -gt 0 ] && shift

MODE="down"
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --stop)    MODE="stop" ;;
    --volumes|-v) MODE="volumes" ;;
    --all)     MODE="all" ;;
    --yes|-y)  ASSUME_YES=1 ;;
    *)         echo "unknown option: $arg" >&2; exit 64 ;;
  esac
done

PRIORITYCLASS="discord-music-bot-critical"   # NAMESPACE comes from k8s_common.sh

case "$TARGET" in
  dev)  EXPECTED_CONTEXT="${K8S_CONTEXT:-docker-desktop}" ;;
  prod) EXPECTED_CONTEXT="${K8S_CONTEXT:-k3s-production}" ;;
  *)    echo "usage: $0 [dev|prod] [--stop|--volumes|--all] [--yes]" >&2; exit 64 ;;
esac
TARGET_LABEL="$TARGET"

# Context + reachability, shared with the build scripts. A dead cluster must not
# look like a successful teardown.
k8s_guard

confirm() {   # confirm <blurb> <word-the-operator-must-type>
    [ "$ASSUME_YES" = "1" ] && return 0
    printf '%s\n' "$1" >&2
    if [ ! -t 0 ]; then
        echo "ERROR: no TTY to confirm on — pass --yes if you really mean it." >&2
        exit 1
    fi
    read -r -p "Type '$2' to continue: " reply
    [ "$reply" = "$2" ] || { echo "Aborted — nothing was deleted." >&2; exit 1; }
}

# Tearing prod down takes the live bot off Discord mid-song; never on a stray keystroke.
if [ "$TARGET" = "prod" ] && [ "$MODE" != "stop" ]; then
    confirm "About to '$MODE' PRODUCTION ($EXPECTED_CONTEXT) — the live bot goes offline." "prod"
fi

case "$MODE" in
  # ── compose `stop` parity: reversible in seconds, state fully intact ────────
  stop)
    echo "Scaling workloads to zero (data, secrets, and manifests all kept)"
    $KUBECTL scale deployment discord-music-bot --replicas=0 2>/dev/null || true
    ;;

  # ── compose `down` parity: workloads gone, PVCs + Secrets + namespace kept ──
  down)
    echo "Deleting workloads (PVCs, Secrets, and the namespace are kept)"
    # Bot first: stop the live Discord session before its dependencies vanish.
    $KUBECTL delete deployment discord-music-bot --ignore-not-found
    $KUBECTL delete statefulset redis --ignore-not-found
    $KUBECTL delete deployment lgtm --ignore-not-found
    $KUBECTL delete service redis lgtm --ignore-not-found
    $KUBECTL delete configmap discord-music-bot-config --ignore-not-found
    echo "Done. './build_k8s_$TARGET.sh' brings it all back; PVC data is reattached."
    ;;

  # ── compose `down -v` parity: the only mode that destroys play history ──────
  volumes)
    confirm "This DELETES every PVC in '$NAMESPACE' on '$EXPECTED_CONTEXT' — including
Redis's guild:{id}:history, which has no TTL and is not recreatable. Back it up
first if you care: see 'History backup / restore' in deploy/k8s/README.md." "delete-data"
    echo "Deleting workloads"
    $KUBECTL delete deployment discord-music-bot --ignore-not-found
    $KUBECTL delete statefulset redis --ignore-not-found
    $KUBECTL delete deployment lgtm --ignore-not-found
    $KUBECTL delete service redis lgtm --ignore-not-found
    $KUBECTL delete configmap discord-music-bot-config --ignore-not-found
    # After the workloads, or the PVCs hang in Terminating on the mounts.
    # --all also catches redis-data-redis-0, which the StatefulSet's
    # volumeClaimTemplate creates and which no manifest deletion would reach.
    echo "Deleting PVCs"
    $KUBECTL delete pvc --all
    echo "Done. Secrets survive; './build_k8s_$TARGET.sh' starts from empty volumes."
    ;;

  # ── Full reset: namespace (Secrets + PVCs with it) and the cluster-scoped PC ─
  all)
    confirm "This DELETES the whole '$NAMESPACE' namespace on '$EXPECTED_CONTEXT':
every PVC (play history included) AND both Secrets. You will have to re-run the
secret bootstrap in deploy/k8s/README.md before the next deploy." "delete-everything"
    $KUBECTL delete namespace "$NAMESPACE" --ignore-not-found --wait=true
    # Cluster-scoped: outlives the namespace, so it needs its own delete.
    kubectl --context "$EXPECTED_CONTEXT" delete priorityclass "$PRIORITYCLASS" --ignore-not-found
    echo "Done. Re-bootstrap namespace + secrets before './build_k8s_$TARGET.sh'."
    ;;
esac
