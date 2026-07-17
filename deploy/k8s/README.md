# Kubernetes deployment

An **additional** deployment pipeline — Docker Compose (`../../docker-compose.yml`,
`./build_docker.sh`) remains fully supported. Same image, same test gate, same secrets;
the pipelines diverge only at the "run it" step. Full design:
`docs/K8S_DEPLOYMENT_PLAN.md` (local doc).

Two clusters, one Kustomize base:

| | **dev** | **prod** |
|---|---|---|
| Cluster | Docker Desktop built-in Kubernetes | k3s on the dedicated Linux server |
| kubectl context | `docker-desktop` | `k3s-production` (renamed — see below) |
| Image source | shared Docker daemon store (local builds) | GHCR, CI-built `sha-` tags only |
| Deploy | `./build_k8s_dev.sh` | `./build_k8s_prod.sh` |

**Only one environment runs at a time** — compose, dev cluster, or prod cluster. One
Discord token means two live bots join voice twice and play double audio. The build
scripts warn about same-machine collisions; dev-vs-prod is an operating rule until the
planned separate dev Discord application/token lands (then it's just a different
`DISCORD_TOKEN` in the dev cluster's Secret).

## Cluster prerequisites

**dev:** Docker Desktop → *Settings → Kubernetes → Enable*. Give the VM ≥ 8 GB
(*Settings → Resources*). `kubectl` ships with it; the `docker-desktop` context is
installed automatically.

**prod:** on the server:

```bash
curl -sfL https://get.k3s.io | sh -    # single node; bundled Traefik is unused
```

Then merge `/etc/rancher/k3s/k3s.yaml` into the operator machine's kubeconfig and
**rename the context** from k3s's default (`default`) to `k3s-production` —
the shared context guard (`k8s_common.sh`) depends on the unambiguous name:

```bash
kubectl config rename-context default k3s-production
```

Firewall the API endpoint (6443) to operator IPs / WireGuard — it must be reachable by
`kubectl`, not by the world.

## Secret bootstrap

**dev: automatic.** `./build_k8s_dev.sh` creates both Secrets from `.env` if they are
absent, so re-provisioning the Docker Desktop cluster (which wipes them — see below)
costs nothing. It is **create-if-missing, not upsert**: a dev Secret you deliberately
diverged from `.env` survives ordinary deploys. `./build_k8s_dev.sh --rotate-secrets`
forces `.env` to overwrite them.

**prod: by hand, always.** `build_k8s_prod.sh` never writes Secrets — it only checks
they exist and aborts with a pointer here if not. Prod credentials are managed
out-of-band, and `.env` holds dev ones; auto-applying it would overwrite a rotated
prod secret with a stale local value or push the dev token to prod. Bootstrap prod
(and any cluster you want set up by hand) with:

```bash
for CTX in docker-desktop k3s-production; do
  kubectl --context "$CTX" create namespace discord-music-bot
  kubectl --context "$CTX" -n discord-music-bot create secret generic discord-music-bot-secrets \
      --from-literal=DISCORD_TOKEN=... \
      --from-literal=SPOTIFY_CLIENT_ID=... \
      --from-literal=SPOTIFY_CLIENT_SECRET=...
  kubectl --context "$CTX" -n discord-music-bot create secret generic grafana-admin \
      --from-literal=GF_SECURITY_ADMIN_PASSWORD=...
done
```

- Both clusters currently get the **same** `DISCORD_TOKEN` (single-token operating
  rule). When the separate dev Discord application lands, re-create the dev cluster's
  secret with its token — no manifest changes, and `build_k8s_dev.sh`'s create-if-missing
  rule leaves it alone (don't pass `--rotate-secrets` after that).
- Prod's Grafana password is a real one (compose uses inline `admin`).
- If the GHCR package is private, prod also needs the pull secret — see
  `overlays/production/pull-secret-patch.yaml` (create the secret, uncomment the patch).

## Deploying

`./build_k8s_dev.sh` and `./build_k8s_prod.sh` (repo root) are the only supported deploy
paths. Each reads top-to-bottom with no target branching; the guards and the deploy
itself are shared via `k8s_common.sh` (sourced, not run) so the
correctness-critical apply exists in exactly one place:

- **dev** — local test gate (black + pytest in Docker), local image build, deploy to
  Docker Desktop. No registry round-trip.
- **prod** — deploy-only, provenance-gated: HEAD must be on `origin/main` and its
  CI-built `ghcr.io/...:sha-<sha>` image must already exist. Production never runs
  bytes that didn't go through `main` + CI.

`k8s_common.sh` injects the image via a transient kustomization; the base
manifest pins a sentinel tag that never exists. A raw `kubectl apply -k deploy/k8s/overlays/<t>` will
therefore leave the bot in ImagePullBackOff — deliberate; use the script.

Deploys are seamless by design: the pod is killed without cleanup, Redis keeps the
voice/text channel state, and the next pod's crash recovery resumes playback mid-song.

## Tearing down — `./k8s_down.sh [dev|prod] [flag]`

The `docker compose down` peer. Modes, least to most destructive:

| Command | Compose analogue | Deletes | Keeps |
|---|---|---|---|
| `./k8s_down.sh dev --stop` | `compose stop` | nothing (scales bot to 0) | everything — resumes in seconds |
| `./k8s_down.sh dev` | `compose down` | Deployments, StatefulSet, Services, ConfigMap | **PVCs, Secrets, namespace** |
| `./k8s_down.sh dev --volumes` | `compose down -v` | the above **+ every PVC** | Secrets, namespace |
| `./k8s_down.sh dev --all` | — | the whole namespace + PriorityClass | nothing (re-bootstrap secrets) |

`--volumes` and `--all` destroy `guild:{id}:history`, which has no TTL and is not
recreatable — both demand a typed confirmation (`--yes` to script it), as does any
non-`--stop` teardown of prod. Back up first (see below).

**Never `kubectl delete -k deploy/k8s/overlays/<t>`** as a "down" — the base includes
`namespace.yaml`, so it deletes the Namespace and cascades into every PVC *and* both
Secrets. That is `compose down` silently behaving like `down -v`; the script exists
precisely to name what it deletes.

After a plain `down`, `./build_k8s_<target>.sh` restores everything and the PVCs
reattach with their data (verified end-to-end 2026-07-16: a sentinel key written
before `down` read back intact after redeploy). A `down` also kills any
`port-forward` you had open on lgtm — restart it after the redeploy.

## Cheat-sheet

```bash
NS="kubectl -n discord-music-bot"   # add --context docker-desktop|k3s-production

$NS get pods                                          # 2/2 = bot + pot-provider
$NS logs deploy/discord-music-bot -c bot -f           # bot logs (-c pot-provider for the sidecar)
$NS port-forward svc/lgtm 3014:3000                   # Grafana → http://localhost:3014
$NS rollout undo deployment/discord-music-bot         # roll back to the previous SHA
$NS rollout restart deployment/discord-music-bot      # force restart on the SAME image
                                                      # (re-running the script on an
                                                      # unchanged SHA is a no-op)
$NS scale deployment/discord-music-bot --replicas=0   # stop the bot (e.g. before
                                                      # starting compose or the other
                                                      # cluster); --replicas=1 resumes
kubectl --context <ctx> apply -k deploy/k8s/overlays/<dev|production> --dry-run=server
```

- **Dev rollback caveat:** `rollout undo` on dev only reaches images still in the local
  daemon store — `docker image prune` deletes old SHA tags; rebuild from the old commit
  as fallback. Prod is immune (GHCR retains every deployed tag).
- **Dirty-worktree tags:** with uncommitted changes, `build_k8s_dev.sh` tags
  `<sha>-dirty.<hash-of-diff>` instead of the bare `<sha>`. Required, not cosmetic:
  kubectl diffs the pod spec's image tag as a *string*, so rebuilding one tag from
  edited source leaves the Deployment `unchanged` — nothing rolls out and the pod keeps
  running old code while the script reports success. (compose is immune; it compares
  resolved image IDs.) The hash is deterministic, so re-running without further edits is
  still a true no-op. Each distinct edit mints an image: `docker image prune` occasionally.
- **otel-lgtm retention:** Tempo/Loki keep data indefinitely by default; set the
  image's retention env vars on the lgtm Deployment (7 days is plenty) before the PVCs
  fill.

## History backup / restore (the §3.8 helper pod)

`guild:{id}:history` is persistent and precious — everything else settles back via
TTL'd keys. PVs are node-local on both clusters (Docker Desktop VM / k3s
`/var/lib/rancher/k3s/storage/`), so the backup is a tar through a helper pod. Run the
backup **on a schedule on prod** (server cron). **Treat dev-cluster PVs as disposable:**
Docker Desktop's kind-mode cluster can be re-provisioned from scratch — namespace,
secrets, and all PVs wiped — by a plain Docker Desktop restart (observed 2026-07-16),
not just the explicit "Reset Kubernetes Cluster" button. Never keep the only copy of
history on the dev cluster; re-bootstrap (namespace + secrets) after any wipe.

**Backup (PVC → tar):**

```bash
kubectl -n discord-music-bot scale statefulset redis --replicas=0
kubectl -n discord-music-bot run pvc-loader --image=alpine --restart=Never \
    --overrides='{"spec":{"containers":[{"name":"pvc-loader","image":"alpine",
      "command":["sleep","300"],
      "volumeMounts":[{"name":"data","mountPath":"/data"}]}],
      "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"redis-data-redis-0"}}]}}'
kubectl -n discord-music-bot wait --for=condition=Ready pod/pvc-loader
kubectl -n discord-music-bot exec pvc-loader -- tar -C /data -cf /tmp/redis-data.tar .
kubectl -n discord-music-bot cp pvc-loader:/tmp/redis-data.tar redis-data.tar
kubectl -n discord-music-bot delete pod pvc-loader
kubectl -n discord-music-bot scale statefulset redis --replicas=1
```

**Restore (tar → PVC):** same helper pod, reversed — `kubectl cp` the tar in, then
`exec pvc-loader -- sh -c 'rm -rf /data/* && tar -C /data -xf /tmp/redis-data.tar'`.
An untested backup isn't one: restore into a scratch namespace and `redis-cli LLEN` a
history key.

**Migrating compose state into a cluster** (one-time, both bots stopped): clean-stop
compose redis (`docker compose stop redis`), tar the named volume out
(`docker run --rm -v discord-music-bot_redis-data:/data -v "$PWD":/backup alpine tar -C /data -cf /backup/redis-data.tar .`),
then the restore procedure above. Never point two pipelines at one shared Redis — two
bots would fight over guild state.

## Break-glass: GHCR unavailable

Prod normally pulls only CI-built GHCR images. If GHCR is down and a deploy can't wait:

```bash
docker save discord-music-bot:<sha> | ssh <server> sudo k3s ctr images import -
```

Then deploy that tag. This bypasses the provenance gate — document why, and re-deploy a
GHCR image when it's back.
