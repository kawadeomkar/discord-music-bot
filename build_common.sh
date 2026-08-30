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
    # GIT_SHA is the caller's to set (build_docker.sh exports it, `just image`
    # exports the tag it computed) for the same explicit-propagation reason as
    # ENVIRONMENT. Defaulted here rather than left unset so a caller that forgets
    # bakes a readable "unknown" instead of an empty string.
    # CHART_EXTRAS rides the environment rather than a positional, so the three-name
    # contract above still holds. ${x+...} tests SET, not non-empty, which is the whole
    # mechanism: `CHART_EXTRAS=` (empty, set) builds the slim variant, and leaving it
    # unset defers to the Dockerfile's charts-included default.
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --build-arg GIT_SHA="${GIT_SHA:-unknown}" \
        ${CHART_EXTRAS+--build-arg CHART_EXTRAS="$CHART_EXTRAS"} \
        ${tag_args[@]+"${tag_args[@]}"} --target runtime -f Dockerfile .
}

# _env_value <NAME> <file> — the effective value of NAME in a .env file: the last
# uncommented assignment (compose is last-wins), with an optional `export ` prefix,
# surrounding quotes and trailing whitespace stripped. Empty output means unset,
# commented out, or explicitly empty — compose treats all three identically, and
# so does every caller here.
_env_value() {
    local name="$1" file="$2"
    grep -E "^[[:space:]]*(export[[:space:]]+)?${name}=" "$file" 2>/dev/null \
        | tail -n1 \
        | sed -E 's/^[[:space:]]*(export[[:space:]]+)?'"${name}"'=//; s/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/' \
        || true
}

# resolve_archive_profile — HISTORY_ARCHIVE_ENABLED into the compose `archive`
# profile, so that flag is the only switch. Exports COMPOSE_PROFILES and
# ARCHIVE_ENABLED (0/1). Compose cannot derive it: profiles activate only from
# COMPOSE_PROFILES or --profile, and `${FLAG:+archive}` yields `archive` for
# `false` too. See docs/ARCHITECTURE.md#history-archive-tier.
#
# The export is unconditional, EMPTY INCLUDED — the process environment beats
# .env, which is how a flag flipped to false overrides an older .env still
# carrying COMPOSE_PROFILES=archive. Only that element is owned; other profiles
# survive. Garbage exits 1 in config.history_archive_enabled's own terms: a
# deploy that disagreed with the bot it deploys is the worst outcome.
resolve_archive_profile() {
    local env_file=".env" flag profiles rest item
    flag="${HISTORY_ARCHIVE_ENABLED:-$(_env_value HISTORY_ARCHIVE_ENABLED "$env_file")}"
    flag="$(printf '%s' "$flag" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    case "$flag" in
        "" | false | 0 | no) ARCHIVE_ENABLED=0 ;;
        true | 1 | yes) ARCHIVE_ENABLED=1 ;;
        *)
            echo "HISTORY_ARCHIVE_ENABLED must be one of true/false, 1/0, or yes/no" >&2
            echo "(case-insensitive); got '$flag'. The bot refuses to start on this" >&2
            echo "value, so the deploy refuses too." >&2
            exit 1
            ;;
    esac

    # Rebuild the list without `archive` rather than appending to it: this runs
    # on every deploy, so a plain append would accumulate duplicates, and a
    # disabled flag has to be able to REMOVE the element it did not add.
    profiles="${COMPOSE_PROFILES:-$(_env_value COMPOSE_PROFILES "$env_file")}"
    rest=""
    while IFS= read -r item; do
        case "$item" in "" | archive) continue ;; esac
        rest="${rest:+$rest,}$item"
    done <<< "$(printf '%s' "$profiles" | tr ',' '\n' | tr -d '[:blank:]')"

    if [ "$ARCHIVE_ENABLED" -eq 1 ]; then
        COMPOSE_PROFILES="${rest:+$rest,}archive"
        echo "History archive: ENABLED — deploying postgres + db-migrate" >&2
    else
        COMPOSE_PROFILES="$rest"
        echo "History archive: disabled — no postgres deployed (the default)" >&2
    fi
    export COMPOSE_PROFILES ARCHIVE_ENABLED
}

# resolve_external_postgres_env — fill EXTERNAL_PG_ENV with `-e POSTGRES_URL=…`
# when the DSN names a database a CONTAINER can reach, empty otherwise. For
# `docker compose run` of db-migrate / db-backfill, which address postgres by
# service name: right for the bundled stack, wrong for an external database.
#
# Passing POSTGRES_URL through unconditionally is worse than either. .env holds
# the HOST-form DSN the host-networked bot and `just run` need, and 127.0.0.1
# inside a compose-network container is that container — the migration then
# fails with connection refused against a healthy database. Loopback is the
# discriminator, and an unparsed DSN errs toward the service name.
resolve_external_postgres_env() {
    EXTERNAL_PG_ENV=()
    local url host
    url="${POSTGRES_URL:-$(_env_value POSTGRES_URL .env)}"
    [ -n "$url" ] || return 0
    # Order matters: scheme, then path/query, THEN credentials, then the port.
    # Credentials are stripped to the LAST `@` so a password containing one does
    # not leave `ss@127.0.0.1` looking like an external host — but only after the
    # path is gone, or a `@` in a database name would eat the host instead.
    # Brackets last, for the IPv6 `[::1]:5432` form.
    host="$(printf '%s' "$url" \
        | sed -E 's#^[a-zA-Z0-9+.-]+://##; s#[/?].*$##; s#^.*@##; s#:[^:]*$##; s#^\[|\]$##g')"
    case "$host" in
        "" | localhost | ::1 | 127.*) return 0 ;;
    esac
    EXTERNAL_PG_ENV=(-e "POSTGRES_URL=$url")
}

# warn_default_postgres_password — compose-path preflight, called only from
# build_docker.sh: the k8s path takes its secret from a Secret, not .env.
#
# Renamed from require_postgres_password, which stopped requiring anything when
# the default landed: it returns 0 on every path now. Safe to rename — the
# file's declared "Fixed API" above names only resolve_environment,
# run_test_gate and build_runtime_image.
#
# The password is not required even when archiving: compose falls back to a
# known default so `docker compose up` works with nothing configured but
# DISCORD_TOKEN. So this warns rather than exits — a build that refused to
# proceed would put back the first-run cliff the default removed.
#
# The bot repeats the warning at startup and on every -ping, which are the
# surfaces an operator actually watches; this one just catches it earlier, at
# build time, with the same remedy.
#
# Skipped when nothing suggests a Postgres is deployed. The archive is opt-in
# (HISTORY_ARCHIVE_ENABLED + the `archive` compose profile), and a credential
# warning about a database that isn't there is noise. The gate is an OR of
# BOTH .env lines, deliberately wider than the bot's own flag-only gate: in
# the drift case — profile active, flag off — postgres runs idle with the
# default password and the bot's warnings are all silenced, so this preflight
# is the one surface left that can still say so. The string comparisons are
# dumb on purpose (truthy spellings / substring on the profile list); they
# mirror config.py's parser without importing it.
warn_default_postgres_password() {
    local env_file=".env"
    if [ ! -f "$env_file" ]; then
        # NOT "compose will use its built-in defaults" — that was wrong and the
        # README (§Requirements) says so: docker-compose.yml declares
        # `env_file: .env`, and Compose treats a MISSING one as an error, not a
        # warning. The password has a fallback; the file itself does not.
        echo "WARNING: $env_file not found. Compose declares env_file: .env and" >&2
        echo "         treats a missing one as an error, so \`docker compose up\`" >&2
        echo "         will fail regardless of any default. Run ./setup_env.sh." >&2
        return 0
    fi
    # The ENVIRONMENT WINS over .env, mirroring Compose's own precedence. Both
    # of these are read by Compose from the process environment first, so
    # `COMPOSE_PROFILES=archive ./build_docker.sh` with a token-only .env really
    # does deploy Postgres — and reading the file alone made this preflight
    # return 0 and say nothing, which is a silent miss in the one function whose
    # entire job is not to miss one.
    local flag profiles
    flag="${HISTORY_ARCHIVE_ENABLED:-$(_env_value HISTORY_ARCHIVE_ENABLED "$env_file")}"
    flag="$(printf '%s' "$flag" | tr '[:upper:]' '[:lower:]')"
    profiles="${COMPOSE_PROFILES:-$(_env_value COMPOSE_PROFILES "$env_file")}"
    case "$flag" in
        true | 1 | yes) : ;; # archive enabled — warn below
        *)
            case "$profiles" in
                *archive*) : ;; # drift case: postgres deployed anyway — warn below
                *) return 0 ;;  # no archive, no deployed postgres — nothing to warn about
            esac
            ;;
    esac
    local value
    value="$(_env_value POSTGRES_PASSWORD "$env_file")"
    if [ -z "$value" ] || [ "$value" = "password" ]; then
        echo "WARNING: POSTGRES_PASSWORD is unset or still the default in $env_file." >&2
        echo "         The play-history database will accept 'password' from" >&2
        echo "         anything that can reach the host's published port." >&2
        echo "         Fix it IN THIS ORDER — the bot's warning reads its DSN, so" >&2
        echo "         doing (2) first silences it while the server still accepts" >&2
        echo "         the old password:" >&2
        echo "           1. docker compose exec postgres psql -U <user> \\" >&2
        echo "                -c \"ALTER USER <user> PASSWORD '<new>'\"" >&2
        echo "           2. ./setup_env.sh --force, with the same value" >&2
        echo "           3. docker compose up -d   (the DSN is baked at create time)" >&2
    fi
}
