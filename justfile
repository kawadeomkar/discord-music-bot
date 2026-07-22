# Developer task index: one verb per recipe.
#
# Why this exists: build.sh used to run lint + tests + image build + deploy as one
# non-negotiable sequence, so linting — 0.13s of actual ruff work — cost a Docker
# image build and two container starts, and deploying meant re-running everything.
# Multi-step *pipelines* still live in the .sh scripts; this file is the index over
# the primitives they compose.
#
# `just check` is the contract: if it passes, CI passes. It is not a copy of
# .github/workflows/ci.yml's lint and test jobs — those jobs CALL these recipes, so
# the two cannot drift.

set shell := ["bash", "-cu"]

# 1.55.0 is what `set minimum-version` itself requires; nothing below needs newer.
# On an older `just` this is an unknown-setting parse error rather than a clean
# message, which is still an error, which is the point.
set minimum-version := '1.55.0'

# Variables are evaluated EAGERLY by default (unlike Make's `=`): a backtick
# assignment would fork on every invocation, `just --list` included. There are none
# today — the git SHA tag lives in build_common.sh's git_sha_tag, so there is one
# definition of "what tag identifies this build" — and this keeps it that way if one
# is ever added.
set lazy

IMAGE := "discord-music-bot"
DOCKER := env('DOCKER', '0')
REPO := justfile_directory()

# Call the venv's binaries directly rather than `poetry run`: poetry re-resolves the
# project on every invocation, which costs ~1.4s and dwarfs ruff's 0.13s of real work.
#
# Which venv, though, is not obvious on a dev box. pyenv-virtualenv auto-activates
# this project's env from .python-version and exports VIRTUAL_ENV — and poetry honours
# an already-activated env over poetry.toml's in-project setting, so `poetry install`
# lands THERE, not in ./.venv. Following VIRTUAL_ENV when it is set keeps `just install`
# and `just lint` pointed at the same interpreter; the ./.venv fallback is what CI
# (which caches that path) and the Dockerfile use. Absolute via REPO so recipes work
# from any subdirectory.
VENV_BIN := if env('VIRTUAL_ENV', '') != '' { env('VIRTUAL_ENV', '') / "bin" } else { REPO / ".venv/bin" }

# ── Where the tools run: local venv (default) or the test image (DOCKER=1) ────
#
#   just check            native, fast — needs Python, Poetry and the venv
#   DOCKER=1 just check   same checks inside the image — needs only Docker and just
#
# DOCKER=1 exists so the project can be handed to someone with no Python toolchain.
# The checks are the same commands either way; only the interpreter they run under
# differs. Note the override must PRECEDE the recipe (`DOCKER=1 just check`, not
# `just check DOCKER=1` — that is a "recipe not found" error).
#
# Mount src/ and tests/ as SUBDIRECTORIES, never the repo root. The image keeps its
# virtualenv at /app/.venv and puts it on PATH, so mounting over /app would shadow the
# venv and every tool below would vanish. pyproject.toml is mounted read-only so
# ruff/pytest/pyright read the working tree's config rather than the copy baked into
# the image.
DOCKER_MOUNTS := '-v "' + REPO + '/src:/app/src" -v "' + REPO + '/tests:/app/tests" -v "' + REPO + '/pyproject.toml:/app/pyproject.toml:ro"'

# Two run modes, and the difference is not cosmetic:
#
#   as the host uid  ruff REWRITES the mounted files. Running as root would leave them
#                    root-owned on the host, which is how a formatter turns into a
#                    permissions incident. pyright only reads, but runs here too so
#                    nothing in this group can write as root.
#   as root          pytest writes .pytest_cache and coverage data into /app, which is
#                    image-owned and NOT mounted — the host uid cannot write there and
#                    pytest fails. Nothing it writes escapes the container, so root is
#                    safe for it specifically.
DOCKER_RUN := 'docker run --rm ' + DOCKER_MOUNTS + ' ' + IMAGE + ':test'
DOCKER_RUN_USER := 'docker run --rm --user "$(id -u):$(id -g)" ' + DOCKER_MOUNTS + ' ' + IMAGE + ':test'

RUFF := if DOCKER == "1" { DOCKER_RUN_USER + ' ruff' } else { VENV_BIN / 'ruff' }

# --pythonpath is not optional here. pyright resolves imports from the interpreter it
# is TOLD about, not the one it runs from: with `[tool.pyright] venvPath/venv` it read
# ./.venv, which on a pyenv box is a different (and stale) environment from the
# $VIRTUAL_ENV that `just install`, `just lint` and `just test` all use — so `just
# types` type-checked against a package set the other recipes never saw. Those keys are
# gone from pyproject.toml; this flag replaces them, and it points at exactly the same
# VENV_BIN as every other recipe. Worse than wrong, the old setup failed SILENTLY: a
# missing .venv makes pyright warn and exit 0.
PYRIGHT := if DOCKER == "1" { DOCKER_RUN_USER + ' pyright' } else { VENV_BIN / 'pyright' + ' --pythonpath ' + VENV_BIN / 'python' }
PYTEST := if DOCKER == "1" { DOCKER_RUN + ' pytest' } else { VENV_BIN / 'pytest' }

[private]
default:
    @{{ just_executable() }} --justfile {{ justfile() }} --list --list-heading $'Recipes (run `just <recipe>`):\n'
    @echo ""
    @echo "Prefix DOCKER=1 to run fmt/lint/types/test inside the test image instead"
    @echo "of a local venv — requires only Docker and just, no Python or Poetry."

# ── Setup ────────────────────────────────────────────────────────────────────

# Create the venv with main + test + lint + dev dependencies
[group('setup')]
install:
    poetry install --with test,lint,dev

# Install the git hooks (ruff on commit, `just check` on push)
[group('setup')]
hooks: _venv
    {{ VENV_BIN }}/pre-commit install

# Bump pinned hook revisions in .pre-commit-config.yaml
[group('setup')]
hooks-update: _venv
    {{ VENV_BIN }}/pre-commit autoupdate

# Run every hook against every file (not just staged ones)
[group('setup')]
hooks-run: _venv
    {{ VENV_BIN }}/pre-commit run --all-files

# Rebuild the test image DOCKER=1 uses (needed after a dependency change)
[group('setup')]
test-image-rebuild:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_environment
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "{{ IMAGE }}:test" --target test -f Dockerfile .

# Fail with an actionable message rather than "No such file or directory". Probes
# pre-commit specifically because that is the only tool the hooks* recipes call.
[private]
_venv:
    @test -x {{ VENV_BIN }}/pre-commit || { echo "No usable venv at {{ VENV_BIN }}/ — run 'just install' first." >&2; exit 1; }

# Make sure the ONE tool this check calls actually exists, on whichever path is selected.
#
# Takes the tool name rather than probing a fixed one. Probing `ruff` as a stand-in for
# "the venv is usable" is wrong: CI's test job installs only main,test,dev, which has no
# ruff, and would fail the guard against a perfectly good venv. Same for anyone who ran
# `poetry install --with test` locally. Naming the tool also makes the error say which
# one is missing.
#
# DELIBERATELY NOT a shebang recipe: a shebang body is written to a temp file and
# executed, which costs ~0.3s on macOS — in front of `just lint`, whose real work is
# 0.13s. Plain lines with continuations cost 0.03s. Parameterising it did not change
# that; `just` de-duplicates by (recipe, arguments), so `check` runs this three times,
# not four.
#
# The image is built only when ABSENT, not when stale: src/ and tests/ are bind-mounted,
# so ordinary code changes need no rebuild. Dependency changes do — run
# `just test-image-rebuild` (or the always-rebuilding `just container-test`) after
# touching pyproject.toml or poetry.lock, or DOCKER=1 keeps using the old dependency set.
[private]
_tools TOOL:
    @if [ "{{ DOCKER }}" = "1" ]; then \
        docker image inspect "{{ IMAGE }}:test" >/dev/null 2>&1 \
            || {{ just_executable() }} --justfile "{{ justfile() }}" test-image-rebuild; \
    else \
        test -x "{{ VENV_BIN }}/{{ TOOL }}" \
            || { echo "{{ TOOL }} not found in {{ VENV_BIN }}/ — run 'just install' first." >&2; exit 1; }; \
    fi

# ── Checks (fast → slow) ─────────────────────────────────────────────────────
#
# One recipe per tool invocation, and `check` chains them. CI runs these as separately
# named steps so GitHub names the failing TOOL in the checks UI, not just "check".

# Format and auto-fix src/ and tests/ (REWRITES files)
[group('check')]
fmt: (_tools 'ruff')
    {{ RUFF }} check --fix src/ tests/
    {{ RUFF }} format src/ tests/

# Check formatting only, no rewrites (~0.04s)
[group('check')]
fmt-check: (_tools 'ruff')
    {{ RUFF }} format --check src/ tests/

# Check lint rules only, no rewrites (~0.05s)
[group('check')]
lint: (_tools 'ruff')
    {{ RUFF }} check src/ tests/

# Type-check src/ AND tests/ with pyright (~6s)
[group('check')]
types: (_tools 'pyright')
    {{ PYRIGHT }}

# Run the test suite with coverage (~13s); extra pytest flags may be appended
[group('check')]
test *ARGS: (_tools 'pytest')
    {{ PYTEST }} --tb=short -q {{ ARGS }}

# Everything CI gates on — run this before pushing
[group('check')]
check: fmt-check lint types test

# `test`, plus the coverage/JUnit artifacts CI's PR-comment action consumes. Defined in
# terms of `test` rather than repeating the pytest invocation, so this can never become
# a second definition of the gate — only reporting flags differ, and they never affect
# pass/fail. `set -o pipefail` lives here rather than in the workflow so it cannot be
# forgotten; without it, `tee` would mask a failing suite.
#
# Under DOCKER=1 only pytest-coverage.txt survives: tee runs on the host, but the xml
# and junit files are written inside the container relative to /app, which is not
# mounted. Do NOT "fix" that by widening DOCKER_MOUNTS — mounting /app shadows the
# image's venv. Mount an explicit artifacts directory if it is ever needed.
#
# [doc] and not a trailing `#` line: `just` takes only the LAST comment line above a
# recipe as its description, so a reasoning block like this one would otherwise show up
# mid-sentence in `just --list`.
[doc('Like `test`, but also writes the coverage/JUnit artifacts CI consumes')]
[group('check')]
test-report *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ just_executable() }} --justfile "{{ justfile() }}" test \
        --cov-report=xml --junitxml=pytest.xml {{ ARGS }} | tee pytest-coverage.txt

# Mirrors CI's container-test job. Its value is proving the IMAGE runs (a runtime stage
# missing a dependency is invisible to `just test`), which is why it is not part of
# `check`.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Build the test image and run the suite inside it')]
[group('check')]
container-test:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_environment
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" -t "{{ IMAGE }}:test" --target test -f Dockerfile .
    docker run --rm "{{ IMAGE }}:test"

# Full local mirror of the CI workflow
[group('check')]
ci: check container-test

# ── Image and deployment ─────────────────────────────────────────────────────
#
# The gate belongs to the *pipeline* (./build_docker.sh), never to these primitives:
# a gate you cannot skip is a gate you route around.

# Build the runtime image as :latest and :<git-sha> — no test gate
[group('build')]
image:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_environment
    build_runtime_image "{{ IMAGE }}:latest" "{{ IMAGE }}:$(git_sha_tag)"

# Deploy an already-built image; pass a git sha to roll back
[group('deploy')]
up TAG='':
    ./deploy_docker.sh {{ TAG }}

# Stop the compose stack (volumes are kept)
[group('deploy')]
down:
    docker compose down

# NOT a deploy. `docker compose restart` stops and starts the EXISTING container with
# the image it already has, so a newly built image is not picked up — the old help text
# said "recreate", which sent `image && restart` down a path that silently kept running
# the old code. Use `just up` to deploy.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Restart the running bot in place — does NOT pick up a new image (use `just up`)')]
[group('deploy')]
restart:
    docker compose restart discord-music-bot

# Follow the bot's logs
[group('deploy')]
logs *ARGS:
    docker compose logs -f discord-music-bot {{ ARGS }}

# Show compose service status
[group('deploy')]
ps:
    docker compose ps

# ── Worktrees ────────────────────────────────────────────────────────────────
#
# Provisioning a worktree is NOT `git worktree add` plus copying the gitignored
# dotfiles, and one file is the reason. The venv's site-packages holds
#
#     discord_music_bot.pth  ->  <absolute path of the tree it was installed from>
#
# so a venv puts THAT tree on sys.path. Share or symlink one into a worktree and its
# tools import the other tree's src/ while you edit this one — a green suite proving
# nothing about the code in front of you. Every worktree therefore gets its own venv.
#
# The sharper trap is .python-version. Copying it looks like the way to make the
# worktree behave like this tree; what it actually does is make pyenv activate the
# SHARED env there, so `poetry install` writes into that env and rewrites the .pth
# above to point at the worktree — breaking this tree from a command run in another
# directory. It is deliberately not copied, and VIRTUAL_ENV is unset below so poetry
# cannot reach the shared env by that route either.

# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Create a sibling worktree on a new branch with its own venv (--env also links .env)')]
[group('setup')]
worktree NAME *FLAGS:
    #!/usr/bin/env bash
    set -euo pipefail

    # Every guard here is an if-statement, not `cond && { ...; }`. As a bare
    # statement that idiom is a set -e landmine: when the condition is FALSE the
    # list returns non-zero and the recipe exits instead of falling through.
    name='{{ NAME }}'
    if ! [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "Invalid name '$name' — letters, digits, dot, dash, underscore, no slashes." >&2
        exit 1
    fi

    # Empty-safe expansion: `for x in {{ FLAGS }}` with no flags is a syntax error on
    # bash 3.2, which is still what /usr/bin/env bash finds on a stock macOS.
    link_env=0
    flags=( {{ FLAGS }} )
    for flag in ${flags[@]+"${flags[@]}"}; do
        case "$flag" in
            --env) link_env=1 ;;
            *) echo "Unknown flag '$flag' — the only flag is --env." >&2; exit 1 ;;
        esac
    done

    repo='{{ REPO }}'
    wt="$(dirname "$repo")/$(basename "$repo")-$name"
    branch="task/$name"

    # Sibling of the repo, never inside it: .dockerignore does not exclude a nested
    # worktree, so one under this tree would join every `docker build .` context as a
    # second full copy of the repo — growing again the moment it has its own .venv.
    if [ -e "$wt" ]; then
        echo "$wt already exists — remove it or pick another name." >&2
        exit 1
    fi
    if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
        echo "Branch $branch already exists." >&2
        exit 1
    fi

    # poetry is installed in the project venv, not in pyenv's global version, so the
    # `poetry` shim resolves to nothing at all once we stop selecting this env — which
    # is exactly what the worktree does. Resolve it absolutely while we still can.
    poetry_bin='{{ VENV_BIN }}/poetry'
    if [ ! -x "$poetry_bin" ]; then
        poetry_bin="$(command -v poetry || true)"
    fi
    if [ ! -x "$poetry_bin" ]; then
        echo "No poetry found — run 'just install' in $repo first." >&2
        exit 1
    fi

    # The BASE interpreter, never {{ VENV_BIN }}/python: `poetry env use` given a
    # venv's python ADOPTS that venv, which is the sharing this whole recipe exists
    # to prevent.
    base_python="$('{{ VENV_BIN }}/python' -c 'import os, sys; print(os.path.join(sys.base_prefix, "bin", "python3"))' 2>/dev/null || true)"
    if [ ! -x "$base_python" ]; then
        base_python="$(command -v python3.14 || true)"
    fi
    if [ ! -x "$base_python" ]; then
        echo "No base Python 3.14 to build the worktree venv from." >&2
        exit 1
    fi

    git -C "$repo" worktree add -b "$branch" "$wt"

    # docs/ is gitignored, so the checkout does not bring it across. Symlinked rather
    # than copied because the plan documents are shared state and two diverging copies
    # is the failure mode.
    if [ -d "$repo/docs" ]; then
        ln -s "$repo/docs" "$wt/docs"
    fi

    # Opt-in, and symlinked for a different reason: .env holds the live Discord token,
    # and one file means deleting a worktree can never strand a copy of it. Off by
    # default because .env is precisely what makes `just up` work in that directory,
    # and `just up` brings the live bot online.
    if [ "$link_env" = 1 ]; then
        if [ -f "$repo/.env" ]; then
            ln -s "$repo/.env" "$wt/.env"
        else
            echo "warning: --env given but $repo/.env does not exist — skipped." >&2
        fi
    fi

    # Subshell so the unset cannot leak. poetry.toml is tracked with in-project = true,
    # so with nothing activated the venv lands at $wt/.venv.
    (
        cd "$wt"
        unset VIRTUAL_ENV
        "$poetry_bin" env use "$base_python"
        "$poetry_bin" install --with test,lint,dev
    )

    # Proves the header's hazard did not happen, rather than assuming it. Had poetry
    # reached the shared env, $wt/.venv would not exist; had it installed in-project
    # against the wrong source root, the .pth would name $repo.
    pth="$(echo "$wt"/.venv/lib/python*/site-packages/discord_music_bot.pth)"
    if [ ! -f "$pth" ] || [ "$(cat "$pth")" != "$wt" ]; then
        echo "FAILED: $wt/.venv does not resolve src/ to the worktree." >&2
        echo "Inspect before using it — the shared venv may have been written to." >&2
        exit 1
    fi

    echo ""
    echo "Worktree:  $wt"
    echo "Branch:    $branch"
    echo "It has its own .venv — run 'just check' from there."
    echo "Remove it with: git worktree remove $wt && git branch -d $branch"
