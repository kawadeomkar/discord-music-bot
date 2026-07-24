# Developer task index: one verb per recipe.
#
# Why this exists: build.sh used to run lint + tests + image build + deploy as one
# non-negotiable sequence, so linting — 0.13s of actual ruff work — cost a Docker
# image build and two container starts, and deploying meant re-running everything.
# Multi-step *pipelines* still live in the .sh scripts; this file is the index over
# the primitives they compose.
#
# `just check` is the contract for CI's lint and test jobs: if it passes, those two
# pass. It is not a copy of them — they CALL these recipes, so the two cannot drift.
# It is NOT the whole pipeline: `container-test` and `build` are separate CI jobs, and
# security.yml audits the lockfile. `just ci` covers the container job too; nothing
# local covers `build` or the audit. See the note above `check`.

set shell := ["bash", "-cu"]

# 1.55.0 is what `set minimum-version` itself requires; nothing below needs newer.
# On an older `just` this is an unknown-setting parse error rather than a clean
# message, which is still an error, which is the point.
set minimum-version := '1.55.0'

# Recipe arguments reach the body as "$@" instead of only as a flattened {{ ARGS }}
# string. Without this, `just test -k "spotify or youtube"` interpolates to
#   pytest -k spotify or youtube
# and pytest reads `or` and `youtube` as test paths — the README documents that
# forwarding, so it has to survive quoting. Recipes that use it must be shebang
# recipes; line-based recipes still interpolate {{ }} as before.
#
# Note this also sets $0 to the recipe's script path. build_common.sh's sourced-only
# guard compares ${BASH_SOURCE[0]} against $0 and still behaves — inside a sourced
# file BASH_SOURCE[0] is that file, never the caller's $0 — so `container-test` and
# `test-image-rebuild` keep working.
set positional-arguments

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

# quote() on the native leg, and it is load-bearing rather than defensive: these
# expand into recipe bodies unquoted (they cannot simply be wrapped in "" at the use
# site, because under DOCKER=1 the same variable is a whole command line, not a path).
# A repo cloned to a path containing a space therefore produced
#   /Users/me/my repo/.venv/bin/ruff check src/
# → `command not found: /Users/me/my`, and _venv's `test -x` guard turned into
# `test: too many arguments`, reporting "No usable venv" against a working one.
RUFF := if DOCKER == "1" { DOCKER_RUN_USER + ' ruff' } else { quote(VENV_BIN / 'ruff') }

# --pythonpath is not optional here. pyright resolves imports from the interpreter it
# is TOLD about, not the one it runs from: with `[tool.pyright] venvPath/venv` it read
# ./.venv, which on a pyenv box is a different (and stale) environment from the
# $VIRTUAL_ENV that `just install`, `just lint` and `just test` all use — so `just
# types` type-checked against a package set the other recipes never saw. Those keys are
# gone from pyproject.toml; this flag replaces them, and it points at exactly the same
# VENV_BIN as every other recipe. Worse than wrong, the old setup failed SILENTLY: a
# missing .venv makes pyright warn and exit 0.
#
# The DOCKER=1 leg names its interpreter too. It used to rely on the image putting
# /app/.venv/bin first on PATH so pyright's implicit "first python found" happened to
# be the right one — which made the container the single caller not covered by the
# rule above, in the one file that states the rule. Anything that prepends another
# Python to PATH in the test stage would have silently repointed it.
PYRIGHT := if DOCKER == "1" { DOCKER_RUN_USER + ' pyright --pythonpath /app/.venv/bin/python' } else { quote(VENV_BIN / 'pyright') + ' --pythonpath ' + quote(VENV_BIN / 'python') }
PYTEST := if DOCKER == "1" { DOCKER_RUN + ' pytest' } else { quote(VENV_BIN / 'pytest') }

[private]
default:
    @{{ quote(just_executable()) }} --justfile {{ quote(justfile()) }} --list --list-heading $'Recipes (run `just <recipe>`):\n'
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
    {{ quote(VENV_BIN / 'pre-commit') }} install

# Bump pinned hook revisions in .pre-commit-config.yaml
[group('setup')]
hooks-update: _venv
    {{ quote(VENV_BIN / 'pre-commit') }} autoupdate

# Run every hook against every file (not just staged ones)
[group('setup')]
hooks-run: _venv
    {{ quote(VENV_BIN / 'pre-commit') }} run --all-files

# Rebuild the test image DOCKER=1 uses (needed after a dependency change)
[group('setup')]
test-image-rebuild:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_environment
    docker build --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --label "dmb.dep-hash=$(dep_hash)" \
        -t "{{ IMAGE }}:test" --target test -f Dockerfile .

# Fail with an actionable message rather than "No such file or directory". Probes
# pre-commit specifically because that is the only tool the hooks* recipes call.
[private]
_venv:
    @test -x {{ quote(VENV_BIN / 'pre-commit') }} || { echo "No usable venv at {{ VENV_BIN }}/ — run 'just install' first." >&2; exit 1; }

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
# The image is rebuilt when absent OR when its dependency set is stale. Source changes
# never trigger it — src/ and tests/ are bind-mounted — but dependency changes must,
# and used not to: pyproject.toml is mounted read-only over a venv baked at build time,
# so `DOCKER=1 just check` after a poetry.lock edit checked the new config against the
# old packages and passed. build_docker.sh then shipped the new dependencies behind a
# green gate that never saw them. dep_hash lives in build_common.sh so the build and
# the staleness check cannot disagree about what "same dependencies" means.
[private]
_tools TOOL:
    @if [ "{{ DOCKER }}" = "1" ]; then \
        source {{ quote(REPO / 'build_common.sh') }}; \
        want="$(dep_hash)"; \
        have="$(docker image inspect --format '{{{{ index .Config.Labels "dmb.dep-hash" }}' "{{ IMAGE }}:test" 2>/dev/null || true)"; \
        { [ -n "$have" ] && [ "$want" = "$have" ]; } \
            || {{ quote(just_executable()) }} --justfile {{ quote(justfile()) }} test-image-rebuild; \
    else \
        test -x "{{ VENV_BIN }}/{{ TOOL }}" \
            || { echo "{{ TOOL }} not found in {{ VENV_BIN }}/ — run 'just install' first." >&2; exit 1; }; \
    fi

# ── Checks (fast → slow) ─────────────────────────────────────────────────────
#
# One recipe per tool invocation, and `check` chains them. CI runs these as separately
# named steps so GitHub names the failing TOOL in the checks UI, not just "check".

# The formatter must run even when the linter still has unfixable findings, which a
# two-line recipe could not do: `ruff check --fix` exits non-zero while anything
# remains unfixed, and just abandons the recipe on the first failing line — so the
# recipe advertised as "REWRITES files" quietly rewrote nothing in exactly the
# situation you reach for it. Status is recorded and re-raised at the end so the
# lint-before-format order (and the failure) both survive.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Format and auto-fix src/ and tests/ (REWRITES files)')]
[group('check')]
fmt: (_tools 'ruff')
    #!/usr/bin/env bash
    set -uo pipefail
    {{ RUFF }} check --fix src/ tests/ || lint_rc=$?
    {{ RUFF }} format src/ tests/
    exit "${lint_rc:-0}"

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
#
# Shebang + "$@" rather than a plain line + {{ ARGS }}, because {{ ARGS }} flattens to
# one space-joined string: `just test -k "spotify or youtube"` reached pytest as
# `-k spotify or youtube`, i.e. `or` and `youtube` as test paths. The ~0.3s a shebang
# body costs on macOS is noise against a 13s suite. See `set positional-arguments`.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Run the test suite with coverage (~13s); extra pytest flags may be appended')]
[group('check')]
test *ARGS: (_tools 'pytest')
    #!/usr/bin/env bash
    set -euo pipefail
    {{ PYTEST }} --tb=short -q "$@"

# Check this file's own formatting (~0.01s)
[group('check')]
fmt-justfile:
    @{{ quote(just_executable()) }} --justfile {{ quote(justfile()) }} --fmt --check

# Both of these were enforced by a comment saying "keep these in step", which is not
# enforcement — and Dependabot is configured to move each half independently (the
# `pip` and `pre-commit` ecosystems open separate PRs). ruff in particular is
# exact-pinned precisely so the hook and CI agree; if the two drift, the commit hook
# reformats to a version `just fmt-check` then rejects, and you get a commit-then-fail
# loop with no explanation. Cheap enough to sit in `check`.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Assert the version/name pins duplicated across two files each (~0.02s)')]
[group('check')]
pins:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ quote(REPO) }}
    fail=0

    want_ruff="$(sed -n 's/^ruff = "\(.*\)"$/\1/p' pyproject.toml)"
    hook_ruff="$(sed -n 's|^ *rev: v\(.*\)$|\1|p' .pre-commit-config.yaml | head -1)"
    if [ -z "$want_ruff" ] || [ "$want_ruff" != "$hook_ruff" ]; then
        echo "ruff pin drift: pyproject.toml=[$want_ruff] .pre-commit-config.yaml rev=[v$hook_ruff]" >&2
        echo "  Bump both in the same commit." >&2
        fail=1
    fi

    just_image="$(sed -n 's/^IMAGE := "\(.*\)"$/\1/p' justfile)"
    sh_image="$(sed -n 's/^IMAGE_NAME="\(.*\)"$/\1/p' build_common.sh)"
    if [ -z "$just_image" ] || [ "$just_image" != "$sh_image" ]; then
        echo "image name drift: justfile IMAGE=[$just_image] build_common.sh IMAGE_NAME=[$sh_image]" >&2
        fail=1
    fi

    exit "$fail"

# What CI's lint and test jobs run — run this before pushing
#
# NOT the whole pipeline, and the difference has bitten: CI also runs `just --fmt
# --check` (now `fmt-justfile`, above, so this no longer omits it), the container-test
# job (`just ci` adds it), the `build` job, and security.yml's lockfile audit. The last
# two have no local equivalent — a green `check` does not promise a green PR.
#
# `python -m compileall src/` used to run in CI and is deliberately not reproduced
# here. It answered "does every file parse", which `ruff check`, pyright and pytest
# collection each already answer for the same file set — including modules nothing
# imports. Dropping it was intentional, not an oversight; this note exists because
# the diff that dropped it did not say so.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc("What CI's lint and test jobs run — run this before pushing")]
[group('check')]
check: fmt-justfile pins fmt-check lint types test

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
    {{ quote(just_executable()) }} --justfile {{ quote(justfile()) }} test \
        --cov-report=xml --junitxml=pytest.xml "$@" | tee pytest-coverage.txt

# Mirrors CI's container-test job. Its value is proving the IMAGE runs (a runtime stage
# missing a dependency is invisible to `just test`), which is why it is not part of
# `check`.
#
# [doc] and not a trailing `#` line — see the note on test-report.
[doc('Build the test image and run the suite inside it')]
[group('check')]
container-test: test-image-rebuild
    #!/usr/bin/env bash
    set -euo pipefail
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
    # Assigned first, not inlined as an argument: a failing command substitution
    # inside an argument does not trip `set -e` (the caller's status is what counts),
    # so a git failure would have built and tagged `discord-music-bot:`.
    tag="$(git_sha_tag)"
    build_runtime_image "{{ IMAGE }}:latest" "{{ IMAGE }}:$tag"

# Deploy an already-built image; pass a git sha to roll back
[group('deploy')]
up TAG='':
    # Quoted: unquoted, `just up '*'` globbed against the repo root and `just up "a b"`
    # passed two arguments. Both ended at the deploy guard's refusal, but naming a tag
    # nobody asked for. Quoting means the empty default arrives as one EMPTY argument
    # rather than none, which is why deploy_docker.sh tests `-n "${1:-}"` and not `$#`.
    ./deploy_docker.sh "{{ TAG }}"

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
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose logs -f discord-music-bot "$@"

# Show compose service status
[group('deploy')]
ps:
    docker compose ps
