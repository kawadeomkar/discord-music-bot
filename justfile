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
# migrations/ is mounted because db_migrate.discover() reads that directory at
# RUNTIME and a test asserts it agrees with EXPECTED_SCHEMA_VERSION. Baked in
# only, a schema change passes `just test` and fails `DOCKER=1 just test`
# against the stale copy — and dep_hash() covers only poetry.lock/pyproject.toml,
# so no rebuild is triggered to explain it.
DOCKER_MOUNTS := '-v "' + REPO + '/src:/app/src" -v "' + REPO + '/tests:/app/tests" -v "' + REPO + '/migrations:/app/migrations:ro" -v "' + REPO + '/pyproject.toml:/app/pyproject.toml:ro"'

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

# The real-Postgres tier, excluded from `test` by its `pg` marker.
#
# Server comes from one of two places, decided by the tests themselves: with
# POSTGRES_TEST_URL set it uses that server (how CI's pg-integration job points
# at its service container), otherwise testcontainers starts postgres:18-alpine
# and Docker must be running. RUN_PG_TESTS=1 is set here so the local invocation
# is just `just test-pg`; the tests also accept POSTGRES_TEST_URL alone.
#
# --no-cov, and not as a shortcut: this tier drives SQL against a real server
# rather than exercising src/ branches, so measuring it under the 80% gate would
# fail the run on a coverage number that means nothing for what it tests.
[doc('Run the real-Postgres integration tier (needs Docker, or POSTGRES_TEST_URL)')]
[group('check')]
test-pg *ARGS: (_tools 'pytest')
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_PG_TESTS=1 {{ PYTEST }} -m pg --no-cov --tb=short -q "$@"

# Opt-in real-Redis tier (testcontainers; needs Docker)
#
# fakeredis executes every stream command the outbox uses and gets FIVE of them
# wrong — all in the safe-looking direction, so the default suite stays green
# while production breaks. See tests/test_redis_integration.py's docstring.
#
# [doc] is not decoration here: without it `just --list` shows the LAST line of
# the comment block above as this recipe's description, which is a docs pointer
# rather than a description. Its sibling test-pg carries both attributes.
[doc('Run the real-Redis integration tier (needs Docker, or REDIS_TEST_URL)')]
[group('check')]
test-redis *ARGS: (_tools 'pytest')
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_REDIS_TESTS=1 {{ PYTEST }} -m redis --no-cov --tb=short -q "$@"

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

    # The pg tier runs against testcontainers locally and a CI service container
    # on GitHub, and the two must be the same server: a schema or type behaviour
    # that differs between majors would pass in one place and fail in the other,
    # with nothing pointing at the version as the cause.
    py_pg="$(sed -n 's/^_PG_IMAGE = "\(.*\)"$/\1/p' tests/test_pg_integration.py)"
    # Anchored to the pg-integration job, not `head -1`: the first
    # `image: postgres:*` in the file happens to be that job's today, and a
    # second postgres service added anywhere above it would silently start
    # comparing the wrong one.
    ci_pg="$(awk '/^  pg-integration:/{f=1} f && /image: postgres:/{print $2; exit}' .github/workflows/ci.yml)"
    if [ -z "$py_pg" ] || [ "$py_pg" != "$ci_pg" ]; then
        echo "postgres image drift: test_pg_integration.py=[$py_pg] ci.yml=[$ci_pg]" >&2
        echo "  Bump both in the same commit." >&2
        fail=1
    fi

    # ...and compose, which is the copy that holds real data. Checking only the
    # two test-side copies left the deployed server free to drift to another
    # major while `just pins` and CI stayed green — exactly the scenario above,
    # but validated against a server nobody runs in production.
    compose_pg="$(awk '/^  postgres:/{f=1} f && /image: postgres:/{print $2; exit}' docker-compose.yml)"
    if [ -z "$compose_pg" ] || [ "$compose_pg" != "$ci_pg" ]; then
        echo "postgres image drift: docker-compose.yml=[$compose_pg] ci.yml=[$ci_pg]" >&2
        echo "  Bump both in the same commit." >&2
        fail=1
    fi

    # Same rule for the redis tier, and this one carries live risk rather than
    # theoretical: `redis:7-alpine` FLOATS (7.4.9 today), Dependabot runs the
    # docker ecosystem at the repo root so it will bump the compose tag on its
    # own, and a hardcoded test image left behind would have the suite asserting
    # one server's behaviour while the bot runs another. That is not
    # hypothetical here — a memory measurement of this very design was taken
    # against the wrong major for exactly that reason.
    py_redis="$(sed -n 's/^_REDIS_IMAGE = "\(.*\)"$/\1/p' tests/test_redis_integration.py)"
    ci_redis="$(awk '/^  redis-integration:/{f=1} f && /image: redis:/{print $2; exit}' .github/workflows/ci.yml)"
    if [ -z "$py_redis" ] || [ "$py_redis" != "$ci_redis" ]; then
        echo "redis image drift: test_redis_integration.py=[$py_redis] ci.yml=[$ci_redis]" >&2
        echo "  Bump both in the same commit." >&2
        fail=1
    fi

    compose_redis="$(awk '/^  redis:/{f=1} f && /image: redis:/{print $2; exit}' docker-compose.yml)"
    if [ -z "$compose_redis" ] || [ "$compose_redis" != "$ci_redis" ]; then
        echo "redis image drift: docker-compose.yml=[$compose_redis] ci.yml=[$ci_redis]" >&2
        echo "  Bump both in the same commit." >&2
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
#
# test-pg and test-redis are here because CI's pg-integration and
# redis-integration jobs are merge gates (`build` needs both), so a green `ci`
# that skipped them would not mean what it says. They need Docker, which
# `container-test` already required of this recipe.
#
# [doc(...)] because `just --list` shows only the LAST comment line, so the
# multi-line reasoning above would otherwise replace this recipe's description
# with "needs Docker, which `container-test` already required of this recipe."
[doc('Full local mirror of CI (check + container-test + test-pg + test-redis)')]
[group('check')]
ci: check container-test test-pg test-redis

# ── Play-history database (Postgres) ─────────────────────────────────────────
#
# All of these assume the ARCHIVE IS ENABLED (HISTORY_ARCHIVE_ENABLED=true in
# .env) and its services are up. Against a disabled stack they fail at connect —
# curt, but honest: there is no database deployed to operate on.
#
# All of these read POSTGRES_URL (or POSTGRES_MIGRATE_URL for db-migrate) from
# the environment, which for a compose deployment means `.env`. They run the
# LOCAL venv's python against whatever that URL points at — operator tools, not
# container commands, so they work the same against the bundled compose Postgres
# and an external one. Never routed through DOCKER=1: that switch exists to run
# the *checks* without a Python toolchain, and pointing a migration at a
# database is not a check.
#
# A value already in the environment WINS over .env, which is why this reads the
# file itself instead of `set -a; . ./.env` — sourcing assigns unconditionally,
# so `POSTGRES_URL=…staging just db-migrate` would migrate the LOCAL database
# and report success.
#
# The reader is deliberately conservative: first `=` splits (so DSN query
# strings survive), `#`/blank lines and non-identifier keys are skipped, a
# trailing CR is stripped (CRLF .env files), and one layer of matched quotes is
# removed to match what sourcing would have done.
_dotenv := '''
    set -euo pipefail
    if [ -f .env ]; then
        while IFS='=' read -r _k _v || [ -n "$_k" ]; do
            case "$_k" in ''|'#'*) continue ;; esac
            case "$_k" in *[!A-Za-z0-9_]*) continue ;; esac
            [ -n "${!_k+x}" ] && continue
            _v="${_v%$'\r'}"
            case "$_v" in
                \"*\") _v="${_v#\"}"; _v="${_v%\"}" ;;
                \'*\') _v="${_v#\'}"; _v="${_v%\'}" ;;
            esac
            export "$_k=$_v"
        done < .env
    fi
    # Build a HOST dsn from the compose parts when POSTGRES_URL is unset. The
    # bundled stack synthesises the bot's URL inside compose (docker-compose.yml
    # builds it from POSTGRES_USER/PASSWORD/DB), so it never lands in .env — and
    # without this every recipe below died with "POSTGRES_URL is not set" on the
    # exact stack the README tells you to run, including db-backfill, which is
    # the mandatory step before the history lists are capped.
    #
    # The password FALLS BACK, exactly as compose does. It did not, and that was
    # a lockout generator rather than a missing convenience: on the stack the
    # default now enables (only DISCORD_TOKEN in .env) every recipe here died
    # with "POSTGRES_URL is unset" and advised ./setup_env.sh — which mints a
    # NEW password, while the postgres volume was already initialised on the
    # default. The next `just run` then built a DSN the database rejects, and
    # because the archive connects lazily the bot started fine and surfaced it
    # much later as a drainer backoff loop. That is precisely the two-step trap
    # .env.example and the -ping advisory warn about, reached by following this
    # file's own advice. Keep this default in step with docker-compose.yml's.
    if [ -z "${POSTGRES_URL:-}" ]; then
        export POSTGRES_URL="postgresql://${POSTGRES_USER:-musicbot}:${POSTGRES_PASSWORD:-password}@127.0.0.1:${POSTGRES_HOST_PORT:-5432}/${POSTGRES_DB:-musicbot}"
    fi
'''

# Bootstrap .env with a generated POSTGRES_PASSWORD — the first command a new
# contributor runs.
[doc('Create/refresh .env with a generated Postgres password')]
[group('dev')]
setup *ARGS:
    ./setup_env.sh {{ ARGS }}

# Start what `just run` connects to: Redis always, Postgres and the migration
# one-shot only when HISTORY_ARCHIVE_ENABLED says so. Replaces having to
# remember which of two `docker compose up` lines matches your .env.
[doc('Start the backing services for `just run` (Postgres only when the archive is enabled)')]
[group('dev')]
services:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_archive_profile
    if [ "$ARCHIVE_ENABLED" -eq 1 ]; then
        docker compose up -d redis postgres db-migrate
    else
        docker compose up -d redis
    fi

# Escape hatch for compose commands this file does not wrap, with the archive
# profile resolved from the flag: `just compose ps`, `just compose logs postgres`.
# A raw `docker compose` still works — it just never deploys the archive tier.
[doc('Run `docker compose` with the archive profile derived from HISTORY_ARCHIVE_ENABLED')]
[group('deploy')]
compose *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_archive_profile
    docker compose "$@"

# Run the bot against the compose-backed services.
#
# `poetry run bot` alone does NOT work from a fresh clone: the bot reads its
# configuration from the ENVIRONMENT and has no .env support, so the documented
# local-run flow died on missing configuration and pointed at a setup_env.sh
# that could not fix it. This recipe supplies the same environment the db-*
# recipes use (.env, then a host DSN built from the compose parts).
#
# POSTGRES_URL is required only while HISTORY_ARCHIVE_ENABLED=true — the archive
# is opt-in and OFF by default, and a disabled bot ignores the DSN entirely
# (setup_hook logs one INFO saying so). The recipe derives one either way
# because it costs nothing and makes flipping the flag a one-line change.
# Bring the services up first with `just services` — it reads the same flag and
# starts Postgres only when the archive is enabled.
[doc('Run the bot locally with .env loaded (services must already be up)')]
[group('dev')]
run:
    #!/usr/bin/env bash
    {{ _dotenv }}
    # _dotenv always derives a DSN now (the password falls back like compose's),
    # so this only fires if someone exported an empty POSTGRES_URL by hand — and
    # only then does it matter, because the bot needs the DSN just while the
    # archive is opted in. Refusing unconditionally would block the DEFAULT
    # configuration, which archives nothing and ignores the variable entirely.
    if [ -z "${POSTGRES_URL:-}" ]; then
        case "$(printf '%s' "${HISTORY_ARCHIVE_ENABLED:-}" | tr '[:upper:]' '[:lower:]')" in
            true | 1 | yes)
                echo "POSTGRES_URL is empty but HISTORY_ARCHIVE_ENABLED is true — setup_hook will refuse to start." >&2
                echo "Unset POSTGRES_URL to let .env supply one, or set it." >&2
                exit 1
                ;;
        esac
    fi
    exec {{ quote(VENV_BIN / 'python') }} -m src.main

# Apply pending schema migrations — the bot refuses to start against an
# unmigrated database (PostgresHistoryArchive._assert_schema_version).
#
# Every deploy already runs this (deploy_docker.sh, before the bot is
# recreated), so this recipe is for external databases and out-of-band runs.
# Re-running applies nothing: versions are recorded in schema_migrations.
[doc('Apply pending play-history schema migrations')]
[group('database')]
db-migrate:
    #!/usr/bin/env bash
    {{ _dotenv }}
    {{ quote(VENV_BIN / 'python') }} -m src.db_migrate

# Copy pre-archive history off the Redis lists into Postgres. Idempotent and
# resumable (ON CONFLICT DO NOTHING). MUST run before the change that caps the
# history lists, which trims the only other copy of exactly what this moves.
#
# This leg needs a local venv. On a Docker-only host use `just db-backfill-docker`
# below — the same image and the same module.
[doc('Backfill pre-archive Redis history into Postgres (--dry-run to preview)')]
[group('database')]
db-backfill *ARGS:
    #!/usr/bin/env bash
    {{ _dotenv }}
    {{ quote(VENV_BIN / 'python') }} -m src.backfill_history "$@"

# The container leg of db-backfill, for hosts with Docker and no Python toolchain.
#
# The profile resolution is what makes it work: `docker compose run` activates
# only the target's own `ops` profile, so db-backfill's `depends_on: postgres`
# is unresolvable ("no such service: postgres") unless `archive` is active too.
# Disabled, this refuses up front rather than surfacing that as a compose error.
[doc('Backfill via the compose one-shot — no local venv needed')]
[group('database')]
db-backfill-docker *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    source ./build_common.sh
    resolve_archive_profile
    if [ "$ARCHIVE_ENABLED" -ne 1 ]; then
        echo "The history archive is disabled — there is no database to backfill into." >&2
        echo "Set HISTORY_ARCHIVE_ENABLED=true in .env and deploy first (just up)." >&2
        exit 1
    fi
    # Same reason as the deploy's migration step: the service's DSN is the
    # compose-network one, and .env's POSTGRES_URL is the host form.
    resolve_external_postgres_env
    docker compose run --rm ${EXTERNAL_PG_ENV[@]+"${EXTERNAL_PG_ENV[@]}"} db-backfill "$@"

# The one Redis-side operator recipe. It exists because XLEN alone cannot tell
# apart four states that call for four different responses, and working that out
# by hand during an incident is the wrong time to learn the commands:
#
#   undelivered        backlog the drainer has not read yet
#   in flight          delivered, insert not committed — normal, unless it is old
#   acked, undeleted   a crash between XACK and XDEL — harmless, self-clearing
#   TOMBSTONED         body trimmed while still pending — these are LOST PLAYS
#
# The last one is the reason this is a recipe rather than a README snippet:
# spotting it means cross-referencing every pending ID against XRANGE, and it is
# the only state that is silent everywhere else — no Postgres row, no
# play_history_rejected row, no log line naming the entry.
[doc('Outbox health: depth, in-flight, stranded entries, lost plays')]
[group('database')]
outbox IDLE_MS='60000':
    #!/usr/bin/env bash
    set -uo pipefail
    r() { docker compose exec -T redis redis-cli "$@" | tr -d '\r'; }
    key=history:outbox
    group=drainers

    kind="$(r TYPE "$key")"
    if [ "$kind" = "none" ]; then
        echo "outbox: key absent — nothing buffered, nothing to do."
        exit 0
    fi
    if [ "$kind" != "stream" ]; then
        # Startup aborts on this, but only the bot sees that; an operator running
        # this recipe should be told the same thing in the same words.
        echo "outbox: WRONGTYPE — '$key' is a $kind, not a stream." >&2
        echo "  A pre-R1 build left a list here. Stop the bot, DEL the key, start it." >&2
        exit 1
    fi

    depth="$(r XLEN "$key")"
    echo "depth (XLEN):        $depth   # entries whose plays are not in Postgres yet"

    summary="$(r XPENDING "$key" "$group" 2>&1)"
    case "$summary" in
        NOGROUP*|*"No such key"*)
            echo "group '$group': MISSING — every read fails NOGROUP." >&2
            echo "  The drainer recreates it on its next tick; if depth is not falling," >&2
            echo "  the bot is down or wedged." >&2
            exit 1 ;;
    esac
    in_flight="$(printf '%s\n' "$summary" | sed -n 1p)"
    echo "in flight (XPENDING): ${in_flight:-0}   # delivered, insert not yet committed"

    # Idle time is the tell for "stranded" — compare against DRAIN_DEADLINE_SECS.
    stranded="$(r XPENDING "$key" "$group" IDLE {{ IDLE_MS }} - + 10)"
    if [ -n "$stranded" ]; then
        echo
        echo "STRANDED — pending longer than {{ IDLE_MS }}ms (id / consumer / idle / deliveries):"
        printf '%s\n' "$stranded" | paste - - - - | sed 's/^/  /'
    fi

    # Tombstones: pending IDs whose bodies are gone. XPENDING's range form prints
    # four lines per entry (id, consumer, idle, delivery-count), so every fourth
    # line from the first is an ID.
    lost=0
    for id in $(r XPENDING "$key" "$group" - + 1000 | awk 'NR % 4 == 1'); do
        if [ -z "$(r XRANGE "$key" "$id" "$id")" ]; then
            [ "$lost" -eq 0 ] && echo && echo "LOST PLAYS — pending, but the body is gone (tombstones):"
            echo "  $id"
            lost=$((lost + 1))
        fi
    done
    if [ "$lost" -gt 0 ]; then
        echo "  $lost entr$([ "$lost" -eq 1 ] && echo y || echo ies) trimmed while in flight." >&2
        echo "  Each is a play that reached no table. Check HISTORY_OUTBOX_MAX." >&2
        exit 1
    fi
    echo
    echo "head (oldest 3):"
    r XRANGE "$key" - + COUNT 3 | sed 's/^/  /'

# Rows Postgres refused, parked by record_rejection. Expected to print NOTHING:
# every entry reaching the drainer is insertable by construction, so a row here
# means the HistoryEntry validator regressed or this build is talking to a schema
# it was not written for. Treat any output as a code defect, not a data problem.
#
# encode(payload, 'escape') rather than convert_from(payload, 'UTF8'): payload is
# bytea precisely because it may hold a NUL or invalid UTF-8, and convert_from
# raises on exactly those (the play_history_rejected block in
# migrations/0001_play_history.sql has the reasoning).
[doc('List play_history rows Postgres refused (expected: nothing)')]
[group('database')]
db-rejects COUNT='10':
    #!/usr/bin/env bash
    {{ _dotenv }}
    docker compose exec -T postgres psql -U "${POSTGRES_USER:-musicbot}" \
        -d "${POSTGRES_DB:-musicbot}" -x -c \
        "SELECT id, rejected_at, guild_id, error_type, error_detail, trace_id,
                encode(payload, 'escape') AS payload
         FROM play_history_rejected
         ORDER BY rejected_at DESC LIMIT {{ COUNT }}"

# Custom-format dump (-Fc): compressed and restorable selectively, unlike plain
# SQL. Writes into backups/, which is gitignored.
[doc('Dump the play-history database to backups/')]
[group('database')]
db-backup:
    #!/usr/bin/env bash
    {{ _dotenv }}
    # A dump is the whole play_history table — every guild id, user id and song
    # title the bot has ever recorded — and the shell's redirect below creates it
    # under the ambient umask, which is 022 on a stock macOS or Ubuntu account.
    # That published the entire history to every local account, in a directory
    # (`backups/`) that then accumulates them. Set here rather than chmod'ing
    # after the fact so the file is never briefly world-readable, and so the
    # `backups/` mkdir is covered by the same rule.
    umask 077
    mkdir -p backups
    out="backups/play_history_$(date +%F_%H%M%S).dump"
    # Dump to a temp file and rename only on success. Redirecting straight into
    # "$out" creates it BEFORE pg_dump runs, so a failed dump left a 0-byte
    # .dump behind — which the README's prune (`ls -1t | tail -n +15`) then
    # keeps as the newest backup while deleting a real older one.
    tmp="$out.partial"
    # -T: no TTY, or the dump arrives with \r\n line endings and is unusable.
    if docker compose exec -T postgres pg_dump -U "${POSTGRES_USER:-musicbot}" -Fc \
        "${POSTGRES_DB:-musicbot}" > "$tmp"; then
        mv "$tmp" "$out"
        echo "wrote $out"
    else
        rm -f "$tmp"
        echo "pg_dump failed; no backup written" >&2
        exit 1
    fi

# Restore a dump produced by db-backup.
#
# Defaults to a SCRATCH database, not the live one — this is the tool the README's
# quarterly restore drill reaches for, and a restore over live drops and reloads
# every row, losing every play since the dump. Overwriting live takes both an
# explicit name and CONFIRM=1.
#
# --clean --if-exists: replace objects rather than collide with existing ones.
# --single-transaction: a failed restore rolls back instead of leaving the
# target with its tables dropped and half its rows reloaded.
[doc('Restore a dump into a scratch DB (or DB=<name> CONFIRM=1 for the live one)')]
[group('database')]
db-restore FILE DB='':
    #!/usr/bin/env bash
    {{ _dotenv }}
    test -f {{ quote(FILE) }} || { echo "no such dump: {{ FILE }}" >&2; exit 1; }
    live="${POSTGRES_DB:-musicbot}"
    target={{ quote(DB) }}
    target="${target:-${live}_restore_check}"
    if [ "$target" = "$live" ] && [ "${CONFIRM:-}" != "1" ]; then
        echo "refusing to overwrite the live database '$live'." >&2
        echo "  drill:    just db-restore {{ FILE }}" >&2
        echo "            (restores into '${live}_restore_check', live untouched)" >&2
        echo "  for real: CONFIRM=1 just db-restore {{ FILE }} $live" >&2
        exit 1
    fi
    user="${POSTGRES_USER:-musicbot}"
    if [ "$target" != "$live" ]; then
        docker compose exec -T postgres createdb -U "$user" "$target" 2>/dev/null \
            || true  # already exists: --clean below replaces its contents
    fi
    echo "restoring {{ FILE }} -> $target"
    docker compose exec -T postgres pg_restore -U "$user" -d "$target" \
        --clean --if-exists --single-transaction < {{ quote(FILE) }}
    echo "restored into '$target'"

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
    # --profile archive is load-bearing, not decoration. `docker compose down`
    # with the profile INACTIVE removes only un-profiled containers and leaves
    # a running postgres behind (plus a "network in use" error) — exactly the
    # just-disabled-the-archive case where the operator most expects the
    # database to stop. Activating a profile whose services have no containers
    # is a no-op, so the always-disabled case is unaffected.
    docker compose --profile archive down

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
