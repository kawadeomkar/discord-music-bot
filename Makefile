# Developer task index: one verb per target.
#
# Why this exists: build.sh used to run lint + tests + image build + deploy as one
# non-negotiable sequence, so "just lint" — 0.13s of actual ruff work — cost a
# Docker image build and two container starts, and deploying meant re-running
# everything. Multi-step *pipelines* still live in the .sh scripts; this file is
# the index over the primitives they compose.
#
# `make check` is the contract: if it passes, CI passes. Its steps are kept
# identical to .github/workflows/ci.yml's lint and test jobs — any divergence
# here is a "green locally, red in CI" bug in waiting.

.DEFAULT_GOAL := help

# Call the venv's binaries directly rather than `poetry run`: poetry re-resolves
# the project on every invocation, which costs ~1.4s and dwarfs ruff's 0.13s of
# real work.
#
# Which venv, though, is not obvious on a dev box. pyenv-virtualenv auto-activates
# this project's env from .python-version and exports VIRTUAL_ENV — and poetry
# honours an already-activated env over poetry.toml's in-project setting, so
# `poetry install` lands THERE, not in ./.venv. Following VIRTUAL_ENV when it is
# set keeps `make install` and `make lint` pointed at the same interpreter; the
# ./.venv fallback is what CI (which caches that path) and the Dockerfile use.
# Both branches are pure Make — no subprocess, so `make help` stays instant.
VENV_BIN := $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin,.venv/bin)
IMAGE    := discord-music-bot

# No GIT_SHA variable here on purpose: the tag comes from build_common.sh's
# git_sha_tag (which appends `-dirty` when the tree has uncommitted changes), so
# there is one definition of "what tag identifies this build". A `:=` here would
# also fork git on EVERY make invocation, `make help` included.

# ── Where the tools run: local venv (default) or the test image (DOCKER=1) ────
#
#   make check          native, fast — needs Python, Poetry and the venv
#   make check DOCKER=1 same checks inside the image — needs ONLY Docker
#
# DOCKER=1 exists so the project can be handed to someone who has nothing but
# Docker installed. The checks are the same commands either way; only the
# interpreter they run under differs.
ifeq ($(DOCKER),1)

# Mount src/ and tests/ as SUBDIRECTORIES, never the repo root. The image keeps
# its virtualenv at /app/.venv and puts it on PATH, so `-v $(CURDIR):/app` would
# shadow the venv and every tool below would vanish. pyproject.toml is mounted
# read-only so ruff/pytest/pyright read the working tree's config rather than the
# copy baked into the image.
DOCKER_MOUNTS := -v "$(CURDIR)/src:/app/src" \
                 -v "$(CURDIR)/tests:/app/tests" \
                 -v "$(CURDIR)/pyproject.toml:/app/pyproject.toml:ro"

# Two run modes, and the difference is not cosmetic:
#
#   as the host uid  ruff REWRITES the mounted files. Running as root would leave
#                    them root-owned on the host, which is how a formatter turns
#                    into a permissions incident. pyright only reads, but runs
#                    here too so nothing in this group can write as root.
#   as root          pytest writes .pytest_cache and coverage data into /app,
#                    which is image-owned and NOT mounted — the host uid cannot
#                    write there and pytest fails. Nothing it writes escapes the
#                    container, so root is safe for it specifically.
DOCKER_RUN      := docker run --rm $(DOCKER_MOUNTS) $(IMAGE):test
DOCKER_RUN_USER := docker run --rm --user "$$(id -u):$$(id -g)" $(DOCKER_MOUNTS) $(IMAGE):test

RUFF      := $(DOCKER_RUN_USER) ruff
PYRIGHT   := $(DOCKER_RUN_USER) pyright
PYTEST    := $(DOCKER_RUN) pytest
TOOLS_DEP := test-image

else

RUFF      := $(VENV_BIN)/ruff
PYRIGHT   := $(VENV_BIN)/pyright --pythonpath $(VENV_BIN)/python
PYTEST    := $(VENV_BIN)/pytest
TOOLS_DEP := _venv

endif

.PHONY: help install fmt lint types test check container-test ci image up down \
        restart logs ps hooks hooks-update hooks-run test-image test-image-rebuild

help: ## Show available targets
	@echo "Targets (run 'make <target>'):"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Add DOCKER=1 to run fmt/lint/types/test inside the test image instead"
	@echo "of a local venv — requires only Docker, no Python or Poetry."

# ── Setup ────────────────────────────────────────────────────────────────────

install: ## Create the in-project venv with main + test + lint dependencies
	poetry install --with test,lint

hooks: _venv ## Install the git hooks (ruff on commit, `make check` on push)
	$(VENV_BIN)/pre-commit install

hooks-update: _venv ## Bump pinned hook revisions in .pre-commit-config.yaml
	$(VENV_BIN)/pre-commit autoupdate

hooks-run: _venv ## Run every hook against every file (not just staged ones)
	$(VENV_BIN)/pre-commit run --all-files

# Fail with an actionable message rather than "No such file or directory".
.PHONY: _venv
_venv:
	@test -x $(VENV_BIN)/ruff || { \
	    echo "No usable venv at $(VENV_BIN)/ — run 'make install' first." >&2; \
	    exit 1; }

# The DOCKER=1 counterpart of _venv: make sure the image the tools live in exists.
#
# Built only when ABSENT, not when stale — src/ and tests/ are bind-mounted, so
# ordinary code changes need no rebuild. Dependency changes do: after touching
# pyproject.toml or poetry.lock, run `make test-image-rebuild` (or the always-
# rebuilding `make container-test`) or DOCKER=1 keeps using the old dependency set.
.PHONY: test-image test-image-rebuild
test-image:
	@docker image inspect $(IMAGE):test >/dev/null 2>&1 || $(MAKE) --no-print-directory test-image-rebuild

test-image-rebuild: ## Rebuild the test image DOCKER=1 uses (needed after a dependency change)
	@bash -c 'source ./build_common.sh && resolve_environment && \
	    docker build --build-arg ENVIRONMENT="$$ENVIRONMENT" -t "$(IMAGE):test" --target test -f Dockerfile .'

# ── Checks (fast → slow) ─────────────────────────────────────────────────────

fmt: $(TOOLS_DEP) ## Format and auto-fix src/ and tests/ (REWRITES files)
	$(RUFF) check --fix src/ tests/
	$(RUFF) format src/ tests/

lint: $(TOOLS_DEP) ## Check formatting + lint rules, no rewrites (~0.1s) — CI's ruff steps
	$(RUFF) format --check src/ tests/
	$(RUFF) check src/ tests/

# --pythonpath is not optional here. pyright resolves imports from the interpreter
# it is TOLD about, not the one it runs from: with `[tool.pyright] venvPath/venv`
# it read ./.venv, which on a pyenv box is a different (and stale) environment
# from the $VIRTUAL_ENV that `make install`, `make lint` and `make test` all use —
# so `make types` type-checked against a package set the other targets never saw.
# Those keys are gone from pyproject.toml; this flag replaces them, and it points
# at exactly the same VENV_BIN as every other target. Worse than wrong, the old
# setup failed SILENTLY: a missing .venv makes pyright warn and exit 0.
types: $(TOOLS_DEP) ## Type-check src/ AND tests/ with pyright (~6s)
	$(PYRIGHT)

test: $(TOOLS_DEP) ## Run the test suite with coverage (~13s)
	$(PYTEST) --tb=short -q

check: lint types test ## Everything CI gates on — run this before pushing

# Mirrors CI's container-test job. Its value is proving the IMAGE runs (a runtime
# stage missing a dependency is invisible to `make test`), which is why it is not
# part of `check`.
container-test: ## Build the test image and run the suite inside it
	@bash -c 'source ./build_common.sh && resolve_environment && \
	    docker build --build-arg ENVIRONMENT="$$ENVIRONMENT" -t "$(IMAGE):test" --target test -f Dockerfile . && \
	    docker run --rm "$(IMAGE):test"'

ci: check container-test ## Full local mirror of the CI workflow

# ── Image and deployment ─────────────────────────────────────────────────────
#
# The gate belongs to the *pipeline* (./build_docker.sh), never to these
# primitives: a gate you cannot skip is a gate you route around.

image: ## Build the runtime image as :latest and :<git-sha> — no test gate
	@bash -c 'source ./build_common.sh && resolve_environment && \
	    build_runtime_image "$(IMAGE):latest" "$(IMAGE):$$(git_sha_tag)"'

up: ## Deploy the already-built image for HEAD (rollback: ./deploy_docker.sh <sha>)
	./deploy_docker.sh

down: ## Stop the compose stack (volumes are kept)
	docker compose down

# NOT a deploy. `docker compose restart` stops and starts the EXISTING container
# with the image it already has, so a newly built image is not picked up — the
# old help text said "recreate", which sent `make image && make restart` down a
# path that silently kept running the old code. Use `make up` to deploy.
restart: ## Restart the running bot in place — does NOT pick up a new image (use `make up`)
	docker compose restart discord-music-bot

logs: ## Follow the bot's logs
	docker compose logs -f discord-music-bot

ps: ## Show compose service status
	docker compose ps
