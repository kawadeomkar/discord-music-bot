import os
import subprocess
import warnings
from enum import Enum
from typing import Final, Optional
from urllib.parse import unquote, urlsplit


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        branch = result.stdout.strip()
        if result.returncode == 0 and branch and branch != "HEAD":
            return branch
    except Exception:
        pass
    # TODO: A detached HEAD turns this advisory warning into a hard test failure.
    # `git rev-parse --abbrev-ref HEAD` prints "HEAD" in any detached checkout — which
    # includes every `git worktree add --detach` — so this branch is taken and the
    # RuntimeWarning is raised at import time. pyproject's `filterwarnings = ["error",
    # ...]` promotes it to an exception, so the whole suite dies at collection with a
    # message about git branch detection rather than anything about the tests. CI is
    # unaffected only because ci.yml sets ENVIRONMENT explicitly, short-circuiting
    # _parse() before this is ever reached; a developer running pytest from a worktree
    # gets no such rescue. Found when a review agent had to export ENVIRONMENT by hand
    # just to run the suite in a detached worktree.
    # Fix: skip the warning when the checkout is legitimately detached, or default
    # ENVIRONMENT for the test session in tests/conftest.py.
    warnings.warn(
        "Could not detect git branch; defaulting ENVIRONMENT to 'development'",
        RuntimeWarning,
        stacklevel=2,
    )
    return "development"


def _parse() -> str:
    raw = os.environ.get("ENVIRONMENT")
    if raw is not None:
        return raw
    branch = _git_branch()
    return "production" if branch == "main" else branch.replace("/", "-")[:50]


ENVIRONMENT: str = _parse()

NOW_PLAYING_UPDATE_INTERVAL_SECS: float = float(
    os.environ.get("NOW_PLAYING_UPDATE_INTERVAL_SECS", "3.0")
)

# -ping's live-edit loop tunables. Constants (not call-time reads) because the
# dashboard reads them every tick and they are deployment-shape settings, not
# feature switches. They live here rather than in ping.py so this module stays
# the one place to answer "what does the bot read from the environment?".
# See docs/PING_METADATA_PLAN.md §5.2/§8.
PING_TICK_SECS: float = float(os.environ.get("PING_TICK_SECS", "1.0"))
PING_DEADLINE_SECS: float = float(os.environ.get("PING_DEADLINE_SECS", "3.0"))

# Opt-in ceiling on the Postgres history outbox, in entries. 0 (the default)
# means unbounded, which IS the durability contract: an entry only leaves the
# outbox once Postgres has it. Operators who would rather lose the oldest
# un-archived plays than let a long Postgres outage grow the non-evictable list
# toward Redis' maxmemory can set a cap; the drainer logs every drop at ERROR.
# Sizing math is in the README's Postgres section (~350 B/entry).
HISTORY_OUTBOX_MAX: int = int(os.environ.get("HISTORY_OUTBOX_MAX", "0"))

# asyncpg's prepared-statement cache size, per connection. The default matches
# asyncpg's own. Set to 0 behind PgBouncer in transaction-pooling mode:
# prepared statements are per-connection state, and transaction pooling hands a
# different backend to each transaction, so a cached statement handle refers to
# something the new backend has never seen.
POSTGRES_STATEMENT_CACHE: int = int(os.environ.get("POSTGRES_STATEMENT_CACHE", "100"))


def postgres_url() -> Optional[str]:
    """The play-history archive's DSN, or None when unset.

    Read at call time (same rationale as spotify_enabled): the bot reads it
    once at startup, so there is no hot path to optimise, and tests can
    monkeypatch the environment per case instead of reloading the module.
    """
    return os.environ.get("POSTGRES_URL")


# The password docker-compose.yml falls back to when .env does not set one, so
# that `docker compose up` works with nothing configured but DISCORD_TOKEN. It
# is a convenience for first-run and a liability everywhere else, which is why
# the bot detects it and complains loudly rather than silently accepting it.
#
# It is only defensible because the compose postgres service publishes on
# 127.0.0.1 — an unauthenticated-in-practice database reachable from the network
# would not be an acceptable default at any level of warning.
DEFAULT_POSTGRES_PASSWORD: Final[str] = "password"


def using_default_postgres_password() -> bool:
    """True when the archive DSN still carries DEFAULT_POSTGRES_PASSWORD.

    Parsed out of POSTGRES_URL rather than read from POSTGRES_PASSWORD directly:
    the bot only ever sees the assembled DSN (compose builds it, and `just run`
    derives it), so the password variable is frequently absent from the bot's own
    environment even when it is set for compose. Reading the DSN is therefore the
    only check that sees what the bot will actually authenticate with.

    Never raises — it feeds a startup warning and a -ping row, and a malformed
    DSN is the archive's problem to report, not this function's.
    """
    url = postgres_url()
    if not url:
        return False
    try:
        password = urlsplit(url).password
    except ValueError:
        return False
    # unquote because SplitResult.password does NOT percent-decode, so a DSN
    # carrying %70assword would otherwise read as a different credential than
    # the identical one written literally.
    return password is not None and unquote(password) == DEFAULT_POSTGRES_PASSWORD


def history_redis_cutover() -> bool:
    """True once Postgres is the source of truth for play history and the
    Redis history list may demote to a bounded display cache (Phase C of
    docs/POSTGRES_HISTORY_PLAN.md).

    Off by default and deliberately a separate switch from POSTGRES_URL: the
    archive being *configured* is not the same as the archive being *complete*.
    Flipping this before `just db-backfill` has run trims away the only copy of
    every play older than HISTORY_CACHE_LIMIT. Runbook order is
    backfill → Phase B reads → this flag.
    """
    return os.environ.get("HISTORY_REDIS_CUTOVER", "").lower() in ("1", "true", "yes")


def spotify_enabled() -> bool:
    """True when Spotify-link support should be active — i.e. both Spotify
    credentials are present in the environment.

    Read at call time (not cached as a module constant) so the value tracks
    the live environment: tests monkeypatch these vars per case, and the bot
    process only ever reads them once at startup, so there is no hot path to
    optimise. Presence, not validity, is the gate here — credentials that are
    set but wrong still count as "enabled" and fail at the first Spotify API
    call with a clear error, rather than silently disabling the feature.
    """
    return bool(
        os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")
    )


# Spotify ID for Rick Astley — "Never Gonna Give You Up". A well-known, permanent
# public track used purely as a probe: on startup, if Spotify credentials are
# present, the bot fetches this track once to confirm the credentials actually
# authenticate against the live API (see MusicBot.cog_load). Any real track ID
# would do; this one is memorable and unlikely to ever be removed.
# Source URL: https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8
SPOTIFY_TEST_TRACK_ID = "4PTG3Z6ehGkBFwjybzWkR8"


class SpotifyStatus(Enum):
    """Runtime state of the Spotify source, resolved once at startup.

    `spotify_enabled()` answers only "are credentials present?". This goes a step
    further and records whether those credentials actually work, so both the
    play-time gate (`MusicBot._require_spotify`) and `-ping` can tell a user
    *why* Spotify links are unavailable.
    """

    # No credentials configured — Spotify support was never turned on.
    DISABLED = "disabled"
    # Credentials are present but Spotify rejected them (or the startup probe
    # failed): links are declined with an "invalid credentials" message.
    INVALID = "invalid"
    # Credentials are present and validated against the live API at startup.
    ENABLED = "enabled"
