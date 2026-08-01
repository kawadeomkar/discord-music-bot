import os
import subprocess
import warnings
from enum import Enum
from typing import Optional


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


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Parse an integer knob from the environment, or raise a named error.

    Three behaviors, each bought by a specific failure:

    Empty reads as unset. `HISTORY_OUTBOX_MAX=` — the bare `KEY=` shape
    .env.example already models for POSTGRES_PASSWORD — otherwise raises
    ValueError here at IMPORT time, which is before main() has run
    setup_telemetry(), so structlog and OTel are not configured yet. Under
    compose (`env_file: .env`, `restart: always`) that is an unstructured
    traceback on stderr and an infinite restart loop with nothing in Loki.
    Same rule postgres_url() applies to a blank DSN.

    Negatives are refused rather than passed through. `-1` is the near-universal
    idiom for "no limit", so it is exactly what an operator reaches for to spell
    out HISTORY_OUTBOX_MAX's default unbounded behavior — and it means the
    opposite downstream: the drainer's cap check is `if not OUTBOX_MAX: return`
    (-1 is truthy) followed by `if depth <= OUTBOX_MAX: return` (never true for
    a negative), so `dropped = depth - OUTBOX_MAX` = depth + 1 and the trim
    takes the entire outbox. That runs on the drain success path too, so a
    healthy system would wipe every un-archived play roughly every 30s, and the
    only signal would be an ERROR line claiming a cap was exceeded.

    The message names the variable because it surfaces with no logger attached:
    a bare `invalid literal for int() with base 10: ''` does not say which of
    the environment's variables is at fault.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


# Opt-in ceiling on the Postgres history outbox, in entries. 0 (the default)
# means unbounded, which IS the durability contract: an entry only leaves the
# outbox once Postgres has it. Operators who would rather lose the oldest
# un-archived plays than let a long Postgres outage grow the non-evictable
# stream toward Redis' maxmemory can set a cap; the drainer logs every drop at
# ERROR.
# A drop here is UNRECOVERABLE, and that is worth knowing before setting a cap.
# The cap destroys the OLDEST outbox entries, and the guild:{id}:history list is
# capped at HISTORY_CACHE_LIMIT, so only the newest few plays per guild have a
# second copy at all. Anything older that the cap discards existed only in the
# outbox and is gone for good — there is no list to re-read it from.
# An outage is not the only thing that trips it: the cap is enforced on the
# drain SUCCESS path too, evaluated after each 100-entry batch, so a healthy
# drainer working through a burst backlog also trims — discarding rows the very
# next cycle would have inserted milliseconds later. Set a cap well above
# BATCH_SIZE x your peak burst, not just above steady-state depth.
# The cap deliberately destroys entries a drainer is already holding, ACKing
# them first so the trim cannot leave them replaying forever. It has to: during
# an outage the drain re-reads the same pending batch every tick, so the OLDEST
# entries are permanently in flight and a cap that refused to cross them would
# never fire at all. See HistoryOutboxDrainer._enforce_cap.
# Sizing: a played-song entry measures ~420 bytes on the wire and ~487 bytes
# stored (MEMORY USAGE ... SAMPLES 0 against redis:7-alpine at 100k stream
# entries), so the compose Redis' 256mb budget holds roughly 525k un-archived
# plays before the non-evictable key becomes the thing that fills it. Redis 8
# stores the same payload in ~424 bytes; measure on the major you deploy.
HISTORY_OUTBOX_MAX: int = _int_env("HISTORY_OUTBOX_MAX", 0)

# asyncpg's prepared-statement cache size, per connection. The default matches
# asyncpg's own. Set to 0 behind PgBouncer in transaction-pooling mode:
# prepared statements are per-connection state, and transaction pooling hands a
# different backend to each transaction, so a cached statement handle refers to
# something the new backend has never seen.
POSTGRES_STATEMENT_CACHE: int = _int_env("POSTGRES_STATEMENT_CACHE", 100)


def postgres_url() -> Optional[str]:
    """The play-history archive's DSN, or None when unset.

    Read at call time (same rationale as spotify_enabled): the bot reads it
    once at startup, so there is no hot path to optimise, and tests can
    monkeypatch the environment per case instead of reloading the module.

    `or None` collapses "unset" and "exported but empty" into one sentinel, so
    the Optional[str] above is honest and every caller has exactly one
    absent-case to handle. A blank line in .env yields "", which is not None but
    is not a DSN either; without this, whether that is caught depends on each
    caller spelling its check as truthiness rather than `is None`. It is not
    load-bearing for any caller today — setup_hook's required-DSN guard is
    `if not postgres_url:` and rejects "" on its own — and that is the point:
    the type should not depend on remembering which form the caller used.
    """
    return os.environ.get("POSTGRES_URL") or None


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
