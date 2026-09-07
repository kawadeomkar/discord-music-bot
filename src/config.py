import math
import os
import subprocess
from enum import Enum
from typing import Final, Optional
from urllib.parse import unquote, urlsplit

# Read from the environment alone so importing runs no subprocess. main() may
# replace it before setup_telemetry(): read `config.ENVIRONMENT`, never import
# the name, or the value binds too early.
ENVIRONMENT: str = os.environ.get("ENVIRONMENT") or "development"


def infer_environment_from_git() -> Optional[str]:
    """Deploy-environment name from the git branch; None when there is no repo,
    no git binary, or a detached HEAD (reported as "HEAD"). Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch or branch == "HEAD":
        return None
    return "production" if branch == "main" else branch.replace("/", "-")[:50]


# Touched by a loop-resident task for the container HEALTHCHECK. Unset (the
# default outside Docker) skips the task.
LIVENESS_FILE: str = os.environ.get("LIVENESS_FILE", "")
LIVENESS_INTERVAL_SECS: float = float(os.environ.get("LIVENESS_INTERVAL_SECS", "15.0"))

NOW_PLAYING_UPDATE_INTERVAL_SECS: float = float(
    os.environ.get("NOW_PLAYING_UPDATE_INTERVAL_SECS", "3.0")
)


def _float_env(name: str, default: float, *, minimum: float) -> float:
    """Float knob from the environment; empty reads as unset. Non-finite is
    refused separately from the floor: `inf` never expires a dashboard deadline
    (the command then holds its concurrency slot forever) and a tick of 0 turns
    the driver's timed wait into a hot spin."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number; got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {raw!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


# Floor for every live-dashboard knob: small enough to stay a tuning knob, large
# enough that the driver's wait is always a real suspension.
_MIN_DASHBOARD_SECS: Final[float] = 0.05

# -ping's live-edit loop (src/dashboard.py). Constants because the driver reads
# them every tick.
PING_TICK_SECS: float = _float_env("PING_TICK_SECS", 1.0, minimum=_MIN_DASHBOARD_SECS)
PING_DEADLINE_SECS: float = _float_env(
    "PING_DEADLINE_SECS", 3.0, minimum=_MIN_DASHBOARD_SECS
)

# -debug's live-edit loop. The Postgres block brackets a 2s sampling window plus a
# Prometheus round trip (~2.2s floor), and a block past the deadline renders
# "timed out" rather than being retried, so keep the deadline well above that.
DEBUG_TICK_SECS: float = _float_env("DEBUG_TICK_SECS", 1.0, minimum=_MIN_DASHBOARD_SECS)
DEBUG_DEADLINE_SECS: float = _float_env(
    "DEBUG_DEADLINE_SECS", 8.0, minimum=_MIN_DASHBOARD_SECS
)

# How long -analytics waits for its chart before sending the card without one.
# Sized for the cold path; expiring is silent. It bounds the caller, not the
# worker — a ProcessPoolExecutor cannot cancel a running call.
# See docs/ARCHITECTURE.md#analytics-rendering.
ANALYTICS_RENDER_DEADLINE_SECS: float = _float_env(
    "ANALYTICS_RENDER_DEADLINE_SECS", 20.0, minimum=_MIN_DASHBOARD_SECS
)

# Higher floor than the dashboards: each tick is a Redis write per PLAYING guild,
# AOF-appended.
_MIN_HEARTBEAT_SECS: Final[float] = 0.5

# How often a playing guild records its playback position; a crash resumes at the
# last heartbeat, so at most this many seconds replay.
HEARTBEAT_INTERVAL_SECS: float = _float_env(
    "HEARTBEAT_INTERVAL_SECS", 3.0, minimum=_MIN_HEARTBEAT_SECS
)


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Integer knob from the environment; empty reads as unset. Negatives are
    refused: `-1` reads as "no limit" but `if not OUTBOX_MAX` is truthy for it
    and `depth <= OUTBOX_MAX` never holds, so the drainer would trim the whole
    outbox. The message names the variable because it surfaces with no logger."""
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


# Opt-in ceiling on the history outbox, in entries; 0 is unbounded (the
# durability contract). A cap destroys the OLDEST entries, which exist nowhere
# else, so every drop logs ERROR. Enforced after each drain batch and against
# entries a drainer is holding (ACKed first, see _enforce_cap): size it well above
# BATCH_SIZE × peak burst. ~625 B stored per entry on redis:7.
HISTORY_OUTBOX_MAX: int = _int_env("HISTORY_OUTBOX_MAX", 0)

# How many guilds on_ready restores at once; the rest wait their turn. Bounds
# the Redis connections recovery draws and staggers the voice connects, so a
# cold start scales with this number rather than with the guild count.
RECOVERY_CONCURRENCY: int = _int_env("RECOVERY_CONCURRENCY", 8, minimum=1)

# asyncpg statement_cache_size per connection. Set 0 behind a transaction-pooling
# PgBouncer, where each transaction lands on a backend that never saw the handle.
POSTGRES_STATEMENT_CACHE: int = _int_env("POSTGRES_STATEMENT_CACHE", 100)


def _parse_bool_env(name: str) -> bool:
    """Strict boolean knob: unset and empty are False, a typo (`=on`) raises
    rather than silently reading as off."""
    raw = os.environ.get(name)
    value = (raw or "").strip().lower()
    if not value:
        return False
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, or yes/no "
        f"(case-insensitive); got {raw!r}"
    )


def history_archive_enabled() -> bool:
    """The consent gate for the Postgres archive: POSTGRES_URL required, every
    play XADDed to history:outbox, the drainer running. Read at call time.
    setup_hook must read it before push_history does, because @_guild_op would
    turn a garbage value into one warning per song instead of a startup abort."""
    return _parse_bool_env("HISTORY_ARCHIVE_ENABLED")


def debug_mode_default() -> bool:
    """Process-wide default for debug mode (observation-only embed footers) for
    guilds that never ran `-debug --enable/--disable`; a persisted per-guild
    choice wins over it. Read once by MusicBot.__init__ so garbage aborts
    startup."""
    return _parse_bool_env("DEBUG_MODE")


def debug_prometheus_url() -> Optional[str]:
    """Prometheus holding this deployment's container metrics, which -debug's
    Postgres block reads CPU/memory from; None leaves that row at `n/a`."""
    return (os.environ.get("DEBUG_PROMETHEUS_URL") or "").strip() or None


def postgres_url() -> Optional[str]:
    """The archive DSN, or None. `or None` folds "exported but empty" (a blank
    .env line) into the absent case."""
    return os.environ.get("POSTGRES_URL") or None


# What docker-compose.yml falls back to when .env sets none. `.env` is the one
# supported place the real password is set; see
# docs/ARCHITECTURE.md#postgres-credential-handling.
DEFAULT_POSTGRES_PASSWORD: Final[str] = "password"


def using_default_postgres_password() -> bool:
    """True when POSTGRES_URL's userinfo carries DEFAULT_POSTGRES_PASSWORD.

    Parsed from the DSN because the bot rarely sees POSTGRES_PASSWORD itself.
    Scoped to the DSN shape this repo's tooling produces; it misses `?password=`
    in the query, an unescaped `@` in the password, and PGPASSWORD, none of which
    compose or `just run` emit. Never raises: it feeds a warning and a -ping row.
    """
    url = postgres_url()
    if not url:
        return False
    try:
        password = urlsplit(url).password
    except ValueError:
        return False
    # SplitResult.password is not percent-decoded; asyncpg decodes, so match it.
    return password is not None and unquote(password) == DEFAULT_POSTGRES_PASSWORD


def spotify_enabled() -> bool:
    """Both Spotify credentials present. Presence only: wrong credentials count
    as enabled and fail at the first API call."""
    return bool(
        os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")
    )


# Fetched by the startup credential probe (MusicBot.cog_load); any permanent
# track would do.
SPOTIFY_TEST_TRACK_ID = "4PTG3Z6ehGkBFwjybzWkR8"


class SpotifyStatus(Enum):
    """Whether the configured Spotify credentials actually work, resolved once at
    startup, so `_require_spotify` and `-ping` can say why links are unavailable."""

    DISABLED = "disabled"  # no credentials configured
    INVALID = "invalid"  # present but rejected by the live API
    ENABLED = "enabled"  # present and validated at startup
