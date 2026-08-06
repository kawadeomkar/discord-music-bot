import math
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
    # `git rev-parse --abbrev-ref HEAD` prints "HEAD" when detached (including every
    # `git worktree add --detach`), and pyproject's `filterwarnings = ["error", ...]`
    # promotes this RuntimeWarning to an import-time exception, killing the suite at
    # collection with an error about git branch detection rather than anything about
    # the tests. CI escapes only because ci.yml sets ENVIRONMENT explicitly.
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


def _float_env(name: str, default: float, *, minimum: float) -> float:
    """Parse a float knob from the environment, or raise a named error.

    Same empty-reads-as-unset rule as _int_env below, and the same reason for raising
    at import time. Non-finite is refused separately from the floor because `float()`
    accepts "inf" and "nan" happily and both defeat the dashboard driver in ways a
    minimum would not catch: `inf` makes the deadline never expire, so the command
    holds its max_concurrency slot forever and every later run in that guild answers
    "already running". A tick of 0 is the other half — it turns the driver's timed
    wait into a hot spin, measured at 0.6 CPU-seconds per wall-second on the loop
    that also carries voice heartbeats.
    """
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


# Floor for every live-dashboard knob. Small enough to stay a tuning knob rather than
# a policy, large enough that the driver's wait is always a real suspension.
_MIN_DASHBOARD_SECS: Final[float] = 0.05

# -ping's live-edit loop tunables. Constants, not call-time reads: the dashboard
# reads them every tick. Here rather than in ping.py so this module stays the one
# place that answers "what does the bot read from the environment?".
PING_TICK_SECS: float = _float_env("PING_TICK_SECS", 1.0, minimum=_MIN_DASHBOARD_SECS)
PING_DEADLINE_SECS: float = _float_env(
    "PING_DEADLINE_SECS", 3.0, minimum=_MIN_DASHBOARD_SECS
)

# The same two knobs for -debug's live-edit loop (src/dashboard.py drives both).
# A longer deadline than -ping's: these collectors do strictly more work per block
# — a Postgres stats query and a Prometheus round trip, against -ping's single
# reachability probe — and a block that misses the deadline renders "timed out"
# rather than being retried, so cutting it short loses real data.
DEBUG_TICK_SECS: float = _float_env("DEBUG_TICK_SECS", 1.0, minimum=_MIN_DASHBOARD_SECS)
DEBUG_DEADLINE_SECS: float = _float_env(
    "DEBUG_DEADLINE_SECS", 8.0, minimum=_MIN_DASHBOARD_SECS
)


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Parse an integer knob from the environment, or raise a named error.

    Empty reads as unset: the bare `KEY=` shape otherwise raises at IMPORT time,
    before setup_telemetry(), so under compose (`restart: always`) it is an
    unstructured traceback and an infinite restart loop with nothing in Loki.

    Negatives are refused. `-1` is the near-universal "no limit" idiom and means
    the OPPOSITE downstream: `if not OUTBOX_MAX` is truthy for -1 and `depth <=
    OUTBOX_MAX` never true, so the drainer's trim takes the entire outbox — on
    the success path too, wiping every un-archived play roughly every 30s. The
    message names the variable because it surfaces with no logger attached.
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


# Opt-in ceiling on the Postgres history outbox, in entries. 0 (the default) is
# unbounded, which is the durability contract — an entry only leaves the outbox
# once Postgres has it. A cap trades that for bounding a non-evictable key during
# a long outage; every drop logs at ERROR and is UNRECOVERABLE, since the cap
# destroys the oldest entries while guild:{id}:history is capped at
# HISTORY_CACHE_LIMIT, so anything older existed only here. Enforced on the drain
# success path too, after each 100-entry batch, and it deliberately destroys
# entries a drainer is holding, ACKing them first (see _enforce_cap) — so size it
# well above BATCH_SIZE x peak burst, not just above steady-state depth.
#
# Sizing: ~420 bytes on the wire, ~487 stored (MEMORY USAGE against
# redis:7-alpine at 100k stream entries), so 256mb holds roughly 525k un-archived
# plays. Redis 8 stores the same payload in ~424 bytes; measure on your major.
HISTORY_OUTBOX_MAX: int = _int_env("HISTORY_OUTBOX_MAX", 0)

# asyncpg's prepared-statement cache size, per connection; the default matches
# asyncpg's own. Set 0 behind PgBouncer in transaction-pooling mode: prepared
# statements are per-connection state and each transaction gets a different
# backend, so a cached handle refers to something that backend has never seen.
POSTGRES_STATEMENT_CACHE: int = _int_env("POSTGRES_STATEMENT_CACHE", 100)


def _parse_bool_env(name: str) -> bool:
    """Parse a boolean knob, or raise naming it. Unset and empty read as False.

    Parsing is STRICT because of the failure direction: a lenient
    anything-but-true-is-False rule turns a typo (`=on`) into an operator who
    believes they flipped the switch while nothing changed.
    """
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
    """True when the operator has opted in to the Postgres history archive.

    The consent gate for long-term storage. Enabled: POSTGRES_URL required at
    startup, every play XADDed to history:outbox, the drainer moving it into
    play_history forever. Disabled — the default — none of that exists; Redis
    behavior is identical either way. Read at call time (once per song at most).

    Parsing is STRICT (see _parse_bool_env) because of the failure direction: a
    typo (`HISTORY_ARCHIVE_ENABLED=on`) would otherwise leave an operator
    believing they enabled archiving while every play goes unrecorded. Unset and
    empty read as False — collection must be a choice.

    setup_hook must call this before any other consumer: the next reader is
    @_guild_op-wrapped push_history, where a garbage value becomes one warning per
    song instead of a startup abort.
    """
    return _parse_bool_env("HISTORY_ARCHIVE_ENABLED")


def debug_mode_default() -> bool:
    """The process-wide default for debug mode — what a guild gets before anyone
    runs `-debug --enable`.

    Debug mode is observation-only: it decorates responses with trace/timing
    metadata and nothing else, so this is safe to leave on. Read ONCE, by
    MusicBot.__init__, which is what makes a garbage value abort startup inside
    load_extension.

    This is the default for a guild that has never chosen, and only for as long as
    it has not: `-debug --enable/--disable` persists to guild:{id}:config and WINS
    over this value from then on, across restarts. Changing this env var moves every
    guild that never chose and none that did.

    Parsed by the same strict table as history_archive_enabled — unset and empty
    are False, and a typo raises rather than silently reading as off — so there is
    one boolean grammar in this file rather than two.
    """
    return _parse_bool_env("DEBUG_MODE")


def debug_prometheus_url() -> Optional[str]:
    """Base URL of a Prometheus that holds this deployment's container metrics, or
    None (the default) to leave the feature off.

    -debug's Postgres block reads container CPU/memory from here, because the bot
    cannot see another container's cgroup and Postgres reports no OS metrics over
    SQL. Compose supplies it once otel-lgtm's Prometheus port is published; unset,
    the row degrades to `n/a (no metrics source)` and nothing else changes.

    Read at call time, like postgres_url, and `or None` collapses unset and
    exported-but-empty into one absent case.
    """
    return (os.environ.get("DEBUG_PROMETHEUS_URL") or "").strip() or None


def postgres_url() -> Optional[str]:
    """The play-history archive's DSN, or None when unset. Read at call time
    (once at startup, so no hot path). `or None` collapses "unset" and "exported
    but empty" into one sentinel, so every caller has a single absent-case: a
    blank line in .env yields "", which is not None but is not a DSN either."""
    return os.environ.get("POSTGRES_URL") or None


# The password docker-compose.yml falls back to when .env sets none, so that
# `docker compose up` works with nothing configured but DISCORD_TOKEN. A
# first-run convenience and a liability everywhere else, hence the loud
# detection; only defensible because compose publishes postgres on 127.0.0.1.
# `.env` is the one supported place the real password is set (setup_env.sh writes
# it; compose and `just run` read it); a per-install POSTGRES_PASSWORD_FILE was
# declined — see docs/ARCHITECTURE.md#postgres-credential-handling.
DEFAULT_POSTGRES_PASSWORD: Final[str] = "password"


def using_default_postgres_password() -> bool:
    """True when the archive DSN still carries DEFAULT_POSTGRES_PASSWORD.

    Parsed out of POSTGRES_URL, not POSTGRES_PASSWORD: the bot only ever sees the
    assembled DSN, so the password variable is frequently absent from its own
    environment. SCOPED to the shape this project's tooling produces — userinfo
    in a DSN assembled from `.env` — so it fails open for three hand-written
    shapes asyncpg accepts and this misses:

      * `?password=` in the query string. asyncpg honours it; to urlsplit the
        query is opaque, so this reads as "no password at all".
      * a password containing an unescaped `@`. asyncpg partitions the netloc on
        the first `@` and urlsplit on the last, so `u:p@ss@host/db` authenticates
        as `p` but reads here as `p@ss`.
      * `PGPASSWORD` exported in the environment. Nothing in this repo sets it.

    None is reachable from compose or `just run`; the full ladder is asyncpg's
    own, in `asyncpg.connect_utils._parse_connect_dsn_and_args`. Never raises —
    it feeds a startup warning and a -ping row.
    """
    url = postgres_url()
    if not url:
        return False
    try:
        password = urlsplit(url).password
    except ValueError:
        return False
    # unquote because SplitResult.password does not percent-decode: a DSN
    # carrying %70assword would otherwise read as a different credential than
    # the identical one written literally. asyncpg decodes it, so we must too.
    return password is not None and unquote(password) == DEFAULT_POSTGRES_PASSWORD


def spotify_enabled() -> bool:
    """True when both Spotify credentials are present in the environment. Read at
    call time, not cached, so it tracks the live environment. Gates on presence,
    not validity: credentials that are set but wrong count as enabled and fail
    loudly at the first API call."""
    return bool(
        os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")
    )


# Startup credential probe (MusicBot.cog_load) fetches this one track to confirm
# the credentials authenticate. Any real ID would do; this one is permanent and
# memorable — Rick Astley, "Never Gonna Give You Up".
SPOTIFY_TEST_TRACK_ID = "4PTG3Z6ehGkBFwjybzWkR8"


class SpotifyStatus(Enum):
    """Runtime state of the Spotify source, resolved once at startup. Where
    `spotify_enabled()` answers "are credentials present?", this records whether
    they actually work, so `MusicBot._require_spotify` and `-ping` can tell a
    user *why* Spotify links are unavailable."""

    # No credentials configured.
    DISABLED = "disabled"
    # Present but rejected (or the probe failed): links declined as invalid.
    INVALID = "invalid"
    # Present and validated against the live API at startup.
    ENABLED = "enabled"
