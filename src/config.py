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

# -ping's live-edit loop tunables. Constants (not call-time reads) because the
# dashboard reads them every tick and they are deployment-shape settings, not
# feature switches. They live here rather than in ping.py so this module stays
# the one place to answer "what does the bot read from the environment?".
PING_TICK_SECS: float = float(os.environ.get("PING_TICK_SECS", "1.0"))
PING_DEADLINE_SECS: float = float(os.environ.get("PING_DEADLINE_SECS", "3.0"))


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Parse an integer knob from the environment, or raise a named error.

    Empty reads as unset. The bare `KEY=` shape otherwise raises ValueError at
    IMPORT time, before main() runs setup_telemetry(), so under compose
    (`env_file: .env`, `restart: always`) it is an unstructured traceback on
    stderr and an infinite restart loop with nothing in Loki.

    Negatives are refused rather than passed through. `-1` is the near-universal
    idiom for "no limit" and means the OPPOSITE downstream: the drainer's cap
    check is `if not OUTBOX_MAX: return` (-1 is truthy) then `if depth <=
    OUTBOX_MAX: return` (never true for a negative), so `dropped = depth -
    OUTBOX_MAX` = depth + 1 and the trim takes the entire outbox — on the drain
    SUCCESS path too, wiping every un-archived play roughly every 30s.

    The message names the variable because it surfaces with no logger attached.
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
# outbox once Postgres has it. A cap trades that for bounding a non-evictable key
# during a long Postgres outage; the drainer logs every drop at ERROR.
#
# A drop is UNRECOVERABLE: the cap destroys the OLDEST entries, and
# guild:{id}:history is capped at HISTORY_CACHE_LIMIT, so anything older than
# that window existed only in the outbox. The cap is enforced on the drain
# SUCCESS path too, after each 100-entry batch, so a healthy drainer working
# through a burst backlog also trims — set a cap well above BATCH_SIZE x peak
# burst, not just above steady-state depth. It deliberately destroys entries a
# drainer is holding (ACKing them first); see HistoryOutboxDrainer._enforce_cap.
#
# Sizing: ~420 bytes on the wire, ~487 stored (MEMORY USAGE against redis:7-alpine
# at 100k stream entries), so 256mb holds roughly 525k un-archived plays. Redis 8
# stores the same payload in ~424 bytes; measure on the major you deploy.
HISTORY_OUTBOX_MAX: int = _int_env("HISTORY_OUTBOX_MAX", 0)

# asyncpg's prepared-statement cache size, per connection. The default matches
# asyncpg's own. Set to 0 behind PgBouncer in transaction-pooling mode:
# prepared statements are per-connection state, and transaction pooling hands a
# different backend to each transaction, so a cached statement handle refers to
# something the new backend has never seen.
POSTGRES_STATEMENT_CACHE: int = _int_env("POSTGRES_STATEMENT_CACHE", 100)


def history_archive_enabled() -> bool:
    """True when the operator has opted in to the Postgres history archive.

    The consent gate for long-term storage. Enabled: POSTGRES_URL required at
    startup, every play XADDed to history:outbox, the drainer moving it into
    play_history forever. Disabled — the DEFAULT — none of that exists. Redis
    behavior is identical in both modes; only the durable tier is optional.

    Read at call time: no hot path (push_history calls it once per song) and tests
    monkeypatch the environment per case instead of reloading the module.

    Parsing is STRICT because of the failure direction: a lenient
    anything-but-true-is-False rule turns a typo (`HISTORY_ARCHIVE_ENABLED=on`)
    into an operator who believes they enabled archiving while every play goes
    unrecorded. Garbage raises, naming the variable, because the message can
    surface with no logger attached. Unset and empty read as False — collection
    must be a choice, so absence of a choice means no collection.

    Validation placement is load-bearing: setup_hook MUST call this before any
    other consumer, because the next reader is push_history, which is
    @_guild_op-wrapped — a garbage value first read there would be swallowed into
    one warning per song instead of aborting startup.
    """
    raw = os.environ.get("HISTORY_ARCHIVE_ENABLED")
    value = (raw or "").strip().lower()
    if not value:
        return False
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(
        f"HISTORY_ARCHIVE_ENABLED must be one of true/false, 1/0, or yes/no "
        f"(case-insensitive); got {raw!r}"
    )


def postgres_url() -> Optional[str]:
    """The play-history archive's DSN, or None when unset.

    Read at call time (same rationale as spotify_enabled): read once at startup,
    so there is no hot path, and tests monkeypatch per case.

    `or None` collapses "unset" and "exported but empty" into one sentinel, so the
    Optional[str] is honest and every caller has one absent-case to handle. A
    blank line in .env yields "", which is not None but is not a DSN either.
    """
    return os.environ.get("POSTGRES_URL") or None


# The password docker-compose.yml falls back to when .env does not set one, so
# that `docker compose up` works with nothing configured but DISCORD_TOKEN. A
# first-run convenience and a liability everywhere else, which is why the bot
# detects it and complains loudly rather than silently accepting it. Only
# defensible because the compose postgres service publishes on 127.0.0.1.
#
# `.env` is the ONE supported place the real password is set: written by
# ./setup_env.sh, read by compose and by `just run`'s DSN derivation. A
# per-install POSTGRES_PASSWORD_FILE was considered and declined — see
# docs/ARCHITECTURE.md#postgres-credential-handling.
DEFAULT_POSTGRES_PASSWORD: Final[str] = "password"


def using_default_postgres_password() -> bool:
    """True when the archive DSN still carries DEFAULT_POSTGRES_PASSWORD.

    Parsed out of POSTGRES_URL rather than read from POSTGRES_PASSWORD directly:
    the bot only ever sees the assembled DSN (compose builds it, and `just run`
    derives it), so the password variable is frequently absent from the bot's own
    environment even when it is set for compose.

    SCOPED, deliberately, to the one shape this project's tooling produces: a
    password in the DSN's userinfo, assembled from `.env`. The cost is fail-OPEN —
    the advisory goes silent while the host stays on the shared default — for
    three shapes asyncpg accepts and this misses, all hand-written DSNs:

      * `?password=` in the query string. asyncpg honours it; to urlsplit the
        query is opaque, so this reads as "no password at all".
      * a password containing an unescaped `@`. asyncpg partitions the netloc on
        the FIRST `@` and urlsplit on the LAST, so `u:p@ss@host/db` authenticates
        as `p` but reads here as `p@ss`.
      * `PGPASSWORD` exported in the environment. Nothing in this repo sets it.

    None is reachable from compose or `just run`. If an external Postgres becomes
    a supported deployment this is the function that grows — the ladder is
    asyncpg's own, in `asyncpg.connect_utils._parse_connect_dsn_and_args`.

    Never raises: it feeds a startup warning and a -ping row, and a crash here
    would take down startup over an advisory.
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
    # the identical one written literally. asyncpg decodes it, so we must too.
    return password is not None and unquote(password) == DEFAULT_POSTGRES_PASSWORD


def spotify_enabled() -> bool:
    """True when both Spotify credentials are present in the environment.

    Read at call time, not cached, so it tracks the live environment (tests
    monkeypatch per case; the bot reads it once at startup, so there is no hot
    path). Gates on presence, not validity: credentials that are set but wrong
    still count as enabled and fail loudly at the first API call.
    """
    return bool(
        os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")
    )


# Startup credential probe (MusicBot.cog_load) fetches this one track to confirm
# the credentials authenticate. Any real ID would do; this one is permanent and
# memorable — Rick Astley, "Never Gonna Give You Up".
# https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8
SPOTIFY_TEST_TRACK_ID = "4PTG3Z6ehGkBFwjybzWkR8"


class SpotifyStatus(Enum):
    """Runtime state of the Spotify source, resolved once at startup.

    Where `spotify_enabled()` answers only "are credentials present?", this
    records whether they actually work, so `MusicBot._require_spotify` and
    `-ping` can tell a user *why* Spotify links are unavailable.
    """

    # No credentials configured.
    DISABLED = "disabled"
    # Present but rejected (or the probe failed): links declined as invalid.
    INVALID = "invalid"
    # Present and validated against the live API at startup.
    ENABLED = "enabled"
