import math
import os
import subprocess
from enum import Enum
from typing import Final, Literal, Optional, TypeIs, cast, get_args, overload
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


# The `minimum=` each _float_env/_int_env call enforced, by variable. The -settings
# registry holds every chat minimum against it, so the floor it checks is the one the
# parse applied.
_ENV_FLOORS: Final[dict[str, float]] = {}


def _float_env(
    name: str, default: float, *, minimum: float, maximum: Optional[float] = None
) -> float:
    """Float knob from the environment; empty reads as unset. Non-finite is
    refused separately from the floor: `inf` never expires a dashboard deadline
    (the command then holds its concurrency slot forever) and a tick of 0 turns
    the driver's timed wait into a hot spin."""
    _ENV_FLOORS[name] = minimum
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        # Checked too: a floor derived from other knobs can rise past a default.
        if default < minimum:
            raise ValueError(f"{name} must be >= {minimum}; its default is {default}")
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number; got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {raw!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}; got {value}")
    return value


# Floor for every live-dashboard knob: small enough to stay a tuning knob, large
# enough that the driver's wait is always a real suspension.
_MIN_DASHBOARD_SECS: Final[float] = 0.05

# -ping's live-edit loop (src/dashboard.py). Env baselines: -ping reads them through
# ping_tick_secs() and ping_deadline_secs(), once per invocation.
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

# The live card a slow collection enqueue shows (src/queue_progress.py). The delay
# marks the unusual rather than narrating every -play: a cache-hit playlist
# resolves in one Redis GET and lands before it fires. Above 2.0s because a
# measured ten-track enqueue ran 2.03s end to end and does not need a card; the
# real trigger is a cliff at ~101 tracks, where a second continuation page makes
# the resolve ~3.1s, so any value between 2.03 and 3.1 behaves identically.
QUEUE_PROGRESS_DELAY_SECS: float = _float_env(
    "QUEUE_PROGRESS_DELAY_SECS", 2.5, minimum=_MIN_DASHBOARD_SECS
)

# The card's own floor, not the dashboards' 0.05: -ping and -debug can share that
# because their deadlines cap the damage at ~8 edits, and this card has none.
# Discord allows 5 edits / 5s per CHANNEL — one bucket, shared with the Now
# Playing bar's 3s cadence, which already spends a third of it.
_MIN_QUEUE_TICK_SECS: Final[float] = 2.0

QUEUE_PROGRESS_TICK_SECS: float = _float_env(
    "QUEUE_PROGRESS_TICK_SECS", 5.0, minimum=_MIN_QUEUE_TICK_SECS
)

# The card's own ceiling. Nothing else bounds the work it watches:
# PLAY_RESOLVE_WAIT_SECS bounds the wait for a slot and deliberately not the
# extraction inside it, and PLACE_TIMEOUT_SECS bounds 0.01s of a 29s command.
# Past it the card says so once and stops editing; it is still deleted when the
# enqueue settles.
# Floored at delay + tick, the two knobs above it: a lower ceiling has passed by the
# card's first tick. Below the floor startup is refused, default included. Each card
# also sets its own ceiling from the values it runs with (queue_progress.card_ceiling).
QUEUE_PROGRESS_MAX_SECS: float = _float_env(
    "QUEUE_PROGRESS_MAX_SECS",
    300.0,
    minimum=QUEUE_PROGRESS_DELAY_SECS + QUEUE_PROGRESS_TICK_SECS,
)


# Higher floor than the dashboards: each tick is a Redis write per PLAYING guild,
# AOF-appended.
_MIN_HEARTBEAT_SECS: Final[float] = 0.5

# How often a playing guild records its playback position; a crash resumes at the
# last heartbeat, so at most this many seconds replay.
HEARTBEAT_INTERVAL_SECS: float = _float_env(
    "HEARTBEAT_INTERVAL_SECS", 3.0, minimum=_MIN_HEARTBEAT_SECS
)

# Every playing tick is a real edit on the channel's 5-edits-per-5s bucket, which the
# bar shares with every other message the bot edits there.
_MIN_NOW_PLAYING_SECS: Final[float] = 1.0

# How often the Now Playing card's progress bar is edited.
NOW_PLAYING_UPDATE_INTERVAL_SECS: float = _float_env(
    "NOW_PLAYING_UPDATE_INTERVAL_SECS", 3.0, minimum=_MIN_NOW_PLAYING_SECS
)

# A probe that does not finish is UNCONFIRMED, so a near-zero cap confirms nothing,
# and three in a row mark the probe path itself as the fault.
_MIN_STREAM_PROBE_SECS: Final[float] = 0.1

# Cap on the pre-playback URL probe. Short because a resolve can pay it twice
# and exceeding it costs a cache entry, not only a verdict; an unconfirmed URL
# still plays, so firing early is cheap.
STREAM_PROBE_TIMEOUT_SECS: float = _float_env(
    "STREAM_PROBE_TIMEOUT_SECS", 2.0, minimum=_MIN_STREAM_PROBE_SECS
)
# Not a knob: the most `-settings bot stream-probe-timeout` accepts. A resolve can
# pay the probe twice, so this is already up to 10s of silence before a song starts.
STREAM_PROBE_TIMEOUT_MAX_SECS: Final[float] = 5.0

# The HEALTHCHECK calls the file stale after 90s (Dockerfile), so the touch cadence
# is capped well under that; the floor keeps the touch a cadence, not a spin.
_MIN_LIVENESS_SECS: Final[float] = 1.0
_MAX_LIVENESS_SECS: Final[float] = 60.0

# Touched by a loop-resident task for the container HEALTHCHECK. Unset (the
# default outside Docker) skips the task.
LIVENESS_FILE: str = os.environ.get("LIVENESS_FILE", "")
LIVENESS_INTERVAL_SECS: float = _float_env(
    "LIVENESS_INTERVAL_SECS",
    15.0,
    minimum=_MIN_LIVENESS_SECS,
    maximum=_MAX_LIVENESS_SECS,
)


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    """Integer knob from the environment; empty reads as unset. Negatives are
    refused: `-1` reads as "no limit" but `if not OUTBOX_MAX` is truthy for it
    and `depth <= OUTBOX_MAX` never holds, so the drainer would trim the whole
    outbox. The message names the variable because it surfaces with no logger."""
    _ENV_FLOORS[name] = minimum
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


# Extraction worker processes (src/ytdlp_pool.py), ~80–120 MB RSS each. The pool is
# sized when it spawns, so this is read once and never changes at runtime.
YTDLP_POOL_WORKERS: int = _int_env("YTDLP_POOL_WORKERS", 4, minimum=1)

# Opt-in ceiling on the history outbox, in entries; 0 is unbounded (the
# durability contract). A cap destroys the OLDEST entries, which exist nowhere
# else, so every drop logs ERROR. Enforced after each drain batch and against
# entries a drainer is holding (ACKed first, see _enforce_cap): size it well above
# BATCH_SIZE × peak burst. ~625 B stored per entry on redis:7.
HISTORY_OUTBOX_MAX: int = _int_env("HISTORY_OUTBOX_MAX", 0)

# asyncpg statement_cache_size per connection. Set 0 behind a transaction-pooling
# PgBouncer, where each transaction lands on a backend that never saw the handle.
POSTGRES_STATEMENT_CACHE: int = _int_env("POSTGRES_STATEMENT_CACHE", 100)

# Zero admits nothing: every -play declined, or every resolve waiting on a slot
# that never opens, with no error to say why.
_MIN_PLAY_COUNT: Final[int] = 1
# A shorter wait declines a request whenever every slot is busy at all.
_MIN_PLAY_RESOLVE_WAIT_SECS: Final[float] = 1.0
# Below it the notice posts, and is taken back, for nearly every -play.
_MIN_PLAY_SLOW_NOTICE_SECS: Final[float] = 0.5

# Per-guild ceiling on -play requests ADMITTED at once. Its unit is one coroutine,
# one open span and one typing keepalive — memory, not pool time, which
# PLAY_RESOLVE_CONCURRENCY below bounds instead.
PLAY_INFLIGHT_MAX: int = _int_env("PLAY_INFLIGHT_MAX", 16, minimum=_MIN_PLAY_COUNT)
# How many admitted requests may hold a yt-dlp worker at once. The pool is
# process-wide and FIFO, so a paste burst in one guild queues every other guild's
# in-band extractions behind it. Half the default pool.
PLAY_RESOLVE_CONCURRENCY: int = _int_env(
    "PLAY_RESOLVE_CONCURRENCY", 2, minimum=_MIN_PLAY_COUNT
)
# Bound on the WAIT for one of those slots, never on the extraction holding it: a
# 5,547-track playlist legitimately runs 99s, and cutting it off would fail the
# request it is serving. See docs/ARCHITECTURE.md#a-resolve-that-has-to-wait.
PLAY_RESOLVE_WAIT_SECS: float = _float_env(
    "PLAY_RESOLVE_WAIT_SECS", 120.0, minimum=_MIN_PLAY_RESOLVE_WAIT_SECS
)
# How long a request resolves before it says so. Above the 1–4s a warm resolve
# takes, so the notice marks the unusual rather than narrating every -play.
PLAY_SLOW_NOTICE_SECS: float = _float_env(
    "PLAY_SLOW_NOTICE_SECS", 6.0, minimum=_MIN_PLAY_SLOW_NOTICE_SECS
)

# The tunables `-settings bot` may override, by the name of the constant holding
# each one's environment value. Literal strings rather than an Enum: a reload of
# this module would mint new enum classes that a registry built earlier fails
# isinstance against.
type FloatKnob = Literal[
    "NOW_PLAYING_UPDATE_INTERVAL_SECS",
    "HEARTBEAT_INTERVAL_SECS",
    "PLAY_SLOW_NOTICE_SECS",
    "PLAY_RESOLVE_WAIT_SECS",
    "STREAM_PROBE_TIMEOUT_SECS",
    "PING_TICK_SECS",
    "PING_DEADLINE_SECS",
    "DEBUG_TICK_SECS",
    "DEBUG_DEADLINE_SECS",
    "ANALYTICS_RENDER_DEADLINE_SECS",
    "QUEUE_PROGRESS_DELAY_SECS",
    "QUEUE_PROGRESS_TICK_SECS",
    "QUEUE_PROGRESS_MAX_SECS",
]
type IntKnob = Literal["PLAY_INFLIGHT_MAX", "PLAY_RESOLVE_CONCURRENCY"]
FLOAT_KNOBS: Final[frozenset[FloatKnob]] = frozenset(get_args(FloatKnob.__value__))
INT_KNOBS: Final[frozenset[IntKnob]] = frozenset(get_args(IntKnob.__value__))


def is_int_knob(knob: FloatKnob | IntKnob) -> TypeIs[IntKnob]:
    return knob in INT_KNOBS


@overload
def baseline(knob: IntKnob) -> int: ...
@overload
def baseline(knob: FloatKnob) -> float: ...
def baseline(knob: FloatKnob | IntKnob) -> float:
    """The knob's value as parsed from the environment, read at call time so a
    reload of this module is what it returns."""
    return cast(float, globals()[knob])


def env_floor(knob: FloatKnob | IntKnob) -> float:
    """The `minimum=` the environment parse enforced for this knob."""
    return _ENV_FLOORS[knob]


# Settable knobs: each accessor returns a stored -settings bot override, else the
# UPPER_CASE env baseline. set_override's one caller in src/ is src/settings.py.
# Consumers call the accessor when the value applies, never the baseline
# (TestBotKnobsAreReadAtCallTime). See docs/ARCHITECTURE.md#settings-resolution.
_FLOAT_OVERRIDES: Final[dict[FloatKnob, float]] = {}
_INT_OVERRIDES: Final[dict[IntKnob, int]] = {}


def now_playing_update_interval_secs() -> float:
    return _FLOAT_OVERRIDES.get(
        "NOW_PLAYING_UPDATE_INTERVAL_SECS", NOW_PLAYING_UPDATE_INTERVAL_SECS
    )


def heartbeat_interval_secs() -> float:
    return _FLOAT_OVERRIDES.get("HEARTBEAT_INTERVAL_SECS", HEARTBEAT_INTERVAL_SECS)


def play_slow_notice_secs() -> float:
    return _FLOAT_OVERRIDES.get("PLAY_SLOW_NOTICE_SECS", PLAY_SLOW_NOTICE_SECS)


def play_inflight_max() -> int:
    return _INT_OVERRIDES.get("PLAY_INFLIGHT_MAX", PLAY_INFLIGHT_MAX)


def play_resolve_concurrency() -> int:
    return _INT_OVERRIDES.get("PLAY_RESOLVE_CONCURRENCY", PLAY_RESOLVE_CONCURRENCY)


def play_resolve_wait_secs() -> float:
    return _FLOAT_OVERRIDES.get("PLAY_RESOLVE_WAIT_SECS", PLAY_RESOLVE_WAIT_SECS)


def stream_probe_timeout_secs() -> float:
    return _FLOAT_OVERRIDES.get("STREAM_PROBE_TIMEOUT_SECS", STREAM_PROBE_TIMEOUT_SECS)


def ping_tick_secs() -> float:
    return _FLOAT_OVERRIDES.get("PING_TICK_SECS", PING_TICK_SECS)


def ping_deadline_secs() -> float:
    return _FLOAT_OVERRIDES.get("PING_DEADLINE_SECS", PING_DEADLINE_SECS)


def debug_tick_secs() -> float:
    return _FLOAT_OVERRIDES.get("DEBUG_TICK_SECS", DEBUG_TICK_SECS)


def debug_deadline_secs() -> float:
    return _FLOAT_OVERRIDES.get("DEBUG_DEADLINE_SECS", DEBUG_DEADLINE_SECS)


def analytics_render_deadline_secs() -> float:
    return _FLOAT_OVERRIDES.get(
        "ANALYTICS_RENDER_DEADLINE_SECS", ANALYTICS_RENDER_DEADLINE_SECS
    )


def queue_progress_delay_secs() -> float:
    return _FLOAT_OVERRIDES.get("QUEUE_PROGRESS_DELAY_SECS", QUEUE_PROGRESS_DELAY_SECS)


def queue_progress_tick_secs() -> float:
    return _FLOAT_OVERRIDES.get("QUEUE_PROGRESS_TICK_SECS", QUEUE_PROGRESS_TICK_SECS)


def queue_progress_max_secs() -> float:
    return _FLOAT_OVERRIDES.get("QUEUE_PROGRESS_MAX_SECS", QUEUE_PROGRESS_MAX_SECS)


@overload
def set_override(knob: IntKnob, value: int) -> None: ...
@overload
def set_override(knob: FloatKnob, value: float) -> None: ...
def set_override(knob: FloatKnob | IntKnob, value: float) -> None:
    """Make `value` what the knob's accessor returns. Checks the type at run time
    too — a bool is refused for either kind, a non-int for an int knob — and
    nothing else: the -settings registry owns the bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{knob} override must be a number; got {value!r}")
    if is_int_knob(knob):
        if not isinstance(value, int):
            raise TypeError(f"{knob} override must be an int; got {value!r}")
        _INT_OVERRIDES[knob] = value
    else:
        _FLOAT_OVERRIDES[knob] = float(value)


def clear_override(knob: FloatKnob | IntKnob) -> None:
    """Return the knob's accessor to its environment baseline."""
    if is_int_knob(knob):
        _INT_OVERRIDES.pop(knob, None)
    else:
        _FLOAT_OVERRIDES.pop(knob, None)


@overload
def override(knob: IntKnob) -> Optional[int]: ...
@overload
def override(knob: FloatKnob) -> Optional[float]: ...
def override(knob: FloatKnob | IntKnob) -> Optional[float]:
    """The knob's override, or None when it runs on its baseline."""
    if is_int_knob(knob):
        return _INT_OVERRIDES.get(knob)
    return _FLOAT_OVERRIDES.get(knob)


@overload
def effective(knob: IntKnob) -> int: ...
@overload
def effective(knob: FloatKnob) -> float: ...
def effective(knob: FloatKnob | IntKnob) -> float:
    """What the knob's accessor returns: the override, else the baseline."""
    value = override(knob)
    return baseline(knob) if value is None else value


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


def owner_ids() -> frozenset[int]:
    """OWNER_IDS: the Discord user ids of the bot's operator, comma- or
    space-separated. Set, it IS the operator list; empty, discord.py looks the
    application's owner up instead. Each must be 17-20 ASCII digits, else this
    raises naming the variable. Read once, in MusicBotApp.__init__."""
    raw = os.environ.get("OWNER_IDS") or ""
    ids: set[int] = set()
    for token in raw.replace(",", " ").split():
        if not (token.isascii() and token.isdecimal() and 17 <= len(token) <= 20):
            raise ValueError(
                "OWNER_IDS must be Discord user ids (17-20 digits), comma- or "
                f"space-separated; got {token!r}"
            )
        ids.add(int(token))
    return frozenset(ids)


def bot_settings_overrides_ignored() -> bool:
    """BOT_SETTINGS_OVERRIDES: `ignore` runs the process on environment and code
    values, never reading bot:{application_id}:config; unset, empty and `apply`
    apply what is stored. Anything else raises, so setup_hook reads it before
    anything that could swallow the error."""
    raw = os.environ.get("BOT_SETTINGS_OVERRIDES")
    value = (raw or "").strip().lower()
    if value in ("", "apply"):
        return False
    if value == "ignore":
        return True
    raise ValueError(
        "BOT_SETTINGS_OVERRIDES must be apply or ignore (case-insensitive); "
        f"got {raw!r}"
    )


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
