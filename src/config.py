import math
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional, cast
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


def _float_env(
    name: str, default: float, *, minimum: float, maximum: Optional[float] = None
) -> float:
    """Float knob from the environment; empty reads as unset. Non-finite is
    refused separately from the floor: `inf` never expires a dashboard deadline
    (the command then holds its concurrency slot forever) and a tick of 0 turns
    the driver's timed wait into a hot spin."""
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


# A settable knob's two values, by environment variable. A handle reads them through
# this module's globals, so one built before a reload reads what the reload parsed.
_BASELINES: Final[dict[str, float]] = {}
_OVERRIDES: Final[dict[str, float]] = {}


@dataclass(frozen=True, slots=True, eq=False)
class Knob[T: (int, float)]:
    """One tunable `-settings bot` may override. Calling it returns the override,
    else the environment value: consumers call it when the value applies, never at
    import (TestBotKnobsAreReadAtCallTime). Identity differs across a reload, so
    key on `env`. See docs/ARCHITECTURE.md#settings-resolution."""

    env: str
    kind: type[T]
    # The `minimum=` the environment parse enforced. The -settings registry holds
    # every chat minimum against it.
    floor: float

    @property
    def field(self) -> str:
        """The knob's field in bot:{application_id}:config."""
        return self.env.lower()

    @property
    def baseline(self) -> T:
        """The value as parsed from the environment."""
        return cast(T, _BASELINES[self.env])

    def __call__(self) -> T:
        return cast(T, _OVERRIDES.get(self.env, _BASELINES[self.env]))

    def override(self) -> Optional[T]:
        """The override, or None while the knob runs on its baseline."""
        return cast(Optional[T], _OVERRIDES.get(self.env))

    def set_override(self, value: T) -> None:
        """Make `value` what the knob returns. Checks the type at run time too — a
        bool is refused for either kind, a non-int for an int knob — and nothing
        else: the -settings registry owns the bounds. src/settings.py is its one
        caller in src/."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{self.env} override must be a number; got {value!r}")
        if self.kind is int and not isinstance(value, int):
            raise TypeError(f"{self.env} override must be an int; got {value!r}")
        _OVERRIDES[self.env] = self.kind(value)

    def clear_override(self) -> None:
        """Return the knob to its environment baseline."""
        _OVERRIDES.pop(self.env, None)


type AnyKnob = Knob[int] | Knob[float]

# Every knob by its config-hash field, in the order declared below.
KNOBS: Final[dict[str, AnyKnob]] = {}


def _secs(name: str, default: float, *, minimum: float) -> Knob[float]:
    """A settable float knob: its environment value parsed, its handle registered."""
    _BASELINES[name] = _float_env(name, default, minimum=minimum)
    knob = Knob(name, float, minimum)
    KNOBS[knob.field] = knob
    return knob


def _count(name: str, default: int, *, minimum: int) -> Knob[int]:
    """A settable int knob, as _secs."""
    _BASELINES[name] = _int_env(name, default, minimum=minimum)
    knob = Knob(name, int, minimum)
    KNOBS[knob.field] = knob
    return knob


# Floor for every live-dashboard knob: small enough to stay a tuning knob, large
# enough that the driver's wait is always a real suspension.
_MIN_DASHBOARD_SECS: Final[float] = 0.05

# -ping's live-edit loop (src/dashboard.py). Env baselines: -ping reads them through
# ping_tick_secs() and ping_deadline_secs(), once per invocation.
ping_tick_secs = _secs("PING_TICK_SECS", 1.0, minimum=_MIN_DASHBOARD_SECS)
ping_deadline_secs = _secs("PING_DEADLINE_SECS", 3.0, minimum=_MIN_DASHBOARD_SECS)

# -debug's live-edit loop. The Postgres block brackets a 2s sampling window plus a
# Prometheus round trip (~2.2s floor), and a block past the deadline renders
# "timed out" rather than being retried, so keep the deadline well above that.
debug_tick_secs = _secs("DEBUG_TICK_SECS", 1.0, minimum=_MIN_DASHBOARD_SECS)
debug_deadline_secs = _secs("DEBUG_DEADLINE_SECS", 8.0, minimum=_MIN_DASHBOARD_SECS)

# How long -analytics waits for its chart before sending the card without one.
# Sized for the cold path; expiring is silent. It bounds the caller, not the
# worker — a ProcessPoolExecutor cannot cancel a running call.
# See docs/ARCHITECTURE.md#analytics-rendering.
analytics_render_deadline_secs = _secs(
    "ANALYTICS_RENDER_DEADLINE_SECS", 20.0, minimum=_MIN_DASHBOARD_SECS
)

# The live card a slow collection enqueue shows (src/queue_progress.py). The delay
# marks the unusual rather than narrating every -play: a cache-hit playlist
# resolves in one Redis GET and lands before it fires. Above 2.0s because a
# measured ten-track enqueue ran 2.03s end to end and does not need a card; the
# real trigger is a cliff at ~101 tracks, where a second continuation page makes
# the resolve ~3.1s, so any value between 2.03 and 3.1 behaves identically.
queue_progress_delay_secs = _secs(
    "QUEUE_PROGRESS_DELAY_SECS", 2.5, minimum=_MIN_DASHBOARD_SECS
)

# The card's own floor, not the dashboards' 0.05: -ping and -debug can share that
# because their deadlines cap the damage at ~8 edits, and this card has none.
# Discord allows 5 edits / 5s per CHANNEL — one bucket, shared with the Now
# Playing bar's 3s cadence, which already spends a third of it.
_MIN_QUEUE_TICK_SECS: Final[float] = 2.0

queue_progress_tick_secs = _secs(
    "QUEUE_PROGRESS_TICK_SECS", 5.0, minimum=_MIN_QUEUE_TICK_SECS
)

# The card's own ceiling: nothing else bounds the work it watches, since
# PLAY_RESOLVE_WAIT_SECS bounds the wait for a slot and not the extraction inside
# it. Past it the card says so once and stops editing, and is deleted when the
# enqueue settles. Floored at delay + tick; docs/ARCHITECTURE.md#queue-progress-card.
queue_progress_max_secs = _secs(
    "QUEUE_PROGRESS_MAX_SECS",
    300.0,
    minimum=queue_progress_delay_secs.baseline + queue_progress_tick_secs.baseline,
)


# Higher floor than the dashboards: each tick is a Redis write per PLAYING guild,
# AOF-appended.
_MIN_HEARTBEAT_SECS: Final[float] = 0.5

# How often a playing guild records its playback position; a crash resumes at the
# last heartbeat, so at most this many seconds replay.
heartbeat_interval_secs = _secs(
    "HEARTBEAT_INTERVAL_SECS", 3.0, minimum=_MIN_HEARTBEAT_SECS
)

# Every playing tick is a real edit on the channel's 5-edits-per-5s bucket, which the
# bar shares with every other message the bot edits there.
_MIN_NOW_PLAYING_SECS: Final[float] = 1.0

# How often the Now Playing card's progress bar is edited.
now_playing_update_interval_secs = _secs(
    "NOW_PLAYING_UPDATE_INTERVAL_SECS", 3.0, minimum=_MIN_NOW_PLAYING_SECS
)

# A probe that does not finish is UNCONFIRMED, so a near-zero cap confirms nothing,
# and three in a row mark the probe path itself as the fault.
_MIN_STREAM_PROBE_SECS: Final[float] = 0.1

# Cap on the pre-playback URL probe. Short because a resolve can pay it twice
# and exceeding it costs a cache entry, not only a verdict; an unconfirmed URL
# still plays, so firing early is cheap.
stream_probe_timeout_secs = _secs(
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


# Extraction worker processes (src/ytdlp_pool.py), ~80–120 MB RSS each. The pool is
# sized when it spawns, so this is read once and never changes at runtime.
YTDLP_POOL_WORKERS: int = _int_env("YTDLP_POOL_WORKERS", 4, minimum=1)

# The ceiling is a typo guard: the value is charged to every start, so a stray
# `50` for `5.0` makes startup look wedged rather than slow. The floor keeps it a
# wait rather than a formality.
_MIN_GUILD_READY_SECS: Final[float] = 0.1
_MAX_GUILD_READY_SECS: Final[float] = 10.0

# How long discord.py waits for ANOTHER GUILD_CREATE before declaring the shard
# ready (its default is 2.0). Read once, at MusicBotApp construction. Raise it if a
# guild is ever missing from bot.guilds at on_ready.
GUILD_READY_TIMEOUT_SECS: float = _float_env(
    "GUILD_READY_TIMEOUT_SECS",
    0.5,
    minimum=_MIN_GUILD_READY_SECS,
    maximum=_MAX_GUILD_READY_SECS,
)

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
play_inflight_max = _count("PLAY_INFLIGHT_MAX", 16, minimum=_MIN_PLAY_COUNT)
# How many admitted requests may hold a yt-dlp worker at once. The pool is
# process-wide and FIFO, so a paste burst in one guild queues every other guild's
# in-band extractions behind it. Half the default pool.
play_resolve_concurrency = _count(
    "PLAY_RESOLVE_CONCURRENCY", 2, minimum=_MIN_PLAY_COUNT
)
# Bound on the WAIT for one of those slots, never on the extraction holding it: a
# 5,547-track playlist legitimately runs 99s, and cutting it off would fail the
# request it is serving. See docs/ARCHITECTURE.md#a-resolve-that-has-to-wait.
play_resolve_wait_secs = _secs(
    "PLAY_RESOLVE_WAIT_SECS", 120.0, minimum=_MIN_PLAY_RESOLVE_WAIT_SECS
)
# How long a request resolves before it says so. Above the 1–4s a warm resolve
# takes, so the notice marks the unusual rather than narrating every -play.
play_slow_notice_secs = _secs(
    "PLAY_SLOW_NOTICE_SECS", 6.0, minimum=_MIN_PLAY_SLOW_NOTICE_SECS
)


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
    guilds that never chose with `-debug --enable/--disable` or `-settings debug`;
    a persisted per-guild choice wins over it, and the operator's `-settings bot
    debug-default` replaces it until restart. Read once by MusicBot.__init__ so
    garbage aborts startup."""
    return _parse_bool_env("DEBUG_MODE")


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
