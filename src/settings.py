"""The -settings machinery: the registry of every setting chat may show or change,
the grammar that reads one line of chat into a request against it, and the objects
that hold what is stored.

config.py reads the environment and holds each tunable's parsed value; this module
decides what chat may set, in what shape and within what range. The registry and
the grammar do no IO, and every refusal carries the text a reply shows, none of it
quoting the input.
"""

import difflib
import math
import re
from collections.abc import (
    Callable,
    Mapping,
)
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal


from src import config, guild_state
from src.guild_state import (
    CONFIG_DOMAIN,
    DEFAULT_ALONE_TIMEOUT_SECS,
    DEFAULT_IDLE_TIMEOUT_SECS,
    DEFAULT_TIMEZONE,
    DEFAULT_VOLUME,
    OFF_SECS,
    BotConfigField,
    BotConfigFieldName,
    ConfigField,
    ConfigFieldName,
    is_config_field,
)
from src.queue_progress import card_ceiling
from src.util import DASHES, fmt_duration, fmt_seconds, get_logger, pluralize

if TYPE_CHECKING:
    # For annotations only: nothing here needs -debug's module at run time.
    pass

log = get_logger(__name__)


class SettingScope(Enum):
    SERVER = "server"
    BOT = "bot"


class SettingKind(Enum):
    DURATION = "duration"  # whole seconds; renders fmt_duration: 0:10, 5:00, 30:00
    SECONDS = "seconds"  # <= 2 decimal places; renders fmt_seconds: 3s, 0.5s, 120s
    SECONDS_OR_OFF = "seconds_or_off"  # SECONDS, or the word off, stored as OFF_SECS
    PERCENT = "percent"  # integer 0-100, "%" optional; renders 80%
    SWITCH = "switch"  # on/off/true/false/enable/disable/yes/no; renders on/off
    TIMEZONE = "timezone"  # canonical IANA name; renders as stored
    COUNT = "count"  # integer; renders 16


class SettingGroup(Enum):
    """Card sections, rendered in this order."""

    PLAYBACK = "Playback"
    LEAVING_VOICE = "Leaving voice"
    MESSAGES = "Messages"
    LIMITS = "Limits"
    DIAGNOSTICS = "Diagnostics"


# Checked on write and on every read. A callable reads process constants only, never
# another setting's value. A server spec's static bounds are CONFIG_DOMAIN's.
type Bound = float | Callable[[], float]

# A value in its spec's unit: seconds, a whole percent, a count, a switch or a zone.
type SettingValue = bool | int | float | str


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingSpec:
    key: str
    aliases: tuple[str, ...]
    scope: SettingScope
    kind: SettingKind
    group: SettingGroup
    # Read off the card; with spaces as "-" it is the key or one of the aliases.
    label: str
    # One clause: the cards print it on every row.
    summary: str
    # A caveat the summary has no room for; only the one-setting view shows it.
    more: str | None = None
    # Completes "It applies ...".
    applies: str
    # None only for debug-default, which is never stored.
    field: ConfigFieldName | BotConfigFieldName | None
    minimum: Bound | None
    maximum: Bound | None
    # Another setting's current value: checked when a value is written, never on read.
    write_minimum: Callable[[], float] | None = None
    write_maximum: Callable[[], float] | None = None
    # One sentence each; an out-of-range refusal appends the side the value missed.
    why_minimum: str | None = None
    why_maximum: str | None = None
    # Complete "has to be between A and B here: ..." when a write-only bound refuses;
    # {bound} is that bound, rendered.
    why_write_minimum: str | None = None
    why_write_maximum: str | None = None
    # Server scope's code default; None where the value follows a bot setting or
    # DEBUG_MODE, and for every bot spec, whose default is its env baseline.
    default: float | str | None = None
    # Bot scope only: the env var it overrides, and the config knob holding it.
    env: str | None = None
    attr: config.FloatKnob | config.IntKnob | None = None


_NUMERIC_KINDS: Final = frozenset(
    {
        SettingKind.DURATION,
        SettingKind.SECONDS,
        SettingKind.SECONDS_OR_OFF,
        SettingKind.PERCENT,
        SettingKind.COUNT,
    }
)


def _play_resolve_concurrency_max() -> float:
    # One worker stays out of any one server's reach.
    return max(1, config.YTDLP_POOL_WORKERS - 1)


def _longest_card_delay() -> float:
    """The longest delay a playlist card can run with: the most any
    queue-progress-delay setting accepts, or the environment's value if longer."""
    maxima = [
        bound(spec.maximum) or 0.0
        for spec in SETTINGS
        if spec.key == "queue-progress-delay"
    ]
    return max([config.baseline("QUEUE_PROGRESS_DELAY_SECS"), *maxima])


def _queue_progress_max_floor() -> float:
    return card_ceiling(
        _longest_card_delay(),
        config.queue_progress_tick_secs(),
        config.env_floor("QUEUE_PROGRESS_MAX_SECS"),
    )


def _queue_progress_tick_ceiling() -> float:
    return max(0.0, (config.queue_progress_max_secs() - _longest_card_delay()) / 2)


SETTINGS: Final[tuple[SettingSpec, ...]] = (
    SettingSpec(
        key="volume",
        aliases=("vol",),
        scope=SettingScope.SERVER,
        kind=SettingKind.PERCENT,
        group=SettingGroup.PLAYBACK,
        label="Volume",
        summary="Playback level.",
        applies="from the next song",
        field=ConfigField.VOLUME,
        minimum=CONFIG_DOMAIN[ConfigField.VOLUME].lo * 100,
        maximum=CONFIG_DOMAIN[ConfigField.VOLUME].hi * 100,
        default=DEFAULT_VOLUME * 100,
    ),
    SettingSpec(
        key="timezone",
        aliases=("tz",),
        scope=SettingScope.SERVER,
        kind=SettingKind.TIMEZONE,
        group=SettingGroup.PLAYBACK,
        label="Timezone",
        summary="Time zone for estimated play times.",
        applies="from the next time a card is drawn",
        field=ConfigField.TIMEZONE,
        minimum=None,
        maximum=None,
        default=DEFAULT_TIMEZONE,
    ),
    SettingSpec(
        key="idle-timeout",
        aliases=("idle", "leave-when-idle"),
        scope=SettingScope.SERVER,
        kind=SettingKind.DURATION,
        group=SettingGroup.LEAVING_VOICE,
        label="Leave when idle",
        summary="How long to stay in voice with nothing queued.",
        applies="the next time the queue runs empty",
        field=ConfigField.IDLE_TIMEOUT,
        minimum=CONFIG_DOMAIN[ConfigField.IDLE_TIMEOUT].lo,
        maximum=CONFIG_DOMAIN[ConfigField.IDLE_TIMEOUT].hi,
        why_minimum=(
            "A shorter wait can end the session while a song is still being looked "
            "up. `-stop` in the bot's voice channel makes it leave right away."
        ),
        why_maximum=(
            "While the bot waits it stays in its channel, and a `-play` from another "
            "channel plays there."
        ),
        default=DEFAULT_IDLE_TIMEOUT_SECS,
    ),
    SettingSpec(
        key="alone-timeout",
        aliases=("alone", "leave-when-alone"),
        scope=SettingScope.SERVER,
        kind=SettingKind.DURATION,
        group=SettingGroup.LEAVING_VOICE,
        label="Leave when alone",
        summary="How long to stay in voice once everyone leaves.",
        more="Music keeps playing while it waits, and those songs count toward history.",
        applies="the next time the channel empties",
        field=ConfigField.ALONE_TIMEOUT,
        minimum=CONFIG_DOMAIN[ConfigField.ALONE_TIMEOUT].lo,
        maximum=CONFIG_DOMAIN[ConfigField.ALONE_TIMEOUT].hi,
        why_maximum=(
            "Music keeps playing while the bot waits alone, and every song counts "
            "toward history."
        ),
        default=DEFAULT_ALONE_TIMEOUT_SECS,
    ),
    SettingSpec(
        key="np-refresh",
        aliases=("progress-bar", "progress-bar-refresh"),
        scope=SettingScope.SERVER,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Progress bar refresh",
        summary="How often the Now Playing bar moves.",
        applies="from the next tick",
        field=ConfigField.NP_REFRESH,
        minimum=CONFIG_DOMAIN[ConfigField.NP_REFRESH].lo,
        maximum=CONFIG_DOMAIN[ConfigField.NP_REFRESH].hi,
        # Never faster than the bot: its value is the channel's edit budget.
        write_minimum=config.now_playing_update_interval_secs,
        why_write_minimum="the bot refreshes no faster than {bound}",
    ),
    SettingSpec(
        key="slow-notice",
        aliases=("lookup-notice",),
        scope=SettingScope.SERVER,
        kind=SettingKind.SECONDS_OR_OFF,
        group=SettingGroup.MESSAGES,
        label="Lookup notice",
        summary="Wait before a slow song lookup posts a notice.",
        applies="from the next -play",
        field=ConfigField.SLOW_NOTICE,
        minimum=CONFIG_DOMAIN[ConfigField.SLOW_NOTICE].lo,
        maximum=CONFIG_DOMAIN[ConfigField.SLOW_NOTICE].hi,
        why_minimum=(
            "Most lookups finish within 4s, so a shorter delay would post the notice "
            "for ordinary ones."
        ),
    ),
    SettingSpec(
        key="queue-progress-delay",
        aliases=("playlist-card",),
        scope=SettingScope.SERVER,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Playlist card",
        summary="Wait before a slow playlist shows a progress card.",
        applies="from the next -play",
        field=ConfigField.QUEUE_PROGRESS_DELAY,
        minimum=CONFIG_DOMAIN[ConfigField.QUEUE_PROGRESS_DELAY].lo,
        maximum=CONFIG_DOMAIN[ConfigField.QUEUE_PROGRESS_DELAY].hi,
        why_minimum=(
            "A ten-track playlist queues in about 2s, so a shorter delay would put "
            "the card up for ordinary ones."
        ),
    ),
    SettingSpec(
        key="debug",
        aliases=("debug-footer",),
        scope=SettingScope.SERVER,
        kind=SettingKind.SWITCH,
        group=SettingGroup.DIAGNOSTICS,
        label="Debug footer",
        summary="Adds trace and bot-load details to every embed here.",
        more="Anyone who can read the channel sees it.",
        applies="immediately",
        field=ConfigField.DEBUG_MODE,
        minimum=None,
        maximum=None,
    ),
    SettingSpec(
        key="np-refresh",
        aliases=("progress-bar", "progress-bar-refresh"),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Progress bar refresh",
        summary="How often the Now Playing bar moves.",
        applies="from the next tick",
        field=BotConfigField.NOW_PLAYING_UPDATE_INTERVAL,
        minimum=3.0,
        maximum=30.0,
        why_minimum=(
            "Every tick edits the card against the channel's rate limit, which the "
            "bot's other messages there share."
        ),
        env="NOW_PLAYING_UPDATE_INTERVAL_SECS",
        attr="NOW_PLAYING_UPDATE_INTERVAL_SECS",
    ),
    SettingSpec(
        key="heartbeat",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.PLAYBACK,
        label="Heartbeat",
        summary="How often a playing server saves its position.",
        applies="from the next tick",
        field=BotConfigField.HEARTBEAT_INTERVAL,
        minimum=2.0,
        maximum=30.0,
        why_minimum="Each beat is a Redis write for every server that is playing.",
        env="HEARTBEAT_INTERVAL_SECS",
        attr="HEARTBEAT_INTERVAL_SECS",
    ),
    SettingSpec(
        key="slow-notice",
        aliases=("lookup-notice",),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Lookup notice",
        summary="Wait before a slow song lookup posts a notice.",
        applies="from the next -play",
        field=BotConfigField.PLAY_SLOW_NOTICE,
        minimum=4.0,
        maximum=60.0,
        why_minimum=(
            "Most lookups finish within 4s, so a shorter delay would post the notice "
            "for ordinary ones."
        ),
        env="PLAY_SLOW_NOTICE_SECS",
        attr="PLAY_SLOW_NOTICE_SECS",
    ),
    SettingSpec(
        key="queue-progress-delay",
        aliases=("playlist-card",),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Playlist card",
        summary="Wait before a slow playlist shows a progress card.",
        applies="from the next -play",
        field=BotConfigField.QUEUE_PROGRESS_DELAY,
        minimum=2.0,
        maximum=60.0,
        why_minimum=(
            "A ten-track playlist queues in about 2s, so a shorter delay would put "
            "the card up for ordinary ones."
        ),
        env="QUEUE_PROGRESS_DELAY_SECS",
        attr="QUEUE_PROGRESS_DELAY_SECS",
    ),
    SettingSpec(
        key="queue-progress-tick",
        aliases=("playlist-card-tick",),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Playlist card tick",
        summary="Shortest gap between the playlist card's edits.",
        applies="from the next card",
        field=BotConfigField.QUEUE_PROGRESS_TICK,
        minimum=3.0,
        maximum=30.0,
        write_maximum=_queue_progress_tick_ceiling,
        why_minimum=(
            "The card shares the channel's rate limit with the Now Playing bar's edits."
        ),
        why_write_maximum=(
            "the card needs two ticks between its longest delay and queue-progress-max, "
            "so a tick of at most {bound}"
        ),
        env="QUEUE_PROGRESS_TICK_SECS",
        attr="QUEUE_PROGRESS_TICK_SECS",
    ),
    SettingSpec(
        key="queue-progress-max",
        aliases=("playlist-card-max",),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.MESSAGES,
        label="Playlist card max",
        summary="How long the playlist card updates before it stops.",
        more='Past it the card says "still working" and stops editing.',
        applies="from the next card",
        field=BotConfigField.QUEUE_PROGRESS_MAX,
        minimum=120.0,
        maximum=900.0,
        write_minimum=_queue_progress_max_floor,
        why_minimum=(
            "A 5,000-track playlist takes about 100s to look up, so a shorter limit "
            "would stall the card on a lookup that is still working."
        ),
        why_write_minimum=(
            "a card waits up to its longest delay and then needs two ticks, "
            "{bound}, before it can stop"
        ),
        env="QUEUE_PROGRESS_MAX_SECS",
        attr="QUEUE_PROGRESS_MAX_SECS",
    ),
    SettingSpec(
        key="play-inflight-max",
        aliases=("inflight-max",),
        scope=SettingScope.BOT,
        kind=SettingKind.COUNT,
        group=SettingGroup.LIMITS,
        label="Inflight max",
        summary="How many -play requests a server can run at once.",
        applies="from the next -play",
        field=BotConfigField.PLAY_INFLIGHT_MAX,
        minimum=1,
        maximum=32,
        env="PLAY_INFLIGHT_MAX",
        attr="PLAY_INFLIGHT_MAX",
    ),
    SettingSpec(
        key="play-resolve-concurrency",
        aliases=("resolve-concurrency",),
        scope=SettingScope.BOT,
        kind=SettingKind.COUNT,
        group=SettingGroup.LIMITS,
        label="Resolve concurrency",
        summary="Lookup workers one server can use at once.",
        applies="once a server's lookups in progress have all finished",
        field=BotConfigField.PLAY_RESOLVE_CONCURRENCY,
        minimum=1,
        maximum=_play_resolve_concurrency_max,
        why_maximum=(
            "One worker stays free, so no one server's lookups can hold the whole pool."
        ),
        env="PLAY_RESOLVE_CONCURRENCY",
        attr="PLAY_RESOLVE_CONCURRENCY",
    ),
    SettingSpec(
        key="play-resolve-wait",
        aliases=("resolve-wait",),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.LIMITS,
        label="Resolve wait",
        summary="How long a -play waits for a lookup worker.",
        applies="from the next wait",
        field=BotConfigField.PLAY_RESOLVE_WAIT,
        minimum=30.0,
        maximum=300.0,
        why_minimum=(
            "A pasted burst of links can queue its last request for about half a "
            "minute on a healthy bot."
        ),
        env="PLAY_RESOLVE_WAIT_SECS",
        attr="PLAY_RESOLVE_WAIT_SECS",
    ),
    SettingSpec(
        key="stream-probe-timeout",
        aliases=("probe-timeout",),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.PLAYBACK,
        label="Probe timeout",
        summary="Time limit for checking a song's stream.",
        applies="from the next check",
        field=BotConfigField.STREAM_PROBE_TIMEOUT,
        minimum=0.5,
        maximum=config.STREAM_PROBE_TIMEOUT_MAX_SECS,
        why_maximum="A song can wait for the check twice before it starts.",
        env="STREAM_PROBE_TIMEOUT_SECS",
        attr="STREAM_PROBE_TIMEOUT_SECS",
    ),
    SettingSpec(
        key="ping-tick",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.DIAGNOSTICS,
        label="Ping tick",
        summary="Shortest gap between -ping's edits.",
        applies="from the next -ping",
        field=BotConfigField.PING_TICK,
        minimum=1.0,
        maximum=5.0,
        env="PING_TICK_SECS",
        attr="PING_TICK_SECS",
    ),
    SettingSpec(
        key="ping-deadline",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.DIAGNOSTICS,
        label="Ping deadline",
        summary="How long -ping waits before marking a check failed.",
        applies="from the next -ping",
        field=BotConfigField.PING_DEADLINE,
        minimum=1.0,
        maximum=30.0,
        env="PING_DEADLINE_SECS",
        attr="PING_DEADLINE_SECS",
    ),
    SettingSpec(
        key="debug-tick",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.DIAGNOSTICS,
        label="Debug tick",
        summary="Shortest gap between -debug's edits.",
        applies="from the next -debug",
        field=BotConfigField.DEBUG_TICK,
        minimum=1.0,
        maximum=5.0,
        env="DEBUG_TICK_SECS",
        attr="DEBUG_TICK_SECS",
    ),
    SettingSpec(
        key="debug-deadline",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.DIAGNOSTICS,
        label="Debug deadline",
        summary='How long a -debug block collects before "timed out".',
        applies="from the next -debug",
        field=BotConfigField.DEBUG_DEADLINE,
        minimum=5.0,
        maximum=60.0,
        why_minimum="The database block alone can take about 4s.",
        env="DEBUG_DEADLINE_SECS",
        attr="DEBUG_DEADLINE_SECS",
    ),
    SettingSpec(
        key="analytics-deadline",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SECONDS,
        group=SettingGroup.DIAGNOSTICS,
        label="Analytics deadline",
        summary="How long -analytics waits for its chart.",
        applies="from the next -analytics",
        field=BotConfigField.ANALYTICS_RENDER_DEADLINE,
        minimum=10.0,
        maximum=120.0,
        why_minimum="The first chart after a restart takes about 6s.",
        env="ANALYTICS_RENDER_DEADLINE_SECS",
        attr="ANALYTICS_RENDER_DEADLINE_SECS",
    ),
    SettingSpec(
        key="debug-default",
        aliases=(),
        scope=SettingScope.BOT,
        kind=SettingKind.SWITCH,
        group=SettingGroup.DIAGNOSTICS,
        label="Debug default",
        summary="Debug footer for servers that haven't chosen.",
        more="It lasts until the bot restarts.",
        applies="immediately",
        field=None,
        minimum=None,
        maximum=None,
        env="DEBUG_MODE",
    ),
)


def _fold(name: str) -> str:
    return name.casefold().replace("_", "-")


def _index(scope: SettingScope) -> Mapping[str, SettingSpec]:
    return MappingProxyType(
        {
            _fold(name): spec
            for spec in SETTINGS
            if spec.scope is scope
            for name in (spec.key, *spec.aliases)
        }
    )


_BY_NAME: Final[Mapping[SettingScope, Mapping[str, SettingSpec]]] = MappingProxyType(
    {scope: _index(scope) for scope in SettingScope}
)


_SERVER_BY_FIELD: Final[Mapping[str, SettingSpec]] = MappingProxyType(
    {
        spec.field: spec
        for spec in SETTINGS
        if spec.scope is SettingScope.SERVER and spec.field is not None
    }
)
_BY_KNOB: Final[Mapping[str, SettingSpec]] = MappingProxyType(
    {spec.attr: spec for spec in SETTINGS if spec.attr is not None}
)


def server_spec(field: ConfigFieldName) -> SettingSpec:
    """The server setting stored in `field`."""
    return _SERVER_BY_FIELD[field]


def knob_spec(knob: config.FloatKnob | config.IntKnob) -> SettingSpec:
    """The bot setting that overrides `knob`."""
    return _BY_KNOB[knob]


def named(token: str, scope: SettingScope) -> SettingSpec | None:
    """The setting a key or alias names in one scope, folded as find() folds it;
    never a close match."""
    return _BY_NAME[scope].get(_fold(token))


def followed_knob(spec: SettingSpec) -> config.FloatKnob | config.IntKnob | None:
    """The bot knob a server setting runs on while unset: the one the bot setting
    of the same key overrides. None for a bot setting, or a server one without."""
    if spec.scope is not SettingScope.SERVER:
        return None
    bot = named(spec.key, SettingScope.BOT)
    return bot.attr if bot is not None else None


# ── Values: bounds, rendering and conversion to what is stored ──────────────


def bound(value: Bound | None) -> float | None:
    """A static bound's value now: a callable is evaluated at the call."""
    return value() if callable(value) else value


def to_stored(spec: SettingSpec, value: float) -> float:
    """A value in the spec's unit as its field stores it: volume is 0.0–1.0."""
    return value / 100 if spec.kind is SettingKind.PERCENT else value


def from_stored(spec: SettingSpec, stored: float) -> float:
    """round(), not int(): int(0.29 * 100) is 28."""
    return round(stored * 100) if spec.kind is SettingKind.PERCENT else stored


def format_value(spec: SettingSpec, value: SettingValue) -> str:
    """Exactly one rendering per kind, and every one parses back to `value`."""
    match spec.kind:
        case SettingKind.DURATION:
            return fmt_duration(int(value))
        case SettingKind.SECONDS:
            return fmt_seconds(float(value))
        case SettingKind.SECONDS_OR_OFF:
            return "off" if value == OFF_SECS else fmt_seconds(float(value))
        case SettingKind.PERCENT:
            return f"{int(value)}%"
        case SettingKind.SWITCH:
            return "on" if value else "off"
        case SettingKind.TIMEZONE:
            return str(value)
        case SettingKind.COUNT:
            return str(int(value))


def in_bounds(spec: SettingSpec, value: SettingValue) -> bool:
    """The one bounds test. A server spec asks its field's CONFIG_DOMAIN, which
    every stored value also passes through on read; a bot spec compares against its
    static bounds. Chained comparisons, so NaN is never in bounds."""
    if spec.kind not in _NUMERIC_KINDS:
        return True
    if isinstance(value, (bool, str)):
        return False
    if spec.kind is SettingKind.COUNT and not isinstance(value, int):
        return False
    if spec.scope is SettingScope.SERVER:
        if spec.field is None or not is_config_field(spec.field):
            return False
        return CONFIG_DOMAIN[spec.field].admits(to_stored(spec, value))
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    return (
        math.isfinite(value)
        and (lo is None or lo <= value)
        and (hi is None or value <= hi)
    )


class RefusalReason(Enum):
    """Recorded on the span as `settings.refused`; the text never is. The parse
    returns the first eight; the command adds the rest."""

    TOO_MUCH = "too_much"
    BAD_SHAPE = "bad_shape"
    OUT_OF_RANGE = "out_of_range"
    UNKNOWN_KEY = "unknown_key"
    RESET_WITHOUT_KEY = "reset_without_key"
    WRONG_SCOPE = "wrong_scope"
    TIMEZONE_REDIRECT = "timezone_redirect"
    FIXED_OFFSET = "fixed_offset"
    DM_NEEDS_BOT = "dm_needs_bot"
    NO_PERMISSION = "no_permission"
    OPERATOR_ONLY = "operator_only"
    OPERATOR_UNCONFIRMED = "operator_unconfirmed"
    OVERRIDES_IGNORED = "overrides_ignored"
    # A bullet-shaped message whose parse was refused: nothing is sent.
    BULLET_SHAPE = "bullet_shape"


@dataclass(frozen=True, slots=True, kw_only=True)
class Refusal:
    reason: RefusalReason
    text: str
    spec: SettingSpec | None = None
    # Which bound an out-of-range value missed.
    side: Literal["minimum", "maximum"] | None = None
    # The scope the request named; None when it was refused before the scope word.
    scope: SettingScope | None = None


@dataclass(frozen=True, slots=True)
class Parsed:
    value: SettingValue


def _range_text(spec: SettingSpec, lo: float | None, hi: float | None) -> str:
    if lo is not None and hi is not None:
        text = f"between **{format_value(spec, lo)}** and **{format_value(spec, hi)}**"
    elif lo is not None:
        text = f"at least **{format_value(spec, lo)}**"
    else:
        text = f"at most **{format_value(spec, hi if hi is not None else 0)}**"
    return text + (", or `off`" if spec.kind is SettingKind.SECONDS_OR_OFF else "")


def _out_of_range(
    spec: SettingSpec,
    side: Literal["minimum", "maximum"],
    lo: float | None,
    hi: float | None,
    *,
    binding: float | None = None,
) -> Refusal:
    """`binding`: the write-only bound on `side` the value missed, when it is that
    side's binding bound."""
    why_write = spec.why_write_minimum if side == "minimum" else spec.why_write_maximum
    text = f"**{spec.label}** has to be {_range_text(spec, lo, hi)}"
    if binding is not None and why_write is not None:
        bold = f"**{format_value(spec, binding)}**"
        text += f" here: {why_write.format(bound=bold)}."
    else:
        why = spec.why_minimum if side == "minimum" else spec.why_maximum
        text += f". {why}" if why else "."
    return Refusal(reason=RefusalReason.OUT_OF_RANGE, text=text, spec=spec, side=side)


def _typeable(spec: SettingSpec, value: float, *, up: bool) -> float:
    """`value` rounded to the finest step the grammar takes for this kind, a
    hundredth of a second or one whole unit, toward the inside of the range. Taken
    to nine decimals first, so a bound that is already a step, like 3.6, whose
    float lies just past it, is not moved one step inward."""
    scale = 100 if spec.kind in (SettingKind.SECONDS, SettingKind.SECONDS_OR_OFF) else 1
    scaled = Fraction(f"{value:.9f}") * scale
    return (math.ceil(scaled) if up else math.floor(scaled)) / scale


def write_range(spec: SettingSpec) -> tuple[float | None, float | None]:
    """The range a write may take now: the static bounds, narrowed by any write-only
    bound, whose value follows another setting. lo > hi when that setting's value
    leaves no room. Each write-only bound is rounded to a value that can be typed."""
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    if spec.write_minimum is not None:
        floor = _typeable(spec, spec.write_minimum(), up=True)
        lo = floor if lo is None else max(lo, floor)
    if spec.write_maximum is not None:
        ceiling = _typeable(spec, spec.write_maximum(), up=False)
        hi = ceiling if hi is None else min(hi, ceiling)
    return lo, hi


def _no_room_why(spec: SettingSpec, lo: float, hi: float) -> str:
    """Why another setting's value leaves a write no range: the write-only bound
    that crossed the static range, rendered."""
    static_lo, static_hi = bound(spec.minimum), bound(spec.maximum)
    if spec.why_write_minimum is not None and (static_hi is None or lo > static_hi):
        return spec.why_write_minimum.format(bound=f"**{format_value(spec, lo)}**")
    if spec.why_write_maximum is not None and (static_lo is None or hi < static_lo):
        return spec.why_write_maximum.format(bound=f"**{format_value(spec, hi)}**")
    return "another setting leaves it no range"


def check_write(spec: SettingSpec, value: SettingValue) -> Refusal | None:
    """in_bounds, then the write-only bounds that follow another setting. A
    refusal names the range a write may take, write-only bounds included."""
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    write_lo, write_hi = write_range(spec)
    if isinstance(value, (bool, str)) or (
        spec.kind is SettingKind.SECONDS_OR_OFF and value == OFF_SECS
    ):
        if in_bounds(spec, value):
            return None
        return _out_of_range(spec, "maximum", write_lo, write_hi)
    if write_lo is not None and write_hi is not None and write_lo > write_hi:
        return Refusal(
            reason=RefusalReason.OUT_OF_RANGE,
            text=(
                f"**{spec.label}** can't be changed right now: "
                f"{_no_room_why(spec, write_lo, write_hi)}."
            ),
            spec=spec,
            side="minimum" if lo is None or write_lo > lo else "maximum",
        )
    if in_bounds(spec, value):
        # In the static range, so a miss here is the write-only bound's.
        if write_lo is not None and value < write_lo:
            return _out_of_range(spec, "minimum", write_lo, write_hi, binding=write_lo)
        if write_hi is not None and value > write_hi:
            return _out_of_range(spec, "maximum", write_lo, write_hi, binding=write_hi)
        return None
    if lo is not None and value < lo:
        dynamic = write_lo if write_lo is not None and write_lo > lo else None
        return _out_of_range(spec, "minimum", write_lo, write_hi, binding=dynamic)
    dynamic = (
        write_hi if write_hi is not None and hi is not None and write_hi < hi else None
    )
    return _out_of_range(spec, "maximum", write_lo, write_hi, binding=dynamic)


# ── The duration grammar ──────────────────────────────────────────────────────
# Nothing reaches Fraction() or int() before its whole value has matched: \d and
# str.isdecimal() both accept '٣', and float() takes '1_000', '1e3', 'inf' and
# surrounding whitespace. fullmatch, because `$` also matches before a trailing
# newline.

_FLAGS: Final = re.ASCII | re.IGNORECASE
_NUMBER: Final = r"\d{1,6}(?:\.\d{1,2})?"
_SECONDS_RE: Final = re.compile(_NUMBER, _FLAGS)
_CLOCK_MS_RE: Final = re.compile(r"(\d{1,5}):([0-5]\d)", _FLAGS)
_CLOCK_HMS_RE: Final = re.compile(r"(\d{1,3}):([0-5]\d):([0-5]\d)", _FLAGS)
# h, then m, then s, each at most once, an optional space before and between.
_UNITS_RE: Final = re.compile(
    rf"(?:(?P<h>{_NUMBER}) ?(?:h|hrs?|hours?))?"
    rf"(?: ?(?P<m>{_NUMBER}) ?(?:m|mins?|minutes?))?"
    rf"(?: ?(?P<s>{_NUMBER}) ?(?:s|secs?|seconds?))?",
    _FLAGS,
)
# A units value with its seconds unit left off: `5m30` means `5:30` or `5m30s`.
_MINUTES_THEN_BARE_RE: Final = re.compile(
    r"(\d{1,5}) ?(?:m|mins?|minutes?) ?([0-5]?\d)", _FLAGS
)
_PERCENT_RE: Final = re.compile(r"(\d{1,3})%?", _FLAGS)
_COUNT_RE: Final = re.compile(r"\d{1,4}", _FLAGS)

_SWITCH_WORDS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "on": True,
        "true": True,
        "enable": True,
        "yes": True,
        "off": False,
        "false": False,
        "disable": False,
        "no": False,
    }
)


def _seconds_total(value: str) -> Fraction | None:
    """Any duration-grammar shape as exact seconds, or None when none matches."""
    if _SECONDS_RE.fullmatch(value):
        return Fraction(value)
    if match := _CLOCK_HMS_RE.fullmatch(value):
        h, m, s = (int(part) for part in match.groups())
        return Fraction(h * 3600 + m * 60 + s)
    if match := _CLOCK_MS_RE.fullmatch(value):
        m, s = (int(part) for part in match.groups())
        return Fraction(m * 60 + s)
    match = _UNITS_RE.fullmatch(value)
    if match is None or not any(match.group(unit) for unit in "hms"):
        return None
    return sum(
        (
            Fraction(match.group(unit) or 0) * scale
            for unit, scale in (("h", 3600), ("m", 60), ("s", 1))
        ),
        Fraction(0),
    )


def _example_seconds(spec: SettingSpec) -> int:
    """A value a write takes now, for a bad-shape hint: the middle of the range, in
    whole minutes once that is past two of them."""
    lo, hi = write_range(spec)
    if lo is None or hi is None or lo > hi:
        return 90
    middle = (lo + hi) / 2
    return round(middle / 60) * 60 if middle >= 120 else max(1, round(middle))


def _units_text(secs: int) -> str:
    h, rest = divmod(secs, 3600)
    m, s = divmod(rest, 60)
    parts = [f"{n}{unit}" for n, unit in ((h, "h"), (m, "m"), (s, "s")) if n]
    return "".join(parts) or "0s"


def _duration_shape_refusal(spec: SettingSpec, value: str) -> Refusal:
    if match := _MINUTES_THEN_BARE_RE.fullmatch(value):
        m, s = int(match.group(1)), int(match.group(2))
        examples = f"`{m}:{s:02d}` or `{m}m{s}s`"
    elif spec.kind is SettingKind.DURATION:
        secs = _example_seconds(spec)
        examples = f"`{fmt_duration(secs)}` or `{_units_text(secs)}`"
    else:
        secs = _example_seconds(spec)
        examples = f"`{secs}` or `{secs}s`"
    if spec.kind is SettingKind.DURATION:
        text = f"**{spec.label}** takes a duration like {examples}."
    elif spec.kind is SettingKind.SECONDS_OR_OFF:
        text = f"**{spec.label}** takes seconds, like {examples}, or `off`."
    else:
        text = f"**{spec.label}** takes seconds, like {examples}."
    return Refusal(reason=RefusalReason.BAD_SHAPE, text=text, spec=spec)


def _parse_time(spec: SettingSpec, value: str) -> Parsed | Refusal:
    if spec.kind is SettingKind.SECONDS_OR_OFF and value.casefold() == "off":
        return Parsed(OFF_SECS)
    total = _seconds_total(value)
    if total is None or (total * 100).denominator != 1:
        return _duration_shape_refusal(spec, value)
    if spec.kind is SettingKind.DURATION and total.denominator != 1:
        example = _example_seconds(spec)
        return Refusal(
            reason=RefusalReason.BAD_SHAPE,
            text=(
                f"**{spec.label}** takes whole seconds, like `{_units_text(example)}` "
                f"or `{fmt_duration(example)}`."
            ),
            spec=spec,
        )
    seconds = float(total)
    if spec.kind is SettingKind.SECONDS_OR_OFF and seconds == OFF_SECS:
        # `0` could mean "at once" as easily as "never", so off has one spelling.
        return _out_of_range(spec, "minimum", bound(spec.minimum), bound(spec.maximum))
    return Parsed(seconds)


def _parse_percent(spec: SettingSpec, value: str) -> Parsed | Refusal:
    if match := _PERCENT_RE.fullmatch(value):
        return Parsed(int(match.group(1)))
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    return Refusal(
        reason=RefusalReason.BAD_SHAPE,
        text=f"**{spec.label}** is a percentage from {int(lo or 0)} to {int(hi or 100)}, like `80`.",
        spec=spec,
    )


def _parse_count(spec: SettingSpec, value: str) -> Parsed | Refusal:
    if _COUNT_RE.fullmatch(value):
        return Parsed(int(value))
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    span = f" from {int(lo)} to {int(hi)}" if lo is not None and hi is not None else ""
    return Refusal(
        reason=RefusalReason.BAD_SHAPE,
        text=f"**{spec.label}** takes a whole number{span}, like `{int(lo or 1)}`.",
        spec=spec,
    )


def _parse_switch(spec: SettingSpec, value: str) -> Parsed | Refusal:
    choice = _SWITCH_WORDS.get(value.casefold())
    if choice is not None:
        return Parsed(choice)
    return Refusal(
        reason=RefusalReason.BAD_SHAPE,
        text=f"**{spec.label}** takes `on` or `off`.",
        spec=spec,
    )


# ── Timezones ───────────────────────────────────────────────────────────────────

# An Area/City name is accepted when its first segment is one of these, and so are
# UTC and GMT. Everything else is refused, including names a host's own tz directory
# adds: available_timezones() walks TZPATH as well as tzdata.
_ZONE_AREAS: Final = frozenset(
    {
        "Africa",
        "America",
        "Antarctica",
        "Arctic",
        "Asia",
        "Atlantic",
        "Australia",
        "Europe",
        "Indian",
        "Pacific",
    }
)
_BARE_ZONES: Final = frozenset({"UTC", "GMT"})
# With spaces and underscores removed first. Etc/GMT+5 is five hours BEHIND UTC.
_FIXED_OFFSET_RE: Final = re.compile(
    r"(?:etc/)?(?:gmt|utc|uct)?[+-]\d{1,2}(?::?\d{2})?", _FLAGS
)


@dataclass(frozen=True, slots=True)
class TimezoneRedirect:
    targets: tuple[str, ...]
    # synonym: another name for UTC or GMT. legacy: an old name with the same
    # rules as its target. abbreviation: a label, not a place.
    reason: Literal["synonym", "legacy", "abbreviation"]


def _synonym(target: str) -> TimezoneRedirect:
    return TimezoneRedirect((target,), "synonym")


def _legacy(target: str) -> TimezoneRedirect:
    return TimezoneRedirect((target,), "legacy")


def _abbreviation(*targets: str) -> TimezoneRedirect:
    return TimezoneRedirect(targets, "abbreviation")


# Every name tzdata carries outside the ten areas that is not UTC, GMT, Factory or a
# fixed offset has an entry, and common abbreviations tzdata lacks have one too.
TIMEZONE_REDIRECTS: Final[Mapping[str, TimezoneRedirect]] = MappingProxyType(
    {
        "Etc/GMT": _synonym("GMT"),
        "Etc/GMT0": _synonym("GMT"),
        "Etc/Greenwich": _synonym("GMT"),
        "GMT0": _synonym("GMT"),
        "Greenwich": _synonym("GMT"),
        "Etc/UTC": _synonym("UTC"),
        "Etc/UCT": _synonym("UTC"),
        "Etc/Universal": _synonym("UTC"),
        "Etc/Zulu": _synonym("UTC"),
        "UCT": _synonym("UTC"),
        "Universal": _synonym("UTC"),
        "Zulu": _synonym("UTC"),
        "Brazil/Acre": _legacy("America/Rio_Branco"),
        "Brazil/DeNoronha": _legacy("America/Noronha"),
        "Brazil/East": _legacy("America/Sao_Paulo"),
        "Brazil/West": _legacy("America/Manaus"),
        "Canada/Atlantic": _legacy("America/Halifax"),
        "Canada/Central": _legacy("America/Winnipeg"),
        "Canada/Eastern": _legacy("America/Toronto"),
        "Canada/Mountain": _legacy("America/Edmonton"),
        "Canada/Newfoundland": _legacy("America/St_Johns"),
        "Canada/Pacific": _legacy("America/Vancouver"),
        "Canada/Saskatchewan": _legacy("America/Regina"),
        "Canada/Yukon": _legacy("America/Whitehorse"),
        "Chile/Continental": _legacy("America/Santiago"),
        "Chile/EasterIsland": _legacy("Pacific/Easter"),
        "Cuba": _legacy("America/Havana"),
        "Egypt": _legacy("Africa/Cairo"),
        "Eire": _legacy("Europe/Dublin"),
        "GB": _legacy("Europe/London"),
        "GB-Eire": _legacy("Europe/London"),
        "Hongkong": _legacy("Asia/Hong_Kong"),
        "Iceland": _legacy("Atlantic/Reykjavik"),
        "Iran": _legacy("Asia/Tehran"),
        "Israel": _legacy("Asia/Jerusalem"),
        "Jamaica": _legacy("America/Jamaica"),
        "Japan": _legacy("Asia/Tokyo"),
        "Kwajalein": _legacy("Pacific/Kwajalein"),
        "Libya": _legacy("Africa/Tripoli"),
        "Mexico/BajaNorte": _legacy("America/Tijuana"),
        "Mexico/BajaSur": _legacy("America/Mazatlan"),
        "Mexico/General": _legacy("America/Mexico_City"),
        "NZ": _legacy("Pacific/Auckland"),
        "NZ-CHAT": _legacy("Pacific/Chatham"),
        "Navajo": _legacy("America/Denver"),
        "PRC": _legacy("Asia/Shanghai"),
        "Poland": _legacy("Europe/Warsaw"),
        "Portugal": _legacy("Europe/Lisbon"),
        "ROC": _legacy("Asia/Taipei"),
        "ROK": _legacy("Asia/Seoul"),
        "Singapore": _legacy("Asia/Singapore"),
        "Turkey": _legacy("Europe/Istanbul"),
        "US/Alaska": _legacy("America/Anchorage"),
        "US/Aleutian": _legacy("America/Adak"),
        "US/Arizona": _legacy("America/Phoenix"),
        "US/Central": _legacy("America/Chicago"),
        "US/East-Indiana": _legacy("America/Indiana/Indianapolis"),
        "US/Eastern": _legacy("America/New_York"),
        "US/Hawaii": _legacy("Pacific/Honolulu"),
        "US/Indiana-Starke": _legacy("America/Indiana/Knox"),
        "US/Michigan": _legacy("America/Detroit"),
        "US/Mountain": _legacy("America/Denver"),
        "US/Pacific": _legacy("America/Los_Angeles"),
        "US/Samoa": _legacy("Pacific/Pago_Pago"),
        "W-SU": _legacy("Europe/Moscow"),
        # In tzdata, as fixed offsets or POSIX rules.
        "CET": _abbreviation("Europe/Paris"),
        "CST6CDT": _abbreviation("America/Chicago"),
        "EET": _abbreviation("Europe/Athens"),
        "EST": _abbreviation("America/New_York"),
        "EST5EDT": _abbreviation("America/New_York"),
        "HST": _abbreviation("Pacific/Honolulu"),
        "MET": _abbreviation("Europe/Paris"),
        "MST": _abbreviation("America/Denver", "America/Phoenix"),
        "MST7MDT": _abbreviation("America/Denver"),
        "PST8PDT": _abbreviation("America/Los_Angeles"),
        "WET": _abbreviation("Europe/Lisbon"),
        # Not in tzdata.
        "PST": _abbreviation("America/Los_Angeles"),
        "PDT": _abbreviation("America/Los_Angeles"),
        "MDT": _abbreviation("America/Denver"),
        "CST": _abbreviation("America/Chicago", "Asia/Shanghai", "America/Havana"),
        "CDT": _abbreviation("America/Chicago"),
        "EDT": _abbreviation("America/New_York"),
        "AKST": _abbreviation("America/Anchorage"),
        "AKDT": _abbreviation("America/Anchorage"),
        "AST": _abbreviation("America/Halifax", "Asia/Riyadh"),
        "ADT": _abbreviation("America/Halifax"),
        "NST": _abbreviation("America/St_Johns"),
        "NDT": _abbreviation("America/St_Johns"),
        "BST": _abbreviation("Europe/London", "Asia/Dhaka"),
        "IST": _abbreviation("Asia/Kolkata", "Europe/Dublin", "Asia/Jerusalem"),
        "CEST": _abbreviation("Europe/Paris"),
        "EEST": _abbreviation("Europe/Athens"),
        "WEST": _abbreviation("Europe/Lisbon"),
        "MSK": _abbreviation("Europe/Moscow"),
        "JST": _abbreviation("Asia/Tokyo"),
        "KST": _abbreviation("Asia/Seoul"),
        "HKT": _abbreviation("Asia/Hong_Kong"),
        "SGT": _abbreviation("Asia/Singapore"),
        "AEST": _abbreviation("Australia/Sydney"),
        "AEDT": _abbreviation("Australia/Sydney"),
        "ACST": _abbreviation("Australia/Adelaide"),
        "ACDT": _abbreviation("Australia/Adelaide"),
        "AWST": _abbreviation("Australia/Perth"),
        "NZST": _abbreviation("Pacific/Auckland"),
        "NZDT": _abbreviation("Pacific/Auckland"),
        "SAST": _abbreviation("Africa/Johannesburg"),
        "WAT": _abbreviation("Africa/Lagos"),
        "CAT": _abbreviation("Africa/Maputo"),
        "EAT": _abbreviation("Africa/Nairobi"),
    }
)
_REDIRECT_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {name.casefold(): name for name in TIMEZONE_REDIRECTS}
)


@lru_cache(maxsize=1)
def _zone_index() -> Mapping[str, str]:
    """Every accepted zone by its casefolded name. Built on first use, never at
    import: it walks the tz database, and a pool worker re-imports this module."""
    return MappingProxyType(
        {
            name.casefold(): name
            for name in guild_state._known_zones()
            if name in _BARE_ZONES
            or ("/" in name and name.partition("/")[0] in _ZONE_AREAS)
        }
    )


@lru_cache(maxsize=1)
def _city_index() -> Mapping[str, tuple[str, ...]]:
    """Every accepted area/city zone by its casefolded last segment. A city can
    have more than one: tzdata keeps old names such as `Asia/Istanbul` beside
    `Europe/Istanbul`."""
    cities: dict[str, list[str]] = {}
    for zone in sorted(_zone_index().values()):
        if "/" in zone:
            cities.setdefault(zone.rsplit("/", 1)[-1].casefold(), []).append(zone)
    return MappingProxyType({city: tuple(zones) for city, zones in cities.items()})


def warm_timezones() -> None:
    """Build the zone indexes ahead of the first -settings timezone, off the event
    loop. Never raises: a failure leaves the build to that first call."""
    try:
        _city_index()
    except Exception as e:  # noqa: BLE001 — a warm-up, retried by the first lookup
        log.warning(f"settings: timezone index not warmed: {e!r}")


def _quote_list(names: tuple[str, ...]) -> str:
    quoted = [f"`{name}`" for name in names]
    return (
        quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} or {quoted[-1]}"
    )


def _redirect_refusal(spec: SettingSpec, name: str) -> Refusal:
    redirect = TIMEZONE_REDIRECTS[name]
    target = redirect.targets[0]
    use = f"Use `-settings timezone {target}`."
    if len(redirect.targets) > 1:
        text = f"`{name}` is used by more than one place. Use {_quote_list(redirect.targets)}."
    elif redirect.reason == "synonym":
        text = f"`{name}` is another name for `{target}`. {use}"
    elif redirect.reason == "legacy":
        text = f"`{name}` is an old name for `{target}`. {use}"
    else:
        text = f"`{name}` is an abbreviation, not a place. {use}"
    return Refusal(reason=RefusalReason.TIMEZONE_REDIRECT, text=text, spec=spec)


def _parse_timezone(spec: SettingSpec, value: str) -> Parsed | Refusal:
    folded = value.strip().replace(" ", "_").casefold()
    if _FIXED_OFFSET_RE.fullmatch(folded.replace("_", "")):
        return Refusal(
            reason=RefusalReason.FIXED_OFFSET,
            text=(
                "Fixed UTC offsets aren't accepted: they ignore daylight saving. Use a "
                "city, like `-settings timezone Asia/Kolkata`, or `UTC`."
            ),
            spec=spec,
        )
    if (canonical := _zone_index().get(folded)) is not None:
        return Parsed(canonical)
    if (redirect := _REDIRECT_KEYS.get(folded)) is not None:
        return _redirect_refusal(spec, redirect)
    if (cities := _city_index().get(folded)) is not None:
        return Refusal(
            reason=RefusalReason.BAD_SHAPE,
            text=f"Timezones are named by area and city. Did you mean {_quote_list(cities)}?",
            spec=spec,
        )
    return Refusal(
        reason=RefusalReason.BAD_SHAPE,
        text=(
            "That isn't a timezone the bot knows. Use a city name like `Europe/London` "
            "or `America/New_York`, or `UTC`."
        ),
        spec=spec,
    )


def parse_value(spec: SettingSpec, value: str) -> Parsed | Refusal:
    """One value, which must match its kind's grammar in full, then the bounds."""
    match spec.kind:
        case SettingKind.DURATION | SettingKind.SECONDS | SettingKind.SECONDS_OR_OFF:
            result = _parse_time(spec, value)
        case SettingKind.PERCENT:
            result = _parse_percent(spec, value)
        case SettingKind.COUNT:
            result = _parse_count(spec, value)
        case SettingKind.SWITCH:
            result = _parse_switch(spec, value)
        case SettingKind.TIMEZONE:
            result = _parse_timezone(spec, value)
    if isinstance(result, Refusal):
        return result
    refusal = check_write(spec, result.value)
    if refusal is None:
        return result
    if spec.kind is SettingKind.DURATION and value.isascii() and value.isdigit():
        # A duration's range prints as a clock, which hides that a bare number is
        # seconds. The hint names the minutes only when they would be accepted.
        n = int(value)
        if check_write(spec, float(n * 60)) is None:
            refusal = replace(
                refusal,
                text=f"{refusal.text} `{n}` is read as seconds; `{n}m` is {n} {pluralize(n, 'minute')}.",
            )
    return refusal


# ── Keys and requests ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Suggestion:
    """No setting has that name; `name` is the closest setting's card name, if any."""

    name: str | None


def card_name(spec: SettingSpec) -> str:
    """The name the cards print for a setting: a server setting's label with
    dashes, which invariant 12 makes one of its names, and a bot setting's key."""
    if spec.scope is SettingScope.SERVER:
        return spec.label.casefold().replace(" ", "-")
    return spec.key


# A bot setting by its environment variable, folded as names are: the operator's
# other word for it, which is suggested rather than accepted.
_BOT_BY_ENV: Final[Mapping[str, SettingSpec]] = MappingProxyType(
    {
        _fold(spec.env): spec
        for spec in SETTINGS
        if spec.scope is SettingScope.BOT and spec.env is not None
    }
)


def find(token: str, scope: SettingScope) -> SettingSpec | Suggestion:
    """A key or alias in one scope, with case and `_`/`-` folded. The suggestion
    comes from that scope only, so a typo never names a bot key to a server."""
    if (spec := named(token, scope)) is not None:
        return spec
    names = _BY_NAME[scope]
    folded = _fold(token)
    if scope is SettingScope.BOT and (by_env := _BOT_BY_ENV.get(folded)) is not None:
        return Suggestion(card_name(by_env))
    close = difflib.get_close_matches(folded, list(names), n=1)
    return Suggestion(card_name(names[close[0]]) if close else None)


def wrong_scope_text(spec: SettingSpec, *, operator: bool) -> str:
    """For a name that exists only in the other scope; the caller knows who asked."""
    if spec.scope is SettingScope.SERVER:
        name = card_name(spec)
        return f"`{name}` is set per server. Use `-settings {name}` in that server."
    if operator:
        return f"`{spec.key}` is bot-wide. Use `-settings bot {spec.key}` to see or change it."
    return (
        f"`{spec.key}` is a bot-wide setting, which only the bot's operator can change. "
        "Run `-settings` for this server's."
    )


def is_bullet_shaped(tail: str) -> bool:
    """Whitespace between the prefix and the command word, as in a markdown list
    item (`- settings page is broken`), which discord.py still dispatches."""
    return tail[:1].isspace()


class SettingsAction(Enum):
    SHOW = "show"
    DETAIL = "detail"
    SET = "set"
    RESET = "reset"


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingsRequest:
    scope: SettingScope
    action: SettingsAction
    spec: SettingSpec | None = None
    value: SettingValue | None = None


# The longest real request, `timezone America/Argentina/ComodRivadavia`, is 41.
_MAX_REQUEST_CHARS: Final = 100
_KEY_SEPARATORS: Final = ("=", ":")

_TOO_MUCH: Final = Refusal(
    reason=RefusalReason.TOO_MUCH,
    text=(
        "That's more than `-settings` understands. Use one line, "
        "`-settings <setting> <value>`, e.g. `-settings volume 80`."
    ),
)
# The example names a setting of the scope asked about.
_RESET_WITHOUT_KEY: Final[dict[SettingScope, Refusal]] = {
    SettingScope.SERVER: Refusal(
        reason=RefusalReason.RESET_WITHOUT_KEY,
        text="Say which setting to reset, like `-settings volume reset`.",
    ),
    SettingScope.BOT: Refusal(
        reason=RefusalReason.RESET_WITHOUT_KEY,
        text="Say which bot setting to reset, like `-settings bot heartbeat reset`.",
    ),
}


def _undash(token: str) -> str:
    """The token without one or two leading dashes; a token of dashes alone, or
    with more than two, is left as it is and matches nothing."""
    body = token.lstrip(DASHES)
    return body if body and len(token) - len(body) <= 2 else token


def _is_reset_word(token: str) -> bool:
    folded = token.casefold()
    return folded in ("reset", "default") or _undash(token).casefold() == "reset"


def _unknown_key(scope: SettingScope, suggestion: Suggestion) -> Refusal:
    listing = "`-settings bot`" if scope is SettingScope.BOT else "`-settings`"
    hint = f" — did you mean `{suggestion.name}`?" if suggestion.name else "."
    return Refusal(
        reason=RefusalReason.UNKNOWN_KEY,
        text=f"There's no setting by that name{hint} Run {listing} for the list.",
    )


def _lookup(key: str, scope: SettingScope) -> SettingSpec | Refusal:
    found = find(key, scope)
    if isinstance(found, SettingSpec):
        return found
    other = SettingScope.BOT if scope is SettingScope.SERVER else SettingScope.SERVER
    if (spec := named(key, other)) is not None:
        return Refusal(
            reason=RefusalReason.WRONG_SCOPE,
            text=wrong_scope_text(spec, operator=False),
            spec=spec,
        )
    return _unknown_key(scope, found)


def _shape_accepted(spec: SettingSpec, value: str) -> bool:
    result = parse_value(spec, value)
    return isinstance(result, Parsed) or result.reason is not RefusalReason.BAD_SHAPE


def parse_settings_args(arg: str, *, tail: str) -> SettingsRequest | Refusal:
    """One `-settings` invocation. `arg` is the consume-rest argument; `tail` is the
    message after the prefix, because discord.py strips a line break between the
    command word and `arg`. A request is one line, and every word must be used."""
    if len(tail.strip().splitlines()) > 1:
        return _TOO_MUCH
    arg = arg.strip()
    if not arg:
        return SettingsRequest(scope=SettingScope.SERVER, action=SettingsAction.SHOW)
    if len(arg) > _MAX_REQUEST_CHARS:
        return _TOO_MUCH

    tokens = arg.split()
    # One `set` among the first two words, when something follows it.
    for i in range(min(2, len(tokens) - 1)):
        if _undash(tokens[i]).casefold() == "set":
            del tokens[i]
            break
    scope = SettingScope.SERVER
    if tokens and _undash(tokens[0]).casefold() == "bot":
        scope, tokens = SettingScope.BOT, tokens[1:]
    result = _parse_scoped(tokens, scope)
    return replace(result, scope=scope) if isinstance(result, Refusal) else result


def _unbracketed(words: list[str]) -> list[str]:
    """A value copied off a card with its `<…>` still around it: one pair, opening
    on the first word and closing on that word or a later one. Words after the
    pair stay, so a value with words left over is still refused as too much."""
    if not words or not words[0].startswith("<"):
        return words
    close = next((i for i, word in enumerate(words) if word.endswith(">")), None)
    if close is None:
        return words
    inside = words[: close + 1]
    inside[0] = inside[0][1:]
    inside[close] = inside[close][:-1]
    return [word for word in inside if word] + words[close + 1 :]


def _parse_scoped(tokens: list[str], scope: SettingScope) -> SettingsRequest | Refusal:
    """The words after the scope: a key, then nothing, a reset word or a value."""
    if not tokens:
        return SettingsRequest(scope=scope, action=SettingsAction.SHOW)

    # `reset <setting>`: the one other word order accepted.
    if _is_reset_word(tokens[0]):
        if len(tokens) == 1:
            return _RESET_WITHOUT_KEY[scope]
        if len(tokens) > 2:
            return _TOO_MUCH
        spec = _lookup(_undash(tokens[1]), scope)
        if isinstance(spec, Refusal):
            return spec
        return SettingsRequest(scope=scope, action=SettingsAction.RESET, spec=spec)

    # `volume=50`, `volume: 50`, `volume = 50`: the key splits at its first separator.
    key_token, rest = _undash(tokens[0]), tokens[1:]
    cut = min(
        (key_token.find(sep) for sep in _KEY_SEPARATORS if sep in key_token), default=-1
    )
    if cut >= 0:
        key_token, remainder = key_token[:cut], key_token[cut + 1 :]
        if remainder:
            rest = [remainder, *rest]
    if rest and rest[0] in _KEY_SEPARATORS:
        rest = rest[1:]

    spec = _lookup(key_token, scope)
    if isinstance(spec, Refusal):
        return spec
    if not rest:
        return SettingsRequest(scope=scope, action=SettingsAction.DETAIL, spec=spec)
    if _is_reset_word(rest[0]):
        if len(rest) > 1:
            return _TOO_MUCH
        return SettingsRequest(scope=scope, action=SettingsAction.RESET, spec=spec)

    joiner = "_" if spec.kind is SettingKind.TIMEZONE else " "
    rest = _unbracketed(rest)
    result = parse_value(spec, joiner.join(rest))
    if isinstance(result, Parsed):
        return SettingsRequest(
            scope=scope, action=SettingsAction.SET, spec=spec, value=result.value
        )
    # Not a value, but one once trailing words are dropped: words left over. A
    # whole value refused for its range or its zone keeps that refusal.
    if result.reason is RefusalReason.BAD_SHAPE and any(
        _shape_accepted(spec, joiner.join(rest[:n])) for n in range(1, len(rest))
    ):
        return _TOO_MUCH
    return result


# ── What a setting accepts ──────────────────────────────────────────────────────


def allowed_text(spec: SettingSpec, *, now: bool = False) -> str:
    """What a spec accepts, as the cards and the detail view print it. A write-time
    bound follows another setting's current value, so only `now` (a render at the
    call) applies it; otherwise this is the static range."""
    match spec.kind:
        case SettingKind.SWITCH:
            return "on or off"
        case SettingKind.TIMEZONE:
            return "a city like Europe/London, or UTC"
    lo, hi = write_range(spec) if now else (bound(spec.minimum), bound(spec.maximum))
    lo, hi = lo or 0, hi or 0
    if lo > hi:
        return f"none right now: {_no_room_why(spec, lo, hi)}"
    off = " or off" if spec.kind is SettingKind.SECONDS_OR_OFF else ""
    return f"{format_value(spec, lo)}–{format_value(spec, hi)}{off}"
