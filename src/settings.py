"""The -settings machinery: the registry of every setting chat may show or change,
the grammar that reads one line of chat into a request against it, and the objects
that hold what is stored.

config.py reads the environment and holds each tunable's parsed value; this module
decides what chat may set, in what shape and within what range. The registry and
the grammar do no IO, and every refusal carries the text a reply shows, none of it
quoting the input. BotSettings applies the operator's stored overrides to config's
accessors; GuildSettings caches every server's guild:{id}:config and is its only
writer.
"""

import asyncio
import difflib
import math
import os
import re
import textwrap
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, get_args
from zoneinfo import ZoneInfo

import redis.asyncio as aioredis
from discord.ext import commands

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
    GuildConfig,
    is_config_field,
    valid_timezone,
)
from src.redis_client import (
    BOT_CONFIG_KEY,
    BotConfigStore,
    GuildRedisStore,
    read_config_writers,
    read_guild_configs,
    scan_guild_config_ids,
)
from src.util import DASHES, fmt_duration, fmt_seconds, get_logger

if TYPE_CHECKING:
    # For annotations only: nothing here needs -debug's module at run time.
    from src.debug import DebugSettings

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
    summary: str
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
    # Completes "has to be between A and B here: ..." when write_minimum refuses;
    # {bound} is that minimum, rendered.
    why_write_minimum: str | None = None
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


def _server_bound(field: ConfigFieldName, side: Literal["lo", "hi"]) -> float:
    return getattr(CONFIG_DOMAIN[field], side)


def _play_resolve_concurrency_max() -> float:
    # One worker stays out of any one server's reach.
    return max(1, config.YTDLP_POOL_WORKERS - 1)


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
        minimum=_server_bound(ConfigField.VOLUME, "lo") * 100,
        maximum=_server_bound(ConfigField.VOLUME, "hi") * 100,
        default=DEFAULT_VOLUME * 100,
    ),
    SettingSpec(
        key="timezone",
        aliases=("tz",),
        scope=SettingScope.SERVER,
        kind=SettingKind.TIMEZONE,
        group=SettingGroup.PLAYBACK,
        label="Timezone",
        summary='The clock "Est. playing at" and "Estimated finish" are shown in.',
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
        summary="How long the bot stays in voice with nothing queued.",
        applies="the next time the queue runs empty",
        field=ConfigField.IDLE_TIMEOUT,
        minimum=_server_bound(ConfigField.IDLE_TIMEOUT, "lo"),
        maximum=_server_bound(ConfigField.IDLE_TIMEOUT, "hi"),
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
        summary=(
            "How long the bot waits alone in voice before leaving. Music keeps playing "
            "while it waits, and those songs count toward history."
        ),
        applies="the next time the channel empties",
        field=ConfigField.ALONE_TIMEOUT,
        minimum=_server_bound(ConfigField.ALONE_TIMEOUT, "lo"),
        maximum=_server_bound(ConfigField.ALONE_TIMEOUT, "hi"),
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
        minimum=_server_bound(ConfigField.NP_REFRESH, "lo"),
        maximum=_server_bound(ConfigField.NP_REFRESH, "hi"),
        # Never faster than the bot: its value is the channel's edit budget.
        write_minimum=config.now_playing_update_interval_secs,
        why_write_minimum="the bot refreshes no faster than {bound}",
    ),
    SettingSpec(
        key="debug",
        aliases=("debug-footer",),
        scope=SettingScope.SERVER,
        kind=SettingKind.SWITCH,
        group=SettingGroup.DIAGNOSTICS,
        label="Debug footer",
        summary=(
            "A footer on every embed here with the trace id, timing and the bot "
            "process's load, readable by anyone in the channel."
        ),
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
        summary="How often the Now Playing bar moves, in every server.",
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
        summary=(
            "How often a playing server records its position; a crash replays at "
            "most this much."
        ),
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
        summary=(
            'How long a lookup for one song runs before "still looking it up" '
            "appears, in every server."
        ),
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
        key="play-inflight-max",
        aliases=("inflight-max",),
        scope=SettingScope.BOT,
        kind=SettingKind.COUNT,
        group=SettingGroup.LIMITS,
        label="Inflight max",
        summary="How many -play requests one server may have in progress at once.",
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
        summary="How many of one server's -play requests may use a lookup worker at once.",
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
        summary="How long a -play waits for a lookup worker before giving up.",
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
        summary="How long the check that a song's stream still works may take.",
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
        summary="The shortest gap between -ping's edits as its checks come back.",
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
        summary="How long -ping waits for a check before marking it failed.",
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
        summary="The longest -debug's card goes without an edit while its blocks arrive.",
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
        summary='How long a -debug block may collect before it shows "timed out".',
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
        summary="How long -analytics waits for its chart before sending the card without one.",
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
        summary=(
            "The debug footer for every server that has not chosen for itself. It "
            "lasts until the bot restarts."
        ),
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


def followed_knob(spec: SettingSpec) -> config.FloatKnob | config.IntKnob | None:
    """The bot knob a server setting runs on while unset: the one the bot setting
    of the same key overrides. None for a bot setting, or a server one without."""
    if spec.scope is not SettingScope.SERVER:
        return None
    bot = _BY_NAME[SettingScope.BOT].get(_fold(spec.key))
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
    BOT_WRITE_UNAVAILABLE = "bot_write_unavailable"
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
    floor: float | None = None,
) -> Refusal:
    """`floor`: the write-only minimum the value missed, when it is the binding one."""
    text = f"**{spec.label}** has to be {_range_text(spec, lo, hi)}"
    if floor is not None and spec.why_write_minimum is not None:
        bold = f"**{format_value(spec, floor)}**"
        text += f" here: {spec.why_write_minimum.format(bound=bold)}."
    else:
        why = spec.why_minimum if side == "minimum" else spec.why_maximum
        text += f". {why}" if why else "."
    return Refusal(reason=RefusalReason.OUT_OF_RANGE, text=text, spec=spec, side=side)


def check_write(spec: SettingSpec, value: SettingValue) -> Refusal | None:
    """in_bounds, then the write-only bounds that follow another setting. A
    refusal names the range a write may take, write-only bounds included."""
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    floor = spec.write_minimum() if spec.write_minimum is not None else None
    ceiling = spec.write_maximum() if spec.write_maximum is not None else None
    write_lo = floor if lo is None else lo if floor is None else max(floor, lo)
    write_hi = ceiling if hi is None else hi if ceiling is None else min(ceiling, hi)
    if isinstance(value, (bool, str)) or (
        spec.kind is SettingKind.SECONDS_OR_OFF and value == OFF_SECS
    ):
        if in_bounds(spec, value):
            return None
        return _out_of_range(spec, "maximum", write_lo, write_hi)
    if in_bounds(spec, value):
        if floor is not None and value < floor:
            return _out_of_range(spec, "minimum", write_lo, write_hi, floor=floor)
        if ceiling is not None and value > ceiling:
            return _out_of_range(spec, "maximum", write_lo, write_hi)
        return None
    if lo is not None and value < lo:
        dynamic = floor if floor is not None and floor > lo else None
        return _out_of_range(spec, "minimum", write_lo, write_hi, floor=dynamic)
    return _out_of_range(spec, "maximum", write_lo, write_hi)


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
    """An in-range value for a bad-shape hint: the middle of the range, in whole
    minutes once that is past two of them."""
    lo, hi = bound(spec.minimum), bound(spec.maximum)
    if lo is None or hi is None:
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


def warm_timezones() -> None:
    """Build the zone index ahead of the first -settings timezone, off the event
    loop. Never raises: a failure leaves the build to that first call."""
    try:
        _zone_index()
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
    return refusal if refusal is not None else result


# ── Keys and requests ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Suggestion:
    """No setting has that name; `key` is the closest one in the scope, if any."""

    key: str | None


def find(token: str, scope: SettingScope) -> SettingSpec | Suggestion:
    """A key or alias in one scope, with case and `_`/`-` folded. The suggestion
    comes from that scope only, so a typo never names a bot key to a server."""
    names = _BY_NAME[scope]
    folded = _fold(token)
    if (spec := names.get(folded)) is not None:
        return spec
    close = difflib.get_close_matches(folded, list(names), n=1)
    return Suggestion(names[close[0]].key if close else None)


def wrong_scope_text(spec: SettingSpec, *, operator: bool) -> str:
    """For a name that exists only in the other scope; the caller knows who asked."""
    if spec.scope is SettingScope.SERVER:
        return f"`{spec.key}` is set per server. Use `-settings {spec.key}` in that server."
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
_RESET_WITHOUT_KEY: Final = Refusal(
    reason=RefusalReason.RESET_WITHOUT_KEY,
    text="Say which setting to reset, like `-settings volume reset`.",
)


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
    hint = f" — did you mean `{suggestion.key}`?" if suggestion.key else "."
    return Refusal(
        reason=RefusalReason.UNKNOWN_KEY,
        text=f"There's no setting by that name{hint} Run {listing} for the list.",
    )


def _lookup(key: str, scope: SettingScope) -> SettingSpec | Refusal:
    found = find(key, scope)
    if isinstance(found, SettingSpec):
        return found
    other = SettingScope.BOT if scope is SettingScope.SERVER else SettingScope.SERVER
    if (spec := _BY_NAME[other].get(_fold(key))) is not None:
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


def _parse_scoped(tokens: list[str], scope: SettingScope) -> SettingsRequest | Refusal:
    """The words after the scope: a key, then nothing, a reset word or a value."""
    if not tokens:
        return SettingsRequest(scope=scope, action=SettingsAction.SHOW)

    # `reset <setting>`: the one other word order accepted.
    if _is_reset_word(tokens[0]):
        if len(tokens) == 1:
            return _RESET_WITHOUT_KEY
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
    result = parse_value(spec, joiner.join(rest))
    if isinstance(result, Parsed):
        return SettingsRequest(
            scope=scope, action=SettingsAction.SET, spec=spec, value=result.value
        )
    # A value that parses once trailing words are dropped had words left over.
    if any(_shape_accepted(spec, joiner.join(rest[:n])) for n in range(1, len(rest))):
        return _TOO_MUCH
    return result


# ── Help ────────────────────────────────────────────────────────────────────────

# help.py's code-block entry layout: wrapped at 48 columns, the summary indented.
_HELP_WIDTH: Final = 48
_HELP_INDENT: Final = "    "


def allowed_text(spec: SettingSpec, *, now: bool = False) -> str:
    """What a spec accepts, as the help and the detail view print it. A write-time
    minimum follows the bot, so only `now` (a render at the call) quotes its value;
    the help is built once, at import, and names it instead."""
    match spec.kind:
        case SettingKind.SWITCH:
            return "on or off"
        case SettingKind.TIMEZONE:
            return "a city like Europe/London, or UTC"
    static_lo = bound(spec.minimum) or 0
    if spec.write_minimum is None:
        lo = format_value(spec, static_lo)
    elif now:
        lo = format_value(spec, max(static_lo, spec.write_minimum()))
    else:
        lo = "the bot's value"
    hi = format_value(spec, bound(spec.maximum) or 0)
    return f"{lo}–{hi}" + (" or off" if spec.kind is SettingKind.SECONDS_OR_OFF else "")


def help_sections() -> list[tuple[str, list[list[str]]]]:
    """The server settings as `-help settings` entries, one per spec: its names and
    range over its summary. Bot settings are the operator's, so none is listed."""
    entries = [
        textwrap.wrap(
            f"{', '.join((spec.key, *spec.aliases))}  {allowed_text(spec)}",
            _HELP_WIDTH,
            subsequent_indent=_HELP_INDENT + "  ",
        )
        + textwrap.wrap(
            spec.summary,
            _HELP_WIDTH,
            initial_indent=_HELP_INDENT,
            subsequent_indent=_HELP_INDENT,
        )
        for spec in SETTINGS
        if spec.scope is SettingScope.SERVER
    ]
    return [("SETTINGS", entries)]


# ── Bot settings: the operator's overrides of config's accessors ─────────────

# Every Redis call a settings path awaits runs under this. The pool has no
# socket_timeout, so a Redis that accepts connections and then stops answering
# would otherwise hold the caller for good.
CONFIG_IO_TIMEOUT_SECS: Final[float] = 2.0

# The bot specs stored in bot:{application_id}:config: every one but debug-default.
_STORED_BOT_SPECS: Final[tuple[SettingSpec, ...]] = tuple(
    spec for spec in SETTINGS if spec.scope is SettingScope.BOT and spec.field
)


def _set_knob(attr: config.FloatKnob | config.IntKnob, value: SettingValue) -> None:
    """config.set_override, narrowed from the registry's value union. A value of
    the wrong type is a registry bug, never user input: parse_value refused that."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{attr} takes a number; got {value!r}")
    if config.is_int_knob(attr):
        if not isinstance(value, int):
            raise TypeError(f"{attr} takes an int; got {value!r}")
        config.set_override(attr, value)
    else:
        config.set_override(attr, value)


class BotSettings:
    """The operator's bot-wide settings for this process. hydrate() applies what
    bot:{application_id}:config holds; apply() and reset() change one setting in
    memory. debug-default is never stored: it is held here, so a cog reload can
    hand it to the new cog's DebugSettings. The one src/ caller of
    config.set_override and config.clear_override. Built in setup_hook.
    See docs/ARCHITECTURE.md#settings-resolution."""

    def __init__(
        self,
        bot: commands.Bot | commands.AutoShardedBot,
        *,
        redis: Optional[aioredis.Redis],
        ignore_stored: bool,
    ) -> None:
        self._bot = bot
        self._redis = redis
        # BOT_SETTINGS_OVERRIDES=ignore: the key is never read, and a stored
        # setting cannot be changed.
        self.ignore_stored = ignore_stored
        # False until a read of the key succeeded, or there is none to make.
        self.hydrated = False
        self._debug_default: Optional[bool] = None
        # apply() and reset() stamp their knob, so a hydrate whose read straddled
        # one leaves that knob as the command set it.
        self._seq = 0
        self._changed_at: dict[str, int] = {}

    def _debug_settings(self) -> Optional[DebugSettings]:
        # Looked up at call time, so a reloaded cog is the one that receives it.
        return getattr(self._bot.get_cog("MusicBot"), "debug_settings", None)

    def _stamp(self, attr: str) -> None:
        self._seq += 1
        self._changed_at[attr] = self._seq

    def apply(self, spec: SettingSpec, value: SettingValue) -> bool:
        """Make `value` this process's setting. False, with nothing changed, for a
        stored setting while stored ones are ignored. A value the registry refuses
        raises ValueError: every caller has already refused it as input."""
        if spec.scope is not SettingScope.BOT or not in_bounds(spec, value):
            raise ValueError(f"{spec.key}={value!r} is not a bot setting value")
        if spec.field is None:
            if not isinstance(value, bool):
                raise TypeError(f"{spec.key} takes on or off; got {value!r}")
            self._debug_default = value
            if (debug_settings := self._debug_settings()) is not None:
                debug_settings.set_default_override(value)
            return True
        if self.ignore_stored:
            return False
        if spec.attr is None:
            raise ValueError(f"{spec.key} names no config knob")
        _set_knob(spec.attr, value)
        self._stamp(spec.attr)
        return True

    def reset(self, spec: SettingSpec) -> bool:
        """Return the setting to its environment value (debug-default: to
        DEBUG_MODE). False, with nothing changed, as for apply()."""
        if spec.scope is not SettingScope.BOT:
            raise ValueError(f"{spec.key} is not a bot setting")
        if spec.field is None:
            self._debug_default = None
            if (debug_settings := self._debug_settings()) is not None:
                debug_settings.set_default_override(None)
            return True
        if self.ignore_stored:
            return False
        if spec.attr is None:
            raise ValueError(f"{spec.key} names no config knob")
        config.clear_override(spec.attr)
        self._stamp(spec.attr)
        return True

    def reapply_debug_default(self, debug_settings: DebugSettings) -> None:
        """Give a newly loaded cog this session's debug-default."""
        if self._debug_default is not None:
            debug_settings.set_default_override(self._debug_default)

    async def hydrate(self) -> None:
        """Apply every stored override, reading under CONFIG_IO_TIMEOUT_SECS.
        Never raises. A failed read leaves every knob on its environment value
        and `hydrated` False, for on_ready to retry. A stored value outside the
        registry's current bounds is skipped with a WARNING, never applied: the
        key outlives builds, and its bounds can change between them."""
        application_id = self._bot.application_id
        if self.hydrated or self._redis is None or application_id is None:
            return
        key = BOT_CONFIG_KEY.format(application_id=application_id)
        if self.ignore_stored:
            log.warning(
                f"BOT_SETTINGS_OVERRIDES=ignore: {key} is not read, so every bot "
                "setting runs on its environment or code value"
            )
            self.hydrated = True
            return
        started = self._seq
        try:
            async with asyncio.timeout(CONFIG_IO_TIMEOUT_SECS):
                stored = await BotConfigStore(self._redis, application_id).read_config()
        except TimeoutError:
            stored = None
        if stored is None:
            log.warning(
                f"bot settings unavailable ({key} could not be read); running on "
                "environment values"
            )
            return
        applied: list[str] = []
        for spec in _STORED_BOT_SPECS:
            if spec.field is None or spec.attr is None or spec.env is None:
                continue
            value: Optional[float] = getattr(stored, spec.field)
            if value is None or self._changed_at.get(spec.attr, 0) > started:
                continue
            if not in_bounds(spec, value):
                log.warning(
                    f"bot setting {spec.key}={value!r} in {key} is outside "
                    f"{allowed_text(spec)}; ignored"
                )
                continue
            _set_knob(spec.attr, value)
            shown = f"{spec.key}={format_value(spec, value)}"
            applied.append(shown)
            if env_value := (os.environ.get(spec.env) or "").strip():
                log.warning(
                    f"bot setting {shown} ({key}) overrides {spec.env}={env_value}; "
                    f"undo with -settings bot {spec.key} reset, just bot-settings "
                    f"reset {application_id}, or BOT_SETTINGS_OVERRIDES=ignore"
                )
        if applied:
            log.info(f"bot settings applied from {key}: {', '.join(applied)}")
        self.hydrated = True


# ── Server settings: the guild:{id}:config cache and its one writer ──────────

ALL_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    get_args(ConfigFieldName.__value__)
)
# One frozenset per shape of known fields (at most 2**7), shared by every entry.
_KNOWN_SHAPES: Final[dict[frozenset[str], frozenset[str]]] = {
    ALL_CONFIG_FIELDS: ALL_CONFIG_FIELDS
}
_UNSET_CONFIG: Final = GuildConfig()
_HYDRATE_RETRY_FIRST_SECS: Final[float] = 1.0
_HYDRATE_RETRY_MAX_SECS: Final[float] = 60.0
# Keys the orphan sweep deletes per start at most; the rest wait for the next.
_ORPHAN_SWEEP_MAX: Final[int] = 500


class WriteMode(Enum):
    SET = "set"  # overwrite the field
    SEED = "seed"  # volume only, HSETNX: restore's one-release legacy migration


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteResult:
    # False: refused (the guild was forgotten, a SEED was superseded or did not
    # persist), and nothing changed.
    applied: bool
    # The store call returned True inside CONFIG_IO_TIMEOUT_SECS. False is "not
    # confirmed", not "not written": a pipeline sends before it reads replies.
    persisted: bool
    # The cached config immediately before this commit.
    previous: Optional[GuildConfig]


@dataclass(frozen=True, slots=True)
class _Entry:
    config: GuildConfig
    # The fields a successful read or a write has covered; the rest read as unset.
    known: frozenset[str]


_EMPTY: Final = _Entry(_UNSET_CONFIG, ALL_CONFIG_FIELDS)


class _GuildLock:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


class _SettingsBot(Protocol):
    @property
    def application_id(self) -> Optional[int]: ...
    def is_closed(self) -> bool: ...


class _SettingsPlayer(Protocol):
    volume: float
    timezone: ZoneInfo


class _SettingsCog(Protocol):
    """What GuildSettings reads off the cog, each at call time."""

    @property
    def redis(self) -> Optional[aioredis.Redis]: ...
    @property
    def mps(self) -> Mapping[int, _SettingsPlayer]: ...
    @property
    def debug_settings(self) -> DebugSettings: ...
    @property
    def bot(self) -> _SettingsBot: ...


class GuildSettings:
    """Every server's guild:{id}:config, cached, and the only writer of that key.

    Hot paths read synchronously (peek and the accessors) and never await. Three
    readers fill the cache — a restore's snapshot (seed), the startup pass
    (hydrate) and the -settings command (load) — and every write, reset and forget
    goes through here, serialized per guild. One sequence counter stamps each
    committed (guild, field), and a read skips a field stamped after it began, so a
    read that straddles a write never undoes it. debug_mode is projected into
    DebugSettings. See docs/ARCHITECTURE.md#settings-resolution."""

    def __init__(self, cog: _SettingsCog) -> None:
        self._cog = cog
        self._entries: dict[int, _Entry] = {}
        self._seq = 0
        self._stamps: dict[tuple[int, str], int] = {}
        self._forgotten_at: dict[int, int] = {}
        # (guild, field) pairs whose last write did not reach Redis.
        self._unpersisted: set[tuple[int, str]] = set()
        # The `started` value of every open reading(), counted.
        self._readers: dict[int, int] = {}
        self._locks: dict[int, _GuildLock] = {}
        self._loads: dict[int, asyncio.Task[Optional[GuildConfig]]] = {}

    # ── Synchronous reads ─────────────────────────────────────────────────────

    def peek(self, guild_id: int) -> Optional[GuildConfig]:
        """The cached config, or None when nothing is cached. A field no read or
        write has covered reads as unset."""
        entry = self._entries.get(guild_id)
        return entry.config if entry is not None else None

    def is_complete(self, guild_id: int) -> bool:
        """True once a successful read has covered every field."""
        entry = self._entries.get(guild_id)
        return entry is not None and entry.known == ALL_CONFIG_FIELDS

    def is_persisted(self, guild_id: int, field: str) -> bool:
        """False while the field's last write had not reached Redis."""
        return (guild_id, field) not in self._unpersisted

    # The accessors: synchronous and total. An unread or unset field is the
    # default; a stored value already passed CONFIG_DOMAIN when it was parsed.

    def idle_timeout_secs(self, guild_id: int) -> float:
        """How long the playback loop waits for the next song before leaving."""
        stored = self.peek(guild_id)
        value = stored.idle_timeout_secs if stored is not None else None
        return DEFAULT_IDLE_TIMEOUT_SECS if value is None else value

    def alone_timeout_secs(self, guild_id: int) -> float:
        """How long the bot waits alone in its voice channel before leaving."""
        stored = self.peek(guild_id)
        value = stored.alone_timeout_secs if stored is not None else None
        return DEFAULT_ALONE_TIMEOUT_SECS if value is None else value

    def np_refresh_secs(self, guild_id: int) -> float:
        """How often the Now Playing bar moves: the bot's value while unset, and
        never faster than it. The bot's value is read at the call."""
        bot = config.now_playing_update_interval_secs()
        stored = self.peek(guild_id)
        value = stored.np_refresh_secs if stored is not None else None
        return bot if value is None else max(value, bot)

    # ── Stamps and registrations ──────────────────────────────────────────────

    @contextmanager
    def reading(self) -> Iterator[int]:
        """Register a read. Yields `started`, the counter's value now: a write that
        commits later stamps above it. Enter it immediately before the read and
        stay inside until the last use of `started`, so pruning keeps every stamp
        it will be compared with."""
        started = self._seq
        self._readers[started] = self._readers.get(started, 0) + 1
        try:
            yield started
        finally:
            if (remaining := self._readers[started] - 1) > 0:
                self._readers[started] = remaining
            else:
                del self._readers[started]
            self._prune()

    def _prune(self) -> None:
        """Drop the stamps and forgets no registered read could see: an absent one
        compares as never set, which is what `stamp > started` already answers for
        every stamp at or below the lowest open `started`, and for any later read."""
        low = min(self._readers, default=self._seq)
        if self._stamps:
            self._stamps = {k: v for k, v in self._stamps.items() if v > low}
        if self._forgotten_at:
            self._forgotten_at = {
                g: v for g, v in self._forgotten_at.items() if v > low
            }

    def _require_open(self, started: int) -> None:
        if started not in self._readers:
            raise ValueError(f"{started} is not an open reading() registration")

    def _store_entry(
        self, guild_id: int, config: GuildConfig, known: frozenset[str]
    ) -> None:
        known = _KNOWN_SHAPES.setdefault(known, known)
        if known is ALL_CONFIG_FIELDS and config == _UNSET_CONFIG:
            self._entries[guild_id] = _EMPTY
        else:
            self._entries[guild_id] = _Entry(config, known)

    # ── Reads into the cache ──────────────────────────────────────────────────

    def _merge(
        self, guild_id: int, config: GuildConfig, started: int
    ) -> frozenset[str]:
        """The merge every reader shares. Nothing for a guild forgotten since the
        read began; otherwise every field not stamped since, and not marked
        unpersisted — an unsaved write keeps its value until a later write reaches
        Redis. Returns the accepted fields; the caller projects debug_mode."""
        if self._forgotten_at.get(guild_id, 0) > started:
            return frozenset()
        accepted = frozenset(
            field
            for field in ALL_CONFIG_FIELDS
            if self._stamps.get((guild_id, field), 0) <= started
            and (guild_id, field) not in self._unpersisted
        )
        if not accepted:
            return accepted
        entry = self._entries.get(guild_id)
        base = entry.config if entry is not None else _UNSET_CONFIG
        merged = replace(base, **{field: getattr(config, field) for field in accepted})
        known = accepted if entry is None else entry.known | accepted
        self._store_entry(guild_id, merged, known)
        return accepted

    def seed(
        self, guild_id: int, config: GuildConfig, *, started: int
    ) -> frozenset[str]:
        """Merge a restore snapshot's config, read inside reading() as `started`.
        Returns the accepted fields: restore assigns volume and timezone, and
        migrates the legacy volume, only for those."""
        self._require_open(started)
        accepted = self._merge(guild_id, config, started)
        if ConfigField.DEBUG_MODE in accepted:
            self._cog.debug_settings.apply_choices(
                {guild_id: config.debug_mode}, persisted=True
            )
        return accepted

    async def hydrate(self, guild_ids: Iterable[int]) -> set[int]:
        """One bounded read of every guild's config, merged. Returns the guilds it
        could not read (a failed or timed-out batch); none without Redis."""
        redis = self._cog.redis
        ids = list(guild_ids)
        if redis is None or not ids:
            return set()
        with self.reading() as started:
            configs = await read_guild_configs(
                redis, ids, batch_timeout=CONFIG_IO_TIMEOUT_SECS
            )
            choices: dict[int, Optional[bool]] = {}
            for guild_id, stored in configs.items():
                if ConfigField.DEBUG_MODE in self._merge(guild_id, stored, started):
                    choices[guild_id] = stored.debug_mode
            if choices:
                self._cog.debug_settings.apply_choices(choices, persisted=True)
        return set(ids) - configs.keys()

    async def retry_hydrate(self, omitted: set[int]) -> None:
        """Re-read the guilds a hydrate left out, backing off, until none is left.
        A guild a seed or load completed meanwhile, or one forgotten since this
        began, drops out: re-reading a departed guild would re-cache it."""
        pending = set(omitted)
        delay = _HYDRATE_RETRY_FIRST_SECS
        with self.reading() as created:
            while pending:
                await asyncio.sleep(delay)
                delay = min(delay * 2, _HYDRATE_RETRY_MAX_SECS)
                pending = {
                    guild_id
                    for guild_id in pending
                    if not self.is_complete(guild_id)
                    and self._forgotten_at.get(guild_id, 0) <= created
                }
                if pending:
                    pending = await self.hydrate(pending)

    async def load(self, guild_id: int) -> Optional[GuildConfig]:
        """The guild's merged config, read now; None when the read failed or timed
        out, which caches nothing. For the -settings command only. Concurrent
        callers share one read, and cancelling one does not cancel it for the rest."""
        job = self._loads.get(guild_id)
        if job is None:
            job = asyncio.ensure_future(self._load(guild_id))
            self._loads[guild_id] = job
            job.add_done_callback(lambda done: self._finish_load(guild_id, done))
        return await asyncio.shield(job)

    def _finish_load(
        self, guild_id: int, done: asyncio.Task[Optional[GuildConfig]]
    ) -> None:
        if self._loads.get(guild_id) is done:
            del self._loads[guild_id]

    async def _load(self, guild_id: int) -> Optional[GuildConfig]:
        redis = self._cog.redis
        if redis is None:
            return None
        with self.reading() as started:
            try:
                async with asyncio.timeout(CONFIG_IO_TIMEOUT_SECS):
                    stored = await GuildRedisStore(redis, guild_id).read_config()
            except TimeoutError:
                stored = None
            if stored is None:
                return None
            if ConfigField.DEBUG_MODE in self._merge(guild_id, stored, started):
                self._cog.debug_settings.apply_choices(
                    {guild_id: stored.debug_mode}, persisted=True
                )
        return self.peek(guild_id)

    # ── The write path ────────────────────────────────────────────────────────

    @asynccontextmanager
    async def _guild_lock(self, guild_id: int) -> AsyncIterator[None]:
        """Per-guild, created on demand and dropped when its last user leaves. Every
        waiter holds the same Lock: an entry deleted under a waiter would let the
        next caller build a second Lock and run beside it."""
        entry = self._locks.get(guild_id)
        if entry is None:
            entry = self._locks[guild_id] = _GuildLock()
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[guild_id]

    async def _store_io(
        self, guild_id: int, call: Callable[[GuildRedisStore], Awaitable[bool]]
    ) -> bool:
        """One store call under CONFIG_IO_TIMEOUT_SECS; False without Redis or on
        a timeout. The store itself never raises (@_guild_op)."""
        redis = self._cog.redis
        if redis is None:
            return False
        try:
            async with asyncio.timeout(CONFIG_IO_TIMEOUT_SECS):
                return await call(GuildRedisStore(redis, guild_id))
        except TimeoutError:
            log.warning(f"[guild:{guild_id}] config write timed out; not confirmed")
            return False

    async def write(
        self,
        guild_id: int,
        change: GuildConfig,
        *,
        mode: WriteMode = WriteMode.SET,
        since: Optional[int] = None,
        player: Optional[_SettingsPlayer] = None,
    ) -> WriteResult:
        """Set the one field `change` sets: the store call, then a synchronous
        commit that stamps it, updates the cache, marks it unpersisted when the
        call was not confirmed, projects debug_mode and assigns the live player's
        volume or timezone. `player` defaults to the guild's registered player.

        SEED (volume only) is restore's migration: `since` is restore's open
        registration, and a volume write or reset stamped after it refuses the
        seed. A SEED that did not persist changes nothing, so the next restore
        retries it. `change` setting anything but exactly one field raises."""
        stored = change.to_redis()
        name = next(iter(stored), None)
        if len(stored) != 1 or name is None or not is_config_field(name):
            raise ValueError(f"a write sets exactly one config field; got {stored}")
        if mode is WriteMode.SEED:
            if name != ConfigField.VOLUME or since is None:
                raise ValueError("SEED writes volume, with `since`")
            self._require_open(since)
        elif since is not None:
            raise ValueError("`since` is for SEED")
        if name == ConfigField.TIMEZONE and not valid_timezone(stored[name]):
            raise ValueError(f"{stored[name]!r} is not a zone this host knows")
        writer = self._cog.bot.application_id
        with self.reading() as started:
            async with self._guild_lock(guild_id):
                previous = self.peek(guild_id)
                if self._forgotten_at.get(guild_id, 0) > started or (
                    since is not None
                    and self._stamps.get((guild_id, ConfigField.VOLUME), 0) > since
                ):
                    return WriteResult(
                        applied=False, persisted=False, previous=previous
                    )
                persisted = await self._store_io(
                    guild_id,
                    lambda store: self._dispatch(store, change, mode, writer),
                )
                if mode is WriteMode.SEED and not persisted:
                    return WriteResult(
                        applied=False, persisted=False, previous=previous
                    )
                self._commit(
                    guild_id,
                    name,
                    getattr(change, name),
                    persisted=persisted,
                    live=mode is WriteMode.SET,
                    player=player,
                )
                return WriteResult(applied=True, persisted=persisted, previous=previous)

    async def reset(
        self,
        guild_id: int,
        field: ConfigFieldName,
        *,
        player: Optional[_SettingsPlayer] = None,
    ) -> WriteResult:
        """Delete one stored choice, on write's order. volume also clears the legacy
        :state copy. The live player returns to the default volume or zone."""
        with self.reading() as started:
            async with self._guild_lock(guild_id):
                previous = self.peek(guild_id)
                if self._forgotten_at.get(guild_id, 0) > started:
                    return WriteResult(
                        applied=False, persisted=False, previous=previous
                    )
                persisted = await self._store_io(
                    guild_id, lambda store: self._dispatch_reset(store, field)
                )
                self._commit(
                    guild_id, field, None, persisted=persisted, live=True, player=player
                )
                return WriteResult(applied=True, persisted=persisted, previous=previous)

    async def forget(self, guild_id: int) -> bool:
        """Delete a departed guild's config and everything cached for it, under the
        guild's lock. Returns whether the DELETE was confirmed. A read that began
        before this, and a write queued behind it, are refused."""
        async with self._guild_lock(guild_id):
            persisted = await self._store_io(
                guild_id, lambda store: store.clear_config()
            )
            self._seq += 1
            self._forgotten_at[guild_id] = self._seq
            self._entries.pop(guild_id, None)
            self._stamps = {k: v for k, v in self._stamps.items() if k[0] != guild_id}
            self._unpersisted = {k for k in self._unpersisted if k[0] != guild_id}
            self._cog.debug_settings.drop(guild_id)
            self._prune()
            return persisted

    @staticmethod
    async def _dispatch(
        store: GuildRedisStore,
        change: GuildConfig,
        mode: WriteMode,
        writer: Optional[int],
    ) -> bool:
        """The one store method each field writes through."""
        if change.volume is not None:
            if mode is WriteMode.SEED:
                return await store.migrate_volume(change.volume, writer=writer)
            return await store.set_volume(change.volume, writer=writer)
        if change.timezone is not None:
            return await store.set_timezone(change.timezone, writer=writer)
        if change.debug_mode is not None:
            return await store.set_debug_mode(change.debug_mode, writer=writer)
        return await store.update_config(change, writer=writer)

    @staticmethod
    async def _dispatch_reset(store: GuildRedisStore, field: ConfigFieldName) -> bool:
        if field == ConfigField.VOLUME:
            return await store.reset_volume()
        return await store.reset_config_fields(field)

    def _commit(
        self,
        guild_id: int,
        field: str,
        value: Optional[bool | float | str],
        *,
        persisted: bool,
        live: bool,
        player: Optional[_SettingsPlayer],
    ) -> None:
        """Synchronous, so nothing interleaves: a restore's gate reads the stamp,
        and the live assignment lands with it."""
        self._seq += 1
        self._stamps[(guild_id, field)] = self._seq
        entry = self._entries.get(guild_id)
        base = entry.config if entry is not None else _UNSET_CONFIG
        known = (entry.known if entry is not None else frozenset()) | {field}
        self._store_entry(guild_id, replace(base, **{field: value}), known)
        if persisted:
            self._unpersisted.discard((guild_id, field))
        else:
            self._unpersisted.add((guild_id, field))
        if field == ConfigField.DEBUG_MODE:
            self._cog.debug_settings.apply_choices(
                {guild_id: value if isinstance(value, bool) else None},
                persisted=persisted,
            )
        if live:
            target = player if player is not None else self._cog.mps.get(guild_id)
            if target is not None and field == ConfigField.VOLUME:
                target.volume = value if isinstance(value, float) else DEFAULT_VOLUME
            elif target is not None and field == ConfigField.TIMEZONE:
                zone = value if isinstance(value, str) else None
                target.timezone = GuildConfig(timezone=zone).tzinfo()
        self._prune()

    async def sweep_orphans(
        self, *, is_member: Callable[[int], bool], application_id: int
    ) -> None:
        """Forget the config of every guild this bot is no longer in and whose
        writer_app_id stamp is this application's: a guild removed while the bot
        was offline never raised on_guild_remove, and its key has no TTL. An
        unstamped key, or another application's sharing this Redis, is never
        deleted. Every call is bounded; at most _ORPHAN_SWEEP_MAX per run, and
        it stops at the first unconfirmed DELETE or when the bot closes. The
        caller runs it once per process, when the guild cache is complete.
        See docs/ARCHITECTURE.md#settings-resolution."""
        redis = self._cog.redis
        if redis is None:
            return
        listed = await scan_guild_config_ids(redis, timeout=CONFIG_IO_TIMEOUT_SECS)
        if listed is None:
            log.warning(
                "orphan config sweep: could not list the config keys; nothing "
                "removed, the next start retries"
            )
            return
        candidates = [guild_id for guild_id in listed if not is_member(guild_id)]
        writers = await read_config_writers(
            redis, candidates, batch_timeout=CONFIG_IO_TIMEOUT_SECS
        )
        owned = [g for g in candidates if writers.get(g) == application_id]
        removed = rejoined = 0
        for guild_id in owned[:_ORPHAN_SWEEP_MAX]:
            if self._cog.bot.is_closed():
                break
            # Re-checked at the delete: a guild re-added since the listing keeps it.
            if is_member(guild_id):
                rejoined += 1
                continue
            if not await self.forget(guild_id):
                break
            removed += 1
        log.info(
            f"orphan config sweep: removed {removed}; skipped "
            f"{len(candidates) - len(owned)} (no stamp, another application's, or "
            f"unread); {len(owned) - removed - rejoined} left for the next start"
        )

    async def aclose(self) -> None:
        """Cancel every load in flight: a merge landing after the cog unloads would
        project debug_mode into a DebugSettings whose sampler already stopped."""
        jobs = list(self._loads.values())
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
