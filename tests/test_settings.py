"""Tests for src/settings.py — the -settings registry, its value grammar and the
request parser."""

import asyncio
import ast
import dataclasses
import datetime
import importlib.resources
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import redis.asyncio as aioredis

from src import config, guild_state, settings
from src.debug import _CONFIG_ALLOWLIST, DebugSettings
from src.guild_state import (
    CONFIG_DOMAIN,
    OFF_SECS,
    BotConfig,
    BotConfigField,
    ConfigField,
)
from src.redis_client import BotConfigStore
from src.musicplayer import _fmt_total_duration
from src.settings import (
    SETTINGS,
    TIMEZONE_REDIRECTS,
    BotSettings,
    Parsed,
    Refusal,
    RefusalReason,
    SettingGroup,
    SettingKind,
    SettingScope,
    SettingSpec,
    SettingsAction,
    SettingsRequest,
    Suggestion,
    bound,
    find,
    format_value,
    from_stored,
    help_sections,
    in_bounds,
    is_bullet_shaped,
    parse_settings_args,
    parse_value,
    warm_timezones,
    wrong_scope_text,
)
from src.util import fmt_duration, fmt_seconds


def _spec(key: str) -> SettingSpec:
    for spec in SETTINGS:
        if spec.key == key:
            return spec
    raise AssertionError(f"no spec {key!r}")


def _local(
    kind: SettingKind,
    *,
    scope: SettingScope = SettingScope.BOT,
    field: guild_state.ConfigFieldName | None = None,
    minimum: float | None = 0.01,
    maximum: float | None = 1_000_000,
) -> SettingSpec:
    """A spec no registry entry stands for yet: the kinds Phase 1 registers none of."""
    return SettingSpec(
        key="test-knob",
        aliases=(),
        scope=scope,
        kind=kind,
        group=SettingGroup.LIMITS,
        label="Test knob",
        summary="A test-only setting.",
        applies="never",
        field=field,
        minimum=minimum,
        maximum=maximum,
    )


def _members(cls: type) -> set[str]:
    return {v for k, v in vars(cls).items() if k.isupper()}


_TIME_KINDS = (SettingKind.DURATION, SettingKind.SECONDS, SettingKind.SECONDS_OR_OFF)
_ZONE_AREAS = settings._ZONE_AREAS


@pytest.fixture
def fresh_zone_index() -> Iterator[None]:
    """Tests that swap the zone source must not leave its index cached."""
    settings._zone_index.cache_clear()
    yield
    settings._zone_index.cache_clear()


# ── Registry invariants ─────────────────────────────────────────────────────────


class TestRegistryInvariants:
    def test_1_names_are_unique_per_scope(self) -> None:
        for scope in SettingScope:
            names = [
                settings._fold(name)
                for spec in SETTINGS
                if spec.scope is scope
                for name in (spec.key, *spec.aliases)
            ]
            assert len(names) == len(set(names)), scope

    def test_2_every_field_is_a_schema_constant(self) -> None:
        """debug-default is the one spec with no field: it is never stored."""
        unstored = [spec.key for spec in SETTINGS if spec.field is None]
        assert unstored == ["debug-default"]
        for spec in SETTINGS:
            if spec.field is None:
                continue
            members = _members(
                ConfigField if spec.scope is SettingScope.SERVER else BotConfigField
            )
            assert spec.field in members, spec.key

    def test_2_every_bot_field_has_exactly_one_spec(self) -> None:
        bot_fields = [
            spec.field
            for spec in SETTINGS
            if spec.scope is SettingScope.BOT and spec.field
        ]
        assert sorted(bot_fields) == sorted(_members(BotConfigField))

    def test_3_every_knob_is_one_bot_spec_above_its_env_floor(self) -> None:
        bot = [spec for spec in SETTINGS if spec.scope is SettingScope.BOT]
        assert [spec.key for spec in bot if spec.attr is None] == ["debug-default"]
        attrs = [spec.attr for spec in bot if spec.attr is not None]
        assert sorted(attrs) == sorted(config.FLOAT_KNOBS | config.INT_KNOBS)
        for spec in bot:
            if spec.attr is None:
                continue
            assert spec.env == spec.attr, spec.key
            minimum = bound(spec.minimum)
            assert minimum is not None, spec.key
            floor = config.env_floor(spec.attr)
            if spec.kind is SettingKind.COUNT:
                assert spec.attr in config.INT_KNOBS, spec.key
                assert minimum >= floor, spec.key
            else:
                # Near the floor a cadence or timeout costs real work per guild; the
                # environment is where an operator makes that choice on purpose.
                assert spec.attr in config.FLOAT_KNOBS, spec.key
                assert minimum > floor, spec.key

    def test_4_copy_is_present_and_short(self) -> None:
        for spec in SETTINGS:
            assert 0 < len(spec.summary) <= 200, spec.key
            assert spec.applies, spec.key
            for why, side in (
                (spec.why_minimum, spec.minimum),
                (spec.why_maximum, spec.maximum),
            ):
                if why is not None:
                    assert len(why) <= 200, spec.key
                    assert side is not None, (
                        f"{spec.key}: a reason for a bound it lacks"
                    )

    def test_5_every_bot_env_has_a_debug_row(self) -> None:
        rows = {var.name for var in _CONFIG_ALLOWLIST}
        for spec in SETTINGS:
            if spec.scope is SettingScope.BOT:
                assert spec.env in rows, spec.key

    def test_6_values_round_trip_at_their_bounds_and_default(self) -> None:
        checked = 0
        for spec in SETTINGS:
            points: list[float | str | bool] = []
            if spec.kind in settings._NUMERIC_KINDS:
                points += [
                    v
                    for v in (bound(spec.minimum), bound(spec.maximum))
                    if v is not None
                ]
            if spec.default is not None:
                points.append(spec.default)
            elif spec.attr is not None:
                # An exported variable may deliberately sit outside the chat range.
                if not (os.environ.get(spec.attr) or "").strip():
                    points.append(config.baseline(spec.attr))
            elif spec.kind is SettingKind.SWITCH:
                points.append(config.debug_mode_default())
            if spec.kind is SettingKind.SECONDS_OR_OFF:
                points.append(OFF_SECS)
            for value in points:
                assert parse_value(spec, format_value(spec, value)) == Parsed(value), (
                    spec.key,
                    value,
                )
                assert in_bounds(spec, value), (spec.key, value)
                checked += 1
        assert checked > len(SETTINGS)

    def test_7_duration_bounds_are_whole_seconds(self) -> None:
        for spec in SETTINGS:
            if spec.kind is SettingKind.DURATION:
                for value in (bound(spec.minimum), bound(spec.maximum), spec.default):
                    assert isinstance(value, (int, float)) and value == int(value), (
                        spec.key
                    )

    def test_8_every_tzdata_name_outside_the_areas_is_handled(self) -> None:
        zones = (
            importlib.resources.files("tzdata").joinpath("zones").read_text().split()
        )
        outside = [
            n for n in zones if n.partition("/")[0] not in _ZONE_AREAS or "/" not in n
        ]
        assert len(outside) > 50, "the zones file changed shape"
        for name in outside:
            assert (
                name in ("UTC", "GMT", "Factory")
                or name in TIMEZONE_REDIRECTS
                or settings._FIXED_OFFSET_RE.fullmatch(name.casefold())
            ), name

    def test_8_redirect_targets_are_accepted_unchanged(self) -> None:
        spec = _spec("timezone")
        for key, redirect in TIMEZONE_REDIRECTS.items():
            for target in redirect.targets:
                assert parse_value(spec, target) == Parsed(target), (key, target)
                if redirect.reason != "synonym":
                    assert target.partition("/")[0] in _ZONE_AREAS, (key, target)
            result = parse_value(spec, key)
            assert isinstance(result, Refusal), key
            assert result.reason is RefusalReason.TIMEZONE_REDIRECT, key

    def test_8_a_same_rules_redirect_keeps_its_offsets(self) -> None:
        """Catches a mistyped target. Abbreviations are exempt: EST is not New York."""
        for key, redirect in TIMEZONE_REDIRECTS.items():
            if redirect.reason == "abbreviation":
                continue
            (target,) = redirect.targets
            for month in (1, 7):
                when = datetime.datetime(2026, month, 1, 12)
                assert ZoneInfo(key).utcoffset(when) == ZoneInfo(target).utcoffset(
                    when
                ), key

    def test_9_server_bounds_are_the_storage_domain(self) -> None:
        for spec in SETTINGS:
            if (
                spec.scope is not SettingScope.SERVER
                or spec.kind not in settings._NUMERIC_KINDS
            ):
                continue
            assert spec.field is not None and guild_state.is_config_field(spec.field)
            domain = CONFIG_DOMAIN[spec.field]
            scale = 100 if spec.kind is SettingKind.PERCENT else 1
            assert (bound(spec.minimum), bound(spec.maximum)) == (
                domain.lo * scale,
                domain.hi * scale,
            )
            assert (spec.kind is SettingKind.SECONDS_OR_OFF) == domain.off, spec.key

    def test_10_wire_names_are_derived(self) -> None:
        for spec in SETTINGS:
            if spec.field is None:
                continue
            if spec.scope is SettingScope.BOT:
                assert spec.env is not None
                assert spec.field == spec.env.lower(), spec.key
            elif spec.key == "debug":
                assert spec.field == "debug_mode"  # older than the registry
            else:
                suffix = "_secs" if spec.kind in _TIME_KINDS else ""
                assert spec.field == spec.key.replace("-", "_") + suffix, spec.key

    def test_11_debug_deadline_outlasts_the_postgres_block(self) -> None:
        from src import debug

        minimum = bound(_spec("debug-deadline").minimum)
        assert minimum is not None
        assert minimum > debug._PG_WINDOW_SECS + debug._PROMETHEUS_TIMEOUT_SECS

    @pytest.mark.parametrize(("workers", "ceiling"), [(1, 1), (2, 1), (4, 3)])
    def test_11_resolve_concurrency_leaves_a_worker_free(
        self, workers: int, ceiling: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "YTDLP_POOL_WORKERS", workers)
        assert bound(_spec("play-resolve-concurrency").maximum) == ceiling

    def test_12_every_label_is_typeable(self) -> None:
        for spec in SETTINGS:
            typed = spec.label.casefold().replace(" ", "-")
            assert typed in (spec.key, *spec.aliases), spec.key


# ── The value grammar ───────────────────────────────────────────────────────────


class TestDurationGrammar:
    @pytest.mark.parametrize(
        ("value", "seconds"),
        [
            ("90", 90),
            ("90s", 90),
            ("90S", 90),
            ("90 sec", 90),
            ("1:30", 90),
            ("0:10", 10),
            ("5:00", 300),
            ("1:00:00", 3600),
            ("10m", 600),
            ("1h30m", 5400),
            ("1h 30m", 5400),
            ("1 hour 30 minutes", 5400),
            ("1h 2m 5s", 3725),
            ("1.5h", 5400),
            ("0.1m", 6),
        ],
    )
    def test_accepted(self, value: str, seconds: float) -> None:
        assert parse_value(_local(SettingKind.DURATION), value) == Parsed(
            float(seconds)
        )

    def test_seconds_take_two_decimal_places(self) -> None:
        assert parse_value(_local(SettingKind.SECONDS), "0.5s") == Parsed(0.5)
        assert parse_value(_local(SettingKind.SECONDS), "10.25") == Parsed(10.25)

    @pytest.mark.parametrize(
        "value",
        [
            "²", "٣", "５", "1e3", "inf", "nan", "-5", "+5", "1_000", ".5", "1,5",
            "0.333", "0x10", "5:0", "5:60", "1:5:00", "5m30", "30m 1h", "1h1h",
            "5 m 30", "10.5s", "90\n", " ", "",
        ],
    )  # fmt: skip
    def test_refused(self, value: str) -> None:
        result = parse_value(_local(SettingKind.DURATION), value)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.BAD_SHAPE

    def test_a_fractional_duration_asks_for_whole_seconds(self) -> None:
        result = parse_value(_local(SettingKind.DURATION), "10.5s")
        assert isinstance(result, Refusal)
        assert "whole seconds" in result.text

    def test_minutes_with_a_bare_number_suggests_both_readings(self) -> None:
        result = parse_value(_local(SettingKind.DURATION), "5m30")
        assert isinstance(result, Refusal)
        assert "`5:30`" in result.text and "`5m30s`" in result.text

    @pytest.mark.parametrize("secs", [10, 59, 60, 61, 300, 3599, 3600, 3725])
    def test_the_bots_own_clock_and_totals_parse_back(self, secs: int) -> None:
        spec = _local(SettingKind.DURATION)
        assert parse_value(spec, fmt_duration(secs)) == Parsed(float(secs))
        assert parse_value(spec, _fmt_total_duration(secs)) == Parsed(float(secs))

    @pytest.mark.parametrize("secs", [0.05, 0.1, 0.25, 0.5, 3, 60, 120, 600])
    def test_fmt_seconds_parses_back(self, secs: float) -> None:
        assert parse_value(_local(SettingKind.SECONDS), fmt_seconds(secs)) == Parsed(
            float(secs)
        )


class TestSecondsOrOff:
    """`off` is the only spelling of OFF_SECS: `0` reads as "at once" as easily as
    "never", and OFF_SECS is falsy while still being a set value."""

    @staticmethod
    def _notice() -> SettingSpec:
        return _local(
            SettingKind.SECONDS_OR_OFF,
            scope=SettingScope.SERVER,
            field=ConfigField.SLOW_NOTICE,
            minimum=4.0,
            maximum=60.0,
        )

    @pytest.mark.parametrize("value", ["off", "OFF", "Off"])
    def test_off_parses_to_off_secs_and_renders_off(self, value: str) -> None:
        spec = self._notice()
        assert parse_value(spec, value) == Parsed(OFF_SECS)
        assert format_value(spec, OFF_SECS) == "off"

    @pytest.mark.parametrize("value", ["0", "0s", "0:00"])
    def test_zero_is_out_of_range_naming_off(self, value: str) -> None:
        result = parse_value(self._notice(), value)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUT_OF_RANGE
        assert result.side == "minimum"
        assert "`off`" in result.text

    def test_in_bounds_admits_off_only_where_the_domain_does(self) -> None:
        assert in_bounds(self._notice(), OFF_SECS)
        np_refresh = _local(
            SettingKind.SECONDS, scope=SettingScope.SERVER, field=ConfigField.NP_REFRESH
        )
        assert not in_bounds(np_refresh, OFF_SECS)
        assert not in_bounds(self._notice(), float("nan"))

    def test_a_bot_bound_refuses_nan_and_inf(self) -> None:
        spec = _spec("heartbeat")
        assert not in_bounds(spec, float("nan"))
        assert not in_bounds(spec, float("inf"))


class TestOutOfRange:
    def test_the_refusal_appends_the_reason_for_the_side_missed(self) -> None:
        spec = _spec("heartbeat")
        low = parse_value(spec, "1s")
        assert isinstance(low, Refusal)
        assert low.side == "minimum"
        assert low.text == (
            "**Heartbeat** has to be between **2s** and **30s**. "
            "Each beat is a Redis write for every server that is playing."
        )
        high = parse_value(spec, "31s")
        assert isinstance(high, Refusal)
        assert high.side == "maximum"
        assert high.text == "**Heartbeat** has to be between **2s** and **30s**."

    def test_a_callable_bound_is_read_at_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = _spec("play-resolve-concurrency")
        monkeypatch.setattr(config, "YTDLP_POOL_WORKERS", 8)
        assert parse_value(spec, "7") == Parsed(7)
        monkeypatch.setattr(config, "YTDLP_POOL_WORKERS", 4)
        assert isinstance(parse_value(spec, "7"), Refusal)

    def test_a_write_bound_narrows_the_range_the_refusal_names(self) -> None:
        spec = dataclasses.replace(
            _local(SettingKind.SECONDS, minimum=0.0, maximum=30.0),
            write_minimum=lambda: 5.0,
        )
        result = parse_value(spec, "2s")
        assert isinstance(result, Refusal)
        assert result.side == "minimum"
        assert "between **5s** and **30s**" in result.text

    def test_a_count_refuses_a_fraction(self) -> None:
        spec = _spec("play-inflight-max")
        assert isinstance(parse_value(spec, "2.5"), Refusal)
        assert not in_bounds(spec, 2.5)


class TestPercent:
    @pytest.mark.parametrize("value", ["80", "80%"])
    def test_accepted(self, value: str) -> None:
        assert parse_value(_spec("volume"), value) == Parsed(80)

    @pytest.mark.parametrize("value", ["0.8", "٨٠"])
    def test_a_fraction_or_a_non_ascii_digit_is_refused(self, value: str) -> None:
        result = parse_value(_spec("volume"), value)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.BAD_SHAPE

    def test_above_100_is_out_of_range(self) -> None:
        result = parse_value(_spec("volume"), "101")
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUT_OF_RANGE

    def test_a_stored_volume_renders_rounded(self) -> None:
        spec = _spec("volume")
        assert format_value(spec, from_stored(spec, 0.29)) == "29%"


class TestSwitch:
    @pytest.mark.parametrize(
        ("value", "on"), [("on", True), ("ENABLE", True), ("no", False)]
    )
    def test_accepted(self, value: str, on: bool) -> None:
        assert parse_value(_spec("debug"), value) == Parsed(on)

    def test_anything_else_is_refused(self) -> None:
        assert isinstance(parse_value(_spec("debug"), "maybe"), Refusal)


class TestTimezone:
    @pytest.mark.parametrize(
        ("value", "canonical"),
        [
            ("Europe/london", "Europe/London"),
            ("america/new york", "America/New_York"),
            ("utc", "UTC"),
            ("Asia/Calcutta", "Asia/Calcutta"),
        ],
    )
    def test_canonicalized(self, value: str, canonical: str) -> None:
        assert parse_value(_spec("timezone"), value) == Parsed(canonical)

    @pytest.mark.parametrize(
        ("value", "names"),
        [
            ("EST", ["`EST`", "America/New_York"]),
            ("est5edt", ["`EST5EDT`", "America/New_York"]),
            ("US/Eastern", ["`US/Eastern`", "America/New_York"]),
            ("Zulu", ["`Zulu`", "`UTC`"]),
            ("IST", ["Asia/Kolkata", "Europe/Dublin", "Asia/Jerusalem"]),
        ],
    )
    def test_redirected_naming_the_target(self, value: str, names: list[str]) -> None:
        result = parse_value(_spec("timezone"), value)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.TIMEZONE_REDIRECT
        for name in names:
            assert name in result.text

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            ("Etc/GMT+5", RefusalReason.FIXED_OFFSET),
            ("UTC+5", RefusalReason.FIXED_OFFSET),
            ("UTC +5:30", RefusalReason.FIXED_OFFSET),
            ("Factory", RefusalReason.BAD_SHAPE),
            ("Mars/Olympus_Mons", RefusalReason.BAD_SHAPE),
        ],
    )
    def test_refused(self, value: str, reason: RefusalReason) -> None:
        result = parse_value(_spec("timezone"), value)
        assert isinstance(result, Refusal)
        assert result.reason is reason

    @pytest.mark.usefixtures("fresh_zone_index")
    def test_a_name_outside_the_areas_is_refused_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host tz directory can add names tzdata lacks; none of them is let in."""
        real = guild_state._known_zones()
        monkeypatch.setattr(
            guild_state, "_known_zones", lambda: real | {"SystemV/AST4"}
        )
        result = parse_value(_spec("timezone"), "SystemV/AST4")
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.BAD_SHAPE


class TestTimezoneIndexIsLazy:
    _LOOKUPS = frozenset({"_known_zones", "_zone_index", "available_timezones"})

    @classmethod
    def _definition_time_calls(cls, tree: ast.Module) -> list[int]:
        """Lines where a lookup runs as the module is imported: module and class
        bodies, decorators and default arguments, but not function bodies."""
        found: list[int] = []

        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for part in (
                    *node.decorator_list,
                    *node.args.defaults,
                    *node.args.kw_defaults,
                ):
                    if part is not None:
                        visit(part)
                return
            if isinstance(node, ast.Lambda):
                return
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if name in cls._LOOKUPS:
                    found.append(node.lineno)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return found

    def test_no_module_walks_the_tz_database_at_import(self) -> None:
        src = Path(settings.__file__).parent
        offenders = {
            path.name: lines
            for path in sorted(src.rglob("*.py"))
            if (lines := self._definition_time_calls(ast.parse(path.read_text())))
        }
        assert offenders == {}

    def test_the_walker_sees_a_module_level_call(self) -> None:
        tree = ast.parse("from x import _zone_index\nINDEX = _zone_index()\n")
        assert self._definition_time_calls(tree) == [2]

    @pytest.mark.usefixtures("fresh_zone_index")
    def test_warm_fills_both_caches(self) -> None:
        guild_state._known_zones.cache_clear()
        warm_timezones()
        assert settings._zone_index.cache_info().currsize == 1
        assert guild_state._known_zones.cache_info().currsize == 1

    @pytest.mark.usefixtures("fresh_zone_index")
    def test_warm_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken() -> frozenset[str]:
            raise OSError("tz database unreadable")

        monkeypatch.setattr(guild_state, "_known_zones", _broken)
        warm_timezones()
        assert settings._zone_index.cache_info().currsize == 0


# ── Keys and requests ───────────────────────────────────────────────────────────


def _request(arg: str, *, tail: str | None = None) -> SettingsRequest | Refusal:
    return parse_settings_args(arg, tail=f"settings {arg}" if tail is None else tail)


def _set(
    key: str, value: settings.SettingValue, scope: SettingScope = SettingScope.SERVER
) -> SettingsRequest:
    return SettingsRequest(
        scope=scope, action=SettingsAction.SET, spec=_spec(key), value=value
    )


class TestFind:
    def test_keys_and_aliases_fold_case_and_underscores(self) -> None:
        assert find("VOL", SettingScope.SERVER) == _spec("volume")
        assert find("Debug_Footer", SettingScope.SERVER) == _spec("debug")

    def test_a_suggestion_comes_from_the_requested_scope_only(self) -> None:
        assert find("volum", SettingScope.SERVER) == Suggestion("volume")
        suggestion = find("heartbeet", SettingScope.SERVER)
        assert isinstance(suggestion, Suggestion)
        assert suggestion.key != "heartbeat"


class TestParseSettingsArgs:
    def test_empty_shows_this_server(self) -> None:
        assert _request("") == SettingsRequest(
            scope=SettingScope.SERVER, action=SettingsAction.SHOW
        )

    def test_bot_alone_shows_the_bot_card(self) -> None:
        assert _request("bot") == SettingsRequest(
            scope=SettingScope.BOT, action=SettingsAction.SHOW
        )

    def test_a_key_alone_is_its_detail(self) -> None:
        assert _request("tz") == SettingsRequest(
            scope=SettingScope.SERVER,
            action=SettingsAction.DETAIL,
            spec=_spec("timezone"),
        )

    @pytest.mark.parametrize(
        ("arg", "tail"),
        [
            ("debug on for the staging bot", None),
            ("volume 80\nthanks", None),
            ("volume 80 x", None),
            ("volume 150 x", None),
            ("volume reset now", None),
            ("reset volume now", None),
            ("timezone Europe/London please", None),
            ("volume " + "8" * 94, None),
            ("volume 50", "settings\nvolume 50"),
        ],
    )
    def test_too_much(self, arg: str, tail: str | None) -> None:
        result = _request(arg, tail=tail)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.TOO_MUCH

    def test_extra_spaces_are_one_separator(self) -> None:
        assert _request("bot play-resolve-wait  1m   30s") == _set(
            "play-resolve-wait", 90.0, SettingScope.BOT
        )

    @pytest.mark.parametrize("arg", ["volume reset", "volume —reset", "volume default"])
    def test_reset(self, arg: str) -> None:
        assert _request(arg) == SettingsRequest(
            scope=SettingScope.SERVER, action=SettingsAction.RESET, spec=_spec("volume")
        )

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("set volume 50", _set("volume", 50)),
            ("bot set heartbeat 5s", _set("heartbeat", 5.0, SettingScope.BOT)),
            ("set bot heartbeat 5s", _set("heartbeat", 5.0, SettingScope.BOT)),
            ("volume=50", _set("volume", 50)),
            ("volume:50", _set("volume", 50)),
            ("volume = 50", _set("volume", 50)),
            ("volume: 50", _set("volume", 50)),
            (
                "bot play-resolve-wait:1:30",
                _set("play-resolve-wait", 90.0, SettingScope.BOT),
            ),
            ("timezone: America/New York", _set("timezone", "America/New_York")),
            ("--volume 50", _set("volume", 50)),
            ("—volume 50", _set("volume", 50)),
            ("--bot heartbeat 5s", _set("heartbeat", 5.0, SettingScope.BOT)),
            ("Debug-Footer on", _set("debug", True)),
        ],
    )
    def test_near_misses_normalize(self, arg: str, expected: SettingsRequest) -> None:
        assert _request(arg) == expected

    @pytest.mark.parametrize(
        ("arg", "scope", "key"),
        [
            ("reset volume", SettingScope.SERVER, "volume"),
            ("bot reset heartbeat", SettingScope.BOT, "heartbeat"),
            ("--reset volume", SettingScope.SERVER, "volume"),
        ],
    )
    def test_reset_before_the_key(
        self, arg: str, scope: SettingScope, key: str
    ) -> None:
        assert _request(arg) == SettingsRequest(
            scope=scope, action=SettingsAction.RESET, spec=_spec(key)
        )

    @pytest.mark.parametrize(
        ("arg", "reason"),
        [
            ("reset", RefusalReason.RESET_WITHOUT_KEY),
            ("bot reset", RefusalReason.RESET_WITHOUT_KEY),
            ("set", RefusalReason.UNKNOWN_KEY),
            ("- volume 50", RefusalReason.UNKNOWN_KEY),
            ("---volume 50", RefusalReason.UNKNOWN_KEY),
        ],
    )
    def test_refused_before_a_key(self, arg: str, reason: RefusalReason) -> None:
        result = _request(arg)
        assert isinstance(result, Refusal)
        assert result.reason is reason

    @pytest.mark.parametrize(
        ("arg", "key"), [("heartbeat 5", "heartbeat"), ("bot volume 50", "volume")]
    )
    def test_wrong_scope_names_the_canonical_key(self, arg: str, key: str) -> None:
        result = _request(arg)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.WRONG_SCOPE
        assert result.spec == _spec(key)
        assert f"`{key}`" in result.text

    def test_a_server_typo_of_a_bot_key_suggests_no_bot_key(self) -> None:
        result = _request("heartbeet 5")
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.UNKNOWN_KEY
        assert "heartbeat" not in result.text

    def test_the_operator_variant_names_the_bot_form(self) -> None:
        text = wrong_scope_text(_spec("heartbeat"), operator=True)
        assert "`-settings bot heartbeat`" in text

    def test_a_value_the_kind_refuses_keeps_its_own_refusal(self) -> None:
        result = _request("volume loud")
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.BAD_SHAPE


class TestBulletShape:
    @pytest.mark.parametrize(
        ("tail", "bullet"),
        [
            (" settings x", True),
            ("\tsettings", True),
            ("settings x", False),
            ("settings", False),
        ],
    )
    def test_whitespace_after_the_prefix(self, tail: str, bullet: bool) -> None:
        assert is_bullet_shaped(tail) is bullet


class TestRefusalsNeverQuoteTheInput:
    """A refusal renders in the channel, so input echoed into it could ping a role,
    break a code fence, or post a link under the bot's name."""

    HOSTILE = (
        "<@123456789012345678>",
        "@everyone",
        "``` break ```",
        "[link](https://evil.example)",
        "HeArTbEaT",
    )

    @staticmethod
    def _refusals(hostile: str) -> list[Refusal]:
        candidates = [
            hostile,
            f"{hostile} 5",
            f"bot {hostile}",
            f"timezone {hostile}",
            f"timezone Europe/London {hostile}",
            f"debug on {hostile}",
            f"volume {hostile}",
            f"bot heartbeat {hostile}",
            f"reset {hostile}",
            f"volume 50\n{hostile}",
            hostile * 400,
        ]
        results = [_request(arg) for arg in candidates]
        return [r for r in results if isinstance(r, Refusal)]

    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_no_refusal_text_contains_the_input(self, hostile: str) -> None:
        refusals = self._refusals(hostile)
        assert len(refusals) >= 8
        reasons = {r.reason for r in refusals}
        assert RefusalReason.TOO_MUCH in reasons
        for refusal in refusals:
            assert hostile not in refusal.text, refusal.reason

    def test_wrong_scope_is_among_them(self) -> None:
        assert any(
            r.reason is RefusalReason.WRONG_SCOPE for r in self._refusals("HeArTbEaT")
        )


class TestHelpSections:
    def test_lists_every_server_key_and_alias_and_no_bot_key(self) -> None:
        ((name, entries),) = help_sections()
        assert name == "SETTINGS"
        text = "\n".join(line for entry in entries for line in entry)
        for spec in SETTINGS:
            names = (spec.key, *spec.aliases)
            if spec.scope is SettingScope.SERVER:
                assert all(n in text for n in names), spec.key
            else:
                assert spec.key not in text.split(), spec.key

    def test_lines_fit_the_help_code_block(self) -> None:
        ((_, entries),) = help_sections()
        assert all(len(line) <= 48 for entry in entries for line in entry)


# ── BotSettings ───────────────────────────────────────────────────────────────

_APP_ID = 123456789012345678


def _bot(*, application_id: int | None = _APP_ID, cog: Any = None) -> MagicMock:
    bot = MagicMock()
    bot.application_id = application_id
    bot.get_cog = MagicMock(return_value=cog)
    return bot


def _cog_with_debug_settings() -> MagicMock:
    cog = MagicMock()
    cog.debug_settings = DebugSettings()
    cog.debug_settings._default = False
    return cog


async def _store(redis: aioredis.Redis, stored: BotConfig) -> None:
    assert await BotConfigStore(redis, _APP_ID).update_config(stored)


class TestBotSettingsHydrate:
    """The stored overrides reach config's accessors, and only in-bounds ones."""

    async def test_in_bounds_overrides_are_applied_and_listed_once(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        await _store(
            fake_redis, BotConfig(heartbeat_interval_secs=5.0, play_inflight_max=4)
        )
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)

        await bot_settings.hydrate()

        assert config.heartbeat_interval_secs() == 5.0
        assert config.play_inflight_max() == 4
        assert bot_settings.hydrated is True
        assert (
            f"bot settings applied from bot:{_APP_ID}:config: "
            "heartbeat=5s, play-inflight-max=4"
        ) in caplog.text

    async def test_a_value_outside_the_current_bounds_is_skipped_with_a_warning(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The key outlives builds: a stored value from a build with other bounds
        must neither apply nor abort."""
        await _store(fake_redis, BotConfig(heartbeat_interval_secs=0.5))
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)

        await bot_settings.hydrate()

        assert config.override("HEARTBEAT_INTERVAL_SECS") is None
        assert "heartbeat=0.5" in caplog.text and "ignored" in caplog.text
        assert bot_settings.hydrated is True

    async def test_an_override_shadowing_a_set_variable_names_every_way_out(
        self,
        fake_redis: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PLAY_INFLIGHT_MAX", "16")
        await _store(fake_redis, BotConfig(play_inflight_max=1))

        await BotSettings(_bot(), redis=fake_redis, ignore_stored=False).hydrate()

        warning = next(
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "overrides" in r.getMessage()
        )
        assert "PLAY_INFLIGHT_MAX=16" in warning
        assert "-settings bot play-inflight-max reset" in warning
        assert f"just bot-settings reset {_APP_ID}" in warning
        assert "BOT_SETTINGS_OVERRIDES=ignore" in warning

    async def test_an_override_of_an_unset_variable_draws_no_warning(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        await _store(fake_redis, BotConfig(play_inflight_max=1))
        await BotSettings(_bot(), redis=fake_redis, ignore_stored=False).hydrate()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    async def test_ignore_makes_no_read_and_warns_once(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        await _store(fake_redis, BotConfig(play_inflight_max=1))
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=True)
        with patch.object(
            BotConfigStore, "read_config", new=AsyncMock(side_effect=AssertionError)
        ) as read:
            await bot_settings.hydrate()
            await bot_settings.hydrate()

        read.assert_not_called()
        assert config.override("PLAY_INFLIGHT_MAX") is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "BOT_SETTINGS_OVERRIDES=ignore" in warnings[0].getMessage()
        assert f"bot:{_APP_ID}:config" in warnings[0].getMessage()

    async def test_no_application_id_makes_no_read(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """Unit tests never log in; the key would otherwise be bot:None:config."""
        bot_settings = BotSettings(
            _bot(application_id=None), redis=fake_redis, ignore_stored=False
        )
        with patch.object(
            BotConfigStore, "read_config", new=AsyncMock(side_effect=AssertionError)
        ) as read:
            await bot_settings.hydrate()
        read.assert_not_called()
        assert bot_settings.hydrated is False

    async def test_a_failed_read_runs_on_environment_values_and_can_retry(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        await _store(fake_redis, BotConfig(heartbeat_interval_secs=5.0))
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        with patch.object(fake_redis, "hgetall", side_effect=RuntimeError("down")):
            await bot_settings.hydrate()

        assert config.override("HEARTBEAT_INTERVAL_SECS") is None
        assert bot_settings.hydrated is False
        assert "running on environment values" in caplog.text

        await bot_settings.hydrate()
        assert config.heartbeat_interval_secs() == 5.0

    async def test_a_stalled_read_gives_up_within_the_timeout(
        self, fake_redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "CONFIG_IO_TIMEOUT_SECS", 0.05)

        async def stalled(*_: object) -> None:
            await asyncio.Event().wait()

        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        with patch.object(BotConfigStore, "read_config", new=stalled):
            async with asyncio.timeout(5):
                await bot_settings.hydrate()
        assert bot_settings.hydrated is False

    async def test_a_knob_changed_during_the_read_keeps_the_new_value(
        self, fake_redis: aioredis.Redis
    ) -> None:
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        stored = BotConfig(heartbeat_interval_secs=5.0, ping_tick_secs=2.0)

        async def read_then_operator_applies(_: object) -> BotConfig:
            bot_settings.apply(_spec("heartbeat"), 7.0)
            return stored

        with patch.object(
            BotConfigStore, "read_config", new=read_then_operator_applies
        ):
            await bot_settings.hydrate()

        assert config.heartbeat_interval_secs() == 7.0
        assert config.ping_tick_secs() == 2.0


class TestBotSettingsApply:
    def test_apply_and_reset_move_the_accessor(self) -> None:
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=False)
        assert bot_settings.apply(_spec("heartbeat"), 5.0) is True
        assert config.heartbeat_interval_secs() == 5.0
        assert bot_settings.reset(_spec("heartbeat")) is True
        assert config.heartbeat_interval_secs() == config.HEARTBEAT_INTERVAL_SECS

    def test_a_count_is_applied_as_an_int(self) -> None:
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=False)
        bot_settings.apply(_spec("play-inflight-max"), 4)
        assert type(config.play_inflight_max()) is int

    @pytest.mark.parametrize(
        ("key", "value"),
        [("heartbeat", 1.0), ("heartbeat", float("nan")), ("play-inflight-max", 2.5)],
    )
    def test_a_value_the_registry_refuses_raises(self, key: str, value: float) -> None:
        """Input is refused before it gets here; reaching apply with one is a bug."""
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=False)
        with pytest.raises(ValueError):
            bot_settings.apply(_spec(key), value)

    def test_a_server_setting_raises(self) -> None:
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=False)
        with pytest.raises(ValueError):
            bot_settings.apply(_spec("volume"), 50)

    def test_while_ignored_a_stored_setting_is_refused_and_unchanged(self) -> None:
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=True)
        config.set_override("HEARTBEAT_INTERVAL_SECS", 9.0)
        assert bot_settings.apply(_spec("heartbeat"), 5.0) is False
        assert bot_settings.reset(_spec("heartbeat")) is False
        assert config.heartbeat_interval_secs() == 9.0


class TestDebugDefaultIsSessionOnly:
    """debug-default decides what every server that never chose publishes, so it
    lasts until a restart and is never stored."""

    async def test_apply_and_reset_reach_the_cogs_debug_settings(self) -> None:
        cog = _cog_with_debug_settings()
        bot_settings = BotSettings(_bot(cog=cog), redis=None, ignore_stored=False)
        try:
            assert bot_settings.apply(_spec("debug-default"), True) is True
            assert cog.debug_settings.default is True
            assert cog.debug_settings.enabled(42) is True
            assert bot_settings.reset(_spec("debug-default")) is True
            assert cog.debug_settings.default is False
        finally:
            await cog.debug_settings.aclose()

    async def test_it_is_settable_while_stored_settings_are_ignored(self) -> None:
        cog = _cog_with_debug_settings()
        bot_settings = BotSettings(_bot(cog=cog), redis=None, ignore_stored=True)
        try:
            assert bot_settings.apply(_spec("debug-default"), True) is True
            assert cog.debug_settings.default is True
        finally:
            await cog.debug_settings.aclose()

    async def test_no_path_touches_the_store(self, fake_redis: aioredis.Redis) -> None:
        cog = _cog_with_debug_settings()
        bot_settings = BotSettings(_bot(cog=cog), redis=fake_redis, ignore_stored=False)
        boom = AsyncMock(side_effect=AssertionError("debug-default reached the store"))
        try:
            with (
                patch.object(BotConfigStore, "update_config", new=boom),
                patch.object(BotConfigStore, "reset_config_fields", new=boom),
            ):
                bot_settings.apply(_spec("debug-default"), True)
                bot_settings.reset(_spec("debug-default"))
            assert (
                await BotConfigStore(fake_redis, _APP_ID).read_config() == BotConfig()
            )
        finally:
            await cog.debug_settings.aclose()

    async def test_a_reloaded_cog_gets_the_session_value_and_a_restart_does_not(
        self,
    ) -> None:
        cog = _cog_with_debug_settings()
        bot_settings = BotSettings(_bot(cog=cog), redis=None, ignore_stored=False)
        reloaded = DebugSettings()
        reloaded._default = False
        restarted = DebugSettings()
        restarted._default = False
        try:
            bot_settings.apply(_spec("debug-default"), True)
            bot_settings.reapply_debug_default(reloaded)
            assert reloaded.default is True

            BotSettings(_bot(), redis=None, ignore_stored=False).reapply_debug_default(
                restarted
            )
            assert restarted.default is False
        finally:
            for debug_settings in (cog.debug_settings, reloaded, restarted):
                await debug_settings.aclose()

    async def test_a_value_set_before_the_cog_loads_is_held(self) -> None:
        bot_settings = BotSettings(_bot(cog=None), redis=None, ignore_stored=False)
        assert bot_settings.apply(_spec("debug-default"), True) is True
        late = DebugSettings()
        late._default = False
        try:
            bot_settings.reapply_debug_default(late)
            assert late.default is True
        finally:
            await late.aclose()
