"""Tests for src/settings.py — the -settings registry, its value grammar and the
request parser."""

import ast
import dataclasses
import datetime
import importlib.resources
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src import config, guild_state, settings
from src.guild_state import (
    CONFIG_DOMAIN,
    OFF_SECS,
    BotConfigField,
    ConfigField,
)
from src.musicplayer import _fmt_total_duration
from src.settings import (
    SETTINGS,
    TIMEZONE_REDIRECTS,
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
    in_bounds,
    is_bullet_shaped,
    parse_settings_args,
    parse_value,
    warm_timezones,
    wrong_scope_text,
)
from src.util import fmt_duration, fmt_seconds
from tests.helpers import members


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


_TIME_KINDS = (SettingKind.DURATION, SettingKind.SECONDS, SettingKind.SECONDS_OR_OFF)
_ZONE_AREAS = settings._ZONE_AREAS


@pytest.fixture
def fresh_zone_index() -> Iterator[None]:
    """Tests that swap the zone source must not leave its indexes cached."""
    settings._zone_index.cache_clear()
    settings._city_index.cache_clear()
    yield
    settings._zone_index.cache_clear()
    settings._city_index.cache_clear()


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
            names = members(
                ConfigField if spec.scope is SettingScope.SERVER else BotConfigField
            )
            assert spec.field in names, spec.key

    def test_2_every_server_field_has_exactly_one_spec(self) -> None:
        """So no stored field lacks a way to show and reset it."""
        server_fields = [
            spec.field for spec in SETTINGS if spec.scope is SettingScope.SERVER
        ]
        assert sorted(f for f in server_fields if f) == sorted(members(ConfigField))

    def test_2_every_bot_field_has_exactly_one_spec(self) -> None:
        bot_fields = [
            spec.field
            for spec in SETTINGS
            if spec.scope is SettingScope.BOT and spec.field
        ]
        assert sorted(bot_fields) == sorted(members(BotConfigField))

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
            # Every row of both cards prints it.
            assert 0 < len(spec.summary) <= 60, spec.key
            if spec.more is not None:
                assert 0 < len(spec.more) <= 200 and spec.more.endswith("."), spec.key
            assert spec.applies, spec.key
            for why, write_bound in (
                (spec.why_write_minimum, spec.write_minimum),
                (spec.why_write_maximum, spec.write_maximum),
            ):
                if why is not None:
                    assert write_bound is not None, spec.key
                    assert "{bound}" in why and len(why) <= 200, spec.key
            for why, side in (
                (spec.why_minimum, spec.minimum),
                (spec.why_maximum, spec.maximum),
            ):
                if why is not None:
                    assert len(why) <= 200, spec.key
                    assert side is not None, (
                        f"{spec.key}: a reason for a bound it lacks"
                    )

    def test_6_values_round_trip_at_their_bounds_and_default(self) -> None:
        # A write-time minimum follows the bot's value. At its lowest, the env
        # floor, a server setting's whole static range is writable.
        for spec in SETTINGS:
            knob = settings.followed_knob(spec)
            if spec.write_minimum is not None and knob and not config.is_int_knob(knob):
                config.set_override(knob, config.env_floor(knob))
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
            elif (knob := spec.attr or settings.followed_knob(spec)) is not None:
                # An exported variable may deliberately sit outside the chat range.
                if not (os.environ.get(knob) or "").strip():
                    points.append(config.baseline(knob))
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

    def test_a_server_setting_without_a_default_follows_its_bot_setting(
        self,
    ) -> None:
        for spec in SETTINGS:
            knob = settings.followed_knob(spec)
            if spec.scope is SettingScope.BOT or spec.default is not None:
                assert knob is None, spec.key
            elif spec.key != "debug":  # follows DEBUG_MODE, which is not a knob
                assert knob is not None, spec.key
                assert [s.attr for s in SETTINGS if s.key == spec.key and s.attr] == [
                    knob
                ]

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

    def test_each_side_of_idle_timeout_says_why(self) -> None:
        spec = _spec("idle-timeout")
        low, high = parse_value(spec, "4m"), parse_value(spec, "45m")
        assert isinstance(low, Refusal) and isinstance(high, Refusal)
        assert low.text == (
            "**Leave when idle** has to be between **5:00** and **30:00**. A shorter "
            "wait can end the session while a song is still being looked up. `-stop` "
            "in the bot's voice channel makes it leave right away."
        )
        assert high.text == (
            "**Leave when idle** has to be between **5:00** and **30:00**. While the "
            "bot waits it stays in its channel, and a `-play` from another channel "
            "plays there."
        )

    def test_alone_timeout_says_why_only_above_its_cap(self) -> None:
        spec = _spec("alone-timeout")
        low, high = parse_value(spec, "5s"), parse_value(spec, "3m")
        assert isinstance(low, Refusal) and isinstance(high, Refusal)
        assert (
            low.text == "**Leave when alone** has to be between **0:10** and **2:00**."
        )
        assert high.text == (
            "**Leave when alone** has to be between **0:10** and **2:00**. Music keeps "
            "playing while the bot waits alone, and every song counts toward history."
        )

    @pytest.mark.parametrize("value", ["2s", "0.5"])
    def test_np_refresh_below_the_bots_value_names_it(self, value: str) -> None:
        """Below the static floor too: the bot's value is the binding minimum."""
        result = parse_value(_spec("np-refresh"), value)
        assert isinstance(result, Refusal)
        assert result.side == "minimum"
        assert result.text == (
            "**Progress bar refresh** has to be between **3s** and **30s** here: the "
            "bot refreshes no faster than **3s**."
        )

    def test_np_refresh_follows_the_bots_value_at_the_write(self) -> None:
        spec = _spec("np-refresh")
        assert parse_value(spec, "4s") == Parsed(4.0)
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 5.0)
        result = parse_value(spec, "4s")
        assert isinstance(result, Refusal)
        assert "between **5s** and **30s** here" in result.text
        assert parse_value(spec, "5s") == Parsed(5.0)
        high = parse_value(spec, "31s")
        assert isinstance(high, Refusal)
        assert (
            high.text
            == "**Progress bar refresh** has to be between **5s** and **30s**."
        )

    def test_the_card_max_leaves_the_longest_delay_two_ticks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chat's own ranges never meet this bound; an environment delay longer than
        any setting accepts does, and a write must not undercut it."""
        spec = _spec("queue-progress-max")
        assert parse_value(spec, "120s") == Parsed(120.0)
        monkeypatch.setattr(config, "QUEUE_PROGRESS_DELAY_SECS", 100.0)
        config.set_override("QUEUE_PROGRESS_TICK_SECS", 15.0)
        result = parse_value(spec, "125s")
        assert isinstance(result, Refusal)
        assert result.side == "minimum"
        assert result.text == (
            "**Playlist card max** has to be between **130s** and **900s** here: a "
            "card waits up to its longest delay and then needs two ticks, **130s**, "
            "before it can stop."
        )
        assert settings.allowed_text(spec, now=True) == "130s–900s"
        assert settings.allowed_text(spec) == "120s–900s"

    def test_the_card_tick_fits_twice_between_the_longest_delay_and_the_max(
        self,
    ) -> None:
        spec = _spec("queue-progress-tick")
        assert parse_value(spec, "25s") == Parsed(25.0)
        config.set_override("QUEUE_PROGRESS_MAX_SECS", 100.0)
        result = parse_value(spec, "25s")
        assert isinstance(result, Refusal)
        assert result.side == "maximum"
        assert "between **3s** and **20s**" in result.text
        assert parse_value(spec, "20s") == Parsed(20.0)
        assert settings.allowed_text(spec, now=True) == "3s–20s"

    def test_a_card_max_with_no_room_for_a_tick_leaves_it_no_range(self) -> None:
        """An environment max below the longest delay plus two minimum ticks: no
        value can be written, and both the refusal and the card say why."""
        spec = _spec("queue-progress-tick")
        config.set_override("QUEUE_PROGRESS_MAX_SECS", 64.0)
        result = parse_value(spec, "3s")
        assert isinstance(result, Refusal)
        assert result.side == "maximum"
        why = (
            "the card needs two ticks between its longest delay and "
            "queue-progress-max, so a tick of at most **2s**"
        )
        assert (
            result.text == f"**Playlist card tick** can't be changed right now: {why}."
        )
        assert settings.allowed_text(spec, now=True) == f"none right now: {why}"

    def test_a_dynamic_floor_is_rounded_to_a_value_that_can_be_typed(self) -> None:
        spec = _spec("np-refresh")
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 3.333)
        assert settings.allowed_text(spec, now=True) == "3.34s–30s"
        assert parse_value(spec, "3.34") == Parsed(3.34)
        assert isinstance(parse_value(spec, "3.33"), Refusal)

    @pytest.mark.parametrize(
        ("key", "attr", "value", "typed"),
        [
            ("np-refresh", "NOW_PLAYING_UPDATE_INTERVAL_SECS", 3.6, "3.6"),
            # A ceiling: (66.02 - 60) / 2 is 3.009999999999998 as a float.
            ("queue-progress-tick", "QUEUE_PROGRESS_MAX_SECS", 66.02, "3.01"),
            # A floor: 60 + 2 * 30.01 is 120.02000000000001 as a float.
            ("queue-progress-max", "QUEUE_PROGRESS_TICK_SECS", 30.01, "120.02"),
        ],
    )
    def test_a_dynamic_bound_that_is_already_a_step_is_not_moved(
        self, key: str, attr: Any, value: float, typed: str
    ) -> None:
        """A bound's float can lie just past the hundredth it names. Rounded as is,
        the bound moved one step inward and refused the value that fits exactly."""
        spec = _spec(key)
        config.set_override(attr, value)
        assert parse_value(spec, typed) == Parsed(float(typed))

    def test_a_bot_refresh_slower_than_any_server_value_leaves_no_range(self) -> None:
        spec = _spec("np-refresh")
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 45.0)
        why = "the bot refreshes no faster than **45s**"
        result = parse_value(spec, "30")
        assert isinstance(result, Refusal)
        assert (
            result.text
            == f"**Progress bar refresh** can't be changed right now: {why}."
        )
        assert settings.allowed_text(spec, now=True) == f"none right now: {why}"

    def test_a_shape_hint_suggests_a_value_the_write_takes(self) -> None:
        spec = _spec("np-refresh")
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 20.0)
        result = parse_value(spec, "fast")
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.BAD_SHAPE
        example = re.search(r"`(\d+)`", result.text)
        assert example is not None
        assert parse_value(spec, example.group(1)) == Parsed(float(example.group(1)))

    @pytest.mark.parametrize("value", ["0", "0s", "0:00"])
    def test_a_server_slow_notice_of_zero_names_off(self, value: str) -> None:
        result = parse_value(_spec("slow-notice"), value)
        assert isinstance(result, Refusal)
        assert result.text == (
            "**Lookup notice** has to be between **4s** and **60s**, or `off`. Most "
            "lookups finish within 4s, so a shorter delay would post the notice for "
            "ordinary ones."
        )

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


class TestHintsForLikelyMistakes:
    def test_a_bare_number_out_of_a_durations_range_says_it_is_seconds(self) -> None:
        result = parse_value(_spec("idle-timeout"), "10")
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUT_OF_RANGE
        assert result.text.endswith("`10` is read as seconds; `10m` is 10 minutes.")

    @pytest.mark.parametrize("value", ["40m", "0:10"])
    def test_a_value_with_a_unit_gets_no_seconds_hint(self, value: str) -> None:
        result = parse_value(_spec("idle-timeout"), value)
        assert isinstance(result, Refusal)
        assert "read as seconds" not in result.text

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("idle-timeout", "5000"),  # past the maximum
            ("idle-timeout", "3"),  # 3m is under 5:00 too
            ("idle-timeout", "200"),  # 200m is past 30:00 too
            ("idle-timeout", "0"),
            ("alone-timeout", "5"),  # 5m is past 2:00
        ],
    )
    def test_no_seconds_hint_names_minutes_that_would_be_refused(
        self, key: str, value: str
    ) -> None:
        result = parse_value(_spec(key), value)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUT_OF_RANGE
        assert "read as seconds" not in result.text

    def test_a_bare_number_for_seconds_kind_gets_no_hint(self) -> None:
        result = parse_value(_spec("slow-notice"), "90")
        assert isinstance(result, Refusal)
        assert "read as seconds" not in result.text

    @pytest.mark.parametrize(
        ("value", "zone"), [("london", "Europe/London"), ("Tokyo", "Asia/Tokyo")]
    )
    def test_a_bare_city_suggests_its_zone(self, value: str, zone: str) -> None:
        result = parse_value(_spec("timezone"), value)
        assert isinstance(result, Refusal)
        assert result.text.endswith(f"Did you mean `{zone}`?")

    def test_a_city_tzdata_lists_under_two_zones_suggests_both(self) -> None:
        result = parse_value(_spec("timezone"), "istanbul")
        assert isinstance(result, Refusal)
        assert result.text.endswith(
            "Did you mean `Asia/Istanbul` or `Europe/Istanbul`?"
        )


class TestTimezoneIndexIsLazy:
    _LOOKUPS = frozenset(
        {"_known_zones", "_zone_index", "_city_index", "available_timezones"}
    )

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
        assert settings._city_index.cache_info().currsize == 1
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
        assert suggestion.name != "heartbeat"

    def test_a_server_suggestion_is_the_name_the_card_prints(self) -> None:
        assert find("leave-idle", SettingScope.SERVER) == Suggestion("leave-when-idle")

    @pytest.mark.parametrize(
        "env", ["NOW_PLAYING_UPDATE_INTERVAL_SECS", "ping_tick_secs"]
    )
    def test_a_bot_settings_environment_variable_suggests_its_key(
        self, env: str
    ) -> None:
        spec = next(
            s for s in SETTINGS if s.scope is SettingScope.BOT and s.env == env.upper()
        )
        assert find(env, SettingScope.BOT) == Suggestion(spec.key)
        assert find(env, SettingScope.SERVER) != Suggestion(spec.key)


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
            ("volume <80> please", None),
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

    @pytest.mark.parametrize(
        ("arg", "reason"),
        [
            ("idle 45 minutes", RefusalReason.OUT_OF_RANGE),
            ("alone-timeout 5 minutes", RefusalReason.OUT_OF_RANGE),
            ("idle 1 h", RefusalReason.OUT_OF_RANGE),
            ("slow-notice 90 s", RefusalReason.OUT_OF_RANGE),
            ("timezone UTC +5", RefusalReason.FIXED_OFFSET),
        ],
    )
    def test_a_spaced_value_keeps_its_own_refusal(
        self, arg: str, reason: RefusalReason
    ) -> None:
        """Every word belongs to the value, so its range or zone refusal stands:
        its first word alone parsing does not make the rest left over."""
        result = _request(arg)
        assert isinstance(result, Refusal)
        assert result.reason is reason

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
        "arg", ["volume <<50>>", "volume <50", "volume 50>", "volume <>"]
    )
    def test_one_pair_of_brackets_comes_off(self, arg: str) -> None:
        result = _request(arg)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.BAD_SHAPE

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
            ("leave-when-idle 10m", _set("idle-timeout", 600.0)),
            ("idle=15 minutes", _set("idle-timeout", 900.0)),
            ("alone-timeout:1:30", _set("alone-timeout", 90.0)),
            ("leave-when-alone 2m", _set("alone-timeout", 120.0)),
            ("Progress-Bar 5s", _set("np-refresh", 5.0)),
            ("lookup-notice off", _set("slow-notice", OFF_SECS)),
            # A command copied off a card with its placeholder's brackets.
            ("volume <50>", _set("volume", 50)),
            ("volume=<50>", _set("volume", 50)),
            ("leave-when-idle <10:00>", _set("idle-timeout", 600.0)),
            ("leave-when-idle < 10:00 >", _set("idle-timeout", 600.0)),
            ("timezone <America/New York>", _set("timezone", "America/New_York")),
            ("bot heartbeat <5s>", _set("heartbeat", 5.0, SettingScope.BOT)),
            ("slow-notice=10.5s", _set("slow-notice", 10.5)),
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

    @pytest.mark.parametrize("scope", list(SettingScope))
    def test_a_reset_without_a_key_suggests_a_reset_of_its_own_scope(
        self, scope: SettingScope
    ) -> None:
        """Copied as written, the example resets a setting of the scope the
        operator or member asked about, never one of the other."""
        result = _request("bot reset" if scope is SettingScope.BOT else "reset")
        assert isinstance(result, Refusal)
        example = re.search(r"`-settings (.+?)`", result.text)
        assert example is not None
        request = _request(example.group(1))
        assert isinstance(request, SettingsRequest)
        assert (request.action, request.scope) == (SettingsAction.RESET, scope)

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

    @pytest.mark.parametrize(
        ("arg", "scope"),
        [
            ("volume 150", SettingScope.SERVER),
            ("heartbeet", SettingScope.SERVER),
            ("bot heartbeet", SettingScope.BOT),
            ("bot heartbeat 1s", SettingScope.BOT),
            ("bot reset", SettingScope.BOT),
            ("volume " + "8" * 94, None),
        ],
    )
    def test_a_refusal_names_the_scope_it_was_made_in(
        self, arg: str, scope: SettingScope | None
    ) -> None:
        """The command keeps a refusal made after `bot` from a non-operator: its
        text can name a bot setting or its range."""
        result = _request(arg)
        assert isinstance(result, Refusal)
        assert result.scope is scope

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


# ── BotSettings ───────────────────────────────────────────────────────────────

_APP_ID = 123456789012345678
