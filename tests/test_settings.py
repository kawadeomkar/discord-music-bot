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
    GuildConfig,
)
from src.redis_client import BotConfigStore, GuildRedisStore
from src.musicplayer import _fmt_total_duration
from src.settings import (
    SETTINGS,
    TIMEZONE_REDIRECTS,
    BotSettings,
    GuildSettings,
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

    def test_2_every_server_field_has_exactly_one_spec(self) -> None:
        """So no stored field lacks a way to show and reset it."""
        server_fields = [
            spec.field for spec in SETTINGS if spec.scope is SettingScope.SERVER
        ]
        assert sorted(f for f in server_fields if f) == sorted(_members(ConfigField))

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
            # Every row of both cards prints it.
            assert 0 < len(spec.summary) <= 60, spec.key
            if spec.more is not None:
                assert 0 < len(spec.more) <= 200 and spec.more.endswith("."), spec.key
            assert spec.applies, spec.key
            if spec.why_write_minimum is not None:
                assert spec.write_minimum is not None, spec.key
                assert "{bound}" in spec.why_write_minimum, spec.key
                assert len(spec.why_write_minimum) <= 200, spec.key
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
        rows = {var.name: var for var in _CONFIG_ALLOWLIST}
        for spec in SETTINGS:
            if spec.scope is SettingScope.BOT:
                assert spec.env in rows, spec.key
                if spec.attr is not None:
                    assert rows[spec.env].knob == spec.attr, spec.key

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


class TestBotSettingsWrite:
    """A -settings bot change: stored, then applied, one at a time."""

    async def test_a_write_stores_then_applies_and_reports_what_it_replaced(
        self, fake_redis: aioredis.Redis
    ) -> None:
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        spec = _spec("heartbeat")
        first = await bot_settings.write(spec, 5.0)
        assert (first.applied, first.persisted, first.previous) == (True, True, None)
        second = await bot_settings.write(spec, 6.0)
        assert second.previous == 5.0
        assert config.heartbeat_interval_secs() == 6.0
        stored = await BotConfigStore(fake_redis, _APP_ID).read_config()
        assert stored == BotConfig(heartbeat_interval_secs=6.0)

    async def test_a_reset_deletes_the_stored_value_and_the_override(
        self, fake_redis: aioredis.Redis
    ) -> None:
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        spec = _spec("play-inflight-max")
        await bot_settings.write(spec, 8)
        result = await bot_settings.write_reset(spec)
        assert (result.persisted, result.previous) == (True, 8)
        assert config.override("PLAY_INFLIGHT_MAX") is None
        assert await BotConfigStore(fake_redis, _APP_ID).read_config() == BotConfig()

    async def test_an_unconfirmed_write_applies_marked_until_one_lands(
        self, fake_redis: aioredis.Redis
    ) -> None:
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        spec = _spec("heartbeat")
        with patch.object(
            BotConfigStore, "update_config", new=AsyncMock(return_value=False)
        ):
            result = await bot_settings.write(spec, 5.0)
        assert (result.applied, result.persisted) == (True, False)
        assert config.heartbeat_interval_secs() == 5.0
        assert not bot_settings.is_persisted(spec)
        await bot_settings.write(spec, 5.0)
        assert bot_settings.is_persisted(spec)

    async def test_a_later_hydrate_keeps_an_unsaved_write(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """Hydration failed, so a later READY reads again. A write made between the
        two did not reach Redis, and that read must not put the stored 4s back."""
        await _store(fake_redis, BotConfig(heartbeat_interval_secs=4.0))
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        spec = _spec("heartbeat")
        with patch.object(
            BotConfigStore, "read_config", new=AsyncMock(return_value=None)
        ):
            await bot_settings.hydrate()
        with patch.object(
            BotConfigStore, "update_config", new=AsyncMock(return_value=False)
        ):
            await bot_settings.write(spec, 10.0)

        await bot_settings.hydrate()

        assert bot_settings.hydrated is True
        assert config.heartbeat_interval_secs() == 10.0
        assert not bot_settings.is_persisted(spec)

    async def test_a_later_hydrate_keeps_an_unsaved_reset(
        self, fake_redis: aioredis.Redis
    ) -> None:
        await _store(fake_redis, BotConfig(play_inflight_max=1))
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        spec = _spec("play-inflight-max")
        with patch.object(
            BotConfigStore, "read_config", new=AsyncMock(return_value=None)
        ):
            await bot_settings.hydrate()
        with patch.object(
            BotConfigStore, "reset_config_fields", new=AsyncMock(return_value=False)
        ):
            await bot_settings.write_reset(spec)

        await bot_settings.hydrate()

        assert config.override("PLAY_INFLIGHT_MAX") is None
        assert not bot_settings.is_persisted(spec)

    async def test_a_stalled_store_reports_not_saved_within_the_timeout(
        self, fake_redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "CONFIG_IO_TIMEOUT_SECS", 0.05)
        never = asyncio.Event()

        async def _stall(*_args: Any, **_kwargs: Any) -> bool:
            await never.wait()
            return True

        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        with patch.object(BotConfigStore, "update_config", new=_stall):
            async with asyncio.timeout(2):
                result = await bot_settings.write(_spec("heartbeat"), 5.0)
        assert result.persisted is False

    async def test_while_ignored_nothing_is_sent_or_changed(
        self, fake_redis: aioredis.Redis
    ) -> None:
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=True)
        update = AsyncMock(return_value=True)
        reset = AsyncMock(return_value=True)
        with (
            patch.object(BotConfigStore, "update_config", new=update),
            patch.object(BotConfigStore, "reset_config_fields", new=reset),
        ):
            written = await bot_settings.write(_spec("heartbeat"), 5.0)
            cleared = await bot_settings.write_reset(_spec("heartbeat"))
        assert (written.applied, cleared.applied) == (False, False)
        update.assert_not_awaited()
        reset.assert_not_awaited()
        assert config.override("HEARTBEAT_INTERVAL_SECS") is None

    async def test_concurrent_writes_end_on_the_last_in_redis_and_in_memory(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """Without the lock a slow first store call would land after the second's
        and leave Redis holding one value while the process runs the other."""
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        spec = _spec("heartbeat")
        real = BotConfigStore.update_config
        release = asyncio.Event()
        calls = 0

        async def _first_is_slow(store: BotConfigStore, change: BotConfig) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                await release.wait()
            return await real(store, change)

        with patch.object(BotConfigStore, "update_config", new=_first_is_slow):
            first = asyncio.create_task(bot_settings.write(spec, 5.0))
            await asyncio.sleep(0)
            second = asyncio.create_task(bot_settings.write(spec, 6.0))
            await asyncio.sleep(0.01)
            release.set()
            await asyncio.gather(first, second)
        assert config.heartbeat_interval_secs() == 6.0
        stored = await BotConfigStore(fake_redis, _APP_ID).read_config()
        assert stored is not None and stored.heartbeat_interval_secs == 6.0

    async def test_a_hydrate_during_the_store_call_does_not_undo_the_write(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """The store call comes before the override, so a startup read that began
        during it sees the knob stamped after it and leaves the knob alone. Applied
        first, that read would put the old stored value back."""
        await _store(fake_redis, BotConfig(heartbeat_interval_secs=4.0))
        bot_settings = BotSettings(_bot(), redis=fake_redis, ignore_stored=False)
        real = BotConfigStore.update_config
        release = asyncio.Event()

        async def _slow(store: BotConfigStore, change: BotConfig) -> bool:
            await release.wait()
            return await real(store, change)

        with patch.object(BotConfigStore, "update_config", new=_slow):
            writing = asyncio.create_task(bot_settings.write(_spec("heartbeat"), 5.0))
            await asyncio.sleep(0)
            await bot_settings.hydrate()
            release.set()
            await writing
        assert config.heartbeat_interval_secs() == 5.0

    @pytest.mark.parametrize("key", ["debug-default", "volume"])
    async def test_only_a_stored_bot_setting_is_written(self, key: str) -> None:
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=False)
        spec = next(s for s in SETTINGS if s.key == key)
        with pytest.raises(ValueError):
            await bot_settings.write(spec, True if key == "debug-default" else 50)

    async def test_a_value_the_registry_refuses_raises(self) -> None:
        bot_settings = BotSettings(_bot(), redis=None, ignore_stored=False)
        with pytest.raises(ValueError):
            await bot_settings.write(_spec("heartbeat"), 1.0)


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


# ── GuildSettings ─────────────────────────────────────────────────────────────

_GUILD = 424242424242424242


@pytest.fixture
async def guild_cog(fake_redis: aioredis.Redis) -> Any:
    """The cog GuildSettings reads, at call time: a real DebugSettings, a player
    registry, the bot's application id and the Redis handle."""
    cog = MagicMock()
    cog.redis = fake_redis
    cog.mps = {}
    cog.bot.application_id = _APP_ID
    cog.debug_settings = DebugSettings()
    cog.debug_settings._default = False
    yield cog
    await cog.debug_settings.aclose()


def _guild_store(redis: aioredis.Redis, guild_id: int = _GUILD) -> GuildRedisStore:
    return GuildRedisStore(redis, guild_id)


class TestGuildSettingsReads:
    async def test_nothing_is_cached_until_a_read_or_write(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        assert guild_settings.peek(_GUILD) is None
        assert guild_settings.is_complete(_GUILD) is False
        assert guild_settings.is_persisted(_GUILD, "volume") is True

    async def test_a_hydrate_caches_what_it_read_and_projects_debug_mode(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        await _guild_store(fake_redis).update_config(
            GuildConfig(idle_timeout_secs=600.0)
        )
        await _guild_store(fake_redis).set_debug_mode(True)
        guild_settings = GuildSettings(guild_cog)

        assert await guild_settings.hydrate([_GUILD]) == set()

        assert guild_settings.peek(_GUILD) == GuildConfig(
            idle_timeout_secs=600.0, debug_mode=True
        )
        assert guild_settings.is_complete(_GUILD)
        assert guild_cog.debug_settings.enabled(_GUILD) is True

    async def test_guilds_that_never_chose_share_one_empty_entry(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        await guild_settings.hydrate([1, 2, 3])
        entries = [guild_settings._entries[g] for g in (1, 2, 3)]
        assert all(entry is settings._EMPTY for entry in entries)
        assert guild_settings.is_complete(1)

    async def test_a_write_replaces_only_its_guilds_shared_entry(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        await guild_settings.hydrate([1, 2])
        await guild_settings.write(1, GuildConfig(alone_timeout_secs=30.0))
        assert guild_settings._entries[1] is not settings._EMPTY
        assert guild_settings._entries[2] is settings._EMPTY
        assert guild_settings.is_complete(1)

    async def test_a_failed_batch_is_returned_and_caches_nothing(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        with patch.object(fake_redis, "pipeline", side_effect=RuntimeError("down")):
            assert await guild_settings.hydrate([1, 2]) == {1, 2}
        assert guild_settings.peek(1) is None

    async def test_a_hydrate_without_redis_reads_nothing(self, guild_cog: Any) -> None:
        guild_cog.redis = None
        assert await GuildSettings(guild_cog).hydrate([1]) == set()

    async def test_a_stalled_hydrate_gives_up_within_the_timeout(
        self,
        guild_cog: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(settings, "CONFIG_IO_TIMEOUT_SECS", 0.05)
        guild_settings = GuildSettings(guild_cog)

        async def stalled(*_: object, **__: object) -> None:
            await asyncio.Event().wait()

        with patch("redis.asyncio.client.Pipeline.execute", new=stalled):
            async with asyncio.timeout(5):
                assert await guild_settings.hydrate([1]) == {1}
        assert guild_settings.peek(1) is None
        assert "config read failed" in caplog.text

    async def test_seed_needs_an_open_registration(self, guild_cog: Any) -> None:
        guild_settings = GuildSettings(guild_cog)
        with pytest.raises(ValueError):
            guild_settings.seed(_GUILD, GuildConfig(), started=0)
        with guild_settings.reading() as started:
            assert guild_settings.seed(_GUILD, GuildConfig(), started=started) == (
                settings.ALL_CONFIG_FIELDS
            )

    async def test_concurrent_loads_share_one_read(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        """Fifty -settings in one guild cost one HGETALL and one connection."""
        await _guild_store(fake_redis).update_config(GuildConfig(np_refresh_secs=5.0))
        guild_settings = GuildSettings(guild_cog)
        real = GuildRedisStore.read_config
        release = asyncio.Event()
        calls = 0

        async def counted(store: GuildRedisStore) -> Any:
            nonlocal calls
            calls += 1
            await release.wait()
            return await real(store)

        with patch.object(GuildRedisStore, "read_config", new=counted):
            callers = [
                asyncio.create_task(guild_settings.load(_GUILD)) for _ in range(50)
            ]
            await asyncio.sleep(0)
            callers[0].cancel()
            release.set()
            results = await asyncio.gather(*callers[1:])

        assert calls == 1
        assert all(r == GuildConfig(np_refresh_secs=5.0) for r in results)
        assert guild_settings._loads == {}

    async def test_a_failed_or_stalled_load_is_none_and_caches_nothing(
        self, guild_cog: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "CONFIG_IO_TIMEOUT_SECS", 0.05)
        guild_settings = GuildSettings(guild_cog)

        async def stalled(_: object) -> None:
            await asyncio.Event().wait()

        with patch.object(GuildRedisStore, "read_config", new=stalled):
            async with asyncio.timeout(5):
                assert await guild_settings.load(_GUILD) is None
        with patch.object(
            GuildRedisStore, "read_config", new=AsyncMock(return_value=None)
        ):
            assert await guild_settings.load(_GUILD) is None
        assert guild_settings.peek(_GUILD) is None

    async def test_aclose_cancels_a_load_in_flight(self, guild_cog: Any) -> None:
        guild_settings = GuildSettings(guild_cog)

        async def stalled(_: object) -> None:
            await asyncio.Event().wait()

        with patch.object(GuildRedisStore, "read_config", new=stalled):
            caller = asyncio.create_task(guild_settings.load(_GUILD))
            await asyncio.sleep(0)
            await guild_settings.aclose()
            with pytest.raises(asyncio.CancelledError):
                await caller
        assert guild_settings._loads == {}


class TestGuildSettingsAccessors:
    """Synchronous and total: what a hot path runs on, from the cache alone."""

    async def test_idle_timeout_is_the_stored_value_or_the_default(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        assert guild_settings.idle_timeout_secs(_GUILD) == 300.0
        # A partial entry: this field is still unknown, so it is the default.
        await guild_settings.write(_GUILD, GuildConfig(timezone="Asia/Tokyo"))
        assert guild_settings.idle_timeout_secs(_GUILD) == 300.0
        await guild_settings.write(_GUILD, GuildConfig(idle_timeout_secs=1800.0))
        assert guild_settings.idle_timeout_secs(_GUILD) == 1800.0
        await guild_settings.reset(_GUILD, ConfigField.IDLE_TIMEOUT)
        assert guild_settings.idle_timeout_secs(_GUILD) == 300.0

    async def test_alone_timeout_is_the_stored_value_or_the_default(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        assert guild_settings.alone_timeout_secs(_GUILD) == 10.0
        await guild_settings.write(_GUILD, GuildConfig(idle_timeout_secs=900.0))
        assert guild_settings.alone_timeout_secs(_GUILD) == 10.0
        await guild_settings.write(_GUILD, GuildConfig(alone_timeout_secs=120.0))
        assert guild_settings.alone_timeout_secs(_GUILD) == 120.0
        await guild_settings.reset(_GUILD, ConfigField.ALONE_TIMEOUT)
        assert guild_settings.alone_timeout_secs(_GUILD) == 10.0

    async def test_np_refresh_is_never_faster_than_the_bot(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        assert guild_settings.np_refresh_secs(_GUILD) == 3.0
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 5.0)
        assert guild_settings.np_refresh_secs(_GUILD) == 5.0  # read at the call
        await guild_settings.write(_GUILD, GuildConfig(np_refresh_secs=10.0))
        assert guild_settings.np_refresh_secs(_GUILD) == 10.0
        await guild_settings.write(_GUILD, GuildConfig(np_refresh_secs=4.0))
        assert guild_settings.np_refresh_secs(_GUILD) == 5.0
        config.clear_override("NOW_PLAYING_UPDATE_INTERVAL_SECS")
        assert guild_settings.np_refresh_secs(_GUILD) == 4.0

    async def test_slow_notice_is_the_bots_while_unset_and_none_when_off(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        assert guild_settings.slow_notice_secs(_GUILD) == 6.0
        config.set_override("PLAY_SLOW_NOTICE_SECS", 8.0)
        assert guild_settings.slow_notice_secs(_GUILD) == 8.0  # read at the call
        await guild_settings.write(_GUILD, GuildConfig(slow_notice_secs=20.0))
        assert guild_settings.slow_notice_secs(_GUILD) == 20.0
        await guild_settings.write(_GUILD, GuildConfig(slow_notice_secs=OFF_SECS))
        assert guild_settings.slow_notice_secs(_GUILD) is None
        await guild_settings.reset(_GUILD, ConfigField.SLOW_NOTICE)
        assert guild_settings.slow_notice_secs(_GUILD) == 8.0

    async def test_the_playlist_card_delay_is_the_bots_while_unset(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        assert guild_settings.queue_progress_delay_secs(_GUILD) == 2.5
        config.set_override("QUEUE_PROGRESS_DELAY_SECS", 4.0)
        assert guild_settings.queue_progress_delay_secs(_GUILD) == 4.0
        await guild_settings.write(_GUILD, GuildConfig(queue_progress_delay_secs=45.0))
        assert guild_settings.queue_progress_delay_secs(_GUILD) == 45.0
        assert guild_settings.queue_progress_delay_secs(_GUILD + 1) == 4.0
        await guild_settings.reset(_GUILD, ConfigField.QUEUE_PROGRESS_DELAY)
        assert guild_settings.queue_progress_delay_secs(_GUILD) == 4.0

    async def test_off_survives_every_read_back(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        """OFF_SECS is 0.0, which is falsy: a truthiness test anywhere between the
        write and the accessor would read a server's `off` as unset and post the
        notice at the bot's delay."""
        await GuildSettings(guild_cog).write(
            _GUILD, GuildConfig(slow_notice_secs=OFF_SECS)
        )
        stored = await _guild_store(fake_redis).read_config()
        assert stored is not None and stored.slow_notice_secs == OFF_SECS

        hydrated = GuildSettings(guild_cog)
        assert await hydrated.hydrate([_GUILD]) == set()
        restored = GuildSettings(guild_cog)
        with restored.reading() as started:
            restored.seed(_GUILD, stored, started=started)
        for read_back in (hydrated, restored):
            peeked = read_back.peek(_GUILD)
            assert peeked is not None and peeked.slow_notice_secs == OFF_SECS
            assert read_back.slow_notice_secs(_GUILD) is None


class TestGuildSettingsWritePath:
    async def test_each_write_reaches_redis_the_cache_and_the_stamp(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        result = await guild_settings.write(_GUILD, GuildConfig(volume=0.4))
        assert result.applied and result.persisted and result.previous is None
        assert await _guild_store(fake_redis).read_config() == GuildConfig(volume=0.4)
        assert guild_settings.peek(_GUILD) == GuildConfig(volume=0.4)

    @pytest.mark.parametrize(
        "change",
        [
            GuildConfig(volume=0.5),
            GuildConfig(timezone="Europe/London"),
            GuildConfig(debug_mode=True),
            GuildConfig(idle_timeout_secs=600.0),
        ],
        ids=lambda c: next(iter(c.to_redis())),
    )
    async def test_every_set_dispatch_stamps_the_application(
        self, guild_cog: Any, fake_redis: aioredis.Redis, change: GuildConfig
    ) -> None:
        await GuildSettings(guild_cog).write(_GUILD, change)
        stamp = await fake_redis.hget(
            _guild_store(fake_redis).config_key(), "writer_app_id"
        )
        assert stamp == str(_APP_ID).encode()

    async def test_a_write_on_an_uncached_guild_is_partial_until_loaded(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        """The other fields read as their defaults, as before the write, and a
        load completes the entry without losing either value."""
        await _guild_store(fake_redis).update_config(
            GuildConfig(alone_timeout_secs=120.0)
        )
        guild_settings = GuildSettings(guild_cog)

        await guild_settings.write(_GUILD, GuildConfig(idle_timeout_secs=900.0))
        assert guild_settings.peek(_GUILD) == GuildConfig(idle_timeout_secs=900.0)
        assert guild_settings.is_complete(_GUILD) is False

        loaded = await guild_settings.load(_GUILD)
        assert loaded == GuildConfig(idle_timeout_secs=900.0, alone_timeout_secs=120.0)
        assert guild_settings.is_complete(_GUILD)

    async def test_the_commit_assigns_the_live_player(self, guild_cog: Any) -> None:
        player = MagicMock()
        guild_cog.mps[_GUILD] = player
        guild_settings = GuildSettings(guild_cog)

        await guild_settings.write(_GUILD, GuildConfig(volume=0.3))
        await guild_settings.write(_GUILD, GuildConfig(timezone="Europe/London"))
        assert player.volume == 0.3
        assert player.timezone == ZoneInfo("Europe/London")

        await guild_settings.reset(_GUILD, ConfigField.VOLUME)
        await guild_settings.reset(_GUILD, ConfigField.TIMEZONE)
        assert player.volume == guild_state.DEFAULT_VOLUME
        assert player.timezone == ZoneInfo(guild_state.DEFAULT_TIMEZONE)

    async def test_a_volume_reset_clears_both_copies(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        await guild_settings.write(_GUILD, GuildConfig(volume=0.3))
        result = await guild_settings.reset(_GUILD, ConfigField.VOLUME)
        assert result.persisted and result.previous == GuildConfig(volume=0.3)
        store = _guild_store(fake_redis)
        assert not await fake_redis.hexists(store.config_key(), "volume")
        assert not await fake_redis.hexists(store.state_key(), "volume")

    async def test_a_debug_write_projects_into_debug_settings(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        await guild_settings.write(_GUILD, GuildConfig(debug_mode=True))
        assert guild_cog.debug_settings.enabled(_GUILD) is True
        await guild_settings.reset(_GUILD, ConfigField.DEBUG_MODE)
        assert guild_cog.debug_settings.has_override(_GUILD) is False

    @pytest.mark.parametrize(
        "change",
        [
            GuildConfig(),
            GuildConfig(volume=0.5, debug_mode=True),
            GuildConfig(idle_timeout_secs=5.0),  # out of domain: already unset
            GuildConfig(timezone="Mars/Olympus"),
        ],
        ids=["none", "two", "out-of-domain", "bad-zone"],
    )
    async def test_a_change_that_is_not_one_valid_field_raises(
        self, guild_cog: Any, change: GuildConfig
    ) -> None:
        """A registry range wider than CONFIG_DOMAIN fails loudly, not silently."""
        with pytest.raises(ValueError):
            await GuildSettings(guild_cog).write(_GUILD, change)

    async def test_since_belongs_to_seed_alone(self, guild_cog: Any) -> None:
        guild_settings = GuildSettings(guild_cog)
        with pytest.raises(ValueError):
            await guild_settings.write(
                _GUILD, GuildConfig(volume=0.5), mode=settings.WriteMode.SEED
            )
        with guild_settings.reading() as started:
            with pytest.raises(ValueError):
                await guild_settings.write(
                    _GUILD, GuildConfig(volume=0.5), since=started
                )
            with pytest.raises(ValueError):
                await guild_settings.write(
                    _GUILD,
                    GuildConfig(debug_mode=True),
                    mode=settings.WriteMode.SEED,
                    since=started,
                )

    async def test_a_stalled_store_reports_not_saved_within_the_timeout(
        self, guild_cog: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "CONFIG_IO_TIMEOUT_SECS", 0.05)

        async def stalled(*_: object, **__: object) -> bool:
            await asyncio.Event().wait()
            return True

        guild_settings = GuildSettings(guild_cog)
        with patch.object(GuildRedisStore, "set_volume", new=stalled):
            async with asyncio.timeout(5):
                result = await guild_settings.write(_GUILD, GuildConfig(volume=0.5))
        assert result.applied and not result.persisted
        assert not guild_settings.is_persisted(_GUILD, "volume")
        assert guild_settings.unsaved(_GUILD) == frozenset({"volume"})
        assert guild_settings.unsaved(_GUILD + 1) == frozenset()

    async def test_without_redis_a_write_applies_unsaved(self, guild_cog: Any) -> None:
        guild_cog.redis = None
        guild_settings = GuildSettings(guild_cog)
        result = await guild_settings.write(_GUILD, GuildConfig(debug_mode=True))
        assert result.applied and not result.persisted
        assert guild_cog.debug_settings.is_persisted(_GUILD) is False


class TestGuildSettingsStamps:
    """One counter orders every read against every write and forget."""

    async def test_forget_during_a_parked_load_recaches_nothing(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        await _guild_store(fake_redis).set_debug_mode(True)
        guild_settings = GuildSettings(guild_cog)
        reading, release = asyncio.Event(), asyncio.Event()
        real = GuildRedisStore.read_config

        async def parked(store: GuildRedisStore) -> Any:
            config = await real(store)
            reading.set()
            await release.wait()
            return config

        with patch.object(GuildRedisStore, "read_config", new=parked):
            load = asyncio.create_task(guild_settings.load(_GUILD))
            await reading.wait()
            assert await guild_settings.forget(_GUILD) is True
            release.set()
            await load

        assert guild_settings.peek(_GUILD) is None
        assert guild_cog.debug_settings.has_override(_GUILD) is False

    async def test_a_write_queued_behind_forget_is_refused(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        """It cannot recreate a no-TTL key for a departed guild."""
        await _guild_store(fake_redis).set_debug_mode(True)
        guild_settings = GuildSettings(guild_cog)
        clearing, release = asyncio.Event(), asyncio.Event()
        real = GuildRedisStore.clear_config

        async def parked(store: GuildRedisStore) -> bool:
            clearing.set()
            await release.wait()
            return await real(store)

        with patch.object(GuildRedisStore, "clear_config", new=parked):
            forget = asyncio.create_task(guild_settings.forget(_GUILD))
            await clearing.wait()
            write = asyncio.create_task(
                guild_settings.write(_GUILD, GuildConfig(idle_timeout_secs=600.0))
            )
            await asyncio.sleep(0)
            # An unrelated read finishing must not prune the forget away.
            await guild_settings.hydrate([7])
            release.set()
            await forget
            result = await write

        assert result.applied is False
        assert await fake_redis.exists(_guild_store(fake_redis).config_key()) == 0
        assert guild_settings.peek(_GUILD) is None

    async def test_a_debug_write_during_a_hydrate_is_kept_and_the_rest_applies(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        await _guild_store(fake_redis).set_volume(0.3)
        await _guild_store(fake_redis).set_debug_mode(False)
        guild_settings = GuildSettings(guild_cog)
        real = settings.read_guild_configs

        async def read_then_write(*args: Any, **kwargs: Any) -> Any:
            configs = await real(*args, **kwargs)
            await guild_settings.write(_GUILD, GuildConfig(debug_mode=True))
            # The hydrate is still registered, so the stamp is still there.
            assert (_GUILD, "debug_mode") in guild_settings._stamps
            return configs

        with patch("src.settings.read_guild_configs", new=read_then_write):
            await guild_settings.hydrate([_GUILD])

        assert guild_settings.peek(_GUILD) == GuildConfig(volume=0.3, debug_mode=True)
        assert guild_cog.debug_settings.enabled(_GUILD) is True
        assert guild_settings._stamps == {}

    async def test_a_write_during_a_load_is_kept(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        """The same rule for the command's reader: the load resolves with the
        older stored value, and skips the field the write stamped meanwhile."""
        await _guild_store(fake_redis).update_config(GuildConfig(np_refresh_secs=5.0))
        await _guild_store(fake_redis).update_config(
            GuildConfig(alone_timeout_secs=30.0)
        )
        guild_settings = GuildSettings(guild_cog)
        real = GuildRedisStore.read_config

        async def read_then_write(store: GuildRedisStore) -> Any:
            stored = await real(store)
            await guild_settings.write(_GUILD, GuildConfig(np_refresh_secs=10.0))
            return stored

        with patch.object(GuildRedisStore, "read_config", new=read_then_write):
            loaded = await guild_settings.load(_GUILD)

        assert loaded == GuildConfig(np_refresh_secs=10.0, alone_timeout_secs=30.0)

    async def test_a_seed_does_not_project_a_debug_choice_written_during_it(
        self, guild_cog: Any
    ) -> None:
        """seed() projects debug_mode apart from the cache merge, so the footer
        needs its own check: a restore snapshot holding the older choice must not
        turn it back on."""
        guild_settings = GuildSettings(guild_cog)
        with guild_settings.reading() as started:
            await guild_settings.write(_GUILD, GuildConfig(debug_mode=False))
            accepted = guild_settings.seed(
                _GUILD, GuildConfig(debug_mode=True), started=started
            )
        assert "debug_mode" not in accepted
        assert guild_cog.debug_settings.enabled(_GUILD) is False

    async def test_a_seed_does_not_project_over_an_unsaved_debug_choice(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        with patch.object(
            GuildRedisStore, "set_debug_mode", new=AsyncMock(return_value=False)
        ):
            await guild_settings.write(_GUILD, GuildConfig(debug_mode=False))
        with guild_settings.reading() as started:
            guild_settings.seed(_GUILD, GuildConfig(debug_mode=True), started=started)
        assert guild_cog.debug_settings.enabled(_GUILD) is False

    async def test_an_unsaved_write_outlives_every_later_read(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        """Its reply promised it applies until restart; a read of the older stored
        value must not quietly undo that."""
        await _guild_store(fake_redis).set_volume(0.3)
        guild_settings = GuildSettings(guild_cog)
        with patch.object(
            GuildRedisStore, "set_volume", new=AsyncMock(return_value=False)
        ):
            await guild_settings.write(_GUILD, GuildConfig(volume=0.8))

        await guild_settings.hydrate([_GUILD])
        await guild_settings.load(_GUILD)
        with guild_settings.reading() as started:
            accepted = guild_settings.seed(
                _GUILD, GuildConfig(volume=0.3), started=started
            )
        assert "volume" not in accepted
        assert guild_settings.peek(_GUILD) == GuildConfig(volume=0.8)

        await guild_settings.write(_GUILD, GuildConfig(volume=0.6))
        assert guild_settings.is_persisted(_GUILD, "volume")
        await guild_settings.hydrate([_GUILD])
        assert guild_settings.peek(_GUILD) == GuildConfig(volume=0.6)

    async def test_nothing_registered_leaves_no_stamps_behind(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        await guild_settings.write(_GUILD, GuildConfig(volume=0.5))
        await guild_settings.forget(_GUILD)
        assert guild_settings._stamps == {}
        assert guild_settings._forgotten_at == {}
        assert guild_settings._readers == {}

    async def test_a_retry_rereads_only_what_was_omitted_and_stops(
        self,
        guild_cog: Any,
        fake_redis: aioredis.Redis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "_HYDRATE_RETRY_FIRST_SECS", 0.001)
        await _guild_store(fake_redis, 1).set_debug_mode(True)
        guild_settings = GuildSettings(guild_cog)
        read: list[list[int]] = []
        real = settings.read_guild_configs

        async def recording(redis: Any, ids: Any, **kwargs: Any) -> Any:
            read.append(sorted(ids))
            if len(read) == 1:
                return {}
            return await real(redis, ids, **kwargs)

        with patch("src.settings.read_guild_configs", new=recording):
            async with asyncio.timeout(5):
                await guild_settings.retry_hydrate({1, 2})

        assert read == [[1, 2], [1, 2]]
        assert guild_settings.is_complete(1) and guild_settings.is_complete(2)

    async def test_a_retry_drops_guilds_completed_or_forgotten_meanwhile(
        self, guild_cog: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "_HYDRATE_RETRY_FIRST_SECS", 0.01)
        guild_settings = GuildSettings(guild_cog)
        read: list[list[int]] = []

        async def recording(redis: Any, ids: Any, **kwargs: Any) -> Any:
            read.append(sorted(ids))
            return {}

        with patch("src.settings.read_guild_configs", new=recording):
            retry = asyncio.create_task(guild_settings.retry_hydrate({1, 2, 3}))
            await asyncio.sleep(0)
            await guild_settings.forget(2)
            with guild_settings.reading() as started:
                guild_settings.seed(3, GuildConfig(), started=started)
            await asyncio.sleep(0.05)
            retry.cancel()

        assert read and all(ids == [1] for ids in read)

    async def test_writes_to_one_guild_never_overlap_and_leave_no_lock(
        self, guild_cog: Any
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        active = 0
        peak = 0
        real = GuildRedisStore.set_debug_mode

        async def measured(store: GuildRedisStore, enabled: bool, **kw: Any) -> bool:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            try:
                return await real(store, enabled, **kw)
            finally:
                active -= 1

        async def clear(store: GuildRedisStore) -> bool:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return True

        with (
            patch.object(GuildRedisStore, "set_debug_mode", new=measured),
            patch.object(GuildRedisStore, "clear_config", new=clear),
        ):
            await asyncio.gather(
                *(
                    guild_settings.write(_GUILD, GuildConfig(debug_mode=bool(i % 2)))
                    for i in range(10)
                ),
                guild_settings.forget(_GUILD),
                *(
                    guild_settings.write(_GUILD, GuildConfig(debug_mode=True))
                    for _ in range(5)
                ),
            )
        assert peak == 1
        assert guild_settings._locks == {}


class TestGuildLockOutlivesItsFirstHolder:
    async def test_a_caller_arriving_after_the_first_release_waits_its_turn(
        self, guild_cog: Any
    ) -> None:
        """The entry survives while anyone waits on it. Dropped on the first
        release, the next caller would build a second Lock and run beside the
        waiter already holding the first."""
        guild_settings = GuildSettings(guild_cog)
        first_in, release_first = asyncio.Event(), asyncio.Event()
        active = peak = calls = 0

        async def measured(store: GuildRedisStore, enabled: bool, **kw: Any) -> bool:
            nonlocal active, peak, calls
            calls += 1
            active += 1
            peak = max(peak, active)
            try:
                if calls == 1:
                    first_in.set()
                    await release_first.wait()
                else:
                    await asyncio.sleep(0.02)
                return True
            finally:
                active -= 1

        with patch.object(GuildRedisStore, "set_debug_mode", new=measured):
            first = asyncio.create_task(
                guild_settings.write(_GUILD, GuildConfig(debug_mode=True))
            )
            await first_in.wait()
            waiter = asyncio.create_task(
                guild_settings.write(_GUILD, GuildConfig(debug_mode=False))
            )
            await asyncio.sleep(0)
            release_first.set()
            await first
            late = asyncio.create_task(
                guild_settings.write(_GUILD, GuildConfig(debug_mode=True))
            )
            await asyncio.gather(waiter, late)

        assert peak == 1
        assert guild_settings._locks == {}


class TestSeedWrite:
    """Restore's one-release volume migration, on the write path."""

    async def test_a_volume_write_since_the_snapshot_supersedes_the_seed(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        guild_settings = GuildSettings(guild_cog)
        with guild_settings.reading() as started:
            await guild_settings.write(_GUILD, GuildConfig(volume=0.8))
            result = await guild_settings.write(
                _GUILD,
                GuildConfig(volume=0.3),
                mode=settings.WriteMode.SEED,
                since=started,
            )
        assert result.applied is False
        assert await _guild_store(fake_redis).read_config() == GuildConfig(volume=0.8)

    async def test_a_seed_caches_without_touching_the_live_player(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        player = MagicMock()
        player.volume = 0.3
        guild_cog.mps[_GUILD] = player
        guild_settings = GuildSettings(guild_cog)
        with guild_settings.reading() as started:
            result = await guild_settings.write(
                _GUILD,
                GuildConfig(volume=0.3),
                mode=settings.WriteMode.SEED,
                since=started,
            )
        assert result.applied and result.persisted
        assert guild_settings.peek(_GUILD) == GuildConfig(volume=0.3)
        assert player.volume == 0.3

    async def test_a_seed_that_did_not_persist_changes_nothing(
        self, guild_cog: Any
    ) -> None:
        """The legacy field survives, so the next restore retries."""
        guild_settings = GuildSettings(guild_cog)
        with (
            patch.object(
                GuildRedisStore, "migrate_volume", new=AsyncMock(return_value=False)
            ),
            guild_settings.reading() as started,
        ):
            result = await guild_settings.write(
                _GUILD,
                GuildConfig(volume=0.3),
                mode=settings.WriteMode.SEED,
                since=started,
            )
        assert result.applied is False
        assert guild_settings.peek(_GUILD) is None
        assert guild_settings.is_persisted(_GUILD, "volume")


class TestOrphanSweep:
    """A guild removed while the bot was offline leaves a key with no TTL. The
    sweep deletes it only when the guild is gone AND this application stamped it:
    a dev and a prod bot can share one Redis, and guild:{id}:config carries no
    application id of its own."""

    @staticmethod
    async def _key(
        redis: aioredis.Redis, guild_id: int, *, writer: int | None
    ) -> GuildRedisStore:
        store = GuildRedisStore(redis, guild_id)
        await store.set_debug_mode(True, writer=writer)
        return store

    async def test_only_a_departed_guild_stamped_by_this_application_goes(
        self,
        guild_cog: Any,
        fake_redis: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        orphan = await self._key(fake_redis, 1, writer=_APP_ID)
        unstamped = await self._key(fake_redis, 2, writer=None)
        other_app = await self._key(fake_redis, 3, writer=999)
        member = await self._key(fake_redis, 4, writer=_APP_ID)
        guild_cog.bot.is_closed = MagicMock(return_value=False)

        await GuildSettings(guild_cog).sweep_orphans(
            is_member=lambda g: g == 4, application_id=_APP_ID
        )

        assert await fake_redis.exists(orphan.config_key()) == 0
        for kept in (unstamped, other_app, member):
            assert await fake_redis.exists(kept.config_key()) == 1
        assert "removed 1; skipped 2" in caplog.text

    async def test_a_guild_rejoined_since_the_listing_keeps_its_config(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        store = await self._key(fake_redis, 1, writer=_APP_ID)
        guild_cog.bot.is_closed = MagicMock(return_value=False)
        checks = 0

        def rejoins(guild_id: int) -> bool:
            nonlocal checks
            checks += 1
            return checks > 1  # gone when listed, back by the delete

        await GuildSettings(guild_cog).sweep_orphans(
            is_member=rejoins, application_id=_APP_ID
        )
        assert await fake_redis.exists(store.config_key()) == 1

    async def test_the_per_run_cap_holds(
        self,
        guild_cog: Any,
        fake_redis: aioredis.Redis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "_ORPHAN_SWEEP_MAX", 2)
        for guild_id in (1, 2, 3):
            await self._key(fake_redis, guild_id, writer=_APP_ID)
        guild_cog.bot.is_closed = MagicMock(return_value=False)

        await GuildSettings(guild_cog).sweep_orphans(
            is_member=lambda g: False, application_id=_APP_ID
        )

        assert len(await redis_client_scan(fake_redis)) == 1

    async def test_it_stops_at_the_first_unconfirmed_delete(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        for guild_id in (1, 2, 3):
            await self._key(fake_redis, guild_id, writer=_APP_ID)
        guild_cog.bot.is_closed = MagicMock(return_value=False)
        with patch.object(
            GuildRedisStore, "clear_config", new=AsyncMock(return_value=False)
        ) as clear:
            await GuildSettings(guild_cog).sweep_orphans(
                is_member=lambda g: False, application_id=_APP_ID
            )
        assert clear.await_count == 1

    async def test_it_stops_when_the_bot_closes(
        self, guild_cog: Any, fake_redis: aioredis.Redis
    ) -> None:
        for guild_id in (1, 2):
            await self._key(fake_redis, guild_id, writer=_APP_ID)
        guild_cog.bot.is_closed = MagicMock(return_value=True)
        await GuildSettings(guild_cog).sweep_orphans(
            is_member=lambda g: False, application_id=_APP_ID
        )
        assert len(await redis_client_scan(fake_redis)) == 2

    async def test_an_unlistable_redis_removes_nothing(
        self,
        guild_cog: Any,
        fake_redis: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await self._key(fake_redis, 1, writer=_APP_ID)
        with patch.object(fake_redis, "scan", side_effect=RuntimeError("down")):
            await GuildSettings(guild_cog).sweep_orphans(
                is_member=lambda g: False, application_id=_APP_ID
            )
        assert len(await redis_client_scan(fake_redis)) == 1
        assert "could not list the config keys" in caplog.text


async def redis_client_scan(redis: aioredis.Redis) -> list[int]:
    from src.redis_client import scan_guild_config_ids

    return await scan_guild_config_ids(redis, timeout=1.0) or []


class TestGuildConfigHasOneWriter:
    """Every store method that writes guild:{id}:config is called from
    src/settings.py alone. A shortcut writing Redis directly would desynchronize
    the cache from its first call."""

    _WRITERS = frozenset(
        {
            "set_volume",
            "migrate_volume",
            "set_debug_mode",
            "set_timezone",
            "update_config",
            "reset_config_fields",
            "reset_volume",
            "clear_config",
        }
    )

    def test_only_settings_calls_them(self) -> None:
        offenders = []
        for path in sorted(Path("src").rglob("*.py")):
            if path.name == "settings.py" and path.parent.name == "src":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in self._WRITERS
                ):
                    offenders.append(f"{path}:{node.lineno} {node.func.attr}")
        assert not offenders

    def test_the_walk_sees_the_writers_it_exempts(self) -> None:
        tree = ast.parse(Path("src/settings.py").read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert self._WRITERS <= called


def _awaits_reaching_guild_settings(tree: ast.AST) -> list[tuple[str, int]]:
    """(function, line) for each await on guild_settings, directly or through a
    local name bound to an expression that mentions it."""
    found: list[tuple[str, int]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        aliases = {
            target.id
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and "guild_settings" in ast.unparse(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(func):
            if not isinstance(node, ast.Await):
                continue
            text = ast.unparse(node.value)
            if "guild_settings" in text or any(
                text.startswith(f"{alias}.") for alias in aliases
            ):
                found.append((func.name, node.lineno))
    return found


class TestHotPathsNeverAwaitSettings:
    """The pool has no socket_timeout, so a hot path that awaited a settings read
    would hang for as long as Redis stalls. Hot paths read synchronously."""

    @pytest.mark.parametrize(
        "name",
        [
            "peek",
            "is_complete",
            "is_persisted",
            "unsaved",
            "reading",
            "seed",
            "idle_timeout_secs",
            "alone_timeout_secs",
            "np_refresh_secs",
            "slow_notice_secs",
            "queue_progress_delay_secs",
        ],
    )
    def test_the_synchronous_surface_is_plain_functions(self, name: str) -> None:
        import inspect

        assert not inspect.iscoroutinefunction(getattr(GuildSettings, name))

    @pytest.mark.parametrize(
        "path",
        [
            "src/recovery.py",
            "src/commands/play.py",
            "src/play_pipeline.py",
            "src/play_placement.py",
        ],
    )
    def test_no_hot_path_module_awaits_it(self, path: str) -> None:
        assert _awaits_reaching_guild_settings(ast.parse(Path(path).read_text())) == []

    def test_debug_build_inputs_does_not_await_it(self) -> None:
        tree = ast.parse(Path("src/commands/debug.py").read_text())
        assert [
            hit
            for hit in _awaits_reaching_guild_settings(tree)
            if hit[0] == "build_inputs"
        ] == []

    def test_the_player_awaits_it_once_for_the_restore_seed(self) -> None:
        tree = ast.parse(Path("src/musicplayer.py").read_text())
        hits = _awaits_reaching_guild_settings(tree)
        assert [name for name, _ in hits] == ["_restore_state"]


# ── Bot knobs are read at call time ───────────────────────────────────────────

_KNOB_NAMES: frozenset[str] = frozenset(config.FLOAT_KNOBS | config.INT_KNOBS)
_ACCESSORS: frozenset[str] = frozenset(name.lower() for name in _KNOB_NAMES)
_OVERRIDE_WRITERS = ("set_override", "clear_override")
# Every function a pool worker runs, by module. A worker re-imports modules with
# environment values only, so an override set in the parent never reaches it.
_WORKER_ENTRIES: dict[str, tuple[str, ...]] = {
    "src/youtube.py": ("_ytdlp_extract", "warm_worker"),
    "src/analytics_render.py": ("render_dashboard",),
    "src/chart_pool.py": ("_warm_worker",),
    "src/ytdlp_pool.py": ("_warmup_noop", "_worker_init"),
}


@dataclasses.dataclass(frozen=True, slots=True)
class _KnobReads:
    baseline: list[str]  # G1
    definition_time: list[str]  # G2
    writers: list[str]  # G3
    in_workers: list[str]  # G4
    worker_entries: set[str]


def _identifier(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan_knob_reads(
    tree: ast.Module,
    where: str,
    *,
    config_module: bool = False,
    writer_module: bool = False,
    worker_entries: tuple[str, ...] = (),
) -> _KnobReads:
    """One pass per module. Identifiers are matched by name, so an aliased import
    (`import src.config as c; c.PING_TICK_SECS`) is caught without tracking it.
    String constants are not identifiers: the registry's env=/attr= and -debug's
    knob= pass. BotConfigField's constants are wire-field names that two count knobs
    share, so an attribute read off that class is not a knob read."""
    reads = _KnobReads([], [], [], [], set())

    def at(node: ast.AST) -> str:
        return f"{where}:{getattr(node, 'lineno', 0)}"

    for node in ast.walk(tree):
        if not config_module:
            if (
                isinstance(node, (ast.Name, ast.Attribute))
                and isinstance(node.ctx, ast.Load)
                and _identifier(node) in _KNOB_NAMES
                and not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "BotConfigField"
                )
            ):
                reads.baseline.append(at(node))
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "src.config"
                and any(a.name in _KNOB_NAMES or a.name == "*" for a in node.names)
            ):
                reads.baseline.append(at(node))
        if (
            not writer_module
            and isinstance(node, ast.Call)
            and _identifier(node.func) in _OVERRIDE_WRITERS
        ):
            reads.writers.append(at(node))

    def visit(node: ast.AST, definition_time: bool) -> None:
        """G2. What runs when a module is imported or a def/class statement runs:
        module and class bodies, decorators and parameter defaults. Function and
        lambda bodies run later; annotations are lazy (PEP 649), and so is a
        `type` alias's value."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                for decorator in node.decorator_list:
                    visit(decorator, definition_time)
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    visit(default, definition_time)
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            for statement in body:
                visit(statement, False)
            return
        if isinstance(node, ast.ClassDef):
            for part in [*node.decorator_list, *node.bases, *node.keywords]:
                visit(part, definition_time)
            for statement in node.body:
                visit(statement, definition_time)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                visit(node.value, definition_time)
            return
        if isinstance(node, (ast.TypeAlias, ast.arg)):
            return
        if (
            definition_time
            and isinstance(node, ast.Call)
            and _identifier(node.func) in _ACCESSORS
        ):
            reads.definition_time.append(at(node))
        for child in ast.iter_child_nodes(node):
            visit(child, definition_time)

    if not config_module:
        visit(tree, True)

    # G4: each entry, and every same-module function it calls by bare name.
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reads.worker_entries.update(e for e in worker_entries if e in functions)
    pending, seen = list(worker_entries), set[str]()
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if _identifier(node) in _KNOB_NAMES or (
                isinstance(node, ast.Call) and _identifier(node.func) in _ACCESSORS
            ):
                reads.in_workers.append(at(node))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                pending.append(node.func.id)
    return reads


@pytest.fixture(scope="module")
def knob_reads() -> _KnobReads:
    """Every src/ module parsed once, from the package's own path, not the CWD."""
    import src

    root = Path(src.__file__).parent
    total = _KnobReads([], [], [], [], set())
    for path in sorted(root.rglob("*.py")):
        where = path.relative_to(root.parent).as_posix()
        reads = _scan_knob_reads(
            ast.parse(path.read_text()),
            where,
            config_module=where == "src/config.py",
            writer_module=where == "src/settings.py",
            worker_entries=_WORKER_ENTRIES.get(where, ()),
        )
        total.baseline.extend(reads.baseline)
        total.definition_time.extend(reads.definition_time)
        total.writers.extend(reads.writers)
        total.in_workers.extend(reads.in_workers)
        total.worker_entries.update(f"{where}:{e}" for e in reads.worker_entries)
    return total


_EVERY_SHAPE = """\
from src import config
from src.config import PING_TICK_SECS
from src.config import *
import src.config as c
a = c.PING_TICK_SECS
b = config.ping_tick_secs()
def f(x=config.ping_tick_secs()) -> config.ping_tick_secs():
    return config.ping_tick_secs() + config.PING_TICK_SECS
@decorate(config.ping_tick_secs())
def g(): pass
class K(Base, flag=config.ping_tick_secs()):
    y = config.ping_tick_secs()
    z: float = config.ping_tick_secs()
    w: config.ping_tick_secs() = 1.0
lam = lambda q=config.ping_tick_secs(): config.ping_tick_secs()
type T = config.ping_tick_secs()
factory = config.ping_tick_secs
BotConfigField.PLAY_INFLIGHT_MAX
PING_TICK_SECS = 2.0
config.set_override("PING_TICK_SECS", 1.0)
clear_override("PING_TICK_SECS")
s = "PING_TICK_SECS"
def _entry():
    return helper()
def helper():
    return config.ping_tick_secs()
"""


class TestBotKnobsAreReadAtCallTime:
    """Consumers call a knob's accessor when its value applies. A baseline read and
    an accessor called at definition time both type-check and both silently ignore
    a -settings bot override, and the consumer tests would likely pass either,
    only slower. See docs/ARCHITECTURE.md#settings-resolution."""

    def test_the_walker_reports_every_shape_and_nothing_else(self) -> None:
        reads = _scan_knob_reads(
            ast.parse(_EVERY_SHAPE), "x", worker_entries=("_entry",)
        )
        assert reads.baseline == ["x:2", "x:3", "x:5", "x:8"]
        assert sorted(reads.definition_time, key=lambda s: int(s[2:])) == [
            "x:6",
            "x:7",
            "x:9",
            "x:11",
            "x:12",
            "x:13",
            "x:15",
        ]
        assert reads.writers == ["x:20", "x:21"]
        assert reads.in_workers == ["x:26"]

    def test_g1_no_module_reads_a_baseline(self, knob_reads: _KnobReads) -> None:
        assert knob_reads.baseline == []

    def test_g2_no_accessor_is_called_at_definition_time(
        self, knob_reads: _KnobReads
    ) -> None:
        assert knob_reads.definition_time == []

    def test_g3_only_settings_writes_an_override(self, knob_reads: _KnobReads) -> None:
        assert knob_reads.writers == []

    def test_g4_no_pool_worker_reads_a_knob(self, knob_reads: _KnobReads) -> None:
        assert knob_reads.in_workers == []
        # A renamed entry would leave this clause checking nothing.
        assert knob_reads.worker_entries == {
            f"{where}:{entry}"
            for where, entries in _WORKER_ENTRIES.items()
            for entry in entries
        }
