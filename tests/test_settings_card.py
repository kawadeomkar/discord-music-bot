"""Tests for src/settings_card.py — the -settings cards, detail view and replies.
Which of them reaches the channel, and for whom, is tests/commands/test_settings.py."""

import re

import pytest

from src import config
from src import settings_card as card
from src.guild_state import DEFAULT_TIMEZONE, OFF_SECS, GuildConfig
from src.settings import (
    SETTINGS,
    Parsed,
    SettingKind,
    SettingScope,
    SettingsAction,
    SettingSpec,
    SettingsRequest,
    bound,
    find,
    format_value,
    parse_settings_args,
    parse_value,
)

# Discord's caps: a field value is 1024 characters, a message 6000 across every
# embed in it. MusicContext.send prepends a Now Playing block of up to three
# embeds, so a card keeps ~1500 of the 6000 for it.
FIELD_LIMIT = 1024
CARD_BUDGET = 6000 - 1500


def _spec(key: str, scope: SettingScope = SettingScope.SERVER) -> SettingSpec:
    found = find(key, scope)
    assert isinstance(found, SettingSpec)
    return found


def _server_rows(
    stored: GuildConfig | None,
    *,
    debug_default: bool = False,
    unsaved: frozenset[str] = frozenset(),
) -> list[tuple[SettingSpec, card.Shown]]:
    return card.server_rows(stored, debug_default=debug_default, unsaved=unsaved)


class TestServerValues:
    def test_unset_is_the_default(self) -> None:
        assert [shown for _, shown in _server_rows(None)] == [
            card.Shown(100.0, "default"),
            card.Shown(DEFAULT_TIMEZONE, "default"),
            card.Shown(300.0, "default"),
            card.Shown(10.0, "default"),
            card.Shown(3.0, "default"),
            card.Shown(6.0, "default"),
            card.Shown(2.5, "default"),
            card.Shown(False, "default"),
        ]

    def test_set_values_render_in_their_unit(self) -> None:
        stored = GuildConfig(
            volume=0.29,
            timezone="Asia/Tokyo",
            idle_timeout_secs=600.0,
            alone_timeout_secs=120.0,
            np_refresh_secs=10.0,
            slow_notice_secs=OFF_SECS,
            queue_progress_delay_secs=45.0,
            debug_mode=False,
        )
        assert [shown for _, shown in _server_rows(stored)] == [
            card.Shown(29, "set here"),
            card.Shown("Asia/Tokyo", "set here"),
            card.Shown(600.0, "set here"),
            card.Shown(120.0, "set here"),
            card.Shown(10.0, "set here"),
            card.Shown(OFF_SECS, "set here"),
            card.Shown(45.0, "set here"),
            card.Shown(False, "set here"),
        ]

    def test_debug_follows_the_bots_current_default_while_unset(self) -> None:
        rows = dict(_server_rows(None, debug_default=True))
        assert rows[_spec("debug")] == card.Shown(True, "default")

    def test_np_refresh_follows_the_bots_current_value_while_unset(self) -> None:
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 5.0)
        rows = dict(_server_rows(None))
        assert rows[_spec("np-refresh")] == card.Shown(5.0, "default")

    def test_a_value_under_the_bots_runs_as_the_bots_and_names_both(self) -> None:
        rows = dict(_server_rows(GuildConfig(np_refresh_secs=4.0)))
        assert rows[_spec("np-refresh")] == card.Shown(4.0, "set here")
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 5.0)
        rows = dict(_server_rows(GuildConfig(np_refresh_secs=4.0)))
        assert rows[_spec("np-refresh")] == card.Shown(5.0, "bot minimum; set here 4s")

    def test_not_saved_replaces_the_source(self) -> None:
        rows = dict(
            _server_rows(GuildConfig(volume=0.5), unsaved=frozenset({"volume"}))
        )
        assert rows[_spec("volume")] == card.Shown(50, "not saved")


class TestBotValues:
    """The operator's labels, the same four renderings -debug's Config block uses."""

    @pytest.mark.parametrize(
        ("override", "env", "expected"),
        [
            (None, None, card.Shown(3.0, "default")),
            (None, "3", card.Shown(3.0, "env")),
            (5.0, "3", card.Shown(5.0, "bot owner; env 3s")),
            (5.0, None, card.Shown(5.0, "bot owner; default 3s")),
        ],
    )
    def test_a_knob(
        self,
        monkeypatch: pytest.MonkeyPatch,
        override: float | None,
        env: str | None,
        expected: card.Shown,
    ) -> None:
        if env is None:
            monkeypatch.delenv("HEARTBEAT_INTERVAL_SECS", raising=False)
        else:
            monkeypatch.setenv("HEARTBEAT_INTERVAL_SECS", env)
        monkeypatch.setattr(config, "HEARTBEAT_INTERVAL_SECS", 3.0)
        if override is not None:
            config.set_override("HEARTBEAT_INTERVAL_SECS", override)
        shown = card.bot_shown(
            _spec("heartbeat", SettingScope.BOT),
            host_debug_default=False,
            debug_default_override=None,
        )
        assert shown == expected

    def test_an_environment_value_outside_chat_range_is_marked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-server install runs the resolve concurrency at the full pool,
        one above what chat accepts. It is honoured, and the card says chat could
        not have set it."""
        spec = _spec("play-resolve-concurrency", SettingScope.BOT)
        monkeypatch.setenv("PLAY_RESOLVE_CONCURRENCY", "4")
        monkeypatch.setattr(config, "PLAY_RESOLVE_CONCURRENCY", 4)
        monkeypatch.setattr(config, "YTDLP_POOL_WORKERS", 4)
        shown = card.bot_shown(
            spec, host_debug_default=False, debug_default_override=None
        )
        assert shown == card.Shown(4, "env, outside chat range")
        config.set_override("PLAY_RESOLVE_CONCURRENCY", 2)
        shown = card.bot_shown(
            spec, host_debug_default=False, debug_default_override=None
        )
        assert shown == card.Shown(2, "bot owner; env 4")

    def test_an_unsaved_knob_says_so(self) -> None:
        config.set_override("HEARTBEAT_INTERVAL_SECS", 5.0)
        shown = card.bot_shown(
            _spec("heartbeat", SettingScope.BOT),
            host_debug_default=False,
            debug_default_override=None,
            persisted=False,
        )
        assert shown == card.Shown(5.0, "not saved")

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            (None, card.Shown(False, "default")),
            (True, card.Shown(True, "bot owner, until restart; default off")),
        ],
    )
    def test_debug_default(self, override: bool | None, expected: card.Shown) -> None:
        shown = card.bot_shown(
            _spec("debug-default", SettingScope.BOT),
            host_debug_default=False,
            debug_default_override=override,
        )
        assert shown == expected


class TestServerCard:
    def test_the_worst_case_fits(self) -> None:
        """Every key set and a hostile 100-character guild name: the title is
        neutralized and clipped, and the card leaves the Now Playing block room."""
        rows = _server_rows(
            GuildConfig(
                volume=1.0,
                timezone="America/Argentina/ComodRivadavia",
                idle_timeout_secs=1800.0,
                alone_timeout_secs=120.0,
                np_refresh_secs=30.0,
                slow_notice_secs=60.0,
                queue_progress_delay_secs=60.0,
                debug_mode=True,
            ),
            unsaved=frozenset(
                {
                    "volume",
                    "timezone",
                    "idle_timeout_secs",
                    "alone_timeout_secs",
                    "np_refresh_secs",
                    "slow_notice_secs",
                    "queue_progress_delay_secs",
                    "debug_mode",
                }
            ),
        )
        embed = card.server_card(
            guild_name="*`[" * 34, rows=rows, read_failed=True, operator=True
        )
        assert len(embed.title or "") <= 256
        assert "`" not in (embed.title or "") and "[" not in (embed.title or "")
        assert all(len(field.value or "") <= FIELD_LIMIT for field in embed.fields)
        assert len(embed) <= CARD_BUDGET

    def test_each_setting_shows_its_value_summary_and_command(self) -> None:
        embed = card.server_card(
            guild_name="Lo-fi Lounge",
            rows=_server_rows(GuildConfig(volume=0.8)),
            read_failed=False,
            operator=False,
        )
        assert embed.title == "Server settings · Lo-fi Lounge"
        assert [f.name for f in embed.fields] == [
            "Playback",
            "Leaving voice",
            "Messages",
            "Diagnostics",
        ]
        assert embed.fields[0].value == (
            "**Volume** · 80% · set here\n"
            "Playback level.\n"
            "`-settings volume <value>` · 0%–100%\n"
            "\n"
            "**Timezone** · America/Los_Angeles · default\n"
            "Time zone for estimated play times.\n"
            "`-settings timezone <value>` · a city like Europe/London, or UTC"
        )
        assert (embed.description or "").endswith(
            "To change one, run the command under it with a value from its range."
        )
        assert embed.footer.text == (
            "-settings <setting> reset puts one back to its default · "
            "-settings <setting> shows one in full"
        )

    def test_a_range_that_follows_the_bot_is_quoted_as_it_stands(self) -> None:
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 5.0)
        embed = card.server_card(
            guild_name="g", rows=_server_rows(None), read_failed=False, operator=False
        )
        assert "`-settings progress-bar-refresh <value>` · 5s–30s" in str(
            embed.to_dict()
        )

    def test_every_command_on_the_card_sets_its_own_setting(self) -> None:
        """Copied off the card with its brackets, and a value from its range."""
        rows = _server_rows(None)
        embed = card.server_card(
            guild_name="g", rows=rows, read_failed=False, operator=False
        )
        text = "\n".join(f.value or "" for f in embed.fields)
        names = re.findall(r"`-settings (\S+) <value>`", text)
        assert len(names) == len(rows)
        for name, (spec, _) in zip(names, rows, strict=True):
            arg = f"{name} <{_lowest(spec)}>"
            request = parse_settings_args(arg, tail=f"settings {arg}")
            assert isinstance(request, SettingsRequest), (arg, request)
            assert (request.spec, request.action) == (spec, SettingsAction.SET), arg

    def test_a_group_past_the_field_limit_continues_in_another_field(self) -> None:
        rows = _server_rows(None)[:1] * 30
        embed = card.server_card(
            guild_name="g", rows=rows, read_failed=False, operator=False
        )
        names = [f.name for f in embed.fields]
        assert len(names) > 1
        assert names == ["Playback"] + ["Playback (cont.)"] * (len(names) - 1)
        assert all(len(f.value or "") <= FIELD_LIMIT for f in embed.fields)
        assert sum((f.value or "").count("**Volume**") for f in embed.fields) == 30


def _lowest(spec: SettingSpec) -> str:
    """A value the setting accepts right now, as someone would type it."""
    if spec.kind is SettingKind.SWITCH:
        return "on"
    if spec.kind is SettingKind.TIMEZONE:
        return "UTC"
    lowest = bound(spec.minimum) or 0.0
    if spec.write_minimum is not None:
        lowest = max(lowest, spec.write_minimum())
    return format_value(spec, lowest)


class TestBotCard:
    def test_the_worst_case_fits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for spec in SETTINGS:
            if spec.scope is SettingScope.BOT and spec.env:
                monkeypatch.setenv(spec.env, "1")
        rows = card.bot_rows(host_debug_default=True, debug_default_override=False)
        embed = card.bot_card(rows=rows, ignored=True)
        assert all(len(field.value or "") <= FIELD_LIMIT for field in embed.fields)
        assert len(embed) <= CARD_BUDGET
        assert (embed.description or "").startswith(
            "Stored bot settings are ignored (`BOT_SETTINGS_OVERRIDES=ignore`)."
        )

    def test_each_setting_shows_its_value_summary_and_command(self) -> None:
        """The server card's shape, not code blocks: a range column would push a
        code-block row past a phone's width."""
        config.set_override("HEARTBEAT_INTERVAL_SECS", 5.0)
        embed = card.bot_card(
            rows=card.bot_rows(host_debug_default=False, debug_default_override=None),
            ignored=False,
        )
        playback = next(f.value or "" for f in embed.fields if f.name == "Playback")
        assert playback.startswith(
            "**Heartbeat** · 5s · bot owner; default 3s\n"
            "How often a playing server saves its position.\n"
            "`-settings bot heartbeat <value>` · 2s–30s"
        )
        assert "```" not in str(embed.to_dict())
        assert embed.footer.text == (
            "-settings bot <setting> reset returns one to the environment · "
            "-settings bot <setting> shows one in full"
        )

    def test_every_command_on_the_card_sets_its_own_setting(self) -> None:
        rows = card.bot_rows(host_debug_default=False, debug_default_override=None)
        embed = card.bot_card(rows=rows, ignored=False)
        text = "\n".join(f.value or "" for f in embed.fields)
        keys = re.findall(r"`-settings bot (\S+) <value>`", text)
        # Grouped on the card, so compared without order.
        assert sorted(keys) == sorted(spec.key for spec, _ in rows)
        for spec, _ in rows:
            arg = f"bot {spec.key} <{_lowest(spec)}>"
            request = parse_settings_args(arg, tail=f"settings {arg}")
            assert isinstance(request, SettingsRequest), (arg, request)
            assert (request.spec, request.action) == (spec, SettingsAction.SET), arg


class TestDetail:
    def test_a_server_setting_that_follows_the_bot(self) -> None:
        """Its caveat, which the cards have no room for, follows the summary."""
        spec = _spec("debug")
        embed = card.detail(spec, card.Shown(True, "set here"), default=False)
        assert embed.description == (
            "**Debug footer** (`debug`; also `debug-footer`) — Adds trace and "
            "bot-load details to every embed here. Anyone who can read the channel "
            "sees it. Current **on** (set here) · Default off (the bot's default) · "
            "Takes on or off · Applies immediately."
        )

    def test_a_duration_setting(self) -> None:
        spec = _spec("idle-timeout")
        embed = card.detail(spec, card.Shown(600.0, "set here"), default=300.0)
        assert embed.description == (
            "**Leave when idle** (`idle-timeout`; also `idle`, `leave-when-idle`) — "
            "How long to stay in voice with nothing queued. Current **10:00** "
            "(set here) · Default 5:00 · Allowed 5:00–30:00 · Applies the next time "
            "the queue runs empty."
        )

    def test_a_setting_that_follows_the_bots_value_quotes_it_live(self) -> None:
        spec = _spec("np-refresh")
        config.set_override("NOW_PLAYING_UPDATE_INTERVAL_SECS", 5.0)
        embed = card.detail(spec, card.Shown(5.0, "default"), default=5.0)
        assert embed.description == (
            "**Progress bar refresh** (`np-refresh`; also `progress-bar`, "
            "`progress-bar-refresh`) — How often the Now Playing bar moves. Current "
            "**5s** (default) · Default 5s (the bot's default) · Allowed 5s–30s · "
            "Applies from the next tick."
        )

    def test_an_off_setting(self) -> None:
        spec = _spec("slow-notice")
        embed = card.detail(spec, card.Shown(OFF_SECS, "set here"), default=6.0)
        assert embed.description == (
            "**Lookup notice** (`slow-notice`; also `lookup-notice`) — Wait before a "
            "slow song lookup posts a notice. Current **off** (set here) · Default 6s "
            "(the bot's default) · Allowed 4s–60s or off · Applies from the next -play."
        )

    def test_a_bot_setting_names_its_baseline_in_the_source(self) -> None:
        spec = _spec("heartbeat", SettingScope.BOT)
        embed = card.detail(spec, card.Shown(5.0, "bot owner; env 3s"), default=None)
        assert embed.description == (
            f"**Heartbeat** (`heartbeat`) — {spec.summary} Current **5s** (bot "
            "owner; env 3s) · Allowed 2s–30s · Applies from the next tick."
        )

    def test_an_unread_store_is_named_first(self) -> None:
        embed = card.detail(
            _spec("timezone"),
            card.Shown(DEFAULT_TIMEZONE, "default"),
            default=DEFAULT_TIMEZONE,
            read_failed=True,
        )
        assert (embed.description or "").startswith("⚠️ Couldn't read")
        assert embed.color == card.DEGRADED_COLOR


class TestReplies:
    def test_a_value_copied_off_a_reply_parses_back(self) -> None:
        """Every value renders through format_value, which round-trips."""
        for key, value in (
            ("volume", 37),
            ("timezone", "Asia/Tokyo"),
            ("idle-timeout", 630.0),
            ("alone-timeout", 95.0),
            ("np-refresh", 12.5),
            ("slow-notice", OFF_SECS),
            ("slow-notice", 7.5),
            ("debug", True),
        ):
            spec = _spec(key)
            text = (
                card.set_reply(
                    spec, value, previous=None, persisted=True, mention="@a"
                ).description
                or ""
            )
            shown = text.split("**")[3]
            assert parse_value(spec, shown) == Parsed(value)

    def test_a_reset_states_the_value_it_returns_to(self) -> None:
        text = card.reset_reply(
            _spec("volume"), 100.0, persisted=False, mention="@a"
        ).description
        assert text == (
            "**Volume** is back to the default, **100%**. ⚠️ It could not be saved "
            "(Redis is unavailable), so it applies until the bot restarts. Changed "
            "by @a."
        )

    @pytest.mark.parametrize("default", [True, False])
    def test_a_debug_reset_discloses_only_when_it_turns_the_footer_on(
        self, default: bool
    ) -> None:
        text = card.reset_reply(
            _spec("debug"), default, persisted=True, mention="@a"
        ).description
        assert ("anyone who can read the channel" in (text or "")) is default

    def test_a_refusal_carries_the_reason_for_the_side_missed(self) -> None:
        spec = _spec("heartbeat", SettingScope.BOT)
        result = parse_value(spec, "1s")
        assert not isinstance(result, Parsed)
        embed = card.refusal(result.text)
        assert embed.color == card.REFUSAL_COLOR
        assert spec.why_minimum is not None
        assert (embed.description or "").endswith(spec.why_minimum)
