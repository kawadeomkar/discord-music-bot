"""Tests for `-settings` (src/commands/settings.py), driven through the cog's
wrapper against a real GuildSettings on fake Redis. The text each builder renders
is pinned in tests/test_settings_card.py; these pin which reply reaches the
channel, who may get it, and what it changed."""

import asyncio
import logging
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import discord
import pytest
from redis.asyncio import Redis

from src import config
from src import settings_card as card
from src.guild_state import DEFAULT_VOLUME, GuildConfig
from src.musicbot import MusicBot
from src.redis_client import GuildRedisStore
from tests.helpers import command_callback

GUILD = 111111111111111111
MENTION = "<@222222222222222222>"


@pytest.fixture
def settings_ctx(mock_ctx: MagicMock) -> MagicMock:
    """Least privilege: no Manage Server, not in voice, not the operator. mock_ctx
    defaults to the owner and mock_author to Manage Server in a voice channel, so
    a denial written on them would pass without the check it names."""
    mock_ctx.bot.is_owner = AsyncMock(return_value=False)
    mock_ctx.author.guild_permissions.manage_guild = False
    mock_ctx.author.voice = None
    mock_ctx.voice_client = None
    mock_ctx.prefix = "-"
    mock_ctx.message.content = "-settings"
    mock_ctx.guild.name = "Lo-fi Lounge"
    mock_ctx.command.name = "settings"
    return mock_ctx


@pytest.fixture
def cog(music_bot_with_redis: MusicBot) -> MusicBot:
    return music_bot_with_redis


@pytest.fixture
def span() -> Iterator[MagicMock]:
    with patch("src.commands.settings.trace.get_current_span") as current:
        yield current.return_value


async def _invoke(
    cog: MusicBot, ctx: MagicMock, arg: str = "", *, tail: str | None = None
) -> None:
    ctx.message.content = "-" + (tail if tail is not None else f"settings {arg}")
    await command_callback(MusicBot.settings)(cog, ctx, arg=arg)


def _embed(ctx: MagicMock) -> discord.Embed:
    return cast(discord.Embed, ctx.send.await_args.kwargs["embed"])


def _text(ctx: MagicMock) -> str:
    return _embed(ctx).description or ""


def _as_operator(ctx: MagicMock) -> MagicMock:
    ctx.bot.is_owner = AsyncMock(return_value=True)
    return ctx


def _with_manage_server(ctx: MagicMock) -> MagicMock:
    ctx.author.guild_permissions.manage_guild = True
    return ctx


def _denied_by_the_operator_check(ctx: MagicMock) -> None:
    ctx.bot.is_owner.assert_awaited_once()
    assert ctx.bot.is_owner.return_value is False


def _in_voice(ctx: MagicMock, *, bot_channel: bool | None) -> None:
    """The author in a voice channel; the bot in the same one (True), another
    (False), or not connected (None)."""
    channel = MagicMock(spec=discord.VoiceChannel)
    ctx.author.voice = MagicMock()
    ctx.author.voice.channel = channel
    if bot_channel is None:
        ctx.voice_client = None
        return
    ctx.voice_client = MagicMock(spec=discord.VoiceClient)
    ctx.voice_client.channel = (
        channel if bot_channel else MagicMock(spec=discord.VoiceChannel)
    )


async def _stored(redis: Redis) -> GuildConfig | None:
    return await GuildRedisStore(redis, GUILD).read_config()


class TestWhoMayChangeAServerSetting:
    async def test_a_member_is_refused(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        await _invoke(cog, settings_ctx, "timezone Europe/London")
        assert _text(settings_ctx) == card.NO_PERMISSION
        _denied_by_the_operator_check(settings_ctx)
        assert await _stored(fake_redis_bot) == GuildConfig()

    async def test_manage_server_changes_it_without_asking_the_operator(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        await _invoke(cog, _with_manage_server(settings_ctx), "timezone Europe/London")
        assert await _stored(fake_redis_bot) == GuildConfig(timezone="Europe/London")
        settings_ctx.bot.is_owner.assert_not_awaited()

    async def test_the_operator_changes_it(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        await _invoke(cog, _as_operator(settings_ctx), "debug on")
        assert await _stored(fake_redis_bot) == GuildConfig(debug_mode=True)
        assert cog.debug_settings.enabled(GUILD) is True

    async def test_a_failed_operator_check_denies_and_is_not_repeated(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        settings_ctx.bot.is_owner = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=503), "boom")
        )
        await _invoke(cog, settings_ctx, "debug on")
        assert _text(settings_ctx) == card.NO_PERMISSION
        await _invoke(cog, settings_ctx, "debug on")
        assert _text(settings_ctx) == card.NO_PERMISSION
        settings_ctx.bot.is_owner.assert_awaited_once()
        assert cog.debug_settings.has_override(GUILD) is False


class TestVolumeTakesTheVoiceGateToo:
    """-volume needs only the voice gate, so -settings volume accepts it: an admin's
    level is otherwise one -volume away from being overwritten anyway."""

    async def test_a_listener_in_the_bots_channel_sets_and_resets_it(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        _in_voice(settings_ctx, bot_channel=True)
        await _invoke(cog, settings_ctx, "volume 50")
        assert await _stored(fake_redis_bot) == GuildConfig(volume=0.5)
        await _invoke(cog, settings_ctx, "volume reset")
        assert await _stored(fake_redis_bot) == GuildConfig()
        settings_ctx.bot.is_owner.assert_not_awaited()

    @pytest.mark.parametrize("arg", ["timezone Europe/London", "idle-timeout 10m"])
    async def test_that_listener_may_not_change_anything_else(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis, arg: str
    ) -> None:
        _in_voice(settings_ctx, bot_channel=True)
        await _invoke(cog, settings_ctx, arg)
        assert _text(settings_ctx) == card.NO_PERMISSION
        _denied_by_the_operator_check(settings_ctx)
        assert await _stored(fake_redis_bot) == GuildConfig()

    async def test_a_listener_in_another_channel_is_told_the_voice_route(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        _in_voice(settings_ctx, bot_channel=False)
        await _invoke(cog, settings_ctx, "volume 50")
        assert _text(settings_ctx) == card.NO_PERMISSION_VOLUME
        _denied_by_the_operator_check(settings_ctx)
        assert await _stored(fake_redis_bot) == GuildConfig()

    async def test_out_of_voice_the_listener_is_refused(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "volume 50")
        assert _text(settings_ctx) == card.NO_PERMISSION_VOLUME
        _denied_by_the_operator_check(settings_ctx)

    async def test_with_the_bot_out_of_voice_any_channel_will_do(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        _in_voice(settings_ctx, bot_channel=None)
        await _invoke(cog, settings_ctx, "volume 50")
        assert await _stored(fake_redis_bot) == GuildConfig(volume=0.5)
        settings_ctx.bot.is_owner.assert_not_awaited()


class TestDirectMessages:
    @pytest.fixture
    def dm(self, settings_ctx: MagicMock) -> MagicMock:
        settings_ctx.guild = None
        settings_ctx.author = MagicMock(spec=discord.User)
        settings_ctx.author.mention = MENTION
        return settings_ctx

    async def test_the_operators_settings_is_the_bot_card(
        self, cog: MusicBot, dm: MagicMock
    ) -> None:
        await _invoke(cog, _as_operator(dm))
        assert _embed(dm).title == "Bot settings"

    async def test_a_setting_without_bot_asks_the_operator_for_the_scope(
        self, cog: MusicBot, dm: MagicMock
    ) -> None:
        await _invoke(cog, _as_operator(dm), "heartbeat 5s")
        assert _text(dm) == card.DM_NEEDS_BOT
        await _invoke(cog, dm, "volume 50")
        assert _text(dm) == card.DM_NEEDS_BOT

    async def test_a_bot_write_is_not_available_yet(
        self, cog: MusicBot, dm: MagicMock
    ) -> None:
        await _invoke(cog, _as_operator(dm), "bot heartbeat 5s")
        assert _text(dm) == card.BOT_WRITE_UNAVAILABLE
        assert config.override("HEARTBEAT_INTERVAL_SECS") is None

    async def test_a_bot_detail_is_shown(self, cog: MusicBot, dm: MagicMock) -> None:
        await _invoke(cog, _as_operator(dm), "bot heartbeat")
        assert _text(dm).startswith("**Heartbeat** (`heartbeat`)")

    async def test_the_session_debug_default_is_labelled_against_debug_mode(
        self, cog: MusicBot, dm: MagicMock
    ) -> None:
        cog.debug_settings.set_default_override(True)
        await _invoke(cog, _as_operator(dm), "bot debug-default")
        assert "Current **on** (bot owner, until restart; default off)" in _text(dm)

    @pytest.mark.parametrize("arg", ["", "bot", "bot heartbeet", "volume 50"])
    async def test_anyone_else_is_told_settings_are_per_server(
        self, cog: MusicBot, dm: MagicMock, arg: str
    ) -> None:
        await _invoke(cog, dm, arg)
        assert _text(dm) == card.DM_PER_SERVER
        _denied_by_the_operator_check(dm)


class TestScopes:
    async def test_a_member_typing_a_bot_key_is_told_it_is_the_operators(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "heartbeat")
        assert "only the bot's operator can change" in _text(settings_ctx)

    async def test_the_operator_typing_a_bot_key_is_given_the_bot_form(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, _as_operator(settings_ctx), "heartbeat 5s")
        assert "`-settings bot heartbeat`" in _text(settings_ctx)

    @pytest.mark.parametrize("operator", [False, True])
    async def test_a_server_key_after_bot_is_named_to_anyone(
        self, cog: MusicBot, settings_ctx: MagicMock, operator: bool
    ) -> None:
        settings_ctx.bot.is_owner = AsyncMock(return_value=operator)
        await _invoke(cog, settings_ctx, "bot volume 50")
        assert "`volume` is set per server" in _text(settings_ctx)

    @pytest.mark.parametrize(
        "arg", ["bot", "bot heartbeat", "bot heartbeet", "bot heartbeat 1s"]
    )
    async def test_a_member_sees_nothing_bot_wide(
        self, cog: MusicBot, settings_ctx: MagicMock, arg: str
    ) -> None:
        """No card, no detail, no suggestion and no range: each would name a bot
        setting to someone who cannot change it."""
        await _invoke(cog, settings_ctx, arg)
        assert _text(settings_ctx) == card.OPERATOR_ONLY
        _denied_by_the_operator_check(settings_ctx)

    async def test_an_unconfirmed_operator_is_told_so(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        settings_ctx.bot.is_owner = AsyncMock(side_effect=RuntimeError("503"))
        await _invoke(cog, settings_ctx, "bot")
        assert _text(settings_ctx) == card.OPERATOR_UNCONFIRMED

    async def test_the_operator_gets_the_bot_typo_suggestion(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, _as_operator(settings_ctx), "bot heartbeet 5s")
        assert "did you mean `heartbeat`?" in _text(settings_ctx)

    async def test_a_bot_write_in_a_server_is_not_available_yet(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, _as_operator(settings_ctx), "bot heartbeat reset")
        assert _text(settings_ctx) == card.BOT_WRITE_UNAVAILABLE


class TestReplies:
    """Each change's reply, with "(was …)" from the write's own previous entry."""

    async def test_volume(self, cog: MusicBot, settings_ctx: MagicMock) -> None:
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "volume 50")
        assert _text(ctx) == (
            "**Volume** is now **50%** for this server (was **100%**, the default). "
            "It applies from the next song. It is saved for this server. "
            f"Changed by {MENTION}."
        )
        await _invoke(cog, ctx, "volume 80%")
        assert "(was **50%**, set here)" in _text(ctx)
        await _invoke(cog, ctx, "volume reset")
        assert _text(ctx) == (
            "**Volume** is back to the default, **100%**. It is saved for this "
            f"server. Changed by {MENTION}."
        )

    async def test_timezone(self, cog: MusicBot, settings_ctx: MagicMock) -> None:
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "tz europe/london")
        assert _text(ctx) == (
            "**Timezone** is now **Europe/London** for this server (was "
            "**America/Los_Angeles**, the default). It applies from the next time a "
            f"card is drawn. It is saved for this server. Changed by {MENTION}."
        )
        await _invoke(cog, ctx, "reset timezone")
        assert _text(ctx) == (
            "**Timezone** is back to the default, **America/Los_Angeles**. It is "
            f"saved for this server. Changed by {MENTION}."
        )

    async def test_idle_timeout(self, cog: MusicBot, settings_ctx: MagicMock) -> None:
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "leave-when-idle 10m")
        assert _text(ctx) == (
            "**Leave when idle** is now **10:00** for this server (was **5:00**, the "
            "default). It applies the next time the queue runs empty. It is saved "
            f"for this server. Changed by {MENTION}."
        )
        assert cog.guild_settings.idle_timeout_secs(GUILD) == 600.0
        await _invoke(cog, ctx, "idle reset")
        assert _text(ctx) == (
            "**Leave when idle** is back to the default, **5:00**. It is saved for "
            f"this server. Changed by {MENTION}."
        )
        assert cog.guild_settings.idle_timeout_secs(GUILD) == 300.0

    async def test_alone_timeout(self, cog: MusicBot, settings_ctx: MagicMock) -> None:
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "alone 1:30")
        assert _text(ctx) == (
            "**Leave when alone** is now **1:30** for this server (was **0:10**, the "
            "default). It applies the next time the channel empties. It is saved for "
            f"this server. Changed by {MENTION}."
        )
        assert cog.guild_settings.alone_timeout_secs(GUILD) == 90.0

    async def test_np_refresh(self, cog: MusicBot, settings_ctx: MagicMock) -> None:
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "progress-bar 10")
        assert _text(ctx) == (
            "**Progress bar refresh** is now **10s** for this server (was **3s**, the "
            "default). It applies from the next tick. It is saved for this server. "
            f"Changed by {MENTION}."
        )
        assert cog.guild_settings.np_refresh_secs(GUILD) == 10.0
        await _invoke(cog, ctx, "np-refresh 2")
        assert _text(ctx) == (
            "**Progress bar refresh** has to be between **3s** and **30s** here: the "
            "bot refreshes no faster than **3s**."
        )
        await _invoke(cog, ctx, "np-refresh reset")
        assert _text(ctx) == (
            "**Progress bar refresh** here is back to the bot's default, which is "
            f"**3s** right now. It is saved for this server. Changed by {MENTION}."
        )

    async def test_debug_on_names_what_it_publishes(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "debug on")
        assert _text(ctx) == (
            "**Debug footer** is now **on** for this server (was **off**, the "
            "default). While it is on, every embed here — including the live Now "
            "Playing card — shows the bot process's load to anyone who can read the "
            f"channel. It is saved for this server. Changed by {MENTION}."
        )
        await _invoke(cog, ctx, "debug off")
        assert "It applies immediately." in _text(ctx)

    async def test_debug_reset_names_the_bots_default_it_now_follows(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        cog.debug_settings.set_default_override(True)
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "debug off")
        await _invoke(cog, ctx, "debug reset")
        assert _text(ctx) == (
            "**Debug footer** here is back to the bot's default, which is **on** "
            f"right now. It is saved for this server. Changed by {MENTION}."
        )
        assert cog.debug_settings.enabled(GUILD) is True

    async def test_an_unsaved_write_says_so(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        ctx = _with_manage_server(settings_ctx)
        with patch.object(
            GuildRedisStore, "set_volume", new=AsyncMock(return_value=False)
        ):
            await _invoke(cog, ctx, "volume 50")
        assert _text(ctx).endswith(
            "⚠️ It could not be saved (Redis is unavailable), so it applies until "
            f"the bot restarts. Changed by {MENTION}."
        )

    async def test_a_stalled_store_replies_not_saved_within_the_timeout(
        self,
        cog: MusicBot,
        settings_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pool has no socket_timeout: without the bound, a Redis that accepts
        and never answers would hold the command, and its reply, for good."""
        monkeypatch.setattr("src.settings.CONFIG_IO_TIMEOUT_SECS", 0.05)
        never = asyncio.Event()

        async def _stall(*args: Any, **kwargs: Any) -> bool:
            await never.wait()
            return True

        ctx = _with_manage_server(settings_ctx)
        with patch.object(GuildRedisStore, "set_timezone", new=_stall):
            async with asyncio.timeout(2):
                await _invoke(cog, ctx, "timezone Europe/London")
        assert "could not be saved" in _text(ctx)
        await _invoke(cog, ctx)
        assert "**Timezone** · Europe/London · not saved" in str(_embed(ctx).fields)

    async def test_what_it_replaced_is_left_out_when_the_stored_values_are_unread(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        await GuildRedisStore(fake_redis_bot, GUILD).set_volume(0.8, writer=None)
        ctx = _with_manage_server(settings_ctx)
        with patch.object(
            GuildRedisStore, "read_config", new=AsyncMock(return_value=None)
        ):
            await _invoke(cog, ctx, "volume 50")
        assert _text(ctx).startswith("**Volume** is now **50%** for this server. It")

    async def test_a_change_logs_old_and_new(
        self,
        cog: MusicBot,
        settings_ctx: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="src.commands.settings"):
            await _invoke(cog, _with_manage_server(settings_ctx), "volume 50")
        assert "settings: volume 100% -> 50%" in caplog.text

    async def test_span_attributes_carry_the_request_never_the_input(
        self, cog: MusicBot, settings_ctx: MagicMock, span: MagicMock
    ) -> None:
        await _invoke(cog, _with_manage_server(settings_ctx), "vol 50")
        recorded = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert recorded == {
            "settings.scope": "server",
            "settings.action": "set",
            "settings.key": "volume",
            "settings.persisted": True,
        }

    async def test_a_refusal_records_its_slug(
        self, cog: MusicBot, settings_ctx: MagicMock, span: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "volume 150")
        span.set_attribute.assert_called_once_with("settings.refused", "out_of_range")


class TestCard:
    async def test_defaults(self, cog: MusicBot, settings_ctx: MagicMock) -> None:
        await _invoke(cog, settings_ctx)
        values = [field.value for field in _embed(settings_ctx).fields]
        assert values == [
            "**Volume** · 100% · default\n**Timezone** · America/Los_Angeles · default",
            "**Leave when idle** · 5:00 · default\n**Leave when alone** · 0:10 · default",
            "**Progress bar refresh** · 3s · default",
            "**Debug footer** · off · default",
        ]

    async def test_stored_values_are_read_before_rendering(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        await GuildRedisStore(fake_redis_bot, GUILD).set_volume(0.8, writer=None)
        await _invoke(cog, settings_ctx)
        assert "**Volume** · 80% · set here" in str(_embed(settings_ctx).fields)

    async def test_an_unreadable_store_says_so_and_shows_defaults(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        await GuildRedisStore(fake_redis_bot, GUILD).set_volume(0.8, writer=None)
        with patch.object(
            GuildRedisStore, "read_config", new=AsyncMock(return_value=None)
        ):
            await _invoke(cog, settings_ctx)
        embed = _embed(settings_ctx)
        assert (embed.description or "").startswith("⚠️ Couldn't read")
        assert embed.color == card.DEGRADED_COLOR
        assert "**Volume** · 100% · default" in str(embed.fields)

    async def test_the_operators_card_points_at_the_bot_card_and_carries_none(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        """A response in the player's home channel becomes the Now Playing host,
        whose embeds ride every progress tick: a bot section there would be
        re-sent every few seconds."""
        await _invoke(cog, _as_operator(settings_ctx))
        settings_ctx.send.assert_awaited_once()
        embed = _embed(settings_ctx)
        assert (embed.description or "").endswith(
            "`-settings bot` shows bot-wide settings."
        )
        assert "heartbeat" not in str(embed.to_dict())

    async def test_a_members_card_has_no_pointer(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx)
        assert "-settings bot" not in str(_embed(settings_ctx).to_dict())

    async def test_rows_are_prose_not_code(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx)
        assert all("```" not in (f.value or "") for f in _embed(settings_ctx).fields)

    async def test_one_settings_detail(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "vol")
        assert _text(settings_ctx) == (
            "**Volume** (`volume`; also `vol`) — Playback level. Current **100%** "
            "(default) · Default 100% · Allowed 0%–100% · Applies from the next song "
            "· Can be changed with Manage Server, or by anyone in the bot's voice "
            "channel."
        )


class TestTheLivePlayer:
    async def test_volume_and_timezone_reach_a_registered_player(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        player = MagicMock()
        player.volume = 1.0
        cog.mps[GUILD] = player
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "volume 30")
        assert player.volume == 0.3
        await _invoke(cog, ctx, "timezone Asia/Tokyo")
        assert player.timezone == ZoneInfo("Asia/Tokyo")
        await _invoke(cog, ctx, "volume reset")
        assert player.volume == DEFAULT_VOLUME

    async def test_no_player_is_built(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, _with_manage_server(settings_ctx), "volume 30")
        assert cog.mps == {}

    async def test_an_existing_players_home_channel_is_left_alone(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        """cog_before_invoke's get_mp() re-homes a player to the invoking channel;
        -settings typed in another channel must not move its Now Playing card."""
        player = MagicMock()
        cog.mps[GUILD] = player
        settings_ctx.command.extras = dict(MusicBot.settings.extras)
        cog.get_mp = MagicMock()
        await cog.cog_before_invoke(settings_ctx)
        cog.get_mp.assert_not_called()
        player.set_context.assert_not_called()


class TestInputIsNeverEchoed:
    @pytest.mark.parametrize(
        "hostile",
        ["<@&123456789012345678>", "@everyone", "``` x ```", "[a](https://evil.test)"],
    )
    async def test_no_reply_contains_it(
        self, cog: MusicBot, settings_ctx: MagicMock, hostile: str
    ) -> None:
        ctx = _with_manage_server(settings_ctx)
        for arg in (
            hostile,
            f"volume {hostile}",
            f"timezone {hostile}",
            f"bot {hostile}",
        ):
            await _invoke(cog, ctx, arg)
            assert hostile not in str(_embed(ctx).to_dict()), arg


class TestTooMuch:
    async def test_words_after_the_value_change_nothing(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        """Release notes whose first line is `- config debug on for the staging bot`
        must not turn debug mode on for this server."""
        ctx = _with_manage_server(settings_ctx)
        write = AsyncMock(return_value=True)
        with patch.object(GuildRedisStore, "set_debug_mode", new=write):
            await _invoke(cog, ctx, "debug on for the staging bot")
        assert "That's more than `-settings` understands" in _text(ctx)
        write.assert_not_awaited()
        assert cog.debug_settings.has_override(GUILD) is False


class TestOneLine:
    async def test_a_line_break_after_the_command_word_is_too_much(
        self, cog: MusicBot, settings_ctx: MagicMock, fake_redis_bot: Redis
    ) -> None:
        """discord.py strips the break between `-settings` and its argument, so
        `arg` alone reads as the one-line `volume 50`; the tail keeps the break."""
        ctx = _with_manage_server(settings_ctx)
        await _invoke(cog, ctx, "volume 50", tail="settings\nvolume 50")
        assert "That's more than `-settings` understands" in _text(ctx)
        assert await _stored(fake_redis_bot) == GuildConfig()


class TestBulletShapedMessages:
    @pytest.mark.parametrize(
        ("tail", "arg"),
        [
            (" settings page is broken", "page is broken"),
            (" config debug on for the staging bot", "debug on for the staging bot"),
            (" settings\n- volume 50", "- volume 50"),
        ],
    )
    async def test_a_refused_list_item_gets_no_reply(
        self,
        cog: MusicBot,
        settings_ctx: MagicMock,
        span: MagicMock,
        caplog: pytest.LogCaptureFixture,
        tail: str,
        arg: str,
    ) -> None:
        ctx = _with_manage_server(settings_ctx)
        with caplog.at_level(logging.DEBUG, logger="src.commands.settings"):
            await _invoke(cog, ctx, arg, tail=tail)
        ctx.send.assert_not_awaited()
        assert cog.guild_settings.peek(GUILD) is None
        assert caplog.text.count("bullet-shaped message ignored") == 1
        span.set_attribute.assert_called_once_with("settings.refused", "bullet_shape")

    async def test_without_the_space_the_hint_is_sent(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "debug on for the staging bot")
        assert "That's more than `-settings` understands" in _text(settings_ctx)

    async def test_a_complete_request_still_answers(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "", tail=" settings")
        assert (_embed(settings_ctx).title or "").startswith("Server settings")

    async def test_a_refusal_only_an_exact_name_reaches_still_answers(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        await _invoke(cog, settings_ctx, "volume 150", tail=" settings volume 150")
        assert _text(settings_ctx).startswith("**Volume** has to be between")


class TestEverySend:
    async def test_mentions_nobody_and_changes_name_their_author(
        self, cog: MusicBot, settings_ctx: MagicMock
    ) -> None:
        ctx = _with_manage_server(settings_ctx)
        changes = [
            "volume 40",
            "volume reset",
            "timezone UTC",
            "tz reset",
            "debug on",
            "debug reset",
        ]
        views = ["", "volume", "bot", "nope", "volume 150"]
        for arg in (*changes, *views):
            await _invoke(cog, ctx, arg)
            if arg in changes:
                assert _text(ctx).endswith(f"Changed by {MENTION}."), arg
        for call in ctx.send.await_args_list:
            mentions = call.kwargs["allowed_mentions"]
            assert mentions.to_dict() == discord.AllowedMentions.none().to_dict()
