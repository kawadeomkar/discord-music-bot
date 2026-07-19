"""Tests for src/main.py — MusicBotApp lifecycle (setup_hook, close, on_ready)."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest
from discord.ext import commands

from src.main import EXTENSIONS, MusicBotApp


@pytest.fixture
def app():
    """Bypass discord.py __init__; wire up minimal internal state so properties work."""
    instance = MusicBotApp.__new__(MusicBotApp)
    instance._redis_pool = None
    instance.redis = None
    instance.history_archive = None
    instance.history_drainer = None
    # BotBase stores cogs in a name-mangled dict; initialize it so the property works.
    instance._BotBase__cogs = {}
    # discord.Client properties (user, guilds, intents) read from _connection.
    conn = MagicMock()
    conn.user = None
    conn.guilds = []
    conn.intents = MagicMock()
    conn.intents.voice_states = True
    instance._connection = conn
    # latency reads self.ws; None → returns float('nan'), which is fine for logging.
    instance.ws = None
    instance.change_presence = AsyncMock()
    return instance


class TestSetupHook:
    async def test_creates_redis_pool(self, app):
        mock_pool = MagicMock()
        with (
            patch("src.main.create_redis_pool", return_value=mock_pool) as mock_create,
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        mock_create.assert_called_once()
        assert app._redis_pool is mock_pool

    async def test_assigns_redis_client(self, app):
        mock_redis = MagicMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=mock_redis),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert app.redis is mock_redis

    async def test_loads_all_extensions(self, app):
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=mock_load),
        ):
            await app.setup_hook()
        assert mock_load.call_count == len(EXTENSIONS)
        for ext in EXTENSIONS:
            mock_load.assert_any_await(ext)

    async def test_no_postgres_url_leaves_archive_off(self, app, monkeypatch):
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert app.history_archive is None
        assert app.history_drainer is None

    async def test_postgres_url_starts_archive_and_drainer(self, app, monkeypatch):
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
        mock_archive = MagicMock()
        mock_drainer = MagicMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch(
                "src.main.PostgresHistoryArchive", return_value=mock_archive
            ) as mock_pg,
            patch(
                "src.main.HistoryOutboxDrainer", return_value=mock_drainer
            ) as mock_dr,
        ):
            await app.setup_hook()
        mock_pg.assert_called_once_with("postgresql://x")
        mock_dr.assert_called_once_with(app.redis, mock_archive)
        mock_drainer.start.assert_called_once()
        assert app.history_archive is mock_archive
        assert app.history_drainer is mock_drainer


class TestClose:
    async def test_closes_redis_pool_when_set(self, app):
        mock_pool = MagicMock()
        app._redis_pool = mock_pool
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_close,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        mock_close.assert_awaited_once_with(mock_pool)

    async def test_skips_close_when_pool_is_none(self, app):
        app._redis_pool = None
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_close,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        mock_close.assert_not_awaited()

    async def test_calls_super_close(self, app):
        app._redis_pool = None
        with patch.object(
            commands.AutoShardedBot, "close", new=AsyncMock()
        ) as mock_super:
            await app.close()
        mock_super.assert_awaited_once()

    async def test_stops_drainer_then_archive_then_pool(self, app):
        # Teardown order matters: the drainer's final drain still needs both
        # the archive and Redis, and the archive pool must go before the
        # Redis pool it never depended on but shuts down alongside.
        order = []
        app.history_drainer = MagicMock(
            stop=AsyncMock(side_effect=lambda: order.append("drainer"))
        )
        app.history_archive = MagicMock(
            close=AsyncMock(side_effect=lambda: order.append("archive"))
        )
        app._redis_pool = MagicMock()
        mock_pool_close = AsyncMock(side_effect=lambda _p: order.append("pool"))
        with (
            patch("src.main.close_redis_pool", new=mock_pool_close),
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        assert order == ["drainer", "archive", "pool"]


class TestHelpFlag:
    """`--help` anywhere in a command message diverts to that command's help
    embed before any other logic runs — global checks, the cog's voice-channel
    gate, argument parsing."""

    def _ctx(self, content: str, *, command_found: bool = True) -> MagicMock:
        ctx = MagicMock()
        ctx.command = MagicMock() if command_found else None
        ctx.message.content = content
        ctx.send_help = AsyncMock()
        return ctx

    async def test_help_flag_diverts_to_command_help(self, app):
        ctx = self._ctx("-play --help")
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        ctx.send_help.assert_awaited_once_with(ctx.command)
        mock_super.assert_not_awaited()

    async def test_help_flag_matches_anywhere_in_the_message(self, app):
        ctx = self._ctx("-play lofi hip hop --help radio")
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        ctx.send_help.assert_awaited_once_with(ctx.command)
        mock_super.assert_not_awaited()

    async def test_without_flag_invokes_normally(self, app):
        ctx = self._ctx("-play lofi hip hop")
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        mock_super.assert_awaited_once_with(ctx)
        ctx.send_help.assert_not_awaited()

    async def test_unknown_command_falls_through(self, app):
        """`-bogus --help` must keep raising CommandNotFound downstream, not
        try to render help for a command that doesn't exist."""
        ctx = self._ctx("-bogus --help", command_found=False)
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        mock_super.assert_awaited_once_with(ctx)
        ctx.send_help.assert_not_awaited()


class TestOnReady:
    @pytest.fixture(autouse=True)
    def _patch_latency(self):
        """AutoShardedClient.latency reads __shards; patch at the class level."""
        with patch.object(
            MusicBotApp, "latency", new_callable=PropertyMock, return_value=0.05
        ):
            yield

    async def test_sets_presence(self, app):
        await app.on_ready()
        app.change_presence.assert_awaited_once()

    async def test_no_error_when_user_is_none(self, app):
        app._connection.user = None
        await app.on_ready()

    async def test_logs_user_info_when_user_set(self, app):
        user = MagicMock()
        user.name = "TestBot"
        user.id = 123456789
        app._connection.user = user
        await app.on_ready()
        app.change_presence.assert_awaited_once()

    async def test_presence_sets_online_status(self, app):
        await app.on_ready()
        call_kwargs = app.change_presence.call_args[1]
        assert call_kwargs["status"] == discord.Status.online
