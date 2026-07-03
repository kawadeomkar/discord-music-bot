"""Tests for src/main.py — MusicBotApp lifecycle (setup_hook, close, on_ready)."""

from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest
from discord.ext import commands

from src.main import EXTENSIONS, MusicBotApp


@pytest.fixture
def app() -> MusicBotApp:
    """Bypass discord.py __init__; wire up minimal internal state so properties work."""
    instance = MusicBotApp.__new__(MusicBotApp)
    instance._redis_pool = None
    instance.redis = None
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
    async def test_creates_redis_pool(self, app: MusicBotApp) -> None:
        mock_pool = MagicMock()
        with (
            patch("src.main.create_redis_pool", return_value=mock_pool) as mock_create,
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        mock_create.assert_called_once()
        assert app._redis_pool is mock_pool

    async def test_assigns_redis_client(self, app: MusicBotApp) -> None:
        mock_redis = MagicMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=mock_redis),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert app.redis is mock_redis

    async def test_loads_all_extensions(self, app: MusicBotApp) -> None:
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


class TestClose:
    async def test_closes_redis_pool_when_set(self, app: MusicBotApp) -> None:
        mock_pool = MagicMock()
        app._redis_pool = mock_pool
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_close,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        mock_close.assert_awaited_once_with(mock_pool)

    async def test_skips_close_when_pool_is_none(self, app: MusicBotApp) -> None:
        app._redis_pool = None
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_close,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        mock_close.assert_not_awaited()

    async def test_calls_super_close(self, app: MusicBotApp) -> None:
        app._redis_pool = None
        with patch.object(
            commands.AutoShardedBot, "close", new=AsyncMock()
        ) as mock_super:
            await app.close()
        mock_super.assert_awaited_once()


class TestOnReady:
    @pytest.fixture(autouse=True)
    def _patch_latency(self) -> Iterator[None]:
        """AutoShardedClient.latency reads __shards; patch at the class level."""
        with patch.object(
            MusicBotApp, "latency", new_callable=PropertyMock, return_value=0.05
        ):
            yield

    async def test_sets_presence(self, app: MusicBotApp) -> None:
        await app.on_ready()
        app.change_presence.assert_awaited_once()

    async def test_no_error_when_user_is_none(self, app: MusicBotApp) -> None:
        app._connection.user = None
        await app.on_ready()

    async def test_logs_user_info_when_user_set(self, app: MusicBotApp) -> None:
        user = MagicMock()
        user.name = "TestBot"
        user.id = 123456789
        app._connection.user = user
        await app.on_ready()
        app.change_presence.assert_awaited_once()

    async def test_presence_sets_online_status(self, app: MusicBotApp) -> None:
        await app.on_ready()
        call_kwargs = app.change_presence.call_args[1]
        assert call_kwargs["status"] == discord.Status.online
