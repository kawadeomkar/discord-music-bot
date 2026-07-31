"""Tests for src/main.py — MusicBotApp lifecycle (setup_hook, close, on_ready)."""

import asyncio
from collections.abc import Iterator
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest
from discord.ext import commands
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.main import EXTENSIONS, MusicBotApp
from tests.helpers import mocked


@pytest.fixture
def app() -> MusicBotApp:
    """Bypass discord.py __init__; wire up minimal internal state so properties work."""
    instance = MusicBotApp.__new__(MusicBotApp)
    instance._redis_pool = None
    instance.redis = None
    # history_archive / history_drainer are deliberately left UNSET. __new__
    # bypasses __init__, and setup_hook is what assigns them — so an unset
    # attribute is exactly the real pre-setup_hook state, which is what
    # close()'s getattr guard exists to survive.
    # BotBase stores cogs in a name-mangled private dict; initialize it so the
    # property works. Set via setattr: the mangled name is deliberately not part
    # of BotBase's declared surface, so it is invisible to the type checker.
    setattr(instance, "_BotBase__cogs", {})
    # discord.Client properties (user, guilds, intents) read from _connection.
    conn = MagicMock()
    conn.user = None
    conn.guilds = []
    conn.intents = MagicMock()
    conn.intents.voice_states = True
    instance._connection = conn
    # latency reads self.ws and returns float('nan') for any falsy value, which
    # is fine for logging. MISSING is discord.py's own "not connected yet"
    # sentinel and is falsy, so it takes that same branch.
    instance.ws = discord.utils.MISSING
    instance.change_presence = AsyncMock()
    return instance


class TestSetupHook:
    @pytest.fixture(autouse=True)
    def postgres_configured(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """POSTGRES_URL is required — setup_hook raises without it. Every test
        in this class that isn't specifically about that refusal needs it set,
        and needs the archive/drainer stubbed so no real pool or task is
        created. Tests that assert on the constructors re-patch them locally.

        ensure_outbox_group is stubbed for the same reason: `app.redis` here is
        a MagicMock, so the real one would await a non-awaitable. Its own
        behaviour — that it runs, that it runs BEFORE the drainer, and that a
        WRONGTYPE aborts startup — is asserted in TestOutboxGroupBootstrap,
        which re-patches it locally."""
        monkeypatch.setenv("POSTGRES_URL", "postgresql://stub")
        with (
            patch("src.main.PostgresHistoryArchive"),
            patch("src.main.HistoryOutboxDrainer"),
            patch("src.main.ensure_outbox_group", new=AsyncMock()),
        ):
            yield

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

    @pytest.mark.parametrize("value", [None, ""])
    async def test_missing_postgres_url_refuses_to_start(
        self,
        app: MusicBotApp,
        monkeypatch: pytest.MonkeyPatch,
        value: Optional[str],
    ) -> None:
        """The archive is a required tier, so an unset (or empty) POSTGRES_URL
        is a startup error, not a degraded mode. Failing here is what stops the
        bot from silently LPUSHing onto an outbox no drainer will ever read."""
        if value is None:
            monkeypatch.delenv("POSTGRES_URL", raising=False)
        else:
            monkeypatch.setenv("POSTGRES_URL", value)
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=mock_load),
            pytest.raises(RuntimeError, match="POSTGRES_URL is not set"),
        ):
            await app.setup_hook()
        # It refuses BEFORE loading the cogs, so no partially-wired bot is left.
        mock_load.assert_not_awaited()

    async def test_postgres_url_starts_archive_and_drainer(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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


class TestOutboxGroupBootstrap:
    """The setup_hook leg of ensure_outbox_group — the wiring, not the helper.

    The helper's own discrimination (BUSYGROUP tolerated, WRONGTYPE raised) is
    asserted in test_redis_client and against real Redis in the redis tier;
    these pin what setup_hook does with it: that it runs, that it runs BEFORE
    the drainer exists, and that a WRONGTYPE aborts startup. That abort is
    load-bearing (golden rule 5): push_history is @_guild_op, so a pre-R1 LIST
    at history:outbox would otherwise be swallowed into one warning per song —
    total history loss reported at debug volume. Startup is the only place the
    signal can be loud, so startup failing IS the feature under test."""

    async def test_wrongtype_aborts_startup_before_anything_starts(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
        wrongtype = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.ensure_outbox_group", new=AsyncMock(side_effect=wrongtype)),
            patch("src.main.PostgresHistoryArchive") as mock_pg,
            patch("src.main.HistoryOutboxDrainer") as mock_dr,
            patch.object(app, "load_extension", new=mock_load),
            pytest.raises(ResponseError, match="WRONGTYPE"),
        ):
            await app.setup_hook()
        # Aborted BEFORE the archive tier or the cogs: a drainer started
        # against a mis-typed key would raise on every read, and a loaded cog
        # would accept plays whose history has nowhere to go.
        mock_pg.assert_not_called()
        mock_dr.assert_not_called()
        mock_load.assert_not_awaited()

    async def test_group_exists_before_the_drainer_starts(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order, not just occurrence: a drainer started first would XREADGROUP
        against a group that does not exist yet and burn its NOGROUP
        heal-once on a plain startup race."""
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
        order: list[str] = []
        ensure = AsyncMock(side_effect=lambda *_: order.append("group"))
        mock_drainer = MagicMock()
        mock_drainer.start = MagicMock(side_effect=lambda: order.append("drainer"))
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.ensure_outbox_group", new=ensure),
            patch("src.main.PostgresHistoryArchive"),
            patch("src.main.HistoryOutboxDrainer", return_value=mock_drainer),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert order == ["group", "drainer"]
        ensure.assert_awaited_once_with(app.redis)

    @pytest.mark.parametrize(
        "exc",
        [
            RedisConnectionError("Error 111 connecting to localhost:6379"),
            RedisTimeoutError("Timeout connecting to server"),
        ],
        ids=["connection-refused", "timeout"],
    )
    async def test_unreachable_redis_degrades_instead_of_aborting(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch, exc: Exception
    ) -> None:
        """The other side of the WRONGTYPE abort, and the reason the two are
        told apart. Redis being DOWN is the case this bot is built to survive
        everywhere else: create_redis_pool connects lazily, so before this probe
        existed an unreachable Redis cost only persistence. Making the probe
        fatal would turn a Redis blip during a deploy into a bot that refuses to
        boot. Safe because _read_batch heals NOGROUP by calling
        ensure_outbox_group itself, so the drainer creates the group on its
        first tick after Redis returns."""
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
        mock_load = AsyncMock()
        mock_drainer = MagicMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.ensure_outbox_group", new=AsyncMock(side_effect=exc)),
            patch("src.main.PostgresHistoryArchive") as mock_pg,
            patch("src.main.HistoryOutboxDrainer", return_value=mock_drainer),
            patch.object(app, "load_extension", new=mock_load),
        ):
            await app.setup_hook()
        # Everything downstream still came up — that is the whole claim.
        mock_pg.assert_called_once()
        mock_drainer.start.assert_called_once()
        mock_load.assert_awaited()


class TestClose:
    @pytest.fixture(autouse=True)
    def stub_telemetry_shutdown(self) -> Iterator[None]:
        """close() awaits shutdown_telemetry in an executor for real, and it blocks on an
        OTLP force_flush. The yt-dlp pool needs no stub: conftest gives each test its own
        thread-backed YtdlpPool, so close() may shut it down for real."""
        with patch("src.telemetry.shutdown_telemetry"):
            yield

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

    async def test_teardown_order_is_drainer_archive_disconnect_pool(
        self, app: MusicBotApp
    ) -> None:
        """Two independent constraints, asserted together because they compose.

        The drainer's final drain reads the outbox and writes Postgres, so it
        must run while BOTH are alive. super().close() disconnects voice clients
        and can still dispatch events, and anything reaching cleanup() in that
        window writes to Redis — so it must run BEFORE the pool closes, or a
        clean shutdown leaves state looking like a crash and the next start runs
        spurious recovery for cleanly-stopped guilds.
        """
        order: list[str] = []

        async def record_super_close() -> None:
            order.append("disconnect")

        async def record_redis_close(_pool: object) -> None:
            order.append("pool")

        app.history_drainer = MagicMock(
            stop=AsyncMock(side_effect=lambda: order.append("drainer"))
        )
        app.history_archive = MagicMock(
            close=AsyncMock(side_effect=lambda: order.append("archive"))
        )
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=record_redis_close),
            patch.object(
                commands.AutoShardedBot, "close", new=lambda self: record_super_close()
            ),
        ):
            await app.close()
        assert order == ["drainer", "archive", "disconnect", "pool"]

    @pytest.mark.parametrize("sick", ["drainer", "archive"])
    async def test_teardown_completes_when_a_step_raises(
        self, app: MusicBotApp, sick: str
    ) -> None:
        """REGRESSION: drainer.stop() and archive.close() were both unguarded,
        and both can raise — a hung Postgres made archive.close() raise
        TimeoutError after 30s, and a drainer task that died with an exception
        made stop() re-raise it. Because _teardown_started is already set, the
        retry path short-circuits, so every step AFTER the raiser was skipped
        permanently: Redis pool left open, discord.py never closed, the yt-dlp
        pool left to its 61s atexit join, and no spans flushed — which hid the
        very failure that caused it.
        """
        boom = AsyncMock(side_effect=RuntimeError(f"{sick} is wedged"))
        app.history_drainer = MagicMock(stop=boom if sick == "drainer" else AsyncMock())
        app.history_archive = MagicMock(
            close=boom if sick == "archive" else AsyncMock()
        )
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_pool_close,
            patch.object(
                commands.AutoShardedBot, "close", new=AsyncMock()
            ) as mock_super,
        ):
            await app.close()  # must not raise
        # Everything downstream of the sick participant still ran.
        mock_pool_close.assert_awaited_once()
        mock_super.assert_awaited_once()

    async def test_teardown_survives_a_failing_disconnect(
        self, app: MusicBotApp
    ) -> None:
        # super().close() is guarded for the same reason as the two above, now
        # that a step follows it: no participant may skip a later one. Before it
        # moved ahead of the pool close it was last, so nothing could be skipped.
        app.history_drainer = MagicMock(stop=AsyncMock())
        app.history_archive = MagicMock(close=AsyncMock())
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_pool_close,
            patch.object(
                commands.AutoShardedBot,
                "close",
                new=AsyncMock(side_effect=RuntimeError("gateway wedged")),
            ),
        ):
            await app.close()  # must not raise
        mock_pool_close.assert_awaited_once()

    async def test_teardown_runs_only_once(self, app: MusicBotApp) -> None:
        # discord.py calls close() from run()'s finally as well as on demand,
        # and its own idempotence check lives inside super().close() — which the
        # reentrancy guard reaches only by short-circuiting to it. Without the
        # guard, a second close() re-runs drainer.stop(), and two concurrent
        # final drains each peek → insert → retire (H2: 6 pushed, 3 archived,
        # outbox empty).
        app.history_drainer = MagicMock(stop=AsyncMock())
        app.history_archive = MagicMock(close=AsyncMock())
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_pool_close,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()) as sup,
        ):
            await app.close()
            await app.close()
        app.history_drainer.stop.assert_awaited_once()
        app.history_archive.close.assert_awaited_once()
        mock_pool_close.assert_awaited_once()
        # super().close() still runs on the second call — discord.py's own
        # teardown is idempotent and expects to be reached.
        assert sup.await_count == 2

    async def test_concurrent_closes_teardown_once(self, app: MusicBotApp) -> None:
        app.history_drainer = MagicMock(stop=AsyncMock())
        app.history_archive = MagicMock(close=AsyncMock())
        app._redis_pool = None
        with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
            await asyncio.gather(app.close(), app.close())
        app.history_drainer.stop.assert_awaited_once()

    async def test_shuts_down_the_ytdlp_pool(self, app: MusicBotApp) -> None:
        """The extraction workers are child processes — a clean close must join them
        rather than leave them orphaned. Asserts the real pool's state rather than a
        mock call: the pool close() reaches is the one conftest installed."""
        import src.youtube as youtube

        app._redis_pool = None
        with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
            await app.close()
        assert youtube.ytdlp_pool.is_closed


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

    async def test_help_flag_diverts_to_command_help(self, app: MusicBotApp) -> None:
        ctx = self._ctx("-play --help")
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        ctx.send_help.assert_awaited_once_with(ctx.command)
        mock_super.assert_not_awaited()

    async def test_help_flag_matches_anywhere_in_the_message(
        self, app: MusicBotApp
    ) -> None:
        ctx = self._ctx("-play lofi hip hop --help radio")
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        ctx.send_help.assert_awaited_once_with(ctx.command)
        mock_super.assert_not_awaited()

    async def test_without_flag_invokes_normally(self, app: MusicBotApp) -> None:
        ctx = self._ctx("-play lofi hip hop")
        with patch.object(
            commands.AutoShardedBot, "invoke", new=AsyncMock()
        ) as mock_super:
            await app.invoke(ctx)
        mock_super.assert_awaited_once_with(ctx)
        ctx.send_help.assert_not_awaited()

    async def test_unknown_command_falls_through(self, app: MusicBotApp) -> None:
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
    def _patch_latency(self) -> Iterator[None]:
        """AutoShardedClient.latency reads __shards; patch at the class level."""
        with patch.object(
            MusicBotApp, "latency", new_callable=PropertyMock, return_value=0.05
        ):
            yield

    async def test_sets_presence(self, app: MusicBotApp) -> None:
        await app.on_ready()
        mocked(app.change_presence).assert_awaited_once()

    async def test_no_error_when_user_is_none(self, app: MusicBotApp) -> None:
        app._connection.user = None
        await app.on_ready()

    async def test_logs_user_info_when_user_set(self, app: MusicBotApp) -> None:
        user = MagicMock()
        user.name = "TestBot"
        user.id = 123456789
        app._connection.user = user
        await app.on_ready()
        mocked(app.change_presence).assert_awaited_once()

    async def test_presence_sets_online_status(self, app: MusicBotApp) -> None:
        await app.on_ready()
        call_kwargs = mocked(app.change_presence).call_args[1]
        assert call_kwargs["status"] == discord.Status.online
