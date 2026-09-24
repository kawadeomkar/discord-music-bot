"""Tests for src/main.py — MusicBotApp lifecycle (setup_hook, close, on_ready)."""

import asyncio
from collections.abc import Iterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest
from discord.ext import commands

from src import config
from src.main import EXTENSIONS, MusicBotApp, intents
from src.musicbot import MusicBot
from tests.helpers import mocked


@pytest.fixture
def app() -> MusicBotApp:
    """Bypass discord.py __init__; wire up minimal internal state so properties work."""
    instance = MusicBotApp.__new__(MusicBotApp)
    instance._redis_pool = None
    instance.redis = None
    instance.bot_settings = None
    instance._bot_settings_hydration = None
    # history_archive / history_drainer / _liveness_task are deliberately left
    # UNSET: __new__ bypasses __init__ and setup_hook is what assigns them, so
    # unset is exactly the pre-setup_hook state close()'s getattr guard exists
    # to survive.
    # BotBase stores cogs in a name-mangled private dict; initialize it so the
    # property works. Set via setattr: the mangled name is deliberately not part
    # of BotBase's declared surface, so it is invisible to the type checker.
    setattr(instance, "_BotBase__cogs", {})
    # discord.Client properties (user, guilds, intents) read from _connection.
    conn = MagicMock()
    conn.user = None
    conn.guilds = []
    # Never logged in: BotSettings.hydrate returns at once without an id.
    conn.application_id = None
    conn.intents = MagicMock()
    conn.intents.voice_states = True
    instance._connection = conn
    # latency reads self.ws and returns float('nan') for any falsy value, which
    # is fine for logging. MISSING is discord.py's own "not connected yet"
    # sentinel and is falsy, so it takes that same branch.
    instance.ws = discord.utils.MISSING
    instance.change_presence = AsyncMock()
    return instance


@pytest.fixture(autouse=True)
def stub_archive_probe() -> Iterator[AsyncMock]:
    """conftest pins HISTORY_ARCHIVE_ENABLED true for the whole suite, so every
    setup_hook test here takes the enabled arm and would spawn a reachability
    probe against a MagicMock archive, left sleeping between attempts. Stubbed for
    the file; TestArchiveReachabilityProbe drives the real coroutine directly."""
    stub = AsyncMock()
    with patch("src.archive_tier.verify_reachable", new=stub):
        yield stub


class TestAppInitDefaults:
    """The real __init__, against the real constructor — no fixture in the way.

    Everywhere else the `app` fixture bypasses __init__, so this is the only
    place `_archive_tier = None` is asserted to be what the constructor actually
    produces. Reverting it to a bare annotation passes every other test in the
    repo and makes the default deployment raise AttributeError on its first
    -play, through the two properties below.
    """

    def test_the_archive_pair_defaults_to_none(self) -> None:
        # Cheap (~1ms): AutoShardedBot.__init__ neither connects nor needs a
        # running loop. Constructed inside the test, never at module scope —
        # see the yt-dlp pool's spawn/forkserver re-import rule.
        app = MusicBotApp()
        assert app.history_archive is None
        assert app.history_drainer is None

    def test_teardown_flag_starts_clear(self) -> None:
        # Same constructor, same reason: close() is one-shot off this flag, and
        # a stale True would skip teardown entirely.
        assert MusicBotApp()._teardown_started is False

    def test_the_shard_ready_wait_comes_from_the_setting(self) -> None:
        """_delay_ready's loop exits only by timing out, so this is charged to every
        start. Read off _connection, which is what discord.py actually consults: the
        kwarg could stop being honoured and a check on our own value would not see
        it. Against config, not a literal, so the two cannot drift."""
        assert (
            MusicBotApp()._connection.guild_ready_timeout
            == config.GUILD_READY_TIMEOUT_SECS
        )

    def test_the_shard_ready_default_is_shorter_than_the_librarys(self) -> None:
        """2.0 is discord.py's default and the whole saving is in not paying it."""
        assert config.GUILD_READY_TIMEOUT_SECS < 2.0


class TestSetupHook:
    @pytest.fixture(autouse=True)
    def postgres_configured(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """POSTGRES_URL is required — setup_hook raises without it — so every test
        here that isn't about that refusal needs it set, plus the archive/drainer
        stubbed so no real pool or task is created. ensure_outbox_group is stubbed
        because `app.redis` is a MagicMock here; its own behaviour is asserted in
        TestOutboxGroupBootstrap, which re-patches it locally."""
        monkeypatch.setenv("POSTGRES_URL", "postgresql://stub")
        with (
            patch("src.archive_tier.PostgresHistoryArchive"),
            patch("src.archive_tier.HistoryOutboxDrainer"),
            patch("src.archive_tier.ensure_outbox_group", new=AsyncMock()),
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

    async def test_prewarms_the_pool_with_the_worker_warm_up(
        self, app: MusicBotApp
    ) -> None:
        """The warm-up is a parameter with a default, so a setup_hook that forgot it
        would still spawn workers, still type-check, and still pay the
        first-YoutubeDL cost on the first -play. Only this pins the wiring."""
        from src.youtube import warm_worker

        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.youtube.ytdlp_pool.prewarm") as mock_prewarm,
        ):
            await app.setup_hook()

        mock_prewarm.assert_called_once_with(warm_worker)

    async def test_startup_checks_the_ytdlp_cache_is_writable(
        self, app: MusicBotApp
    ) -> None:
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.youtube.ytdlp_pool.prewarm"),
            patch("src.youtube.warn_if_cache_unwritable") as check,
        ):
            await app.setup_hook()

        check.assert_called_once_with()

    @pytest.mark.parametrize("value", [None, ""])
    async def test_missing_postgres_url_refuses_to_start(
        self,
        app: MusicBotApp,
        monkeypatch: pytest.MonkeyPatch,
        value: Optional[str],
    ) -> None:
        """An enabled archive requires its database (the suite default pins
        HISTORY_ARCHIVE_ENABLED=true), so an unset (or empty) POSTGRES_URL is
        a startup error, not a degraded mode. Failing here is what stops the
        bot from silently XADDing onto an outbox no drainer will ever read."""
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
        # It refuses before loading the cogs, so no partially-wired bot is left.
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
                "src.archive_tier.PostgresHistoryArchive", return_value=mock_archive
            ) as mock_pg,
            patch(
                "src.archive_tier.HistoryOutboxDrainer", return_value=mock_drainer
            ) as mock_dr,
        ):
            await app.setup_hook()
        mock_pg.assert_called_once_with("postgresql://x")
        mock_dr.assert_called_once_with(app.redis, mock_archive)
        mock_drainer.start.assert_called_once()
        assert app.history_archive is mock_archive
        assert app.history_drainer is mock_drainer


class TestBotSettingsStartup:
    """BOT_SETTINGS_OVERRIDES is read before the pool, and hydration never holds
    setup_hook."""

    @pytest.fixture(autouse=True)
    def archive_disabled(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")

    async def test_garbage_aborts_before_the_pool(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BOT_SETTINGS_OVERRIDES", "ignored")
        with (
            patch("src.main.create_redis_pool") as create,
            pytest.raises(ValueError, match="BOT_SETTINGS_OVERRIDES"),
        ):
            await app.setup_hook()
        create.assert_not_called()

    async def test_setup_hook_returns_while_hydration_is_stalled(
        self, app: MusicBotApp
    ) -> None:
        """Every knob runs on its environment value until the read lands."""
        stalled = asyncio.Event()

        async def never_answers(_: object) -> None:
            await stalled.wait()

        app._connection.application_id = 123456789012345678
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.archive_tier.outbox_depth", new=AsyncMock(return_value=0)),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.settings.BotConfigStore.read_config", new=never_answers),
        ):
            async with asyncio.timeout(5):
                await app.setup_hook()
            hydration = app._bot_settings_hydration
            assert hydration is not None and not hydration.done()
            assert app.bot_settings is not None
            assert app.bot_settings.hydrated is False
            assert (
                config.heartbeat_interval_secs()
                == config.heartbeat_interval_secs.baseline
            )
            hydration.cancel()

    async def test_the_flag_reaches_bot_settings(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BOT_SETTINGS_OVERRIDES", "ignore")
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.archive_tier.outbox_depth", new=AsyncMock(return_value=0)),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert app.bot_settings is not None
        assert app.bot_settings.ignore_stored is True

    async def test_on_ready_retries_a_failed_hydration_once_it_has_ended(
        self, app: MusicBotApp
    ) -> None:
        bot_settings = MagicMock()
        bot_settings.hydrated = False
        bot_settings.hydrate_until_read = AsyncMock()
        app.bot_settings = bot_settings
        with patch.object(
            MusicBotApp, "latency", new_callable=PropertyMock, return_value=0.05
        ):
            await app.on_ready()
            task = app._bot_settings_hydration
            assert task is not None
            await task
            # Still running from the last READY: no second read beside it.
            parked = asyncio.create_task(asyncio.Event().wait())
            app._bot_settings_hydration = parked
            await app.on_ready()
            assert app._bot_settings_hydration is parked
            parked.cancel()
            bot_settings.hydrated = True
            app._bot_settings_hydration = None
            await app.on_ready()
        bot_settings.hydrate_until_read.assert_awaited_once()


class TestClose:
    @staticmethod
    def _tier(aclose: Optional[AsyncMock] = None) -> MagicMock:
        """The archive tier as close() sees it: one participant with one aclose().
        What order the three things INSIDE it come down in is asserted in
        tests/test_archive_tier.py, which is where that order now lives."""
        return MagicMock(aclose=aclose or AsyncMock())

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

    async def test_a_pending_bot_settings_read_is_cancelled(
        self, app: MusicBotApp
    ) -> None:
        """A retry still sleeping on its backoff would outlive the Redis pool."""
        app._redis_pool = None
        pending = asyncio.create_task(asyncio.Event().wait())
        app._bot_settings_hydration = pending
        with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
            await app.close()
        await asyncio.gather(pending, return_exceptions=True)
        assert pending.cancelled()
        assert app._bot_settings_hydration is None

    async def test_calls_super_close(self, app: MusicBotApp) -> None:
        app._redis_pool = None
        with patch.object(
            commands.AutoShardedBot, "close", new=AsyncMock()
        ) as mock_super:
            await app.close()
        mock_super.assert_awaited_once()

    async def test_close_with_the_archive_tier_disabled(self, app: MusicBotApp) -> None:
        """The disabled shape: __init__ assigned None and setup_hook left it
        that way. The `is not None` guards skip the tier's two steps entirely
        and everything downstream still runs — same guards, second meaning
        (the first is surviving a close() before setup_hook ever assigned)."""
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_pool_close,
            patch.object(
                commands.AutoShardedBot, "close", new=AsyncMock()
            ) as mock_super,
        ):
            await app.close()
        mock_pool_close.assert_awaited_once()
        mock_super.assert_awaited_once()

    async def test_teardown_order_is_drainer_archive_disconnect_pool(
        self, app: MusicBotApp
    ) -> None:
        """Two independent constraints, asserted together because they compose:
        the drainer's final drain reads the outbox and writes Postgres, so it
        needs both alive; super().close() can still dispatch events whose
        cleanup() writes Redis, so it must run before the pool closes or a clean
        shutdown leaves state looking like a crash.
        """
        order: list[str] = []

        async def record_super_close() -> None:
            order.append("disconnect")

        async def record_redis_close(_pool: object) -> None:
            order.append("pool")

        app._archive_tier = self._tier(
            AsyncMock(side_effect=lambda: order.append("tier"))
        )
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=record_redis_close),
            patch.object(
                commands.AutoShardedBot, "close", new=lambda self: record_super_close()
            ),
        ):
            await app.close()
        assert order == ["tier", "disconnect", "pool"]

    async def test_teardown_completes_when_a_step_raises(
        self, app: MusicBotApp
    ) -> None:
        """regression: the archive steps were unguarded and can raise (a hung
        Postgres, a drainer task that died). Because _teardown_started is already
        set the retry path short-circuits, so every step after the raiser was
        skipped permanently: Redis pool left open, discord.py never closed, the
        yt-dlp pool left to its 61s atexit join, and no spans flushed — which hid
        the very failure that caused it. The tier guards its own three; this is
        the guard around the tier itself.
        """
        app._archive_tier = self._tier(
            AsyncMock(side_effect=RuntimeError("tier is wedged"))
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
        # that a step follows it: no participant may skip a later one.
        app._archive_tier = self._tier()
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
        # discord.py calls close() from run()'s finally as well as on demand, and
        # its own idempotence check lives inside super().close(). Without the
        # reentrancy guard a second close() re-runs the tier's final drain, and
        # two concurrent drains each peek → insert → retire.
        tier = self._tier()
        app._archive_tier = tier
        app._redis_pool = MagicMock()
        with (
            patch("src.main.close_redis_pool", new=AsyncMock()) as mock_pool_close,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()) as sup,
        ):
            await app.close()
            await app.close()
        assert tier.aclose.await_count == 1
        mock_pool_close.assert_awaited_once()
        # super().close() still runs on the second call — discord.py's own
        # teardown is idempotent and expects to be reached.
        assert sup.await_count == 2

    async def test_concurrent_closes_teardown_once(self, app: MusicBotApp) -> None:
        tier = self._tier()
        app._archive_tier = tier
        app._redis_pool = None
        with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
            await asyncio.gather(app.close(), app.close())
        assert tier.aclose.await_count == 1

    async def test_shuts_down_the_ytdlp_pool(self, app: MusicBotApp) -> None:
        """The extraction workers are child processes — a clean close must join them
        rather than leave them orphaned. Asserts the real pool's state rather than a
        mock call: the pool close() reaches is the one conftest installed."""
        import src.youtube as youtube

        app._redis_pool = None
        with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
            await app.close()
        assert youtube.ytdlp_pool.is_closed

    async def test_closes_the_stream_probe_session(self, app: MusicBotApp) -> None:
        """The probe session lives for the life of the process, so close() is the
        only thing that releases it. Asserts the real module global rather than a
        mock call: the conftest fixture closes sessions after every test, so a
        close() that stopped calling it would otherwise leave the suite green."""
        import src.youtube as youtube

        session = youtube._get_probe_session()
        app._redis_pool = None
        with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
            await app.close()

        assert session.closed
        assert youtube._probe_session is None

    async def test_a_failing_probe_session_close_does_not_skip_telemetry(
        self, app: MusicBotApp
    ) -> None:
        """close_probe_session() no longer swallows its own errors, so the call
        site is the only guard — and if it were missing, a socket already gone
        would cost the span flush, the record of the failed shutdown."""
        app._redis_pool = None
        with patch(
            "src.youtube.close_probe_session",
            new=AsyncMock(side_effect=OSError("already gone")),
        ):
            with patch("src.telemetry.shutdown_telemetry") as shutdown:
                with patch.object(commands.AutoShardedBot, "close", new=AsyncMock()):
                    await app.close()  # must not raise

        shutdown.assert_called_once()

    async def test_a_failing_chart_pool_close_does_not_cost_the_span_flush(
        self, app: MusicBotApp
    ) -> None:
        """Every step in close() is individually guarded because a hung Postgres once
        made archive.close() raise, and _teardown_started short-circuits the retry —
        so every step after the raiser was skipped for good. The chart pool sits
        between the yt-dlp pool and the probe session; a raise here would cost the
        span flush, which is the record of the failed shutdown."""
        app._redis_pool = None
        with (
            patch(
                "src.chart_pool.chart_pool.aclose",
                new=AsyncMock(side_effect=RuntimeError("join wedged")),
            ),
            patch("src.telemetry.shutdown_telemetry") as shutdown,
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()  # must not raise

        shutdown.assert_called_once()

    async def test_the_chart_pool_is_closed_on_the_way_down(
        self, app: MusicBotApp
    ) -> None:
        """It is never spawned on a bot that never ran -analytics, in which case this
        only flips the closed flag — but an unclosed one that WAS spawned leaves a
        worker to the 61s atexit join aclose() exists to bound."""
        app._redis_pool = None
        with (
            patch("src.chart_pool.chart_pool.aclose", new=AsyncMock()) as aclose,
            patch("src.telemetry.shutdown_telemetry"),
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        aclose.assert_awaited_once()


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

    async def test_the_help_path_borrows_a_span(
        self, app: MusicBotApp, music_bot: MusicBot
    ) -> None:
        """MusicContext.send reads cog._active_spans[id(ctx)] for the trace id and
        elapsed time. This path skips dispatch, so cog_before_invoke never opens
        one."""
        ctx = self._ctx("-play --help")
        seen: dict[str, object] = {}

        async def _capture(_command: object) -> None:
            seen["active"] = music_bot._active_spans.get(id(ctx))

        ctx.send_help = AsyncMock(side_effect=_capture)
        with patch.object(app, "get_cog", return_value=music_bot):
            await app.invoke(ctx)

        assert seen["active"] is not None
        assert music_bot._active_spans == {}  # and closed on the way out

    async def test_the_bare_help_command_borrows_one_too(
        self, app: MusicBotApp, music_bot: MusicBot
    ) -> None:
        """`-help` dispatches normally, but discord.py owns it rather than the cog,
        so the hooks never fire."""
        ctx = self._ctx("-help")
        ctx.command.cog = None
        seen: dict[str, object] = {}

        async def _invoke(_ctx: object) -> None:
            seen["active"] = music_bot._active_spans.get(id(ctx))

        with (
            patch.object(app, "get_cog", return_value=music_bot),
            patch.object(
                commands.AutoShardedBot, "invoke", new=AsyncMock(side_effect=_invoke)
            ),
        ):
            await app.invoke(ctx)

        assert seen["active"] is not None
        assert music_bot._active_spans == {}

    async def test_an_ordinary_command_is_left_to_the_cog_hooks(
        self, app: MusicBotApp, music_bot: MusicBot
    ) -> None:
        """Double-opening would leak: cog_before_invoke writes the same key and
        cog_after_invoke pops it once."""
        ctx = self._ctx("-play lofi")
        seen: dict[str, object] = {}

        async def _invoke(_ctx: object) -> None:
            seen["active"] = music_bot._active_spans.get(id(ctx))

        with (
            patch.object(app, "get_cog", return_value=music_bot),
            patch.object(
                commands.AutoShardedBot, "invoke", new=AsyncMock(side_effect=_invoke)
            ),
        ):
            await app.invoke(ctx)

        assert seen["active"] is None

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


class TestCommandNotFound:
    """Unknown commands are dropped without a log line; everything else keeps
    discord.py's handling. The prefix is a bare `-` with strip_after_prefix, so a
    markdown bullet ("- milk") dispatches CommandNotFound for `milk`, which the
    default handler logs at ERROR with a traceback."""

    def _ctx(self, invoked_with: str) -> MagicMock:
        ctx = MagicMock()
        ctx.invoked_with = invoked_with
        return ctx

    async def test_unknown_command_is_dropped(self, app: MusicBotApp) -> None:
        with patch.object(
            commands.AutoShardedBot, "on_command_error", new=AsyncMock()
        ) as mock_super:
            await app.on_command_error(
                self._ctx("milk"),
                commands.CommandNotFound('Command "milk" is not found'),
            )
        mock_super.assert_not_awaited()

    async def test_every_other_error_is_delegated(self, app: MusicBotApp) -> None:
        """The guard is one isinstance, not a blanket swallow: anything else must
        reach the default, whose command/cog checks are what stop
        MusicBot.cog_command_error's errors being logged a second time."""
        ctx = self._ctx("play")
        error = commands.CheckFailure("nope")
        with patch.object(
            commands.AutoShardedBot, "on_command_error", new=AsyncMock()
        ) as mock_super:
            await app.on_command_error(ctx, error)
        mock_super.assert_awaited_once_with(ctx, error)

    async def test_the_logged_token_is_bounded(self, app: MusicBotApp) -> None:
        """invoked_with is one whitespace-free token and nothing caps its length —
        a 2,000-character dash-prefixed message must not log whole."""
        with (
            patch.object(commands.AutoShardedBot, "on_command_error", new=AsyncMock()),
            patch("src.main.log") as mock_log,
        ):
            await app.on_command_error(
                self._ctx("x" * 2000), commands.CommandNotFound("...")
            )
        logged = mock_log.debug.call_args.args[0]
        # The cap itself, not a bound loose enough to survive widening it.
        assert "x" * 32 in logged
        assert "x" * 33 not in logged

    @pytest.mark.parametrize("token", ["milk", "pn", "playnow", "paly"])
    async def test_no_token_earns_a_reply(self, app: MusicBotApp, token: str) -> None:
        """Dropping is silent for the user too — no token gets a did-you-mean. A
        reply here is reachable from ordinary chat: with strip_after_prefix, a
        bullet reading "- pn" lands in exactly this branch."""
        ctx = self._ctx(token)
        with patch.object(commands.AutoShardedBot, "on_command_error", new=AsyncMock()):
            await app.on_command_error(ctx, commands.CommandNotFound("..."))
        ctx.send.assert_not_called()


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


class TestLivenessHeartbeat:
    """`restart: always` only sees the process exit, so a wedged event loop
    stays "up" while answering nothing. The touch is what makes that visible to
    the container HEALTHCHECK."""

    async def test_touches_the_file_on_each_tick(
        self, app: MusicBotApp, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "bot-alive"
        monkeypatch.setattr(config, "LIVENESS_FILE", str(target))

        async def _sleep(_s: Any) -> None:
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", new=_sleep):
            with pytest.raises(asyncio.CancelledError):
                await app._liveness_heartbeat()

        assert target.exists()

    async def test_unwritable_path_does_not_kill_the_bot(
        self, app: MusicBotApp, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unwritable path degrades to no liveness signal; the healthcheck
        fails on its own rather than the process dying over a touch."""
        monkeypatch.setattr(
            config, "LIVENESS_FILE", str(tmp_path / "nonexistent-dir" / "f")
        )

        async def _sleep(_s: Any) -> None:
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", new=_sleep):
            with pytest.raises(asyncio.CancelledError):
                await app._liveness_heartbeat()  # must not raise OSError

    async def test_setup_hook_skips_the_task_when_unconfigured(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset outside Docker; nothing reads the file there. Driven through
        the disabled arm so this stays a liveness test rather than a second
        assertion about the archive's Postgres requirement."""
        monkeypatch.setattr(config, "LIVENESS_FILE", "")
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.archive_tier.outbox_depth", new=AsyncMock(return_value=0)),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert getattr(app, "_liveness_task", None) is None

    async def test_close_cancels_the_task(self, app: MusicBotApp) -> None:
        async def _forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_forever())
        app._liveness_task = task
        app._redis_pool = None
        with (
            patch("src.telemetry.shutdown_telemetry"),
            patch.object(commands.AutoShardedBot, "close", new=AsyncMock()),
        ):
            await app.close()
        assert task.cancelled() or task.cancelling()
        assert app._liveness_task is None


class TestIntents:
    """The declared gateway contract. Each assertion matches code that stops
    working if the flag is dropped, and none of it fails anywhere but at
    runtime, against Discord."""

    def test_only_the_needed_intents_are_requested(self) -> None:
        assert {f for f, v in intents if v} == {
            "guilds",
            "voice_states",
            "guild_messages",
            "dm_messages",
            "message_content",
            "members",
        }

    def test_presences_is_not_requested(self) -> None:
        """Privileged, and blocks verification past 100 guilds. Sending our own
        presence through change_presence() needs no intent, so nothing here
        wants it."""
        assert intents.presences is False

    def test_message_events_are_received(self) -> None:
        """message_content alone is NOT enough: without guild_messages the
        events never arrive and no prefix command works at all."""
        assert intents.guild_messages is True
        assert intents.message_content is True

    def test_dm_messages_are_received(self) -> None:
        """-help renders a DM-safe embed and -debug has a reply written for the
        no-guild case (test_debug.py: "-debug is DM-reachable"). Dropping this
        leaves both unreachable in production with every test still green."""
        assert intents.dm_messages is True


class TestChartPoolWarm:
    """setup_hook warms the chart worker so the first -analytics does not pay
    matplotlib's import in front of a user — but only where the command is reachable."""

    async def test_the_chart_pool_is_warmed_when_the_archive_is_on(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "true")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@h:5432/d")
        with (
            patch("src.chart_pool.warm") as warm,
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.main.start_archive_tier", new=AsyncMock(return_value=None)),
            patch("src.main.create_redis_pool"),
            patch("src.youtube.ytdlp_pool.prewarm"),
        ):
            await app.setup_hook()
        warm.assert_called_once()

    async def test_a_default_deployment_never_spawns_a_chart_worker(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """-analytics is archive-gated, so a bot with the archive off can never reach
        that pool. Paying ~60MB for it would be pure waste, and the archive is OFF by
        default — this is the narrowing that makes warming affordable at all."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        with (
            patch("src.chart_pool.warm") as warm,
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.main.start_archive_tier", new=AsyncMock(return_value=None)),
            patch("src.main.create_redis_pool"),
            patch("src.youtube.ytdlp_pool.prewarm"),
        ):
            await app.setup_hook()
        warm.assert_not_called()

    async def test_the_chart_pool_is_warmed_after_the_ytdlp_pool(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order is load-bearing: yt-dlp's prewarm is what brings the forkserver up and
        pays the entry-module import in it, so warming after costs a ~21ms fork rather
        than repeating ~1.7s of imports."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "true")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@h:5432/d")
        order: list[str] = []
        with (
            patch("src.chart_pool.warm", side_effect=lambda: order.append("chart")),
            patch(
                "src.youtube.ytdlp_pool.prewarm",
                side_effect=lambda *a, **k: order.append("ytdlp"),
            ),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.main.start_archive_tier", new=AsyncMock(return_value=None)),
            patch("src.main.create_redis_pool"),
        ):
            await app.setup_hook()
        assert order == ["ytdlp", "chart"]

    async def test_slim_image_with_the_archive_on_warns_at_startup(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The slim image pairs a working -analytics with no chart. Nothing else says
        so at startup: warm() returns silently and the only other signal is a
        per-invocation line, so the operator sees a chartless card and reads it as a
        render failure."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "true")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@h:5432/d")
        with (
            patch("src.chart_pool.chart_available", return_value=False),
            patch("src.chart_pool.warm"),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.main.start_archive_tier", new=AsyncMock(return_value=None)),
            patch("src.main.create_redis_pool"),
            patch("src.youtube.ytdlp_pool.prewarm"),
            patch("src.main.log.warning") as warn,
        ):
            await app.setup_hook()
        assert any("matplotlib is not installed" in str(c) for c in warn.call_args_list)

    async def test_a_full_image_warns_about_nothing(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pairing above is the ONLY thing that warning describes — firing it on a
        normal deployment would train the operator to ignore it."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "true")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://u:p@h:5432/d")
        with (
            patch("src.chart_pool.chart_available", return_value=True),
            patch("src.chart_pool.warm"),
            patch.object(app, "load_extension", new=AsyncMock()),
            patch("src.main.start_archive_tier", new=AsyncMock(return_value=None)),
            patch("src.main.create_redis_pool"),
            patch("src.youtube.ytdlp_pool.prewarm"),
            patch("src.main.log.warning") as warn,
        ):
            await app.setup_hook()
        assert not any("matplotlib" in str(c) for c in warn.call_args_list), (
            warn.call_args_list
        )


class TestChartPoolWarmCallable:
    def test_warm_does_nothing_without_matplotlib(self) -> None:
        """No worker is spawned to discover the import will fail — that would leave a
        ~60MB process resident for nothing."""
        import src.chart_pool as cp

        with (
            patch.object(cp, "chart_available", return_value=False),
            patch.object(cp.chart_pool, "prewarm") as prewarm,
        ):
            cp.warm()
        prewarm.assert_not_called()

    def test_warm_submits_the_matplotlib_importing_callable(self) -> None:
        """The default no-op would spawn the worker and warm nothing that matters:
        matplotlib is imported inside render_dashboard, so a no-op never touches it."""
        import src.chart_pool as cp

        with (
            patch.object(cp, "chart_available", return_value=True),
            patch.object(cp.chart_pool, "prewarm") as prewarm,
        ):
            cp.warm()
        prewarm.assert_called_once_with(cp._warm_worker)

    def test_the_warm_callable_is_importable_by_qualified_name(self) -> None:
        """It is pickled to the worker by name, so it must be module-level — a closure
        or a local would fail at submit time, on the startup path."""
        import pickle

        import src.chart_pool as cp

        assert pickle.loads(pickle.dumps(cp._warm_worker)) is cp._warm_worker
