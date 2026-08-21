"""Tests for src/main.py — MusicBotApp lifecycle (setup_hook, close, on_ready)."""

import asyncio
from collections.abc import Iterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
import pytest
import redis.asyncio as aioredis
from discord.ext import commands
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src import config
from src.main import EXTENSIONS, MusicBotApp, intents
from src.musicbot import MusicBot
from src.redis_client import HISTORY_OUTBOX_KEY
from tests.helpers import mocked


@pytest.fixture
def app() -> MusicBotApp:
    """Bypass discord.py __init__; wire up minimal internal state so properties work."""
    instance = MusicBotApp.__new__(MusicBotApp)
    instance._redis_pool = None
    instance.redis = None
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
    conn.intents = MagicMock()
    conn.intents.voice_states = True
    instance._connection = conn
    # latency reads self.ws and returns float('nan') for any falsy value, which
    # is fine for logging. MISSING is discord.py's own "not connected yet"
    # sentinel and is falsy, so it takes that same branch.
    instance.ws = discord.utils.MISSING
    instance.change_presence = AsyncMock()
    return instance


class TestAppInitDefaults:
    """The real __init__, against the real constructor — no fixture in the way.

    Everywhere else the `app` fixture bypasses __init__, and the disabled-mode
    tests below seed both attributes by hand, so `assert app.history_archive is
    None` would otherwise assert against a value the fixture wrote. Reverting the
    two assignments to bare annotations passes every other test in the repo and
    makes the default deployment raise AttributeError on its first -play.
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


class TestSetupHookDisabledArchive:
    """The disabled arm — the SHIP default, inverted from the suite-wide
    conftest pin per test here. No Postgres requirement, no archive/drainer
    objects, no consumer group, and the leftover-outbox probe in place of the
    enabled arm's fail-fast pair."""

    @pytest.fixture(autouse=True)
    def archive_disabled(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flip the flag to the ship default and model the post-__init__ attrs:
        the app fixture bypasses __init__ (which assigns None) and the disabled
        arm deliberately never assigns them, so seed None to make "still None
        after setup_hook" a real assertion. TestAppInitDefaults above is what
        pins that None is genuinely what __init__ produces."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        app.history_archive = None
        app.history_drainer = None

    async def test_starts_without_postgres_url(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline behavior: a token-only environment boots. The enabled
        arm's RuntimeError is the consent trade — no archive, no requirement."""
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.outbox_depth", new=AsyncMock(return_value=0)),
            patch.object(app, "load_extension", new=mock_load),
        ):
            await app.setup_hook()
        assert app.history_archive is None
        assert app.history_drainer is None
        mock_load.assert_awaited()

    async def test_a_present_postgres_url_is_ignored_with_a_note(
        self,
        app: MusicBotApp,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The flag wins over URL presence: compose interpolates POSTGRES_URL
        whether or not the archive profile is active, so a DSN is not consent.
        One INFO tells the operator who expected archiving why there is none."""
        caplog.set_level("INFO")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.outbox_depth", new=AsyncMock(return_value=0)),
            patch("src.main.PostgresHistoryArchive") as mock_pg,
            patch("src.main.HistoryOutboxDrainer") as mock_dr,
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        mock_pg.assert_not_called()
        mock_dr.assert_not_called()
        assert "POSTGRES_URL is set but ignored" in caplog.text

    async def test_no_outbox_group_or_key_is_created(
        self, app: MusicBotApp, fake_redis: aioredis.Redis
    ) -> None:
        """ensure_outbox_group is real and simply not called: creating the
        group would MKSTREAM the non-evictable key into existence, which is
        collection infrastructure the operator declined. EXISTS, not XLEN —
        the key must not exist at all."""
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=fake_redis),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert await fake_redis.exists(HISTORY_OUTBOX_KEY) == 0

    async def test_no_default_password_warning_when_disabled(
        self,
        app: MusicBotApp,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no Postgres deployed, a credential warning about it is noise
        that trains operators to ignore warnings — the startup ERROR gates on
        the flag (as do the -ping surfaces)."""
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.outbox_depth", new=AsyncMock(return_value=0)),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert "still the default" not in caplog.text

    async def test_garbage_flag_aborts_startup_before_anything(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validation placement is the feature: the flag is read first, so a typo
        aborts startup with a named error instead of surfacing inside
        @_guild_op-wrapped push_history, where the ValueError would be swallowed
        into one warning per song while archiving silently stayed off."""
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "maybe")
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()) as mock_pool,
            patch("src.main.get_redis", return_value=MagicMock()),
            patch.object(app, "load_extension", new=mock_load),
            pytest.raises(ValueError, match="HISTORY_ARCHIVE_ENABLED"),
        ):
            await app.setup_hook()
        # Before the Redis pool, before the cogs: nothing is half-started.
        mock_pool.assert_not_called()
        mock_load.assert_not_awaited()


class TestLeftoverOutboxWarning:
    """A previously-enabled archive left entries behind and the bot now starts
    disabled. Warn — never auto-delete (an accidental toggle must not destroy
    un-archived plays), never stay silent (a non-evictable key holding exactly
    the data the operator opted out of keeping must not linger invisibly)."""

    @pytest.fixture(autouse=True)
    def archive_disabled(
        self, app: MusicBotApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same shape as TestSetupHookDisabledArchive's fixture, for the same
        # reason; TestAppInitDefaults pins that None is what __init__ assigns.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        app.history_archive = None
        app.history_drainer = None

    async def test_warns_with_the_depth_when_entries_remain(
        self,
        app: MusicBotApp,
        fake_redis: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await fake_redis.xadd(HISTORY_OUTBOX_KEY, {b"e": b"play"})
        await fake_redis.xadd(HISTORY_OUTBOX_KEY, {b"e": b"play"})
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=fake_redis),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert "2 entries" in caplog.text
        assert "NEVER drain" in caplog.text
        # Both remedies named: draining them and discarding them are operator
        # decisions, and the warning is only useful if it says what to do.
        assert "Re-enable" in caplog.text
        assert "DEL history:outbox" in caplog.text

    async def test_an_empty_outbox_is_silent(
        self,
        app: MusicBotApp,
        fake_redis: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=fake_redis),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert "history:outbox" not in caplog.text

    @pytest.mark.parametrize(
        "exc",
        [
            RedisConnectionError("Error 111 connecting to localhost:6379"),
            RedisTimeoutError("Timeout connecting to server"),
        ],
        ids=["connection-refused", "timeout"],
    )
    async def test_an_unreachable_redis_skips_the_probe(
        self,
        app: MusicBotApp,
        caplog: pytest.LogCaptureFixture,
        exc: Exception,
    ) -> None:
        """Same rule as the enabled arm's group probe: a Redis blip must not
        stop the music. The probe reuses the RAISING outbox_depth helper and
        handles the error here — golden rule 5's split, unchanged."""
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=MagicMock()),
            patch("src.main.outbox_depth", new=AsyncMock(side_effect=exc)),
            patch.object(app, "load_extension", new=mock_load),
        ):
            await app.setup_hook()
        assert "could not reach Redis" in caplog.text
        mock_load.assert_awaited()  # startup still completed

    async def test_a_wrongtype_key_downgrades_to_a_warning(
        self,
        app: MusicBotApp,
        fake_redis: aioredis.Redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The enabled path ABORTS on a mis-shaped outbox; disabled, the XADD leg
        is off and the key is inert, so warn and serve.

        STAGED, not patched: a mocked `outbox_depth(side_effect=ResponseError)`
        proved only that `except ResponseError` catches what is handed to it, and
        stayed green through any change to which helper the probe calls. XLEN
        models the real ResponseError faithfully, so the path runs here honestly.
        """
        # A list at the stream's key: the real leftover shape, from a build
        # predating the switch to a stream outbox.
        await fake_redis.rpush(HISTORY_OUTBOX_KEY, b"legacy-entry")
        mock_load = AsyncMock()
        with (
            patch("src.main.create_redis_pool", return_value=MagicMock()),
            patch("src.main.get_redis", return_value=fake_redis),
            patch.object(app, "load_extension", new=mock_load),
        ):
            await app.setup_hook()
        assert "not a stream" in caplog.text
        mock_load.assert_awaited()  # startup still completed


class TestOutboxGroupBootstrap:
    """The setup_hook leg of ensure_outbox_group — the wiring, not the helper.

    The helper's own discrimination (BUSYGROUP tolerated, WRONGTYPE raised) is
    asserted in test_redis_client and against real Redis; these pin that it runs,
    that it runs before the drainer, and that a WRONGTYPE aborts startup. The abort
    has to happen there: push_history is @_guild_op, so a leftover list would
    otherwise be swallowed into one warning per song."""

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
        # Aborted before the archive tier or the cogs: a drainer started
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
        """The other side of the WRONGTYPE abort, and why the two are told apart.
        Redis being DOWN is what this bot survives everywhere else, so a fatal
        probe would turn a blip during a deploy into a bot that refuses to boot.
        Safe because _read_batch heals NOGROUP by calling ensure_outbox_group
        itself, so the drainer creates the group once Redis returns."""
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


class TestDefaultPostgresPassword:
    """compose falls back to a known password so `docker compose up` works with
    only a Discord token. The bot pays for that convenience with noise: an ERROR
    at every startup and a standing -ping warning, until it is changed.
    """

    async def test_default_postgres_password_logs_an_error_but_starts(
        self,
        app: MusicBotApp,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Loud, not fatal. compose defaults the password so `docker compose up`
        works with only a Discord token — refusing to start would put the
        first-run cliff straight back, so the bot runs and complains instead."""
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        with (
            patch("src.main.create_redis_pool"),
            patch("src.main.get_redis"),
            patch("src.main.ensure_outbox_group", new=AsyncMock()),
            patch("src.main.HistoryOutboxDrainer"),
            patch("src.main.PostgresHistoryArchive"),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert any(r.levelname == "ERROR" for r in caplog.records)
        assert "still the default" in caplog.text
        # The remedy has to be the one that works: an .env edit alone leaves an
        # initialized volume on its old password and locks the bot out.
        assert "ALTER USER" in caplog.text
        assert "up -d" in caplog.text  # the container recreate, or the DSN is stale

        # And in the order that works. This check reads the bot's DSN, so
        # `setup_env.sh --force` silences it while the server still accepts the
        # old password — running that first walks the operator through the one
        # window where they are exposed and nothing says so.
        assert caplog.text.index("ALTER USER") < caplog.text.index("setup_env.sh")
        assert "IN THIS ORDER" in caplog.text

    async def test_a_real_postgres_password_logs_nothing(
        self,
        app: MusicBotApp,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:9f3a1c@127.0.0.1:5432/musicbot"
        )
        with (
            patch("src.main.create_redis_pool"),
            patch("src.main.get_redis"),
            patch("src.main.ensure_outbox_group", new=AsyncMock()),
            patch("src.main.HistoryOutboxDrainer"),
            patch("src.main.PostgresHistoryArchive"),
            patch.object(app, "load_extension", new=AsyncMock()),
        ):
            await app.setup_hook()
        assert "still the default" not in caplog.text


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

    async def test_close_with_the_archive_tier_disabled(self, app: MusicBotApp) -> None:
        """The disabled shape: __init__ assigned None and setup_hook left it
        that way. The `is not None` guards skip the tier's two steps entirely
        and everything downstream still runs — same guards, second meaning
        (the first is surviving a close() before setup_hook ever assigned)."""
        app.history_archive = None
        app.history_drainer = None
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
        """regression: drainer.stop() and archive.close() were both unguarded and
        both can raise (a hung Postgres, a drainer task that died). Because
        _teardown_started is already set the retry path short-circuits, so every
        step after the raiser was skipped permanently: Redis pool left open,
        discord.py never closed, the yt-dlp pool left to its 61s atexit join, and
        no spans flushed — which hid the very failure that caused it.
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
        # that a step follows it: no participant may skip a later one.
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
        # discord.py calls close() from run()'s finally as well as on demand, and
        # its own idempotence check lives inside super().close(). Without the
        # reentrancy guard a second close() re-runs drainer.stop(), and two
        # concurrent final drains each peek → insert → retire.
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
            patch("src.main.outbox_depth", new=AsyncMock(return_value=0)),
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
