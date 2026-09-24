"""Tests for src/archive_tier.py — starting and stopping the history archive.

Driven through `start_archive_tier` rather than `setup_hook`: the tier is what
these assert about, and main.py's own wiring (that it reads the flag first, that
a raising tier aborts before the cogs load) stays in test_main.py.
"""

import asyncio
from collections.abc import Iterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src import archive_tier
from src.archive_tier import ArchiveTier, start_archive_tier, verify_reachable
from src.redis_client import HISTORY_OUTBOX_KEY

UNREACHABLE = [
    RedisConnectionError("Error 111 connecting to localhost:6379"),
    RedisTimeoutError("Timeout connecting to server"),
]
UNREACHABLE_IDS = ["connection-refused", "timeout"]


@pytest.fixture(autouse=True)
def quiet_probe() -> Iterator[AsyncMock]:
    """The enabled arm spawns a reachability probe against whatever archive it
    built — a MagicMock in most tests here, which would leave a task sleeping
    between attempts. TestReachabilityProbe shadows this to drive the real one."""
    stub = AsyncMock()
    with patch.object(archive_tier, "verify_reachable", new=stub):
        yield stub


class TestDisabledArm:
    """The SHIP default, inverted from the suite-wide conftest pin per test here.
    No Postgres requirement, no archive or drainer, no consumer group, and the
    leftover-outbox probe in place of the enabled arm's fail-fast pair."""

    async def test_it_builds_no_tier(
        self, fake_redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline behavior: a token-only environment boots. The enabled
        arm's RuntimeError is the consent trade — no archive, no requirement."""
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert await start_archive_tier(fake_redis, enabled=False) is None

    async def test_a_present_postgres_url_is_ignored_with_a_note(
        self,
        fake_redis: aioredis.Redis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The flag wins over URL presence: compose interpolates POSTGRES_URL
        whether or not the archive profile is active, so a DSN is not consent.
        One INFO tells the operator who expected archiving why there is none."""
        caplog.set_level("INFO")
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")
        with (
            patch("src.archive_tier.PostgresHistoryArchive") as mock_pg,
            patch("src.archive_tier.HistoryOutboxDrainer") as mock_dr,
        ):
            assert await start_archive_tier(fake_redis, enabled=False) is None
        mock_pg.assert_not_called()
        mock_dr.assert_not_called()
        assert "POSTGRES_URL is set but ignored" in caplog.text

    async def test_no_outbox_group_or_key_is_created(
        self, fake_redis: aioredis.Redis
    ) -> None:
        """ensure_outbox_group is real and simply not called: creating the
        group would MKSTREAM the non-evictable key into existence, which is
        collection infrastructure the operator declined. EXISTS, not XLEN —
        the key must not exist at all."""
        await start_archive_tier(fake_redis, enabled=False)
        assert await fake_redis.exists(HISTORY_OUTBOX_KEY) == 0

    async def test_no_default_password_warning_when_disabled(
        self,
        fake_redis: aioredis.Redis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no Postgres deployed, a credential warning about it is noise
        that trains operators to ignore warnings — the startup ERROR gates on
        the flag (as do the -ping surfaces)."""
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        await start_archive_tier(fake_redis, enabled=False)
        assert "still the default" not in caplog.text


class TestLeftoverOutbox:
    """A previously-enabled archive left entries behind and the bot now starts
    disabled. Warn — never auto-delete (an accidental toggle must not destroy
    un-archived plays), never stay silent (a non-evictable key holding exactly
    the data the operator opted out of keeping must not linger invisibly)."""

    async def test_warns_with_the_depth_when_entries_remain(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        await fake_redis.xadd(HISTORY_OUTBOX_KEY, {b"e": b"play"})
        await fake_redis.xadd(HISTORY_OUTBOX_KEY, {b"e": b"play"})
        await start_archive_tier(fake_redis, enabled=False)
        assert "2 entries" in caplog.text
        assert "NEVER drain" in caplog.text
        # Both remedies named: draining them and discarding them are operator
        # decisions, and the warning is only useful if it says what to do.
        assert "Re-enable" in caplog.text
        assert "DEL history:outbox" in caplog.text

    async def test_an_empty_outbox_is_silent(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        await start_archive_tier(fake_redis, enabled=False)
        assert "history:outbox" not in caplog.text

    @pytest.mark.parametrize("exc", UNREACHABLE, ids=UNREACHABLE_IDS)
    async def test_an_unreachable_redis_skips_the_probe(
        self, caplog: pytest.LogCaptureFixture, exc: Exception
    ) -> None:
        """Same rule as the enabled arm's group probe: a Redis blip must not
        stop the music. The probe reuses the RAISING outbox_depth helper and
        handles the error here — golden rule 5's split, unchanged."""
        with patch("src.archive_tier.outbox_depth", new=AsyncMock(side_effect=exc)):
            await start_archive_tier(MagicMock(), enabled=False)
        assert "could not reach Redis" in caplog.text

    async def test_a_wrongtype_key_downgrades_to_a_warning(
        self, fake_redis: aioredis.Redis, caplog: pytest.LogCaptureFixture
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
        await start_archive_tier(fake_redis, enabled=False)
        assert "not a stream" in caplog.text


class TestEnabledArm:
    """The opt-in arm: the DSN it requires, the group it creates before anything
    can write, and what it does when Redis is unreachable versus mis-shaped."""

    @pytest.fixture(autouse=True)
    def postgres_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_URL", "postgresql://x")

    async def test_a_missing_dsn_refuses_to_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail fast: a bot silently running without the archive would XADD every
        song-end onto an outbox nobody drains."""
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        with pytest.raises(RuntimeError, match="POSTGRES_URL is not set"):
            await start_archive_tier(MagicMock(), enabled=True)

    async def test_a_wrongtype_outbox_aborts_before_anything_starts(self) -> None:
        wrongtype = ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )
        with (
            patch(
                "src.archive_tier.ensure_outbox_group",
                new=AsyncMock(side_effect=wrongtype),
            ),
            patch("src.archive_tier.PostgresHistoryArchive") as mock_pg,
            patch("src.archive_tier.HistoryOutboxDrainer") as mock_dr,
            pytest.raises(ResponseError, match="WRONGTYPE"),
        ):
            await start_archive_tier(MagicMock(), enabled=True)
        # A drainer started against a mis-typed key would raise on every read.
        mock_pg.assert_not_called()
        mock_dr.assert_not_called()

    async def test_the_group_exists_before_the_drainer_starts(self) -> None:
        """Order, not just occurrence: a drainer started first would XREADGROUP
        against a group that does not exist yet and burn its NOGROUP
        heal-once on a plain startup race."""
        order: list[str] = []
        ensure = AsyncMock(side_effect=lambda *_: order.append("group"))
        drainer = MagicMock()
        drainer.start = MagicMock(side_effect=lambda: order.append("drainer"))
        redis = MagicMock()
        with (
            patch("src.archive_tier.ensure_outbox_group", new=ensure),
            patch("src.archive_tier.PostgresHistoryArchive"),
            patch("src.archive_tier.HistoryOutboxDrainer", return_value=drainer),
        ):
            await start_archive_tier(redis, enabled=True)
        assert order == ["group", "drainer"]
        ensure.assert_awaited_once_with(redis)

    @pytest.mark.parametrize("exc", UNREACHABLE, ids=UNREACHABLE_IDS)
    async def test_unreachable_redis_degrades_instead_of_aborting(
        self, exc: Exception
    ) -> None:
        """The other side of the WRONGTYPE abort, and why the two are told apart.
        Redis being DOWN is what this bot survives everywhere else, so a fatal
        probe would turn a blip during a deploy into a bot that refuses to boot.
        Safe because _read_batch heals NOGROUP by calling ensure_outbox_group
        itself, so the drainer creates the group once Redis returns."""
        drainer = MagicMock()
        with (
            patch(
                "src.archive_tier.ensure_outbox_group", new=AsyncMock(side_effect=exc)
            ),
            patch("src.archive_tier.PostgresHistoryArchive") as mock_pg,
            patch("src.archive_tier.HistoryOutboxDrainer", return_value=drainer),
        ):
            tier = await start_archive_tier(MagicMock(), enabled=True)
        # Everything downstream still came up — that is the whole claim.
        assert tier is not None
        mock_pg.assert_called_once()
        drainer.start.assert_called_once()

    async def test_the_default_password_logs_an_error_but_starts(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Loud, not fatal. compose defaults the password so `docker compose up`
        works with only a Discord token — refusing to start would put the
        first-run cliff straight back, so the tier runs and complains instead."""
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://musicbot:password@127.0.0.1:5432/musicbot"
        )
        with (
            patch("src.archive_tier.ensure_outbox_group", new=AsyncMock()),
            patch("src.archive_tier.PostgresHistoryArchive"),
            patch("src.archive_tier.HistoryOutboxDrainer"),
        ):
            tier = await start_archive_tier(MagicMock(), enabled=True)
        assert tier is not None
        assert "still the default" in caplog.text

    async def test_it_spawns_the_probe_against_the_archive_it_built(
        self, quiet_probe: AsyncMock
    ) -> None:
        with (
            patch("src.archive_tier.ensure_outbox_group", new=AsyncMock()),
            patch("src.archive_tier.PostgresHistoryArchive") as mock_pg,
            patch("src.archive_tier.HistoryOutboxDrainer"),
        ):
            tier = await start_archive_tier(MagicMock(), enabled=True)
            assert tier is not None
            await tier.probe
        quiet_probe.assert_awaited_once_with(mock_pg.return_value)


class TestReachabilityProbe:
    """A bare `docker compose up` never activates the `archive` profile, so no
    Postgres is deployed — but compose interpolates POSTGRES_URL before profile
    filtering, so the required-DSN check above still passes and the lazy pool
    does not connect until the first song end. Every play until then XADDs onto
    history:outbox, which carries no TTL and cannot be evicted."""

    @pytest.fixture(autouse=True)
    def quiet_probe(self) -> Iterator[None]:
        """Shadow the file-wide stub: these drive the real coroutine."""
        yield None

    @staticmethod
    def _archive(health_check: Any) -> MagicMock:
        archive = MagicMock()
        archive.health_check = health_check
        return archive

    async def test_a_reachable_database_logs_one_info_and_stops(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        health = AsyncMock()
        with caplog.at_level("INFO"):
            await verify_reachable(self._archive(health))
        assert health.await_count == 1
        assert "the archive is live" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    async def test_it_retries_until_the_database_answers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A cold `up` starts the bot seconds before Postgres passes its
        healthcheck, so the first attempts failing is the NORMAL case and must
        not produce the error."""
        health = AsyncMock(side_effect=[OSError("refused"), OSError("refused"), None])
        with caplog.at_level("INFO"), patch("asyncio.sleep", new=AsyncMock()):
            await verify_reachable(self._archive(health))
        assert health.await_count == 3
        assert "the archive is live" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    async def test_an_unreachable_database_logs_one_error_after_every_attempt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        health = AsyncMock(side_effect=OSError("no route"))
        with caplog.at_level("INFO"), patch("asyncio.sleep", new=AsyncMock()):
            await verify_reachable(self._archive(health))
        assert health.await_count == archive_tier._PROBE_ATTEMPTS
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert "the archive is live" not in caplog.text

    async def test_the_error_names_the_outbox_and_the_way_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The operator reading this line has a bot that looks healthy. It has to
        say what is accumulating and which command deploys the database, or it
        sends them looking at Discord."""
        health = AsyncMock(side_effect=OSError("no route"))
        with caplog.at_level("ERROR"), patch("asyncio.sleep", new=AsyncMock()):
            await verify_reachable(self._archive(health))
        message = caplog.text
        assert "history:outbox" in message
        assert "archive" in message and "profile" in message
        assert "just up" in message
        # The failure itself, not just the diagnosis: a wrong password and an
        # absent container read identically without it.
        assert "OSError" in message and "no route" in message

    async def test_one_hanging_attempt_does_not_consume_the_whole_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """health_check bounds its own connect, but a route that blackholes packets
        can outlast it; without the per-attempt cap the loop never reaches its
        report."""
        monkeypatch.setattr(archive_tier, "_PROBE_STEP_TIMEOUT_SECS", 0.01)
        monkeypatch.setattr(archive_tier, "_PROBE_ATTEMPTS", 1)

        async def _hang() -> None:
            await asyncio.Event().wait()  # never set

        async with asyncio.timeout(5):
            await verify_reachable(self._archive(_hang))

    async def test_it_never_raises_out(self) -> None:
        """Nothing awaits the task, so an escaping exception would surface only as
        an asyncio "exception was never retrieved" at garbage-collection time."""
        health = AsyncMock(side_effect=RuntimeError("unexpected"))
        with patch("asyncio.sleep", new=AsyncMock()):
            await verify_reachable(self._archive(health))

    async def test_a_cancellation_is_not_swallowed(self) -> None:
        """aclose() cancels the probe; swallowing that would mark the task complete
        while it was still between attempts."""
        started = asyncio.Event()

        async def _fail() -> None:
            started.set()
            raise OSError("refused")

        task = asyncio.create_task(verify_reachable(self._archive(_fail)))
        await started.wait()
        await asyncio.sleep(0)  # let it reach the inter-attempt sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()


class TestTeardown:
    """`aclose()` owns the order the three come down in. Before this was one
    object the sequence lived in MusicBotApp.close(), interleaved with Redis and
    discord.py steps, held together by nothing."""

    @staticmethod
    def _tier(
        *,
        probe: Optional[asyncio.Task[None]] = None,
        drainer: Optional[MagicMock] = None,
        archive: Optional[MagicMock] = None,
    ) -> ArchiveTier:
        async def _idle() -> None:
            await asyncio.Event().wait()

        return ArchiveTier(
            archive=archive or MagicMock(close=AsyncMock()),
            drainer=drainer or MagicMock(stop=AsyncMock()),
            probe=probe or asyncio.create_task(_idle()),
        )

    async def test_it_cancels_the_probe_before_closing_the_archive(self) -> None:
        """Ordering, not merely presence: the probe reads the archive's pool, and
        _ensure() refuses once close() has latched it shut, so a probe outliving
        the archive spends its remaining attempts failing for the wrong reason."""
        order: list[str] = []

        async def _probe() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order.append("probe")
                raise

        archive = MagicMock(
            close=AsyncMock(side_effect=lambda: order.append("archive"))
        )
        drainer = MagicMock(stop=AsyncMock(side_effect=lambda: order.append("drainer")))
        tier = self._tier(
            probe=asyncio.create_task(_probe()), drainer=drainer, archive=archive
        )
        await asyncio.sleep(0)
        await tier.aclose()
        assert order == ["probe", "drainer", "archive"]

    @pytest.mark.parametrize("sick", ["probe", "drainer", "archive"], ids=str)
    async def test_one_sick_participant_does_not_skip_the_others(
        self, sick: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A hung Postgres once made archive.close() raise after 30s. Each step is
        guarded separately so the ones after it still run."""

        async def _boom() -> None:
            raise RuntimeError("sick")

        async def _probe_body() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if sick == "probe":
                    raise RuntimeError("sick") from None
                raise

        drainer = MagicMock(stop=_boom if sick == "drainer" else AsyncMock())
        archive = MagicMock(close=_boom if sick == "archive" else AsyncMock())
        tier = self._tier(
            probe=asyncio.create_task(_probe_body()), drainer=drainer, archive=archive
        )
        await asyncio.sleep(0)
        await tier.aclose()  # must not raise
        if sick != "drainer":
            drainer.stop.assert_awaited_once()
        if sick != "archive":
            archive.close.assert_awaited_once()
        assert "shutdown failed" in caplog.text

    async def test_a_settled_probe_is_not_an_error(self) -> None:
        """The common case: the probe reported minutes ago and the task is done.
        cancel_task no-ops on it."""
        done: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))
        await done
        drainer = MagicMock(stop=AsyncMock())
        archive = MagicMock(close=AsyncMock())
        await self._tier(probe=done, drainer=drainer, archive=archive).aclose()
        drainer.stop.assert_awaited_once()
        archive.close.assert_awaited_once()
