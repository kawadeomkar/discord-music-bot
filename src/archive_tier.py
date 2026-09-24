"""Starting and stopping the history-archive tier: the enabled arm's two checks,
the outbox consumer group, the archive + drainer + reachability probe, and the
order those unwind in.

`main.py` decides WHETHER to run the tier — it reads the flag first, before
anything else can consume it — and this decides what running it means. The
disabled arm lives here too, because "no Postgres is deployed" is a state of the
tier rather than an absence of one, and it has its own reporting to do.
See docs/ARCHITECTURE.md#history-archive-tier.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src import config
from src.history_archive import HistoryOutboxDrainer, PostgresHistoryArchive
from src.redis_client import HISTORY_CACHE_LIMIT, ensure_outbox_group, outbox_depth
from src.util import cancel_task, get_logger

if TYPE_CHECKING:
    import redis.asyncio as aioredis

log = get_logger(__name__)

# The reachability probe. 12 x 5s outlasts a cold `up`, where the bot is ready
# seconds before Postgres passes its healthcheck (interval 5s), so the error
# fires only for a database that is not coming at all.
_PROBE_ATTEMPTS = 12
_PROBE_INTERVAL_SECS = 5.0
# Caps one attempt above health_check's own 10s connect bound.
_PROBE_STEP_TIMEOUT_SECS = 15.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ArchiveTier:
    """A running archive tier: the archive, the drainer feeding it, and the
    startup probe watching it. One object, so the three cannot be torn down out
    of order — `aclose()` owns that sequence."""

    archive: PostgresHistoryArchive
    drainer: HistoryOutboxDrainer
    probe: asyncio.Task[None]

    async def aclose(self) -> None:
        """Stop the tier, in the only order that works.

        The probe first: it reads the archive's pool, and `_ensure()` refuses
        once `close()` has latched it shut, so a probe outliving the archive
        spends its remaining attempts failing for a reason nobody asked about.
        Then the drainer, whose final drain needs Redis AND the archive alive.
        Then the archive.

        Each step is guarded separately: a hung Postgres once made
        `archive.close()` raise after 30s, and a step that raises must not take
        the ones after it with it.
        """
        try:
            await cancel_task(self.probe)
        except Exception as e:
            log.warning(f"archive probe shutdown failed: {e}")
        try:
            await self.drainer.stop()
        except Exception as e:
            log.warning(f"history drainer shutdown failed: {e}")
        try:
            await self.archive.close()
        except Exception as e:
            log.warning(f"history archive shutdown failed: {e}")


async def start_archive_tier(
    redis: aioredis.Redis, *, enabled: bool
) -> Optional[ArchiveTier]:
    """The tier for this process, or None when the archive is off (the default).

    `enabled` is passed rather than read here: `setup_hook` reads the flag before
    anything else can consume it, because the parser raises on garbage and the
    next reader would be `@_guild_op`-wrapped `push_history`, which swallows it
    into a warning per song.
    """
    if not enabled:
        await _report_disabled(redis)
        return None
    # Fail fast: a bot silently running without the archive would XADD every
    # song-end onto an outbox nobody drains. The remedy names `just run`
    # because this process reads only the environment.
    postgres_url = config.postgres_url()
    if not postgres_url:
        raise RuntimeError(
            "POSTGRES_URL is not set but HISTORY_ARCHIVE_ENABLED is true — "
            "the enabled archive requires its database. Under docker "
            "compose it is supplied for you; for a local run use `just "
            "run`, which loads .env and derives the URL from it. Otherwise "
            "export POSTGRES_URL yourself "
            "(postgresql://user:password@host:5432/dbname). To run without "
            "the archive instead, remove HISTORY_ARCHIVE_ENABLED."
        )
    if config.using_default_postgres_password():
        _warn_default_password()
    # Create the group before anything can write: push_history is @_guild_op-
    # wrapped, so a WRONGTYPE at history:outbox would be swallowed into one
    # warning per song while every play was lost. Only here is it loud.
    # An UNREACHABLE Redis must not abort startup — the pool connects lazily,
    # and _read_batch heals NOGROUP on its first tick after Redis returns.
    try:
        await ensure_outbox_group(redis)
    except (RedisConnectionError, RedisTimeoutError) as e:
        log.warning(f"outbox group probe could not reach Redis: {e}")
    # Lazy: no connection is made here, so startup never blocks on Postgres.
    archive = PostgresHistoryArchive(postgres_url)
    drainer = HistoryOutboxDrainer(redis, archive)
    drainer.start()
    # A task, not an await: the lazy pool above keeps startup off Postgres and
    # this must too.
    probe = asyncio.create_task(verify_reachable(archive))
    return ArchiveTier(archive=archive, drainer=drainer, probe=probe)


async def verify_reachable(archive: PostgresHistoryArchive) -> None:
    """Retry `health_check` for about a minute — longer where attempts hang
    rather than refuse — then log ONE error naming the cause.

    An enabled archive with no database reachable is otherwise silent until a
    play lands: the DSN is interpolated whether or not the `archive` compose
    profile deployed Postgres behind it, so the required-URL check above passes
    and the lazy pool connects for the first time at the first song end. Until
    then every play XADDs onto the non-evictable `history:outbox`.

    Never raises (nothing awaits it) and never repeats — once plays are moving
    the drainer's backoff loop owns the reporting.
    """
    last_error: Optional[Exception] = None
    started = asyncio.get_running_loop().time()
    for attempt in range(_PROBE_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_PROBE_INTERVAL_SECS)
        try:
            # health_check bounds its own connect at 10s; this caps a route that
            # hangs past it, so one dead attempt cannot swallow the whole run.
            async with asyncio.timeout(_PROBE_STEP_TIMEOUT_SECS):
                await archive.health_check()
        except Exception as e:  # noqa: BLE001 — reported once, below
            last_error = e
            continue
        log.info("History archive probe: Postgres answered, the archive is live")
        return
    # Measured, not computed from the constants: the first attempt does not
    # sleep, and a route that hangs rather than refusing stretches the run by
    # up to the step timeout per attempt. Either arithmetic would print a
    # number the operator's own clock disagrees with.
    waited = round(asyncio.get_running_loop().time() - started)
    log.error(
        f"HISTORY_ARCHIVE_ENABLED is true but Postgres has not answered in "
        f"{waited}s ({type(last_error).__name__}: {last_error}). Every play is "
        "being XADDed onto history:outbox, which carries no TTL and is not an "
        "eviction candidate, and nothing is draining it. If this stack came up "
        "with a bare `docker compose up`, the `archive` profile was never "
        "activated and no Postgres was deployed: bring it up with `just up`, "
        "which derives the profile from the flag. If POSTGRES_URL names an "
        "external database, check that it is reachable from this host."
    )


def _warn_default_password() -> None:
    """Loud but not fatal: compose defaults POSTGRES_PASSWORD so a token-only
    `docker compose up` works."""
    log.error(
        "POSTGRES_PASSWORD is still the default "
        f"({config.DEFAULT_POSTGRES_PASSWORD!r}). The play-history "
        "database accepts it from anything that can reach the host's "
        "published port. Fix it IN THIS ORDER: (1) change the server "
        'itself — `docker compose exec postgres psql -U <user> -c "ALTER '
        "USER <user> PASSWORD '<new>'\"`; (2) put the same value in "
        ".env via `./setup_env.sh --force`; (3) `docker compose up -d` "
        "to recreate the bot with the new DSN. To start clean instead, "
        "drop ONLY the database volume — `docker compose down && docker "
        "volume rm discord-music-bot_postgres-data` — not `down -v`, "
        "which also removes the Redis volume holding plays that are not "
        "durable in Postgres yet. The order matters: this "
        "warning reads the bot's DSN, so doing (2) first silences it "
        "while the database still accepts the old password. And "
        "Postgres reads POSTGRES_PASSWORD only when initializing an "
        "EMPTY data directory, so editing .env alone never changes the "
        "server — it just locks the bot out of its own database."
    )


async def _report_disabled(redis: aioredis.Redis) -> None:
    """The disabled arm (the default): say so once and warn about leftovers.
    No consumer-group creation, which would MKSTREAM the non-evictable
    outbox key into existence."""
    # States what IS retained: guild:{id}:history is PERSISTed, so an
    # opted-out deployment still holds 50 plays per guild indefinitely.
    log.info(
        "History archive disabled (the default; HISTORY_ARCHIVE_ENABLED=true "
        f"opts in). Plays are kept only in the per-guild Redis list behind "
        f"-history — the newest {HISTORY_CACHE_LIMIT} per guild, retained "
        "until deleted (no expiry); nothing is written to Postgres."
    )
    if config.postgres_url():
        # Compose interpolates POSTGRES_URL whether or not the archive
        # profile is active, so a DSN here is not consent.
        log.info(
            "POSTGRES_URL is set but ignored: the archive is enabled by "
            "HISTORY_ARCHIVE_ENABLED=true, never by URL presence."
        )
    await _warn_if_outbox_left_over(redis)


async def _warn_if_outbox_left_over(redis: aioredis.Redis) -> None:
    """One WARNING when a previously-enabled archive left outbox entries in
    a non-evictable key that will never drain. Never auto-deleted. The error
    handler for the raising outbox_depth helper: an unreachable Redis skips
    the probe, and a WRONGTYPE only warns since the XADD leg is off."""
    try:
        depth = await outbox_depth(redis)
    except (RedisConnectionError, RedisTimeoutError) as e:
        log.warning(f"leftover-outbox probe could not reach Redis: {e}")
        return
    except ResponseError as e:
        log.warning(
            f"history:outbox exists but is not a stream ({e}). With the "
            "archive disabled it is inert; `DEL history:outbox` with the "
            "bot stopped clears it."
        )
        return
    if depth > 0:
        log.warning(
            f"history:outbox still holds {depth} entries from a "
            "previously-enabled archive. They were buffered for Postgres, "
            "sit in a non-evictable key, and will NEVER drain while the "
            "archive is disabled. Re-enable HISTORY_ARCHIVE_ENABLED to "
            "drain them into the archive, or discard them with `DEL "
            "history:outbox` (inspect first: `just outbox`)."
        )
