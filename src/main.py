import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import discord
from discord.ext import commands
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src import config
from src.config import spotify_enabled

from src.help import MusicHelpCommand
from src.history_archive import HistoryOutboxDrainer, PostgresHistoryArchive
from src.redis_client import (
    HISTORY_CACHE_LIMIT,
    close_redis_pool,
    create_redis_pool,
    ensure_outbox_group,
    get_redis,
    outbox_depth,
)
from src.util import get_logger

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.musicbot import MusicBot
    from src.musicplayer import MusicPlayer

log = get_logger(__name__)


def _build_intents() -> discord.Intents:
    """The gateway events this bot subscribes to. `presences` is absent: it is
    privileged and nothing reads a presence update (`change_presence()` needs
    no intent)."""
    intents = discord.Intents.none()
    # Guild/channel/voice-client cache; on_voice_state_update + VoiceChannel.members.
    intents.guilds = True
    intents.voice_states = True
    # Prefix commands dispatch from on_message; -help and -debug answer DMs.
    intents.guild_messages = True
    intents.dm_messages = True
    # Both privileged. Without message_content no command ever matches; members
    # backs guild.get_member() for queue requesters and VoiceChannel.members.
    intents.message_content = True
    intents.members = True
    return intents


intents = _build_intents()
EXTENSIONS = ("src.musicbot",)


class MusicContext(commands.Context):
    """Context whose send() keeps the Now Playing block at the bottom of the
    channel: responses lead with the NP block, then their own embeds, and the
    previous host is retired. Also the one choke point where every command
    response gets its debug footer."""

    async def send(
        self, content: Optional[str] = None, **kwargs: Any
    ) -> discord.Message:
        # Decoration mutates the caller's embeds in place, so the early return
        # below stays a verbatim pass-through.
        self._decorate_for_debug(kwargs)
        mp = self._np_player()
        if mp is None:
            return await super().send(content, **kwargs)
        embeds_kwarg = kwargs.pop("embeds", None)
        single = kwargs.pop("embed", None)
        if single is not None and embeds_kwarg is not None:
            # discord.py's own send() contract
            raise TypeError("cannot pass both embed and embeds parameter to send()")
        own: list[discord.Embed] = list(embeds_kwarg or [])
        if single is not None:
            own.append(single)
        song = mp.current_song  # the song the block below is built for
        block = mp.np_embed_block()
        # ≤10 is Discord's per-message embed cap (worst case here is 3).
        attached = bool(block) and len(own) + len(block) <= 10
        embeds = block + own if attached else own
        if embeds:
            message = await super().send(content, embeds=embeds, **kwargs)
        else:
            message = await super().send(content, **kwargs)
        if attached:
            # The send may have crossed a song boundary; the gate sheds a stale
            # block from the just-sent message instead of adopting it.
            mp._adopt_np_host_if_current(message, own, song)
        return message

    def _decorate_for_debug(self, kwargs: dict[str, Any]) -> None:
        """Add the debug footer to this response's own embeds. The NP block is
        decorated by the player at build time instead, so the progress tick
        cannot re-render it back to bare. See docs/ARCHITECTURE.md#debug-footer-seams."""
        cog = self._music_cog()
        if cog is None:
            return
        own = [
            e
            for e in (kwargs.get("embed"), *(kwargs.get("embeds") or ()))
            if e is not None
        ]
        if not own:
            return
        # Absent for a send outside any command: no elapsed time, no span.
        active = cog._active_spans.get(id(self))
        cog.debug_settings.decorate(
            own,
            self.guild,
            span=active.span if active is not None else None,
            elapsed_ms=(time.monotonic() - active.started) * 1000
            if active is not None
            else None,
        )

    def _music_cog(self) -> Optional[MusicBot]:
        from src.musicbot import MusicBot

        cog = self.bot.get_cog("MusicBot")
        return cog if isinstance(cog, MusicBot) else None

    def _np_player(self) -> Optional[MusicPlayer]:
        """The guild's MusicPlayer when attaching is appropriate: guild message,
        cog loaded, a song live, and this channel is the player's home channel."""
        if self.guild is None:
            return None
        cog = self._music_cog()
        if cog is None:
            return None
        mp = cog.mps.get(self.guild.id)
        if mp is None or mp.current_song is None:
            return None
        if self.channel.id != mp.home_channel.id:
            return None
        return mp


# Ceiling on the off-loop span flush at close(); the exporter's own timeout is
# 30s, so this fires only when the collector hangs past it.
_TELEMETRY_SHUTDOWN_TIMEOUT = 35.0


# GH #5: AutoShardedBot multi-shards in one process; shard_count=None lets Discord
# assign it. setup_hook is a subclass override — discord.py calls it before connecting.
class MusicBotApp(commands.AutoShardedBot):
    def __init__(self) -> None:
        super().__init__(
            command_prefix="-",
            intents=intents,
            description="Plays YouTube, Spotify and SoundCloud audio in voice channels.",
            strip_after_prefix=True,
            help_command=MusicHelpCommand(),
        )
        self._redis_pool = None
        self.redis = None
        self._liveness_task: Optional[asyncio.Task] = None
        # Built in setup_hook while HISTORY_ARCHIVE_ENABLED; None is the disabled
        # shape (the default) and every consumer handles it.
        self.history_archive: Optional[PostgresHistoryArchive] = None
        self.history_drainer: Optional[HistoryOutboxDrainer] = None
        self._teardown_started = False  # close() runs at most once

    async def _liveness_heartbeat(self) -> None:
        """Touch LIVENESS_FILE on a fixed cadence for the container HEALTHCHECK.
        Dependency-free: it answers only "is the event loop still turning", so
        a Redis or Discord blip cannot mark the bot dead."""
        path = Path(config.LIVENESS_FILE)
        while True:
            try:
                path.touch()
            except OSError as e:
                # Degrades to no signal; the healthcheck fails on its own.
                log.warning(f"Liveness touch failed for {path}: {e}")
            await asyncio.sleep(config.LIVENESS_INTERVAL_SECS)

    async def setup_hook(self) -> None:
        # Read first: the parser raises on garbage, and the next reader would be
        # @_guild_op-wrapped push_history, which swallows it into a warning per song.
        archive_enabled = config.history_archive_enabled()
        # Before the pool and extensions so the file exists in the start-period.
        if config.LIVENESS_FILE:
            self._liveness_task = asyncio.create_task(self._liveness_heartbeat())
        self._redis_pool = create_redis_pool()
        self.redis = get_redis(self._redis_pool)
        if archive_enabled:
            await self._setup_history_archive(self.redis)
        else:
            await self._report_archive_disabled()
        for extension in EXTENSIONS:
            await self.load_extension(extension)
        # Fire-and-forget, so the first -play skips spawn + yt-dlp import latency.
        from src.youtube import ytdlp_pool

        ytdlp_pool.prewarm()
        # The chart worker, only with the archive on, after the prewarm that
        # brings the forkserver up. Fire-and-forget.
        if archive_enabled:
            from src.chart_pool import chart_available, warm as warm_chart_pool

            if not chart_available():
                # The slim image with the archive on. -analytics still answers
                # (the chart is an attachment), and a card without its chart is
                # indistinguishable from a failed render, so warn here.
                log.warning(
                    "matplotlib is not installed, so -analytics will answer without "
                    "its chart — deploy the image tag without the -slim suffix"
                )
            try:
                warm_chart_pool()
            except Exception as e:
                # A raise here would abort startup over an optional feature.
                log.warning(f"chart pool warm failed: {e}")

    async def _setup_history_archive(self, redis: aioredis.Redis) -> None:
        """The enabled arm: required DSN, default-password advisory, outbox
        consumer group, archive + drainer. `redis` is a parameter because the
        checker's narrowing does not cross the method boundary."""
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
        # Loud but not fatal: compose defaults POSTGRES_PASSWORD so a token-only
        # `docker compose up` works.
        if config.using_default_postgres_password():
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
        self.history_archive = archive
        self.history_drainer = HistoryOutboxDrainer(redis, archive)
        self.history_drainer.start()

    async def _report_archive_disabled(self) -> None:
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
        await self._warn_if_outbox_left_over()

    async def _warn_if_outbox_left_over(self) -> None:
        """One WARNING when a previously-enabled archive left outbox entries in
        a non-evictable key that will never drain. Never auto-deleted. The error
        handler for the raising outbox_depth helper: an unreachable Redis skips
        the probe, and a WRONGTYPE only warns since the XADD leg is off."""
        if self.redis is None:
            return
        try:
            depth = await outbox_depth(self.redis)
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

    async def get_context(
        self,
        origin: Union[discord.Message, discord.Interaction],
        /,
        *,
        cls: type[commands.Context[Any]] = MusicContext,
    ) -> commands.Context[Any]:
        # Typed against discord.py's signature: `Any` would make drift uncheckable.
        return await super().get_context(origin, cls=cls)

    async def invoke(self, ctx: commands.Context, /) -> None:
        command = ctx.command
        # `--help` anywhere in the message short-circuits to that command's help
        # embed before checks, the voice gate and argument parsing.
        short_circuit = command is not None and "--help" in ctx.message.content
        # Neither help path reaches cog_before_invoke, so both borrow a span from
        # the cog. See MusicBot.traced_help.
        if short_circuit or (command is not None and command.cog is None):
            from src.musicbot import MusicBot

            cog = self.get_cog("MusicBot")
            span = (
                cog.traced_help(ctx)
                if isinstance(cog, MusicBot)
                else contextlib.nullcontext()
            )
            async with span:
                if short_circuit and command is not None:
                    await ctx.send_help(command)
                else:
                    await super().invoke(ctx)
            return
        await super().invoke(ctx)

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError, /
    ) -> None:
        """Drop unknown commands; hand every other error back to discord.py. The
        prefix is a bare `-`, so a markdown bullet ("- milk") dispatches the
        command `milk`, which the default handler logs at ERROR with a traceback."""
        if isinstance(error, commands.CommandNotFound):
            # invoked_with is one token of unbounded length.
            log.debug(f"Unknown command: {str(ctx.invoked_with)[:32]!r}")
            return
        await super().on_command_error(ctx, error)

    async def on_ready(self) -> None:
        activity = discord.Game(name="music", type=3)
        await self.change_presence(status=discord.Status.online, activity=activity)
        if self.user:
            log.info(f"Bot: {self.user.name} # {self.user.id}")
        log.info(f"Environment: {config.ENVIRONMENT}")
        log.info(f"Bot cogs: {list(self.cogs.keys())}")
        log.info(f"Bot guilds: {len(self.guilds)} | latency: {self.latency:.2f}s")
        # FIXME: labelled "Bot commands:" but logs the `voice_states` intent flag.
        # Drop it, or log `sorted(c.qualified_name for c in self.walk_commands())`.
        log.info(f"Bot commands: {self.intents.voice_states}")

    async def close(self) -> None:
        # discord.py's own idempotency check lives in super().close(), reached only
        # partway through; without this guard a second close() re-runs the whole
        # sequence, and two concurrent final drains retire each other's entries.
        if getattr(self, "_teardown_started", False):
            await super().close()
            return
        self._teardown_started = True
        # getattr throughout: run()'s finally calls close() even when setup_hook
        # raised, possibly before __init__ completed, and a bare read would mask
        # the original startup error. First so a slow teardown stops reporting alive.
        liveness = getattr(self, "_liveness_task", None)
        if liveness is not None:
            liveness.cancel()
            self._liveness_task = None
        # Ordering: drainer before archive and Redis (its final drain needs both);
        # super().close() before the Redis pool (it disconnects voice clients and can
        # still dispatch on_voice_state_update, whose cleanup() must reach a live
        # pool or the next start runs spurious recovery for stopped guilds).
        # Every step is individually guarded: a hung participant must not skip
        # the steps after it, and _teardown_started prevents any retry.
        drainer = getattr(self, "history_drainer", None)
        if drainer is not None:
            try:
                await drainer.stop()
            except Exception as e:
                log.warning(f"history drainer shutdown failed: {e}")
        archive = getattr(self, "history_archive", None)
        if archive is not None:
            try:
                await archive.close()
            except Exception as e:
                log.warning(f"history archive shutdown failed: {e}")
        try:
            await super().close()
        except Exception as e:
            log.warning(f"discord client shutdown failed: {e}")
        if self._redis_pool is not None:
            try:
                await close_redis_pool(self._redis_pool)
            except Exception as e:
                log.warning(f"redis pool shutdown failed: {e}")
        loop = asyncio.get_running_loop()
        # Awaited directly: each aclose() owns its off-loop join and bounds the wait.
        from src.chart_pool import chart_pool
        from src.youtube import close_probe_session, ytdlp_pool

        try:
            await ytdlp_pool.aclose()
        except Exception as e:
            log.warning(f"yt-dlp pool shutdown failed: {e}")
        try:
            # Only flips the closed flag on a bot that never ran -analytics.
            await chart_pool.aclose()
        except Exception as e:
            log.warning(f"chart pool shutdown failed: {e}")
        try:
            await close_probe_session()
        except Exception as e:
            log.warning(f"stream-probe session shutdown failed: {e}")
        # Exception, not BaseException, above: a second Ctrl-C abandons the rest,
        # which is what it asks for. shutdown_telemetry blocks up to 30s on its
        # own flush timeout; the bound here is for a collector that hangs past it.
        from src.telemetry import shutdown_telemetry

        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, shutdown_telemetry),
                _TELEMETRY_SHUTDOWN_TIMEOUT,
            )
        except TimeoutError:
            log.warning(
                f"telemetry shutdown did not finish within "
                f"{_TELEMETRY_SHUTDOWN_TIMEOUT:.0f}s; exiting without a full flush"
            )


def main() -> None:
    # With ENVIRONMENT unset, name it after the git branch. Must precede
    # setup_telemetry(), which stamps the value onto the OTel resource.
    if not os.environ.get("ENVIRONMENT"):
        inferred = config.infer_environment_from_git()
        if inferred is not None:
            config.ENVIRONMENT = inferred

    from src.telemetry import setup_telemetry

    setup_telemetry()  # first: configures structlog before any get_logger() resolves

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    # Presence only; MusicBot.cog_load probes the credentials against the live API.
    if spotify_enabled():
        log.info(
            "Spotify credentials found — validating against Spotify API on startup"
        )
    else:
        log.warning(
            "Spotify source disabled — set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET to enable Spotify links"
        )
    # Never at module scope: yt-dlp pool workers re-import this module under
    # spawn/forkserver.
    bot = MusicBotApp()
    bot.run(token)


if __name__ == "__main__":
    main()
