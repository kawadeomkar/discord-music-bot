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
from src.config import ENVIRONMENT, spotify_enabled

# Reaches yt_dlp transitively (src.debug -> src.ping -> yt_dlp.version), and the
# console script imports THIS module at module scope — so the yt-dlp pool's
# spawn/forkserver bootstrap re-imports it too, measured at ~+500ms per bootstrap.
# Kept because it is paid once at forkserver bootstrap on the deploy target (Linux),
# happens inside the fire-and-forget prewarm() rather than on the loop, and is
# arguably the point: prewarm()'s promise is that the first -play does not absorb
# yt-dlp import latency, which was only half true before.
from src.debug import decorate_embeds, strip_debug_footers
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
    """The gateway events this bot subscribes to.

    `presences` is deliberately absent: it is privileged — which blocks bot
    verification past 100 guilds — and nothing here reads a presence update.
    `change_presence()` SENDS our own and needs no intent for it.
    """
    intents = discord.Intents.none()
    # Guild/channel/voice-client cache. Everything else assumes it.
    intents.guilds = True
    # on_voice_state_update, plus the VoiceChannel.members behind the
    # alone-in-channel disconnect timer.
    intents.voice_states = True
    # Prefix commands dispatch from on_message, so the events have to arrive at
    # all — message_content alone does not deliver them.
    intents.guild_messages = True
    # -help renders in a DM and -debug answers one; both are pinned by tests.
    intents.dm_messages = True
    # Privileged. Without it every message arrives with empty content and no
    # command ever matches.
    intents.message_content = True
    # Privileged. guild.get_member() rehydrates a queue entry's requester and
    # backs VoiceChannel.members; the event-driven cache alone goes patchy
    # across a restart.
    intents.members = True
    return intents


intents = _build_intents()
EXTENSIONS = ("src.musicbot",)


class MusicContext(commands.Context):
    """Context whose send() keeps the Now Playing block at the bottom of the channel:
    responses lead with the NP block, then their own embeds, and the previous host is
    retired (deleted if dedicated, strip-edited otherwise). Attaching at send time
    keeps the response and the block one atomic message.

    It is also where debug mode's footer is applied — the one choke point through
    which every command response passes."""

    async def send(
        self, content: Optional[str] = None, **kwargs: Any
    ) -> discord.Message:
        # Ahead of the NP branch, so both send paths carry it. Decoration MUTATES
        # the caller's embeds rather than reshaping kwargs, which is what lets the
        # early return below stay a verbatim pass-through.
        self._decorate_for_debug(kwargs)
        mp = self._np_player()
        if mp is None:
            return await super().send(content, **kwargs)
        embeds_kwarg = kwargs.pop("embeds", None)
        single = kwargs.pop("embed", None)
        if single is not None and embeds_kwarg is not None:
            # match discord.py's own send() contract instead of silently merging
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
            # The send's await may have crossed a song boundary; the gate sheds
            # a stale block from the just-sent message instead of adopting it.
            mp._adopt_np_host_if_current(message, own, song)
        return message

    def _decorate_for_debug(self, kwargs: dict[str, Any]) -> None:
        """Add the debug footer to this response's own embeds while the guild has
        debug mode on. The NP block is decorated by the player instead, at build
        time, so the progress tick cannot re-render it back to bare.
        See docs/ARCHITECTURE.md#debug-footer-seams."""
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
        guild_id = self.guild.id if self.guild else None
        if not cog.debug_settings.enabled(guild_id):
            # Strip rather than return: a cached embed built while debug mode was
            # on must lose its suffix once it is off.
            strip_debug_footers(own)
            return
        # Keyed by id(ctx), and this IS the ctx. Absent for a send outside any
        # command, which just means no elapsed time and no span to name.
        active = cog._active_spans.get(id(self))
        decorate_embeds(
            own,
            span=active.span if active is not None else None,
            elapsed_ms=(time.monotonic() - active.started) * 1000
            if active is not None
            else None,
            shard_id=self.guild.shard_id if self.guild else None,
            runtime=cog.debug_settings.snapshot,
        )

    def _music_cog(self) -> Optional["MusicBot"]:
        from src.musicbot import MusicBot

        cog = self.bot.get_cog("MusicBot")
        return cog if isinstance(cog, MusicBot) else None

    def _np_player(self) -> Optional["MusicPlayer"]:
        """The guild's MusicPlayer, only when attaching is appropriate: guild message,
        MusicBot cog loaded, player exists, a song is live, and this channel is the
        player's home channel (the host never leaves it)."""
        if self.guild is None:
            return None
        cog = self._music_cog()
        if cog is None:
            return None
        mp = cog.mps.get(self.guild.id)
        if mp is None or mp.current_song is None:
            return None
        if self.channel.id != mp._channel.id:
            return None
        return mp


# GH #5: AutoShardedBot multi-shards in one process. Discord requires sharding at 2500
# guilds (plan at ~1500); shard_count=None lets Discord assign it. setup_hook is a
# subclass override, not a @bot.event — discord.py 2.x calls it before connecting.
class MusicBotApp(commands.AutoShardedBot):
    def __init__(self) -> None:
        super().__init__(
            command_prefix="-",
            intents=intents,
            description="Plays YouTube, Spotify and SoundCloud audio in voice channels.",
            strip_after_prefix=True,
            # DefaultHelpCommand's plaintext codeblock can't show aliases and
            # clashes with the all-embed responses.
            help_command=MusicHelpCommand(),
        )
        self._redis_pool = None
        self.redis = None
        self._liveness_task: Optional[asyncio.Task] = None
        # Postgres play-history archive + its outbox drainer, built in setup_hook while
        # HISTORY_ARCHIVE_ENABLED (the opt-in consent gate for long-term storage). None
        # is the disabled shape — the default — and every consumer handles it:
        # musicplayer wires a None notify, -ping renders Postgres OFF, close() skips it.
        self.history_archive: Optional[PostgresHistoryArchive] = None
        self.history_drainer: Optional[HistoryOutboxDrainer] = None
        # See close(): teardown runs at most once, no matter how many times
        # discord.py or a signal handler calls close().
        self._teardown_started = False

    async def _liveness_heartbeat(self) -> None:
        """Touch LIVENESS_FILE on a fixed cadence for the container HEALTHCHECK.

        Answers one question: is the event loop still turning. A wedged loop
        stops touching the file and the healthcheck fails on the stale mtime.
        Probing Redis or Discord here would report the bot dead over a
        dependency blip, so this stays dependency-free.
        """
        path = Path(config.LIVENESS_FILE)
        while True:
            try:
                path.touch()
            except OSError as e:
                # An unwritable path degrades to no liveness signal; the
                # healthcheck then fails on its own rather than the bot dying here.
                log.warning(f"Liveness touch failed for {path}: {e}")
            await asyncio.sleep(config.LIVENESS_INTERVAL_SECS)

    async def setup_hook(self) -> None:
        # The archive flag is read first, before anything else can consume it: the
        # parser raises on garbage and the next reader would be push_history, which is
        # @_guild_op-wrapped and would swallow that ValueError into one warning per
        # song. Startup is the only place the signal can be loud.
        archive_enabled = config.history_archive_enabled()
        # Ahead of the pool and the extensions so the file exists early in the
        # HEALTHCHECK's start-period, and after the flag read, which stays first.
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
        # Spawn extraction workers now so the first -play doesn't pay
        # process-spawn + yt-dlp-import latency. Fire-and-forget.
        from src.youtube import ytdlp_pool

        ytdlp_pool.prewarm()

    async def _setup_history_archive(self, redis: "aioredis.Redis") -> None:
        """The enabled arm: required DSN, default-password advisory, outbox consumer
        group, archive + drainer. The operator opted in, so a bot that cannot deliver
        the archive must say so loudly. `redis` is a parameter rather than a self.redis
        read only because the checker's narrowing does not cross the method boundary."""
        # Fail fast rather than degrade: a bot that silently ran without the archive
        # would keep XADDing every song-end onto an outbox nobody drains, surfacing days
        # later as an un-evictable Redis stream instead of at the misconfigured moment.
        postgres_url = config.postgres_url()
        if not postgres_url:
            # The remedy names `just run`, not setup_env.sh: this process reads only
            # the environment and has no .env support.
            raise RuntimeError(
                "POSTGRES_URL is not set but HISTORY_ARCHIVE_ENABLED is true — "
                "the enabled archive requires its database. Under docker "
                "compose it is supplied for you; for a local run use `just "
                "run`, which loads .env and derives the URL from it. Otherwise "
                "export POSTGRES_URL yourself "
                "(postgresql://user:password@host:5432/dbname). To run without "
                "the archive instead, remove HISTORY_ARCHIVE_ENABLED."
            )
        # Loud, but not fatal: compose defaults POSTGRES_PASSWORD so `docker compose up`
        # works with nothing configured but DISCORD_TOKEN, and refusing to start would
        # put the first-run cliff straight back. The message spells out the order because
        # Postgres reads the variable only when it INITIALIZES an empty data directory —
        # editing .env against an existing volume just locks the bot out of its own db.
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
        # Create the outbox consumer group before anything can write to it. The second
        # fail-fast, and it has to be at startup: push_history is @_guild_op-wrapped, so
        # a WRONGTYPE from a pre-stream list at history:outbox would be swallowed into one
        # warning per song while both legs of its transaction failed — total history
        # loss, reported as a warning. ensure_outbox_group tolerates only BUSYGROUP;
        # WRONGTYPE propagates, remedied by `DEL history:outbox` with the bot stopped.
        #
        # An UNREACHABLE Redis is the opposite case and must not abort startup: the pool
        # connects lazily, so an outage otherwise costs only persistence, and a fatal
        # probe would turn every deploy-time blip into a bot that refuses to boot.
        # Degrading is safe because _read_batch heals NOGROUP itself, by calling this
        # same function on its first tick after Redis returns.
        try:
            await ensure_outbox_group(redis)
        except (RedisConnectionError, RedisTimeoutError) as e:
            log.warning(f"outbox group probe could not reach Redis: {e}")
        # Lazy inside: no connection is made here, so startup never blocks on Postgres —
        # the drainer's backoff loop absorbs an unreachable database. That is what keeps
        # "required" from meaning "Postgres must be up before the bot can start".
        archive = PostgresHistoryArchive(postgres_url)
        self.history_archive = archive
        self.history_drainer = HistoryOutboxDrainer(redis, archive)
        self.history_drainer.start()

    async def _report_archive_disabled(self) -> None:
        """The disabled arm — the default. Say so once, and warn about leftovers. No
        POSTGRES_URL requirement and no consumer-group creation (which would MKSTREAM
        the non-evictable outbox key into existence); history_archive/history_drainer
        stay None, and push_history's XADD leg reads the same flag, so nothing accrues.
        """
        # States what is retained, not just what is not: push_history PERSISTs
        # guild:{id}:history — no TTL, ever, by design — so an opted-out deployment
        # still retains requester ids, titles and timestamps for 50 plays per guild,
        # indefinitely. An operator asked to erase a user's data has to know that the
        # only copy is in Redis.
        log.info(
            "History archive disabled (the default; HISTORY_ARCHIVE_ENABLED=true "
            f"opts in). Plays are kept only in the per-guild Redis list behind "
            f"-history — the newest {HISTORY_CACHE_LIMIT} per guild, retained "
            "until deleted (no expiry); nothing is written to Postgres."
        )
        if config.postgres_url():
            # The flag wins over URL presence, explicitly: compose interpolates
            # POSTGRES_URL whether or not the archive profile is active, so a DSN here
            # is not evidence of consent. One INFO so an operator who set it expecting
            # an archive is not left guessing.
            log.info(
                "POSTGRES_URL is set but ignored: the archive is enabled by "
                "HISTORY_ARCHIVE_ENABLED=true, never by URL presence."
            )
        await self._warn_if_outbox_left_over()

    async def _warn_if_outbox_left_over(self) -> None:
        """One WARNING when a previously-enabled archive left outbox entries behind: they
        sit in a non-evictable key and will never drain. Never auto-deleted — an
        accidental toggle would destroy un-archived plays irreversibly. This is the error
        handler for the RAISING outbox_depth helper: an unreachable Redis skips the probe
        and a WRONGTYPE only warns, since with the producer's XADD leg off a mis-shaped
        key is inert.
        """
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
        # Typed against discord.py's signature, not `Any`: `Any` on an override
        # parameter makes signature drift against the base class uncheckable.
        return await super().get_context(origin, cls=cls)

    async def invoke(self, ctx: commands.Context, /) -> None:
        command = ctx.command
        # `--help` ANYWHERE in the raw message short-circuits to that command's help
        # embed, before checks, the cog's voice gate and argument parsing — so `-play
        # --help` answers from outside a voice channel instead of searching for it.
        short_circuit = command is not None and "--help" in ctx.message.content
        # Neither help path reaches cog_before_invoke — the short-circuit skips
        # dispatch, and discord.py owns the help command — so both borrow a span
        # from the cog. See MusicBot.traced_help.
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

    async def on_ready(self) -> None:
        activity = discord.Game(name="music", type=3)
        await self.change_presence(status=discord.Status.online, activity=activity)
        if self.user:
            log.info(f"Bot: {self.user.name} # {self.user.id}")
        log.info(f"Environment: {ENVIRONMENT}")
        log.info(f"Bot cogs: {list(self.cogs.keys())}")
        log.info(f"Bot guilds: {len(self.guilds)} | latency: {self.latency:.2f}s")
        # FIXME: this line is labelled "Bot commands:" but logs the `voice_states`
        # intent flag (a bool), not the commands. Drop it, or log
        # `sorted(c.qualified_name for c in self.walk_commands())`.
        log.info(f"Bot commands: {self.intents.voice_states}")

    async def close(self) -> None:
        # Reentrancy guard, ahead of everything: discord.py's close() is idempotent but
        # its check lives in super().close(), which this override only reaches partway
        # through teardown, so a second close() would re-run the whole sequence first.
        # That matters most for drainer.stop() — two concurrent final drains each peek →
        # insert → retire, and the second retires entries the first never inserted.
        if getattr(self, "_teardown_started", False):
            await super().close()
            return
        self._teardown_started = True
        # getattr for the same reason as the rest: a close() from run()'s finally
        # can land before __init__ finished. First in the sequence so a slow
        # teardown stops reporting itself alive.
        liveness = getattr(self, "_liveness_task", None)
        if liveness is not None:
            liveness.cancel()
            self._liveness_task = None
        # Two ordering constraints, on different pairs, so they compose:
        #   drainer before archive and Redis — its final drain reads the outbox and
        #     writes Postgres, so both have to still be alive for it.
        #   super().close() before the Redis pool — it disconnects voice clients and can
        #     still dispatch events; an on_voice_state_update landing in that window runs
        #     cleanup(), whose clear_connection()/refresh_ttl() would hit a dead pool and
        #     be swallowed, so an ORDERLY shutdown persists state as if the bot had
        #     crashed and the next start runs spurious recovery for stopped guilds.
        # The archive may close before that disconnect: the cleanup path it can trigger
        # touches Redis only, and a song ending there still drains on the next start.
        #
        # getattr, not plain reads: discord.py calls close() from run()'s finally even
        # when setup_hook RAISED — possibly before __init__ completed — so a bare read
        # would mask the original startup error. `is not None` also encodes the disabled
        # archive's None pair.
        #
        # Every step is individually guarded so teardown completes even when one
        # participant is sick. These CAN raise — a hung Postgres made archive.close()
        # raise TimeoutError after 30s — and _teardown_started short-circuits the retry,
        # so every step after the raiser was skipped for good: Redis pool left open,
        # discord.py never closed, the yt-dlp pool left to its 61s atexit join, no spans
        # flushed. No step may skip a later one.
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
        # Awaited directly rather than via the executor below — only aclose() knows
        # which half blocks; it owns its off-loop join and bounds the wait.
        from src.youtube import close_probe_session, ytdlp_pool

        try:
            await ytdlp_pool.aclose()
        except Exception as e:
            # Guarded like every step above it: aclose() already swallows its own join
            # timeout, so this arm is for the unexpected — which would otherwise cost
            # the span flush below, the record of the failed shutdown.
            log.warning(f"yt-dlp pool shutdown failed: {e}")
        try:
            await close_probe_session()
        except Exception as e:
            log.warning(f"stream-probe session shutdown failed: {e}")
        # shutdown_telemetry has no async form and blocks up to 30s flushing spans,
        # so it needs the executor hop.
        from src.telemetry import shutdown_telemetry

        await loop.run_in_executor(None, shutdown_telemetry)


def main() -> None:
    from src.telemetry import setup_telemetry

    setup_telemetry()  # must be first — configures structlog before any get_logger() call resolves

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    # Spotify is optional: without credentials only Spotify links are rejected. This
    # reports only whether credentials were PROVIDED; MusicBot.cog_load probes them
    # against the live API and logs enabled/invalid.
    if spotify_enabled():
        log.info(
            "Spotify credentials found — validating against Spotify API on startup"
        )
    else:
        log.warning(
            "Spotify source disabled — set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET to enable Spotify links"
        )
    # Not at module scope: yt-dlp pool workers re-import this module under
    # spawn/forkserver, so a module-level MusicBotApp() would build a full
    # AutoShardedBot in every worker. main() runs only in the parent.
    bot = MusicBotApp()
    bot.run(token)


if __name__ == "__main__":
    main()
