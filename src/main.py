import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import discord
from discord.ext import commands
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src import config
from src.config import ENVIRONMENT, spotify_enabled
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

    from src.musicplayer import MusicPlayer

log = get_logger(__name__)

intents = discord.Intents.all()
intents.message_content = True
EXTENSIONS = ("src.musicbot",)


class MusicContext(commands.Context):
    """Context whose send() keeps the Now Playing block at the bottom of the
    channel: responses lead with the NP block, then their own embeds, and the
    previous host is retired (deleted if dedicated, strip-edited otherwise).
    Attaching at send time rather than post-send keeps the response and the
    block one atomic message."""

    async def send(
        self, content: Optional[str] = None, **kwargs: Any
    ) -> discord.Message:
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

    def _np_player(self) -> Optional["MusicPlayer"]:
        """The guild's MusicPlayer, only when attaching is appropriate: guild
        message, MusicBot cog loaded, player exists, a song is live, and this
        channel is the player's home channel (the host never leaves it)."""
        from src.musicbot import MusicBot

        if self.guild is None:
            return None
        cog = self.bot.get_cog("MusicBot")
        if not isinstance(cog, MusicBot):
            return None
        mp = cog.mps.get(self.guild.id)
        if mp is None or mp.current_song is None:
            return None
        if self.channel.id != mp._channel.id:
            return None
        return mp


# GH #5: AutoShardedBot multi-shards in one process. Discord requires sharding
# at 2500 guilds (plan at ~1500); shard_count=None lets Discord assign it.
# setup_hook is a subclass override, not a @bot.event — discord.py 2.x calls it
# before the bot connects.
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
        # Postgres play-history archive + its outbox drainer — one pair per
        # process, built in setup_hook while the archive is enabled
        # (HISTORY_ARCHIVE_ENABLED, the opt-in consent gate for long-term
        # storage). None is the disabled shape — the DEFAULT — and every
        # consumer handles it: musicplayer wires a None outbox notify, -ping
        # renders the Postgres row OFF, close() skips the tier. Enabled, the
        # pair is created in setup_hook (not here) because that is where the
        # event loop exists to start the drainer on.
        self.history_archive: Optional[PostgresHistoryArchive] = None
        self.history_drainer: Optional[HistoryOutboxDrainer] = None
        # See close(): teardown runs at most once, no matter how many times
        # discord.py or a signal handler calls close().
        self._teardown_started = False

    async def setup_hook(self) -> None:
        # The archive flag is read FIRST, before anything else can consume it:
        # the parser raises on garbage, and the next reader would be
        # push_history, which is @_guild_op-wrapped and would swallow that
        # ValueError into one warning per song. Startup is the only place the
        # signal can be loud — and once this read has succeeded, the
        # environment cannot change for the life of the process, so every
        # later call-time read returns the same proven-parseable value.
        archive_enabled = config.history_archive_enabled()
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
        """The enabled arm — today's full archive tier, unchanged: required
        DSN, default-password advisory, outbox consumer group, archive +
        drainer. The operator opted in (HISTORY_ARCHIVE_ENABLED=true), so a
        bot that cannot deliver the archive must say so loudly.

        redis is a parameter rather than a self.redis read only because the
        checker's narrowing (setup_hook assigns it two lines above the call)
        does not cross the method boundary."""
        # Fail fast rather than degrade. A bot that silently ran without the
        # archive would keep accepting plays while every song-end quietly
        # XADDed onto an outbox nobody drains — the failure would surface
        # days later as an un-evictable Redis stream, not at the moment of
        # misconfiguration.
        postgres_url = config.postgres_url()
        if not postgres_url:
            # The remedy has to be something that actually works: this process
            # reads the ENVIRONMENT and has no .env support, so pointing at
            # setup_env.sh (which only writes .env) sent operators in a circle.
            raise RuntimeError(
                "POSTGRES_URL is not set but HISTORY_ARCHIVE_ENABLED is true — "
                "the enabled archive requires its database. Under docker "
                "compose it is supplied for you; for a local run use `just "
                "run`, which loads .env and derives the URL from it. Otherwise "
                "export POSTGRES_URL yourself "
                "(postgresql://user:password@host:5432/dbname). To run without "
                "the archive instead, remove HISTORY_ARCHIVE_ENABLED."
            )
        # Loud, but NOT fatal. compose defaults POSTGRES_PASSWORD to a known
        # value so `docker compose up` works with nothing configured but
        # DISCORD_TOKEN — refusing to start would take that away and put the
        # first-run cliff straight back. So the bot runs and complains instead,
        # here and on every -ping, until it is changed.
        #
        # Changing it is not just an .env edit, and the message says so: Postgres
        # reads POSTGRES_PASSWORD only when it INITIALIZES an empty data
        # directory, so an existing postgres-data volume keeps the password it
        # was created with. Editing .env alone against an initialized volume
        # leaves the bot unable to authenticate against its own database — the
        # confusing failure this warning exists to pre-empt.
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
        # Create the outbox consumer group before anything can write to it. The
        # SECOND fail-fast, and it has to be here rather than in the drainer,
        # because the loud signal is only available at startup: push_history is
        # a @_guild_op method, so a WRONGTYPE from a pre-R1 LIST at
        # history:outbox would be swallowed into one warning per song while
        # BOTH legs of its transaction — the outbox and guild:{id}:history —
        # silently failed. XLEN would raise too, so the depth alarm could never
        # fire either. Total history loss, reported as a warning.
        #
        # ensure_outbox_group tolerates only BUSYGROUP; WRONGTYPE propagates
        # out of setup_hook exactly as the missing-POSTGRES_URL check above
        # does. The remedy is `DEL history:outbox` with the bot stopped.
        #
        # An UNREACHABLE Redis is the opposite case and must not abort startup.
        # create_redis_pool above connects lazily, so before this line a Redis
        # outage cost the bot only its persistence — it still served commands
        # and still played music. Making the group probe fatal would have turned
        # every Redis blip during a deploy into a bot that refuses to boot, which
        # is a strictly worse outage than the degraded mode this file is built
        # around everywhere else — the same degrade-don't-die rule every
        # GuildRedisStore method follows. Degrading is safe here for one
        # specific reason: _read_batch heals NOGROUP once by calling this same
        # function, so the drainer creates the group itself on its first tick
        # after Redis returns. Nothing is lost in between either — push_history
        # cannot reach Redis during the outage either way.
        try:
            await ensure_outbox_group(redis)
        except (RedisConnectionError, RedisTimeoutError) as e:
            log.warning(f"outbox group probe could not reach Redis: {e}")
        # Lazy inside: no connection is made here, so startup never blocks on
        # Postgres — the drainer's backoff loop absorbs an unreachable database.
        # That is what keeps "required" from meaning "Postgres must be up before
        # the bot can start".
        archive = PostgresHistoryArchive(postgres_url)
        self.history_archive = archive
        self.history_drainer = HistoryOutboxDrainer(redis, archive)
        self.history_drainer.start()

    async def _report_archive_disabled(self) -> None:
        """The disabled arm — the default. Say so once, and warn about
        leftovers.

        No POSTGRES_URL requirement, no consumer-group creation (which would
        MKSTREAM the non-evictable outbox key into existence), no archive or
        drainer objects: history_archive/history_drainer stay None, and every
        consumer handles that shape (musicplayer wires a None notify, -ping
        renders OFF, close() skips the tier). push_history's XADD leg reads
        the same flag, so nothing accumulates for Postgres either.
        """
        # SAYS WHAT THE RETENTION IS, not merely what it is not. "Nothing is
        # written to long-term storage" was true about Postgres and misleading
        # about the whole: push_history PERSISTs guild:{id}:history — no TTL,
        # ever, by design — so an opted-out deployment does retain requester
        # ids, titles and timestamps for 50 plays per guild, indefinitely. An
        # operator who read the short version and was later asked to erase a
        # user's data would follow README's erasure step, remove the
        # postgres-data volume, and delete nothing at all, because the only
        # copy is in Redis. The number and the "no expiry" are the load-bearing
        # parts; README's "Disabling — and erasing" carries the command.
        log.info(
            "History archive disabled (the default; HISTORY_ARCHIVE_ENABLED=true "
            f"opts in). Plays are kept only in the per-guild Redis list behind "
            f"-history — the newest {HISTORY_CACHE_LIMIT} per guild, retained "
            "until deleted (no expiry); nothing is written to Postgres."
        )
        if config.postgres_url():
            # The flag wins over URL presence, explicitly. Compose interpolates
            # POSTGRES_URL into the bot's environment whether or not the
            # archive profile is active, so a DSN here is NOT evidence the
            # operator wanted archiving — and URL-presence gating would be
            # implicit consent, the exact thing the opt-in removes. One INFO so
            # an operator who set the URL expecting an archive is not left
            # guessing.
            log.info(
                "POSTGRES_URL is set but ignored: the archive is enabled by "
                "HISTORY_ARCHIVE_ENABLED=true, never by URL presence."
            )
        await self._warn_if_outbox_left_over()

    async def _warn_if_outbox_left_over(self) -> None:
        """One WARNING when a previously-enabled archive left outbox entries
        behind.

        Those plays were buffered for an archive that is now off: they sit in
        a non-evictable key and will never drain. The remedy stays an operator
        decision — re-enable to drain, or DEL to discard. Auto-deleting would
        let an accidental toggle destroy un-archived plays irreversibly, and
        silence would leave exactly the data the operator opted out of keeping
        lingering invisibly; the warning is the middle path.

        Degrade-don't-die, via the RAISING outbox_depth helper — golden rule
        5's split holds: the outbox helpers raise and the caller is the error
        handler, so no new swallowing store method. An unreachable Redis skips
        the probe (same rule as the enabled arm's group probe: a Redis blip
        must not stop the music). A WRONGTYPE — a pre-R1 list at the key —
        downgrades to a warning rather than the enabled path's startup abort:
        with the producer's XADD leg off, a mis-shaped key is inert.
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
        # `--help` ANYWHERE in the raw message content short-circuits straight to
        # that command's help embed, before checks, the cog's voice gate
        # (validate_commands) and argument parsing — so `-play --help` answers from
        # outside a voice channel instead of searching YouTube for "--help".
        if ctx.command is not None and "--help" in ctx.message.content:
            await ctx.send_help(ctx.command)
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
        # Reentrancy guard, ahead of everything. discord.py's own close() is
        # idempotent, but its check lives in super().close() — which this
        # override only reaches partway through teardown, so a second close()
        # would re-run the whole sequence first. That matters most for
        # drainer.stop(): two concurrent final drains each peek → insert →
        # retire, and the second retires entries the first never inserted
        # (H2). The drainer now defends itself as well; this is the belt over
        # those braces, and it also spares the archive/Redis pools a redundant
        # second close.
        #
        # getattr for the same reason the two reads below use it: close() must
        # be the one method that cannot itself raise, or it masks whatever
        # actually went wrong.
        if getattr(self, "_teardown_started", False):
            await super().close()
            return
        self._teardown_started = True
        # Order is load-bearing in BOTH directions, and the two constraints order
        # different pairs, so they compose:
        #
        #   drainer BEFORE archive and Redis — its final drain reads the outbox
        #     and writes Postgres, so both have to still be alive for it.
        #   super().close() BEFORE the Redis pool — it disconnects voice clients
        #     and can still dispatch events in flight, and an
        #     on_voice_state_update landing in that window runs cleanup(), which
        #     calls store.clear_connection()/refresh_ttl(). With Redis already
        #     closed those writes hit a dead pool and are swallowed as warnings, so
        #     an ORDERLY shutdown persists state as if the bot had crashed and the
        #     next start runs spurious crash recovery for cleanly-stopped guilds.
        #
        # The archive can close before the disconnect: the cleanup path the
        # disconnect can trigger touches Redis only, and a song ending in that
        # window still lands on the outbox and drains on the next start.
        #
        # getattr, not a plain attribute read: discord.py calls close() from run()'s
        # finally even when setup_hook RAISED — possibly before __init__ completed —
        # and a bare read would throw AttributeError out of close() and mask the
        # original startup error. The `is not None` checks below carry a second
        # meaning: with the archive disabled the pair is None by design.
        #
        # Each step is individually guarded, for the same reason: teardown must run
        # to completion even when one participant is sick. These CAN raise — a hung
        # Postgres made archive.close() raise TimeoutError after 30s — and because
        # _teardown_started is already set the retry path short-circuits, so every
        # step after the raiser was skipped for good: Redis pool left open,
        # discord.py never closed, the yt-dlp pool left to its 61s atexit join and
        # no spans flushed, hiding the very failure that caused it. No step may
        # skip a later one.
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
        # which half blocks. It owns its off-loop join and bounds the wait so a
        # stuck extraction can't hang exit.
        from src.youtube import ytdlp_pool

        try:
            await ytdlp_pool.aclose()
        except Exception as e:
            # Guarded like every step above it. aclose() already swallows its own
            # join timeout, so this arm is for the unexpected — which would
            # otherwise cost the span flush below, the record of the failed
            # shutdown.
            log.warning(f"yt-dlp pool shutdown failed: {e}")
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
    # Spotify is optional: without credentials only Spotify links are rejected.
    # Logged here so a missing credential is visible at startup rather than as a
    # per-link error later. This reports only whether credentials were PROVIDED;
    # MusicBot.cog_load probes them against the live API and logs ENABLED/INVALID.
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
    # AutoShardedBot in every worker as an import side effect. main() runs only in
    # the parent, so the bot is built exactly once.
    bot = MusicBotApp()
    bot.run(token)


if __name__ == "__main__":
    main()
