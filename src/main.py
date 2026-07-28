import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import discord
from discord.ext import commands

from src.config import ENVIRONMENT, spotify_enabled
from src.help import MusicHelpCommand
from src.redis_client import close_redis_pool, create_redis_pool, get_redis
from src.util import get_logger

if TYPE_CHECKING:
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

    async def setup_hook(self) -> None:
        self._redis_pool = create_redis_pool()
        self.redis = get_redis(self._redis_pool)
        for extension in EXTENSIONS:
            await self.load_extension(extension)
        # Spawn extraction workers now so the first -play doesn't pay
        # process-spawn + yt-dlp-import latency. Fire-and-forget.
        from src.youtube import ytdlp_pool

        ytdlp_pool.prewarm()

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
        if self._redis_pool is not None:
            await close_redis_pool(self._redis_pool)
        await super().close()
        loop = asyncio.get_running_loop()
        # Awaited directly rather than via the executor below — only aclose() knows
        # which half blocks. It owns its off-loop join and bounds the wait so a
        # stuck extraction can't hang exit.
        from src.youtube import ytdlp_pool

        await ytdlp_pool.aclose()
        # No async form, and blocks up to 30s flushing spans — needs the executor hop.
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
