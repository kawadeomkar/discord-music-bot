import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import discord
from discord.ext import commands

from src.config import ENVIRONMENT
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
    """Context whose send() keeps the Now Playing embed block glued to the
    bottom of the channel: while a song is live, every command response in the
    player's channel leads with the NP block, followed by the response's own
    embeds, and the message that previously hosted the block is retired
    (deleted if it was a dedicated NP message, strip-edited otherwise).
    Attaching at send time — rather than a post-send edit — makes the response
    and the NP block one atomic message, so the bar is never even momentarily
    buried. Full design: docs/NOW_PLAYING_EMBED_ATTACH_PLAN.md §3."""

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
        # ≤10 is Discord's per-message embed cap — defensive; never expected
        # to trip with this bot's embed counts (worst case is 3).
        attached = bool(block) and len(own) + len(block) <= 10
        embeds = block + own if attached else own
        if embeds:
            message = await super().send(content, embeds=embeds, **kwargs)
        else:
            message = await super().send(content, **kwargs)
        if attached:
            # Gate on the song still being current — the send's await may have
            # crossed a song boundary, making the attached block stale (the
            # gate sheds it from the just-sent message instead of adopting).
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


# Issue #5: AutoShardedBot handles multi-shard within a single process.
# Discord requires sharding at 2500 guilds; plan migration at ~1500.
# shard_count=None lets Discord auto-assign the correct number.
#
# setup_hook is a method override on the Bot subclass, NOT a @bot.event dispatcher.
# In discord.py 2.x, setup_hook is invoked by the library before the bot connects.
class MusicBotApp(commands.AutoShardedBot):
    def __init__(self) -> None:
        super().__init__(
            command_prefix="-",
            intents=intents,
            description="Plays YouTube, Spotify and SoundCloud audio in voice channels.",
            strip_after_prefix=True,
            # Replaces discord.py's DefaultHelpCommand, whose plaintext codeblock
            # output cannot show aliases and clashes with the all-embed responses.
            help_command=MusicHelpCommand(),
        )
        self._redis_pool = None
        self.redis = None

    async def setup_hook(self) -> None:
        self._redis_pool = create_redis_pool()
        self.redis = get_redis(self._redis_pool)
        for extension in EXTENSIONS:
            await self.load_extension(extension)
        # Spawn the yt-dlp extraction workers before the first -play so it doesn't
        # pay process-spawn + yt-dlp-import latency. Non-blocking (fire-and-forget).
        from src.youtube import prewarm_ytdlp_pool

        prewarm_ytdlp_pool()

    async def get_context(
        self,
        origin: Union[discord.Message, discord.Interaction],
        /,
        *,
        cls: type[commands.Context[Any]] = MusicContext,
    ) -> commands.Context[Any]:
        # Written against discord.py's own signature rather than `Any`: `Any` on an
        # override parameter makes the override unconditionally LSP-compatible, so
        # signature drift against the base class becomes uncheckable.
        return await super().get_context(origin, cls=cls)

    async def invoke(self, ctx: commands.Context, /) -> None:
        # `--help` anywhere in a command message short-circuits straight to
        # that command's help embed, before any other step runs — global
        # checks, the cog's voice-channel gate (validate_commands), argument
        # parsing. So `-play --help` answers from outside a voice channel
        # instead of searching YouTube for the string "--help".
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
        log.info(f"Bot commands: {self.intents.voice_states}")

    async def close(self) -> None:
        if self._redis_pool is not None:
            await close_redis_pool(self._redis_pool)
        await super().close()
        loop = asyncio.get_running_loop()
        # Both of these block (joining worker processes / flushing spans for up to
        # 30s), so run them off the event loop.
        from src.youtube import shutdown_ytdlp_pool

        await loop.run_in_executor(None, shutdown_ytdlp_pool)
        from src.telemetry import shutdown_telemetry

        await loop.run_in_executor(None, shutdown_telemetry)


def main() -> None:
    from src.telemetry import setup_telemetry

    setup_telemetry()  # must be first — configures structlog before any get_logger() call resolves

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    if not os.getenv("SPOTIFY_CLIENT_ID"):
        raise ValueError("SPOTIFY_CLIENT_ID environment variable is not set")
    if not os.getenv("SPOTIFY_CLIENT_SECRET"):
        raise ValueError("SPOTIFY_CLIENT_SECRET environment variable is not set")
    # Constructed here, not at module scope: the yt-dlp ProcessPoolExecutor workers
    # re-import this module under the spawn/forkserver start method, and a module-level
    # MusicBotApp() would build a full AutoShardedBot (all of discord.py, the help
    # command) in every worker purely as an import side effect. main() runs only in the
    # parent, so the bot is built exactly once.
    bot = MusicBotApp()
    bot.run(token)


if __name__ == "__main__":
    main()
