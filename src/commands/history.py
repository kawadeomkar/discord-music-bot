"""`-history` — the recently played songs, newest first."""

import discord
from discord.ext import commands

from src.redis_client import HISTORY_CACHE_LIMIT
from src.util import notice_embed
from src.guild_history import GuildHistory, history_embeds


HISTORY_MIN_LIMIT = 1


# Pinned to HISTORY_CACHE_LIMIT. recent() serves the command from the Redis list
# alone, which holds exactly that many entries, so a larger ceiling here returns a
# short page instead of failing. Raise both together or neither.
HISTORY_MAX_LIMIT = HISTORY_CACHE_LIMIT


# 8 song embeds + the ≤2-embed NP block MusicContext.send may prepend = Discord's
# per-message cap of 10, so the block always fits and is never shed.
HISTORY_EMBEDS_PER_MESSAGE = 8


class HistoryFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    limit: int = 10


async def run(
    ctx: commands.Context, flags: HistoryFlags, *, history: GuildHistory
) -> None:
    """`-history` — the recently played songs, newest first.

    Takes the guild's GuildHistory rather than its player: this command reads one
    capped list and renders it, and the class that owns that list is right above.
    """
    if not (HISTORY_MIN_LIMIT <= flags.limit <= HISTORY_MAX_LIMIT):
        await ctx.send(
            embed=notice_embed(
                f"--limit must be between {HISTORY_MIN_LIMIT} and {HISTORY_MAX_LIMIT}",
                discord.Color.red(),
            )
        )
        return
    entries = await history.recent(flags.limit)
    if not entries:
        await ctx.send(
            embed=notice_embed("No songs have been played yet.", discord.Color.orange())
        )
        return
    embeds = history_embeds(entries)
    # 8 per message keeps every chunk within Discord's 10-embed cap once
    # MusicContext.send prepends the ≤2-embed NP block. Each chunk goes through
    # ctx.send — never bare channel.send in the player's channel — so the
    # adopt/retire machinery walks the block down to the last chunk.
    for start in range(0, len(embeds), HISTORY_EMBEDS_PER_MESSAGE):
        await ctx.send(embeds=embeds[start : start + HISTORY_EMBEDS_PER_MESSAGE])
