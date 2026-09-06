"""`-history` — the recently played songs, newest first."""

import discord
from discord.ext import commands

from src.redis_client import HISTORY_CACHE_LIMIT
from src.util import notice_embed
from src.guild_history import GuildHistory, history_embeds


HISTORY_MIN_LIMIT = 1


# Pinned to HISTORY_CACHE_LIMIT: the Redis list holds exactly that many, so a
# larger ceiling returns a short page. Raise both together or neither.
HISTORY_MAX_LIMIT = HISTORY_CACHE_LIMIT

# 8 + the ≤2-embed NP block MusicContext.send prepends = Discord's cap of 10.
HISTORY_EMBEDS_PER_MESSAGE = 8


class HistoryFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    limit: int = 10


async def run(
    ctx: commands.Context, flags: HistoryFlags, *, history: GuildHistory
) -> None:
    """`-history` — the recently played songs, newest first."""
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
    # Each chunk goes through ctx.send, never bare channel.send, so the
    # adopt/retire machinery walks the NP block down to the last chunk.
    for start in range(0, len(embeds), HISTORY_EMBEDS_PER_MESSAGE):
        await ctx.send(embeds=embeds[start : start + HISTORY_EMBEDS_PER_MESSAGE])
