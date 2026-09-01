"""`-leaderboard` — the guild's most-played songs over a window."""

import time
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import commands

from src.redis_client import cache_get, cache_set
from src.util import background_typing, notice_embed, pluralize

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.history_archive import ArchiveReader
from src.leaderboard import (
    CACHE_TTL_SECS,
    MAX_DAYS,
    TOP_N,
    build_embed,
    cache_key,
    from_cache,
    to_cache,
)


class LeaderboardFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    days: int = 0  # 0 = all-time; otherwise a rolling now - N*86400 window


async def run(
    ctx: commands.Context,
    flags: LeaderboardFlags,
    *,
    archive: Optional[ArchiveReader],
    redis: Optional[aioredis.Redis],
) -> None:
    """The whole of `-leaderboard`, minus the error handling the cog keeps.

    The cog resolves what discord.py owns — the flags, the archive, the Redis
    handle — and hands them over; nothing here reaches back into MusicBot. Raising
    is the contract: the caller's `except` renders the board's failure copy.
    """
    # A local: ctx.guild is a property, so narrowing it would not survive the
    # awaits below.
    guild = ctx.guild
    if guild is None:
        await ctx.send(
            embed=notice_embed(
                "Leaderboards are per server — use this in a server channel.",
                discord.Color.orange(),
            )
        )
        return
    if archive is None:
        await ctx.send(
            embed=notice_embed(
                "This server's host has not enabled the long-term play "
                "archive, so there is no leaderboard data.",
                discord.Color.orange(),
            )
        )
        return
    if not 0 <= flags.days <= MAX_DAYS:
        await ctx.send(
            embed=notice_embed(
                f"--days must be between 1 and {MAX_DAYS}. "
                "Omit it, or pass 0, for all-time.",
                discord.Color.red(),
            )
        )
        return
    key = cache_key(guild.id, flags.days, TOP_N)
    board = from_cache(await cache_get(redis, key), top_n=TOP_N)
    if board is None:
        since = time.time() - flags.days * 86400 if flags.days else 0.0
        async with background_typing(ctx):
            board = await archive.leaderboard(guild.id, TOP_N, since_epoch=since)
        await cache_set(redis, key, to_cache(board), CACHE_TTL_SECS)
    embed = build_embed(board, days=flags.days, guild=guild)
    if embed is None:
        window = (
            f"in the last {flags.days} {pluralize(flags.days, 'day')}"
            if flags.days
            else "yet"
        )
        await ctx.send(
            embed=notice_embed(
                f"Nothing has been archived {window} — play something first!",
                discord.Color.orange(),
            )
        )
        return
    await ctx.send(embed=embed)
