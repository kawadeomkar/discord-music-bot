"""`-leaderboard` — the guild's most-played songs over a window."""

import time
from typing import TYPE_CHECKING, Final, Optional

import discord
from discord.ext import commands

from src.redis_client import cache_get, cache_set
from src.util import background_typing, notice_embed, pluralize, refund_cooldown

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.history_archive import ArchiveReader
from src.leaderboard import (
    CACHE_TTL_SECS,
    TOP_N,
    build_embed,
    cache_key,
    from_cache,
    to_cache,
)

# The windows `--days` lands on; 0 stays all-time. Six cache keys per guild,
# where a free integer made every distinct N a fresh aggregate pass.
WINDOWS_DAYS: Final[tuple[int, ...]] = (1, 7, 30, 90, 365)


def resolve_days(requested: int) -> Optional[int]:
    """The window `--days N` rounds to: 0 is all-time, a positive N the nearest
    entry of WINDOWS_DAYS (ties to the shorter), a negative N is None (refused)."""
    if requested == 0:
        return 0
    if requested < 0:
        return None
    return min(WINDOWS_DAYS, key=lambda w: (abs(w - requested), w))


def windows_copy() -> str:
    """ "1, 7, 30, 90 or 365" — the help and the refusal quote one list."""
    *head, last = WINDOWS_DAYS
    return ", ".join(str(d) for d in head) + f" or {last}"


class LeaderboardFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    days: int = 0  # 0 = all-time; otherwise rounded onto WINDOWS_DAYS


async def run(
    ctx: commands.Context,
    flags: LeaderboardFlags,
    *,
    archive: Optional[ArchiveReader],
    redis: Optional[aioredis.Redis],
) -> None:
    """The whole of `-leaderboard`, minus the error handling the cog keeps."""
    # A local: ctx.guild is a property, so its narrowing would not survive an await.
    guild = ctx.guild
    # The cooldown protects Postgres; the three refusals below never reach it.
    if guild is None:
        refund_cooldown(ctx)
        await ctx.send(
            embed=notice_embed(
                "Leaderboards are per server — use this in a server channel.",
                discord.Color.orange(),
            )
        )
        return
    if archive is None:
        refund_cooldown(ctx)
        await ctx.send(
            embed=notice_embed(
                "This server's host has not enabled the long-term play "
                "archive, so there is no leaderboard data.",
                discord.Color.orange(),
            )
        )
        return
    days = resolve_days(flags.days)
    if days is None:
        refund_cooldown(ctx)
        await ctx.send(
            embed=notice_embed(
                f"--days takes a positive number, rounded to {windows_copy()} "
                "days. Omit it, or pass 0, for all-time.",
                discord.Color.red(),
            )
        )
        return
    key = cache_key(guild.id, days, TOP_N)
    board = from_cache(await cache_get(redis, key), top_n=TOP_N)
    if board is None:
        since = time.time() - days * 86400 if days else 0.0
        async with background_typing(ctx):
            board = await archive.leaderboard(guild.id, TOP_N, since_epoch=since)
        await cache_set(redis, key, to_cache(board), CACHE_TTL_SECS)
    embed = build_embed(board, days=days, guild=guild)
    if embed is None:
        window = f"in the last {days} {pluralize(days, 'day')}" if days else "yet"
        await ctx.send(
            embed=notice_embed(
                f"Nothing has been archived {window} — play something first!",
                discord.Color.orange(),
            )
        )
        return
    if days != flags.days:
        # The title names the window served; this names the one asked for.
        embed.set_footer(
            text=f"{embed.footer.text} · --days {flags.days} rounded to {days}."
        )
    await ctx.send(embed=embed)
