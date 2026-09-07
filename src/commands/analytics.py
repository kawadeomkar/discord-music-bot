"""`-analytics` — the guild's listening card: six panels and a chart."""

import asyncio
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import commands

from src.redis_client import cache_get, cache_set
from src.util import background_typing, notice_embed, refund_cooldown

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.history_archive import ArchiveReader
from src.analytics_card import (
    DEFAULT_DAYS,
    TOP_N,
    cache_key,
    cache_ttl_secs,
    empty_notice,
    from_cache,
    invalid_days_notice,
    render_chart,
    resolve_days,
    send_card,
    to_cache,
)
from src.telemetry import get_tracer

_tracer = get_tracer(__name__)


class AnalyticsFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    days: int = DEFAULT_DAYS


async def run(
    ctx: commands.Context,
    flags: AnalyticsFlags,
    *,
    archive: Optional[ArchiveReader],
    redis: Optional[aioredis.Redis],
    tasks: set[asyncio.Task],
) -> None:
    """The whole of `-analytics`, minus the error handling the cog keeps."""
    # A local: ctx.guild is a property, so its narrowing would not survive an await.
    guild = ctx.guild
    # The cooldown protects Postgres; the three refusals below never reach it.
    if guild is None:
        refund_cooldown(ctx)
        await ctx.send(
            embed=notice_embed(
                "Analytics are per server — use this in a server channel.",
                discord.Color.orange(),
            )
        )
        return
    if archive is None:
        refund_cooldown(ctx)
        await ctx.send(
            embed=notice_embed(
                "This server's host has not enabled the long-term play "
                "archive, so there is nothing to chart.",
                discord.Color.orange(),
            )
        )
        return
    days = resolve_days(flags.days)
    if days is None:
        # Before the cache and the archive: an unlisted window must not take a read slot.
        refund_cooldown(ctx)
        await ctx.send(embed=notice_embed(invalid_days_notice(), discord.Color.red()))
        return
    key = cache_key(guild.id, days)
    # Typing spans the query and the render.
    async with background_typing(ctx):
        metrics = from_cache(await cache_get(redis, key))
        if metrics is None:
            with _tracer.start_as_current_span("analytics.query") as span:
                span.set_attribute("analytics.days", days)
                try:
                    metrics = await archive.analytics(guild.id, days=days, top_n=TOP_N)
                except Exception:
                    # A query that failed produced nothing to rate-limit; the
                    # cog's handler renders the error, the guild keeps its slot.
                    refund_cooldown(ctx)
                    raise
                span.set_attribute("analytics.plays", metrics.plays)
            ttl = cache_ttl_secs(metrics)
            if ttl > 0:
                # Non-positive: the day turned while the query ran.
                await cache_set(redis, key, to_cache(metrics), ttl)
        if metrics.is_empty:
            await ctx.send(
                embed=notice_embed(
                    empty_notice(days, metrics.bucket_unit),
                    discord.Color.orange(),
                )
            )
            return
        # After the archive call returns, so its slot and connection are released.
        png = await render_chart(ctx, metrics, redis=redis, tasks=tasks)
    await send_card(ctx, metrics, png, guild)
