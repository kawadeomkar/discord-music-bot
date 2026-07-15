import asyncio
import contextlib
from typing import Any, List, Optional

import discord
import structlog
from opentelemetry.trace import StatusCode


def queue_message(songs: List[str]) -> str:
    capped = songs[:10]
    msg = "\n".join([f"{i + 1}: {capped[i]}" for i in range(len(capped))])
    if len(songs) > 10:
        msg += "\n..."
    return msg


def trace_footer(span: Any) -> Optional[str]:
    span_ctx = span.get_span_context()
    return f"trace: {format(span_ctx.trace_id, '032x')}" if span_ctx.is_valid else None


async def cancel_task(task: Optional[asyncio.Task]) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def record_span_error(span: Any, e: Exception) -> None:
    span.record_exception(e)
    span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")


def latency_color(ms: float) -> discord.Color:
    if ms <= 50:
        return discord.Color(0x44FF44)
    if ms <= 100:
        return discord.Color(0xFFD000)
    if ms <= 200:
        return discord.Color(0xFF6600)
    return discord.Color(0x990000)


def notice_embed(
    message: str,
    color: Optional[discord.Color] = None,
    *,
    title: Optional[str] = None,
) -> discord.Embed:
    """Build a lightweight single-message embed for short status/notice replies.

    The one place that turns a plain status string ("Shuffled!", "Volume set…",
    validation errors) into an embed. Every command response must be an embed
    now that MusicContext.send funnels responses and prepends the Now Playing
    block: a bare `content` string would render as loose text above the block,
    breaking the uniform embed stack. Pairs with the richer send_embed (which
    forces a title/description split) for the one-liner case where a body-only
    embed reads best.
    """
    return discord.Embed(title=title, description=message, color=color)


async def send_embed(
    destination: discord.abc.Messageable,
    title: str,
    description: str,
    color: Optional[discord.Color] = None,
    footer: Optional[str] = None,
    thumbnail: Optional[str] = None,
    fields: Optional[List[tuple[str, str, bool]]] = None,
) -> discord.Message:
    embed = discord.Embed(title=title, description=description, color=color)
    if footer:
        embed.set_footer(text=footer)
    if thumbnail:
        embed.set_thumbnail(url=thumbnail)
    for name, value, inline in fields or []:
        embed.add_field(name=name, value=value, inline=inline)
    return await destination.send(embed=embed)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


log = get_logger(__name__)
