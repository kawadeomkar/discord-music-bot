import asyncio
import contextlib
from typing import Any, Optional
from collections.abc import Coroutine

import discord
import structlog
from opentelemetry.trace import Span, StatusCode

from src.guild_state import HistoryEntry


def queue_message(songs: list[str]) -> str:
    capped = songs[:10]
    msg = "\n".join([f"{i + 1}: {capped[i]}" for i in range(len(capped))])
    if len(songs) > 10:
        msg += "\n..."
    return msg


def trace_id_of(span: Span) -> str:
    """The span's trace id as 32 hex chars, or "" when the span is not recording. Empty
    string rather than None: every consumer stores this in a column or log field that is
    text, so an absent trace and an unset one should not be two cases downstream."""
    span_ctx = span.get_span_context()
    return format(span_ctx.trace_id, "032x") if span_ctx.is_valid else ""


def trace_footer(span: Span) -> Optional[str]:
    """Return an embed-footer string identifying the current trace, or None if untraced."""
    trace_id = trace_id_of(span)
    return f"trace: {trace_id}" if trace_id else None


async def cancel_task(task: Optional[asyncio.Task]) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def spawn_background(
    coro: Coroutine[Any, Any, Any], tasks: set[asyncio.Task[Any]]
) -> asyncio.Task[Any]:
    """Create a fire-and-forget task tracked in `tasks`, auto-discarded on completion."""
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


def record_span_error(span: Span, e: Exception) -> None:
    """Record an exception on a span and mark its status as ERROR."""
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
    """Turn a plain status string ("Shuffled!", validation errors) into an embed. Every
    command response must be an embed: MusicContext.send prepends the Now Playing block,
    and a bare `content` string would render as loose text above it. send_embed is the
    pair for anything needing a title/description split."""
    return discord.Embed(title=title, description=message, color=color)


async def send_embed(
    destination: discord.abc.Messageable,
    title: str,
    description: str,
    color: Optional[discord.Color] = None,
    footer: Optional[str] = None,
    thumbnail: Optional[str] = None,
    fields: Optional[list[tuple[str, str, bool]]] = None,
) -> discord.Message:
    embed = discord.Embed(title=title, description=description, color=color)
    if footer:
        embed.set_footer(text=footer)
    if thumbnail:
        embed.set_thumbnail(url=thumbnail)
    for name, value, inline in fields or []:
        embed.add_field(name=name, value=value, inline=inline)
    return await destination.send(embed=embed)


def fmt_duration(secs: int) -> str:
    """Compact clock rendering: 225 → "3:45", 3725 → "1:02:05"."""
    m, s = divmod(max(0, secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """The noun form matching `count`: pluralize(1, "song") → "song",
    pluralize(3, "song") → "songs". `plural` overrides the default `+ "s"`."""
    if count == 1:
        return singular
    return plural if plural is not None else singular + "s"


# Discord's hard embed-title limit. An over-length title 400s the whole send(),
# silently no-opping -history or failing the now-playing send/edit.
EMBED_TITLE_LIMIT = 256


def truncate_embed_title(title: str) -> str:
    """Clip a title to Discord's embed-title limit, ellipsizing if clipped."""
    if len(title) <= EMBED_TITLE_LIMIT:
        return title
    return title[: EMBED_TITLE_LIMIT - 1] + "…"


def history_embeds(entries: list[HistoryEntry]) -> list[discord.Embed]:
    """One embed per played song, in the given (newest-first) order. Layout: numbered
    title, raw webpage_url on its own line (Discord auto-links it), then played/duration,
    requester, and — when known — the played-at timestamp as viewer-local <t:…:f>."""
    embeds = []
    for i, entry in enumerate(entries, start=1):
        lines = []
        if entry.webpage_url:
            lines.append(entry.webpage_url)
        requested_by = (
            f"<@{entry.requester_id}>"
            if entry.requester_id
            else (entry.requester_name or "unknown")
        )
        meta = (
            f"{fmt_duration(entry.played_secs)} / {fmt_duration(entry.duration_secs)}"
            f" · requested by {requested_by}"
        )
        # played_at == 0 means unknown (absent on the wire); <t:0:f> would render
        # "1 January 1970", so omit the timestamp instead.
        if entry.played_at:
            meta += f" · <t:{int(entry.played_at)}:f>"
        lines.append(meta)
        title = truncate_embed_title(f"{i}. {entry.title}")
        embed = discord.Embed(
            title=title,
            description="\n".join(lines),
            color=discord.Color.blue(),
        )
        if entry.thumbnail:
            embed.set_thumbnail(url=entry.thumbnail)
        embeds.append(embed)
    return embeds


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


log = get_logger(__name__)
