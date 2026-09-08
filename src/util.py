import asyncio
import contextlib
import re
from typing import Any, Final, Optional
from collections.abc import AsyncGenerator, Coroutine

import discord
import structlog
from discord.ext import commands
from opentelemetry.trace import Span, SpanContext, StatusCode, get_current_span
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)


# Discord's embed field-value cap; it rejects the WHOLE send past it.
_FIELD_VALUE_MAX = 1024
_TRUNCATION_MARK = "..."


def queue_message(songs: list[str]) -> str:
    """Numbered song list for an embed field, bounded by count AND by length
    (ten 100-char titles compose to 1040). Either overflow ends in the mark."""
    lines: list[str] = []
    used = 0
    budget = _FIELD_VALUE_MAX - len(_TRUNCATION_MARK) - 1
    for i, song in enumerate(songs[:10]):
        line = f"{i + 1}: {song}"
        if used + len(line) > budget:
            # A single line over budget is truncated, not dropped.
            if not lines:
                lines.append(line[:budget])
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(songs):
        lines.append(_TRUNCATION_MARK)
    return "\n".join(lines)


def trace_id_of(span: Span) -> str:
    """The span's trace id as 32 hex chars, or "" when not recording — every
    consumer stores it in a text column or log field."""
    span_ctx = span.get_span_context()
    return format(span_ctx.trace_id, "032x") if span_ctx.is_valid else ""


def trace_footer(span: Span) -> Optional[str]:
    """Return an embed-footer string identifying the current trace, or None if untraced."""
    trace_id = trace_id_of(span)
    return f"trace: {trace_id}" if trace_id else None


_TRACE_PROPAGATOR: Final = TraceContextTextMapPropagator()


def current_traceparent() -> str:
    """The current span as a W3C traceparent, or "" when nothing is being traced.
    inject() skips only a context equal to INVALID_SPAN_CONTEXT, hence the check."""
    if not get_current_span().get_span_context().is_valid:
        return ""
    carrier: dict[str, str] = {}
    _TRACE_PROPAGATOR.inject(carrier)
    return carrier.get("traceparent", "")


def traceparent_context(traceparent: str) -> Optional[SpanContext]:
    """The span a traceparent names, for Span.add_link(). None when the value is
    absent or unparseable: extract() answers an invalid span rather than raising."""
    if not traceparent:
        return None
    span_ctx = get_current_span(
        _TRACE_PROPAGATOR.extract({"traceparent": traceparent})
    ).get_span_context()
    return span_ctx if span_ctx.is_valid else None


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


async def _typing_keepalive(ctx: commands.Context) -> None:
    try:
        async with ctx.typing():
            await asyncio.sleep(3600)  # held open until cancelled
    # Exception only: background_typing() cancels this, and swallowing the
    # CancelledError would mark the task completed and stall a shutdown here.
    except Exception:
        pass  # cosmetic — never let typing failures surface


@contextlib.asynccontextmanager
async def background_typing(ctx: commands.Context) -> AsyncGenerator[None]:
    """Non-blocking ctx.typing(): the POST runs in a background task so the body
    starts immediately. The whole Typing CM lives inside that task."""
    task = asyncio.create_task(_typing_keepalive(ctx))
    try:
        yield
    finally:
        task.cancel()


def refund_cooldown(ctx: commands.Context) -> None:
    """Hand back the cooldown discord.py charged in prepare(). The guard is for
    the checker: inside a command body `ctx.command` is never None."""
    if ctx.command is not None:
        ctx.command.reset_cooldown(ctx)


def user_facing_reason(e: BaseException, fallback: str) -> str:
    """The one line a member may see for `e`: its `user_message` when the type
    vets its own copy (ExtractionError, PlaylistInputError, the Spotify errors),
    else `fallback`. A raw message can carry yt-dlp's bug-report boilerplate or
    an endpoint; the full text still reaches the span and the logs."""
    message = getattr(e, "user_message", None)
    if isinstance(message, str) and message:
        return message
    return fallback


def record_span_error(span: Span, e: Exception) -> None:
    """Record an exception on a span and mark its status as ERROR."""
    span.record_exception(e)
    span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")


def notice_embed(
    message: str,
    color: Optional[discord.Color] = None,
    *,
    title: Optional[str] = None,
) -> discord.Embed:
    """A plain status string as an embed. Every command response must be one:
    MusicContext.send prepends the Now Playing block, and a bare `content`
    string would render as loose text above it."""
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


def first_sendable_channel(
    guild: discord.Guild,
) -> Optional[discord.TextChannel]:
    """A text channel the bot may post in — the system channel when it qualifies,
    else the first that does. None when the bot is not in the member cache or
    can post nowhere; callers stay silent, these notices are advisory."""
    if guild.me is None:
        return None
    if (
        guild.system_channel is not None
        and guild.system_channel.permissions_for(guild.me).send_messages
    ):
        return guild.system_channel
    return next(
        (
            ch
            for ch in guild.text_channels
            if ch.permissions_for(guild.me).send_messages
        ),
        None,
    )


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


# Discord's hard limits: an over-length title, footer or field value 400s the
# whole send(). The field cap matters most for lists a user can grow (removed
# songs), where the 400 lands after the command has already mutated state.
EMBED_TITLE_LIMIT = 256
FOOTER_LIMIT = 2048
EMBED_FIELD_LIMIT = 1024

# One row of a multi-row field of user-supplied titles; ten share one field.
ECHO_ROW_MAX = 70
# One title rendered inline in a notice's text — bold, or beside a link.
INLINE_TITLE_MAX = 200


# Control characters end a rendered embed line early; they have no visible form.
_LABEL_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")


def safe_label(text: str, limit: int) -> str:
    """Attacker-influenceable text (a search term, a yt-dlp title) rendered into
    an embed without styling it or forging a link. Three neutralizations
    `escape_markdown` lacks: `[`/`]` pick a masked link's label, a backtick
    closes the caller's code span, and `ignore_links` defaults to True. Cap
    BEFORE escaping: cutting after can split an escape pair."""
    flattened = _LABEL_UNSAFE.sub(" ", text)
    clipped = truncate(flattened, limit)
    neutralized = clipped.replace("[", "(").replace("]", ")").replace("`", "'")
    return discord.utils.escape_markdown(neutralized, ignore_links=False)


def truncate(text: str, limit: int) -> str:
    """Clip to `limit` characters, ellipsizing if clipped."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def truncate_embed_title(title: str) -> str:
    """Clip a title to Discord's embed-title limit, ellipsizing if clipped."""
    return truncate(title, EMBED_TITLE_LIMIT)


# Debug mode's suffix starts a line of its own. See docs/ARCHITECTURE.md#debug-footer-seams.
FOOTER_SUFFIX_SEP = "\n"


def join_footer(base: str, suffix: str) -> str:
    """`base` and `suffix` as one footer, the suffix on a line of its own. The
    clip falls on the base; a suffix leaving no room drops the base whole
    (truncate() to a limit of zero would return a lone ellipsis)."""
    if not (base and suffix):
        return truncate(suffix or base, FOOTER_LIMIT)
    room = FOOTER_LIMIT - len(suffix) - len(FOOTER_SUFFIX_SEP)
    if room <= 0:
        return truncate(suffix, FOOTER_LIMIT)
    return f"{truncate(base, room)}{FOOTER_SUFFIX_SEP}{suffix}"


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


log = get_logger(__name__)
