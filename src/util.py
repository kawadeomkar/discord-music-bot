import asyncio
import contextlib
import re
from typing import Any, Final, Optional
from collections.abc import AsyncGenerator, Coroutine

import discord
import structlog
from discord.ext import commands
from opentelemetry.trace import Span, StatusCode


# Discord's embed field-value cap; it rejects the WHOLE send past it. The -remove
# reply lands after the songs are gone, so a rejected send reports a false failure.
_FIELD_VALUE_MAX = 1024
_TRUNCATION_MARK = "..."


def queue_message(songs: list[str]) -> str:
    """Numbered song list for an embed field, bounded by count AND by length: ten
    100-char titles compose to 1040. Either overflow ends in the same trailing
    mark; the count is in the embed's title."""
    lines: list[str] = []
    used = 0
    budget = _FIELD_VALUE_MAX - len(_TRUNCATION_MARK) - 1
    for i, song in enumerate(songs[:10]):
        line = f"{i + 1}: {song}"
        if used + len(line) > budget:
            # A single line over budget is truncated, not dropped: an empty
            # field would say nothing about what was taken.
            if not lines:
                lines.append(line[:budget])
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(songs):
        lines.append(_TRUNCATION_MARK)
    return "\n".join(lines)


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
    """Cancel `task` and wait for it, swallowing ITS CancelledError — not the
    caller's. A cancellation aimed at this coroutine (a place timeout, a teardown)
    arrives at the same await; swallowed, asyncio.timeout has nothing to convert
    and the caller runs past its bound. cancelling() counts cancel() calls against
    the CURRENT task, which only an outside canceller made."""
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise


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
    # Exception only, not CancelledError: background_typing() cancels this on the way
    # out, and letting that propagate is what marks the task genuinely cancelled
    # rather than completed. Swallowing it would stop a shutdown at this frame.
    except Exception:
        pass  # cosmetic — never let typing failures surface


# Refcounted per channel: concurrent -play requests enter this once per link, and
# typing is a CHANNEL state — N keepalives would POST the same route N times.
_TYPING_HOLDS: dict[int, int] = {}
_TYPING_TASKS: dict[int, asyncio.Task[None]] = {}


@contextlib.asynccontextmanager
async def background_typing(ctx: commands.Context) -> AsyncGenerator[None]:
    """Non-blocking ctx.typing(): the first POST /typing runs in a background task
    so the command body starts immediately; the keepalive is cancelled when the
    last holder finishes, so it does not blink off when the first of several
    concurrent commands returns. The whole CM lives inside the task."""
    key = ctx.channel.id
    _TYPING_HOLDS[key] = _TYPING_HOLDS.get(key, 0) + 1
    if key not in _TYPING_TASKS:
        _TYPING_TASKS[key] = asyncio.create_task(_typing_keepalive(ctx))
    try:
        yield
    finally:
        _TYPING_HOLDS[key] -= 1
        if _TYPING_HOLDS[key] <= 0:
            del _TYPING_HOLDS[key]
            task = _TYPING_TASKS.pop(key, None)
            if task is not None:
                task.cancel()


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
    """Turn a plain status string ("Shuffled!", validation errors) into an embed. Every
    command response must be an embed: MusicContext.send prepends the Now Playing block,
    and a bare `content` string would render as loose text above it. send_embed is the
    pair for anything needing a title/description split."""
    return discord.Embed(title=title, description=message, color=color)


def build_embed(
    title: str,
    description: str,
    color: Optional[discord.Color] = None,
    footer: Optional[str] = None,
    thumbnail: Optional[str] = None,
    fields: Optional[list[tuple[str, str, bool]]] = None,
) -> discord.Embed:
    """The embed send_embed sends, for a caller that builds its reply in one place
    and sends it from another."""
    embed = discord.Embed(title=title, description=description, color=color)
    if footer:
        embed.set_footer(text=footer)
    if thumbnail:
        embed.set_thumbnail(url=thumbnail)
    for name, value, inline in fields or []:
        embed.add_field(name=name, value=value, inline=inline)
    return embed


async def send_embed(
    destination: discord.abc.Messageable,
    title: str,
    description: str,
    color: Optional[discord.Color] = None,
    footer: Optional[str] = None,
    thumbnail: Optional[str] = None,
    fields: Optional[list[tuple[str, str, bool]]] = None,
) -> discord.Message:
    return await destination.send(
        embed=build_embed(title, description, color, footer, thumbnail, fields)
    )


def first_sendable_channel(
    guild: discord.Guild,
) -> Optional[discord.TextChannel]:
    """A text channel in `guild` the bot may post in — the system channel when it
    qualifies, else the first that does. For notices with no channel of their own,
    e.g. telling a guild that the channels it was playing in were deleted.

    None when the bot is not in the guild's member cache or can post nowhere; the
    caller stays silent rather than raising, since these messages are advisory."""
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


# Discord's hard embed-title limit. An over-length title 400s the whole send(),
# silently no-opping -history or failing the now-playing send/edit.
EMBED_TITLE_LIMIT = 256

# The same, for footer text. Lives here, not in debug.py: ping.py writes footers too
# and debug.py already imports ping.py, so importing it back closes a hard cycle.
FOOTER_LIMIT = 2048

# The same again, for a field VALUE. A field built from a list the user can grow
# (removed songs, dropped positions) has no natural ceiling, and the 400 lands
# after the command has already mutated state.
EMBED_FIELD_LIMIT = 1024


# Control characters end a rendered embed line early, hiding whatever follows.
# Flattened rather than escaped — they have no visible form.
_LABEL_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")


def safe_label(text: str, limit: int) -> str:
    """Attacker-influenceable text — a user's search term, a yt-dlp title, an
    uploader name — rendered into an embed without being able to style it or
    forge a link.

    Three neutralizations before the escape, because `escape_markdown` covers none
    of them: `[`/`]` are not in its set and are what picks a masked link's label; a
    backtick closes any code span the caller wrapped this in; and `ignore_links`
    defaults to TRUE, passing a whole http(s) token through untouched.

    Cap BEFORE escaping: cutting after can split an escape pair and leave a
    trailing backslash that eats the next character."""
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


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


log = get_logger(__name__)
