import asyncio
import contextlib
import re
from typing import Any, Final, Optional
from collections.abc import AsyncGenerator, Coroutine

import discord
import structlog
from discord.ext import commands
from opentelemetry.trace import Span, StatusCode


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


async def _typing_keepalive(ctx: commands.Context) -> None:
    try:
        async with ctx.typing():
            await asyncio.sleep(3600)  # held open until cancelled
    # Exception only, not CancelledError: background_typing() cancels this on the way
    # out, and letting that propagate is what marks the task genuinely cancelled
    # rather than completed. Swallowing it would stop a shutdown at this frame.
    except Exception:
        pass  # cosmetic — never let typing failures surface


@contextlib.asynccontextmanager
async def background_typing(ctx: commands.Context) -> AsyncGenerator[None]:
    """Non-blocking ctx.typing(): the first POST /typing runs in a background task so
    the command body starts immediately, and the keepalive is cancelled when the body
    finishes. The whole CM lives inside the task — never enter/exit Typing manually
    across tasks."""
    task = asyncio.create_task(_typing_keepalive(ctx))
    try:
        yield
    finally:
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


async def send_embed(
    destination: discord.abc.Messageable,
    title: str,
    description: str,
    color: Optional[discord.Color] = None,
    footer: Optional[str] = None,
    thumbnail: Optional[str] = None,
    fields: Optional[list[tuple[str, str, bool]]] = None,
) -> discord.Message:
    # Clamped here, the last point before Discord — see EMBED_DESCRIPTION_LIMIT.
    embed = discord.Embed(
        title=truncate(title, EMBED_TITLE_LIMIT),
        description=truncate_escaped(description, EMBED_DESCRIPTION_LIMIT),
        color=color,
    )
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
# after the command has already mutated state — so clamp at the boundary rather
# than trusting the caller's arithmetic about how long its inputs can get.
EMBED_FIELD_LIMIT = 1024

# The same again, for a DESCRIPTION. Enforced in send_embed rather than at each
# caller for the reason _field exists: the three list-built descriptions (a
# queued playlist, a queued collection, a cleared queue) are ten escaped rows
# wide, so nothing but the row-width constant keeps them under this — and the
# 400 lands after the enqueue or the clear has already committed, telling the
# user the command failed for work that happened.
EMBED_DESCRIPTION_LIMIT = 4096


# Control characters end a rendered embed line early, which is how text can hide
# whatever follows it. Flattened rather than escaped — they have no visible form.
_LABEL_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")


def safe_label(text: str, limit: int) -> str:
    """Attacker-influenceable text — a user's search term, a yt-dlp title, an
    uploader name — rendered into an embed without being able to style it or
    forge a link.

    Three neutralizations, then the escape, because `escape_markdown` covers none
    of them: `[` and `]` are not in its set at all and are the half of a masked
    link that picks the label; a backtick closes any code span the caller wrapped
    this in, where a backslash means nothing; and `ignore_links` defaults to TRUE,
    passing an entire http(s) token through untouched.

    Cap BEFORE escaping — escaping first and cutting after can split an escape
    pair and leave a trailing backslash that eats the next character."""
    flattened = _LABEL_UNSAFE.sub(" ", text)
    clipped = truncate(flattened, limit)
    neutralized = clipped.replace("[", "(").replace("]", ")").replace("`", "'")
    return discord.utils.escape_markdown(neutralized, ignore_links=False)


def truncate(text: str, limit: int) -> str:
    """Clip to `limit` characters, ellipsizing if clipped."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def truncate_escaped(text: str, limit: int) -> str:
    """truncate(), for text that has already been through escape_markdown.

    A field or description built by joining N escaped rows has a length no single
    row's cap controls, so this clamp fires. A cut landing between a backslash and
    the character it escapes leaves an orphan that escapes the ellipsis instead,
    rendering the clip as a literal character. An ODD trailing run of backslashes
    is that orphan; an even run is escaped backslashes and is kept."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if (len(cut) - len(cut.rstrip("\\"))) % 2:
        cut = cut[:-1]
    return cut + "…"


def truncate_embed_title(title: str) -> str:
    """Clip a title to Discord's embed-title limit, ellipsizing if clipped."""
    return truncate(title, EMBED_TITLE_LIMIT)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


log = get_logger(__name__)
