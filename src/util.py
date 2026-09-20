import asyncio
import contextlib
import re
import time
from typing import Any, Final, Optional
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator

import discord
import structlog
from discord.ext import commands
from opentelemetry.trace import Span, SpanContext, StatusCode, get_current_span
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)


# (done_so_far, total_if_known) — what a long resolve reports as it walks. `done`
# is ABSOLUTE, never a delta, so a dropped report self-corrects on the next one.
# MAY BE CALLED OFF THE EVENT LOOP: Spotify calls it on the loop, the yt-dlp
# transport from YtdlpPool's drain thread, so an implementation must be
# synchronous and non-blocking. Declared here because spotify.py and youtube.py
# know nothing about guilds and must not import what renders it.
ProgressFn = Callable[[int, Optional[int]], None]


# Discord's embed field-value cap; it rejects the WHOLE send past it.
_FIELD_VALUE_MAX = 1024
_TRUNCATION_MARK = "..."


# One MORE than queue_message renders. It appends its mark only while
# `len(lines) < len(songs)`, so a caller that slices before escaping hands it
# this many: the eleventh row is never rendered and exists only to be counted.
QUEUE_MESSAGE_ROWS_PLUS_ONE = 11


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


async def set_within(event: asyncio.Event, secs: float) -> bool:
    """True when `event` was set inside `secs`. A cancellation aimed at this
    coroutine still propagates — asyncio.timeout only converts its own."""
    try:
        async with asyncio.timeout(secs):
            await event.wait()
    except TimeoutError:
        return False
    return True


async def set_when_set(source: asyncio.Event, target: asyncio.Event) -> None:
    """Set `target` once `source` is set. Run as a task and cancelled by its
    owner; it holds nothing, so an unawaited cancel is enough."""
    await source.wait()
    target.set()


async def join_task(task: asyncio.Task[Any]) -> None:
    """Wait out a task that was told to stop by a SIGNAL rather than cancelled,
    swallowing what it raises — never this coroutine's own cancellation, for the
    reason cancel_task spells out. Cancelling instead would orphan whatever the
    task cleans up in its own finally.

    Shielded, because a bare `await task` is itself a cancellation vector into
    that task: Task.cancel() cancels whatever the task is waiting ON, and
    awaiting a task directly makes it the _fut_waiter. So a caller cancelled
    here would cancel the very task it is waiting out, mid-cleanup."""
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
    except Exception as e:
        log.warning(f"joined background task failed: {e!r}")


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


class PoolSlotUnavailable(Exception):
    """A `pool_slot` context manager could not be entered in time.

    The contract between whoever owns a slot and the extraction that holds one.
    The bound belongs to the CALLER that supplied the slot, never to the shared
    job that caller happened to start, so the extraction single-flight re-elects
    a leader on this rather than failing every joiner: a joiner holds no worker
    of its own, and may have supplied no slot at all.
    See docs/ARCHITECTURE.md#where-the-resolve-bound-is-taken."""


# One transient message per channel per KIND, exclusive: a card and a
# slow-resolve notice are different kinds and coexist, but sixteen cards do not.
# PLAY_INFLIGHT_MAX is 16 and requests resolve concurrently, and discord.py
# sleeps a throttled channel bucket internally — so the cost lands on the
# confirmations, not here. See docs/ARCHITECTURE.md#queue-progress-card.
_CLAIMED_CHANNELS: dict[str, set[int]] = {}


@contextlib.contextmanager
def channel_claim(kind: str, channel_id: int) -> Iterator[bool]:
    """Take exclusive use of a channel for one kind of transient message.

    Yields False when another holder has it, and then holds nothing: the loser
    says nothing rather than queueing behind the winner, because two identical
    messages tell the user less than one. Claim at the moment of SENDING, never
    at entry — a request that settles inside its own delay never sends, and must
    not hold the slot a slow sibling needs."""
    held = _CLAIMED_CHANNELS.setdefault(kind, set())
    claimed = channel_id not in held
    if claimed:
        held.add(channel_id)
    try:
        yield claimed
    finally:
        if claimed:
            held.discard(channel_id)


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


def refund_cooldown(ctx: commands.Context) -> None:
    """Hand back the cooldown discord.py charged in prepare(). The guard is for
    the checker: inside a command body `ctx.command` is never None."""
    if ctx.command is not None:
        ctx.command.reset_cooldown(ctx)


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


# Square emoji blocks: the done portion renders in a different colour from the
# remainder. Width is low because each block glyph is much wider than a dash.
# Public, because the queue-progress card quantizes its count against it.
BAR_WIDTH = 10
_BAR_FILL_DONE = "🟦"
_BAR_FILL_REMAINING = "⬜"
_BAR_HEAD = "🔘"


def progress_bar(ratio: float, *, width: int = BAR_WIDTH) -> str:
    """The glyph run for a ratio, clamped to 0..1: filled cells, then a head marking
    the position reached, then what remains. A width of 0 renders nothing."""
    if width <= 0:
        return ""
    ratio = max(0.0, min(1.0, ratio))
    # One short of the end, so the head has a cell at 100% and never renders with
    # filled cells behind it.
    filled = min(width - 1, int(ratio * width))
    return (
        _BAR_FILL_DONE * filled + _BAR_HEAD + _BAR_FILL_REMAINING * (width - filled - 1)
    )


def progress_line(
    position: float,
    total: float,
    *,
    label: Callable[[float], str],
    width: int = BAR_WIDTH,
) -> str:
    """`label(position)` <bar> `label(total)` — the Now Playing bar, for anything
    with a position and a known end. The position is clamped to 0..total before it
    is labelled, so the left label never reads past the right one. "" without a
    positive total."""
    if total <= 0:
        return ""
    position = max(0.0, min(position, total))
    bar = progress_bar(position / total, width=width)
    return f"`{label(position)}` {bar} `{label(total)}`"


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """The noun form matching `count`: pluralize(1, "song") → "song",
    pluralize(3, "song") → "songs". `plural` overrides the default `+ "s"`."""
    if count == 1:
        return singular
    return plural if plural is not None else singular + "s"


def fmt_seconds(secs: float) -> str:
    """Seconds as the shortest text that reads back as the same float: 3.0 → "3s",
    0.25 → "0.25s". repr() is that shortest form; only its trailing ".0" goes."""
    return repr(float(secs)).removesuffix(".0") + "s"


# Discord's hard limits: an over-length title, footer or field value 400s the
# whole send(). The field cap matters most for lists a user can grow (removed
# songs), where the 400 lands after the command has already mutated state.
EMBED_TITLE_LIMIT = 256
FOOTER_LIMIT = 2048
EMBED_FIELD_LIMIT = 1024

# Bound on one echoed needle, which owns a field to itself. Discord renders
# markdown in field values, so what a user typed goes through safe_label first.
ECHO_MAX = 200

# One row of a multi-row field of user-supplied titles; ten share the budget one
# echoed needle gets.
ECHO_ROW_MAX = 70


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


def verbatim_code(text: str, limit: int) -> Optional[str]:
    """`text` as an inline code span a user can copy back exactly, or None when no
    span can hold it: past `limit`, or carrying a backtick or a control character.
    Unescaped because Discord shows a code span's content literally, so
    safe_label's backslashes would be copied along with the text."""
    if len(text) > limit or "`" in text or _LABEL_UNSAFE.search(text):
        return None
    return f"`{text}`"


def truncate(text: str, limit: int) -> str:
    """Clip to `limit` characters, ellipsizing if clipped."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def truncate_embed_title(title: str) -> str:
    """Clip a title to Discord's embed-title limit, ellipsizing if clipped."""
    return truncate(title, EMBED_TITLE_LIMIT)


# Every dash Unicode offers that a keyboard or a paste substitutes for ASCII `-`.
# iOS turns a typed `--` into a single em dash.
DASHES: Final[str] = "-‐‑‒–—―−"


def codeblock_fields(name: str, lines: list[str]) -> list[tuple[str, str]]:
    """Lines as one or more codeblock fields within Discord's 1024-char field
    cap. Splits rather than truncates: a silently clipped config listing reads
    as a complete one."""
    fence = 8  # "```\n" + "\n```"
    fields: list[tuple[str, str]] = []
    chunk: list[str] = []
    size = 0
    for line in lines:
        line = truncate(line, EMBED_FIELD_LIMIT - fence)
        if chunk and size + len(line) + 1 + fence > EMBED_FIELD_LIMIT:
            fields.append((name if not fields else f"{name} (cont.)", _fence(chunk)))
            chunk, size = [], 0
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        fields.append((name if not fields else f"{name} (cont.)", _fence(chunk)))
    return fields


def _fence(lines: list[str]) -> str:
    return "```\n" + "\n".join(lines) + "\n```"


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


# How long a failed owner lookup is remembered. Each lookup is an
# application_info() REST call, which discord.py retries for up to ~25s on a 5xx.
OWNER_LOOKUP_RETRY_SECS: Final[float] = 60.0
_owner_lookup_retry_at = 0.0


def owner_lookup_backing_off() -> bool:
    """True while a failed owner lookup is remembered, when is_operator answers
    False without asking: a reply can say the operator could not be confirmed."""
    return time.monotonic() < _owner_lookup_retry_at


async def is_operator(ctx: commands.Context) -> bool:
    """Is the caller the bot's operator (discord.py's is_owner)? Fails CLOSED and
    never raises. is_owner() looks the owner up with application_info() unless a
    lookup already succeeded, and RAISES when that fails;
    a failure answers False for OWNER_LOOKUP_RETRY_SECS without asking again."""
    global _owner_lookup_retry_at
    if owner_lookup_backing_off():
        return False
    try:
        return await ctx.bot.is_owner(ctx.author)
    except Exception as e:  # noqa: BLE001 — an unreachable owner is not an owner
        _owner_lookup_retry_at = time.monotonic() + OWNER_LOOKUP_RETRY_SECS
        log.warning(
            f"owner check failed, denying for {OWNER_LOOKUP_RETRY_SECS:.0f}s: "
            f"{type(e).__name__}: {e}"
        )
        return False
