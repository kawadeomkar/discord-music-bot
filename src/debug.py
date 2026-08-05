"""Debug mode: the footer every reply grows while it is on.

OBSERVATION-ONLY, and that is the whole design constraint. Nothing here changes
playback, caching, queueing or persistence — only what the bot shows. It is what
keeps "test with debug on, ship with debug off" a valid methodology: nothing you
validated changes when the toggle flips.
"""

from collections.abc import Sequence
from typing import Optional

import discord
from opentelemetry import trace

from src.util import get_logger, trace_id_of, truncate

log = get_logger(__name__)

# Discord's hard cap on an embed's footer text.
FOOTER_LIMIT = 2048


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 · FOOTER DECORATION
# ════════════════════════════════════════════════════════════════════════════
# What debug mode actually does to ordinary traffic: every command response grows
# a footer identifying the request. The trace id is the point — it is already the
# join key for every log line and span, so pasting one out of Discord finds the
# exact request in Loki/Tempo. -ping is not decorated: it bypasses this seam.

_DEBUG_MARK = "🐞"


def debug_footer(
    *,
    span: Optional[trace.Span] = None,
    elapsed_ms: Optional[float] = None,
    shard_id: Optional[int] = None,
    skip_trace: bool = False,
) -> str:
    """The debug suffix, or "" when nothing is known worth showing.

    Every part is optional because every part has an absent case: a send outside
    any command has no elapsed time, a DM has no shard, and an unsampled span has
    no trace id.
    """
    parts: list[str] = []
    if elapsed_ms is not None:
        parts.append(f"{round(elapsed_ms)} ms")
    if shard_id is not None:
        parts.append(f"shard {shard_id}")
    if not skip_trace and span is not None and (trace_id := trace_id_of(span)):
        parts.append(f"trace {trace_id}")
    if not parts:
        return ""
    return f"{_DEBUG_MARK} " + " · ".join(parts)


def decorate_embeds(
    embeds: Sequence[discord.Embed],
    *,
    span: Optional[trace.Span] = None,
    elapsed_ms: Optional[float] = None,
    shard_id: Optional[int] = None,
) -> None:
    """Append the debug footer to each embed, IN PLACE.

    Mutating is safe — embeds are freshly constructed per response everywhere in
    this codebase — and it is what lets MusicContext.send decorate both of its send
    paths without either of them reshaping its kwargs.
    """
    for embed in embeds:
        existing = embed.footer.text or ""
        suffix = debug_footer(
            span=span,
            elapsed_ms=elapsed_ms,
            shard_id=shard_id,
            # Error embeds already carry one from _command_error. The same id twice
            # in one footer reads as two different traces.
            skip_trace="trace:" in existing or "trace " in existing,
        )
        if not suffix:
            continue
        text = f"{existing} · {suffix}" if existing else suffix
        embed.set_footer(
            text=truncate(text, FOOTER_LIMIT), icon_url=embed.footer.icon_url
        )
