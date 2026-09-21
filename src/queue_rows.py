"""One queued item as a row of text, and the ETA walk that runs down the rows.

Pure: no player, no queue, no Discord call. `-queue` and the queued-collection
card list items through here, so a listing reads the same either way. The
single-entry cards are their own renderer over the same fields — a row and a
card are different shapes. See docs/ARCHITECTURE.md#queue-rows.
"""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Optional, Union

import discord

from src.guild_queue import QueueItem
from src.util import fmt_duration, safe_label
from src.youtube import QueueObject

# Nothing bounds a yt-dlp title or uploader, and every row of a listing shares
# one 4096-character embed description.
ROW_TITLE_MAX = 200
ROW_BYLINE_MAX = 200
# Rows a listing shows before its "... and N more".
ROW_LIMIT = 10
# Characters a listing's rows may use. Ten rows at both caps, with their links,
# pass 4096 on their own, so the count is not the only bound.
ROWS_BUDGET = 3400


@dataclass(frozen=True, slots=True, kw_only=True)
class EtaWalk:
    """Accumulator for the queue's ETA walk; `now` is invariant across a walk and
    passed alongside. Frozen: advancing is `replace()` + rebind."""

    cumulative_secs: int
    uncertain: bool

    def advance(self, remaining: Optional[int]) -> EtaWalk:
        """The next walk state after an item whose remaining time is `remaining`,
        or None when its duration is unknown."""
        if remaining is None:
            return replace(self, uncertain=True)
        return replace(self, cumulative_secs=self.cumulative_secs + remaining)

    def advance_estimate(self, secs: int) -> EtaWalk:
        """The next state after an item whose length is someone else's figure: the
        seconds count, and every time after it is approximate."""
        return replace(
            self, cumulative_secs=self.cumulative_secs + secs, uncertain=True
        )


def fmt_total_duration(secs: int) -> str:
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    return " ".join(parts) or "0s"


def eta_at(now: datetime.datetime, secs: int) -> datetime.datetime:
    """`secs` of real time after `now`, in `now`'s zone. Adding a timedelta to an
    aware datetime is WALL-CLOCK arithmetic, so a span crossing a DST transition
    lands an hour out and renders a local time that may not exist."""
    return (now.astimezone(datetime.UTC) + datetime.timedelta(seconds=secs)).astimezone(
        now.tzinfo
    )


def fmt_clock_time(dt: datetime.datetime) -> str:
    """A wall-clock time with the zone it is in, read off the datetime."""
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    # tzname(), not strftime("%Z"): same output, ~50x cheaper, and this runs on
    # every NP tick. None is possible for a naive datetime, hence the `or ""`.
    return f"{hour}:{dt.minute:02d} {ampm} {dt.tzname() or ''}".rstrip()


def fmt_eta(est_dt: datetime.datetime, uncertain: bool) -> str:
    prefix = "~" if uncertain else ""
    return f"{prefix}**{fmt_clock_time(est_dt)}**"


def requester_mention(
    requester: Optional[Union[discord.User, discord.Member]],
) -> str:
    return requester.mention if requester else "Unknown"


def remaining_secs(item: QueueObject) -> Optional[int]:
    """A queued item's expected playtime: full duration, minus the resume offset
    for a resume entry, which plays only its tail."""
    if item.duration is None:
        return None
    if item.is_resume and item.ts:
        return max(0, item.duration - item.ts)
    return item.duration


def advance_walk(walk: EtaWalk, item: QueueItem) -> EtaWalk:
    """The walk after `item` has played. An unresolved search counts the length it
    carries, which is Spotify's and not the YouTube match's, as an estimate."""
    if isinstance(item, QueueObject):
        return walk.advance(remaining_secs(item))
    if item.duration is not None:
        return walk.advance_estimate(item.duration)
    return walk.advance(None)


def queue_runtime(items: Sequence[QueueItem]) -> tuple[int, bool]:
    """Total remaining playtime of queued items, and whether the total is
    approximate — set when a duration is unknown, and when one is Spotify's
    estimate rather than the played track's. Either way it renders with a "~"."""
    total_secs = 0
    partial = False
    for item in items:
        if isinstance(item, QueueObject):
            remaining = remaining_secs(item)
        else:
            # A search's length is an estimate: it counts, and the total says so.
            remaining = item.duration
            partial = True
        if remaining is not None:
            total_secs += remaining
        else:
            partial = True
    return total_secs, partial


def queue_row(
    item: QueueItem,
    index: int,
    *,
    now: datetime.datetime,
    walk: EtaWalk,
    byline: bool = True,
) -> str:
    """One listing row: `index`, the linked title, its length and the clock time
    `walk` gives it; with `byline`, a second line naming the channel and the
    requester. `walk` is the state BEFORE this item. An unresolved search renders
    the same row from its display fields (the artists stand in for the channel),
    and one that carries none is its search text and "resolving..."."""
    eta = fmt_eta(eta_at(now, walk.cumulative_secs), walk.uncertain)
    if isinstance(item, QueueObject):
        if item.is_resume and item.ts:
            note = f"  ·  ⏮ resumes at `{fmt_duration(item.ts)}`"
        elif item.ts:
            note = f"  ·  starts at `{item.ts}s`"
        else:
            note = ""
        who = requester_mention(item.requester)
        by = "Unknown channel"
    elif item.title:
        note = ""
        who = f"<@{item.requester_id}>" if item.requester_id else "Unknown"
        by = "Unknown artist"
    else:
        search = safe_label(
            (item.ytsearch or item.url or "?").removeprefix("ytsearch:"), ROW_TITLE_MAX
        )
        return f"`{index}` {search} · *resolving...*"
    # Capped and sanitized: a "]" in a masked link's label would close it early.
    title = safe_label(item.title or "", ROW_TITLE_MAX) or "Unknown"
    linked = (
        f"[**{title}**]({item.webpage_url})" if item.webpage_url else f"**{title}**"
    )
    dur = fmt_duration(item.duration) if item.duration is not None else "?:??"
    line = f"`{index}` {linked} · `{dur}`{note} · Est. playing at {eta}"
    if not byline:
        return line
    return f"{line}\n{safe_label(item.uploader or '', ROW_BYLINE_MAX) or by} · {who}"


def queue_rows(
    items: Sequence[QueueItem],
    *,
    first_index: int,
    now: datetime.datetime,
    walk: EtaWalk,
    byline: bool = True,
    limit: int = ROW_LIMIT,
    budget: int = ROWS_BUDGET,
) -> str:
    """The first rows of `items`, numbered from `first_index`, chaining the ETA
    walk from `walk`, then "... and N more" for the rest. Bounded by `limit` rows
    AND by `budget` characters, so a listing of long titles ends early rather
    than overflowing the description. Two-line rows are set apart by a blank
    line, one-line rows are not. "" for no items."""
    gap = "\n\n" if byline else "\n"
    rows: list[str] = []
    used = 0
    for offset, item in enumerate(items[:limit]):
        row = queue_row(item, first_index + offset, now=now, walk=walk, byline=byline)
        # The gap only exists between rows, so the first is charged for none: `used`
        # is exactly what join() will return.
        extra = len(row) if not rows else len(gap) + len(row)
        # The first row always shows: a listing of one row is never empty.
        if rows and used + extra > budget:
            break
        rows.append(row)
        used += extra
        walk = advance_walk(walk, item)
    more = len(items) - len(rows)
    if more > 0:
        rows.append(f"*... and {more} more*")
    return gap.join(rows)
