"""`-analytics` — the window allowlist, the Redis cache codec, and the embed.

Everything here is pure: it takes an `AnalyticsMetrics` and returns keys, dicts or an
embed. The command stays on the MusicBot cog, the same split `leaderboard.py` makes.

Every human-authored string the command shows renders HERE, never in the chart image:
the runtime image ships no system fonts and matplotlib's bundled face covers no CJK,
Thai or emoji. The embed gets Discord's full-Unicode stack, masked links and live
mentions. See docs/ARCHITECTURE.md#analytics-rendering.

Named `analytics_card` because `guild_state.Analytics` already exists — a per-song
enqueue stamp — and the two would sit one capital apart.
"""

import hashlib
import re
import time
from typing import Any, Final, Optional

import discord
import orjson
from discord.ext import commands

from src import analytics_render
from src.guild_state import (
    WAIT_UNAVAILABLE,
    AnalyticsMetrics,
    CompletionBucket,
    DailyPoint,
    DurationBucket,
    HeatCell,
    SourceCompletion,
    SourceDay,
    TopArtist,
    TopListener,
    TopSong,
)
from src.util import fmt_duration, pluralize, safe_label

# The four windows -analytics answers, and the whole of its key space. A closed key
# space is what makes the day-long TTL a rate bound: one recompute per guild per
# window per day, worst case. See docs/ARCHITECTURE.md#analytics-rendering.
ALLOWED_DAYS: Final[tuple[int, ...]] = (7, 30, 90, 365)
DEFAULT_DAYS: Final[int] = 30

# Three lists plus a topline share Discord's 4096-character description limit with
# the Now Playing block. Five keeps the worst case near 3,450; the arithmetic picks
# this constant, so redo it before raising it.
TOP_N: Final[int] = 5

# Bumped on any change to the cached shape. The codec defaults missing fields rather
# than rejecting them, so without this a rolling deploy would decode an old entry into
# a valid-looking card with wrong values.
_CACHE_VERSION: Final[int] = 1
# Bumped on any change to the FIGURE, independently of the aggregate: a layout change
# must not be served from an entry the previous layout produced.
_PNG_CACHE_VERSION: Final[int] = 1

_TITLE_MAX: Final[int] = 50
_NAME_MAX: Final[int] = 40
_URL_MAX: Final[int] = 150
# A paren, whitespace or control character inside a masked-link URL ends the markdown
# early. Sibling of leaderboard.py's `linkable`, which is private to that module; keep
# the two in step.
_LABEL_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")

_DAY_SECS: Final[int] = 86400

# The chart image's attachment name. One constant because it is written in three
# places that must agree — the File, the embed's set_image, and the test that pins
# the pair — and Discord resolves attachment:// by exact filename.
IMAGE_FILENAME: Final[str] = "analytics.png"


class AnalyticsFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    days: int = DEFAULT_DAYS


def cache_key(guild_id: int, days: int) -> str:
    """Keyed by window, and by nothing else. There is deliberately no `{tz}`
    component: the query takes no zone parameter, so a second key dimension would
    be meaningless. Worst case per guild is four windows."""
    return f"analytics:agg:v{_CACHE_VERSION}:{guild_id}:{days}"


def png_cache_key(guild_id: int, days: int, digest: str) -> str:
    """Keyed to a digest of the aggregate the PNG was rendered from, and to the
    renderer's own version, so a stale PNG misses rather than pairing an old chart
    with fresh numbers."""
    return (
        f"analytics:png:v{_PNG_CACHE_VERSION}.{analytics_render.RENDER_VERSION}"
        f":{guild_id}:{days}:{digest}"
    )


def cache_ttl_secs(metrics: AnalyticsMetrics, now: Optional[float] = None) -> int:
    """Seconds until the next UTC midnight, when the answer changes.

    The window is N complete days, so the artifact is immutable for its whole life.
    Derived from the query's own clock read, so a non-positive result means the day
    turned while it ran and the caller skips the write."""
    now = time.time() if now is None else now
    if metrics.today_start_epoch <= 0:
        return 0
    return int(metrics.today_start_epoch + _DAY_SECS - now)


# The cache wire format, spelled out once: field -> record type and the names that
# reach Redis, so an attribute rename breaks the decode into a miss.
_WIRE: Final[dict[str, tuple[type, tuple[str, ...]]]] = {
    "daily": (DailyPoint, ("day", "plays", "listen_secs")),
    "daily_by_source": (SourceDay, ("day", "source", "plays")),
    "heat": (HeatCell, ("dow", "hour", "plays")),
    "completion": (CompletionBucket, ("source", "bucket", "plays")),
    "durations": (DurationBucket, ("minutes", "plays")),
    "source_completion": (
        SourceCompletion,
        ("source", "played_secs", "duration_secs"),
    ),
    "top_listeners": (
        TopListener,
        ("requester_id", "requester_name", "plays", "played_secs"),
    ),
    "top_artists": (TopArtist, ("uploader", "plays", "played_secs")),
    "top_songs": (
        TopSong,
        ("title", "webpage_url", "query_source", "plays", "played_secs"),
    ),
}

# The scalar half, same rule, each with the callable that coerces it on the way back
# in — so a version skew is a cache miss rather than a later TypeError.
_SCALARS: Final[dict[str, Any]] = {
    "days": int,
    "window_start_epoch": float,
    "window_end_epoch": float,
    "today_start_epoch": float,
    "bucket_unit": str,
    "archived_days": int,
    "plays": int,
    "listen_secs": int,
    "unique_songs": int,
    "unique_listeners": int,
    "unique_artists": int,
    "wait_p50_secs": float,
    "livestream_plays": int,
}


def to_cache(metrics: AnalyticsMetrics) -> dict:
    """Plain dicts and lists for orjson."""
    out: dict[str, Any] = {k: getattr(metrics, k) for k in _SCALARS}
    out["wait_pcts"] = list(metrics.wait_pcts)
    for key, (_, fields) in _WIRE.items():
        out[key] = [
            {f: getattr(row, f) for f in fields} for row in getattr(metrics, key)
        ]
    return out


def from_cache(raw: object) -> Optional[AnalyticsMetrics]:
    """Rebuild a cached AnalyticsMetrics. None means MALFORMED, never "empty": an
    empty window is a valid cached value, and caching it is what stops an idle guild
    re-querying Postgres on every invocation. Do not test truthiness."""
    if not isinstance(raw, dict):
        return None
    try:
        kwargs: dict[str, Any] = {k: cast(raw[k]) for k, cast in _SCALARS.items()}
        kwargs["wait_pcts"] = tuple(float(v) for v in raw.get("wait_pcts", []))
        for key, (cls, fields) in _WIRE.items():
            rows = raw.get(key, [])
            if not isinstance(rows, list):
                return None
            kwargs[key] = tuple(cls(**{f: r[f] for f in fields}) for r in rows)
        return AnalyticsMetrics(**kwargs)
    except KeyError, TypeError, ValueError:
        return None


def aggregate_digest(metrics: AnalyticsMetrics) -> str:
    """A short, stable fingerprint of the aggregate, for the PNG key.

    Sorted keys so the digest depends on the VALUES rather than on dict insertion
    order — a cache hit rebuilds the metrics through from_cache, and the two paths
    must agree or every hit would re-render. blake2b at 8 bytes: this is a cache
    discriminator, not a security boundary, and 16 hex characters keep the key short.
    """
    blob = orjson.dumps(to_cache(metrics), option=orjson.OPT_SORT_KEYS)
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def resolve_days(requested: int) -> Optional[int]:
    """The requested window, or None when it is not one of the four.

    A rejection must never reach Postgres and must never take a read slot, so this
    runs before anything else in the command body."""
    return requested if requested in ALLOWED_DAYS else None


def _linkable(url: str) -> bool:
    return bool(
        url
        and len(url) <= _URL_MAX
        and not any(c in url for c in "() \t")
        and not _LABEL_UNSAFE.search(url)
    )


def _fmt_wait(secs: float) -> str:
    """A queue wait, or the reason there isn't one. WAIT_UNAVAILABLE is not 0s: every
    queued_at in the window was the epoch-0 backfill sentinel, and a confident "0s"
    would read as "songs play instantly here"."""
    if secs <= WAIT_UNAVAILABLE:
        return "n/a"
    return f"{secs:.0f}s" if secs < 60 else fmt_duration(int(secs))


def _line_listener(rank: int, t: TopListener, guild: Optional[discord.Guild]) -> str:
    """A mention while the listener is still in the guild, their archived name once
    they leave — Discord renders a mention for a non-member as a raw id."""
    who = f"<@{t.requester_id}>"
    if guild is not None and guild.get_member(t.requester_id) is None:
        who = safe_label(t.requester_name, _NAME_MAX) or "unknown"
    return (
        f"**{rank}.** {who} — {fmt_duration(t.played_secs)} · "
        f"{t.plays} {pluralize(t.plays, 'song')}"
    )


def _line_artist(rank: int, t: TopArtist) -> str:
    name = safe_label(t.uploader, _NAME_MAX) or "Unknown"
    return (
        f"**{rank}.** {name} — {fmt_duration(t.played_secs)} · "
        f"{t.plays} {pluralize(t.plays, 'play')}"
    )


def _line_song(rank: int, t: TopSong) -> str:
    # A blank title is a real archived value (the zero-value convention), and an
    # empty masked-link label renders as an invisible link.
    title = safe_label(t.title, _TITLE_MAX) or "Unknown"
    label = f"[{title}]({t.webpage_url})" if _linkable(t.webpage_url) else title
    return (
        f"**{rank}.** {label} — {fmt_duration(t.played_secs)} · "
        f"{t.plays} {pluralize(t.plays, 'play')}"
    )


def window_label(metrics: AnalyticsMetrics) -> str:
    """What the title says the card covers: the requested window, and the days with
    plays beside it when they differ.

    FlagConverter silently defaults an input it does not recognise, so the period is
    named where the answer is read."""
    period = metrics.period_label
    if 0 < metrics.archived_days < metrics.days:
        # "with plays", not "archived": first_play is min() over the window SLICE, so
        # a guild archiving for a year that went quiet through July reports 5 here.
        # Coverage would need a probe outside the window; this is what the number is.
        return (
            f"{period} ({metrics.archived_days} "
            f"{pluralize(metrics.archived_days, 'day')} with plays)"
        )
    return period


def _topline(m: AnalyticsMetrics) -> str:
    per = "week" if m.bucket_unit == "week" else "day"
    buckets = max(len(m.daily), 1)
    # Rounded, not floored: 59 plays over 30 active days is 2 a day, and // renders
    # it as "1 plays" -- understating by a third and disagreeing with its own noun.
    avg = round(m.plays / buckets)
    return (
        f"**{m.plays}** {pluralize(m.plays, 'play')} · "
        f"**{fmt_duration(m.listen_secs)}** listened · "
        f"**{m.unique_songs}** unique {pluralize(m.unique_songs, 'song')} · "
        f"**{m.unique_listeners}** {pluralize(m.unique_listeners, 'listener')} · "
        f"**{m.unique_artists}** {pluralize(m.unique_artists, 'artist')}\n"
        f"Median queue wait **{_fmt_wait(m.wait_p50_secs)}** · "
        f"**{avg}** {pluralize(avg, 'play')} on the average active {per}"
    )


def build_embed(
    metrics: AnalyticsMetrics,
    *,
    guild: Optional[discord.Guild] = None,
    image_filename: Optional[str] = None,
    chart_note: Optional[str] = None,
) -> discord.Embed:
    """The card. `image_filename` attaches the chart; without it this is the
    embed-only build — which is both what P2 ships and the permanent fallback for
    every way the render can fail, so the numbers survive a chart that does not.

    Sections go in the DESCRIPTION, not in fields: a field value caps at 1024
    characters and five masked-link lines do not reliably fit, while the 4096-char
    description does. leaderboard.py records the same finding.
    """
    sections = [_topline(metrics)]
    if metrics.top_listeners:
        sections.append(
            "**Top listeners**\n"
            + "\n".join(
                _line_listener(i, t, guild)
                for i, t in enumerate(metrics.top_listeners, start=1)
            )
        )
    if metrics.top_artists:
        sections.append(
            "**Top artists**\n"
            + "\n".join(
                _line_artist(i, t) for i, t in enumerate(metrics.top_artists, start=1)
            )
        )
    if metrics.top_songs:
        sections.append(
            "**Top songs**\n"
            + "\n".join(
                _line_song(i, t) for i, t in enumerate(metrics.top_songs, start=1)
            )
        )
    embed = discord.Embed(
        title=f"📊 Analytics — {window_label(metrics)}",
        description="\n\n".join(sections),
        color=discord.Color.blurple(),
    )
    if image_filename:
        embed.set_image(url=f"attachment://{image_filename}")
    embed.set_footer(text=footer_text(metrics, chart_note))
    return embed


def footer_text(metrics: AnalyticsMetrics, note: Optional[str] = None) -> str:
    """Times are UTC and the window ends yesterday — both stated, because both are
    surprising. A UTC day boundary is 17:00 in US/Pacific, so an evening's listening
    can fall across two daily bars, and a guild whose only plays are from this
    morning is empty to this command."""
    ends = "the last complete week" if metrics.bucket_unit == "week" else "yesterday"
    text = (
        f"Long-term archive · times in UTC · window ends {ends}, "
        "so today is not included."
    )
    return f"{text}\n{note}" if note else text


def empty_notice(days: int, bucket_unit: str) -> str:
    if bucket_unit == "week":
        weeks = -(-days // 7)
        span, before = f"{weeks} complete {pluralize(weeks, 'week')}", "this week"
    else:
        span, before = f"{days} complete {pluralize(days, 'day')}", "today"
    return (
        f"Nothing has been archived in the {span} before {before} — "
        "play something first! "
        "(Today's plays appear here tomorrow; `-history` shows them now.)"
    )


def invalid_days_notice() -> str:
    allowed = ", ".join(str(d) for d in ALLOWED_DAYS)
    return (
        f"`--days` must be one of: {allowed}. "
        f"Omit it for the default of {DEFAULT_DAYS}."
    )
