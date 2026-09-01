"""`-leaderboard` — the board's tunables, its Redis result cache codec, and its
renderer.

Two halves. The first is PURE: it takes a `Leaderboard` (from history_archive)
and returns strings, dicts or an embed. The second is `run()`, the command body —
it reads the cache, queries the archive and sends. Only the wrapper stays on the
MusicBot cog in musicbot.py, because discord.py needs the command there and
because the error-embed policy is the cog's.

NOT in util.py, which is imported by the yt-dlp worker graph and must stay free
of asyncpg — this module reads history_archive's row types, so it may not move
there. See docs/ARCHITECTURE.md#history-archive-tier.

Names are unqualified (`cache_key`, `build_embed`) because the module already
says leaderboard; musicbot imports the module, not the names.
"""

import re
from typing import TYPE_CHECKING, Final, Optional
from urllib.parse import urlsplit

import discord

from src.history_archive import Leaderboard, RequesterLeader, SongLeader
from src.sources import (
    QUERY_SOURCE_SEARCH,
    QUERY_SOURCE_SOUNDCLOUD,
    QUERY_SOURCE_SPOTIFY,
    QUERY_SOURCE_YOUTUBE,
)
from src.util import fmt_duration, pluralize, truncate

if TYPE_CHECKING:
    pass

TOP_N: Final[int] = 10
MAX_DAYS: Final[int] = 3650
# Bounds Postgres to one aggregate pass per guild per window per minute, whatever
# the table size. TTL'd, so the key is a legitimate volatile-lru eviction
# candidate — losing it costs one re-query.
CACHE_TTL_SECS: Final[int] = 60
# Bumped on any change to the cached shape. The codec defaults missing fields
# rather than rejecting them, so without this a rolling deploy would decode an
# old entry into a valid-looking board with wrong values.
_CACHE_VERSION: Final[int] = 2
# Masked-link label budget. escape_markdown can double it, so 50 holds all twenty
# lines under 3 KB — inside the 4096-char description limit, and inside the 6000
# characters Discord counts across EVERY embed in the message, which this shares
# with the ≤2-embed Now Playing block MusicContext.send prepends. The "· via
# <source>" tail adds at most 20 characters to each of the ten song lines.
_TITLE_MAX: Final[int] = 50
# Past this the URL is dropped from the line rather than budgeted for.
_URL_MAX: Final[int] = 150
# The link's host, rendered beside the label.
_HOST_MAX: Final[int] = 32
# Display names for the query-source tokens sources.py mints for the services it
# special-cases. Everything else is a bare host and renders as itself.
_QUERY_SOURCE_LABELS: Final[dict[str, str]] = {
    QUERY_SOURCE_SEARCH: "search",
    QUERY_SOURCE_SPOTIFY: "Spotify",
    QUERY_SOURCE_YOUTUBE: "YouTube",
    QUERY_SOURCE_SOUNDCLOUD: "SoundCloud",
}
# Characters that end a masked link early: a newline splits the line and leaks
# the rest of the markdown as its own text, and U+2028/9 do the same on some
# clients. Flattened rather than dropped so words do not run together.
_LABEL_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")


def cache_key(guild_id: int, days: int, top_n: int) -> str:
    """Keyed by row count as well as window: raising TOP_N must not render a
    short board from a cache entry the previous limit produced."""
    return f"leaderboard:v{_CACHE_VERSION}:{guild_id}:{days}:{top_n}"


def to_cache(board: Leaderboard) -> dict:
    """Plain dicts for orjson. Field names spelled out so a dataclass rename
    cannot silently change the cache shape."""
    return {
        "requesters": [
            {
                "requester_id": r.requester_id,
                "requester_name": r.requester_name,
                "plays": r.plays,
                "played_secs": r.played_secs,
            }
            for r in board.requesters
        ],
        "songs": [
            {
                "title": s.title,
                "webpage_url": s.webpage_url,
                "duration_secs": s.duration_secs,
                "query_source": s.query_source,
                "plays": s.plays,
                "played_secs": s.played_secs,
            }
            for s in board.songs
        ],
    }


def from_cache(raw: object, *, top_n: int) -> Optional[Leaderboard]:
    """Rebuild a cached Leaderboard. None means MALFORMED, never "empty": an
    empty board is a valid cached value and caching it is what stops an idle
    guild re-querying Postgres on every invocation. Do not test truthiness.

    Both boards are capped at `top_n` on the way in: the entry is decoded before
    anything checks its size, so the cap is what stops an oversized value —
    written by another build, or by anything else holding the key — from
    rendering more rows than the command promises."""
    if not isinstance(raw, dict):
        return None
    try:
        return Leaderboard(
            requesters=tuple(
                RequesterLeader(
                    requester_id=int(r["requester_id"]),
                    requester_name=str(r.get("requester_name", "")),
                    plays=int(r["plays"]),
                    played_secs=int(r["played_secs"]),
                )
                for r in raw.get("requesters", [])[:top_n]
            ),
            songs=tuple(
                SongLeader(
                    title=str(s.get("title", "")),
                    webpage_url=str(s.get("webpage_url", "")),
                    duration_secs=int(s.get("duration_secs", 0)),
                    query_source=str(s.get("query_source", "")),
                    plays=int(s["plays"]),
                    played_secs=int(s["played_secs"]),
                )
                for s in raw.get("songs", [])[:top_n]
            ),
        )
    except KeyError, TypeError, ValueError:
        return None


def _sanitize_label(text: str) -> str:
    """Render-safe archive text — a song title or a requester's stored name.
    Flatten the control characters that end a line early, cap, neutralize the
    brackets that break a masked link (escape_markdown does not cover them),
    then escape the rest so the text cannot bold or strike its line.

    Cap BEFORE escaping: escaping first and cutting after can split an escape
    pair and leave a trailing backslash that eats the next character. The cap
    ellipsizes — two songs sharing a 50-character prefix would otherwise render
    as the same line."""
    flattened = _LABEL_UNSAFE.sub(" ", text)
    clipped = truncate(flattened, _TITLE_MAX)
    return discord.utils.escape_markdown(clipped.replace("[", "(").replace("]", ")"))


def _link_host(url: str) -> str:
    """Host of a song's link, empty when it has none. Rendered beside the label
    because both halves of a masked link come from the archive: without it a
    played song's title can name a destination its URL does not go to."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    return host.removeprefix("www.")[:_HOST_MAX]


def _line_requester(
    rank: int, r: RequesterLeader, guild: Optional[discord.Guild]
) -> str:
    """A mention while the requester is still in the guild, their archived name
    once they leave — Discord renders a mention for a non-member as a raw id."""
    who = f"<@{r.requester_id}>"
    if guild is not None and guild.get_member(r.requester_id) is None:
        who = _sanitize_label(r.requester_name) or "unknown"
    # "listened" labels the ranking key. A bare clock reads as a track length,
    # which is the other duration on the songs board two lines down.
    return (
        f"**{rank}.** {who} — {fmt_duration(r.played_secs)} listened · "
        f"{r.plays} {pluralize(r.plays, 'song')}"
    )


def _source_label(token: str) -> str:
    """Display name for a query-source token. Known services get their own
    spelling; anything else is a host, which reads fine as itself (tiktok.com).
    Empty stays empty — the caller drops the segment rather than printing
    "unknown", which every pre-feature row would carry."""
    return _QUERY_SOURCE_LABELS.get(token, token)


def _line_song(rank: int, s: SongLeader) -> str:
    # A blank title is a real archived value (the zero-value convention), and an
    # empty masked-link label renders as an invisible link.
    title = _sanitize_label(s.title) or "Unknown"
    url = s.webpage_url
    # A paren, whitespace or control character inside a masked-link URL ends the
    # markdown early; such URLs (and over-long ones) render as a plain title.
    linkable = (
        url
        and len(url) <= _URL_MAX
        and not any(c in url for c in "() \t")
        and not _LABEL_UNSAFE.search(url)
    )
    if linkable:
        host = _link_host(url)
        label = f"[{title}]({url})" + (f" `{host}`" if host else "")
    else:
        label = title
    # Both clocks are labelled: the first is the ranking key (time this server
    # spent on the song), the second the track's own length, and unlabelled they
    # render as two interchangeable durations.
    line = (
        f"**{rank}.** {label} — {fmt_duration(s.played_secs)} listened · "
        f"{s.plays} {pluralize(s.plays, 'play')} · track {fmt_duration(s.duration_secs)}"
    )
    # The host chip names where the LINK goes; this names how the song was
    # ASKED for, which webpage_url cannot answer — a Spotify link and a
    # plaintext search both resolve to youtube.com.
    source = _sanitize_label(_source_label(s.query_source))
    return f"{line} · via {source}" if source else line


def build_embed(
    board: Leaderboard, *, days: int = 0, guild: Optional[discord.Guild] = None
) -> Optional[discord.Embed]:
    """One embed, both boards in the DESCRIPTION: an embed field caps at 1024
    characters and ten masked-link lines do not reliably fit, while the 4096-char
    description does. Sections render independently; both empty -> None, and the
    caller sends the nothing-archived notice."""
    sections: list[str] = []
    if board.requesters:
        rows = [
            _line_requester(i, r, guild)
            for i, r in enumerate(board.requesters, start=1)
        ]
        sections.append("**Top listeners**\n" + "\n".join(rows))
    if board.songs:
        rows = [_line_song(i, s) for i, s in enumerate(board.songs, start=1)]
        sections.append("**Top songs**\n" + "\n".join(rows))
    if not sections:
        return None
    # The period is always named, including all-time. FlagConverter silently
    # defaults days=0 for every input it does not recognise — `--days=7`, a bare
    # `--days`, a positional `7` — so an unnamed title would render a dropped
    # window as an all-time board the requester reads as their window.
    period = f"last {days} {pluralize(days, 'day')}" if days else "all time"
    embed = discord.Embed(
        title=f"🏆 Leaderboard — {period}",
        description="\n\n".join(sections),
        color=discord.Color.gold(),
    )
    embed.set_footer(
        text="Totals cover songs saved to this server's long-term archive."
    )
    return embed
