import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, Optional, Union
from urllib.parse import parse_qs, urlsplit

from src.guild_state import ANALYTICS_ZERO, Analytics
from src.util import get_logger, safe_label

log = get_logger(__name__)

# YouTube's older share format: ?t=1m30s, ?t=90s, ?t=1h2m3s. Still widely
# present in the wild — every "copy link at current time" from an older client
# emits it, and old messages are re-pasted for years.
_HMS_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")

# Quoted back to the user when their timestamp does not parse. Lives beside the
# regex so the accepted shapes have one definition.
TIMESTAMP_FORMATS: Final = "`90`, `90s`, `1m30s`, `2h30m15s`"


def parse_timestamp(raw: str) -> Optional[int]:
    """Seconds from a YouTube `t`/`ts` value, or None if it isn't one.

    Accepts both shapes YouTube has shipped: bare seconds ("90") and the
    colon-free HMS form ("1m30s", "90s", "1h2m3s"). Returns None rather than
    raising, because a timestamp is an optional refinement of a URL — an
    unparseable one must degrade to "play from the start", never to "this
    wasn't a URL at all" (which is what `int(v)` raising used to cause: the
    ValueError escaped parse_url and parse_input converted the whole link into
    a YouTube *search* for the URL text).
    """
    raw = raw.strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    match = _HMS_RE.fullmatch(raw)
    # `fullmatch` on an all-optional pattern also matches the empty string and
    # any leftover garbage matches nothing, so require at least one group.
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


# The unparseable `t=` echoed back to the user. Short: it is quoted inside a
# sentence, and a pasted URL fragment can be arbitrarily long.
_TIMESTAMP_ECHO_MAX = 40


def timestamp_warning(
    source: Union["SpotifySource", "YTSource", "SoundcloudSource"],
) -> Optional[str]:
    """One line naming a `t=` value that did not parse, or None.

    Lives beside the parser so the accepted shapes have one definition and the
    sentence quoting them cannot drift from what parse_timestamp takes. Returns
    text rather than an embed: this module stays free of discord, and the cog
    owns how a notice is rendered.

    Stated rather than silent, for the reason the playlist branch reports its
    skipped count: something the user wrote in their own URL changed where the
    song starts, and nothing else in the response accounts for it."""
    if not isinstance(source, YTSource) or source.bad_timestamp is None:
        return None
    # safe_label, not the raw value: it is rendered inside a code span below and
    # a backtick in it would close that span.
    shown = safe_label(source.bad_timestamp, _TIMESTAMP_ECHO_MAX)
    return (
        f"⚠️ Couldn't read the timestamp `{shown}` in that link — starting from "
        f"the beginning. YouTube's `t=` takes {TIMESTAMP_FORMATS}."
    )


class URLSource(Enum):
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"
    SOUNDCLOUD = "soundcloud"
    # Any host we don't special-case (tiktok, vimeo, bandcamp, …). yt-dlp's ~1800
    # extractors are the source of truth, so the URL goes straight to it and is rejected
    # only if yt-dlp reports the site unsupported (see YTDL.yt_source).
    OTHER = "other"
    # No URL was given at all — the input is a plaintext search term.
    SEARCH = "search"


# The persisted "how was this asked for" token, archived per play as
# play_history.query_source. Constants for the services parse_url special-cases
# rather than a parsed host, so youtu.be collapses onto the service it is and
# only the generic branch parses a hostname at all.
QUERY_SOURCE_YOUTUBE: Final[str] = "youtube.com"
QUERY_SOURCE_SPOTIFY: Final[str] = "spotify.com"
QUERY_SOURCE_SOUNDCLOUD: Final[str] = "soundcloud.com"
QUERY_SOURCE_SEARCH: Final[str] = "search"

# The token domain, mirrored by play_history's query_source CHECK. Bounded here
# because this is the only producer: HistoryEntry clamps anything else to the
# unknown sentinel rather than storing it.
_QUERY_HOST_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9.-]{1,64}")


def normalize_query_host(host: str) -> str:
    """A parsed host as the archive stores it: lowercased, leading `www.` dropped,
    empty when it is not host-shaped. parse_url's domain group admits `_`, `+` and
    `|`, so this filters rather than merely formats."""
    cleaned = host.strip().lower().removeprefix("www.")
    return cleaned if _QUERY_HOST_RE.fullmatch(cleaned) else ""


class SpotifyType(Enum):
    TRACK = "track"
    PLAYLIST = "playlist"


class YTType(Enum):
    TRACK = "track"
    PLAYLIST = "playlist"


@dataclass(frozen=True)
class SpotifySource:
    type: SpotifyType
    id: str
    process: bool = True
    stype: URLSource = URLSource.SPOTIFY


# slots: one of these is retained per unresolved Spotify-playlist track, so a
# 1000-track playlist holds 1000 until each is dequeued. Measures 344 B -> 120 B
# per instance. Keep the class free of __dict__ readers (asdict/vars) and off
# any pickle path; it crosses to Redis as SearchQueueEntry JSON.
@dataclass(frozen=True, slots=True)
class YTSource:
    """A YouTube track or playlist: either a pasted `url` or a `ytsearch:` term in
    `ytsearch`, with an optional `ts` start offset. `list_id` is the playlist ID, set
    when type == YTType.PLAYLIST; `index` is that playlist's 1-based start position,
    set only on the playlist branch because it means nothing without a list.
    `video_id` is the `v=` of a playlist link, kept only so the enqueue path can
    tell whether `ts` belongs to the track it is about to queue first."""

    url: Optional[str] = None
    ytsearch: Optional[str] = None
    ts: Optional[int] = None
    process: Optional[bool] = None
    stype: URLSource = URLSource.YOUTUBE
    type: YTType = YTType.TRACK
    list_id: Optional[str] = None
    index: Optional[int] = None
    video_id: Optional[str] = None
    # The raw `t`/`ts` value that failed to parse, set only when no usable
    # timestamp was found. Read once at the command layer to warn the requester
    # and never persisted, so it carries none of the propagation contracts the
    # fields below do.
    bad_timestamp: Optional[str] = None
    # Ask-time analytics (guild_state.Analytics), carried onto the QueueObject
    # this resolves into. Parse-layer minting leaves the zero value — the real
    # one arrives per-track in spotify_playlist_to_ytsearch, or rides the
    # yt_source/yt_playlist call for sources resolved directly.
    #
    # CONTRACT: the default is for the PARSE layer, which runs before the mint.
    # Anything handing a YTSource on to be queued must pass a real value —
    # nothing re-mints downstream, so an omission persists 0.0/0 to Redis and to
    # play_history with no error and no log line.
    analytics: Analytics = ANALYTICS_ZERO
    # What the user typed, for -remove to match on. Same contract as analytics:
    # the parse layer leaves None, and an old wire entry rehydrates as None too, so
    # an omission downstream is silent rather than reported.
    user_input: Optional[str] = None
    # How the song was asked for, set at parse time (see query_source_of). The one
    # source type that carries it: this covers pasted links, plaintext searches and
    # Spotify-playlist tracks alike, and it is the only one that survives into Redis
    # (as SearchQueueEntry), so a lazily-resolved Spotify track still archives as
    # Spotify rather than as the YouTube URL it resolves into.
    query_source: str = ""

    @property
    def playlist_url(self) -> str:
        """Canonical playlist URL for a type=PLAYLIST source: the pasted URL, else
        rebuilt from list_id. One spelling for the enqueue/interject/resolve paths."""
        return self.url or f"https://www.youtube.com/playlist?list={self.list_id}"


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO: SoundCloud timestamp links are ignored, so the track always starts at 0:00.
    # parse_url() reads `t`/`ts` for youtube.com only, so this field is never
    # populated — silently, for a link shape that works on YouTube.
    url: str
    ts: Optional[int] = None
    process: bool = False
    stype: URLSource = URLSource.SOUNDCLOUD


def query_source_of(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> str:
    """The query-source token for a parsed input. YTSource carries its own; the
    other two are consumed at resolve time and never persisted, so their token is
    a constant of their type and they need no field."""
    if isinstance(source, YTSource):
        return source.query_source
    if isinstance(source, SpotifySource):
        return QUERY_SOURCE_SPOTIFY
    return QUERY_SOURCE_SOUNDCLOUD


def spotify_playlist_to_ytsearch(
    titles: list[str], *, analytics: Analytics, origin: str
) -> list[YTSource]:
    """Spotify playlist tracks as lazy YouTube searches. The Spotify token, the
    ask-time analytics and `origin` are set here because it is the last point that
    knows where these came from — each resolves to a YouTube URL at dequeue.
    `analytics` is the head's; per-track positions are derived from it, as in
    yt_playlist. `origin` is the collection link the user pasted."""
    return [
        YTSource(
            ytsearch=f"ytsearch:{title}",
            process=True,
            query_source=QUERY_SOURCE_SPOTIFY,
            analytics=replace(analytics, queue_position=analytics.queue_position + i),
            user_input=origin,
        )
        for i, title in enumerate(titles)
    ]


def _playlist_index(raw: str) -> Optional[int]:
    """YouTube's 1-based `index=` param, or None when it is unparseable or below 1.

    Never raises: parse_url's ValueError means "not a URL at all" and sends
    parse_input to search for the link's own text, which is the wrong answer for
    a playlist whose index happens to be malformed."""
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def _last(args: dict[str, list[str]], key: str) -> Optional[str]:
    """Last value of a repeated query key. Matches a left-to-right scan in which
    each assignment overwrites the one before."""
    values = args.get(key)
    return values[-1] if values else None


def parse_url(url: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Parse a URL into a source dataclass. Raises ValueError if no domain matches.
    domain regex groups: 1/2 = http/www prefix, 3 = domain, 4 = path."""
    # Inside a character class `+` and `|` are literals, so a wider spelling
    # accepts "you+tube|com" as a hostname. `-` is included because hostnames
    # carry it: without it re.search starts matching after the hyphen and
    # "my-site.com" is read as the host "site.com", which is what lands in the
    # archive's query_source.
    domain_re = r"(https:\/\/)?(www\.)?([\w.-]+)\/([^?]*)"

    domain_match = re.search(domain_re, url)

    if not domain_match:
        raise ValueError(f"Not a recognised URL: {url!r}")

    domain = domain_match.group(3)
    # Query parsing via urllib rather than a hand-rolled regex: it handles
    # percent-encoding and repeated keys correctly, and it works on scheme-less
    # input ("youtu.be/x?t=90") because urlsplit only needs the "?" to find the
    # query — which is exactly the shape users paste.
    args = parse_qs(urlsplit(url).query)

    if domain in ("youtube.com", "youtu.be"):
        ts: Optional[int] = None
        list_id: Optional[str] = None
        # `t` and `ts` are the same parameter under two names; `ts` wins if a URL
        # somehow carries both.
        unparsed: list[str] = []
        for key in ("t", "ts"):
            for raw in args.get(key, []):
                parsed = parse_timestamp(raw)
                if parsed is None:
                    # Keep the URL and start at 0:00. An unparseable timestamp
                    # must not raise: parse_url's ValueError means "not a URL",
                    # which sends parse_input to search for the link's own text.
                    log.info(f"Ignoring unparseable timestamp {raw!r} in {url!r}")
                    unparsed.append(raw)
                else:
                    ts = parsed
        # Only worth reporting when nothing usable was found: a URL carrying both
        # a bad `t` and a good `ts` does start where the user asked.
        bad_timestamp = unparsed[-1] if ts is None and unparsed else None
        list_id = _last(args, "list")
        raw_index = _last(args, "index")
        index: Optional[int] = (
            _playlist_index(raw_index) if raw_index is not None else None
        )
        video_id = _last(args, "v")
        if list_id is not None:
            return YTSource(
                url,
                ts=ts,
                process=False,
                type=YTType.PLAYLIST,
                list_id=list_id,
                index=index,
                video_id=video_id,
                bad_timestamp=bad_timestamp,
                query_source=QUERY_SOURCE_YOUTUBE,
            )
        return YTSource(
            url,
            ts=ts,
            process=False,
            bad_timestamp=bad_timestamp,
            query_source=QUERY_SOURCE_YOUTUBE,
        )
    elif domain in ("open.spotify.com", "spotify.com"):
        path = domain_match.group(4).split("/")
        try:
            spotify_type = SpotifyType(path[0])
        except ValueError:
            raise Exception(f"Unknown Spotify track type: {path}")
        log.info(f"Spotify source ID: {path[1]}")
        return SpotifySource(spotify_type, path[1], process=True)
    elif domain in ("soundcloud.com",):
        return SoundcloudSource(url, process=True)
    elif "." in domain:
        # Looks like a real domain but isn't special-cased: hand it to yt-dlp rather
        # than maintain a whitelist, so an unsupported site surfaces yt-dlp's own
        # "Unsupported URL" via YTDL.yt_source. Routed like a bare YouTube watch URL, so
        # nothing downstream needs to know it's generic. The host is what distinguishes
        # tiktok from vimeo in the archive, so it is the only branch that parses one.
        return YTSource(
            url=url,
            process=True,
            stype=URLSource.OTHER,
            query_source=normalize_query_host(domain),
        )
    else:
        # Regex matched but the "host" has no dot (e.g. "98" from the search term
        # "98/99"). ValueError makes parse_input fall back to a YouTube search.
        raise ValueError(f"Not a recognised URL: {url!r}")


def unquote_argument(text: str) -> str:
    """Drop one matched pair of surrounding quotes.

    `-play` takes a consume-rest argument, which discord.py's `read_rest()` hands
    through with the quotes: parse_url then drags the trailing one into the path and
    yt-dlp rejects it, and a quoted search stores `"some song"` as the origin, which
    `-remove some song` cannot match.

    Only a whole argument wrapped at both ends, and never down to nothing — a lone
    quote or an empty pair is text the user typed. Runs here and at the command,
    which unquotes the value it stamps `origin` from, so it must be safe twice."""
    for quote in ('"', "'"):
        if len(text) > 2 and text.startswith(quote) and text.endswith(quote):
            return text[1:-1]
    return text


def parse_input(user_input: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Top-level entry point for command input: tries parse_url, falls back to ytsearch.
    parse_url is attempted only for single-word input, since URLs never contain spaces; a
    single-word term with a slash ("98/99") still reaches it but raises ValueError on the
    dotless host and falls back to search.

    Reads only what the caller passes. It used to re-derive the search from the raw
    message content, which was possible only while `-play` bound one word — and which
    counted the whitespace between prefix and argument as part of the term, so a
    double-spaced `-play  <link>` was searched for as text instead of parsed as a URL.
    A caller that strips a flag off the front now gets the answer for what remains."""
    args = user_input.split()
    if len(args) == 1:
        try:
            # args[0], not user_input: the token without whatever whitespace
            # surrounded it, since parse_url hands this straight to YTSource.url.
            return parse_url(unquote_argument(args[0]))
        except ValueError:
            pass
    ytsearch = unquote_argument(" ".join(args))
    return YTSource(
        ytsearch=f"ytsearch:{ytsearch}",
        process=True,
        stype=URLSource.SEARCH,
        query_source=QUERY_SOURCE_SEARCH,
    )
