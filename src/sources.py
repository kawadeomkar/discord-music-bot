import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, Optional, Union

from src.guild_state import ANALYTICS_ZERO, Analytics
from src.util import get_logger

log = get_logger(__name__)


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
    ALBUM = "album"


class YTType(Enum):
    TRACK = "track"
    PLAYLIST = "playlist"


@dataclass(frozen=True)
class SpotifySource:
    type: SpotifyType
    id: str
    si: Optional[str] = None
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
    # Ask-time analytics (guild_state.Analytics), carried onto the QueueObject
    # this resolves into. Parse-layer minting leaves the zero value — the real
    # one arrives per-track in spotify_titles_to_ytsearch, or rides the
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
    # Who asked for this track. An ID, not a Member: this module is discord-free,
    # and a live member could not survive the Redis round-trip anyway. None on the
    # parse-time sources, which resolve inside the command that built them, and on
    # entries queued before this field existed — both fall back to _last_author,
    # which is what they have always resolved to.
    requester_id: Optional[int] = None

    @property
    def playlist_url(self) -> str:
        """Canonical playlist URL for a type=PLAYLIST source: the pasted URL, else
        rebuilt from list_id. One spelling for the enqueue/playnow/resolve paths."""
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


def spotify_titles_to_ytsearch(
    titles: list[str], requester_id: int, *, analytics: Analytics, origin: str
) -> list[YTSource]:
    """Spotify album or playlist tracks as lazy YouTube searches. The Spotify token,
    the requester, the ask-time analytics and `origin` are set here because it is
    the last point that knows where these came from — each resolves to a YouTube
    URL at dequeue, minutes to an hour after the command that asked for it returned.
    `analytics` is the head's; per-track positions are derived from it, as in
    yt_playlist. `origin` is the collection link the user pasted.

    requester_id is required rather than defaulted on purpose: a caller that omits
    it silently attributes every track to whoever ran a command most recently,
    which is exactly the defect this parameter exists to close."""
    return [
        YTSource(
            ytsearch=f"ytsearch:{title}",
            process=True,
            query_source=QUERY_SOURCE_SPOTIFY,
            requester_id=requester_id,
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


class UnsupportedSpotifyLinkError(Exception):
    """A Spotify URL naming a type this bot does not queue (/artist/, /show/, …).

    Deliberately not a ValueError: parse_input catches ValueError and falls back
    to a YouTube search, which would turn an /artist/ link into a nonsense
    `ytsearch:https://open.spotify.com/...` query instead of an error the user
    can act on.
    """

    @property
    def user_message(self) -> str:
        """The message is already written for the user; exposing it under the
        name _command_error's allowlist renders keeps the embed free of the
        `**UnsupportedSpotifyLinkError:**` class-name prefix."""
        return str(self)


def parse_url(
    url: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Parse a URL into a source dataclass. Raises ValueError if no domain matches,
    and UnsupportedSpotifyLinkError for a Spotify link type the bot does not queue
    (not a ValueError — see that class's docstring). `message` is the full message
    content. domain regex groups: 1/2 = http/www prefix, 3 = domain, 4 = path."""
    domain_re = r"(https:\/\/)?(www\.)?([\w+|\.]+)\/([^?]*)"
    args_re = r"(\?|\&)([^=]+)\=([^&]+)"

    domain_match = re.search(domain_re, url)
    args_match = re.findall(args_re, url)

    if not domain_match:
        raise ValueError(f"Not a recognised URL: {url!r}")

    domain = domain_match.group(3)

    if domain in ("youtube.com", "youtu.be"):
        ts: Optional[int] = None
        list_id: Optional[str] = None
        index: Optional[int] = None
        video_id: Optional[str] = None
        for _, k, v in args_match:
            if k == "ts" or k == "t":
                ts = int(v)
            elif k == "list":
                list_id = v
            elif k == "index":
                index = _playlist_index(v)
            elif k == "v":
                video_id = v
        if list_id is not None:
            return YTSource(
                url,
                ts=ts,
                process=False,
                type=YTType.PLAYLIST,
                list_id=list_id,
                index=index,
                video_id=video_id,
                query_source=QUERY_SOURCE_YOUTUBE,
            )
        return YTSource(url, ts=ts, process=False, query_source=QUERY_SOURCE_YOUTUBE)
    elif domain in ("open.spotify.com", "spotify.com"):
        path = domain_match.group(4).split("/")
        # Spotify's web player and mobile share sheet prefix a locale segment
        # (/intl-de/album/…) for every non-English client. It carries no
        # routing information — drop it, or the share URL half the world copies
        # is rejected while the hand-trimmed one works.
        if path and path[0].startswith("intl-"):
            path = path[1:]
        kind = path[0] if path else ""
        try:
            spotify_type = SpotifyType(kind)
        except ValueError:
            values = [t.value for t in SpotifyType]
            supported = ", ".join(values[:-1]) + f" or {values[-1]}"
            # `from None`: the ValueError above is an implementation detail of
            # the enum lookup; chaining it turns the user-facing error into a
            # two-traceback log entry (the exact noise the incident log shows).
            raise UnsupportedSpotifyLinkError(
                f"Spotify {kind!r} links aren't supported — try a {supported} link"
            ) from None
        if len(path) < 2 or not path[1]:
            # A bare /album or /intl-de/track: the type without an id. The
            # IndexError this used to raise rendered as a stack trace.
            raise UnsupportedSpotifyLinkError(
                f"That Spotify {kind} link has no id — copy the full link"
            ) from None
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

    `-play`/`-playnow` take consume-rest arguments, which discord.py's `read_rest()`
    hands through with the quotes: parse_url then drags the trailing one into the
    path and yt-dlp rejects it, and a quoted search stores `"some song"` as the
    origin, which `-remove some song` cannot match.

    Only a whole argument wrapped at both ends, and never down to nothing — a lone
    quote or an empty pair is text the user typed. Runs here and at the command,
    which unquotes the value it stamps `origin` from, so it must be safe twice."""
    for quote in ('"', "'"):
        if len(text) > 2 and text.startswith(quote) and text.endswith(quote):
            return text[1:-1]
    return text


def parse_input(
    user_input: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Top-level entry point for command input: tries parse_url, falls back to ytsearch.
    parse_url is attempted only for single-word input, since URLs never contain spaces; a
    single-word term with a slash ("98/99") still reaches it but raises ValueError on the
    dotless host and falls back to search."""
    args = message.split(" ")[1:]
    if len(args) == 1:
        try:
            return parse_url(unquote_argument(user_input), message)
        except ValueError:
            pass
    ytsearch = unquote_argument(" ".join(args))
    return YTSource(
        ytsearch=f"ytsearch:{ytsearch}",
        process=True,
        stype=URLSource.SEARCH,
        query_source=QUERY_SOURCE_SEARCH,
    )
