import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional, Union

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


@dataclass(frozen=True)
class YTSource:
    """A YouTube track or playlist: either a pasted `url` or a `ytsearch:` term in
    `ytsearch`, with an optional `ts` start offset. `list_id` is the playlist ID, set
    when type == YTType.PLAYLIST."""

    url: Optional[str] = None
    ytsearch: Optional[str] = None
    ts: Optional[int] = None
    process: Optional[bool] = None
    stype: URLSource = URLSource.YOUTUBE
    type: YTType = YTType.TRACK
    list_id: Optional[str] = None
    # Enqueue stamps, carried onto the QueueObject this resolves into. Frozen,
    # so MusicPlayer stamps by building a replace()d copy.
    queued_at: float = 0.0
    queue_position: int = 0
    # How the song was asked for, set at parse time (see query_source_of). The one
    # source type that carries it: this covers pasted links, plaintext searches and
    # Spotify-playlist tracks alike, and it is the only one that survives into Redis
    # (as SearchQueueEntry), so a lazily-resolved Spotify track still archives as
    # Spotify rather than as the YouTube URL it resolves into.
    query_source: str = ""

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


def spotify_playlist_to_ytsearch(titles: list[str]) -> list[YTSource]:
    """Spotify playlist tracks as lazy YouTube searches. The Spotify token is
    stamped here because it is the last point that knows where these came from —
    each resolves to a YouTube URL at dequeue."""
    return [
        YTSource(
            ytsearch=f"ytsearch:{title}",
            process=True,
            query_source=QUERY_SOURCE_SPOTIFY,
        )
        for title in titles
    ]


def parse_url(
    url: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Parse a URL into a source dataclass. Raises ValueError if no domain matches.
    `message` is the full message content. domain regex groups: 1/2 = http/www prefix,
    3 = domain, 4 = path."""
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
        for _, k, v in args_match:
            if k == "ts" or k == "t":
                ts = int(v)
            elif k == "list":
                list_id = v
        if list_id is not None:
            return YTSource(
                url,
                ts=ts,
                process=False,
                type=YTType.PLAYLIST,
                list_id=list_id,
                query_source=QUERY_SOURCE_YOUTUBE,
            )
        return YTSource(url, ts=ts, process=False, query_source=QUERY_SOURCE_YOUTUBE)
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
            return parse_url(user_input, message)
        except ValueError:
            pass
    ytsearch = " ".join(args)
    return YTSource(
        ytsearch=f"ytsearch:{ytsearch}",
        process=True,
        stype=URLSource.SEARCH,
        query_source=QUERY_SOURCE_SEARCH,
    )
