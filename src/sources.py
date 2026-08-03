import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

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


def spotify_titles_to_ytsearch(titles: list[str]) -> list[YTSource]:
    return [YTSource(ytsearch=f"ytsearch:{title}", process=True) for title in titles]


class UnsupportedSpotifyLinkError(Exception):
    """A Spotify URL naming a type this bot does not queue (/artist/, /show/, …).

    Deliberately NOT a ValueError: parse_input catches ValueError and falls back
    to a YouTube search, which would turn an /artist/ link into a nonsense
    `ytsearch:https://open.spotify.com/...` query instead of an error the user
    can act on.
    """


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
        for _, k, v in args_match:
            if k == "ts" or k == "t":
                ts = int(v)
            elif k == "list":
                list_id = v
        if list_id is not None:
            return YTSource(
                url, ts=ts, process=False, type=YTType.PLAYLIST, list_id=list_id
            )
        return YTSource(url, ts=ts, process=False)
    elif domain in ("open.spotify.com", "spotify.com"):
        path = domain_match.group(4).split("/")
        try:
            spotify_type = SpotifyType(path[0])
        except ValueError:
            supported = ", ".join(t.value for t in SpotifyType)
            # `from None`: the ValueError above is an implementation detail of
            # the enum lookup; chaining it turns the user-facing error into a
            # two-traceback log entry (the exact noise the incident log shows).
            raise UnsupportedSpotifyLinkError(
                f"Spotify {path[0]!r} links aren't supported — try a {supported} link"
            ) from None
        log.info(f"Spotify source ID: {path[1]}")
        return SpotifySource(spotify_type, path[1], process=True)
    elif domain in ("soundcloud.com",):
        return SoundcloudSource(url, process=True)
    elif "." in domain:
        # Looks like a real domain but isn't special-cased: hand it to yt-dlp rather
        # than maintain a whitelist, so an unsupported site surfaces yt-dlp's own
        # "Unsupported URL" via YTDL.yt_source. Routed like a bare YouTube watch URL, so
        # nothing downstream needs to know it's generic.
        return YTSource(url=url, process=True, stype=URLSource.OTHER)
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
    return YTSource(ytsearch=f"ytsearch:{ytsearch}", process=True)
