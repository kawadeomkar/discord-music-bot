import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from src.util import get_logger

log = get_logger(__name__)


class URLSource(Enum):
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"
    SOUNDCLOUD = "soundcloud"


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
    """
    :param url: YT URL
    :param ytsearch: youtube search
    :param ts: timestamp
    :param list_id: YouTube playlist ID (present when type == YTType.PLAYLIST)
    """

    url: Optional[str] = None
    ytsearch: Optional[str] = None
    ts: Optional[int] = None
    process: Optional[bool] = None
    stype: URLSource = URLSource.YOUTUBE
    type: YTType = YTType.TRACK
    list_id: Optional[str] = None


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO: SoundCloud timestamp links are silently ignored. parse_url() extracts `t`/`ts`
    # for youtube.com only, so this ts field is never populated for a SoundCloud URL and
    # the track always starts at 0:00 — a user who pastes a link with a timestamp gets no
    # seek and no explanation, while the same link shape works for YouTube.
    url: str
    ts: Optional[int] = None
    process: bool = False
    stype: URLSource = URLSource.SOUNDCLOUD


def spotify_playlist_to_ytsearch(titles: List[str]) -> List[YTSource]:
    return [YTSource(ytsearch=f"ytsearch:{title}", process=True) for title in titles]


def parse_url(
    url: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    Parse a URL into a source dataclass. Raises ValueError if no domain is matched.

    domain regex (4 groups):
        group 1/2: http/www prefix
        group 3: domain
        group 4: path

    :param url: URL to be parsed
    :param message: full message content (used for Spotify si param extraction)
    :return: source
    """
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
            raise Exception(f"Unknown Spotify track type: {path}")
        log.info(f"Spotify source ID: {path[1]}")
        return SpotifySource(spotify_type, path[1], process=True)
    elif domain in ("soundcloud.com",):
        return SoundcloudSource(url, process=True)
    else:
        raise Exception(f"Domain not supported {domain}")


def parse_input(
    user_input: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    Top-level entry point for command input. Tries parse_url; falls back to ytsearch.

    Only attempts parse_url when the command argument is a single word (a bare
    link) — URLs never contain spaces, so multi-word input is always a search
    query. This also avoids the loose domain regex misidentifying search terms
    like "98/99" as a URL with an unsupported domain.

    :param user_input: the URL or search term from the command argument
    :param message: full message content (used to extract the search query)
    :return: source
    """
    args = message.split(" ")[1:]
    if len(args) == 1:
        try:
            return parse_url(user_input, message)
        except ValueError:
            pass
    ytsearch = " ".join(args)
    return YTSource(ytsearch=f"ytsearch:{ytsearch}", process=True)
