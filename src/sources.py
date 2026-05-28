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


@dataclass(frozen=True)
class SpotifySource:
    type: str
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
    """

    url: Optional[str] = None
    ytsearch: Optional[str] = None
    ts: Optional[int] = None
    process: Optional[bool] = None
    stype: URLSource = URLSource.YOUTUBE


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO timestamp regex
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
        for _, k, v in args_match:
            if k == "ts" or k == "t":
                ts = int(v)
        return YTSource(url, ts=ts, process=False)
    elif domain in ("open.spotify.com", "spotify.com"):
        path = domain_match.group(4).split("/")
        if path[0] not in ("playlist", "track"):
            raise Exception(f"Unknown Spotify track type: {path}")
        log.info(f"Spotify source ID: {path[1]}")
        return SpotifySource(path[0], path[1], process=True)
    elif domain in ("soundcloud.com",):
        return SoundcloudSource(url, process=True)
    else:
        raise Exception(f"Domain not supported {domain}")


def parse_input(
    user_input: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    Top-level entry point for command input. Tries parse_url; falls back to ytsearch.

    :param user_input: the URL or search term from the command argument
    :param message: full message content (used to extract the search query)
    :return: source
    """
    try:
        return parse_url(user_input, message)
    except ValueError:
        ytsearch = " ".join(message.split(" ")[1:])
        return YTSource(ytsearch=f"ytsearch:{ytsearch}", process=True)
