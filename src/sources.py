from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import re


class URLSource(Enum):
    SPOTIFY: str = "spotify"
    YOUTUBE: str = "youtube"
    SOUNDCLOUD: str = "soundcloud"


@dataclass(frozen=True)
class SpotifySource:
    type: str
    id: str
    si: str = None
    process: bool = True
    stype: str = URLSource.SPOTIFY


@dataclass(frozen=True)
class YTSource:
    """
    :param url: YT URL
    :param ytsearch: youtube search
    :param ts: timestamp
    """
    url: str = None
    ytsearch: str = None
    ts: int = None
    process: bool = None
    stype: str = URLSource.YOUTUBE


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO timestamp regex
    url: str
    ts: int = None
    process: bool = False
    stype: str = URLSource.SOUNDCLOUD


def spotify_playlist_to_ytsearch(titles: List[str]) -> List[YTSource]:
    return [YTSource(ytsearch=f"ytsearch:{title}", process=True) for title in titles]


def parse_url(url: str, message: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    domain regex (4 groups):
        group 1/2: http/www prefix
        group 3: domain
        group 4: path

    :param url: URL to be parsed
    :param message: full message content search
    :return: source
    """
    domain_re = r'(https:\/\/)?(www\.)?([\w+|\.]+)\/([^?]*)'
    args_re = r'(\?|\&)([^=]+)\=([^&]+)'

    domain_match = re.search(domain_re, url)
    args_match = re.findall(args_re, url)

    if not domain_match:
        ytsearch = " ".join(message.split(" ")[1:])
        ytsearch = f"ytsearch:{ytsearch}"
        return YTSource(ytsearch=ytsearch, process=True)

    domain = domain_match.group(3)

    if domain in ("youtube.com", "youtu.be"):
        ts = None
        for _, k, v in args_match:
            if k == 'ts' or k == 't':
                ts = int(v)
        return YTSource(url, ts=ts, process=False)
    elif domain in ("open.spotify.com", "spotify.com"):
        path = domain_match.group(4).split("/")
        if path[0] not in ("playlist", "track"):
            raise Exception(f"Unknown Spotify track type: {path}")
        print(path[1])
        return SpotifySource(path[0], path[1], process=True)
    elif domain in ("soundcloud.com"):
        return SoundcloudSource(url, process=True)
    else:
        raise Exception(f"Domain not supported {domain}")
