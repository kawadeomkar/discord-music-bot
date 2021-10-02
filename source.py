from dataclasses import dataclass
from enum import Enum
from typing import Union

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
    stype: str = URLSource.SPOTIFY


@dataclass(frozen=True)
class YTSource:
    url: str
    ts: str = None
    stype: str = URLSource.YOUTUBE


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO
    url: str
    stype: str = URLSource.YOUTUBE


def parse_url(url: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    domain regex (4 groups):
        group 1/2: http/www prefix
        group 3: domain
        group 4: path

    :param url: URL to be parsed
    :return: source
    """
    domain_re = r'(https:\/\/)?(www\.)?(.+)(\.com)?\/([^?]*)'
    args_re = r'(\?|\&)([^=]+)\=([^&]+)'

    domain_match = re.search(domain_re, url)
    args_match = re.findall(args_re, url)

    if not domain_match or not len(domain_match) == 4:
        return None

    if domain_match.group(3) in ("youtube", "youtu.be"):
        ts = None
        for _, k, v in args_match:
            if k == 'ts':
                ts = v
        return YTSource(url, ts=ts)
    elif domain_match.group(3) in ("open.spotify", "spotify"):


    elif domain_match.group()
