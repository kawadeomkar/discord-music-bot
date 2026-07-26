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
    # Any other host we don't special-case (tiktok, twitter/x, vimeo, bandcamp,
    # twitch clips, …). We don't maintain the list — yt-dlp's ~1800 extractors are
    # the source of truth, so the URL is handed straight to it and only rejected if
    # yt-dlp itself reports the site as unsupported (see YTDL.yt_source).
    OTHER = "other"


class UnsupportedSourceError(ValueError):
    """A URL we recognised the host of, but cannot play.

    Subclasses ValueError deliberately: parse_input treats ValueError as
    "not a URL, fall back to a YouTube search", which is the right behaviour
    for a dotless host but NOT for this — so parse_input catches this type
    FIRST and re-raises. Making it a ValueError anyway keeps every existing
    `except ValueError` caller (and any future one) from letting an
    unsupported link escape as a bare, untyped Exception the way
    `Exception("Unknown Spotify track type: …")` used to.
    """


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

    @property
    def playlist_url(self) -> str:
        """The canonical playlist URL for a type=PLAYLIST source: the original
        URL if one was pasted, else rebuilt from list_id. The single spelling
        of the `url or ".../playlist?list={list_id}"` fallback the enqueue,
        playnow, and resolve paths all need."""
        return self.url or f"https://www.youtube.com/playlist?list={self.list_id}"


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO: SoundCloud timestamp links are ignored, so the track always starts at 0:00.
    # parse_url() extracts the `t`/`ts` query param for youtube.com only, so this ts
    # field is never populated for a SoundCloud URL. A user who pastes a SoundCloud link
    # with a timestamp gets no seek and no explanation — the identical link shape works
    # for YouTube, which makes the inconsistency look like a bug rather than a gap.
    url: str
    ts: Optional[int] = None
    process: bool = False
    stype: URLSource = URLSource.SOUNDCLOUD


def spotify_playlist_to_ytsearch(titles: list[str]) -> list[YTSource]:
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
        # Spotify's share sheet emits a locale segment for most non-US clients:
        # open.spotify.com/intl-de/track/<id>. It is presentational only — the
        # type and ID that follow are identical — but it lands in path[0], so
        # every such link failed the type lookup below. That broke *track*
        # links, a supported feature, for much of the world.
        if path and path[0].startswith("intl-"):
            path = path[1:]
        try:
            spotify_type = SpotifyType(path[0])
        except ValueError, IndexError:
            # Typed, not a bare Exception: this reaches the user as a command
            # error, and "Unknown Spotify track type: ['episode', '4rOo…']"
            # told them nothing they could act on.
            raise UnsupportedSourceError(
                f"Unsupported Spotify link type {path[0]!r} — "
                "tracks, albums and playlists are supported."
                if path
                else f"Not a recognisable Spotify link: {url!r}"
            )
        if len(path) < 2 or not path[1]:
            raise UnsupportedSourceError(f"Spotify link is missing an ID: {url!r}")
        log.info(f"Spotify source ID: {path[1]}")
        return SpotifySource(spotify_type, path[1], process=True)
    elif domain in ("soundcloud.com",):
        return SoundcloudSource(url, process=True)
    elif "." in domain:
        # Not a host we special-case, but it looks like a real domain. Rather than
        # maintain a whitelist of yt-dlp's ~1800 supported sites, hand the raw URL to
        # yt-dlp and let it decide: a supported site (tiktok, vimeo, twitch clips, …)
        # just plays, and a genuinely unsupported one surfaces yt-dlp's own
        # "Unsupported URL" as a clear message from YTDL.yt_source. Routed exactly like
        # a bare YouTube watch URL — resolved to a QueueObject in queue_source before it
        # ever reaches the queue — so no downstream path needs to know it's generic.
        return YTSource(url=url, process=True, stype=URLSource.OTHER)
    else:
        # The domain regex matched but the "host" has no dot (e.g. "98" from a search
        # term like "98/99"). That's not a URL — raise ValueError so parse_input falls
        # back to a YouTube search instead of shipping a bogus host to yt-dlp.
        raise ValueError(f"Not a recognised URL: {url!r}")


def parse_input(
    user_input: str, message: str
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    Top-level entry point for command input. Tries parse_url; falls back to ytsearch.

    Only attempts parse_url when the command argument is a single word (a bare
    link) — URLs never contain spaces, so multi-word input is always a search
    query. A single-word search term that happens to contain a slash (e.g. "98/99")
    still reaches parse_url, but its dotless "host" raises ValueError there and is
    caught below, falling back to search rather than being shipped to yt-dlp.

    :param user_input: the URL or search term from the command argument
    :param message: full message content (used to extract the search query)
    :return: source
    """
    args = message.split(" ")[1:]
    if len(args) == 1:
        try:
            return parse_url(user_input, message)
        except UnsupportedSourceError:
            # Ordered before the ValueError arm below (which this subclasses):
            # a recognised-but-unplayable link must surface its own message, not
            # be silently retried as `ytsearch:https://open.spotify.com/...`,
            # which spends a full search extraction to play something arbitrary.
            raise
        except ValueError:
            pass
    ytsearch = " ".join(args)
    return YTSource(ytsearch=f"ytsearch:{ytsearch}", process=True)
