import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
from urllib.parse import parse_qs, urlsplit

from src.util import get_logger

log = get_logger(__name__)

# YouTube's older share format: ?t=1m30s, ?t=90s, ?t=1h2m3s. Still widely
# present in the wild — every "copy link at current time" from an older client
# emits it, and old messages are re-pasted for years.
_HMS_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


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


class URLSource(Enum):
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"
    SOUNDCLOUD = "soundcloud"
    # Any other host we don't special-case (tiktok, twitter/x, vimeo, bandcamp,
    # twitch clips, …). We don't maintain the list — yt-dlp's ~1800 extractors are
    # the source of truth, so the URL is handed straight to it and only rejected if
    # yt-dlp itself reports the site as unsupported (see YTDL.yt_source).
    OTHER = "other"


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


def parse_url(url: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """
    Parse a URL into a source dataclass. Raises ValueError if no domain is matched.

    domain regex (3 groups):
        group 1/2: http/www prefix
        group 3: domain
        group 4: path

    :param url: URL to be parsed
    :return: source
    """
    # `[\w.]+`, not `[\w+|\.]+`: inside a character class `+` and `|` are
    # literals, so the old spelling accepted "you+tube|com" as a hostname.
    domain_re = r"(https:\/\/)?(www\.)?([\w.]+)\/([^?]*)"

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
        # `t` and `ts` are the same parameter under two names; last one wins,
        # matching the old left-to-right loop.
        for key in ("t", "ts"):
            for raw in args.get(key, []):
                parsed = parse_timestamp(raw)
                if parsed is None:
                    # Keep the URL and start from 0:00. Previously this raised
                    # out of parse_url and the link degraded into a *search*
                    # for its own text — which usually still played, but paid a
                    # full search extraction and dropped the seek silently.
                    log.info(f"Ignoring unparseable timestamp {raw!r} in {url!r}")
                else:
                    ts = parsed
        list_ids = args.get("list", [])
        list_id = list_ids[0] if list_ids else None
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
            return parse_url(user_input)
        except ValueError:
            pass
    ytsearch = " ".join(args)
    return YTSource(ytsearch=f"ytsearch:{ytsearch}", process=True)
