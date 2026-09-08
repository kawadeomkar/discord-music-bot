import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, Optional, Union
from urllib.parse import parse_qs, urlsplit

from src.guild_state import ANALYTICS_ZERO, Analytics
from src.util import get_logger, safe_label

log = get_logger(__name__)

# YouTube's older share format (?t=1m30s, ?t=90s, ?t=1h2m3s), still emitted by
# older clients and re-pasted for years.
_HMS_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")

# Quoted back to the user when their timestamp does not parse; beside the regex
# so the accepted shapes have one definition.
TIMESTAMP_FORMATS: Final = "`90`, `90s`, `1m30s`, `2h30m15s`"


def parse_timestamp(raw: str) -> Optional[int]:
    """Seconds from a YouTube `t`/`ts` value (bare seconds or the colon-free
    HMS form), or None. Never raises: an unparseable timestamp must degrade to
    "play from the start", not to "this wasn't a URL" — a ValueError out of
    parse_url turns the whole link into a YouTube search for its text."""
    raw = raw.strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    match = _HMS_RE.fullmatch(raw)
    # An all-optional pattern also matches the empty string, so require a group.
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


# The unparseable `t=` is quoted inside a sentence, and a pasted URL fragment
# can be arbitrarily long.
_TIMESTAMP_ECHO_MAX = 40


def timestamp_warning(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> Optional[str]:
    """One line naming a `t=` value that did not parse, or None. Text rather
    than an embed: this module stays free of discord. Stated because something
    the user wrote in their own URL changed where the song starts."""
    if not isinstance(source, YTSource) or source.bad_timestamp is None:
        return None
    # safe_label: rendered inside a code span, and a backtick would close it.
    shown = safe_label(source.bad_timestamp, _TIMESTAMP_ECHO_MAX)
    return (
        f"⚠️ Couldn't read the timestamp `{shown}` in that link — starting from "
        f"the beginning. YouTube's `t=` takes {TIMESTAMP_FORMATS}."
    )


class URLSource(Enum):
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"
    SOUNDCLOUD = "soundcloud"
    # Any host not special-cased (tiktok, vimeo, …): handed straight to yt-dlp,
    # rejected only if it reports the site unsupported (YTDL.yt_source).
    OTHER = "other"
    # No URL at all — a plaintext search term.
    SEARCH = "search"


# The persisted "how was this asked for" token (play_history.query_source).
# Constants for the special-cased services, so youtu.be collapses onto its
# service; only the generic branch parses a hostname.
QUERY_SOURCE_YOUTUBE: Final[str] = "youtube.com"
QUERY_SOURCE_SPOTIFY: Final[str] = "spotify.com"
QUERY_SOURCE_SOUNDCLOUD: Final[str] = "soundcloud.com"
QUERY_SOURCE_SEARCH: Final[str] = "search"

# The token domain, mirrored by play_history's query_source CHECK; HistoryEntry
# clamps anything else to the unknown sentinel.
_QUERY_HOST_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9.-]{1,64}")


def normalize_query_host(host: str) -> str:
    """A parsed host as the archive stores it: lowercased, leading `www.`
    dropped, empty when not host-shaped (parse_url's domain group admits `_`,
    `+` and `|`)."""
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


# slots: one instance is retained per unresolved Spotify-playlist track (344 B
# -> 120 B each). Keep the class free of __dict__ readers (asdict/vars) and off
# any pickle path; it crosses to Redis as SearchQueueEntry JSON.
@dataclass(frozen=True, slots=True)
class YTSource:
    """A YouTube track or playlist: a pasted `url` or a `ytsearch:` term, with
    an optional `ts` start offset. `list_id`, `index` (1-based start position)
    and `video_id` (the `v=` of a playlist link, so the enqueue path can tell
    whether `ts` belongs to the track it queues first) are set on the playlist
    branch only."""

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
    # timestamp was found. Read once at the command layer, never persisted.
    bad_timestamp: Optional[str] = None
    # Ask-time analytics, carried onto the QueueObject this resolves into. The
    # default is for the PARSE layer, which runs before the mint: anything
    # handing a YTSource on to be queued must pass a real value — nothing
    # re-mints downstream, so an omission persists 0.0/0 with no log line.
    analytics: Analytics = ANALYTICS_ZERO
    # What the user typed, for -remove to match on. Same contract as analytics:
    # the parse layer leaves None, and an omission downstream is silent.
    user_input: Optional[str] = None
    # How the song was asked for (query_source_of). The one source type that
    # carries it, because it is the only one that survives into Redis, so a
    # lazily-resolved Spotify track still archives as Spotify.
    query_source: str = ""

    @property
    def playlist_url(self) -> str:
        """Canonical playlist URL for a type=PLAYLIST source: the pasted URL,
        else rebuilt from list_id."""
        return self.url or f"https://www.youtube.com/playlist?list={self.list_id}"


@dataclass(frozen=True)
class SoundcloudSource:
    # TODO: SoundCloud timestamp links are ignored, so the track always starts at 0:00.
    # parse_url() reads `t`/`ts` for youtube.com only, so `ts` is never populated.
    url: str
    ts: Optional[int] = None
    process: bool = False
    stype: URLSource = URLSource.SOUNDCLOUD


def query_source_of(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> str:
    """The query-source token for a parsed input. YTSource carries its own; the
    other two are consumed at resolve time, so their token is a constant."""
    if isinstance(source, YTSource):
        return source.query_source
    if isinstance(source, SpotifySource):
        return QUERY_SOURCE_SPOTIFY
    return QUERY_SOURCE_SOUNDCLOUD


def spotify_playlist_to_ytsearch(
    titles: list[str], *, analytics: Analytics, origin: str
) -> list[YTSource]:
    """Spotify playlist tracks as lazy YouTube searches, each resolved at
    dequeue. The Spotify token, the ask-time analytics (the head's; per-track
    positions derive from it) and `origin` (the pasted collection link) are set
    here, the last point that knows where these came from."""
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
    """YouTube's 1-based `index=` param, or None when unparseable or below 1.
    Never raises: parse_url's ValueError means "not a URL at all"."""
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def _last(args: dict[str, list[str]], key: str) -> Optional[str]:
    """Last value of a repeated query key."""
    values = args.get(key)
    return values[-1] if values else None


def parse_url(url: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Parse a URL into a source dataclass. Raises ValueError if no domain matches.
    domain regex groups: 1/2 = http/www prefix, 3 = domain, 4 = path."""
    # `-` must be in the class: without it "my-site.com" is read as the host
    # "site.com", which lands in the archive's query_source.
    domain_re = r"(https:\/\/)?(www\.)?([\w.-]+)\/([^?]*)"

    domain_match = re.search(domain_re, url)

    if not domain_match:
        raise ValueError(f"Not a recognised URL: {url!r}")

    domain = domain_match.group(3)
    # urllib handles percent-encoding and repeated keys, and works on
    # scheme-less input ("youtu.be/x?t=90"), which is what users paste.
    args = parse_qs(urlsplit(url).query)

    if domain in ("youtube.com", "youtu.be"):
        ts: Optional[int] = None
        list_id: Optional[str] = None
        # `t` and `ts` are one parameter under two names; `ts` wins if both.
        unparsed: list[str] = []
        for key in ("t", "ts"):
            for raw in args.get(key, []):
                parsed = parse_timestamp(raw)
                if parsed is None:
                    # Keep the URL and start at 0:00 (see parse_timestamp).
                    log.info(f"Ignoring unparseable timestamp {raw!r} in {url!r}")
                    unparsed.append(raw)
                else:
                    ts = parsed
        # Reported only when nothing usable was found: a bad `t` beside a good
        # `ts` does start where the user asked.
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
    if domain in ("open.spotify.com", "spotify.com"):
        path = domain_match.group(4).split("/")
        try:
            spotify_type = SpotifyType(path[0])
        except ValueError:
            raise Exception(f"Unknown Spotify track type: {path}")
        log.info(f"Spotify source ID: {path[1]}")
        return SpotifySource(spotify_type, path[1], process=True)
    if domain in ("soundcloud.com",):
        return SoundcloudSource(url, process=True)
    if "." in domain:
        # A real-looking domain that is not special-cased goes to yt-dlp, which
        # rejects an unsupported site itself (YTDL.yt_source). Routed like a
        # bare YouTube watch URL; the host is what the archive records.
        return YTSource(
            url=url,
            process=True,
            stype=URLSource.OTHER,
            query_source=normalize_query_host(domain),
        )
    # A dotless "host" (e.g. "98" from the search term "98/99"): ValueError
    # makes parse_input fall back to a YouTube search.
    raise ValueError(f"Not a recognised URL: {url!r}")


def unquote_argument(text: str) -> str:
    """Drop one matched pair of surrounding quotes. discord.py's `read_rest()`
    hands consume-rest arguments through with their quotes: parse_url drags the
    trailing one into the path, and a quoted search stores `"some song"` as the
    origin `-remove some song` cannot match. Only a whole argument wrapped at
    both ends, never down to nothing; runs twice (here and at the command), so
    it must be safe twice."""
    for quote in ('"', "'"):
        if len(text) > 2 and text.startswith(quote) and text.endswith(quote):
            return text[1:-1]
    return text


def parse_input(user_input: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Entry point for command input: parse_url for single-word input (URLs
    never contain spaces), else ytsearch. A single word with a slash ("98/99")
    reaches parse_url, raises on the dotless host, and falls back to search.
    Reads only what the caller passes, so a stripped flag is gone."""
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
