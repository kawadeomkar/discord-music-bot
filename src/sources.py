import re
from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal, Optional, Union
from urllib.parse import parse_qs, urlsplit

from src.guild_state import ANALYTICS_ZERO, Analytics
from src.util import get_logger, safe_label

if TYPE_CHECKING:
    # Annotation only: parsing stays free of the Spotify client.
    from src.spotify import SpotifyTrack

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
# Likewise the path segment an unsupported Spotify link names as its type.
_SPOTIFY_KIND_ECHO_MAX = 40
# Spotify ids are base62.
_SPOTIFY_ID_RE = re.compile(r"[A-Za-z0-9]+")


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
    dropped, empty when not host-shaped (parse_url's host group admits `_` and
    non-ASCII letters)."""
    cleaned = host.strip().lower().removeprefix("www.")
    return cleaned if _QUERY_HOST_RE.fullmatch(cleaned) else ""


class SpotifyType(Enum):
    TRACK = "track"
    PLAYLIST = "playlist"
    ALBUM = "album"


class UnsupportedSpotifyLinkError(Exception):
    """A Spotify URL this bot cannot queue: a type it does not take (/artist/,
    /show/), a type with no id, or an id that is not one. Not a ValueError, which
    parse_input reads as "not a URL" and turns into a YouTube search for the link
    itself."""

    @property
    def user_message(self) -> str:
        """The message, already written for the channel, under the name
        _command_error renders without the class-name prefix."""
        return str(self)


def is_mix(list_id: str) -> bool:
    """A YouTube Mix, which yt-dlp walks one window at a time. A curated
    `RDCLAK5uy_` list shares the prefix but has a page and a header count, and the
    tab extractor walks it without repeating itself."""
    return list_id.startswith("RD") and not list_id.startswith("RDCLAK5uy_")


class YTType(Enum):
    TRACK = "track"
    PLAYLIST = "playlist"


@dataclass(frozen=True)
class SpotifySource:
    type: SpotifyType
    id: str
    process: bool = True
    stype: URLSource = URLSource.SPOTIFY

    @property
    def url(self) -> str:
        """The canonical open.spotify.com link, without the locale or the share
        parameters the pasted one may carry."""
        return f"https://open.spotify.com/{self.type.value}/{self.id}"


# slots: one instance is retained per unresolved Spotify collection track (344 B
# -> 176 B each). Keep the class free of __dict__ readers (asdict/vars) and off
# any pickle path; it crosses to Redis as SearchQueueEntry JSON.
@dataclass(frozen=True, slots=True)
class YTSource:
    """A YouTube track or playlist: a pasted `url` or a `ytsearch:` term, with
    an optional `ts` start offset. `list_id`, `index` (1-based start position)
    and `video_id` (the `v=` of a playlist link, or a youtu.be link's path, so
    the enqueue path can tell whether `ts` belongs to the track it queues first)
    are set on the playlist branch only."""

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
    # Who queued a lazy search, as an ID: this module is discord-free, and the value
    # has to survive Redis. None on parse-time sources, which resolve inside the
    # command that built them, and on entries queued before the field existed.
    requester_id: Optional[int] = None
    # What a listing shows while the search is unresolved, under the names a
    # resolved song uses: the track's own title, its artists, its length in
    # seconds, and the page the title links to. A Spotify track's; None on a
    # typed search. The length is Spotify's, so an ETA built on it is an estimate.
    title: Optional[str] = None
    uploader: Optional[str] = None
    duration: Optional[int] = None
    webpage_url: Optional[str] = None

    @property
    def playlist_url(self) -> str:
        """Canonical playlist URL for a type=PLAYLIST source: the pasted URL,
        else rebuilt from list_id. One spelling for the enqueue/interject/resolve
        paths."""
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


# What the replies call a collection. A Literal, so a reply that takes one cannot
# be handed a guess, and one with no default cannot forget it.
CollectionNoun = Literal["playlist", "album"]


def collection_noun(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> CollectionNoun:
    """What the replies call a collection: a Spotify album is the one that is not
    a playlist."""
    is_album = isinstance(source, SpotifySource) and source.type is SpotifyType.ALBUM
    return "album" if is_album else "playlist"


def spotify_playlist_to_ytsearch(
    titles: list[str],
    *,
    analytics: Analytics,
    origin: str,
    requester_id: int,
    tracks: Sequence[SpotifyTrack] = (),
) -> list[YTSource]:
    """Spotify album or playlist tracks as lazy YouTube searches, each resolved
    at dequeue. The Spotify token, the ask-time analytics (the head's; per-track
    positions derive from it), `origin` (the pasted collection link) and the
    requester are set here, the last point that knows where these came from.
    `requester_id` has no default: a track without one is attributed at dequeue to
    whoever ran a command most recently. `tracks` is `titles` as a listing shows
    them, index for index; without it the searches carry no display fields."""
    rows: Sequence[Optional[SpotifyTrack]] = (
        tracks if len(tracks) == len(titles) else [None] * len(titles)
    )
    # One joined byline per distinct artist tuple: an album is usually one artist,
    # and a fresh join per track is the only string this pass keeps for the life of
    # the queue (measured ~800 KiB over 10,000 tracks).
    bylines: dict[tuple[str, ...], Optional[str]] = {}

    def byline(row: SpotifyTrack) -> Optional[str]:
        key = tuple(row.artists)
        if key not in bylines:
            bylines[key] = ", ".join(key) or None
        return bylines[key]

    return [
        YTSource(
            ytsearch=f"ytsearch:{title}",
            process=True,
            query_source=QUERY_SOURCE_SPOTIFY,
            analytics=replace(analytics, queue_position=analytics.queue_position + i),
            user_input=origin,
            requester_id=requester_id,
            title=row.name if row else None,
            uploader=byline(row) if row else None,
            duration=row.duration_secs if row else None,
            webpage_url=row.url if row else None,
        )
        for i, (title, row) in enumerate(zip(titles, rows))
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


# A token longer than this is searched for, never read as a link.
LINK_MAX_CHARS: Final = 2048

# Anchored, so re.match tries one start position, and every quantified run stops
# at a literal its own class cannot match, so even a failing match is linear. A
# scheme-less token is a link only when a dotted host is followed by `/`, which
# keeps `will.i.am` a search. See docs/ARCHITECTURE.md#source-resolution.
_LINK_RE: Final = re.compile(
    r"(?:https?://)?(?:www\.)?([\w-]+(?:\.[\w-]+)+)/", re.IGNORECASE
)

# `spotify:track:<id>`, and the legacy `spotify:user:<name>:playlist:<id>`.
_SPOTIFY_URI_RE: Final = re.compile(
    r"spotify:(?:user:[^:]+:)?(track|album|playlist|artist|episode|show)"
    r":([0-9A-Za-z]{22})"
)
_SPOTIFY_HOSTS: Final = ("open.spotify.com", "spotify.com", "play.spotify.com")

# What a sentence leaves on the end of a pasted link.
_TRAILING_PUNCTUATION: Final = ".,!?;:"
# The single-character pairs a link is wrapped in: `<link>` suppresses Discord's
# embed, `` `link` `` is a code span, and a sentence puts one in parentheses.
_WRAPPER_PAIRS: Final = ("<>", "``", "()")
# The deepest nesting a message renders is `||[text](<link>)||`.
_UNWRAP_PASSES: Final = 3


def _unwrap(token: str) -> str:
    """The link inside the markup a message wraps one in: the pairs above,
    `||link||` (a spoiler) and `[text](link)` (a masked link). String methods
    only, so the cost stays linear in the token."""
    for _ in range(_UNWRAP_PASSES):
        token = token.rstrip(_TRAILING_PUNCTUATION)
        if len(token) > 2 and token[0] + token[-1] in _WRAPPER_PAIRS:
            token = token[1:-1]
        elif len(token) > 4 and token.startswith("||") and token.endswith("||"):
            token = token[2:-2]
        elif token.startswith("[") and token.endswith(")") and "](" in token:
            token = token[token.index("](") + 2 : -1]
        else:
            return token
    return token.rstrip(_TRAILING_PUNCTUATION)


def _is_spotify_uri(token: str) -> bool:
    return token[:8].lower() == "spotify:"


def is_link(text: str) -> bool:
    """Whether parse_url reads this unwrapped token as a link rather than words
    to search with. The link-or-text decisions made outside the parser ask this,
    so none of them can disagree with the route a `-play` took."""
    if len(text) > LINK_MAX_CHARS or len(text.split(None, 1)) != 1:
        return False
    if _is_spotify_uri(text):
        return _SPOTIFY_URI_RE.fullmatch(text) is not None
    return _LINK_RE.match(text) is not None


def parse_url(url: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Parse one token into a source dataclass. Raises ValueError when the token
    is not a link, which parse_input answers with a search, and
    UnsupportedSpotifyLinkError for a Spotify link that is not a track or
    playlist."""
    return _parse_link(url, decode_nested=True)


def _parse_link(
    token: str, *, decode_nested: bool
) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    if len(token) > LINK_MAX_CHARS:
        raise ValueError("Too long to be a link")
    link = _unwrap(token)
    if _is_spotify_uri(link):
        uri = _SPOTIFY_URI_RE.fullmatch(link)
        if uri is None:
            raise ValueError(f"Not a recognised URL: {token!r}")
        return _spotify_source(uri.group(1), uri.group(2))
    match = _LINK_RE.match(link)
    if match is None:
        raise ValueError(f"Not a recognised URL: {token!r}")
    # Hosts are case-insensitive and yt-dlp's extractors are not, so the link
    # handed on has its scheme and host lowercased; the path keeps its case.
    link = link[: match.end()].lower() + link[match.end() :]
    host = match.group(1).lower()
    # urlsplit reads a scheme-less link's host as its path without the `//`.
    parts = urlsplit(link if "://" in link[: match.end()] else f"//{link}")
    args = parse_qs(parts.query)

    if host in ("youtube.com", "youtu.be"):
        if (
            decode_nested
            and host == "youtube.com"
            and parts.path == "/attribution_link"
        ):
            # A share link carrying the path it redirects to, decoded once.
            inner = _last(args, "u")
            if inner is not None and inner.startswith("/"):
                return _parse_link(
                    f"https://www.youtube.com{inner}", decode_nested=False
                )
        return _youtube_source(link, host, parts.path, args)
    if host in _SPOTIFY_HOSTS:
        return _spotify_source(*_spotify_path(parts.path))
    if host == "soundcloud.com":
        return SoundcloudSource(link, process=True)
    # A host that is not special-cased goes to yt-dlp, which rejects an
    # unsupported site itself (YTDL.yt_source). Routed like a bare YouTube watch
    # URL; the host is what the archive records.
    return YTSource(
        url=link,
        process=True,
        stype=URLSource.OTHER,
        query_source=normalize_query_host(host),
    )


def _youtube_source(
    link: str, host: str, path: str, args: dict[str, list[str]]
) -> YTSource:
    ts: Optional[int] = None
    # `t` and `ts` are one parameter under two names; `ts` wins if both.
    unparsed: list[str] = []
    for key in ("t", "ts"):
        for raw in args.get(key, []):
            parsed = parse_timestamp(raw)
            if parsed is None:
                # Keep the URL and start at 0:00 (see parse_timestamp).
                log.info(f"Ignoring unparseable timestamp {raw!r} in {link!r}")
                unparsed.append(raw)
            else:
                ts = parsed
    # Reported only when nothing usable was found: a bad `t` beside a good
    # `ts` does start where the user asked.
    bad_timestamp = unparsed[-1] if ts is None and unparsed else None
    list_id = _last(args, "list")
    if list_id is None:
        return YTSource(
            link,
            ts=ts,
            process=False,
            bad_timestamp=bad_timestamp,
            query_source=QUERY_SOURCE_YOUTUBE,
        )
    raw_index = _last(args, "index")
    # youtu.be carries its video in the path, where `v=` never appears.
    video_id = path[1:].split("/", 1)[0] if host == "youtu.be" else _last(args, "v")
    return YTSource(
        link,
        ts=ts,
        process=False,
        type=YTType.PLAYLIST,
        list_id=list_id,
        index=_playlist_index(raw_index) if raw_index is not None else None,
        video_id=video_id or None,
        bad_timestamp=bad_timestamp,
        query_source=QUERY_SOURCE_YOUTUBE,
    )


def _spotify_path(path: str) -> tuple[str, str]:
    """(kind, id) from a Spotify link's path. The `intl-<locale>` segment a
    localized client adds, the `embed` player's, and the legacy
    `/user/<name>/playlist/<id>` shape all name the same item."""
    segments = path.strip("/").split("/")
    if len(segments) >= 4 and segments[0] == "user" and segments[2] == "playlist":
        return "playlist", segments[3]
    if segments[0].startswith("intl-"):
        segments = segments[1:]
    if segments and segments[0] in ("embed", "embed-podcast"):
        segments = segments[1:]
    kind = segments[0] if segments else ""
    return kind, segments[1] if len(segments) > 1 else ""


def _spotify_source(kind: str, spotify_id: str) -> SpotifySource:
    """One Spotify link's source, or the refusal that says which part of it the
    bot cannot use. The supported list is built from the enum, so adding a type
    cannot leave a stale message behind."""
    try:
        spotify_type = SpotifyType(kind)
    except ValueError:
        values = [t.value for t in SpotifyType]
        supported = ", ".join(values[:-1]) + f" or {values[-1]}"
        # The segment is the user's own text on its way into an embed.
        shown = safe_label(kind, _SPOTIFY_KIND_ECHO_MAX)
        # `from None`: the enum lookup's ValueError is not the user's error.
        raise UnsupportedSpotifyLinkError(
            f"Spotify '{shown}' links aren't supported — try a {supported} link."
            if shown
            else f"That Spotify link doesn't point at a {supported}."
        ) from None
    if not spotify_id:
        raise UnsupportedSpotifyLinkError(
            f"That Spotify {kind} link has no id — copy the full link."
        )
    if not _SPOTIFY_ID_RE.fullmatch(spotify_id):
        # The id goes into the API path, the cache key and the card's link.
        raise UnsupportedSpotifyLinkError(
            f"That Spotify {kind} link's id doesn't look right — copy the link "
            "again from Spotify."
        )
    log.info(f"Spotify source ID: {spotify_id}")
    return SpotifySource(spotify_type, spotify_id, process=True)


def unquote_argument(text: str) -> str:
    """Drop one matched pair of surrounding quotes. discord.py's `read_rest()`
    hands consume-rest arguments through with their quotes: parse_url drags the
    trailing one into the path, and a quoted search stores `"some song"` as the
    origin `-remove some song` cannot match. Only a whole argument wrapped at
    both ends, never down to nothing; runs twice (here and at the command), so
    it must be safe twice. `<link>` is the same case: Discord's own wrapper for a
    link whose embed is suppressed, which -remove already strips from its needle."""
    for opener, closer in (('"', '"'), ("'", "'")):
        if len(text) > 2 and text.startswith(opener) and text.endswith(closer):
            return text[1:-1]
    if len(text) > 2 and text[0] == "<" and text[-1] == ">" and " " not in text:
        return text[1:-1]
    return text


def parse_input(user_input: str) -> Union[SpotifySource, YTSource, SoundcloudSource]:
    """Entry point for command input. What is one token once Discord's wrappers
    are off goes to parse_url (a masked link's text can hold spaces, a link
    cannot); anything else, or a token parse_url does not read as a link, is a
    YouTube search. Reads only what the caller passes, so a stripped flag is
    gone."""
    text = unquote_argument(" ".join(user_input.split()))
    link = _unwrap(text)
    if len(link.split(None, 1)) == 1:
        try:
            return parse_url(link)
        except ValueError:
            pass
    return YTSource(
        ytsearch=f"ytsearch:{text}",
        process=True,
        stype=URLSource.SEARCH,
        query_source=QUERY_SOURCE_SEARCH,
    )
