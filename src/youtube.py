import asyncio
import contextlib
import copy
import os
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from contextlib import AbstractAsyncContextManager
from typing import Any, Optional, TypedDict, Union, cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import discord
import yt_dlp as youtube_dl
from yarl import URL
from yt_dlp.utils import UnsupportedError, YoutubeDLError

import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.guild_state import ANALYTICS_ZERO, Analytics
from src.redis_client import cache_del, cache_get, cache_set
from src.telemetry import get_tracer
from src.util import current_traceparent, fmt_duration, get_logger, spawn_background
from src.ytdlp_pool import YtdlpPool

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# The process's one extraction pool. A module-level *binding*, not mutable state:
# production never reassigns it and the lifecycle lives on the object
# (src/ytdlp_pool.py). Tests patch this name to swap in a thread-pool-backed instance.
ytdlp_pool = YtdlpPool()


class ExtractionError(Exception):
    """A yt-dlp failure, flattened so it survives the process boundary: yt-dlp's own
    errors store sys.exc_info(), so __dict__ carries a live traceback and will not
    pickle. Every field here must have a default — BaseException.__reduce__ rebuilds
    as `cls(*args)`, so a required positional raises TypeError while UNPICKLING in the
    parent, killing the executor's result thread and bricking the pool permanently.
    """

    def __init__(
        self,
        message: str = "",
        original_type: str = "",
        expected: bool = False,
        video_id: str = "",
        cause_type: str = "",
        unsupported: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.original_type = original_type
        self.expected = expected
        self.video_id = video_id
        self.cause_type = cause_type
        # yt-dlp rejected the URL's site (UnsupportedError), classified in the worker
        # where the original structure still exists; yt_source reads it to surface
        # "not a site I can play" instead of the generic extractor error.
        self.unsupported = unsupported

    @property
    def user_message(self) -> str:
        """The only yt-dlp text safe to show a user; the full message still reaches the
        span and logs via record_span_error. expected=True is yt-dlp's own user-facing
        reason ("Private video", a geo-block), shown minus its "ERROR: " prefix;
        expected=False can carry yt-dlp's bug-report boilerplate, so it degrades to a
        generic line.
        """
        if not self.expected:
            return "Couldn't load this track — the extractor hit an unexpected error."
        prefix = "ERROR: "
        message = self.message
        if message.startswith(prefix):
            message = message[len(prefix) :]
        return message or "Couldn't load this track."


def _classify_ytdlp_error(e: BaseException) -> ExtractionError:
    """Mine the classification that exists only here, inside the worker."""
    inner = None
    exc_info = getattr(e, "exc_info", None)
    if isinstance(exc_info, tuple) and len(exc_info) == 3:
        inner = exc_info[1]
    cause = getattr(inner, "cause", None) or getattr(e, "cause", None)
    # extract_info wraps an UnsupportedError in a DownloadError carrying the original in
    # exc_info, so check both outer and inner. Flattened to a bool because the
    # UnsupportedError type itself cannot cross the process boundary.
    unsupported = isinstance(e, UnsupportedError) or isinstance(inner, UnsupportedError)
    return ExtractionError(
        message=str(e),
        original_type=type(e).__name__,
        expected=bool(getattr(inner, "expected", False)),
        video_id=str(getattr(inner, "video_id", "") or ""),
        cause_type=type(cause).__name__ if cause is not None else "",
        unsupported=unsupported,
    )


class _YTDLVideoInfoRequired(TypedDict):
    """`url`/`webpage_url` are the only fields this codebase accesses via
    direct subscript (`data["url"]`) rather than `.get()` — yt-dlp always
    populates both once `data` is narrowed to a single video."""

    url: str
    webpage_url: str


class YTDLVideoMetadata(TypedDict, total=False):
    """The descriptive half of an info-dict — everything but the two identity fields.
    Split out because _enrich_queueobject() and _record_serving_format() read only
    these; the full YTDLVideoInfo would demand a `url`/`webpage_url` they never touch."""

    title: str
    uploader: str
    uploader_url: str
    upload_date: str
    thumbnail: str
    description: str
    # float, not int: yt-dlp's SoundCloud extractor emits `float_or_none(scale=1000)`
    # (fixtures show 942.762) and this bot accepts SoundcloudSource. Every read below
    # wraps this in int() — that is the conversion, not a redundancy.
    duration: float
    tags: list[str]
    view_count: int
    like_count: int
    dislike_count: int
    abr: float
    asr: int
    acodec: str
    # Format-shape fields, mirroring the trio in _STREAM_CACHE_FIELDS — what
    # _record_serving_format reads to tell a healthy audio-only serve from a
    # degraded muxed/HLS one.
    format_id: str
    protocol: str
    vcodec: str


class YTDLVideoInfo(YTDLVideoMetadata, _YTDLVideoInfoRequired, total=False):
    """A single video's fields, once yt_source() has unwrapped "entries". Only
    url/webpage_url are guaranteed — any other field may be absent per
    extractor/client. Mirrors _STREAM_CACHE_FIELDS field-for-field, plus the one key
    below that no extraction produces; required keys go in _YTDLVideoInfoRequired,
    descriptive ones in YTDLVideoMetadata.
    """

    # Stamped by _cache_stream, never by yt-dlp: the trace of the extraction that
    # minted this URL. Absent on a fresh extraction.
    traceparent: str
    # Not from yt-dlp: stamped by _cache_stream when the URL probed PLAYABLE, so a
    # play seconds later can skip re-probing what was just confirmed. Absent on a
    # fresh info-dict and on entries written before the stamp existed.
    probed_at: float


class YTDLEntry(YTDLVideoMetadata, total=False):
    """One leaf of yt-dlp's info-dict tree: a search result's full video, or a flat
    playlist's sparser `id`/`title`/`url` shape (_YTDL_PLAYLIST_OPTS, extract_flat).
    Both fit because every key is optional. Deliberately not recursive — yt_source
    skips nested playlists (`_type == "playlist"`) rather than descending.
    """

    url: str
    webpage_url: str
    id: str
    _type: str
    # Flat-entry fields: `channel` is a lockupViewModel result's `uploader`, and
    # `live_status` is the only warning that a live entry's `duration` is None.
    channel: str
    live_status: str


class YTDLExtractResult(YTDLEntry, total=False):
    """What _ytdlp_extract/_slim_info return before narrowing: a YTDLEntry that MAY
    carry `entries` (search/playlist profiles do, stream profiles don't), so it cannot
    promise `url` the way YTDLVideoInfo does. Call sites cast() once the shape is known.
    """

    entries: list[Optional[YTDLEntry]]


# Collections no caller reads once process=True has hoisted the *served* format's
# fields to the top level: the whole `formats` ladder plus thumbnails/captions/etc.,
# commonly 100 KB-1 MB pickled worker->parent per extraction, so it is dropped in the
# worker. `_STREAM_CACHE_FIELDS` is the exhaustive list of what callers do consume.
_UNUSED_INFO_COLLECTIONS = frozenset(
    {
        "formats",
        "requested_formats",
        "requested_downloads",
        "thumbnails",
        "automatic_captions",
        "subtitles",
        "heatmap",
        "chapters",
    }
)


# Bound once at import to the real staticmethod rather than looked up per call, so
# slimming survives tests patching `youtube_dl.YoutubeDL` wholesale (which would
# otherwise stub out sanitize_info).
_sanitize_info = youtube_dl.YoutubeDL.sanitize_info


# Every character RFC 3986 allows in a URI unencoded, percent included. A yt-dlp
# `url` outside this set needs quoting — a raw space or non-ASCII reaches here from
# the sites URLSource.OTHER allows. Inside it, re-quoting breaks a signed HLS path.
_URI_SAFE = re.compile(r"^[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$")


def _probe_target(stream_url: str) -> URL:
    """The URL to probe. Pre-encoded when the string is already a valid URI, so yarl
    does not decode the %3D/%3B an HLS manifest signs inside its own path; quoted
    normally otherwise, since encoded=True would emit those bytes into the request
    line verbatim and earn a 400 for a URL ffmpeg would have played."""
    if _URI_SAFE.match(stream_url):
        return URL(stream_url, encoded=True)
    return URL(stream_url)


def _lift_thumbnail(info: dict[str, Any]) -> None:
    """Keep the one thumbnail URL callers render before `thumbnails` leaves with the
    other large collections. yt-dlp orders that list ascending by size, so the last
    entry is the largest; a `thumbnail` the extractor set itself wins."""
    if info.get("thumbnail"):
        return
    thumbs = info.get("thumbnails")
    if isinstance(thumbs, list) and thumbs:
        last = thumbs[-1]
        if isinstance(last, dict) and isinstance(last.get("url"), str):
            info["thumbnail"] = last["url"]


def _slim_info(info: Any) -> Optional[YTDLExtractResult]:
    """Make a yt-dlp result cheap and safe to ship back from the worker: sanitize_info()
    reduces the live objects a process=True info-dict carries (LazyList format ladders,
    a _YDLLogger, callables) to JSON primitives, without which every extraction fails on
    an opaque pickling error. The large collections it keeps but no caller reads are
    dropped here too, top level and per `entries` element — after the one thumbnail URL
    callers render is lifted out of one of them.
    """
    info = _sanitize_info(info)
    if not isinstance(info, dict):
        # extract_info and sanitize_info only ever return a dict or None.
        return None
    _lift_thumbnail(info)
    for key in _UNUSED_INFO_COLLECTIONS:
        info.pop(key, None)
    entries = info.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                _lift_thumbnail(entry)
                for key in _UNUSED_INFO_COLLECTIONS:
                    entry.pop(key, None)
    # cast, not a bare annotation: the checker cannot verify yt-dlp's untyped dict
    # conforms, and `grep cast(` is how those assertions are audited.
    return cast(YTDLExtractResult, info)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractRequest:
    """Everything one yt-dlp extraction needs, as a single picklable payload. kw_only
    because `download` and `process` are both bool, so a positional pair could transpose
    silently — same value, wrong meaning, invisible to pyright. Frozen + slots because it
    crosses a process boundary; use dataclasses.replace() for variants.
    """

    url: str
    opts: Any
    download: bool = False
    # True at every current call site. `process=False` returns flat metadata with no
    # format selection, which is why the direct-URL play path stopped using it.
    process: bool = True


def _ytdlp_extract(req: ExtractRequest) -> Optional[YTDLExtractResult]:
    """Extraction worker run in the process pool. Top-level so it's picklable and
    named in tracebacks. Takes the whole request as one argument so no call in the
    chain depends on the order of two interchangeable bools."""
    url, opts = req.url, req.opts
    download, process = req.download, req.process
    # YoutubeDL.__init__ keeps the params dict by reference and writes into it
    # (js_runtimes, http_headers, …); the copy keeps the opts profile immutable
    # across repeated extractions within a worker.
    try:
        result = youtube_dl.YoutubeDL(copy.copy(opts)).extract_info(
            url, download=download, process=process
        )
    except YoutubeDLError as e:
        # `from e`, not `from None`: the stdlib stringifies the whole chain into the
        # parent's __cause__ (_RemoteTraceback), preserving the original traceback.
        raise _classify_ytdlp_error(e) from e
    # Slimmed in the worker, not the parent, to keep the unpicklable/oversized
    # payload from ever entering the pool's result queue.
    return _slim_info(result)


async def _run_extract(req: ExtractRequest) -> Optional[YTDLExtractResult]:
    """Await a yt-dlp extraction on the shared process pool — the single call site for
    _ytdlp_extract, so every extraction path keeps the pool binding in one place. Both
    module-level names it reads (`ytdlp_pool`, `_ytdlp_extract`) are resolved per call,
    not captured; that is what keeps the seams the test suite patches working."""
    return await ytdlp_pool.run(_ytdlp_extract, req)


def warm_worker() -> None:
    """Handed to prewarm(), so it runs once per pool worker. The first YoutubeDL a
    process builds discovers plugins and probes for a JS runtime; measured at 136 ms
    against 61 ms for the next two in the same process, so paying the ~75 ms delta at
    startup keeps it off the first -play each worker serves. Top-level so it is
    picklable, like _ytdlp_extract."""
    # cast, as at every other yt-dlp boundary: the opts profiles are the plain dicts
    # yt-dlp accepts, against a params TypedDict the checker cannot match them to.
    youtube_dl.YoutubeDL(cast(Any, copy.copy(_YTDL_STREAM_OPTS)))


class _YtdlpLogger:
    """Routes yt-dlp's own diagnostics into our logger instead of dropping them.
    yt-dlp announces what *precedes* an outage as warnings — formats skipped for a
    missing GVS PO token, the SABR-only streaming experiment, signature/n-challenge
    failures — so those are the early-warning system. Progress chatter goes nowhere.
    """

    def debug(self, msg: str) -> None:
        # Both [debug] lines and ordinary per-video chatter ("Downloading android vr
        # player API JSON") land here. Neither earns a line per song.
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        log.warning(f"yt-dlp: {msg}")

    def error(self, msg: str) -> None:
        log.error(f"yt-dlp: {msg}")


_YTDLP_LOGGER = _YtdlpLogger()

# Client strategy. Relevant when bumping yt-dlp.
#
# We pick NO client: `default` is yt-dlp's own list, and tracking it is the point —
# upstream moves it when YouTube breaks a client. Names here are what `default`
# resolved to, not configuration: RE-VERIFY ON EVERY BUMP. Today, `visionos,web`.
#
# visionos carries playback (no PO token, no JS player, audio-only opus). The deno +
# yt-dlp-ejs extras and the bgutil sidecar exist only to keep `web` usable as a
# fallback: yt-dlp drops `web` without a JS runtime, and its formats are withheld
# without a GVS token. The bgutil pin tracks the compose image tag by hand.
# Revoked URLs are separate: _resolve_playable_stream()'s probe-and-re-extract.
# See docs/ARCHITECTURE.md#yt-dlp-client-strategy.
#
# `-tv_simply` is a no-op against today's defaults; kept as a guard if it returns.
_EXTRACTOR_ARGS = {
    "youtube": {
        "player_client": ["default", "-tv_simply"],
    },
    # Skips the ~880 KB youtube.com homepage a search or playlist extraction opens
    # with — 0.35s measured per search — for a ytcfg only cookie-authenticated
    # playlists read. youtube:search and youtube:tab both read this key.
    "youtubetab": {"skip": ["webpage"]},
    # The plugin's own default is already 127.0.0.1:4416; set explicitly so a
    # deployment where the provider lives elsewhere overrides via env, not code.
    "youtubepot-bgutilhttp": {
        "base_url": [os.environ.get("POT_PROVIDER_URL", "http://127.0.0.1:4416")],
    },
}

# Shared base opts for both extraction paths.
_YTDL_BASE_OPTS = {
    "quiet": True,  # keep yt-dlp off stdout; diagnostics reach us via `logger` instead
    "no_warnings": False,  # warnings are the early-warning system — see _YtdlpLogger
    "logger": _YTDLP_LOGGER,
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "source_address": "0.0.0.0",
    "socket_timeout": 30,
    "extractor_args": _EXTRACTOR_ARGS,
    # rm_cachedir intentionally absent: keeping yt-dlp's JS player cache means the
    # signature-decryption JS is fetched only on a new player version, not per call.
}

# Used by yt_stream / prefetch_stream: webpage_url → CDN stream URL.
# check_formats=False skips HEAD requests probing format availability.
# Format ladder: audio-only when available, else a *small* muxed format — ffmpeg's -vn
# keeps only the audio, so plain `best` would stream 1080p (~120MB/song) to throw the
# picture away, while 360p muxed (itag 18 / HLS 93) carries the same mp4a audio for a
# tenth of that. Bare `best` is the last rung, for videos with nothing ≤360p.
_YTDL_STREAM_OPTS = {
    **_YTDL_BASE_OPTS,
    "format": "bestaudio/best[height<=360]/best",
    "check_formats": False,
    "retries": 10,
}

# Used by yt_source: the unified single-extraction play path. One stream-opts extraction
# returns identity AND a playable stream URL, so a single call populates both the
# ytdl:source and ytdl:stream caches. Default_search is what the stream opts lack for
# bare search queries; retries stays at 10 because this call serves playback.
_YTDL_STREAM_SEARCH_OPTS = {
    **_YTDL_STREAM_OPTS,
    "default_search": "auto",
}

# Used by yt_playlist: entry metadata for every video in a playlist without
# extracting each one's stream URL. noplaylist=False overrides the base option so
# yt-dlp processes the full playlist, not just the first video.
# extract_flat is "in_playlist", not True: a pasted watch?v=…&list=… URL resolves
# to a url_result pointing at the playlist, and True stops there — yielding a
# _type="url" with no entries, so the playlist queued nothing. "in_playlist"
# resolves that one hop and still leaves the entries inside the playlist flat.
_YTDL_PLAYLIST_OPTS = {
    **_YTDL_BASE_OPTS,
    "noplaylist": False,
    "extract_flat": "in_playlist",
}

# yt_source(flat=True): one search POST answering with id/title/duration/uploader and
# a thumbnail, with no watch page and no player call. The `process=True` it runs under
# is load-bearing — see docs/ARCHITECTURE.md#resolve-mode.
_YTDL_FLAT_SEARCH_OPTS = {
    **_YTDL_PLAYLIST_OPTS,
    "default_search": "auto",
}

# Legacy alias kept so any external callers that imported YTDL_OPTS still work.
YTDL_OPTS = _YTDL_STREAM_OPTS

# Search-query → (webpage_url, title) cache lifetime, and the age past which a hit is
# still served but revalidated behind the reply. Entries are ~200 bytes and TTL'd, so
# they are eviction-safe; what used to bound them at an hour was ranking drift, which
# the revalidation now handles without charging a repeat play for it.
# See docs/ARCHITECTURE.md#source-cache-freshness.
_YT_SOURCE_TTL = 86400  # 24 hours
_YT_SOURCE_FRESH_SECS = 3600  # 1 hour

# Playlist entry-list cache lifetime. Short, and deliberately not the source cache's
# day: a playlist is editable, and the entry count and order are what the callers'
# `&index=` handling and every "Queued playlist — N songs" line are built from. It
# buys the two cases that hurt — the same collection pasted twice, and a re-paste
# inside the window, which at 5,547 entries is a 99s extraction.
_YT_PLAYLIST_TTL = 900  # 15 minutes

# Revalidations in flight. Module-level because the refresh outlives the command that
# noticed the staleness — the reply has already been served from the stale entry.
_SOURCE_REVALIDATIONS: set[asyncio.Task[Any]] = set()

# Ceiling on stream-URL caching. YouTube revokes these well before the `expire` they
# carry (see _stream_url_ttl), so this — not `expire` — keeps a dead URL from being
# replayed: re-extracting costs seconds, serving a revoked URL costs the song.
_STREAM_URL_MAX_TTL = 1800  # 30 minutes

# Cap on the pre-playback URL probe. Short because a resolve can pay it TWICE (cached
# entry, then the URL that replaces it) and exceeding it costs a cache entry rather
# than only a verdict. An unconfirmed URL still plays, so firing early is cheap.
_STREAM_PROBE_TIMEOUT = float(os.environ.get("STREAM_PROBE_TIMEOUT_SECS", "2.0"))

# Consecutive UNCONFIRMED verdicts before the probe itself, rather than the URLs, is
# treated as the fault. Its failure modes (blocked egress, DNS, a stalled loop) are
# process-wide, so past this the cached-entry drop is suppressed and cached URLs are
# served as-is.
_UNCONFIRMED_STREAK_LIMIT = 3

# How long a PLAYABLE verdict stands in for a fresh probe. The resolve probes a URL
# and the play probes it again seconds later, against a signature good for half an
# hour. Short, because within it a revocation costs one ffmpeg spawn — `produced_audio`
# is what catches that, exactly as it does for a URL revoked between probe and read.
_PROBE_REUSE_SECS = 10.0

# Ceiling on how long an UNCONFIRMED URL may be cached. It is cached at all because
# probe failures are process-wide: declining the write would stop anything repopulating
# the cache and put every song through a fresh extraction. This bounds a wrong entry.
_UNCONFIRMED_STREAM_TTL = 120  # 2 minutes

# Fresh extractions one resolve may spend. A re-mint returns the same CDN host and the
# same format, so it cures a revoked signature and nothing else: a url that probes dead
# seconds after minting is refused for a reason an identical call cannot vary, and
# yt-dlp has already retried the player API internally (`extractor_retries`). The
# cached-entry drop is deliberately NOT charged against this.
_MAX_STREAM_EXTRACTIONS = 1

# Fields to persist in the stream URL cache — strips ephemeral/large fields.
_STREAM_CACHE_FIELDS = frozenset(
    {
        "url",
        "webpage_url",
        "title",
        "uploader",
        "uploader_url",
        "upload_date",
        "thumbnail",
        "description",
        "duration",
        "tags",
        "view_count",
        "like_count",
        "dislike_count",
        "abr",
        "asr",
        "acodec",
        # Format-shape fields — how _record_serving_format tells a healthy audio-only
        # serve from a degraded muxed/HLS one; kept so cache hits stay attributable.
        "format_id",
        "protocol",
        "vcodec",
    }
)


# Once per format per process, so a real outage doesn't warn on every song.
# Optional[str] because an info-dict can omit format_id — that gets its own slot.
_DEGRADED_FORMAT_WARNED: set[Optional[str]] = set()


def _record_serving_format(data: YTDLVideoMetadata) -> None:
    """Record the shape of the format a song will play from. yt-dlp strips per-format
    client attribution (`__yt_dlp_client`) before formats leave the extractor, so the
    format shape is the signal instead: audio-only (vcodec "none") is healthy, while
    muxed or HLS means the audio-only primary stopped serving and a fallback took over
    — one warning, since playback continues and nothing else surfaces it. A missing
    vcodec (pre-upgrade cache entries) counts as healthy. Phrased by SHAPE, never by
    client name, so it survives yt-dlp changing what `default` resolves to.
    """
    span = trace.get_current_span()
    format_id = data.get("format_id")
    span.set_attribute("ytdl.format_id", str(format_id))
    span.set_attribute("ytdl.protocol", str(data.get("protocol")))
    audio_only = data.get("vcodec") in (None, "none")
    span.set_attribute("ytdl.audio_only", audio_only)
    if not audio_only and format_id not in _DEGRADED_FORMAT_WARNED:
        _DEGRADED_FORMAT_WARNED.add(format_id)
        log.warning(
            f"songs are being served a muxed A/V format "
            f"(format_id={format_id}, protocol={data.get('protocol')}) — the "
            "audio-only primary is degraded and the player is on the fallback ladder"
        )


def _source_cache_key(search: str) -> str:
    """The ytdl:source key for a query. Case-folded so "Destiny" and "destiny " reach
    one entry — but never for a URL: YouTube video ids are case-sensitive, so `?v=aB`
    and `?v=Ab` would share an entry and the second would be served the first's song
    for the whole TTL."""
    query = search.strip()
    return f"ytdl:source:{query if '://' in query else query.lower()}"


def _source_entry_is_stale(cached: Any) -> bool:
    """Whether a source-cache hit is past _YT_SOURCE_FRESH_SECS and worth refreshing
    behind the reply. An entry from a build that stamped nothing reads as fresh — it
    carries that build's one-hour TTL too, so it expires before this could matter, and
    reading it as stale would revalidate every entry in the cache once on deploy."""
    stamped = cached.get("cached_at") if isinstance(cached, dict) else None
    if not isinstance(stamped, (int, float)):
        return False
    return time.time() - stamped > _YT_SOURCE_FRESH_SECS


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceIdentity:
    """What a resolve learns about a song before any format is chosen: the five fields
    ytdl:source stores and the enqueue card shows. kw_only — title, uploader and
    thumbnail are adjacent strings that would transpose silently."""

    webpage_url: str
    title: str
    duration: Optional[int]
    uploader: Optional[str]
    thumbnail: Optional[str]


def _identity_to_wire(identity: SourceIdentity) -> dict[str, Any]:
    """One definition of a cached identity's wire shape, shared by the source cache
    and the playlist cache so neither can drift from what reads it back."""
    return {
        "webpage_url": identity.webpage_url,
        "title": identity.title,
        "duration": identity.duration,
        "uploader": identity.uploader,
        "thumbnail": identity.thumbnail,
    }


def _identity_from_wire(entry: dict[str, Any]) -> SourceIdentity:
    """Rebuild a cached identity. Explicit `.get()`s rather than `SourceIdentity(**entry)`
    so an entry written before a field existed still parses."""
    return SourceIdentity(
        webpage_url=entry.get("webpage_url", ""),
        title=entry.get("title", ""),
        duration=entry.get("duration"),
        uploader=entry.get("uploader"),
        thumbnail=entry.get("thumbnail"),
    )


def _source_cache_value(identity: SourceIdentity) -> dict[str, Any]:
    """A source-cache entry: the identity plus the stamp _source_entry_is_stale reads."""
    return {**_identity_to_wire(identity), "cached_at": time.time()}


def _playlist_cache_key(url: str) -> str:
    """Keyed on the playlist's `list=` id rather than the pasted URL: one collection
    arrives as /playlist?list=X, as a watch link carrying `&index=`, and with a `&t=`
    on it, and those must share an entry rather than fragmenting the cache per entry
    point. A URL carrying no `list=` falls back to itself."""
    list_id = parse_qs(urlparse(url).query).get("list", [""])[0]
    return f"ytdl:playlist:{list_id or url}"


def _stream_cache_key(webpage_url: str) -> str:
    return f"ytdl:stream:{webpage_url}"


def _stream_url_ttl(stream_url: str) -> Optional[int]:
    """How long a stream URL may be cached, or None when it isn't worth caching.
    `expire` advertises a 6-hour window but YouTube revokes long before it, so
    _STREAM_URL_MAX_TTL is what bounds this in practice. `expire` is a query param on
    https formats but a path
    segment (`/expire/<epoch>/`) on the HLS manifests the muxed rung serves; missing either
    leaves that rung re-extracting 3-5s on every play, so both forms are read.
    """
    try:
        parsed = urlparse(stream_url)
        expire = int(parse_qs(parsed.query).get("expire", [0])[0])
        if not expire:
            match = re.search(r"/expire/(\d+)(?:/|$)", parsed.path)
            expire = int(match.group(1)) if match else 0
        ttl = min(expire - int(time.time()) - 1800, _STREAM_URL_MAX_TTL)
        return ttl if ttl > 60 else None
    # Bare `except A, B:` is PEP 758 (3.14+) tuple-catch syntax, not the py2
    # form — see guild_state._b_float's note on ruff's py314 normalization.
    except ValueError, IndexError:
        return None


class StreamProbe(Enum):
    """What a pre-playback probe learned about a stream URL. UNCONFIRMED must stay
    distinct: as DEAD it fails songs over a blocked probe, as PLAYABLE the URL is
    cached unverified and one unreachable CDN edge takes a song out for the full TTL.
    """

    PLAYABLE = "playable"
    DEAD = "dead"
    UNCONFIRMED = "unconfirmed"


# Consecutive UNCONFIRMED verdicts, process-wide. Reset by any probe that actually
# reached the host — a single completed probe proves the path works.
_unconfirmed_streak = 0


def probe_path_looks_broken() -> bool:
    """True once enough probes in a row failed to complete that the probe, rather than
    the URLs, is the thing in doubt. Callers use this to stop acting on UNCONFIRMED."""
    return _unconfirmed_streak >= _UNCONFIRMED_STREAK_LIMIT


def _record_probe_outcome(probe: StreamProbe) -> StreamProbe:
    global _unconfirmed_streak
    if probe is StreamProbe.UNCONFIRMED:
        _unconfirmed_streak += 1
        if _unconfirmed_streak == _UNCONFIRMED_STREAK_LIMIT:
            log.warning(
                f"{_unconfirmed_streak} stream probes in a row did not complete — "
                "treating the probe path as unhealthy and trusting cached URLs until "
                "one succeeds"
            )
    else:
        _unconfirmed_streak = 0
    return probe


# One session for every stream probe. The probe closes each connection rather
# than pooling it, so this saves the connector and SSL-context construction a
# per-call session paid — not a handshake. See docs/ARCHITECTURE.md#stream-probe-session
_probe_session: Optional[aiohttp.ClientSession] = None
# One-shot, like MusicBotApp.close()'s own _teardown_started: the playback loop
# outlives close_probe_session() by up to the 30s span flush, and rebuilding on
# a call from there would strand a session nothing closes.
_probe_session_closed = False


class ProbeSessionClosed(RuntimeError):
    """Raised when a probe is attempted after the session is closed for good."""


def _get_probe_session() -> aiohttp.ClientSession:
    """The process's probe session, created on first use so it binds to the
    running loop. DummyCookieJar is load-bearing, not tidiness: a status-line
    probe never needs a cookie, and `-play <any url>` reaches this session, so a
    real jar lets one guild set a `Domain=com` cookie that is then replayed to
    googlevideo for every guild until restart. Rebuilt if closed from outside,
    but never after close_probe_session(). Safe to call twice."""
    global _probe_session
    if _probe_session_closed:
        raise ProbeSessionClosed("stream-probe session is closed")
    if _probe_session is None or _probe_session.closed:
        _probe_session = aiohttp.ClientSession(
            # limit=0: one connector now serves every guild, and the default 100
            # would queue the 101st probe against its own 2s budget and report a
            # healthy URL as UNCONFIRMED. Costs nothing — the probe never pools.
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=_STREAM_PROBE_TIMEOUT),
            cookie_jar=aiohttp.DummyCookieJar(),
            # The Location header is parsed as URL(loc, encoded=not this): a redirect
            # gets the same pre-encoded treatment _probe_target gives the first hop.
            requote_redirect_url=False,
        )
    return _probe_session


async def close_probe_session() -> None:
    """Release the probe session for good. Called from MusicBotApp.close();
    safe to call twice. Errors propagate — the call site guards this step, like
    every other step in close()."""
    global _probe_session, _probe_session_closed
    _probe_session_closed = True
    session, _probe_session = _probe_session, None
    if session is not None and not session.closed:
        await session.close()


async def _probe_stream_url(stream_url: str) -> StreamProbe:
    """What YouTube will do with this stream URL right now. A revoked URL makes ffmpeg
    403 and exit, which discord.py cannot tell from a song that simply ended — silence,
    nothing logged. So probe exactly as ffmpeg opens it: a plain GET, no Range. A revoked
    URL still answers 206 to a *ranged* GET and googlevideo rejects HEAD, so either would
    report a dead URL as healthy. The body is never read.

    A probe that never completed says nothing about the URL, hence UNCONFIRMED rather
    than DEAD — the caller still plays it and lets ffmpeg judge.

    The URL goes in pre-encoded: yarl requotes a plain string, which decodes the
    %3D/%3B inside an HLS manifest's SIGNED path and earns a 403. yt-dlp emits these
    fully encoded.
    """
    if not stream_url:
        return StreamProbe.DEAD
    try:
        session = _get_probe_session()
        # read_bufsize=0 + close(), not release(): only the status line matters,
        # and aiohttp otherwise fills its StreamReader from the moment headers land
        # until the transport is paused — audio we pay for and discard. Nothing is
        # lost: an unread body means the connection could not be pooled anyway.
        async with session.get(_probe_target(stream_url), read_bufsize=0) as response:
            # Only a definite client-side refusal is DEAD. 429 and 5xx say "not
            # right now" exactly as a timeout does, and routing them to DEAD would
            # delete the cache entry and refuse a song ffmpeg's own -reconnect
            # would very likely have played.
            status = response.status
            response.close()
            if status < 400:
                return _record_probe_outcome(StreamProbe.PLAYABLE)
            if status == 429 or status >= 500:
                log.warning(
                    f"stream URL probe got HTTP {status}, treating as "
                    "unconfirmed rather than revoked"
                )
                return _record_probe_outcome(StreamProbe.UNCONFIRMED)
            return _record_probe_outcome(StreamProbe.DEAD)
    except ProbeSessionClosed:
        # Shutdown, not a defect: the loop can reach here while close() waits on
        # the span flush. Nothing to judge and nothing to cache.
        return _record_probe_outcome(StreamProbe.UNCONFIRMED)
    except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
        # State the fact only — the three call sites have three different policies,
        # and each logs its own.
        log.warning(f"stream URL probe did not complete: {e}")
        return _record_probe_outcome(StreamProbe.UNCONFIRMED)
    except Exception as e:
        # Anything outside the network set is a bug in this function, not evidence
        # about the URL. Still UNCONFIRMED, because no probe defect may cost a song
        # — but ERROR, or a dead session reads as a flaky CDN forever.
        log.error(
            f"stream URL probe failed unexpectedly: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return _record_probe_outcome(StreamProbe.UNCONFIRMED)


async def _cache_stream(
    redis: Optional[aioredis.Redis],
    cache_key: str,
    data: YTDLVideoInfo,
    *,
    max_ttl: Optional[int] = None,
    probed: bool = False,
) -> bool:
    """Persist a probed stream URL. True when an entry was written, False when the URL
    isn't worth caching (no usable expiry). `max_ttl` caps the lifetime below the URL's
    own — used for a URL that could not be confirmed. `probed` stamps the entry as
    CONFIRMED playable just now, which is what lets a play seconds later skip its own
    probe; an unconfirmed entry must never carry it."""
    # Absent keys are dropped, not written as None: `{"title": None}` would contradict
    # YTDLVideoInfo, which types title as str and treats absent fields as *missing*.
    stripped: dict[str, Any] = {
        k: data[k] for k in _STREAM_CACHE_FIELDS if data.get(k) is not None
    }
    # The span here is the extraction that minted the URL; the song that plays it
    # links back. See docs/ARCHITECTURE.md#observability.
    if traceparent := current_traceparent():
        stripped["traceparent"] = traceparent
    if probed:
        stripped["probed_at"] = time.time()
    ttl = _stream_url_ttl(data.get("url", ""))
    if ttl:
        if max_ttl is not None:
            ttl = min(ttl, max_ttl)
        await cache_set(redis, cache_key, stripped, ttl)
        return True
    return False


def _probe_is_recent(data: YTDLVideoInfo) -> bool:
    """Whether this cached entry's URL was confirmed playable inside
    _PROBE_REUSE_SECS. Only a PLAYABLE verdict stamps one, so an unconfirmed entry
    reads False and is probed as before. The window is short because the URL is what
    it re-checks — and `produced_audio` still catches a revocation inside it, at the
    cost of one ffmpeg spawn."""
    stamped = data.get("probed_at")
    return stamped is not None and time.time() - stamped < _PROBE_REUSE_SECS


async def _probe_and_cache(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> bool:
    """Success-path post-processing for a full stream extraction: record the serving
    format, probe the URL, cache it. True when an entry was written. Shared by
    prefetch_stream and yt_source so both write identical entries.

    A DEAD URL is never cached. An UNCONFIRMED one is, for _UNCONFIRMED_STREAM_TTL
    only: probe failures are process-wide, so declining the write would stop anything
    repopulating the cache and put every play through a fresh extraction."""
    span = trace.get_current_span()
    _record_serving_format(data)
    if _stream_url_ttl(data.get("url", "")) is None:
        # Uncacheable (no usable expiry — e.g. SoundCloud): probing would spend a
        # network round only for _cache_stream to decline the write anyway.
        return False
    probe = await _probe_stream_url(data.get("url", ""))
    span.set_attribute("ytdl.stream_probe", probe.value)
    if probe is StreamProbe.PLAYABLE:
        return await _cache_stream(redis, cache_key, data, probed=True)
    if probe is StreamProbe.UNCONFIRMED:
        log.warning(
            "could not confirm a freshly extracted stream URL — caching it for "
            f"{_UNCONFIRMED_STREAM_TTL}s only"
        )
        return await _cache_stream(
            redis, cache_key, data, max_ttl=_UNCONFIRMED_STREAM_TTL
        )
    return False


# Stream-cache warms in flight, keyed by their cache key. yt_source starts one from
# the extraction it already paid for and does NOT await it — the reply needs identity,
# not a probed URL. Everything that reads the stream cache joins through
# _stream_cache_get, so the warm is shared rather than raced into a second extraction
# of the same URL. See docs/ARCHITECTURE.md#warming-the-stream-cache.
_INFLIGHT_STREAM_WARMS: dict[str, asyncio.Future[bool]] = {}


def _start_stream_warm(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> asyncio.Future[bool]:
    """Probe and cache one stream URL in the background, or return the job already
    doing it. Nobody is obliged to await the result, so the done-callback retrieves
    any exception itself: an unretrieved one would surface at collection time, in a
    task with no caller left to name."""
    running = _INFLIGHT_STREAM_WARMS.get(cache_key)
    if running is not None:
        return running
    job = asyncio.ensure_future(_probe_and_cache(redis, cache_key, data))
    _INFLIGHT_STREAM_WARMS[cache_key] = job

    # Registered before anything can await the job, so the key is gone by the time a
    # joiner resumes and the next miss starts a fresh warm rather than joining a
    # settled one — the same ordering _extract_once depends on.
    def _settle(finished: asyncio.Future[bool]) -> None:
        _INFLIGHT_STREAM_WARMS.pop(cache_key, None)
        if not finished.cancelled() and finished.exception() is not None:
            log.warning(
                f"stream cache warm failed for {cache_key}: {finished.exception()!r}"
            )

    job.add_done_callback(_settle)
    return job


async def _stream_cache_get(
    redis: Optional[aioredis.Redis], cache_key: str
) -> Optional[YTDLVideoInfo]:
    """Read the stream cache, joining a warm still in flight for this key rather than
    starting a second extraction of the URL it is about to write. Shielded: the warm
    is shared, so one caller's cancellation must not take it from the others."""
    cached = cast(Optional[YTDLVideoInfo], await cache_get(redis, cache_key))
    if cached is not None:
        return cached
    warm = _INFLIGHT_STREAM_WARMS.get(cache_key)
    if warm is None:
        return None
    trace.get_current_span().set_attribute("ytdl.joined_stream_warm", True)
    with contextlib.suppress(Exception):
        await asyncio.shield(warm)
    return cast(Optional[YTDLVideoInfo], await cache_get(redis, cache_key))


async def invalidate_stream_cache(
    redis: Optional[aioredis.Redis], webpage_url: str
) -> bool:
    """Drop a song's cached stream URL so the next play re-extracts a fresh one.
    Returns whether an entry existed to drop."""
    return await cache_del(redis, _stream_cache_key(webpage_url))


@dataclass(frozen=True, slots=True)
class NpHostRef:
    """The live Now Playing host an interrupted fragment left behind, so the
    fragment's resume tail can dispose of that frozen card when it starts.

    Runtime only — a live Message cannot be serialized, and own_embeds cannot be
    reconstructed from ids, so the wire fields alone can never strip-edit a
    retirement (see MusicPlayer._retire_np_host)."""

    message: discord.Message
    own_embeds: list[discord.Embed]
    dedicated: bool


@dataclass
class QueueObject:
    """Song metadata in a queue before it's processed by YTDL"""

    webpage_url: str
    title: str
    requester: Union[discord.User, discord.Member]
    ts: Optional[int] = None
    user_input: Optional[str] = None
    duration: Optional[int] = None  # seconds, from yt-dlp at enqueue time
    uploader: Optional[str] = None  # YouTube channel name
    thumbnail: Optional[str] = None
    # False only for the crash-recovered "current song" that restore_crashed() re-queues:
    # it was never RPUSHed to the Redis queue list (it lives in current_song_url state),
    # so the loop must skip its redis_pop_for(). Read via guild_queue.is_persisted().
    persisted: bool = True
    # ── interjection flags ──
    # Queued by an interjection. Attribution only — they stack, so nothing reads
    # this except the span attribute.
    interjected: bool = False
    # The rebuilt tail of an interrupted song (ts = interrupt position). Drives which
    # notice the loop's start path sends: "Resuming…" for these, "Starting song at
    # Xs" for an ordinary ?t= entry.
    is_resume: bool = False
    # The interrupted song was paused at interjection time: the loop re-pauses
    # immediately after vc.play() so it returns parked.
    start_paused: bool = False
    # ── Ask-time analytics, set at construction and carried thereafter ──
    # queued_at + queue_position in one frozen container (guild_state.Analytics).
    # yt_source/yt_playlist REQUIRE it, so a QueueObject leaves them complete;
    # the default exists for rehydration and the carry sites, which always pass
    # a real value explicitly.
    analytics: Analytics = ANALYTICS_ZERO
    # How the song was asked for — "search", or the host of the pasted link.
    # Classified by src.sources at parse time and carried from there; "" = unknown.
    query_source: str = ""
    # Unix epoch when the audio started, stamped by the loop at vc.play(). A resume
    # tail INHERITS it, so every fragment of one play records the same start;
    # 0.0 = not played yet, which makes the stamp idempotent across fragments.
    played_at: float = 0.0
    # ── The Now Playing card the interrupted fragment left frozen ──
    # Set on a resume tail at the fragment's iteration end (see MusicPlayer's
    # _pending_resume_tail) and consumed when the tail starts. The ids survive a
    # restart; the ref does not, and only the ref can strip-edit a response host.
    # 0/0/False = nothing to clean up.
    np_message_id: int = 0
    np_channel_id: int = 0  # from message.channel.id — NEVER the home channel
    np_dedicated: bool = False  # a pure NP message (deletable) vs a response
    np_host_ref: Optional[NpHostRef] = field(default=None, repr=False)


def _enrich_queueobject(qo: QueueObject, data: YTDLVideoMetadata) -> None:
    """Back-fill QueueObject fields that couldn't be populated at enqueue time: a
    ytdl:source cache entry written before a field existed, or a flat entry whose
    renderer omitted one, hold None until their TTL lapses. prefetch_stream() has the
    full data and writes it back onto the same instance for queue_embed().
    """
    fetched_duration = data.get("duration")
    if qo.duration is None and fetched_duration is not None:
        qo.duration = int(fetched_duration)
    if qo.uploader is None:
        qo.uploader = data.get("uploader")
    if qo.thumbnail is None:
        qo.thumbnail = data.get("thumbnail")


# The served format's fields: present on a processed entry, never on a flat one. One
# reaching the flat path means the single-flight key stopped separating profiles, and
# its `url` is a CDN address rather than a watch page.
_PROCESSED_ENTRY_MARKERS = ("format_id", "protocol", "acodec")


def _flat_entry_fields(entry: YTDLEntry) -> Optional[SourceIdentity]:
    """The identity a flat search entry yields, or None when the entry cannot stand
    for a plain song — the caller then takes the full path. webpage_url is derived
    from `id` rather than read from `url`, so the flat and full paths agree by
    construction: it is the stream-cache key, what -remove matches, and part of
    play_history's dedup tuple."""
    video_id = entry.get("id")
    if not video_id:
        return None
    if any(entry.get(marker) is not None for marker in _PROCESSED_ENTRY_MARKERS):
        log.warning("Refusing a processed entry on the flat path: %s", video_id)
        return None
    # Every flat entry is `_type: "url"`, playlists and channels included, so the
    # extractor key is what separates a video from a collection. Absent falls through
    # to the guards below.
    if entry.get("ie_key") not in (None, "Youtube"):
        return None
    if entry.get("live_status") in ("is_live", "is_upcoming"):
        return None
    # Falsy, not just None: a zero duration renders a 0:00 card and stands in
    # ytdl:source for an hour.
    raw_duration = entry.get("duration")
    if not raw_duration:
        return None
    # A missing title means the renderer changed shape. Declined rather than filled
    # with the id, which would reach the card, the queue entry and play_history and
    # stay there — _enrich_queueobject does not back-fill titles.
    title = entry.get("title")
    if not title:
        return None
    return SourceIdentity(
        webpage_url=f"https://www.youtube.com/watch?v={video_id}",
        title=title,
        duration=int(raw_duration),
        uploader=entry.get("uploader") or entry.get("channel"),
        thumbnail=entry.get("thumbnail"),
    )


def _queue_object_from_flat_entry(
    entry: YTDLEntry,
    requester: Union[discord.User, discord.Member],
    *,
    query_source: str,
    analytics: Analytics,
    user_input: Optional[str],
    ts: Optional[int] = None,
) -> Optional[QueueObject]:
    """Build a QueueObject from a flat search entry, or None when the entry cannot be
    queued as a plain song."""
    identity = _flat_entry_fields(entry)
    if identity is None:
        return None
    return _queue_object_from_identity(
        identity,
        requester,
        query_source=query_source,
        analytics=analytics,
        user_input=user_input,
        ts=ts,
    )


def _queue_object_from_identity(
    identity: SourceIdentity,
    requester: Union[discord.User, discord.Member],
    *,
    query_source: str,
    analytics: Analytics,
    user_input: Optional[str],
    ts: Optional[int] = None,
) -> QueueObject:
    """The one place a resolved identity becomes a queue entry — shared by the flat
    path, the full path and a source-cache hit, so all three build the same object."""
    return QueueObject(
        identity.webpage_url,
        identity.title,
        requester,
        ts=ts,
        user_input=user_input,
        duration=identity.duration,
        uploader=identity.uploader,
        thumbnail=identity.thumbnail,
        query_source=query_source,
        analytics=analytics,
    )


_INFLIGHT_EXTRACTS: dict[str, asyncio.Future[Optional[YTDLExtractResult]]] = {}

_prefetch_gate: Optional[asyncio.Semaphore] = None
_prefetch_gate_loop: Optional[asyncio.AbstractEventLoop] = None


def prefetch_warm_slot() -> asyncio.Semaphore:
    """The bound on enqueue-time stream warms, which are spawned one per song and
    share the pool with the in-band resolves of every OTHER guild's playback loop.
    Half the workers, so one guild's paste burst can never hold all of them.
    Rebuilt when the running loop changes — a Semaphore binds to the first loop
    that awaits it and refuses another."""
    global _prefetch_gate, _prefetch_gate_loop
    loop = asyncio.get_running_loop()
    if _prefetch_gate is None or _prefetch_gate_loop is not loop:
        _prefetch_gate = asyncio.Semaphore(max(1, ytdlp_pool.max_workers // 2))
        _prefetch_gate_loop = loop
    return _prefetch_gate


def _inflight_key(cache_key: str, profile: str) -> str:
    """Single-flight key: the cache key the lookup missed on, plus the shape of the
    request. Results of different profiles are not interchangeable — a flat entry's
    `url` is the watch page, a processed one's is the CDN stream URL.
    See docs/ARCHITECTURE.md#resolve-mode."""
    return f"{cache_key}|{profile}"


def _held(slot: Optional[asyncio.Semaphore]) -> AbstractAsyncContextManager[Any]:
    """`slot` as an async context manager, or nothing to hold. A resolve reached
    outside a command (a lazy entry at dequeue, a test) passes None."""
    return slot if slot is not None else contextlib.nullcontext()


async def _gated_extract(
    request: ExtractRequest, pool_slot: Optional[asyncio.Semaphore]
) -> Optional[YTDLExtractResult]:
    """One extraction, holding the requesting guild's pool slot for as long as it
    runs. The wait for the slot belongs to the job rather than to _extract_once, so
    that a queued job is still a registered one and its key still deduplicates."""
    async with _held(pool_slot):
        return await _run_extract(request)


async def _extract_once(
    key: str, request: ExtractRequest, *, pool_slot: Optional[asyncio.Semaphore] = None
) -> Optional[YTDLExtractResult]:
    """One extraction per distinct query at a time, process-wide: N users pasting the
    same link are N identical jobs against a four-worker pool, racing to write one
    cache entry. The first caller starts the job and the rest await its outcome,
    exception included. Every caller reaches it through shield, the leader included:
    the key carries no guild, so one guild's cancellation must not reach another's.

    `pool_slot` is the guild's bound on workers held at once, and it is taken HERE
    rather than around the caller's whole resolve: a cache hit and a joined job hold
    no worker, and queueing those behind two extractions is what the bound is meant
    to prevent, not to cause. See docs/ARCHITECTURE.md#where-the-resolve-bound-is-taken."""
    running = _INFLIGHT_EXTRACTS.get(key)
    if running is not None:
        # A joiner holds no worker of its own — the leader's job is the one running.
        trace.get_current_span().set_attribute("ytdl.extract_shared", True)
        return await asyncio.shield(running)
    # The slot is taken INSIDE the job, so nothing is awaited between the read above
    # and the write below. Awaiting the slot here instead loses the single flight
    # exactly when it is worth most: with the semaphore full, every caller for one
    # key reads an empty registry, queues, and starts a job of its own.
    job = asyncio.ensure_future(_gated_extract(request, pool_slot))
    _INFLIGHT_EXTRACTS[key] = job
    # Registered before the shield, so it runs first: the key is gone before any
    # awaiter resumes, and a re-extraction starts a fresh job.
    job.add_done_callback(lambda _f: _INFLIGHT_EXTRACTS.pop(key, None))
    return await asyncio.shield(job)


async def _extract_for_source(
    key: str,
    request: ExtractRequest,
    search: str,
    *,
    pool_slot: Optional[asyncio.Semaphore] = None,
) -> Optional[YTDLExtractResult]:
    """yt_source's extraction, shared by its two request profiles so that an input
    yt-dlp refuses fails identically whichever one asked."""
    try:
        return await _extract_once(key, request, pool_slot=pool_slot)
    except ExtractionError as e:
        # parse_url whitelists no domains, so any dotted host lands here for yt-dlp
        # to accept or reject. An unrecognised site arrives flattened as `unsupported`
        # (UnsupportedError cannot cross the process boundary).
        if e.unsupported:
            trace.get_current_span().set_attribute("ytdl.unsupported_url", True)
            raise Exception(
                f"This link isn't from a site I can play: {search}. Try a "
                "YouTube, Spotify, or SoundCloud link, another yt-dlp-supported "
                "site, or just search by name."
            ) from e
        raise


def _first_video_entry(data: YTDLExtractResult) -> YTDLEntry:
    """A wrapper carries the video in `entries`; a lone-video result already is the
    entry. A result with no usable entry falls back to the wrapper."""
    if "entries" not in data:
        return data
    # TODO: Validate search results have a usable audio format before accepting.
    # An entry wins purely by being the first non-playlist result — nothing
    # checks for an https audio URL at a usable bitrate, so a format-less or
    # low-quality entry is accepted here and only blows up at stream time,
    # looking unrelated.
    for entry in data["entries"]:
        if entry and entry.get("_type", None) != "playlist":
            return entry
    return data


def _playlist_tracks(data: YTDLExtractResult, url: str) -> list[SourceIdentity]:
    """The playable entries of a flat playlist extraction, in order. Entries yt-dlp
    could not describe are dropped HERE, before the cache, so a hit cannot resurrect
    a deleted video the miss already refused."""
    # Optional in the element type, not re-annotated on the loop target: yt-dlp emits
    # a null entry for a deleted/private video, which is what the guard below skips —
    # declaring it non-optional excluded that case.
    entries: list[Optional[YTDLEntry]] = data.get("entries") or []
    tracks: list[SourceIdentity] = []
    for i, entry in enumerate(entries):
        if not entry:
            log.warning("Skipping null entry at playlist index %d for %s", i, url)
            continue
        video_id = entry.get("id")
        if not video_id:
            log.warning(
                "Skipping entry at playlist index %d (title=%r) — missing video ID for %s",
                i,
                entry.get("title"),
                url,
            )
            continue
        # A flat entry already carries what the card shows. duration is None on a
        # live entry; uploader falls back to channel on lockupViewModel entries.
        raw_duration = entry.get("duration")
        tracks.append(
            SourceIdentity(
                # Derived from `id`, never read from `url`: that is /shorts/{id} for a
                # Short, a second stream-cache key and a second play_history row for
                # one video.
                webpage_url=f"https://www.youtube.com/watch?v={video_id}",
                title=entry.get("title") or video_id,
                duration=int(raw_duration) if raw_duration is not None else None,
                uploader=entry.get("uploader") or entry.get("channel"),
                thumbnail=entry.get("thumbnail"),
            )
        )
    return tracks


async def _revalidate_source(
    redis: Optional[aioredis.Redis], cache_key: str, search: str
) -> None:
    """Refresh a stale source entry behind the reply that was served from it.

    Searches only: what goes stale is the query → video mapping YouTube's ranking
    decides, and a link's mapping is the link. One flat POST, so the refresh costs a
    fraction of the resolve it saves the next play. Nothing here may raise into the
    task — the entry it failed to refresh is still the one being served."""
    try:
        data = await _extract_once(
            _inflight_key(cache_key, "flat"),
            ExtractRequest(url=search, opts=_YTDL_FLAT_SEARCH_OPTS),
        )
        entry = _first_video_entry(data) if data is not None else None
        identity = _flat_entry_fields(entry) if entry is not None else None
        if identity is None:
            # A live result, one without a duration, or none at all. The stale entry
            # outlives this attempt rather than being dropped: it still plays.
            return
        await cache_set(redis, cache_key, _source_cache_value(identity), _YT_SOURCE_TTL)
        log.debug(f"refreshed a stale source cache entry: {cache_key}")
    except Exception as e:
        log.warning(f"source cache revalidation failed for {search!r}: {e!r}")


class YTDL(discord.FFmpegOpusAudio):
    FFMPEG_OPTS = {
        "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        "options": "-vn",
    }

    def __init__(
        self,
        channel: discord.TextChannel,
        url: str,
        *,
        data: YTDLVideoInfo,
        requester: Optional[Union[discord.User, discord.Member]] = None,
        start_offset: int = 0,
        before_options: Optional[str] = None,
        options: Optional[str] = None,
        interjected: bool = False,
        is_resume: bool = False,
        start_paused: bool = False,
        analytics: Analytics = ANALYTICS_ZERO,
        query_source: str = "",
        user_input: Optional[str] = None,
        persisted: bool = True,
        played_at: float = 0.0,
        np_message_id: int = 0,
        np_channel_id: int = 0,
        np_dedicated: bool = False,
        np_host_ref: Optional[NpHostRef] = None,
    ) -> None:
        super().__init__(
            url, executable="ffmpeg", before_options=before_options, options=options
        )

        self.requester = requester
        self.channel = channel
        # Seconds skipped via FFmpeg -ss; audio position = start_offset + elapsed.
        self.start_offset: int = start_offset
        # Interjection flags carried through from the QueueObject (see its fields).
        self.interjected: bool = interjected
        self.is_resume: bool = is_resume
        self.start_paused: bool = start_paused
        # Ask-time analytics carried from the QueueObject so the history entry
        # this song produces records where it started, not when it played.
        self.analytics: Analytics = analytics
        self.query_source: str = query_source
        # What the user typed, carried so -remove still matches a song that has
        # become a playing one — a resume tail is rebuilt from here.
        self.user_input: Optional[str] = user_input
        # Whether this song's entry is still on the Redis list: the loop reads it to
        # decide whether settling its claim LPOPs (QueueObject.persisted).
        self.persisted: bool = persisted
        # 0.0 until the loop stamps it at vc.play(); nonzero on a resume tail,
        # which inherits the interrupted song's start (see QueueObject.played_at).
        self.played_at: float = played_at
        # The previous fragment's frozen NP card, carried so the loop can dispose
        # of it once this one is up (see QueueObject's fields).
        self.np_message_id: int = np_message_id
        self.np_channel_id: int = np_channel_id
        self.np_dedicated: bool = np_dedicated
        self.np_host_ref: Optional[NpHostRef] = np_host_ref

        self.data = data
        self.uploader = data.get("uploader")
        self.uploader_url = data.get("uploader_url")
        self.date = data.get("upload_date") or "00000000"
        self.upload_date = self.date[6:8] + "." + self.date[4:6] + "." + self.date[0:4]
        self.title = data.get("title")
        self.thumbnail = data.get("thumbnail")
        self.description = data.get("description")
        # `or 0`, not a dict default: yt-dlp sets "duration" to None (not absent)
        # for livestreams and some age-gated videos, so data.get("duration", 0)
        # would hand int() a None and raise.
        self.duration_secs: int = int(data.get("duration") or 0)
        # fmt_duration, not str(timedelta): the latter spells 3m30s "0:03:30", which
        # left the recovered embed and presence card disagreeing with the bar's "3:30".
        self.duration = fmt_duration(self.duration_secs)
        self.tags = data.get("tags")
        self.webpage_url = data.get("webpage_url")
        self.views = data.get("view_count")
        self.likes = data.get("like_count")
        self.dislikes = data.get("dislike_count")
        self.url = data.get("url")
        self.abr = data.get("abr")
        self.asr = data.get("asr")
        self.acodec = data.get("acodec")

        self._frames_read: int = 0

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def read(self) -> bytes:
        """Read the next audio frame, tracking frame count for elapsed_secs."""
        data = super().read()
        if data:
            self._frames_read += 1
        return data

    @property
    def produced_audio(self) -> bool:
        """False when ffmpeg exited without delivering a frame — the stream never opened
        (typically a 403 on a revoked URL). discord.py hands that to `after` exactly like
        a finished song, so the frame count is the only thing that tells them apart."""
        return self._frames_read > 0

    @property
    def elapsed_secs(self) -> float:
        """Seconds of audio actually delivered to the player so far. Frozen during
        any pause — explicit (`-pause`) or involuntary (voice reconnect stall) —
        because AudioPlayer simply doesn't call read() during either."""
        return self._frames_read * (discord.opus.Encoder.FRAME_LENGTH / 1000.0)

    @property
    def position_secs(self) -> float:
        """True audio position: seconds skipped via FFmpeg -ss plus seconds actually
        delivered, frozen during any pause since elapsed_secs is. The single source of
        truth for every position surface — bar, presence, pause confirmation, history —
        so a ?t= start or a crash-recovered resume can't report different positions."""
        return self.start_offset + self.elapsed_secs

    @classmethod
    @_tracer.start_as_current_span("ytdl.prefetch_stream")
    async def prefetch_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis] = None,
    ) -> bool:
        """Eagerly populate the stream URL cache for a queued song, so yt_stream() is a
        cache hit by the time it plays. No-op with no redis or an already-cached URL;
        errors are logged and swallowed — yt_stream() recovers by extracting fresh.

        Returns False ONLY when an extraction was attempted and produced nothing, so
        an interjection can decline a head it cannot prove playable. Anything
        unprovable — no Redis — answers True.
        """
        trace.get_current_span().set_attribute("ytdl.url", qo.webpage_url)
        if redis is None:
            trace.get_current_span().set_attribute("ytdl.skipped", True)
            return True
        cache_key = _stream_cache_key(qo.webpage_url)
        cached = await _stream_cache_get(redis, cache_key)
        already_cached = cached is not None
        trace.get_current_span().set_attribute("ytdl.already_cached", already_cached)
        if already_cached:
            _enrich_queueobject(qo, cached)
            return True
        try:
            # Single-video cast: _YTDL_STREAM_OPTS on a watch URL never yields a
            # search/playlist wrapper, so the result is always a lone video here.
            # Single-flighted with the loop's own resolve — a joined caller shares the
            # winner's dict, so neither path may mutate it.
            data = cast(
                Optional[YTDLVideoInfo],
                await _extract_once(
                    _inflight_key(cache_key, "stream"),
                    ExtractRequest(url=qo.webpage_url, opts=_YTDL_STREAM_OPTS),
                ),
            )
            trace.get_current_span().set_attribute(
                "ytdl.extract_success", data is not None
            )
        except Exception as e:
            trace.get_current_span().record_exception(e)
            trace.get_current_span().set_status(
                StatusCode.ERROR, f"prefetch_stream failed: {e}"
            )
            log.warning(f"prefetch_stream failed for {qo.webpage_url}: {e}")
            return False
        if data is not None:
            # Shielded like every other join: a cancelled prefetch (every bulk queue
            # mutation cancels one) must leave the warm running for whoever else is
            # waiting on this key. The CancelledError still propagates from here.
            await asyncio.shield(_start_stream_warm(redis, cache_key, data))
            _enrich_queueobject(qo, data)
            return True
        return False

    @classmethod
    async def _resolve_playable_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis],
        *,
        allow_reextract: bool = True,
    ) -> YTDLVideoInfo:
        """Resolve a song to stream data whose URL YouTube will actually serve. Every URL
        is probed first, because a revoked one fails in the worst way: ffmpeg 403s and
        exits, discord.py reports a completed song, and the player advances in silence
        with nothing logged. A revoked URL is dropped from the cache and re-extracted
        once; once is enough.

        UNCONFIRMED is not DEAD: the URL still plays (ffmpeg is the judge) and is
        cached only briefly. A cached one is dropped and re-extracted for a freshly
        signed URL on the same edge and format, which cures an early revocation. That
        drop is FREE, never charged against _MAX_STREAM_EXTRACTIONS.

        Two brakes stop it becoming self-inflicted load: once the probe path looks
        broken process-wide the cached URL is served untouched, and
        `allow_reextract=False` (the background prefetch) declines to re-extract.
        """
        span = trace.get_current_span()
        cache_key = _stream_cache_key(qo.webpage_url)

        data = await _stream_cache_get(redis, cache_key)
        span.set_attribute("ytdl.cache_hit", data is not None)

        extractions = 0
        while True:
            extracted_fresh = False
            if data is None:
                if extractions >= _MAX_STREAM_EXTRACTIONS:
                    break
                # Single-video cast, as in prefetch_stream, and single-flighted with
                # it. _extract_once pops its key first, so this retry gets a fresh job.
                data = cast(
                    Optional[YTDLVideoInfo],
                    await _extract_once(
                        _inflight_key(cache_key, "stream"),
                        ExtractRequest(url=qo.webpage_url, opts=_YTDL_STREAM_OPTS),
                    ),
                )
                extractions += 1
                span.set_attribute("ytdl.extracted_fresh", True)
                if data is None:
                    raise RuntimeError("Could not extract stream data")
                extracted_fresh = True

            if not extracted_fresh and _probe_is_recent(data):
                # The resolve probed this URL seconds ago and cached the verdict with
                # it. Re-probing spends a network round trip to re-confirm a signature
                # good for half an hour.
                span.set_attribute("ytdl.probe_reused", True)
                _record_serving_format(data)
                return data

            probe = await _probe_stream_url(data.get("url", ""))
            span.set_attribute("ytdl.stream_probe", probe.value)

            if probe is StreamProbe.PLAYABLE:
                _record_serving_format(data)
                if extracted_fresh:
                    await _cache_stream(redis, cache_key, data, probed=True)
                return data

            if probe is StreamProbe.UNCONFIRMED:
                if extracted_fresh:
                    # Nowhere better to go: play it and cache it briefly, so the next
                    # play is not forced through the same extraction.
                    _record_serving_format(data)
                    await _cache_stream(
                        redis, cache_key, data, max_ttl=_UNCONFIRMED_STREAM_TTL
                    )
                    return data
                if probe_path_looks_broken() or not allow_reextract:
                    # The probe, not the URL, is what is in doubt — or a caller that
                    # must not block on an extraction. Serve what we have.
                    log.warning(
                        f"serving the cached stream URL for {qo.webpage_url} unverified "
                        f"(probe unhealthy={probe_path_looks_broken()}, "
                        f"reextract_allowed={allow_reextract})"
                    )
                    _record_serving_format(data)
                    return data
                log.warning(
                    f"could not confirm the cached stream URL for {qo.webpage_url} "
                    "— dropping it from the cache and re-extracting"
                )
                await cache_del(redis, cache_key)
                data = None
                continue

            if probe is not StreamProbe.DEAD:
                # Deliberately loud: the enum makes the last branch a catch-all, so a
                # fourth member would silently inherit "YouTube revoked this URL",
                # deleting cache entries with no type error and no failing test.
                raise AssertionError(f"unhandled stream probe verdict: {probe}")

            if not extracted_fresh:
                # Only a cached URL has an entry to drop — a fresh one is cached
                # exclusively on probe success, above.
                log.warning(
                    f"YouTube revoked the cached stream URL for {qo.webpage_url} "
                    "— dropping it from the cache and re-extracting"
                )
                await cache_del(redis, cache_key)
            elif extractions < _MAX_STREAM_EXTRACTIONS:
                log.warning(
                    f"freshly extracted stream URL for {qo.webpage_url} probed "
                    "dead — re-extracting with the budget that remains"
                )
            else:
                log.warning(
                    f"freshly extracted stream URL for {qo.webpage_url} probed "
                    "dead — giving up"
                )
            data = None

        raise RuntimeError(
            f"YouTube refused the audio stream for {qo.webpage_url} even after re-extracting"
        )

    @classmethod
    @_tracer.start_as_current_span("ytdl.yt_stream")
    async def yt_stream(
        cls,
        qo: QueueObject,
        channel: discord.TextChannel,
        *,
        volume: float = 1.0,
        redis: Optional[aioredis.Redis] = None,
        allow_reextract: bool = True,
    ) -> YTDL:
        """Resolve a queued song to a playable YTDL source, using the Redis
        stream-URL cache if present and extracting fresh via yt-dlp otherwise.

        `allow_reextract=False` keeps an unconfirmable cached URL rather than dropping
        and re-extracting it — for the background prefetch, whose cancellation is what
        every bulk mutation waits on, and which must not put an uninterruptible
        executor job in that path."""
        trace.get_current_span().set_attribute("ytdl.url", qo.webpage_url)

        data = await cls._resolve_playable_stream(
            qo, redis, allow_reextract=allow_reextract
        )

        ffmpeg_opts = cls.FFMPEG_OPTS.copy()
        if qo.ts is not None:
            ffmpeg_opts["options"] += f" -ss {qo.ts}"
            # No user notice here. This runs at CONSTRUCTION, which prefetch does
            # while the previous song is still playing, so announcing "Starting song
            # at Xs" from here fired at the wrong moment. MusicPlayer's start path
            # announces it — alongside "Resuming…", which was already moved there.
        if volume != 1.0:
            ffmpeg_opts["options"] += f" -filter:a volume={volume}"

        return cls(
            channel,
            data["url"],
            data=data,
            requester=qo.requester,
            start_offset=qo.ts or 0,
            before_options=ffmpeg_opts["before_options"],
            options=ffmpeg_opts["options"],
            interjected=qo.interjected,
            is_resume=qo.is_resume,
            start_paused=qo.start_paused,
            analytics=qo.analytics,
            query_source=qo.query_source,
            user_input=qo.user_input,
            persisted=qo.persisted,
            played_at=qo.played_at,
            np_message_id=qo.np_message_id,
            np_channel_id=qo.np_channel_id,
            np_dedicated=qo.np_dedicated,
            np_host_ref=qo.np_host_ref,
        )

    @classmethod
    @_tracer.start_as_current_span("ytdl.yt_source")
    async def yt_source(
        cls,
        requester: Union[discord.User, discord.Member],
        search: str,
        *,
        query_source: str,
        analytics: Analytics,
        user_input: Optional[str],
        download: bool = False,
        ts: Optional[int] = None,
        redis: Optional[aioredis.Redis] = None,
        flat: bool = False,
        pool_slot: Optional[asyncio.Semaphore] = None,
    ) -> QueueObject:
        """Resolve a search term or URL to a QueueObject via yt-dlp, using the
        Redis source cache if present.

        flat=True answers a SEARCH from one search POST when the first result is a
        plain video with a duration — no watch page, no player call, and no stream
        URL, which the prefetch fills later. Anything else takes the full extraction.

        query_source, analytics and user_input are REQUIRED so the QueueObject
        leaves here complete — a default would let a new call site forget them and
        write a plausible zero, permanently indistinguishable from a real value.

        user_input is what the user typed; None falls back to `search`, which is the
        same string only for a direct -play of one song. For an expanded collection
        `search` is a title this code generated, so the fallback would lose the album
        link -remove matches on."""
        origin = user_input if user_input is not None else search
        trace.get_current_span().set_attribute("ytdl.search", search)
        # ts is excluded — a per-request playback offset, not part of the identity.
        cache_key = _source_cache_key(search)

        if redis is not None:
            cached = await cache_get(redis, cache_key)
            if cached is not None:
                trace.get_current_span().set_attribute("ytdl.source_cache_hit", True)
                trace.get_current_span().set_attribute(
                    "ytdl.result_title", cached.get("title", "")
                )
                stale = _source_entry_is_stale(cached)
                trace.get_current_span().set_attribute("ytdl.source_stale", stale)
                if stale and "://" not in search.strip():
                    # Served now, refreshed behind the reply: what ages is the
                    # ranking a search resolved through, and a link's mapping is
                    # the link. Not awaited — this play uses the entry it has.
                    spawn_background(
                        _revalidate_source(redis, cache_key, search),
                        _SOURCE_REVALIDATIONS,
                    )
                return _queue_object_from_identity(
                    SourceIdentity(
                        webpage_url=cached["webpage_url"],
                        title=cached["title"],
                        duration=cached.get("duration"),
                        uploader=cached.get("uploader"),
                        thumbnail=cached.get("thumbnail"),
                    ),
                    requester,
                    query_source=query_source,
                    analytics=analytics,
                    user_input=origin,
                    ts=ts,
                )

        trace.get_current_span().set_attribute("ytdl.source_cache_hit", False)

        if flat:
            # Metadata only, no format selection. The stream URL is extracted later,
            # by the prefetch queue_put spawns.
            trace.get_current_span().set_attribute("ytdl.flat", True)
            flat_data = await _extract_for_source(
                _inflight_key(cache_key, "flat"),
                ExtractRequest(url=search, opts=_YTDL_FLAT_SEARCH_OPTS),
                search,
                pool_slot=pool_slot,
            )
            flat_qobj = (
                _queue_object_from_flat_entry(
                    _first_video_entry(flat_data),
                    requester,
                    query_source=query_source,
                    analytics=analytics,
                    user_input=origin,
                    ts=ts,
                )
                if flat_data is not None
                else None
            )
            if flat_qobj is not None:
                trace.get_current_span().set_attribute(
                    "ytdl.result_title", flat_qobj.title
                )
                if redis is not None:
                    await cache_set(
                        redis,
                        cache_key,
                        _source_cache_value(
                            SourceIdentity(
                                webpage_url=flat_qobj.webpage_url,
                                title=flat_qobj.title,
                                duration=flat_qobj.duration,
                                uploader=flat_qobj.uploader,
                                thumbnail=flat_qobj.thumbnail,
                            )
                        ),
                        _YT_SOURCE_TTL,
                    )
                return flat_qobj
            # A live entry, one without a duration, or no result: take the full path
            # on top of the flat POST already spent, re-fetching the same video.
            trace.get_current_span().set_attribute("ytdl.flat_fallback", True)

        # The only path for a link: one stream-opts call yields identity AND a
        # playable stream URL, filling the ytdl:source and ytdl:stream caches from one
        # network round. It runs processed — data["url"] is the selected format.
        data = await _extract_for_source(
            _inflight_key(cache_key, "full"),
            ExtractRequest(
                url=search, opts=_YTDL_STREAM_SEARCH_OPTS, download=download
            ),
            search,
            pool_slot=pool_slot,
        )
        if data is None:
            # TODO: Replace the bare Exception on yt-dlp failure with typed errors.
            # Every failure mode raises the same untyped "Could not find song", so
            # callers cannot tell
            # "no such video" from "extractor broken" from "network down" — all three
            # render the identical embed and nothing can retry selectively.
            raise Exception("Could not find song")

        # Separate from `data` because a leaf (YTDLEntry) is not assignable back to the
        # result type, and "raw result" vs "chosen entry" are two things.
        selected: YTDLEntry = _first_video_entry(data)
        if download:
            # TODO: Implement or remove yt_source's dead download=True parameter.
            # It is accepted but does nothing — the file is never named (prepare_filename)
            # or returned, so a caller passing it silently gets streaming behavior.
            pass

        # `selected` is one entry now. cast(), not a bare annotation: it asserts what
        # the checker cannot verify, and `grep cast(` audits those assertions.
        video_data = cast(YTDLVideoInfo, selected)

        webpage_url = video_data["webpage_url"]
        title = video_data.get("title", "")
        raw_duration = video_data.get("duration")
        duration = int(raw_duration) if raw_duration is not None else None
        uploader = video_data.get("uploader")
        thumbnail = video_data.get("thumbnail")
        trace.get_current_span().set_attribute("ytdl.result_title", title)
        identity = SourceIdentity(
            webpage_url=webpage_url,
            title=title,
            duration=duration,
            uploader=uploader,
            thumbnail=thumbnail,
        )

        if redis is not None:
            await cache_set(
                redis, cache_key, _source_cache_value(identity), _YT_SOURCE_TTL
            )
            # Warms the stream cache from the same extraction, so queue_put's
            # prefetch_stream is a cache hit. STARTED, not awaited: the reply needs
            # identity, and the probe behind that write is a network round trip
            # bounded only by _STREAM_PROBE_TIMEOUT. Whoever reaches the cache first
            # joins the same job through _stream_cache_get.
            _start_stream_warm(redis, _stream_cache_key(webpage_url), video_data)
            trace.get_current_span().set_attribute("ytdl.stream_warm_started", True)

        return _queue_object_from_identity(
            identity,
            requester,
            query_source=query_source,
            analytics=analytics,
            user_input=origin,
            ts=ts,
        )

    @staticmethod
    @_tracer.start_as_current_span("ytdl.yt_playlist")
    async def yt_playlist(
        url: str,
        requester: Union[discord.User, discord.Member],
        *,
        query_source: str,
        analytics: Analytics,
        user_input: str,
        redis: Optional[aioredis.Redis] = None,
        pool_slot: Optional[asyncio.Semaphore] = None,
    ) -> list[QueueObject]:
        """Fetch flat entry metadata for every video in a YouTube playlist.

        query_source, analytics and user_input are REQUIRED (see yt_source).
        `analytics` is the head's — track positions are derived per kept track
        below. `user_input` is the playlist link the user pasted, carried onto every
        track so -remove can match it.

        Cached and single-flighted: this is the most expensive resolve in the system
        (99s at 5,547 entries), and two users pasting one collection used to run two
        of them. See docs/ARCHITECTURE.md#the-playlist-cache."""
        span = trace.get_current_span()
        span.set_attribute("ytdl.url", url)
        cache_key = _playlist_cache_key(url)
        cached = await cache_get(redis, cache_key)
        span.set_attribute("ytdl.playlist_cache_hit", cached is not None)
        if cached is not None:
            tracks = [_identity_from_wire(entry) for entry in cached]
        else:
            data = await _extract_once(
                _inflight_key(cache_key, "playlist"),
                ExtractRequest(url=url, opts=_YTDL_PLAYLIST_OPTS),
                pool_slot=pool_slot,
            )
            if data is None:
                raise Exception(f"Could not fetch YouTube playlist: {url}")
            tracks = _playlist_tracks(data, url)
            if tracks:
                # An empty result is not cached: a private or unavailable playlist
                # extracts to no entries too, and a quarter hour is a long time to
                # answer a retry with the same nothing.
                await cache_set(
                    redis,
                    cache_key,
                    [_identity_to_wire(track) for track in tracks],
                    _YT_PLAYLIST_TTL,
                )
        span.set_attribute("ytdl.playlist_size", len(tracks))
        # Positions derive from the KEPT tracks, so the entries _playlist_tracks
        # dropped leave no gaps. replace() so a field added to Analytics is carried.
        return [
            _queue_object_from_identity(
                track,
                requester,
                query_source=query_source,
                analytics=replace(
                    analytics, queue_position=analytics.queue_position + offset
                ),
                user_input=user_input,
            )
            for offset, track in enumerate(tracks)
        ]
