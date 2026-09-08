import asyncio
import copy
import os
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
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
from src.util import current_traceparent, fmt_duration, get_logger
from src.ytdlp_pool import YtdlpPool

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# The process's one extraction pool; its lifecycle lives on the object. Tests
# patch this name to swap in a thread-pool-backed instance.
ytdlp_pool = YtdlpPool()


class ExtractionError(Exception):
    """A yt-dlp failure flattened to cross the process boundary (yt-dlp's own
    errors carry a live traceback and will not pickle). Every field must have a
    default: BaseException.__reduce__ rebuilds as `cls(*args)`, and a required
    positional raises while UNPICKLING in the parent, bricking the pool."""

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
        # UnsupportedError, classified in the worker where the original type still
        # exists; yt_source reads it to say "not a site I can play".
        self.unsupported = unsupported

    @property
    def user_message(self) -> str:
        """The only yt-dlp text safe to show a user. expected=True is yt-dlp's own
        user-facing reason ("Private video"), minus its "ERROR: " prefix;
        expected=False can carry bug-report boilerplate, so it degrades to a
        generic line. The full message still reaches the span and logs."""
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
    # extract_info wraps an UnsupportedError in a DownloadError carrying the
    # original in exc_info, so check both.
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
    """The two fields read by direct subscript; yt-dlp always populates both
    once `data` is narrowed to a single video."""

    url: str
    webpage_url: str


class YTDLVideoMetadata(TypedDict, total=False):
    """The descriptive half of an info-dict, what _enrich_queueobject() and
    _record_serving_format() read."""

    title: str
    uploader: str
    uploader_url: str
    upload_date: str
    thumbnail: str
    description: str
    # float, not int: yt-dlp's SoundCloud extractor emits 942.762-style values.
    # Every read wraps this in int().
    duration: float
    tags: list[str]
    view_count: int
    like_count: int
    dislike_count: int
    abr: float
    asr: int
    acodec: str
    # Format shape, mirrored in _STREAM_CACHE_FIELDS: how _record_serving_format
    # tells an audio-only serve from a degraded muxed/HLS one.
    format_id: str
    protocol: str
    vcodec: str


class YTDLVideoInfo(YTDLVideoMetadata, _YTDLVideoInfoRequired, total=False):
    """A single video's fields once yt_source() has unwrapped "entries". Only
    url/webpage_url are guaranteed. Mirrors _STREAM_CACHE_FIELDS field-for-field
    plus `traceparent`."""

    # Stamped by _cache_stream, never by yt-dlp: the trace of the extraction that
    # minted this URL. Absent on a fresh extraction.
    traceparent: str


class YTDLEntry(YTDLVideoMetadata, total=False):
    """One leaf of yt-dlp's info-dict tree: a search result's full video, or a
    flat playlist's sparser id/title/url shape. Not recursive — yt_source skips
    nested playlists (`_type == "playlist"`)."""

    url: str
    webpage_url: str
    id: str
    _type: str


class YTDLExtractResult(YTDLEntry, total=False):
    """What _ytdlp_extract returns before narrowing: a YTDLEntry that MAY carry
    `entries`, so it cannot promise `url`. Call sites cast() once the shape is
    known."""

    entries: list[Optional[YTDLEntry]]


# Collections no caller reads once process=True has hoisted the served format's
# fields to the top level; commonly 100 KB-1 MB pickled worker->parent, so
# dropped in the worker. _STREAM_CACHE_FIELDS is what callers do consume.
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


# Bound at import so slimming survives tests patching `youtube_dl.YoutubeDL`.
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
    """Make a yt-dlp result cheap and safe to ship back from the worker:
    sanitize_info() reduces the live objects of a process=True info-dict to JSON
    primitives (without which every extraction fails to pickle), then the large
    unread collections are dropped, top level and per `entries` element — after
    the one thumbnail URL callers render is lifted out of one of them."""
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
    # cast: the checker cannot verify yt-dlp's untyped dict conforms.
    return cast(YTDLExtractResult, info)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractRequest:
    """One yt-dlp extraction as a single picklable payload. kw_only because
    `download` and `process` are both bool and a positional pair could
    transpose silently."""

    url: str
    opts: Any
    download: bool = False
    # True at every call site: process=False does no format selection.
    process: bool = True


def _ytdlp_extract(req: ExtractRequest) -> Optional[YTDLExtractResult]:
    """Extraction worker run in the process pool. Top-level so it is picklable."""
    url, opts = req.url, req.opts
    download, process = req.download, req.process
    # YoutubeDL.__init__ keeps the params dict by reference and writes into it;
    # the copy keeps the opts profile immutable across a worker's extractions.
    try:
        result = youtube_dl.YoutubeDL(copy.copy(opts)).extract_info(
            url, download=download, process=process
        )
    except YoutubeDLError as e:
        # `from e`: the stdlib stringifies the chain into the parent's __cause__.
        raise _classify_ytdlp_error(e) from e
    # Slimmed here so the unpicklable payload never enters the result queue.
    return _slim_info(result)


async def _run_extract(req: ExtractRequest) -> Optional[YTDLExtractResult]:
    """The single call site for _ytdlp_extract. Both module-level names it reads
    (`ytdlp_pool`, `_ytdlp_extract`) are resolved per call, never captured —
    tests patch them."""
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
    """Routes yt-dlp's diagnostics into our logger. yt-dlp announces what
    precedes an outage as warnings (formats skipped for a missing PO token,
    SABR-only streaming, signature failures), so those are the early-warning
    system. Progress chatter goes nowhere."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        log.warning(f"yt-dlp: {msg}")

    def error(self, msg: str) -> None:
        log.error(f"yt-dlp: {msg}")


_YTDLP_LOGGER = _YtdlpLogger()

# Client strategy — re-verify on every yt-dlp bump. No client is named: `default`
# is yt-dlp's own list, which upstream moves when YouTube breaks a client. Today
# it resolves to `visionos,web`. visionos carries playback (no PO token, no JS
# player, audio-only opus). The deno + yt-dlp-ejs extras and the bgutil sidecar
# exist only to keep `web` usable as the fallback: yt-dlp drops `web` without a
# JS runtime, and its formats are withheld without a GVS token. The bgutil pin
# tracks the compose image tag by hand. Revoked URLs are a separate mechanism
# (_resolve_playable_stream's probe-and-re-extract). `-tv_simply` is a no-op
# against today's default; kept as a guard if it returns.
# See docs/ARCHITECTURE.md#yt-dlp-client-strategy.
_EXTRACTOR_ARGS = {
    "youtube": {
        "player_client": ["default", "-tv_simply"],
    },
    # Without this, a search or playlist extraction opens by downloading the
    # ~880 KB homepage for a ytcfg only cookie-authenticated playlists read, and
    # we send none: 0.35s of every search. youtube:search and youtube:tab both
    # read this key, so one entry covers both.
    "youtubetab": {"skip": ["webpage"]},
    # Set explicitly so a provider living elsewhere overrides via env, not code.
    "youtubepot-bgutilhttp": {
        "base_url": [os.environ.get("POT_PROVIDER_URL", "http://127.0.0.1:4416")],
    },
}

_YTDL_BASE_OPTS = {
    "quiet": True,  # diagnostics reach us via `logger`, not stdout
    "no_warnings": False,  # warnings are the early-warning system (_YtdlpLogger)
    "logger": _YTDLP_LOGGER,
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "source_address": "0.0.0.0",
    "socket_timeout": 30,
    "extractor_args": _EXTRACTOR_ARGS,
    # No rm_cachedir: yt-dlp's JS player cache means the signature JS is fetched
    # only on a new player version, not per call.
}

# webpage_url → CDN stream URL. The 360p cap matters on the muxed rung: ffmpeg's
# -vn discards the picture, and plain `best` would stream ~120 MB of 1080p per
# song where 360p (itag 18 / HLS 93) carries the same mp4a audio. Bare `best` is
# the last rung, for videos with nothing ≤360p.
_YTDL_STREAM_OPTS = {
    **_YTDL_BASE_OPTS,
    "format": "bestaudio/best[height<=360]/best",
    "check_formats": False,
    "retries": 10,
}

# yt_source's unified extraction: one stream-opts call returns identity AND a
# playable URL, so both caches are written from one network round.
_YTDL_STREAM_SEARCH_OPTS = {
    **_YTDL_STREAM_OPTS,
    "default_search": "auto",
}

# yt_playlist: entry metadata without per-video stream extraction. extract_flat
# is "in_playlist", not True: a watch?v=…&list=… URL resolves to a url_result
# pointing at the playlist, and True stops there with no entries.
_YTDL_PLAYLIST_OPTS = {
    **_YTDL_BASE_OPTS,
    "noplaylist": False,
    "extract_flat": "in_playlist",
}

# Alias kept for external callers.
YTDL_OPTS = _YTDL_STREAM_OPTS

# Search-query → identity cache lifetime: short enough to follow YouTube ranking
# changes, long enough to skip the 3-4s search on repeat plays.
_YT_SOURCE_TTL = 3600  # 1 hour

# Ceiling on stream-URL caching. YouTube revokes well before the `expire` a URL
# carries, so this — not `expire` — keeps a dead URL from being replayed.
_STREAM_URL_MAX_TTL = 1800  # 30 minutes

# Cap on the pre-playback URL probe. Short because a resolve can pay it twice
# and exceeding it costs a cache entry, not only a verdict; an unconfirmed URL
# still plays, so firing early is cheap.
_STREAM_PROBE_TIMEOUT = float(os.environ.get("STREAM_PROBE_TIMEOUT_SECS", "2.0"))

# Consecutive UNCONFIRMED verdicts before the probe itself, not the URLs, is
# treated as the fault (blocked egress, DNS): past this, cached URLs are served
# as-is instead of dropped.
_UNCONFIRMED_STREAK_LIMIT = 3

# Ceiling on caching an UNCONFIRMED URL. Cached at all because probe failures
# are process-wide, and declining the write would stop anything repopulating
# the cache.
_UNCONFIRMED_STREAM_TTL = 120  # 2 minutes

# Fresh extractions one resolve may spend. A re-mint returns the same CDN host
# and format, so it cures a revoked signature and nothing else; yt-dlp already
# retries the player API internally. The cached-entry drop is not charged here.
_MAX_STREAM_EXTRACTIONS = 1

# Fields persisted in the stream URL cache.
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
        # Format shape, so cache hits stay attributable (_record_serving_format).
        "format_id",
        "protocol",
        "vcodec",
    }
)


# Once per format per process, so an outage does not warn on every song.
# Optional[str] because an info-dict can omit format_id.
_DEGRADED_FORMAT_WARNED: set[Optional[str]] = set()


def _record_serving_format(data: YTDLVideoMetadata) -> None:
    """Record the shape of the format a song will play from. yt-dlp strips
    per-format client attribution, so the shape is the signal: audio-only
    (vcodec "none") is healthy, muxed or HLS means the primary stopped serving
    and a fallback took over — one warning, since playback continues. A missing
    vcodec (older cache entries) counts as healthy."""
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


def _stream_cache_key(webpage_url: str) -> str:
    return f"ytdl:stream:{webpage_url}"


def _stream_url_ttl(stream_url: str) -> Optional[int]:
    """How long a stream URL may be cached, or None when it is not worth caching.
    `expire` is a query param on https formats but a path segment
    (`/expire/<epoch>/`) on the HLS manifests the muxed rung serves; missing
    either leaves that rung re-extracting on every play."""
    try:
        parsed = urlparse(stream_url)
        expire = int(parse_qs(parsed.query).get("expire", [0])[0])
        if not expire:
            match = re.search(r"/expire/(\d+)(?:/|$)", parsed.path)
            expire = int(match.group(1)) if match else 0
        ttl = min(expire - int(time.time()) - 1800, _STREAM_URL_MAX_TTL)
        return ttl if ttl > 60 else None
    # PEP 758 tuple catch (3.14+), normalized by ruff.
    except ValueError, IndexError:
        return None


class StreamProbe(Enum):
    """What a pre-playback probe learned about a stream URL. UNCONFIRMED must
    stay distinct: as DEAD it fails songs over a blocked probe, as PLAYABLE the
    URL is cached unverified and one unreachable CDN edge takes a song out for
    the full TTL."""

    PLAYABLE = "playable"
    DEAD = "dead"
    UNCONFIRMED = "unconfirmed"


# Consecutive UNCONFIRMED verdicts, process-wide; any completed probe resets it.
_unconfirmed_streak = 0


def probe_path_looks_broken() -> bool:
    """True once enough probes in a row failed to complete that the probe, not
    the URLs, is in doubt; callers then stop acting on UNCONFIRMED."""
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


# One session for every stream probe; the probe closes each connection, so this
# saves connector and SSL-context construction, not a handshake.
# See docs/ARCHITECTURE.md#stream-probe-session
_probe_session: Optional[aiohttp.ClientSession] = None
# One-shot: the playback loop outlives close_probe_session() by up to the 30s
# span flush, and a rebuild from there would strand a session nothing closes.
_probe_session_closed = False


class ProbeSessionClosed(RuntimeError):
    """Raised when a probe is attempted after the session is closed for good."""


def _get_probe_session() -> aiohttp.ClientSession:
    """The process's probe session, created on first use so it binds to the
    running loop. DummyCookieJar is required: `-play <any url>` reaches this
    session, so a real jar lets one guild set a `Domain=com` cookie that is
    replayed to googlevideo for every guild until restart. Rebuilt if closed
    from outside, never after close_probe_session()."""
    global _probe_session
    if _probe_session_closed:
        raise ProbeSessionClosed("stream-probe session is closed")
    if _probe_session is None or _probe_session.closed:
        _probe_session = aiohttp.ClientSession(
            # limit=0: the default 100 would queue the 101st probe against its
            # own 2s budget and report a healthy URL as UNCONFIRMED.
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=_STREAM_PROBE_TIMEOUT),
            cookie_jar=aiohttp.DummyCookieJar(),
            # The Location header is parsed as URL(loc, encoded=not this): a redirect
            # gets the same pre-encoded treatment _probe_target gives the first hop.
            requote_redirect_url=False,
        )
    return _probe_session


async def close_probe_session() -> None:
    """Release the probe session for good (MusicBotApp.close()); safe to call
    twice. Errors propagate — the call site guards this step."""
    global _probe_session, _probe_session_closed
    _probe_session_closed = True
    session, _probe_session = _probe_session, None
    if session is not None and not session.closed:
        await session.close()


async def _probe_stream_url(stream_url: str) -> StreamProbe:
    """What YouTube will do with this URL right now. A revoked URL makes ffmpeg
    403 and exit, which discord.py cannot tell from a song that ended, so probe
    exactly as ffmpeg opens it: a plain GET, no Range (a revoked URL still
    answers 206 to a ranged GET, and googlevideo rejects HEAD). The body is
    never read. A probe that never completed is UNCONFIRMED, not DEAD.

    The URL goes in pre-encoded: yarl requotes a plain string, which decodes the
    %3D/%3B inside an HLS manifest's SIGNED path and earns a 403 on a URL ffmpeg
    plays. yt-dlp emits these fully encoded."""
    if not stream_url:
        return StreamProbe.DEAD
    try:
        session = _get_probe_session()
        # read_bufsize=0 + close(), not release(): only the status line matters,
        # read_bufsize=0 + close(), not release(): only the status line matters,
        # and aiohttp otherwise buffers audio from the moment headers land.
        async with session.get(_probe_target(stream_url), read_bufsize=0) as response:
            # Only a definite client-side refusal is DEAD: 429 and 5xx say "not
            # right now", as a timeout does, and ffmpeg's -reconnect would very
            # likely have played the song.
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
        # Shutdown: the loop can reach here while close() waits on the flush.
        return _record_probe_outcome(StreamProbe.UNCONFIRMED)
    except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
        # The fact only — the three call sites each log their own policy.
        log.warning(f"stream URL probe did not complete: {e}")
        return _record_probe_outcome(StreamProbe.UNCONFIRMED)
    except Exception as e:
        # A bug in this function, not evidence about the URL: still UNCONFIRMED
        # (no probe defect may cost a song), but ERROR so a dead session does not
        # read as a flaky CDN forever.
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
) -> bool:
    """Persist a probed stream URL. True when an entry was written, False when
    the URL has no usable expiry. `max_ttl` caps the lifetime below the URL's
    own (an unconfirmed URL)."""
    # Absent keys are dropped, not written as None: YTDLVideoInfo types title as
    # str and treats absent fields as missing.
    stripped: dict[str, Any] = {
        k: data[k] for k in _STREAM_CACHE_FIELDS if data.get(k) is not None
    }
    # The extraction that minted the URL; the song that plays it links back.
    # See docs/ARCHITECTURE.md#observability.
    if traceparent := current_traceparent():
        stripped["traceparent"] = traceparent
    ttl = _stream_url_ttl(data.get("url", ""))
    if ttl:
        if max_ttl is not None:
            ttl = min(ttl, max_ttl)
        await cache_set(redis, cache_key, stripped, ttl)
        return True
    return False


async def _probe_and_cache(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> bool:
    """Success-path post-processing for a fresh stream extraction: record the
    serving format, probe, cache. True when an entry was written. A DEAD URL is
    never cached; an UNCONFIRMED one for _UNCONFIRMED_STREAM_TTL only."""
    span = trace.get_current_span()
    _record_serving_format(data)
    if _stream_url_ttl(data.get("url", "")) is None:
        # Uncacheable (no usable expiry, e.g. SoundCloud): a probe would spend a
        # network round only for _cache_stream to decline the write.
        return False
    probe = await _probe_stream_url(data.get("url", ""))
    span.set_attribute("ytdl.stream_probe", probe.value)
    if probe is StreamProbe.PLAYABLE:
        return await _cache_stream(redis, cache_key, data)
    if probe is StreamProbe.UNCONFIRMED:
        log.warning(
            "could not confirm a freshly extracted stream URL — caching it for "
            f"{_UNCONFIRMED_STREAM_TTL}s only"
        )
        return await _cache_stream(
            redis, cache_key, data, max_ttl=_UNCONFIRMED_STREAM_TTL
        )
    return False


async def invalidate_stream_cache(
    redis: Optional[aioredis.Redis], webpage_url: str
) -> bool:
    """Drop a song's cached stream URL so the next play re-extracts. Returns
    whether an entry existed."""
    return await cache_del(redis, _stream_cache_key(webpage_url))


@dataclass(frozen=True, slots=True)
class NpHostRef:
    """The live Now Playing host an interrupted fragment left behind, for its
    resume tail to dispose of. Runtime only: a Message cannot be serialized and
    own_embeds cannot be rebuilt from ids, so the wire fields alone can never
    strip-edit a retirement (MusicPlayer._retire_np_host)."""

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
    # False only for the crash-recovered head restore_crashed() re-queues: it was
    # never RPUSHed to the Redis list, so the loop must skip its redis_pop_for().
    # Read via guild_queue.is_persisted().
    persisted: bool = True
    # ── interjection flags ──
    # `interjected` is attribution only (span attribute); `is_resume` marks the
    # rebuilt tail of an interrupted song (ts = interrupt position) and selects
    # the "Resuming…" notice; `start_paused` re-pauses right after vc.play() so
    # a song paused at interjection returns parked.
    interjected: bool = False
    is_resume: bool = False
    start_paused: bool = False
    # Ask-time analytics. yt_source/yt_playlist REQUIRE it; the default exists
    # for rehydration and the carry sites, which always pass a real value.
    analytics: Analytics = ANALYTICS_ZERO
    # "search", or the host of the pasted link; "" = unknown (src.sources).
    query_source: str = ""
    # Epoch when the audio started, stamped by the loop at vc.play(). A resume
    # tail INHERITS it so every fragment of one play records the same start;
    # 0.0 = not played yet.
    played_at: float = 0.0
    # The NP card the interrupted fragment left frozen, set on a resume tail at
    # the fragment's iteration end and consumed when the tail starts. The ids
    # survive a restart, the ref does not, and only the ref can strip-edit a
    # response host. 0/0/False = nothing to clean up.
    np_message_id: int = 0
    np_channel_id: int = 0  # from message.channel.id — NEVER the home channel
    np_dedicated: bool = False  # a pure NP message (deletable) vs a response
    np_host_ref: Optional[NpHostRef] = field(default=None, repr=False)


def _enrich_queueobject(qo: QueueObject, data: YTDLVideoMetadata) -> None:
    """Back-fill fields unknown at enqueue time (flat playlist entries carry no
    duration/uploader/thumbnail) onto the same instance queue_embed() reads."""
    fetched_duration = data.get("duration")
    if qo.duration is None and fetched_duration is not None:
        qo.duration = int(fetched_duration)
    if qo.uploader is None:
        qo.uploader = data.get("uploader")
    if qo.thumbnail is None:
        qo.thumbnail = data.get("thumbnail")


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
    `url` is the watch page, a processed one's is the CDN stream URL — so sharing one
    job between them would persist the wrong URL as a song's identity."""
    return f"{cache_key}|{profile}"


async def _extract_once(
    key: str, request: ExtractRequest
) -> Optional[YTDLExtractResult]:
    """One extraction per distinct query at a time, process-wide: N users pasting
    the same link are N identical jobs against a four-worker pool, racing to write
    one cache entry. The first caller extracts and the rest await its outcome,
    exception included — a failure answers their query too. Keyed by _inflight_key;
    the entry is removed before the result is published, so the next call after a
    failure extracts again."""
    running = _INFLIGHT_EXTRACTS.get(key)
    if running is not None:
        trace.get_current_span().set_attribute("ytdl.extract_shared", True)
        return await asyncio.shield(running)
    future: asyncio.Future[Optional[YTDLExtractResult]] = (
        asyncio.get_running_loop().create_future()
    )
    _INFLIGHT_EXTRACTS[key] = future
    try:
        data = await _run_extract(request)
    except BaseException as e:
        _INFLIGHT_EXTRACTS.pop(key, None)
        if not future.done():
            future.set_exception(e)
        # Retrieved by every waiter, or by nobody — asyncio logs an unretrieved
        # future exception at GC otherwise, and a lone caller is the common case.
        future.exception()
        raise
    _INFLIGHT_EXTRACTS.pop(key, None)
    if not future.done():
        future.set_result(data)
    return data


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
        # Interjection flags carried from the QueueObject (see its field
        # comments). A resume tail and _neutralize_prefetch rebuild a QueueObject
        # from these, so every field the queue entry has must survive here.
        self.interjected: bool = interjected
        self.is_resume: bool = is_resume
        self.start_paused: bool = start_paused
        self.analytics: Analytics = analytics
        self.query_source: str = query_source
        self.user_input: Optional[str] = user_input
        self.persisted: bool = persisted
        self.played_at: float = played_at
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
        # for livestreams, and int(None) raises.
        self.duration_secs: int = int(data.get("duration") or 0)
        # fmt_duration everywhere, so the embeds and the bar agree on "3:30".
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
        """False when ffmpeg exited without delivering a frame (typically a 403
        on a revoked URL). discord.py hands that to `after` like a finished
        song, so the frame count is what tells them apart."""
        return self._frames_read > 0

    @property
    def elapsed_secs(self) -> float:
        """Seconds of audio delivered to the player so far. Frozen during any
        pause, because AudioPlayer does not call read() then."""
        return self._frames_read * (discord.opus.Encoder.FRAME_LENGTH / 1000.0)

    @property
    def position_secs(self) -> float:
        """True audio position: the -ss offset plus seconds delivered. The single
        source of truth for every position surface (bar, presence, pause
        confirmation, history)."""
        return self.start_offset + self.elapsed_secs

    @classmethod
    @_tracer.start_as_current_span("ytdl.prefetch_stream")
    async def prefetch_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis] = None,
    ) -> None:
        """Populate the stream URL cache for a queued song so yt_stream() is a
        cache hit by the time it plays. No-op with no redis or an already-cached
        URL; errors are logged and swallowed (yt_stream() extracts fresh)."""
        trace.get_current_span().set_attribute("ytdl.url", qo.webpage_url)
        if redis is None:
            trace.get_current_span().set_attribute("ytdl.skipped", True)
            return
        cache_key = _stream_cache_key(qo.webpage_url)
        cached: Optional[YTDLVideoInfo] = await cache_get(redis, cache_key)
        already_cached = cached is not None
        trace.get_current_span().set_attribute("ytdl.already_cached", already_cached)
        if already_cached:
            _enrich_queueobject(qo, cached)
            return
        try:
            # Single-video cast: stream opts on a watch URL never yield a
            # search/playlist wrapper. Single-flighted with the playback loop's
            # own resolve of the same song; a joined caller shares the winner's
            # dict, so neither path may mutate it.
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
            return
        if data is not None:
            await _probe_and_cache(redis, cache_key, data)
            _enrich_queueobject(qo, data)

    @classmethod
    async def _resolve_playable_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis],
        *,
        allow_reextract: bool = True,
    ) -> YTDLVideoInfo:
        """Resolve a song to stream data whose URL YouTube will serve. Every URL
        is probed first (a revoked one fails as silence, nothing logged); a
        revoked cached URL is dropped and re-extracted once.

        UNCONFIRMED is not DEAD: the URL still plays and is cached only briefly.
        A cached one is dropped and re-extracted for a freshly signed URL on the
        same edge, which cures an early revocation; that drop is never charged
        against _MAX_STREAM_EXTRACTIONS. Two brakes: once the probe path looks
        broken process-wide the cached URL is served untouched, and
        `allow_reextract=False` (the background prefetch) declines to re-extract.
        """
        span = trace.get_current_span()
        cache_key = _stream_cache_key(qo.webpage_url)

        data: Optional[YTDLVideoInfo] = await cache_get(redis, cache_key)
        span.set_attribute("ytdl.cache_hit", data is not None)

        extractions = 0
        while True:
            extracted_fresh = False
            if data is None:
                if extractions >= _MAX_STREAM_EXTRACTIONS:
                    break
                # Single-video cast, as in prefetch_stream, and single-flighted with
                # it. _extract_once pops its key before publishing the result, so the
                # re-extraction this loop performs after a DEAD probe starts a fresh
                # job rather than rejoining the one that produced the dead URL.
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

            probe = await _probe_stream_url(data.get("url", ""))
            span.set_attribute("ytdl.stream_probe", probe.value)

            if probe is StreamProbe.PLAYABLE:
                _record_serving_format(data)
                if extracted_fresh:
                    await _cache_stream(redis, cache_key, data)
                return data

            if probe is StreamProbe.UNCONFIRMED:
                if extracted_fresh:
                    # Nowhere better to go: play it, cache it briefly.
                    _record_serving_format(data)
                    await _cache_stream(
                        redis, cache_key, data, max_ttl=_UNCONFIRMED_STREAM_TTL
                    )
                    return data
                if probe_path_looks_broken() or not allow_reextract:
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
                # A fourth enum member must not silently inherit "revoked" below.
                raise AssertionError(f"unhandled stream probe verdict: {probe}")

            if not extracted_fresh:
                # Only a cached URL has an entry to drop.
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
        """Resolve a queued song to a playable YTDL source, from the stream-URL
        cache when present. `allow_reextract=False` keeps an unconfirmable
        cached URL rather than re-extracting: the background prefetch, whose
        cancellation every bulk mutation waits on, must not put an
        uninterruptible executor job in that path."""
        trace.get_current_span().set_attribute("ytdl.url", qo.webpage_url)

        data = await cls._resolve_playable_stream(
            qo, redis, allow_reextract=allow_reextract
        )

        ffmpeg_opts = cls.FFMPEG_OPTS.copy()
        if qo.ts is not None:
            # No user notice here: prefetch constructs this while the previous
            # song still plays. MusicPlayer's start path announces the offset.
            ffmpeg_opts["options"] += f" -ss {qo.ts}"
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
    ) -> QueueObject:
        """Resolve a search term or URL to a QueueObject, from the source cache
        when present. query_source, analytics and user_input are REQUIRED so the
        QueueObject leaves complete — a default would let a call site write a
        plausible zero. user_input None falls back to `search`, which is what
        the user typed only for a direct -play of one song; for an expanded
        collection `search` is a generated title, not the link -remove matches."""
        origin = user_input if user_input is not None else search
        trace.get_current_span().set_attribute("ytdl.search", search)
        # Normalised so "Destiny" and "destiny " both hit; ts is a per-request
        # playback offset, not part of the identity.
        cache_key = f"ytdl:source:{search.strip().lower()}"

        if redis is not None:
            cached = await cache_get(redis, cache_key)
            if cached is not None:
                trace.get_current_span().set_attribute("ytdl.source_cache_hit", True)
                trace.get_current_span().set_attribute(
                    "ytdl.result_title", cached.get("title", "")
                )
                return QueueObject(
                    cached["webpage_url"],
                    cached["title"],
                    requester,
                    ts=ts,
                    user_input=origin,
                    duration=cached.get("duration"),
                    uploader=cached.get("uploader"),
                    thumbnail=cached.get("thumbnail"),
                    query_source=query_source,
                    analytics=analytics,
                )

        trace.get_current_span().set_attribute("ytdl.source_cache_hit", False)

        # One stream-opts extraction yields identity AND a playable URL, filling
        # both caches from one network round. process stays True: unprocessed,
        # data["url"] is absent and the stream-cache write silently never happens.
        try:
            data = await _extract_once(
                _inflight_key(cache_key, "full"),
                ExtractRequest(
                    url=search, opts=_YTDL_STREAM_SEARCH_OPTS, download=download
                ),
            )
        except ExtractionError as e:
            # parse_url whitelists no domains; an unrecognised site arrives as
            # `unsupported`, flattened in the worker.
            if e.unsupported:
                trace.get_current_span().set_attribute("ytdl.unsupported_url", True)
                raise Exception(
                    f"This link isn't from a site I can play: {search}. Try a "
                    "YouTube, Spotify, or SoundCloud link, another yt-dlp-supported "
                    "site, or just search by name."
                ) from e
            raise
        if data is None:
            # TODO: Replace the bare Exception on yt-dlp failure with typed errors.
            # Every failure mode raises the same untyped "Could not find song", so
            # callers cannot tell "no such video" from "extractor broken" from
            # "network down", and nothing can retry selectively.
            raise Exception("Could not find song")

        # A wrapper carries the video in `entries`; a lone-video result already
        # is the entry.
        selected: YTDLEntry = data
        if "entries" in data:
            # TODO: Validate search results have a usable audio format before accepting.
            # An entry wins by being the first non-playlist result — nothing
            # checks for an https audio URL, so a format-less entry is accepted
            # here and only fails at stream time, looking unrelated.
            for entry in data["entries"]:
                if entry and entry.get("_type", None) != "playlist":
                    selected = entry
                    break
        if download:
            # TODO: Implement or remove yt_source's dead download=True parameter.
            # It is accepted but does nothing — the file is never named or
            # returned, so a caller passing it silently gets streaming behavior.
            pass

        # cast: `selected` is one entry now, which the checker cannot verify.
        video_data = cast(YTDLVideoInfo, selected)

        webpage_url = video_data["webpage_url"]
        title = video_data.get("title", "")
        raw_duration = video_data.get("duration")
        duration = int(raw_duration) if raw_duration is not None else None
        uploader = video_data.get("uploader")
        thumbnail = video_data.get("thumbnail")
        trace.get_current_span().set_attribute("ytdl.result_title", title)

        if redis is not None:
            await cache_set(
                redis,
                cache_key,
                {
                    "webpage_url": webpage_url,
                    "title": title,
                    "duration": duration,
                    "uploader": uploader,
                    "thumbnail": thumbnail,
                },
                _YT_SOURCE_TTL,
            )
            # Warm the stream cache from the same extraction. Awaited, not
            # spawned, so the write lands before queue_put's prefetch_stream
            # reads. A failed probe never fails yt_source.
            stream_cached = await _probe_and_cache(
                redis, _stream_cache_key(webpage_url), video_data
            )
            trace.get_current_span().set_attribute("ytdl.stream_cached", stream_cached)

        return QueueObject(
            webpage_url,
            title,
            requester,
            ts=ts,
            user_input=origin,
            duration=duration,
            uploader=uploader,
            thumbnail=thumbnail,
            query_source=query_source,
            analytics=analytics,
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
    ) -> list[QueueObject]:
        """Flat entry metadata for every video in a YouTube playlist. The three
        keyword fields are REQUIRED (see yt_source); `analytics` is the head's,
        with per-track positions derived below, and `user_input` is the pasted
        playlist link, carried onto every track for -remove."""
        trace.get_current_span().set_attribute("ytdl.url", url)
        data = await _run_extract(ExtractRequest(url=url, opts=_YTDL_PLAYLIST_OPTS))
        if data is None:
            raise Exception(f"Could not fetch YouTube playlist: {url}")
        # Optional element: yt-dlp emits a null entry for a deleted/private video.
        entries: list[Optional[YTDLEntry]] = data.get("entries") or []
        trace.get_current_span().set_attribute("ytdl.playlist_size", len(entries))
        qobjs: list[QueueObject] = []
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
            title = entry.get("title") or video_id
            video_url = (
                entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            )
            # A flat entry already carries what the queue card shows, so read it
            # here rather than leaving every track blank until its prefetch lands.
            # duration is None on a live entry; uploader falls back to channel.
            raw_duration = entry.get("duration")
            # Offset by tracks KEPT, never the enumerate index: skipped null
            # entries must not leave gaps in queue_position. replace() so a field
            # added to Analytics later is carried here.
            qobjs.append(
                QueueObject(
                    video_url,
                    title,
                    requester,
                    user_input=user_input,
                    duration=int(raw_duration) if raw_duration is not None else None,
                    uploader=entry.get("uploader") or entry.get("channel"),
                    thumbnail=entry.get("thumbnail"),
                    query_source=query_source,
                    analytics=replace(
                        analytics,
                        queue_position=analytics.queue_position + len(qobjs),
                    ),
                )
            )
        return qobjs
