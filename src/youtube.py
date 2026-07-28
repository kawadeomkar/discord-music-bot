import copy
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, TypedDict, Union, cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import discord
import yt_dlp as youtube_dl
from yt_dlp.utils import UnsupportedError, YoutubeDLError

import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.redis_client import cache_del, cache_get, cache_set
from src.telemetry import get_tracer
from src.util import fmt_duration, get_logger, notice_embed
from src.ytdlp_pool import YtdlpPool

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# The process's one extraction pool. A module-level *binding*, not mutable state:
# production never reassigns it and the lifecycle lives on the object
# (src/ytdlp_pool.py). Tests patch this name to swap in a thread-pool-backed
# instance — one seam, one place (tests/conftest.py).
ytdlp_pool = YtdlpPool()


class ExtractionError(Exception):
    """A yt-dlp failure, flattened so it survives the process boundary.

    yt-dlp's own errors cannot be pickled: ExtractorError.__init__ stores
    sys.exc_info(), so __dict__ carries a live traceback (and a _YDLLogger via
    .cause/.ie), and the real reason gets replaced by a pickling error. This
    carries the same information as flat fields, classified in the worker where
    the original structure still exists.

    Every field MUST have a default — that is what makes it picklable. The default
    BaseException.__reduce__ rebuilds it as `cls(*args)` (args being just the
    message) and restores the rest from __dict__, so a required positional raises
    TypeError on the PARENT side while unpickling, killing the executor's result
    thread and breaking the pool permanently. A round-trip test in
    tests/test_youtube.py guards this; an explicit __reduce__ would be redundant.
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
        # yt-dlp rejected the URL's site (UnsupportedError). Classified in the worker
        # where the original structure still exists; yt_source reads it to surface
        # "not a site I can play" instead of the generic extractor error. Defaulted
        # like every field so the pickle round-trip stays intact.
        self.unsupported = unsupported

    @property
    def user_message(self) -> str:
        """The line to show a Discord user. The full message always reaches
        the span/logs via record_span_error; this is only what is safe to surface.

        expected=True is yt-dlp's own user-facing reason ("Video unavailable",
        "Private video", a geo-block): show it, minus the "ERROR: " prefix.
        expected=False is an extractor/network fault whose raw text can carry
        yt-dlp's "report this ... on github.com/yt-dlp" bug-report boilerplate,
        which must never reach a user — so it degrades to a generic line.
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
    """The descriptive half of an info-dict — everything but the two identity
    fields. Split out because _enrich_queueobject() and _record_serving_format()
    read only these; the full YTDLVideoInfo would demand a `url`/`webpage_url`
    they never touch."""

    title: str
    uploader: str
    uploader_url: str
    upload_date: str
    thumbnail: str
    description: str
    # float, not int: yt-dlp's SoundCloud extractor emits
    # `'duration': float_or_none(scale=1000)` (fixtures show 942.762) and this bot
    # accepts SoundcloudSource. Every read below wraps this in int() — that is the
    # conversion, not a redundancy.
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
    extractor/client. Mirrors _STREAM_CACHE_FIELDS field-for-field.

    total=False on the empty body keeps a key added here optional like every other;
    required keys go in _YTDLVideoInfoRequired, descriptive ones in
    YTDLVideoMetadata.
    """


class YTDLEntry(YTDLVideoMetadata, total=False):
    """One leaf of yt-dlp's info-dict tree: a search result's full video, or a flat
    playlist's sparser `id`/`title`/`url` shape (_YTDL_PLAYLIST_OPTS, extract_flat).
    Both fit because every key is optional.

    Deliberately NOT recursive. yt-dlp *can* nest a playlist inside a playlist, but
    yt_source skips those (`_type == "playlist"`) rather than descending — a
    self-referential `entries` field would advertise nesting nothing here reads.
    """

    url: str
    webpage_url: str
    id: str
    _type: str


class YTDLExtractResult(YTDLEntry, total=False):
    """What _ytdlp_extract/_slim_info return before narrowing: a YTDLEntry that MAY
    carry `entries` (search/playlist profiles do, stream profiles don't), so it
    cannot promise `url` the way YTDLVideoInfo does. Call sites cast() once the
    shape is known — to YTDLVideoInfo for streams, after picking an entry for
    yt_source.
    """

    entries: list[Optional[YTDLEntry]]


# Collections no caller reads once process=True has hoisted the *served* format's
# fields to the top level. A real result carries the whole `formats` ladder plus
# thumbnails/captions/subtitles/heatmap — commonly 100 KB-1 MB of nested data,
# pickled worker->parent on every extraction, so it is dropped in the worker.
# `_STREAM_CACHE_FIELDS` is the exhaustive list of what callers do consume.
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


def _slim_info(info: Any) -> Optional[YTDLExtractResult]:
    """Make a yt-dlp result cheap and safe to ship back from the worker.

    A raw process=True info-dict carries live objects (LazyList format ladders, a
    _YDLLogger, callables) that would fail *every* extraction with an opaque
    pickling error; sanitize_info() reduces it to JSON primitives (LazyList->list,
    any non-primitive->repr, keeping only str/int/float/bool/list/dict/None). That
    closes the same contract for the return value that ExtractionError closes for
    the exception path.

    sanitize_info keeps the large collections no caller reads, so those are dropped
    here — top level and per `entries` element. Non-dict results pass through.
    """
    info = _sanitize_info(info)
    if not isinstance(info, dict):
        # extract_info and sanitize_info only ever return a dict or None.
        return None
    for field in _UNUSED_INFO_COLLECTIONS:
        info.pop(field, None)
    entries = info.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                for field in _UNUSED_INFO_COLLECTIONS:
                    entry.pop(field, None)
    # cast, not a bare annotation: the checker cannot verify yt-dlp's untyped dict
    # conforms, and `grep cast(` is how those assertions are audited.
    return cast(YTDLExtractResult, info)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractRequest:
    """Everything one yt-dlp extraction needs, as a single picklable payload.

    kw_only is load-bearing, not frozen: `download` and `process` are both bool, so
    a positional pair could transpose silently — same value, wrong meaning,
    invisible to pyright. (Plain frozen does NOT give this: `ExtractRequest("u",
    opts, True, False)` would be accepted.) Use dataclasses.replace() for variants.

    Frozen + slots because it crosses a process boundary: it must be picklable
    (verified against a real spawn pool) and must not be mutated by a worker in a
    way the parent never sees. `opts` is a plain options dict and must stay one.
    """

    url: str
    opts: Any
    download: bool = False
    # True at every current call site. `process=False` returns flat metadata with
    # no format selection, which is why the direct-URL play path stopped using it
    # (see yt_source). Kept as one of extract_info's own two switches.
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
    """Await a yt-dlp extraction on the shared process pool — the single call site
    for _ytdlp_extract, so every extraction path (prefetch_stream,
    _resolve_playable_stream, yt_source, yt_playlist) keeps the pool binding in
    one place. Takes the request whole so a new extraction option never ripples
    through this signature.

    Both module-level names it reads (`ytdlp_pool`, `_ytdlp_extract`) are resolved
    per call, not captured — that is what keeps the two seams the test suite patches
    (src.youtube.ytdlp_pool in tests/conftest.py, src.youtube._ytdlp_extract in ~29
    tests) working through this indirection."""
    return await ytdlp_pool.run(_ytdlp_extract, req)


class _YtdlpLogger:
    """Routes yt-dlp's own diagnostics into our logger instead of dropping them.

    yt-dlp announces what *precedes* an outage as warnings: formats skipped for a
    missing GVS PO token, "YouTube may have enabled the SABR-only streaming
    experiment", signature/n-challenge failures. Those were previously silenced
    (no_warnings), so the first sign of YouTube changing the rules would have been
    users reporting that songs no longer play. Progress chatter still goes nowhere —
    only warnings and errors.
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

# Client strategy: android_vr primary, web_safari as a *working* fallback.
#
# yt-dlp resolves `default` by JS-runtime availability: ('android_vr',) without one,
# ('android_vr', 'web_safari') with one. Deno (yt-dlp's `deno` extra, landing in the
# venv scripts dir where yt-dlp looks first) plus yt-dlp-ejs make web_safari's
# signature/n challenges solvable. Verified 2026-07: web_safari serves *muxed* formats
# only (HLS 91-96, https 18) — the `/best` leg of the selector picks one and ffmpeg's
# -vn drops the video. GVS PO tokens — minted by the bgutil-pot-provider sidecar
# (docker-compose.yml, port 4416) via the bgutil-ytdlp-pot-provider plugin, whose
# pyproject pin moves in lockstep with that compose image tag — are not yet
# *enforced* for muxed formats, but that is YouTube's documented
# trajectory (PO-Token-Guide: HLS exempt "currently"), and the sidecar is what keeps
# this fallback alive when it flips. android_vr needs none of it and stays first;
# fetch_pot=auto consults the provider only when needed.
#
# Degradation ladder — every rung lands on a previously-working configuration:
#   android_vr healthy → pre-fallback behavior (audio-only, e.g. 251/opus)
#   android_vr out     → web_safari muxed audio; WARNING via _record_serving_format
#   sidecar down       → plugin warns; web_safari works until POT enforcement lands
#   Deno broken        → yt-dlp reverts to the JS-less default (android_vr only)
# Revoked URLs are separate: _resolve_playable_stream()'s probe-and-re-extract, which
# now has two clients to heal from.
#
# `-tv_simply` is a no-op against today's defaults; kept as a guard if it returns.
_EXTRACTOR_ARGS = {
    "youtube": {
        "player_client": ["default", "-tv_simply"],
    },
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
#
# Format ladder: audio-only when available (healthy android_vr); otherwise a *small*
# muxed format, because ffmpeg's -vn keeps only the audio — plain `best` would stream
# 1080p (~120MB/song) to throw the picture away, while 360p muxed (itag 18 / HLS 93)
# carries the same mp4a audio for a tenth of that. Bare `best` is the final rung for
# videos with nothing ≤360p.
_YTDL_STREAM_OPTS = {
    **_YTDL_BASE_OPTS,
    "format": "bestaudio/best[height<=360]/best",
    "check_formats": False,
    "retries": 10,
}

# Used by yt_source: the unified single-extraction play path. One stream-opts
# extraction returns both the video's identity AND a playable stream URL, so a
# single call populates the ytdl:source and
# ytdl:stream caches — the previous source-opts search did a full extraction anyway
# and discarded the stream data, making prefetch_stream hit YouTube twice per cold
# play. default_search is what the stream opts lack for bare search queries; retries
# stays at 10 because this call now serves playback, not just metadata.
_YTDL_STREAM_SEARCH_OPTS = {
    **_YTDL_STREAM_OPTS,
    "default_search": "auto",
}

# Used by yt_playlist: entry metadata for every video in a playlist without
# extracting each one's stream URL. noplaylist=False overrides the base option so
# yt-dlp processes the full playlist, not just the first video.
_YTDL_PLAYLIST_OPTS = {
    **_YTDL_BASE_OPTS,
    "noplaylist": False,
    "extract_flat": True,
}

# Legacy alias kept so any external callers that imported YTDL_OPTS still work.
YTDL_OPTS = _YTDL_STREAM_OPTS

# Search-query → (webpage_url, title) cache lifetime. Short enough to pick up
# YouTube ranking changes, long enough to skip the 3-4s search on repeat plays.
_YT_SOURCE_TTL = 3600  # 1 hour

# Ceiling on stream-URL caching. YouTube revokes these well before the `expire` they
# carry (see _stream_url_ttl), so this — not `expire` — keeps a dead URL from being
# replayed: re-extracting costs seconds, serving a revoked URL costs the song.
_STREAM_URL_MAX_TTL = 1800  # 30 minutes

# Cap on the pre-playback URL probe: generous enough for a slow CDN, short enough
# never to add a noticeable pause before a song starts.
_STREAM_PROBE_TIMEOUT = 5.0  # seconds

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


# format_ids already warned about — once per format per process, so a real
# android_vr outage doesn't warn on every song. Optional[str] because an info-dict
# can omit format_id; that case gets its own dedupe slot rather than being dropped.
_DEGRADED_FORMAT_WARNED: set[Optional[str]] = set()


def _record_serving_format(data: YTDLVideoMetadata) -> None:
    """Record the shape of the format a song will play from.

    yt-dlp strips per-format client attribution (`__yt_dlp_client`) before formats
    leave the extractor, so *which* client served a song isn't observable — and the
    format shape is the sharper signal anyway. Healthy is audio-only (vcodec ==
    "none", bestaudio from android_vr); a muxed or HLS selection means android_vr
    degraded to muxed-only (yt-dlp#16150) or web_safari took over. Either way the
    primary path is degraded, which is worth one warning since playback continues
    and nothing else surfaces it.

    A missing vcodec (pre-upgrade cache entries) counts as healthy: never warn on a
    song that may be fine.
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
            "primary audio-only path (android_vr) is degraded and the player is "
            "on the fallback ladder"
        )


def _stream_cache_key(webpage_url: str) -> str:
    return f"ytdl:stream:{webpage_url}"


def _stream_url_ttl(stream_url: str) -> Optional[int]:
    """How long a stream URL may be cached, or None when it isn't worth caching.

    `expire` advertises a 6-hour window, but YouTube revokes URLs long before it — a
    DRM-restricted track was observed 403ing within the hour while `expire` still
    claimed five hours left, and trusting it meant replaying that revoked URL for the
    whole TTL. So _STREAM_URL_MAX_TTL is what bounds this in practice; `expire` only
    shortens it further, near the end of a URL's life.

    `ip` sits inside `sparams` (HMAC-signed), so URLs are also bound to the IP that
    extracted them and can never be reused from another host.

    `expire` is a query param on https formats but a path segment
    (`/expire/<epoch>/`) on the HLS manifests the degraded web_safari rung serves.
    Both forms are read; missing either leaves that rung uncached, silently
    re-extracting 3-5s on every play.
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


async def _stream_url_playable(stream_url: str) -> bool:
    """True when YouTube will actually serve this stream URL to ffmpeg right now.

    ffmpeg reports a revoked URL by 403ing and exiting, which discord.py cannot tell
    from a song that simply ended — so a dead URL plays as silence with no error
    anywhere. This probe is what makes that visible while it can still be fixed.

    It MUST open the request exactly as ffmpeg does: a plain GET, no Range header. A
    revoked URL still answers 206 to a *ranged* GET while refusing the open-ended
    one, so probing with a Range header (or HEAD, which googlevideo rejects) reports
    a dead URL as healthy. The body is never read, so this costs only the status line.
    """
    if not stream_url:
        return False
    try:
        timeout = aiohttp.ClientTimeout(total=_STREAM_PROBE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(stream_url) as response:
                return response.status < 400
    except Exception as e:
        # A probe that never completed is evidence about the network, not the URL.
        # Assume playable and let ffmpeg judge — a probe failure must never be the
        # reason a song refuses to play.
        log.warning(f"stream URL probe failed, assuming playable: {e}")
        return True


async def _cache_stream(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> bool:
    """Persist a stream URL already probed and found playable. True when an entry was
    written, False when the URL isn't worth caching (no usable expiry)."""
    # Absent keys are dropped, not written as None: readers all use .get() so the two
    # look alike downstream, but `{"title": None}` would contradict YTDLVideoInfo,
    # which types title as str and treats absent fields as *missing*. This keeps what
    # comes back out of cache_get() conformant to the type it is read as.
    stripped = {k: data[k] for k in _STREAM_CACHE_FIELDS if data.get(k) is not None}
    ttl = _stream_url_ttl(data.get("url", ""))
    if ttl:
        await cache_set(redis, cache_key, stripped, ttl)
        return True
    return False


async def _probe_and_cache(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> bool:
    """Success-path post-processing for a full stream extraction: record the serving
    format, probe the URL, cache it when playable. True when an entry was written.

    Shared by prefetch_stream and yt_source's unified extraction
    so both write identical entries. Only a proven-
    playable URL earns one — caching a revoked URL hands yt_stream a dead one."""
    _record_serving_format(data)
    if _stream_url_ttl(data.get("url", "")) is None:
        # Uncacheable (no usable expiry — e.g. SoundCloud): probing would spend a
        # network round only for _cache_stream to decline the write anyway.
        return False
    if await _stream_url_playable(data.get("url", "")):
        return await _cache_stream(redis, cache_key, data)
    return False


async def invalidate_stream_cache(
    redis: Optional[aioredis.Redis], webpage_url: str
) -> None:
    """Drop a song's cached stream URL so the next play re-extracts a fresh one."""
    await cache_del(redis, _stream_cache_key(webpage_url))


@dataclass
class QueueObject:
    """Song metadata in a queue before its processed by YTDL"""

    webpage_url: str
    title: str
    requester: Union[discord.User, discord.Member]
    ts: Optional[int] = None
    user_input: Optional[str] = None
    duration: Optional[int] = None  # seconds, from yt-dlp at enqueue time
    uploader: Optional[str] = None  # YouTube channel name
    thumbnail: Optional[str] = None
    # False only for the crash-recovered "current song" that
    # GuildQueue.restore_crashed() re-queues (from MusicPlayer._restore_state): it
    # was never RPUSHed to the Redis queue list (it lives in current_song_url
    # state), so the loop must skip its GuildQueue.redis_pop_for(). Read through
    # guild_queue.is_persisted(), never getattr.
    persisted: bool = True
    # ── -playnow interjection flags ──
    # Queued via -playnow. A later -playnow REPLACES a playing interjection (no
    # resume entry is built for it) instead of stacking.
    interjected: bool = False
    # The rebuilt tail of an interrupted song (ts = interrupt position). Drives
    # notice wording and suppresses yt_stream's "Starting song at Xs" — the loop
    # announces "Resuming…" when the entry actually starts.
    is_resume: bool = False
    # The interrupted song was paused at interjection time: the loop re-pauses
    # immediately after vc.play() so it returns parked.
    start_paused: bool = False


def _enrich_queueobject(qo: QueueObject, data: YTDLVideoMetadata) -> None:
    """Back-fill QueueObject fields that couldn't be populated at enqueue time.

    yt_source()'s unified extraction returns complete metadata, but other paths
    produce sparse QueueObjects: yt_playlist()'s flat entries carry no
    duration/uploader/thumbnail, and pre-unified ytdl:source cache entries may hold
    None until their TTL lapses. prefetch_stream() has the full data and writes it
    back onto the same instance so queue_embed() sees the enriched values.
    """
    fetched_duration = data.get("duration")
    if qo.duration is None and fetched_duration is not None:
        qo.duration = int(fetched_duration)
    if qo.uploader is None:
        qo.uploader = data.get("uploader")
    if qo.thumbnail is None:
        qo.thumbnail = data.get("thumbnail")


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
    ) -> None:
        super().__init__(
            url, executable="ffmpeg", before_options=before_options, options=options
        )

        self.requester = requester
        self.channel = channel
        # Seconds skipped via FFmpeg -ss; audio position = start_offset + elapsed.
        self.start_offset: int = start_offset
        # -playnow flags carried through from the QueueObject (see its fields).
        self.interjected: bool = interjected
        self.is_resume: bool = is_resume
        self.start_paused: bool = start_paused

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
        # Same clock rendering as the progress bar and every other printed
        # duration. str(timedelta(...)) spells 3m30s "0:03:30", which left the
        # recovered embed and presence card disagreeing with the bar's "3:30".
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
        """False when ffmpeg exited without delivering a frame — the stream never
        opened (typically a 403 on a revoked URL). discord.py hands that to `after`
        exactly like a finished song, so the frame count is the only thing telling a
        song that played from one that silently never started."""
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
        delivered, frozen during any pause since elapsed_secs is. The single source
        of truth for every position surface — progress bar, Activity presence, pause
        confirmation — so a song started via ?t= or resumed by crash recovery can't
        report different positions in different places. (The loop's crash-recovery
        math mirrors this by backdating play_start_epoch by start_offset.)"""
        return self.start_offset + self.elapsed_secs

    @classmethod
    @_tracer.start_as_current_span("ytdl.prefetch_stream")
    async def prefetch_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis] = None,
    ) -> None:
        """Eagerly populate the stream URL cache for a queued song.

        Spawned at enqueue time so yt_stream() is a cache hit by the time the song
        plays. No-op with no redis or an already-cached URL. Errors are logged and
        swallowed — yt_stream() recovers by extracting fresh at play time.
        """
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
            # Single-video cast: _YTDL_STREAM_OPTS on a watch URL never yields a
            # search/playlist wrapper, so the result is always a lone video here.
            data = cast(
                Optional[YTDLVideoInfo],
                await _run_extract(
                    ExtractRequest(url=qo.webpage_url, opts=_YTDL_STREAM_OPTS)
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
    ) -> YTDLVideoInfo:
        """Resolve a song to stream data whose URL YouTube will actually serve.

        Every URL is probed before it reaches ffmpeg, because a revoked one fails in
        the worst way: ffmpeg 403s and exits, discord.py reports a completed song,
        and the player advances in silence with nothing logged — and because the URL
        was cached, every later -play failed identically for the life of the entry.

        A revoked URL is dropped from the cache and re-extracted once; once is
        enough, since re-extraction reliably produced a playable URL.
        """
        span = trace.get_current_span()
        cache_key = _stream_cache_key(qo.webpage_url)

        data: Optional[YTDLVideoInfo] = await cache_get(redis, cache_key)
        span.set_attribute("ytdl.cache_hit", data is not None)

        for attempt in range(2):
            extracted_fresh = False
            if data is None:
                # Single-video cast, as in prefetch_stream.
                data = cast(
                    Optional[YTDLVideoInfo],
                    await _run_extract(
                        ExtractRequest(url=qo.webpage_url, opts=_YTDL_STREAM_OPTS)
                    ),
                )
                span.set_attribute("ytdl.extracted_fresh", True)
                if data is None:
                    raise RuntimeError("Could not extract stream data")
                extracted_fresh = True

            if await _stream_url_playable(data.get("url", "")):
                _record_serving_format(data)
                if extracted_fresh:
                    await _cache_stream(redis, cache_key, data)
                return data

            span.set_attribute("ytdl.stream_url_revoked", True)
            if not extracted_fresh:
                # Only a cached URL has an entry to drop — a fresh one is cached
                # exclusively on probe success, above.
                log.warning(
                    f"YouTube revoked the cached stream URL for {qo.webpage_url} "
                    "— dropping it from the cache and re-extracting"
                )
                await cache_del(redis, cache_key)
            elif attempt == 0:
                log.warning(
                    f"freshly extracted stream URL for {qo.webpage_url} probed "
                    "dead — re-extracting once"
                )
            else:
                log.warning(
                    f"freshly extracted stream URL for {qo.webpage_url} probed "
                    "dead again — giving up"
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
    ) -> "YTDL":
        """Resolve a queued song to a playable YTDL source, using the Redis
        stream-URL cache if present and extracting fresh via yt-dlp otherwise."""
        trace.get_current_span().set_attribute("ytdl.url", qo.webpage_url)

        data = await cls._resolve_playable_stream(qo, redis)

        ffmpeg_opts = cls.FFMPEG_OPTS.copy()
        if qo.ts is not None:
            ffmpeg_opts["options"] += f" -ss {qo.ts}"
            # Resume entries skip this construction-time notice: prefetch builds
            # them while the interjected song still plays, so it would fire at the
            # wrong moment — the loop announces "Resuming…" at actual start.
            if not qo.is_resume:
                await channel.send(
                    embed=notice_embed(
                        f"Starting song at {qo.ts} seconds", discord.Color.blue()
                    )
                )
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
        )

    @classmethod
    @_tracer.start_as_current_span("ytdl.yt_source")
    async def yt_source(
        cls,
        requester: Union[discord.User, discord.Member],
        search: str,
        *,
        download: bool = False,
        ts: Optional[int] = None,
        redis: Optional[aioredis.Redis] = None,
    ) -> QueueObject:
        """Resolve a search term or URL to a QueueObject via yt-dlp, using the
        Redis source cache if present."""
        trace.get_current_span().set_attribute("ytdl.search", search)
        # Normalised so "Destiny" and "destiny " both hit. ts is excluded — a
        # per-request playback offset, not part of the video identity.
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
                    user_input=search,
                    duration=cached.get("duration"),
                    uploader=cached.get("uploader"),
                    thumbnail=cached.get("thumbnail"),
                )

        trace.get_current_span().set_attribute("ytdl.source_cache_hit", False)

        # Unified single extraction: one stream-opts
        # call yields identity AND a playable stream URL, populating both the
        # ytdl:source and ytdl:stream caches from one network round.
        # process=True is hardcoded: an unprocessed extract_info does NO format
        # selection, so data["url"] would be absent and the stream-cache write below
        # would silently never happen for direct-URL plays. The page + player fetch
        # is paid either way; processing costs only format-selection CPU (~tens of
        # ms) and eliminates prefetch_stream's second extraction.
        try:
            data = await _run_extract(
                ExtractRequest(
                    url=search, opts=_YTDL_STREAM_SEARCH_OPTS, download=download
                )
            )
        except ExtractionError as e:
            # parse_url whitelists no domains — any dotted host lands here for yt-dlp
            # to accept or reject. An unrecognised site is rejected with
            # UnsupportedError, flattened to ExtractionError.unsupported in the worker
            # (_classify_ytdlp_error) since the type can't cross the boundary. Surface
            # something actionable; any other ExtractionError is a real failure and is
            # re-raised for _command_error to render via user_message.
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
            # callers cannot tell
            # "no such video" from "extractor broken" from "network down" — all three
            # render the identical embed and nothing can retry selectively.
            raise Exception("Could not find song")

        # A wrapper carries the video in `entries`; a lone-video result already is the
        # entry. Separate from `data` because a leaf (YTDLEntry) is not assignable back
        # to the result type, and "raw result" vs "chosen entry" are two things.
        selected: YTDLEntry = data
        if "entries" in data:
            # TODO: Validate search results have a usable audio format before accepting.
            # An entry wins purely by being the first non-playlist result — nothing
            # checks for an https audio URL at a usable bitrate, so a format-less or
            # low-quality entry is accepted here and only blows up at stream time,
            # looking unrelated.
            for entry in data["entries"]:
                if entry and entry.get("_type", None) != "playlist":
                    selected = entry
                    break
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
            # Warm the stream cache from the same extraction, which is what makes
            # queue_put's prefetch_stream a cache-hit no-op instead of a second
            # extraction. Awaited, not spawned, so the write lands before
            # prefetch_stream's cache_get can race it. A failed probe never fails
            # yt_source: the song enqueues on identity alone and
            # _resolve_playable_stream re-extracts at dequeue.
            stream_cached = await _probe_and_cache(
                redis, _stream_cache_key(webpage_url), video_data
            )
            trace.get_current_span().set_attribute("ytdl.stream_cached", stream_cached)

        return QueueObject(
            webpage_url,
            title,
            requester,
            ts=ts,
            user_input=search,
            duration=duration,
            uploader=uploader,
            thumbnail=thumbnail,
        )

    @staticmethod
    @_tracer.start_as_current_span("ytdl.yt_playlist")
    async def yt_playlist(
        url: str,
        requester: Union[discord.User, discord.Member],
    ) -> list[QueueObject]:
        """Fetch flat entry metadata for every video in a YouTube playlist."""
        trace.get_current_span().set_attribute("ytdl.url", url)
        data = await _run_extract(ExtractRequest(url=url, opts=_YTDL_PLAYLIST_OPTS))
        if data is None:
            raise Exception(f"Could not fetch YouTube playlist: {url}")
        # Optional in the element type, not re-annotated on the loop target: yt-dlp
        # emits a null entry for a deleted/private video, which is exactly what the
        # guard below skips — declaring it non-optional excluded that case.
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
            qobjs.append(QueueObject(video_url, title, requester))
        return qobjs
