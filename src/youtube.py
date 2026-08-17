import copy
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional, TypedDict, Union, cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import discord
import yt_dlp as youtube_dl
from yt_dlp.utils import UnsupportedError, YoutubeDLError

import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.guild_state import ANALYTICS_ZERO, Analytics
from src.redis_client import cache_del, cache_get, cache_set
from src.telemetry import get_tracer
from src.util import fmt_duration, get_logger
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


class AudioCandidate(TypedDict, total=False):
    """One rung of a song's audio ladder, mined in the worker before `formats` is
    dropped. Field-for-field the shape a candidate promotes onto the info-dict it came
    from (_CANDIDATE_FIELDS), so a fallback URL that wins the probe is indistinguishable
    downstream from one yt-dlp selected itself.
    """

    url: str
    format_id: str
    acodec: str
    abr: float
    asr: int
    protocol: str
    vcodec: str
    audio_channels: int


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
    # Channel count of the served format. Read by _passthrough_codec, which must
    # refuse a >2-channel serve: `-c:a copy` also copies OpusHead, so a 5.1 stream
    # reaches Discord as 6-channel multistream and clients decode only the front pair.
    audio_channels: int
    # Format-shape fields, mirroring the trio in _STREAM_CACHE_FIELDS — what
    # _record_serving_format reads to tell a healthy audio-only serve from a
    # degraded muxed/HLS one.
    format_id: str
    protocol: str
    vcodec: str
    # The song's audio ladder, best first, candidate[0] being the format above.
    # Mined in the worker (_mine_audio_candidates) and walked at probe time, so a
    # revoked URL costs one sideways probe instead of the song.
    audio_candidates: list[AudioCandidate]


class YTDLVideoInfo(YTDLVideoMetadata, _YTDLVideoInfoRequired, total=False):
    """A single video's fields, once yt_source() has unwrapped "entries". Only
    url/webpage_url are guaranteed — any other field may be absent per
    extractor/client. Mirrors _STREAM_CACHE_FIELDS field-for-field; required keys go in
    _YTDLVideoInfoRequired, descriptive ones in YTDLVideoMetadata.
    """


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
# The one thing kept out of `formats` is the fallback audio ladder, mined into
# `audio_candidates` right before this drop (see _slim_info).
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

# How many audio URLs one extraction keeps, INCLUDING the format yt-dlp selected. A
# typical YouTube video exposes five audio formats (139 m4a-48k, 140 m4a-128k, and
# opus 249/250/251), so three keeps the selection plus its two best alternatives and
# drops the rungs nobody wants to land on. Only the alternatives are stored — the
# selected one is already at the top level (see _candidate_ladder) — so this costs
# two cached candidates, ~1.28 KB each, URL-dominated.
_STREAM_CANDIDATES = 3

# The per-format fields a candidate carries. Exactly the shape a winner promotes onto
# the info-dict, which is why the tuple is shared by both directions: everything
# downstream (_record_serving_format, YTDL's abr/asr/acodec) then describes the URL
# actually being played rather than the one yt-dlp originally selected.
_CANDIDATE_FIELDS = (
    "url",
    "format_id",
    "acodec",
    "abr",
    "asr",
    "protocol",
    "vcodec",
    "audio_channels",
)


def _candidate_shape(fmt: dict[str, Any]) -> AudioCandidate:
    """Slim one format down to the candidate fields it actually carries. Absent keys
    stay absent rather than becoming None — the same rule _cache_stream follows, so a
    promoted candidate never contradicts YTDLVideoInfo's non-optional types."""
    return cast(
        AudioCandidate, {k: fmt[k] for k in _CANDIDATE_FIELDS if fmt.get(k) is not None}
    )


def _mine_audio_candidates(node: dict[str, Any]) -> list[AudioCandidate]:
    """The audio URLs worth trying for one video, best first — mined in the worker
    because _slim_info drops `formats` and these URLs exist nowhere else afterwards.

    yt-dlp sorts `formats` worst→best, so the walk is reversed. Storyboards also carry
    `vcodec: none` and are excluded by the acodec test; `-drc` (dynamic-range-compressed)
    variants and foreign-language dubs are skipped so a fallback never quietly changes
    what the song sounds like. A muxed selection means the audio-only path is already
    degraded (see _record_serving_format), so that rung gets no ladder at all.

    See docs/ARCHITECTURE.md#stream-retry-ladder.
    """
    formats = node.get("formats")
    if not isinstance(formats, list) or not node.get("url"):
        # No format list (extract_flat entries) or no selected stream: nothing to mine.
        return []
    if node.get("vcodec") not in (None, "none"):
        # A muxed rung already means the audio-only path is degraded; walking sideways
        # across muxed formats is not a recovery worth having, so it gets no
        # alternatives — only the implicit rung 0 _candidate_ladder synthesizes.
        return []
    language = node.get("language")
    ladder: list[AudioCandidate] = []
    for fmt in reversed(formats):
        if not isinstance(fmt, dict) or not fmt.get("url"):
            continue
        if fmt.get("vcodec") not in (None, "none"):
            continue
        if fmt.get("acodec") in (None, "none"):
            continue
        if str(fmt.get("format_id") or "").endswith("-drc"):
            continue
        if language is not None and fmt.get("language") not in (None, language):
            continue
        ladder.append(_candidate_shape(fmt))
    # candidate[0] must BE the selected format, so "candidate 0" and "the URL the
    # probe already validated" stay the same thing. Normally the sort agrees; when it
    # doesn't, the selection wins and its duplicate is dropped from the tail.
    # The selected format is NOT stored: it already sits at the top level of the very
    # dict this list rides on, and _candidate_ladder synthesizes it back as rung 0.
    # Storing it duplicated a ~1.1 KB signed URL byte for byte (measured: 2,048 B per
    # entry, enough to push the entry past jemalloc's 8192 size class), and made
    # "candidate 0 is the selected format" an invariant to maintain by hand rather
    # than one that cannot be violated.
    selected_id = node.get("format_id")
    if ladder and ladder[0].get("format_id") == selected_id:
        ladder = ladder[1:]
    else:
        ladder = [c for c in ladder if c.get("format_id") != selected_id]
    return ladder[: _STREAM_CANDIDATES - 1]


def select_search_entry(
    entries: list[Optional[YTDLEntry]],
) -> Optional[YTDLEntry]:
    """The entry of a search result that would actually be played, or None if the
    result holds nothing playable.

    Prefers an entry carrying a stream URL, with the first non-playlist entry as the
    fallback so a shape this code does not recognise still plays rather than failing
    outright.

    Defensive, NOT a fix for yt_source's format-validation TODO: every search this bot
    builds is a bare `ytsearch:` (sources.py), which yt-dlp maps to
    _get_n_results(query, 1) — so `entries` holds exactly one item and there is nothing
    to choose between. It earns its keep only on multi-entry shapes (a `ytsearchN:`, or
    an extractor returning several), and asking for N is not free: process=True
    extracts every result, so ytsearch3 triples a 3-5s search.

    Public because `just ytdl-formats` answers "what would the bot play?" and has to
    make the same choice — two copies of this rule would drift, and the diagnostic
    would then answer for a song the bot does not pick.
    """
    playable = [
        entry for entry in entries if entry and entry.get("_type", None) != "playlist"
    ]
    if not playable:
        return None
    return next((entry for entry in playable if entry.get("url")), playable[0])


def _slim_info(info: Any) -> Optional[YTDLExtractResult]:
    """Make a yt-dlp result cheap and safe to ship back from the worker: sanitize_info()
    reduces the live objects a process=True info-dict carries (LazyList format ladders,
    a _YDLLogger, callables) to JSON primitives, without which every extraction fails on
    an opaque pickling error. The large collections it keeps but no caller reads are
    dropped here too, top level and per `entries` element.

    The audio ladder is mined BEFORE that drop and kept as `audio_candidates`: `formats`
    is one of the dropped collections, so this is the only point in the process where a
    fallback URL still exists.
    """
    info = _sanitize_info(info)
    if not isinstance(info, dict):
        # extract_info and sanitize_info only ever return a dict or None.
        return None
    entries = info.get("entries")
    nodes = [info]
    if isinstance(entries, list):
        nodes.extend(entry for entry in entries if isinstance(entry, dict))
    for node in nodes:
        candidates = _mine_audio_candidates(node)
        if candidates:
            node["audio_candidates"] = candidates
        for key in _UNUSED_INFO_COLLECTIONS:
            node.pop(key, None)
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
# android_vr is primary: it needs no PO token and serves audio-only formats. yt-dlp
# resolves `default` by JS-runtime availability, so shipping Deno (yt-dlp's `deno`
# extra, landing where yt-dlp looks first) plus yt-dlp-ejs turns ('android_vr',) into
# ('android_vr', 'web_safari') and makes the fallback's signature/n challenges solvable.
# web_safari serves *muxed* formats only (HLS 91-96, https 18) — the `/best` leg picks
# one and ffmpeg's -vn drops the video. The bgutil-pot-provider sidecar mints PO tokens
# via the bgutil-ytdlp-pot-provider plugin, whose pyproject pin must move in lockstep
# with that compose image tag.
#
# Degradation ladder — every rung lands on a previously-working configuration:
#   android_vr healthy → audio-only (e.g. 251/opus)
#   android_vr out     → web_safari muxed audio; WARNING via _record_serving_format
#   sidecar down       → plugin warns; web_safari works until POT enforcement lands
#   Deno broken        → yt-dlp reverts to the JS-less default (android_vr only)
# Revoked URLs are separate: _resolve_playable_stream()'s probe-and-re-extract.
# See docs/ARCHITECTURE.md#yt-dlp-client-strategy.
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
        # Cached because _passthrough_codec refuses a >2-channel serve, and a cache
        # hit must reach that decision with the same information a fresh extraction
        # had — absent means re-encode, so dropping it here would silently disable
        # passthrough rather than fail loudly.
        "audio_channels",
        # Format-shape fields — how _record_serving_format tells a healthy audio-only
        # serve from a degraded muxed/HLS one; kept so cache hits stay attributable.
        "format_id",
        "protocol",
        "vcodec",
        # The mined fallback ladder. Measured on a real 4-format extraction with
        # 1,151-char URLs: 1.28 KB/rung, taking the entry from 4.66 KB to 8.49 KB of
        # payload — but 5.22 KB to 10.34 KB RESIDENT, because 8.49 KB lands just past
        # jemalloc's 8192 size class. Redis memory is a step, not a size.
        # Cached so a revoked URL costs a sideways probe rather than a 3-5s
        # re-extraction; this key is TTL'd and evictable, so it carries no
        # golden-rule-12 obligation.
        "audio_candidates",
    }
)


# format_ids already warned about — once per format per process, so a real
# android_vr outage doesn't warn on every song. Optional[str] because an info-dict
# can omit format_id; that case gets its own dedupe slot rather than being dropped.
_DEGRADED_FORMAT_WARNED: set[Optional[str]] = set()


def _record_serving_format(data: YTDLVideoMetadata) -> None:
    """Record the shape of the format a song will play from. yt-dlp strips per-format
    client attribution (`__yt_dlp_client`) before formats leave the extractor, so the
    format shape is the signal instead: audio-only (vcodec "none") is healthy, while
    muxed or HLS means android_vr degraded (yt-dlp#16150) or web_safari took over — one
    warning, since playback continues and nothing else surfaces it. A missing vcodec
    (pre-upgrade cache entries) counts as healthy.
    """
    span = trace.get_current_span()
    format_id = data.get("format_id")
    span.set_attribute("ytdl.format_id", str(format_id))
    span.set_attribute("ytdl.protocol", str(data.get("protocol")))
    audio_only = data.get("vcodec") in (None, "none")
    span.set_attribute("ytdl.audio_only", audio_only)
    # Whether this serve is eligible for the remux path — attributed here rather
    # than only at construction, so a guild that stopped getting passthrough
    # (android_vr degraded to muxed AAC) is visible in the same place as why.
    span.set_attribute(
        "ytdl.opus_passthrough", _passthrough_codec(data, 1.0) is not None
    )
    if not audio_only and format_id not in _DEGRADED_FORMAT_WARNED:
        _DEGRADED_FORMAT_WARNED.add(format_id)
        log.warning(
            f"songs are being served a muxed A/V format "
            f"(format_id={format_id}, protocol={data.get('protocol')}) — the "
            "primary audio-only path (android_vr) is degraded and the player is "
            "on the fallback ladder"
        )


# The only formats this bot remuxes rather than re-encodes: YouTube's Opus itags,
# which are 20ms-framed stereo. The allowlist IS the frame-duration check — yt-dlp
# exposes no frame duration in the info-dict, so there is nothing to test at runtime
# and "opus" alone cannot be trusted to mean 20ms (see _passthrough_codec).
_PASSTHROUGH_FORMAT_IDS = frozenset({"249", "250", "251"})

# OpusHead + OpusTags, which RFC 7845 mandates at the head of every Ogg Opus stream.
# discord.py yields them from read() like audio, so YTDL discounts exactly two.
_OGG_HEADER_PACKETS = 2


def _passthrough_codec(data: YTDLVideoMetadata, volume: float) -> Optional[str]:
    """ "copy" when this song can be remuxed instead of re-encoded, else None (which
    leaves discord.py's `-c:a libopus` default).

    Discord speaks Opus and ~90% of what YouTube serves here IS Opus (format 251),
    so the encoder was spending a full lossy generation — 129 kbps opus decoded and
    re-encoded to 128 kbps — plus the CPU, to arrive at the same thing. discord.py's
    own route to the copy path is from_probe(), an ffprobe round trip per song; the
    extraction already told us the codec and it is in the stream cache, so this
    decides from data we have.

    Every clause is load-bearing, and three of them exist because `-c:a copy` also
    discards the `-ac 2 -ar 48000 -b:a 128k` discord.py always passes:

      volume        a volume filter has to touch samples, so it forces the encoder.
                    ffmpeg REFUSES `-c:a copy` alongside `-filter:a` outright (exit
                    234, zero bytes), so this is not a preference — yt_stream derives
                    the filter from this same call to keep the two from ever drifting.
      audio_channels  a 5.1 serve copied verbatim reaches Discord as 6-channel
                    multistream Opus (OpusHead mapping_family=1) and clients decode
                    only the front pair, silently losing centre-channel vocals.
                    yt-dlp ranks `channels` ABOVE `acodec` when sorting formats, so
                    `bestaudio` really does select itag 338 on videos that carry it.
                    Absent must mean re-encode: not every extractor populates it.
      format_id     packet duration. `read()` counts packets and every position
                    surface is frames x 20ms, so a 60ms-framed Opus source (legal,
                    and what SoundCloud's http_opus can be) plays at 3x speed and
                    reads a third of its true position. Nothing in the info-dict
                    reports frame duration, so the safe set is named explicitly.

    Measured before adopting: copy and libopus produce identical packet counts for
    YouTube's 20ms Opus (213.10s of packets for a 213s song, both), and copy with
    the two-pass seek lands within 0.02s of the target. Three accepted differences,
    none of which ffmpeg warns about, since no encoder is instantiated to ignore
    them: a mono source stays mono rather than being upmixed; the source's own
    bitrate is kept instead of being capped at 128k (measured ~30% more egress on a
    251); and discord.py's `-fec true -packet_loss 15` no longer embeds in-band
    forward error correction, so packet loss on the voice path produces dropouts
    libopus used to conceal. YouTube's files carry no FEC of their own, so that last
    one is a real regression for lossy listeners and the reason to keep this gate
    narrow rather than widen it.
    """
    if volume != 1.0:
        return None
    if not str(data.get("acodec") or "").startswith("opus"):
        return None
    if data.get("audio_channels") not in (1, 2):
        return None
    if str(data.get("format_id") or "") not in _PASSTHROUGH_FORMAT_IDS:
        return None
    return "copy"


def _stream_cache_key(webpage_url: str) -> str:
    return f"ytdl:stream:{webpage_url}"


def _stream_url_ttl(stream_url: str) -> Optional[int]:
    """How long a stream URL may be cached, or None when it isn't worth caching.
    `expire` advertises a 6-hour window but YouTube revokes long before it (observed: a
    403 within the hour with five hours still claimed), so _STREAM_URL_MAX_TTL is what
    bounds this in practice. `expire` is a query param on https formats but a path
    segment (`/expire/<epoch>/`) on the HLS manifests web_safari serves; missing either
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


async def _stream_url_playable(stream_url: str) -> bool:
    """True when YouTube will actually serve this stream URL to ffmpeg right now. A
    revoked URL makes ffmpeg 403 and exit, which discord.py cannot tell from a song that
    simply ended — silence, nothing logged. So probe with a plain GET, no Range: a
    revoked URL still answers 206 to a *ranged* GET and googlevideo rejects HEAD, so
    either would report a dead URL as healthy. The body is never read.

    For a song carrying a start offset this is deliberately STRICTER than ffmpeg's own
    open, which is a range request once `-ss` moved to the input side (see yt_stream).
    That is the safe direction — a URL that serves a full GET serves a ranged one — and
    it is why the probe was not "corrected" to match.
    """
    if not stream_url:
        return False
    try:
        timeout = aiohttp.ClientTimeout(total=_STREAM_PROBE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(stream_url) as response:
                return response.status < 400
    except Exception as e:
        # A probe that never completed is evidence about the network, not the URL:
        # assume playable and let ffmpeg judge.
        log.warning(f"stream URL probe failed, assuming playable: {e}")
        return True


async def _entry_ttl(redis: Optional[aioredis.Redis], cache_key: str) -> Optional[int]:
    """Seconds left on a cache entry, or None when that cannot be established. Its one
    caller uses it as a ceiling, and None means "no ceiling", so every uncertain answer
    — no Redis, a key with no expiry, a key already gone, a failed call — maps to None
    rather than to a number that would shorten a legitimate write."""
    if redis is None:
        return None
    try:
        ttl = await redis.ttl(cache_key)
    except Exception as e:
        log.warning(f"could not read TTL for {cache_key}: {e}")
        return None
    return ttl if isinstance(ttl, int) and ttl > 0 else None


async def _cache_stream(
    redis: Optional[aioredis.Redis],
    cache_key: str,
    data: YTDLVideoInfo,
    *,
    max_ttl: Optional[int] = None,
) -> bool:
    """Persist a stream URL already probed and found playable. True when an entry was
    written, False when the URL isn't worth caching (no usable expiry).

    `max_ttl` bounds the write for a caller re-caching URLs it did not just extract
    (a ladder promotion). _STREAM_URL_MAX_TTL is measured from now, so an unbounded
    rewrite would hand a 29-minute-old extraction another full 30 minutes and widen
    exactly the "probe says 200, ffmpeg 403s" window the probe exists to close.
    """
    # Absent keys are dropped, not written as None: `{"title": None}` would contradict
    # YTDLVideoInfo, which types title as str and treats absent fields as *missing*.
    stripped = {k: data[k] for k in _STREAM_CACHE_FIELDS if data.get(k) is not None}
    ttl = _stream_url_ttl(data.get("url", ""))
    if ttl is not None and max_ttl is not None:
        ttl = min(ttl, max_ttl)
    if ttl and ttl > 0:
        await cache_set(redis, cache_key, stripped, ttl)
        return True
    return False


async def _probe_and_cache(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> bool:
    """Success-path post-processing for a full stream extraction: record the serving
    format, probe the URL, cache it when playable. True when an entry was written.
    Shared by prefetch_stream and yt_source so both write identical entries; only a
    proven-playable URL earns one, since caching a revoked URL hands yt_stream a dead
    one."""
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
    # ── -playnow interjection flags ──
    # Queued via -playnow. Attribution only — interjections stack, so nothing reads
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
    # ── Stream-retry state, RUNTIME ONLY (never on the Redis wire) ──
    # Plays already spent on this song whose stream never opened, and the formats
    # that failed. Deliberately not persisted: a crash resetting the counter only
    # costs a restarted bot a few more attempts at a dead song, which is not worth
    # the QueueEntryField/parse-path surface. See MusicPlayer._retry_failed_stream.
    stream_attempts: int = 0
    failed_format_ids: frozenset[str] = frozenset()
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
    """Back-fill QueueObject fields that couldn't be populated at enqueue time:
    yt_playlist()'s flat entries carry no duration/uploader/thumbnail, and pre-unified
    ytdl:source cache entries may hold None until their TTL lapses. prefetch_stream()
    has the full data and writes it back onto the same instance for queue_embed().
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
        codec: Optional[str] = None,
        user_input: Optional[str] = None,
        persisted: bool = True,
        stream_attempts: int = 0,
        failed_format_ids: frozenset[str] = frozenset(),
        interjected: bool = False,
        is_resume: bool = False,
        start_paused: bool = False,
        analytics: Analytics = ANALYTICS_ZERO,
        query_source: str = "",
        played_at: float = 0.0,
        np_message_id: int = 0,
        np_channel_id: int = 0,
        np_dedicated: bool = False,
        np_host_ref: Optional[NpHostRef] = None,
    ) -> None:
        # codec=None keeps discord.py's default (`-c:a libopus`); "copy" selects the
        # remux path. See yt_stream, which decides which one a song gets.
        super().__init__(
            url,
            executable="ffmpeg",
            before_options=before_options,
            options=options,
            codec=codec,
        )

        self.requester = requester
        self.channel = channel
        # Seconds skipped via FFmpeg -ss; audio position = start_offset + elapsed.
        self.start_offset: int = start_offset
        # What the user typed, and whether a Redis queue entry backs this song —
        # both carried from the QueueObject because a playing song can become one
        # again (a neutralized prefetch, an interjection's resume tail, a stream
        # retry). Dropping user_input loses the only record of the collection link
        # -remove matches on. persisted is read by the playback loop to decide
        # whether settling this song's claim LPOPs: a crash-recovered head, which
        # was never on the Redis list, defaulted to True here retires an entry that
        # was never its own.
        self.user_input: Optional[str] = user_input
        self.persisted: bool = persisted
        # Retry state carried from the QueueObject: how many plays this song has
        # already spent failing to open a stream, and which formats failed.
        self.stream_attempts: int = stream_attempts
        self.failed_format_ids: frozenset[str] = failed_format_ids
        # -playnow flags carried through from the QueueObject (see its fields).
        self.interjected: bool = interjected
        self.is_resume: bool = is_resume
        self.start_paused: bool = start_paused
        # Ask-time analytics carried from the QueueObject so the history entry
        # this song produces records where it started, not when it played.
        self.analytics: Analytics = analytics
        self.query_source: str = query_source
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
        self._packets_read: int = 0

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def read(self) -> bytes:
        """Read the next packet, counting the AUDIO ones for elapsed_secs.

        The first two packets are never audio: RFC 7845 requires OpusHead and OpusTags
        at the start of every Ogg Opus stream, and discord.py's
        OggStream.iter_packets() yields them like any other packet. Counting them put a
        constant 40ms phantom into every position surface, and — worse — made
        produced_audio true for a stream that opened and encoded NOTHING (measured: an
        `-ss` past the end exits 0 having emitted exactly these two), so such a stream
        was reported as a finished song rather than reaching the retry ladder.
        """
        data = super().read()
        if data:
            self._packets_read += 1
            if self._packets_read > _OGG_HEADER_PACKETS:
                self._frames_read += 1
        return data

    @property
    def produced_audio(self) -> bool:
        """False when ffmpeg exited without delivering an audio frame — the stream never
        opened (typically a 403 on a revoked URL), or opened and produced only its
        container headers. discord.py hands either to `after` exactly like a finished
        song, so this count is the only thing that tells them apart."""
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
    ) -> None:
        """Eagerly populate the stream URL cache for a queued song, so yt_stream() is a
        cache hit by the time it plays. No-op with no redis or an already-cached URL;
        errors are logged and swallowed — yt_stream() recovers by extracting fresh.
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

    @staticmethod
    def _candidate_ladder(data: YTDLVideoInfo) -> list[AudioCandidate]:
        """The audio URLs to try for this song, best first.

        Rung 0 is always synthesized from the top level, because that IS the selected
        format — `audio_candidates` stores only the alternatives behind it. So "rung 0
        and the URL the probe already validated are the same thing" holds by
        construction rather than by maintenance, and entries cached before the ladder
        existed (or extractors exposing no format list) need no special case: they
        simply have no alternatives."""
        if not data.get("url"):
            return []
        alternatives = data.get("audio_candidates") or []
        return [_candidate_shape(cast(dict[str, Any], data)), *alternatives]

    @classmethod
    async def _probe_candidate_ladder(
        cls, data: YTDLVideoInfo, *, deprioritize: frozenset[str] = frozenset()
    ) -> Optional[int]:
        """Probe the ladder in order and promote the first URL YouTube will actually
        serve onto `data`, returning its index (None = every rung is dead).

        A fallback that wins is hoisted wholesale — url and format shape both — so
        everything downstream describes what is really playing. The candidates ahead of
        it are dropped: they are known dead for this extraction, and a caller that
        re-caches the entry must not make the next play re-probe them.

        `deprioritize` names formats that failed a previous play of this song (see
        MusicPlayer._retry_failed_stream). They are moved to the BACK of the walk, not
        removed: a single-format video would otherwise have nothing left to try, and
        the common failure — a URL revoked between the probe and ffmpeg's first read —
        is cured by a fresh URL for the same format. Sideways first, in place second.
        """
        ladder = cls._candidate_ladder(data)
        if deprioritize:
            tainted = [c for c in ladder if str(c.get("format_id")) in deprioritize]
            ladder = [
                c for c in ladder if str(c.get("format_id")) not in deprioritize
            ] + tainted
        span = trace.get_current_span()
        span.set_attribute("ytdl.candidates", len(ladder))
        for index, candidate in enumerate(ladder):
            if not await _stream_url_playable(candidate.get("url", "")):
                continue
            if candidate.get("format_id") != data.get("format_id"):
                log.warning(
                    f"audio format {data.get('format_id')} is not being served for "
                    f"{data.get('webpage_url')} — falling back to "
                    f"{candidate.get('format_id')}"
                )
                # cast to plain dict: the checker cannot see that AudioCandidate's
                # keys are a subset of YTDLVideoInfo's, though _CANDIDATE_FIELDS
                # makes them so by construction.
                # Replace, not merge. _candidate_shape drops keys the format does
                # not carry, so a rung with no `abr`/`asr` would otherwise inherit
                # the DEAD format's numbers and the Now Playing footer would
                # advertise a bitrate the stream is not using.
                promoted = cast(dict[str, Any], data)
                for stale in _CANDIDATE_FIELDS:
                    promoted.pop(stale, None)
                promoted.update(candidate)
            # Unconditional, NOT `if index:`. The walk is reordered when
            # `deprioritize` is non-empty, so a promotion can land at index 0 —
            # and then the entry would carry the winner's url/format_id over a
            # ladder still headed by the format that just failed. A caller that
            # re-caches that (every retry does: it invalidates first, so the
            # resolve is always a fresh extraction) makes the next play probe the
            # blacklisted rung first, throwing away what the retry learned and
            # warning about the wrong format for the rest of the TTL.
            # Rotated, not truncated: the rungs ahead of the winner move to the
            # BACK. Deleting them made two promotions leave a single ~50kbps rung
            # pinned for the rest of the TTL with no fallback left — worse than no
            # cache, which at least re-extracted and restored the top format. A
            # rung also dies for reasons that pass (a transient 503 from one CDN
            # host), and demotion lets it recover instead of being written off.
            # The winner is now rung 0 (it was just promoted to the top level), so
            # only what follows it is stored — rotated rather than truncated, since a
            # rung can die for reasons that pass (one CDN host 503ing) and deserves a
            # later try rather than being written off for the rest of the TTL.
            data["audio_candidates"] = ladder[index + 1 :] + ladder[:index]
            span.set_attribute("ytdl.candidate_index", index)
            return index
        return None

    @classmethod
    async def _resolve_playable_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis],
    ) -> YTDLVideoInfo:
        """Resolve a song to stream data whose URL YouTube will actually serve. Every URL
        is probed first, because a revoked one fails in the worst way: ffmpeg 403s and
        exits, discord.py reports a completed song, and the player advances in silence
        with nothing logged.

        Retry sideways before retrying in place: one extraction already carries the
        whole audio ladder, so a dead URL costs a ~100ms probe of the next format
        rather than a 3-5s re-extraction — which would only re-select the same format
        anyway, curing a stale URL but not a format YouTube has stopped serving. Only
        when every rung is dead is the entry dropped and re-extracted; once is enough.
        """
        span = trace.get_current_span()
        cache_key = _stream_cache_key(qo.webpage_url)

        data: Optional[YTDLVideoInfo] = await cache_get(redis, cache_key)
        span.set_attribute("ytdl.cache_hit", data is not None)
        # Read BEFORE any promotion rewrites the entry: it is the ceiling a re-cache
        # of not-freshly-extracted URLs must respect (see _cache_stream's max_ttl).
        remaining_ttl = await _entry_ttl(redis, cache_key) if data is not None else None

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

            winner = await cls._probe_candidate_ladder(
                data, deprioritize=qo.failed_format_ids
            )
            if winner is not None:
                _record_serving_format(data)
                # A promotion rewrites the entry too: left alone, a cached ladder
                # whose head is dead re-probes that same dead URL on every play
                # until the TTL lapses.
                if extracted_fresh or winner > 0:
                    await _cache_stream(
                        redis,
                        cache_key,
                        data,
                        max_ttl=None if extracted_fresh else remaining_ttl,
                    )
                return data

            span.set_attribute("ytdl.stream_url_revoked", True)
            if not extracted_fresh:
                # Only a cached URL has an entry to drop — a fresh one is cached
                # exclusively on probe success, above.
                log.warning(
                    f"YouTube revoked every cached audio URL for {qo.webpage_url} "
                    "— dropping the entry and re-extracting"
                )
                await cache_del(redis, cache_key)
            elif attempt == 0:
                log.warning(
                    f"every freshly extracted audio URL for {qo.webpage_url} probed "
                    "dead — re-extracting once"
                )
            else:
                log.warning(
                    f"every freshly extracted audio URL for {qo.webpage_url} probed "
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
        if qo.ts:
            # Two-pass seek, and BOTH halves are load-bearing — measured, because the
            # obvious single-sided forms are each wrong in one direction:
            #
            #   output-side alone   ffmpeg opens the stream at 0:00 and decodes its
            #                       way to the offset, so a crash-recovered song 40min
            #                       in pulls 40min of audio through the CDN before its
            #                       first frame. Accurate, and slow.
            #   input-side alone    a real HTTP range request, but it lands on the
            #                       nearest webm cluster BEFORE the target: measured
            #                       5-10s early across offsets, on live googlevideo and
            #                       on the same stream saved to disk, under both libopus
            #                       and copy. `-accurate_seek` is already the default
            #                       and does not fix it. Fast, and up to 10s wrong —
            #                       which position_secs (start_offset + elapsed) would
            #                       then overstate on every surface, and a -playnow
            #                       resume would replay.
            #
            # Input seek for the range request, then `-ss 0` on the output to drop the
            # pre-roll it lands in (that pre-roll carries negative timestamps, which is
            # what makes 0 the right threshold). Measured back to +/-0.02s at every
            # offset, with the deep range request intact.
            ffmpeg_opts["before_options"] += f" -ss {qo.ts}"
            ffmpeg_opts["options"] += " -ss 0"
            # No user notice here. This runs at CONSTRUCTION, which prefetch does
            # while the previous song is still playing, so announcing "Starting song
            # at Xs" from here fired at the wrong moment. MusicPlayer's start path
            # announces it — alongside "Resuming…", which was already moved there.
        # One decision, two consequences. ffmpeg refuses `-c:a copy` alongside any
        # filtergraph (exit 234, zero bytes, and the player would report it as
        # "YouTube refused the stream" and burn the whole retry budget on a purely
        # local bug), so the filter is gated on the codec rather than re-deriving
        # the same volume test in a second place that could drift from the first.
        codec = _passthrough_codec(data, volume)
        if codec is None and volume != 1.0:
            ffmpeg_opts["options"] += f" -filter:a volume={volume}"

        return cls(
            channel,
            data["url"],
            data=data,
            requester=qo.requester,
            start_offset=qo.ts or 0,
            before_options=ffmpeg_opts["before_options"],
            options=ffmpeg_opts["options"],
            codec=codec,
            user_input=qo.user_input,
            persisted=qo.persisted,
            stream_attempts=qo.stream_attempts,
            failed_format_ids=qo.failed_format_ids,
            interjected=qo.interjected,
            is_resume=qo.is_resume,
            start_paused=qo.start_paused,
            analytics=qo.analytics,
            query_source=qo.query_source,
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
        """Resolve a search term or URL to a QueueObject via yt-dlp, using the
        Redis source cache if present.

        query_source, analytics and user_input are REQUIRED so the QueueObject
        leaves here complete — a default would let a new call site forget them and
        write a plausible zero, permanently indistinguishable from a real value.

        user_input is what the user typed; None falls back to `search`, which is the
        same string only for a direct -play of one song. For an expanded collection
        `search` is a title this code generated, so the fallback would lose the album
        link -remove matches on."""
        origin = user_input if user_input is not None else search
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
                    user_input=origin,
                    duration=cached.get("duration"),
                    uploader=cached.get("uploader"),
                    thumbnail=cached.get("thumbnail"),
                    query_source=query_source,
                    analytics=analytics,
                )

        trace.get_current_span().set_attribute("ytdl.source_cache_hit", False)

        # Unified single extraction: one stream-opts call yields identity AND a playable
        # stream URL, filling both the ytdl:source and ytdl:stream caches from one
        # network round. process=True is hardcoded — an unprocessed extract_info does no
        # format selection, so data["url"] would be absent and the stream-cache write
        # below would silently never happen for direct-URL plays.
        try:
            data = await _run_extract(
                ExtractRequest(
                    url=search, opts=_YTDL_STREAM_SEARCH_OPTS, download=download
                )
            )
        except ExtractionError as e:
            # parse_url whitelists no domains — any dotted host lands here for yt-dlp
            # to accept or reject. An unrecognised site arrives as
            # ExtractionError.unsupported (flattened in the worker, since
            # UnsupportedError can't cross the boundary); any other one is re-raised.
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
            chosen = select_search_entry(data["entries"])
            if chosen is not None:
                selected = chosen
        # TODO: Validate that a search result carries a usable audio format.
        # A result yt-dlp could select no format for is still accepted here and only
        # fails at stream time, where the error looks unrelated to the search. The
        # warning below makes it attributable but does not prevent it; the fix is to
        # reject the entry and search again, which needs more than one candidate.
        if not selected.get("url"):
            log.warning(
                f"search result for {search} carries no stream URL "
                f"(title={selected.get('title')!r}) — playback will likely fail"
            )
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
            # Warm the stream cache from the same extraction, so queue_put's
            # prefetch_stream is a cache-hit no-op instead of a second extraction.
            # Awaited, not spawned, so the write lands before prefetch_stream's
            # cache_get can race it. A failed probe never fails yt_source.
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
        """Fetch flat entry metadata for every video in a YouTube playlist.

        query_source, analytics and user_input are REQUIRED (see yt_source).
        `analytics` is the head's — track positions are derived per kept track
        below. `user_input` is the playlist link the user pasted, carried onto every
        track so -remove can match it."""
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
            # Offset by tracks KEPT (len(qobjs)), never the enumerate index — the
            # skipped null entries above must not leave gaps in queue_position.
            # replace() so a field added to Analytics later is carried here.
            qobjs.append(
                QueueObject(
                    video_url,
                    title,
                    requester,
                    user_input=user_input,
                    query_source=query_source,
                    analytics=replace(
                        analytics,
                        queue_position=analytics.queue_position + len(qobjs),
                    ),
                )
            )
        return qobjs
