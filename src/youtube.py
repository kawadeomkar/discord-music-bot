import asyncio
import copy
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, TypedDict, Union, cast
from urllib.parse import parse_qs, urlparse

import aiohttp
import discord
import yt_dlp as youtube_dl

import redis.asyncio as aioredis
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.redis_client import cache_del, cache_get, cache_set
from src.telemetry import get_tracer
from src.util import fmt_duration, get_logger, notice_embed

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# ISSUE: Move yt-dlp extraction off threads and onto a ProcessPoolExecutor.
# yt-dlp extraction is only half I/O — JSON parsing, signature decryption and format
# selection are all GIL-bound Python, so the 8 threads here contend for the GIL rather
# than running in parallel. Every extraction therefore steals time from the event loop
# that is also serving voice heartbeats, and concurrent plays across guilds degrade one
# another. The extract worker (_ytdlp_extract) is already a top-level, picklable
# function, so the swap is mostly mechanical.
# Design: docs/ARCHITECTURE_PLAN.md §3.1.
_YTDLP_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ytdlp")


def _ytdlp_extract(url: str, opts: Any, download: bool, process: bool) -> Any:
    """Dedicated thread-pool worker for yt-dlp extraction. Top-level so it's named in tracebacks."""
    # YoutubeDL.__init__ keeps the params dict by reference and writes into it
    # (js_runtimes, http_headers, ...); the copy keeps the module-level opts
    # profiles immutable across the 8 pool threads.
    return youtube_dl.YoutubeDL(copy.copy(opts)).extract_info(
        url, download=download, process=process
    )


class _YtdlpLogger:
    """Routes yt-dlp's own diagnostics into our logger instead of dropping them.

    yt-dlp announces the things that *precede* an outage as warnings: formats skipped for a
    missing GVS PO token, "YouTube may have enabled the SABR-only streaming experiment",
    signature / n-challenge solving failures. Those were previously silenced (no_warnings),
    so the first sign of YouTube changing the rules would have been users reporting that
    songs no longer play. Per-video progress chatter still goes nowhere — only warnings and
    errors are worth a log line.
    """

    def debug(self, msg: str) -> None:
        # yt-dlp funnels both its [debug] lines and its ordinary per-video chatter
        # ("Downloading android vr player API JSON") here. Neither earns a line per song.
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
# ('android_vr', 'web_safari') with one. We ship Deno via yt-dlp's `deno` extra (the
# binary lands in the venv scripts dir, where yt-dlp looks first) and yt-dlp-ejs via
# the `default` extra, so web_safari's signature/n challenges are solvable. What that
# buys today (verified 2026-07): web_safari serves *muxed* formats only (HLS itags
# 91-96 and https itag 18) — the `/best` leg of our `bestaudio/best` selector picks
# one and ffmpeg's -vn drops the video. GVS Proof-of-Origin tokens — minted per
# visitor session by the bgutil-pot-provider sidecar (docker-compose.yml, port 4416)
# via the bgutil-ytdlp-pot-provider plugin — are not yet *enforced* for those muxed
# formats, but that enforcement is YouTube's documented trajectory (PO-Token-Guide:
# HLS exempt "currently"); the sidecar is what keeps this fallback alive when it
# flips. android_vr needs none of this — PO tokens are "Not required" for it — and
# stays first; fetch_pot=auto means the provider is only consulted when needed.
#
# Degradation ladder — every rung lands on a previously-working configuration:
#   android_vr healthy   → exactly the pre-fallback behavior (audio-only, e.g. 251/opus)
#   android_vr out       → web_safari serves muxed audio; WARNING via _record_serving_format
#   sidecar down         → plugin warns; web_safari keeps working until POT enforcement lands
#   Deno broken          → yt-dlp reverts to the JS-less default (android_vr only)
# Revoked URLs are handled separately by _resolve_playable_stream()'s probe-and-
# re-extract — which now has two clients to heal from. See docs/PO_TOKEN_SIDECAR_PLAN.md.
#
# `-tv_simply` is a no-op against today's defaults; kept as a guard in case it is added back.
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
    # rm_cachedir intentionally absent — yt-dlp's JS player cache is kept across
    # calls so the signature-decryption JS is only fetched when YouTube publishes
    # a new player version, not on every extraction.
}

# Used by yt_stream / prefetch_stream: resolves a webpage_url to a CDN stream URL.
# check_formats=False skips HEAD requests that probe format URL availability.
#
# Format ladder: audio-only when available (the healthy android_vr path); otherwise a
# *small* muxed format — ffmpeg's -vn keeps only its audio, so on the fallback rung
# picking plain `best` would stream 1080p video (~120MB/song) just to throw the
# picture away, while 360p muxed (itag 18 / HLS 93) carries the same mp4a audio for
# ~a tenth of that. Bare `best` stays as the final rung for videos with nothing ≤360p.
_YTDL_STREAM_OPTS = {
    **_YTDL_BASE_OPTS,
    "format": "bestaudio/best[height<=360]/best",
    "check_formats": False,
    "retries": 10,
}

# Used by yt_source: the unified single-extraction play path
# (docs/PERFORMANCE_PLAN.md §2.1). One stream-opts extraction returns the video's
# identity (webpage_url/title/duration/…) AND a selected, playable stream URL, so a
# single yt-dlp call populates both the ytdl:source and ytdl:stream caches — the
# previous source-opts search performed a full extraction anyway (including format
# selection) and then discarded the stream data, forcing prefetch_stream to hit
# YouTube a second time for every cold play. default_search is what the stream opts
# lack for resolving bare search queries; retries stays at the stream value (10)
# because this call now serves playback, not just metadata.
_YTDL_STREAM_SEARCH_OPTS = {
    **_YTDL_STREAM_OPTS,
    "default_search": "auto",
}

# Used by yt_playlist: fetches entry metadata for all videos in a playlist without
# individually extracting each video's stream URL. noplaylist=False overrides the
# base option so yt-dlp processes the full playlist rather than just the first video.
_YTDL_PLAYLIST_OPTS = {
    **_YTDL_BASE_OPTS,
    "noplaylist": False,
    "extract_flat": True,
}

# Legacy alias kept so any external callers that imported YTDL_OPTS still work.
YTDL_OPTS = _YTDL_STREAM_OPTS

# How long to cache a search-query → (webpage_url, title) resolution.
# Short enough to pick up YouTube ranking changes; long enough to meaningfully
# skip the 3-4s yt-dlp search on repeat plays of the same track.
_YT_SOURCE_TTL = 3600  # 1 hour

# Ceiling on how long a resolved stream URL may be cached. YouTube revokes these
# well before the `expire` they carry (see _stream_url_ttl), so this — not `expire`
# — is what keeps a dead URL from being replayed. Re-extracting costs a few seconds;
# serving a revoked URL costs the song.
_STREAM_URL_MAX_TTL = 1800  # 30 minutes

# Cap on the pre-playback URL probe. Generous enough not to trip on a slow CDN,
# short enough that it never adds a noticeable pause before a song starts.
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


class _YTDLVideoInfoRequired(TypedDict):
    """`url`/`webpage_url` are the only fields this codebase accesses via
    direct subscript (`data["url"]`) rather than `.get()` — yt-dlp always
    populates both once `data` is narrowed to a single video."""

    url: str
    webpage_url: str


class YTDLVideoMetadata(TypedDict, total=False):
    """The descriptive half of an info-dict: everything except the two
    identity fields. Split out because helpers like _enrich_queueobject() and
    _record_serving_format() read only these — typing them against the full
    YTDLVideoInfo would demand a `url`/`webpage_url` they never touch."""

    title: str
    uploader: str
    uploader_url: str
    upload_date: str
    thumbnail: str
    description: str
    # float, not int: yt-dlp's SoundCloud extractor emits
    # `'duration': float_or_none(scale=1000)` (its own fixtures show 942.762), and this
    # bot accepts SoundcloudSource. Every read below wraps this in int() — that is the
    # conversion, not a redundancy.
    duration: float
    tags: list[str]
    view_count: int
    like_count: int
    dislike_count: int
    abr: float
    asr: int
    acodec: str
    # Format-shape fields, mirroring the same trio in _STREAM_CACHE_FIELDS:
    # what _record_serving_format reads to tell a healthy audio-only serve
    # from a degraded muxed/HLS one.
    format_id: str
    protocol: str
    vcodec: str


class YTDLVideoInfo(YTDLVideoMetadata, _YTDLVideoInfoRequired, total=False):
    """The subset of yt-dlp's info-dict fields this codebase actually reads,
    once `data` has been narrowed to a single video (see yt_source()'s
    "entries" un-wrapping). Everything but `url`/`webpage_url` is optional:
    yt-dlp's own dict is not a stable, fully-populated contract — any field
    may be absent depending on extractor/client. Mirrors
    _STREAM_CACHE_FIELDS field-for-field.

    `total=False` is on an empty body deliberately. It changes nothing today,
    but without it a key added directly here later would silently be
    *required* — the opposite of every other field in this hierarchy, and a
    break that shows up as a type error far from the line that caused it.
    Required keys belong in _YTDLVideoInfoRequired; descriptive ones in
    YTDLVideoMetadata.
    """


class YTDLFlatPlaylistEntry(TypedDict, total=False):
    """One entry from a flat playlist listing (_YTDL_PLAYLIST_OPTS,
    extract_flat=True) — a much sparser shape than YTDLVideoInfo."""

    id: str
    title: str
    url: str


# format_ids already warned about by _record_serving_format — once per format per
# process, so a real android_vr outage doesn't emit a warning for every song.
# Optional[str], not str: an info-dict can omit format_id, and that case gets its
# own dedupe slot (one warning for all id-less degraded serves) rather than being
# silently dropped.
_DEGRADED_FORMAT_WARNED: set[Optional[str]] = set()


def _record_serving_format(data: YTDLVideoMetadata) -> None:
    """Record the shape of the format a song will play from.

    yt-dlp strips per-format client attribution (`__yt_dlp_client`) before formats
    leave the extractor, so *which client* served a song is not directly observable.
    The format shape is the sharper signal anyway: the healthy path is an audio-only
    format (vcodec == "none", bestaudio from android_vr); a muxed or HLS selection
    means either android_vr degraded to muxed-only (yt-dlp#16150's "ONLY -f=18" mode)
    or web_safari is serving as the fallback. Both mean the primary path is degraded —
    worth one warning, since playback itself continues and nothing else surfaces it.

    A missing vcodec (pre-upgrade cache entries) is treated as healthy: never warn on
    a song that may be fine.
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
    """Returns how long a stream URL may be cached, or None when it isn't worth caching.

    The `expire` query param advertises a 6-hour window, but YouTube revokes URLs long before
    it: a DRM-restricted track's URL was observed serving 403 within the hour while `expire`
    still claimed five hours left. Trusting `expire` meant one revoked URL was replayed for
    its whole TTL, so every -play of that song failed. The cap is therefore what bounds this
    in practice — `expire` only ever shortens it further, near the end of a URL's life.

    The `ip` parameter is inside `sparams` (HMAC-signed), so URLs are also bound to the IP that
    extracted them and can never be reused from another host.

    `expire` lives in the query string on https formats, but HLS manifest URLs — the muxed
    formats the degraded web_safari rung serves — carry it as a path segment
    (`/expire/<epoch>/`). Both forms are read; missing either would leave that rung uncached,
    silently re-extracting 3-5s on every play.
    """
    try:
        parsed = urlparse(stream_url)
        expire = int(parse_qs(parsed.query).get("expire", [0])[0])
        if not expire:
            match = re.search(r"/expire/(\d+)(?:/|$)", parsed.path)
            expire = int(match.group(1)) if match else 0
        ttl = min(expire - int(time.time()) - 1800, _STREAM_URL_MAX_TTL)
        return ttl if ttl > 60 else None
    except ValueError, IndexError:
        return None


async def _stream_url_playable(stream_url: str) -> bool:
    """True when YouTube will actually serve this stream URL to ffmpeg right now.

    ffmpeg reports a revoked URL by 403ing and exiting, which discord.py cannot distinguish
    from a song that simply ended — so a dead URL plays as silence with no error anywhere.
    Probing here is what makes that failure visible while we can still do something about it.

    The probe must open the request exactly the way ffmpeg does — a plain GET with no Range
    header — because that is the only question whose answer matches ffmpeg's. A revoked URL
    still answers 206 to a *ranged* GET while refusing the open-ended one, so probing with a
    Range header (or with HEAD, which googlevideo rejects outright) reports a dead URL as
    healthy. The body is never read: aiohttp holds it until asked, so the status line is all
    this costs.
    """
    if not stream_url:
        return False
    try:
        timeout = aiohttp.ClientTimeout(total=_STREAM_PROBE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(stream_url) as response:
                return response.status < 400
    except Exception as e:
        # A probe that never completed is evidence about the network, not about the
        # URL. Assume playable and let ffmpeg be the judge — a probe failure must
        # never be the reason a song refuses to play.
        log.warning(f"stream URL probe failed, assuming playable: {e}")
        return True


async def _cache_stream(
    redis: Optional[aioredis.Redis], cache_key: str, data: YTDLVideoInfo
) -> bool:
    """Persist a stream URL that has been probed and found playable.

    Returns True when an entry was written; False when the URL isn't worth caching
    (no usable expiry — see _stream_url_ttl)."""
    # Absent keys are dropped rather than written as None. Every reader goes
    # through .get(), so the two are indistinguishable downstream — but writing
    # `{"title": None}` would contradict YTDLVideoInfo, which types title as str
    # and documents absent fields as *missing*. This makes the value that comes
    # back out of cache_get() actually conform to the type it is read as.
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
    format, probe the stream URL, and cache it when playable.

    Shared by prefetch_stream and yt_source's unified extraction
    (docs/PERFORMANCE_PLAN.md §2.1) so both write identical cache entries. Only a URL
    that has been proven playable earns a cache entry — caching an already-revoked one
    would hand yt_stream a dead URL it then has to discard. Returns True when a cache
    entry was written."""
    _record_serving_format(data)
    if _stream_url_ttl(data.get("url", "")) is None:
        # Uncacheable URL (no usable expiry — e.g. SoundCloud): probing it would spend
        # an awaited network round only for _cache_stream to decline the write anyway.
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
    # False for the crash-recovered "current song" MusicPlayer._restore_state()
    # re-queues via GuildQueue.restore_crashed() — it was never RPUSHed to
    # Redis's queue list (it's tracked separately via current_song_url state),
    # so the playback loop must skip the matching GuildQueue.redis_pop_for() for
    # it. Read through the guild_queue.is_persisted() helper, never getattr.
    persisted: bool = True
    # ── -playnow interjection flags (docs/PLAYNOW_PROPOSAL.md) ──
    # True for a song queued via -playnow. A later -playnow REPLACES a playing
    # interjection (no resume entry is built for it) instead of stacking.
    interjected: bool = False
    # True for the rebuilt tail of an interrupted song (ts = interrupt
    # position). Drives display/notice wording and suppresses yt_stream's
    # construction-time "Starting song at Xs" notice — the loop announces
    # "Resuming…" when the entry actually starts instead.
    is_resume: bool = False
    # True when the interrupted song was paused at interjection time: the loop
    # re-pauses immediately after vc.play() so the song returns parked.
    start_paused: bool = False


def _enrich_queueobject(qo: QueueObject, data: YTDLVideoMetadata) -> None:
    """Back-fill QueueObject fields that couldn't be populated at enqueue time.

    yt_source()'s unified extraction returns complete metadata, but other
    enqueue paths still produce sparse QueueObjects: yt_playlist()'s flat
    entries carry no duration/uploader/thumbnail, and ytdl:source cache
    entries written by pre-unified code may hold None for those fields until
    their TTL lapses. prefetch_stream() has the complete data from full
    extraction — this helper writes it back onto the same QueueObject
    instance so queue_embed() sees the enriched values.
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
        # -playnow flags, carried through from the QueueObject (see its field
        # comments): interjected drives replace semantics, is_resume/start_paused
        # drive the loop's resume announcement and re-pause on start.
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
        # `or 0` rather than a dict default: yt-dlp sets "duration" to None (not
        # absent) for livestreams and some age-gated videos, so a plain
        # data.get("duration", 0) would hand int() a None and raise.
        self.duration_secs: int = int(data.get("duration") or 0)
        # Same clock rendering as the progress bar and every other duration the
        # bot prints. Was str(timedelta(...)), which spells 3m30s "0:03:30" and
        # left the recovered now-playing embed and the "Listening to" presence
        # card disagreeing with the bar's own "3:30" label.
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
        """False when ffmpeg exited without ever delivering a frame — the stream never
        opened (typically a 403 on a revoked URL). discord.py hands that to the `after`
        callback exactly like a song that finished, so the frame count is the only thing
        that distinguishes a song that played from one that silently never started."""
        return self._frames_read > 0

    @property
    def elapsed_secs(self) -> float:
        """Seconds of audio actually delivered to the player so far. Frozen during
        any pause — explicit (`-pause`) or involuntary (voice reconnect stall) —
        because AudioPlayer simply doesn't call read() during either."""
        return self._frames_read * (discord.opus.Encoder.FRAME_LENGTH / 1000.0)

    @property
    def position_secs(self) -> float:
        """True audio position: seconds skipped via FFmpeg -ss (start_offset)
        plus seconds actually delivered (elapsed_secs); frozen during any pause
        since elapsed_secs is. The single source of truth for every position
        surface — progress bar, Activity presence, pause confirmation — so a
        song started via ?t= or resumed mid-stream by crash recovery can't
        report different positions in different places. (The playback loop's
        crash-recovery math mirrors this by backdating play_start_epoch by
        start_offset.)"""
        return self.start_offset + self.elapsed_secs

    @classmethod
    @_tracer.start_as_current_span("ytdl.prefetch_stream")
    async def prefetch_stream(
        cls,
        qo: QueueObject,
        redis: Optional[aioredis.Redis] = None,
    ) -> None:
        """Eagerly populate the stream URL cache for a queued song.

        Spawned as a background task at enqueue time so yt_stream() is a cache
        hit by the time the song is ready to play. No-op when redis is None or
        the URL is already cached. Errors are logged and swallowed — yt_stream()
        recovers by extracting fresh at play time.
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
        loop = asyncio.get_running_loop()
        try:
            data: Optional[YTDLVideoInfo] = await loop.run_in_executor(
                _YTDLP_POOL,
                _ytdlp_extract,
                qo.webpage_url,
                _YTDL_STREAM_OPTS,
                False,
                True,
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

        Every URL is probed before it reaches ffmpeg, because a revoked one fails in the
        worst possible way: ffmpeg 403s and exits, discord.py reports that as a completed
        song, and the player advances in silence with nothing logged. Since the URL was
        cached, every later -play of that song replayed it and failed the same way, which
        is what pinned one song to a permanent failure for the life of its cache entry.

        A revoked URL is dropped from the cache and re-extracted once. Once is enough:
        re-extracting a video whose cached URL had died reliably produced a playable one.
        """
        span = trace.get_current_span()
        loop = asyncio.get_running_loop()
        cache_key = _stream_cache_key(qo.webpage_url)

        data: Optional[YTDLVideoInfo] = await cache_get(redis, cache_key)
        span.set_attribute("ytdl.cache_hit", data is not None)

        for attempt in range(2):
            extracted_fresh = False
            if data is None:
                data = await loop.run_in_executor(
                    _YTDLP_POOL,
                    _ytdlp_extract,
                    qo.webpage_url,
                    _YTDL_STREAM_OPTS,
                    False,
                    True,
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
                # Only a cached URL has a cache entry to drop — a fresh one is
                # cached exclusively on probe success, above.
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
            # Resume entries skip this construction-time notice: prefetch
            # constructs them while the interjected song is still playing, so
            # it would fire at the wrong moment — the playback loop announces
            # "Resuming…" when the entry actually starts instead.
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
        # Cache key: normalise search so "Destiny" and "destiny " both hit.
        # ts is intentionally excluded — it is a per-request playback offset,
        # not part of the video identity.
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
        loop = asyncio.get_running_loop()

        # Unified single extraction (docs/PERFORMANCE_PLAN.md §2.1): one stream-opts
        # call yields identity AND a playable stream URL, so both the ytdl:source and
        # ytdl:stream caches are populated from this single network round.
        # process=True is hardcoded — an unprocessed extract_info performs NO format
        # selection, so data["url"] would be absent and the stream-cache write below
        # would silently never happen for direct-URL plays (which used to arrive here
        # with process=False). For a single watch URL the page + player fetch is paid
        # either way; processing adds only format-selection CPU (~tens of ms), no
        # extra network, and it eliminates prefetch_stream's second extraction.
        data = await loop.run_in_executor(
            _YTDLP_POOL,
            _ytdlp_extract,
            search,
            _YTDL_STREAM_SEARCH_OPTS,
            download,
            True,
        )
        if data is None:
            # TODO: Replace the bare Exception on yt-dlp failure with typed errors.
            # Every failure mode raises the same untyped Exception("Could not find
            # song"), so callers cannot distinguish "no such video" from "extractor
            # broken" from "network down". All three render the identical generic error
            # embed to the user, and nothing upstream can retry selectively or degrade
            # differently per cause.
            # Pairs with the error-handling consolidation in docs/ARCHITECTURE_PLAN.md §3.3.
            raise Exception("Could not find song")

        if "entries" in data:
            # TODO: Validate search results have a usable audio format before accepting.
            # An entry wins purely by being the first non-playlist result — nothing
            # checks that it carries an https audio URL at a usable bitrate. A
            # format-less or low-quality entry is therefore selected here and only
            # blows up later, at stream time, where the failure looks unrelated.
            for entry in data["entries"]:
                if entry and entry.get("_type", None) != "playlist":
                    data = entry
                    break
        if download:
            # TODO: Implement or remove yt_source's dead download=True parameter.
            # The parameter is accepted by the signature but does nothing: the file is
            # never named (prepare_filename) or handed back to the caller, so anyone
            # passing download=True silently gets streaming behavior and no error.
            pass

        # `data` is genuinely Any before this point (video dict vs search/playlist wrapper —
        # see _ytdlp_extract's docstring); the narrowing above always leaves it single-video shaped.
        # cast(), not a bare annotation: this asserts something the checker cannot verify,
        # and `grep cast(` is how those assertions are audited in this codebase.
        video_data = cast(YTDLVideoInfo, data)

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
            # Warm the stream cache from the same extraction — this is what makes
            # queue_put's prefetch_stream a cache-hit no-op instead of a second
            # YouTube extraction. Awaited (not spawned) so the write has landed
            # before prefetch_stream's cache_get can race it. A failed probe never
            # fails yt_source: the song enqueues on identity alone and dequeue-time
            # _resolve_playable_stream re-extracts, exactly the pre-§2.1 behavior.
            stream_cached = await _probe_and_cache(
                redis, _stream_cache_key(webpage_url), data
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
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            _YTDLP_POOL,
            _ytdlp_extract,
            url,
            _YTDL_PLAYLIST_OPTS,
            False,
            True,
        )
        if data is None:
            raise Exception(f"Could not fetch YouTube playlist: {url}")
        # Optional in the element type, not re-annotated on the loop target: yt-dlp
        # emits a null entry for a deleted/private video, which is what the guard
        # below skips. Declaring it non-optional there excluded exactly the case the
        # next line handles.
        entries: list[Optional[YTDLFlatPlaylistEntry]] = data.get("entries") or []
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
