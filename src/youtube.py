import asyncio
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import parse_qs, urlparse

import discord
import yt_dlp as youtube_dl

import redis.asyncio as aioredis

from src.redis_client import cache_get, cache_set
from src.spotify import Spotify
from src.util import get_logger

log = get_logger(__name__)

_YTDLP_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ytdlp")


def _ytdlp_extract(url: str, opts: Any, download: bool, process: bool) -> Any:
    """Dedicated thread-pool worker for yt-dlp extraction. Top-level so it's named in tracebacks."""
    return youtube_dl.YoutubeDL(opts).extract_info(
        url, download=download, process=process
    )


# TODO: PO token may be required eventually
PO_TOKEN = ""
# TODO: postprocessing ffmpeg, audio format, etc.

YTDL_OPTS = {
    "format": "bestaudio/best",
    "extractaudio": True,
    "verbose": True,
    "outtmpl": "%(extractor)s-%(id)s-%(title)s.%(ext)s",
    # "restrictfilenames": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "logtostderr": True,
    # "quiet": True,
    # "no_warnings": True,
    "default_search": "auto",
    "source_address": "0.0.0.0",
    "extractor_args": {
        "youtube": {
            "player_client": ["default", "-tv_simply"],
            # TODO: at the moment, we aren't being blocked by PO token requirements with `tv_simply`
            # This is where the PO token is passed: 'client_name+token_value'
            #'po_token': f'mweb.player+{PO_TOKEN}'
            # This specifies which player client to use (mweb is often reliable)
            #'po_token': f'mweb.gvs+{PO_TOKEN}',
        }
    },
    # Forces ffmpeg to attempt reconnection if the peer drops the connection mid-stream
    "external_downloader": "ffmpeg",
    "external_downloader_args": {
        "ffmpeg_i": [
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
        ]
    },
    # Buffers more data to handle transient network hiccups
    "socket_timeout": 30,
    "retries": 10,
    # Helps bypass server-side termination due to old cache/tokens
    "rm_cachedir": True,
}

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
    }
)


def _stream_url_ttl(stream_url: str) -> Optional[int]:
    """Returns seconds until stream URL expiry minus 30-min safety margin, or None if too short.

    YouTube CDN URLs carry a 6-hour expiry window (empirically confirmed). The `ip` parameter
    is included in `sparams` (HMAC-signed) so URLs are cryptographically bound to the IP that
    extracted them — they cannot be reused from a different host. With a 300s margin, the minimum
    URL lifetime on a cache hit is only ~5 minutes, which is shorter than many songs. 1800s (30
    min) ensures cache hits remain valid for songs up to ~30 minutes.
    """
    try:
        expire = int(parse_qs(urlparse(stream_url).query).get("expire", [0])[0])
        ttl = expire - int(time.time()) - 1800
        return ttl if ttl > 60 else None
    except (ValueError, IndexError):
        return None


@dataclass
class QueueObject:
    """Song metadata in a queue before its processed by YTDL"""

    webpage_url: str
    title: str
    requester: Union[discord.User, discord.Member]
    ts: Optional[int] = None


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
        data: dict,
        requester=None,
        before_options: Optional[str] = None,
        options: Optional[str] = None,
    ):
        super().__init__(
            url, executable="ffmpeg", before_options=before_options, options=options
        )

        self.requester = requester
        self.channel = channel

        self.data = data
        self.uploader = data.get("uploader")
        self.uploader_url = data.get("uploader_url")
        self.date = data.get("upload_date") or "00000000"
        self.upload_date = self.date[6:8] + "." + self.date[4:6] + "." + self.date[0:4]
        self.title = data.get("title")
        self.thumbnail = data.get("thumbnail")
        self.description = data.get("description")
        self.duration = str(datetime.timedelta(seconds=int(data.get("duration", "0"))))
        self.tags = data.get("tags")
        self.webpage_url = data.get("webpage_url")
        self.views = data.get("view_count")
        self.likes = data.get("like_count")
        self.dislikes = data.get("dislike_count")
        self.url = data.get("url")
        self.abr = data.get("abr")
        self.asr = data.get("asr")
        self.acodec = data.get("acodec")

    def __getitem__(self, item: str):
        return self.__getattribute__(item)

    @classmethod
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
        if redis is None:
            return
        cache_key = f"ytdl:stream:{qo.webpage_url}"
        if await cache_get(redis, cache_key) is not None:
            return
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                _YTDLP_POOL, _ytdlp_extract, qo.webpage_url, YTDL_OPTS, False, True
            )
        except Exception as e:
            log.warning(f"prefetch_stream failed for {qo.webpage_url}: {e}")
            return
        if data is not None:
            stripped = {k: data.get(k) for k in _STREAM_CACHE_FIELDS}
            ttl = _stream_url_ttl(data.get("url", ""))
            if ttl:
                await cache_set(redis, cache_key, stripped, ttl)

    @classmethod
    async def yt_stream(
        cls,
        qo: QueueObject,
        channel: discord.TextChannel,
        *,
        volume: float = 1.0,
        redis: Optional[aioredis.Redis] = None,
    ):
        loop = asyncio.get_running_loop()

        # ── Cache check ───────────────────────────────────────────────────────
        cache_key = f"ytdl:stream:{qo.webpage_url}"
        data = await cache_get(redis, cache_key)

        # ── Extract (only if cache miss) ──────────────────────────────────────
        if data is None:
            data = await loop.run_in_executor(
                _YTDLP_POOL, _ytdlp_extract, qo.webpage_url, YTDL_OPTS, False, True
            )
            if data is not None:
                stripped = {k: data.get(k) for k in _STREAM_CACHE_FIELDS}
                ttl = _stream_url_ttl(data.get("url", ""))
                if ttl:
                    await cache_set(redis, cache_key, stripped, ttl)

        if data is None:
            raise RuntimeError("Could not extract stream data")

        ffmpeg_opts = cls.FFMPEG_OPTS.copy()
        if qo.ts is not None:
            ffmpeg_opts["options"] += f" -ss {qo.ts}"
            await channel.send(f"Starting song at {qo.ts} seconds")
        if volume != 1.0:
            ffmpeg_opts["options"] += f" -filter:a volume={volume}"

        return cls(
            channel,
            data["url"],
            data=data,
            requester=qo.requester,
            before_options=ffmpeg_opts["before_options"],
            options=ffmpeg_opts["options"],
        )

    @classmethod
    async def yt_source(
        cls,
        requester: Union[discord.User, discord.Member],
        search: str,
        process: bool,
        *,
        download: bool = False,
        ts: Optional[int] = None,
    ) -> QueueObject:
        loop = asyncio.get_running_loop()

        # process=True to resolve all unresolved references (urls), need for ytsearch
        data = await loop.run_in_executor(
            _YTDLP_POOL, _ytdlp_extract, search, YTDL_OPTS, download, process
        )
        if data is None:
            # TODO: create custom YTDL exceptions
            raise Exception("Could not find song")

        if "entries" in data:  # TOOD: narrow down to https urls and right bitrate
            for entry in data["entries"]:
                if entry and entry.get("_type", None) != "playlist":
                    data = entry
                    break
        if download:
            # TODO: Handle downloading?
            # ytdl.prepare_filename(data)
            pass
        return QueueObject(data["webpage_url"], data["title"], requester, ts=ts)
