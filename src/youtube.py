import asyncio
import datetime
from dataclasses import dataclass
from typing import Union

import discord
import yt_dlp as youtube_dl
from discord.ext import commands

from src.spotify import Spotify


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


ytdl = youtube_dl.YoutubeDL(YTDL_OPTS)


@dataclass
class QueueObject:
    """Song metadata in a queue before its processed by YTDL"""

    webpage_url: str
    title: str
    requester: Union[discord.User, discord.Member]
    ts: int = None


class YTDL(discord.FFmpegOpusAudio):
    FFMPEG_OPTS = {
        "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        "options": "-vn",
    }

    def __init__(
        self,
        ctx: commands.Context,
        url: str,
        *,
        data: dict,
        requester=None,
        before_options: str = None,
        options: str = None,
    ):
        super().__init__(
            url, executable="ffmpeg", before_options=before_options, options=options
        )

        self.requester = requester
        self.channel = ctx.channel

        self.data = data
        self.uploader = data.get("uploader")
        self.uploader_url = data.get("uploader_url")
        self.date = data.get("upload_date")
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
    async def yt_stream(
        cls,
        qo: QueueObject,
        ctx: commands.Context,
        *,
        loop: asyncio.BaseEventLoop = None,
    ):
        loop = loop or asyncio.get_event_loop()
        requester = qo.requester or ctx.author
        data = await loop.run_in_executor(
            None,
            lambda: ytdl.extract_info(qo.webpage_url, download=False, process=True),
        )
        ffmpeg_opts = cls.FFMPEG_OPTS.copy()
        if qo.ts is not None:
            ffmpeg_opts["options"] += f" -ss {qo.ts}"
            await ctx.send(f"Starting song at {qo.ts} seconds")

        return cls(
            ctx,
            data["url"],
            data=data,
            requester=requester,
            before_options=ffmpeg_opts["before_options"],
            options=ffmpeg_opts["options"],
        )

    @classmethod
    async def yt_source(
        cls,
        ctx: commands.Context,
        search: str,
        process: bool,
        *,
        loop: asyncio.BaseEventLoop = None,
        download=False,
        ts: int = None,
    ) -> QueueObject:
        loop = loop or asyncio.get_event_loop()

        # process=True to resolve all unresolved references (urls), need for ytsearch
        data = await loop.run_in_executor(
            None, lambda: ytdl.extract_info(search, download=download, process=process)
        )
        # print(data)
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
        return QueueObject(data["webpage_url"], data["title"], ctx.author, ts=ts)
