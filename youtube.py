from discord.ext import commands

import asyncio
import datetime
import discord
import youtube_dl

# TODO: postprocessing ffmpeg, audio format, etc.
YTDL_OPTS = {
    'format': 'bestaudio/best',
    'extractaudio': True,
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0',
}

FFMPEG_OPTS = {
    'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5',
    'options': '-vn',
}

ytdl = youtube_dl.YoutubeDL(YTDL_OPTS)


class YTDL(discord.PCMVolumeTransformer):
    def __init__(self, ctx: commands.Context, source: discord.FFmpegPCMAudio, *, data: dict,
                 volume: float = 0.5, requester=None):
        print(source)
        super().__init__(source, volume)

        self.requester = requester or ctx.author
        self.channel = ctx.channel

        self.data = data
        self.uploader = data.get('uploader')
        self.uploader_url = data.get('uploader_url')
        self.date = data.get('upload_date')
        self.upload_date = self.date[6:8] + '.' + self.date[4:6] + '.' + self.date[0:4]
        self.title = data.get('title')
        self.thumbnail = data.get('thumbnail')
        self.description = data.get('description')
        self.duration = str(datetime.timedelta(seconds=int(data.get('duration', '0'))))
        self.tags = data.get('tags')
        self.webpage_url = data.get('webpage_url')
        self.views = data.get('view_count')
        self.likes = data.get('like_count')
        self.dislikes = data.get('dislike_count')
        self.stream_url = data.get('url')

    def __getitem__(self, item: str):
        return self.__getattribute__(item)

    @classmethod
    async def yt_stream(cls, data, ctx, *, loop=None):
        loop = loop or asyncio.get_event_loop()
        requester = data['requester']
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(data['webpage_url'],
                                                                          download=False,
                                                                          process=True))
        print("streaming: " + data['url'])
        print(ctx)
        return cls(ctx,
                   discord.FFmpegPCMAudio(data['url'], **FFMPEG_OPTS, executable="ffmpeg"),
                   data=data,
                   requester=requester)

    # TODO: handle downloading?
    @classmethod
    async def yt_url(cls, url, ctx, *, loop: asyncio.BaseEventLoop = None, download=False,
                     ytsearch: str = None):
        loop = loop or asyncio.get_event_loop()
        process = False

        if "https" not in url and ytsearch:
            url = " ".join(ytsearch.split(" ")[1:])
            url = f"ytsearch:{url}"
            process = True

        # process=True to resolve all unresolved references (urls), need for ytsearch
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url,
                                                                          download=download,
                                                                          process=process))
        if data is None:
            # TODO: create custom YTDL exceptions
            raise Exception("Could not find song")

        if 'entries' in data:  # TOOD: narrow down to https urls and right bitrate
            for entry in data['entries']:
                if entry and entry.get('_type', None) != 'playlist':
                    data = entry
                    break

        # ctx.message.guild.voice_client.play(
        # discord.FFmpegPCMAudio(source=url, **FFMPEG_OPTS, executable="ffmpeg"))

        return cls(ctx,
                   discord.FFmpegPCMAudio(ytdl.prepare_filename(data)),
                   data=data,
                   requester=ctx.author) \
            if download else {'webpage_url': data['webpage_url'],
                              'requester': ctx.author,
                              'title': data['title']}
