from discord.ext import commands

import asyncio
import discord
import youtube_dl

YTDL_OPTS = {
    'format': 'bestaudio/best',
    'extractaudio': True,
    'audioformat': 'mp3',
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
    def __init__(self, ctx: commands.Context, source: discord.FFmpegPCMAudio, *, data: dict, volume: float = 0.5):
        super().__init__(source, volume)

        self.requester = ctx.author
        self.channel = ctx.channel

        self.data = data
        self.uploader = data.get('uploader')
        self.uploader_url = data.get('uploader_url')
        date = data.get('upload_date')
        self.upload_date = date[6:8] + '.' + date[4:6] + '.' + date[0:4]
        self.title = data.get('title')
        self.thumbnail = data.get('thumbnail')
        self.description = data.get('description')
        #self.duration = self.parse_duration(int(data.get('duration')))
        self.tags = data.get('tags')
        self.url = data.get('webpage_url')
        self.views = data.get('view_count')
        self.likes = data.get('like_count')
        self.dislikes = data.get('dislike_count')
        self.stream_url = data.get('url')



    # TODO: search, stream
    @classmethod
    async def yt_url(cls, url, ctx, *, loop: asyncio.BaseEventLoop=None, stream=False):
        loop = loop or asyncio.get_event_loop()
        process = False
        if "https" not in url:
            url = f"ytsearch:{url}"
            process = True
        # process=True to resolve all unresolved references (urls), need for ytsearch
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=False, process=process))

        if data is None:
            # TODO: create exceptions
            raise Exception("Could not find song")
        #print(data.keys())
        if 'entries' in data:
            for entry in data['entries']:
                if entry and 'formats' in entry and entry.get('_type', None) != 'playlist':
                    data = entry
                    print(entry.keys())
                    url = entry['formats'][0]['url']
                    break
        else:
            url = data['formats'][0]['url']
        print(data)
        ctx.message.guild.voice_client.play(discord.FFmpegPCMAudio(source=url, executable="ffmpeg"))
        await ctx.send(f'**Now playing:** {data["title"]}')
