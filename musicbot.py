from discord.ext import commands
from musicplayer import MusicPlayer

# music players
from youtube import YTDL

import discord
import os


class MusicBot(commands.Cog):
    """
    class for music bot
    """
    __slots__ = ('bot', 'mps')

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.mps = {}

    def get_mp(self, ctx: commands.Context) -> MusicPlayer:
        if ctx.guild.id in self.mps:
            return self.mps[ctx.guild.id]
        self.mps[ctx.guild.id] = MusicPlayer(self.bot, ctx)
        return self.mps[ctx.guild.id]

    async def cleanup(self, guild: discord.Guild) -> None:
        if guild.voice_client:
            await guild.voice_client.disconnect()
        self.mps.pop(guild.id, None)

    async def cog_before_invoke(self, ctx):
        self.get_mp(ctx)

    async def validate_commands(self, ctx: commands.Context) -> None:
        if isinstance(ctx.author, discord.User):
            await ctx.send(f'You must be a member of this channel {ctx.author}')
            raise commands.CommandError(f'User {ctx.author} must be a member of this channel.')

        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send(f'You are not connected to a voice channel, you silly baka {ctx.author}')
            raise commands.CommandError(f'User {ctx.author} is not connected to a voice channel.')

        if not ctx.command == "play" and ctx.voice_client and ctx.voice_client.channel != ctx.author.voice.channel:
            await ctx.send(f'Bot is already being used in channel {ctx.voice_client.channel}')
            raise commands.CommandError('Bot is already in a voice channel.')

    @commands.command(name='play', aliases=['p', 'pl', 'pla', 'sing'], help='play a youtube song')
    @commands.before_invoke(validate_commands)
    async def play(self, ctx: commands.Context, url):
        if not ctx.guild.voice_client:
            await ctx.invoke(self.join)

        mp = self.get_mp(ctx)

        print(type(ctx.message.content))

        # only support youtube link for now
        async with ctx.typing():
            try:
                source = await YTDL.yt_url(url, ctx, loop=self.bot.loop, ytsearch=ctx.message.content)
                await mp.queue.put(source)
            except Exception as e:
                await ctx.send(f"Exception caught: {e}")

    @commands.command(name='stop', aliases=['s'], help='stop current song')
    @commands.before_invoke(validate_commands)
    async def stop(self, ctx: commands.Context):
        if ctx.message.guild.voice_client.is_playing():
            ctx.message.guild.voice_client.stop()

    @commands.command(name='pause', aliases=['po'], help='pause the current song')
    @commands.before_invoke(validate_commands)
    async def pause(self, ctx: commands.Context):
        if ctx.message.guild.voice_client.is_playing():
            await ctx.message.guild.voice_client.pause()

    @commands.command(name='join', aliases=['j'], help='join the channel')
    @commands.before_invoke(validate_commands)
    async def join(self, ctx: commands.Context):
        channel, guild = ctx.message.author.voice.channel, ctx.message.guild
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if not voice_client:
            await channel.connect()
        if not ctx.author.voice.channel == ctx.voice_client.channel:
            await ctx.voice_client.move_to(channel)


b = commands.Bot(
    command_prefix='-',
    intents=discord.Intents().all(),  # TODO: narrow down
    description='omkars bad music bot lol')


@b.event
async def on_ready():
    print(f'Bot :\n{b.user.name}\n{b.user.id}')


if __name__ == '__main__':
    b.add_cog(MusicBot(b))
    b.run(os.getenv("DISCORD_TOKEN"))
