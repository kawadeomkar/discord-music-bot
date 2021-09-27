from discord.ext import commands

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

    def get_mp(self, ctx: commands.Context):
        if ctx.guild.id in self.mps:
            return self.mps[ctx.guild.id]
        pass

    async def cleanup(self, guild: discord.Guild):
        if guild.voice_client:
            await guild.voice_client.disconnect()
        self.mps.pop(guild.id, None)

    async def validate_commands(self, ctx: commands.Context):
        if isinstance(ctx.author, discord.User):
            await ctx.send(f'You must be a member of this channel {ctx.author}')
            raise commands.CommandError(f'User {ctx.author} must be a member of this channel.')

        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send(f'You are not connected to a voice channel, you silly baka {ctx.author}')
            raise commands.CommandError(f'User {ctx.author} is not connected to a voice channel.')

        # if ctx.voice_client and ctx.voice_client.channel != ctx.author.voice.channel:
        #    await ctx.send(f'Bot is already being used in channel {ctx.voice_client.channel}')
        #    raise commands.CommandError('Bot is already in a voice channel.')

    @commands.command(name='p', aliases=['pl', 'pla', 'play', 'sing'], help='play a youtube song')
    @commands.before_invoke(validate_commands)
    async def play(self, ctx: commands.Context, url):

        channel, guild = ctx.message.author.voice.channel, ctx.message.guild
        voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        print(voice_client)
        print(channel)

        if not voice_client:
            await channel.connect()
        if not ctx.author.voice.channel == ctx.voice_client.channel:
            await ctx.voice_client.move_to(channel)
        # only support youtube link for now
        async with ctx.typing():
            try:
                source = await YTDL.yt_url(url, ctx, loop=self.bot.loop)
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
        if not ctx.message.author.voice:
            await ctx.send("Must be connected to a voice channel")
            return
        else:
            await ctx.author.voice.channel.connect()


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
