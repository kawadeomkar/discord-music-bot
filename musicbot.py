from discord.ext import commands
#from dotenv import load_dotenv

# music players
from youtube import YTDL

import discord
import os
import youtube_dl


intents = discord.Intents().all()
client = discord.Client(intents=intents)
bot = commands.Bot(
                command_prefix='-',
                intents=discord.Intents().all(), # TODO: narrow down
                description='omkars shitty music bot lol')

@bot.command(name='p', pass_context=True, help='play a youtube song')
async def play(ctx: commands.Context, url):
    if not ctx.message.author.voice:
        await ctx.send('Must be connected to a voice channel')
        return

    channel, guild = ctx.message.author.voice.channel, ctx.message.guild
    voice_client = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    print(voice_client)
    print(channel)

    if not voice_client:
        await channel.connect()
    if not ctx.author.voice.channel == ctx.voice_client.channel:
        await ctx.voice_client.move_to(channel)
    # only support youtube link for now
    async with ctx.typing():
        try:
            source = await YTDL.yt_url(url, ctx, loop=bot.loop)
        except Exception as e:
            await ctx.send(f"Exception caught: {e}")


@bot.command(name='stop', help='stop current song')
async def stop(ctx: commands.Context):
    if ctx.message.guild.voice_client.is_playing():
        ctx.message.guild.voice_client.stop()


@bot.command(name='join', invoke_without_subcommand=True)
async def join(ctx: commands.Context):
    if not ctx.message.author.voice:
        await ctx.send("Must be connected to a voice channel")
        return
    else:
        await ctx.author.voice.channel.connect()


if __name__ == '__main__':
    bot.run(os.getenv("DISCORD_TOKEN"))
