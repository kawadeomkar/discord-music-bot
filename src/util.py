import logging
import random
import sys
from typing import List

from discord.ext import commands


async def send_queue_phrases(ctx: commands.Context):
    if ctx.message.author.name == "pineapplecat":
        phrases = [
            "great choice king! :3",
            "my god you gigachad, impressive choice",
            "splendid choice pogdaddy",
            "turbo taste fam",
            "terrific taste turbo chad",
            "vibrations are retro daddy",
        ]
        await ctx.send(f"{random.choice(phrases)}")
    elif ctx.message.author.name == "Bryan":
        await ctx.send(f"terrible choice bryan, cringepilled taste beta simp")


def queue_message(songs: List[str]) -> str:
    capped = songs[:10]
    msg = "\n".join([f"{i + 1}: {capped[i]}" for i in range(len(capped))])
    if len(songs) > 10:
        msg += "\n..."
    return msg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = get_logger(__name__)
