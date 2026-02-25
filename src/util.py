import functools
import logging
import random
import sys
from typing import List

from discord.ext import commands


def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        sig = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        log.info(f"Calling {func.__name__}({sig})")
        value = func(*args, **kwargs)
        log.info(f"{func.__name__!r} returned {value!r}")
        return value

    return wrapper_debug


def validate_input(*expected_args):
    def validate_outer(func):
        @functools.wraps(func)
        def validate_wrap(*args, **kwargs):
            for exp in expected_args:
                if exp not in args or exp not in kwargs:
                    raise Exception(f"Expected argument does not exist {exp}")
            return func(*args, **kwargs)

        return validate_wrap

    return validate_outer


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
    msg = "\n".join([f"{i}: {songs[i-1]}" for i in range(1, len(songs[:10]))])
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
