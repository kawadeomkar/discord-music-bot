"""`-debug` — the diagnostic snapshot card, or the per-guild debug-mode toggle."""

from typing import TYPE_CHECKING, Optional, cast

import discord
from discord.ext import commands

from src.config import (
    debug_prometheus_url,
    history_archive_enabled,
    using_default_postgres_password,
)
from src.redis_client import GuildRedisStore
from src.util import (
    notice_embed,
)

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the cog
    # is only named in annotations. Same guard recovery.py and musicplayer.py use.
    from src.musicbot import MusicBot
from src.debug import (
    ArchiveStatsReader,
    DebugAction,
    DebugInputs,
    parse_debug_arg,
    run_debug_dashboard,
    unknown_arg_message,
)
from src.util import get_logger

log = get_logger(__name__)


async def run(ctx: commands.Context, arg: str, *, cog: MusicBot) -> None:
    """`-debug` — the snapshot card, or the per-guild toggle.

    Takes the cog: the snapshot reads the player registry, the Redis handle and the
    archive off it, and the toggle writes the guild's stored choice through it.
    """
    action = parse_debug_arg(arg)
    if action is None:
        await ctx.send(
            embed=notice_embed(unknown_arg_message(arg), discord.Color.red())
        )
        return
    if action is not DebugAction.STATUS:
        await toggle(ctx, action, cog=cog)
        return
    inputs = await build_inputs(ctx, cog=cog)
    # No typing indicator and no ctx.send: the dashboard sends its own
    # skeleton immediately and edits it as blocks land, so the reply IS the
    # acknowledgement. It uses channel.send to stay off the Now Playing host,
    # which an edit loop must not own (src/dashboard.py).
    await run_debug_dashboard(ctx, inputs)


async def toggle(ctx: commands.Context, action: DebugAction, *, cog: MusicBot) -> None:
    """Apply `--enable`/`--disable` to the invoking guild and confirm it."""
    if ctx.guild is None:
        await ctx.send(
            embed=notice_embed(
                "Debug mode is set per server, so it can't be toggled from a "
                f"direct message. It is currently "
                f"**{'on' if cog.debug_settings.default else 'off'}** here, following "
                "the host's `DEBUG_MODE` default.",
                discord.Color.orange(),
            )
        )
        return
    # A moderator action: the toggle is guild-wide and every member sees the
    # result on every reply, so it is not the invoking user's to make alone.
    # Reading `-debug` stays open to everyone; only writing is gated.
    author = ctx.author
    may_toggle = (
        isinstance(author, discord.Member) and author.guild_permissions.manage_guild
    ) or await is_operator(ctx)
    if not may_toggle:
        await ctx.send(
            embed=notice_embed(
                "Debug mode changes what **every** embed in this server looks "
                "like, so switching it needs the Manage Server permission. Run "
                "`-debug` on its own to see the current state.",
                discord.Color.red(),
            )
        )
        return
    enabled = action is DebugAction.ENABLE
    persisted = await cog.debug_settings.toggle(cog.redis, ctx.guild.id, enabled)
    # Say which kind of change this was. A guild told "on" that quietly reverts
    # on the next restart reads as the bot ignoring them, so a degraded write is
    # named rather than rounded up to success.
    durability = (
        "The setting is saved for this server."
        if persisted
        else "⚠️ It could not be saved (Redis is unavailable), so it applies "
        "until the bot restarts."
    )
    # Names what enabling publishes, at the moment the choice is made: the
    # footer reports the whole process's load, and the Now Playing card carries
    # it passively to everyone who can read the channel while music plays.
    scope = (
        " While it is on, every embed here — including the live Now Playing "
        "card — shows the bot process's load to anyone who can read the channel."
        if enabled
        else ""
    )
    await ctx.send(
        embed=notice_embed(
            f"Debug mode is now **{'on' if enabled else 'off'}** for this "
            "server. Embeds "
            + ("carry" if enabled else "no longer carry")
            + " a debug footer; nothing about playback changes either way."
            + scope
            + " "
            + durability,
            discord.Color.blue(),
        )
    )


async def is_operator(ctx: commands.Context) -> bool:
    """Is the caller the bot owner? Fails CLOSED.

    `is_owner()` falls through to an `application_info()` REST call when neither
    owner_id nor owner_ids is configured (MusicBotApp sets neither), and it RAISES
    rather than returning False — a diagnostic must not disclose the host just
    because Discord blinked. discord.py caches the answer onto the bot afterwards,
    so this is one round trip per process, not per command.
    """
    try:
        return await ctx.bot.is_owner(ctx.author)
    except Exception as e:  # noqa: BLE001 — an unreachable owner is not an owner
        log.warning(f"owner check failed, denying: {type(e).__name__}: {e}")
        return False


async def build_inputs(ctx: commands.Context, *, cog: MusicBot) -> DebugInputs:
    """Everything the snapshot cannot reach on its own (src/debug.py importing
    MusicBot would be a cycle)."""
    guild_id = ctx.guild.id if ctx.guild else None
    archive_enabled = history_archive_enabled()
    operator = await is_operator(ctx)
    # Asked symmetrically — not only when the password IS the default — because a
    # row that renders for False and vanishes for True makes its own absence the
    # answer. Moot while the whole Checks block is owner-only, and it stays right
    # if that ever loosens.
    default_password = (
        (using_default_postgres_password() and archive_enabled) if operator else None
    )
    return DebugInputs(
        debug_enabled=cog.debug_settings.enabled(guild_id),
        debug_overridden=cog.debug_settings.has_override(guild_id),
        debug_persisted=cog.debug_settings.is_persisted(guild_id),
        players=len(cog.mps),
        player=cog.mps.get(guild_id) if guild_id is not None else None,
        redis=cog.redis,
        store=GuildRedisStore(cog.redis, guild_id)
        if cog.redis is not None and guild_id is not None
        else None,
        # Structural: PostgresHistoryArchive satisfies ArchiveStatsReader, and
        # a cog built without an archive (tests, disabled tier) passes None.
        archive=cast(Optional[ArchiveStatsReader], cog.history_archive),
        archive_enabled=archive_enabled,
        prometheus_url=debug_prometheus_url(),
        operator=operator,
        default_password=default_password,
        # Gated on `operator`: the card withholds its Runtime block from a
        # non-owner and says so, so the footer must not print those figures.
        debug_suffix=cog.debug_suffix(ctx, host_metrics=operator),
    )
