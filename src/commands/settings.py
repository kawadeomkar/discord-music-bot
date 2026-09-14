"""`-settings` — this server's settings, one setting in detail, a change to one,
or (for the bot's operator) the bot-wide settings. The grammar and the registry
are src/settings.py; the embeds are src/settings_card.py."""

from dataclasses import replace
from typing import TYPE_CHECKING, Final, Optional

import discord
from discord.ext import commands
from opentelemetry import trace

from src import settings_card as card
from src.guild_state import ConfigField, ConfigFieldName, GuildConfig, is_config_field
from src.play_placement import check_voice_permissions
from src.settings import (
    ALL_CONFIG_FIELDS,
    SETTINGS,
    BotSettings,
    Refusal,
    RefusalReason,
    SettingScope,
    SettingsAction,
    SettingSpec,
    SettingsRequest,
    SettingValue,
    find,
    format_value,
    is_bullet_shaped,
    parse_settings_args,
    to_stored,
    wrong_scope_text,
)
from src.util import get_logger, is_operator, owner_lookup_backing_off

if TYPE_CHECKING:
    # A runtime import would close the cycle: musicbot imports this module.
    from src.musicbot import MusicBot

log = get_logger(__name__)

# A bullet-shaped message (`- settings page is broken`) refused for one of these
# was likely never meant for the bot, so it gets no reply. The rest are only
# reachable by naming a setting exactly.
_QUIET_WHEN_BULLETED: Final = frozenset(
    {
        RefusalReason.TOO_MUCH,
        RefusalReason.BAD_SHAPE,
        RefusalReason.UNKNOWN_KEY,
        RefusalReason.RESET_WITHOUT_KEY,
    }
)
_NO_MENTIONS: Final = discord.AllowedMentions.none()


async def run(ctx: commands.Context, arg: str, *, cog: MusicBot) -> None:
    """`-settings`. Takes the cog for its GuildSettings and DebugSettings. Every
    Redis read or write it makes is bounded by CONFIG_IO_TIMEOUT_SECS."""
    span = trace.get_current_span()
    tail = ctx.message.content[len(ctx.prefix or "") :]
    parsed = parse_settings_args(arg, tail=tail)
    if isinstance(parsed, Refusal):
        if is_bullet_shaped(tail) and parsed.reason in _QUIET_WHEN_BULLETED:
            log.debug(
                f"settings: bullet-shaped message ignored ({parsed.reason.value})"
            )
            span.set_attribute("settings.refused", RefusalReason.BULLET_SHAPE.value)
            return
    else:
        span.set_attribute("settings.scope", parsed.scope.value)
        span.set_attribute("settings.action", parsed.action.value)
        if parsed.spec is not None:
            span.set_attribute("settings.key", parsed.spec.key)

    if ctx.guild is None:
        await _in_dm(ctx, parsed, cog=cog)
    elif isinstance(parsed, Refusal):
        await _refuse_parse(ctx, parsed)
    elif parsed.scope is SettingScope.BOT:
        await _bot_scope(ctx, parsed, cog=cog)
    elif parsed.action in (SettingsAction.SHOW, SettingsAction.DETAIL):
        await _show_server(ctx, parsed, cog=cog)
    else:
        await _change_server(ctx, parsed, cog=cog)


async def _send(ctx: commands.Context, embed: discord.Embed) -> None:
    await ctx.send(embed=embed, allowed_mentions=_NO_MENTIONS)


async def _refuse(ctx: commands.Context, reason: RefusalReason, text: str) -> None:
    trace.get_current_span().set_attribute("settings.refused", reason.value)
    await _send(ctx, card.refusal(text))


async def _refuse_operator(ctx: commands.Context) -> None:
    """A bot-scope request from someone is_operator did not confirm."""
    if owner_lookup_backing_off():
        await _refuse(
            ctx, RefusalReason.OPERATOR_UNCONFIRMED, card.OPERATOR_UNCONFIRMED
        )
    else:
        await _refuse(ctx, RefusalReason.OPERATOR_ONLY, card.OPERATOR_ONLY)


async def _refuse_parse(ctx: commands.Context, refusal: Refusal) -> None:
    """A parse refusal in a server. One made after `bot` can name a bot setting
    or its range, so only the operator sees it; `bot volume` names only a server
    setting, and a bot setting typed without `bot` only what was typed."""
    spec = refusal.spec
    wrong_scope = refusal.reason is RefusalReason.WRONG_SCOPE and spec is not None
    if (
        refusal.scope is SettingScope.BOT
        and not wrong_scope
        and not await is_operator(ctx)
    ):
        await _refuse_operator(ctx)
        return
    text = refusal.text
    if wrong_scope and spec is not None and spec.scope is SettingScope.BOT:
        text = wrong_scope_text(spec, operator=await is_operator(ctx))
    await _refuse(ctx, refusal.reason, text)


async def _in_dm(
    ctx: commands.Context, parsed: SettingsRequest | Refusal, *, cog: MusicBot
) -> None:
    """A DM has no server, so it reaches only the bot scope, and only for the
    operator: `-settings` alone shows the bot card, and a setting named without
    `bot` is never read as bot scope."""
    if not await is_operator(ctx):
        await _refuse(ctx, RefusalReason.DM_NEEDS_BOT, card.DM_PER_SERVER)
        return
    if isinstance(parsed, SettingsRequest):
        if parsed.scope is SettingScope.BOT:
            await _bot_scope(ctx, parsed, cog=cog, operator=True)
        elif parsed.action is SettingsAction.SHOW:
            await _send(ctx, _bot_card(cog))
        else:
            await _refuse(ctx, RefusalReason.DM_NEEDS_BOT, card.DM_NEEDS_BOT)
    elif parsed.scope is SettingScope.SERVER:
        await _refuse(ctx, RefusalReason.DM_NEEDS_BOT, card.DM_NEEDS_BOT)
    else:
        await _refuse(ctx, parsed.reason, parsed.text)


def _bot_settings(cog: MusicBot) -> Optional[BotSettings]:
    """MusicBotApp's, built in setup_hook; None only on a bot that never ran it."""
    found = getattr(cog.bot, "bot_settings", None)
    return found if isinstance(found, BotSettings) else None


def _bot_card(cog: MusicBot) -> discord.Embed:
    bot_settings = _bot_settings(cog)
    unsaved = frozenset(
        spec.key
        for spec in SETTINGS
        if bot_settings is not None
        and spec.scope is SettingScope.BOT
        and not bot_settings.is_persisted(spec)
    )
    return card.bot_card(
        rows=card.bot_rows(
            host_debug_default=cog.debug_settings.host_default,
            debug_default_override=cog.debug_settings.default_override,
            unsaved=unsaved,
        ),
        ignored=bot_settings is not None and bot_settings.ignore_stored,
    )


async def _bot_scope(
    ctx: commands.Context,
    request: SettingsRequest,
    *,
    cog: MusicBot,
    operator: bool = False,
) -> None:
    """The bot card, a bot setting's detail, or a bot write: the operator's only."""
    if not (operator or await is_operator(ctx)):
        await _refuse_operator(ctx)
        return
    spec = request.spec
    bot_settings = _bot_settings(cog)
    if request.action is SettingsAction.SHOW:
        await _send(ctx, _bot_card(cog))
        return
    assert spec is not None  # the parse sets a detail's, a set's and a reset's
    if request.action is SettingsAction.DETAIL:
        shown = card.bot_shown(
            spec,
            host_debug_default=cog.debug_settings.host_default,
            debug_default_override=cog.debug_settings.default_override,
            persisted=bot_settings is None or bot_settings.is_persisted(spec),
        )
        await _send(ctx, card.detail(spec, shown, default=None))
        return
    if bot_settings is None:
        raise RuntimeError("bot settings are not set up on this bot")
    if spec.field is None:
        await _change_debug_default(ctx, request, cog=cog, bot_settings=bot_settings)
        return
    if bot_settings.ignore_stored:
        await _refuse(ctx, RefusalReason.OVERRIDES_IGNORED, card.OVERRIDES_IGNORED)
        return
    if request.action is SettingsAction.SET:
        assert request.value is not None  # the parse sets a SET's value
        result = await bot_settings.write(spec, request.value)
        embed = card.bot_set_reply(
            spec,
            request.value,
            previous=result.previous,
            persisted=result.persisted,
            mention=ctx.author.mention,
        )
    else:
        result = await bot_settings.write_reset(spec)
        embed = card.bot_reset_reply(
            spec, persisted=result.persisted, mention=ctx.author.mention
        )
    trace.get_current_span().set_attribute("settings.persisted", result.persisted)
    before = "env" if result.previous is None else format_value(spec, result.previous)
    after = "env" if request.value is None else format_value(spec, request.value)
    log.info(
        f"settings: bot {spec.key} {before} -> {after}",
        action=request.action.value,
        persisted=result.persisted,
    )
    await _send(ctx, embed)


async def _change_debug_default(
    ctx: commands.Context,
    request: SettingsRequest,
    *,
    cog: MusicBot,
    bot_settings: BotSettings,
) -> None:
    """debug-default is session-only: nothing is stored, so it changes even while
    stored bot settings are ignored."""
    spec = request.spec
    assert spec is not None
    value = request.value if isinstance(request.value, bool) else None
    if request.action is SettingsAction.SET and value is not None:
        bot_settings.apply(spec, value)
    else:
        bot_settings.reset(spec)
    debug_settings = cog.debug_settings
    guilds = cog.bot.guilds
    following = sum(1 for guild in guilds if not debug_settings.has_override(guild.id))
    shown = "DEBUG_MODE" if value is None else format_value(spec, value)
    log.info(f"settings: bot debug-default -> {shown}")
    await _send(
        ctx,
        card.debug_default_reply(
            value,
            host_default=debug_settings.host_default,
            following=following,
            total=len(guilds),
            mention=ctx.author.mention,
        ),
    )


def _field(spec: SettingSpec) -> ConfigFieldName:
    if spec.field is None or not is_config_field(spec.field):
        raise ValueError(f"{spec.key} is not a server setting")
    return spec.field


async def _read(cog: MusicBot, guild_id: int) -> bool:
    """Make sure this server's stored values are cached; False when they could
    not be read, so what is cached may be missing some."""
    settings = cog.guild_settings
    return settings.is_complete(guild_id) or await settings.load(guild_id) is not None


def _unsaved(cog: MusicBot, guild_id: int) -> frozenset[str]:
    settings = cog.guild_settings
    return frozenset(
        field
        for field in ALL_CONFIG_FIELDS
        if not settings.is_persisted(guild_id, field)
    )


async def _show_server(
    ctx: commands.Context, request: SettingsRequest, *, cog: MusicBot
) -> None:
    guild = ctx.guild
    assert guild is not None  # run() sends a DM elsewhere
    read = await _read(cog, guild.id)
    rows = card.server_rows(
        cog.guild_settings.peek(guild.id),
        debug_default=cog.debug_settings.default,
        unsaved=_unsaved(cog, guild.id),
    )
    if request.spec is None:
        embed = card.server_card(
            guild_name=guild.name,
            rows=rows,
            read_failed=not read,
            operator=await is_operator(ctx),
        )
        await _send(ctx, embed)
        return
    spec = request.spec
    shown = next(shown for row, shown in rows if row is spec)
    default = card.server_default(spec, debug_default=cog.debug_settings.default)
    await _send(ctx, card.detail(spec, shown, default=default, read_failed=not read))


async def _may_change(ctx: commands.Context, spec: SettingSpec) -> bool:
    """Manage Server; for volume, -volume's voice gate; then the operator. The
    cheap checks first: the operator check can be a REST call."""
    author = ctx.author
    if isinstance(author, discord.Member) and author.guild_permissions.manage_guild:
        return True
    if spec.field == ConfigField.VOLUME:
        vc = ctx.voice_client
        voice_client = vc if isinstance(vc, discord.VoiceClient) else None
        if check_voice_permissions(author, voice_client, "volume") is None:
            return True
    return await is_operator(ctx)


async def _change_server(
    ctx: commands.Context, request: SettingsRequest, *, cog: MusicBot
) -> None:
    guild = ctx.guild
    spec = request.spec
    assert guild is not None and spec is not None  # run() and the parse ensure both
    if not await _may_change(ctx, spec):
        text = (
            card.NO_PERMISSION_VOLUME
            if spec.field == ConfigField.VOLUME
            else card.NO_PERMISSION
        )
        await _refuse(ctx, RefusalReason.NO_PERMISSION, text)
        return
    field = _field(spec)
    settings = cog.guild_settings
    debug_default = cog.debug_settings.default
    # The reply's "(was …)" comes from the write's own previous entry, which only
    # a complete cache makes true.
    known = await _read(cog, guild.id)
    if request.action is SettingsAction.SET:
        assert request.value is not None  # the parse sets a SET's value
        value: SettingValue = request.value
        change = replace(GuildConfig(), **{field: _stored(spec, value)})
        result = await settings.write(guild.id, change)
    else:
        value = card.server_default(spec, debug_default=debug_default)
        result = await settings.reset(guild.id, field)
    trace.get_current_span().set_attribute("settings.persisted", result.persisted)
    if not result.applied:
        await _send(ctx, card.not_applied())
        return
    previous = (
        card.server_shown(
            spec, result.previous, debug_default=debug_default, persisted=True
        )
        if known
        else None
    )
    log.info(
        f"settings: {spec.key} "
        f"{format_value(spec, previous.value) if previous else '?'} -> "
        f"{format_value(spec, value)}",
        action=request.action.value,
        persisted=result.persisted,
    )
    if request.action is SettingsAction.SET:
        # The operator can change either scope, so their server write names the
        # bot-wide one; a server admin's does not.
        dual = isinstance(find(spec.key, SettingScope.BOT), SettingSpec)
        embed = card.set_reply(
            spec,
            value,
            previous=previous,
            persisted=result.persisted,
            mention=ctx.author.mention,
            bot_form=dual and await is_operator(ctx),
        )
    else:
        embed = card.reset_reply(
            spec, value, persisted=result.persisted, mention=ctx.author.mention
        )
    await _send(ctx, embed)


def _stored(spec: SettingSpec, value: SettingValue) -> bool | float | str:
    """A parsed value as its GuildConfig field holds it."""
    if isinstance(value, (bool, str)):
        return value
    return to_stored(spec, value)
