"""-settings' embeds: the server card, the bot card, one setting's detail, and the
reply to every change and refusal. Pure: each builder takes what the command
resolved and returns an embed. Every value renders through the registry's
format_value, so a value copied off a card parses back. See src/settings.py for
the registry itself."""

import os
from dataclasses import dataclass
from typing import Final, Optional

import discord

from src import config
from src.guild_state import ConfigField, GuildConfig
from src.settings import (
    SETTINGS,
    SettingKind,
    SettingScope,
    SettingSpec,
    SettingValue,
    allowed_text,
    followed_knob,
    format_value,
    from_stored,
)
from src.util import (
    codeblock_fields,
    notice_embed,
    safe_label,
    truncate_embed_title,
)

CHANGE_COLOR: Final = discord.Color.blue()
REFUSAL_COLOR: Final = discord.Color.red()
DEGRADED_COLOR: Final = discord.Color.orange()

SET_HERE: Final = "set here"
DEFAULT: Final = "default"
NOT_SAVED: Final = "not saved"
BOT_MINIMUM: Final = "bot minimum"

_SERVER_SPECS: Final = tuple(s for s in SETTINGS if s.scope is SettingScope.SERVER)
_BOT_SPECS: Final = tuple(s for s in SETTINGS if s.scope is SettingScope.BOT)

_CARD_INTRO: Final = (
    "Anyone can view these. Changing one needs Manage Server; volume can also be "
    "changed by anyone in the bot's voice channel. The bot's operator can change "
    "them too."
)
_CARD_UNREAD: Final = (
    "⚠️ Couldn't read this server's saved settings just now, so these are the "
    "defaults. Try again in a moment."
)
_SAVED: Final = "It is saved for this server."
_NOT_SAVED: Final = (
    "⚠️ It could not be saved (Redis is unavailable), so it applies until the bot "
    "restarts."
)
# -debug --enable's sentence: what turning the footer on publishes.
_DEBUG_DISCLOSURE: Final = (
    "While it is on, every embed here — including the live Now Playing card — shows "
    "the bot process's load to anyone who can read the channel."
)


@dataclass(frozen=True, slots=True)
class Shown:
    """A value in its spec's unit, and the source label it renders with."""

    value: SettingValue
    source: str


def server_default(spec: SettingSpec, *, debug_default: bool) -> SettingValue:
    """What a server setting runs on while this server has not set it: its own
    default, or the bot's current value for one that follows the bot."""
    if spec.default is not None:
        return spec.default
    if spec.field == ConfigField.DEBUG_MODE:
        return debug_default
    if (knob := followed_knob(spec)) is not None:
        return config.effective(knob)
    raise ValueError(f"{spec.key} has no server default")


def server_shown(
    spec: SettingSpec,
    stored: Optional[GuildConfig],
    *,
    debug_default: bool,
    persisted: bool,
) -> Shown:
    """The value a server setting renders with, from the cached config. A stored
    value under its write-time minimum renders that minimum, which is what runs,
    and names the stored value: `5s`, `bot minimum; set here 3s`."""
    raw = getattr(stored, spec.field) if stored is not None and spec.field else None
    if raw is None:
        value = server_default(spec, debug_default=debug_default)
    elif isinstance(raw, (bool, str)):
        value = raw
    else:
        value = from_stored(spec, raw)
    source = DEFAULT if raw is None else SET_HERE
    if (
        raw is not None
        and spec.write_minimum is not None
        and isinstance(value, float)
        and value < (floor := spec.write_minimum())
    ):
        source = f"{BOT_MINIMUM}; set here {format_value(spec, value)}"
        value = floor
    return Shown(value, source if persisted else NOT_SAVED)


def _env_set(spec: SettingSpec) -> bool:
    # The test the env parse applies: unset and blank both leave the default.
    return bool(spec.env and (os.environ.get(spec.env) or "").strip())


def bot_shown(
    spec: SettingSpec,
    *,
    host_debug_default: bool,
    debug_default_override: Optional[bool],
) -> Shown:
    """A bot setting as the operator's views render it: the value in force, and
    where it comes from — `bot owner; env 3s` for an override."""
    origin = "env" if _env_set(spec) else "default"
    if spec.attr is None:
        base = format_value(spec, host_debug_default)
        if debug_default_override is None:
            return Shown(host_debug_default, origin)
        return Shown(
            debug_default_override, f"bot owner, until restart; {origin} {base}"
        )
    override = config.override(spec.attr)
    baseline = config.baseline(spec.attr)
    if override is None:
        return Shown(baseline, origin)
    return Shown(override, f"bot owner; {origin} {format_value(spec, baseline)}")


# ── Cards ────────────────────────────────────────────────────────────────────


def _hint_name(specs: tuple[SettingSpec, ...]) -> str:
    """A name typed with dashes, read off the card: the first label with a space."""
    for spec in specs:
        if " " in spec.label:
            return spec.label.casefold().replace(" ", "-")
    return specs[0].key


def server_card(
    *,
    guild_name: str,
    rows: list[tuple[SettingSpec, Shown]],
    read_failed: bool,
    operator: bool,
) -> discord.Embed:
    """This server's settings, one prose line each under its group. `rows` is in
    registry order; the operator's card ends with a pointer to -settings bot."""
    description = f"{_CARD_UNREAD}\n\n{_CARD_INTRO}" if read_failed else _CARD_INTRO
    if operator:
        description += "\n`-settings bot` shows bot-wide settings."
    embed = discord.Embed(
        title=truncate_embed_title(f"Server settings · {safe_label(guild_name, 100)}"),
        description=description,
        color=DEGRADED_COLOR if read_failed else CHANGE_COLOR,
    )
    for group in dict.fromkeys(spec.group for spec, _ in rows):
        lines = [
            f"**{spec.label}** · {format_value(spec, shown.value)} · {shown.source}"
            for spec, shown in rows
            if spec.group is group
        ]
        embed.add_field(name=group.value, value="\n".join(lines), inline=False)
    name = _hint_name(tuple(spec for spec, _ in rows))
    embed.set_footer(
        text=(
            f"Names work with dashes: -settings {name} to see one · "
            f"-settings {name} <value> · -settings {name} reset"
        )
    )
    return embed


def bot_card(*, rows: list[tuple[SettingSpec, Shown]], ignored: bool) -> discord.Embed:
    """The bot-wide settings, for the operator only, as code-block rows in
    -debug's Config form."""
    description = (
        "Bot-wide: each applies in every server. They can't be changed from chat yet."
    )
    if ignored:
        description = (
            "Stored bot settings are ignored (`BOT_SETTINGS_OVERRIDES=ignore`).\n\n"
            + description
        )
    embed = discord.Embed(
        title="Bot settings", description=description, color=CHANGE_COLOR
    )
    width = max(len(spec.key) for spec, _ in rows)
    for group in dict.fromkeys(spec.group for spec, _ in rows):
        lines = [
            f"{spec.key:<{width}}  {format_value(spec, shown.value)} ({shown.source})"
            for spec, shown in rows
            if spec.group is group
        ]
        for name, value in codeblock_fields(group.value, lines):
            embed.add_field(name=name, value=value, inline=False)
    embed.set_footer(text="-settings bot <setting> shows one setting in full")
    return embed


def _names(spec: SettingSpec) -> str:
    also = f"; also {', '.join(f'`{a}`' for a in spec.aliases)}" if spec.aliases else ""
    return f"`{spec.key}`{also}"


def _accepts(spec: SettingSpec) -> str:
    text = allowed_text(spec, now=True)
    if spec.kind in (SettingKind.SWITCH, SettingKind.TIMEZONE):
        return f"Takes {text}"
    return f"Allowed {text}"


def detail(
    spec: SettingSpec,
    shown: Shown,
    *,
    default: Optional[SettingValue],
    read_failed: bool = False,
) -> discord.Embed:
    """One setting: names, summary, current value and source, default, range and
    when it applies. `default` is None for a bot setting, whose source label
    already names its baseline."""
    parts = [f"Current **{format_value(spec, shown.value)}** ({shown.source})"]
    if default is not None:
        follows = " (the bot's default)" if spec.default is None else ""
        parts.append(f"Default {format_value(spec, default)}{follows}")
    parts += [_accepts(spec), f"Applies {spec.applies}"]
    if spec.field == ConfigField.VOLUME:
        parts.append(
            "Can be changed with Manage Server, or by anyone in the bot's voice channel"
        )
    text = f"**{spec.label}** ({_names(spec)}) — {spec.summary} " + " · ".join(parts)
    return discord.Embed(
        description=f"{_CARD_UNREAD}\n\n{text}." if read_failed else f"{text}.",
        color=DEGRADED_COLOR if read_failed else CHANGE_COLOR,
    )


# ── Replies ──────────────────────────────────────────────────────────────────


def _changed_by(mention: str) -> str:
    return f"Changed by {mention}."


def set_reply(
    spec: SettingSpec,
    value: SettingValue,
    *,
    previous: Optional[Shown],
    persisted: bool,
    mention: str,
) -> discord.Embed:
    """A server setting changed. `previous` is None when this server's stored
    values could not be read, so what it replaced is unknown."""
    text = f"**{spec.label}** is now **{format_value(spec, value)}** for this server"
    if previous is not None:
        was = "the default" if previous.source == DEFAULT else previous.source
        text += f" (was **{format_value(spec, previous.value)}**, {was})"
    text += "."
    if spec.field == ConfigField.DEBUG_MODE and value is True:
        text += f" {_DEBUG_DISCLOSURE}"
    else:
        text += f" It applies {spec.applies}."
    text += f" {_SAVED if persisted else _NOT_SAVED} {_changed_by(mention)}"
    return notice_embed(text, CHANGE_COLOR)


def reset_reply(
    spec: SettingSpec, default: SettingValue, *, persisted: bool, mention: str
) -> discord.Embed:
    """A server setting back to what it runs on unset: its own default, or the
    bot's current value for one that follows the bot."""
    shown = format_value(spec, default)
    if spec.default is None:
        text = (
            f"**{spec.label}** here is back to the bot's default, which is "
            f"**{shown}** right now."
        )
    else:
        text = f"**{spec.label}** is back to the default, **{shown}**."
    text += f" {_SAVED if persisted else _NOT_SAVED} {_changed_by(mention)}"
    return notice_embed(text, CHANGE_COLOR)


def not_applied() -> discord.Embed:
    """A write refused because the server's settings were cleared under it."""
    return notice_embed(
        "Nothing changed: this server's settings were cleared while the change was "
        "being made.",
        DEGRADED_COLOR,
    )


def refusal(text: str) -> discord.Embed:
    return notice_embed(text, REFUSAL_COLOR)


DM_PER_SERVER: Final = "Server settings are per server — use this in a server channel."
DM_NEEDS_BOT: Final = (
    "In a direct message, name the scope: `-settings bot <setting> <value>`. Server "
    "settings are changed in the server."
)
NO_PERMISSION: Final = (
    "Changing server settings needs the **Manage Server** permission. Run "
    "`-settings` to see the current values."
)
NO_PERMISSION_VOLUME: Final = (
    "Changing the volume needs the **Manage Server** permission, or being in the "
    "bot's voice channel (any voice channel while it isn't in one). Join it and try "
    "again, or use `-volume`."
)
OPERATOR_ONLY: Final = (
    "Bot-wide settings can only be changed by the bot's operator. Run `-settings` "
    "for this server's."
)
OPERATOR_UNCONFIRMED: Final = (
    "Couldn't confirm the bot's operator just now, so bot-wide settings can't be "
    "changed. Try again in a minute."
)
BOT_WRITE_UNAVAILABLE: Final = "Bot settings can't be changed from chat yet."


def server_rows(
    stored: Optional[GuildConfig],
    *,
    debug_default: bool,
    unsaved: frozenset[str],
) -> list[tuple[SettingSpec, Shown]]:
    """Every server setting with the value it renders, in registry order."""
    return [
        (
            spec,
            server_shown(
                spec,
                stored,
                debug_default=debug_default,
                persisted=not (spec.field and spec.field in unsaved),
            ),
        )
        for spec in _SERVER_SPECS
    ]


def bot_rows(
    *, host_debug_default: bool, debug_default_override: Optional[bool]
) -> list[tuple[SettingSpec, Shown]]:
    """Every bot setting with the value it renders, in registry order."""
    return [
        (
            spec,
            bot_shown(
                spec,
                host_debug_default=host_debug_default,
                debug_default_override=debug_default_override,
            ),
        )
        for spec in _BOT_SPECS
    ]
