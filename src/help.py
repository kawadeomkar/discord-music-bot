"""Embed-based help command.

discord.py ships two built-in implementations (DefaultHelpCommand,
MinimalHelpCommand); both format through a Paginator into plain text, so an
embed-only bot has to subclass HelpCommand directly. Only the dispatch methods
are overridden — command_callback still does the resolution, which means
`-help p` (an alias) and `-help MusicBot` (the cog) keep working for free.

Per-command copy lives on the commands themselves (brief/help/usage/extras in
src/musicbot.py) rather than in a table here, so a new command shows up in the
help output as soon as it is declared.
"""

from typing import Any, List, Mapping, Optional

import discord
from discord.ext import commands

from src.util import notice_embed

HELP_COLOR = discord.Color.blurple()

# Display order of the extras["category"] buckets in the command list.
CATEGORY_ORDER: tuple[str, ...] = ("Playback", "Queue", "Utility")
CATEGORY_EMOJI: dict[str, str] = {
    "Playback": "▶️",
    "Queue": "🎶",
    "Utility": "🔧",
    "Other": "📌",
}
UNCATEGORISED = "Other"

# Discord's hard cap on an embed field value.
_FIELD_LIMIT = 1024

SOURCES = (
    "**YouTube** — video links, playlist links, or plain words to search with. "
    "A `?t=` / `?ts=` timestamp starts the song at that offset.\n"
    "**Spotify** — track and playlist links. Each title is matched to its "
    "YouTube audio, so a playlist may take a moment to queue.\n"
    "**SoundCloud** — track links."
)

TIPS = (
    "• `play` pulls the bot into your voice channel — no need to `join` first.\n"
    "• The bot disconnects on its own 10 seconds after the last person leaves.\n"
    "• The **Now Playing** card re-anchors itself to the bottom of the channel "
    "so its live progress bar is never buried by other messages.\n"
    "• Queue, history and volume are saved per server and restored if the bot "
    "restarts mid-song.\n"
    "• A volume change applies from the **next** song onwards."
)


class MusicHelpCommand(commands.HelpCommand):
    """Renders the command list and per-command help as embeds."""

    def __init__(self, **options: Any) -> None:
        super().__init__(
            command_attrs={
                "name": "help",
                "aliases": ["commands"],
                "brief": "show this message",
                "help": (
                    "Shows the full command list, or detailed help for a single "
                    "command — its description, usage, aliases and examples. "
                    "Aliases work here too, so `-help np` is the same as `-help now`."
                ),
                "usage": "[command]",
                "extras": {
                    "category": "Utility",
                    "examples": ["-help", "-help play", "-help np"],
                },
            },
            **options,
        )

    def get_destination(self) -> discord.abc.Messageable:  # type: ignore[override]
        """The invoking Context, not its channel.

        The base implementation returns ``context.channel``, whose bare send()
        would bury the Now Playing host message mid-song. MusicContext.send
        keeps the NP block glued to the bottom of the channel — see
        docs/NOW_PLAYING_EMBED_ATTACH_PLAN.md.

        The base signature promises a MessageableChannel; a Context is only
        Messageable, which is all this class ever uses it for (send()).
        """
        return self.context

    # ── formatting helpers ────────────────────────────────────────────────────

    @property
    def prefix(self) -> str:
        return self.context.clean_prefix

    def get_command_signature(self, command: commands.Command, /) -> str:
        """`-play <url or search terms>`.

        The base implementation inlines aliases into the name as
        `-[play|p|sing] …`; aliases get their own field here, so keep the
        signature to the canonical name plus its arguments. Command.signature
        returns the `usage=` kwarg verbatim when one is set.
        """
        return f"{self.prefix}{command.qualified_name} {command.signature}".strip()

    def _extras(self, command: commands.Command) -> dict:
        return command.extras or {}

    def _category(self, command: commands.Command) -> str:
        category = self._extras(command).get("category", UNCATEGORISED)
        return category if category in CATEGORY_ORDER else UNCATEGORISED

    def _aliases(self, command: commands.Command) -> str:
        return " ".join(f"`{alias}`" for alias in command.aliases)

    def _entry(self, command: commands.Command) -> str:
        """One command's two-line block in the command list."""
        signature = f"**`{self.get_command_signature(command).strip()}`**"
        aliases = self._aliases(command)
        header = f"{signature}  ·  {aliases}" if aliases else signature
        summary = command.brief or command.short_doc or "no description"
        return f"{header}\n{summary}"

    def _add_command_field(
        self, embed: discord.Embed, category: str, entries: List[str]
    ) -> None:
        """Add one category field, splitting across continuation fields rather
        than letting Discord reject an over-long value (>1024 chars)."""
        emoji = CATEGORY_EMOJI.get(category, CATEGORY_EMOJI[UNCATEGORISED])
        name = f"{emoji} {category}"
        chunk: List[str] = []
        length = 0
        for entry in entries:
            # +1 for the newline joining this entry to the previous one.
            if chunk and length + len(entry) + 1 > _FIELD_LIMIT:
                embed.add_field(name=name, value="\n".join(chunk), inline=False)
                name = f"{emoji} {category} (cont.)"
                chunk, length = [], 0
            chunk.append(entry)
            length += len(entry) + 1
        if chunk:
            embed.add_field(name=name, value="\n".join(chunk), inline=False)

    # ── dispatch ──────────────────────────────────────────────────────────────

    async def send_bot_help(
        self, mapping: Mapping[Optional[commands.Cog], List[commands.Command]], /
    ) -> None:
        prefix = self.prefix
        everything = [cmd for cmds in mapping.values() for cmd in cmds]
        visible = await self.filter_commands(everything, sort=True)

        embed = discord.Embed(
            title="🎵 Music Bot — command reference",
            description=(
                "Plays audio from YouTube, Spotify and SoundCloud in your voice "
                f"channel.\n\nEvery command starts with `{prefix}` — for example "
                f"`{prefix}play lofi hip hop`. Run `{prefix}help <command>` for a "
                "command's usage, aliases and examples."
            ),
            color=HELP_COLOR,
        )

        buckets: dict[str, List[str]] = {}
        for command in visible:
            buckets.setdefault(self._category(command), []).append(self._entry(command))

        ordered = [c for c in CATEGORY_ORDER if c in buckets]
        ordered += [c for c in buckets if c not in CATEGORY_ORDER]
        for category in ordered:
            self._add_command_field(embed, category, buckets[category])

        embed.add_field(name="🎧 What you can play", value=SOURCES, inline=False)
        embed.add_field(name="💡 Good to know", value=TIPS, inline=False)
        embed.set_footer(
            text=f"{len(visible)} commands · {prefix}help <command> for details"
        )
        await self.get_destination().send(embed=embed)

    async def send_cog_help(self, cog: commands.Cog, /) -> None:
        # Every command lives in the single MusicBot cog, so `-help MusicBot` is
        # just the full list — no separate, near-identical rendering.
        await self.send_bot_help(self.get_bot_mapping())

    async def send_command_help(self, command: commands.Command, /) -> None:
        prefix = self.prefix
        extras = self._extras(command)
        category = self._category(command)
        emoji = CATEGORY_EMOJI.get(category, CATEGORY_EMOJI[UNCATEGORISED])

        embed = discord.Embed(
            title=f"{emoji} {prefix}{command.qualified_name}",
            description=command.help or command.brief or "no description",
            color=HELP_COLOR,
        )
        embed.add_field(
            name="Usage",
            value=f"`{self.get_command_signature(command).strip()}`",
            inline=False,
        )
        if command.aliases:
            embed.add_field(
                name="Aliases",
                value=" ".join(f"`{prefix}{alias}`" for alias in command.aliases),
                inline=False,
            )
        examples: List[str] = extras.get("examples", [])
        if examples:
            embed.add_field(
                name="Examples",
                value="\n".join(f"`{example}`" for example in examples),
                inline=False,
            )
        note: Optional[str] = extras.get("note")
        if note:
            embed.add_field(name="Note", value=note, inline=False)
        embed.set_footer(text=f"{category} · {prefix}help for the full command list")
        await self.get_destination().send(embed=embed)

    async def send_group_help(self, group: commands.Group, /) -> None:
        # No command groups exist today; degrade to the single-command embed
        # rather than falling back to the base class's plaintext output.
        await self.send_command_help(group)

    async def send_error_message(self, error: str, /) -> None:
        await self.get_destination().send(
            embed=notice_embed(
                f"{error}\nRun `{self.prefix}help` to see every command.",
                discord.Color.red(),
            )
        )

    def command_not_found(self, string: str, /) -> str:
        return f'No command called "{string}".'
