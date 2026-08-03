"""Man-page-styled embed help command.

Both built-ins format through a Paginator into plain text, so an embed-only bot must
subclass HelpCommand directly; only the dispatch methods are overridden, so
command_callback still resolves `-help p` (alias) and `-help MusicBot` (cog) for free.
Layout borrows from man(1) — caps section headers, hanging-indent entries — because a
column-aligned table's grid read poorly at Discord widths. Per-command copy lives on
the commands themselves (brief/help/usage/extras in src/musicbot.py).
"""

import textwrap
from typing import Any, Optional
from collections.abc import Mapping, Sequence

import discord
from discord.ext import commands

from src.util import notice_embed

HELP_COLOR = discord.Color.blurple()

# Display order: categories as rendered, and within each by frequency of use (daily
# verbs first, housekeeping last) — not alphabetically, which put `pause` above `play`.
CATEGORY_COMMANDS: dict[str, tuple[str, ...]] = {
    "Playback": ("play", "playnow", "pause", "resume", "skip", "stop", "volume"),
    "Queue": ("queue", "now", "history", "shuffle", "remove", "clear", "jump"),
    "Utility": ("help", "join", "ping"),
}
CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_COMMANDS)
UNCATEGORISED = "Other"

# Discord's hard cap on an embed field value.
_FIELD_LIMIT = 1024

# Entries live in code blocks — the only construct Discord renders monospace, so the
# only place a hanging indent survives. But Discord soft-wraps code blocks at the embed
# width (~54 chars on desktop, less on mobile) and restarts at column 0, so hard-wrapping
# narrower than any common width keeps the wrapping ours.
_WIDTH = 48
_INDENT = "    "
_FENCE = "```"

SOURCES = (
    "**YouTube** — video links, playlist links, or plain words to search with. "
    "A `?t=` / `?ts=` timestamp starts the song at that offset.\n"
    "**Spotify** — track and playlist links. Each title is matched to its "
    "YouTube audio, so a playlist may take a moment to queue.\n"
    "**SoundCloud** — track links."
)

TIPS = (
    "• Add `--help` to any command — `-play --help` — for its manual, the "
    "same as `-help play`.\n"
    "• `play` pulls the bot into your voice channel — no need to `join` first.\n"
    "• The bot disconnects on its own 10 seconds after the last person leaves.\n"
    "• The **Now Playing** card re-anchors itself to the bottom of the channel "
    "so its live progress bar is never buried by other messages.\n"
    "• Queue, history and volume are saved per server and restored if the bot "
    "restarts mid-song.\n"
    "• A volume change applies from the **next** song onwards."
)


class MusicHelpCommand(commands.HelpCommand):
    """Renders the command list and per-command help as man(1)-styled embeds."""

    def __init__(self, **options: Any) -> None:
        super().__init__(
            command_attrs={
                "name": "help",
                "aliases": ["commands"],
                "brief": "show this message",
                "help": (
                    "Shows the full command list, or detailed help for a single "
                    "command — its description, usage, aliases and examples. "
                    "Aliases work here too, so `-help np` is the same as `-help now`. "
                    "Adding `--help` to any command does the same thing: "
                    "`-play --help` is `-help play`."
                ),
                "usage": "[command]",
                "extras": {
                    "category": "Utility",
                    "examples": ["-help", "-help play", "-help np", "-play --help"],
                },
            },
            **options,
        )

    # Every send goes through self.context, never self.get_destination(): the inherited
    # one returns context.channel, whose bare send() would bury the Now Playing host
    # mid-song. Overriding get_destination() would be the natural hook, but its base
    # promises a MessageableChannel and a Context is only Messageable.

    # ── formatting helpers ────────────────────────────────────────────────────

    @property
    def prefix(self) -> str:
        return self.context.clean_prefix

    def get_command_signature(self, command: commands.Command, /) -> str:
        """`-play <url|search>` — the canonical form only. The base inlines aliases as
        `-[play|p|sing] …`; here each alias gets its own SYNOPSIS line, or joins the
        comma list heading a list entry, the way man pages write `-h, --help`.
        Command.signature returns the `usage=` kwarg verbatim when one is set.
        """
        return f"{self.prefix}{command.qualified_name} {command.signature}".strip()

    def _extras(self, command: commands.Command) -> dict[str, Any]:
        return command.extras or {}

    def _category(self, command: commands.Command) -> str:
        category = self._extras(command).get("category", UNCATEGORISED)
        return category if category in CATEGORY_ORDER else UNCATEGORISED

    def _rank(self, command: commands.Command) -> tuple[int, str]:
        """Sort key placing a command at its CATEGORY_COMMANDS position;
        commands missing from the ranking sink to the end, alphabetically."""
        order = CATEGORY_COMMANDS.get(self._category(command), ())
        try:
            return (order.index(command.qualified_name), command.qualified_name)
        except ValueError:
            return (len(order), command.qualified_name)

    def _forms(self, command: commands.Command) -> list[str]:
        """Every way to invoke the command, canonical name first."""
        return [
            f"{self.prefix}{name}"
            for name in (command.qualified_name, *command.aliases)
        ]

    def _entry_lines(self, command: commands.Command) -> list[str]:
        """One command as a hanging-indent entry, the way man(1) lists options:

            -play, -p, -sing <url|search>
                queue a song and start playing

        Overflow wraps rather than truncates, and a wrapped heading continues two spaces
        past the summary indent so the two can't be confused.
        """
        heading = f"{', '.join(self._forms(command))} {command.signature}".strip()
        summary = command.brief or command.short_doc or "no description"
        return textwrap.wrap(
            heading, _WIDTH, subsequent_indent=_INDENT + "  "
        ) + textwrap.wrap(
            summary, _WIDTH, initial_indent=_INDENT, subsequent_indent=_INDENT
        )

    def _add_entries_field(
        self, embed: discord.Embed, name: str, entries: Sequence[list[str]]
    ) -> None:
        """Add one section of entries (blank line between them), spilling into
        "(cont.)" fields rather than letting Discord reject a >1024-char value."""
        # The fences and their newlines count toward the value Discord measures.
        budget = _FIELD_LIMIT - (2 * len(_FENCE) + 2)

        def size(lines: Sequence[str]) -> int:
            return sum(len(line) + 1 for line in lines)

        field_name = name
        chunk: list[str] = []
        for lines in entries:
            spaced = lines if not chunk else ["", *lines]
            if chunk and size(chunk) + size(spaced) > budget:
                embed.add_field(name=field_name, value=self._fence(chunk), inline=False)
                field_name = f"{name} (cont.)"
                chunk = list(lines)
            else:
                chunk.extend(spaced)
        if chunk:
            embed.add_field(name=field_name, value=self._fence(chunk), inline=False)

    def _fence(self, lines: Sequence[str]) -> str:
        return f"{_FENCE}\n" + "\n".join(lines) + f"\n{_FENCE}"

    # ── dispatch ──────────────────────────────────────────────────────────────

    async def send_bot_help(
        self, mapping: Mapping[Optional[commands.Cog], list[commands.Command]], /
    ) -> None:
        prefix = self.prefix
        everything = [cmd for cmds in mapping.values() for cmd in cmds]
        visible = await self.filter_commands(everything, sort=True)

        embed = discord.Embed(
            title="MUSICBOT(1)",
            description=(
                "**musicbot** — plays YouTube, Spotify and SoundCloud audio "
                "in your voice channel"
            ),
            color=HELP_COLOR,
        )
        embed.add_field(
            name="SYNOPSIS",
            value=self._fence(
                [f"{prefix}<command> [argument ...]", f"{prefix}help [command]"]
            ),
            inline=False,
        )

        buckets: dict[str, list[commands.Command]] = {}
        for command in visible:
            buckets.setdefault(self._category(command), []).append(command)
        ordered = [c for c in CATEGORY_ORDER if c in buckets]
        ordered += [c for c in buckets if c not in CATEGORY_ORDER]
        for category in ordered:
            self._add_entries_field(
                embed,
                f"{category.upper()} COMMANDS",
                [
                    self._entry_lines(command)
                    for command in sorted(buckets[category], key=self._rank)
                ],
            )

        embed.add_field(name="SOURCES", value=SOURCES, inline=False)
        embed.add_field(name="NOTES", value=TIPS, inline=False)
        embed.set_footer(
            text=f"{len(visible)} commands · {prefix}help <command> for details"
        )
        await self.context.send(embed=embed)

    async def send_cog_help(self, cog: commands.Cog, /) -> None:
        # One cog holds every command, so `-help MusicBot` is just the full list.
        await self.send_bot_help(self.get_bot_mapping())

    async def send_command_help(self, command: commands.Command, /) -> None:
        prefix = self.prefix
        extras = self._extras(command)
        category = self._category(command)

        embed = discord.Embed(
            title=f"{prefix}{command.qualified_name}(1)",
            # The NAME section, as man(1) writes it: name — one-line summary.
            description=(
                f"**{command.qualified_name}** — "
                f"{command.brief or command.short_doc or 'no description'}"
            ),
            color=HELP_COLOR,
        )
        embed.add_field(
            name="SYNOPSIS",
            # One line per invocable form, aliases included, as a man SYNOPSIS.
            value=self._fence(
                [f"{form} {command.signature}".strip() for form in self._forms(command)]
            ),
            inline=False,
        )
        embed.add_field(
            name="DESCRIPTION",
            value=command.help or command.brief or "no description",
            inline=False,
        )
        examples: list[str] = extras.get("examples", [])
        if examples:
            embed.add_field(name="EXAMPLES", value=self._fence(examples), inline=False)
        note: Optional[str] = extras.get("note")
        if note:
            embed.add_field(name="NOTES", value=note, inline=False)
        embed.set_footer(text=f"{category} · {prefix}help for the full command list")
        await self.context.send(embed=embed)

    async def send_group_help(self, group: commands.Group, /) -> None:
        # No groups exist today; degrade to the single-command embed rather than
        # falling back to the base class's plaintext output.
        await self.send_command_help(group)

    async def send_error_message(self, error: str, /) -> None:
        await self.context.send(
            embed=notice_embed(
                f"{error}\nRun `{self.prefix}help` to see every command.",
                discord.Color.red(),
            )
        )

    def command_not_found(self, string: str, /) -> str:
        return f'No command called "{string}".'
