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

import textwrap
from typing import Any, List, Mapping, Optional, Sequence, Tuple

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

# The command list is a monospace table. Discord has no table primitive, so columns
# can only be aligned inside a code block — and a code block does not soft-wrap: past
# roughly 60 characters it scrolls sideways on mobile. Hence the width budget, and
# hence the deliberately terse usage= strings on the commands (`<url|search>`, not
# `<url or search terms>`): one long signature would otherwise set the column width
# for every row. Cells that still overflow are wrapped rather than truncated — the
# table may grow a line, but it never silently loses text.
_TABLE_WIDTH = 62
_MIN_DESC_WIDTH = 20
# Aliases are capped rather than measured: one long alias list (`np, rn, nowplaying`)
# would otherwise claim a third of the table and starve the description column into
# wrapping every row. Capped, it wraps instead — and it is the least important column.
_MAX_ALIAS_WIDTH = 13
_GUTTER = "  "
_HEADERS: Tuple[str, str, str] = ("Command", "Aliases", "Description")
_FENCE = "```"

# A row's cell triple: signature, aliases, one-line summary.
Row = Tuple[str, str, str]
# Column widths: command, aliases, description.
Widths = Tuple[int, int, int]

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

    def _row(self, command: commands.Command) -> Row:
        return (
            self.get_command_signature(command).strip(),
            ", ".join(command.aliases),
            command.brief or command.short_doc or "no description",
        )

    def _widths(self, rows: Sequence[Row]) -> Widths:
        """Column widths, measured across *every* command rather than per category,
        so the separate category tables share one grid and read as a single table."""
        command_width = max(len(_HEADERS[0]), *(len(row[0]) for row in rows))
        alias_width = min(
            _MAX_ALIAS_WIDTH,
            max(len(_HEADERS[1]), *(len(row[1]) for row in rows)),
        )
        description_width = max(
            _MIN_DESC_WIDTH,
            _TABLE_WIDTH - command_width - alias_width - 2 * len(_GUTTER),
        )
        return command_width, alias_width, description_width

    def _line(self, cells: Sequence[str], widths: Widths) -> str:
        command, alias, description = cells
        return (
            f"{command:<{widths[0]}}{_GUTTER}{alias:<{widths[1]}}{_GUTTER}{description}"
        ).rstrip()

    def _head(self, widths: Widths) -> List[str]:
        rule = "-" * (widths[0] + widths[1] + widths[2] + 2 * len(_GUTTER))
        return [self._line(_HEADERS, widths), rule]

    def _row_lines(self, row: Row, widths: Widths) -> List[str]:
        """One command's row — several physical lines when a cell has to wrap."""
        columns = [
            textwrap.wrap(cell, width) or [""] for cell, width in zip(row, widths)
        ]
        height = max(len(column) for column in columns)
        return [
            self._line(
                [column[i] if i < len(column) else "" for column in columns], widths
            )
            for i in range(height)
        ]

    def _add_table_field(
        self, embed: discord.Embed, category: str, rows: Sequence[Row], widths: Widths
    ) -> None:
        """Add one category's table, continuing into further fields rather than
        letting Discord reject an over-long value (>1024 chars). Each continuation
        repeats the header, so a split table still has labelled columns."""
        emoji = CATEGORY_EMOJI.get(category, CATEGORY_EMOJI[UNCATEGORISED])
        name = f"{emoji} {category}"
        head = self._head(widths)
        # The fences and their newlines are part of the field value Discord measures.
        budget = _FIELD_LIMIT - (2 * len(_FENCE) + 2)

        def size(lines: Sequence[str]) -> int:
            return sum(len(line) + 1 for line in lines)

        chunk = list(head)
        for row in rows:
            lines = self._row_lines(row, widths)
            if len(chunk) > len(head) and size(chunk) + size(lines) > budget:
                embed.add_field(name=name, value=self._fence(chunk), inline=False)
                name = f"{emoji} {category} (cont.)"
                chunk = list(head)
            chunk.extend(lines)
        if len(chunk) > len(head):
            embed.add_field(name=name, value=self._fence(chunk), inline=False)

    def _fence(self, lines: Sequence[str]) -> str:
        return f"{_FENCE}\n" + "\n".join(lines) + f"\n{_FENCE}"

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

        buckets: dict[str, List[Row]] = {}
        for command in visible:
            buckets.setdefault(self._category(command), []).append(self._row(command))

        if buckets:
            widths = self._widths([row for rows in buckets.values() for row in rows])
            ordered = [c for c in CATEGORY_ORDER if c in buckets]
            ordered += [c for c in buckets if c not in CATEGORY_ORDER]
            for category in ordered:
                self._add_table_field(embed, category, buckets[category], widths)

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
