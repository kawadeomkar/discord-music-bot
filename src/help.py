"""Man-page-styled embed help command.

discord.py ships two built-in implementations (DefaultHelpCommand,
MinimalHelpCommand); both format through a Paginator into plain text, so an
embed-only bot has to subclass HelpCommand directly. Only the dispatch methods
are overridden — command_callback still does the resolution, which means
`-help p` (an alias) and `-help MusicBot` (the cog) keep working for free.

The layout borrows from man(1): caps section headers (NAME, SYNOPSIS,
DESCRIPTION, EXAMPLES, NOTES) and a command list of hanging-indent entries —
every form of a command on one line, its summary indented beneath — rather
than a column-aligned table, whose grid read poorly at Discord widths.

Per-command copy lives on the commands themselves (brief/help/usage/extras in
src/musicbot.py) rather than in a table here, so a new command shows up in the
help output as soon as it is declared.
"""

import textwrap
from typing import Any, List, Mapping, Optional, Sequence

import discord
from discord.ext import commands

from src.util import notice_embed

HELP_COLOR = discord.Color.blurple()

# The command list, in display order: categories as rendered, and within each
# category the commands by importance/frequency of use — the daily verbs first,
# housekeeping last — not alphabetically, which put `pause` above `play`.
CATEGORY_COMMANDS: dict[str, tuple[str, ...]] = {
    "Playback": ("play", "playnow", "pause", "resume", "skip", "stop", "volume"),
    "Queue": ("queue", "now", "history", "stats", "shuffle", "remove", "clear", "jump"),
    "Utility": ("help", "join", "ping"),
}
CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORY_COMMANDS)
UNCATEGORISED = "Other"

# Discord's hard cap on an embed field value.
_FIELD_LIMIT = 1024

# Command entries live in code blocks — the only construct Discord renders in
# a monospace grid, and so the only place a hanging indent survives. But
# Discord soft-wraps code blocks at the embed's rendered width (roughly 54
# monospace characters on desktop, less on mobile), and its wrap restarts at
# column 0 — destroying the indent. Hard-wrapping narrower than any common
# embed width keeps the wrapping ours, so continuations stay indented.
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
        """`-play <url|search>` — the canonical form only.

        The base implementation inlines aliases into the name as
        `-[play|p|sing] …`; here each alias is instead its own SYNOPSIS line
        (per-command help) or joins the comma list heading the command's list
        entry, the way man pages write `-h, --help`. Command.signature returns
        the `usage=` kwarg verbatim when one is set.
        """
        return f"{self.prefix}{command.qualified_name} {command.signature}".strip()

    def _extras(self, command: commands.Command) -> dict:
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

    def _forms(self, command: commands.Command) -> List[str]:
        """Every way to invoke the command, canonical name first."""
        return [
            f"{self.prefix}{name}"
            for name in (command.qualified_name, *command.aliases)
        ]

    def _entry_lines(self, command: commands.Command) -> List[str]:
        """One command as a hanging-indent entry, the way man(1) lists options:

            -play, -p, -sing <url|search>
                queue a song and start playing

        Cells that overflow the width budget wrap rather than truncate — an
        entry may grow a line, but it never silently loses text. A wrapped
        heading continues two spaces past the summary indent so the two can't
        be mistaken for each other.
        """
        heading = f"{', '.join(self._forms(command))} {command.signature}".strip()
        summary = command.brief or command.short_doc or "no description"
        return textwrap.wrap(
            heading, _WIDTH, subsequent_indent=_INDENT + "  "
        ) + textwrap.wrap(
            summary, _WIDTH, initial_indent=_INDENT, subsequent_indent=_INDENT
        )

    def _add_entries_field(
        self, embed: discord.Embed, name: str, entries: Sequence[List[str]]
    ) -> None:
        """Add one section of entries (blank line between them), continuing into
        "(cont.)" fields rather than letting Discord reject an over-long value
        (>1024 chars)."""
        # The fences and their newlines are part of the field value Discord measures.
        budget = _FIELD_LIMIT - (2 * len(_FENCE) + 2)

        def size(lines: Sequence[str]) -> int:
            return sum(len(line) + 1 for line in lines)

        field_name = name
        chunk: List[str] = []
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
        self, mapping: Mapping[Optional[commands.Cog], List[commands.Command]], /
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

        buckets: dict[str, List[commands.Command]] = {}
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
        await self.get_destination().send(embed=embed)

    async def send_cog_help(self, cog: commands.Cog, /) -> None:
        # Every command lives in the single MusicBot cog, so `-help MusicBot` is
        # just the full list — no separate, near-identical rendering.
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
            # One line per invocable form, aliases included — how a man page's
            # SYNOPSIS lists every spelling of a command.
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
        examples: List[str] = extras.get("examples", [])
        if examples:
            embed.add_field(name="EXAMPLES", value=self._fence(examples), inline=False)
        note: Optional[str] = extras.get("note")
        if note:
            embed.add_field(name="NOTES", value=note, inline=False)
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
