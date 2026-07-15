"""Tests for src/help.py — MusicHelpCommand man-page-styled embed rendering."""

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from src.help import (
    CATEGORY_COMMANDS,
    CATEGORY_ORDER,
    MusicHelpCommand,
    _WIDTH as WIDTH,
)
from src.musicbot import MusicBot

# Discord's hard caps: an embed field value is 1024 chars, a description 4096.
FIELD_LIMIT = 1024
DESCRIPTION_LIMIT = 4096


@pytest.fixture
async def bot():
    """A real Bot with the real cog, so help reflects the actual command table."""
    instance = commands.Bot(
        command_prefix="-",
        intents=discord.Intents.none(),
        help_command=MusicHelpCommand(),
    )
    await instance.add_cog(MusicBot(instance))
    return instance


@pytest.fixture
def ctx(bot):
    """Stub context that captures what the help command sends."""
    context = MagicMock()
    context.bot = bot
    context.clean_prefix = "-"
    context.guild = None
    context.command = None
    context.send = AsyncMock()
    return context


@pytest.fixture
def help_command(bot, ctx):
    # copy() is what discord.py does per invocation (issue #2123).
    hc = bot.help_command.copy()
    hc.context = ctx
    return hc


def sent_embed(ctx) -> discord.Embed:
    ctx.send.assert_called_once()
    return ctx.send.call_args.kwargs["embed"]


class TestGetDestination:
    def test_returns_context_not_channel(self, help_command, ctx):
        """Must route through MusicContext.send so the Now Playing block stays
        glued to the bottom of the channel — context.channel would bury it."""
        assert help_command.get_destination() is ctx


class TestBotHelp:
    async def test_lists_every_visible_command(self, help_command, ctx, bot):
        await help_command.command_callback(ctx, command=None)
        embed = sent_embed(ctx)
        body = "\n".join(f.value or "" for f in embed.fields)
        for command in bot.commands:
            assert f"-{command.name}" in body

    async def test_shows_every_alias(self, help_command, ctx, bot):
        await help_command.command_callback(ctx, command=None)
        body = "\n".join(f.value or "" for f in sent_embed(ctx).fields)
        for command in bot.commands:
            for alias in command.aliases:
                assert f"-{alias}" in body, f"alias {alias} of {command.name} missing"

    async def test_documents_sources_and_behaviour(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        body = "\n".join(f.value or "" for f in sent_embed(ctx).fields)
        for topic in ("YouTube", "Spotify", "SoundCloud", "Now Playing"):
            assert topic in body

    async def test_respects_discord_size_limits(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        embed = sent_embed(ctx)
        assert len(embed.description or "") <= DESCRIPTION_LIMIT
        for field in embed.fields:
            assert len(field.value or "") <= FIELD_LIMIT
        assert len(embed) <= 6000


class TestCommandList:
    """The command list renders the way man(1) renders OPTIONS: a hanging-indent
    entry per command — every invocable form on the heading line, the summary
    indented beneath — inside a code block, the only construct Discord renders
    in monospace so the indent survives."""

    def _sections(self, ctx) -> dict[str, list[str]]:
        """The code-block body of each *COMMANDS field, as lines."""
        sections = {}
        for field in sent_embed(ctx).fields:
            value = field.value or ""
            if "COMMANDS" in (field.name or "") and value.startswith("```"):
                sections[field.name or ""] = value.strip("`").strip("\n").splitlines()
        return sections

    async def test_synopsis_comes_first(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        fields = sent_embed(ctx).fields
        assert fields[0].name == "SYNOPSIS"
        assert "-<command> [argument ...]" in (fields[0].value or "")

    async def test_one_section_per_category_in_order(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        names = list(self._sections(ctx))
        assert names == [f"{category.upper()} COMMANDS" for category in CATEGORY_ORDER]

    async def test_heading_lines_list_every_form_of_the_command(
        self, help_command, ctx
    ):
        """Aliases join the heading comma-list the way man writes `-h, --help`,
        and arguments survive unwrapped — not discord.py's `-[play|p|sing]`."""
        await help_command.command_callback(ctx, command=None)
        body = "\n".join("\n".join(lines) for lines in self._sections(ctx).values())
        assert "-play, -p, -sing <url|search>" in body
        assert "-volume, -v, -vol, -sound <0-100>" in body
        assert "-now, -np, -rn, -nowplaying" in body
        assert "[play|p|sing]" not in body

    async def test_entries_follow_importance_order(self, help_command, ctx):
        """Within a category, commands render by importance/frequency of use
        (play before playnow before pause…), not alphabetically — which put
        `pause` above `play`."""
        await help_command.command_callback(ctx, command=None)
        for name, lines in self._sections(ctx).items():
            category = name.removesuffix(" COMMANDS").capitalize()
            rendered = [
                line.split(",")[0].split()[0].lstrip("-")
                for line in lines
                if line.startswith("-")
            ]
            expected = [c for c in CATEGORY_COMMANDS[category] if c in rendered]
            assert rendered == expected, f"{name}: {rendered}"

    async def test_entries_hang_indent(self, help_command, ctx):
        """Heading lines sit at column 0 and start with the prefix; summary
        lines are indented — reading the left edge scans the command names."""
        await help_command.command_callback(ctx, command=None)
        for lines in self._sections(ctx).values():
            for line in lines:
                if not line:
                    continue  # blank separator between entries
                assert line.startswith("-") or line.startswith("    "), repr(line)

    async def test_no_line_exceeds_the_width_budget(self, help_command, ctx):
        """A code block does not soft-wrap — an over-long line scrolls sideways
        on mobile instead. Cells wrap so that never happens."""
        await help_command.command_callback(ctx, command=None)
        for lines in self._sections(ctx).values():
            for line in lines:
                assert len(line) <= WIDTH, f"{len(line)} chars: {line!r}"

    async def test_wrapping_never_drops_text(self, help_command, ctx, bot):
        """Summaries wrap rather than truncate: reading the indented lines top
        to bottom reassembles every brief in full."""
        await help_command.command_callback(ctx, command=None)
        indented = []
        for lines in self._sections(ctx).values():
            indented += [line.strip() for line in lines if line.startswith("    ")]
        reassembled = " ".join(" ".join(indented).split())

        assert "…" not in reassembled and "..." not in reassembled
        for command in bot.commands:
            assert command.brief in reassembled, f"{command.name}'s brief was mangled"

    async def test_cog_help_renders_the_full_list(self, help_command, ctx):
        await help_command.command_callback(ctx, command="MusicBot")
        assert sent_embed(ctx).title == "MUSICBOT(1)"


class TestCommandHelp:
    async def test_renders_man_sections(self, help_command, ctx):
        await help_command.command_callback(ctx, command="play")
        embed = sent_embed(ctx)
        assert embed.title == "-play(1)"
        # NAME, as man writes it: name — one-line summary.
        assert (embed.description or "").startswith("**play** — ")
        fields = {f.name: f.value or "" for f in embed.fields}
        assert "SoundCloud" in fields["DESCRIPTION"]
        assert "-play never gonna give you up" in fields["EXAMPLES"]

    async def test_synopsis_lists_every_form(self, help_command, ctx):
        """One line per spelling, aliases included — man-page SYNOPSIS style,
        instead of a separate Aliases blurb the reader has to recombine with
        the usage line themselves."""
        await help_command.command_callback(ctx, command="play")
        fields = {f.name: f.value or "" for f in sent_embed(ctx).fields}
        lines = fields["SYNOPSIS"].strip("`").strip("\n").splitlines()
        assert lines == [
            "-play <url|search>",
            "-p <url|search>",
            "-sing <url|search>",
        ]

    async def test_resolves_an_alias_to_its_command(self, help_command, ctx):
        await help_command.command_callback(ctx, command="np")
        assert sent_embed(ctx).title == "-now(1)"

    async def test_command_without_aliases_has_single_synopsis_line(
        self, help_command, ctx
    ):
        await help_command.command_callback(ctx, command="shuffle")
        fields = {f.name: f.value or "" for f in sent_embed(ctx).fields}
        assert fields["SYNOPSIS"].strip("`").strip("\n").splitlines() == ["-shuffle"]

    async def test_every_command_has_help_metadata(self, bot):
        """A new command must not silently land in the help output bare — and
        its long help has to fit the DESCRIPTION field's 1024-char cap."""
        for command in bot.commands:
            assert command.brief, f"{command.name} is missing brief="
            assert command.help, f"{command.name} is missing help="
            assert len(command.help) <= FIELD_LIMIT, f"{command.name} help too long"
            category = (command.extras or {}).get("category")
            assert category in CATEGORY_ORDER, f"{command.name} category={category!r}"
            # …and must be placed in its category's importance ranking.
            assert (
                command.name in CATEGORY_COMMANDS[category]
            ), f"{command.name} missing from CATEGORY_COMMANDS[{category!r}]"
            note = (command.extras or {}).get("note")
            assert note is None or len(note) <= FIELD_LIMIT


class TestErrors:
    async def test_unknown_command_sends_red_embed(self, help_command, ctx):
        await help_command.command_callback(ctx, command="bogus")
        embed = sent_embed(ctx)
        assert embed.color == discord.Color.red()
        assert "bogus" in (embed.description or "")


class TestHelpFlagEndToEnd:
    async def test_play_dash_dash_help_renders_the_play_man_page(self):
        """`-play --help` through the real MusicBotApp.invoke lands on the same
        embed as `-help play` — the flag diverts before argument parsing, so
        the extra words never reach the play command."""
        from discord.ext.commands.view import StringView

        from src.main import MusicBotApp, MusicContext

        app = MusicBotApp()
        await app.add_cog(MusicBot(app))
        message = MagicMock()
        message.content = "-play lofi hip hop --help"
        context = MusicContext(
            prefix="-",
            view=StringView(message.content),
            bot=app,
            message=message,
            invoked_with="play",
            command=app.all_commands["play"],
        )
        context.send = AsyncMock()

        await app.invoke(context)

        embed = context.send.call_args.kwargs["embed"]
        assert embed.title == "-play(1)"
