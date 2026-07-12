"""Tests for src/help.py — MusicHelpCommand embed rendering."""

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from src.help import CATEGORY_ORDER, MusicHelpCommand
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

    async def test_shows_aliases(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        body = "\n".join(f.value or "" for f in sent_embed(ctx).fields)
        for alias in ("p", "sing", "sk", "np", "nowplaying", "rm", "vol", "summon"):
            assert f"`{alias}`" in body

    async def test_signature_shows_arguments(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        body = "\n".join(f.value or "" for f in sent_embed(ctx).fields)
        assert "-play <url or search terms>" in body
        assert "-volume <0-100>" in body
        # Aliases are their own column; they must not be inlined into the name
        # the way discord.py's base get_command_signature does it.
        assert "[play|p|sing]" not in body

    async def test_categories_render_in_order(self, help_command, ctx):
        await help_command.command_callback(ctx, command=None)
        names = [f.name or "" for f in sent_embed(ctx).fields]
        positions = [
            next(i for i, n in enumerate(names) if category in n)
            for category in CATEGORY_ORDER
        ]
        assert positions == sorted(positions)

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

    async def test_cog_help_renders_the_full_list(self, help_command, ctx):
        await help_command.command_callback(ctx, command="MusicBot")
        assert "command reference" in (sent_embed(ctx).title or "")


class TestCommandHelp:
    async def test_renders_long_description_and_examples(self, help_command, ctx):
        await help_command.command_callback(ctx, command="play")
        embed = sent_embed(ctx)
        assert embed.title == "▶️ -play"
        assert "SoundCloud" in (embed.description or "")
        fields = {f.name: f.value or "" for f in embed.fields}
        assert fields["Usage"] == "`-play <url or search terms>`"
        assert "`-p`" in fields["Aliases"] and "`-sing`" in fields["Aliases"]
        assert "-play never gonna give you up" in fields["Examples"]

    async def test_resolves_an_alias_to_its_command(self, help_command, ctx):
        await help_command.command_callback(ctx, command="np")
        assert sent_embed(ctx).title == "🎶 -now"

    async def test_omits_alias_field_for_command_without_aliases(
        self, help_command, ctx
    ):
        await help_command.command_callback(ctx, command="shuffle")
        names = [f.name for f in sent_embed(ctx).fields]
        assert "Aliases" not in names

    async def test_every_command_has_help_metadata(self, bot):
        """A new command must not silently land in the help output bare."""
        for command in bot.commands:
            assert command.brief, f"{command.name} is missing brief="
            assert command.help, f"{command.name} is missing help="
            category = (command.extras or {}).get("category")
            assert category in CATEGORY_ORDER, f"{command.name} category={category!r}"


class TestErrors:
    async def test_unknown_command_sends_red_embed(self, help_command, ctx):
        await help_command.command_callback(ctx, command="bogus")
        embed = sent_embed(ctx)
        assert embed.color == discord.Color.red()
        assert "bogus" in (embed.description or "")
