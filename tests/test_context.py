"""Tests for MusicContext (src/main.py) — the ctx.send override that keeps the
Now Playing embed block attached to the newest bot message while a song is live
(docs/NOW_PLAYING_EMBED_ATTACH_PLAN.md §3)."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands

from src.main import MusicContext
from src.musicbot import MusicBot


@pytest.fixture
def music_bot_cog(mock_bot):
    """Minimal real MusicBot instance — _np_player's isinstance check needs the
    actual class, not a MagicMock."""
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog.redis = None
    cog._active_spans = {}
    cog._alone_timers = {}
    cog._restore_tasks = set()
    return cog


@pytest.fixture
def mctx(mock_bot, mock_guild, mock_message):
    """MusicContext without discord.py's full Context construction — guild/
    channel/author are properties over .message, so setting message suffices."""
    ctx = object.__new__(MusicContext)
    ctx.bot = mock_bot
    mock_message.guild = mock_guild
    ctx.message = mock_message
    return ctx


@pytest.fixture
def live_mp(music_bot_cog, mock_bot, mock_guild, mock_channel):
    """A guild MusicPlayer (mocked) with a live song, wired into the cog lookup
    that MusicContext._np_player performs."""
    mock_bot.get_cog = MagicMock(return_value=music_bot_cog)
    mp = MagicMock()
    mp.current_song = MagicMock()
    mp._channel = mock_channel  # same channel object the ctx sends to
    mp.np_embed_block.return_value = [
        discord.Embed(title="NP"),
        discord.Embed(title="Up next"),
    ]
    music_bot_cog.mps[mock_guild.id] = mp
    return mp


def _parent_send(sent):
    return patch.object(commands.Context, "send", new=AsyncMock(return_value=sent))


class TestMusicContextAttach:
    async def test_block_leads_single_embed(self, mctx, live_mp):
        sent = MagicMock(spec=discord.Message)
        own = discord.Embed(title="Queue")
        with _parent_send(sent) as parent:
            message = await mctx.send(embed=own)

        embeds = parent.call_args.kwargs["embeds"]
        assert [e.title for e in embeds] == ["NP", "Up next", "Queue"]
        assert "embed" not in parent.call_args.kwargs  # folded into embeds
        live_mp._adopt_np_host.assert_called_once_with(sent, [own])
        assert message is sent

    async def test_block_leads_embeds_list(self, mctx, live_mp):
        sent = MagicMock(spec=discord.Message)
        own = [discord.Embed(title="A"), discord.Embed(title="B")]
        with _parent_send(sent) as parent:
            await mctx.send(embeds=own)

        embeds = parent.call_args.kwargs["embeds"]
        assert [e.title for e in embeds] == ["NP", "Up next", "A", "B"]
        live_mp._adopt_np_host.assert_called_once_with(sent, own)

    async def test_content_only_message_carries_block(self, mctx, live_mp):
        """Plain-text responses need no embed conversion — content and embeds
        coexist on one message (settled decision §4 of the plan)."""
        sent = MagicMock(spec=discord.Message)
        with _parent_send(sent) as parent:
            await mctx.send("shuffling...")

        args = parent.call_args
        assert args.args == ("shuffling...",)
        assert [e.title for e in args.kwargs["embeds"]] == ["NP", "Up next"]
        live_mp._adopt_np_host.assert_called_once_with(sent, [])

    async def test_skips_attach_when_embed_cap_would_be_exceeded(self, mctx, live_mp):
        """Defensive ≤10 guard: never expected to trip, but it must skip the
        attach rather than fail the send."""
        sent = MagicMock(spec=discord.Message)
        own = [discord.Embed(title=f"e{i}") for i in range(9)]
        with _parent_send(sent) as parent:
            await mctx.send(embeds=own)

        assert len(parent.call_args.kwargs["embeds"]) == 9
        live_mp._adopt_np_host.assert_not_called()


class TestMusicContextVanillaFallthrough:
    """Each no-attach guard falls through to a vanilla send, kwargs untouched."""

    @staticmethod
    async def _assert_vanilla(mctx, live_mp=None):
        sent = MagicMock(spec=discord.Message)
        own = discord.Embed(title="Queue")
        with _parent_send(sent) as parent:
            message = await mctx.send(embed=own)
        assert parent.call_args.kwargs["embed"] is own
        assert "embeds" not in parent.call_args.kwargs
        if live_mp is not None:
            live_mp._adopt_np_host.assert_not_called()
        return message

    async def test_dm_message(self, mctx, live_mp):
        mctx.message.guild = None
        await self._assert_vanilla(mctx, live_mp)

    async def test_cog_not_loaded(self, mctx, live_mp, mock_bot):
        mock_bot.get_cog = MagicMock(return_value=None)
        await self._assert_vanilla(mctx, live_mp)

    async def test_no_player_for_guild(self, mctx, live_mp, music_bot_cog):
        music_bot_cog.mps.clear()
        await self._assert_vanilla(mctx, live_mp)

    async def test_no_live_song(self, mctx, live_mp):
        live_mp.current_song = None
        await self._assert_vanilla(mctx, live_mp)

    async def test_different_channel_never_steals_host(self, mctx, live_mp):
        other_channel = MagicMock(spec=discord.TextChannel)
        live_mp._channel = other_channel  # distinct MagicMock → distinct .id
        await self._assert_vanilla(mctx, live_mp)
