"""Tests for MusicContext (src/main.py) — the ctx.send override that keeps the
Now Playing embed block attached to the newest bot message while a song is live
."""

import time
from contextlib import AbstractContextManager
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands
from opentelemetry import trace

from src.config import SpotifyStatus
from src.debug import DebugSettings
from src.main import MusicBotApp, MusicContext
from src.musicbot import MusicBot
from src.recovery import VoiceWatchdog
from tests.helpers import mocked


@pytest.fixture
def music_bot_cog(mock_bot: MagicMock) -> MusicBot:
    """Minimal real MusicBot instance — _np_player's isinstance check needs the
    actual class, not a MagicMock."""
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog._spotify_status = SpotifyStatus.ENABLED
    cog.redis = None
    cog._active_spans = {}
    cog.voice_watchdog = VoiceWatchdog(cog)
    cog._restore_tasks = set()
    # Off by default: send() asks the cog on every response, so a debug-on cog
    # here would decorate the embeds these attach tests compare.
    # Constructed, not hand-assembled: DebugSettings owns its own field set, so a
    # fixture that listed them would drift the moment one is added — which is
    # exactly how the cog's old __slots__ fell three attributes behind.
    cog.debug_settings = DebugSettings()
    cog.debug_settings._default = False
    return cog


@pytest.fixture
def mctx(
    mock_bot: MagicMock, mock_guild: MagicMock, mock_message: MagicMock
) -> MusicContext:
    """MusicContext without discord.py's full Context construction — guild/
    channel/author are properties over .message, so setting message suffices."""
    ctx = object.__new__(MusicContext)
    ctx.bot = mock_bot
    mock_message.guild = mock_guild
    ctx.message = mock_message
    return ctx


@pytest.fixture
def live_mp(
    music_bot_cog: MusicBot,
    mock_bot: MagicMock,
    mock_guild: MagicMock,
    mock_channel: MagicMock,
) -> MagicMock:
    """A guild MusicPlayer (mocked) with a live song, wired into the cog lookup
    that MusicContext._np_player performs."""
    mock_bot.get_cog = MagicMock(return_value=music_bot_cog)
    mp = MagicMock()
    mp.current_song = MagicMock()
    mp.home_channel = mock_channel  # same channel object the ctx sends to
    mp.np_embed_block.return_value = [
        discord.Embed(title="NP"),
        discord.Embed(title="Up next"),
    ]
    music_bot_cog.mps[mock_guild.id] = mp
    return mp


def _parent_send(sent: Any) -> AbstractContextManager[AsyncMock]:
    return patch.object(commands.Context, "send", new=AsyncMock(return_value=sent))


class TestMusicContextAttach:
    async def test_block_leads_single_embed(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        sent = MagicMock(spec=discord.Message)
        own = discord.Embed(title="Queue")
        with _parent_send(sent) as parent:
            message = await mctx.send(embed=own)

        embeds = parent.call_args.kwargs["embeds"]
        assert [e.title for e in embeds] == ["NP", "Up next", "Queue"]
        assert "embed" not in parent.call_args.kwargs  # folded into embeds
        live_mp._adopt_np_host_if_current.assert_called_once_with(
            sent, [own], live_mp.current_song
        )
        assert message is sent

    async def test_block_leads_embeds_list(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        sent = MagicMock(spec=discord.Message)
        own = [discord.Embed(title="A"), discord.Embed(title="B")]
        with _parent_send(sent) as parent:
            await mctx.send(embeds=own)

        embeds = parent.call_args.kwargs["embeds"]
        assert [e.title for e in embeds] == ["NP", "Up next", "A", "B"]
        live_mp._adopt_np_host_if_current.assert_called_once_with(
            sent, own, live_mp.current_song
        )

    async def test_content_only_message_carries_block(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        """Plain-text responses need no embed conversion — content and embeds
        coexist on one message."""
        sent = MagicMock(spec=discord.Message)
        with _parent_send(sent) as parent:
            await mctx.send("shuffling...")

        args = parent.call_args
        assert args.args == ("shuffling...",)
        assert [e.title for e in args.kwargs["embeds"]] == ["NP", "Up next"]
        live_mp._adopt_np_host_if_current.assert_called_once_with(
            sent, [], live_mp.current_song
        )

    async def test_skips_attach_when_embed_cap_would_be_exceeded(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        """Defensive ≤10 guard: never expected to trip, but it must skip the
        attach rather than fail the send."""
        sent = MagicMock(spec=discord.Message)
        own = [discord.Embed(title=f"e{i}") for i in range(9)]
        with _parent_send(sent) as parent:
            await mctx.send(embeds=own)

        assert len(parent.call_args.kwargs["embeds"]) == 9
        live_mp._adopt_np_host_if_current.assert_not_called()

    async def test_raises_on_embed_and_embeds_together(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        """Parity with discord.py's send() contract — mixing embed= and
        embeds= raises rather than being silently merged."""
        with _parent_send(MagicMock(spec=discord.Message)) as parent:
            with pytest.raises(TypeError):
                await mctx.send(
                    embed=discord.Embed(title="A"), embeds=[discord.Embed(title="B")]
                )
        parent.assert_not_awaited()

    async def test_content_only_send_without_block_omits_embeds(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        """Defensive else-branch: a live player whose block comes back empty
        falls through to a plain content send with no embeds kwarg."""
        live_mp.np_embed_block.return_value = []
        sent = MagicMock(spec=discord.Message)
        with _parent_send(sent) as parent:
            await mctx.send("ok")
        assert parent.call_args.args == ("ok",)
        assert "embeds" not in parent.call_args.kwargs
        live_mp._adopt_np_host_if_current.assert_not_called()


class TestMusicContextVanillaFallthrough:
    """Each no-attach guard falls through to a vanilla send, kwargs untouched."""

    @staticmethod
    async def _assert_vanilla(
        mctx: MusicContext, live_mp: Optional[MagicMock] = None
    ) -> discord.Message:
        sent = MagicMock(spec=discord.Message)
        own = discord.Embed(title="Queue")
        with _parent_send(sent) as parent:
            message = await mctx.send(embed=own)
        assert parent.call_args.kwargs["embed"] is own
        assert "embeds" not in parent.call_args.kwargs
        if live_mp is not None:
            live_mp._adopt_np_host_if_current.assert_not_called()
        return message

    async def test_dm_message(self, mctx: MusicContext, live_mp: MagicMock) -> None:
        mctx.message.guild = None
        await self._assert_vanilla(mctx, live_mp)

    async def test_cog_not_loaded(
        self, mctx: MusicContext, live_mp: MagicMock, mock_bot: MagicMock
    ) -> None:
        mock_bot.get_cog = MagicMock(return_value=None)
        await self._assert_vanilla(mctx, live_mp)

    async def test_no_player_for_guild(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        music_bot_cog.mps.clear()
        await self._assert_vanilla(mctx, live_mp)

    async def test_no_live_song(self, mctx: MusicContext, live_mp: MagicMock) -> None:
        live_mp.current_song = None
        await self._assert_vanilla(mctx, live_mp)

    async def test_different_channel_never_steals_host(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        other_channel = MagicMock(spec=discord.TextChannel)
        live_mp.home_channel = other_channel  # distinct MagicMock → distinct .id
        await self._assert_vanilla(mctx, live_mp)


class TestGetContextWiring:
    async def test_bot_builds_music_context(self) -> None:
        """MusicBotApp.get_context defaults cls to MusicContext — the wiring
        that routes every command response through the attach hook."""
        app = object.__new__(MusicBotApp)  # avoid full Bot construction
        origin = MagicMock(spec=discord.Message)
        built = MagicMock(spec=MusicContext)
        with patch.object(
            commands.AutoShardedBot, "get_context", new=AsyncMock(return_value=built)
        ) as parent:
            result = await app.get_context(origin)
        parent.assert_awaited_once_with(origin, cls=MusicContext)
        assert result is built


class TestDebugDecoration:
    """Debug mode's footer, applied at the one choke point every command response
    passes through. Off, nothing here happens at all — which is what the whole
    suite's undecorated embed assertions depend on."""

    @staticmethod
    def _enable(cog: MusicBot, guild_id: int) -> None:
        cog.debug_settings._overrides[guild_id] = True

    async def test_off_leaves_embeds_untouched(
        self, mctx: MusicContext, live_mp: MagicMock
    ) -> None:
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert own.footer.text is None

    async def test_decorates_on_the_np_attach_path(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)) as parent:
            await mctx.send(embed=own)
        assert own.footer.text is not None
        assert own.footer.text.startswith("🐞")
        # This layer decorates the response's OWN embeds only; the block arrives
        # already decorated from np_embed_block (stubbed here, so it arrives bare)
        # and passes through untouched. The real thing is asserted against a real
        # player in test_musicplayer.py::TestPlayerDebugDecoration.
        block = parent.call_args.kwargs["embeds"][:2]
        assert [e.footer.text for e in block] == [None, None]

    async def test_decorates_on_the_vanilla_path(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """The early return exists for every no-attach guard; a response that skips
        the NP block must still carry the footer."""
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        live_mp.current_song = None  # forces the vanilla fallthrough
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)) as parent:
            await mctx.send(embed=own)
        assert parent.call_args.kwargs["embed"] is own  # kwargs shape untouched
        assert (own.footer.text or "").startswith("🐞")

    async def test_resending_a_cached_embed_refreshes_rather_than_stacks(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """Decoration mutates in place and play_message is cached, so repeat -now
        sends the same object again. Each send replaces the suffix."""
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        live_mp.current_song = None
        cached = discord.Embed(title="Now playing: x")  # stands in for play_message
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=cached)
            await mctx.send(embed=cached)
        assert (cached.footer.text or "").count("🐞") == 1

    async def test_decorates_every_embed_of_a_multi_embed_response(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        own = [discord.Embed(title="A"), discord.Embed(title="B")]
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embeds=own)
        assert all((e.footer.text or "").startswith("🐞") for e in own)

    async def test_existing_footer_is_appended_to_not_replaced(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        own = discord.Embed(title="Board")
        own.set_footer(text="top 10 · last 30 days")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert (own.footer.text or "").startswith("top 10 · last 30 days · 🐞")

    async def test_shard_id_is_reported(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        mocked(mctx.guild).shard_id = 3
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert "shard 3" in (own.footer.text or "")

    async def test_content_only_response_needs_no_embed(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """Nothing to decorate is not an error — and this layer must not reach for
        the NP block standing in for the response (the player owns that)."""
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        with _parent_send(MagicMock(spec=discord.Message)) as parent:
            await mctx.send("shuffling...")
        assert all(e.footer.text is None for e in parent.call_args.kwargs["embeds"])

    async def test_no_active_command_omits_elapsed_and_trace(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """A send outside any command has no _active_spans entry. It still gets a
        footer — the shard is knowable — but nothing is fabricated."""
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        mocked(mctx.guild).shard_id = 0
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert own.footer.text == "🐞 shard 0"

    async def test_elapsed_is_measured_at_each_send(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """Two sends in one command show two increasing numbers: the footer times
        the phases of a command, not the command."""
        from src.musicbot import ActiveCommand

        self._enable(music_bot_cog, mocked(mctx.guild).id)
        span = trace.INVALID_SPAN
        music_bot_cog._active_spans[id(mctx)] = ActiveCommand(
            span=span, token=cast(Any, None), started=time.monotonic() - 0.5
        )
        first, second = discord.Embed(title="A"), discord.Embed(title="B")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=first)
            await mctx.send(embed=second)
        assert "ms" in (first.footer.text or "")
        elapsed = [
            int((e.footer.text or "").split(" ms")[0].split("🐞 ")[1])
            for e in (first, second)
        ]
        assert 500 <= elapsed[0] <= elapsed[1]

    async def test_trace_id_is_never_printed_twice(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """Error embeds already carry one from _command_error. Two copies of the
        same id in one footer read as two different traces."""
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        own = discord.Embed(title="Command failed")
        own.set_footer(text="trace: 0af7651916cd43dd8448eb211c80319c")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert (own.footer.text or "").count("0af7651916cd43dd8448eb211c80319c") == 1

    async def test_footer_is_truncated_to_discord_s_limit(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        self._enable(music_bot_cog, mocked(mctx.guild).id)
        own = discord.Embed(title="Long")
        own.set_footer(text="x" * 2100)
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert len(own.footer.text or "") == 2048

    async def test_per_guild_scope_holds_at_the_send_seam(
        self, mctx: MusicContext, live_mp: MagicMock, music_bot_cog: MusicBot
    ) -> None:
        """An enable in one server must not decorate another's replies."""
        self._enable(music_bot_cog, 4242424242)
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert own.footer.text is None

    async def test_no_cog_loaded_is_not_an_error(
        self, mctx: MusicContext, mock_bot: MagicMock
    ) -> None:
        mock_bot.get_cog = MagicMock(return_value=None)
        own = discord.Embed(title="Queue")
        with _parent_send(MagicMock(spec=discord.Message)):
            await mctx.send(embed=own)
        assert own.footer.text is None
