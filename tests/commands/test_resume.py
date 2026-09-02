"""Tests for `-resume` (src/commands/resume.py)."""

import contextlib
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import discord

from src.musicbot import MusicBot
from tests.helpers import (
    command_callback,
    no_typing,
    paused_vc,
    playing_vc,
)


class TestResumeCommand:
    async def test_resumes_when_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.resume = AsyncMock()
        mp.rehost_np_after_resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.resume)(music_bot, mock_ctx)
        mp.resume.assert_awaited_once_with(vc)

    async def test_rehosts_np_block_after_resume(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """If the -pause confirmation hosts the block, resume re-hosts it so
        "⏸️ Paused at…" becomes plain history instead of sitting beneath a
        live, advancing bar."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.resume = AsyncMock()
        mp.rehost_np_after_resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.resume)(music_bot, mock_ctx)
        mp.rehost_np_after_resume.assert_awaited_once()

    async def test_noop_when_not_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=False)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.resume)(music_bot, mock_ctx)
        mp.resume.assert_not_awaited()

    async def test_notice_when_already_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Silence was the old answer on every no-op branch; a reply has to say
        why nothing happened."""
        mock_ctx.voice_client = playing_vc()
        mp = MagicMock()
        mp.resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mp.resume.assert_not_awaited()
        sent = mock_ctx.send.await_args.kwargs["embed"]
        assert "Already playing" in sent.description

    async def test_nothing_paused_notice_gives_no_queue_advice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Exact match, not a substring: this branch also covers the seconds
        between two songs, so any "use -play to queue a song" advice added here
        would be telling a user with a full queue that it is empty."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = False
        vc.is_paused.return_value = False
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.resume = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mp.resume.assert_not_awaited()
        sent = mock_ctx.send.await_args.kwargs["embed"]
        assert sent.description == "Nothing is paused."

    @staticmethod
    def _cold_mp(embed: Optional[discord.Embed]) -> MagicMock:
        """MusicPlayer stand-in for -resume's disconnected path: the restore wait
        and the playback-gate hold both have to be enterable/awaitable.

        Every attribute the command BRANCHES on is set explicitly, for the reason
        conftest.mock_bot spells out — an auto-vivified one answers both `is not
        None` and `if x` with True, which would silently route every test down the
        Redis-down wording and the wedged-player rebuild."""
        mp = MagicMock()
        mp.store = MagicMock()
        mp.restore_read_failed = False
        mp.can_rejoin_cold = MagicMock(return_value=True)
        mp.playback_holds = 1  # the hold this command itself takes
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.defer_playback = MagicMock(return_value=contextlib.nullcontext())
        mp.build_rejoin_resume_embed = MagicMock(return_value=embed)
        # Explicitly awaitable: the teardown suppresses Exception, so a plain
        # MagicMock would fail its await and be swallowed as a pass.
        mp.repark_crashed_head = AsyncMock()
        return mp

    @staticmethod
    def _recording_hold(calls: list[str]) -> MagicMock:
        """defer_playback() stand-in that records its own entry and exit, so a
        test can assert what ran *inside* the hold. Asserting the mock was called
        proves only that the context manager was built — it stays green with the
        join and the send moved outside it, which is the whole regression."""

        class _Hold:
            async def __aenter__(self) -> None:
                calls.append("hold-enter")

            async def __aexit__(self, *_a: object) -> None:
                calls.append("hold-exit")

        return MagicMock(side_effect=lambda: _Hold())

    def _join_sets_voice_client(
        self, mock_ctx: MagicMock, calls: Optional[list[str]] = None
    ) -> AsyncMock:
        """ctx.invoke stub standing in for a join that succeeds — the real one
        leaves a voice client behind, which is what the command checks."""

        async def fake_invoke(*_a: Any, **_kw: Any) -> None:
            if calls is not None:
                calls.append("join")
            mock_ctx.voice_client = paused_vc()

        return AsyncMock(side_effect=fake_invoke)

    async def test_joins_when_disconnected(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The gap this closes: a -stop or an eject leaves the queue in Redis, and
        -resume used to answer a bot that was out of voice with silence."""
        mock_ctx.voice_client = None
        embed = discord.Embed(title="▶️ Resumed from queue")
        mp = self._cold_mp(embed)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = self._join_sets_voice_client(mock_ctx)

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mock_ctx.invoke.assert_awaited_once_with(music_bot.join)
        assert mock_ctx.send.await_args.kwargs["embed"] is embed

    async def test_waits_for_restore_before_reading_the_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The queue lives in Redis until restore replays it in memory, so a
        build before that wait would find it empty and report nothing to resume."""
        mock_ctx.voice_client = None
        calls: list[str] = []

        def build() -> discord.Embed:
            calls.append("build")
            return discord.Embed(title="▶️ Resumed from queue")

        mp = self._cold_mp(None)

        async def restored(**_kw: object) -> bool:
            calls.append("restore")
            return True

        mp.wait_for_restore = AsyncMock(side_effect=restored)
        mp.build_rejoin_resume_embed = MagicMock(side_effect=build)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = self._join_sets_voice_client(mock_ctx, calls)

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        # join last: the embed describes the queue head, and the head is gone
        # once the gate opens behind the join.
        assert calls == ["restore", "build", "join"]

    async def test_join_and_response_run_inside_the_playback_hold(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The hold is what stops the restored head from starting — and posting
        its own Now Playing card — before the response explaining the join.
        Asserting only that defer_playback() was called passes with both the join
        and the send moved outside the hold, which is exactly the regression."""
        mock_ctx.voice_client = None
        calls: list[str] = []
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        mp.defer_playback = self._recording_hold(calls)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = self._join_sets_voice_client(mock_ctx, calls)
        mock_ctx.send = AsyncMock(side_effect=lambda **_kw: calls.append("send"))

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        assert calls == ["hold-enter", "join", "send", "hold-exit"]

    async def test_reports_nothing_to_resume_without_joining(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing came back from Redis, so joining would park the bot in a
        channel to sit silent until the 300s idle disconnect."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(None)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mock_ctx.invoke.assert_not_awaited()
        sent = mock_ctx.send.await_args.kwargs["embed"]
        assert "Nothing to resume" in sent.description

    async def test_names_the_outage_instead_of_claiming_the_queue_is_gone(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """No store means restore never read anything, so the display is empty for
        a reason that says nothing about the queue — which is intact under its 24h
        TTL. "Nothing was left from a previous session" would assert what the bot
        could not know."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(None)
        mp.store = None
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mock_ctx.invoke.assert_not_awaited()
        sent = mock_ctx.send.await_args.kwargs["embed"]
        assert "Can't reach the queue store" in sent.description
        assert "no queue was left" not in sent.description

    async def test_cleans_up_when_the_join_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join swallows its own Exceptions, so an escape means its error
        REPORTING failed — a Forbidden out of ctx.send in a channel the bot cannot
        post embeds to. defer_playback opens the gate as it unwinds either way, and
        a loop woken with no voice client fails its vc assertion once per restored
        song."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock(side_effect=RuntimeError("send failed"))
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mp.repark_crashed_head.assert_awaited_once()

    async def test_cleans_up_when_the_join_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join swallows its own errors, so a failure shows up as a still-absent
        voice client. cog_before_invoke already started loop(), which would park
        on a gate nothing will open for 300s."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()  # leaves voice_client None
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_not_awaited()

    async def test_reparks_the_recovered_head_after_tearing_the_player_down(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A crash-recovered song lives only in the player restore built it in, so
        the teardown above is what would lose it. The order is the fix: cleanup()'s
        clear_connection() HDELs the fields the re-park writes, so a re-park that
        ran first would be wiped by the teardown it exists to survive."""
        mock_ctx.voice_client = None
        calls: list[str] = []
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        mp.repark_crashed_head = AsyncMock(side_effect=lambda: calls.append("repark"))
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()  # leaves voice_client None
        music_bot.cleanup = AsyncMock(side_effect=lambda _g: calls.append("cleanup"))

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        assert calls == ["cleanup", "repark"]

    async def test_leaves_the_player_alone_when_another_command_holds_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A second hold is a concurrent cold -play mid-join on this same player.
        Tearing it down pops the mps entry that command is still driving and drops
        the queue it is about to front-insert into."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        mp.playback_holds = 2  # this command's, plus the other command's
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()  # leaves voice_client None
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        music_bot.cleanup.assert_not_awaited()
        mp.repark_crashed_head.assert_not_awaited()

    async def test_a_registered_but_unconnected_voice_client_is_a_failed_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """discord.py registers the voice client on the guild BEFORE the handshake,
        so the type alone does not mean connected — and vc.play() on a half-open one
        raises once per restored song."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        half_open = MagicMock(spec=discord.VoiceClient)
        half_open.is_connected.return_value = False

        async def fake_invoke(*_a: Any, **_kw: Any) -> None:
            mock_ctx.voice_client = half_open

        mock_ctx.invoke = AsyncMock(side_effect=fake_invoke)
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_not_awaited()

    async def test_rebuilds_a_wedged_player_before_rejoining(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A player still holding a song with no voice client kept running after an
        eject that never reached on_voice_state_update. Rejoining around it would
        announce a resume its wedged loop is never going to deliver."""
        mock_ctx.voice_client = None
        wedged = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        wedged.can_rejoin_cold = MagicMock(return_value=False)
        rebuilt = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        music_bot.get_mp = MagicMock(side_effect=[wedged, rebuilt])
        music_bot.cleanup = AsyncMock()
        mock_ctx.invoke = self._join_sets_voice_client(mock_ctx)

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        # The rebuilt player is the one driven from there on, not the wedged one.
        rebuilt.build_rejoin_resume_embed.assert_called_once()
        wedged.build_rejoin_resume_embed.assert_not_called()

    async def test_reports_a_restore_that_has_not_landed_yet(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The pool sets no socket_timeout, so a stalled Redis leaves the restore
        pending forever. Waiting it out is a command that never answers at all."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(discord.Embed(title="▶️ Resumed from queue"))
        mp.wait_for_restore = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mock_ctx.invoke.assert_not_awaited()
        assert "Still loading" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_names_the_outage_when_the_restore_could_not_read(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A store that exists but could not be read leaves the same empty display
        as a guild with no saved queue. Reporting the second is telling a guild its
        queue is gone on the strength of a failed pipeline."""
        mock_ctx.voice_client = None
        mp = self._cold_mp(None)
        mp.restore_read_failed = True
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()

        with no_typing("src.commands.resume.background_typing"):
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mock_ctx.invoke.assert_not_awaited()
        sent = mock_ctx.send.await_args.kwargs["embed"]
        assert "Can't reach the queue store" in sent.description
        assert "no queue was left" not in sent.description
