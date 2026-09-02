"""Tests for `-skip` (src/commands/skip.py)."""

from unittest.mock import AsyncMock, MagicMock

import discord

from src.musicbot import MusicBot
from src.musicplayer import MusicPlayer
from tests.helpers import (
    command_callback,
)


class TestSkipCommand:
    async def test_stops_voice_client_if_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        await command_callback(MusicBot.skip)(music_bot, mock_ctx)
        vc.stop.assert_called_once()

    async def test_marks_the_stop_as_deliberate_before_stopping(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A skip inside ffmpeg's startup window is byte for byte what a stream whose
        host never answered looks like; without the marker the player drops the cached
        URL of a perfectly good song. Marked BEFORE vc.stop(), which fires `after`
        immediately."""
        order: list[str] = []
        # spec'd: a bare MagicMock invents note_deliberate_stop, so renaming the real
        # method would leave this green while -skip silently stopped marking.
        mp = MagicMock(spec=MusicPlayer)
        mp.note_deliberate_stop = MagicMock(side_effect=lambda: order.append("mark"))
        music_bot.mps[mock_ctx.guild.id] = mp

        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock(side_effect=lambda: order.append("stop"))
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        assert order == ["mark", "stop"]

    async def test_skip_without_a_player_still_stops(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Read from mps, never get_mp(): constructing a player here would start a
        playback loop as a side effect of stopping a song."""
        music_bot.mps.pop(mock_ctx.guild.id, None)
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        vc.stop.assert_called_once()
        assert mock_ctx.guild.id not in music_bot.mps

    async def test_playing_skip_sends_no_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """An ordinary skip is self-evident — the music changes. Only the
        silent (paused) case earns a channel message."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        await command_callback(MusicBot.skip)(music_bot, mock_ctx)
        mock_ctx.send.assert_not_awaited()

    async def test_noop_when_neither_playing_nor_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock()
        mock_ctx.voice_client = vc
        await command_callback(MusicBot.skip)(music_bot, mock_ctx)
        vc.stop.assert_not_called()
        mock_ctx.send.assert_not_awaited()

    async def test_stops_voice_client_if_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """is_playing() is False while paused — gating on it alone made -skip a
        total no-op on a paused song."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        mp = MagicMock()
        mp.current_song = MagicMock(title="Paused Song", position_secs=83.4)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        vc.stop.assert_called_once()
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏭")

    async def test_paused_skip_sends_notice_naming_song_and_position(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A paused song makes no sound, so stopping it gives no audible cue
        that the command did anything."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        mp = MagicMock(spec=MusicPlayer)
        mp.current_song = MagicMock(title="Paused Song", position_secs=83.4)
        # Registered, not patched onto get_mp: skip reads the player it already has
        # and must never construct one.
        music_bot.mps[mock_ctx.guild.id] = mp

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Paused Song" in embed.description
        assert "1:23" in embed.description  # frozen position, not 83.4

    async def test_paused_skip_without_current_song_sends_no_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing to name — still stop, but don't invent a notice."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        mp = MagicMock()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        vc.stop.assert_called_once()
        mock_ctx.send.assert_not_awaited()

    async def test_noop_when_no_voice_client(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The isinstance guard: a None/partial voice client must not reach
        is_playing()/is_paused()."""
        mock_ctx.voice_client = None
        await command_callback(MusicBot.skip)(music_bot, mock_ctx)
        mock_ctx.send.assert_not_awaited()

    async def test_invoked_as_subcommand_suppresses_reaction(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """invoked_parents is non-empty when skip runs as part of another
        command — the reaction belongs to the parent's message, not ours."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock()
        mock_ctx.invoked_parents = ["parent"]
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        vc.stop.assert_called_once()
        mock_ctx.message.add_reaction.assert_not_awaited()

    async def test_reports_error_when_stop_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        vc.stop = MagicMock(side_effect=RuntimeError("voice gone"))
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        music_bot._command_error = AsyncMock()

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        music_bot._command_error.assert_awaited_once()

    async def test_captures_song_before_stopping(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The loop's song-end bookkeeping clears current_song, so the title
        must be read before stop() — reading after would name nothing."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_ctx.invoked_parents = []
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()

        mp = MagicMock(spec=MusicPlayer)
        mp.current_song = MagicMock(title="Paused Song", position_secs=83.4)
        music_bot.mps[mock_ctx.guild.id] = mp
        # Simulate the playback loop racing ahead the instant we stop.
        vc.stop = MagicMock(side_effect=lambda: setattr(mp, "current_song", None))

        await command_callback(MusicBot.skip)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Paused Song" in embed.description
