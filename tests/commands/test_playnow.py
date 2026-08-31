"""Tests for `-playnow` (src/commands/playnow.py)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from src import play_pipeline
from src.musicbot import MusicBot
from src.musicplayer import InterjectOutcome
from src.youtube import QueueObject
from tests.helpers import (
    command_callback,
)


_ORIGIN = "https://yt.com/v=origin"


class TestPlaynow:
    @pytest.fixture
    def live_mp(self) -> MagicMock:
        """A MusicPlayer mock with a song currently playing."""
        from src.musicplayer import InterjectOutcome

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=False,
            )
        )
        return mp

    @pytest.fixture
    def live_vc(self) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return vc

    async def test_idle_delegates_to_play(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.invoke = AsyncMock()

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        mock_ctx.invoke.assert_awaited_once_with(music_bot.play, url="test")

    async def test_no_voice_client_delegates_to_play(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live_mp: MagicMock
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = None
        mock_ctx.invoke = AsyncMock()

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        mock_ctx.invoke.assert_awaited_once_with(music_bot.play, url="test")

    async def test_live_song_interjects(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.content = "-playnow test song"
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        assert qobj.interjected is True
        # The origin reaches the song through yt_source's required user_input, not
        # a post-hoc assignment — so with queue_source mocked out, assert it was
        # PASSED. A real yt_source stamps it (see test_youtube).
        origin_call = play_pipeline.queue_source.await_args
        assert origin_call is not None
        assert origin_call.kwargs["origin"] == "test"
        live_mp.interject.assert_awaited_once_with(qobj, live_vc, resume_paused=True)
        # Confirmation embed names both songs and the resume position.
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Urgent" in embed.title
        assert "Original Song" in embed.description
        assert "2:31" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_paused_wording(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """-playnow restores exactly what it interrupted, so a paused song is
        announced as returning paused. returns_paused is what the wording keys
        off — was_paused alone is the observed state and is also True on the
        -play path, where the song comes back playing."""
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=True,
                returns_paused=True,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "return paused" in embed.description

    async def test_near_end_wording(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The one outcome-wording branch with no coverage. Worth pinning now:
        re-keys these branches off
        returns_paused, and an unpinned branch could silently change text."""
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Almost Done",
                resume_position=None,
                was_paused=False,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Almost Done" in embed.description
        assert "nearly finished" in embed.description
        assert "will not resume" in embed.description

    async def test_interjecting_over_an_interjection_promises_a_return(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Interjections stack, so a song that was itself queued via -playnow gets
        the ordinary resume wording. This used to be its own branch announcing
        "it will not return" — the reply must never say that of a parked song."""
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Old Interjection",
                resume_position=151,
                was_paused=False,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Old Interjection" in embed.description
        assert "will resume at" in embed.description
        assert "will not return" not in embed.description

    async def test_interject_none_front_enqueues_with_confirmation(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Song ended mid-resolve: the resolved qobj is front-inserted directly, not
        by re-invoking -play, which would re-parse, re-resolve and (for playlists)
        enqueue every track right after the first-track-only notice. The user still
        gets a confirmation embed."""
        live_mp.interject = AsyncMock(return_value=None)
        live_mp.queue_put_front = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.invoke = AsyncMock()
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        mock_ctx.invoke.assert_not_awaited()
        # The player's wrapper, not queue.put_front directly — same plumbing as
        # every other insert; the stream was warmed, so it must not prefetch again.
        live_mp.queue_put_front.assert_awaited_once_with(qobj, prefetch=False)
        # interject() also returns None when the loop moved on to a DIFFERENT
        # song, which this insert waits behind: one, not the 0 an interjection
        # would have had, and not the queue depth — it goes to the front.
        assert qobj.analytics.queue_position == 1
        # The interjection marker must not leak onto a normally queued song —
        # a later -playnow would otherwise "replace" it without a resume entry.
        assert qobj.interjected is False
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Playing next" in embed.title
        assert "already ended" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_spotify_playlist_interjects_first_track_only(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        mock_ctx.message.content = f"-playnow {url}"
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(return_value=["First Song", "Second"])
        qobj = QueueObject("https://yt.com/v=first", "First Song", mock_ctx.author)

        with patch(
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=qobj)
        ) as ys:
            await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url=url)

        music_bot.spotify.playlist.assert_awaited_once_with("37i9dQZF1DXcBWIGoYBM5M")
        ys.assert_awaited_once()
        assert ys.call_args.args[1] == "ytsearch:First Song"
        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is qobj
        # First-track notice + confirmation.
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        assert any("first track" in d for d in notices)

    async def test_yt_playlist_interjects_first_track_only(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://www.youtube.com/playlist?list=PLtest123"
        mock_ctx.message.content = f"-playnow {url}"
        first = QueueObject("https://yt.com/v=1", "Track One", mock_ctx.author)
        second = QueueObject("https://yt.com/v=2", "Track Two", mock_ctx.author)

        with patch(
            "src.play_pipeline.YTDL.yt_playlist",
            new=AsyncMock(return_value=[first, second]),
        ):
            await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url=url)

        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is first

    async def test_error_shows_command_error(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        live_mp.interject.assert_not_awaited()
        mock_ctx.send.assert_awaited()  # error embed

    async def test_warms_stream_cache_before_interjecting(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The stream-URL cache is warmed before interject stops the current
        song — a cache miss at dequeue would otherwise put yt-dlp dead air
        between the interrupt and the playnow song starting."""
        from src.musicplayer import InterjectOutcome

        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        order: list[str] = []
        prefetch = AsyncMock(side_effect=lambda *a, **k: order.append("prefetch"))
        outcome = InterjectOutcome(
            interrupted_title="Original Song",
            resume_position=151,
            was_paused=False,
        )

        def _interject_effect(*args: Any, **kwargs: Any) -> InterjectOutcome:
            order.append("interject")
            return outcome

        live_mp.interject = AsyncMock(side_effect=_interject_effect)

        with patch("src.play_pipeline.YTDL.prefetch_stream", new=prefetch):
            await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        prefetch.assert_awaited_once_with(qobj, redis=music_bot.redis)
        assert order == ["prefetch", "interject"]
