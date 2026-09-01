"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""

from src.musicplayer import MusicPlayer
import redis.asyncio as aioredis
import asyncio
import inspect
import orjson
from typing import Any, Optional, cast
from collections.abc import Coroutine, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from tests.helpers import (
    connected_vc,
    mock_mp,
    no_typing,
    paused_vc,
    playing_vc,
)
from discord.ext import commands

import src.debug as debug_mode
from src.config import SpotifyStatus
from src.guild_state import Analytics, HistoryEntry
from src.musicbot import (
    RESTORE_WAIT_SECS,
    EmptyPlaylistError,
    MusicBot,
    PlaylistIndexError,
    ResolvedSpotifyPlaylist,
    ResolvedYoutubePlaylist,
    SpotifyDisabledError,
    _check_voice_permissions,
)
from src.sources import (
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    parse_url,
    timestamp_warning,
)
from src.musicplayer import InterjectOutcome
from src.spotify import SpotifyAuthError
from src.youtube import YTDL, QueueObject
from tests.helpers import (
    command_callback,
    make_mock_task,
    queue_object,
)

# Ask-time analytics for direct queue_source/_enqueue_playlist calls — the real
# command paths mint this at dispatch from ctx.message.created_at + enqueue_depth.
_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=0)
_ORIGIN = "https://yt.com/v=origin"


class TestCommandErrorRendering:
    """_command_error must not leak yt-dlp's raw error text to the user."""

    async def test_extraction_error_renders_its_user_message(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        from src.youtube import ExtractionError

        err = ExtractionError(
            "ERROR: [youtube] v9: Video unavailable",
            original_type="DownloadError",
            expected=True,
        )
        with (
            patch("src.musicbot.send_embed", new=AsyncMock()) as send_embed,
            patch("src.musicbot.record_span_error"),
        ):
            await music_bot._command_error(mock_ctx, err)

        assert (call := send_embed.await_args) is not None
        detail = call.args[2]
        assert detail == "[youtube] v9: Video unavailable"
        assert "ExtractionError" not in detail  # not the raw type: message form

    async def test_unexpected_extraction_error_is_generic(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        from src.youtube import ExtractionError

        err = ExtractionError(
            "ERROR: boom; please report this issue on https://github.com/yt-dlp/yt-dlp",
            expected=False,
        )
        with (
            patch("src.musicbot.send_embed", new=AsyncMock()) as send_embed,
            patch("src.musicbot.record_span_error"),
        ):
            await music_bot._command_error(mock_ctx, err)

        assert (call := send_embed.await_args) is not None
        detail = call.args[2]
        assert "github.com" not in detail
        assert "unexpected error" in detail

    async def test_a_plain_exception_still_renders_type_and_message(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        with (
            patch("src.musicbot.send_embed", new=AsyncMock()) as send_embed,
            patch("src.musicbot.record_span_error"),
        ):
            await music_bot._command_error(mock_ctx, ValueError("nope"))

        assert (call := send_embed.await_args) is not None
        assert call.args[2] == "**ValueError:** nope"


class TestCheckVoicePermissions:
    def test_rejects_non_member_user(self) -> None:
        user = MagicMock(spec=discord.User)
        assert _check_voice_permissions(user, None, "play") is not None

    def test_rejects_member_not_in_voice_channel(self) -> None:
        member = MagicMock(spec=discord.Member)
        member.voice = None
        assert _check_voice_permissions(member, None, "play") is not None

    def test_rejects_wrong_voice_channel_for_non_play(self) -> None:
        member = MagicMock(spec=discord.Member)
        channel_a = MagicMock()
        channel_b = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel_a
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel_b
        assert _check_voice_permissions(member, vc, "skip") is not None

    def test_allows_play_in_different_channel(self) -> None:
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()  # different from member's channel — OK for play
        assert _check_voice_permissions(member, vc, "play") is None

    def test_passes_valid_member_in_correct_channel(self) -> None:
        member = MagicMock(spec=discord.Member)
        channel = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel
        assert _check_voice_permissions(member, vc, "skip") is None

    def test_passes_when_no_voice_client(self) -> None:
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        assert _check_voice_permissions(member, None, "skip") is None


class TestQueueSource:
    async def test_spotify_playlist_returns_list(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(return_value=["Song A", "Song B"])
        result = await music_bot.queue_source(
            mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert result == ResolvedSpotifyPlaylist(titles=["Song A", "Song B"])

    async def test_spotify_track_calls_yt_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        fake_qobj = QueueObject("https://yt.com/v=1", "My Track", mock_ctx.author)
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.track = AsyncMock(return_value="My Track Artist")
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, QueueObject)

    async def test_youtube_url_calls_yt_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(url="https://yt.com/watch?v=abc", process=False)
        fake_qobj = QueueObject(
            "https://yt.com/watch?v=abc", "YT Song", mock_ctx.author
        )
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, QueueObject)


class TestQuerySourceClassification:
    """Every path from parsed input to an enqueueable object hands the token to
    the extraction call, which REQUIRES it. yt_source cannot derive it: it is
    given a search string, and for Spotify that string is a YouTube title query
    indistinguishable from a plaintext search."""

    @staticmethod
    def _passed_query_source(spy: AsyncMock) -> str:
        assert spy.await_args is not None
        return spy.await_args.kwargs["query_source"]

    async def test_spotify_track(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        fake_qobj = QueueObject("https://yt.com/v=1", "My Track", mock_ctx.author)
        assert music_bot.spotify is not None
        music_bot.spotify.track = AsyncMock(return_value="My Track Artist")
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert self._passed_query_source(spy) == "spotify.com"

    async def test_plaintext_search(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = parse_input("never gonna give you up", "-play never gonna give you up")
        fake_qobj = QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert self._passed_query_source(spy) == "search"

    async def test_generic_host_link(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        url = "https://www.tiktok.com/@user/video/1234567890"
        source = parse_input(url, f"-play {url}")
        fake_qobj = QueueObject(url, "Clip", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert self._passed_query_source(spy) == "tiktok.com"

    async def test_youtube_playlist_classifies_every_track(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # One token for the whole playlist: yt_playlist stamps it onto each
        # QueueObject it builds, so the call carries it once.
        url = "https://www.youtube.com/playlist?list=PLabc"
        source = parse_input(url, f"-play {url}")
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"T{i}", mock_ctx.author)
            for i in range(3)
        ]
        spy = AsyncMock(return_value=tracks)
        with patch("src.musicbot.YTDL.yt_playlist", new=spy):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert self._passed_query_source(spy) == "youtube.com"

    @staticmethod
    def _yt_tracks(author: MagicMock, count: int) -> list[QueueObject]:
        return [
            QueueObject(f"https://yt.com/watch?v=v{i}", f"T{i}", author)
            for i in range(count)
        ]

    async def test_playlist_index_drops_the_tracks_before_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A link copied at position 4 queues from #4, not from the top."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 6)
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert [t.title for t in result.tracks] == ["T3", "T4", "T5"]
        assert result.skipped == 3

    async def test_playlist_index_rebases_the_kept_positions(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Positions are assigned at construction, BEFORE the index slice, so the
        slice must rebase what it keeps. Without that, an &index=4 link archives
        every kept track three deeper than it actually waited — invisible unless
        a test carries an index."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4"
        source = parse_input(url, f"-play {url}")
        tracks = [
            QueueObject(
                f"https://yt.com/watch?v=v{i}",
                f"T{i}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1752530000.5, queue_position=i),
            )
            for i in range(6)
        ]
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert [t.analytics.queue_position for t in result.tracks] == [0, 1, 2]
        # keep_first_only is -playnow's; -play enqueues the whole tail.
        assert len(result.tracks) == 3
        # The ask time is untouched by the slice — one instant for the command.
        assert all(t.analytics.queued_at == 1752530000.5 for t in result.tracks)

    async def test_playlist_index_rebase_preserves_a_nonzero_base(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The rebase subtracts the dropped count, it does not zero the field: a
        playlist queued behind two songs still waits behind them."""
        url = "https://www.youtube.com/watch?v=v2&list=PLabc&index=3"
        source = parse_input(url, f"-play {url}")
        tracks = [
            QueueObject(
                f"https://yt.com/watch?v=v{i}",
                f"T{i}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1752530000.5, queue_position=2 + i),
            )
            for i in range(5)
        ]
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert [t.analytics.queue_position for t in result.tracks] == [2, 3, 4]

    async def test_playlist_index_1_queues_everything(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """index=1 is the first song, so it drops nothing — the common shape,
        since YouTube stamps it onto a share copied at the top."""
        url = "https://www.youtube.com/watch?v=v0&list=PLabc&index=1"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 3)
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert len(result.tracks) == 3
        assert result.skipped == 0

    async def test_playlist_index_past_the_end_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Not a silent empty enqueue: an out-of-range index would otherwise
        report "Queued playlist — 0 songs" and queue nothing."""
        url = "https://www.youtube.com/watch?v=v9&list=PLabc&index=9"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 3)
        with (
            patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)),
            pytest.raises(PlaylistIndexError) as excinfo,
        ):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert (excinfo.value.index, excinfo.value.total) == (9, 3)

    async def test_empty_playlist_raises_instead_of_queueing_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The same guard -playnow already had: a playlist that resolves to
        nothing is an error, not a successful enqueue of zero songs."""
        url = "https://www.youtube.com/playlist?list=PLabc"
        source = parse_input(url, f"-play {url}")
        with (
            patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=[])),
            pytest.raises(EmptyPlaylistError),
        ):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )

    async def test_playlist_timestamp_applies_to_the_linked_video(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`t=` names an offset into the `v=` video, and `index=` makes that
        video the head of the queue — so the offset lands on it."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4&t=90"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 6)
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert result.tracks[0].title == "T3"
        assert result.tracks[0].ts == 90
        assert all(t.ts is None for t in result.tracks[1:])

    async def test_playlist_timestamp_ignored_when_head_is_a_different_video(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """No index, so the queue starts at track 1 — which is not the video the
        offset belongs to. Seeking it would start the wrong song mid-way."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&t=30"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 6)
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert all(t.ts is None for t in result.tracks)

    async def test_playlist_index_error_embed_names_both_numbers(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The embed the user actually sees: their index and the real length,
        rendered from user_message rather than as "ValueError: …"."""
        await music_bot._command_error(
            mock_ctx, PlaylistIndexError(99, 16), title="Failed to queue song"
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "**#99**" in embed.description
        assert "**16 songs**" in embed.description
        assert "1 to 16" in embed.description
        assert "PlaylistIndexError" not in embed.description

    async def test_playlist_index_error_singular_total(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A one-song playlist offers no range — "from 1 to 1" would read as a
        bug in the message rather than as advice."""
        message = PlaylistIndexError(4, 1).user_message
        assert "**1 song**" in message
        assert "1 to 1" not in message

    async def test_empty_playlist_embed_explains_itself(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """EmptyPlaylistError renders as advice, not as "ValueError: …" — the
        same treatment as its PlaylistIndexError sibling."""
        await music_bot._command_error(
            mock_ctx, EmptyPlaylistError(), title="Failed to queue song"
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "no songs I can queue" in embed.description
        assert "ValueError" not in embed.description
        assert "EmptyPlaylistError" not in embed.description

    async def test_playnow_honours_the_playlist_index(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """-playnow interjects the track the link was copied at, not track 1."""
        url = "https://www.youtube.com/watch?v=v2&list=PLabc&index=3"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 5)
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            result = await music_bot._resolve_playnow_source(
                mock_ctx, source, origin=_ORIGIN
            )
        assert result.title == "T2"
        notice = mock_ctx.send.await_args.kwargs["embed"].description
        assert "#3" in notice

    async def test_playnow_index_past_the_end_reports_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """-playnow shares the guard, and its own error path renders the same
        embed under its own title."""
        url = "https://www.youtube.com/watch?v=v9&list=PLabc&index=9"
        source = parse_input(url, f"-play {url}")
        tracks = self._yt_tracks(mock_ctx.author, 3)
        with (
            patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)),
            pytest.raises(PlaylistIndexError) as excinfo,
        ):
            await music_bot._resolve_playnow_source(mock_ctx, source, origin=_ORIGIN)

        await music_bot._command_error(
            mock_ctx, excinfo.value, title="Failed to play song now"
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert embed.title == "Failed to play song now"
        assert "**#9**" in embed.description
        assert "**3 songs**" in embed.description

    async def test_playnow_spotify_playlist_bypasses_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # _resolve_playnow_source resolves both playlist shapes directly, so a
        # token passed only from queue_source would leave these two unclassified.
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None
        music_bot.spotify.playlist = AsyncMock(return_value=["Song A", "Song B"])
        fake_qobj = QueueObject("https://yt.com/v=1", "Song A", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot._resolve_playnow_source(mock_ctx, source, origin=_ORIGIN)
        assert self._passed_query_source(spy) == "spotify.com"

    async def test_playnow_youtube_playlist_bypasses_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        url = "https://www.youtube.com/playlist?list=PLabc"
        source = parse_input(url, f"-play {url}")
        tracks = [QueueObject("https://yt.com/v=1", "T", mock_ctx.author)]
        spy = AsyncMock(return_value=tracks)
        with patch("src.musicbot.YTDL.yt_playlist", new=spy):
            await music_bot._resolve_playnow_source(mock_ctx, source, origin=_ORIGIN)
        assert self._passed_query_source(spy) == "youtube.com"

    async def test_playnow_indexed_playlist_rebases_only_the_track_it_keeps(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """-playnow interjects exactly one track. Rebasing the rest is work whose
        only consumer throws it away, so keep_first_only trims first — and the
        one survivor still lands at 0, the depth an interjection actually has."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4"
        source = parse_input(url, f"-playnow {url}")
        tracks = [
            QueueObject(
                f"https://yt.com/watch?v=v{i}",
                f"T{i}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1752530000.5, queue_position=i),
            )
            for i in range(6)
        ]
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            kept = await music_bot._resolve_playnow_source(
                mock_ctx, source, origin=_ORIGIN
            )

        assert kept is tracks[3]
        assert kept.analytics.queue_position == 0
        # The discarded tail keeps its construction-time positions — untouched,
        # which is the whole point of trimming before the rebase.
        assert [t.analytics.queue_position for t in tracks[4:]] == [4, 5]

    async def test_playnow_analytics_is_depth_zero(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # An interjection plays immediately by definition, so -playnow reads no
        # queue depth at all — and its queued_at is still the ask time.
        url = "https://www.youtube.com/watch?v=abc"
        source = parse_input(url, f"-playnow {url}")
        fake_qobj = QueueObject(url, "Song", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot._resolve_playnow_source(mock_ctx, source, origin=_ORIGIN)
        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        assert analytics.queue_position == 0
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()


class TestSpotifyDisabled:
    """When Spotify isn't usable — no credentials (self.spotify is None, status
    disabled) or credentials rejected at startup (status invalid) — any Spotify
    source must raise SpotifyDisabledError, while every other source keeps
    working."""

    def test_require_spotify_returns_client_when_enabled(
        self, music_bot: MusicBot
    ) -> None:
        assert music_bot._require_spotify() is music_bot.spotify

    def test_require_spotify_raises_when_no_credentials(
        self, music_bot: MusicBot
    ) -> None:
        music_bot.spotify = None
        music_bot._spotify_status = SpotifyStatus.DISABLED
        with pytest.raises(SpotifyDisabledError) as exc:
            music_bot._require_spotify()
        assert exc.value.status is SpotifyStatus.DISABLED

    def test_require_spotify_raises_when_credentials_invalid(
        self, music_bot: MusicBot
    ) -> None:
        """Credentials were present (client built) but rejected at startup: the
        gate still refuses, and the error reports invalid rather than disabled."""
        music_bot._spotify_status = SpotifyStatus.INVALID
        with pytest.raises(SpotifyDisabledError) as exc:
            music_bot._require_spotify()
        assert exc.value.status is SpotifyStatus.INVALID

    async def test_spotify_playlist_raises_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.spotify = None
        music_bot._spotify_status = SpotifyStatus.DISABLED
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        with pytest.raises(SpotifyDisabledError):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )

    async def test_spotify_track_raises_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.spotify = None
        music_bot._spotify_status = SpotifyStatus.DISABLED
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        with pytest.raises(SpotifyDisabledError):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )

    async def test_spotify_track_raises_when_credentials_invalid(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Even with a live client object, an invalid status short-circuits the
        source before any Spotify API call is attempted."""
        music_bot._spotify_status = SpotifyStatus.INVALID
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        with pytest.raises(SpotifyDisabledError):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )

    async def test_non_spotify_source_unaffected_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A YouTube link still resolves normally with Spotify turned off."""
        music_bot.spotify = None
        music_bot._spotify_status = SpotifyStatus.DISABLED
        source = YTSource(url="https://yt.com/watch?v=abc", process=False)
        fake_qobj = QueueObject(
            "https://yt.com/watch?v=abc", "YT Song", mock_ctx.author
        )
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        assert isinstance(result, QueueObject)

    def test_disabled_error_message_is_actionable(self) -> None:
        msg = str(SpotifyDisabledError(SpotifyStatus.DISABLED))
        assert "SPOTIFY_CLIENT_ID" in msg
        assert "without" in msg
        assert "SoundCloud" in msg or "search" in msg

    def test_invalid_error_message_distinguishes_bad_credentials(self) -> None:
        msg = str(SpotifyDisabledError(SpotifyStatus.INVALID))
        assert "SPOTIFY_CLIENT_ID" in msg
        assert "rejected" in msg

    async def test_youtube_search_uses_ytsearch(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(ytsearch="ytsearch:test song", process=True)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ) as mock_yt:
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        call_args = mock_yt.call_args
        assert call_args[0][1] == "ytsearch:test song"

    async def test_youtube_playlist_calls_yt_playlist(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest123",
            process=False,
            type=YTType.PLAYLIST,
            list_id="PLtest123",
        )
        fake_qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author),
            QueueObject("https://yt.com/watch?v=2", "Track 2", mock_ctx.author),
        ]
        with patch(
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=fake_qobjs)
        ) as mock_playlist:
            result = await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        mock_playlist.assert_awaited_once_with(
            "https://www.youtube.com/playlist?list=PLtest123",
            mock_ctx.author,
            # "" because this YTSource is built directly rather than by
            # parse_input, which is what classifies. See TestQuerySourceClassification.
            query_source="",
            analytics=_ANALYTICS,
            user_input=_ORIGIN,
        )
        assert result == ResolvedYoutubePlaylist(tracks=fake_qobjs)

    async def test_youtube_playlist_raises_if_list_id_missing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """queue_source raises ValueError (not AssertionError) when list_id is None."""
        source = YTSource(
            url="https://www.youtube.com/watch?v=abc",
            process=False,
            type=YTType.PLAYLIST,
            list_id=None,
        )
        with pytest.raises(ValueError, match="list_id"):
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )

    async def test_youtube_playlist_preserves_full_url(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        full_url = "https://www.youtube.com/watch?v=XfHbPIx42uo&list=RDXfHbPIx42uo&start_radio=1"
        source = YTSource(
            url=full_url,
            process=False,
            type=YTType.PLAYLIST,
            list_id="RDXfHbPIx42uo",
        )
        fake_qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)
        ]
        with patch(
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=fake_qobjs)
        ) as mock_playlist:
            await music_bot.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
            )
        mock_playlist.assert_awaited_once_with(
            full_url,
            mock_ctx.author,
            query_source="",
            analytics=_ANALYTICS,
            user_input=_ORIGIN,
        )


class TestEnqueuePlaylist:
    @staticmethod
    def _make_enqueue_mp(mock_ctx: MagicMock) -> MagicMock:
        mp = MagicMock()
        mp.queue_put = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()
        return mp

    # ── YouTube playlist path ─────────────────────────────────────────────────

    async def test_yt_sends_embed_with_song_count_and_playlist_url(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author),
            QueueObject("https://yt.com/watch?v=2", "Track 2", mock_ctx.author),
        ]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "2 songs" in embed.title
        assert source.url in embed.description
        assert "Track 1" in embed.description

    async def test_yt_embed_states_the_skipped_songs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A shorter queue than the playlist needs an explanation, and only the
        user's own `index=` provides one."""
        source = YTSource(
            url="https://www.youtube.com/watch?v=x&list=PLtest&index=4",
            type=YTType.PLAYLIST,
            list_id="PLtest",
            index=4,
        )
        qobjs = [QueueObject("https://yt.com/watch?v=4", "Track 4", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs, skipped=3),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Starting at #4" in embed.description
        assert "skipped 3 earlier songs" in embed.description

    async def test_yt_embed_omits_the_skip_line_when_nothing_was_skipped(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Starting at" not in embed.description

    async def test_yt_singular_song_count_in_title(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Only Track", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "1 song" in embed.title
        assert "1 songs" not in embed.title

    async def test_yt_calls_queue_put_with_prefetch_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put.assert_awaited_once()
        _, call_kwargs = mp.queue_put.call_args
        assert call_kwargs.get("prefetch") is False

    # ── Spotify playlist path ─────────────────────────────────────────────────

    async def test_spotify_sends_queued_playlist_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        titles = ["Song A", "Song B", "Song C"]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=titles),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Queued playlist" in embed.title
        assert "Song A" in embed.description

    async def test_spotify_calls_queue_put_with_prefetch_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        titles = ["Song A", "Song B"]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=titles),
            mp,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put.assert_awaited_once()
        _, call_kwargs = mp.queue_put.call_args
        assert call_kwargs.get("prefetch") is False


# ── New coverage: __init__, get_mp, cleanup, validate_commands, commands, on_ready ──


class TestMusicBotInit:
    @pytest.fixture(autouse=True)
    def _spotify_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "x")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "y")

    def test_sets_bot_attribute(self, mock_bot: MagicMock) -> None:
        assert MusicBot(mock_bot).bot is mock_bot

    def test_mps_starts_empty(self, mock_bot: MagicMock) -> None:
        assert MusicBot(mock_bot).mps == {}

    def test_voice_watchdog_starts_with_no_timers(self, mock_bot: MagicMock) -> None:
        assert MusicBot(mock_bot).voice_watchdog._timers == {}

    def test_reads_redis_from_bot(self, mock_bot: MagicMock) -> None:
        mock_redis = MagicMock()
        mock_bot.redis = mock_redis
        assert MusicBot(mock_bot).redis is mock_redis

    def test_reads_history_archive_from_bot(self, mock_bot: MagicMock) -> None:
        # -ping's Postgres row reads this. MusicBotApp.setup_hook builds the
        # archive before load_extension constructs the cog, so it is always set
        # on a real bot.
        archive = MagicMock()
        mock_bot.history_archive = archive
        assert MusicBot(mock_bot).history_archive is archive

    def test_missing_history_archive_is_none_not_an_error(
        self, mock_bot: MagicMock
    ) -> None:
        # A bot without the attribute (tests, or a cog built outside MusicBotApp)
        # must still construct — the Postgres row degrades to n/a instead.
        del mock_bot.history_archive
        assert MusicBot(mock_bot).history_archive is None


class TestGetMp:
    def test_returns_existing_player_and_sets_context(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.set_context = MagicMock()
        music_bot.mps[mock_guild.id] = mp
        result = music_bot.get_mp(mock_ctx)
        assert result is mp
        mp.set_context.assert_called_once_with(mock_ctx)

    def test_creates_new_player_for_unknown_guild(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mock_mp = MagicMock()
        mock_mp.start = MagicMock()
        with patch("src.musicbot.MusicPlayer.from_context", return_value=mock_mp):
            result = music_bot.get_mp(mock_ctx)
        assert result is mock_mp
        mock_mp.start.assert_called_once()
        assert mock_guild.id in music_bot.mps


class TestCleanup:
    @staticmethod
    def _make_minimal_mp(
        music_bot: MusicBot, mock_guild: MagicMock, **overrides: Any
    ) -> MagicMock:
        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = None
        mp.store = None
        mp._progress_task = None
        mp._pause_debounce_task = None
        mp._heartbeat_task = None
        mp.retire_np_host_on_stop = AsyncMock()
        mp.update_activity = AsyncMock()
        # None = nothing was playing, the shape most of these tests mean. A bare
        # MagicMock attribute would read truthy and put a non-awaitable into
        # cleanup's gather; tests about the teardown itself override it.
        mp.claim_current_song_for_history = MagicMock(return_value=None)
        mp.history.add = AsyncMock()
        for attr, val in overrides.items():
            setattr(mp, attr, val)
        music_bot.mps[mock_guild.id] = mp
        return mp

    async def test_does_not_cancel_current_task(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """cleanup() skips cancellation when the alone-timer is the running task (self-cancel guard)."""
        current = asyncio.current_task()
        assert current is not None, "test must run inside an asyncio.Task"
        music_bot.voice_watchdog._timers[mock_guild.id] = (
            current  # simulate countdown calling cleanup on itself
        )

        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        # If the guard were missing, current_task().cancel() would have been called
        # and this coroutine would receive CancelledError at the next await.
        assert not current.cancelled()
        assert mock_guild.id not in music_bot.voice_watchdog._timers

    async def test_disconnects_voice_client(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client.disconnect = AsyncMock()
        await music_bot.cleanup(mock_guild)
        mock_guild.voice_client.disconnect.assert_awaited_once()

    async def test_removes_guild_from_mps(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        assert mock_guild.id not in music_bot.mps

    async def test_cancels_in_flight_prefetch_task(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _prefetch_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_cancels_in_flight_progress_task(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _progress_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_cancels_in_flight_pause_debounce_task(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _pause_debounce_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_cancels_in_flight_heartbeat_task(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Left running it keeps writing last_position_secs — and refreshing the
        state key's TTL — over fields clear_connection() is about to delete."""
        task = AsyncMock(spec=asyncio.Task)
        task.done.return_value = False
        task.cancel = MagicMock()
        self._make_minimal_mp(music_bot, mock_guild, _heartbeat_task=task)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        task.cancel.assert_called_once()

    async def test_retires_np_host_after_task_cancellation(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """-stop / alone-disconnect must dispose of the NP host (delete a
        dedicated NP message, strip a response host) so no message keeps a
        mid-song bar frozen by the stop — and only after the progress/loop
        tasks are down, so no tick can race the retire."""
        call_order: list[str] = []

        class _AwaitableTask:
            def done(self) -> bool:
                return False

            def cancel(self, msg: Any = None) -> None:
                call_order.append("cancel")

            def __await__(self) -> Iterator[Any]:
                return iter([])  # completes immediately, no exception

        mp = self._make_minimal_mp(
            music_bot, mock_guild, _progress_task=_AwaitableTask()
        )
        mp.retire_np_host_on_stop = AsyncMock(
            side_effect=lambda: call_order.append("retire")
        )
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        assert call_order == ["cancel", "retire"]

    async def test_resets_activity_after_disconnecting(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """A stopped bot must not keep advertising the song it stopped. The reset
        runs after the disconnect: until the voice client is gone it still registers
        as playing, so update_activity() bails and the stale presence survives."""
        call_order: list[str] = []
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mp.update_activity = AsyncMock(
            side_effect=lambda song: call_order.append("activity")
        )
        mock_guild.voice_client.disconnect = AsyncMock(
            side_effect=lambda force: call_order.append("disconnect")
        )

        await music_bot.cleanup(mock_guild)

        assert call_order == ["disconnect", "activity"]
        mp.update_activity.assert_awaited_once_with(None)

    async def test_cancels_running_alone_timer(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        timer = make_mock_task()
        music_bot.voice_watchdog._timers[mock_guild.id] = timer

        self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        timer.cancel.assert_called_once()
        assert mock_guild.id not in music_bot.voice_watchdog._timers

    async def test_noop_cleanup_does_not_error_without_timer(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        # No timer on the watchdog — cleanup must not raise KeyError.
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)  # guild not in mps either — pure noop

    async def test_clears_store_connection_on_cleanup(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        store = MagicMock()
        store.clear_connection = AsyncMock()
        store.refresh_ttl = AsyncMock()
        mp = self._make_minimal_mp(music_bot, mock_guild, store=store)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        mp.store.clear_connection.assert_awaited_once()
        mp.store.refresh_ttl.assert_awaited_once()

    async def test_records_the_song_abandoned_mid_play(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Every teardown converges here — -stop, the alone-disconnect, and an
        external voice kick — so one write site covers all three. Without it the
        song a listener was hearing when the bot left is lost outright: it is not
        in the queue, and the loop is cancelled before its own write site runs."""
        entry = HistoryEntry(guild_id=mock_guild.id, title="Abandoned")
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mp.claim_current_song_for_history = MagicMock(return_value=entry)
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        mp.history.add.assert_awaited_once_with(entry)

    async def test_records_before_the_np_host_is_retired(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The claim reads the live NP host, and retire_np_host_on_stop() disposes
        of it — so the claim has to happen first or the entry loses the ids."""
        order: list[str] = []
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mp.claim_current_song_for_history = MagicMock(
            side_effect=lambda: (
                order.append("claim") or HistoryEntry(guild_id=mock_guild.id, title="x")
            )
        )
        mp.retire_np_host_on_stop = AsyncMock(
            side_effect=lambda: order.append("retire")
        )
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        assert order == ["claim", "retire"]

    async def test_every_command_opens_a_span(self, music_bot: MusicBot) -> None:
        """Repo convention: every command body runs inside its own
        bot.<name> span, or its errors and timings never reach Tempo. -remove
        shipped without one and nothing noticed, so this asserts the decorator SET
        rather than any single member of it."""

        missing = []
        for cmd in music_bot.get_commands():
            src = inspect.getsource(cmd.callback)
            header = src.split("async def ", 1)[0]
            if f'start_as_current_span("bot.{cmd.name}")' not in header:
                missing.append(cmd.name)

        assert missing == [], f"commands with no span: {missing}"

    async def test_the_claim_precedes_every_await_in_cleanup(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The claim is synchronous SO THAT no await can open a window for the
        loop to record the same song. Ordering against the retire is not enough to
        pin that: moving the claim after the teardown gather leaves the retire
        later still. This asserts the claim runs before the first suspension."""
        order: list[str] = []
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mp.claim_current_song_for_history = MagicMock(
            side_effect=lambda: (
                order.append("claim") or HistoryEntry(guild_id=mock_guild.id, title="x")
            )
        )

        async def _cancel(task: Any) -> None:
            order.append("await")

        mp.retire_np_host_on_stop = AsyncMock(
            side_effect=lambda: order.append("retire")
        )
        mock_guild.voice_client = None

        with patch("src.musicbot.cancel_task", new=_cancel):
            await music_bot.cleanup(mock_guild)

        assert order[0] == "claim", order

    async def test_the_history_write_never_delays_the_disconnect(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """The write is Redis + outbox IO and nothing below reads it. Inside the
        teardown gather it delayed the silence -stop asked for, because gather
        completes on its SLOWEST member — measured at 20s against an unreachable
        host, unbounded against one that accepts and then stalls."""
        order: list[str] = []
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mp.claim_current_song_for_history = MagicMock(
            return_value=HistoryEntry(guild_id=mock_guild.id, title="x")
        )
        mp.history.add = AsyncMock(side_effect=lambda _e: order.append("history"))
        vc = MagicMock()
        vc.disconnect = AsyncMock(side_effect=lambda **_kw: order.append("disconnect"))
        mock_guild.voice_client = vc

        await music_bot.cleanup(mock_guild)

        assert order == ["disconnect", "history"]

    async def test_nothing_playing_writes_no_history(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        # The helper's default: a teardown with no live song must not invent one.
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        mp.history.add.assert_not_awaited()

    async def test_noop_when_guild_not_in_mps(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)  # must not raise

    async def test_cancels_player_task_before_disconnect(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """_player must be cancelled before disconnect() so the loop cannot wake up
        and start the next song between voice_client.stop() firing and the loop
        being cancelled (the root cause of the brief-next-song-on-stop bug)."""
        call_order: list[str] = []

        class _AwaitableTask:
            """Minimal awaitable task double: done()=False, cancel() tracked, await=noop."""

            def done(self) -> bool:
                return False

            def cancel(self, msg: Optional[str] = None) -> None:
                call_order.append("cancel")

            def __await__(self) -> Iterator[None]:
                return iter([])  # completes immediately, no exception

        mp = MagicMock()
        mp._prefetch_task = None
        mp._restore_task = None
        mp._player = _AwaitableTask()
        mp.store = None
        mp.retire_np_host_on_stop = AsyncMock()
        mp.claim_current_song_for_history = MagicMock(return_value=None)
        music_bot.mps[mock_guild.id] = mp

        async def _disconnect(**_kw: Any) -> None:
            call_order.append("disconnect")

        mock_guild.voice_client.disconnect = AsyncMock(side_effect=_disconnect)

        await music_bot.cleanup(mock_guild)

        assert call_order.index("cancel") < call_order.index("disconnect"), (
            "player task must be cancelled before voice disconnect"
        )


class TestCogBeforeInvoke:
    async def test_calls_get_mp(self, music_bot: MusicBot, mock_ctx: MagicMock) -> None:
        mock_mp = MagicMock()
        mock_mp.store = None  # skip the channel-persistence branch
        music_bot.get_mp = MagicMock(return_value=mock_mp)
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_called_once_with(mock_ctx)

    async def test_persists_text_channel_when_channel_changes(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """set_connection is called when the command arrives from a new text channel."""
        old_channel = MagicMock(spec=discord.TextChannel)
        new_channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = new_channel

        store = MagicMock()
        store.set_connection = AsyncMock()

        mp = MagicMock()
        mp.home_channel = old_channel
        mp.store = store
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()
        vc.channel.id = 555
        mock_guild.voice_client = vc

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_awaited_once_with(vc.channel.id, new_channel.id)

    async def test_no_persist_when_channel_unchanged(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """set_connection is not called when the text channel hasn't changed."""
        channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = channel

        store = MagicMock()
        store.set_connection = AsyncMock()

        mp = MagicMock()
        mp.home_channel = channel  # same object → no change
        mp.store = store
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_not_awaited()

    async def test_no_persist_when_no_voice_client(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """set_connection is not called when the bot isn't in a voice channel yet."""
        old_channel = MagicMock(spec=discord.TextChannel)
        new_channel = MagicMock(spec=discord.TextChannel)
        mock_ctx.channel = new_channel

        store = MagicMock()
        store.set_connection = AsyncMock()

        mp = MagicMock()
        mp.home_channel = old_channel
        mp.store = store
        music_bot.mps[mock_guild.id] = mp
        music_bot.get_mp = MagicMock(return_value=mp)

        mock_guild.voice_client = None  # not connected yet

        await music_bot.cog_before_invoke(mock_ctx)

        store.set_connection.assert_not_awaited()

    async def test_returns_early_when_guild_is_none(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """cog_before_invoke must not call get_mp (which asserts guild is not None) in a DM."""
        mock_ctx.guild = None
        music_bot.get_mp = MagicMock()
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_not_called()


class TestValidateCommands:
    async def test_raises_command_error_when_not_in_voice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "skip"
        mock_ctx.author.voice = None
        mock_ctx.send = AsyncMock()
        with pytest.raises(commands.CommandError):
            await music_bot.validate_commands(mock_ctx)
        mock_ctx.send.assert_awaited_once()

    async def test_passes_when_member_in_voice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "play"
        # mock_ctx.author has voice set by conftest
        await music_bot.validate_commands(mock_ctx)  # must not raise


class TestMaxConcurrencyNotice:
    async def test_reports_already_running(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "ping"
        await music_bot.cog_command_error(
            mock_ctx, commands.MaxConcurrencyReached(1, commands.BucketType.guild)
        )
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "already running" in embed.description
        assert "ping" in embed.description

    async def test_a_cooldown_says_how_long_is_left(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """-analytics is the only command with a cooldown, and this arm is the only
        place a user learns why they were refused. Deleting it falls through to the
        generic handler, and zeroing retry_after reads as "try again now"."""
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "analytics"
        await music_bot.cog_command_error(
            mock_ctx,
            commands.CommandOnCooldown(
                commands.Cooldown(1, 30.0), 12.0, commands.BucketType.guild
            ),
        )
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "12" in embed.description


def _running(cog: MusicBot, coro_name: str) -> bool:
    """True when `cog` has a tracked background task running that coroutine.

    Tests that used to count `_restore_tasks` broke whenever a new fire-and-forget
    task was added, for a reason they had no opinion about. Naming the coroutine
    keeps the assertion about the thing under test.
    """
    return any(coro_name in repr(t.get_coro()) for t in cog._restore_tasks)


class TestCogUnloadReleasesSpotify:
    """The Spotify client's session lives for the life of the process, so the
    cog's unload is the only thing that releases it."""

    async def test_cog_unload_closes_the_spotify_session(
        self, music_bot: MusicBot
    ) -> None:
        """Nothing else asserts this: the conftest fixture closes stray sessions
        after every test, so dropping the call from cog_unload leaves the whole
        suite green."""
        music_bot._restore_tasks = set()
        await music_bot.cog_unload()
        cast(Any, music_bot.spotify).aclose.assert_awaited_once()

    async def test_a_failing_step_does_not_skip_the_ones_after_it(
        self, music_bot: MusicBot
    ) -> None:
        """Cog._eject only logs what this raises and BotBase.close swallows that,
        so an unguarded early failure would silently skip every later step and
        nothing would say so — the same shape that once made a hung Postgres
        permanently skip the rest of MusicBotApp.close()."""
        music_bot._restore_tasks = set()
        music_bot.debug_settings = MagicMock()
        music_bot.debug_settings.aclose = AsyncMock(side_effect=OSError("boom"))

        with patch.object(
            debug_mode, "close_prometheus_session", new=AsyncMock()
        ) as prom:
            await music_bot.cog_unload()  # must not raise

        prom.assert_awaited_once()
        cast(Any, music_bot.spotify).aclose.assert_awaited_once()

    async def test_cog_unload_skips_a_disabled_spotify(
        self, music_bot: MusicBot
    ) -> None:
        """The shipping default: no credentials means `spotify` is None, and the
        unload must not raise on the guard that exists for exactly that case."""
        music_bot.spotify = None
        music_bot._restore_tasks = set()
        await music_bot.cog_unload()  # must not raise


class TestCogLoadSpotifyValidation:
    """cog_load spawns the credential probe as a background task so startup is
    never blocked, and the probe resolves _spotify_status without ever raising."""

    async def test_no_op_when_spotify_disabled(self, music_bot: MusicBot) -> None:
        music_bot.spotify = None
        music_bot._spotify_status = SpotifyStatus.DISABLED
        music_bot._restore_tasks = set()
        await music_bot.cog_load()
        assert music_bot._spotify_status is SpotifyStatus.DISABLED
        assert not _running(music_bot, "_validate_spotify_credentials")

    async def test_cog_load_spawns_probe_without_blocking(
        self, music_bot: MusicBot
    ) -> None:
        """cog_load returns immediately (leaving status enabled) and the probe
        runs as a tracked background task, not inline."""
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.validate = AsyncMock(return_value=None)
        music_bot._spotify_status = SpotifyStatus.ENABLED
        music_bot._restore_tasks = set()

        await music_bot.cog_load()
        # Spawned, not awaited. Named rather than counted: cog_load also spawns
        # the per-guild config hydration, and a bare count would make this test
        # fail for a reason it does not care about.
        assert _running(music_bot, "_validate_spotify_credentials")

        await asyncio.gather(*music_bot._restore_tasks)  # let the probe finish
        music_bot.spotify.validate.assert_awaited_once()
        assert music_bot._spotify_status is SpotifyStatus.ENABLED

    async def test_valid_credentials_stay_enabled(self, music_bot: MusicBot) -> None:
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.validate = AsyncMock(return_value=None)
        music_bot._spotify_status = SpotifyStatus.ENABLED
        await music_bot._validate_spotify_credentials()
        music_bot.spotify.validate.assert_awaited_once()
        assert music_bot._spotify_status is SpotifyStatus.ENABLED

    async def test_auth_error_flips_to_invalid(self, music_bot: MusicBot) -> None:
        """Only an authentication rejection disables Spotify."""
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.validate = AsyncMock(side_effect=SpotifyAuthError(400))
        music_bot._spotify_status = SpotifyStatus.ENABLED
        await music_bot._validate_spotify_credentials()  # must not raise
        assert music_bot._spotify_status is SpotifyStatus.INVALID

    async def test_network_error_leaves_enabled(self, music_bot: MusicBot) -> None:
        """A non-auth failure is inconclusive: Spotify stays enabled."""
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.validate = AsyncMock(
            side_effect=OSError("connection refused")
        )
        music_bot._spotify_status = SpotifyStatus.ENABLED
        await music_bot._validate_spotify_credentials()  # must not raise
        assert music_bot._spotify_status is SpotifyStatus.ENABLED

    async def test_timeout_leaves_enabled(self, music_bot: MusicBot) -> None:
        """A probe timeout is inconclusive (not an auth rejection): stays enabled."""
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.validate = AsyncMock(side_effect=asyncio.TimeoutError)
        music_bot._spotify_status = SpotifyStatus.ENABLED
        await music_bot._validate_spotify_credentials()  # must not raise
        assert music_bot._spotify_status is SpotifyStatus.ENABLED


class TestPlayCommand:
    """Tests for play()'s cold-join parallelism. asyncio.Future stands in for the
    join_task: unlike AsyncMock it is directly awaitable, matching what the real
    Task does when the code says `await join_task`."""

    async def test_cold_join_creates_task_and_awaits_after_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join is launched as a task; join_task is awaited after queue_source."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        # Resolved Future: done() is True, await returns immediately.
        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task) as mock_create,
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mock_create.assert_called_once()
        music_bot.queue_source.assert_awaited_once()
        music_bot._enqueue_single.assert_awaited_once()

    async def test_warm_path_skips_join_task(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """When already in voice, no join task is created and queue_source runs directly."""
        mock_ctx.voice_client = playing_vc()
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task") as mock_create,
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mock_create.assert_not_called()
        music_bot.queue_source.assert_awaited_once()

    async def test_cold_join_cancels_inflight_join_when_queue_source_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """queue_source fails while join is still running → join task cancelled, then cleanup()."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = None

        # Pending Future: done() is False; cancel() marks it cancelled so the
        # subsequent `await join_task` in the guard raises CancelledError (suppressed).
        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()  # error embed shown

    async def test_cold_join_cleans_up_when_join_done_before_queue_source_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join completes first, then queue_source fails → cleanup() called (handles ghost connection)."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = MagicMock(
            spec=discord.VoiceClient
        )  # join already established voice

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)  # done() is True
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_not_called()  # already done, nothing to cancel
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()

    async def test_cold_join_cancels_and_cleans_up_partial_connection(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join in-flight but voice partially established → cancel join task, then cleanup()."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = MagicMock(spec=discord.VoiceClient)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()  # pending, done() is False
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()


class TestPlayAnalytics:
    """The ask-time Analytics -play mints and hands to queue_source.

    Asserted on the call rather than on a returned object: queue_source is what
    carries the value into every construction site, and nothing downstream
    restamps it, so the hand-off IS the behavior."""

    async def test_warm_path_carries_the_ask_time_and_the_player_depth(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = playing_vc()
        mp = mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot.queue_source = spy
        music_bot._enqueue_single = AsyncMock()

        with no_typing("src.musicbot.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        # The message snowflake, NOT time.time(): gateway delivery lag is real
        # time the user waited, and so is the 1-4s resolve that follows.
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()
        assert analytics.queue_position == 7

    async def test_cold_path_is_depth_zero_without_reading_the_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A cold-start song front-inserts ahead of the restored queue and plays
        first, so its depth is 0 by construction — the queue is never asked."""
        mock_ctx.voice_client = None
        mp = mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)  # would be wrong if read
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot.queue_source = spy
        music_bot._enqueue_single = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = connected_vc()
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()
        assert analytics.queue_position == 0
        mp.enqueue_depth.assert_not_called()

    async def test_warm_path_reads_the_depth_after_the_restore_lands(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Crash recovery reconnects voice BEFORE restore_entries() replays the
        queue. Reading the depth in that window records 0 behind a
        queue about to reappear, so the read waits the restore out."""
        mock_ctx.voice_client = playing_vc()
        mp = mock_mp()
        restored = False

        async def _land_the_restore(**_kw: Any) -> bool:
            nonlocal restored
            restored = True
            return True

        mp.wait_for_restore = AsyncMock(side_effect=_land_the_restore)
        # What the real queue answers on either side of the restore.
        mp.enqueue_depth = MagicMock(side_effect=lambda: 12 if restored else 0)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot.queue_source = spy
        music_bot._enqueue_single = AsyncMock()

        with no_typing("src.musicbot.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        assert spy.await_args.kwargs["analytics"].queue_position == 12


class TestPlayWhilePaused:
    """-play on a paused song interjects instead of appending
    . Appending would leave the bot silent
        with the request buried behind a paused song."""

    def _paused_mp(self) -> MagicMock:
        mp = mock_mp()
        mp.current_song = MagicMock(title="Paused Song")
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Paused Song",
                resume_position=83,
                was_paused=True,
                returns_paused=False,
            )
        )
        return mp

    async def test_interjects_with_resume_paused_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New Song", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)
        music_bot._enqueue_single = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()

        with (
            no_typing("src.musicbot.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_awaited_once()
        assert mp.interject.await_args.kwargs["resume_paused"] is False
        music_bot._enqueue_single.assert_not_awaited()
        mp.queue_put.assert_not_awaited()

    async def test_wording_says_resume_not_return_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The song was paused but comes back playing — announcing "will return
        paused" would be wrong. This is why returns_paused exists separately
        from was_paused."""
        mock_ctx.voice_client = paused_vc()
        music_bot.get_mp = MagicMock(return_value=self._paused_mp())
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        mock_ctx.message.add_reaction = AsyncMock()

        with (
            no_typing("src.musicbot.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Paused Song" in embed.description
        assert "1:23" in embed.description
        assert "will resume from there" in embed.description
        assert "return paused" not in embed.description

    async def test_playing_song_is_not_interjected(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression guard: -play on a *playing* bot still appends."""
        mock_ctx.voice_client = playing_vc()
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()

        with no_typing("src.musicbot.background_typing"), patch("asyncio.create_task"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        music_bot._enqueue_single.assert_awaited_once()

    async def test_paused_without_current_song_falls_through(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing to interrupt — take the ordinary append path rather than
        building an interjection around a song that isn't there."""
        mock_ctx.voice_client = paused_vc()
        mp = self._paused_mp()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()

        with no_typing("src.musicbot.background_typing"), patch("asyncio.create_task"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        music_bot._enqueue_single.assert_awaited_once()

    async def test_resume_during_resolution_appends_instead(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A -resume landing during the 1-4s extraction removes the reason to
        interject, so the resolved track is appended rather than interrupting a
        song the user just chose to keep playing."""
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        mp.enqueue_depth = MagicMock(return_value=9)
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)
        music_bot._enqueue_single = AsyncMock()

        async def _resolve_then_resume(*a: Any, **kw: Any) -> None:
            vc.is_paused.return_value = False  # user hit -resume mid-extraction
            return None

        with (
            no_typing("src.musicbot.background_typing"),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        music_bot._enqueue_single.assert_awaited_once()
        assert qobj.interjected is False  # must not trigger replace semantics later
        # Re-minted for the append: the 0 minted for an interjection would claim
        # this song played immediately when it waited behind the whole queue.
        assert qobj.analytics.queue_position == 9

    async def test_resolution_failure_leaves_paused_song_untouched(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Resolution happens before interject, so a failed lookup never stops
        the paused song."""
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

        with no_typing("src.musicbot.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        vc.stop.assert_not_called()
        assert mp.current_song is not None
        mock_ctx.send.assert_awaited()  # error embed

    async def test_playlist_collapses_to_first_track(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unlike the disconnected path (whole playlist front-inserted), an
        interjection collapses to one track so the paused song's return is not
        delayed indefinitely — and says so."""
        mock_ctx.voice_client = paused_vc()
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        mock_ctx.message.add_reaction = AsyncMock()
        # Distinct sentinel, not tracks[0]: if the URL ever stops parsing as a
        # playlist, _resolve_playnow_source falls through to queue_source, and
        # the identity assertion below catches it. (Stubbing it at all is also
        # a network guard — an unstubbed one runs a real yt-dlp extraction.)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject(
                "https://yt.com/v=fell-through", "X", mock_ctx.author
            )
        )
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"
        # parse_input splits the full message to count args — an unset MagicMock
        # content makes every URL fall back to the ytsearch branch.
        mock_ctx.message.content = f"-play {url}"

        with (
            no_typing("src.musicbot.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_awaited_once()
        assert mp.interject.await_args.args[0] is tracks[0]
        sent = mock_ctx.send.await_args_list + mock_ctx.send.call_args_list
        notices = [
            c.kwargs["embed"].description
            for c in sent
            if c.kwargs.get("embed") is not None
        ]
        assert any("first track" in (d or "") for d in notices), notices


class TestPlayFrontInsertion:
    """-play on a disconnected bot means "play this", not "play whatever was
    left over": the requested song jumps ahead of the queue persisted by a
    previous -stop, which resumes behind it."""

    async def test_cold_path_enqueues_at_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        single_call = music_bot._enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["front"] is True

    async def test_cold_path_queues_nothing_when_the_join_never_connected(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join swallows its own failures, so the cold path used to front-insert
        onto a join that never landed — handing loop() a song it can only raise on
        once the gate opens, once per restored entry, with the Redis mirror keeping
        everything it drains."""
        mock_ctx.voice_client = None  # the stub join leaves it that way
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        with (
            no_typing("src.musicbot.background_typing"),
            patch(
                "asyncio.create_task", side_effect=lambda c: (c.close(), join_task)[1]
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        music_bot._enqueue_single.assert_not_awaited()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mp.repark_crashed_head.assert_awaited_once()

    async def test_cold_path_queues_nothing_when_the_restore_never_lands(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """put_front LPUSHes the mirror while restore_entries replays already-listed
        entries in memory only, so inserting against a restore that never read its
        snapshot double-queues the song. Not landing is a reason not to insert."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        mp = mock_mp()
        mp.wait_for_restore = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        music_bot._enqueue_single.assert_not_awaited()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        assert "wasn't queued" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_cold_path_reparks_the_recovered_head_when_extraction_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The cleanup this path already ran drops the player holding the only copy
        of a crash-recovered song — restore cleared its state fields the moment it
        re-queued it. Order matters: clear_connection() HDELs what the re-park
        writes."""
        mock_ctx.voice_client = None
        calls: list[str] = []
        mp = mock_mp()
        mp.repark_crashed_head = AsyncMock(side_effect=lambda: calls.append("repark"))
        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock(side_effect=lambda _g: calls.append("cleanup"))

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        with (
            no_typing("src.musicbot.background_typing"),
            patch(
                "asyncio.create_task", side_effect=lambda c: (c.close(), join_task)[1]
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert calls == ["cleanup", "repark"]

    async def test_warm_path_enqueues_at_back(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression guard: a -play on a connected bot keeps append semantics."""
        mock_ctx.voice_client = playing_vc()
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        with no_typing("src.musicbot.background_typing"), patch("asyncio.create_task"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        single_call = music_bot._enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["front"] is False
        # No playback hold on the warm path — the gate is already open. It DOES
        # wait on the restore, on the SAME bound as every other queue-mutating
        # command: enqueueing ahead of entries restore_entries has not replayed
        # yet misaligns every later LPOP, so this is not just an analytics read.
        mp.defer_playback.assert_not_called()
        assert (
            mp.wait_for_restore.await_args is not None
            and mp.wait_for_restore.await_args.kwargs["timeout"] == RESTORE_WAIT_SECS
        )

    async def test_cold_path_waits_for_restore_before_enqueueing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Ordering: put_front LPUSHes the Redis mirror while
        restore_entries replays already-listed entries in memory only, so
        inserting before restore reads its snapshot double-queues the song."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        calls: list[str] = []
        mp = mock_mp()

        async def restored(**_kw: object) -> bool:
            calls.append("restore")
            return True

        mp.wait_for_restore = AsyncMock(side_effect=restored)
        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock(
            side_effect=lambda *a, **kw: calls.append("enqueue")
        )
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert calls == ["restore", "enqueue"]

    async def test_cold_path_holds_playback_gate_across_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join opens the gate as soon as the handshake lands — the hold is what
        stops the restored head from starting while queue_source is still
        extracting."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = mock_mp()
        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.defer_playback.assert_called_once()

    async def test_front_single_uses_queue_put_front_and_sends_resume_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = mock_mp(qsize=3)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(mock_ctx, qobj, mp, front=True)

        mp.queue_put_front.assert_awaited_once_with(qobj)
        mp.queue_put.assert_not_awaited()
        # The song being started is handed to the builder: it is the only thing
        # in this response that names it (no Now Playing block exists yet).
        mp.build_resume_notice_embed.assert_called_once_with(qobj)
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed is mp.build_resume_notice_embed.return_value

    async def test_front_single_sends_nothing_when_nothing_persisted(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """No restored queue means no resumption to announce, and the notice
        exists only to explain a restore — the 👍 plus the Now Playing message
        that follows are the whole response."""
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = mock_mp(qsize=0)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(mock_ctx, qobj, mp, front=True)

        mp.queue_put_front.assert_awaited_once_with(qobj)
        mock_ctx.send.assert_not_awaited()

    async def test_front_playlist_inserts_all_tracks_in_order(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unlike -playnow (first track only), -play front-inserts a playlist in
        full — nothing is playing here to delay the return of."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks),
            mp,
            front=True,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put_front.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()

    async def test_cold_path_routes_playlist_through_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """End-to-end wiring for the playlist half of the cold path: play()'s
        list branch must carry front=True into _enqueue_playlist. Previously
        only _enqueue_playlist was tested directly, leaving this dispatch —
        and the decision that a playlist front-inserts in full — unpinned."""
        mock_ctx.voice_client = None
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]

        music_bot.queue_source = AsyncMock(return_value=ResolvedYoutubePlaylist(tracks))
        music_bot._enqueue_playlist = AsyncMock()
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.musicbot.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        music_bot._enqueue_single.assert_not_awaited()
        pl_call = music_bot._enqueue_playlist.await_args
        assert pl_call is not None
        assert pl_call.kwargs["front"] is True
        assert pl_call.args[2] == ResolvedYoutubePlaylist(tracks)

    async def test_front_insert_after_restore_orders_both_legs(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """End to end against a real GuildQueue and fake Redis: the requested song
        leads, the persisted entries follow in order, both legs agree. Also the
        double-queue regression — put_front LPUSHes onto the same Redis list
        restore_entries replays, so inserting before the snapshot read queues twice."""
        assert music_player.store is not None
        for title in ("Persisted One", "Persisted Two"):
            await fake_redis.rpush(
                music_player.store.queue_key(),
                orjson.dumps(
                    {
                        "webpage_url": f"https://yt.com/v={title}",
                        "title": title,
                        "requester_id": mock_author.id,
                        "ts": None,
                    }
                ),
            )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        await music_player._restore_state()
        assert music_player.queue.qsize() == 2

        qobj = QueueObject("https://yt.com/v=new", "New Song", mock_author)
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("src.youtube.YTDL.prefetch_stream", new=AsyncMock()):
            await music_bot._enqueue_single(mock_ctx, qobj, music_player, front=True)

        titles = [
            queue_object(item).title for item in music_player.queue.display_items()
        ]
        assert titles == ["New Song", "Persisted One", "Persisted Two"]
        assert titles.count("New Song") == 1

        stored = [
            orjson.loads(raw)["title"]
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert stored == ["New Song", "Persisted One", "Persisted Two"]

        # The notice counts the RESTORED entries only. Building it after the
        # front insert would say 3 and include the song the Now Playing block
        # is already announcing.
        notice = mock_ctx.send.await_args.kwargs["embed"]
        queued = next(f for f in notice.fields if f.name == "Queued")
        assert queued.value == "**2** songs"


class TestEnqueueSingle:
    @staticmethod
    def _playing_mp(head: Any = None) -> MagicMock:
        """A player with a song live and `head` at the queue front. The default
        head is a fresh Mock, i.e. NOT the song being queued."""
        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue.peek_next = MagicMock(
            return_value=head if head is not None else MagicMock()
        )
        mp.queue_put = AsyncMock()
        mp.repin_now_playing = AsyncMock(return_value=True)
        return mp

    async def test_reposts_the_block_when_the_song_becomes_the_head(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The block's "Up next" card and the confirmation render the same body, so
        a song that lands at the head would be described twice in one message. The
        live block is re-hosted instead and no confirmation is sent."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.repin_now_playing.assert_awaited_once()
        mp.build_queued_song_embed.assert_not_called()
        mock_ctx.send.assert_not_awaited()

    async def test_sends_confirmation_when_something_is_already_queued(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp()  # head is some other song

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.repin_now_playing.assert_not_awaited()
        mp.build_queued_song_embed.assert_called_once_with(qobj, warning=None)
        assert (
            mock_ctx.send.await_args.kwargs["embed"]
            is mp.build_queued_song_embed.return_value
        )

    async def test_confirmation_when_repin_reports_no_live_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """repin_now_playing() answers False when the song ended mid-send — it
        already disposed of its message, so the confirmation is the honest reply."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)
        mp.repin_now_playing = AsyncMock(return_value=False)

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.repin_now_playing.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"]
            is mp.build_queued_song_embed.return_value
        )

    async def test_warning_gets_its_own_message_on_the_repin_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The re-hosted block has no description of its own to carry the warning."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)

        await music_bot._enqueue_single(mock_ctx, qobj, mp, warning="watch out")

        mp.repin_now_playing.assert_awaited_once()
        assert "watch out" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_warning_rides_the_confirmation_when_one_is_sent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp()

        await music_bot._enqueue_single(mock_ctx, qobj, mp, warning="watch out")

        mp.build_queued_song_embed.assert_called_once_with(qobj, warning="watch out")

    async def test_enqueues_before_reading_the_head(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The reply's shape depends on the put having landed, so the put is
        awaited ahead of it rather than gathered with it. Read against a queue
        whose head only appears once queue_put has run."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=None)
        mp.queue.peek_next = MagicMock(return_value=None)

        async def _put(_: Any) -> None:
            mp.queue.peek_next = MagicMock(return_value=qobj)

        mp.queue_put = AsyncMock(side_effect=_put)

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.repin_now_playing.assert_awaited_once()

    async def test_no_queued_embed_when_nothing_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.build_queued_song_embed.assert_not_called()
        mp.repin_now_playing.assert_not_called()
        mock_ctx.send.assert_not_awaited()


class TestSetup:
    @pytest.fixture(autouse=True)
    def _spotify_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "x")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "y")

    async def test_adds_music_bot_cog(self) -> None:
        from src.musicbot import setup

        mock_bot = AsyncMock()
        await setup(mock_bot)
        mock_bot.add_cog.assert_awaited_once()


class TestCommandArgumentBinding:
    """`-play`, `-playnow` and `-remove` all consume the rest of the line, because
    a positional binds ONE WORD: `-play` stores its argument as the origin
    `-remove` matches on, so `-play never gonna give you up` would record
    `"never"` and `-remove never` would become a wildcard over every song starting
    with it. Asserted on the callback signature, since that is where the binding
    lives."""

    @pytest.mark.parametrize("name", ["play", "playnow", "remove"])
    def test_the_argument_consumes_the_rest_of_the_line(self, name: str) -> None:

        callback = getattr(MusicBot, name).callback
        param = list(inspect.signature(callback).parameters.values())[2]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"-{name}'s {param.name} is positional; discord.py will bind one word"
        )

    @pytest.mark.parametrize(
        "typed,expected",
        [
            ("never gonna give you up", "never gonna give you up"),
            ("some song   ", "some song"),  # read_rest keeps trailing whitespace
        ],
    )
    async def test_what_the_user_typed_reaches_queue_source_whole(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        typed: str,
        expected: str,
    ) -> None:
        """The end-to-end C1 guards: the origin -remove matches on is the whole
        line, stripped. Previously it was the first word."""
        mock_ctx.message.content = f"-play {typed}"
        mock_ctx.voice_client = connected_vc()
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        with no_typing("src.musicbot.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=typed)

        call = music_bot.queue_source.await_args
        assert call is not None
        assert call.kwargs["origin"] == expected


# ── -playnow ──────────────────────────────────────────────────────────────────


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
        music_bot.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        assert qobj.interjected is True
        # The origin reaches the song through yt_source's required user_input, not
        # a post-hoc assignment — so with queue_source mocked out, assert it was
        # PASSED. A real yt_source stamps it (see test_youtube).
        origin_call = music_bot.queue_source.await_args
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
        music_bot.queue_source = AsyncMock(
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
        music_bot.queue_source = AsyncMock(
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
        music_bot.queue_source = AsyncMock(
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
        music_bot.queue_source = AsyncMock(return_value=qobj)

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
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=qobj)
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
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=[first, second])
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
        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

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
        music_bot.queue_source = AsyncMock(return_value=qobj)

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

        with patch("src.musicbot.YTDL.prefetch_stream", new=prefetch):
            await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url="test")

        prefetch.assert_awaited_once_with(qobj, redis=music_bot.redis)
        assert order == ["prefetch", "interject"]


class TestDebugCommand:
    """The `-debug` command surface: toggle semantics, per-guild scoping, and the
    argument grammar. What the snapshot RENDERS is tests/test_debug.py's job."""

    async def test_status_sends_the_snapshot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """channel.send, not ctx.send: the snapshot is a live-edited dashboard now,
        and an edit loop must not own the Now Playing host."""
        mock_ctx.guild.voice_client = None
        await command_callback(MusicBot.debug)(music_bot, mock_ctx)
        mock_ctx.channel.send.assert_awaited_once()
        embed = mock_ctx.channel.send.call_args.kwargs["embeds"][0]
        assert embed.title == "🐞 Debug snapshot"

    async def test_enable_then_disable_round_trips(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        guild_id = mock_ctx.guild.id
        assert music_bot.debug_settings.enabled(guild_id) is False
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        assert music_bot.debug_settings.enabled(guild_id) is True
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--disable")
        assert music_bot.debug_settings.enabled(guild_id) is False

    async def test_toggle_is_scoped_to_the_invoking_guild(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Per-guild is the blast-radius containment behind the Manage Server gate:
        an enable typed in one server must not decorate another's replies."""
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is True
        assert music_bot.debug_settings.enabled(424242424242424242) is False

    async def test_env_default_applies_where_no_override_exists(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.debug_settings._default = True
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is True
        assert music_bot.debug_settings.enabled(None) is True
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--disable")
        # The override wins over the default, and only for this guild.
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is False
        assert music_bot.debug_settings.enabled(424242424242424242) is True

    async def test_dm_toggle_explains_the_scope_instead_of_toggling(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild = None
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "per server" in embed.description
        assert music_bot.debug_settings._overrides == {}

    async def test_dm_status_still_renders(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild = None
        await command_callback(MusicBot.debug)(music_bot, mock_ctx)
        embed = mock_ctx.channel.send.call_args.kwargs["embeds"][0]
        assert embed.title == "🐞 Debug snapshot"

    async def test_bad_argument_answers_with_usage(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="enable")
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "--enable" in embed.description
        assert music_bot.debug_settings._overrides == {}

    async def test_collection_failure_becomes_an_error_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The command body's try/except → _command_error, like every other
        command: a broken snapshot must not surface as a silent no-reply."""
        with patch("src.debug.run_debug_dashboard", side_effect=RuntimeError("boom")):
            await command_callback(MusicBot.debug)(music_bot, mock_ctx)
        # The failure reply still goes through ctx.send — _command_error is not a
        # dashboard, so it keeps the ordinary NP-host-aware path.
        embed = mock_ctx.send.call_args[1]["embed"]
        assert embed.title == "Command failed"


class TestDebugTogglePermission:
    """Reading `-debug` is open to everyone; WRITING the toggle is not. It is
    guild-wide and every member sees the result on every reply."""

    @staticmethod
    def _plain_member(mock_ctx: MagicMock) -> None:
        mock_ctx.author.guild_permissions.manage_guild = False
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)

    async def test_a_member_without_manage_server_cannot_toggle(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._plain_member(mock_ctx)
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        assert music_bot.debug_settings._overrides == {}
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is False

    async def test_the_refusal_names_the_permission(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._plain_member(mock_ctx)
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Manage Server" in embed.description

    async def test_a_plain_member_cannot_turn_it_OFF_either(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Both directions: a member who could disable it could also hide a
        moderator's deliberate enable."""
        music_bot.debug_settings._overrides[mock_ctx.guild.id] = True
        self._plain_member(mock_ctx)
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--disable")
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is True

    async def test_manage_server_may_toggle(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.author.guild_permissions.manage_guild = True
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is True

    async def test_the_bot_owner_may_toggle_without_manage_server(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.author.guild_permissions.manage_guild = False
        mock_ctx.bot.is_owner = AsyncMock(return_value=True)
        await command_callback(MusicBot.debug)(music_bot, mock_ctx, arg="--enable")
        assert music_bot.debug_settings.enabled(mock_ctx.guild.id) is True

    async def test_reading_the_snapshot_stays_open_to_a_plain_member(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._plain_member(mock_ctx)
        mock_ctx.guild.voice_client = None
        await command_callback(MusicBot.debug)(music_bot, mock_ctx)
        embed = mock_ctx.channel.send.call_args.kwargs["embeds"][0]
        assert embed.title == "🐞 Debug snapshot"


class TestDebugInputs:
    """What the cog HANDS the snapshot. Everything the renderer shows is decided
    here, including who is allowed to see it."""

    async def test_reports_the_cogs_actual_state(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Any
    ) -> None:
        guild_id = mock_ctx.guild.id
        player = MagicMock()
        music_bot.mps = {guild_id: player, 999: MagicMock()}
        music_bot.redis = fake_redis
        music_bot.debug_settings._overrides[guild_id] = True

        inputs = await music_bot._debug_inputs(mock_ctx)

        assert inputs.debug_enabled is True
        assert inputs.debug_overridden is True
        assert inputs.players == 2
        assert inputs.player is player
        assert inputs.redis is fake_redis
        assert inputs.store is not None and inputs.store.guild_id == guild_id

    async def test_a_guild_with_no_override_is_not_marked_overridden(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        inputs = await music_bot._debug_inputs(mock_ctx)
        assert inputs.debug_overridden is False

    async def test_operator_follows_is_owner(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.bot.is_owner = AsyncMock(return_value=True)
        assert (await music_bot._debug_inputs(mock_ctx)).operator is True
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)
        assert (await music_bot._debug_inputs(mock_ctx)).operator is False

    async def test_an_unreachable_owner_check_denies_rather_than_discloses(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """is_owner() RAISES when application_info() fails — it does not return
        False. A diagnostic must not open up because Discord blinked."""
        mock_ctx.bot.is_owner = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=503), "boom")
        )
        inputs = await music_bot._debug_inputs(mock_ctx)
        assert inputs.operator is False
        assert inputs.default_password is None

    @pytest.mark.parametrize("using_default", [True, False])
    async def test_a_non_owner_gets_no_password_row_in_either_state(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        using_default: bool,
    ) -> None:
        """Symmetry is the point. Suppressing only the True case makes the row's
        ABSENCE the answer: no row + archive on == the compose default is in use."""
        monkeypatch.setattr(
            "src.musicbot.using_default_postgres_password", lambda: using_default
        )
        monkeypatch.setattr("src.musicbot.history_archive_enabled", lambda: True)
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)
        assert (await music_bot._debug_inputs(mock_ctx)).default_password is None

    async def test_an_owner_gets_the_password_row(
        self, music_bot: MusicBot, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.musicbot.using_default_postgres_password", lambda: True
        )
        monkeypatch.setattr("src.musicbot.history_archive_enabled", lambda: True)
        mock_ctx.bot.is_owner = AsyncMock(return_value=True)
        assert (await music_bot._debug_inputs(mock_ctx)).default_password is True

    async def test_a_dm_has_no_guild_scoped_state(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Any
    ) -> None:
        mock_ctx.guild = None
        music_bot.redis = fake_redis
        inputs = await music_bot._debug_inputs(mock_ctx)
        assert inputs.player is None
        assert inputs.store is None
        assert inputs.debug_overridden is False


class TestDebugObservesWithoutCreating:
    """debug.py's module docstring promises OBSERVATION-ONLY. cog_before_invoke
    calls get_mp(), which CREATES a player — so -debug has to be exempt, or the
    snapshot reports a player it manufactured and starts a restore on an idle guild."""

    async def test_the_real_command_carries_the_flag(self) -> None:
        """The exemption is driven off extras, so the flag has to be ON the command.
        Asserting it here rather than restating the literal keeps the test from
        passing on a command that lost it."""
        assert MusicBot.debug.extras.get("observation_only") is True
        assert MusicBot.play.extras.get("observation_only") is None

    async def test_analytics_carries_the_flag_too(self) -> None:
        """-analytics reads the archive and never touches voice. Without the flag
        cog_before_invoke builds a player for it, which starts _restore_state() and
        then parks on the 300s gate before tearing itself down — observed in the
        deployed bot as a gate timeout logged under command=analytics."""
        assert MusicBot.analytics.extras.get("observation_only") is True

    async def test_debug_does_not_create_a_player(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.command.extras = {"observation_only": True}
        mock_ctx.guild.voice_client = None
        music_bot.get_mp = MagicMock()
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_not_called()

    async def test_other_commands_still_get_their_player(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.command.extras = {}
        mock_ctx.guild.voice_client = None
        music_bot.get_mp = MagicMock()
        await music_bot.cog_before_invoke(mock_ctx)
        music_bot.get_mp.assert_called_once()

    async def test_an_idle_guild_reports_no_player(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The observable consequence: `player no` is reachable. It was not before
        — get_mp() had always populated mps by the time the snapshot read it."""
        music_bot.mps = {}
        inputs = await music_bot._debug_inputs(mock_ctx)
        assert inputs.player is None
        assert inputs.players == 0


class TestTimestampWarningReachesTheUser:
    @staticmethod
    def _bad_ts_source() -> Any:
        return parse_url("https://youtu.be/a?t=bogus")

    async def test_it_rides_the_queued_song_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        mp.queue.qsize = MagicMock(return_value=3)  # something already queued
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, warning=timestamp_warning(self._bad_ts_source())
        )

        # The card is the player's to build now; the cog's job is handing the
        # warning over. That it lands under the ETA is asserted on the builder.
        assert "bogus" in mp.build_queued_song_embed.call_args.kwargs["warning"]

    async def test_it_gets_its_own_message_when_no_embed_is_sent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """An idle bot plays the first song immediately and sends no "Queued
        song" embed at all. Riding that embed alone would drop the warning in
        the most ordinary case there is."""
        mp = mock_mp()
        mp.queue.qsize = MagicMock(return_value=0)
        mock_ctx.voice_client = connected_vc()
        mock_ctx.voice_client.is_playing = MagicMock(return_value=False)
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, warning=timestamp_warning(self._bad_ts_source())
        )

        mp.build_queued_song_embed.assert_not_called()
        sent = [c.kwargs["embed"] for c in mock_ctx.send.await_args_list]
        assert any("bogus" in (e.description or "") for e in sent)
