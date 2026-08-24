"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""

from src.musicplayer import MusicPlayer
import redis.asyncio as aioredis
import asyncio
import contextlib
import orjson
from types import SimpleNamespace
from contextlib import AbstractContextManager
from typing import Any, Optional, cast
from collections.abc import AsyncIterator, Coroutine, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands
from redis.asyncio import Redis

import src.debug as debug_mode
from src.config import SpotifyStatus
from src.guild_history import GuildHistory
from src.guild_queue import QueueItem, RemoveMode, RemoveOutcome, item_label
from src.guild_state import Analytics, HistoryEntry
from src.musicbot import (
    _echo,
    _rebase_positions,
    HISTORY_MAX_LIMIT,
    RESTORE_WAIT_SECS,
    EmptyPlaylistError,
    HistoryFlags,
    MusicBot,
    PlaylistIndexError,
    ResolvedSpotifyPlaylist,
    ResolvedYoutubePlaylist,
    SpotifyDisabledError,
    _front_insert_depth,
)
from src.play_placement import (
    PlaceResult,
    Placement,
    PlayArgs,
    PlayMode,
    PlayRequest,
    _GuildPlays,
    check_voice_permissions,
    join_succeeded,
    play_key,
    play_takes_the_queue,
    split_play_args,
)
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import EMBED_FIELD_LIMIT
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
    described,
    make_mock_task,
    mocked,
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


class TestJoinSucceeded:
    """The check both cold-start commands gate their insert on. Its whole reason to
    exist is the still-connecting case: a type-only check passes there and hands the
    loop a client vc.play() raises on, once per restored song."""

    @staticmethod
    def _ctx(voice_client: object) -> MagicMock:
        ctx = MagicMock(spec=commands.Context)
        ctx.voice_client = voice_client
        return ctx

    def test_connected_client_succeeds(self) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_connected.return_value = True
        assert join_succeeded(self._ctx(vc)) is True

    def test_still_connecting_client_fails(self) -> None:
        # discord.py registers the client on the guild BEFORE the handshake, so this
        # is a real state a concurrent cold -play leaves behind — not a mock artifact.
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_connected.return_value = False
        assert join_succeeded(self._ctx(vc)) is False

    def test_absent_client_fails(self) -> None:
        # join swallows its own failures, so a failed join arrives as None.
        assert join_succeeded(self._ctx(None)) is False

    def test_non_voice_client_fails(self) -> None:
        assert join_succeeded(self._ctx(MagicMock())) is False


class TestCheckVoicePermissions:
    def test_rejects_non_member_user(self) -> None:
        user = MagicMock(spec=discord.User)
        assert check_voice_permissions(user, None, "play") is not None

    def test_rejects_member_not_in_voice_channel(self) -> None:
        member = MagicMock(spec=discord.Member)
        member.voice = None
        assert check_voice_permissions(member, None, "play") is not None

    def test_rejects_wrong_voice_channel_for_non_play(self) -> None:
        member = MagicMock(spec=discord.Member)
        channel_a = MagicMock()
        channel_b = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel_a
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel_b
        assert check_voice_permissions(member, vc, "skip") is not None

    def test_allows_play_in_different_channel(self) -> None:
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()  # different from member's channel — OK for play
        assert check_voice_permissions(member, vc, "play") is None

    def test_passes_valid_member_in_correct_channel(self) -> None:
        member = MagicMock(spec=discord.Member)
        channel = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel
        assert check_voice_permissions(member, vc, "skip") is None

    def test_passes_when_no_voice_client(self) -> None:
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        assert check_voice_permissions(member, None, "skip") is None

    def test_rejects_an_interjecting_play_in_a_different_channel(self) -> None:
        """-play's exemption is for QUEUEING into a session running elsewhere,
        which costs its listeners nothing. An interjection STOPS what that channel
        is hearing, so it is gated like every other command — otherwise a member
        in channel B can cut the song for a room they are not in."""
        member = MagicMock(spec=discord.Member)
        member.voice = MagicMock()
        member.voice.channel = MagicMock()
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock()  # not the member's channel
        assert (
            check_voice_permissions(member, vc, "play", queue_control=True) is not None
        )

    def test_an_interjecting_play_in_the_same_channel_is_fine(self) -> None:
        member = MagicMock(spec=discord.Member)
        channel = MagicMock()
        member.voice = MagicMock()
        member.voice.channel = channel
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel
        assert check_voice_permissions(member, vc, "play", queue_control=True) is None


class TestPlayTakesTheQueue:
    """What the voice gate reads to decide whether a -play is queue control.

    It runs as a before_invoke hook, which discord.py calls AFTER parsing —
    Command.prepare() runs _parse_arguments and then call_before_hooks — so the
    parsed argument is available in ctx.kwargs by then."""

    @staticmethod
    def _ctx(url: str) -> MagicMock:
        ctx = MagicMock()
        ctx.kwargs = {"url": url}
        return ctx

    @staticmethod
    def _vc(*, paused: bool, channel: Optional[MagicMock] = None) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_paused.return_value = paused
        # Explicit: check_voice_permissions compares this against the author's,
        # and an unset one on a spec'd mock raises rather than answering.
        vc.channel = (
            channel if channel is not None else MagicMock(spec=discord.VoiceChannel)
        )
        return vc

    def test_no_voice_client_never_interjects(self) -> None:
        assert play_takes_the_queue(self._ctx("--now song"), None) is False

    def test_the_flag_says_so(self) -> None:
        assert (
            play_takes_the_queue(self._ctx("--now song"), self._vc(paused=False))
            is True
        )

    def test_a_plain_play_appends(self) -> None:
        assert play_takes_the_queue(self._ctx("song"), self._vc(paused=False)) is False

    def test_the_next_flag_says_so_too(self) -> None:
        """`--next` interrupts nothing, but it decides what the channel hears when
        the current song ends — which is queue control, the thing -skip, -shuffle,
        -remove and -clear are all gated on. -play's exemption exists because
        APPENDING costs the other channel's listeners nothing, and this does not
        append."""
        assert (
            play_takes_the_queue(self._ctx("--next song"), self._vc(paused=False))
            is True
        )

    def test_a_paused_song_interjects_without_the_flag(self) -> None:
        """-play on a paused song interrupts it to bring it back playing, so it
        has always been an interjection — and has always carried -play's
        exemption. Closing that is why this reads the pause state too."""
        assert play_takes_the_queue(self._ctx("song"), self._vc(paused=True)) is True

    def test_a_trailing_flag_is_not_the_flag(self) -> None:
        """Same leading-token rule as everywhere else: the gate must agree with
        the body about what counts, or one refuses what the other would append."""
        assert (
            play_takes_the_queue(self._ctx("song --now"), self._vc(paused=False))
            is False
        )

    def test_a_command_with_no_url_argument_falls_out(self) -> None:
        ctx = MagicMock()
        ctx.kwargs = {}
        assert play_takes_the_queue(ctx, self._vc(paused=False)) is False


class TestValidateCommandsGatesQueueControl:
    """The hook wires the two halves together. Both are tested apart above, and
    both stay green if the hook stops passing the flag — so this is what proves a
    cross-channel `-p --now` / `-p --next` is actually refused end to end."""

    @staticmethod
    def _ctx(url: str, *, same_channel: bool) -> MagicMock:
        ctx = MagicMock()
        ctx.kwargs = {"url": url}
        ctx.command.name = "play"
        channel = MagicMock()
        ctx.author = MagicMock(spec=discord.Member)
        ctx.author.voice = MagicMock()
        ctx.author.voice.channel = channel
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = channel if same_channel else MagicMock()
        vc.is_paused.return_value = False
        ctx.voice_client = vc
        ctx.send = AsyncMock()
        return ctx

    @pytest.mark.parametrize("flag", ["--now", "--next"])
    async def test_a_cross_channel_queue_control_is_refused(
        self, music_bot: MusicBot, flag: str
    ) -> None:
        ctx = self._ctx(f"{flag} song", same_channel=False)
        with pytest.raises(commands.CommandError):
            await music_bot.validate_commands(ctx)
        assert "already being used" in described(ctx.send.call_args.kwargs["embed"])

    async def test_a_cross_channel_plain_play_is_still_allowed(
        self, music_bot: MusicBot
    ) -> None:
        """The exemption survives for queueing, which is what it was for."""
        ctx = self._ctx("song", same_channel=False)
        await music_bot.validate_commands(ctx)
        ctx.send.assert_not_awaited()

    @pytest.mark.parametrize("flag", ["--now", "--next"])
    async def test_a_same_channel_queue_control_is_allowed(
        self, music_bot: MusicBot, flag: str
    ) -> None:
        ctx = self._ctx(f"{flag} song", same_channel=True)
        await music_bot.validate_commands(ctx)
        ctx.send.assert_not_awaited()


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
        source = parse_input("never gonna give you up")
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
        source = parse_input(url)
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
        source = parse_input(url)
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
        source = parse_input(url)
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
        source = parse_input(url)
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
        # keep_first_only is the interjection's; -play enqueues the whole tail.
        assert len(result.tracks) == 3
        # The ask time is untouched by the slice — one instant for the command.
        assert all(t.analytics.queued_at == 1752530000.5 for t in result.tracks)

    async def test_playlist_index_rebase_preserves_a_nonzero_base(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The rebase subtracts the dropped count, it does not zero the field: a
        playlist queued behind two songs still waits behind them."""
        url = "https://www.youtube.com/watch?v=v2&list=PLabc&index=3"
        source = parse_input(url)
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
        source = parse_input(url)
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
        source = parse_input(url)
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
        """The same guard the interjection path already had: a playlist that resolves to
        nothing is an error, not a successful enqueue of zero songs."""
        url = "https://www.youtube.com/playlist?list=PLabc"
        source = parse_input(url)
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
        source = parse_input(url)
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
        source = parse_input(url)
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

    async def test_interjection_honours_the_playlist_index(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """An interjection plays the track the link was copied at, not track 1."""
        url = "https://www.youtube.com/watch?v=v2&list=PLabc&index=3"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 5)
        with patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)):
            head, rest = await music_bot._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN
            )
        assert head.title == "T2"
        # The tracks after it come too, in order.
        assert [queue_object(item).title for item in rest] == ["T3", "T4"]
        notice = mock_ctx.send.await_args.kwargs["embed"].description
        assert "#3" in notice

    async def test_interjection_index_past_the_end_reports_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The interjection path shares the guard, and its own error path renders the same
        embed under its own title."""
        url = "https://www.youtube.com/watch?v=v9&list=PLabc&index=9"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 3)
        with (
            patch("src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)),
            pytest.raises(PlaylistIndexError) as excinfo,
        ):
            await music_bot._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN
            )

        await music_bot._command_error(
            mock_ctx, excinfo.value, title="Failed to play song now"
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert embed.title == "Failed to play song now"
        assert "**#9**" in embed.description
        assert "**3 songs**" in embed.description

    async def test_interjection_spotify_playlist_bypasses_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # _resolve_interjection_source resolves both playlist shapes directly, so a
        # token passed only from queue_source would leave these two unclassified.
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None
        music_bot.spotify.playlist = AsyncMock(return_value=["Song A", "Song B"])
        fake_qobj = QueueObject("https://yt.com/v=1", "Song A", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN
            )
        assert self._passed_query_source(spy) == "spotify.com"

    async def test_interjection_youtube_playlist_bypasses_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        url = "https://www.youtube.com/playlist?list=PLabc"
        source = parse_input(url)
        tracks = [QueueObject("https://yt.com/v=1", "T", mock_ctx.author)]
        spy = AsyncMock(return_value=tracks)
        with patch("src.musicbot.YTDL.yt_playlist", new=spy):
            await music_bot._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN
            )
        assert self._passed_query_source(spy) == "youtube.com"

    async def test_interjection_indexed_playlist_rebases_every_kept_track(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The head lands at 0 — the depth an interjection actually has — and the
        tracks behind it count up from there. Without the rebase an `&index=4` link
        would file every track three deeper than it played."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4"
        source = parse_input(url)
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
            head, rest = await music_bot._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN
            )

        assert head is tracks[3]
        assert head.analytics.queue_position == 0
        assert [queue_object(item).analytics.queue_position for item in rest] == [1, 2]

    async def test_interjection_analytics_is_depth_zero(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # An interjection plays immediately by definition, so it reads no
        # queue depth at all — and its queued_at is still the ask time.
        url = "https://www.youtube.com/watch?v=abc"
        source = parse_input(url)
        fake_qobj = QueueObject(url, "Song", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.musicbot.YTDL.yt_source", new=spy):
            await music_bot._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN
            )
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
            _admit(music_bot, mock_ctx, mp),
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
            _admit(music_bot, mock_ctx, mp),
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
            _admit(music_bot, mock_ctx, mp),
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
            _admit(music_bot, mock_ctx, mp),
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
            _admit(music_bot, mock_ctx, mp),
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
            _admit(music_bot, mock_ctx, mp),
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
            _admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put.assert_awaited_once()
        _, call_kwargs = mp.queue_put.call_args
        assert call_kwargs.get("prefetch") is False


class TestJoinChannelPersistence:
    async def test_join_writes_channel_ids_to_redis(
        self,
        music_bot_with_redis: MusicBot,
        mock_ctx: MagicMock,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:
        """Calling join should persist voice and text channel IDs to Redis."""
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 777000000000000001
        voice_channel.connect = AsyncMock()
        mock_ctx.author.voice.channel = voice_channel
        mock_guild.change_voice_state = AsyncMock()
        mock_guild.voice_client = None

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.id = 777000000000000002
        mock_ctx.channel = text_channel

        mp = MagicMock()
        mp.store = MagicMock()
        mp.store.set_connection = AsyncMock()
        music_bot_with_redis.mps[mock_guild.id] = mp

        # join is a @commands.command — call the underlying callback directly.
        mock_ctx.voice_client = None  # bot not yet in channel
        with (
            patch.object(discord.VoiceChannel, "connect", new=AsyncMock()),
            patch.object(mock_ctx, "invoke", new=AsyncMock()),
        ):
            music_bot_with_redis.get_mp = MagicMock(return_value=mp)
            await command_callback(MusicBot.join)(music_bot_with_redis, mock_ctx)

        mp.store.set_connection.assert_awaited_once_with(
            voice_channel.id, text_channel.id
        )
        # Voice is up — a queue persisted by a previous -stop resumes.
        mp.open_playback_gate.assert_called_once()


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
        import inspect

        missing = []
        for cmd in music_bot.get_commands():
            src = inspect.getsource(cmd.callback)
            header = src.split("async def ", 1)[0]
            if f'start_as_current_span("bot.{cmd.name}")' not in header:
                missing.append(cmd.name)

        assert missing == [], f"commands with no span: {missing}"

    async def test_cleanup_retires_the_player_before_its_first_await(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Verdict ① of _place reads MusicPlayer.retired, and cleanup() is its only
        producer. Every test of that verdict sets the flag by hand, so deleting
        this call — or moving it past the teardown gather — leaves a -play that
        resolved across a -stop placing into a torn-down player: the entry lands in
        the Redis mirror alone and the next restore resurrects it."""
        order: list[str] = []
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mp.mark_retired = MagicMock(side_effect=lambda: order.append("retire"))

        async def _cancel(task: Any) -> None:
            order.append("await")

        mock_guild.voice_client = None
        with patch("src.musicbot.cancel_task", new=_cancel):
            await music_bot.cleanup(mock_guild)

        mp.mark_retired.assert_called_once()
        assert order and order[0] == "retire", order

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


class TestStopCommand:
    async def test_stop_adds_wave_reaction(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        music_bot.cleanup = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("discord.utils.get", return_value=vc):
            await command_callback(MusicBot.stop)(music_bot, mock_ctx)
        mock_ctx.message.add_reaction.assert_awaited_once_with("👋")

    async def test_stop_calls_cleanup(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        music_bot.cleanup = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        with patch("discord.utils.get", return_value=vc):
            await command_callback(MusicBot.stop)(music_bot, mock_ctx)
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)

    async def test_stop_does_not_call_skip(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """stop must not invoke skip — skip fires voice_client.stop() which triggers
        the after callback and gives the playback loop a window to start the next song.
        """
        music_bot.cleanup = AsyncMock()
        music_bot.skip = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        with patch("discord.utils.get", return_value=vc):
            await command_callback(MusicBot.stop)(music_bot, mock_ctx)
        music_bot.skip.assert_not_called()

    async def test_stop_noop_when_no_voice_client(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        music_bot.cleanup = AsyncMock()
        with patch("discord.utils.get", return_value=None):
            await command_callback(MusicBot.stop)(music_bot, mock_ctx)
        music_bot.cleanup.assert_not_awaited()


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
        mp._channel = old_channel
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
        mp._channel = channel  # same object → no change
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
        mp._channel = old_channel
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


class TestPauseCommand:
    async def test_pauses_when_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_awaited_once_with(vc)

    async def test_sends_confirmation_embed_when_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        embed = discord.Embed(title="⏸️ Paused")
        mp.build_pause_confirmation_embed = MagicMock(return_value=embed)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once_with(embed=embed)

    async def test_no_confirmation_sent_when_embed_is_none(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        mock_ctx.voice_client = vc
        mock_ctx.message.add_reaction = AsyncMock()
        mp = MagicMock()
        mp.pause = AsyncMock()
        mp.build_pause_confirmation_embed = MagicMock(return_value=None)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mock_ctx.send.assert_not_awaited()

    async def test_noop_when_not_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        mock_ctx.voice_client = vc
        mp = MagicMock()
        mp.pause = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.pause)(music_bot, mock_ctx)
        mp.pause.assert_not_awaited()


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
        mock_ctx.voice_client = _playing_vc(mock_ctx)
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
            mock_ctx.voice_client = _paused_vc(mock_ctx)

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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
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

        with _no_typing():
            await command_callback(MusicBot.resume)(music_bot, mock_ctx)

        mock_ctx.invoke.assert_not_awaited()
        sent = mock_ctx.send.await_args.kwargs["embed"]
        assert "Can't reach the queue store" in sent.description
        assert "no queue was left" not in sent.description


class TestVolumeCommand:
    @staticmethod
    def _description(ctx: MagicMock) -> str:
        return cast(str, ctx.send.await_args.kwargs["embed"].description)

    async def test_sets_player_volume(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.store.set_volume = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        mp.store.set_volume.assert_awaited_once_with(0.5)
        mock_ctx.send.assert_awaited()

    async def test_volume_persists_nothing_without_store(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.store = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        assert mp.volume == 0.5
        mock_ctx.send.assert_awaited()
        assert "could not be saved" in self._description(mock_ctx)

    async def test_a_successful_write_says_it_is_saved(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.store.set_volume = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        description = self._description(mock_ctx)
        assert "saved for this server" in description
        assert "could not be saved" not in description

    async def test_a_failed_write_is_reported_not_claimed(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """set_volume returns False when the write did not land, and the help
        promises the level survives a restart. Confirming it anyway is the exact
        failure the debug toggle fixed: a setting that quietly reverts."""
        mp = MagicMock()
        mp.store.set_volume = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "50")
        description = self._description(mock_ctx)
        assert "could not be saved" in description
        # Still applied to this process's player, as the debug toggle is.
        assert mp.volume == 0.5

    async def test_rejects_non_numeric_string(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "loud")
        mock_ctx.send.assert_awaited()

    async def test_rejects_out_of_range(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.volume)(music_bot, mock_ctx, "150")
        mock_ctx.send.assert_awaited()


def _history_entries(n: int) -> list[HistoryEntry]:
    """n entries, oldest-first (the order GuildHistory stores them)."""
    return [
        HistoryEntry(
            title=f"Song {i}",
            webpage_url=f"https://yt.com/v={i}",
            duration_secs=200,
            played_secs=200,
            requester_id=i + 1,
            requester_name=f"user{i}",
            played_at=1000.0 + i,
        )
        for i in range(n)
    ]


def _flags(limit: int = 10) -> SimpleNamespace:
    """Stand-in for a parsed HistoryFlags (FlagConverter can't be constructed
    directly; the command body only reads .limit)."""
    return SimpleNamespace(limit=limit)


class TestHistoryCommand:
    def _mp_with_history(self, music_bot: MusicBot, entries: Any) -> MagicMock:
        mp = MagicMock()
        history = GuildHistory(None, on_outbox_push=lambda: None)
        # No store, so the in-memory deque is the whole read path — these tests are
        # about rendering (ordering, chunking, limits, fields), not which leg served
        # them; that is TestHistoryReadsRedis, where the legs DISAGREE. Beware the
        # trap the old Postgres double hit: recent() swallows every error, so a
        # broken fixture reads exactly like graceful degradation and all 14 tests
        # passed against the cache while covering none of the read path.
        history.restore(list(reversed(entries)))
        mp.history = history
        music_bot.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_empty_history_sends_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._mp_with_history(music_bot, [])
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "No songs have been played yet" in embed.description

    async def test_shows_most_recent_newest_first(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._mp_with_history(music_bot, _history_entries(15))
        await command_callback(MusicBot.history)(
            music_bot, mock_ctx, flags=_flags(limit=3)
        )
        mock_ctx.send.assert_awaited_once()
        embeds = mock_ctx.send.call_args[1]["embeds"]
        # Most recent 3 of 15, newest first — not the oldest 3.
        assert [e.title for e in embeds] == [
            "1. Song 14",
            "2. Song 13",
            "3. Song 12",
        ]

    async def test_default_limit_chunks_at_eight_embeds(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # 10 embeds + the ≤2-embed NP block must stay under Discord's 10-embed
        # cap, so the response is chunked 8 + 2, every chunk via ctx.send.
        self._mp_with_history(music_bot, _history_entries(12))
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        assert mock_ctx.send.await_count == 2
        first, second = mock_ctx.send.await_args_list
        assert len(first.kwargs["embeds"]) == 8
        assert len(second.kwargs["embeds"]) == 2

    async def test_limit_smaller_than_history_returns_that_many(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._mp_with_history(music_bot, _history_entries(5))
        await command_callback(MusicBot.history)(
            music_bot, mock_ctx, flags=_flags(limit=50)
        )
        embeds = mock_ctx.send.call_args[1]["embeds"]
        assert len(embeds) == 5

    @pytest.mark.parametrize("bad_limit", [0, -3, 51])
    async def test_out_of_range_limit_rejected(
        self, music_bot: MusicBot, mock_ctx: MagicMock, bad_limit: int
    ) -> None:
        self._mp_with_history(music_bot, _history_entries(5))
        await command_callback(MusicBot.history)(
            music_bot, mock_ctx, flags=_flags(limit=bad_limit)
        )
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "--limit must be between 1 and 50" in embed.description
        mocked(music_bot.get_mp).assert_not_called()

    async def test_song_embeds_carry_thumbnail_and_metadata(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        entry = HistoryEntry(
            title="Rich Song",
            webpage_url="https://yt.com/v=rich",
            duration_secs=242,
            played_secs=225,
            requester_id=42,
            requester_name="Omkar",
            thumbnail="https://i.ytimg.com/t.jpg",
            played_at=1752530000.0,
        )
        self._mp_with_history(music_bot, [entry])
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embeds"][0]
        assert embed.thumbnail.url == "https://i.ytimg.com/t.jpg"
        lines = embed.description.splitlines()
        assert lines[0] == "https://yt.com/v=rich"
        assert lines[1] == "3:45 / 4:02 · requested by <@42> · <t:1752530000:f>"

    def test_flag_defaults(self) -> None:
        # -h with no flags must parse to limit=10.
        assert HistoryFlags.get_flags()["limit"].default == 10

    def test_max_limit_never_exceeds_the_redis_window(self) -> None:
        """The merge's completeness argument, pinned rather than commented: the
        union holds the true newest `limit` only while the Redis window is at least
        as deep as anything this command accepts. Raising HISTORY_MAX_LIMIT alone
        raises nothing — it silently starts returning short pages."""
        assert HISTORY_MAX_LIMIT <= HISTORY_CACHE_LIMIT

    @pytest.mark.parametrize(
        "name", ["history", "ping", "leaderboard", "debug", "resume"]
    )
    def test_the_command_is_capped_at_one_render_per_guild(self, name: str) -> None:
        """`-history` is the heaviest send in the bot (up to 8 song embeds plus the
        NP block), so unbounded concurrent renders rate-limit a guild out of its own
        channel — and deleting the decorator that prevents it left the suite green.
        `wait=False` is half the point: queueing the extra invocations still issues
        every send, so they must be declined outright. `-leaderboard` carries it for
        a second reason: it draws on the same Postgres pool as the drainer, and
        `-debug` for a third: a Postgres stats query, a Prometheus round trip and
        two Redis reads, live-editing under an 8s deadline. `-resume` for a fourth:
        two racing on a disconnected bot both read `voice_client is None`, so
        validate_commands' "already being used in channel X" check cannot fire for
        either — both join, and the second MOVES the bot to its own author's channel.
        `-play` is guarded for a fifth reason — two concurrent invocations against
        a PAUSED song both read it live and both park a resume tail, so one play
        comes back twice — but is NOT in this list: it needs a bucket per
        PLACEMENT, which the decorator cannot express, because prepare() acquires
        before the argument is parsed. See TestPlayConcurrencyBuckets.

        command_callback() strips decorators everywhere else in this file, so this
        is the only place any of these guards is reachable at all."""
        guard = getattr(MusicBot, name)._max_concurrency
        assert guard is not None
        assert guard.number == 1
        assert guard.per is commands.BucketType.guild
        assert guard.wait is False

    def test_help_copy_states_the_real_retention_window(self) -> None:
        """The user-facing copy must name the window the command actually keeps: 50
        is the retention cap AND the display cap, and the copy once promised
        permanent retention in the configuration that now ships by default. Pins
        that the constant stays interpolated, so raising the window cannot leave the
        copy quoting the old one; the negative assertion names the false claim."""
        help_text = MusicBot.history.help
        assert help_text is not None
        assert str(HISTORY_MAX_LIMIT) in help_text
        assert "permanently" not in help_text
        # The archive caveat belongs in NOTES, and it is the one place the word
        # is honest: Postgres retention really is permanent when enabled.
        note = (MusicBot.history.extras or {}).get("note", "")
        assert "permanently" in note


class TestHistoryReadsRedis:
    """That `-history` renders the Redis leg at all, asserted at the command level.

    Every failure inside recent() degrades to the leg below by design, so a command
    test can render a perfectly correct embed off the in-memory deque while the Redis
    read raises on every invocation. Making the legs DISAGREE is the only way to tell
    which one reached the user. Postgres is not on this read path at all."""

    async def test_the_rendered_songs_come_from_redis_not_just_the_cache(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Any
    ) -> None:

        store = GuildRedisStore(fake_redis, guild_id=1)
        stored = _history_entries(2)  # Song 0 (t=1000), Song 1 (t=1001)
        for entry in stored:
            await store.push_history(entry)
        # Older than both, and present only in the deque — so its position in
        # the output says which legs ran: absent means the Redis leg never got
        # read, first means the cache won, last means both legs merged and
        # sorted, which is the contract.
        cache_only = HistoryEntry(
            title="CACHE ONLY",
            webpage_url="https://yt.com/v=cache",
            duration_secs=1,
            played_secs=1,
            requester_id=1,
            requester_name="u",
            played_at=1.0,
        )
        mp = MagicMock()
        history = GuildHistory(store, on_outbox_push=lambda: None)
        history.restore([cache_only])
        mp.history = history
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        titles = [e.title for e in mock_ctx.send.call_args[1]["embeds"]]
        assert titles == ["1. Song 1", "2. Song 0", "3. CACHE ONLY"]

    async def test_a_broken_store_double_is_now_visible(
        self, music_bot: MusicBot, mock_ctx: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The guard on the guard: a stand-in without get_history stays green
        # forever, so this pins that the same mistake now leaves a visible warning
        # instead of a false claim that the read path is covered.
        mp = MagicMock()
        mp.history = GuildHistory(cast(Any, object()), on_outbox_push=lambda: None)
        mp.history.restore(_history_entries(1))
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        assert "redis read failed" in caplog.text
        assert "AttributeError" in caplog.text


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


class TestClearCommand:
    async def test_sends_empty_message_when_queue_already_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=[])
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        mp.queue_clear.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "The queue is already empty."
        )

    async def test_an_unrestored_queue_is_never_cleared(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """clear() destroys the Redis mirror while reading the IN-MEMORY display,
        so against an unrestored player it deletes a saved queue it cannot see —
        and an interjection stack loses its rows too, because _flush_played records
        from that same empty display. validate_commands only requires the AUTHOR
        in voice, so a cold player is reachable."""
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=[])
        mp.wait_for_restore = AsyncMock(return_value=False)  # snapshot not read
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.clear)(music_bot, mock_ctx)

        mp.queue_clear.assert_not_awaited()
        assert "Still loading" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_sends_embed_with_cleared_songs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        cleared = ["Song A - https://yt.com/1", "Song B - https://yt.com/2"]
        mp = MagicMock()
        mp.queue_clear = AsyncMock(return_value=cleared)
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.message.add_reaction = AsyncMock()
        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        mp.queue_clear.assert_awaited_once()
        mock_ctx.message.add_reaction.assert_awaited_once_with("🗑️")
        call_kwargs = mock_ctx.send.call_args[1]
        embed = call_kwargs["embed"]
        assert "2 songs removed" in embed.title
        assert "Song A" in embed.description


def _admit(
    music_bot: MusicBot,
    ctx: MagicMock,
    mp: Any,
    *,
    mode: PlayMode = PlayMode.NORMAL,
    query: str = "test",
) -> PlayRequest:
    """Register a request as play() would, so a helper that inserts under the
    place lock can be called directly.

    A bare MagicMock player answers `retired` with a truthy mock, which place()
    reads as torn down; pin it the way _mock_mp() does."""
    if isinstance(mp, MagicMock) and isinstance(mp.retired, MagicMock):
        mp.retired = False
    return music_bot._plays.register(ctx, query=query, mp=mp, mode=mode)


def _no_typing() -> AbstractContextManager[MagicMock]:
    """Stub play()'s background_typing wrapper with an inert async CM: TestPlayCommand
    patches asyncio.create_task as a join-task spy, and without this the typing
    keepalive hits the same patch, polluting call counts and taking the fake join
    future. The wrapper itself is covered by TestBackgroundTyping."""
    return patch(
        "src.musicbot.background_typing",
        MagicMock(return_value=contextlib.nullcontext()),
    )


def _in_authors_channel(vc: MagicMock, ctx: Optional[MagicMock]) -> MagicMock:
    """Seat a voice-client double in the author's channel, or somewhere else.

    Queue control (`--now`, `--next`, a paused song) is gated on the bot being in
    the author's channel, at dispatch AND again at the insert. A double with no
    channel reads as "somewhere else" and is refused, so a test that means
    "allowed" has to say where the bot is."""
    vc.channel = (
        ctx.author.voice.channel
        if ctx is not None
        else MagicMock(spec=discord.VoiceChannel)
    )
    return vc


def _connected_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client, nothing playing — what a successful cold join leaves
    behind. is_connected is explicit: the cold path checks it, because discord.py
    registers the client on the guild before the handshake completes."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return _in_authors_channel(vc, ctx)


def _playing_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client, actively playing. Both flags must be set explicitly:
    an unstubbed is_paused() returns a truthy Mock, silently sending -play down the
    interjection branch instead of the append path."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return _in_authors_channel(vc, ctx)


def _paused_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client with a song parked paused. is_connected is explicit:
    -resume's rejoin checks it, and an auto-vivified one answers True by accident
    rather than by choice."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = True
    vc.is_connected.return_value = True
    return _in_authors_channel(vc, ctx)


def _mock_mp(qsize: int = 0) -> MagicMock:
    """MusicPlayer stand-in for the -play cold path, with the playback-gate
    hooks awaitable: play() takes defer_playback() as an async context manager
    and awaits wait_for_restore() before front-inserting."""
    mp = MagicMock()
    mp.defer_playback = MagicMock(return_value=contextlib.nullcontext())
    mp.wait_for_restore = AsyncMock(return_value=True)
    # Numeric, not auto-vivified: _abandon_cold_start COMPARES this, and a Mock
    # raises TypeError there rather than answering.
    mp.playback_holds = 1  # the hold this command itself takes
    # Explicit, not auto-vivified: _place() reads it, and a MagicMock is truthy.
    mp.retired = False
    mp.queue.generation = 0
    mp.repark_crashed_head = AsyncMock()
    # Awaitable, not auto-vivified: _interject_flow settles the prefetch BEFORE it
    # takes the place lock, and a bare Mock is not awaitable there.
    mp.settle_prefetch = AsyncMock()
    mp.queue_put_front = AsyncMock()
    mp.queue_put_next = AsyncMock()
    mp.queue_put = AsyncMock()
    mp.queue.qsize = MagicMock(return_value=qsize)
    # False, not auto-vivified: an unset Mock is TRUTHY, which would say a consumer
    # is holding a song on an idle bot — the state that decides both the front-insert
    # depth and whether the confirmation claims the song starts now.
    mp.queue.claim_outstanding = MagicMock(return_value=False)
    # A real title, not a Mock: the --next confirmation interpolates it into the
    # embed body, and a Mock renders as its repr there.
    mp.current_song = None
    # Numeric for the same reason as playback_holds: this lands in
    # Analytics.queue_position and rides to Postgres through HistoryEntry's
    # integer clamp, which a Mock raises on rather than answering.
    mp.enqueue_depth = MagicMock(return_value=qsize)
    # Numeric for that reason too: _cold_start_left_something_playable compares it
    # to decide whether a late refusal may disconnect the session, and an
    # auto-vivified Mock is truthy — which spares every teardown these tests pin.
    mp.queue.display_size = MagicMock(return_value=qsize)
    # Mirrors the real builder's contract: a notice only when the restore
    # actually left something in the queue (see build_resume_notice_embed).
    mp.build_resume_notice_embed = MagicMock(
        return_value=discord.Embed(title="❗ Resumed from queue") if qsize else None
    )
    return mp


# ── -play --now routing ───────────────────────────────────────────────────────


class TestNowFlagRouting:
    """The six rows of the branch matrix, over one command body.

    Two shallow commands became one branchy body, so this is what stops a later
    edit quietly swapping a leg. Every row also asserts _command_error was not
    awaited: the body wraps everything, so a row can otherwise pass on a
    TypeError from an under-configured mock and prove nothing."""

    @staticmethod
    def _vc(
        *, playing: bool = False, paused: bool = False, ctx: Optional[MagicMock] = None
    ) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = playing
        vc.is_paused.return_value = paused
        vc.is_connected.return_value = True
        return _in_authors_channel(vc, ctx)

    def _wire(
        self, music_bot: MusicBot, mock_ctx: MagicMock, *, live: bool
    ) -> MagicMock:
        mp = _mock_mp()
        mp.current_song = MagicMock() if live else None
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()
        music_bot._interject_flow = AsyncMock()
        music_bot._abandon_cold_start = AsyncMock()
        music_bot._command_error = AsyncMock()
        mock_ctx.invoke = AsyncMock()
        seams = MagicMock()
        seams.mp = mp
        seams.queue_source = music_bot.queue_source
        seams.enqueue_single = music_bot._enqueue_single
        seams.interject = music_bot._interject_flow
        seams.command_error = music_bot._command_error
        seams.abandon = music_bot._abandon_cold_start
        return seams

    @pytest.mark.parametrize(
        "flag,connected,playing,paused,interjects,resume_paused,enqueued_as",
        [
            # flag      connected playing paused  interjects  resume_paused  placement
            #
            # `enqueued_as` is None wherever no enqueue is expected at all, which
            # happens for two different reasons: an interjection never reaches
            # _enqueue_single, and a disconnected row abandons at join_succeeded
            # because the mocked join leaves no voice client.
            ("", False, False, False, False, None, None),
            ("", True, True, False, False, None, Placement.TAIL),
            ("", True, False, True, True, False, None),
            ("", True, False, False, False, None, Placement.TAIL),
            ("--now ", False, False, False, False, None, None),
            ("--now ", True, True, False, True, True, None),
            ("--now ", True, False, True, True, True, None),
            # Connected with nothing live: no song to interrupt, so `--now` cannot
            # interject, but it still jumps the queue. That state lasts every
            # song-resolve and the whole of a restored queue's first song.
            ("--now ", True, False, False, False, None, Placement.NEXT),
            # `--next` never interjects, on any row. The paused one is the carve-out
            # that matters: plain `-play` interjects there because the request would
            # otherwise be buried behind a paused song, and with `--next` it is not
            # buried — it IS next, so stopping the song the user chose to keep would
            # be the opposite of what they typed.
            ("--next ", False, False, False, False, None, None),
            ("--next ", True, True, False, False, None, Placement.NEXT),
            ("--next ", True, False, True, False, None, Placement.NEXT),
            ("--next ", True, False, False, False, None, Placement.NEXT),
        ],
    )
    async def test_the_branch_matrix(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        flag: str,
        connected: bool,
        playing: bool,
        paused: bool,
        interjects: bool,
        resume_paused: Optional[bool],
        enqueued_as: Optional[Placement],
    ) -> None:
        live = connected and (playing or paused)
        seams = self._wire(music_bot, mock_ctx, live=live)
        mock_ctx.voice_client = (
            self._vc(playing=playing, paused=paused) if connected else None
        )

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"{flag}never gonna give you up"
            )

        seams.command_error.assert_not_awaited()
        if interjects:
            seams.interject.assert_awaited_once()
            assert (
                seams.interject.await_args.kwargs.get("resume_paused", True)
                is resume_paused
            )
        else:
            seams.interject.assert_not_awaited()
            seams.queue_source.assert_awaited_once()
        if enqueued_as is None:
            seams.enqueue_single.assert_not_awaited()
        else:
            assert seams.enqueue_single.await_args.kwargs["placement"] is enqueued_as

    async def test_next_does_not_tear_the_player_down_when_the_restore_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The trap in sharing the cold path's restore guard.

        Every front insert has to wait out an in-flight restore — put_front LPUSHes
        the same Redis list restore_entries replays, so inserting against an unread
        snapshot double-queues the song. But the cold path answers a failed wait
        with _abandon_cold_start, which cancels the player's tasks and disconnects
        it. That is right for a join that never landed and catastrophic here: it
        would stop the music over a Redis blink.
        """
        seams = self._wire(music_bot, mock_ctx, live=True)
        seams.mp.wait_for_restore = AsyncMock(return_value=False)
        mock_ctx.voice_client = self._vc(playing=True, ctx=mock_ctx)

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next song"
            )

        seams.abandon.assert_not_awaited()
        seams.enqueue_single.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "saved queue" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_next_records_the_depth_it_actually_waits_behind(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """It goes to the front, so it waits behind the playing song and nothing
        else — never the queue depth enqueue_depth() would report. That number is
        written to Postgres once and never revisited."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        seams.mp.enqueue_depth = MagicMock(return_value=17)
        mock_ctx.voice_client = self._vc(playing=True, ctx=mock_ctx)
        # The real insert: the depth is minted there, not at the ask.
        music_bot._enqueue_single = MusicBot._enqueue_single.__get__(music_bot)

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next song"
            )

        queued = seams.mp.queue_put_next.await_args.args[0]
        assert queued.analytics.queue_position == 1
        seams.mp.enqueue_depth.assert_not_called()

    @pytest.mark.parametrize("flag", ["", "--now "])
    async def test_a_connected_but_idle_player_does_not_interject(
        self, music_bot: MusicBot, mock_ctx: MagicMock, flag: str
    ) -> None:
        """A live current_song is necessary but not sufficient.

        current_song outlives the song it names — the loop clears it after the
        song ends — so a voice client that is neither playing nor paused has
        nothing to interrupt. Without the playing-or-paused term, `--now` here
        would build a resume entry for a song that already finished and replay
        its final seconds. The matrix rows above all pair a live current_song
        with a live client, so they cannot see this."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        mock_ctx.voice_client = self._vc(playing=False, paused=False)

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"{flag}song"
            )

        seams.interject.assert_not_awaited()
        seams.queue_source.assert_awaited_once()
        seams.command_error.assert_not_awaited()

    async def test_a_paused_song_returns_playing_without_the_flag(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The ONE semantic difference between the two legs, and the only thing
        the merge could silently lose. `-play` on a paused song brings it back
        PLAYING — "-play means play" — where `--now` restores it paused."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        mock_ctx.voice_client = self._vc(paused=True, ctx=mock_ctx)

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="song")

        kwargs = seams.interject.await_args.kwargs
        assert kwargs["resume_paused"] is False
        # require_paused is the second difference: a -resume landing during the
        # 1-4s resolve removes the reason to interject, so that leg appends
        # instead. The --now leg has no such reason to lose.
        assert kwargs["require_paused"] is True

    async def test_the_flag_leg_does_not_require_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=True)
        mock_ctx.voice_client = self._vc(playing=True, ctx=mock_ctx)

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now song")

        kwargs = seams.interject.await_args.kwargs
        assert kwargs.get("require_paused", False) is False

    @pytest.mark.parametrize("live", [True, False])
    async def test_the_origin_never_carries_the_flag(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live: bool
    ) -> None:
        """What -remove matches on is the query without the flag, on BOTH the
        interject row and the ordinary row — the ordinary path passes origin=url
        at three separate call sites, and one is not a proxy for the others.

        A leak here fails silently: the enqueue succeeds, the reply is normal,
        and only `-remove never gonna give you up` later finds nothing."""
        seams = self._wire(music_bot, mock_ctx, live=live)
        mock_ctx.voice_client = (
            self._vc(playing=True, ctx=mock_ctx) if live else _connected_vc()
        )
        typed = "never gonna give you up"

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {typed}"
            )

        seams.command_error.assert_not_awaited()
        if live:
            # _interject_flow resolves its own source; it receives the query.
            assert seams.interject.await_args.args[1] == typed
        else:
            assert seams.queue_source.await_args.kwargs["origin"] == typed

    async def test_the_flag_leaves_a_link_parseable_as_a_link(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`-p --now <link>` must reach parse_url: with the flag stripped the link
        is one token, so it parses as a URL and not as a two-word search."""
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        url = "https://youtu.be/dQw4w9WgXcQ"

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        source = seams.queue_source.await_args.args[1]
        assert isinstance(source, YTSource)
        assert source.url == url
        assert source.ytsearch is None

    async def test_a_dash_typo_asks_instead_of_searching(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = _connected_vc(mock_ctx)

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="-now song")

        seams.queue_source.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "--now" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_the_flag_with_nothing_behind_it_asks(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = _connected_vc(mock_ctx)

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now")

        seams.queue_source.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "url" in mock_ctx.send.call_args.kwargs["embed"].description

    @pytest.mark.parametrize(
        "live,title",
        [(True, "Failed to play song now"), (False, "Failed to queue song")],
    )
    async def test_the_error_title_names_the_branch_not_the_flag(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live: bool, title: str
    ) -> None:
        """`-p --now x` on an idle bot queues like any other -play, so "failed to
        play song now" would describe an interjection that never happened."""
        seams = self._wire(music_bot, mock_ctx, live=live)
        mock_ctx.voice_client = (
            self._vc(playing=True, ctx=mock_ctx) if live else _connected_vc()
        )
        boom = RuntimeError("nope")
        seams.interject.side_effect = boom
        seams.queue_source.side_effect = boom

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now song")

        assert seams.command_error.await_args.kwargs["title"] == title


# ── -play argument parsing ────────────────────────────────────────────────────


class TestSplitPlayArgs:
    """`--now`/`--next` comes off the front of -play's argument, or not at all.

    The leading-token rule is the whole design: the remainder becomes both the
    YouTube search and the origin `-remove` matches on, so a flag stripped from the
    middle of a line would leave a value the user never typed."""

    @pytest.mark.parametrize(
        "argument,mode,query",
        [
            ("never gonna give you up", PlayMode.NORMAL, "never gonna give you up"),
            ("--now never gonna give you up", PlayMode.NOW, "never gonna give you up"),
            ("--next never gonna give", PlayMode.NEXT, "never gonna give"),
            ("--NOW song", PlayMode.NOW, "song"),
            ("--NeXt song", PlayMode.NEXT, "song"),
            ("--now   https://youtu.be/x", PlayMode.NOW, "https://youtu.be/x"),
            ("--next   https://youtu.be/x", PlayMode.NEXT, "https://youtu.be/x"),
            ("--now", PlayMode.NOW, ""),
            ("--next", PlayMode.NEXT, ""),
            ("  --now  song  ", PlayMode.NOW, "song"),
            ("  --next  song  ", PlayMode.NEXT, "song"),
            # A word that merely starts with a flag is a search, not a flag — and
            # not a typo either, on both the two-dash and one-dash side.
            ("--nowhere man", PlayMode.NORMAL, "--nowhere man"),
            ("-nowhere man", PlayMode.NORMAL, "-nowhere man"),
            ("--nextdoor", PlayMode.NORMAL, "--nextdoor"),
            ("-nextdoor", PlayMode.NORMAL, "-nextdoor"),
            # Trailing and repeated flags stay in the text: only the head is read.
            ("song --now", PlayMode.NORMAL, "song --now"),
            ("song --next", PlayMode.NORMAL, "song --next"),
            ("--now --now song", PlayMode.NOW, "--now song"),
            # The two are mutually exclusive by construction, so the second is
            # search text like any other repeat — it does not combine, and it does
            # not override.
            ("--now --next song", PlayMode.NOW, "--next song"),
            ("--next --now song", PlayMode.NEXT, "--now song"),
            ("", PlayMode.NORMAL, ""),
            ("   ", PlayMode.NORMAL, ""),
        ],
    )
    def test_the_head_decides(self, argument: str, mode: PlayMode, query: str) -> None:
        args = split_play_args(argument)
        assert (args.mode, args.query) == (mode, query)
        assert args.dash_typo is None

    @pytest.mark.parametrize(
        "argument,meant",
        [
            ("-now song", "--now"),  # one ASCII hyphen
            ("–now song", "--now"),  # en dash
            ("—now song", "--now"),  # em dash — what iOS turns a typed `--` into
            ("―now song", "--now"),  # horizontal bar
            ("-–now song", "--now"),  # mixed pair
            ("—NOW song", "--now"),  # the typo is case-insensitive too
            ("-now", "--now"),  # nothing behind it
            ("-next song", "--next"),
            ("—next song", "--next"),
            ("–NEXT song", "--next"),
            ("-–next song", "--next"),
            ("-next", "--next"),
        ],
    )
    def test_a_dash_away_from_a_flag_asks(self, argument: str, meant: str) -> None:
        """These cannot be anything but a misspelt flag, so the command asks rather
        than searching YouTube for the user's own flag — and it names the one it
        thinks was meant, which is the only reason dash_typo carries a string."""
        args = split_play_args(argument)
        assert args.dash_typo == meant
        assert args.mode is PlayMode.NORMAL

    @pytest.mark.parametrize("argument", ["--now song", "--next song"])
    def test_a_real_flag_is_never_read_as_a_typo(self, argument: str) -> None:
        """Ordering inside split_play_args is load-bearing: `--now` satisfies the
        near-miss pattern too (two dashes is within `{1,2}`), so the exact-match
        lookup has to run first or every correct invocation would be answered with
        a did-you-mean."""
        args = split_play_args(argument)
        assert args.dash_typo is None
        assert args.mode is not PlayMode.NORMAL

    @pytest.mark.parametrize(
        "argument",
        [
            "now thats what i call music",
            "now",
            "nowhere",
            "now --now",
            "next to me",
            "next",
            "nextdoor",
        ],
    )
    def test_a_bare_flag_word_is_a_search(self, argument: str) -> None:
        """The did-you-mean deliberately stops at the dash. `-p now thats what i
        call music` and `-p next to me` are real searches, and guessing there would
        break them."""
        args = split_play_args(argument)
        assert (args.mode, args.dash_typo, args.query) == (
            PlayMode.NORMAL,
            None,
            argument,
        )

    def test_the_query_keeps_its_case(self) -> None:
        """Only the head is lowercased to match the flag — the search is what the
        user typed, since it is also the origin -remove matches on."""
        assert split_play_args("--now Never Gonna GIVE").query == "Never Gonna GIVE"

    def test_it_splits_on_any_whitespace(self) -> None:
        """Discord messages carry newlines; the head is a token, not everything up
        to the first space."""
        assert split_play_args("--next\nsong") == PlayArgs(
            mode=PlayMode.NEXT, query="song"
        )

    def test_play_args_is_immutable(self) -> None:
        """Frozen: the split happens once at the top of the body and every consumer
        downstream — the gate, the branch, the origin — reads that same value."""
        args = split_play_args("--now song")
        with pytest.raises(AttributeError):
            setattr(args, "mode", PlayMode.NORMAL)


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
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = _connected_vc(mock_ctx)  # what a real join leaves
            return join_task

        with (
            _no_typing(),
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
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        with _no_typing(), patch("asyncio.create_task") as mock_create:
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
        music_bot.get_mp = MagicMock(return_value=_mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
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
        music_bot.get_mp = MagicMock(return_value=_mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
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
        music_bot.get_mp = MagicMock(return_value=_mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()


class TestPlayAnalytics:
    """The Analytics -play mints: the ask time at dispatch, handed to queue_source
    for every construction site, and the depth at the insert, under the place
    lock, where it is the position the song actually takes."""

    async def test_warm_path_carries_the_ask_time_and_mints_the_depth_at_the_insert(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        mp = _mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot.queue_source = spy

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        # The message snowflake, NOT time.time(): gateway delivery lag is real
        # time the user waited, and so is the 1-4s resolve that follows.
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()
        # Not read at the ask — two requests resolving together would both read
        # the same depth. The insert reads it.
        assert analytics.queue_position == 0
        queued = mp.queue_put.await_args.args[0]
        assert queued.analytics.queue_position == 7

    async def test_cold_path_is_depth_zero_without_reading_the_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A cold-start song front-inserts ahead of the restored queue and plays
        first, so its depth is 0 by construction — the queue is never asked."""
        mock_ctx.voice_client = None
        mp = _mock_mp()
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
            mock_ctx.voice_client = _connected_vc(mock_ctx)
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
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
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        mp = _mock_mp()
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

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        queued = mp.queue_put.await_args.args[0]
        assert queued.analytics.queue_position == 12


class TestPlayWhilePaused:
    """-play on a paused song interjects instead of appending
    . Appending would leave the bot silent
        with the request buried behind a paused song."""

    def _paused_mp(self) -> MagicMock:
        mp = _mock_mp()
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
        vc = _paused_vc(mock_ctx)
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New Song", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)
        music_bot._enqueue_single = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()

        with _no_typing(), patch.object(YTDL, "prefetch_stream", new=AsyncMock()):
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
        mock_ctx.voice_client = _paused_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=self._paused_mp())
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        mock_ctx.message.add_reaction = AsyncMock()

        with _no_typing(), patch.object(YTDL, "prefetch_stream", new=AsyncMock()):
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
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()

        with _no_typing(), patch("asyncio.create_task"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        music_bot._enqueue_single.assert_awaited_once()

    async def test_paused_without_current_song_falls_through(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing to interrupt — take the ordinary append path rather than
        building an interjection around a song that isn't there."""
        mock_ctx.voice_client = _paused_vc(mock_ctx)
        mp = self._paused_mp()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()

        with _no_typing(), patch("asyncio.create_task"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        music_bot._enqueue_single.assert_awaited_once()

    async def test_resume_during_resolution_appends_instead(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A -resume landing during the 1-4s extraction removes the reason to
        interject, so the resolved track is appended rather than interrupting a
        song the user just chose to keep playing."""
        vc = _paused_vc(mock_ctx)
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        mp.enqueue_depth = MagicMock(return_value=9)
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)

        async def _resolve_then_resume(*a: Any, **kw: Any) -> None:
            vc.is_paused.return_value = False  # user hit -resume mid-extraction
            return None

        with (
            _no_typing(),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        mp.queue_put.assert_awaited_once_with(qobj)
        assert qobj.interjected is False  # must not trigger replace semantics later
        # Re-minted for the append: the 0 minted for an interjection would claim
        # this song played immediately when it waited behind the whole queue.
        assert qobj.analytics.queue_position == 9

    async def test_a_resume_mid_resolve_still_queues_the_rest_of_the_playlist(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The append path has a playlist behind it too, and dropping it is silent.

        A `-resume` landing during the extraction turns the interjection into an
        ordinary append — but the head arrived with the rest of its playlist, and
        without this the tail is discarded with no error, no log line, and a reply
        that says the song was queued. The tracks follow the head to the tail rather
        than front-inserting: the head just went to the back of the queue, and the
        playlist belongs behind it either way."""
        vc = _paused_vc(mock_ctx)
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        music_bot._enqueue_single = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()
        url = "https://www.youtube.com/playlist?list=PLabc"

        async def _resolve_then_resume(*a: Any, **kw: Any) -> None:
            vc.is_paused.return_value = False  # user hit -resume mid-extraction
            return None

        with (
            _no_typing(),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_not_awaited()
        single_call = music_bot._enqueue_single.await_args
        assert single_call is not None
        assert single_call.args[1] is tracks[0]
        # Behind the head, through the same insert, so the two land in order.
        assert list(single_call.kwargs["follow_on"]) == tracks[1:]
        # And SAID so. Queueing a playlist behind a reply that names one song is
        # how 199 tracks arrive unannounced.
        note = single_call.kwargs["note"]
        assert "**3** songs" in note
        assert "-remove" in note

    async def test_a_resume_mid_resolve_restamps_the_tail_it_moved(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The tail's ask-time depths were minted for a FRONT insert (1..N-1), and
        this path sends it to the back instead. play_history keeps whatever number
        is on them forever, so the head being re-minted and the tail not left one
        row at the real depth and the rest claiming the front of the queue."""
        vc = _paused_vc(mock_ctx)
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        mp.enqueue_depth = MagicMock(return_value=20)
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(
                f"https://yt.com/v={i}",
                f"Track {i}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1.0, queue_position=i),
            )
            for i in range(3)
        ]
        mock_ctx.message.add_reaction = AsyncMock()

        async def _resolve_then_resume(*a: Any, **kw: Any) -> None:
            vc.is_paused.return_value = False
            return None

        with (
            _no_typing(),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="https://www.youtube.com/playlist?list=PLabc"
            )

        head_call, tail_call = mp.queue_put.await_args_list
        assert head_call.args[0] is tracks[0]
        assert tracks[0].analytics.queue_position == 20
        assert [item.analytics.queue_position for item in tail_call.args[0]] == [21, 22]

    async def test_resolution_failure_leaves_paused_song_untouched(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Resolution happens before interject, so a failed lookup never stops
        the paused song."""
        vc = _paused_vc(mock_ctx)
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        vc.stop.assert_not_called()
        assert mp.current_song is not None
        mock_ctx.send.assert_awaited()  # error embed

    async def test_playlist_interjects_head_first_and_queues_the_rest(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The head interrupts and the rest queue behind it, so the paused song
        comes back after the WHOLE playlist. That is the deliberate call, and the
        confirmation both states it and names the one command that undoes it."""
        mock_ctx.voice_client = _paused_vc(mock_ctx)
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        mock_ctx.message.add_reaction = AsyncMock()
        # Distinct sentinel, not tracks[0]: if the URL ever stops parsing as a
        # playlist, _resolve_interjection_source falls through to queue_source, and
        # the identity assertion below catches it. (Stubbing it at all is also
        # a network guard — an unstubbed one runs a real yt-dlp extraction.)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject(
                "https://yt.com/v=fell-through", "X", mock_ctx.author
            )
        )
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_awaited_once()
        assert mp.interject.await_args.args[0] is tracks[0]
        assert list(mp.interject.await_args.kwargs["follow_on"]) == tracks[1:]
        sent = mock_ctx.send.await_args_list + mock_ctx.send.call_args_list
        notices = [
            c.kwargs["embed"].description
            for c in sent
            if c.kwargs.get("embed") is not None
        ]
        # TWO, not three: the head is playing now, and a playing song has no queue
        # object — its entry was LPOPed at start — so -remove cannot reach it.
        # Counting it would offer an undo that leaves the interrupting track behind.
        assert any("**2** songs" in (d or "") for d in notices), notices
        assert any("-remove" in (d or "") for d in notices), notices
        assert any("`-skip`" in (d or "") for d in notices), notices


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
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = _connected_vc(mock_ctx)  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        single_call = music_bot._enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["placement"] is Placement.COLD_FRONT

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
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        with (
            _no_typing(),
            patch(
                "asyncio.create_task", side_effect=lambda c: (c.close(), join_task)[1]
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        music_bot._enqueue_single.assert_not_awaited()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mp.repark_crashed_head.assert_awaited_once()

    async def test_a_cold_start_beats_the_next_flag_to_the_placement(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`-p --next` on a disconnected bot is a COLD_FRONT, not a NEXT.

        Both front-insert, so the queue ends up the same either way and the
        difference is entirely in the reply: COLD_FRONT sends the resume notice,
        which is the ONLY thing naming the song about to start (the gate is still
        shut, so there is no Now Playing block yet), while NEXT would send "Playing
        next" — true of nothing here, since the song is what plays. The precedence
        is one `if`/`elif` ordering and nothing else pins it."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = _connected_vc(mock_ctx)  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next test"
            )

        single_call = music_bot._enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["placement"] is Placement.COLD_FRONT

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
        mp = _mock_mp()
        mp.wait_for_restore = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = _connected_vc(mock_ctx)
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
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
        mp = _mock_mp()
        mp.repark_crashed_head = AsyncMock(side_effect=lambda: calls.append("repark"))
        music_bot.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock(side_effect=lambda _g: calls.append("cleanup"))

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        with (
            _no_typing(),
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
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        with _no_typing(), patch("asyncio.create_task"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        single_call = music_bot._enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["placement"] is Placement.TAIL
        # No playback hold on the warm path — the gate is already open. It DOES
        # wait on the restore, on the same bound as every placement: a put()
        # landing before restore_entries replays leaves the deque holding this
        # song ahead of entries Redis lists behind it.
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
        mp = _mock_mp()

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
            mock_ctx.voice_client = _connected_vc(mock_ctx)  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
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

        mp = _mock_mp()
        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = _connected_vc(mock_ctx)  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.defer_playback.assert_called_once()

    async def test_front_single_uses_queue_put_front_and_sends_resume_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = _mock_mp(qsize=3)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.COLD_FRONT,
        )

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
        mp = _mock_mp(qsize=0)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.COLD_FRONT,
        )

        mp.queue_put_front.assert_awaited_once_with(qobj)
        mock_ctx.send.assert_not_awaited()

    async def test_front_playlist_inserts_all_tracks_in_order(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unlike `--now` (first track only), plain -play front-inserts a playlist in
        full — nothing is playing here to delay the return of."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = _mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks),
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.COLD_FRONT,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put_front.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()

    async def test_next_playlist_inserts_all_tracks_through_queue_put_next(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`--next` takes a playlist in FULL. queue_put_next rather than
        queue_put_front because a song IS playing here — the loop's prefetch holds a
        claim a plain front-insert would land behind, and the playlist would start
        one song late."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = _mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks),
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put_next.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        # Said, not implied: "Queued playlist" alone reads as "at the back".
        assert "plays next" in mock_ctx.send.call_args.kwargs["embed"].title

    @pytest.mark.parametrize(
        "placement,warmed",
        [
            (Placement.NEXT, True),
            (Placement.TAIL, False),
            (Placement.COLD_FRONT, False),
        ],
    )
    async def test_only_a_next_playlist_warms_its_first_track(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        placement: Placement,
        warmed: bool,
    ) -> None:
        """The bulk enqueue warms nothing, which is right for the tail and wrong for
        the head under `--next`: queue_put_next just killed the loop's one-ahead
        prefetch and the loop will not spawn another until its next iteration, so
        the song promised to play next would reach the handoff with an empty
        stream cache and pay a full in-band extraction — the dead air the
        three-phase pipeline exists to remove. The HEAD only, or N concurrent
        extractions mint URLs that expire before playback reaches them."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = _mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        with patch.object(YTDL, "prefetch_stream", new=AsyncMock()) as warm:
            await music_bot._enqueue_playlist(
                mock_ctx,
                source,
                ResolvedYoutubePlaylist(tracks),
                mp,
                _admit(music_bot, mock_ctx, mp),
                placement=placement,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
            )

        assert warm.await_count == (1 if warmed else 0)
        if warmed:
            warmed_call = warm.await_args
            assert warmed_call is not None
            assert warmed_call.args[0] is tracks[0]

    async def test_cold_path_routes_playlist_through_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """End-to-end wiring for the playlist half of the cold path: play()'s
        list branch must carry the COLD_FRONT placement into _enqueue_playlist.
        Previously
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
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = _connected_vc(mock_ctx)  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        music_bot._enqueue_single.assert_not_awaited()
        pl_call = music_bot._enqueue_playlist.await_args
        assert pl_call is not None
        assert pl_call.kwargs["placement"] is Placement.COLD_FRONT
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
            await music_bot._enqueue_single(
                mock_ctx,
                qobj,
                music_player,
                _admit(music_bot, mock_ctx, music_player),
                placement=Placement.COLD_FRONT,
            )

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
    @pytest.mark.parametrize("placement", list(Placement))
    async def test_only_a_cold_front_builds_a_resume_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock, placement: Placement
    ) -> None:
        """The reason placement is a value and not a `front: bool`.

        `build_resume_notice_embed` says "Resumed from queue … N songs from the
        previous session resume after it", which is true only for a disconnected
        bot waking a persisted queue. Parametrized over every member on purpose:
        a placement added later that front-inserts while the bot is playing would
        otherwise inherit that copy silently, and it returns None only when the
        queue is EMPTY — so it renders exactly on the case that would be wrong.
        """
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = _mock_mp(qsize=3)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, _admit(music_bot, mock_ctx, mp), placement=placement
        )

        assert mp.build_resume_notice_embed.called is (
            placement is Placement.COLD_FRONT
        )

    async def test_next_inserts_through_queue_put_next_and_names_what_it_waits_on(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """queue_put_next, not queue_put_front: the loop's prefetch holds a claim a
        plain front-insert would land behind, and the song would play second while
        the embed said "next".

        And no "Est. playing at": estimated_playing_at() seeds from the current
        song's FULL duration as a proxy for what is left of it, which is fine at the
        back of a queue and badly wrong for the very next slot. Naming the song it
        waits behind is exact."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = _mock_mp(qsize=3)
        mp.current_song = MagicMock(title="Current Banger")
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = _connected_vc(mock_ctx)

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
        )

        mp.queue_put_next.assert_awaited_once_with(qobj)
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        mp.estimated_playing_at.assert_not_called()
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Playing next" in embed.title
        assert "Urgent" in embed.title
        assert "Current Banger" in embed.description

    async def test_next_says_playback_is_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`--next` deliberately does NOT interject a paused song, so the bot stays
        silent afterwards. Nothing else in the response would explain that."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = _mock_mp()
        mp.current_song = MagicMock(title="Paused Song")
        mock_ctx.message.add_reaction = AsyncMock()
        vc = _in_authors_channel(MagicMock(spec=discord.VoiceClient), mock_ctx)
        vc.is_paused.return_value = True
        mock_ctx.voice_client = vc

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
        )

        assert "-resume" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_next_on_an_idle_bot_says_it_starts_now(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """With nothing playing there is nothing to be next to — a front insert
        into an empty queue IS an append, which is what lets `--next` need no
        special case for an idle bot."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = _mock_mp()
        mp.current_song = None
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = _connected_vc(mock_ctx)

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
        )

        assert "starts now" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_next_during_the_handoff_does_not_claim_to_start_now(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """loop() takes the prefetch result out of its slot and nulls it BEFORE it
        assigns current_song, and the claim it took stays open until the commit.
        Across that window nothing to neutralize exists and current_song is None,
        so put_front lands the song behind the one about to start — while the
        confirmation, reading current_song alone, told the user it starts now."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = _mock_mp()
        mp.current_song = None
        mp.queue.claim_outstanding = MagicMock(return_value=True)
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = _connected_vc(mock_ctx)

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
        )

        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert "starts now" not in description
        assert "Plays after the song starting now." in description

    async def test_a_long_title_cannot_400_the_confirmation(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Discord rejects the WHOLE send past 256 title chars, and qobj.title is
        yt-dlp metadata from arbitrary sites — not just YouTube's 100-char ceiling.
        Unguarded, `-p --next <long-titled video>` queues the song and then reports
        nothing at all."""
        qobj = QueueObject("https://yt.com/v=1", "T" * 400, mock_ctx.author)
        mp = _mock_mp()
        mp.current_song = None
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = _connected_vc(mock_ctx)

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
        )

        assert len(mock_ctx.send.call_args.kwargs["embed"].title) <= 256

    def test_the_front_insert_depth_counts_an_open_claim(self) -> None:
        """Same window, on the number that goes to Postgres forever: the song is
        queued behind the one about to play, so the ask-time depth is 1. Reading
        current_song alone recorded 0 — an insert that waited behind nothing."""
        mp = _mock_mp()
        mp.current_song = None
        mp.queue.claim_outstanding = MagicMock(return_value=True)

        assert _front_insert_depth(mp) == 1

    async def test_shows_queued_embed_with_eta_when_song_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = _in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, _admit(music_bot, mock_ctx, mp)
        )

        mp.estimated_playing_at.assert_called_once()
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Est. playing at **7:42 PM PST**" in embed.description

    async def test_no_queued_embed_when_nothing_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, _admit(music_bot, mock_ctx, mp)
        )

        mp.estimated_playing_at.assert_not_called()
        mock_ctx.send.assert_not_awaited()

    async def test_queued_embed_has_thumbnail_when_present(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = _in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject(
            "https://yt.com/v=1",
            "Test Song",
            mock_ctx.author,
            thumbnail="https://img.youtube.com/vi/1/0.jpg",
        )

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, _admit(music_bot, mock_ctx, mp)
        )

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert embed.thumbnail.url == "https://img.youtube.com/vi/1/0.jpg"

    async def test_queued_embed_has_no_thumbnail_when_absent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = _in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(
            mock_ctx, qobj, mp, _admit(music_bot, mock_ctx, mp)
        )

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert embed.thumbnail.url is None


class TestNowCommand:
    async def test_repins_now_playing_when_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """-now re-hosts the live NP block at the bottom of the channel (the
        old host is retired) instead of sending a static snapshot embed."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = mock_ctx.channel  # invoked from the player's home channel
        mp.repin_now_playing = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()
        mock_ctx.send.assert_not_awaited()

    async def test_repins_live_block_when_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """-now while paused repins the live block rather than replying "No songs
        are currently playing" — an intentional behaviour change, not a side effect
        of making the embed live."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=True)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = mock_ctx.channel
        mp.repin_now_playing = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()

    async def test_cross_channel_sends_static_embed_where_invoked(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """-now from a channel other than the player's home channel must
        answer THERE with a static snapshot — the host never leaves home, so
        repinning would leave the invoking channel with no response at all."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = MagicMock()  # distinct from ctx.channel → distinct .id
        static = discord.Embed(title="NP snapshot")
        mp._build_now_playing_embed = MagicMock(return_value=static)
        mp.repin_now_playing = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        mp.repin_now_playing.assert_not_awaited()
        mp._build_now_playing_embed.assert_called_once_with(mp.current_song)
        mock_ctx.send.assert_awaited_once_with(embed=static)

    async def test_falls_back_when_repin_reports_no_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """The song can end between the liveness check and the repin —
        repin_now_playing() returns False and -now must still respond
        instead of silently doing nothing."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=True)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild

        mp = MagicMock()
        mp.current_song = MagicMock()
        mp._channel = mock_ctx.channel
        mp.play_message = None
        mp.repin_now_playing = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        mp.repin_now_playing.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "No songs are currently playing."
        )

    async def test_sends_not_playing_when_no_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        mock_guild.voice_client = None
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.play_message = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "No songs are currently playing."
        )

    async def test_now_reports_nothing_playing_after_song_ends(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """After a song finishes, loop() nulls both current_song and
        play_message — the recovery-snapshot elif must not serve the finished
        song's embed as "Now playing"."""
        vc = object.__new__(discord.VoiceClient)
        vc.is_playing = MagicMock(return_value=False)
        vc.is_paused = MagicMock(return_value=False)
        mock_guild.voice_client = vc
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.current_song = None
        mp.play_message = None
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        assert (
            mock_ctx.send.await_args.kwargs["embed"].description
            == "No songs are currently playing."
        )

    async def test_sends_restored_snapshot_during_recovery_window(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        """current_song isn't live yet (crash-recovery window), but a
        now-playing snapshot survived the restart via play_message."""
        mock_guild.voice_client = None
        mock_ctx.guild = mock_guild
        mp = MagicMock()
        mp.current_song = None
        mp.play_message = discord.Embed(title="Now Playing")
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.now)(music_bot, mock_ctx)
        mock_ctx.send.assert_awaited_once_with(embed=mp.play_message)


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


# ── Queue command ─────────────────────────────────────────────────────────────


class TestQueueCommand:
    async def test_always_sends_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        embed = discord.Embed(
            title="Queue", description="Songs: **0**\n\n*The queue is empty.*"
        )
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=embed)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.queue)(music_bot, mock_ctx)

        mock_ctx.send.assert_awaited_once()
        call_kwargs = mock_ctx.send.call_args[1]
        assert "embed" in call_kwargs
        assert call_kwargs["embed"] is embed

    async def test_sends_embed_when_queue_is_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        embed = discord.Embed(
            title="Queue", description="Songs: **0**\n\n*The queue is empty.*"
        )
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=embed)
        mp.song_queue = []
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.queue)(music_bot, mock_ctx)

        mock_ctx.send.assert_awaited_once()

    async def test_delegates_to_mp_get_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.queue)(music_bot, mock_ctx)

        mp.queue_embed.assert_called_once()


# ── Remove command ────────────────────────────────────────────────────────────


def _removed_song(n: int, query_source: str = "") -> QueueObject:
    """A stand-in for what queue_remove hands back — the reply reads query_source
    off these to name how it matched."""
    return QueueObject(
        f"https://yt.com/v={n}", f"Song {n}", MagicMock(), query_source=query_source
    )


class TestEchoIsSafeInAnEmbed:
    """`-remove`'s argument is user text echoed back into an embed, and Discord
    renders markdown in descriptions and field values. Escaping it is not
    cosmetic: unescaped, any member can make the bot post a styled masked link
    under its own name, with no queue state required."""

    def test_a_masked_link_cannot_form(self) -> None:
        """Asserted on the OUTCOME, not on the escape: what matters is that no
        `](` pair survives to pick a link's label and destination. Escaping alone
        does not cover it — brackets are not in escape_markdown's set."""
        attack = "[Free Discord Nitro](https://evil.example/phish)"
        out = _echo(attack)
        assert "[" not in out and "]" not in out

    def test_markdown_riding_behind_a_url_is_neutralized(self) -> None:
        """escape_markdown defaults to ignore_links=True, which passes any http(s)
        URL through UNTOUCHED — so an attack prefixed with a bare link reaches the
        embed verbatim unless safe_label overrides that default."""
        attack = "https://x.com/`[FREE NITRO](https://evil.example/phish)"
        out = _echo(attack)
        assert "[" not in out and "]" not in out
        assert "`" not in out

    def test_emphasis_behind_a_url_is_escaped(self) -> None:
        """What `ignore_links=False` still buys, now that the brackets are
        neutralized outright: escape_markdown's URL exemption covers the WHOLE
        token, so emphasis after a scheme renders styled unless the flag is off.
        Pinned separately because the masked-link tests above pass either way."""
        out = _echo("https://x.com/**bold**_em_")
        assert "\\*\\*" in out
        assert "\\_" in out

    def test_a_backtick_cannot_close_the_code_span(self) -> None:
        """Two call sites wrap this in a code span, and Discord gives a backslash
        NO meaning inside one — so an ESCAPED backtick still closes the span and
        renders everything after it. The backtick has to go, not be escaped."""
        out = _echo("foo` **bold** `bar")
        assert "`" not in out
        assert "\\*\\*" in out

    def test_control_characters_cannot_end_the_line_early(self) -> None:
        """A control character truncates the rendered line, hiding whatever the
        needle put after it."""
        assert _echo("a\x00b\x1fc\x7fd") == "a b c d"

    def test_the_echo_is_bounded_well_inside_the_field_cap(self) -> None:
        """Discord 400s the whole send past 1024 chars in a field value, and
        escaping can double the length. The removal has already committed by then,
        so the user sees "Command failed" for a removal that happened. `*`, not
        `x`: escaping leaves `x` alone and would not exercise the doubling."""
        # A LITERAL bound, and tight: the old 1024 passed at five times the real
        # value, so raising _ECHO_MAX kept this green while pushing the composed
        # field — ten of these in ONE 1024-char value — past the cap. safe_label
        # caps BEFORE escaping, so 200 raw chars become at most 400 plus the
        # ellipsis.
        assert len(_echo("*" * 5000)) <= 410

    def test_an_ordinary_needle_is_unchanged_apart_from_the_span(self) -> None:
        assert _echo("never gonna give you up") == "never gonna give you up"


class TestItemLabelNamesEveryItemType:
    """The Songs field exists because one argument can now take out a whole
    playlist and there is no undo. `YTSource` has no `.title` at all, so reaching
    for it rendered every unresolved Spotify-playlist track as `?` — the exact
    case the field was added for, and the one the -remove help now advertises."""

    def test_a_resolved_song_uses_its_title(self, mock_author: MagicMock) -> None:
        item = QueueObject("https://yt.com/v=1", "Real Title", mock_author)
        assert item_label(item) == "Real Title"

    def test_an_unresolved_search_uses_its_search_text(self) -> None:
        item = YTSource(ytsearch="ytsearch:Artist - Song", process=True)
        assert item_label(item) == "Artist - Song"

    def test_an_unresolved_link_falls_back_to_the_url(self) -> None:
        item = YTSource(url="https://yt.com/v=2", process=True)
        assert item_label(item) == "https://yt.com/v=2"


class TestRemoveReplyStaysInsideDiscordsCaps:
    """Every field of the `-remove` reply is built from a list the USER sizes —
    the removed songs and their positions — and the send happens AFTER
    queue_remove() has already mutated memory and Redis. So an over-length field
    is not cosmetic: Discord 400s the whole send, `_command_error` reports
    "Command failed", and the user is told nothing happened to a queue that has
    already been irreversibly changed. Asserted on the ASSEMBLED embed, since ten
    individually-capped echoes still share one field."""

    @staticmethod
    def _fields(mock_ctx: MagicMock) -> list[discord.embeds.EmbedProxy]:
        return list(mock_ctx.send.await_args_list[0][1]["embed"].fields)

    async def _run(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        *,
        removed: list[QueueItem],
        positions: list[int],
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=removed, positions=positions, mode=RemoveMode.RESOLVED
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/v=0"
        )

    async def test_ten_long_titles_fit_the_songs_field(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """99 characters is INSIDE YouTube's own 100-char title limit, so ten
        ordinary songs overflow the 1024-char field with no crafted content."""
        songs: list[QueueItem] = [
            QueueObject(f"https://yt.com/v={i}", "A" * 99, MagicMock())
            for i in range(10)
        ]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 11))
        )
        for field in self._fields(mock_ctx):
            assert len(field.value or "") <= EMBED_FIELD_LIMIT, field.name

    async def test_a_markdown_heavy_title_cannot_blow_the_field(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Escaping roughly doubles a title of pure markdown characters, which is
        the shape a hostile uploader picks."""
        songs: list[QueueItem] = [
            QueueObject(f"https://yt.com/v={i}", "*" * 200, MagicMock())
            for i in range(10)
        ]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 11))
        )
        for field in self._fields(mock_ctx):
            assert len(field.value or "") <= EMBED_FIELD_LIMIT, field.name

    async def test_a_playlists_worth_of_positions_fits(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """One `-remove <playlist link>` drops every track the link added. A raw
        join passes 1024 characters at 227 positions — well inside what a real
        playlist holds."""
        songs: list[QueueItem] = [_removed_song(i) for i in range(240)]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 241))
        )
        fields = {f.name: f.value or "" for f in self._fields(mock_ctx)}
        positions_field = next(v for k, v in fields.items() if k and "removed" in k)
        assert len(positions_field) <= EMBED_FIELD_LIMIT
        # The count is still honest about what went, even though the list is cut.
        assert "180 more" in positions_field

    async def test_the_whole_embed_stays_under_the_total_cap(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Discord caps an embed at 6000 characters across every part, so three
        fields each legal on their own can still fail together."""
        songs: list[QueueItem] = [
            QueueObject(f"https://yt.com/v={i}", "*" * 200, MagicMock())
            for i in range(240)
        ]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 241))
        )
        assert len(mock_ctx.send.await_args_list[0][1]["embed"]) <= 6000


class TestCommandArgumentBinding:
    """`-play` and `-remove` both consume the rest of the line.

    A positional binds ONE WORD. `-play` stores its argument as the origin
    `-remove` matches on, so a positional there meant `-play never gonna give you
    up` recorded `"never"` — the help's own example matched nothing, and
    `-remove never` became a wildcard over every song starting with that word.

    Asserted on the callback signature rather than through a parsed message,
    because the binding is a property of the signature and the tests that missed
    this were the ones that hand-built the value instead."""

    @pytest.mark.parametrize("name", ["play", "remove"])
    def test_the_argument_consumes_the_rest_of_the_line(self, name: str) -> None:
        import inspect

        callback = getattr(MusicBot, name).callback
        param = list(inspect.signature(callback).parameters.values())[2]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"-{name}'s {param.name} is positional; discord.py will bind one word"
        )

    @pytest.mark.parametrize(
        "typed,expected",
        [
            ("never gonna give you up", "never gonna give you up"),
            # discord.py strips a consume-rest argument before the body sees it;
            # this pins that play() does not re-introduce the whitespace when it
            # hands the origin on.
            ("some song   ", "some song"),
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
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=typed)

        call = music_bot.queue_source.await_args
        assert call is not None
        assert call.kwargs["origin"] == expected


class TestRemoveCommand:
    async def test_a_failure_becomes_a_command_error(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Every other command wraps its body; -remove was the one that did not, so
        a raise escaped to discord.py's generic handler — logged server-side, and
        answered with nothing the user could act on or quote. _command_error is
        what renders the embed and puts the trace id in its footer."""
        mp = MagicMock()
        mp.queue_remove = AsyncMock(side_effect=RuntimeError("queue exploded"))
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._command_error = AsyncMock()

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        music_bot._command_error.assert_awaited_once()
        recorded = mocked(music_bot._command_error).await_args
        assert recorded is not None
        # The real exception, not one manufactured by the handler — that is what
        # puts a usable type and message in the log and on the span.
        assert isinstance(recorded.args[1], RuntimeError)

    async def test_no_url_sends_usage_message(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.remove)(music_bot, mock_ctx, needle=None)

        mock_ctx.send.assert_awaited_once()
        msg = mock_ctx.send.call_args.kwargs["embed"].description
        assert "-remove" in msg

    async def test_no_match_sends_not_found_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(removed=[], positions=[])
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=notfound"
        )

        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "No queued songs found" in embed.description

    async def test_match_sends_removal_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[2],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        calls = mock_ctx.send.await_args_list
        # First call: removal embed
        first_kwargs = calls[0][1]
        assert "embed" in first_kwargs
        removal_embed = first_kwargs["embed"]
        assert "Removed" in removal_embed.title

    async def test_lazy_playlist_tracks_are_named_not_rendered_as_unknown(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A Spotify playlist enqueues one lazy YTSource per track, and YTSource has
        no `title` at all — so the Songs field rendered `?` for every track of the
        collection, which is the case it was added for. The field exists because a
        bare count leaves the user unable to tell whether it took what they meant,
        and there is no undo."""
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[
                    YTSource(ytsearch=f"ytsearch:Track {i} Artist") for i in range(3)
                ],
                positions=[1, 2, 3],
                mode=RemoveMode.ORIGIN,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://open.spotify.com/playlist/abc"
        )

        songs = next(
            f
            for f in mock_ctx.send.await_args_list[0][1]["embed"].fields
            if f.name == "Songs"
        )
        assert "Track 0 Artist" in (songs.value or "")
        assert "ytsearch:" not in (songs.value or "")
        assert "?" not in (songs.value or "")

    async def test_match_sends_updated_queue_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        queue_embed = discord.Embed(title="Queue")
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[1],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=queue_embed)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        calls = mock_ctx.send.await_args_list
        assert len(calls) == 2
        second_kwargs = calls[1][1]
        assert "embed" in second_kwargs
        assert second_kwargs["embed"] is queue_embed

    async def test_match_adds_trash_reaction(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[1],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        mock_ctx.message.add_reaction.assert_awaited_once_with("🗑️")

    async def test_an_origin_match_explains_itself(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """One argument removing eight songs needs a reason on screen, or it reads
        as the bot having removed more than it was asked to."""
        album = "https://open.spotify.com/album/abc123"
        removed: list[QueueItem] = [_removed_song(i, "spotify.com") for i in range(8)]
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=removed,
                positions=list(range(1, 9)),
                mode=RemoveMode.ORIGIN,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(music_bot, mock_ctx, needle=album)

        fields = {
            f.name: f.value for f in mock_ctx.send.await_args_list[0][1]["embed"].fields
        }
        assert (
            fields["Matched"] == f"{album} — the spotify.com link you queued them with"
        )

    async def test_a_search_match_is_quoted_not_linked(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Search text is not a URL — angle-bracketing it would render as a broken
        link, and "them" would be wrong for the single song it took."""
        song = _removed_song(1, "search")
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[song], positions=[2], mode=RemoveMode.ORIGIN
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="never gonna give you up"
        )

        fields = {
            f.name: f.value for f in mock_ctx.send.await_args_list[0][1]["embed"].fields
        }
        assert fields["Matched"] == (
            "never gonna give you up — the search you queued it with"
        )

    async def test_removal_embed_names_what_it_matched(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[3],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        url = "https://yt.com/watch?v=abc"

        await command_callback(MusicBot.remove)(music_bot, mock_ctx, needle=url)

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        fields = {f.name: f.value for f in removal_embed.fields}
        # Escaped, and deliberately NOT wrapped in a code span: escaping inside
        # one renders the backslashes literally, and a bare URL auto-links —
        # which the angle-bracket form this replaced also did.
        assert fields["Matched"] == url

    async def test_removal_embed_shows_positions(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(2)],
                positions=[1, 4],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        field_values = [f.value for f in removal_embed.fields]
        assert any("1" in v and "4" in v for v in field_values)

    async def test_removal_embed_color_is_orange(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[1],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        assert removal_embed.colour == discord.Color.orange()


# ── -play --now ──────────────────────────────────────────────────────────────


class TestNowFlag:
    """`-p --now` end to end: resolve one song, interrupt, and report.

    The assertions are unchanged from when a separate command drove this
    path — what they pin is the interjection itself, and only its entry point
    moved."""

    @pytest.fixture
    def live_mp(self) -> MagicMock:
        """A MusicPlayer mock with a song currently playing.

        Built on _mock_mp() rather than a bare MagicMock: the not-live rows now
        run play()'s real cold-start machinery, which needs defer_playback as an
        async CM and an awaitable wait_for_restore."""
        from src.musicplayer import InterjectOutcome

        mp = _mock_mp()
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
    def live_vc(self, mock_ctx: MagicMock) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return _in_authors_channel(vc, mock_ctx)

    async def test_two_concurrent_now_flags_interject_one_at_a_time(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Two `--now` resolving together, driven through the command rather than
        through _interject_flow directly.

        Overlapping, each would park a resume tail for a song the other had
        already stopped, and one play's history row goes with it — the race
        max_concurrency used to close by serializing the whole body. Only an
        end-to-end drive proves the command still takes the lock that replaced it.
        """
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        music_bot._command_error = AsyncMock()
        gate = asyncio.Event()

        async def _resolve(_ctx: Any, _source: Any, **kw: Any) -> QueueObject:
            await gate.wait()
            return QueueObject(
                f"https://yt.com/v={kw['origin']}", kw["origin"], mock_ctx.author
            )

        music_bot.queue_source = AsyncMock(side_effect=_resolve)

        inside = 0
        overlapped = False

        async def _interject(*_a: Any, **_k: Any) -> InterjectOutcome:
            nonlocal inside, overlapped
            inside += 1
            overlapped = overlapped or inside > 1
            await asyncio.sleep(0)  # a suspension point inside the hold
            inside -= 1
            return InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=False,
            )

        live_mp.interject = AsyncMock(side_effect=_interject)

        with _no_typing():
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(
                        music_bot, mock_ctx, url=f"--now s{n}"
                    )
                )
                for n in (1, 2)
            ]
            await _settle()
            assert music_bot.queue_source.await_count == 2  # both resolving at once
            gate.set()
            await asyncio.gather(*tasks)

        assert live_mp.interject.await_count == 2  # neither was dropped
        assert not overlapped
        music_bot._command_error.assert_not_awaited()

    async def test_idle_runs_the_ordinary_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing live to interrupt, so this queues like any other -play — through
        the same body, so the checks, the hooks and the bucket all apply."""
        mp = _mock_mp()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot._enqueue_single = AsyncMock()
        music_bot._interject_flow = AsyncMock()
        music_bot._command_error = AsyncMock()

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        music_bot._interject_flow.assert_not_awaited()
        music_bot._enqueue_single.assert_awaited_once()
        # Without this the test passes on a TypeError from an under-configured
        # mock: the body swallows everything into _command_error.
        music_bot._command_error.assert_not_awaited()

    async def test_no_voice_client_runs_the_ordinary_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live_mp: MagicMock
    ) -> None:
        """A live current_song is not enough — with no voice client there is
        nothing to interrupt, so this takes the cold-start path instead."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = None
        mock_ctx.invoke = AsyncMock()
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot._interject_flow = AsyncMock()
        music_bot._abandon_cold_start = AsyncMock()
        music_bot._command_error = AsyncMock()

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        music_bot._interject_flow.assert_not_awaited()
        music_bot.queue_source.assert_awaited_once()
        music_bot._command_error.assert_not_awaited()

    async def test_live_song_interjects(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        assert qobj.interjected is True
        # The origin reaches the song through yt_source's required user_input, not
        # a post-hoc assignment — so with queue_source mocked out, assert it was
        # PASSED. A real yt_source stamps it (see test_youtube).
        origin_call = music_bot.queue_source.await_args
        assert origin_call is not None
        assert origin_call.kwargs["origin"] == "test"
        live_mp.interject.assert_awaited_once_with(
            qobj, live_vc, resume_paused=True, follow_on=[]
        )
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
        """`--now` restores exactly what it interrupted, so a paused song is
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

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

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

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

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
        """Interjections stack, so a song that was itself interjected gets
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

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

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
        live_mp.queue_put_next = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.invoke = AsyncMock()
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        music_bot.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        # Resolved once: a second resolve would re-parse and, for a playlist,
        # enqueue every track again.
        music_bot.queue_source.assert_awaited_once()
        # queue_put_next, not queue_put_front: the embed promises "play next", and
        # a bare front-insert only delivers that with nothing queued — the loop's
        # prefetch holds a claim the insert would land behind. interject() returned
        # None without reaching its own neutralize, so this path owes it.
        # prefetch=False — the stream was warmed, so it must not warm again.
        live_mp.queue_put_next.assert_awaited_once_with([qobj], prefetch=False)
        # interject() also returns None when the loop moved on to a DIFFERENT
        # song, which this insert waits behind: one, not the 0 an interjection
        # would have had, and not the queue depth — it goes to the front.
        assert qobj.analytics.queue_position == 1
        # The interjection marker must not leak onto a normally queued song —
        # a later interjection would otherwise "replace" it without a resume entry.
        assert qobj.interjected is False
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Playing next" in embed.title
        assert "already ended" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_only_the_head_of_a_playlist_is_marked_interjected(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """`interjected` means "this song cut the line", which is true of the head
        and of nothing behind it — the rest queued the way any -play's tracks do.
        It is attribution only today (an `interject.over_interjection` span
        attribute), so marking them all would be wrong quietly: every track of a
        500-song `--now` filed as its own interjection."""
        # Captured AT the call: the marker is a mutable field, and the
        # interject-returned-None path deliberately clears the head's afterwards.
        marks: list[list[bool]] = []

        async def _record(qobj: QueueObject, _vc: Any, **kw: Any) -> None:
            marks.append([qobj.interjected, *(i.interjected for i in kw["follow_on"])])
            return None

        live_mp.interject = AsyncMock(side_effect=_record)
        live_mp.queue_put_next = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.add_reaction = AsyncMock()
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(
                music_bot,
                mock_ctx,
                url="--now https://www.youtube.com/playlist?list=PLabc",
            )

        assert marks == [[True, False, False]]

    async def test_a_playlist_that_fell_through_says_how_many_it_queued(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Same fall-through, with a playlist behind it: all N front-insert while
        the reply named Track 1 alone. Nothing was interrupted here, so the head is
        QUEUED — it counts, and -remove reaches it."""
        live_mp.interject = AsyncMock(return_value=None)
        live_mp.queue_put_next = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.add_reaction = AsyncMock()
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        url = "https://www.youtube.com/playlist?list=PLabc"

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        live_mp.queue_put_next.assert_awaited_once_with(tracks, prefetch=False)
        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert "**3** songs" in description
        assert "-remove" in description
        # No -skip caveat: nothing is playing that -remove cannot reach.
        assert "`-skip`" not in description

    async def test_spotify_playlist_interjects_head_and_queues_the_rest(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(return_value=["First Song", "Second"])
        qobj = QueueObject("https://yt.com/v=first", "First Song", mock_ctx.author)

        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=qobj)
        ) as ys:
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        music_bot.spotify.playlist.assert_awaited_once_with("37i9dQZF1DXcBWIGoYBM5M")
        ys.assert_awaited_once()
        assert ys.call_args.args[1] == "ytsearch:First Song"
        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is qobj
        # Only the HEAD is resolved to a playable song: the rest stay lazy YouTube
        # searches, which is what keeps a 100-track album from paying 100 searches
        # before a note is heard.
        follow_on = live_mp.interject.call_args.kwargs["follow_on"]
        assert [item.ytsearch for item in follow_on] == ["ytsearch:Second"]
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        # ONE: the tail. The head is playing, so -remove cannot take it back out.
        assert any("**1** song" in d for d in notices), notices
        assert any("`-skip`" in d for d in notices), notices

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
        first = QueueObject("https://yt.com/v=1", "Track One", mock_ctx.author)
        second = QueueObject("https://yt.com/v=2", "Track Two", mock_ctx.author)

        with patch(
            "src.musicbot.YTDL.yt_playlist", new=AsyncMock(return_value=[first, second])
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

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

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        live_mp.interject.assert_not_awaited()
        mock_ctx.send.assert_awaited()  # error embed

    async def test_the_prefetch_settles_before_the_place_lock(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """interject()'s own neutralize cancels a prefetch that can be pinned in
        the yt-dlp executor. Run inside _place that wait IS the guild's lock, held
        while every sibling -play in the guild burns its own bound against it and
        reports a Redis outage that never happened."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        music_bot.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        order: list[str] = []
        live_mp.settle_prefetch = AsyncMock(
            side_effect=lambda *a, **k: order.append("settle")
        )
        real_place = music_bot._plays.place

        @contextlib.asynccontextmanager
        async def _spy(req: PlayRequest) -> AsyncIterator[PlaceResult]:
            order.append("place")
            async with real_place(req) as verdict:
                yield verdict

        music_bot._plays.place = _spy
        with _no_typing(), patch.object(YTDL, "prefetch_stream", new=AsyncMock()):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        assert order == ["settle", "place"]

    async def test_warms_stream_cache_before_interjecting(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The stream-URL cache is warmed before interject stops the current
        song — a cache miss at dequeue would otherwise put yt-dlp dead air
        between the interrupt and the interjected song starting."""
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
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

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


class TestShuffleWaitsForTheRestore:
    """-shuffle waits for the restore like every other queue-mutating command:
    shuffle() REBUILDS the mirror from memory, so running it before
    restore_entries() has replayed the saved queue writes an unrestored deque over
    it and deletes the persisted entries outright."""

    async def test_it_refuses_until_the_restore_lands(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=False)
        mp.queue_shuffle = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        with _no_typing():
            await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_not_awaited()
        assert "Still loading" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_it_shuffles_once_the_restore_has_landed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_shuffle = AsyncMock(return_value="Shuffled!")
        music_bot.get_mp = MagicMock(return_value=mp)

        with _no_typing():
            await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_awaited_once()


class TestTimestampWarningReachesTheUser:
    @staticmethod
    def _bad_ts_source() -> Any:
        return parse_url("https://youtu.be/a?t=bogus")

    async def test_it_rides_the_queued_song_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = _mock_mp()
        mp.queue.qsize = MagicMock(return_value=3)  # something already queued
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            warning=timestamp_warning(self._bad_ts_source()),
        )

        mock_ctx.send.assert_awaited_once()
        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert "bogus" in description
        assert "Est. playing at" in description  # folded in, not replacing it

    @pytest.mark.parametrize(
        "placement", [Placement.COLD_FRONT, Placement.NEXT], ids=lambda p: p.name
    )
    async def test_it_rides_the_flag_confirmations_too(
        self, music_bot: MusicBot, mock_ctx: MagicMock, placement: Placement
    ) -> None:
        """ "Every exit sends it either way" is the contract, and the flag legs
        build their own embeds. `-p --next <link>?t=bogus` would otherwise lose the
        only word the user gets that the timestamp was ignored."""
        mp = _mock_mp()
        mp.queue.qsize = MagicMock(return_value=3)
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await music_bot._enqueue_single(
            mock_ctx,
            qobj,
            mp,
            _admit(music_bot, mock_ctx, mp),
            placement=placement,
            warning=timestamp_warning(self._bad_ts_source()),
        )

        said = " ".join(
            (c.kwargs["embed"].description or "")
            for c in mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        )
        assert "bogus" in said

    async def test_it_gets_its_own_message_when_no_embed_is_sent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """An idle bot plays the first song immediately and sends no "Queued
        song" embed at all. Riding that embed alone would drop the warning in
        the most ordinary case there is."""
        mp = _mock_mp()
        mp.queue.qsize = MagicMock(return_value=0)
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        mock_ctx.voice_client.is_playing = MagicMock(return_value=False)
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        with patch("src.musicbot.send_embed", new=AsyncMock()) as send:
            await music_bot._enqueue_single(
                mock_ctx,
                qobj,
                mp,
                _admit(music_bot, mock_ctx, mp),
                warning=timestamp_warning(self._bad_ts_source()),
            )

        send.assert_not_awaited()
        sent = [c.kwargs["embed"] for c in mock_ctx.send.await_args_list]
        assert any("bogus" in (e.description or "") for e in sent)


# ── The place section: resolve concurrently, insert under one short lock ──────


async def _settle(ticks: int = 12) -> None:
    """Let every runnable task reach its next suspension point."""
    for _ in range(ticks):
        await asyncio.sleep(0)


def _gated_resolve(qobj: QueueObject, gate: asyncio.Event) -> AsyncMock:
    """A queue_source that resolves `qobj` once `gate` is set."""

    async def _resolve(*_a: Any, **_kw: Any) -> QueueObject:
        await gate.wait()
        return qobj

    return AsyncMock(side_effect=_resolve)


@contextlib.contextmanager
def _recording_span() -> Iterator[MagicMock]:
    """A span double in place of the current span. Its context is invalid so
    structlog's OTel processor, which reads the same global, skips the trace-id
    format it would otherwise apply to a MagicMock."""
    with patch("src.musicbot.trace.get_current_span") as current:
        span = current.return_value
        span.get_span_context.return_value.is_valid = False
        yield span


async def _stalled_put(*_a: Any, **_k: Any) -> None:
    """A put against a Redis that accepts and never answers."""
    await asyncio.sleep(5)


def _song(n: int, ctx: MagicMock) -> QueueObject:
    return QueueObject(f"https://yt.com/v={n}", f"Song {n}", ctx.author)


def _holding_mp() -> MagicMock:
    """_mock_mp() whose defer_playback really counts holds, for the cold-start
    tests: _abandon_cold_start reads playback_holds, and two participants must
    see each other's."""
    mp = _mock_mp()
    mp.playback_holds = 0

    @contextlib.asynccontextmanager
    async def _hold() -> AsyncIterator[None]:
        mp.playback_holds += 1
        try:
            yield
        finally:
            mp.playback_holds -= 1

    mp.defer_playback = MagicMock(side_effect=_hold)
    return mp


class TestResolveConcurrency:
    """PLAY_INFLIGHT_MAX bounds what a guild holds in memory; this bounds what it
    holds of the shared, process-wide yt-dlp pool."""

    async def test_a_guild_holds_at_most_that_many_workers_at_once(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Without it one guild's paste burst takes every worker for as many waves
        as it has links, and the jobs queued behind include the playback loop's own
        in-band extractions in OTHER guilds."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        live = 0
        peak = 0
        release = asyncio.Event()

        async def _resolve(*_a: Any, **_k: Any) -> QueueObject:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await release.wait()
            live -= 1
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_resolve)
        with _no_typing(), patch("src.play_placement.PLAY_RESOLVE_CONCURRENCY", 2):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in range(5)
            ]
            await _settle()
            assert peak == 2, peak  # not 5
            release.set()
            await asyncio.gather(*tasks)

        assert mp.queue_put.await_count == 5  # and all of them still land


class TestPlayRegistry:
    """PlayRegistry.register / retire and the per-guild state they keep."""

    def test_register_is_synchronous_to_the_insert(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = _mock_mp()
        first = _admit(music_bot, mock_ctx, mp)
        second = _admit(music_bot, mock_ctx, mp)

        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        # Held by identity, in arrival order: two requests for one query from one
        # author are two requests, and the drop reports read this order back.
        assert plays.inflight == [first, second]
        assert first.generation == mp.queue.generation

    def test_beyond_the_cap_the_request_is_declined(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = _mock_mp()
        with patch("src.play_placement.PLAY_INFLIGHT_MAX", 2):
            _admit(music_bot, mock_ctx, mp)
            _admit(music_bot, mock_ctx, mp)
            with (
                _recording_span() as span,
                pytest.raises(commands.MaxConcurrencyReached) as excinfo,
            ):
                _admit(music_bot, mock_ctx, mp)

        # Recorded before the cap check: the declined request carries the count
        # it would have joined, and nothing else counts declines.
        span.set_attribute.assert_any_call("play.inflight", 3)
        span.set_attribute.assert_any_call("play.declined", True)
        assert excinfo.value.number == 2  # the cap, not 1: the wording keys on it
        assert len(music_bot._plays._guilds[play_key(mock_ctx)].inflight) == 2

    def test_another_guild_has_its_own_count(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = _mock_mp()
        other = MagicMock()
        other.guild = MagicMock()
        other.guild.id = mock_ctx.guild.id + 1
        with patch("src.play_placement.PLAY_INFLIGHT_MAX", 1):
            _admit(music_bot, mock_ctx, mp)
            _admit(music_bot, other, mp)  # no raise

    def test_the_registry_is_dropped_once_idle(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = _mock_mp()
        req = _admit(music_bot, mock_ctx, mp)
        key = play_key(mock_ctx)
        assert key in music_bot._plays._guilds

        music_bot._plays.retire(req)
        assert not music_bot._plays._guilds

    async def test_the_registry_outlives_its_requests_while_a_join_runs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = _mock_mp()
        req = _admit(music_bot, mock_ctx, mp)
        key = play_key(mock_ctx)
        done = asyncio.Event()

        async def _join(*_a: Any, **_k: Any) -> None:
            await done.wait()

        mock_ctx.invoke = AsyncMock(side_effect=_join)
        music_bot._restore_tasks = set()

        join, owns_join = music_bot._plays.cold_join(
            req,
            joiner=lambda: mock_ctx.invoke(music_bot.join),
            tracked=music_bot._restore_tasks,
        )
        assert owns_join  # the first request to find no client creates it
        music_bot._plays.retire(req)
        assert key in music_bot._plays._guilds  # the join still runs

        done.set()
        await join
        await _settle()
        assert not music_bot._plays._guilds

    async def test_the_cap_raise_escapes_the_command_body(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Raised in play() before _play's try/except: caught there it would
        render as "Failed to queue song"; from here it reaches cog_command_error
        and the existing decline notice."""
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._command_error = AsyncMock()
        with patch("src.play_placement.PLAY_INFLIGHT_MAX", 1):
            _admit(music_bot, mock_ctx, mp)
            with _no_typing(), pytest.raises(commands.MaxConcurrencyReached):
                await command_callback(MusicBot.play)(music_bot, mock_ctx, url="x")
        music_bot._command_error.assert_not_awaited()

    async def test_the_decline_names_the_cap_not_a_single_slot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "play"
        await music_bot.cog_command_error(
            mock_ctx, commands.MaxConcurrencyReached(16, commands.BucketType.guild)
        )
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "Too many" in text and "resolving" in text


class TestResolveThenPlace:
    """Requests resolve together and insert one at a time, each where the queue
    is when its own resolve finishes."""

    def _warm(self, music_bot: MusicBot, mock_ctx: MagicMock) -> MagicMock:
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_requests_resolve_concurrently(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()
        music_bot.queue_source = _gated_resolve(_song(1, mock_ctx), gate)

        with _no_typing():
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await _settle()
            # Both are inside the resolver; neither has placed.
            assert music_bot.queue_source.await_count == 2
            mp.queue_put.assert_not_awaited()
            assert len(music_bot._plays._guilds[play_key(mock_ctx)].inflight) == 2

            gate.set()
            await asyncio.gather(*tasks)

        assert mp.queue_put.await_count == 2
        assert not music_bot._plays._guilds

    async def test_a_short_resolve_places_before_a_long_one(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Decision 2.1: no ticket order. The song asked second lands first when
        the collection asked first is still extracting."""
        mp = self._warm(music_bot, mock_ctx)
        slow_gate = asyncio.Event()
        slow, fast = _song(1, mock_ctx), _song(2, mock_ctx)

        async def _resolve(_ctx: Any, _source: Any, **_kw: Any) -> QueueObject:
            if _kw["origin"] == "slow":
                await slow_gate.wait()
                return slow
            return fast

        music_bot.queue_source = AsyncMock(side_effect=_resolve)

        with _no_typing():
            first = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="slow")
            )
            await _settle()
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="fast")
            assert mp.queue_put.await_args.args[0] is fast
            slow_gate.set()
            await first

        assert [c.args[0] for c in mp.queue_put.await_args_list] == [fast, slow]

    async def test_the_reply_is_sent_outside_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A 429 on the channel must not hold every other -play in the guild: the
        confirmation goes out after the lock is released."""
        mp = self._warm(music_bot, mock_ctx)
        mp.queue.qsize = MagicMock(return_value=3)  # so a confirmation is sent
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        send_gate = asyncio.Event()
        sends = 0

        async def _blocked_send(**_kw: Any) -> None:
            nonlocal sends
            sends += 1
            if sends == 1:
                await send_gate.wait()

        mock_ctx.send = AsyncMock(side_effect=_blocked_send)

        with _no_typing():
            first = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            )
            await _settle()
            assert sends == 1  # stuck in its reply
            assert not music_bot._plays._guilds[play_key(mock_ctx)].lock.locked()
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="b")
            assert mp.queue_put.await_count == 2  # placed behind a blocked reply
            send_gate.set()
            await first

    async def test_the_confirmation_is_rendered_outside_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Only the put and the depth on it belong under the hold. Rendering an
        "Est. playing at" walks the whole queue, and the hold is shared by every
        -play in the guild — under it, one long queue's walk is time every sibling
        spends waiting for a lock it is not using."""
        mp = self._warm(music_bot, mock_ctx)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        held: list[bool] = []

        def _eta() -> str:
            plays = music_bot._plays._guilds[play_key(mock_ctx)]
            held.append(plays.lock.locked())
            return "soon"

        mp.estimated_playing_at = MagicMock(side_effect=_eta)

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_awaited_once()  # it did place, so the ETA was rendered
        assert held == [False]

    async def test_a_refusal_is_sent_outside_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mock_ctx.author.voice = None  # left during the resolve
        send_gate = asyncio.Event()

        async def _blocked_send(**_k: Any) -> None:
            await send_gate.wait()

        mock_ctx.send = AsyncMock(side_effect=_blocked_send)

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            )
            await _settle()
            mock_ctx.send.assert_awaited_once()
            assert not music_bot._plays._guilds[play_key(mock_ctx)].lock.locked()
            send_gate.set()
            await task

        mp.queue_put.assert_not_awaited()

    async def test_a_served_request_records_resolve_and_place_wait(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._warm(music_bot, mock_ctx)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))

        with _no_typing(), _recording_span() as span:
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        recorded = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert recorded["play.inflight"] == 1
        assert recorded["play.resolve_secs"] >= 0
        assert recorded["play.place_wait_secs"] >= 0
        assert "play.declined" not in recorded
        assert "play.dropped_by" not in recorded

    async def test_the_place_timeout_reports_and_releases(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A Redis stall inside the put: the request reports, the lock is free
        for the next one, and nothing else is sent."""
        mp = self._warm(music_bot, mock_ctx)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mp.queue_put = AsyncMock(side_effect=_stalled_put)

        with _no_typing(), patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mock_ctx.send.assert_awaited_once()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "queue is busy" in text
        # queue_put appends to the deque BEFORE it awaits the mirror, so a stall
        # here may well have queued the song. Claiming it did not, and inviting a
        # retry, is what mints the duplicate.
        assert "wasn't queued" not in text
        assert "-queue" in text
        assert not music_bot._plays._guilds  # retired, lock gone with it

    async def test_a_cold_start_that_stalls_at_the_insert_is_abandoned(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._abandon_cold_start = AsyncMock()
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mp.queue_put_front = AsyncMock(side_effect=_stalled_put)

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = _connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with _no_typing(), patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        music_bot._abandon_cold_start.assert_awaited_once()
        assert "queue is busy" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_a_stalled_tail_does_not_disconnect_what_its_siblings_queued(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The hold is one Redis round trip long, so a guild bursting to the
        admission cap serializes that many and the last request can spend its whole
        budget waiting. Torn down there, one refusal disconnects a joined session
        and takes every song the other fifteen placed with it."""
        mock_ctx.voice_client = None
        mp = _mock_mp(qsize=3)  # siblings placed while this one waited
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._abandon_cold_start = AsyncMock()
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mp.queue_put_front = AsyncMock(side_effect=_stalled_put)

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = _connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with _no_typing(), patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        music_bot._abandon_cold_start.assert_not_awaited()
        assert "queue is busy" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_a_refused_tail_does_not_disconnect_either(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The other late exit, reached by a verdict rather than a clock. Same
        session, same siblings, same reason not to take it down."""
        mock_ctx.voice_client = None
        mp = _mock_mp(qsize=3)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._abandon_cold_start = AsyncMock()
        gate = asyncio.Event()

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_slow)

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = _connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            )
            await _settle()
            mock_ctx.author.voice = None  # left while it resolved
            gate.set()
            await task

        mp.queue_put_front.assert_not_awaited()
        music_bot._abandon_cold_start.assert_not_awaited()

    async def test_a_connection_lost_mid_resolve_is_still_torn_down(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The other half of the guard. A queue with songs in it is only a reason
        to stay if there is still a connection to play them over: released with
        none, the loop fails its `vc` assertion once per restored song while Redis
        keeps every entry, and the next restore does it again."""
        mock_ctx.voice_client = None
        mp = _mock_mp(qsize=3)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._abandon_cold_start = AsyncMock()
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mp.queue_put_front = AsyncMock(side_effect=_stalled_put)

        async def _join(*_a: Any, **_k: Any) -> None:
            vc = _connected_vc(mock_ctx)
            # Connected for the post-join check, gone by the time the stall asks:
            # a kick between the handshake and the lock.
            vc.is_connected = MagicMock(side_effect=[True, False])
            mock_ctx.voice_client = vc

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with _no_typing(), patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        music_bot._abandon_cold_start.assert_awaited_once()

    async def test_a_stall_waiting_for_the_lock_says_the_song_is_absent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The timeout is outer to the lock: a request parked behind a stalled
        sibling gives up on its own clock, not the sibling's — and having never
        acquired the lock it wrote nothing, so this half may say so. The other
        half may not: see the put-stall test, where the deque is already
        appended and a retry would mint a duplicate."""
        mp = self._warm(music_bot, mock_ctx)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        plays = _GuildPlays()
        music_bot._plays._guilds[play_key(mock_ctx)] = plays
        await plays.lock.acquire()  # a sibling holds it and never returns

        with _no_typing(), patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_not_awaited()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "queue is busy" in text
        assert "wasn't queued" in text  # nothing was written, so it may say so
        plays.lock.release()

    async def test_an_author_who_left_during_the_resolve_does_not_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)

        async def _leave(*_a: Any, **_kw: Any) -> QueueObject:
            mock_ctx.author.voice = None
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_leave)

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_not_awaited()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "not connected to a voice channel" in text

    async def test_queue_position_is_the_depth_at_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Decision 2.3. Two asks read the same depth; two inserts do not."""
        mp = self._warm(music_bot, mock_ctx)
        placed: list[QueueObject] = []

        async def _put(obj: Any, **_k: Any) -> None:
            placed.append(obj)

        mp.queue_put = AsyncMock(side_effect=_put)
        mp.enqueue_depth = MagicMock(side_effect=lambda: 4 + len(placed))
        gate = asyncio.Event()
        songs = iter([_song(1, mock_ctx), _song(2, mock_ctx)])

        async def _resolve(*_a: Any, **_kw: Any) -> QueueObject:
            await gate.wait()
            return next(songs)

        music_bot.queue_source = AsyncMock(side_effect=_resolve)

        with _no_typing():
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await _settle()
            gate.set()
            await asyncio.gather(*tasks)

        assert [q.analytics.queue_position for q in placed] == [4, 5]

    async def test_now_interjects_without_waiting_for_a_collection(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The lane plan's §12 row 3, removed rather than accepted: --now lands
        while the collection asked before it is still resolving."""
        mp = self._warm(music_bot, mock_ctx)
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Old",
                resume_position=10,
                was_paused=False,
                returns_paused=False,
            )
        )
        gate = asyncio.Event()
        music_bot.queue_source = _gated_resolve(_song(1, mock_ctx), gate)
        music_bot._resolve_interjection_source = AsyncMock(
            return_value=(_song(2, mock_ctx), [])
        )

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            collection = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="list")
            )
            await _settle()
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now x")
            mp.interject.assert_awaited_once()
            assert not collection.done()
            gate.set()
            await collection


class TestPlaylistPositionsAreMintedAtTheInsert:
    """A playlist's queue_position is the slot each track actually takes. Minted
    at resolve it is the depth the queue had 1-99s earlier, and it rides to
    Postgres unchallenged."""

    def _wire(self, music_bot: MusicBot, mock_ctx: MagicMock, depth: int) -> MagicMock:
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        mp.enqueue_depth = MagicMock(return_value=depth)
        return mp

    @pytest.mark.parametrize("depth", [0, 4])
    async def test_a_youtube_playlist_lands_on_the_depth_at_the_insert(
        self, music_bot: MusicBot, mock_ctx: MagicMock, depth: int
    ) -> None:
        mp = self._wire(music_bot, mock_ctx, depth)
        tracks = [
            QueueObject(
                f"https://yt.com/v={n}",
                f"T{n}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1.0, queue_position=n),
            )
            for n in range(3)
        ]
        music_bot.queue_source = AsyncMock(
            return_value=ResolvedYoutubePlaylist(tracks=tracks, skipped=0)
        )
        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="https://yt.com/playlist?list=x"
            )

        queued = mp.queue_put.await_args.args[0]
        assert [q.analytics.queue_position for q in queued] == [
            depth,
            depth + 1,
            depth + 2,
        ]

    async def test_a_head_that_moved_during_the_wait_is_rebased_under_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The provisional mint happens outside the lock, so another request can
        place before this one's turn. The depth read UNDER the lock is the slot the
        collection actually takes; the one minted against is stale."""
        mp = self._wire(music_bot, mock_ctx, 0)
        calls = {"n": 0}

        def _depth() -> int:
            calls["n"] += 1
            return 4 if calls["n"] == 1 else 9  # a sibling placed in between

        mp.enqueue_depth = MagicMock(side_effect=_depth)
        tracks = [
            QueueObject(
                f"https://yt.com/v={n}",
                f"T{n}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1.0, queue_position=n),
            )
            for n in range(3)
        ]
        music_bot.queue_source = AsyncMock(
            return_value=ResolvedYoutubePlaylist(tracks=tracks, skipped=0)
        )
        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="https://yt.com/playlist?list=x"
            )

        queued = mp.queue_put.await_args.args[0]
        assert [q.analytics.queue_position for q in queued] == [9, 10, 11]

    async def test_the_rebase_is_skipped_when_the_head_has_not_moved(self) -> None:
        """The O(N) pass is one dataclass copy per track — milliseconds of
        synchronous event-loop time at 5,000 of them, and under the place lock
        every sibling -play waits it out. Minting before the lock and re-basing
        under it makes the common case free."""
        tracks = [
            QueueObject(
                f"https://yt.com/v={n}", f"T{n}", MagicMock(), analytics=_ANALYTICS
            )
            for n in range(3)
        ]

        assert _rebase_positions(tracks, 7, 7) is tracks  # same list, no copies

        moved = _rebase_positions(tracks, 7, 9)
        assert moved is not tracks
        assert [q.analytics.queue_position for q in moved] == [9, 10, 11]


class TestPlacementRevalidation:
    """What the insert re-checks, and what a command can still take back."""

    def _resolving(
        self, music_bot: MusicBot, mock_ctx: MagicMock, gate: asyncio.Event
    ) -> MagicMock:
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_slow)
        return mp

    async def test_a_now_whose_author_changed_channels_does_not_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """validate_commands gates --now on the bot's own channel, because it stops
        what that channel is hearing. The re-check after the resolve has to carry
        the same queue_control, or walking to another channel during a 1-99s
        extraction buys an exemption the command never had."""
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)
        mp.current_song = None

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now x")
            )
            await _settle()
            mock_ctx.author.voice.channel = MagicMock(spec=discord.VoiceChannel)
            gate.set()
            await task

        mp.queue_put_next.assert_not_awaited()
        mp.interject.assert_not_called()  # bare Mock: no await-flavoured assert
        assert (
            "already being used" in mock_ctx.send.await_args.kwargs["embed"].description
        )

    async def test_a_plain_play_keeps_its_exemption(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Queueing into a session running elsewhere costs its listeners nothing,
        so -play alone is exempt at dispatch and must stay exempt here."""
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="x")
            )
            await _settle()
            mock_ctx.author.voice.channel = MagicMock(spec=discord.VoiceChannel)
            gate.set()
            await task

        mp.queue_put.assert_awaited_once()

    async def test_remove_reaches_a_request_still_resolving(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The window this branch opened: a resolve runs to 99s, and the argument
        the user is taking back is the one they just typed. Matched by origin, the
        way the queue matches it."""
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[], positions=[], mode=RemoveMode.RESOLVED
            )
        )

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="Bad Song")
            )
            await _settle()
            await command_callback(MusicBot.remove)(
                music_bot, mock_ctx, needle="bad song"
            )
            gate.set()
            await task

        mp.queue_put.assert_not_awaited()
        said = [
            c.kwargs["embed"].description or ""
            for c in mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert any("still being looked up" in t for t in said), said
        assert any("`-remove` ran while it was resolving" in t for t in said), said

    async def test_remove_leaves_an_unrelated_request_alone(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[], positions=[], mode=RemoveMode.RESOLVED
            )
        )

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="Keep This")
            )
            await _settle()
            await command_callback(MusicBot.remove)(
                music_bot, mock_ctx, needle="something else"
            )
            gate.set()
            await task

        mp.queue_put.assert_awaited_once()

    async def test_a_placed_request_is_not_reported_as_dropped(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """PlayRegistry.retire runs in play()'s finally, AFTER the confirmation is sent, so
        a request whose song is already queued is still in the registry. Stamping it
        names it in -clear's own dropped field beside the song it just cleared."""
        mp = _mock_mp(3)  # something queued, so a confirmation is actually sent
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        gate = asyncio.Event()

        async def _slow_send(*_a: Any, **_k: Any) -> MagicMock:
            await gate.wait()
            return MagicMock()

        # Blocked BEFORE the task starts, so it parks after the put and before
        # the retire: exactly the window where a placed request is still listed.
        mock_ctx.send = AsyncMock(side_effect=_slow_send)
        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="queued one")
            )
            await _settle()
            mp.queue_put.assert_awaited_once()  # it placed
            assert music_bot._plays._guilds[
                play_key(mock_ctx)
            ].inflight  # and is listed
            dropped = music_bot._plays.inflight(play_key(mock_ctx), "clear")
            gate.set()
            await task

        assert dropped == []  # no command can take it back now

    async def test_a_cold_start_refused_at_the_lock_is_torn_down(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The join already put the bot in the channel. Every other exit tears it
        down; without this one it sits in an empty channel until the 300s idle."""
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()
        music_bot._abandon_cold_start = AsyncMock()
        gate = asyncio.Event()

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = _connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_slow)
        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="x")
            )
            await _settle()
            mock_ctx.author.voice = None  # left while it resolved
            gate.set()
            await task

        mp.queue_put_front.assert_not_awaited()
        music_bot._abandon_cold_start.assert_awaited()

    async def test_a_failed_reaction_does_not_fail_a_queued_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Send Messages without Add Reactions is a common split, and by the time
        the reaction runs the song is IN the queue. Raising there renders "Failed
        to queue song" over a song that plays."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mock_ctx.message.add_reaction = AsyncMock(
            side_effect=discord.Forbidden(MagicMock(status=403), "no reactions")
        )

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_awaited_once()
        for call_ in mock_ctx.send.await_args_list:
            embed = call_.kwargs.get("embed")
            assert embed is None or "Failed to queue" not in (embed.title or "")

    async def test_a_body_timeout_is_not_reported_as_a_stalled_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """_place catches TimeoutError, which since 3.11 IS asyncio.TimeoutError.
        Anything inside the body raising one — a nested wait_for, a guard added
        later — would otherwise be reclassified as a Redis stall."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mp.queue_put = AsyncMock(side_effect=TimeoutError("someone else's deadline"))

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        text = mock_ctx.send.await_args.kwargs["embed"].description or ""
        assert "queue is busy" not in text
        assert "Failed to queue song" in (
            mock_ctx.send.await_args.kwargs["embed"].title or ""
        )


class TestPlayShowsTyping:
    async def test_the_body_runs_inside_the_typing_indicator(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Every other -play test patches background_typing away, so deleting the
        `async with` leaves the suite green and the bot silent for the 1-99s a
        resolve takes."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        entered: list[str] = []

        @contextlib.asynccontextmanager
        async def _typing(_ctx: Any) -> AsyncIterator[None]:
            entered.append("typing")
            yield

        with patch("src.musicbot.background_typing", new=_typing):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        assert entered == ["typing"]


class TestSpanDecoratorsNameTheirFunction:
    """A helper inserted between a decorator and the function it was written for
    silently inherits its span: bot.enqueue_playlist once measured a --next-only
    stream warm while the playlist enqueue itself went untraced."""

    async def test_every_tracer_decorator_wraps_the_function_it_names(self) -> None:
        import inspect
        import re

        import src.musicbot as module

        lines = inspect.getsource(module).splitlines()
        spans = [
            (i, m.group(1))
            for i, line in enumerate(lines)
            if (m := re.search(r'start_as_current_span\("bot\.([a-z_]+)"\)', line))
        ]
        assert len(spans) > 15, "the decorators moved; re-anchor this test"
        for i, name in spans:
            following = next(line.strip() for line in lines[i + 1 :] if line.strip())
            defined = re.match(r"(?:async )?def (\w+)", following)
            assert defined, (name, following)
            assert defined.group(1) in (name, f"_{name}"), (name, defined.group(1))


class TestPlaceRefuses:
    """The four checks a resolved request can fail at the lock, and what each
    side says about it."""

    def _warm(self, music_bot: MusicBot, mock_ctx: MagicMock) -> MagicMock:
        mock_ctx.voice_client = _playing_vc(mock_ctx)
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        # -clear on a MagicMock player: bump the generation the way clear() does.
        async def _clear() -> list[Any]:
            mp.queue.generation += 1
            return []

        mp.queue_clear = AsyncMock(side_effect=_clear)
        return mp

    async def test_clear_during_the_resolve_drops_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()
        music_bot.queue_source = _gated_resolve(_song(1, mock_ctx), gate)

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="my song")
            )
            await _settle()
            await command_callback(MusicBot.clear)(music_bot, mock_ctx)
            gate.set()
            await task

        mp.queue_put.assert_not_awaited()
        # Two reports: -clear names the request it dropped, and the request
        # names the command that dropped it.
        descriptions = [
            c.kwargs["embed"].description for c in mock_ctx.send.await_args_list
        ]
        clear_embed = mock_ctx.send.await_args_list[0].kwargs["embed"]
        assert clear_embed.fields[0].name == "1 play request dropped"
        assert "my song" in clear_embed.fields[0].value
        assert any("`-clear` ran while it was resolving" in d for d in descriptions)

    async def test_clear_reports_even_when_the_queue_was_already_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        _admit(music_bot, mock_ctx, mp, query="pending")

        await command_callback(MusicBot.clear)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "already empty" not in (embed.description or "") or embed.fields
        # The title counts what the command actually took. "Queue cleared — 0 songs
        # removed" over a field naming a dropped request contradicts itself.
        assert embed.title == "Cleared — 1 play request dropped"
        assert embed.fields[0].name == "1 play request dropped"

    async def test_clear_with_nothing_to_drop_still_says_already_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._warm(music_bot, mock_ctx)
        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        assert "already empty" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_stop_during_the_resolve_drops_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()
        music_bot.queue_source = _gated_resolve(_song(1, mock_ctx), gate)

        async def _cleanup(_guild: Any) -> None:
            mp.retired = True  # what cleanup() stamps before any await

        music_bot.cleanup = AsyncMock(side_effect=_cleanup)
        mock_ctx.message.add_reaction = AsyncMock()

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="my song")
            )
            await _settle()
            with patch("discord.utils.get", return_value=_playing_vc()):
                await command_callback(MusicBot.stop)(music_bot, mock_ctx)
            gate.set()
            await task

        mp.queue_put.assert_not_awaited()
        embeds = [c.kwargs["embed"] for c in mock_ctx.send.await_args_list]
        assert embeds[0].title == "Stopped"
        assert embeds[0].fields[0].name == "1 play request dropped"
        assert any(
            "`-stop` ran while it was resolving" in (e.description or "")
            for e in embeds
        )

    async def test_stop_drops_a_cold_start_before_there_is_a_client_to_find(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unconditional: the voice-client check that gates the teardown does
        not gate the report, since a cold start is resolving before any client
        exists."""
        mp = self._warm(music_bot, mock_ctx)
        _admit(music_bot, mock_ctx, mp, query="cold one")
        music_bot.cleanup = AsyncMock()

        with patch("discord.utils.get", return_value=None):
            await command_callback(MusicBot.stop)(music_bot, mock_ctx)

        music_bot.cleanup.assert_not_awaited()
        assert mock_ctx.send.await_args.kwargs["embed"].fields[0].name == (
            "1 play request dropped"
        )

    async def test_a_request_stop_reported_dropped_does_not_then_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """On this leg -stop retires no player and bumps no generation — there is
        no voice client to clean up — so the stamp is the ONLY invalidation. Left
        as a label the channel is told the request was dropped and it queues,
        joins and plays anyway."""
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_slow)
        music_bot.cleanup = AsyncMock()

        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="cold one")
            )
            await _settle()
            with patch("discord.utils.get", return_value=None):
                await command_callback(MusicBot.stop)(music_bot, mock_ctx)
            gate.set()
            await task

        music_bot.cleanup.assert_not_awaited()  # nothing was retired
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        said = mock_ctx.send.await_args.kwargs["embed"].description or ""
        assert "`-stop` ran while it was resolving" in said

    async def test_a_kick_during_the_resolve_drops_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """No command stamped it, so the message names the session, not a
        command."""
        mp = self._warm(music_bot, mock_ctx)

        async def _kicked(*_a: Any, **_kw: Any) -> QueueObject:
            mp.retired = True
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_kicked)

        with _no_typing(), _recording_span() as span:
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_not_awaited()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "the session ended" in text
        span.set_attribute.assert_any_call("play.dropped_by", "session")

    @pytest.mark.parametrize("url", ["song", "--next song", "--now song"])
    async def test_a_dropped_request_never_places(
        self, music_bot: MusicBot, mock_ctx: MagicMock, url: str
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        mp.current_song = MagicMock()
        mp.interject = AsyncMock()

        async def _cleared(*_a: Any, **_kw: Any) -> Any:
            mp.queue.generation += 1
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_cleared)

        async def _cleared_pair(*_a: Any, **_kw: Any) -> Any:
            mp.queue.generation += 1
            return _song(1, mock_ctx), []

        music_bot._resolve_interjection_source = AsyncMock(side_effect=_cleared_pair)

        with _no_typing(), patch.object(YTDL, "prefetch_stream", new=AsyncMock()):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.queue_put.assert_not_awaited()
        mp.queue_put_next.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        mp.interject.assert_not_awaited()
        assert (
            "queue was cleared" in mock_ctx.send.await_args.kwargs["embed"].description
        )

    async def test_a_dropped_collection_never_places(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        tracks = [_song(1, mock_ctx), _song(2, mock_ctx)]

        async def _cleared(*_a: Any, **_kw: Any) -> ResolvedYoutubePlaylist:
            mp.queue.generation += 1
            return ResolvedYoutubePlaylist(tracks)

        music_bot.queue_source = AsyncMock(side_effect=_cleared)

        with _no_typing():
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="https://www.youtube.com/playlist?list=PLx"
            )

        mp.queue_put.assert_not_awaited()
        mock_ctx.send.assert_awaited_once()
        assert (
            "queue was cleared" in mock_ctx.send.await_args.kwargs["embed"].description
        )

    async def test_an_interjection_that_stalls_at_the_lock_reports_too(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The interject route has no gate hold to unwind; it reports the same
        notice from _play's own handler."""
        mp = self._warm(music_bot, mock_ctx)
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(side_effect=_stalled_put)
        music_bot._resolve_interjection_source = AsyncMock(
            return_value=(_song(1, mock_ctx), [])
        )

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now x")

        assert "queue is busy" in mock_ctx.send.await_args.kwargs["embed"].description
        assert not music_bot._plays._guilds

    async def test_a_request_placed_before_the_clear_is_an_ordinary_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The boundary: placed means in the queue, and -clear's generation
        bump is what a later placer refuses on, not an earlier one."""
        mp = self._warm(music_bot, mock_ctx)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
        mp.queue_put.assert_awaited_once()

        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        assert not music_bot._plays._guilds  # nothing in flight to name


class TestColdStartSingleflight:
    """One -join per guild, shared by every request that found no voice client."""

    def _cold(
        self, music_bot: MusicBot, mock_ctx: MagicMock, *, join_gate: asyncio.Event
    ) -> MagicMock:
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()

        async def _join(*_a: Any, **_k: Any) -> None:
            await join_gate.wait()
            mock_ctx.voice_client = _connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)
        return mp

    async def _two_cold_starts(
        self, music_bot: MusicBot, mock_ctx: MagicMock, join_gate: asyncio.Event
    ) -> tuple[MagicMock, list[asyncio.Task[None]]]:
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        songs = iter([_song(1, mock_ctx), _song(2, mock_ctx)])
        music_bot.queue_source = AsyncMock(side_effect=lambda *_a, **_k: next(songs))
        tasks = [
            asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
            )
            for n in (1, 2)
        ]
        await _settle()
        return mp, tasks

    async def test_two_cold_starts_share_one_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        join_gate = asyncio.Event()
        with _no_typing():
            mp, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            mock_ctx.invoke.assert_awaited_once()
            assert mp.playback_holds == 2
            join_gate.set()
            await asyncio.gather(*tasks)

        mock_ctx.invoke.assert_awaited_once()
        assert mp.queue_put_front.await_count == 2  # both ahead of the leftovers
        assert mp.playback_holds == 0
        assert not music_bot._plays._guilds

    async def _two_cold_starts_onto_a_failed_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock, join_gate: asyncio.Event
    ) -> tuple[MagicMock, list[asyncio.Task[None]]]:
        """Both requests share one join that leaves no voice client. join()
        swallows its own error into _command_error, so the task completes and the
        waiters see only its absence."""
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()

        async def _failed_join(*_a: Any, **_k: Any) -> None:
            await join_gate.wait()  # and no voice client, ever

        mock_ctx.invoke = AsyncMock(side_effect=_failed_join)
        songs = iter([_song(1, mock_ctx), _song(2, mock_ctx)])
        music_bot.queue_source = AsyncMock(side_effect=lambda *_a, **_k: next(songs))
        tasks = [
            asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
            )
            for n in (1, 2)
        ]
        await _settle()
        return mp, tasks

    async def test_a_cancelled_play_still_retires_its_slot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """_play catches Exception, so play() only unwinds abnormally on a
        BaseException — which no test reaching the registry produced. Each leak
        costs a slot, and PLAY_INFLIGHT_MAX of them decline the guild until
        restart."""
        join_gate = asyncio.Event()
        with _no_typing():
            _, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            tasks[0].cancel()
            await _settle()
            join_gate.set()
            await tasks[1]
            await _settle()

        assert not music_bot._plays._guilds

    async def test_a_stale_join_callback_does_not_null_its_replacement(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A cancelled join settles a tick later and its done-callback runs with a
        replacement already installed. Clearing the slot unconditionally there
        hands the NEXT cold start an empty slot, and it spawns a second concurrent
        -join for the same guild — the one thing the singleflight exists to stop."""
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()
        gate = asyncio.Event()

        async def _join(*_a: Any, **_k: Any) -> None:
            await gate.wait()
            mock_ctx.voice_client = _connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)
        key = play_key(mock_ctx)
        plays = music_bot._plays._guilds.setdefault(key, _GuildPlays())
        first = self._make_join(music_bot, mock_ctx, plays)
        first.cancel()
        second = self._make_join(music_bot, mock_ctx, plays)
        await _settle()  # the cancelled task's callback has now run

        assert plays.join is second
        gate.set()
        with contextlib.suppress(Exception):
            await second

    @staticmethod
    def _make_join(
        music_bot: MusicBot, mock_ctx: MagicMock, plays: _GuildPlays
    ) -> asyncio.Task[Any]:
        req = PlayRequest(
            ctx=mock_ctx,
            guild_id=play_key(mock_ctx),
            query="x",
            mp=music_bot.get_mp(mock_ctx),
            generation=0,
            mode=PlayMode.NORMAL,
        )
        plays.inflight.append(req)
        join, _ = music_bot._plays.cold_join(
            req,
            joiner=lambda: mock_ctx.invoke(music_bot.join),
            tracked=music_bot._restore_tasks,
        )
        return join

    async def test_a_request_arriving_mid_handshake_joins_the_cold_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """discord.py registers the voice client BEFORE the handshake completes, so
        for most of a join an arriving request sees one. Read as warm it takes no
        gate hold and is invisible to _abandon_cold_start's census, and a creator
        whose resolve then fails tears the player down under it."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        songs = iter([_song(1, mock_ctx), _song(2, mock_ctx)])
        music_bot.queue_source = AsyncMock(side_effect=lambda *_a, **_k: next(songs))

        with _no_typing():
            first = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="s1")
            )
            await _settle()
            # Mid-handshake: the client exists, the join has not finished.
            mock_ctx.voice_client = _connected_vc(mock_ctx)
            second = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="s2")
            )
            await _settle()
            assert mp.playback_holds == 2, "the second request took no gate hold"
            join_gate.set()
            await asyncio.gather(first, second)

        mock_ctx.invoke.assert_awaited_once()  # still one join

    async def test_a_request_that_only_waited_reports_the_failed_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The P0: join() reports into the creator's context, so a request that
        merely awaited that join used to return None and say nothing at all."""
        join_gate = asyncio.Event()
        with _no_typing():
            mp, tasks = await self._two_cold_starts_onto_a_failed_join(
                music_bot, mock_ctx, join_gate
            )
            join_gate.set()
            await asyncio.gather(*tasks)

        mp.queue_put_front.assert_not_awaited()
        said = [
            call.kwargs["embed"].description
            for call in mock_ctx.send.await_args_list
            if call.kwargs.get("embed") is not None
        ]
        told = [text for text in said if "Couldn't join the voice channel" in text]
        # Exactly one: the waiter speaks for itself, the creator does not repeat
        # what join() already said in this channel.
        assert len(told) == 1, said
        assert "s2" in told[0]  # and it names the song that was dropped

    async def test_a_failed_join_names_itself_on_the_span(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join() swallows its own error, so without this the drop leaves no
        trace at all: no status, no attribute, no resolve_secs."""
        join_gate = asyncio.Event()
        with _no_typing(), _recording_span() as span:
            _, tasks = await self._two_cold_starts_onto_a_failed_join(
                music_bot, mock_ctx, join_gate
            )
            join_gate.set()
            await asyncio.gather(*tasks)

        assert ("play.dropped_by", "join_failed") in [
            recorded.args for recorded in span.set_attribute.call_args_list
        ]

    async def test_a_join_that_lands_reports_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The notice rides the failure, not the wait: a shared join that works
        leaves both requests on the ordinary confirmation path."""
        join_gate = asyncio.Event()
        with _no_typing():
            mp, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            join_gate.set()
            await asyncio.gather(*tasks)

        assert mp.queue_put_front.await_count == 2
        for call_ in mock_ctx.send.await_args_list:
            embed = call_.kwargs.get("embed")
            if embed is not None:
                assert "Couldn't join the voice channel" not in (
                    embed.description or ""
                )

    async def test_a_stall_records_itself_on_the_span_and_the_log(
        self, music_bot: MusicBot, mock_ctx: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_PlaceStalled becomes a notice at both catch sites, so it never reaches
        _command_error's log-and-record: without this the failure most likely to be
        systemic — a Redis that accepts and stalls — leaves no operator signal."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        mp.queue_put = AsyncMock(side_effect=_stalled_put)

        with (
            _no_typing(),
            _recording_span() as span,
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        recorded = [call.args for call in span.set_attribute.call_args_list]
        assert ("play.verdict", "stalled") in recorded
        # Set inside the lock on every other path, so a request that never
        # acquired it would otherwise report no wait at all.
        assert any(name == "play.place_wait_secs" for name, _ in recorded)
        span.record_exception.assert_called()
        assert any("Place stalled" in r.message for r in caplog.records)

    @pytest.mark.parametrize("in_voice, wire", [(True, "place"), (False, "voice")])
    async def test_the_span_names_the_verdict_even_when_no_command_dropped_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock, in_voice: bool, wire: str
    ) -> None:
        """dropped_by names a COMMAND; a voice refusal has none, so without a
        verdict attribute it is indistinguishable from a successful place."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.queue_source = AsyncMock(return_value=_song(1, mock_ctx))
        if not in_voice:
            mock_ctx.author.voice = None

        with _no_typing(), _recording_span() as span:
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        assert ("play.verdict", wire) in [
            call.args for call in span.set_attribute.call_args_list
        ]

    async def test_a_dropped_request_names_the_song_it_lost(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Three resolving requests get three of these; identical text leaves the
        author unable to tell which one died."""
        mp = _mock_mp()
        mock_ctx.voice_client = _connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        gate = asyncio.Event()

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return _song(1, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_slow)
        with _no_typing():
            task = asyncio.create_task(
                command_callback(MusicBot.play)(
                    music_bot, mock_ctx, url="never gonna give you up"
                )
            )
            await _settle()
            mp.retired = True  # -stop, a kick, the alone-watchdog
            gate.set()
            await task

        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "the session ended" in text
        assert "never gonna give you up" in text

    @pytest.mark.parametrize("cancelled", [0, 1], ids=["creator", "participant"])
    async def test_a_participant_cannot_cancel_the_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock, cancelled: int
    ) -> None:
        join_gate = asyncio.Event()
        music_bot._abandon_cold_start = AsyncMock()
        with _no_typing():
            mp, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            tasks[cancelled].cancel()
            await _settle()
            join = music_bot._plays._guilds[play_key(mock_ctx)].join
            assert join is not None and not join.cancelled()
            join_gate.set()
            await tasks[1 - cancelled]

        assert tasks[cancelled].cancelled()
        assert mp.queue_put_front.await_count == 1  # the survivor placed

    async def test_alone_a_failed_resolve_cancels_its_own_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The single-request rule is unchanged: nobody else waits on the join,
        and a teardown under a still-connecting join leaves join() to rebuild
        the player it then finds missing."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        music_bot.queue_source = AsyncMock(side_effect=RuntimeError("no such song"))
        music_bot._abandon_cold_start = AsyncMock()
        music_bot._command_error = AsyncMock()

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            await _settle()

        music_bot._abandon_cold_start.assert_awaited_once()
        assert not music_bot._plays._guilds  # the cancelled join cleared its slot
        assert mp.playback_holds == 0

    async def test_a_failed_resolve_leaves_a_shared_join_running(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The creator's resolve fails while a participant waits on the join it
        created: the join is the participant's now, and so is the teardown."""
        join_gate = asyncio.Event()
        fail_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)

        async def _resolve(_ctx: Any, _source: Any, **kw: Any) -> QueueObject:
            if kw["origin"] == "bad":
                await fail_gate.wait()
                raise RuntimeError("no such song")
            return _song(2, mock_ctx)

        music_bot.queue_source = AsyncMock(side_effect=_resolve)
        music_bot._command_error = AsyncMock()
        music_bot.cleanup = AsyncMock()

        with _no_typing():
            creator = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="bad")
            )
            await _settle()
            participant = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="good")
            )
            await _settle()
            assert mp.playback_holds == 2
            fail_gate.set()
            await creator
            join = music_bot._plays._guilds[play_key(mock_ctx)].join
            assert join is not None and not join.cancelling()
            join_gate.set()
            await participant

        music_bot.cleanup.assert_not_awaited()
        mp.queue_put_front.assert_awaited_once()
        assert mp.playback_holds == 0

    async def test_a_join_being_cancelled_is_not_handed_to_the_next_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A creator that fails alone cancels its join, which settles a tick
        later. A request arriving in that tick gets a fresh join, not one about
        to raise at it."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        outcomes = iter([RuntimeError("no such song"), _song(2, mock_ctx)])

        async def _resolve(*_a: Any, **_k: Any) -> QueueObject:
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        music_bot.queue_source = AsyncMock(side_effect=_resolve)
        music_bot._command_error = AsyncMock()
        music_bot.cleanup = AsyncMock()

        with _no_typing():
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await _settle()
            # call_count: the dying join never ran, so it was never awaited.
            assert mock_ctx.invoke.call_count == 2  # a second join, not the dying one
            join_gate.set()
            await asyncio.gather(*tasks)

        mp.queue_put_front.assert_awaited_once()
        assert mp.playback_holds == 0

    async def test_a_failed_join_is_torn_down_exactly_once(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Last one out: the first failure sees the other's hold and skips, the
        second sees none and tears down."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        songs = iter([_song(1, mock_ctx), _song(2, mock_ctx)])
        music_bot.queue_source = AsyncMock(side_effect=lambda *_a, **_k: next(songs))
        music_bot.cleanup = AsyncMock()

        async def _failed_join(*_a: Any, **_k: Any) -> None:
            await join_gate.wait()  # voice_client stays None: no usable client

        mock_ctx.invoke = AsyncMock(side_effect=_failed_join)

        with _no_typing():
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await _settle()
            join_gate.set()
            await asyncio.gather(*tasks)

        music_bot.cleanup.assert_awaited_once()
        mp.queue_put_front.assert_not_awaited()

    def test_teardown_decision_and_hold_release_have_no_await_between_them(
        self,
    ) -> None:
        """_abandon_cold_start's skip reads the other participant's hold, and the
        hold is released by leaving the stack. An await between the two — a
        notice, a log flush — lets both participants skip."""
        import inspect

        lines = inspect.getsource(MusicBot._resolve_and_place).splitlines()
        decisions = [
            i
            for i, line in enumerate(lines)
            if "await self._abandon_cold_start(" in line
        ]
        assert decisions, "the cold-start block moved; re-anchor this test"
        for i in decisions:
            following = next(
                line.strip()
                for line in lines[i + 1 :]
                if line.strip() and not line.strip().startswith("#")
            )
            assert following.startswith(("return", "raise")), (i, following)
        # And the decision itself: _abandon_cold_start reads the OTHER
        # participant's hold and returns on it. An await between those two lets
        # that participant reach its own decision while this hold is still
        # counted, and then both skip and nobody tears the player down.
        guard = inspect.getsource(MusicBot._abandon_cold_start).splitlines()
        read = next(n for n, line in enumerate(guard) if "playback_holds > 1" in line)
        assert guard[read + 1].strip().startswith("return"), guard[read + 1]
