"""Tests for src/musicbot.py — voice permission validation, queue source dispatch, and latency color."""

from src.musicplayer import MusicPlayer
import redis.asyncio as aioredis
import asyncio
import contextlib
import orjson
from types import SimpleNamespace
from contextlib import AbstractContextManager
from typing import Any, Optional, cast
from collections.abc import AsyncGenerator, Coroutine, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands
from redis.asyncio import Redis

from src.config import SpotifyStatus
from src.guild_history import GuildHistory
from src.guild_queue import QueueItem, RemoveMode, RemoveOutcome
from src.guild_state import Analytics, HistoryEntry
from src.musicbot import (
    RESTORE_WAIT_SECS,
    _echo,
    _removed_label,
    HISTORY_MAX_LIMIT,
    EmptyPlaylistError,
    HistoryFlags,
    MusicBot,
    PlaylistIndexError,
    ResolvedYoutubePlaylist,
    SpotifyCollectionPager,
    SpotifyDisabledError,
    _ENQUEUE_MAX_WAITERS,
    _ENQUEUE_WAIT_SECS,
    _GuildEnqueueLock,
    _CollectionDrain,
    _check_voice_permissions,
    _is_collection,
    _join_succeeded,
)
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import EMBED_FIELD_LIMIT
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
)
from src.musicplayer import InterjectOutcome, _PLAYBACK_GATE_TIMEOUT
from src.spotify import (
    SpotifyAuthError,
    SpotifyCollection,
    SpotifyRateLimitError,
    SpotifyRequestError,
    TrackPage,
)
from src.youtube import YTDL, QueueObject
from tests.helpers import (
    command_callback,
    make_mock_task,
    mocked,
    queue_object,
)

# Ask-time analytics for direct queue_source/_enqueue_playlist calls — the real
# command paths mint this at dispatch from ctx.message.created_at + enqueue_depth.
_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=0)
_ORIGIN = "https://yt.com/v=origin"


# ── Streamed-collection test helpers ──────────────────────────────────────────


def _scollection(
    kind: SpotifyType,
    *,
    total: int,
    name: Optional[str] = "200% Electronica",
    thumbnail: Optional[str] = "https://i.scdn.co/image/640",
) -> SpotifyCollection:
    """A SpotifyCollection fixture. Playlist collections carry no identity
    fields, matching what /v1/playlists/{id}/tracks can actually yield."""
    if kind is SpotifyType.PLAYLIST:
        return SpotifyCollection(kind=kind, id="cid", total=total)
    return SpotifyCollection(
        kind=kind,
        id="cid",
        total=total,
        name=name,
        artists=["ESPRIT 空想", "George Clanton"],
        thumbnail=thumbnail,
        release_date="2017-11-17",
    )


def _spage(
    collection: SpotifyCollection, titles: list[str], *, is_last: bool
) -> TrackPage:
    return TrackPage(collection=collection, titles=titles, is_last=is_last)


async def _sgen(
    pages: list[TrackPage],
    *,
    fail_at: Optional[int] = None,
    yielded: Optional[list[int]] = None,
) -> AsyncGenerator[TrackPage]:
    """An async generator of TrackPages. fail_at=i raises before yielding
    pages[i] (a mid-drain page-fetch failure); `yielded` records indices as
    they go out, so a test can prove the drain stopped consuming."""
    for i, page in enumerate(pages):
        if fail_at is not None and i == fail_at:
            raise SpotifyAuthError(401, "page fetch failed mid-drain")
        if yielded is not None:
            yielded.append(i)
        yield page


def _collection_mp(
    bot: MusicBot,
    ctx: MagicMock,
    *,
    generation: int = 0,
    backlog: bool = False,
) -> MagicMock:
    """MusicPlayer stand-in for the streamed-enqueue paths.

    Registered in bot.mps under ctx.guild.id because _begin_collection_enqueue
    checks that the player it was handed is still the guild's live one — an
    unregistered mock is indistinguishable from a guild torn down during the
    page-1 fetch, and would take the abandon path instead of the one under
    test."""
    mp = MagicMock()
    mp.queue_put = AsyncMock(return_value=True)
    mp.queue_put_front = AsyncMock(return_value=True)
    mp.queue.generation = generation
    mp.queue.has_restored_backlog = AsyncMock(return_value=backlog)
    bot.mps[ctx.guild.id] = mp
    return mp


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

    async def test_spotify_request_error_renders_its_user_message(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Every page of every collection is a door to this error; the raw args
        carry the endpoint and the offset, which are not the user's business."""
        err = SpotifyRequestError(
            503, "https://api.spotify.com/v1/albums/6WgS/tracks", {"offset": 50}
        )
        with (
            patch("src.musicbot.send_embed", new=AsyncMock()) as send_embed,
            patch("src.musicbot.record_span_error"),
        ):
            await music_bot._command_error(mock_ctx, err)

        assert (call := send_embed.await_args) is not None
        detail = call.args[2]
        assert detail == err.user_message
        assert "api.spotify.com" not in detail
        assert "offset" not in detail

    async def test_unsupported_spotify_link_renders_without_class_name(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The class exists to give the user "an error they can act on";
        falling to the generic arm would prefix it with
        `**UnsupportedSpotifyLinkError:**`, which is noise on the way there."""
        from src.sources import UnsupportedSpotifyLinkError

        err = UnsupportedSpotifyLinkError(
            "Spotify 'artist' links aren't supported — try a track, "
            "playlist or album link"
        )
        with (
            patch("src.musicbot.send_embed", new=AsyncMock()) as send_embed,
            patch("src.musicbot.record_span_error"),
        ):
            await music_bot._command_error(mock_ctx, err)

        assert (call := send_embed.await_args) is not None
        detail = call.args[2]
        assert detail == err.user_message
        assert "UnsupportedSpotifyLinkError" not in detail

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
        assert _join_succeeded(self._ctx(vc)) is True

    def test_still_connecting_client_fails(self) -> None:
        # discord.py registers the client on the guild BEFORE the handshake, so this
        # is a real state a concurrent cold -play leaves behind — not a mock artifact.
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_connected.return_value = False
        assert _join_succeeded(self._ctx(vc)) is False

    def test_absent_client_fails(self) -> None:
        # join swallows its own failures, so a failed join arrives as None.
        assert _join_succeeded(self._ctx(None)) is False

    def test_non_voice_client_fails(self) -> None:
        assert _join_succeeded(self._ctx(MagicMock())) is False


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
    async def test_spotify_playlist_returns_pager(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        col = _scollection(SpotifyType.PLAYLIST, total=2)
        gen = _sgen([_spage(col, ["Song A", "Song B"], is_last=True)])
        music_bot.spotify.playlist_stream = MagicMock(return_value=gen)

        result = await music_bot.queue_source(
            mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
        )

        assert isinstance(result, SpotifyCollectionPager)
        assert result.kind is SpotifyType.PLAYLIST
        music_bot.spotify.playlist_stream.assert_called_once_with("pid123")
        # Lazy: page 1 is fetched only when the enqueue path starts the pager.
        page1 = await anext(result.pages)
        assert page1.titles == ["Song A", "Song B"]
        await result.aclose()

    async def test_spotify_album_returns_pager(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None
        col = _scollection(SpotifyType.ALBUM, total=1)
        gen = _sgen([_spage(col, ["Track 0 Artist"], is_last=True)])
        music_bot.spotify.album_stream = MagicMock(return_value=gen)

        result = await music_bot.queue_source(
            mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
        )

        assert isinstance(result, SpotifyCollectionPager)
        assert result.kind is SpotifyType.ALBUM
        music_bot.spotify.album_stream.assert_called_once_with("aid123")
        # album_stream chosen, not playlist_stream — the two endpoints and
        # unwraps differ; routing an album through playlist_stream
        # would KeyError on the missing ["track"] wrapper.
        await result.aclose()

    async def test_pager_is_lazy_leaving_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """S2 inverted the old H7 guard: queue_source must return WITHOUT
        starting the pager. No task exists to cancel or retrieve on abandon
        paths, and every fetch — page 1 included — happens inside
        _begin_collection_enqueue's bounded anext. An eager fetch sneaking
        back in would resurrect the hand-managed lifecycle S2 removed."""
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None
        started = asyncio.Event()
        col = _scollection(SpotifyType.ALBUM, total=1)

        async def gen() -> AsyncGenerator[TrackPage]:
            started.set()  # first line of the generator body = the HTTP call site
            yield _spage(col, ["T"], is_last=True)

        music_bot.spotify.album_stream = MagicMock(return_value=gen())

        result = await music_bot.queue_source(
            mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert isinstance(result, SpotifyCollectionPager)
        await asyncio.sleep(0)  # a tick an eager task would have used to run
        assert not started.is_set()
        # The fetch happens exactly when the enqueue path asks for it.
        page1 = await anext(result.pages)
        assert started.is_set() and page1.titles == ["T"]
        await result.aclose()

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
        # _resolve_playnow_source resolves both collection shapes directly, so a
        # token passed only from queue_source would leave these two unclassified.
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None
        col = _scollection(SpotifyType.PLAYLIST, total=2)
        music_bot.spotify.playlist_stream = MagicMock(
            return_value=_sgen([_spage(col, ["Song A", "Song B"], is_last=True)])
        )
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
        assert result == ResolvedYoutubePlaylist(
            tracks=fake_qobjs,
            playlist_url="https://www.youtube.com/playlist?list=PLtest123",
        )

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
        url = "https://www.youtube.com/playlist?list=PLtest"
        qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author),
            QueueObject("https://yt.com/watch?v=2", "Track 2", mock_ctx.author),
        ]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx, ResolvedYoutubePlaylist(tracks=qobjs, playlist_url=url), mp
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "2 songs" in embed.title
        assert url in embed.description
        assert "Track 1" in embed.description

    async def test_yt_embed_states_the_skipped_songs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A shorter queue than the playlist needs an explanation, and only the
        user's own `index=` provides one."""
        url = "https://www.youtube.com/watch?v=x&list=PLtest&index=4"
        qobjs = [QueueObject("https://yt.com/watch?v=4", "Track 4", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            ResolvedYoutubePlaylist(tracks=qobjs, playlist_url=url, skipped=3),
            mp,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Starting at #4" in embed.description
        assert "skipped 3 earlier songs" in embed.description

    async def test_yt_embed_omits_the_skip_line_when_nothing_was_skipped(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            ResolvedYoutubePlaylist(
                tracks=qobjs,
                playlist_url="https://www.youtube.com/playlist?list=PLtest",
            ),
            mp,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Starting at" not in embed.description

    async def test_yt_singular_song_count_in_title(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Only Track", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            ResolvedYoutubePlaylist(
                tracks=qobjs,
                playlist_url="https://www.youtube.com/playlist?list=PLtest",
            ),
            mp,
        )

        embed = mock_ctx.send.call_args[1]["embed"]
        assert "1 song" in embed.title
        assert "1 songs" not in embed.title

    async def test_yt_calls_queue_put_with_prefetch_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        qobjs = [QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author)]
        mp = self._make_enqueue_mp(mock_ctx)

        await music_bot._enqueue_playlist(
            mock_ctx,
            ResolvedYoutubePlaylist(
                tracks=qobjs,
                playlist_url="https://www.youtube.com/playlist?list=PLtest",
            ),
            mp,
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
        mp.retire_np_host_on_stop = AsyncMock()
        mp.update_activity = AsyncMock()
        # cleanup() awaits this on every teardown (the stream-preemption bump).
        mp.queue.bump_generation = AsyncMock()
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

    async def test_cleanup_bumps_queue_generation(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """cleanup() is the single teardown choke point — -stop, a kick, the
        alone-disconnect timer, the gate timeout, and play's error path all
        funnel through it. The generation bump must land here (not in the
        stop/clear commands alone) so an in-flight collection drain is refused
        after ANY teardown; without it the drain keeps RPUSHing the Redis
        mirror of a guild being torn down."""
        mp = self._make_minimal_mp(music_bot, mock_guild)
        mock_guild.voice_client = None
        await music_bot.cleanup(mock_guild)
        mp.queue.bump_generation.assert_awaited_once()

    async def test_cleanup_bumps_generation_before_cancelling_tasks(
        self, music_bot: MusicBot, mock_guild: MagicMock
    ) -> None:
        """Bumping before the cancellations is what stops a page landing after
        teardown: bumped after, a drain page parked on the queue mutex commits
        to the mirror of a guild mid-teardown."""
        events: list[str] = []

        async def parked() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events.append("cancelled")
                raise

        task = asyncio.create_task(parked())
        await asyncio.sleep(0)  # park it
        mp = self._make_minimal_mp(music_bot, mock_guild, _prefetch_task=task)
        mp.queue.bump_generation = AsyncMock(side_effect=lambda: events.append("bump"))
        mock_guild.voice_client = None

        await music_bot.cleanup(mock_guild)

        assert events == ["bump", "cancelled"]

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
        mp.queue.bump_generation = AsyncMock()
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

        mp = MagicMock()
        mp.current_song = MagicMock(title="Paused Song", position_secs=83.4)
        music_bot.get_mp = MagicMock(return_value=mp)

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

        mp = MagicMock()
        mp.current_song = MagicMock(title="Paused Song", position_secs=83.4)
        music_bot.get_mp = MagicMock(return_value=mp)
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
        mock_ctx.voice_client = _playing_vc()
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
            mock_ctx.voice_client = _paused_vc()

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
        and a -playnow stack loses its rows too, because _flush_played records
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


def _no_typing() -> AbstractContextManager[MagicMock]:
    """Stub play()'s background_typing wrapper with an inert async CM: TestPlayCommand
    patches asyncio.create_task as a join-task spy, and without this the typing
    keepalive hits the same patch, polluting call counts and taking the fake join
    future. The wrapper itself is covered by TestBackgroundTyping."""
    return patch(
        "src.musicbot.background_typing",
        MagicMock(return_value=contextlib.nullcontext()),
    )


def _connected_vc() -> MagicMock:
    """Connected voice client, nothing playing — what a successful cold join leaves
    behind. is_connected is explicit: the cold path checks it, because discord.py
    registers the client on the guild before the handshake completes."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return vc


def _playing_vc() -> MagicMock:
    """Connected voice client, actively playing. Both flags must be set explicitly:
    an unstubbed is_paused() returns a truthy Mock, silently sending -play down the
    interjection branch instead of the append path."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return vc


def _paused_vc() -> MagicMock:
    """Connected voice client with a song parked paused. is_connected is explicit:
    -resume's rejoin checks it, and an auto-vivified one answers True by accident
    rather than by choice."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = True
    vc.is_connected.return_value = True
    return vc


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
    mp.repark_crashed_head = AsyncMock()
    mp.queue_put_front = AsyncMock()
    mp.queue_put = AsyncMock()
    mp.queue.qsize = MagicMock(return_value=qsize)
    # Numeric for the same reason as playback_holds: this lands in
    # Analytics.queue_position and rides to Postgres through HistoryEntry's
    # integer clamp, which a Mock raises on rather than answering.
    mp.enqueue_depth = MagicMock(return_value=qsize)
    # Mirrors the real builder's contract: a notice only when the restore
    # actually left something in the queue (see build_resume_notice_embed).
    mp.build_resume_notice_embed = MagicMock(
        return_value=discord.Embed(title="❗ Resumed from queue") if qsize else None
    )
    return mp


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
            mock_ctx.voice_client = _connected_vc()  # what a real join leaves
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
        mock_ctx.voice_client = _playing_vc()
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
    """The ask-time Analytics -play mints and hands to queue_source.

    Asserted on the call rather than on a returned object: queue_source is what
    carries the value into every construction site, and nothing downstream
    restamps it, so the hand-off IS the behavior."""

    async def test_warm_path_carries_the_ask_time_and_the_player_depth(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = _playing_vc()
        mp = _mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        music_bot.queue_source = spy
        music_bot._enqueue_single = AsyncMock()

        with _no_typing():
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
            mock_ctx.voice_client = _connected_vc()
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
        mock_ctx.voice_client = _playing_vc()
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
        music_bot._enqueue_single = AsyncMock()

        with _no_typing():
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        assert spy.await_args.kwargs["analytics"].queue_position == 12


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
        # The collection path resumes instead of interjecting.
        mp.resume = AsyncMock()
        mp.rehost_np_after_resume = AsyncMock()
        return mp

    async def test_interjects_with_resume_paused_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = _paused_vc()
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
        mock_ctx.voice_client = _paused_vc()
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
        mock_ctx.voice_client = _playing_vc()
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
        mock_ctx.voice_client = _paused_vc()
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
        vc = _paused_vc()
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
            _no_typing(),
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
        vc = _paused_vc()
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

    async def test_collection_queues_in_full_and_resumes(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A paused -play <collection> must not interject.

        Interjection resolves to exactly one song, so collapsing here would
        discard the rest of the playlist and answer with "use -play for the
        full playlist" — the command the user just ran. It queues in full and
        playback resumes afterwards instead, so -play still means play.
        """
        vc = _paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        order: list[str] = []
        mp.queue_put.side_effect = lambda *a, **k: order.append("enqueue") or True
        mp.resume.side_effect = lambda *a, **k: order.append("resume")
        mock_ctx.message.add_reaction = AsyncMock()
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"
        # parse_input splits the full message to count args — an unset MagicMock
        # content makes every URL fall back to the ytsearch branch.
        mock_ctx.message.content = f"-play {url}"

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_not_awaited()
        mp.queue_put.assert_awaited_once()
        assert list(mp.queue_put.await_args.args[0]) == tracks
        mp.resume.assert_awaited_once_with(vc)
        # Enqueue strictly BEFORE resume: resuming first would restart the
        # paused song only for a failed resolve to leave the user with
        # playback they did not ask to change. Both calls were asserted;
        # their order was not, so hoisting the resume stayed green.
        assert order == ["enqueue", "resume"]
        sent = mock_ctx.send.await_args_list + mock_ctx.send.call_args_list
        notices = [
            c.kwargs["embed"].description
            for c in sent
            if c.kwargs.get("embed") is not None
        ]
        assert not any("first track" in (d or "") for d in notices), notices

    async def test_resume_skipped_when_a_resume_already_landed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """_resume_after_collection mirrors -resume's guard instead of calling
        it, so a -resume that lands while the collection queues is a no-op
        here rather than a second resume() + a spurious NP-host migration on
        an already-playing song. The guard never evaluated False anywhere in
        the suite, so deleting it was green."""
        vc = _paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        def _user_resumed_mid_drain(*a: Any, **k: Any) -> bool:
            # The -resume lands during the enqueue: by the time
            # _resume_after_collection re-reads the state, playback is live.
            vc.is_playing.return_value = True
            vc.is_paused.return_value = False
            return True

        mp.queue_put.side_effect = _user_resumed_mid_drain
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        mock_ctx.message.add_reaction = AsyncMock()
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"
        mock_ctx.message.content = f"-play {url}"

        with (
            _no_typing(),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.queue_put.assert_awaited_once()
        mp.resume.assert_not_awaited()
        mp.rehost_np_after_resume.assert_not_awaited()


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
            mock_ctx.voice_client = _connected_vc()  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
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
            mock_ctx.voice_client = _connected_vc()
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
        mock_ctx.voice_client = _playing_vc()
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        music_bot.queue_source = AsyncMock(return_value=fake_qobj)
        music_bot._enqueue_single = AsyncMock()
        mp = _mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        with _no_typing(), patch("asyncio.create_task"):
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
            mock_ctx.voice_client = _connected_vc()  # what a real join leaves
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
            mock_ctx.voice_client = _connected_vc()  # what a real join leaves
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
        mp = _mock_mp(qsize=0)
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
        mp = _mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._enqueue_playlist(
            mock_ctx,
            ResolvedYoutubePlaylist(
                tracks, playlist_url="https://yt.com/playlist?list=X"
            ),
            mp,
            front=True,
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

        music_bot.queue_source = AsyncMock(
            return_value=ResolvedYoutubePlaylist(
                tracks, playlist_url="https://yt.com/playlist?list=X"
            )
        )
        music_bot._enqueue_playlist = AsyncMock()
        music_bot._enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=_mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = _connected_vc()  # what a real join leaves
            return join_task

        with _no_typing(), patch("asyncio.create_task", side_effect=fake_create_task):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        music_bot._enqueue_single.assert_not_awaited()
        pl_call = music_bot._enqueue_playlist.await_args
        assert pl_call is not None
        assert pl_call.kwargs["front"] is True
        assert pl_call.args[1] == ResolvedYoutubePlaylist(
            tracks, playlist_url="https://yt.com/playlist?list=X"
        )

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
    async def test_shows_queued_embed_with_eta_when_song_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

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

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        mp.estimated_playing_at.assert_not_called()
        mock_ctx.send.assert_not_awaited()

    async def test_queued_embed_has_thumbnail_when_present(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
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

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert embed.thumbnail.url == "https://img.youtube.com/vi/1/0.jpg"

    async def test_queued_embed_has_no_thumbnail_when_absent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()
        mp.estimated_playing_at.return_value = "**7:42 PM PST**"

        await music_bot._enqueue_single(mock_ctx, qobj, mp)

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
        `](` pair can survive to pick a link's label and destination. Escaping
        alone never covered this — brackets are not in escape_markdown's set."""
        attack = "[Free Discord Nitro](https://evil.example/phish)"
        out = _echo(attack)
        assert "[" not in out and "]" not in out

    def test_markdown_riding_behind_a_url_is_neutralized(self) -> None:
        """The regression that shipped: escape_markdown defaults to
        ignore_links=True, which passes any http(s) URL through UNTOUCHED — so an
        attack prefixed with a bare link was echoed verbatim, and the earlier
        bracket test passed only because its attack started at the bracket."""
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
        so the user sees "Command failed" for a removal that happened.

        `*`, not `x`: escaping leaves `x` alone, so the old input never exercised
        the doubling this test's own docstring is about."""
        assert len(_echo("*" * 5000)) <= 1024

    def test_an_ordinary_needle_is_unchanged_apart_from_the_span(self) -> None:
        assert _echo("never gonna give you up") == "never gonna give you up"


class TestRemovedLabelNamesEveryItemType:
    """The Songs field exists because one argument can now take out a whole
    playlist and there is no undo. `YTSource` has no `.title` at all, so reaching
    for it rendered every unresolved Spotify-playlist track as `?` — the exact
    case the field was added for, and the one the -remove help now advertises."""

    def test_a_resolved_song_uses_its_title(self, mock_author: MagicMock) -> None:
        item = QueueObject("https://yt.com/v=1", "Real Title", mock_author)
        assert _removed_label(item) == "Real Title"

    def test_an_unresolved_search_uses_its_search_text(self) -> None:
        item = YTSource(ytsearch="ytsearch:Artist - Song", process=True)
        assert _removed_label(item) == "Artist - Song"

    def test_an_unresolved_link_falls_back_to_the_url(self) -> None:
        item = YTSource(url="https://yt.com/v=2", process=True)
        assert _removed_label(item) == "https://yt.com/v=2"


class TestRemoveReplyStaysInsideDiscordsCaps:
    """Every field of the `-remove` reply is built from a list the USER sizes —
    the removed songs and their positions — and the send happens AFTER
    queue_remove() has already mutated memory and Redis. So an over-length field
    is not a cosmetic bug: Discord 400s the whole send, `_command_error` reports
    "Command failed", and the user is told nothing happened to a queue that has
    already been irreversibly changed.

    Asserted on the ASSEMBLED embed rather than on `_echo` alone. The bug these
    pin shipped past a test that checked one echo against the cap while ten of
    them shared the same field."""

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
        """99 characters is INSIDE YouTube's own 100-char title limit, so this
        needs no crafted content — ten ordinary songs used to overflow at 1030."""
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
    """`-play`, `-playnow` and `-remove` all consume the rest of the line.

    A positional binds ONE WORD. `-play` stores its argument as the origin
    `-remove` matches on, so a positional there meant `-play never gonna give you
    up` recorded `"never"` — the help's own example matched nothing, and
    `-remove never` became a wildcard over every song starting with that word.

    Asserted on the callback signature rather than through a parsed message,
    because the binding is a property of the signature and the tests that missed
    this were the ones that hand-built the value instead."""

    @pytest.mark.parametrize("name", ["play", "playnow", "remove"])
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
        mock_ctx.voice_client = _connected_vc()
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
        col = _scollection(SpotifyType.PLAYLIST, total=2)
        # Page 1 only, via the stream: one HTTP call, generator abandoned
        # under aclosing, no cache write (the pagers cache on full drain only).
        music_bot.spotify.playlist_stream = MagicMock(
            return_value=_sgen([_spage(col, ["First Song", "Second"], is_last=True)])
        )
        qobj = QueueObject("https://yt.com/v=first", "First Song", mock_ctx.author)

        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=qobj)
        ) as ys:
            await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url=url)

        music_bot.spotify.playlist_stream.assert_called_once_with(
            "37i9dQZF1DXcBWIGoYBM5M"
        )
        ys.assert_awaited_once()
        assert ys.call_args.args[1] == "ytsearch:First Song"
        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is qobj
        # First-track notice + confirmation.
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs and c.kwargs["embed"].description
        ]
        assert any("first track" in d for d in notices)

    async def test_spotify_album_interjects_first_track_with_album_copy(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """-playnow on an album: page 1 of the stream, first track only, and
        the notice says album — not a playlist embed with different words."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        mock_ctx.message.content = f"-playnow {url}"
        assert music_bot.spotify is not None
        col = _scollection(SpotifyType.ALBUM, total=11)
        consumed: list[int] = []
        music_bot.spotify.album_stream = MagicMock(
            return_value=_sgen(
                [
                    _spage(
                        col,
                        ["Iridescence ESPRIT 空想", "Second Track"],
                        is_last=False,
                    ),
                    _spage(col, ["Third Track", "Fourth Track"], is_last=True),
                ],
                yielded=consumed,
            )
        )
        qobj = QueueObject("https://yt.com/v=first", "Iridescence", mock_ctx.author)

        with patch(
            "src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=qobj)
        ) as ys:
            await command_callback(MusicBot.playnow)(music_bot, mock_ctx, url=url)

        music_bot.spotify.album_stream.assert_called_once_with("6WgSCcRfaXuBVfM2TpV0Kl")
        # Page-1-only, pinned against a MULTI-page stream: a mutation that
        # drains everything before taking track 1 fetches page 2.
        assert consumed == [0]
        ys.assert_awaited_once()
        assert ys.call_args.args[1] == "ytsearch:Iridescence ESPRIT 空想"
        live_mp.interject.assert_awaited_once()
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs and c.kwargs["embed"].description
        ]
        assert any("Albums can't be interjected" in n for n in notices)
        assert any("full album" in n for n in notices)

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


# ── Streamed collection enqueue ──────────────────────────────────────────────


class TestIsCollection:
    def test_spotify_album_and_playlist_are_collections(self) -> None:
        assert _is_collection(SpotifySource(SpotifyType.ALBUM, "a")) is True
        assert _is_collection(SpotifySource(SpotifyType.PLAYLIST, "p")) is True

    def test_spotify_track_is_not(self) -> None:
        assert _is_collection(SpotifySource(SpotifyType.TRACK, "t")) is False

    def test_yt_playlist_is_a_collection_but_track_is_not(self) -> None:
        assert _is_collection(YTSource(type=YTType.PLAYLIST, list_id="PL1")) is True
        assert _is_collection(YTSource(url="https://yt.com/watch?v=1")) is False

    def test_soundcloud_and_search_are_not(self) -> None:
        assert _is_collection(SoundcloudSource(url="https://sc.com/a/t")) is False
        assert _is_collection(YTSource(ytsearch="ytsearch:despacito")) is False


class TestEnqueueGuardrails:
    def test_bounded_wait_stays_well_under_the_gate_timeout(self) -> None:
        """G3 tripwire: a lock-waiter's player already exists (get_mp runs in
        cog_before_invoke, before any command body), so on a disconnected bot
        its 300s gate clock is ticking WHILE it waits. A wait bound at or above
        the gate timeout lets the player be torn down under a still-waiting
        command, which then enqueues into a dead guild."""
        assert _ENQUEUE_WAIT_SECS < _PLAYBACK_GATE_TIMEOUT
        # "well under": leave real headroom, not a 1-second technicality.
        assert _ENQUEUE_WAIT_SECS <= _PLAYBACK_GATE_TIMEOUT / 2

    def test_timeout_chain_nests(self) -> None:
        """The full documented nesting: one HTTP request fits inside a drain
        leg, a drain leg inside a waiter's patience. Only the gate leg was
        pinned before; the branch that added the middle constants deleted the
        only assertion on _HTTP_TIMEOUT's value, so any of them could drift
        past its neighbour with a green suite. (The lock HOLD spans several
        legs and may legitimately exceed _ENQUEUE_WAIT_SECS — see the comment
        on _COLLECTION_DRAIN_TIMEOUT_SECS — but each leg must stay inside the
        next bound up or its timeout fires inside an in-flight request.)"""
        from src.musicbot import _COLLECTION_DRAIN_TIMEOUT_SECS
        from src.spotify import _HTTP_TIMEOUT

        assert _HTTP_TIMEOUT.total is not None
        assert _HTTP_TIMEOUT.total < _COLLECTION_DRAIN_TIMEOUT_SECS
        assert _COLLECTION_DRAIN_TIMEOUT_SECS < _ENQUEUE_WAIT_SECS


class TestAcquireEnqueueSlot:
    async def test_uncontended_acquire_returns_held_slot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        slot = await music_bot._acquire_enqueue_slot(mock_ctx)
        assert slot is not None
        assert slot.lock.locked()
        mock_ctx.send.assert_not_awaited()  # no ack when nobody is ahead
        slot.lock.release()

    async def test_waiter_gets_ack_then_the_slot_in_order(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        first = await music_bot._acquire_enqueue_slot(mock_ctx)
        assert first is not None

        waiter = asyncio.create_task(music_bot._acquire_enqueue_slot(mock_ctx))
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # let the waiter send its ack and park
        acks = [c.kwargs["embed"].description for c in mock_ctx.send.call_args_list]
        assert any("Waiting for another album/playlist" in a for a in acks)
        assert not waiter.done()

        first.lock.release()
        second = await waiter
        assert second is not None
        assert second.lock.locked()
        assert second.waiters == 0  # count restored after the wait
        second.lock.release()

    async def test_bounded_wait_times_out_with_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        first = await music_bot._acquire_enqueue_slot(mock_ctx)
        assert first is not None
        try:
            with patch("src.musicbot._ENQUEUE_WAIT_SECS", 0.01):
                slot = await music_bot._acquire_enqueue_slot(mock_ctx)
            assert slot is None
            notices = [
                c.kwargs["embed"].description for c in mock_ctx.send.call_args_list
            ]
            assert any("Still queueing a large collection" in n for n in notices)
            assert first.waiters == 0  # the timed-out waiter deregistered
        finally:
            first.lock.release()

    async def test_backlog_cap_declines_immediately(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        first = await music_bot._acquire_enqueue_slot(mock_ctx)
        assert first is not None
        try:
            first.waiters = _ENQUEUE_MAX_WAITERS
            slot = await music_bot._acquire_enqueue_slot(mock_ctx)
            assert slot is None
            notices = [
                c.kwargs["embed"].description for c in mock_ctx.send.call_args_list
            ]
            assert any("Too many albums/playlists" in n for n in notices)
        finally:
            first.waiters = 0
            first.lock.release()

    async def test_backlog_cap_reached_through_real_waiters(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The cap's two halves were only tested in isolation: the decline
        above compares against a HAND-SET counter, so moving `waiters += 1`
        below the ack send — a plausible "acknowledge first" refactor — or
        deleting the increment/decrement pair entirely kept every assertion
        green while a gateway burst of collections stacked unbounded. Here the
        counter is driven only by real waiting acquires."""

        def _wctx() -> MagicMock:
            ctx = MagicMock()
            ctx.guild.id = mock_ctx.guild.id
            ctx.send = AsyncMock()
            return ctx

        holder = await music_bot._acquire_enqueue_slot(mock_ctx)
        assert holder is not None
        waiter_tasks: list[asyncio.Task[Any]] = []
        try:
            for _ in range(_ENQUEUE_MAX_WAITERS):
                waiter_tasks.append(
                    asyncio.create_task(music_bot._acquire_enqueue_slot(_wctx()))
                )
            # Let each waiter send its ack and park on the acquire.
            for _ in range(10):
                await asyncio.sleep(0)
            # Counted by the production increments, not by this test.
            assert holder.waiters == _ENQUEUE_MAX_WAITERS

            declined = await music_bot._acquire_enqueue_slot(mock_ctx)
            assert declined is None
            notices = [
                c.kwargs["embed"].description for c in mock_ctx.send.call_args_list
            ]
            assert any("Too many albums/playlists" in n for n in notices)
        finally:
            # Unwind FIFO: each waiter wins the lock in turn and releases it.
            holder.lock.release()
            for task in waiter_tasks:
                slot = await task
                assert slot is not None
                slot.lock.release()
        # The decrement wire: every waiter that stopped waiting was uncounted.
        assert holder.waiters == 0

    async def test_fast_path_is_bounded_during_release_handoff(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """During asyncio.Lock's release→wakeup handoff, locked() reads False
        while the woken waiter is still queued, so the fast path's acquire()
        parks behind it. It must park BOUNDED — without the timeout this call
        blocks unboundedly with no notice, no cap, no waiter count."""
        monkeypatch.setattr("src.musicbot._ENQUEUE_WAIT_SECS", 0.05)
        entry = music_bot._enqueue_locks.setdefault(
            mock_ctx.guild.id, _GuildEnqueueLock()
        )
        await entry.lock.acquire()  # the drain holding the slot
        waiter = asyncio.create_task(entry.lock.acquire())  # a parked waiter
        await asyncio.sleep(0)
        entry.lock.release()  # the handoff window
        assert not entry.lock.locked()  # ...in which locked() reads False

        slot = await music_bot._acquire_enqueue_slot(mock_ctx)

        assert slot is None  # timed out instead of parking forever
        notices = [c.kwargs["embed"].description for c in mock_ctx.send.call_args_list]
        assert any("Still queueing" in n for n in notices)
        await waiter  # the woken waiter did get the lock
        entry.lock.release()


class TestPlayAdmission:
    """The tier rule at the top of play(): collections take the per-guild
    lock, singles never do (M5 → b)."""

    async def test_single_track_never_takes_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        mock_ctx.message.content = "-play despacito"
        music_bot._acquire_enqueue_slot = AsyncMock()
        music_bot._play_resolved = AsyncMock()

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="despacito")

        music_bot._acquire_enqueue_slot.assert_not_awaited()
        music_bot._play_resolved.assert_awaited_once()

    async def test_collection_takes_and_releases_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        mock_ctx.message.content = f"-play {url}"
        held_during_body: list[bool] = []

        async def probe(*args: Any, **kwargs: Any) -> None:
            held_during_body.append(
                music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()
            )

        music_bot._play_resolved = AsyncMock(side_effect=probe)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        music_bot._play_resolved.assert_awaited_once()
        # Held ACROSS the body — an acquire-release-immediately would pass a
        # released-after check alone — then released.
        assert held_during_body == [True]
        entry = music_bot._enqueue_locks[mock_ctx.guild.id]
        assert not entry.lock.locked()

    async def test_declined_admission_skips_resolution(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        mock_ctx.message.content = f"-play {url}"
        music_bot._acquire_enqueue_slot = AsyncMock(return_value=None)
        music_bot._play_resolved = AsyncMock()

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        music_bot._play_resolved.assert_not_awaited()

    async def test_lock_released_when_resolution_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        mock_ctx.message.content = f"-play {url}"
        music_bot._play_resolved = AsyncMock(side_effect=RuntimeError("boom"))

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        entry = music_bot._enqueue_locks[mock_ctx.guild.id]
        assert not entry.lock.locked()


class TestSpotifyCollectionPagerAclose:
    """The lazy pager's close contract (S2 removed the eager page-1 task and
    with it the settled-task arms this class used to cover): construction
    performs no work, a started generator is finalized by aclose(), and the
    no-op arms — unstarted, already-finished, double-close — must not raise."""

    async def test_construction_starts_nothing_and_aclose_is_a_noop(self) -> None:
        """The S2 property itself: building a pager fires no I/O, so an
        abandon path that never started it (a failed voice join) has nothing
        to cancel, retrieve, or finalize — and wastes no Spotify request."""
        entered: list[bool] = []

        async def _gen() -> AsyncGenerator[TrackPage]:
            entered.append(True)
            yield _spage(
                _scollection(SpotifyType.ALBUM, total=1), ["T A"], is_last=True
            )

        resolved = SpotifyCollectionPager(SpotifyType.ALBUM, _gen())
        assert entered == []  # lazy: nothing ran at construction
        await resolved.aclose()  # unstarted: must not raise…
        assert entered == []  # …and must not start it either

    async def test_aclose_finalizes_a_started_generator(self) -> None:
        """A pager suspended at a yield is the state every begin-then-bail
        path leaves behind; aclose() must run its finally now rather than
        deferring to asyncgen hooks (whose warning is a hard failure under
        filterwarnings=["error"]). Double-close is a no-op."""
        closed: list[bool] = []

        async def _gen() -> AsyncGenerator[TrackPage]:
            try:
                yield _spage(
                    _scollection(SpotifyType.ALBUM, total=2), ["T A"], is_last=False
                )
                yield _spage(
                    _scollection(SpotifyType.ALBUM, total=2), ["U A"], is_last=True
                )
            finally:
                closed.append(True)

        resolved = SpotifyCollectionPager(SpotifyType.ALBUM, _gen())
        await anext(resolved.pages)  # start it, park at the first yield
        await resolved.aclose()
        assert closed == [True]
        await resolved.aclose()  # already finished: must not raise
        assert closed == [True]


class TestPlayCollectionIntegration:
    """-play <collection> driven end-to-end through the REAL glue: play()'s
    command body, a real SpotifyCollectionPager from the real queue_source, a
    real GuildQueue, and the fake-redis mirror. Every other class in this file
    mocks at least one of those layers, which left the dispatch in
    _play_resolved, the gate/tail ordering, and the cross-page mirror order
    pinned by nothing."""

    _URL = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"

    def _wire(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
        pages: AsyncGenerator[TrackPage],
    ) -> None:
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.mps[mock_ctx.guild.id] = music_player
        music_bot.spotify.album_stream = MagicMock(return_value=pages)
        mock_ctx.message.content = f"-play {self._URL}"

    @staticmethod
    def _join_lands(mock_ctx: MagicMock) -> AsyncMock:
        """ctx.invoke for a join that CONNECTS. A bare AsyncMock leaves
        voice_client None, which _join_succeeded rejects — the cold path then
        abandons and nothing reaches the queue."""

        async def _invoke(*_args: object, **_kwargs: object) -> None:
            mock_ctx.voice_client = _connected_vc()

        return AsyncMock(side_effect=_invoke)

    async def test_page1_failure_reports_and_closes_the_generator(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
    ) -> None:
        """Spotify down on the page-1 call — the mainline outage path.

        queue_source returns the lazy stream; _begin_collection_enqueue's
        anext starts it, the raise happens there, drain stays None, and
        _play_resolved's finally aclose()s a generator that already finished
        raising — a no-op that must not mask the real error. The user must
        still be told, and the enqueue lock released.
        """

        async def _boom() -> AsyncGenerator[TrackPage]:
            raise SpotifyAuthError(401, "page 1 failed")
            yield  # pragma: no cover - unreachable, makes this a generator

        self._wire(music_bot, music_player, mock_ctx, _boom())

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert music_player.queue.qsize() == 0
        # The lock is released even though the command failed.
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()
        errors = [
            c.kwargs["embed"]
            for c in mock_ctx.send.call_args_list + mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert errors, "the user was told nothing"

    async def test_join_failure_leaves_pager_unstarted_and_cleans_up(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
    ) -> None:
        """queue_source SUCCEEDS and the voice join then raises. Under the
        lazy pager (S2) this path is inert by construction: page 1 has not
        been fetched, so there is no in-flight task to cancel and no Spotify
        request wasted — the property the eager design bought with a
        hand-managed lifecycle. The generator must never be STARTED by the
        abandon path either (aclose on an unstarted generator is a no-op),
        and full cleanup must still run."""
        col = _scollection(SpotifyType.ALBUM, total=4)
        entered: list[bool] = []

        async def gen() -> AsyncGenerator[TrackPage]:
            entered.append(True)
            yield _spage(col, ["T0 A", "T1 A"], is_last=False)
            yield _spage(col, ["T2 A", "T3 A"], is_last=True)

        self._wire(music_bot, music_player, mock_ctx, gen())
        mock_ctx.voice_client = None  # front path: the join actually runs
        mock_ctx.invoke = AsyncMock(
            side_effect=discord.ClientException("voice channel is full")
        )

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert entered == [], "a failed join must not cost a Spotify fetch"
        # The join failure ran full cleanup: player gone, nothing queued, and
        # the enqueue lock released for the next command.
        assert mock_ctx.guild.id not in music_bot.mps
        assert music_player.queue.qsize() == 0
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()

    async def test_join_without_a_client_abandons_before_the_pager_starts(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
    ) -> None:
        """A join that returns WITHOUT connecting — _join_succeeded's case,
        which leaves by `return` rather than by raising. The collection must
        take the same exit as a raising join: nothing queued, the player torn
        down, the lock released, and no Spotify fetch spent on a bot that is
        not in voice."""
        col = _scollection(SpotifyType.ALBUM, total=4)
        entered: list[bool] = []

        async def gen() -> AsyncGenerator[TrackPage]:
            entered.append(True)
            yield _spage(col, ["T0 A", "T1 A"], is_last=True)

        self._wire(music_bot, music_player, mock_ctx, gen())
        mock_ctx.voice_client = None
        mock_ctx.invoke = AsyncMock()  # join reports its own failure, connects nothing

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert entered == [], "a join that connected nothing must not cost a fetch"
        assert mock_ctx.guild.id not in music_bot.mps
        assert music_player.queue.qsize() == 0
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()

    async def test_unreadable_restore_abandons_the_collection_and_says_so(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The bounded restore wait timing out on the front path. Front-inserting
        a collection against an unread snapshot double-queues every page, so the
        whole collection is abandoned — and unlike the join paths, this one owes
        the user a notice, because nothing else reported anything."""
        col = _scollection(SpotifyType.ALBUM, total=4)
        entered: list[bool] = []

        async def gen() -> AsyncGenerator[TrackPage]:
            entered.append(True)
            yield _spage(col, ["T0 A", "T1 A"], is_last=True)

        self._wire(music_bot, music_player, mock_ctx, gen())
        mock_ctx.voice_client = None
        mock_ctx.invoke = self._join_lands(mock_ctx)
        # The real timeout, not a stubbed return: an event that is never set
        # against a 10ms budget can only expire.
        music_player._restore_complete.clear()
        monkeypatch.setattr("src.musicbot.RESTORE_WAIT_SECS", 0.01)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert entered == [], "an unread restore must not cost a Spotify fetch"
        assert music_player.queue.qsize() == 0
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()
        told = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list + mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert any("saved queue" in (d or "") for d in told), told

    async def test_begin_bailing_after_page1_finalizes_the_generator(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
    ) -> None:
        """The abandon path that still carries the leak risk under the lazy
        pager: _begin_collection_enqueue consumed page 1 (the generator is
        started, suspended at its yield) and then bailed — here, the
        empty-collection raise. drain stays None, so _play_resolved's finally
        must aclose() the suspended generator NOW; leaving it to asyncgen-hook
        finalization is a hard suite failure under filterwarnings=["error"],
        but only at GC time, which is why the finally needs its own test."""
        col = _scollection(SpotifyType.ALBUM, total=0)
        closed: list[bool] = []

        async def gen() -> AsyncGenerator[TrackPage]:
            try:
                yield _spage(col, [], is_last=True)  # empty album → raise
            finally:
                closed.append(True)

        self._wire(music_bot, music_player, mock_ctx, gen())

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert closed == [True], "the suspended generator was never finalized"
        assert music_player.queue.qsize() == 0
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()
        errors = [
            c.kwargs["embed"]
            for c in mock_ctx.send.call_args_list + mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert errors, "the user was told nothing"

    async def test_album_play_drains_all_pages_to_queue_and_mirror_in_order(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
        fake_redis: Redis,
    ) -> None:
        """The mirror-order guarantee the design promises: pages land on the
        Redis leg contiguously and in collection order, via the real
        queue_put → GuildQueue → push_queue_batch chain."""
        col = _scollection(SpotifyType.ALBUM, total=6)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=False),
            _spage(col, ["T4 A", "T5 A"], is_last=True),
        ]
        self._wire(music_bot, music_player, mock_ctx, _sgen(pages))

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert music_player.queue.qsize() == 6
        assert music_player.store is not None
        stored = [
            orjson.loads(raw)["ytsearch"]
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert stored == [f"ytsearch:T{i} A" for i in range(6)]
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()

    async def test_gate_hold_releases_after_page_one_before_tail_drains(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
    ) -> None:
        """The latency property the streaming design exists for: page 1 is
        consumed inside the playback-gate hold, the tail strictly after it.
        _playback_holds is the real hold count defer_playback maintains, read
        at each yield — moving _drain_collection_tail back inside the
        AsyncExitStack turns the tail's 0s into 1s."""
        col = _scollection(SpotifyType.ALBUM, total=6)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=False),
            _spage(col, ["T4 A", "T5 A"], is_last=True),
        ]
        holds_at_yield: list[int] = []

        async def gen() -> AsyncGenerator[TrackPage]:
            for page in pages:
                holds_at_yield.append(music_player._playback_holds)
                yield page

        self._wire(music_bot, music_player, mock_ctx, gen())
        mock_ctx.voice_client = None  # front path: the gate hold is live
        mock_ctx.invoke = self._join_lands(mock_ctx)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert music_player.queue.qsize() == 6
        assert holds_at_yield == [1, 0, 0]

    async def test_front_backlog_buffers_collection_ahead_on_both_legs(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
        """front=True with a persisted backlog takes the buffered path: the
        whole collection lands ahead of the backlog on BOTH legs. Dropping
        front= from the _begin_collection_enqueue dispatch appends behind 'Old'
        instead."""
        await music_player.queue_put(
            QueueObject("https://yt.com/v=old", "Old", mock_author), prefetch=False
        )
        col = _scollection(SpotifyType.ALBUM, total=4)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=True),
        ]
        self._wire(music_bot, music_player, mock_ctx, _sgen(pages))
        mock_ctx.voice_client = None
        mock_ctx.invoke = self._join_lands(mock_ctx)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert music_player.queue.qsize() == 5
        assert music_player.store is not None
        stored = [
            orjson.loads(raw)
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert [d.get("ytsearch") or d.get("title") for d in stored] == [
            "ytsearch:T0 A",
            "ytsearch:T1 A",
            "ytsearch:T2 A",
            "ytsearch:T3 A",
            "Old",
        ]

    async def test_failed_enqueue_closes_pager_before_play_returns(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
    ) -> None:
        """When _begin_collection_enqueue raises (empty collection), the finally
        in _play_resolved must aclose() the stream before play returns — the
        generator's own finally is the proof. Leaving finalization to asyncgen
        GC hooks runs it too late for this assert (and, under
        filterwarnings=error, loudly)."""
        col = _scollection(SpotifyType.ALBUM, total=0)
        closed = asyncio.Event()

        async def gen() -> AsyncGenerator[TrackPage]:
            try:
                yield _spage(col, [], is_last=True)
            finally:
                closed.set()

        self._wire(music_bot, music_player, mock_ctx, gen())

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert closed.is_set()
        assert music_player.queue.qsize() == 0
        assert not music_bot._enqueue_locks[mock_ctx.guild.id].lock.locked()
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        assert any("no queueable tracks" in (n or "") for n in notices)

    async def test_clear_mid_drain_stops_the_pager_and_queue_stays_empty(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
        fake_redis: Redis,
    ) -> None:
        """A -clear landing between pages: the next put is refused against the
        real GuildQueue and the drain abandons WITHOUT refilling either leg —
        the clear-then-refill regression, checked end-to-end."""
        col = _scollection(SpotifyType.ALBUM, total=6)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=False),
            _spage(col, ["T4 A", "T5 A"], is_last=True),
        ]
        page3_requested = False

        async def gen() -> AsyncGenerator[TrackPage]:
            nonlocal page3_requested
            yield pages[0]
            await music_player.queue.clear()  # -clear lands mid-drain
            yield pages[1]
            page3_requested = True
            yield pages[2]

        self._wire(music_bot, music_player, mock_ctx, gen())

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)

        assert music_player.queue.qsize() == 0
        assert music_player.store is not None
        assert await fake_redis.lrange(music_player.store.queue_key(), 0, -1) == []
        assert not page3_requested  # the drain stopped consuming
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        assert any("Queueing stopped" in (n or "") for n in notices)

    async def test_single_play_mid_drain_lands_at_current_tail(
        self,
        music_bot: MusicBot,
        music_player: MusicPlayer,
        mock_ctx: MagicMock,
        fake_redis: Redis,
        mock_author: MagicMock,
    ) -> None:
        """M5→b's other half: a single -play issued mid-drain neither waits on
        the collection lock nor lands after the still-arriving collection — it
        appends at the CURRENT tail, between pages."""
        col = _scollection(SpotifyType.ALBUM, total=6)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=False),
            _spage(col, ["T4 A", "T5 A"], is_last=True),
        ]
        release = asyncio.Event()

        async def gen() -> AsyncGenerator[TrackPage]:
            yield pages[0]
            await release.wait()  # the drain parks here, slot held
            yield pages[1]
            yield pages[2]

        self._wire(music_bot, music_player, mock_ctx, gen())
        collection_task = asyncio.create_task(
            command_callback(MusicBot.play)(music_bot, mock_ctx, url=self._URL)
        )
        async with asyncio.timeout(5):
            while music_player.queue.qsize() < 2:  # page 1 has landed
                await asyncio.sleep(0)

        single = QueueObject("https://yt.com/v=s", "Single", mock_author)
        mock_ctx.message.content = "-play single song"
        with (
            patch("src.musicbot.YTDL.yt_source", new=AsyncMock(return_value=single)),
            patch("src.musicplayer.YTDL.prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="single song"
            )
            assert music_player.queue.qsize() == 3  # landed immediately, no wait

            release.set()
            await collection_task

        assert music_player.store is not None
        stored = [
            orjson.loads(raw)
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert [d.get("ytsearch") or d.get("title") for d in stored] == [
            "ytsearch:T0 A",
            "ytsearch:T1 A",
            "Single",
            "ytsearch:T2 A",
            "ytsearch:T3 A",
            "ytsearch:T4 A",
            "ytsearch:T5 A",
        ]


class TestCollectionRequesterAttribution:
    """2026-08-07: a 22-track album queued by one user played and archived
    entirely as someone else, because the lazy YTSource entries carried no
    requester and the resolve at dequeue fell back to whoever had typed most
    recently. Every path that mints one must stamp ctx.author."""

    async def test_page1_stamps_the_requester(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.ALBUM, total=2)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, ["A x", "B x"], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        queued = mp.queue_put.call_args[0][0]
        assert [y.requester_id for y in queued] == [mock_ctx.author.id] * 2
        await resolved.aclose()

    async def test_buffered_front_path_stamps_the_requester(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.ALBUM, total=2)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, ["A x", "B x"], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx)
        mp.queue.has_restored_backlog.return_value = True
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=True
        )

        queued = mp.queue_put_front.call_args[0][0]
        assert [y.requester_id for y in queued] == [mock_ctx.author.id] * 2
        await resolved.aclose()

    async def test_tail_pages_stamp_the_original_requester(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The exposed one: the tail drain runs after the playback gate opened,
        so other users' commands are landing while it is still enqueueing. Every
        page belongs to whoever ran the command, not to the newest typist."""
        col = _scollection(SpotifyType.PLAYLIST, total=3)
        pages = [
            _spage(col, ["T0 A"], is_last=False),
            _spage(col, ["T1 A"], is_last=False),
            _spage(col, ["T2 A"], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        stamped = [
            y.requester_id
            for call in mp.queue_put.await_args_list
            for y in call.args[0]
        ]
        assert stamped == [mock_ctx.author.id] * 3


class TestCollectionAnalyticsAndOrigin:
    """The streamed path mints its own YTSources page by page, so the ask-time
    analytics and the pasted link have to be threaded down to it — nothing
    re-mints them later, and an omission persists 0.0/0 to Redis and to
    play_history with no error, and leaves `-remove <the collection link>`
    matching nothing."""

    async def test_page1_carries_the_analytics_and_the_origin(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.ALBUM, total=2)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, ["A x", "B x"], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        queued = mp.queue_put.call_args[0][0]
        assert [y.user_input for y in queued] == [_ORIGIN] * 2
        assert all(y.analytics.queued_at == _ANALYTICS.queued_at for y in queued)
        # The head's depth fans out across the page, as yt_playlist does.
        assert [y.analytics.queue_position for y in queued] == [
            _ANALYTICS.queue_position,
            _ANALYTICS.queue_position + 1,
        ]

    async def test_buffered_front_path_carries_them_too(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.ALBUM, total=2)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, ["A x", "B x"], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx, backlog=True)
        mock_ctx.message.add_reaction = AsyncMock()

        await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=True
        )

        queued = mp.queue_put_front.call_args[0][0]
        assert [y.user_input for y in queued] == [_ORIGIN] * 2
        assert [y.analytics.queue_position for y in queued] == [
            _ANALYTICS.queue_position,
            _ANALYTICS.queue_position + 1,
        ]

    async def test_tail_positions_continue_instead_of_restarting(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Each page re-derives its base from the head plus what this collection
        has already enqueued. Passing the head's analytics straight through would
        record every page as waiting at the head's depth — track 200 of an album
        claiming it played immediately."""
        col = _scollection(SpotifyType.PLAYLIST, total=4)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        positions = [
            y.analytics.queue_position
            for call in mp.queue_put.await_args_list
            for y in call.args[0]
        ]
        base = _ANALYTICS.queue_position
        assert positions == [base, base + 1, base + 2, base + 3]
        origins = [
            y.user_input for call in mp.queue_put.await_args_list for y in call.args[0]
        ]
        assert origins == [_ORIGIN] * 4


class TestBeginCollectionEnqueue:
    async def test_streaming_puts_page1_and_returns_drain_state(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.PLAYLIST, total=250)
        pages = [
            _spage(col, [f"T{i} A" for i in range(100)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(100, 200)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(200, 250)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx, generation=7)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is not None
        assert drain.generation == 7
        assert drain.enqueued == 100
        assert drain.total == 250
        assert drain.completion_notice is True
        mp.queue_put.assert_awaited_once()
        args, kwargs = mp.queue_put.call_args
        assert [y.ytsearch for y in args[0]] == [f"ytsearch:T{i} A" for i in range(100)]
        assert kwargs["prefetch"] is False
        assert kwargs["expected_generation"] == 7
        mp.queue_put_front.assert_not_awaited()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "~250" in embed.description
        await resolved.aclose()

    async def test_slow_page1_is_bounded_and_reports_cleanly(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Page 1 was the one fetch outside every deadline: unbounded, a
        rate-limited identity call holds the enqueue lock (and, on the front
        path, the playback gate) for http_call's full retry ladder — ~150s
        against waiters that give up at 60. The bound converts it into the
        same clean per-leg budget the drains already have."""
        col = _scollection(SpotifyType.PLAYLIST, total=100)

        async def _hung() -> AsyncGenerator[TrackPage]:
            await asyncio.sleep(30)  # page-1 fetch that never returns in time
            yield _spage(col, ["T0 A"], is_last=True)

        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _hung())
        mp = _collection_mp(music_bot, mock_ctx)

        with patch("src.musicbot._COLLECTION_DRAIN_TIMEOUT_SECS", 0.05):
            with pytest.raises(ValueError, match="took too long"):
                await music_bot._begin_collection_enqueue(
                    mock_ctx,
                    resolved,
                    mp,
                    analytics=_ANALYTICS,
                    origin=_ORIGIN,
                    front=False,
                )

        # Nothing was queued — the failure precedes any put.
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        # The abandon path (play's finally) still owns generator cleanup.
        await resolved.aclose()

    async def test_album_gets_album_embed_with_art_and_exact_count(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.ALBUM, total=11)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM,
            _sgen([_spage(col, [f"T{i} A" for i in range(11)], is_last=True)]),
        )
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is not None
        assert drain.completion_notice is False  # album count was exact upfront
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Queued album — 200% Electronica" in embed.title
        assert "ESPRIT 空想, George Clanton" in embed.description
        assert "11 songs" in embed.description
        # The track list is the confirmation the user actually reads; main had
        # an equivalent assertion that was deleted with Spotify.playlist(), and
        # without it queue_message(titles) can be dropped with a green suite.
        assert "T0 A" in embed.description
        assert embed.thumbnail.url == "https://i.scdn.co/image/640"
        await resolved.aclose()

    async def test_single_page_playlist_reports_exact_enqueued_count(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """L6: total counts skipped episodes; a drained page 1 reports what was
        actually queued."""
        col = _scollection(SpotifyType.PLAYLIST, total=5)  # 2 items were skipped
        resolved = SpotifyCollectionPager(
            SpotifyType.PLAYLIST,
            _sgen([_spage(col, ["A x", "B y", "C z"], is_last=True)]),
        )
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is not None
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Queued playlist — 3 songs" in embed.title
        assert "A x" in embed.description  # see the album test's note
        await resolved.aclose()

    async def test_empty_collection_raises_before_anything_queues(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.ALBUM, total=0)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, [], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx)

        with pytest.raises(ValueError, match="album has no queueable tracks"):
            await music_bot._begin_collection_enqueue(
                mock_ctx,
                resolved,
                mp,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                front=False,
            )

        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        mock_ctx.send.assert_not_awaited()
        await resolved.aclose()

    async def test_refused_page1_returns_none_with_notice_and_no_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A clear/teardown landed between the generation snapshot and the
        put: nothing was queued, so no enqueue embed and no 👍 — but the user
        is told, not silently swallowed."""
        col = _scollection(SpotifyType.ALBUM, total=11)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, ["T A"], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx)
        mp.queue_put = AsyncMock(return_value=False)

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is None
        assert mock_ctx.send.await_count == 1
        notice = mock_ctx.send.call_args.kwargs["embed"].description
        assert "Queueing stopped" in notice
        mock_ctx.message.add_reaction.assert_not_awaited()
        await resolved.aclose()

    async def test_buffered_drain_timeout_queues_what_arrived(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The buffered drain holds the playback gate, so it must be bounded.

        Left unbounded it can run to the gate's own 300s timeout, which tears
        the player down and refuses the finished put_front — losing the whole
        collection after five minutes of silence. Timing out keeps what
        arrived.
        """
        col = _scollection(SpotifyType.PLAYLIST, total=500)

        async def _slow() -> AsyncGenerator[TrackPage]:
            yield _spage(col, [f"T{i} A" for i in range(100)], is_last=False)
            yield _spage(col, [f"T{i} A" for i in range(100, 200)], is_last=False)
            await asyncio.sleep(30)  # never arrives within the budget
            yield _spage(col, ["never A"], is_last=True)

        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _slow())
        mp = _collection_mp(music_bot, mock_ctx, backlog=True)
        mock_ctx.message.add_reaction = AsyncMock()

        with patch("src.musicbot._COLLECTION_DRAIN_TIMEOUT_SECS", 0.05):
            drain = await music_bot._begin_collection_enqueue(
                mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=True
            )

        assert drain is None  # buffered path never returns a tail
        mp.queue_put_front.assert_awaited_once()
        assert len(mp.queue_put_front.await_args.args[0]) == 200
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list + mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert any("taking too long" in (d or "") for d in notices), notices
        await resolved.aclose()

    async def test_notification_failure_keeps_the_drain(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A failed embed or 👍 must not discard the tail.

        Page 1 is already committed when the notification runs, so raising here
        would return None, the caller would aclose() the generator, and the
        collection would truncate to page 1 while play reported "Failed to
        queue song" for songs that are playing. A channel without Add Reactions
        makes that deterministic for every -play <collection>.
        """
        col = _scollection(SpotifyType.PLAYLIST, total=250)
        pages = [
            _spage(col, [f"T{i} A" for i in range(100)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(100, 250)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx, generation=2)
        mock_ctx.message.add_reaction = AsyncMock(
            side_effect=discord.Forbidden(MagicMock(status=403), "Missing Permissions")
        )

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is not None
        assert drain.generation == 2
        assert drain.enqueued == 100
        assert drain.total == 250
        mp.queue_put.assert_awaited_once()
        await resolved.aclose()

    async def test_teardown_during_page1_fetch_queues_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A teardown landing inside the page-1 round-trip must not enqueue.

        cleanup() pops the player before bumping the generation, so a snapshot
        taken after the pop reads the post-teardown value and every page then
        matches and commits onto a dead guild's persisted mirror — a "Queued
        album" success embed after the user pressed -stop. The generation
        cannot catch this one; only the player-identity check can.
        """
        col = _scollection(SpotifyType.ALBUM, total=11)
        mp = _collection_mp(music_bot, mock_ctx, generation=5)

        async def _gen() -> AsyncGenerator[TrackPage]:
            # The teardown happens while page 1 is in flight: cleanup() pops
            # the player, then bumps the generation the snapshot has not read
            # yet.
            del music_bot.mps[mock_ctx.guild.id]
            mp.queue.generation = 6
            yield _spage(col, [f"T{i} A" for i in range(11)], is_last=True)

        resolved = SpotifyCollectionPager(SpotifyType.ALBUM, _gen())

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is None
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        assert mock_ctx.send.await_count == 1
        assert "Queueing stopped" in mock_ctx.send.call_args.kwargs["embed"].description
        mock_ctx.message.add_reaction.assert_not_awaited()
        await resolved.aclose()

    async def test_front_with_backlog_buffers_into_one_put_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Ordering matrix row 3: restored entries exist, so the whole
        collection buffers into one put_front — successive streamed put_fronts
        would invert page order (put_front inserts at the head)."""
        col = _scollection(SpotifyType.PLAYLIST, total=150)
        pages = [
            _spage(col, [f"T{i} A" for i in range(100)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(100, 150)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx, generation=3, backlog=True)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=True
        )

        assert drain is None  # nothing left to drain — the buffer consumed it all
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_awaited_once()
        args, kwargs = mp.queue_put_front.call_args
        assert [y.ytsearch for y in args[0]] == [f"ytsearch:T{i} A" for i in range(150)]
        assert kwargs["prefetch"] is False
        assert kwargs["expected_generation"] == 3
        # The success side is user-visible too: embed + 👍 — inverting the
        # refusal guard would silence exactly this.
        assert mock_ctx.send.await_count == 1
        assert "150 songs" in mock_ctx.send.call_args.kwargs["embed"].title
        mock_ctx.message.add_reaction.assert_awaited_once()

    async def test_refused_buffered_put_front_notifies_and_returns_none(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The buffered arm of the M6 fix: the user waited through the whole
        fetch — a refusal here must say so, and send no enqueue embed."""
        col = _scollection(SpotifyType.ALBUM, total=4)
        pages = [
            _spage(col, ["T0 A", "T1 A"], is_last=False),
            _spage(col, ["T2 A", "T3 A"], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.ALBUM, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx, backlog=True)
        mp.queue_put_front = AsyncMock(return_value=False)

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=True
        )

        assert drain is None
        assert mock_ctx.send.await_count == 1
        notice = mock_ctx.send.call_args.kwargs["embed"].description
        assert "Queueing stopped" in notice
        mock_ctx.message.add_reaction.assert_not_awaited()
        await resolved.aclose()

    async def test_empty_non_last_page1_streams_on(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A playlist whose first 100 items are all skipped (episodes/nulls)
        yields an EMPTY non-last page 1: that streams on — only an empty LAST
        page means the collection has nothing. Tightening the
        guard to `not page1.titles` kills such playlists."""
        col = _scollection(SpotifyType.PLAYLIST, total=101)
        pages = [
            _spage(col, [], is_last=False),
            _spage(col, ["T A"], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx)

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is not None  # did not raise "no queueable tracks"
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)
        assert drain.enqueued == 1

    async def test_album_embed_title_is_truncated(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Discord rejects >256-char embed titles, failing the whole send, so
        the album name goes through truncate_embed_title."""
        col = _scollection(SpotifyType.ALBUM, total=1, name="X" * 300)
        resolved = SpotifyCollectionPager(
            SpotifyType.ALBUM, _sgen([_spage(col, ["T A"], is_last=True)])
        )
        mp = _collection_mp(music_bot, mock_ctx)

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )

        assert drain is not None
        embed = mock_ctx.send.call_args[1]["embed"]
        assert len(embed.title) <= 256

    async def test_front_without_backlog_streams(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Ordering matrix row 2: memory and mirror empty — appending to an
        empty queue IS front insertion, so the fast streamed path applies."""
        col = _scollection(SpotifyType.ALBUM, total=60)
        pages = [
            _spage(col, [f"T{i} A" for i in range(50)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(50, 60)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.ALBUM, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx, backlog=False)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=True
        )

        assert drain is not None
        mp.queue.has_restored_backlog.assert_awaited_once()
        mp.queue_put.assert_awaited_once()
        mp.queue_put_front.assert_not_awaited()
        await resolved.aclose()


class TestDrainCollectionTail:
    async def test_failed_completion_notice_does_not_raise(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A Discord error sending the SUCCESS notice must not escape to
        play's error handler, which would report a fully successful enqueue
        as 'Failed to queue song'."""
        col = _scollection(SpotifyType.PLAYLIST, total=200)
        pages = [
            _spage(col, ["T0 A"], is_last=False),
            _spage(col, ["T1 A"], is_last=True),
        ]
        mp = _collection_mp(music_bot, mock_ctx)
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None and drain.completion_notice

        mock_ctx.send = AsyncMock(side_effect=RuntimeError("discord hiccup"))
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)  # must not raise

        assert drain.enqueued == 2  # the enqueue itself completed

    async def test_tail_drains_all_pages_in_order_with_batch_flags(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        col = _scollection(SpotifyType.PLAYLIST, total=250)
        pages = [
            _spage(col, [f"T{i} A" for i in range(100)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(100, 200)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(200, 250)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx, generation=4)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        assert mp.queue_put.await_count == 3
        all_titles = [
            y.ytsearch for call in mp.queue_put.await_args_list for y in call.args[0]
        ]
        assert all_titles == [f"ytsearch:T{i} A" for i in range(250)]
        for call in mp.queue_put.await_args_list:
            assert call.kwargs["prefetch"] is False
            assert call.kwargs["expected_generation"] == 4
        assert drain.enqueued == 250
        # Multi-page playlist → completion notice with the REAL count (G5/L6).
        final = mock_ctx.send.call_args_list[-1].kwargs["embed"].description
        assert "finished queueing — 250 songs" in final

    async def test_album_tail_sends_no_completion_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The album embed carried the exact count upfront — a completion
        notice would be noise."""
        col = _scollection(SpotifyType.ALBUM, total=60)
        pages = [
            _spage(col, [f"T{i} A" for i in range(50)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(50, 60)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.ALBUM, _sgen(pages))
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None
        sends_before = mock_ctx.send.await_count
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        assert mock_ctx.send.await_count == sends_before  # no extra message

    async def test_refused_page_abandons_quietly_and_stops_consuming(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The H2 regression guard at the tail: a -clear (or any teardown)
        refuses the next compare-and-put; the drain stops, tells the user what
        landed, and never fetches further pages."""
        col = _scollection(SpotifyType.PLAYLIST, total=400)
        yielded: list[int] = []
        pages = [
            _spage(col, [f"T{i} A" for i in range(100)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(100, 200)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(200, 300)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(300, 400)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(
            SpotifyType.PLAYLIST, _sgen(pages, yielded=yielded)
        )
        mp = _collection_mp(music_bot, mock_ctx)
        # page 1 lands; first tail page lands; second tail page refused.
        mp.queue_put = AsyncMock(side_effect=[True, True, False])
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None
        await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        assert mp.queue_put.await_count == 3
        assert drain.enqueued == 200  # the refused page was never counted
        assert yielded == [0, 1, 2]  # page 4 was never even fetched
        final = mock_ctx.send.call_args_list[-1].kwargs["embed"].description
        assert "Queueing stopped" in final
        assert "200 songs" in final

    async def test_tail_timeout_keeps_what_queued_and_releases(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The tail holds the per-guild enqueue lock, so it must be bounded.

        Unbounded, a slow collection stalls -shuffle and every YouTube-playlist
        enqueue in the guild until they are declined at _ENQUEUE_WAIT_SECS. It
        returns rather than raising: the songs already queued ARE queued, so
        play's handler must not report the command as failed.
        """
        col = _scollection(SpotifyType.PLAYLIST, total=400)

        async def _slow() -> AsyncGenerator[TrackPage]:
            yield _spage(col, [f"T{i} A" for i in range(100)], is_last=False)
            await asyncio.sleep(30)
            yield _spage(col, ["never A"], is_last=True)

        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _slow())
        mp = _collection_mp(music_bot, mock_ctx)
        drain = _CollectionDrain(
            resolved=resolved,
            generation=0,
            enqueued=100,
            total=400,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            completion_notice=True,
        )

        with patch("src.musicbot._COLLECTION_DRAIN_TIMEOUT_SECS", 0.05):
            await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list + mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert any("taking too long" in (d or "") for d in notices), notices
        # The completion notice must not also fire — the drain did not complete.
        assert not any("finished queueing" in (d or "") for d in notices), notices
        await resolved.aclose()

    async def test_deadline_never_cancels_a_queue_put(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The drain deadline bounds page FETCHES only, like the buffered
        path. queue_put is a two-leg mutation whose Redis leg suspends
        inside the queue mutex — a deadline expiring there cancels between
        the in-memory legs and the mirror, and the desync self-heals only on
        the next shuffle/remove. A put slower than the whole budget must
        still commit; the timeout lands on the NEXT fetch instead.
        """
        col = _scollection(SpotifyType.PLAYLIST, total=300)

        async def _pages() -> AsyncGenerator[TrackPage]:
            yield _spage(col, ["p1 A"], is_last=False)
            yield _spage(col, [f"T{i} A" for i in range(100)], is_last=False)
            await asyncio.sleep(30)  # the fetch the expired deadline lands on
            yield _spage(col, ["tail A"], is_last=True)

        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _pages())
        # Production always consumes page 1 (_begin_collection_enqueue's
        # anext) before a drain exists; the tail starts from page 2.
        await anext(resolved.pages)
        mp = _collection_mp(music_bot, mock_ctx)
        put_completed: list[int] = []

        async def slow_put(items: Any, **kwargs: Any) -> bool:
            # Slower than the entire drain budget below — inside the old
            # timeout scope this await is where the cancellation landed.
            await asyncio.sleep(0.2)
            put_completed.append(len(items))
            return True

        mp.queue_put = AsyncMock(side_effect=slow_put)
        drain = _CollectionDrain(
            resolved=resolved,
            generation=0,
            enqueued=100,
            total=300,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            completion_notice=True,
        )

        with patch("src.musicbot._COLLECTION_DRAIN_TIMEOUT_SECS", 0.05):
            await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        # Page 2's put ran to completion despite outliving the deadline; the
        # budget then expired on the page-3 fetch, not mid-put.
        assert put_completed == [100]
        assert drain.enqueued == 200
        await resolved.aclose()

    async def test_rate_limited_tail_does_not_advise_rerunning(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The generic failure copy says "re-running will re-add the first N".
        For a 429 that is the worst possible advice: the re-run refetches every
        page from 1 and doubles the load that earned the limit."""
        col = _scollection(SpotifyType.PLAYLIST, total=400)

        async def _limited() -> AsyncGenerator[TrackPage]:
            yield _spage(col, [f"T{i} A" for i in range(100)], is_last=False)
            raise SpotifyRateLimitError(7.0)

        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _limited())
        mp = _collection_mp(music_bot, mock_ctx)
        drain = _CollectionDrain(
            resolved=resolved,
            generation=0,
            enqueued=100,
            total=400,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            completion_notice=True,
        )

        await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list + mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert any("rate-limiting" in (d or "") for d in notices), notices
        assert not any("Re-running" in (d or "") for d in notices), notices
        await resolved.aclose()

    async def test_midstream_failure_sends_honest_notice_and_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """H3 → (c): partial failure reports what landed and what a re-run
        does. No resume machinery — resume-from-offset is unsound for
        playlists (mutability shifts offsets)."""
        col = _scollection(SpotifyType.PLAYLIST, total=300)
        pages = [
            _spage(col, [f"T{i} A" for i in range(100)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(100, 200)], is_last=False),
            _spage(col, [f"T{i} A" for i in range(200, 300)], is_last=True),
        ]
        resolved = SpotifyCollectionPager(SpotifyType.PLAYLIST, _sgen(pages, fail_at=2))
        mp = _collection_mp(music_bot, mock_ctx)
        mock_ctx.message.add_reaction = AsyncMock()

        drain = await music_bot._begin_collection_enqueue(
            mock_ctx, resolved, mp, analytics=_ANALYTICS, origin=_ORIGIN, front=False
        )
        assert drain is not None
        with pytest.raises(SpotifyAuthError):
            await music_bot._drain_collection_tail(mock_ctx, mp, drain)

        assert drain.enqueued == 200
        final = mock_ctx.send.call_args_list[-1].kwargs["embed"].description
        assert "Queued 200 of ~300" in final
        assert "re-add the first 200" in final


class TestShuffleSerialized:
    async def test_shuffle_acquires_and_releases_the_slot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        entry = _GuildEnqueueLock()
        await entry.lock.acquire()
        music_bot._acquire_enqueue_slot = AsyncMock(return_value=entry)
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_shuffle = AsyncMock(return_value="Shuffled!")
        music_bot.get_mp = MagicMock(return_value=mp)
        # The M1 guard shuffles only the guild's REGISTERED player.
        music_bot.mps[mock_ctx.guild.id] = mp
        mock_ctx.message.add_reaction = AsyncMock()

        await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        music_bot._acquire_enqueue_slot.assert_awaited_once()
        mp.queue_shuffle.assert_awaited_once()
        assert not entry.lock.locked()  # released even on the happy path

    async def test_slot_released_when_shuffle_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A leaked slot is forever: every later collection/-shuffle in the
        guild eats the full wait timeout then declines."""
        entry = _GuildEnqueueLock()
        await entry.lock.acquire()
        music_bot._acquire_enqueue_slot = AsyncMock(return_value=entry)
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_shuffle = AsyncMock(side_effect=RuntimeError("boom"))
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.mps[mock_ctx.guild.id] = mp

        await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        assert not entry.lock.locked()

    async def test_shuffle_bails_when_player_torn_down_during_wait(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The wait can end BECAUSE of a teardown (cleanup() aborts the drain
        holding the slot). Shuffling the popped player would rebuild the Redis
        mirror from a dead snapshot — resurrecting a queue -stop deliberately
        left persisted."""
        entry = _GuildEnqueueLock()
        await entry.lock.acquire()
        music_bot._acquire_enqueue_slot = AsyncMock(return_value=entry)
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_shuffle = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.mps.pop(mock_ctx.guild.id, None)  # cleanup() popped it mid-wait

        await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_not_awaited()
        assert not entry.lock.locked()
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        assert any("while shuffle waited" in (n or "") for n in notices)

    async def test_declined_shuffle_does_not_run(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot._acquire_enqueue_slot = AsyncMock(return_value=None)
        mp = MagicMock()
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_shuffle = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_not_awaited()


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
    """-shuffle was the one queue-mutating command with no restore wait and no
    comment saying why.

    shuffle() REBUILDS the mirror from memory, so running it before
    restore_entries() has replayed the saved queue writes an unrestored deque over
    it — deleting the persisted entries outright. Alone it merely reported "at
    least 3 songs" against a queue it could not see; combined with an enqueue past
    the same window it destroys one."""

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
        # Registered, so the M1 teardown guard lets the shuffle through.
        music_bot.mps[mock_ctx.guild.id] = mp
        mock_ctx.message.add_reaction = AsyncMock()

        with _no_typing():
            await command_callback(MusicBot.shuffle)(music_bot, mock_ctx)

        mp.queue_shuffle.assert_awaited_once()
