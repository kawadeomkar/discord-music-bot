"""Tests for the -play pipeline (src/play_pipeline.py)."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from src import play_pipeline
from src.guild_state import Analytics
from src.config import SpotifyStatus
from src.musicbot import MusicBot, SpotifyDisabledError
from src.play_placement import Placement
from src.play_pipeline import (
    EmptyPlaylistError,
    PlaylistIndexError,
    ResolvedSpotifyPlaylist,
    ResolvedYoutubePlaylist,
    _rebase_positions,
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
from src.youtube import QueueObject
from tests.helpers import (
    admit,
    command_callback,
    connected_vc,
    in_authors_channel,
    no_typing,
    mock_mp,
    queue_object,
)


_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=0)


_ORIGIN = "https://yt.com/v=origin"


class TestQueueSource:
    async def test_spotify_playlist_returns_list(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(return_value=["Song A", "Song B"])
        result = await play_pipeline.queue_source(
            mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )
        assert isinstance(result, QueueObject)


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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs, skipped=3),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=titles),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
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

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=titles),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        mp.queue_put.assert_awaited_once()
        _, call_kwargs = mp.queue_put.call_args
        assert call_kwargs.get("prefetch") is False


class TestEnqueueSingle:
    @staticmethod
    def _playing_mp(head: Any = None) -> MagicMock:
        """A player with a song live and `head` at the queue front. The default
        head is a fresh Mock, i.e. NOT the song being queued."""
        mp = mock_mp(qsize=0)
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
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            cog=music_bot,
        )

        mp.repin_now_playing.assert_awaited_once()
        mp.build_queued_song_embed.assert_not_called()
        mock_ctx.send.assert_not_awaited()

    async def test_sends_confirmation_when_something_is_already_queued(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp()  # head is some other song

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            cog=music_bot,
        )

        mp.repin_now_playing.assert_not_awaited()
        mp.build_queued_song_embed.assert_called_once_with(qobj, note="", warning=None)
        assert (
            mock_ctx.send.await_args.kwargs["embed"]
            is mp.build_queued_song_embed.return_value
        )

    async def test_confirmation_when_repin_reports_no_live_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """repin_now_playing() answers False when the song ended mid-send — it
        already disposed of its message, so the confirmation is the honest reply."""
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)
        mp.repin_now_playing = AsyncMock(return_value=False)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            cog=music_bot,
        )

        mp.repin_now_playing.assert_awaited_once()
        assert (
            mock_ctx.send.await_args.kwargs["embed"]
            is mp.build_queued_song_embed.return_value
        )

    async def test_warning_gets_its_own_message_on_the_repin_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The re-hosted block has no description of its own to carry the warning."""
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            warning="watch out",
            cog=music_bot,
        )

        mp.repin_now_playing.assert_awaited_once()
        assert "watch out" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_warning_rides_the_confirmation_when_one_is_sent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp()

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            warning="watch out",
            cog=music_bot,
        )

        mp.build_queued_song_embed.assert_called_once_with(
            qobj, note="", warning="watch out"
        )

    async def test_enqueues_before_reading_the_head(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The reply's shape depends on the put having landed, so the put is
        awaited ahead of it rather than gathered with it. Read against a queue
        whose head only appears once queue_put has run."""
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=None)
        mp.queue.peek_next = MagicMock(return_value=None)

        async def _put(_: Any) -> None:
            mp.queue.peek_next = MagicMock(return_value=qobj)

        mp.queue_put = AsyncMock(side_effect=_put)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            cog=music_bot,
        )

        mp.repin_now_playing.assert_awaited_once()

    async def test_no_queued_embed_when_nothing_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue_put = AsyncMock()

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            cog=music_bot,
        )

        mp.build_queued_song_embed.assert_not_called()
        mp.repin_now_playing.assert_not_called()
        mock_ctx.send.assert_not_awaited()


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

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            warning=timestamp_warning(self._bad_ts_source()),
            cog=music_bot,
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
        mock_ctx.voice_client = connected_vc(mock_ctx)
        mock_ctx.voice_client.is_playing = MagicMock(return_value=False)
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            warning=timestamp_warning(self._bad_ts_source()),
            cog=music_bot,
        )

        mp.build_queued_song_embed.assert_not_called()
        sent = [c.kwargs["embed"] for c in mock_ctx.send.await_args_list]
        assert any("bogus" in (e.description or "") for e in sent)

    @pytest.mark.parametrize(
        "placement", [Placement.COLD_FRONT, Placement.NEXT], ids=lambda p: p.name
    )
    async def test_it_rides_the_flag_confirmations_too(
        self, music_bot: MusicBot, mock_ctx: MagicMock, placement: Placement
    ) -> None:
        """ "Every exit sends it either way" is the contract, and the flag legs
        build their own embeds. `-p --next <link>?t=bogus` would otherwise lose the
        only word the user gets that the timestamp was ignored."""
        mp = mock_mp()
        mp.queue.qsize = MagicMock(return_value=3)
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=placement,
            warning=timestamp_warning(self._bad_ts_source()),
            cog=music_bot,
        )

        said = " ".join(
            (c.kwargs["embed"].description or "")
            for c in mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        )
        assert "bogus" in said


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
        with patch("src.play_pipeline.YTDL.yt_source", new=spy):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )
        assert self._passed_query_source(spy) == "spotify.com"

    async def test_plaintext_search(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = parse_input("never gonna give you up")
        fake_qobj = QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.play_pipeline.YTDL.yt_source", new=spy):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )
        assert self._passed_query_source(spy) == "search"

    async def test_generic_host_link(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        url = "https://www.tiktok.com/@user/video/1234567890"
        source = parse_input(url)
        fake_qobj = QueueObject(url, "Clip", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.play_pipeline.YTDL.yt_source", new=spy):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=spy):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert [t.analytics.queue_position for t in result.tracks] == [0, 1, 2]
        # The interjection keeps the whole tail now; -play enqueues it too.
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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            patch(
                "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
            ),
            pytest.raises(PlaylistIndexError) as excinfo,
        ):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            patch("src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=[])),
            pytest.raises(EmptyPlaylistError),
        ):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )

    async def test_playlist_timestamp_applies_to_the_linked_video(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`t=` names an offset into the `v=` video, and `index=` makes that
        video the head of the queue — so the offset lands on it."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4&t=90"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 6)
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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

    async def test_an_interjection_honours_the_playlist_index(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`--now` starts at the track the link was copied at, not track 1 — and
        the rest of the collection follows it rather than being discarded."""
        url = "https://www.youtube.com/watch?v=v2&list=PLabc&index=3"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 5)
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            result = await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )
        head, follow_on = result
        assert head.title == "T2"
        # The tail is kept now: `--now` takes the whole collection, and the
        # interrupted song returns after the last of it.
        assert [cast(QueueObject, t).title for t in follow_on] == ["T3", "T4"]
        notice = mock_ctx.send.await_args.kwargs["embed"].description
        assert "#3" in notice

    async def test_interjection_index_past_the_end_reports_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """the interjection path shares the guard, and its own error path renders the same
        embed under its own title."""
        url = "https://www.youtube.com/watch?v=v9&list=PLabc&index=9"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 3)
        with (
            patch(
                "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
            ),
            pytest.raises(PlaylistIndexError) as excinfo,
        ):
            await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
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
        with patch("src.play_pipeline.YTDL.yt_source", new=spy):
            await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )
        assert self._passed_query_source(spy) == "spotify.com"

    async def test_interjection_youtube_playlist_bypasses_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        url = "https://www.youtube.com/playlist?list=PLabc"
        source = parse_input(url)
        tracks = [QueueObject("https://yt.com/v=1", "T", mock_ctx.author)]
        spy = AsyncMock(return_value=tracks)
        with patch("src.play_pipeline.YTDL.yt_playlist", new=spy):
            await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )
        assert self._passed_query_source(spy) == "youtube.com"

    async def test_interjection_indexed_playlist_rebases_only_the_track_it_keeps(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The head lands at 0, the depth an interjection actually has, and every
        track kept behind it is rebased off that — the dropped ones never enqueue,
        so an `&index=N` link must not record the survivors N-1 too deep."""
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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            kept = await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )

        head, follow_on = kept
        assert head is tracks[3]
        assert head.analytics.queue_position == 0
        # Rebased kept-relative, so the tail reads 1, 2 rather than 4, 5.
        assert follow_on == tracks[4:]
        assert [t.analytics.queue_position for t in follow_on] == [1, 2]

    async def test_interjection_analytics_is_depth_zero(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # An interjection plays immediately by definition, so -playnow reads no
        # queue depth at all — and its queued_at is still the ask time.
        url = "https://www.youtube.com/watch?v=abc"
        source = parse_input(url)
        fake_qobj = QueueObject(url, "Song", mock_ctx.author)
        spy = AsyncMock(return_value=fake_qobj)
        with patch("src.play_pipeline.YTDL.yt_source", new=spy):
            await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
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
        music_bot.spotify_status = SpotifyStatus.DISABLED
        with pytest.raises(SpotifyDisabledError) as exc:
            music_bot._require_spotify()
        assert exc.value.status is SpotifyStatus.DISABLED

    def test_require_spotify_raises_when_credentials_invalid(
        self, music_bot: MusicBot
    ) -> None:
        """Credentials were present (client built) but rejected at startup: the
        gate still refuses, and the error reports invalid rather than disabled."""
        music_bot.spotify_status = SpotifyStatus.INVALID
        with pytest.raises(SpotifyDisabledError) as exc:
            music_bot._require_spotify()
        assert exc.value.status is SpotifyStatus.INVALID

    async def test_spotify_playlist_raises_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.spotify = None
        music_bot.spotify_status = SpotifyStatus.DISABLED
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        with pytest.raises(SpotifyDisabledError):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )

    async def test_spotify_track_raises_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.spotify = None
        music_bot.spotify_status = SpotifyStatus.DISABLED
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        with pytest.raises(SpotifyDisabledError):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )

    async def test_spotify_track_raises_when_credentials_invalid(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Even with a live client object, an invalid status short-circuits the
        source before any Spotify API call is attempted."""
        music_bot.spotify_status = SpotifyStatus.INVALID
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        with pytest.raises(SpotifyDisabledError):
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )

    async def test_non_spotify_source_unaffected_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A YouTube link still resolves normally with Spotify turned off."""
        music_bot.spotify = None
        music_bot.spotify_status = SpotifyStatus.DISABLED
        source = YTSource(url="https://yt.com/watch?v=abc", process=False)
        fake_qobj = QueueObject(
            "https://yt.com/watch?v=abc", "YT Song", mock_ctx.author
        )
        with patch(
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ):
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=fake_qobj)
        ) as mock_yt:
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=fake_qobjs)
        ) as mock_playlist:
            result = await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
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
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=fake_qobjs)
        ) as mock_playlist:
            await play_pipeline.queue_source(
                mock_ctx, source, analytics=_ANALYTICS, origin=_ORIGIN, cog=music_bot
            )
        mock_playlist.assert_awaited_once_with(
            full_url,
            mock_ctx.author,
            query_source="",
            analytics=_ANALYTICS,
            user_input=_ORIGIN,
        )


class TestInterjectionCollectionHandling:
    """`--now` takes the whole collection: the head interrupts, the tail follows."""

    @staticmethod
    def _yt_tracks(author: MagicMock, count: int) -> list[QueueObject]:
        return [
            QueueObject(f"https://yt.com/watch?v=v{i}", f"T{i}", author)
            for i in range(count)
        ]

    async def test_interjection_honours_the_playlist_index(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """An interjection plays the track the link was copied at, not track 1."""
        url = "https://www.youtube.com/watch?v=v2&list=PLabc&index=3"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 5)
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            head, rest = await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )
        assert head.title == "T2"
        # The tracks after it come too, in order.
        assert [queue_object(item).title for item in rest] == ["T3", "T4"]
        notice = mock_ctx.send.await_args.kwargs["embed"].description
        assert "#3" in notice

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
        with patch(
            "src.play_pipeline.YTDL.yt_playlist", new=AsyncMock(return_value=tracks)
        ):
            head, rest = await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )

        assert head is tracks[3]
        assert head.analytics.queue_position == 0
        assert [queue_object(item).analytics.queue_position for item in rest] == [1, 2]


class TestPlaylistPositionsAreMintedAtTheInsert:
    """A playlist's queue_position is the slot each track actually takes. Minted
    at resolve it is the depth the queue had 1-99s earlier, and it rides to
    Postgres unchallenged."""

    def _wire(self, music_bot: MusicBot, mock_ctx: MagicMock, depth: int) -> MagicMock:
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
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
        play_pipeline.queue_source = AsyncMock(
            return_value=ResolvedYoutubePlaylist(tracks=tracks, skipped=0)
        )
        with no_typing("src.commands.play.background_typing"):
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
        play_pipeline.queue_source = AsyncMock(
            return_value=ResolvedYoutubePlaylist(tracks=tracks, skipped=0)
        )
        with no_typing("src.commands.play.background_typing"):
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
