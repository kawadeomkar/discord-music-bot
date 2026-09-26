"""Tests for the -play pipeline (src/play_pipeline.py)."""

import ast
import asyncio
import pathlib
import contextlib
import inspect
from collections.abc import AsyncIterator
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from src import play_pipeline
from src.guild_state import Analytics
from src.config import SpotifyStatus
from src.musicbot import MusicBot, SpotifyDisabledError
from src.musicplayer import MusicPlayer
from src.play_placement import (
    PlaceResult,
    Placement,
    PlayRequest,
    ResolveMode,
    resolve_mode_for,
)
from src.guild_queue import GuildQueue, RemoveMode, remove_matcher
from src.play_pipeline import (
    EmptyPlaylistError,
    PlaylistIndexError,
    ResolvedSpotifyPlaylist,
    ResolvedYoutubePlaylist,
    _rebase_positions,
    collection_note,
    effective_start_offset,
    past_end_refusal,
    start_offset_refusal,
)
from src.redis_client import GuildRedisStore
from src.util import ECHO_MAX, ECHO_ROW_MAX, EMBED_DESCRIPTION_LIMIT
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    parse_url,
    timestamp_warning,
)
from src.spotify import SpotifyPlaylist, SpotifyTrack
from src.youtube import YTDL, QueueObject
from tests.helpers import (
    admit,
    command_callback,
    connected_vc,
    in_authors_channel,
    no_typing,
    mock_mp,
    queue_object,
    stub_yt_playlist,
)


_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=0)


_ORIGIN = "https://yt.com/v=origin"


class TestQueueSource:
    async def test_spotify_playlist_returns_list(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(
            return_value=SpotifyPlaylist(
                name=None,
                titles=["Song A", "Song B"],
                duration_secs=0,
                duration_partial=False,
                unavailable=0,
            )
        )
        result = await play_pipeline.queue_source(
            mock_ctx,
            source,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            mode=ResolveMode.FLAT_OK,
            cog=music_bot,
        )
        assert result == ResolvedSpotifyPlaylist(titles=["Song A", "Song B"])

    async def test_a_spotify_playlist_carries_what_the_card_reports(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        row = SpotifyTrack(
            name="Song A", artists=["A"], duration_secs=122, url="https://sp/a"
        )
        music_bot.spotify.playlist = AsyncMock(
            return_value=SpotifyPlaylist(
                name="Biteki",
                titles=["Song A"],
                duration_secs=122,
                duration_partial=True,
                unavailable=2,
                tracks=[row],
            )
        )
        result = await play_pipeline.queue_source(
            mock_ctx,
            source,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            mode=ResolveMode.FLAT_OK,
            cog=music_bot,
        )
        assert result == ResolvedSpotifyPlaylist(
            titles=["Song A"],
            name="Biteki",
            unavailable=2,
            tracks=[row],
        )

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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )
        assert isinstance(result, QueueObject)


_ROWS = "`1` [**Song A**](https://x) · `3:00` · Est. playing at **9:41 PM PDT**"


def _enqueue_mp(mock_ctx: MagicMock) -> MagicMock:
    """The player enqueue_playlist places into. Spec'd, so a MusicPlayer attribute
    the enqueue starts reading raises here rather than auto-vivifying."""
    mp = MagicMock(spec=MusicPlayer)
    mp.retired = False
    mp.queue = MagicMock()
    mp.queue.generation = 0
    mp.queue_put = AsyncMock()
    mp.queue_put_front = AsyncMock()
    mp.queue_put_next = AsyncMock()
    mp.settle_prefetch = AsyncMock()
    # A str, not auto-vivified: the card joins it into its description.
    mp.playlist_facts = MagicMock(return_value="Total Duration: **3m**")
    # The shared rows, rendered by the player after the put.
    mp.queued_rows = MagicMock(return_value=_ROWS)
    # An int, not auto-vivified: the card derives "Songs ahead" from it. Mirrors
    # the real lookup's fallback, which is the depth the insert saw.
    mp.queued_slot = MagicMock(side_effect=lambda _tracks, *, ahead: ahead + 1)
    mp.queue.display_size = MagicMock(return_value=0)
    mp.enqueue_depth = MagicMock(return_value=0)
    mock_ctx.message.add_reaction = AsyncMock()
    return mp


class TestTheStartOffsetReachesTheSong:
    """`--timestamp` is applied inside queue_source, so a link's `t=` and the flag
    reach a song by one route."""

    @staticmethod
    def _yt(**kwargs: Any) -> Any:
        return YTSource(url="https://youtu.be/x", process=False, **kwargs)

    def test_the_flag_beats_the_links_own_timestamp(self) -> None:
        """A link whose `t=` did not take is the likeliest reason to reach for the
        flag, so the flag cannot lose to it."""
        assert effective_start_offset(self._yt(ts=43), 92) == 92
        assert effective_start_offset(self._yt(ts=43), None) == 43
        assert effective_start_offset(self._yt(), None) is None

    def test_a_zero_offset_is_the_flag_and_not_its_absence(self) -> None:
        """0 is falsy; reading it by truthiness would silently restore the link's
        own `t=` for `-p -ts 0 <link with ?t=43>`."""
        assert effective_start_offset(self._yt(ts=43), 0) == 0

    def test_a_spotify_source_carries_no_timestamp_of_its_own(self) -> None:
        """SpotifySource has no `ts` field at all, so the narrowing is what keeps
        this from raising."""
        source = SpotifySource(type=SpotifyType.TRACK, id="tid")
        assert effective_start_offset(source, None) is None
        assert effective_start_offset(source, 92) == 92

    async def test_a_search_is_resolved_at_the_offset(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A search is the input with no `t=` to carry one, which is the whole
        reason the flag exists."""
        source = parse_input("never gonna give you up")
        with patch.object(YTDL, "yt_source", new=AsyncMock()) as yt_source:
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                start_offset=92,
                cog=music_bot,
            )
        assert yt_source.await_args is not None
        assert yt_source.await_args.kwargs["ts"] == 92

    async def test_a_soundcloud_link_is_resolved_at_the_offset(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """parse_url never populates SoundcloudSource.ts, so the flag is the only
        start offset a SoundCloud track can get."""
        with patch.object(YTDL, "yt_source", new=AsyncMock()) as yt_source:
            await play_pipeline.queue_source(
                mock_ctx,
                SoundcloudSource(url="https://soundcloud.com/a/b"),
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                start_offset=92,
                cog=music_bot,
            )
        assert yt_source.await_args is not None
        assert yt_source.await_args.kwargs["ts"] == 92

    async def test_a_watch_link_with_a_list_starts_its_queued_head(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The shape YouTube's share button emits while a playlist is queued. The
        `t=` on this exact link already starts track 4 at its offset, so the flag
        does the same rather than disagreeing with it."""
        source = parse_url("https://youtube.com/watch?v=vid4&list=PL1&index=4")
        tracks = [
            queue_object(
                QueueObject(f"https://yt.com/v=vid{n}", f"S{n}", mock_ctx.author)
            )
            for n in range(1, 6)
        ]
        with patch.object(YTDL, "yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                start_offset=92,
                cog=music_bot,
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        # index=4 dropped the three ahead of it, so the head IS the `v=` video.
        assert [t.webpage_url for t in result.tracks] == [
            "https://yt.com/v=vid4",
            "https://yt.com/v=vid5",
        ]
        assert [t.ts for t in result.tracks] == [92, None]

    async def test_it_does_not_land_on_a_head_the_link_does_not_name(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Without a matching `index=` the queue starts at track 1, usually a
        different song — the same guard the link's own `t=` passes through."""
        source = parse_url("https://youtube.com/watch?v=vid4&list=PL1")
        tracks = [
            queue_object(
                QueueObject(f"https://yt.com/v=vid{n}", f"S{n}", mock_ctx.author)
            )
            for n in range(1, 6)
        ]
        with patch.object(YTDL, "yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                start_offset=92,
                cog=music_bot,
            )
        assert isinstance(result, ResolvedYoutubePlaylist)
        assert [t.ts for t in result.tracks] == [None] * 5


class TestStartOffsetRefusals:
    """The two refusals, at the two points they can be answered: the input shape
    before the join, and the resolved duration after it."""

    def test_a_collection_is_refused_off_the_parsed_source(self) -> None:
        """Answered before the join and before any extraction: one offset over N
        songs names none of them."""
        for link in (
            "https://youtube.com/playlist?list=PL1",
            "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
            "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl",
        ):
            assert start_offset_refusal(parse_url(link)) is not None

    def test_it_calls_an_album_an_album(self) -> None:
        refusal = start_offset_refusal(
            parse_url("https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl")
        )
        assert refusal is not None and "album" in refusal

    @pytest.mark.parametrize(
        "link",
        [
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=vid4&list=PL1&index=4",
            "https://soundcloud.com/a/b",
            "https://open.spotify.com/track/1",
        ],
    )
    def test_one_song_and_a_named_video_are_allowed(self, link: str) -> None:
        assert start_offset_refusal(parse_url(link)) is None

    def test_a_search_is_allowed(self) -> None:
        assert start_offset_refusal(parse_input("never gonna give you up")) is None

    def test_a_youtu_be_link_carrying_a_list_names_its_video(self) -> None:
        """youtu.be puts the video in the path, where `v=` never appears, so its
        `video_id` comes from there — the offset can name the queued head exactly
        as it does on a watch link."""
        source = parse_url("https://youtu.be/dQw4w9WgXcQ?list=PL1")
        assert isinstance(source, YTSource) and source.video_id == "dQw4w9WgXcQ"
        assert start_offset_refusal(source) is None

    @pytest.mark.parametrize(
        "uri,refused",
        [
            ("spotify:track:4uLU6hMCjMI75M1A2tKUQC", False),
            ("spotify:album:6WgSCcRfaXuBVfM2TpV0Kl", True),
            ("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M", True),
        ],
    )
    def test_a_spotify_uri_is_judged_like_its_link(
        self, uri: str, refused: bool
    ) -> None:
        """A `spotify:` URI reaches the same SpotifySource as the open.spotify.com
        link, so one offset rule covers both spellings."""
        assert (start_offset_refusal(parse_url(uri)) is not None) is refused

    def test_the_two_refusals_read_the_same_input_set(self) -> None:
        """They sit in two modules and run at two points in the flow; a source one
        refuses and the other resolves would drop the offset in silence."""
        for link in (
            "https://youtube.com/playlist?list=PL1",
            "https://youtube.com/watch?v=v&list=PL1",
            "https://youtu.be/dQw4w9WgXcQ?list=PL1",
            "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
            "https://youtu.be/x",
        ):
            source = parse_url(link)
            refused = start_offset_refusal(source) is not None
            assert refused is (
                play_pipeline.is_collection(source)
                and not (isinstance(source, YTSource) and bool(source.video_id))
            )

    def test_an_offset_at_or_past_the_end_is_refused(self, mock_ctx: MagicMock) -> None:
        qobj = QueueObject(
            "https://yt.com/v=1", "Test Song", mock_ctx.author, duration=210
        )
        assert past_end_refusal(qobj, 210) is not None
        assert past_end_refusal(qobj, 900) is not None
        assert past_end_refusal(qobj, 209) is None

    def test_an_unknown_or_zero_duration_rules_nothing_out(
        self, mock_ctx: MagicMock
    ) -> None:
        """Livestreams report both, and an offset that cannot be checked stands —
        ffmpeg judges it."""
        for duration in (None, 0):
            qobj = QueueObject(
                "https://yt.com/v=1", "Live", mock_ctx.author, duration=duration
            )
            assert past_end_refusal(qobj, 9000) is None

    def test_no_flag_means_no_refusal(self, mock_ctx: MagicMock) -> None:
        """A link's own `?t=` past the end keeps starting at 0:00 as it always
        has; only the explicit flag is answered."""
        qobj = QueueObject(
            "https://yt.com/v=1", "T", mock_ctx.author, duration=60, ts=900
        )
        assert past_end_refusal(qobj, None) is None

    def test_a_collection_the_offset_never_reached_is_not_measured(self) -> None:
        assert past_end_refusal(ResolvedSpotifyPlaylist(titles=["A"]), 9000) is None
        assert past_end_refusal(ResolvedYoutubePlaylist([]), 9000) is None

    def test_a_playlist_head_the_offset_landed_on_is_measured(
        self, mock_ctx: MagicMock
    ) -> None:
        """A `watch?v=…&list=…` link starts its queued head, so the ordinary
        placement has to answer it the way the interjection does — there the head
        arrives as a bare QueueObject and would be checked either way."""
        head = QueueObject(
            "https://yt.com/v=v4", "S4", mock_ctx.author, duration=210, ts=9000
        )
        playlist = ResolvedYoutubePlaylist([head])
        assert past_end_refusal(playlist, 9000) is not None
        assert past_end_refusal(ResolvedYoutubePlaylist([head]), 60) is None

    def test_it_names_both_clocks(self, mock_ctx: MagicMock) -> None:
        qobj = QueueObject(
            "https://yt.com/v=1", "Test Song", mock_ctx.author, duration=210
        )
        refusal = past_end_refusal(qobj, 7470)
        assert refusal is not None
        assert "2:04:30" in refusal and "3:30" in refusal
        assert "Test Song" in refusal

    def test_the_title_cannot_forge_a_link_in_the_refusal(
        self, mock_ctx: MagicMock
    ) -> None:
        """The title is yt-dlp's text on its way into an embed description."""
        qobj = QueueObject(
            "https://yt.com/v=1", "[a](http://evil)`x`", mock_ctx.author, duration=60
        )
        refusal = past_end_refusal(qobj, 900)
        assert refusal is not None
        assert "[a](http://evil)" not in refusal and "`x`" not in refusal


class TestEnqueuePlaylist:
    async def _enqueue(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        source: Any,
        resolved: Any,
        placement: Placement = Placement.TAIL,
    ) -> None:
        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            resolved,
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            placement=placement,
            cog=music_bot,
        )

    # ── Facts shared by both paths ────────────────────────────────────────────

    async def test_spotify_links_the_playlist_name_and_passes_its_length(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)
        resolved = ResolvedSpotifyPlaylist(
            titles=["Song A"],
            name="Biteki",
            tracks=[
                SpotifyTrack(
                    name="Song A", artists=["A"], duration_secs=3723, url="https://sp/a"
                )
            ],
        )

        await self._enqueue(music_bot, mock_ctx, mp, source, resolved)

        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert "[**Biteki**](https://open.spotify.com/playlist/pid123)" in description
        mp.playlist_facts.assert_called_once_with(
            ahead=0, runtime=(3723, True), eta=False
        )

    async def test_a_playlist_card_has_no_artist_line_and_no_cover(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await self._enqueue(
            music_bot,
            mock_ctx,
            _enqueue_mp(mock_ctx),
            SpotifySource(type=SpotifyType.PLAYLIST, id="pid123"),
            ResolvedSpotifyPlaylist(titles=["T"], name="Biteki"),
        )

        card = mock_ctx.send.call_args.kwargs["embed"]
        assert card.title == "Queued playlist — 1 song"
        assert "\nby " not in card.description
        assert card.thumbnail.url is None

    async def test_youtube_links_the_playlist_title_and_sums_its_tracks(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        tracks = [
            QueueObject(
                "https://yt.com/watch?v=1", "T1", mock_ctx.author, duration=100
            ),
            QueueObject("https://yt.com/watch?v=2", "T2", mock_ctx.author),
        ]
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot,
            mock_ctx,
            mp,
            source,
            ResolvedYoutubePlaylist(tracks=tracks, title="Road Trip"),
        )

        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert f"[**Road Trip**]({source.playlist_url})" in description
        mp.playlist_facts.assert_called_once_with(
            ahead=0, runtime=(100, True), eta=False
        )

    async def test_the_facts_sit_between_the_heading_and_the_titles(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot,
            mock_ctx,
            mp,
            source,
            ResolvedSpotifyPlaylist(titles=["Song A"], name="Biteki"),
        )

        lines = mock_ctx.send.call_args.kwargs["embed"].description.split("\n")
        assert lines[1].startswith("[**Biteki**]")
        assert lines[2] == "Total Duration: **3m**"
        assert lines[3] == ""
        assert lines[4] == _ROWS

    @pytest.mark.parametrize(
        "name,heading",
        [
            (
                "Biteki びてき ",
                "[**Biteki びてき**](https://open.spotify.com/playlist/pid123)",
            ),
            ("   ", "https://open.spotify.com/playlist/pid123"),
        ],
    )
    async def test_the_name_is_stripped_so_its_bold_renders(
        self, music_bot: MusicBot, mock_ctx: MagicMock, name: str, heading: str
    ) -> None:
        """Spotify returns "Biteki びてき " with its trailing space, and Discord
        shows `**name **` with the asterisks."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot,
            mock_ctx,
            mp,
            source,
            ResolvedSpotifyPlaylist(titles=["A"], name=name),
        )

        lines = mock_ctx.send.call_args.kwargs["embed"].description.split("\n")
        assert lines[1] == heading

    async def test_a_nameless_playlist_shows_its_bare_link(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The name request can fail; the link still says which playlist it was."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot, mock_ctx, mp, source, ResolvedSpotifyPlaylist(titles=["A"])
        )

        lines = mock_ctx.send.call_args.kwargs["embed"].description.split("\n")
        assert lines[1] == "https://open.spotify.com/playlist/pid123"
        assert lines[2] == "Total Duration: **3m**"

    @pytest.mark.parametrize(
        "placement,ahead",
        [(Placement.TAIL, 4), (Placement.NEXT, 0), (Placement.COLD_FRONT, 0)],
    )
    async def test_songs_ahead_is_the_queue_behind_a_tail_insert_only(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        placement: Placement,
        ahead: int,
    ) -> None:
        """Read under the place lock; both front placements go ahead of the queue."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)
        mp.queue.display_size = MagicMock(return_value=4)
        mp.settle_prefetch = AsyncMock()
        mp.queue_put_next = AsyncMock()
        mp.queue_put_front = AsyncMock()

        await self._enqueue(
            music_bot,
            mock_ctx,
            mp,
            source,
            ResolvedSpotifyPlaylist(titles=["A"]),
            placement=placement,
        )

        assert mp.playlist_facts.call_args.kwargs["ahead"] == ahead

    async def test_unavailable_tracks_get_a_red_notice_above_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """One message, notice first: two separate sends race each other."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot,
            mock_ctx,
            mp,
            source,
            ResolvedSpotifyPlaylist(titles=["A", "B"], unavailable=3),
        )

        mock_ctx.send.assert_awaited_once()
        notice, card = mock_ctx.send.await_args.kwargs["embeds"]
        assert notice.color == discord.Color.red()
        assert (
            notice.description == "Skipped **3** unavailable songs from this playlist."
        )
        assert card.title is not None and "Queued playlist" in card.title

    async def test_one_unavailable_video_is_singular(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest",
            type=YTType.PLAYLIST,
            list_id="PLtest",
        )
        tracks = [QueueObject("https://yt.com/watch?v=1", "T1", mock_ctx.author)]
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot,
            mock_ctx,
            mp,
            source,
            ResolvedYoutubePlaylist(tracks=tracks, unavailable=1),
        )

        notice, _ = mock_ctx.send.await_args.kwargs["embeds"]
        assert (
            notice.description == "Skipped **1** unavailable song from this playlist."
        )

    async def test_nothing_unavailable_sends_the_card_alone(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await self._enqueue(
            music_bot, mock_ctx, mp, source, ResolvedSpotifyPlaylist(titles=["A"])
        )

        mock_ctx.send.assert_awaited_once()
        assert "embeds" not in mock_ctx.send.await_args.kwargs
        assert "unavailable" not in mock_ctx.send.await_args.kwargs["embed"].description

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
        mp = _enqueue_mp(mock_ctx)

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
        assert embed.description.endswith(_ROWS)
        # The rows come from the slot the tracks took, which also feeds \"Songs ahead\".
        mp.queued_slot.assert_called_once_with(qobjs, ahead=0)
        (tracks_arg,), kwargs = mp.queued_rows.call_args
        assert list(tracks_arg) == list(qobjs) and kwargs["first"] == 1
        # The rows are bounded by the room the rest of the description left.
        assert 0 < kwargs["budget"] <= EMBED_DESCRIPTION_LIMIT

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
        mp = _enqueue_mp(mock_ctx)

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
        mp = _enqueue_mp(mock_ctx)

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
        mp = _enqueue_mp(mock_ctx)

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
        mp = _enqueue_mp(mock_ctx)

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
        mp = _enqueue_mp(mock_ctx)

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
        assert embed.description.endswith(_ROWS)

    async def test_a_spotify_playlist_stamps_who_queued_every_track(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """These resolve at dequeue, when _last_author is whoever typed most
        recently: each track has to carry the requester from here."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=["Song A", "Song B"]),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        queued = mp.queue_put.call_args[0][0]
        assert [y.requester_id for y in queued] == [mock_ctx.author.id] * 2

    async def test_a_spotify_playlist_says_how_many_it_queued(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The card shows ten titles and an ellipsis whatever the size, so the
        count is the only thing that tells 300 songs from 12 — and it is the only
        place a user can see that a playlist over 100 tracks no longer stops
        there. The YouTube branch has always said it."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=[f"Song {n}" for n in range(300)]),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        assert "300 songs" in mock_ctx.send.call_args[1]["embed"].title

    async def test_one_queued_song_is_not_pluralized(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=["Song A"]),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        assert "1 song" in mock_ctx.send.call_args[1]["embed"].title
        assert "1 songs" not in mock_ctx.send.call_args[1]["embed"].title

    async def test_the_rows_are_the_players_for_exactly_what_was_queued(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The card lists through MusicPlayer.queued_rows, after the put, so its
        rows are -queue's: the same objects, in the slots they took."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)
        order: list[str] = []
        mp.queue_put.side_effect = lambda *_a, **_k: order.append("put")
        mp.queued_rows.side_effect = lambda *_a, **_k: order.append("rows") or _ROWS
        mp.queue.display_size = MagicMock(return_value=4)

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedSpotifyPlaylist(titles=[f"Song {n}" for n in range(300)]),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        assert order == ["put", "rows"]
        (queued,) = mp.queue_put.await_args.args
        rows_call = mp.queued_rows.call_args
        assert rows_call.args[0] is queued or list(rows_call.args[0]) == list(queued)
        assert rows_call.kwargs["first"] == 5

    async def test_the_whole_description_is_bounded_not_just_the_rows(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """ROWS_BUDGET bounds the rows; the heading, the facts line and a timestamp
        warning are appended around them. Each is capped on its own, but Discord
        rejects the SUM, and the send happens after the songs are already queued."""
        mp = _enqueue_mp(mock_ctx)
        mp.queued_rows = MagicMock(
            side_effect=lambda _t, *, first, budget: "R" * budget
        )
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="p" * 22)
        resolved = ResolvedSpotifyPlaylist(
            titles=["Song A"],
            name="N" * 300,
            artists=["A" * 300],
            tracks=[
                SpotifyTrack(
                    name="Song A", artists=["A"], duration_secs=1, url="https://sp/a"
                )
            ],
        )

        await self._enqueue(music_bot, mock_ctx, mp, source, resolved)

        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert len(description) <= EMBED_DESCRIPTION_LIMIT

    async def test_the_card_escapes_no_track_titles_of_its_own(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Escaping 10,000 titles to show ten measured 53ms of event-loop time,
        right before the place lock. The rows are the shared helper's, which
        formats only what it shows, so the card itself touches the heading alone."""
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        mp = _enqueue_mp(mock_ctx)
        seen: list[str] = []

        def _spy(text: str, width: int) -> str:
            seen.append(text)
            return text

        with patch.object(play_pipeline, "safe_label", side_effect=_spy):
            await play_pipeline.enqueue_playlist(
                mock_ctx,
                source,
                ResolvedSpotifyPlaylist(
                    titles=[f"Song {n}" for n in range(500)], name="Biteki"
                ),
                mp,
                admit(music_bot, mock_ctx, mp),
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                cog=music_bot,
            )

        assert seen == ["Biteki"]

    async def test_spotify_calls_queue_put_with_prefetch_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        titles = ["Song A", "Song B"]
        mp = _enqueue_mp(mock_ctx)

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

    async def test_a_next_playlist_settles_the_prefetch_off_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Both branches of _enqueue_playlist take the place lock, and both reach
        queue_put_next's neutralize under it — so the settle is hoisted once, above
        the branch, for the same reason the single-track route hoists its own."""
        qobjs = [
            QueueObject("https://yt.com/watch?v=1", "Track 1", mock_ctx.author),
            QueueObject("https://yt.com/watch?v=2", "Track 2", mock_ctx.author),
        ]
        mp = _enqueue_mp(mock_ctx)
        order: list[str] = []
        mp.queue_put_next = AsyncMock()
        mp.settle_prefetch = AsyncMock(side_effect=lambda: order.append("settle"))
        real_place = music_bot._plays.place

        @contextlib.asynccontextmanager
        async def _spy(req: PlayRequest) -> AsyncIterator[PlaceResult]:
            order.append("place")
            async with real_place(req) as verdict:
                yield verdict

        music_bot._plays.place = _spy
        await play_pipeline.enqueue_playlist(
            mock_ctx,
            YTSource(url="https://www.youtube.com/playlist?list=PLx", list_id="PLx"),
            ResolvedYoutubePlaylist(tracks=qobjs),
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            placement=Placement.NEXT,
            cog=music_bot,
        )

        assert order == ["settle", "place"]

    async def test_a_lazy_spotify_head_is_not_warmed(self, music_bot: MusicBot) -> None:
        """--next warms the playlist head because queue_put_next killed the loop's
        one-ahead prefetch. A Spotify collection's head is still a YTSource — it
        resolves at dequeue — and prefetch_stream reads a webpage_url it has not
        got, which would raise AFTER the tracks are already queued."""
        head = YTSource(ytsearch="ytsearch:song one")
        with patch.object(YTDL, "prefetch_stream", new=AsyncMock()) as warm:
            await play_pipeline._warm_front_track([head], Placement.NEXT, cog=music_bot)
        warm.assert_not_awaited()

    @pytest.mark.parametrize("placement", list(Placement))
    async def test_a_remove_during_a_playlist_put_takes_every_track_or_none(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        fake_redis: Any,
        placement: Placement,
    ) -> None:
        """placement_split_probe.py, once per placement. A `-remove <playlist link>`
        that reaches the queue mutex while the playlist's insert waits on it runs
        after the whole insert, never between its tracks."""
        link = "https://www.youtube.com/playlist?list=PLsplit"
        queue = GuildQueue(
            mock_ctx.guild, GuildRedisStore(fake_redis, guild_id=mock_ctx.guild.id)
        )
        mp = mock_mp()
        mp.queue = queue

        async def _put(obj: Any, *, prefetch: bool = True) -> None:
            await queue.put(list(obj), batch=not prefetch)

        async def _put_front(obj: Any, *, prefetch: bool = True) -> None:
            await queue.put_front(list(obj))

        mp.queue_put = AsyncMock(side_effect=_put)
        mp.queue_put_front = AsyncMock(side_effect=_put_front)
        mp.queue_put_next = AsyncMock(side_effect=_put_front)
        tracks = [
            QueueObject(
                f"https://yt.com/v={n}", f"T{n}", mock_ctx.author, user_input=link
            )
            for n in range(4)
        ]
        source = YTSource(
            url=link, process=False, type=YTType.PLAYLIST, list_id="PLsplit"
        )

        await queue._mutex.acquire()  # the loop's commit hold
        placing = asyncio.create_task(
            play_pipeline.enqueue_playlist(
                mock_ctx,
                source,
                ResolvedYoutubePlaylist(tracks),
                mp,
                admit(music_bot, mock_ctx, mp),
                analytics=_ANALYTICS,
                origin=link,
                placement=placement,
                cog=music_bot,
            )
        )
        for _ in range(50):
            await asyncio.sleep(0)
            if queue._mutex._waiters:
                break
        assert queue._mutex._waiters, "the insert never reached the queue mutex"
        removing = asyncio.create_task(queue.remove(remove_matcher(link)))
        await asyncio.sleep(0)
        queue._mutex.release()
        await placing
        outcome = await removing

        assert len(outcome.removed) == 4
        assert queue.display_items() == []


class TestEnqueueSingle:
    async def test_a_remove_during_the_put_takes_the_head_and_tail_together(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Any
    ) -> None:
        """A `-remove <playlist link>` that reaches the queue mutex while this
        placement waits on it must find the whole playlist or none of it: one
        between a head put and a tail put took the head and let the tail land."""
        link = "https://www.youtube.com/playlist?list=PLsplit"
        queue = GuildQueue(
            mock_ctx.guild, GuildRedisStore(fake_redis, guild_id=mock_ctx.guild.id)
        )
        mp = mock_mp()
        mp.queue = queue

        async def _put(obj: Any, *, prefetch: bool = True) -> None:
            await queue.put(obj if isinstance(obj, list) else [obj], batch=not prefetch)

        mp.queue_put = AsyncMock(side_effect=_put)
        head, *tail = [
            QueueObject(
                f"https://yt.com/v={n}", f"T{n}", mock_ctx.author, user_input=link
            )
            for n in range(4)
        ]

        await queue._mutex.acquire()  # the loop's commit hold
        placing = asyncio.create_task(
            play_pipeline.enqueue_single(
                mock_ctx,
                head,
                mp,
                admit(music_bot, mock_ctx, mp),
                follow_on=tail,
                cog=music_bot,
            )
        )
        for _ in range(50):
            await asyncio.sleep(0)
            if queue._mutex._waiters:
                break
        removing = asyncio.create_task(queue.remove(remove_matcher(link)))
        await asyncio.sleep(0)
        queue._mutex.release()
        await placing
        outcome = await removing

        assert len(outcome.removed) == 4
        assert queue.display_items() == []

    async def test_a_head_and_its_tail_go_in_one_put(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        head = QueueObject("https://yt.com/v=0", "Head", mock_ctx.author)
        tail = [
            QueueObject(f"https://yt.com/v={n}", f"T{n}", mock_ctx.author)
            for n in (1, 2)
        ]

        await play_pipeline.enqueue_single(
            mock_ctx,
            head,
            mp,
            admit(music_bot, mock_ctx, mp),
            follow_on=tail,
            cog=music_bot,
        )

        (call,) = mp.queue_put.await_args_list
        assert [item.webpage_url for item in call.args[0]] == [
            "https://yt.com/v=0",
            "https://yt.com/v=1",
            "https://yt.com/v=2",
        ]
        assert call.kwargs == {"prefetch": False}

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
        mp.queue.display_size.return_value = 0
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


class TestASongBehindTheInFlightHeadIsConfirmed:
    async def test_a_claimed_song_still_resolving_counts_as_something_ahead(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """While the loop holds the only queued song and resolves it, nothing is
        pending and nothing is playing, and this -play queues behind it anyway."""
        mock_ctx.voice_client = connected_vc(mock_ctx)
        mp = mock_mp()
        mp.queue.qsize = MagicMock(return_value=0)
        mp.queue.display_size = MagicMock(return_value=1)
        mp.queue.peek_next = MagicMock(return_value=None)
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        await play_pipeline.enqueue_single(
            mock_ctx, qobj, mp, admit(music_bot, mock_ctx, mp), cog=music_bot
        )

        mp.build_queued_song_embed.assert_called_once()


class TestTimestampWarningReachesTheUser:
    @staticmethod
    def _bad_ts_source() -> Any:
        return parse_url("https://youtu.be/a?t=bogus")

    async def test_it_rides_the_queued_song_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp(qsize=3)  # something already queued
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
        mp = mock_mp(qsize=3)
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        spy = stub_yt_playlist(tracks)
        with patch("src.play_pipeline.YTDL.yt_playlist", new=spy):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
            patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)),
            pytest.raises(PlaylistIndexError) as excinfo,
        ):
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
            patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist([])),
            pytest.raises(EmptyPlaylistError),
        ):
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )

    async def test_playlist_timestamp_applies_to_the_linked_video(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`t=` names an offset into the `v=` video, and `index=` makes that
        video the head of the queue — so the offset lands on it."""
        url = "https://www.youtube.com/watch?v=v3&list=PLabc&index=4&t=90"
        source = parse_input(url)
        tracks = self._yt_tracks(mock_ctx.author, 6)
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
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
            patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)),
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
        music_bot.spotify.playlist = AsyncMock(
            return_value=SpotifyPlaylist(
                name=None,
                titles=["Song A", "Song B"],
                duration_secs=0,
                duration_partial=False,
                unavailable=0,
            )
        )
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
        spy = stub_yt_playlist(tracks)
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )

    async def test_spotify_track_raises_when_disabled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.spotify = None
        music_bot.spotify_status = SpotifyStatus.DISABLED
        source = SpotifySource(type=SpotifyType.TRACK, id="tid123")
        with pytest.raises(SpotifyDisabledError):
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
            "src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(fake_qobjs)
        ) as mock_playlist:
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )
        mock_playlist.assert_awaited_once_with(
            "https://www.youtube.com/playlist?list=PLtest123",
            mock_ctx.author,
            # "" because this YTSource is built directly rather than by
            # parse_input, which is what classifies. See TestQuerySourceClassification.
            query_source="",
            analytics=_ANALYTICS,
            user_input=_ORIGIN,
            redis=music_bot.redis,
            on_progress=None,
            pool_slot=None,
        )
        assert result == ResolvedYoutubePlaylist(tracks=fake_qobjs)

    async def test_a_youtube_playlist_carries_its_title_and_unavailable_count(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = YTSource(
            url="https://www.youtube.com/playlist?list=PLtest123",
            type=YTType.PLAYLIST,
            list_id="PLtest123",
        )
        tracks = [QueueObject("https://yt.com/watch?v=1", "T1", mock_ctx.author)]
        with patch(
            "src.play_pipeline.YTDL.yt_playlist",
            new=stub_yt_playlist(tracks, title="Road Trip", unavailable=3),
        ):
            result = await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )
        assert result == ResolvedYoutubePlaylist(
            tracks=tracks, title="Road Trip", unavailable=3
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
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
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
            "src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(fake_qobjs)
        ) as mock_playlist:
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )
        mock_playlist.assert_awaited_once_with(
            full_url,
            mock_ctx.author,
            query_source="",
            analytics=_ANALYTICS,
            user_input=_ORIGIN,
            redis=music_bot.redis,
            on_progress=None,
            pool_slot=None,
        )


class TestInterjectionCollectionHandling:
    """`--now` takes the whole collection: the head interrupts, the tail follows."""

    async def test_a_spotify_tail_stamps_who_queued_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.PLAYLIST, id="pid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(
            return_value=SpotifyPlaylist(
                name=None,
                titles=["Head", "Two", "Three"],
                duration_secs=0,
                duration_partial=False,
                unavailable=0,
            )
        )
        head = QueueObject("https://yt.com/v=h", "Head", mock_ctx.author)
        with patch(
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=head)
        ):
            _, rest = await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )

        assert [cast(YTSource, y).requester_id for y in rest] == [
            mock_ctx.author.id
        ] * 2

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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
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
        with patch("src.play_pipeline.YTDL.yt_playlist", new=stub_yt_playlist(tracks)):
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


class TestResolveModeThreading:
    """Which inputs may answer from search metadata alone, and who decides. The
    decision cannot be read off the input: interjection resolves through the same
    helper, and its head must be playable before the current song is stopped."""

    @staticmethod
    def _passed_flat(spy: AsyncMock) -> bool:
        assert spy.await_args is not None
        return spy.await_args.kwargs["flat"]

    async def _resolve(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        source: Any,
        mode: ResolveMode,
    ) -> AsyncMock:
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        with patch("src.play_pipeline.YTDL.yt_source", new=spy):
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=mode,
                cog=music_bot,
            )
        return spy

    async def test_a_search_goes_flat(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        spy = await self._resolve(
            music_bot, mock_ctx, parse_input("take on me"), ResolveMode.FLAT_OK
        )
        assert self._passed_flat(spy) is True

    async def test_a_spotify_track_goes_flat(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """It resolves to a YouTube title search, so it has the same cheap mode."""
        assert music_bot.spotify is not None
        music_bot.spotify.track = AsyncMock(return_value="My Track Artist")
        spy = await self._resolve(
            music_bot,
            mock_ctx,
            SpotifySource(type=SpotifyType.TRACK, id="tid123"),
            ResolveMode.FLAT_OK,
        )
        assert self._passed_flat(spy) is True

    async def test_a_youtube_link_never_goes_flat(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A link has no cheap mode: its cost is the watch page, and skipping that
        fails YouTube's bot check."""
        spy = await self._resolve(
            music_bot,
            mock_ctx,
            YTSource(url="https://yt.com/watch?v=abc", process=False),
            ResolveMode.FLAT_OK,
        )
        assert self._passed_flat(spy) is False

    async def test_a_soundcloud_link_never_goes_flat(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        spy = await self._resolve(
            music_bot,
            mock_ctx,
            SoundcloudSource(url="https://soundcloud.com/a/b"),
            ResolveMode.FLAT_OK,
        )
        assert self._passed_flat(spy) is False

    async def test_full_mode_refuses_flat_even_for_a_search(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        spy = await self._resolve(
            music_bot, mock_ctx, parse_input("take on me"), ResolveMode.FULL
        )
        assert self._passed_flat(spy) is False

    def test_only_a_cold_start_must_resolve_full(self) -> None:
        """The policy has one name so it has one test, and the test enumerates
        Placement so a member added later cannot inherit FLAT_OK in silence. A cold
        start plays its song immediately: resolving it flat moves the failure past
        the join and parks the bot in a channel with an empty queue. Interjection is
        not a Placement — it never reaches this function."""
        assert resolve_mode_for(Placement.COLD_FRONT) is ResolveMode.FULL
        assert [p for p in Placement if resolve_mode_for(p) is ResolveMode.FLAT_OK] == [
            Placement.TAIL,
            Placement.NEXT,
        ]

    async def test_an_interjection_head_resolves_full(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """interject() stops the current song; a head that turns out to be
        unplayable would have stopped it for nothing."""
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        with patch.object(play_pipeline, "queue_source", new=spy):
            await play_pipeline._resolve_interjection_source(
                mock_ctx, parse_input("take on me"), origin=_ORIGIN, cog=music_bot
            )
        assert spy.await_args is not None
        assert spy.await_args.kwargs["mode"] is ResolveMode.FULL


class TestTheResumeNoticeDescribesARestoredQueue:
    """It calls what sits behind the head 'the previous session'."""

    async def test_a_sibling_that_landed_suppresses_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Two cold starts in one burst: the second would file the song the same
        user pasted a second earlier under the previous session."""
        mp = mock_mp()
        mp.queue.qsize = MagicMock(return_value=1)
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=2", "Second", mock_ctx.author)

        first = admit(music_bot, mock_ctx, mp)
        first.placed = True
        second = admit(music_bot, mock_ctx, mp)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            second,
            placement=Placement.COLD_FRONT,
            cog=music_bot,
        )

        mp.build_resume_notice_embed.assert_not_called()
        mp.build_queued_song_embed.assert_called()

    async def test_a_lone_cold_start_still_gets_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=1", "First", mock_ctx.author)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.COLD_FRONT,
            cog=music_bot,
        )

        mp.build_resume_notice_embed.assert_called_once()


class TestSpanDecoratorsNameTheirOwnFunction:
    """`bot.queue_source` once decorated `plays_after_note`, a microsecond string
    helper inserted directly beneath it, while the 29s resolve it was written for
    emitted no span at all. A decorator naming one function and sitting on another
    is invisible to every behavioural test, so it is asserted from the source."""

    # Every module that decorates a function with a span name. The bug is not
    # specific to play_pipeline — it is what a same-signature insert above a
    # decorated function does anywhere — so the guard covers all of them.
    _MODULES = (
        "src/play_pipeline.py",
        "src/play_placement.py",
        "src/musicplayer.py",
        "src/queue_progress.py",
        "src/youtube.py",
        "src/spotify.py",
    )

    @staticmethod
    def _decorations(source: Optional[str] = None) -> list[tuple[str, str]]:
        """(span name, decorated function) for every tracer decorator in a module."""
        tree = ast.parse(
            source if source is not None else inspect.getsource(play_pipeline)
        )
        found: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                called = decorator.func
                if (
                    not isinstance(called, ast.Attribute)
                    or called.attr != "start_as_current_span"
                ):
                    continue
                (name,) = decorator.args
                assert isinstance(name, ast.Constant)
                span_name = name.value
                assert isinstance(span_name, str)
                found.append((span_name, node.name))
        return found

    def test_the_resolve_is_the_one_bot_queue_source_wraps(self) -> None:
        assert ("bot.queue_source", "queue_source") in self._decorations()

    def test_every_span_name_matches_the_function_under_it(self) -> None:
        decorations = self._decorations()
        assert len(decorations) >= 5
        for span_name, func_name in decorations:
            assert span_name == f"bot.{func_name.lstrip('_')}"

    def test_no_module_decorates_a_function_it_does_not_name(self) -> None:
        """Same rule, every module. The prefix differs per module — `bot.`,
        `player.`, `spotify.` — and a tail may abbreviate (`player.prefetch` over
        `_prefetch_next_song`), so what is asserted is that one NAMES the other:
        either is a substring of the other. `bot.queue_source` sitting on
        `plays_after_note`, the bug this encodes, shares nothing either way."""
        checked = 0
        for path in self._MODULES:
            for span_name, func_name in self._decorations(
                pathlib.Path(path).read_text()
            ):
                checked += 1
                tail = span_name.split(".")[-1]
                func = func_name.lstrip("_")
                assert tail in func or func in tail, (path, span_name, func_name)
        assert checked >= 10


class TestCollectionNote:
    """The undo a --now playlist offers has to work when copied: -remove compares
    links literally, and every Mix link carries an underscore in `start_radio`."""

    @staticmethod
    def _copied(note: str) -> str:
        return note.split("`-remove ", 1)[1].split("`", 1)[0]

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1",
            "https://open.spotify.com/playlist/37i9dQZF1DX*0XUsuxWHRQ",
            "https://www.youtube.com/playlist?list=PL_a_b~c|d",
        ],
    )
    def test_the_copied_command_removes_the_playlist(
        self, mock_ctx: MagicMock, url: str
    ) -> None:
        note = collection_note(url, 483, head_playing=False, noun="playlist")
        track = QueueObject(
            "https://www.youtube.com/watch?v=x", "X", mock_ctx.author, user_input=url
        )
        assert remove_matcher(self._copied(note))(track) is RemoveMode.ORIGIN

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/" + "a" * ECHO_MAX,
            "https://example.com/`-skip`",
            "https://example.com/a\nb",
        ],
    )
    def test_a_link_no_code_span_can_hold_is_named_instead(self, url: str) -> None:
        note = collection_note(url, 2, head_playing=False, noun="playlist")
        assert "`-remove` followed by the link you pasted" in note
        assert url not in note


class TestSearchesForAPlaylist:
    async def test_built_a_chunk_per_loop_turn_with_positions_that_count_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk per loop turn, numbering the tracks as one call would."""
        monkeypatch.setattr(play_pipeline, "_SEARCH_BUILD_CHUNK", 2)
        ticks = 0

        async def _tick() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        ticker = asyncio.create_task(_tick())
        await asyncio.sleep(0)
        started = ticks
        try:
            tracks = await play_pipeline._searches_for(
                [f"T{n}" for n in range(5)],
                analytics=Analytics(queued_at=1.0, queue_position=7),
                origin="https://open.spotify.com/playlist/x",
                requester_id=7,
            )
        finally:
            ticker.cancel()

        assert [t.analytics.queue_position for t in tracks] == [7, 8, 9, 10, 11]
        assert [t.ytsearch for t in tracks] == [f"ytsearch:T{n}" for n in range(5)]
        assert ticks - started >= 2

    async def test_display_rows_stay_with_their_titles_across_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rows are sliced per chunk while the titles are sliced separately, so
        a chunk boundary is where the two can part company — and a row read against
        the wrong title shows one track's length and link under another's name."""
        monkeypatch.setattr(play_pipeline, "_SEARCH_BUILD_CHUNK", 2)
        rows = [
            SpotifyTrack(
                name=f"T{n}",
                artists=["A"],
                duration_secs=100 + n,
                url=f"https://sp/{n}",
            )
            for n in range(5)
        ]
        tracks = await play_pipeline._searches_for(
            [f"T{n}" for n in range(5)],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=7,
            rows=rows,
        )
        assert [t.title for t in tracks] == [f"T{n}" for n in range(5)]
        assert [t.duration for t in tracks] == [100 + n for n in range(5)]
        assert [t.webpage_url for t in tracks] == [f"https://sp/{n}" for n in range(5)]

    async def test_rows_that_do_not_pair_with_the_titles_are_dropped_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Losing every row beats showing one track's length under another's name,
        but it degrades the whole card to the pre-upgrade look, so it says so."""
        tracks = await play_pipeline._searches_for(
            [f"T{n}" for n in range(3)],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=7,
            rows=[SpotifyTrack(name="T0", artists=["A"], duration_secs=1, url=None)],
        )
        assert [t.title for t in tracks] == [None, None, None]
        assert "do not pair" in caplog.text
        assert "1 rows, 3 titles" in caplog.text


def _album_walk(**overrides: Any) -> SpotifyPlaylist:
    fields: dict[str, Any] = {
        "name": "Discovery",
        "titles": ["One More Time", "Aerodynamic"],
        "duration_secs": 531,
        "duration_partial": False,
        "unavailable": 0,
        "artists": ["Daft Punk"],
        "thumbnail": "https://i.scdn.co/cover",
    }
    fields.update(overrides)
    # A walk returns one row per title, and SpotifyPlaylist refuses any other
    # pairing, so a test that overrides the titles gets rows matching them.
    fields.setdefault(
        "tracks",
        [
            SpotifyTrack(
                name=t, artists=["Daft Punk"], duration_secs=180, url=f"https://sp/{i}"
            )
            for i, t in enumerate(fields["titles"])
        ],
    )
    return SpotifyPlaylist(**fields)


class TestSpotifyAlbum:
    """An album resolves and places exactly as a playlist does; what differs is the
    endpoint walked and what the replies call it."""

    async def test_queue_source_walks_the_album_and_carries_its_identity(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.album = AsyncMock(return_value=_album_walk())
        music_bot.spotify.playlist = AsyncMock()
        report = MagicMock()

        result = await play_pipeline.queue_source(
            mock_ctx,
            source,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            mode=ResolveMode.FLAT_OK,
            on_progress=report,
            cog=music_bot,
        )

        music_bot.spotify.album.assert_awaited_once_with("aid123", on_progress=report)
        music_bot.spotify.playlist.assert_not_awaited()
        assert result == ResolvedSpotifyPlaylist(
            titles=["One More Time", "Aerodynamic"],
            name="Discovery",
            artists=["Daft Punk"],
            thumbnail="https://i.scdn.co/cover",
            tracks=_album_walk().tracks,
        )

    async def test_an_empty_album_queues_nothing_and_says_so(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.album = AsyncMock(return_value=_album_walk(titles=[]))
        with pytest.raises(EmptyPlaylistError) as raised:
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )

        assert raised.value.user_message.startswith("That album has no songs")
        assert "playlist" not in raised.value.user_message
        assert "video" not in raised.value.user_message

    async def test_a_disabled_spotify_refuses_an_album_before_any_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        music_bot.spotify_status = SpotifyStatus.DISABLED
        with pytest.raises(SpotifyDisabledError):
            await play_pipeline.queue_source(
                mock_ctx,
                source,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                mode=ResolveMode.FLAT_OK,
                cog=music_bot,
            )

    async def _enqueue(
        self, music_bot: MusicBot, mock_ctx: MagicMock, resolved: Any
    ) -> MagicMock:
        """Returns the player, for what the enqueue asked of it."""
        mp = _enqueue_mp(mock_ctx)
        await play_pipeline.enqueue_playlist(
            mock_ctx,
            SpotifySource(type=SpotifyType.ALBUM, id="aid123"),
            resolved,
            mp,
            admit(music_bot, mock_ctx, mp),
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )
        return mp

    async def test_the_card_names_the_album_its_artists_and_its_cover(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = await self._enqueue(
            music_bot,
            mock_ctx,
            ResolvedSpotifyPlaylist(
                titles=["One More Time", "Aerodynamic"],
                name="Discovery",
                artists=["Daft Punk", "Romanthony"],
                thumbnail="https://i.scdn.co/cover",
                tracks=[
                    SpotifyTrack(
                        name="One More Time",
                        artists=["Daft Punk"],
                        duration_secs=320,
                        url="https://sp/1",
                    ),
                    SpotifyTrack(
                        name="Aerodynamic",
                        artists=["Daft Punk"],
                        duration_secs=211,
                        url="https://sp/2",
                    ),
                ],
            ),
        )

        card = mock_ctx.send.call_args.kwargs["embed"]
        assert card.title == "Queued album — 2 songs"
        lines = card.description.split("\n")
        assert lines[1] == "[**Discovery**](https://open.spotify.com/album/aid123)"
        assert lines[2] == "by Daft Punk, Romanthony"
        # The facts line is the player's; what reaches it is the album's runtime.
        assert lines[3] == mp.playlist_facts.return_value
        # Approximate: the lengths are Spotify's, not the YouTube matches'.
        mp.playlist_facts.assert_called_once_with(
            ahead=0, runtime=(531, True), eta=False
        )
        assert card.thumbnail.url == "https://i.scdn.co/cover"

    async def test_an_artist_name_cannot_style_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await self._enqueue(
            music_bot,
            mock_ctx,
            ResolvedSpotifyPlaylist(
                titles=["T"], name="N", artists=["[click](https://evil.example)"]
            ),
        )

        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert "[click](https://evil.example)" not in description

    async def test_a_walk_spotify_ended_early_says_so_above_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The card's count is what was queued; only this says the link holds more."""
        await self._enqueue(
            music_bot, mock_ctx, ResolvedSpotifyPlaylist(titles=["A"], short=True)
        )

        notice, card = mock_ctx.send.await_args.kwargs["embeds"]
        assert notice.color == discord.Color.orange()
        assert "stopped sending this album early" in notice.description
        assert card.title == "Queued album — 1 song"

    async def test_a_whole_walk_sends_the_card_alone(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await self._enqueue(
            music_bot, mock_ctx, ResolvedSpotifyPlaylist(titles=["A"], short=False)
        )

        assert "embeds" not in mock_ctx.send.await_args.kwargs

    async def test_queue_source_carries_a_short_walk(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.album = AsyncMock(return_value=_album_walk(short=True))

        result = await play_pipeline.queue_source(
            mock_ctx,
            source,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            mode=ResolveMode.FLAT_OK,
            cog=music_bot,
        )

        assert isinstance(result, ResolvedSpotifyPlaylist) and result.short

    async def test_now_over_a_short_walk_says_so_before_it_interrupts(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.album = AsyncMock(
            return_value=_album_walk(titles=["Head", "Two"], short=True)
        )
        head = QueueObject("https://yt.com/v=h", "Head", mock_ctx.author)
        with patch(
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=head)
        ):
            await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )

        notice = mock_ctx.send.await_args.kwargs["embed"].description
        assert "stopped sending this album early" in notice

    async def test_a_compilations_artist_line_is_clamped_to_a_row(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await self._enqueue(
            music_bot,
            mock_ctx,
            ResolvedSpotifyPlaylist(
                titles=["T"], name="N", artists=[f"Artist {i}" for i in range(40)]
            ),
        )

        lines = mock_ctx.send.call_args.kwargs["embed"].description.split("\n")
        assert lines[2].startswith("by Artist 0, Artist 1")
        assert len(lines[2]) <= len("by ") + ECHO_ROW_MAX

    async def test_the_unavailable_notice_calls_it_an_album(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await self._enqueue(
            music_bot, mock_ctx, ResolvedSpotifyPlaylist(titles=["A"], unavailable=1)
        )

        notice, _ = mock_ctx.send.await_args.kwargs["embeds"]
        assert notice.description == "Skipped **1** unavailable song from this album."

    async def test_every_track_carries_the_album_link_and_the_requester(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = await self._enqueue(
            music_bot, mock_ctx, ResolvedSpotifyPlaylist(titles=["A", "B", "C"])
        )

        (tracks,) = mp.queue_put.await_args.args
        assert [t.ytsearch for t in tracks] == [
            "ytsearch:A",
            "ytsearch:B",
            "ytsearch:C",
        ]
        assert {t.user_input for t in tracks} == {_ORIGIN}
        assert {t.requester_id for t in tracks} == {mock_ctx.author.id}

    async def test_the_queued_searches_carry_what_a_listing_shows(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        rows = [
            SpotifyTrack(
                name="DNA.",
                artists=["Kendrick Lamar"],
                duration_secs=185,
                url="https://open.spotify.com/track/dna",
            ),
            SpotifyTrack(name="YAH.", artists=[], duration_secs=None, url=None),
        ]
        mp = await self._enqueue(
            music_bot,
            mock_ctx,
            ResolvedSpotifyPlaylist(
                titles=["DNA. Kendrick Lamar", "YAH. Kendrick Lamar"], tracks=rows
            ),
        )

        (tracks,) = mp.queue_put.await_args.args
        assert [(t.title, t.uploader, t.duration, t.webpage_url) for t in tracks] == [
            ("DNA.", "Kendrick Lamar", 185, "https://open.spotify.com/track/dna"),
            ("YAH.", None, None, None),
        ]
        # What is searched for is still the name with its artists.
        assert tracks[0].ytsearch == "ytsearch:DNA. Kendrick Lamar"

    async def test_queue_source_carries_the_walks_rows(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        rows = [SpotifyTrack(name="One", artists=["A"], duration_secs=1, url=None)]
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.album = AsyncMock(
            return_value=_album_walk(titles=["One A"], tracks=rows)
        )

        result = await play_pipeline.queue_source(
            mock_ctx,
            source,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            mode=ResolveMode.FLAT_OK,
            cog=music_bot,
        )

        assert isinstance(result, ResolvedSpotifyPlaylist) and result.tracks == rows

    async def test_now_takes_the_whole_album_head_first(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        source = SpotifySource(type=SpotifyType.ALBUM, id="aid123")
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.album = AsyncMock(
            return_value=_album_walk(titles=["Head", "Two", "Three"])
        )
        head = QueueObject("https://yt.com/v=h", "Head", mock_ctx.author)
        with patch(
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=head)
        ) as resolve:
            got, rest = await play_pipeline._resolve_interjection_source(
                mock_ctx, source, origin=_ORIGIN, cog=music_bot
            )

        assert got is head
        assert (call := resolve.await_args) is not None
        assert call.args[1] == "ytsearch:Head"
        assert [cast(YTSource, y).ytsearch for y in rest] == [
            "ytsearch:Two",
            "ytsearch:Three",
        ]

    def test_the_undo_note_calls_it_an_album(self) -> None:
        url = "https://open.spotify.com/album/aid123"
        note = collection_note(url, 12, head_playing=False, noun="album")
        assert "from the album" in note
        assert "takes the whole album back out" in note

    @pytest.mark.parametrize(
        "source,noun",
        [
            (SpotifySource(type=SpotifyType.ALBUM, id="a"), "album"),
            (SpotifySource(type=SpotifyType.PLAYLIST, id="p"), "playlist"),
            (YTSource(type=YTType.PLAYLIST, list_id="PLx"), "playlist"),
        ],
    )
    def test_only_a_spotify_album_is_called_an_album(
        self, source: Any, noun: str
    ) -> None:
        assert play_pipeline.collection_noun(source) == noun
