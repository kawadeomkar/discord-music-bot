"""Tests for src/youtube.py — QueueObject, YTDL config, yt_source, yt_stream, and stream cache."""

import asyncio
import logging
import redis.asyncio as aioredis
import pickle
from dataclasses import FrozenInstanceError, replace
import threading
import time
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any, Optional, cast
from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import discord
import orjson
import pytest
from redis.asyncio import Redis
from yarl import URL
from yt_dlp.utils import DownloadError, UnsupportedError

from src.telemetry import configure_worker_logging
from src.guild_state import Analytics
from src.youtube import (
    YTDL,
    YTDL_OPTS,
    QueueObject,
    _DEGRADED_FORMAT_WARNED,
    _STREAM_CACHE_FIELDS,
    _UNUSED_INFO_COLLECTIONS,
    _YTDL_PLAYLIST_OPTS,
    _YTDL_STREAM_OPTS,
    _YTDL_STREAM_SEARCH_OPTS,
    _enrich_queueobject,
    _record_serving_format,
    _run_extract,
    ExtractRequest,
    _slim_info,
    _EXTRACTOR_ARGS,
    _probe_stream_url,
    _UNCONFIRMED_STREAK_LIMIT,
    _UNCONFIRMED_STREAM_TTL,
    probe_path_looks_broken,
    StreamProbe,
    _stream_url_ttl,
    _ytdlp_extract,
    _YtdlpLogger,
    YTDLVideoInfo,
    YTDLVideoMetadata,
)
from tests.helpers import noop_ffmpeg_init

# Ask-time analytics for direct yt_source/yt_playlist calls — the command paths
# mint this at dispatch; both params are REQUIRED so a call site cannot forget.
_ANALYTICS = Analytics(queued_at=1752529000.5, queue_position=0)


@pytest.fixture(autouse=True)
def _suppress_ytdl_del(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch discord.AudioSource.__del__ to a no-op for every test in this module.

    YTDL tests stub FFmpegOpusAudio.__init__ so no FFmpeg spawns, leaving _process
    unset; GC then runs __del__ → cleanup() → _kill_process(), which reads it and
    raises AttributeError. Suppressing __del__ avoids that without touching src."""
    monkeypatch.setattr(discord.AudioSource, "__del__", lambda self: None)


@pytest.fixture(autouse=True)
def playable_urls(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Treat every stream URL as playable unless a test says otherwise.

    yt_stream() probes each URL before ffmpeg; unpatched that is a real HTTP call to
    a fake googlevideo host. Revocation tests set the mock's return_value to
    StreamProbe.DEAD, and probe-blocked tests to StreamProbe.UNCONFIRMED."""
    probe = AsyncMock(return_value=StreamProbe.PLAYABLE)
    monkeypatch.setattr("src.youtube._probe_stream_url", probe)
    return probe


def _fake_ytdl_data(**overrides: Any) -> YTDLVideoInfo:
    base = {
        "url": f"https://r2.googlevideo.com/stream?expire={int(time.time()) + 7200}",
        "webpage_url": "https://www.youtube.com/watch?v=test",
        "title": "Test Song",
        "upload_date": "20240101",
        "duration": 180,
        "uploader": "Test Channel",
        "uploader_url": "",
        "thumbnail": "https://img.yt.com/test.jpg",
        "description": "",
        "tags": [],
        "view_count": 1000,
        "like_count": 100,
        "dislike_count": 5,
        "abr": 128,
        "asr": 44100,
        "acodec": "opus",
    }
    base.update(overrides)
    return cast(YTDLVideoInfo, base)


class TestYTDLGetItem:
    def test_getitem_returns_attribute(self, ytdl_instance: Callable[..., Any]) -> None:
        song = ytdl_instance()
        assert song["title"] == "Test Song"
        assert song["webpage_url"] == "https://www.youtube.com/watch?v=test"

    def test_getitem_returns_uploader(self, ytdl_instance: Callable[..., Any]) -> None:
        song = ytdl_instance()
        assert song["uploader"] == "Test Channel"


class TestYTDLDuration:
    def test_duration_uses_clock_format(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        # Same rendering as the progress bar's labels — not timedelta's
        # "0:03:00", which disagreed with the bar for the same song.
        song = ytdl_instance({"duration": 180})
        assert song.duration == "3:00"
        assert song.duration_secs == 180

    def test_duration_over_an_hour_keeps_hours(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance({"duration": 3725})
        assert song.duration == "1:02:05"

    def test_null_duration_does_not_raise(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        """yt-dlp sets "duration" to None (present, not absent) for livestreams
        and some age-gated videos. The old int(data.get("duration", "0")) got
        None past its default and raised TypeError, failing the whole
        construction."""
        song = ytdl_instance({"duration": None})
        assert song.duration_secs == 0
        assert song.duration == "0:00"

    def test_missing_duration_key_does_not_raise(
        self, ytdl_instance: Callable[..., Any], mock_channel: MagicMock
    ) -> None:
        data = _fake_ytdl_data()
        del data["duration"]
        with patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init):
            song = YTDL(mock_channel, data["url"], data=data)
        assert song.duration_secs == 0


class TestYTDLElapsedSecs:
    """Elapsed-time tracking by counting YTDL.read() calls — deterministic, no
    time-mocking. Patches the parent FFmpegOpusAudio.read() (what super().read()
    resolves to) rather than the real _packet_iter, which noop_ffmpeg_init
    never sets up."""

    def test_zero_before_any_read(self, ytdl_instance: Callable[..., Any]) -> None:
        song = ytdl_instance()
        assert song.elapsed_secs == 0.0

    def test_increments_by_20ms_per_frame_with_data(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            song.read()
        assert song.elapsed_secs == pytest.approx(0.02)

    def test_accumulates_across_multiple_reads(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            for _ in range(5):
                song.read()
        assert song.elapsed_secs == pytest.approx(0.10)

    def test_does_not_increment_on_empty_read(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b""):
            song.read()
            song.read()
        assert song.elapsed_secs == 0.0

    def test_read_returns_underlying_data(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            assert song.read() == b"opus-frame"


class TestYTDLPositionSecs:
    """position_secs = start_offset + elapsed_secs — the single source of
    truth for every position surface (progress bar, Activity presence, pause
    confirmation), so a -ss/?t= song can't report different positions in
    different places."""

    def test_equals_elapsed_when_no_offset(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            for _ in range(5):
                song.read()
        assert song.position_secs == song.elapsed_secs == pytest.approx(0.10)

    def test_includes_start_offset(self, ytdl_instance: Callable[..., Any]) -> None:
        song = ytdl_instance()
        song.start_offset = 90
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            for _ in range(5):
                song.read()
        assert song.position_secs == pytest.approx(90.10)

    def test_offset_only_before_any_read(
        self, ytdl_instance: Callable[..., Any]
    ) -> None:
        song = ytdl_instance()
        song.start_offset = 90
        assert song.position_secs == 90.0


class TestYtStreamCarriesTheQueueObjectsFields:
    """`YTDL.yt_stream` is the hop where a queue entry becomes a playing song, and
    CLAUDE.md's queue-entry-field recipe names it as one of the three sites a new
    field is silently dropped at. `user_input` reached it late: it is what
    `-remove` matches on, and a -playnow resume tail is rebuilt from the YTDL, so
    losing it here leaves the parked track un-removable by origin."""

    @staticmethod
    async def _played(qobj: QueueObject) -> YTDL:
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        with (
            patch("src.youtube._ytdlp_extract", return_value=_fake_ytdl_data()),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            return await YTDL.yt_stream(qobj, channel)

    async def test_user_input_survives_the_hop(self, mock_ctx: MagicMock) -> None:
        album = "https://open.spotify.com/album/abc123"
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            user_input=album,
        )

        assert (await self._played(qobj)).user_input == album

    def test_no_queueobject_field_is_silently_left_behind(self) -> None:
        """Reflective, so a field added to QueueObject tomorrow fails HERE rather
        than at playback — the hand-written list below it enumerates six fields and
        cannot notice a seventh. Anything genuinely not meant to cross gets named
        in the allow-list, with the reason."""
        import dataclasses
        import inspect

        # Fields that legitimately do not cross into YTDL.
        not_carried = {
            # The identity/metadata YTDL rebuilds from the yt-dlp payload itself.
            "webpage_url",
            "title",
            "duration",
            "uploader",
            "thumbnail",
            # Renamed at the boundary: ts -> start_offset (FFmpeg -ss seconds).
            "ts",
            # Runtime-only NP handle; a live Message cannot be carried on a source.
            "np_host_ref",
        }
        carried = {f.name for f in dataclasses.fields(QueueObject)} - not_carried
        params = set(inspect.signature(YTDL.__init__).parameters)
        missing = sorted(carried - params)
        assert not missing, (
            f"QueueObject fields with no YTDL.__init__ keyword: {missing}. "
            "Add them there, assign them, and pass them from yt_stream — or list "
            "them in not_carried with a reason."
        )

    async def test_every_carried_field_arrives(self, mock_ctx: MagicMock) -> None:
        """A field added to QueueObject and forgotten here dies at playback, where
        every read of it happens. Asserted together so an omission fails rather
        than needing to be noticed."""
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            user_input="typed",
            query_source="search",
            interjected=True,
            is_resume=True,
            start_paused=True,
            persisted=False,
            played_at=12.5,
        )

        song = await self._played(qobj)

        assert (
            song.user_input,
            song.query_source,
            song.interjected,
            song.is_resume,
            song.start_paused,
            song.persisted,
            song.played_at,
        ) == ("typed", "search", True, True, True, False, 12.5)

    async def test_persisted_survives_the_hop(self, mock_ctx: MagicMock) -> None:
        """`_neutralize_prefetch` reads `persisted` back off the playing song to
        rebuild a QueueObject, so a YTDL without it raises AttributeError on every
        `-playnow` over a COMPLETED prefetch — failing the command and stranding
        the claim, which the next commit then settles onto the wrong song.

        False is the value that matters: it marks the crash-recovered head, whose
        entry is NOT on the Redis list, and a rebuild defaulting to True writes
        that head into the mirror, where its dequeue never LPOPs."""
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            persisted=False,
        )

        assert (await self._played(qobj)).persisted is False


class TestQueueObject:
    def test_required_fields(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            webpage_url="https://www.youtube.com/watch?v=abc",
            title="My Song",
            requester=mock_author,
        )
        assert qobj.webpage_url == "https://www.youtube.com/watch?v=abc"
        assert qobj.title == "My Song"
        assert qobj.requester is mock_author

    def test_ts_defaults_to_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/watch?v=1", "Title", mock_author)
        assert qobj.ts is None

    def test_ts_can_be_set(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/watch?v=1", "Title", mock_author, ts=90)
        assert qobj.ts == 90

    def test_optional_fields_default_to_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/watch?v=1", "Title", mock_author)
        assert qobj.user_input is None
        assert qobj.duration is None
        assert qobj.uploader is None

    def test_optional_fields_can_be_set(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            "https://yt.com/watch?v=1",
            "Title",
            mock_author,
            user_input="search term",
            duration=180,
            uploader="My Channel",
        )
        assert qobj.user_input == "search term"
        assert qobj.duration == 180
        assert qobj.uploader == "My Channel"

    def test_is_dataclass(self, mock_author: MagicMock) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(QueueObject)

    def test_equality(self, mock_author: MagicMock) -> None:
        q1 = QueueObject("https://yt.com/watch?v=1", "Song", mock_author)
        q2 = QueueObject("https://yt.com/watch?v=1", "Song", mock_author)
        assert q1 == q2

    def test_inequality_different_url(self, mock_author: MagicMock) -> None:
        q1 = QueueObject("https://yt.com/watch?v=1", "Song", mock_author)
        q2 = QueueObject("https://yt.com/watch?v=2", "Song", mock_author)
        assert q1 != q2


class TestEnrichQueueObject:
    def test_sets_duration_when_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"duration": 180, "uploader": "Chan"})
        assert qobj.duration == 180

    def test_does_not_overwrite_existing_duration(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author, duration=120)
        _enrich_queueobject(qobj, {"duration": 999, "uploader": "Chan"})
        assert qobj.duration == 120

    def test_sets_uploader_when_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"uploader": "My Channel"})
        assert qobj.uploader == "My Channel"

    def test_does_not_overwrite_existing_uploader(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            "https://yt.com/v=1", "Song", mock_author, uploader="Original"
        )
        _enrich_queueobject(qobj, {"uploader": "New Channel"})
        assert qobj.uploader == "Original"

    def test_handles_missing_keys_gracefully(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {})
        assert qobj.duration is None
        assert qobj.uploader is None
        assert qobj.thumbnail is None

    def test_sets_thumbnail_when_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"thumbnail": "https://img.yt.com/x.jpg"})
        assert qobj.thumbnail == "https://img.yt.com/x.jpg"

    def test_does_not_overwrite_existing_thumbnail(
        self, mock_author: MagicMock
    ) -> None:
        qobj = QueueObject(
            "https://yt.com/v=1",
            "Song",
            mock_author,
            thumbnail="https://img.yt.com/original.jpg",
        )
        _enrich_queueobject(qobj, {"thumbnail": "https://img.yt.com/new.jpg"})
        assert qobj.thumbnail == "https://img.yt.com/original.jpg"

    def test_duration_cast_to_int(self, mock_author: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"duration": 180.7})
        assert qobj.duration == 180
        assert isinstance(qobj.duration, int)


class TestYTDLOpts:
    def test_format_prefers_audio_only_then_small_muxed(self) -> None:
        """bestaudio is the healthy audio-only path; the ≤360p middle rung keeps the
        muxed fallback rung from streaming 1080p video
        just for ffmpeg -vn to discard."""
        assert YTDL_OPTS["format"] == "bestaudio/best[height<=360]/best"

    def test_noplaylist_is_true(self) -> None:
        assert YTDL_OPTS["noplaylist"] is True

    def test_source_address_is_ipv4_any(self) -> None:
        assert YTDL_OPTS["source_address"] == "0.0.0.0"

    def test_default_search_is_auto(self) -> None:
        # Default_search belongs to yt_source's unified search opts, not the stream opts
        assert _YTDL_STREAM_SEARCH_OPTS["default_search"] == "auto"
        assert "default_search" not in _YTDL_STREAM_OPTS

    def test_ytdlp_warnings_are_not_suppressed(self) -> None:
        """yt-dlp's warnings are the early-warning system for YouTube changing the rules
        ("formats skipped", "SABR-only experiment"). Silencing them again would mean the
        first sign of an outage is users reporting that songs stopped playing."""
        assert YTDL_OPTS["no_warnings"] is False
        assert isinstance(YTDL_OPTS["logger"], _YtdlpLogger)


class TestYtdlpLogger:
    def test_warnings_and_errors_reach_the_log(self) -> None:
        with patch("src.youtube.log") as mock_log:
            _YtdlpLogger().warning("web client https formats have been skipped")
            _YtdlpLogger().error("boom")
        assert "skipped" in mock_log.warning.call_args.args[0]
        assert "boom" in mock_log.error.call_args.args[0]

    def test_per_video_chatter_is_dropped(self) -> None:
        """One line per song for "Downloading android vr player API JSON" is noise."""
        with patch("src.youtube.log") as mock_log:
            _YtdlpLogger().debug("[debug] Loading youtube player")
            _YtdlpLogger().info("Downloading android vr player API JSON")
        mock_log.warning.assert_not_called()
        mock_log.error.assert_not_called()

    def test_retries_is_set(self) -> None:
        assert YTDL_OPTS["retries"] > 0

    def test_socket_timeout_is_set(self) -> None:
        assert YTDL_OPTS["socket_timeout"] > 0

    def test_extractor_args_include_youtube(self) -> None:
        assert "youtube" in YTDL_OPTS["extractor_args"]

    def test_extractor_args_point_at_pot_provider(self) -> None:
        """The bgutil plugin is what lets the fallback client serve audio;
        losing this key silently reverts the fallback to token-less (video-only)."""
        pot_args = YTDL_OPTS["extractor_args"]["youtubepot-bgutilhttp"]
        assert pot_args["base_url"] == ["http://127.0.0.1:4416"]

    def test_stream_opts_have_format(self) -> None:
        assert _YTDL_STREAM_OPTS["format"] == "bestaudio/best[height<=360]/best"

    def test_unified_search_opts_carry_stream_format(self) -> None:
        """yt_source's single extraction must select a playable stream — the unified
        play path populates the ytdl:stream cache from
        the same call, which only works with the stream format ladder and its retry
        budget. Dropping the format key would silently revert to double extraction."""
        assert _YTDL_STREAM_SEARCH_OPTS["format"] == _YTDL_STREAM_OPTS["format"]
        assert _YTDL_STREAM_SEARCH_OPTS["retries"] == _YTDL_STREAM_OPTS["retries"]

    def test_no_verbose_or_rm_cachedir(self) -> None:
        for opts in (_YTDL_STREAM_SEARCH_OPTS, _YTDL_STREAM_OPTS):
            assert not opts.get("verbose")
            assert not opts.get("rm_cachedir")


class TestYTDLFfmpegOpts:
    def test_before_options_has_reconnect_flag(self) -> None:
        assert "-reconnect" in YTDL.FFMPEG_OPTS["before_options"]

    def test_options_strips_video(self) -> None:
        assert "-vn" in YTDL.FFMPEG_OPTS["options"]


class TestYTSource:
    async def test_yt_source_returns_queue_object(self, mock_ctx: MagicMock) -> None:
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Extracted Title",
        }

        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author,
                "ytsearch:test song",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        assert isinstance(result, QueueObject)
        assert result.title == "Extracted Title"
        assert result.webpage_url == "https://www.youtube.com/watch?v=test123"
        assert result.requester is mock_ctx.author

    async def test_yt_source_sets_thumbnail_fresh_extraction(
        self, mock_ctx: MagicMock
    ) -> None:
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Extracted Title",
            "thumbnail": "https://img.yt.com/test123.jpg",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author,
                "ytsearch:test song",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        assert result.thumbnail == "https://img.yt.com/test123.jpg"

    async def test_yt_source_raises_when_no_data(self, mock_ctx: MagicMock) -> None:
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = None
            with pytest.raises(Exception, match="Could not find song"):
                await YTDL.yt_source(
                    mock_ctx.author,
                    "ytsearch:nothing",
                    query_source="youtube.com",
                    analytics=_ANALYTICS,
                    user_input=None,
                )

    async def test_yt_source_unsupported_url_gives_friendly_error(
        self, mock_ctx: MagicMock
    ) -> None:
        """yt-dlp raises UnsupportedError for an unrecognised site, wrapped in a
        DownloadError. _classify_ytdlp_error flattens that to an ExtractionError with
        .unsupported set, and yt_source keys off the flag for an actionable message."""
        url = "https://example.com/not-media"
        # Mirrors yt-dlp's own wrapping: a DownloadError carrying the cause in
        # exc_info. cast() because its ExcInfo type wants a non-None traceback we
        # don't have and don't need (only exc_info[1] is read). _ytdlp_extract runs
        # in-process here, so the worker-side classification runs for real.
        cause = UnsupportedError(url)
        wrapped = DownloadError(
            "ERROR: Unsupported URL", cast(Any, (type(cause), cause, None))
        )

        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.side_effect = wrapped
            with pytest.raises(Exception, match="isn't from a site I can play"):
                await YTDL.yt_source(
                    mock_ctx.author,
                    url,
                    query_source="youtube.com",
                    analytics=_ANALYTICS,
                    user_input=None,
                )

    async def test_yt_source_reraises_non_unsupported_extraction_error(
        self, mock_ctx: MagicMock
    ) -> None:
        """A yt-dlp failure that is not an unsupported-site error (e.g. a network
        failure) is re-raised untouched as the classified ExtractionError — only a
        genuine UnsupportedError is remapped to the friendly message. _command_error
        renders the ExtractionError via its user_message."""
        from src.youtube import ExtractionError

        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.side_effect = DownloadError(
                "ERROR: unable to download webpage"
            )
            with pytest.raises(ExtractionError) as caught:
                await YTDL.yt_source(
                    mock_ctx.author,
                    "ytsearch:test",
                    query_source="youtube.com",
                    analytics=_ANALYTICS,
                    user_input=None,
                )
        assert caught.value.unsupported is False
        assert "unable to download webpage" in caught.value.message
        assert "isn't from a site I can play" not in str(caught.value)

    async def test_yt_source_picks_first_entry_from_playlist(
        self, mock_ctx: MagicMock
    ) -> None:
        fake_data = {
            "entries": [
                {
                    "webpage_url": "https://www.youtube.com/watch?v=entry1",
                    "title": "Entry One",
                    "_type": "video",
                },
                {
                    "webpage_url": "https://www.youtube.com/watch?v=entry2",
                    "title": "Entry Two",
                    "_type": "video",
                },
            ]
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author,
                "ytsearch:test",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        assert result.title == "Entry One"
        assert "entry1" in result.webpage_url

    async def test_yt_source_skips_playlist_type_entries(
        self, mock_ctx: MagicMock
    ) -> None:
        fake_data = {
            "entries": [
                {
                    "webpage_url": "https://www.youtube.com/playlist?list=abc",
                    "title": "A Playlist",
                    "_type": "playlist",
                },
                {
                    "webpage_url": "https://www.youtube.com/watch?v=real_video",
                    "title": "Real Video",
                    "_type": "video",
                },
            ]
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author,
                "ytsearch:test",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        assert result.title == "Real Video"

    async def test_yt_source_sets_user_input_fresh_extraction(
        self, mock_ctx: MagicMock
    ) -> None:
        """user_input is set to the search string on fresh extraction."""
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Song Title",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author,
                "my search query",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        assert result.user_input == "my search query"

    async def test_yt_source_sets_user_input_cache_hit(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """user_input is set to the search string even on a Redis cache hit."""
        import orjson as _orjson

        cached = {
            "webpage_url": "https://yt.com/v=cached",
            "title": "Cached Song",
            "duration": 120,
            "uploader": "Chan",
        }
        await fake_redis.set(
            "ytdl:source:cached search", _orjson.dumps(cached), ex=3600
        )
        result = await YTDL.yt_source(
            mock_ctx.author,
            "cached search",
            redis=fake_redis,
            query_source="youtube.com",
            analytics=_ANALYTICS,
            user_input=None,
        )
        assert result.user_input == "cached search"

    async def test_yt_source_sets_thumbnail_cache_hit(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """thumbnail is restored from the cached entry on a Redis cache hit."""
        import orjson as _orjson

        cached = {
            "webpage_url": "https://yt.com/v=cached",
            "title": "Cached Song",
            "duration": 120,
            "uploader": "Chan",
            "thumbnail": "https://img.yt.com/cached.jpg",
        }
        await fake_redis.set(
            "ytdl:source:cached search", _orjson.dumps(cached), ex=3600
        )
        result = await YTDL.yt_source(
            mock_ctx.author,
            "cached search",
            redis=fake_redis,
            query_source="youtube.com",
            analytics=_ANALYTICS,
            user_input=None,
        )
        assert result.thumbnail == "https://img.yt.com/cached.jpg"

    async def test_yt_source_caches_thumbnail_for_next_lookup(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """A fresh extraction's thumbnail is written to the cache, not just returned."""
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Song",
            "thumbnail": "https://img.yt.com/fresh.jpg",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            await YTDL.yt_source(
                mock_ctx.author,
                "some search",
                redis=fake_redis,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        result = await YTDL.yt_source(
            mock_ctx.author,
            "some search",
            redis=fake_redis,
            query_source="youtube.com",
            analytics=_ANALYTICS,
            user_input=None,
        )
        assert result.thumbnail == "https://img.yt.com/fresh.jpg"

    async def test_yt_source_passes_timestamp(self, mock_ctx: MagicMock) -> None:
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=ts_test",
            "title": "Timestamped Song",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author,
                "https://yt.com/watch?v=ts_test",
                ts=45,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        assert result.ts == 45

    async def test_yt_source_passes_download_flag(self, mock_ctx: MagicMock) -> None:
        fake_data = {
            "webpage_url": "https://yt.com/v=dl",
            "title": "Download Song",
        }
        with patch(
            "src.youtube._ytdlp_extract", return_value=fake_data
        ) as mock_extract:
            result = await YTDL.yt_source(
                mock_ctx.author,
                "https://yt.com/v=dl",
                download=True,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        # download rides on the request object, not a positional bool
        req = mock_extract.call_args[0][0]
        assert req.download is True
        assert result.title == "Download Song"


class TestYTPlaylistAnalytics:
    """yt_playlist builds every track complete — one ask time for the command, and
    a position per track derived from the head's."""

    @staticmethod
    def _entries(*ids: Optional[str]) -> dict[str, Any]:
        """One flat playlist reply. None is yt-dlp's deleted/private video; the
        empty string is an entry that arrives without an id."""
        return {
            "entries": [
                None
                if i is None
                else {"title": "no id"}
                if i == ""
                else {"id": i, "title": f"T{i}"}
                for i in ids
            ]
        }

    async def test_positions_count_up_from_the_head(self, mock_ctx: MagicMock) -> None:
        with patch(
            "src.youtube._ytdlp_extract", return_value=self._entries("a", "b", "c")
        ):
            tracks = await YTDL.yt_playlist(
                "https://yt.com/playlist?list=X",
                mock_ctx.author,
                query_source="youtube.com",
                analytics=Analytics(queued_at=1752529000.5, queue_position=2),
                user_input="https://yt.com/playlist?list=PL1",
            )
        assert [t.analytics.queue_position for t in tracks] == [2, 3, 4]
        assert all(t.analytics.queued_at == 1752529000.5 for t in tracks)
        assert all(t.query_source == "youtube.com" for t in tracks)

    async def test_skipped_entries_leave_no_gap_in_the_positions(
        self, mock_ctx: MagicMock
    ) -> None:
        """The offset counts tracks KEPT, not the loop index: yt-dlp emits a null
        entry for a deleted/private video, and an enumerate-based offset would
        archive positions nobody ever waited at (and skip a number entirely)."""
        with patch(
            "src.youtube._ytdlp_extract",
            # A deleted video, and one missing its ID — both skipped.
            return_value=self._entries("a", None, "b", "", "c"),
        ):
            tracks = await YTDL.yt_playlist(
                "https://yt.com/playlist?list=X",
                mock_ctx.author,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input="https://yt.com/playlist?list=PL1",
            )
        assert [t.title for t in tracks] == ["Ta", "Tb", "Tc"]
        assert [t.analytics.queue_position for t in tracks] == [0, 1, 2]


class TestYTSourceUnifiedExtraction:
    """The unified single-extraction play path: one stream-opts yt-dlp call
    populates both the ytdl:source and ytdl:stream
    caches, making queue_put's prefetch_stream a cache-hit no-op instead of a
    second YouTube extraction."""

    async def test_always_extracts_with_process_true(self, mock_ctx: MagicMock) -> None:
        """process=True is hardcoded — the process=True trap. Direct URLs used to flow with
        process=False, and an unprocessed extract_info performs no format selection,
        so data["url"] would be absent and the stream-cache write would silently
        never happen for direct-URL plays."""
        fake_data = _fake_ytdl_data()
        with patch(
            "src.youtube._ytdlp_extract", return_value=fake_data
        ) as mock_extract:
            await YTDL.yt_source(
                mock_ctx.author,
                "https://yt.com/watch?v=direct",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        req = mock_extract.call_args[0][0]
        assert req.opts is _YTDL_STREAM_SEARCH_OPTS
        assert req.process is True

    async def test_fresh_extraction_writes_both_caches(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """One cold yt_source call must leave both a ytdl:source and a ytdl:stream
        entry behind — the absence of the stream key means the second extraction
        is back."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni1")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            await YTDL.yt_source(
                mock_ctx.author,
                "unified search",
                redis=fake_redis,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        source_entry = await fake_redis.get("ytdl:source:unified search")
        stream_entry = await fake_redis.get("ytdl:stream:https://yt.com/v=uni1")
        assert source_entry is not None
        assert stream_entry is not None
        cached = orjson.loads(stream_entry)
        assert cached["url"] == fake_data["url"]
        assert cached["title"] == "Test Song"

    async def test_stream_cache_hit_for_prefetch_after_yt_source(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """prefetch_stream must not re-extract a song yt_source just resolved: the
        unified extraction makes the enqueue-time prefetch one Redis GET."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni2")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            qobj = await YTDL.yt_source(
                mock_ctx.author,
                "prefetch noop search",
                redis=fake_redis,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        with patch("src.youtube._ytdlp_extract") as mock_extract:
            await YTDL.prefetch_stream(qobj, redis=fake_redis)
        mock_extract.assert_not_called()

    async def test_dead_probe_skips_stream_cache_but_returns_qobj(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A failed probe never fails yt_source: the song enqueues on identity
        alone (source cache written), and dequeue-time re-extraction handles the
        stream — exactly the pre-unification behavior."""
        playable_urls.return_value = StreamProbe.DEAD
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni3")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            result = await YTDL.yt_source(
                mock_ctx.author,
                "dead probe search",
                redis=fake_redis,
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )

        assert isinstance(result, QueueObject)
        assert result.webpage_url == "https://yt.com/v=uni3"
        assert await fake_redis.get("ytdl:source:dead probe search") is not None
        assert await fake_redis.get("ytdl:stream:https://yt.com/v=uni3") is None

    async def test_uncacheable_url_skips_stream_cache(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A stream URL with no usable expiry (e.g. SoundCloud) is not worth caching —
        _probe_and_cache skips the playability probe entirely (it would be an awaited
        network round on the -play path only for _cache_stream to decline the write)
        and yt_source degrades gracefully, no special-casing."""
        fake_data = _fake_ytdl_data(
            url="https://cf-media.sndcdn.com/abc.128.mp3",
            webpage_url="https://soundcloud.com/artist/track",
        )
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            result = await YTDL.yt_source(
                mock_ctx.author,
                "https://soundcloud.com/artist/track",
                redis=fake_redis,
                query_source="soundcloud.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        assert isinstance(result, QueueObject)
        assert (
            await fake_redis.get("ytdl:stream:https://soundcloud.com/artist/track")
            is None
        )
        playable_urls.assert_not_awaited()

    async def test_no_probe_without_redis(
        self, mock_ctx: MagicMock, playable_urls: AsyncMock
    ) -> None:
        """Without Redis there is nothing to cache — the probe's network GET must
        be skipped entirely."""
        fake_data = _fake_ytdl_data()
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            await YTDL.yt_source(
                mock_ctx.author,
                "no redis search",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        playable_urls.assert_not_awaited()

    async def test_fresh_extraction_populates_full_metadata(
        self, mock_ctx: MagicMock
    ) -> None:
        """The unified extraction is a full one — duration/uploader/thumbnail come
        back on the first call, no prefetch enrichment needed."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni4")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            result = await YTDL.yt_source(
                mock_ctx.author,
                "metadata search",
                query_source="youtube.com",
                analytics=_ANALYTICS,
                user_input=None,
            )
        assert result.duration == 180
        assert result.uploader == "Test Channel"
        assert result.thumbnail == "https://img.yt.com/test.jpg"


class TestYTStreamRuntimeError:
    async def test_raises_when_extract_returns_none(self, mock_ctx: MagicMock) -> None:
        qobj = QueueObject("https://yt.com/v=none", "None Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        with patch("src.youtube._ytdlp_extract", return_value=None):
            with pytest.raises(RuntimeError, match="Could not extract stream data"):
                await YTDL.yt_stream(qobj, channel)


class TestYTStream:
    async def test_yt_stream_returns_ytdl_instance(self, mock_ctx: MagicMock) -> None:
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author
        )

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            result = await YTDL.yt_stream(qobj, channel)

        assert isinstance(result, YTDL)
        assert result.title == "Test Song"

    async def test_yt_stream_appends_volume_filter_when_not_default(
        self, mock_ctx: MagicMock
    ) -> None:
        """volume != 1.0 must append -filter:a to ffmpeg options."""
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author
        )

        captured_options = {}

        def capture_init(
            self: Any,
            url: str,
            *,
            executable: str,
            before_options: str,
            options: str,
        ) -> None:
            noop_ffmpeg_init(self)
            captured_options["options"] = options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel, volume=0.5)

        assert "volume=0.5" in captured_options["options"]

    async def test_yt_stream_appends_seek_when_ts_set(
        self, mock_ctx: MagicMock
    ) -> None:
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author, ts=90
        )

        captured_options = {}

        def capture_init(
            self: Any,
            url: str,
            *,
            executable: str,
            before_options: str,
            options: str,
        ) -> None:
            noop_ffmpeg_init(self)
            captured_options["options"] = options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel)

        assert "-ss 90" in captured_options["options"]

    async def test_yt_stream_carries_ts_as_start_offset(
        self, mock_ctx: MagicMock
    ) -> None:
        """QueueObject.ts must survive onto the YTDL object — loop() backdates
        play_start_epoch by it so crash recovery resumes at the true position."""
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author, ts=90
        )

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            result = await YTDL.yt_stream(qobj, channel)

        assert result.start_offset == 90

    async def test_yt_stream_start_offset_zero_without_ts(
        self, mock_ctx: MagicMock
    ) -> None:
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author
        )

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            result = await YTDL.yt_stream(qobj, channel)

        assert result.start_offset == 0

    async def test_yt_stream_carries_the_enqueue_stamps(
        self, mock_ctx: MagicMock
    ) -> None:
        """The only hop from QueueObject into the object the loop plays, and so
        the only route by which queued_at/queue_position reach
        HistoryEntry.from_song, the outbox and play_history. Zeroing it left the
        whole suite green while the feature recorded 0/0 for every play."""
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            analytics=Analytics(queued_at=1752529000.5, queue_position=4),
        )

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            result = await YTDL.yt_stream(qobj, channel)

        assert result.analytics.queued_at == 1752529000.5
        assert result.analytics.queue_position == 4


class TestStreamUrlTtl:
    def test_caps_ttl_regardless_of_expire(self) -> None:
        """A URL claiming hours of life is still only cached for the cap: YouTube
        revokes long before `expire`, so trusting it replays a revoked URL for
        hours and the song fails every time."""
        future = int(time.time()) + 7200  # 2h from now
        url = f"https://r2.googlevideo.com/stream?expire={future}&other=x"
        assert _stream_url_ttl(url) == 1800

    def test_expire_shortens_ttl_below_the_cap(self) -> None:
        """Near the end of a URL's life `expire` binds instead of the cap."""
        future = int(time.time()) + 2400  # 40m from now
        url = f"https://r2.googlevideo.com/stream?expire={future}&other=x"
        ttl = _stream_url_ttl(url)
        assert ttl is not None
        assert 2400 - 1800 - 5 <= ttl <= 2400 - 1800 + 5

    def test_reads_expire_from_hls_manifest_path_segment(self) -> None:
        """HLS manifest URLs — the muxed formats the degraded fallback rung
        serves — carry expire as a path segment, not a query param. Missing it
        would leave the entire fallback rung uncached: a full re-extract on
        every play of every degraded song."""
        future = int(time.time()) + 7200
        url = (
            "https://manifest.googlevideo.com/api/manifest/hls_playlist"
            f"/expire/{future}/ei/abcdefgh/id/xyz/playlist/index.m3u8"
        )
        assert _stream_url_ttl(url) == 1800

    def test_returns_none_when_no_expire_param(self) -> None:
        ttl = _stream_url_ttl("https://r2.googlevideo.com/stream?other=x")
        assert ttl is None

    def test_returns_none_when_already_expired(self) -> None:
        past = int(time.time()) - 100
        url = f"https://r2.googlevideo.com/stream?expire={past}"
        assert _stream_url_ttl(url) is None

    def test_returns_none_when_ttl_too_short(self) -> None:
        soon = int(time.time()) + 30  # 30s — below 60s threshold
        url = f"https://r2.googlevideo.com/stream?expire={soon}"
        assert _stream_url_ttl(url) is None

    def test_returns_none_on_non_numeric_expire(self) -> None:
        ttl = _stream_url_ttl("https://r2.googlevideo.com/stream?expire=notanumber")
        assert ttl is None


class TestRevokedStreamUrl:
    """The regression this guards: YouTube revoked a cached stream URL, the bot replayed
    it on every -play of that song, and each attempt died silently in ffmpeg."""

    async def _cache(
        self, fake_redis: aioredis.Redis, webpage_url: str, title: str = "Revoked Song"
    ) -> None:
        await fake_redis.set(
            f"ytdl:stream:{webpage_url}",
            orjson.dumps(_fake_ytdl_data(webpage_url=webpage_url, title=title)),
            ex=1800,
        )

    async def test_revoked_cached_url_is_dropped_and_re_extracted(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        webpage_url = "https://yt.com/v=revoked"
        await self._cache(fake_redis, webpage_url)
        # The cached URL is dead; the freshly extracted replacement plays.
        playable_urls.side_effect = [StreamProbe.DEAD, StreamProbe.PLAYABLE]
        fresh = _fake_ytdl_data(webpage_url=webpage_url, title="Fresh Song")
        qobj = QueueObject(webpage_url, "Revoked Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)

        with (
            patch("src.youtube._ytdlp_extract", return_value=fresh) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(qobj, channel, redis=fake_redis)

        mock_extract.assert_called_once()
        assert song.title == "Fresh Song"
        # Re-cached with the URL that actually played, not the revoked one.
        raw = await fake_redis.get(f"ytdl:stream:{webpage_url}")
        assert raw is not None
        cached = orjson.loads(raw)
        assert cached["url"] == fresh["url"]

    async def test_unconfirmed_cached_url_is_dropped_and_re_extracted(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The live incident, end to end: a cached URL whose host accepts nothing makes
        the probe hit its own deadline. Leaving the entry is what made ONE song
        unplayable for its whole TTL, since every replay re-read it and never reached
        the DEAD-only re-extract. Re-extracting escapes onto a different CDN edge."""
        webpage_url = "https://yt.com/v=blackholed"
        await self._cache(fake_redis, webpage_url)
        # Probe blocked on the cached URL; the replacement's edge answers.
        playable_urls.side_effect = [StreamProbe.UNCONFIRMED, StreamProbe.PLAYABLE]
        fresh = _fake_ytdl_data(webpage_url=webpage_url, title="Fresh Edge")
        qobj = QueueObject(webpage_url, "Stuck Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)

        with (
            patch("src.youtube._ytdlp_extract", return_value=fresh) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(qobj, channel, redis=fake_redis)

        mock_extract.assert_called_once()
        assert song.title == "Fresh Edge"
        raw = await fake_redis.get(f"ytdl:stream:{webpage_url}")
        assert raw is not None
        assert orjson.loads(raw)["url"] == fresh["url"]

    async def test_unconfirmed_fresh_url_still_plays_but_is_not_cached(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A probe that never completed says nothing about the URL, so the song still
        plays — re-extracting would only mint another URL nobody can probe. It caches,
        but only briefly: refusing outright stops the cache repopulating for as long as
        probes keep failing, which multiplies extraction load against YouTube."""
        webpage_url = "https://yt.com/v=unprobeable"
        playable_urls.return_value = StreamProbe.UNCONFIRMED
        qobj = QueueObject(webpage_url, "Unprobeable Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url, title="Plays"),
            ) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(qobj, channel, redis=fake_redis)

        assert song.title == "Plays"
        # Exactly one extraction: an unconfirmed FRESH URL has nowhere better to go.
        mock_extract.assert_called_once()
        # Cached, but capped well under the 30-minute ceiling a confirmed URL earns.
        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is not None
        ttl = await fake_redis.ttl(f"ytdl:stream:{webpage_url}")
        assert 0 < ttl <= _UNCONFIRMED_STREAM_TTL

    async def test_raises_when_youtube_refuses_even_a_fresh_url(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Both attempts refused — surface it so the player reports a failed song
        instead of handing ffmpeg a URL that will 403 into silence."""
        webpage_url = "https://yt.com/v=always_dead"
        await self._cache(fake_redis, webpage_url)
        playable_urls.return_value = StreamProbe.DEAD
        qobj = QueueObject(webpage_url, "Dead Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
            pytest.raises(RuntimeError, match="refused the audio stream"),
        ):
            await YTDL.yt_stream(qobj, channel, redis=fake_redis)

        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is None

    async def test_unplayable_fresh_url_is_never_cached(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """prefetch_stream must not cache a URL that is already dead."""
        webpage_url = "https://yt.com/v=prefetch_dead"
        qobj = QueueObject(webpage_url, "Prefetch Song", mock_ctx.author)

        with (
            patch(
                "src.youtube._probe_stream_url",
                AsyncMock(return_value=StreamProbe.DEAD),
            ),
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
        ):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)

        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is None

    async def test_unconfirmed_fresh_url_is_cached_only_briefly(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """The prefetch/yt_source half of the same rule. An unconfirmed URL was only
        un-refused, never proven, so it gets minutes rather than the full ceiling —
        long enough to keep the cache working through a blip, short enough that a wrong
        entry cannot hold a song down."""
        webpage_url = "https://yt.com/v=prefetch_unconfirmed"
        qobj = QueueObject(webpage_url, "Prefetch Song", mock_ctx.author)

        with (
            patch(
                "src.youtube._probe_stream_url",
                AsyncMock(return_value=StreamProbe.UNCONFIRMED),
            ),
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
        ):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)

        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is not None
        ttl = await fake_redis.ttl(f"ytdl:stream:{webpage_url}")
        assert 0 < ttl <= _UNCONFIRMED_STREAM_TTL

    async def test_dropped_cache_entry_stays_gone_when_the_replacement_is_uncacheable(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Pins the cache_del itself. Every other unconfirmed-cached test lets the
        replacement URL be written, which overwrites the key and hides whether the drop
        happened at all. Here the fresh URL carries no expiry, so _cache_stream declines
        the write and only an actual delete can leave the key absent."""
        webpage_url = "https://yt.com/v=drop_pinned"
        await self._cache(fake_redis, webpage_url)
        playable_urls.side_effect = [StreamProbe.UNCONFIRMED, StreamProbe.PLAYABLE]
        # No `expire` -> _stream_url_ttl returns None -> nothing is cached for it.
        fresh = _fake_ytdl_data(webpage_url=webpage_url, url="https://cdn/no-expiry")
        qobj = QueueObject(webpage_url, "Drop", mock_ctx.author)

        with (
            patch("src.youtube._ytdlp_extract", return_value=fresh),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is None

    async def test_unconfirmed_cached_url_does_not_spend_the_reextract_budget(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A probe that never completed is not evidence, so dropping the entry it
        describes must be free. Charged against the budget, the resolve would leave the
        loop having never asked yt-dlp anything."""
        webpage_url = "https://yt.com/v=budget"
        await self._cache(fake_redis, webpage_url)
        playable_urls.side_effect = [
            StreamProbe.UNCONFIRMED,  # cached entry: free drop
            StreamProbe.PLAYABLE,  # the one real extraction still gets its chance
        ]
        qobj = QueueObject(webpage_url, "Budget", mock_ctx.author)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url, title="Fresh"),
            ) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert song.title == "Fresh"
        assert mock_extract.call_count == 1  # the drop cost nothing

    async def test_gives_up_once_the_extraction_budget_is_spent(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The budget is extractions, not loop turns, and one dead FRESH url is the
        ceiling: a re-mint returns the same edge and format, so whatever refused that
        url is not something an identical call can vary."""
        webpage_url = "https://yt.com/v=budget_exhausted"
        await self._cache(fake_redis, webpage_url)
        playable_urls.side_effect = [
            StreamProbe.UNCONFIRMED,  # cached entry: free drop, no budget spent
            StreamProbe.DEAD,  # the one real extraction
            StreamProbe.DEAD,  # unused at budget 1; keeps a raised budget failing on
            # the call-count assertion rather than on an exhausted mock
        ]
        qobj = QueueObject(webpage_url, "Exhausted", mock_ctx.author)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
            pytest.raises(RuntimeError, match="refused the audio stream"),
        ):
            await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert mock_extract.call_count == 1

    async def test_the_extraction_budget_is_a_dial_not_a_hardcoded_two(
        self,
        mock_ctx: MagicMock,
        fake_redis: aioredis.Redis,
        playable_urls: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The loop reads _MAX_STREAM_EXTRACTIONS rather than counting turns, so raising
        it restores multi-extraction recovery with no other edit. Pinned because the
        shipped value is 1, which leaves the "budget remains" branch unreachable — and
        an unreachable branch rots silently, so the one change that would revive it has
        to be covered."""
        monkeypatch.setattr("src.youtube._MAX_STREAM_EXTRACTIONS", 2)
        webpage_url = "https://yt.com/v=dial"
        playable_urls.side_effect = [
            StreamProbe.DEAD,  # first real extraction
            StreamProbe.PLAYABLE,  # the raised budget buys a second, and it wins
        ]
        qobj = QueueObject(webpage_url, "Dial", mock_ctx.author)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url, title="Second"),
            ) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert song.title == "Second"
        assert mock_extract.call_count == 2

    async def test_unhealthy_probe_path_serves_the_cached_url_untouched(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Once enough probes in a row fail to complete, the probe — not the URL — is
        what is broken. Dropping every entry then converts a local fault into a global
        one: nothing repopulates the cache and every song re-extracts."""
        webpage_url = "https://yt.com/v=streak"
        await self._cache(fake_redis, webpage_url, title="Cached Original")
        playable_urls.return_value = StreamProbe.UNCONFIRMED
        for _ in range(_UNCONFIRMED_STREAK_LIMIT):
            await _probe_stream_url("https://anything")
        assert probe_path_looks_broken()

        qobj = QueueObject(webpage_url, "Streak", mock_ctx.author)
        with (
            patch("src.youtube._ytdlp_extract") as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert song.title == "Cached Original"
        mock_extract.assert_not_called()
        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is not None

    async def test_a_completed_probe_clears_the_unhealthy_streak(self) -> None:
        """One probe that reaches the host proves the path works; the streak must not
        latch, or a single bad minute would disable probing until restart."""
        with patch("src.youtube._probe_stream_url", new=_probe_stream_url):
            for _ in range(_UNCONFIRMED_STREAK_LIMIT):
                await _probe_stream_url("")  # empty URL -> DEAD, a completed verdict
        assert not probe_path_looks_broken()

    async def test_prefetch_never_re_extracts_an_unconfirmed_cached_url(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """_cancel_prefetch() awaits this task and an executor job cannot be
        interrupted, so a re-extraction here would put an uninterruptible wait in front
        of every -clear/-shuffle/-remove. The play-time resolve decides instead."""
        webpage_url = "https://yt.com/v=prefetch_noreextract"
        await self._cache(fake_redis, webpage_url, title="Cached Original")
        playable_urls.return_value = StreamProbe.UNCONFIRMED
        qobj = QueueObject(webpage_url, "Prefetch", mock_ctx.author)

        with (
            patch("src.youtube._ytdlp_extract") as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(
                qobj,
                AsyncMock(spec=discord.TextChannel),
                redis=fake_redis,
                allow_reextract=False,
            )

        assert song.title == "Cached Original"
        mock_extract.assert_not_called()

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (200, StreamProbe.PLAYABLE),
            (206, StreamProbe.PLAYABLE),
            # 400 exactly: the boundary. Without it `< 400` can drift to `<= 400`
            # and nothing notices, because every other refusal is 403/404.
            (400, StreamProbe.DEAD),
            (403, StreamProbe.DEAD),
            (404, StreamProbe.DEAD),
            (429, StreamProbe.UNCONFIRMED),
            (503, StreamProbe.UNCONFIRMED),
        ],
    )
    async def test_probe_maps_status_to_verdict(
        self, status: int, expected: StreamProbe
    ) -> None:
        """Pins the whole mapping, not just the refusal. Nothing else in the suite
        drives a 2xx through the real probe — every other caller patches it out — so a
        success that stopped mapping to PLAYABLE would only surface as songs silently
        never being cached. 429 and 5xx say "not right now" exactly as a timeout does,
        so they must not delete a cache entry."""
        response = MagicMock()
        response.status = status
        session = MagicMock()
        session.get.return_value.__aenter__ = AsyncMock(return_value=response)
        session.get.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("src.youtube._get_probe_session", return_value=session):
            assert await _probe_stream_url("https://cdn/x") is expected

    async def test_a_probe_bug_is_logged_at_error_not_swallowed_as_a_verdict(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A defect in this function is not evidence about the URL. It still answers
        UNCONFIRMED — no probe bug may cost a song — but at ERROR, or a session left
        dead reads as a flaky CDN for the life of the process. A real TypeError was
        swallowed as UNCONFIRMED once; only the log level tells them apart."""
        session = MagicMock()
        session.get.side_effect = TypeError("unexpected keyword")
        with patch("src.youtube._get_probe_session", return_value=session):
            with caplog.at_level(logging.ERROR):
                assert await _probe_stream_url("https://cdn/x") is (
                    StreamProbe.UNCONFIRMED
                )
        assert any(
            r.levelno == logging.ERROR and "failed unexpectedly" in r.getMessage()
            for r in caplog.records
        )

    async def test_a_network_failure_stays_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The ordinary case must not be promoted alongside it — a blocked probe is
        routine and would drown the signal the ERROR arm exists to carry."""
        session = MagicMock()
        session.get.side_effect = aiohttp.ClientConnectionError("refused")
        with patch("src.youtube._get_probe_session", return_value=session):
            with caplog.at_level(logging.DEBUG):
                assert await _probe_stream_url("https://cdn/x") is (
                    StreamProbe.UNCONFIRMED
                )
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    async def test_resolve_records_the_probe_verdict_on_the_span(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The verdict is the trace-side signal for the whole incident class, and it is
        what a Loki/Tempo query joins a user report to. Nothing else asserts it, so it
        can be dropped or misnamed with every behavioural test still green."""
        webpage_url = "https://yt.com/v=span_attr"
        playable_urls.return_value = StreamProbe.UNCONFIRMED
        qobj = QueueObject(webpage_url, "Span", mock_ctx.author)
        attrs: dict[str, Any] = {}
        span = MagicMock()
        span.set_attribute = lambda k, v: attrs.__setitem__(k, v)
        # The patch is module-wide, so ytdlp_pool's context carrier sees this span too
        # and formats its ids. An invalid context makes it return {} instead.
        span.get_span_context.return_value.is_valid = False

        with (
            patch("src.youtube.trace.get_current_span", return_value=span),
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert attrs["ytdl.stream_probe"] == StreamProbe.UNCONFIRMED.value

    async def test_probe_and_cache_records_the_verdict_on_the_span(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """The prefetch/yt_source half. Without it, "we silently stopped caching
        everything because probes are failing" is indistinguishable in a trace from
        "uncacheable, no usable expiry" and from a revoked URL."""
        webpage_url = "https://yt.com/v=span_prefetch"
        qobj = QueueObject(webpage_url, "SpanPrefetch", mock_ctx.author)
        attrs: dict[str, Any] = {}
        span = MagicMock()
        span.set_attribute = lambda k, v: attrs.__setitem__(k, v)
        # The patch is module-wide, so ytdlp_pool's context carrier sees this span too
        # and formats its ids. An invalid context makes it return {} instead.
        span.get_span_context.return_value.is_valid = False

        with (
            patch("src.youtube.trace.get_current_span", return_value=span),
            patch(
                "src.youtube._probe_stream_url",
                AsyncMock(return_value=StreamProbe.UNCONFIRMED),
            ),
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
        ):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)

        assert attrs["ytdl.stream_probe"] == StreamProbe.UNCONFIRMED.value

    async def test_unconfirmed_fresh_url_still_records_its_serving_format(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """_record_serving_format is the once-per-format warning that a muxed/HLS
        fallback has taken over — the early-warning system for YouTube-side change.
        Dropping it on this path would silence it with no other symptom."""
        webpage_url = "https://yt.com/v=fmt_unconfirmed"
        playable_urls.return_value = StreamProbe.UNCONFIRMED
        qobj = QueueObject(webpage_url, "Fmt", mock_ctx.author)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
            patch("src.youtube._record_serving_format") as mock_record,
        ):
            await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        mock_record.assert_called()

    async def test_unhandled_probe_verdict_raises_rather_than_being_treated_as_dead(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The DEAD arm is reached by elimination. A fourth StreamProbe member would
        silently inherit "YouTube revoked this URL" — deleting cache entries and
        mislabelling spans — so the arm asserts what it is rather than assuming."""
        webpage_url = "https://yt.com/v=bogus_verdict"
        # Verdict-shaped (it has .value, which the span attribute reads) but not a
        # member — exactly what adding a fourth StreamProbe would look like here.
        playable_urls.return_value = cast(
            StreamProbe, SimpleNamespace(value="not-a-verdict")
        )
        qobj = QueueObject(webpage_url, "Bogus", mock_ctx.author)

        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
            pytest.raises(AssertionError, match="unhandled stream probe verdict"),
        ):
            await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

    async def test_probe_opens_the_request_the_way_ffmpeg_does(self) -> None:
        """A revoked URL still answers 206 to a *ranged* GET while refusing
        the open-ended one ffmpeg actually sends. Probing with a Range header (or HEAD)
        reports a dead URL as healthy — which is the bug this whole path exists to catch.
        """
        response = MagicMock()
        response.status = 403
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession.get", return_value=ctx) as mock_get:
            probe = await _probe_stream_url("https://r2.googlevideo.com/s")
        assert probe is StreamProbe.DEAD

        assert "headers" not in mock_get.call_args.kwargs

    async def test_probe_failure_is_unconfirmed_not_dead(self) -> None:
        """A probe that cannot complete must never be the reason a song refuses to play.
        UNCONFIRMED keeps it playing; not being PLAYABLE keeps it out of the cache."""
        with patch(
            "aiohttp.ClientSession.get", side_effect=OSError("network unreachable")
        ):
            probe = await _probe_stream_url("https://r2.googlevideo.com/stream")

        assert probe is StreamProbe.UNCONFIRMED
        assert probe is not StreamProbe.DEAD

    async def test_probe_timeout_is_unconfirmed(self) -> None:
        """The shape the live incident took: the edge accepted nothing, so the probe hit
        its own deadline rather than getting a status back."""
        with patch("aiohttp.ClientSession.get", side_effect=asyncio.TimeoutError()):
            probe = await _probe_stream_url("https://r2.googlevideo.com/stream")

        assert probe is StreamProbe.UNCONFIRMED

    async def test_empty_url_is_dead(self) -> None:
        assert await _probe_stream_url("") is StreamProbe.DEAD


class TestStreamCache:
    async def test_cache_hit_skips_executor(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """Second yt_stream call with same URL should use Redis cache."""
        future = int(time.time()) + 7200
        cached_data = _fake_ytdl_data(
            url=f"https://r2.googlevideo.com/stream?expire={future}",
            webpage_url="https://yt.com/v=cache_hit",
            title="Cached Song",
        )
        await fake_redis.set(
            "ytdl:stream:https://yt.com/v=cache_hit",
            orjson.dumps(cached_data),
            ex=3600,
        )
        qobj = QueueObject("https://yt.com/v=cache_hit", "Cached Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()

        with (
            patch("src.youtube._ytdlp_extract") as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            await YTDL.yt_stream(qobj, channel, redis=fake_redis)
        mock_extract.assert_not_called()

    async def test_cache_miss_calls_executor_and_populates_cache(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """On cache miss, executor is called and result is written to Redis."""
        fake_data = _fake_ytdl_data(
            webpage_url="https://yt.com/v=cache_miss",
            title="Miss Song",
        )
        qobj = QueueObject("https://yt.com/v=cache_miss", "Miss Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            await YTDL.yt_stream(qobj, channel, redis=fake_redis)

        mock_extract.assert_called_once()
        cached = await fake_redis.get("ytdl:stream:https://yt.com/v=cache_miss")
        assert cached is not None

    async def test_cache_graceful_on_redis_error(self, mock_ctx: MagicMock) -> None:
        """Redis failure during cache check must not crash yt_stream; executor is called."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=err")
        bad_redis = AsyncMock()
        bad_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        qobj = QueueObject("https://yt.com/v=err", "Error Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data) as mock_extract,
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            await YTDL.yt_stream(qobj, channel, redis=bad_redis)

        mock_extract.assert_called_once()


class TestRecordServingFormat:
    """_record_serving_format is the fallback-ladder telemetry: an audio-only serve is
    business as usual; a muxed A/V serve means the primary path is degraded — either
    the audio-only primary fell back to muxed-only or the fallback is serving."""

    @pytest.fixture(autouse=True)
    def _reset_warned_formats(self) -> Iterator[None]:
        _DEGRADED_FORMAT_WARNED.clear()
        yield
        _DEGRADED_FORMAT_WARNED.clear()

    def test_audio_only_format_never_warns(self) -> None:
        with patch("src.youtube.log") as mock_log:
            _record_serving_format(
                {"format_id": "251", "protocol": "https", "vcodec": "none"}
            )
        mock_log.warning.assert_not_called()

    def test_muxed_format_warns_once_per_format(self) -> None:
        """A real primary-client outage affects every song — one warning per format, not
        one per song."""
        muxed: YTDLVideoMetadata = {
            "format_id": "18",
            "protocol": "https",
            "vcodec": "avc1.42001E",
        }
        with patch("src.youtube.log") as mock_log:
            _record_serving_format(muxed)
            _record_serving_format(muxed)
        assert mock_log.warning.call_count == 1
        assert "format_id=18" in mock_log.warning.call_args.args[0]

    def test_distinct_muxed_formats_each_warn(self) -> None:
        with patch("src.youtube.log") as mock_log:
            _record_serving_format(
                {"format_id": "18", "protocol": "https", "vcodec": "avc1.42001E"}
            )
            _record_serving_format(
                {"format_id": "96", "protocol": "m3u8_native", "vcodec": "avc1.640028"}
            )
        assert mock_log.warning.call_count == 2

    def test_missing_vcodec_is_treated_as_healthy(self) -> None:
        """Cache entries written before vcodec was persisted must never warn —
        the song they describe may be perfectly healthy."""
        with patch("src.youtube.log") as mock_log:
            _record_serving_format({"format_id": "251", "protocol": "https"})
        mock_log.warning.assert_not_called()

    def test_format_shape_survives_the_cache_strip(self) -> None:
        """The shape fields must be in _STREAM_CACHE_FIELDS, or cache-hit plays would
        lose attribution and the degraded-primary signal would only fire on misses."""
        assert {"format_id", "protocol", "vcodec"} <= _STREAM_CACHE_FIELDS


class TestPrefetchStream:
    async def test_populates_cache_on_miss(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """prefetch_stream calls yt-dlp and writes to Redis when key is absent."""
        fake_data = _fake_ytdl_data(
            webpage_url="https://yt.com/v=pf1", title="Prefetch Song"
        )
        qobj = QueueObject("https://yt.com/v=pf1", "Prefetch Song", mock_ctx.author)

        with patch(
            "src.youtube._ytdlp_extract", return_value=fake_data
        ) as mock_extract:
            await YTDL.prefetch_stream(qobj, redis=fake_redis)

        mock_extract.assert_called_once()
        cached = await fake_redis.get("ytdl:stream:https://yt.com/v=pf1")
        assert cached is not None
        assert orjson.loads(cached)["title"] == "Prefetch Song"

    async def test_no_op_when_redis_none(self, mock_ctx: MagicMock) -> None:
        """prefetch_stream returns immediately when redis is None — no exception."""
        qobj = QueueObject("https://yt.com/v=pf2", "No Redis", mock_ctx.author)
        with patch("src.youtube._ytdlp_extract") as mock_extract:
            await YTDL.prefetch_stream(qobj, redis=None)
        mock_extract.assert_not_called()

    async def test_no_op_when_already_cached(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """prefetch_stream skips yt-dlp extraction when the key is already in Redis."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=pf3")
        await fake_redis.set(
            "ytdl:stream:https://yt.com/v=pf3",
            orjson.dumps(fake_data),
            ex=3600,
        )
        qobj = QueueObject("https://yt.com/v=pf3", "Already Cached", mock_ctx.author)
        with patch("src.youtube._ytdlp_extract") as mock_extract:
            await YTDL.prefetch_stream(qobj, redis=fake_redis)
        mock_extract.assert_not_called()

    async def test_swallows_extraction_errors(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """prefetch_stream does not propagate yt-dlp exceptions. The failure is an
        ExtractionError because that is what production raises — the worker flattens
        yt-dlp's own unpicklable errors — so a bare Exception here would assert a
        shape the code can no longer produce."""
        from src.youtube import ExtractionError

        qobj = QueueObject("https://yt.com/v=pf4", "Error Song", mock_ctx.author)
        with patch(
            "src.youtube._ytdlp_extract",
            side_effect=ExtractionError(
                "ERROR: [youtube] pf4: Video unavailable",
                original_type="DownloadError",
                expected=True,
                video_id="pf4",
            ),
        ):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)
        cached = await fake_redis.get("ytdl:stream:https://yt.com/v=pf4")
        assert cached is None

    async def test_skips_write_when_ttl_too_short(
        self, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        """prefetch_stream does not cache a URL that is already near expiry."""
        soon = int(time.time()) + 30  # 30s — below the 60s threshold
        fake_data = _fake_ytdl_data(
            url=f"https://r2.googlevideo.com/stream?expire={soon}",
            webpage_url="https://yt.com/v=pf5",
        )
        qobj = QueueObject("https://yt.com/v=pf5", "Nearly Expired", mock_ctx.author)
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)
        cached = await fake_redis.get("ytdl:stream:https://yt.com/v=pf5")
        assert cached is None


class TestYTStreamPlaynowFlags:
    async def test_flags_carried_onto_ytdl(self, mock_ctx: MagicMock) -> None:
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            ts=90,
            is_resume=True,
            start_paused=True,
        )

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            result = await YTDL.yt_stream(qobj, channel)

        assert result.is_resume is True
        assert result.start_paused is True
        assert result.interjected is False

    @pytest.mark.parametrize("is_resume", [False, True])
    async def test_no_notice_is_sent_from_construction_but_the_seek_remains(
        self, mock_ctx: MagicMock, is_resume: bool
    ) -> None:
        """yt_stream sends no user notice any more — every one moved to the loop's
        start path (test_musicplayer.py::TestStartOffsetAnnounce). The -ss seek stays.

        `is_resume=False` is the case that changed: resume entries were already
        silent here, so covering only those would pass against the old code too."""
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            ts=151,
            is_resume=is_resume,
        )

        captured_options = {}

        def capture_init(
            self: discord.FFmpegOpusAudio,
            url: str,
            *,
            executable: str,
            before_options: Optional[str],
            options: Optional[str],
        ) -> None:
            noop_ffmpeg_init(self)
            captured_options["options"] = options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel)

        channel.send.assert_not_awaited()
        assert "-ss 151" in captured_options["options"]


class TestPotProviderCompatibility:
    """The bgutil PO-token plugin against the yt-dlp actually installed.

    Nothing else can catch this pair drifting. `bgutil-ytdlp-pot-provider` declares NO
    `Requires-Dist` on yt-dlp at all, so neither Poetry nor pip can detect a mismatch,
    and CLAUDE.md rule 6a's hand check covers plugin <-> sidecar IMAGE, not plugin <->
    host library. A nightly yt-dlp is exactly what moves the plugin API, and the symptom
    is YouTube playback failing in production rather than a red build.
    """

    def test_bgutil_providers_register_with_the_installed_yt_dlp(self) -> None:
        from yt_dlp.YoutubeDL import YoutubeDL

        YoutubeDL({"quiet": True})  # triggers plugin discovery
        from yt_dlp.extractor.youtube.pot._registry import _pot_providers

        registered = set(_pot_providers.value)
        assert "BgUtilHTTP" in registered, (
            f"bgutil PO-token provider did not register: {sorted(registered)}. "
            "The plugin and this yt-dlp disagree; PO-token minting is down, which "
            "surfaces as YouTube playback failures, not as a build error."
        )

    def test_extractor_args_name_no_client(self) -> None:
        """The strategy is to track yt-dlp's own default, so this config must keep
        naming none. A hardcoded client here would pin the bot to one nobody upstream
        is defending — the failure the client-strategy comment exists to prevent."""
        clients = _EXTRACTOR_ARGS["youtube"]["player_client"]
        assert clients[0] == "default"
        assert not [c for c in clients if not c.startswith("-") and c != "default"]


class TestProcessBoundaryContract:
    """Everything that must survive being pickled to a worker process.

    The suite runs extraction on an in-process ThreadPoolExecutor (see conftest), so
    nothing else exercises the pickling production performs on every submit. These
    assert the contract directly, in microseconds; the expensive half is
    TestRealWorkerProcess in tests/test_ytdlp_pool.py. What they catch: an
    unpicklable value in an opts profile (a logger, a session, a lambda), or a
    submitted top-level function becoming a closure or bound method — each breaks
    every extraction in production while a green suite says nothing."""

    @pytest.mark.parametrize(
        "name,opts",
        [
            ("stream", _YTDL_STREAM_OPTS),
            ("search", _YTDL_STREAM_SEARCH_OPTS),
            ("playlist", _YTDL_PLAYLIST_OPTS),
        ],
    )
    def test_opts_profile_survives_a_round_trip(
        self, name: str, opts: dict[str, Any]
    ) -> None:
        """Every profile is an argument to _ytdlp_extract, so it is pickled per call.
        Round-tripped, not just dumped: a value that serialises but cannot
        reconstruct fails only in the worker, as an opaque BrokenProcessPool."""
        restored = pickle.loads(pickle.dumps(opts))
        assert restored.keys() == opts.keys(), f"{name} profile lost keys"

    def test_extract_worker_is_picklable_by_reference(self) -> None:
        """_ytdlp_extract is pickled by qualified name, not by value — so it must stay
        a module-level function. `is` rather than `==`: pickle resolves the name on the
        far side, and only a real module-level lookup round-trips to the same object."""
        assert pickle.loads(pickle.dumps(_ytdlp_extract)) is _ytdlp_extract

    def test_worker_logging_initializer_is_picklable_by_reference(self) -> None:
        """ProcessPoolExecutor pickles `initializer` to every worker, so a closure or
        bound method here breaks pool construction rather than one extraction. The
        initializer is `_worker_init`, not configure_worker_logging itself; asserting
        the wrong one passes while missing the real contract."""
        from src.ytdlp_pool import _worker_init

        assert pickle.loads(pickle.dumps(_worker_init)) is _worker_init
        # the function it delegates to must survive the boundary too
        assert (
            pickle.loads(pickle.dumps(configure_worker_logging))
            is configure_worker_logging
        )


def _realistic_raw_info(**overrides: Any) -> dict[str, Any]:
    """A process=True-shaped info dict carrying what a flat _fake_ytdl_data() never
    has: a genuinely unpicklable live object, a LazyList format ladder, and the large
    collections no caller reads — what _ytdlp_extract must not return as-is."""
    from yt_dlp.utils import LazyList

    base: dict[str, Any] = {
        "url": f"https://r2.googlevideo.com/stream?expire={int(time.time()) + 7200}",
        "webpage_url": "https://www.youtube.com/watch?v=test",
        "title": "Test Song",
        "duration": 180,
        "uploader": "Test Channel",
        "thumbnail": "https://img.yt.com/test.jpg",
        "abr": 128,
        "asr": 44100,
        "acodec": "opus",
        "format_id": "251",
        "protocol": "https",
        "vcodec": "none",
        # A LazyList of format dicts — what yt-dlp stores `formats` as; sanitize_info
        # must materialise it to a plain list before it can cross the boundary.
        "formats": LazyList(iter([{"format_id": "251", "url": "https://x"}])),
        "thumbnails": [{"url": "https://img/1.jpg"}, {"url": "https://img/2.jpg"}],
        "automatic_captions": {"en": [{"url": "https://c"}]},
        "heatmap": [{"start_time": 0.0, "value": 1.0}],
        # A live, genuinely-unpicklable object under a private key — a lock stands in
        # for the loggers/tracebacks/callables a real info-dict carries. This is what
        # makes the raw dict impossible to ship back untouched.
        "__lock": threading.Lock(),
    }
    base.update(overrides)
    return base


class TestSlimInfoReturnContract:
    """The success-path return value crosses the process boundary on *every* extraction.

    TestProcessBoundaryContract covers the arguments and the callable; this covers the
    info-dict _ytdlp_extract returns, which the rest of the suite never exercises
    against a realistic shape. _slim_info is what makes it picklable at all."""

    def test_a_realistic_raw_info_dict_is_genuinely_unpicklable(self) -> None:
        """Guards the premise: if this ever starts pickling on its own, the fix below is
        no longer needed and this test should be revisited, not deleted silently.
        """
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(_realistic_raw_info())

    def test_slimmed_info_round_trips_through_pickle(self) -> None:
        """_ytdlp_extract's return value survives the boundary. The
        pool pickles results synchronously, so a value that fails here fails *every*
        extraction with an opaque pickling error."""
        # cast to a plain dict: these assertions poke raw content, not the narrowed
        # YTDLExtractResult contract, and a realistic raw dict is never slimmed to None.
        slim = cast(dict[str, Any], _slim_info(_realistic_raw_info()))
        restored = pickle.loads(pickle.dumps(slim))
        assert restored["url"] == slim["url"]
        assert restored["webpage_url"] == slim["webpage_url"]

    def test_slimmed_info_keeps_every_field_callers_read(self) -> None:
        """No consumed field is lost to slimming. _STREAM_CACHE_FIELDS is the exhaustive
        set of info-dict keys this codebase reads; each one present in the raw dict must
        survive, unchanged."""
        raw = _realistic_raw_info()
        slim = cast(dict[str, Any], _slim_info(raw))
        for field in _STREAM_CACHE_FIELDS:
            if field in raw:
                assert slim.get(field) == raw[field], f"{field} lost or altered"

    def test_slimmed_info_drops_the_oversized_collections(self) -> None:
        """The performance half: the large lists no caller reads are gone, so they are
        not serialised worker->parent on every extraction."""
        slim = cast(dict[str, Any], _slim_info(_realistic_raw_info()))
        for field in _UNUSED_INFO_COLLECTIONS:
            assert field not in slim, f"{field} should have been dropped"

    def test_slimming_preserves_search_entries_but_slims_each(self) -> None:
        """A search/playlist wrapper's `entries` list must survive (yt_source unwraps it),
        but each entry carries its own formats ladder that must be dropped too."""
        wrapper = {
            "_type": "playlist",
            "entries": [_realistic_raw_info(), None, _realistic_raw_info()],
        }
        slim = cast(dict[str, Any], _slim_info(wrapper))
        entries = slim["entries"]
        assert len(entries) == 3
        assert entries[1] is None  # null entries (deleted/private) are preserved
        for entry in (entries[0], entries[2]):
            assert entry["webpage_url"] == "https://www.youtube.com/watch?v=test"
            for field in _UNUSED_INFO_COLLECTIONS:
                assert field not in entry
        pickle.loads(pickle.dumps(slim))  # the whole wrapper still round-trips

    def test_slim_info_passes_none_through(self) -> None:
        """A failed extract_info returns None; callers branch on `data is None`, so
        slimming must not turn it into anything else."""
        assert _slim_info(None) is None


class TestExtractionErrorClassification:
    """_ytdlp_extract's worker-side rewrap of yt-dlp's own (unpicklable) errors.

    DownloadError's exc_info holds a live traceback, so it cannot cross the boundary
    and the parent would get an opaque pickling error instead of the reason. The
    worker re-raises a flat ExtractionError while the structure still exists; the far
    half is TestErrorAcrossBoundary in tests/test_ytdlp_pool.py."""

    def _real_downloaderror(self) -> Any:
        """A DownloadError shaped exactly like yt-dlp's: exc_info holds the ExtractorError
        that carries .expected / .video_id. Built without any network."""
        import sys

        from yt_dlp.utils import DownloadError, ExtractorError

        try:
            raise ExtractorError("Video unavailable", video_id="vid42", expected=True)
        except ExtractorError:
            return DownloadError(
                "ERROR: [youtube] vid42: Video unavailable",
                sys.exc_info(),  # type: ignore[arg-type]
            )

    def test_downloaderror_is_reclassified_with_its_fields_mined(self) -> None:
        from src.youtube import ExtractionError, _ytdlp_extract

        real_error = self._real_downloaderror()
        fake_ydl = MagicMock()
        fake_ydl.extract_info.side_effect = real_error

        with patch("src.youtube.youtube_dl.YoutubeDL", return_value=fake_ydl):
            with pytest.raises(ExtractionError) as caught:
                _ytdlp_extract(ExtractRequest(url="http://x", opts=_YTDL_STREAM_OPTS))

        err = caught.value
        assert err.original_type == "DownloadError"
        assert err.video_id == "vid42"
        assert err.expected is True
        assert "Video unavailable" in err.message
        # `raise ... from e` — the original error is preserved as the cause, so the
        # worker traceback reaches the parent (as _RemoteTraceback across a real boundary).
        assert err.__cause__ is real_error

    def test_non_ytdlp_errors_are_not_swallowed(self) -> None:
        """Only yt-dlp's own errors get rewrapped. A programming error (KeyError, …) must
        propagate unchanged so it is not silently relabelled as an extraction failure.
        """
        from src.youtube import ExtractionError, _ytdlp_extract

        fake_ydl = MagicMock()
        fake_ydl.extract_info.side_effect = KeyError("bug")

        with patch("src.youtube.youtube_dl.YoutubeDL", return_value=fake_ydl):
            with pytest.raises(KeyError):
                _ytdlp_extract(ExtractRequest(url="http://x", opts=_YTDL_STREAM_OPTS))
        assert not isinstance(KeyError("bug"), ExtractionError)

    def test_extractionerror_survives_pickle_round_trip(self) -> None:
        """loads(dumps(...)), not dumps alone: a multi-arg __init__ without __reduce__
        serialises fine and fails only on the *parent* side while unpickling, which is
        exactly how the naive fix bricks the pool. Values, not just keys."""
        from src.youtube import ExtractionError

        err = ExtractionError(
            "ERROR: [youtube] v9: Video unavailable",
            original_type="DownloadError",
            expected=True,
            video_id="v9",
            cause_type="NameResolutionError",
            unsupported=True,
        )
        back = pickle.loads(pickle.dumps(err))

        assert isinstance(back, ExtractionError)
        assert str(back) == "ERROR: [youtube] v9: Video unavailable"
        assert back.message == err.message
        assert back.original_type == "DownloadError"
        assert back.expected is True
        assert back.video_id == "v9"
        assert back.cause_type == "NameResolutionError"
        assert back.unsupported is True

    def test_unsupported_url_is_classified_from_the_wrapped_cause(self) -> None:
        """yt-dlp raises UnsupportedError, then extract_info re-raises a DownloadError
        carrying it in exc_info. The worker classifies .unsupported off that wrapped
        inner error so the flag survives the (otherwise unpicklable) boundary."""
        import sys

        from src.youtube import ExtractionError, _ytdlp_extract

        try:
            raise UnsupportedError("https://example.com/not-media")
        except UnsupportedError:
            wrapped = DownloadError("ERROR: Unsupported URL", sys.exc_info())  # type: ignore[arg-type]

        fake_ydl = MagicMock()
        fake_ydl.extract_info.side_effect = wrapped
        with patch("src.youtube.youtube_dl.YoutubeDL", return_value=fake_ydl):
            with pytest.raises(ExtractionError) as caught:
                _ytdlp_extract(ExtractRequest(url="http://x", opts=_YTDL_STREAM_OPTS))
        assert caught.value.unsupported is True

    def test_non_unsupported_error_is_not_flagged_unsupported(self) -> None:
        """A garden-variety DownloadError (network failure, video unavailable) must
        classify with unsupported=False — only genuine UnsupportedError sets it."""
        from src.youtube import ExtractionError, _ytdlp_extract

        fake_ydl = MagicMock()
        fake_ydl.extract_info.side_effect = self._real_downloaderror()
        with patch("src.youtube.youtube_dl.YoutubeDL", return_value=fake_ydl):
            with pytest.raises(ExtractionError) as caught:
                _ytdlp_extract(ExtractRequest(url="http://x", opts=_YTDL_STREAM_OPTS))
        assert caught.value.unsupported is False

    def test_user_message_shows_the_reason_for_an_expected_error(self) -> None:
        """expected=True is yt-dlp's own user-facing reason; show it minus the prefix."""
        from src.youtube import ExtractionError

        err = ExtractionError(
            "ERROR: [youtube] v9: Video unavailable",
            original_type="DownloadError",
            expected=True,
        )
        assert err.user_message == "[youtube] v9: Video unavailable"

    def test_user_message_is_generic_for_an_unexpected_error(self) -> None:
        """expected=False can carry yt-dlp's 'please report this issue on github.com/yt-dlp'
        boilerplate — never surface the raw text; full detail stays in the span/logs."""
        from src.youtube import ExtractionError

        err = ExtractionError(
            "ERROR: [generic] x: Unable to download webpage; please report this issue on "
            "https://github.com/yt-dlp/yt-dlp/issues",
            original_type="DownloadError",
            expected=False,
        )
        assert err.user_message == (
            "Couldn't load this track — the extractor hit an unexpected error."
        )
        assert "github.com" not in err.user_message

    def test_user_message_falls_back_when_expected_but_empty(self) -> None:
        from src.youtube import ExtractionError

        assert (
            ExtractionError("", expected=True).user_message
            == "Couldn't load this track."
        )


class TestExtractRequest:
    """The request object that crosses the process boundary. Its whole purpose is
    that `download` and `process` — two interchangeable bools — cannot be
    transposed; kw_only provides that, frozen and slots are for the crossing. Each
    is asserted separately: a `@dataclass(frozen=True)` dropping kw_only would look
    correct in review while silently restoring the original defect."""

    def test_flags_cannot_be_passed_positionally(self) -> None:
        """The defect this type exists to prevent: without kw_only,
        `ExtractRequest("u", opts, True, False)` is accepted and means
        download=True, process=False — the exact silent swap."""
        with pytest.raises(TypeError):
            ExtractRequest("http://x", {}, True, False)  # pyright: ignore[reportCallIssue]

    def test_is_frozen(self) -> None:
        """A worker must not mutate a request in a way the parent never sees."""
        req = ExtractRequest(url="http://x", opts={})
        with pytest.raises(FrozenInstanceError):
            req.download = True  # pyright: ignore[reportAttributeAccessIssue]

    def test_defaults_are_download_false_process_true(self) -> None:
        req = ExtractRequest(url="http://x", opts={})
        assert req.download is False
        assert req.process is True

    def test_survives_a_pickle_round_trip(self) -> None:
        """It is submitted to a ProcessPoolExecutor, so it must pickle: an
        unpicklable field added later breaks every extraction at once, and can brick
        the pool permanently. Asserted field-by-field rather than with `==` because a
        real opts profile carries a live `_YtdlpLogger`, which has no __eq__ and so
        compares by identity — equality would fail for an unrelated reason."""
        req = ExtractRequest(
            url="http://x", opts=dict(_YTDL_STREAM_OPTS), download=True
        )

        back = pickle.loads(pickle.dumps(req))

        assert (back.url, back.download, back.process) == (
            req.url,
            req.download,
            req.process,
        )
        assert sorted(back.opts) == sorted(req.opts)

    def test_pickles_with_a_plain_opts_dict(self) -> None:
        """With no live objects in opts the round-trip is fully equal — this is
        the assertion the test above would make if opts were plain data."""
        req = ExtractRequest(url="http://x", opts={"quiet": True}, process=False)
        assert pickle.loads(pickle.dumps(req)) == req

    def test_replace_derives_a_variant(self) -> None:
        """dataclasses.replace is how a caller varies one option without
        respelling the rest — the ergonomic that keeps growth cheap."""
        base = ExtractRequest(url="http://x", opts={})
        assert replace(base, download=True) == ExtractRequest(
            url="http://x", opts={}, download=True
        )
        assert base.download is False  # original untouched


class TestRunExtract:
    """`_run_extract` — the single call site for _ytdlp_extract, through which every
    extraction path routes. It forwards one opaque request, so these assert the
    plumbing (request reaches the worker unchanged, both seams stay patchable)
    rather than argument order, which the type system enforces."""

    async def test_forwards_the_request_unchanged(self) -> None:
        req = ExtractRequest(url="https://yt.com/v=x", opts={"quiet": True})
        with patch("src.youtube.ytdlp_pool") as mock_pool:
            mock_pool.run = AsyncMock(return_value={"title": "T"})
            await _run_extract(req)

        fn, forwarded = mock_pool.run.call_args[0]
        assert forwarded is req  # not rebuilt, not copied, not spread

    async def test_submits_the_module_level_ytdlp_extract(self) -> None:
        """The callable must be resolved per call, not captured at def time: ~29
        tests and tests/conftest.py patch `src.youtube._ytdlp_extract` and
        `src.youtube.ytdlp_pool`, and capturing either at import would make every
        one of those patches silently ineffective."""
        sentinel = MagicMock(name="patched_extract")
        with patch("src.youtube.ytdlp_pool") as mock_pool:
            mock_pool.run = AsyncMock(return_value=None)
            with patch("src.youtube._ytdlp_extract", sentinel):
                await _run_extract(ExtractRequest(url="https://yt.com/v=x", opts={}))

        assert mock_pool.run.call_args[0][0] is sentinel

    async def test_uses_the_pool_bound_at_call_time(self) -> None:
        """Same seam for the pool itself — conftest swaps in a thread pool."""
        req = ExtractRequest(url="https://yt.com/v=x", opts={})
        with patch("src.youtube.ytdlp_pool") as first:
            first.run = AsyncMock(return_value=None)
            await _run_extract(req)
        with patch("src.youtube.ytdlp_pool") as second:
            second.run = AsyncMock(return_value=None)
            await _run_extract(req)

        first.run.assert_awaited_once()
        second.run.assert_awaited_once()

    @pytest.mark.parametrize("download", [True, False])
    @pytest.mark.parametrize("process", [True, False])
    async def test_both_flags_reach_the_worker_independently(
        self, download: bool, process: bool
    ) -> None:
        """All four combinations still asserted: the type system stops a swap at
        construction, but nothing stops _run_extract from dropping or defaulting
        a field on the way to the pool."""
        req = ExtractRequest(
            url="https://yt.com/v=x", opts={}, download=download, process=process
        )
        with patch("src.youtube.ytdlp_pool") as mock_pool:
            mock_pool.run = AsyncMock(return_value=None)
            await _run_extract(req)

        forwarded = mock_pool.run.call_args[0][1]
        assert forwarded.download is download
        assert forwarded.process is process

    async def test_returns_the_pool_result_unchanged(self) -> None:
        """No post-processing here — _slim_info already ran in the worker."""
        payload = {"webpage_url": "https://yt.com/v=x", "title": "Song"}
        with patch("src.youtube.ytdlp_pool") as mock_pool:
            mock_pool.run = AsyncMock(return_value=payload)
            result = await _run_extract(
                ExtractRequest(url="https://yt.com/v=x", opts={})
            )
        assert result is payload

    async def test_none_result_is_passed_through(self) -> None:
        """A None extraction is a normal "nothing found" outcome, not an error."""
        with patch("src.youtube.ytdlp_pool") as mock_pool:
            mock_pool.run = AsyncMock(return_value=None)
            assert (
                await _run_extract(ExtractRequest(url="https://yt.com/v=x", opts={}))
                is None
            )

    async def test_pool_exception_propagates(self) -> None:
        """_classify_ytdlp_error already ran in the worker; _run_extract must not
        add a second layer of swallowing that would hide it from callers."""
        with patch("src.youtube.ytdlp_pool") as mock_pool:
            mock_pool.run = AsyncMock(side_effect=DownloadError("boom"))
            with pytest.raises(DownloadError):
                await _run_extract(ExtractRequest(url="https://yt.com/v=x", opts={}))


class TestProbeSessionSharing:
    """The probe holds one process-wide session. It pools nothing (the body goes
    unread), so what these pin is the lifecycle: reuse, replacement and close."""

    async def test_probe_session_is_reused(self) -> None:
        import src.youtube as youtube

        first = youtube._get_probe_session()
        second = youtube._get_probe_session()
        assert first is second
        await youtube.close_probe_session()

    async def test_close_clears_the_global(self) -> None:
        import src.youtube as youtube

        session = youtube._get_probe_session()
        await youtube.close_probe_session()
        assert youtube._probe_session is None
        assert session.closed

    async def test_a_closed_session_is_replaced(self) -> None:
        """Defensive: if something closes it out from under us, the next probe
        must build a new one rather than raise on a dead session."""
        import src.youtube as youtube

        first = youtube._get_probe_session()
        await first.close()
        second = youtube._get_probe_session()
        assert second is not first
        await youtube.close_probe_session()

    async def test_close_without_a_session_is_a_noop(self) -> None:
        import src.youtube as youtube

        await youtube.close_probe_session()
        await youtube.close_probe_session()  # must not raise

    async def test_a_probe_after_close_does_not_rebuild(self) -> None:
        """close() is followed by a span flush that yields the loop for up to 30s,
        and nothing tears the players down first — so the playback loop can reach a
        probe after this ran. Rebuilding there strands a session nothing closes."""
        import src.youtube as youtube

        youtube._get_probe_session()
        await youtube.close_probe_session()

        with pytest.raises(youtube.ProbeSessionClosed):
            youtube._get_probe_session()
        assert youtube._probe_session is None

    async def test_a_probe_after_close_is_unconfirmed_not_an_error(self) -> None:
        """Shutdown is not a probe defect: it must not take the ERROR arm, and it
        must not answer DEAD — that would drop a cache entry on the way out."""
        import src.youtube as youtube

        await youtube.close_probe_session()
        assert await _probe_stream_url("https://cdn/x") is StreamProbe.UNCONFIRMED

    async def test_probe_connector_is_unbounded(self) -> None:
        """One connector serves every guild now. aiohttp's default limit of 100
        would queue the 101st probe against its own 2s budget and report a healthy
        URL as UNCONFIRMED — which _unconfirmed_streak then counts process-wide."""
        import src.youtube as youtube

        session = youtube._get_probe_session()
        try:
            assert session.connector is not None
            assert session.connector.limit == 0
        finally:
            await youtube.close_probe_session()

    async def test_probe_session_keeps_no_cookies(self) -> None:
        """One session serves every guild, and `-play <any url>` reaches it via the
        generic extractor — so a default CookieJar would let one guild set a
        `Domain=com` cookie (aiohttp applies no public-suffix check) that is then
        replayed to googlevideo for everyone until restart. Enough cookie bytes
        turns every probe into a 400, which maps to DEAD, drops the cache entry and
        burns the re-extraction. Nothing self-heals: probe_path_looks_broken()
        watches UNCONFIRMED streaks and a DEAD verdict resets that counter."""
        import src.youtube as youtube

        session = youtube._get_probe_session()
        try:
            hostile = SimpleCookie()
            hostile["pwn"] = "AAAA"
            hostile["pwn"]["domain"] = "com"
            hostile["pwn"]["path"] = "/"
            session.cookie_jar.update_cookies(hostile, URL("https://evil.com/x"))

            assert len(session.cookie_jar) == 0
            replayed = session.cookie_jar.filter_cookies(
                URL("https://rr3---sn-4g5e6nez.googlevideo.com/videoplayback")
            )
            assert dict(replayed) == {}
        finally:
            await youtube.close_probe_session()
