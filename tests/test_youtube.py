"""Tests for src/youtube.py — QueueObject, YTDL config, yt_source, yt_stream, and stream cache."""

import redis.asyncio as aioredis
import pickle
from dataclasses import FrozenInstanceError, replace
import threading
import time
from typing import Any, Optional, cast
from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from redis.asyncio import Redis
from yt_dlp.utils import DownloadError, UnsupportedError

from src.telemetry import configure_worker_logging
from src.guild_state import Analytics
from src.youtube import (
    YTDL,
    YTDL_OPTS,
    QueueObject,
    _CANDIDATE_FIELDS,
    _DEGRADED_FORMAT_WARNED,
    _STREAM_CACHE_FIELDS,
    _STREAM_CANDIDATES,
    _UNUSED_INFO_COLLECTIONS,
    _mine_audio_candidates,
    _YTDL_PLAYLIST_OPTS,
    _YTDL_STREAM_OPTS,
    _YTDL_STREAM_SEARCH_OPTS,
    _enrich_queueobject,
    _record_serving_format,
    _run_extract,
    ExtractRequest,
    _slim_info,
    _stream_url_playable,
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
    a fake googlevideo host. Revocation tests set the mock's return_value to False."""
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr("src.youtube._stream_url_playable", probe)
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
        """bestaudio is the healthy android_vr path; the ≤360p middle rung keeps the
        muxed fallback (web_safari / degraded android_vr) from streaming 1080p video
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
        """The bgutil plugin is what lets web_safari serve audio as a fallback client;
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


class TestSearchEntrySelection:
    """Which of a search's entries gets played. The old rule — first non-playlist
    entry — accepted a result yt-dlp had selected no format for, which then failed at
    stream time looking unrelated to the search."""

    async def _pick(self, mock_ctx: MagicMock, entries: list[Any]) -> QueueObject:
        with patch(
            "src.youtube._ytdlp_extract",
            return_value={"_type": "playlist", "entries": entries},
        ):
            return await YTDL.yt_source(
                mock_ctx.author,
                "some song",
                query_source="search",
                analytics=_ANALYTICS,
                user_input="some song",
            )

    async def test_an_entry_without_a_stream_url_is_passed_over(
        self, mock_ctx: MagicMock
    ) -> None:
        formatless = {"webpage_url": "https://yt.com/v=noformat", "title": "No Format"}
        playable = _fake_ytdl_data(webpage_url="https://yt.com/v=ok", title="Playable")

        result = await self._pick(mock_ctx, [formatless, playable])

        assert result.title == "Playable"

    async def test_playlists_and_null_entries_are_still_skipped(
        self, mock_ctx: MagicMock
    ) -> None:
        playable = _fake_ytdl_data(webpage_url="https://yt.com/v=ok", title="Playable")

        result = await self._pick(
            mock_ctx, [None, {"_type": "playlist", "url": "https://x"}, playable]
        )

        assert result.title == "Playable"

    async def test_the_first_entry_still_wins_when_none_carries_a_url(
        self, mock_ctx: MagicMock
    ) -> None:
        """The fallback is the old rule, so an entry shape this code does not
        recognise plays as before rather than failing outright."""
        first = {"webpage_url": "https://yt.com/v=first", "title": "First"}
        second = {"webpage_url": "https://yt.com/v=second", "title": "Second"}

        result = await self._pick(mock_ctx, [first, second])

        assert result.title == "First"


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
        playable_urls.return_value = False
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
            codec: Optional[str] = None,
        ) -> None:
            noop_ffmpeg_init(self)
            captured_options["options"] = options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel, volume=0.5)

        assert "volume=0.5" in captured_options["options"]

    async def test_yt_stream_seeks_on_both_sides_when_ts_is_set(
        self, mock_ctx: MagicMock
    ) -> None:
        """Two-pass seek: `-ss N` before -i for the range request, `-ss 0` after it to
        drop the pre-roll the input seek lands in. Input-side alone measured 5-10s
        early (webm cluster granularity) — which position_secs would then overstate
        on every surface. The volume filter must NOT follow the seek to the input
        side: ffmpeg silently ignores -filter:a placed ahead of the input."""
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
            codec: Optional[str] = None,
        ) -> None:
            noop_ffmpeg_init(self)
            captured_options["options"] = options
            captured_options["before_options"] = before_options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel, volume=0.5)

        assert "-ss 90" in captured_options["before_options"]
        assert "-ss 0" in captured_options["options"]
        assert "-ss 90" not in captured_options["options"]
        assert "volume=0.5" in captured_options["options"]
        assert "volume" not in captured_options["before_options"]

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
        """HLS manifest URLs — the muxed formats the degraded web_safari rung
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
        playable_urls.side_effect = [False, True]
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

    async def test_raises_when_youtube_refuses_even_a_fresh_url(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Both attempts refused — surface it so the player reports a failed song
        instead of handing ffmpeg a URL that will 403 into silence."""
        webpage_url = "https://yt.com/v=always_dead"
        await self._cache(fake_redis, webpage_url)
        playable_urls.return_value = False
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
            patch("src.youtube._stream_url_playable", AsyncMock(return_value=False)),
            patch(
                "src.youtube._ytdlp_extract",
                return_value=_fake_ytdl_data(webpage_url=webpage_url),
            ),
        ):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)

        assert await fake_redis.get(f"ytdl:stream:{webpage_url}") is None

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
            assert await _stream_url_playable("https://r2.googlevideo.com/s") is False

        assert "headers" not in mock_get.call_args.kwargs

    async def test_probe_failure_assumes_playable(self) -> None:
        """A probe that cannot complete is a statement about the network, not the URL —
        it must never be the reason a song refuses to play."""
        with patch(
            "aiohttp.ClientSession.get", side_effect=OSError("network unreachable")
        ):
            assert (
                await _stream_url_playable("https://r2.googlevideo.com/stream") is True
            )

    async def test_empty_url_is_not_playable(self) -> None:
        assert await _stream_url_playable("") is False


class TestCandidateLadderWalk:
    """A revoked URL used to cost a 3-5s re-extraction that re-selected the same
    format — curing a stale URL but not a format YouTube stopped serving. The ladder
    walks sideways first: next format, ~100ms probe, same extraction."""

    def _laddered(self, webpage_url: str, *format_ids: str) -> YTDLVideoInfo:
        """Stream data whose ladder is the given formats, best first, with the head
        hoisted as the selected format — the shape _cache_stream persists."""
        ladder = [_fmt(fid, "opus", 129.0, 48000) for fid in format_ids]
        return _fake_ytdl_data(
            webpage_url=webpage_url,
            audio_candidates=ladder,
            **{k: ladder[0][k] for k in _CANDIDATE_FIELDS},
        )

    async def _cache(self, fake_redis: aioredis.Redis, data: YTDLVideoInfo) -> None:
        await fake_redis.set(
            f"ytdl:stream:{data['webpage_url']}", orjson.dumps(data), ex=1800
        )

    async def _play(
        self, fake_redis: aioredis.Redis, webpage_url: str, author: Any, **kwargs: Any
    ) -> YTDL:
        qobj = QueueObject(webpage_url, "Laddered Song", author)
        with patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init):
            return await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis, **kwargs
            )

    async def test_a_dead_head_falls_sideways_without_re_extracting(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The headline behavior: the second rung plays, and yt-dlp is never called."""
        url = "https://yt.com/v=sideways"
        await self._cache(fake_redis, self._laddered(url, "251", "140", "249"))
        playable_urls.side_effect = [False, True]

        with patch("src.youtube._ytdlp_extract") as mock_extract:
            song = await self._play(fake_redis, url, mock_ctx.author)

        mock_extract.assert_not_called()
        assert song.data.get("format_id") == "140"
        assert song.url == song.data["url"]

    async def test_a_deprioritized_promotion_still_rewrites_the_ladder(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The retry path's cache coherence, and it is the COMMON path rather than an
        edge case: _retry_failed_stream invalidates the entry first, so every retry
        resolves fresh, walks with the failed format deprioritized, and re-caches.

        The reorder means the winner can land at index 0 while still differing from
        the format the entry was built around — so a rewrite gated on a non-zero
        index would persist `format_id: 140` over a ladder still headed by 251, and
        the next play would probe the blacklisted rung first, discard what the retry
        learned, and warn about the wrong format until the TTL lapsed.
        """
        url = "https://yt.com/v=retryladder"
        playable_urls.side_effect = [True]  # 140, walked first, is healthy

        qobj = QueueObject(
            url,
            "Laddered Song",
            mock_ctx.author,
            failed_format_ids=frozenset({"251"}),
        )
        with (
            patch(
                "src.youtube._ytdlp_extract",
                return_value=self._laddered(url, "251", "140", "249"),
            ),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            song = await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert song.data.get("format_id") == "140"
        raw = await fake_redis.get(f"ytdl:stream:{url}")
        assert raw is not None, "the promotion must have re-cached the entry"
        cached = orjson.loads(cast(bytes, raw))
        assert cached["format_id"] == "140"
        assert cached["audio_candidates"][0]["format_id"] == "140", (
            "the cached ladder head must be the URL the probe validated"
        )
        # 251 is not dropped — it is demoted, since the usual cause is a URL revoked
        # between probe and first read, which a fresh extraction cures.
        assert "251" in [c["format_id"] for c in cached["audio_candidates"]]

    async def test_the_winner_is_promoted_wholesale(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Not just the URL: the whole format shape, so _record_serving_format and
        YTDL's abr/asr/acodec describe what is actually playing."""
        url = "https://yt.com/v=promote"
        data = self._laddered(url, "251", "140")
        cast(dict[str, Any], data)["audio_candidates"][1].update(
            acodec="mp4a.40.2", abr=130.0, asr=44100
        )
        await self._cache(fake_redis, data)
        playable_urls.side_effect = [False, True]

        song = await self._play(fake_redis, url, mock_ctx.author)

        assert song.acodec == "mp4a.40.2"
        assert song.asr == 44100

    async def test_a_promotion_rewrites_the_cache_entry(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Without the rewrite, the next play re-probes the same dead URL every time
        until the TTL lapses — re-diagnosing a failure already known."""
        url = "https://yt.com/v=rewrite"
        await self._cache(fake_redis, self._laddered(url, "251", "140", "249"))
        playable_urls.side_effect = [False, True]

        await self._play(fake_redis, url, mock_ctx.author)

        raw = await fake_redis.get(f"ytdl:stream:{url}")
        assert raw is not None
        cached = orjson.loads(raw)
        assert cached["format_id"] == "140"
        # The dead rung ahead of the winner is gone; the ones behind it survive.
        assert [c["format_id"] for c in cached["audio_candidates"]] == ["140", "249"]

    async def test_a_healthy_head_rewrites_nothing(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """The common path stays one probe and zero writes."""
        url = "https://yt.com/v=healthy"
        await self._cache(fake_redis, self._laddered(url, "251", "140"))
        playable_urls.return_value = True

        with patch("src.youtube.cache_set") as mock_set:
            song = await self._play(fake_redis, url, mock_ctx.author)

        mock_set.assert_not_called()
        assert playable_urls.await_count == 1
        assert song.data.get("format_id") == "251"

    async def test_a_whole_dead_ladder_falls_back_to_re_extraction(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Sideways first, in place second: only when every rung is dead does the
        entry get dropped and re-extracted."""
        url = "https://yt.com/v=all_dead"
        await self._cache(fake_redis, self._laddered(url, "251", "140"))
        fresh = self._laddered(url, "251")
        cast(dict[str, Any], fresh)["title"] = "Fresh Song"
        playable_urls.side_effect = [False, False, True]

        with patch("src.youtube._ytdlp_extract", return_value=fresh) as mock_extract:
            song = await self._play(fake_redis, url, mock_ctx.author)

        mock_extract.assert_called_once()
        assert song.title == "Fresh Song"

    async def test_a_fresh_extraction_walks_its_own_ladder(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A cache miss gets the same treatment, and caches the rung that survived."""
        url = "https://yt.com/v=fresh_ladder"
        playable_urls.side_effect = [False, True]

        with patch(
            "src.youtube._ytdlp_extract", return_value=self._laddered(url, "251", "140")
        ):
            song = await self._play(fake_redis, url, mock_ctx.author)

        assert song.data.get("format_id") == "140"
        raw = await fake_redis.get(f"ytdl:stream:{url}")
        assert raw is not None
        assert orjson.loads(raw)["format_id"] == "140"

    async def test_a_failed_format_is_tried_last_not_skipped(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A format that just 403'd mid-play is suspect, so the retry walks the
        others first — but it is deprioritized, never removed: a single-format
        video would otherwise have nothing left to try."""
        url = "https://yt.com/v=retry"
        await self._cache(fake_redis, self._laddered(url, "251", "140"))
        playable_urls.return_value = True

        qobj = QueueObject(
            url, "Retrying Song", mock_ctx.author, failed_format_ids=frozenset({"251"})
        )
        with patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init):
            song = await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert song.data.get("format_id") == "140"

    async def test_the_only_format_is_still_tried_after_it_failed(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Filtering instead of reordering would make the retry a guaranteed
        failure here — and "revoked between the probe and ffmpeg's first read", the
        common case, is cured by a fresh URL for the same format."""
        url = "https://yt.com/v=single"
        await self._cache(fake_redis, self._laddered(url, "251"))
        playable_urls.return_value = True

        qobj = QueueObject(
            url, "Single Format", mock_ctx.author, failed_format_ids=frozenset({"251"})
        )
        with patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init):
            song = await YTDL.yt_stream(
                qobj, AsyncMock(spec=discord.TextChannel), redis=fake_redis
            )

        assert song.data.get("format_id") == "251"

    async def test_entries_cached_before_the_ladder_existed_still_play(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """Wire-compat: a pre-upgrade entry carries one URL and no candidates, which
        must read as a one-rung ladder rather than an empty one."""
        url = "https://yt.com/v=legacy"
        legacy = _fake_ytdl_data(webpage_url=url, title="Legacy Song")
        await self._cache(fake_redis, legacy)
        playable_urls.return_value = True

        song = await self._play(fake_redis, url, mock_ctx.author)

        assert song.title == "Legacy Song"
        assert song.url == legacy["url"]


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
    android_vr fell back to muxed-only (yt-dlp#16150) or web_safari is serving."""

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
        """A real android_vr outage affects every song — one warning per format, not
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


class TestOpusPassthrough:
    """Discord speaks Opus and ~90% of what YouTube serves here already IS Opus, so
    the encoder was spending a full lossy generation to arrive at the same thing.
    `codec="copy"` remuxes instead. discord.py resolves `codec` to 'copy' only for
    ('opus', 'libopus', 'copy') and to 'libopus' for everything else, INCLUDING None
    — so the gate is a plain codec string, and None is today's behavior."""

    async def _ffmpeg_args(self, **data_overrides: Any) -> tuple[Optional[str], str]:
        """The codec AND the output options yt_stream hands FFmpegOpusAudio.

        Both, together, because they encode one invariant across two code paths:
        ffmpeg REFUSES `-c:a copy` alongside any filtergraph. Capturing only the
        codec is what let a mutation that always appends the volume filter survive
        the whole suite.
        """
        captured: dict[str, Any] = {}

        def capture_init(
            self: Any,
            url: str,
            *,
            codec: Optional[str] = None,
            options: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            noop_ffmpeg_init(self)
            captured["codec"] = codec
            captured["options"] = options or ""

        volume = data_overrides.pop("volume", 1.0)
        # A remuxable YouTube serve, spelled out: opus, stereo, and one of the itags
        # known to be 20ms-framed. Every clause of the gate is defeatable per test.
        data: dict[str, Any] = {
            "acodec": "opus",
            "audio_channels": 2,
            "format_id": "251",
        }
        data.update(data_overrides)
        with (
            patch("src.youtube._ytdlp_extract", return_value=_fake_ytdl_data(**data)),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(
                QueueObject("https://yt.com/v=x", "Song", MagicMock()),
                AsyncMock(spec=discord.TextChannel),
                volume=volume,
            )
        return captured["codec"], captured["options"]

    async def _codec(self, **data_overrides: Any) -> Optional[str]:
        """The codec yt_stream hands FFmpegOpusAudio for this serve."""
        codec, _ = await self._ffmpeg_args(**data_overrides)
        return codec

    async def test_an_opus_serve_is_remuxed(self) -> None:
        assert await self._codec() == "copy"

    async def test_an_aac_serve_is_still_encoded(self) -> None:
        """The muxed fallback rungs (itag 18, HLS 91-96) carry AAC."""
        assert await self._codec(acodec="mp4a.40.2", format_id="18") is None

    async def test_a_volume_filter_forces_the_encoder(self) -> None:
        """A filter has to touch samples, so it cannot ride the copy path."""
        assert await self._codec(volume=0.5) is None

    async def test_a_missing_codec_is_not_assumed_to_be_opus(self) -> None:
        """Pre-upgrade cache entries and extractors that report nothing must fall to
        the encoder, not be guessed into a remux that ffmpeg would then refuse."""
        assert await self._codec(acodec=None) is None

    async def test_a_surround_serve_is_never_remuxed(self) -> None:
        """`-c:a copy` copies OpusHead too, so a 5.1 stream reaches Discord as
        6-channel multistream and clients decode only the front pair — centre-channel
        vocals silently vanish. yt-dlp sorts `channels` ABOVE `acodec`, so bestaudio
        really does select itag 338 on videos that carry it; `-ac 2` used to downmix.
        """
        assert await self._codec(audio_channels=6, format_id="338") is None

    async def test_an_unknown_channel_count_is_not_assumed_to_be_stereo(self) -> None:
        """Not every extractor populates audio_channels, and the safe reading of
        absent is "re-encode" — the encoder path is correct for every input."""
        assert await self._codec(audio_channels=None) is None

    async def test_a_mono_serve_is_remuxed(self) -> None:
        """Mono stays mono rather than being upmixed by `-ac 2`, which is the honest
        rendering of the source and decodes correctly at the client."""
        assert await self._codec(audio_channels=1) == "copy"

    async def test_an_opus_format_outside_the_allowlist_is_encoded(self) -> None:
        """Packet duration is the reason. `read()` counts packets and every position
        surface is frames x 20ms, but Opus may legally be 60ms-framed — which plays
        at 3x speed and reads a third of its true position. yt-dlp reports no frame
        duration, so anything but YouTube's known-20ms itags takes the encoder.
        SoundCloud's http_opus is exactly this case.
        """
        assert await self._codec(format_id="http_opus_0_0") is None

    async def test_the_volume_filter_and_the_copy_codec_are_mutually_exclusive(
        self,
    ) -> None:
        """The invariant, asserted directly rather than inferred from two separate
        tests: ffmpeg exits 234 with zero bytes on `-c:a copy` plus `-filter:a`, and
        the player would blame YouTube and burn the whole retry budget for it.
        """
        for volume in (1.0, 0.5, 2.0):
            codec, options = await self._ffmpeg_args(volume=volume)
            assert codec != "copy" or "-filter:a" not in options, (
                f"volume={volume} produced codec={codec!r} options={options!r}"
            )


class TestYTStreamCarriedFields:
    """A playing song becomes a QueueObject again — a neutralized prefetch, an
    interjection's resume tail, a stream retry — so anything QueueObject carries and
    YTDL does not is silently dropped at that rebuild. `user_input` and `persisted`
    have each been lost this way before."""

    async def _stream(self, qobj: QueueObject) -> YTDL:
        with (
            patch("src.youtube._ytdlp_extract", return_value=_fake_ytdl_data()),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init),
        ):
            return await YTDL.yt_stream(qobj, AsyncMock(spec=discord.TextChannel))

    async def test_user_input_survives(self, mock_ctx: MagicMock) -> None:
        """The only surviving record of the collection link -remove matches on: a
        search entry's ytsearch is a title this code generated."""
        song = await self._stream(
            QueueObject(
                "https://www.youtube.com/watch?v=test",
                "Test Song",
                mock_ctx.author,
                user_input="https://open.spotify.com/album/abc",
            )
        )
        assert song.user_input == "https://open.spotify.com/album/abc"

    async def test_persisted_survives(self, mock_ctx: MagicMock) -> None:
        """A crash-recovered song was never RPUSHed to the Redis list. Rebuilt as
        persisted=True, its next dequeue LPOPs an entry that belongs to an unrelated
        still-queued song — deleting it, with no error."""
        song = await self._stream(
            QueueObject(
                "https://www.youtube.com/watch?v=test",
                "Test Song",
                mock_ctx.author,
                persisted=False,
            )
        )
        assert song.persisted is False

    async def test_retry_state_survives(self, mock_ctx: MagicMock) -> None:
        """The loop reads both off the PLAYING song to decide whether to retry."""
        song = await self._stream(
            QueueObject(
                "https://www.youtube.com/watch?v=test",
                "Test Song",
                mock_ctx.author,
                stream_attempts=2,
                failed_format_ids=frozenset({"251"}),
            )
        )
        assert song.stream_attempts == 2
        assert song.failed_format_ids == frozenset({"251"})


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
            codec: Optional[str] = None,
        ) -> None:
            noop_ffmpeg_init(self)
            captured_options["before_options"] = before_options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel)

        channel.send.assert_not_awaited()
        assert "-ss 151" in captured_options["before_options"]


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
        "audio_channels": 2,
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


def _audio_ladder() -> list[dict[str, Any]]:
    """A YouTube format list shaped the way a real one is: sorted worst→best, and
    carrying every kind of entry a naive audio filter would wrongly admit —
    storyboards (audio-less but `vcodec: none`), a DRC variant, a dubbed track, and
    the video-only rungs."""
    return [
        {"format_id": "sb0", "url": "https://sb", "vcodec": "none", "acodec": "none"},
        _fmt("139", "mp4a.40.5", 49.0, 22050),
        _fmt("249", "opus", 46.0, 48000),
        _fmt("251-drc", "opus", 129.0, 48000),
        _fmt("251-1", "opus", 129.0, 48000, language="es"),
        _fmt("140", "mp4a.40.2", 130.0, 44100),
        _fmt("251", "opus", 129.0, 48000, language="en"),
        {"format_id": "137", "url": "https://137", "vcodec": "avc1", "acodec": "none"},
    ]


def _fmt(
    format_id: str, acodec: str, abr: float, asr: int, **extra: Any
) -> dict[str, Any]:
    """One audio-only format, with the expire param _stream_url_ttl needs to consider
    the URL cacheable."""
    return {
        "format_id": format_id,
        "url": f"https://r2.googlevideo.com/{format_id}?expire={int(time.time()) + 7200}",
        "vcodec": "none",
        "acodec": acodec,
        "abr": abr,
        "asr": asr,
        "protocol": "https",
        "audio_channels": 2,
        **extra,
    }


class TestAudioCandidateMining:
    """The fallback ladder is mined in the worker because _slim_info drops `formats` —
    after that the alternative URLs exist nowhere, which is why one dead URL used to
    cost the whole song."""

    def _mine(self, **overrides: Any) -> list[dict[str, Any]]:
        fields: dict[str, Any] = {
            "formats": _audio_ladder(),
            "format_id": "251",
            "language": "en",
            **overrides,
        }
        return cast(
            list[dict[str, Any]], _mine_audio_candidates(_realistic_raw_info(**fields))
        )

    def test_keeps_the_top_rungs_best_first(self) -> None:
        """yt-dlp sorts worst→best, so the ladder is walked in reverse: the selected
        format leads, then the next-best real audio formats."""
        assert [c["format_id"] for c in self._mine()] == ["251", "140", "249"]

    def test_candidate_zero_is_the_selected_format(self) -> None:
        """Candidate 0 and the URL the probe already validated must be the same
        thing, or a promotion would be recorded for a URL nothing moved to."""
        candidates = self._mine()
        assert candidates[0]["format_id"] == "251"
        assert candidates[0]["acodec"] == "opus"

    def test_selection_leads_even_when_the_sort_disagrees(self) -> None:
        """Selection wins over sort order, and never appears twice."""
        candidates = self._mine(format_id="140")
        assert [c["format_id"] for c in candidates] == ["140", "251", "249"]

    def test_storyboards_and_drc_and_dubs_are_excluded(self) -> None:
        """Storyboards carry `vcodec: none` too (the acodec test is what removes
        them); a `-drc` variant or a foreign-language dub would silently change what
        the song sounds like."""
        ids = [c["format_id"] for c in self._mine()]
        assert "sb0" not in ids and "137" not in ids
        assert "251-drc" not in ids and "251-1" not in ids

    def test_a_storyboard_is_excluded_by_the_filter_and_not_by_the_cap(self) -> None:
        """Deliberately fewer real audio formats than _STREAM_CANDIDATES, so the slice
        cannot be what drops the storyboard — only the acodec test can.

        yt-dlp writes the STRING "none" here, not None, so narrowing that check to an
        identity test against None removes the only branch that ever fires in
        production. A storyboard promoted as audio would probe 200 (it is a real
        JPEG mosaic) and be served to the voice channel as the song.
        """
        candidates = self._mine(
            formats=[
                {
                    "format_id": "sb0",
                    "url": "https://sb0",
                    "vcodec": "none",
                    "acodec": "none",
                },
                {
                    "format_id": "sb1",
                    "url": "https://sb1",
                    "vcodec": "none",
                    "acodec": "none",
                },
                _fmt("251", "opus", 129.0, 48000),
            ],
            language=None,
        )
        assert [c["format_id"] for c in candidates] == ["251"]
        assert len(candidates) < _STREAM_CANDIDATES

    def test_language_less_formats_survive_a_language_selection(self) -> None:
        """Absent language means "the only track", not "a different one"."""
        assert "140" in [c["format_id"] for c in self._mine()]

    def test_muxed_selection_gets_no_ladder(self) -> None:
        """The muxed rung already means the audio-only path is degraded; walking
        sideways across muxed formats is not a recovery worth having."""
        candidates = self._mine(format_id="18", vcodec="avc1.42001E")
        assert len(candidates) == 1
        assert candidates[0]["format_id"] == "18"

    def test_no_format_list_mines_nothing(self) -> None:
        """extract_flat playlist entries carry no formats and no stream URL."""
        assert _mine_audio_candidates({"url": "https://x", "title": "t"}) == []
        assert _mine_audio_candidates({"formats": _audio_ladder()}) == []

    def test_candidates_are_capped(self) -> None:
        assert len(self._mine()) <= _STREAM_CANDIDATES

    def test_only_the_candidate_fields_are_carried(self) -> None:
        """A candidate promotes wholesale onto the info-dict, so a stray key here
        would overwrite a top-level field with a per-format one. Every candidate field
        is also a cached field, or a promotion would not survive the round trip."""
        assert set(self._mine()[0]) == set(_CANDIDATE_FIELDS)
        assert set(_CANDIDATE_FIELDS) <= set(_STREAM_CACHE_FIELDS)


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

    def test_slimming_mines_the_audio_ladder_before_dropping_formats(self) -> None:
        """The ONE integration point of the whole retry ladder.

        `formats` is dropped in the same pass, so if the ladder is not attached here
        it exists nowhere afterwards and every fallback URL is gone for good. Nothing
        else in the suite covers it — the mining tests hand-build `audio_candidates`
        into a fixture, and every extraction test patches out _slim_info's caller — so
        without this the entire feature can be deleted with the suite still green.
        """
        raw = _realistic_raw_info(
            formats=_audio_ladder(), format_id="251", language="en"
        )
        slim = cast(dict[str, Any], _slim_info(raw))
        assert [c["format_id"] for c in slim["audio_candidates"]] == [
            "251",
            "140",
            "249",
        ]
        assert "formats" not in slim
        # Cheap enough to ship, and picklable — it crosses the boundary every time.
        pickle.loads(pickle.dumps(slim))

    def test_each_search_entry_is_mined_too(self) -> None:
        """yt_source narrows to an entry, so an unmined one reaches playback with no
        ladder and silently falls back to single-URL behaviour."""
        wrapper = {
            "_type": "playlist",
            "entries": [
                _realistic_raw_info(
                    formats=_audio_ladder(), format_id="251", language="en"
                )
            ],
        }
        slim = cast(dict[str, Any], _slim_info(wrapper))
        entry = slim["entries"][0]
        assert [c["format_id"] for c in entry["audio_candidates"]] == [
            "251",
            "140",
            "249",
        ]
        assert "formats" not in entry

    def test_an_entry_with_no_formats_gets_no_ladder_key(self) -> None:
        """extract_flat playlist entries carry no `formats`, and must not gain an
        empty key — _candidate_ladder's legacy path is what serves them."""
        slim = cast(dict[str, Any], _slim_info({"url": "https://x", "title": "t"}))
        assert "audio_candidates" not in slim

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
