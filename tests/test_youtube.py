"""Tests for src/youtube.py — QueueObject, YTDL config, yt_source, yt_stream, and stream cache."""

import redis.asyncio as aioredis
import pickle
import time
from typing import Any, Optional, cast
from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from redis.asyncio import Redis

from src.telemetry import configure_worker_logging
from src.youtube import (
    YTDL,
    YTDL_OPTS,
    QueueObject,
    _DEGRADED_FORMAT_WARNED,
    _STREAM_CACHE_FIELDS,
    _YTDL_PLAYLIST_OPTS,
    _YTDL_STREAM_OPTS,
    _YTDL_STREAM_SEARCH_OPTS,
    _enrich_queueobject,
    _record_serving_format,
    _stream_url_playable,
    _stream_url_ttl,
    _ytdlp_extract,
    _YtdlpLogger,
    YTDLVideoInfo,
    YTDLVideoMetadata,
)
from tests.helpers import noop_ffmpeg_init


@pytest.fixture(autouse=True)
def _suppress_ytdl_del(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch discord.AudioSource.__del__ to a no-op for every test in this module.

    YTDL tests patch discord.FFmpegOpusAudio.__init__ to return_value=None so that
    no real FFmpeg process is spawned. This leaves _process unset on the instance.
    When Python GC collects the object, AudioSource.__del__ → FFmpegAudio.cleanup()
    → _kill_process() → _check_process_returncode() accesses self._process and raises
    AttributeError. Suppressing __del__ here avoids that crash without touching
    production code.
    """
    monkeypatch.setattr(discord.AudioSource, "__del__", lambda self: None)


@pytest.fixture(autouse=True)
def playable_urls(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Treat every stream URL as playable unless a test says otherwise.

    yt_stream() probes each URL before handing it to ffmpeg. Left unpatched that is a
    real HTTP call to a fake googlevideo host in every test. Tests that exercise the
    revocation path set the returned mock's return_value to False.
    """
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
    """Elapsed-time tracking via YTDL.read() call counting — see Design §1 of
    docs/NOW_PLAYING_PROGRESS_BAR_PLAN.md. Deterministic call counting, no
    time-mocking needed: patches the parent FFmpegOpusAudio.read() (which
    super().read() resolves to) directly rather than relying on the real
    _packet_iter, since noop_ffmpeg_init doesn't set that up.
    """

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
        # default_search belongs to yt_source's unified search opts, not the stream opts
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
        play path (docs/PERFORMANCE_PLAN.md §2.1) populates the ytdl:stream cache from
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
            result = await YTDL.yt_source(mock_ctx.author, "ytsearch:test song")

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
            result = await YTDL.yt_source(mock_ctx.author, "ytsearch:test song")
        assert result.thumbnail == "https://img.yt.com/test123.jpg"

    async def test_yt_source_raises_when_no_data(self, mock_ctx: MagicMock) -> None:
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = None
            with pytest.raises(Exception, match="Could not find song"):
                await YTDL.yt_source(mock_ctx.author, "ytsearch:nothing")

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
            result = await YTDL.yt_source(mock_ctx.author, "ytsearch:test")

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
            result = await YTDL.yt_source(mock_ctx.author, "ytsearch:test")

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
            result = await YTDL.yt_source(mock_ctx.author, "my search query")
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
            mock_ctx.author, "cached search", redis=fake_redis
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
            mock_ctx.author, "cached search", redis=fake_redis
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
            await YTDL.yt_source(mock_ctx.author, "some search", redis=fake_redis)

        result = await YTDL.yt_source(mock_ctx.author, "some search", redis=fake_redis)
        assert result.thumbnail == "https://img.yt.com/fresh.jpg"

    async def test_yt_source_passes_timestamp(self, mock_ctx: MagicMock) -> None:
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=ts_test",
            "title": "Timestamped Song",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author, "https://yt.com/watch?v=ts_test", ts=45
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
                mock_ctx.author, "https://yt.com/v=dl", download=True
            )
        # download=True is passed as the 3rd positional arg to _ytdlp_extract
        call_args = mock_extract.call_args[0]
        assert call_args[2] is True
        assert result.title == "Download Song"


class TestYTSourceUnifiedExtraction:
    """The unified single-extraction play path (docs/PERFORMANCE_PLAN.md §2.1):
    one stream-opts yt-dlp call populates BOTH the ytdl:source and ytdl:stream
    caches, making queue_put's prefetch_stream a cache-hit no-op instead of a
    second YouTube extraction."""

    async def test_always_extracts_with_process_true(self, mock_ctx: MagicMock) -> None:
        """process=True is hardcoded — the §2.1 trap. Direct URLs used to flow with
        process=False, and an unprocessed extract_info performs no format selection,
        so data["url"] would be absent and the stream-cache write would silently
        never happen for direct-URL plays."""
        fake_data = _fake_ytdl_data()
        with patch(
            "src.youtube._ytdlp_extract", return_value=fake_data
        ) as mock_extract:
            await YTDL.yt_source(mock_ctx.author, "https://yt.com/watch?v=direct")
        opts, process = mock_extract.call_args[0][1], mock_extract.call_args[0][3]
        assert opts is _YTDL_STREAM_SEARCH_OPTS
        assert process is True

    async def test_fresh_extraction_writes_both_caches(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """One cold yt_source call must leave both a ytdl:source and a ytdl:stream
        entry behind — the absence of the stream key means the second extraction
        is back."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni1")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            await YTDL.yt_source(mock_ctx.author, "unified search", redis=fake_redis)

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
        """prefetch_stream must not re-extract a song yt_source just resolved —
        the whole point of §2.1 is that the enqueue-time prefetch becomes one
        Redis GET."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni2")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            qobj = await YTDL.yt_source(
                mock_ctx.author, "prefetch noop search", redis=fake_redis
            )
        with patch("src.youtube._ytdlp_extract") as mock_extract:
            await YTDL.prefetch_stream(qobj, redis=fake_redis)
        mock_extract.assert_not_called()

    async def test_dead_probe_skips_stream_cache_but_returns_qobj(
        self, mock_ctx: MagicMock, fake_redis: aioredis.Redis, playable_urls: AsyncMock
    ) -> None:
        """A failed probe never fails yt_source: the song enqueues on identity
        alone (source cache written), and dequeue-time re-extraction handles the
        stream — exactly the pre-§2.1 behavior."""
        playable_urls.return_value = False
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni3")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            result = await YTDL.yt_source(
                mock_ctx.author, "dead probe search", redis=fake_redis
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
            await YTDL.yt_source(mock_ctx.author, "no redis search")
        playable_urls.assert_not_awaited()

    async def test_fresh_extraction_populates_full_metadata(
        self, mock_ctx: MagicMock
    ) -> None:
        """The unified extraction is a full one — duration/uploader/thumbnail come
        back on the first call, no prefetch enrichment needed."""
        fake_data = _fake_ytdl_data(webpage_url="https://yt.com/v=uni4")
        with patch("src.youtube._ytdlp_extract", return_value=fake_data):
            result = await YTDL.yt_source(mock_ctx.author, "metadata search")
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


class TestStreamUrlTtl:
    def test_caps_ttl_regardless_of_expire(self) -> None:
        """A URL claiming hours of life is still only cached for the cap.

        YouTube revokes stream URLs long before their `expire`; trusting it meant a
        revoked URL was replayed for hours and the song failed every time.
        """
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
        """Load-bearing: a revoked URL still answers 206 to a *ranged* GET while refusing
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
        """prefetch_stream does not propagate yt-dlp exceptions."""
        qobj = QueueObject("https://yt.com/v=pf4", "Error Song", mock_ctx.author)
        with patch(
            "src.youtube._ytdlp_extract", side_effect=Exception("network error")
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

    async def test_resume_entry_suppresses_ts_notice_but_keeps_seek(
        self, mock_ctx: MagicMock
    ) -> None:
        """Prefetch constructs resume entries mid-interjection — the
        construction-time notice would fire at the wrong moment, so the loop
        announces the resume instead. The -ss seek itself must remain."""
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test",
            "Test Song",
            mock_ctx.author,
            ts=151,
            is_resume=True,
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

    async def test_plain_ts_entry_still_sends_notice(self, mock_ctx: MagicMock) -> None:
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
            await YTDL.yt_stream(qobj, channel)

        channel.send.assert_awaited_once()


class TestProcessBoundaryContract:
    """Everything that must survive being pickled to a worker process.

    The suite runs extraction on an in-process ThreadPoolExecutor (see conftest), so
    nothing else here ever exercises the pickling that the production
    ProcessPoolExecutor performs on every submit. These tests are the cheap half of
    that coverage gap — they assert the contract directly, in microseconds, without
    spawning anything. The expensive half (a real worker actually spawning) is
    docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §7.

    What they catch: adding an unpicklable value to an opts profile — a logger
    instance, a session, a lambda, a compiled callback — or turning a submitted
    top-level function into a closure/lambda/bound method. Either breaks every
    extraction in production the moment it ships, while a green suite says nothing.
    """

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

        Round-tripped rather than merely dumped: a value that serialises but does not
        reconstruct (a class whose module moved, say) fails only in the worker, where
        the failure surfaces as an opaque BrokenProcessPool.
        """
        restored = pickle.loads(pickle.dumps(opts))
        assert restored.keys() == opts.keys(), f"{name} profile lost keys"

    def test_extract_worker_is_picklable_by_reference(self) -> None:
        """_ytdlp_extract is pickled by qualified name, not by value — so it must stay
        a module-level function. `is` rather than `==`: pickle resolves the name on the
        far side, and only a real module-level lookup round-trips to the same object."""
        assert pickle.loads(pickle.dumps(_ytdlp_extract)) is _ytdlp_extract

    def test_worker_logging_initializer_is_picklable_by_reference(self) -> None:
        """ProcessPoolExecutor pickles `initializer` to every worker. A closure or a
        bound method here breaks pool construction rather than one extraction."""
        assert (
            pickle.loads(pickle.dumps(configure_worker_logging))
            is configure_worker_logging
        )
