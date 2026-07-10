"""Tests for src/youtube.py — QueueObject, YTDL config, yt_source, yt_stream, and stream cache."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from src.youtube import (
    YTDL,
    YTDL_OPTS,
    QueueObject,
    _YTDL_SOURCE_OPTS,
    _YTDL_STREAM_OPTS,
    _enrich_queueobject,
    _stream_url_ttl,
)
from tests.helpers import noop_ffmpeg_init


@pytest.fixture(autouse=True)
def _suppress_ytdl_del(monkeypatch):
    """Patch discord.AudioSource.__del__ to a no-op for every test in this module.

    YTDL tests patch discord.FFmpegOpusAudio.__init__ to return_value=None so that
    no real FFmpeg process is spawned. This leaves _process unset on the instance.
    When Python GC collects the object, AudioSource.__del__ → FFmpegAudio.cleanup()
    → _kill_process() → _check_process_returncode() accesses self._process and raises
    AttributeError. Suppressing __del__ here avoids that crash without touching
    production code.
    """
    monkeypatch.setattr(discord.AudioSource, "__del__", lambda self: None)


def _fake_ytdl_data(**overrides):
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
    return base


class TestYTDLGetItem:
    def test_getitem_returns_attribute(self, ytdl_instance):
        song = ytdl_instance()
        assert song["title"] == "Test Song"
        assert song["webpage_url"] == "https://www.youtube.com/watch?v=test"

    def test_getitem_returns_uploader(self, ytdl_instance):
        song = ytdl_instance()
        assert song["uploader"] == "Test Channel"


class TestYTDLElapsedSecs:
    """Elapsed-time tracking via YTDL.read() call counting — see Design §1 of
    docs/NOW_PLAYING_PROGRESS_BAR_PLAN.md. Deterministic call counting, no
    time-mocking needed: patches the parent FFmpegOpusAudio.read() (which
    super().read() resolves to) directly rather than relying on the real
    _packet_iter, since noop_ffmpeg_init doesn't set that up.
    """

    def test_zero_before_any_read(self, ytdl_instance):
        song = ytdl_instance()
        assert song.elapsed_secs == 0.0

    def test_increments_by_20ms_per_frame_with_data(self, ytdl_instance):
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            song.read()
        assert song.elapsed_secs == pytest.approx(0.02)

    def test_accumulates_across_multiple_reads(self, ytdl_instance):
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            for _ in range(5):
                song.read()
        assert song.elapsed_secs == pytest.approx(0.10)

    def test_does_not_increment_on_empty_read(self, ytdl_instance):
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b""):
            song.read()
            song.read()
        assert song.elapsed_secs == 0.0

    def test_read_returns_underlying_data(self, ytdl_instance):
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            assert song.read() == b"opus-frame"


class TestYTDLPositionSecs:
    """position_secs = start_offset + elapsed_secs — the single source of
    truth for every position surface (progress bar, Activity presence, pause
    confirmation), so a -ss/?t= song can't report different positions in
    different places."""

    def test_equals_elapsed_when_no_offset(self, ytdl_instance):
        song = ytdl_instance()
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            for _ in range(5):
                song.read()
        assert song.position_secs == song.elapsed_secs == pytest.approx(0.10)

    def test_includes_start_offset(self, ytdl_instance):
        song = ytdl_instance()
        song.start_offset = 90
        with patch.object(discord.FFmpegOpusAudio, "read", return_value=b"opus-frame"):
            for _ in range(5):
                song.read()
        assert song.position_secs == pytest.approx(90.10)

    def test_offset_only_before_any_read(self, ytdl_instance):
        song = ytdl_instance()
        song.start_offset = 90
        assert song.position_secs == 90.0


class TestQueueObject:
    def test_required_fields(self, mock_author):
        qobj = QueueObject(
            webpage_url="https://www.youtube.com/watch?v=abc",
            title="My Song",
            requester=mock_author,
        )
        assert qobj.webpage_url == "https://www.youtube.com/watch?v=abc"
        assert qobj.title == "My Song"
        assert qobj.requester is mock_author

    def test_ts_defaults_to_none(self, mock_author):
        qobj = QueueObject("https://yt.com/watch?v=1", "Title", mock_author)
        assert qobj.ts is None

    def test_ts_can_be_set(self, mock_author):
        qobj = QueueObject("https://yt.com/watch?v=1", "Title", mock_author, ts=90)
        assert qobj.ts == 90

    def test_optional_fields_default_to_none(self, mock_author):
        qobj = QueueObject("https://yt.com/watch?v=1", "Title", mock_author)
        assert qobj.user_input is None
        assert qobj.duration is None
        assert qobj.uploader is None

    def test_optional_fields_can_be_set(self, mock_author):
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

    def test_is_dataclass(self, mock_author):
        import dataclasses

        assert dataclasses.is_dataclass(QueueObject)

    def test_equality(self, mock_author):
        q1 = QueueObject("https://yt.com/watch?v=1", "Song", mock_author)
        q2 = QueueObject("https://yt.com/watch?v=1", "Song", mock_author)
        assert q1 == q2

    def test_inequality_different_url(self, mock_author):
        q1 = QueueObject("https://yt.com/watch?v=1", "Song", mock_author)
        q2 = QueueObject("https://yt.com/watch?v=2", "Song", mock_author)
        assert q1 != q2


class TestEnrichQueueObject:
    def test_sets_duration_when_none(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"duration": 180, "uploader": "Chan"})
        assert qobj.duration == 180

    def test_does_not_overwrite_existing_duration(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author, duration=120)
        _enrich_queueobject(qobj, {"duration": 999, "uploader": "Chan"})
        assert qobj.duration == 120

    def test_sets_uploader_when_none(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"uploader": "My Channel"})
        assert qobj.uploader == "My Channel"

    def test_does_not_overwrite_existing_uploader(self, mock_author):
        qobj = QueueObject(
            "https://yt.com/v=1", "Song", mock_author, uploader="Original"
        )
        _enrich_queueobject(qobj, {"uploader": "New Channel"})
        assert qobj.uploader == "Original"

    def test_handles_missing_keys_gracefully(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {})
        assert qobj.duration is None
        assert qobj.uploader is None
        assert qobj.thumbnail is None

    def test_sets_thumbnail_when_none(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"thumbnail": "https://img.yt.com/x.jpg"})
        assert qobj.thumbnail == "https://img.yt.com/x.jpg"

    def test_does_not_overwrite_existing_thumbnail(self, mock_author):
        qobj = QueueObject(
            "https://yt.com/v=1",
            "Song",
            mock_author,
            thumbnail="https://img.yt.com/original.jpg",
        )
        _enrich_queueobject(qobj, {"thumbnail": "https://img.yt.com/new.jpg"})
        assert qobj.thumbnail == "https://img.yt.com/original.jpg"

    def test_duration_cast_to_int(self, mock_author):
        qobj = QueueObject("https://yt.com/v=1", "Song", mock_author)
        _enrich_queueobject(qobj, {"duration": 180.7})
        assert qobj.duration == 180
        assert isinstance(qobj.duration, int)


class TestYTDLOpts:
    def test_format_is_bestaudio(self):
        assert YTDL_OPTS["format"] == "bestaudio/best"

    def test_noplaylist_is_true(self):
        assert YTDL_OPTS["noplaylist"] is True

    def test_source_address_is_ipv4_any(self):
        assert YTDL_OPTS["source_address"] == "0.0.0.0"

    def test_default_search_is_auto(self):
        # default_search belongs to the source (search) opts, not the stream opts
        assert _YTDL_SOURCE_OPTS["default_search"] == "auto"

    def test_retries_is_set(self):
        assert YTDL_OPTS["retries"] > 0

    def test_socket_timeout_is_set(self):
        assert YTDL_OPTS["socket_timeout"] > 0

    def test_extractor_args_include_youtube(self):
        assert "youtube" in YTDL_OPTS["extractor_args"]

    def test_stream_opts_have_format(self):
        assert _YTDL_STREAM_OPTS["format"] == "bestaudio/best"

    def test_source_opts_no_format(self):
        # yt_source only needs metadata; format resolution is deferred to yt_stream
        assert "format" not in _YTDL_SOURCE_OPTS

    def test_no_verbose_or_rm_cachedir(self):
        for opts in (_YTDL_SOURCE_OPTS, _YTDL_STREAM_OPTS):
            assert not opts.get("verbose")
            assert not opts.get("rm_cachedir")


class TestYTDLFfmpegOpts:
    def test_before_options_has_reconnect_flag(self):
        assert "-reconnect" in YTDL.FFMPEG_OPTS["before_options"]

    def test_options_strips_video(self):
        assert "-vn" in YTDL.FFMPEG_OPTS["options"]


class TestYTSource:
    async def test_yt_source_returns_queue_object(self, mock_ctx):
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Extracted Title",
        }

        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author, "ytsearch:test song", process=True
            )

        assert isinstance(result, QueueObject)
        assert result.title == "Extracted Title"
        assert result.webpage_url == "https://www.youtube.com/watch?v=test123"
        assert result.requester is mock_ctx.author

    async def test_yt_source_sets_thumbnail_fresh_extraction(self, mock_ctx):
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Extracted Title",
            "thumbnail": "https://img.yt.com/test123.jpg",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author, "ytsearch:test song", process=True
            )
        assert result.thumbnail == "https://img.yt.com/test123.jpg"

    async def test_yt_source_raises_when_no_data(self, mock_ctx):
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = None
            with pytest.raises(Exception, match="Could not find song"):
                await YTDL.yt_source(mock_ctx.author, "ytsearch:nothing", process=True)

    async def test_yt_source_picks_first_entry_from_playlist(self, mock_ctx):
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
                mock_ctx.author, "ytsearch:test", process=True
            )

        assert result.title == "Entry One"
        assert "entry1" in result.webpage_url

    async def test_yt_source_skips_playlist_type_entries(self, mock_ctx):
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
                mock_ctx.author, "ytsearch:test", process=True
            )

        assert result.title == "Real Video"

    async def test_yt_source_sets_user_input_fresh_extraction(self, mock_ctx):
        """user_input is set to the search string on fresh extraction."""
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Song Title",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author, "my search query", process=True
            )
        assert result.user_input == "my search query"

    async def test_yt_source_sets_user_input_cache_hit(self, mock_ctx, fake_redis):
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
            mock_ctx.author, "cached search", process=True, redis=fake_redis
        )
        assert result.user_input == "cached search"

    async def test_yt_source_sets_thumbnail_cache_hit(self, mock_ctx, fake_redis):
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
            mock_ctx.author, "cached search", process=True, redis=fake_redis
        )
        assert result.thumbnail == "https://img.yt.com/cached.jpg"

    async def test_yt_source_caches_thumbnail_for_next_lookup(
        self, mock_ctx, fake_redis
    ):
        """A fresh extraction's thumbnail is written to the cache, not just returned."""
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=test123",
            "title": "Song",
            "thumbnail": "https://img.yt.com/fresh.jpg",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            await YTDL.yt_source(
                mock_ctx.author, "some search", process=True, redis=fake_redis
            )

        result = await YTDL.yt_source(
            mock_ctx.author, "some search", process=True, redis=fake_redis
        )
        assert result.thumbnail == "https://img.yt.com/fresh.jpg"

    async def test_yt_source_passes_timestamp(self, mock_ctx):
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=ts_test",
            "title": "Timestamped Song",
        }
        with patch("src.youtube.youtube_dl.YoutubeDL") as mock_cls:
            mock_cls.return_value.extract_info.return_value = fake_data
            result = await YTDL.yt_source(
                mock_ctx.author, "https://yt.com/watch?v=ts_test", process=False, ts=45
            )

        assert result.ts == 45

    async def test_yt_source_passes_download_flag(self, mock_ctx):
        fake_data = {
            "webpage_url": "https://yt.com/v=dl",
            "title": "Download Song",
        }
        with patch(
            "src.youtube._ytdlp_extract", return_value=fake_data
        ) as mock_extract:
            result = await YTDL.yt_source(
                mock_ctx.author, "https://yt.com/v=dl", process=True, download=True
            )
        # download=True is passed as the 3rd positional arg to _ytdlp_extract
        call_args = mock_extract.call_args[0]
        assert call_args[2] is True
        assert result.title == "Download Song"


class TestYTStreamRuntimeError:
    async def test_raises_when_extract_returns_none(self, mock_ctx):
        qobj = QueueObject("https://yt.com/v=none", "None Song", mock_ctx.author)
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        with patch("src.youtube._ytdlp_extract", return_value=None):
            with pytest.raises(RuntimeError, match="Could not extract stream data"):
                await YTDL.yt_stream(qobj, channel)


class TestYTStream:
    async def test_yt_stream_returns_ytdl_instance(self, mock_ctx):
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

    async def test_yt_stream_appends_volume_filter_when_not_default(self, mock_ctx):
        """volume != 1.0 must append -filter:a to ffmpeg options."""
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author
        )

        captured_options = {}

        def capture_init(self, url, *, executable, before_options, options):
            noop_ffmpeg_init(self)
            captured_options["options"] = options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel, volume=0.5)

        assert "volume=0.5" in captured_options["options"]

    async def test_yt_stream_appends_seek_when_ts_set(self, mock_ctx):
        fake_data = _fake_ytdl_data()
        channel = AsyncMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        qobj = QueueObject(
            "https://www.youtube.com/watch?v=test", "Test Song", mock_ctx.author, ts=90
        )

        captured_options = {}

        def capture_init(self, url, *, executable, before_options, options):
            noop_ffmpeg_init(self)
            captured_options["options"] = options

        with (
            patch("src.youtube._ytdlp_extract", return_value=fake_data),
            patch.object(discord.FFmpegOpusAudio, "__init__", new=capture_init),
        ):
            await YTDL.yt_stream(qobj, channel)

        assert "-ss 90" in captured_options["options"]

    async def test_yt_stream_carries_ts_as_start_offset(self, mock_ctx):
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

    async def test_yt_stream_start_offset_zero_without_ts(self, mock_ctx):
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
    def test_returns_seconds_minus_margin(self):
        future = int(time.time()) + 7200  # 2h from now
        url = f"https://r2.googlevideo.com/stream?expire={future}&other=x"
        ttl = _stream_url_ttl(url)
        assert ttl is not None
        assert 7200 - 1800 - 5 <= ttl <= 7200 - 1800 + 5

    def test_returns_none_when_no_expire_param(self):
        ttl = _stream_url_ttl("https://r2.googlevideo.com/stream?other=x")
        assert ttl is None

    def test_returns_none_when_already_expired(self):
        past = int(time.time()) - 100
        url = f"https://r2.googlevideo.com/stream?expire={past}"
        assert _stream_url_ttl(url) is None

    def test_returns_none_when_ttl_too_short(self):
        soon = int(time.time()) + 30  # 30s — below 60s threshold
        url = f"https://r2.googlevideo.com/stream?expire={soon}"
        assert _stream_url_ttl(url) is None

    def test_returns_none_on_non_numeric_expire(self):
        ttl = _stream_url_ttl("https://r2.googlevideo.com/stream?expire=notanumber")
        assert ttl is None


class TestStreamCache:
    async def test_cache_hit_skips_executor(self, mock_ctx, fake_redis):
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
        self, mock_ctx, fake_redis
    ):
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

    async def test_cache_graceful_on_redis_error(self, mock_ctx):
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


class TestPrefetchStream:
    async def test_populates_cache_on_miss(self, mock_ctx, fake_redis):
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

    async def test_no_op_when_redis_none(self, mock_ctx):
        """prefetch_stream returns immediately when redis is None — no exception."""
        qobj = QueueObject("https://yt.com/v=pf2", "No Redis", mock_ctx.author)
        with patch("src.youtube._ytdlp_extract") as mock_extract:
            await YTDL.prefetch_stream(qobj, redis=None)
        mock_extract.assert_not_called()

    async def test_no_op_when_already_cached(self, mock_ctx, fake_redis):
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

    async def test_swallows_extraction_errors(self, mock_ctx, fake_redis):
        """prefetch_stream does not propagate yt-dlp exceptions."""
        qobj = QueueObject("https://yt.com/v=pf4", "Error Song", mock_ctx.author)
        with patch(
            "src.youtube._ytdlp_extract", side_effect=Exception("network error")
        ):
            await YTDL.prefetch_stream(qobj, redis=fake_redis)
        cached = await fake_redis.get("ytdl:stream:https://yt.com/v=pf4")
        assert cached is None

    async def test_skips_write_when_ttl_too_short(self, mock_ctx, fake_redis):
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
