"""Tests for src/youtube.py — QueueObject, YTDL config, and yt_source."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.youtube import YTDL, YTDL_OPTS, QueueObject


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


class TestYTDLOpts:
    def test_format_is_bestaudio(self):
        assert YTDL_OPTS["format"] == "bestaudio/best"

    def test_noplaylist_is_true(self):
        assert YTDL_OPTS["noplaylist"] is True

    def test_source_address_is_ipv4_any(self):
        assert YTDL_OPTS["source_address"] == "0.0.0.0"

    def test_default_search_is_auto(self):
        assert YTDL_OPTS["default_search"] == "auto"

    def test_retries_is_set(self):
        assert YTDL_OPTS["retries"] > 0

    def test_socket_timeout_is_set(self):
        assert YTDL_OPTS["socket_timeout"] > 0

    def test_extractor_args_include_youtube(self):
        assert "youtube" in YTDL_OPTS["extractor_args"]

    def test_ffmpeg_reconnect_args_present(self):
        ffmpeg_args = YTDL_OPTS["external_downloader_args"]["ffmpeg_i"]
        assert "-reconnect" in ffmpeg_args
        assert "-reconnect_streamed" in ffmpeg_args


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

        with patch("src.youtube.ytdl.extract_info", return_value=fake_data):
            result = await YTDL.yt_source(
                mock_ctx, "ytsearch:test song", process=True
            )

        assert isinstance(result, QueueObject)
        assert result.title == "Extracted Title"
        assert result.webpage_url == "https://www.youtube.com/watch?v=test123"
        assert result.requester is mock_ctx.author

    async def test_yt_source_raises_when_no_data(self, mock_ctx):
        with patch("src.youtube.ytdl.extract_info", return_value=None):
            with pytest.raises(Exception, match="Could not find song"):
                await YTDL.yt_source(mock_ctx, "ytsearch:nothing", process=True)

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
        with patch("src.youtube.ytdl.extract_info", return_value=fake_data):
            result = await YTDL.yt_source(mock_ctx, "ytsearch:test", process=True)

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
        with patch("src.youtube.ytdl.extract_info", return_value=fake_data):
            result = await YTDL.yt_source(mock_ctx, "ytsearch:test", process=True)

        assert result.title == "Real Video"

    async def test_yt_source_passes_timestamp(self, mock_ctx):
        fake_data = {
            "webpage_url": "https://www.youtube.com/watch?v=ts_test",
            "title": "Timestamped Song",
        }
        with patch("src.youtube.ytdl.extract_info", return_value=fake_data):
            result = await YTDL.yt_source(
                mock_ctx, "https://yt.com/watch?v=ts_test", process=False, ts=45
            )

        assert result.ts == 45
