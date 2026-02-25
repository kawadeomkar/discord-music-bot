"""Tests for src/sources.py — URL parsing and source type detection."""
import pytest

from src.sources import (
    SoundcloudSource,
    SpotifySource,
    URLSource,
    YTSource,
    parse_url,
    spotify_playlist_to_ytsearch,
)


class TestParseUrlYouTube:
    def test_youtube_watch_url(self):
        result = parse_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "-play https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE
        assert result.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result.ts is None
        assert result.process is False

    def test_youtube_watch_url_with_t_param(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.ts == 42

    def test_youtube_watch_url_with_ts_param(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ts=120"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.ts == 120

    def test_youtu_be_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE
        assert result.url == url
        assert result.ts is None

    def test_youtu_be_with_timestamp(self):
        url = "https://youtu.be/dQw4w9WgXcQ?t=60"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.ts == 60

    def test_youtube_without_www(self):
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE


class TestParseUrlSpotify:
    def test_spotify_track(self):
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == "track"
        assert result.id == "4cOdK2wGLETKBW3PvgPWqT"
        assert result.stype == URLSource.SPOTIFY
        assert result.process is True

    def test_spotify_playlist(self):
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == "playlist"
        assert result.id == "37i9dQZF1DXcBWIGoYBM5M"
        assert result.stype == URLSource.SPOTIFY

    def test_spotify_track_with_si_param(self):
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT?si=abc123"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == "track"
        assert result.id == "4cOdK2wGLETKBW3PvgPWqT"

    def test_unknown_spotify_type_raises(self):
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(Exception, match="Unknown Spotify track type"):
            parse_url(url, f"-play {url}")


class TestParseUrlSoundcloud:
    def test_soundcloud_url(self):
        url = "https://soundcloud.com/artist/track-name"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SoundcloudSource)
        assert result.stype == URLSource.SOUNDCLOUD
        assert result.url == url
        assert result.process is True

    def test_soundcloud_ts_defaults_to_none(self):
        url = "https://soundcloud.com/artist/track"
        result = parse_url(url, f"-play {url}")
        assert result.ts is None


class TestParseUrlTextSearch:
    def test_plain_text_becomes_ytsearch(self):
        result = parse_url(
            "never gonna give you up", "-play never gonna give you up"
        )
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:never gonna give you up"
        assert result.process is True
        assert result.url is None

    def test_multi_word_search(self):
        result = parse_url("bohemian rhapsody queen", "-play bohemian rhapsody queen")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:bohemian rhapsody queen"

    def test_single_word_search(self):
        result = parse_url("beethoven", "-play beethoven")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:beethoven"


class TestParseUrlErrors:
    def test_unsupported_domain_raises(self):
        url = "https://example.com/video/123"
        with pytest.raises(Exception, match="Domain not supported"):
            parse_url(url, f"-play {url}")

    def test_vimeo_raises(self):
        url = "https://vimeo.com/12345678"
        with pytest.raises(Exception, match="Domain not supported"):
            parse_url(url, f"-play {url}")


class TestSpotifyPlaylistToYTSearch:
    def test_converts_titles_to_ytsearch(self):
        titles = ["Never Gonna Give You Up Rick Astley", "Bohemian Rhapsody Queen"]
        result = spotify_playlist_to_ytsearch(titles)

        assert len(result) == 2
        assert all(isinstance(r, YTSource) for r in result)
        assert result[0].ytsearch == "ytsearch:Never Gonna Give You Up Rick Astley"
        assert result[1].ytsearch == "ytsearch:Bohemian Rhapsody Queen"

    def test_all_results_have_process_true(self):
        titles = ["Song A", "Song B", "Song C"]
        result = spotify_playlist_to_ytsearch(titles)
        assert all(r.process is True for r in result)

    def test_empty_list_returns_empty(self):
        assert spotify_playlist_to_ytsearch([]) == []

    def test_single_title(self):
        result = spotify_playlist_to_ytsearch(["Only Song Artist"])
        assert len(result) == 1
        assert result[0].ytsearch == "ytsearch:Only Song Artist"

    def test_url_field_is_none(self):
        result = spotify_playlist_to_ytsearch(["Song"])
        assert result[0].url is None
