"""Tests for src/sources.py — URL parsing and source type detection."""

import pytest

from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    URLSource,
    YTSource,
    YTType,
    parse_input,
    parse_url,
    spotify_playlist_to_ytsearch,
)


class TestParseUrlYouTube:
    def test_youtube_watch_url(self) -> None:
        result = parse_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "-play https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE
        assert result.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result.ts is None
        assert result.process is False

    def test_youtube_watch_url_with_t_param(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.ts == 42

    def test_youtube_watch_url_with_ts_param(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ts=120"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.ts == 120

    def test_youtu_be_short_url(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE
        assert result.url == url
        assert result.ts is None

    def test_youtu_be_with_timestamp(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ?t=60"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.ts == 60

    def test_youtube_without_www(self) -> None:
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE

    def test_youtube_watch_url_is_track_by_default(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.TRACK
        assert result.list_id is None

    def test_youtube_url_with_list_param_is_playlist(self) -> None:
        url = "https://www.youtube.com/watch?v=jOLT6ukrQSg&list=RDEMfxur2p8gn1zGJ2gwGBdjQg"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "RDEMfxur2p8gn1zGJ2gwGBdjQg"
        assert result.url == url

    def test_youtube_playlist_url_is_playlist(self) -> None:
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"

    def test_youtube_playlist_preserves_timestamp(self) -> None:
        url = "https://www.youtube.com/watch?v=abc&list=PLtest&t=30"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "PLtest"
        assert result.ts == 30


class TestParseUrlSpotify:
    def test_spotify_track(self) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.TRACK
        assert result.id == "4cOdK2wGLETKBW3PvgPWqT"
        assert result.stype == URLSource.SPOTIFY
        assert result.process is True

    def test_spotify_playlist(self) -> None:
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.PLAYLIST
        assert result.id == "37i9dQZF1DXcBWIGoYBM5M"
        assert result.stype == URLSource.SPOTIFY

    def test_spotify_track_with_si_param(self) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT?si=abc123"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.TRACK
        assert result.id == "4cOdK2wGLETKBW3PvgPWqT"

    def test_unknown_spotify_type_raises(self) -> None:
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(Exception, match="Unknown Spotify track type"):
            parse_url(url, f"-play {url}")


class TestParseUrlSoundcloud:
    def test_soundcloud_url(self) -> None:
        url = "https://soundcloud.com/artist/track-name"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SoundcloudSource)
        assert result.stype == URLSource.SOUNDCLOUD
        assert result.url == url
        assert result.process is True

    def test_soundcloud_ts_defaults_to_none(self) -> None:
        url = "https://soundcloud.com/artist/track"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SoundcloudSource)
        assert result.ts is None


class TestParseUrlErrors:
    def test_plain_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Not a recognised URL"):
            parse_url("never gonna give you up", "-play never gonna give you up")

    def test_dotless_host_raises_value_error(self) -> None:
        """A search term like "98/99" matches the domain regex with a dotless
        "host" of "98" — not a real URL, so it raises ValueError and parse_input
        falls back to search rather than shipping it to yt-dlp."""
        with pytest.raises(ValueError, match="Not a recognised URL"):
            parse_url("98/99", "-play 98/99")


class TestParseUrlOther:
    """Domains we don't special-case are handed to yt-dlp rather than rejected."""

    def test_unknown_domain_becomes_generic_ytdlp_source(self) -> None:
        url = "https://example.com/video/123"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url

    def test_vimeo_becomes_generic_ytdlp_source(self) -> None:
        url = "https://vimeo.com/12345678"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url

    def test_tiktok_becomes_generic_ytdlp_source(self) -> None:
        url = "https://www.tiktok.com/@user/video/1234567890"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url


class TestParseInput:
    def test_plain_text_becomes_ytsearch(self) -> None:
        result = parse_input("never gonna give you up", "-play never gonna give you up")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:never gonna give you up"
        assert result.process is True
        assert result.url is None

    def test_multi_word_search(self) -> None:
        result = parse_input("bohemian rhapsody queen", "-play bohemian rhapsody queen")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:bohemian rhapsody queen"

    def test_single_word_search(self) -> None:
        result = parse_input("beethoven", "-play beethoven")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:beethoven"

    def test_valid_url_is_parsed_directly(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_input(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.url == url

    def test_spotify_url_is_parsed_directly(self) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        result = parse_input(url, f"-play {url}")
        assert isinstance(result, SpotifySource)

    def test_search_term_with_slash_does_not_hit_domain_regex(self) -> None:
        """Regression: "98/99 sorisa" was misparsed as a URL with domain "98",
        raising "Domain not supported 98" instead of falling back to search."""
        result = parse_input("98/99", "-p 98/99 sorisa")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:98/99 sorisa"
        assert result.url is None

    def test_single_word_with_slash_still_tries_url_parse(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_input(url, f"-p {url}")
        assert isinstance(result, YTSource)
        assert result.url == url

    def test_single_word_dotless_slash_falls_back_to_search(self) -> None:
        """A lone "98/99" (no dot, no scheme) is not a URL — parse_url raises
        ValueError and parse_input recovers with a YouTube search."""
        result = parse_input("98/99", "-p 98/99")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:98/99"
        assert result.url is None

    def test_single_word_unknown_domain_is_parsed_as_url(self) -> None:
        """A bare link on a non-special-cased site routes straight to yt-dlp."""
        url = "https://www.tiktok.com/@user/video/1234567890"
        result = parse_input(url, f"-p {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url


class TestSpotifyPlaylistToYTSearch:
    def test_converts_titles_to_ytsearch(self) -> None:
        titles = ["Never Gonna Give You Up Rick Astley", "Bohemian Rhapsody Queen"]
        result = spotify_playlist_to_ytsearch(titles)

        assert len(result) == 2
        assert all(isinstance(r, YTSource) for r in result)
        assert result[0].ytsearch == "ytsearch:Never Gonna Give You Up Rick Astley"
        assert result[1].ytsearch == "ytsearch:Bohemian Rhapsody Queen"

    def test_all_results_have_process_true(self) -> None:
        titles = ["Song A", "Song B", "Song C"]
        result = spotify_playlist_to_ytsearch(titles)
        assert all(r.process is True for r in result)

    def test_empty_list_returns_empty(self) -> None:
        assert spotify_playlist_to_ytsearch([]) == []

    def test_single_title(self) -> None:
        result = spotify_playlist_to_ytsearch(["Only Song Artist"])
        assert len(result) == 1
        assert result[0].ytsearch == "ytsearch:Only Song Artist"

    def test_url_field_is_none(self) -> None:
        result = spotify_playlist_to_ytsearch(["Song"])
        assert result[0].url is None


class TestYTSourcePlaylistUrl:
    """`YTSource.playlist_url` — the single spelling of the
    `url or ".../playlist?list={list_id}"` fallback that the enqueue, playnow
    and resolve paths all need."""

    def test_pasted_url_wins_over_rebuild(self) -> None:
        """A user-pasted URL is returned verbatim — it may carry an index, a
        video id or a radio mix that the rebuilt form would discard."""
        url = "https://www.youtube.com/watch?v=abc&list=PLtest&index=4"
        src = YTSource(url=url, list_id="PLtest", type=YTType.PLAYLIST)
        assert src.playlist_url == url

    def test_rebuilds_from_list_id_when_no_url(self) -> None:
        src = YTSource(url=None, list_id="PLtest", type=YTType.PLAYLIST)
        assert src.playlist_url == "https://www.youtube.com/playlist?list=PLtest"

    def test_empty_url_falls_back_to_rebuild(self) -> None:
        """The implementation is `self.url or ...`, so an empty string — falsy,
        not None — must take the rebuild path rather than returning "". A
        `self.url is not None` regression would yield an empty URL and a
        silently broken enqueue."""
        src = YTSource(url="", list_id="PLtest", type=YTType.PLAYLIST)
        assert src.playlist_url == "https://www.youtube.com/playlist?list=PLtest"

    def test_property_is_not_gated_on_playlist_type(self) -> None:
        """Documents that the property does NOT assert type == PLAYLIST: a TRACK
        source with a url returns it unchanged. Callers are responsible for only
        reading this on playlist sources."""
        src = YTSource(url="https://yt.com/watch?v=one", type=YTType.TRACK)
        assert src.playlist_url == "https://yt.com/watch?v=one"

    def test_no_url_and_no_list_id_stringifies_none(self) -> None:
        """Unguarded edge, pinned rather than endorsed: with both fields unset the
        f-string interpolates the literal "None". Reachable only by hand-building a
        PLAYLIST source without a list_id, which parse_url never does; adding a
        guard should change this test, so the change stays deliberate."""
        src = YTSource(type=YTType.PLAYLIST)
        assert src.playlist_url == "https://www.youtube.com/playlist?list=None"

    def test_parse_url_output_yields_a_usable_playlist_url(self) -> None:
        """End-to-end with the real parser, not a hand-built dataclass."""
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.playlist_url == url

    def test_rebuilt_url_is_accepted_by_the_parser(self) -> None:
        """Round-trip: the rebuilt form must itself parse back to the same
        playlist, so a rebuilt URL is safe to hand to any download path."""
        src = YTSource(url=None, list_id="PLround", type=YTType.PLAYLIST)
        reparsed = parse_url(src.playlist_url, f"-play {src.playlist_url}")
        assert isinstance(reparsed, YTSource)
        assert reparsed.type == YTType.PLAYLIST
        assert reparsed.list_id == "PLround"
