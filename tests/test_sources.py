"""Tests for src/sources.py — URL parsing and source type detection."""

import re
import time
from collections.abc import Callable
from typing import NamedTuple, Optional

import pytest

from src.guild_state import Analytics
from src.sources import (
    LINK_MAX_CHARS,
    is_link,
    unquote_argument,
    MAX_START_OFFSET_SECS,
    QUERY_SOURCE_SEARCH,
    START_OFFSET_FORMATS,
    TIMESTAMP_FORMATS,
    QUERY_SOURCE_SOUNDCLOUD,
    QUERY_SOURCE_SPOTIFY,
    QUERY_SOURCE_YOUTUBE,
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    UnsupportedSpotifyLinkError,
    URLSource,
    YTSource,
    YTType,
    normalize_query_host,
    parse_input,
    parse_start_offset,
    parse_timestamp,
    parse_url,
    query_source_of,
    spotify_playlist_to_ytsearch,
    timestamp_warning,
)
from src.spotify import SpotifyTrack
from src.youtube import _source_cache_key


class TestParseUrlYouTube:
    def test_youtube_watch_url(self) -> None:
        result = parse_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE
        assert result.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result.ts is None
        assert result.process is False

    def test_youtube_watch_url_with_t_param(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.ts == 42

    def test_youtube_watch_url_with_ts_param(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ts=120"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.ts == 120

    def test_youtu_be_short_url(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE
        assert result.url == url
        assert result.ts is None

    def test_youtu_be_with_timestamp(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ?t=60"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.ts == 60

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("90", 90),
            ("90s", 90),
            ("1m30s", 90),
            ("2m", 120),
            ("1h", 3600),
            ("1h2m3s", 3723),
            ("0", 0),
            ("1H2M3S", 3723),  # YouTube has emitted uppercase
        ],
    )
    def test_hms_timestamp_forms_are_parsed(self, raw: str, expected: int) -> None:
        """YouTube's older share format. These used to raise ValueError out of
        parse_url, which parse_input read as "not a URL" — so the link became a
        SEARCH for its own text and the seek was silently dropped."""
        url = f"https://youtu.be/dQw4w9WgXcQ?t={raw}"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.ts == expected

    @pytest.mark.parametrize("raw", ["abc", "1m2x", "", "m", "s", "-30", "1.5"])
    def test_unparseable_timestamp_keeps_the_url(self, raw: str) -> None:
        """Degrade to "play from the start", never to "this wasn't a URL"."""
        url = f"https://youtu.be/dQw4w9WgXcQ?t={raw}"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.url == url  # still a URL, not a ytsearch
        assert result.ts is None

    def test_unparseable_timestamp_does_not_fall_back_to_search(self) -> None:
        """The end-to-end shape of the bug: through parse_input, a bad
        timestamp must not turn the link into `ytsearch:<the url>`."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=notatime"
        result = parse_input(url)
        assert isinstance(result, YTSource)
        assert result.ytsearch is None
        assert result.url == url

    def test_timestamp_and_playlist_together(self) -> None:
        url = "https://www.youtube.com/watch?v=abc&list=PLtest&t=1m30s"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.ts == 90
        assert result.list_id == "PLtest"
        assert result.type == YTType.PLAYLIST

    def test_youtube_without_www(self) -> None:
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.YOUTUBE

    def test_youtube_watch_url_is_track_by_default(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.TRACK
        assert result.list_id is None

    def test_youtube_url_with_list_param_is_playlist(self) -> None:
        url = "https://www.youtube.com/watch?v=jOLT6ukrQSg&list=RDEMfxur2p8gn1zGJ2gwGBdjQg"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "RDEMfxur2p8gn1zGJ2gwGBdjQg"
        assert result.url == url

    def test_youtube_playlist_url_is_playlist(self) -> None:
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"

    def test_youtube_playlist_preserves_timestamp(self) -> None:
        url = "https://www.youtube.com/watch?v=abc&list=PLtest&t=30"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "PLtest"
        assert result.ts == 30


class TestParseUrlPlaylistIndex:
    """`index=` is YouTube's 1-based position of the video the link was copied
    at. Parsed only on the playlist branch and never allowed to raise — a
    ValueError out of parse_url means "not a URL" and searches for the link text.
    """

    def test_index_is_parsed_from_a_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=abc&list=PLtest&index=4"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.index == 4
        assert result.video_id == "abc"

    def test_index_is_none_when_absent(self) -> None:
        url = "https://www.youtube.com/playlist?list=PLtest"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.index is None
        assert result.video_id is None

    def test_index_is_not_carried_by_a_bare_track(self) -> None:
        """No list, no playlist — the index has nothing to index into."""
        url = "https://www.youtube.com/watch?v=abc&index=4"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.TRACK
        assert result.index is None

    @pytest.mark.parametrize("raw", ["0", "-2", "abc", "4.5"])
    def test_unusable_index_parses_as_none_not_an_error(self, raw: str) -> None:
        """A malformed index degrades to "no index" instead of raising: the
        alternative sends the whole link to ytsearch as plain text."""
        url = f"https://www.youtube.com/watch?v=abc&list=PLtest&index={raw}"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "PLtest"
        assert result.index is None


class TestParseUrlSpotify:
    def test_spotify_track(self) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.TRACK
        assert result.id == "4cOdK2wGLETKBW3PvgPWqT"
        assert result.stype == URLSource.SPOTIFY
        assert result.process is True

    def test_spotify_playlist(self) -> None:
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.PLAYLIST
        assert result.id == "37i9dQZF1DXcBWIGoYBM5M"
        assert result.stype == URLSource.SPOTIFY

    def test_spotify_track_with_si_param(self) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT?si=abc123"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.TRACK
        assert result.id == "4cOdK2wGLETKBW3PvgPWqT"

    def test_spotify_album(self) -> None:
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.ALBUM
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"
        assert result.stype == URLSource.SPOTIFY
        assert result.process is True

    def test_spotify_album_without_open_subdomain(self) -> None:
        url = "https://spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.ALBUM
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"

    def test_spotify_album_with_si_param(self) -> None:
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl?si=abc123"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.ALBUM
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"

    @pytest.mark.parametrize(
        ("locale", "kind", "expected_type"),
        [
            ("intl-de", "album", SpotifyType.ALBUM),
            ("intl-pt", "track", SpotifyType.TRACK),
            ("intl-ja", "playlist", SpotifyType.PLAYLIST),
        ],
    )
    def test_intl_locale_prefix_is_dropped(
        self, locale: str, kind: str, expected_type: SpotifyType
    ) -> None:
        """Spotify's own share sheet emits /intl-xx/ links for every
        non-English client — rejecting them rejects the URL half the world
        copies, with copy telling the user to paste what they just pasted."""
        url = f"https://open.spotify.com/{locale}/{kind}/6WgSCcRfaXuBVfM2TpV0Kl"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert result.type is expected_type
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"

    def test_intl_prefixed_unsupported_type_still_raises(self) -> None:
        """The locale strip must expose the real type, not blindly accept."""
        url = "https://open.spotify.com/intl-fr/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url)
        assert "'artist'" in str(exc_info.value)

    @pytest.mark.parametrize(
        "url",
        [
            # A bare /album: the len(path) < 2 leg.
            "https://open.spotify.com/album",
            # A trailing slash: len(path) == 2 with path[1] == "", so only the
            # `not path[1]` half of the guard catches it. Untested, that half
            # could be dropped with the suite green — and a SpotifySource with
            # an empty id resolves to a 404 the user cannot act on.
            "https://open.spotify.com/album/",
            "https://open.spotify.com/intl-de/track/",
        ],
    )
    def test_spotify_link_without_id_raises_cleanly(self, url: str) -> None:
        with pytest.raises(UnsupportedSpotifyLinkError, match="has no id"):
            parse_url(url)

    def test_unknown_spotify_type_raises(self) -> None:
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url)
        # The message names the supported types so the user can act on it —
        # as a sentence ("or"), not a bare comma join.
        assert "'artist'" in str(exc_info.value)
        assert "track, playlist or album" in str(exc_info.value)

    def test_user_message_is_the_message(self) -> None:
        """_command_error renders `user_message` for allowlisted classes; the
        property existing is what keeps the class-name prefix out of the
        embed."""
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url)
        assert exc_info.value.user_message == str(exc_info.value)

    def test_unknown_spotify_type_is_not_a_value_error(self) -> None:
        """Regression guard: parse_input catches ValueError and falls back to a
        YouTube search. If this error ever becomes a ValueError, an /artist/
        link silently turns into `ytsearch:https://open.spotify.com/...`."""
        url = "https://open.spotify.com/show/4rOoJ6Egrf8K2IrywzwOMk"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url)
        assert not isinstance(exc_info.value, ValueError)

    def test_unknown_spotify_type_suppresses_exception_chain(self) -> None:
        """`from None`: the enum-lookup ValueError is an implementation detail;
        chaining it doubles the traceback in every error log."""
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True

    @pytest.mark.parametrize(
        "segment",
        ["**x**||y||`z`", "\u202e" * 1970, "\\" * 1970, "[x](y)"],
        ids=["markdown", "bidi-run", "backslash-run", "brackets"],
    )
    def test_the_echoed_type_cannot_style_or_overflow_the_embed(
        self, segment: str
    ) -> None:
        """The message is an embed description: Discord renders markdown there and
        400s the send past 4,096 characters, which leaves the user with no reply."""
        with pytest.raises(UnsupportedSpotifyLinkError) as raised:
            parse_url(f"https://open.spotify.com/{segment}/1")

        message = raised.value.user_message
        assert len(message) < 300
        for mark in ("**", "||", "`", "[", "]"):
            assert mark not in message.replace("\\" + mark[0], "")

    @pytest.mark.parametrize(
        "url",
        [
            "https://open.spotify.com/",
            "https://open.spotify.com/intl-de",
            "https://open.spotify.com/intl-de/",
        ],
    )
    def test_a_link_with_no_type_is_not_quoted_as_an_empty_one(self, url: str) -> None:
        with pytest.raises(UnsupportedSpotifyLinkError) as raised:
            parse_url(url)

        assert "''" not in raised.value.user_message
        assert "track, playlist or album" in raised.value.user_message

    @pytest.mark.parametrize(
        "url",
        [
            "https://open.spotify.com/album/ID&x=1",
            "https://open.spotify.com/album/../../v1/me",
            "https://open.spotify.com/album/ID>",
        ],
    )
    def test_an_id_that_is_not_base62_is_refused(self, url: str) -> None:
        """The id goes into the API path, the cache key and the card's link."""
        with pytest.raises(UnsupportedSpotifyLinkError, match="doesn't look right"):
            parse_url(url)

    def test_a_fragment_is_split_off_rather_than_refused(self) -> None:
        """`#frag` is not part of the id and never reaches it, so the link plays
        instead of being turned away for an id that only looked wrong."""
        source = parse_url("https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl#x")
        assert isinstance(source, SpotifySource)
        assert (source.type, source.id) == (
            SpotifyType.ALBUM,
            "6WgSCcRfaXuBVfM2TpV0Kl",
        )

    def test_a_link_with_its_embed_suppressed_parses_like_the_bare_one(self) -> None:
        source = parse_input("<https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl>")
        assert isinstance(source, SpotifySource)
        assert source.id == "6WgSCcRfaXuBVfM2TpV0Kl"

    def test_the_canonical_url_drops_the_locale_and_the_share_parameters(self) -> None:
        source = parse_url(
            "https://open.spotify.com/intl-de/album/6WgSCcRfaXuBVfM2TpV0Kl?si=abc"
        )
        assert isinstance(source, SpotifySource)
        assert source.url == "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"

    def test_parse_input_does_not_search_youtube_for_an_unsupported_link(self) -> None:
        with pytest.raises(UnsupportedSpotifyLinkError):
            parse_input("https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d")

    @pytest.mark.parametrize(
        "path",
        [
            "track/4cOdK2wGLETKBW3PvgPWqT",
            "intl-de/track/4cOdK2wGLETKBW3PvgPWqT",
            "intl-pt-br/track/4cOdK2wGLETKBW3PvgPWqT",
            "embed/track/4cOdK2wGLETKBW3PvgPWqT",
            "intl-ja/embed/track/4cOdK2wGLETKBW3PvgPWqT/",
        ],
    )
    def test_locale_and_embed_segments_name_the_same_track(self, path: str) -> None:
        """Localized clients prefix `intl-<locale>`, and the embed player `embed/`.
        Real locales answer 200 on that path, so it is what users paste."""
        result = parse_url(f"https://open.spotify.com/{path}")
        assert result == SpotifySource(SpotifyType.TRACK, "4cOdK2wGLETKBW3PvgPWqT")

    @pytest.mark.parametrize(
        "link",
        [
            "https://play.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
            "https://open.spotify.com/user/spotify/playlist/37i9dQZF1DXcBWIGoYBM5M",
            "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
            "spotify:user:spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
        ],
    )
    def test_legacy_hosts_paths_and_uris_name_the_same_playlist(
        self, link: str
    ) -> None:
        result = parse_input(link)
        assert result == SpotifySource(SpotifyType.PLAYLIST, "37i9dQZF1DXcBWIGoYBM5M")

    def test_a_uri_that_is_not_a_spotify_item_stays_a_search(self) -> None:
        """Only the exact URI shape is Spotify's; anything else spelled with the
        prefix is words."""
        assert parse_input("spotify:wrapped").stype is URLSource.SEARCH


class TestSpotifyRefusal:
    """A Spotify link the bot cannot queue fails the command with a sentence, in
    parse_input, before any join or interruption (tests/commands/test_play.py)."""

    @pytest.mark.parametrize(
        "link",
        [
            "https://open.spotify.com/user/someone",
            "https://open.spotify.com/",
            "https://open.spotify.com/socialsession/abc?feature=campfire_invite",
        ],
    )
    def test_anything_else_on_the_host_is_refused(self, link: str) -> None:
        """A profile or a Jam invite. A ValueError here would search YouTube for
        the link's text."""
        with pytest.raises(
            UnsupportedSpotifyLinkError, match="aren't supported|doesn't point at"
        ):
            parse_input(link)

    def test_a_track_link_cut_off_before_its_id_says_so(self) -> None:
        with pytest.raises(UnsupportedSpotifyLinkError, match="has no id"):
            parse_input("https://open.spotify.com/track/")

    def test_it_is_not_a_value_error(self) -> None:
        """parse_input's search fallback catches ValueError."""
        assert not issubclass(UnsupportedSpotifyLinkError, ValueError)


class TestParseUrlSoundcloud:
    def test_soundcloud_url(self) -> None:
        url = "https://soundcloud.com/artist/track-name"
        result = parse_url(url)
        assert isinstance(result, SoundcloudSource)
        assert result.stype == URLSource.SOUNDCLOUD
        assert result.url == url
        assert result.process is True

    def test_soundcloud_ts_defaults_to_none(self) -> None:
        url = "https://soundcloud.com/artist/track"
        result = parse_url(url)
        assert isinstance(result, SoundcloudSource)
        assert result.ts is None


class TestParseUrlErrors:
    def test_plain_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Not a recognised URL"):
            parse_url("never gonna give you up")

    def test_dotless_host_raises_value_error(self) -> None:
        """A search term like "98/99" matches the domain regex with a dotless
        "host" of "98" — not a real URL, so it raises ValueError and parse_input
        falls back to search rather than shipping it to yt-dlp."""
        with pytest.raises(ValueError, match="Not a recognised URL"):
            parse_url("98/99")


class TestParseUrlOther:
    """Domains we don't special-case are handed to yt-dlp rather than rejected."""

    def test_unknown_domain_becomes_generic_ytdlp_source(self) -> None:
        url = "https://example.com/video/123"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url

    def test_vimeo_becomes_generic_ytdlp_source(self) -> None:
        url = "https://vimeo.com/12345678"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url

    def test_tiktok_becomes_generic_ytdlp_source(self) -> None:
        url = "https://www.tiktok.com/@user/video/1234567890"
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url


class TestParseInput:
    def test_plain_text_becomes_ytsearch(self) -> None:
        result = parse_input("never gonna give you up")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:never gonna give you up"
        assert result.process is True
        assert result.url is None

    def test_multi_word_search(self) -> None:
        result = parse_input("bohemian rhapsody queen")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:bohemian rhapsody queen"

    def test_single_word_search(self) -> None:
        result = parse_input("beethoven")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:beethoven"

    def test_valid_url_is_parsed_directly(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_input(url)
        assert isinstance(result, YTSource)
        assert result.url == url

    def test_spotify_url_is_parsed_directly(self) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
        result = parse_input(url)
        assert isinstance(result, SpotifySource)

    def test_search_term_with_slash_does_not_hit_domain_regex(self) -> None:
        """Regression: "98/99 sorisa" was misparsed as a URL with domain "98",
        raising "Domain not supported 98" instead of falling back to search."""
        result = parse_input("98/99 sorisa")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:98/99 sorisa"
        assert result.url is None

    @pytest.mark.parametrize(
        "padded",
        [
            "  https://youtu.be/dQw4w9WgXcQ  ",
            "\thttps://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ\n",
        ],
    )
    def test_surrounding_whitespace_still_parses_as_a_url(self, padded: str) -> None:
        """A padded link is one token, so it reaches parse_url. The URL that comes
        back is the token without the padding, because it is handed straight to
        yt-dlp."""
        result = parse_input(padded)
        assert isinstance(result, YTSource)
        assert result.url == "https://youtu.be/dQw4w9WgXcQ"
        assert result.ytsearch is None

    def test_the_search_comes_from_the_argument_alone(self) -> None:
        """parse_input reads nothing but what it is handed, which is what lets a
        caller strip a flag off the front and get the answer for what remains."""
        result = parse_input("never gonna give you up")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:never gonna give you up"

    def test_internal_whitespace_collapses_in_the_search(self) -> None:
        """Runs of whitespace inside the term are separators, not content."""
        result = parse_input("never   gonna\tgive")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:never gonna give"

    def test_single_word_with_slash_still_tries_url_parse(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_input(url)
        assert isinstance(result, YTSource)
        assert result.url == url

    def test_single_word_dotless_slash_falls_back_to_search(self) -> None:
        """A lone "98/99" (no dot, no scheme) is not a URL — parse_url raises
        ValueError and parse_input recovers with a YouTube search."""
        result = parse_input("98/99")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:98/99"
        assert result.url is None

    def test_single_word_unknown_domain_is_parsed_as_url(self) -> None:
        """A bare link on a non-special-cased site routes straight to yt-dlp."""
        url = "https://www.tiktok.com/@user/video/1234567890"
        result = parse_input(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert result.url == url


_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=3)
_ORIGIN = "https://open.spotify.com/album/abc123"
_REQUESTER = 424242424242424242


class TestSpotifyPlaylistToYTSearch:
    def test_converts_titles_to_ytsearch(self) -> None:
        titles = ["Never Gonna Give You Up Rick Astley", "Bohemian Rhapsody Queen"]
        result = spotify_playlist_to_ytsearch(
            titles, analytics=_ANALYTICS, origin=_ORIGIN, requester_id=_REQUESTER
        )

        assert len(result) == 2
        assert all(isinstance(r, YTSource) for r in result)
        assert result[0].ytsearch == "ytsearch:Never Gonna Give You Up Rick Astley"
        assert result[1].ytsearch == "ytsearch:Bohemian Rhapsody Queen"

    def test_all_results_have_process_true(self) -> None:
        titles = ["Song A", "Song B", "Song C"]
        result = spotify_playlist_to_ytsearch(
            titles, analytics=_ANALYTICS, origin=_ORIGIN, requester_id=_REQUESTER
        )
        assert all(r.process is True for r in result)

    def test_empty_list_returns_empty(self) -> None:
        assert (
            spotify_playlist_to_ytsearch(
                [], analytics=_ANALYTICS, origin=_ORIGIN, requester_id=_REQUESTER
            )
            == []
        )

    def test_single_title(self) -> None:
        result = spotify_playlist_to_ytsearch(
            ["Only Song Artist"],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=_REQUESTER,
        )
        assert len(result) == 1
        assert result[0].ytsearch == "ytsearch:Only Song Artist"

    def test_url_field_is_none(self) -> None:
        result = spotify_playlist_to_ytsearch(
            ["Song"], analytics=_ANALYTICS, origin=_ORIGIN, requester_id=_REQUESTER
        )
        assert result[0].url is None

    def test_per_track_positions_derive_from_the_head(self) -> None:
        # The head's analytics fans out: same ask-time queued_at on every track,
        # positions incrementing from the head's — a playlist behind 3 songs
        # waits at 3, 4, 5.
        result = spotify_playlist_to_ytsearch(
            ["a", "b", "c"],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=_REQUESTER,
        )
        assert [r.analytics.queue_position for r in result] == [3, 4, 5]
        assert all(r.analytics.queued_at == 1752530000.5 for r in result)

    def test_every_track_carries_the_requester(self) -> None:
        """These resolve at dequeue, minutes to an hour after the command returned.
        Without the ID the resolve attributes each track to whoever ran a command
        most recently."""
        result = spotify_playlist_to_ytsearch(
            ["a", "b"], analytics=_ANALYTICS, origin=_ORIGIN, requester_id=_REQUESTER
        )
        assert [r.requester_id for r in result] == [_REQUESTER] * 2

    def test_the_requester_has_no_default(self) -> None:
        with pytest.raises(TypeError, match="requester_id"):
            spotify_playlist_to_ytsearch(["a"], analytics=_ANALYTICS, origin=_ORIGIN)  # pyright: ignore[reportCallIssue]

    def test_a_track_row_becomes_the_searchs_display_fields(self) -> None:
        rows = [
            SpotifyTrack(
                name="LOYALTY.",
                artists=["Kendrick Lamar", "Rihanna"],
                duration_secs=227,
                url="https://open.spotify.com/track/abc",
            )
        ]
        (source,) = spotify_playlist_to_ytsearch(
            ["LOYALTY. Kendrick Lamar Rihanna"],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=7,
            tracks=rows,
        )
        assert source.ytsearch == "ytsearch:LOYALTY. Kendrick Lamar Rihanna"
        assert (source.title, source.uploader, source.duration, source.webpage_url) == (
            "LOYALTY.",
            "Kendrick Lamar, Rihanna",
            227,
            "https://open.spotify.com/track/abc",
        )

    def test_one_artist_tuple_yields_one_shared_byline(self) -> None:
        """An album is usually one artist, and the joined byline is the only string
        this pass keeps for the life of the queue — a fresh join per track was
        ~800 KiB over 10,000 of them."""
        rows = [
            SpotifyTrack(
                name=f"T{i}", artists=["Daft Punk"], duration_secs=180, url=None
            )
            for i in range(50)
        ]
        built = spotify_playlist_to_ytsearch(
            [f"T{i} Daft Punk" for i in range(50)],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=7,
            tracks=rows,
        )
        assert {s.uploader for s in built} == {"Daft Punk"}
        assert len({id(s.uploader) for s in built}) == 1

    def test_a_different_artist_tuple_gets_its_own_byline(self) -> None:
        rows = [
            SpotifyTrack(name="A", artists=["X"], duration_secs=1, url=None),
            SpotifyTrack(name="B", artists=["X", "Y"], duration_secs=1, url=None),
            SpotifyTrack(name="C", artists=[], duration_secs=1, url=None),
        ]
        built = spotify_playlist_to_ytsearch(
            ["A X", "B X Y", "C"],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=7,
            tracks=rows,
        )
        assert [s.uploader for s in built] == ["X", "X, Y", None]

    def test_without_rows_a_search_has_nothing_to_show(self) -> None:
        (source,) = spotify_playlist_to_ytsearch(
            ["a"], analytics=_ANALYTICS, origin=_ORIGIN, requester_id=7
        )
        assert (source.title, source.uploader, source.duration, source.webpage_url) == (
            None,
            None,
            None,
            None,
        )


class TestYTSourcePlaylistUrl:
    """`YTSource.playlist_url` — the single spelling of the
    `url or ".../playlist?list={list_id}"` fallback that the enqueue, interject
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
        """Documents that the property does not assert type == PLAYLIST: a TRACK
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
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.playlist_url == url

    def test_rebuilt_url_is_accepted_by_the_parser(self) -> None:
        """Round-trip: the rebuilt form must itself parse back to the same
        playlist, so a rebuilt URL is safe to hand to any download path."""
        src = YTSource(url=None, list_id="PLround", type=YTType.PLAYLIST)
        reparsed = parse_url(src.playlist_url)
        assert isinstance(reparsed, YTSource)
        assert reparsed.type == YTType.PLAYLIST
        assert reparsed.list_id == "PLround"


class TestParseTimestamp:
    """Direct coverage of the helper, independent of URL shape."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("45", 45),
            ("45s", 45),
            ("3m", 180),
            ("3m20s", 200),
            ("2h", 7200),
            ("2h30m", 9000),
            ("2h30m15s", 9015),
            ("  90  ", 90),
            ("100000", 100000),
        ],
    )
    def test_valid_forms(self, raw: str, expected: int) -> None:
        assert parse_timestamp(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "abc",
            "1x",
            "h",
            "hms",
            "1:30",  # colon form is not something YouTube emits in ?t=
            "1m 30s",
            "-5",
            "1.5s",
            "s30",
        ],
    )
    def test_invalid_forms_return_none(self, raw: str) -> None:
        assert parse_timestamp(raw) is None

    def test_all_optional_pattern_rejects_empty_match(self) -> None:
        """The HMS regex is entirely optional groups, so it also matches ""
        — the "at least one group" guard is what stops that being 0 seconds."""
        assert parse_timestamp("") is None

    @pytest.mark.parametrize("raw", ["²", "①", "₁", "¹²"])
    def test_a_digit_int_rejects_is_not_a_timestamp(self, raw: str) -> None:
        """str.isdigit() is True for these and int() rejects them. This promises
        never to raise, and the flag parser calls it as a front door — outside any
        handler that would turn a ValueError into an answer."""
        assert parse_timestamp(raw) is None

    def test_a_non_ascii_decimal_digit_still_parses(self) -> None:
        """`\\d` is the Nd category, which int() takes: narrowing to ASCII would
        make the bare-seconds form reject what the clock form accepts."""
        assert parse_timestamp("١٢") == 12


class TestParseStartOffset:
    """`--timestamp`'s grammar: the clock a user reads off a player, on top of
    everything `t=` takes."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1:32", 92),
            ("0:07", 7),
            ("2:04:30", 7470),
            ("1:00:00", 3600),
            ("120:00", 7200),  # two-part: 120 minutes, its only reading
            ("0:00", 0),
            ("  1:32  ", 92),
            # Everything parse_timestamp takes comes through unchanged.
            ("90", 90),
            ("90s", 90),
            ("1m30s", 90),
            ("2h30m15s", 9015),
        ],
    )
    def test_valid_forms(self, raw: str, expected: int) -> None:
        assert parse_start_offset(raw) == expected

    @pytest.mark.parametrize(
        "raw", ["1:75", "1:60", "2:70:00", "1:99:00", "1:2", "1:", ":30", "", "abc"]
    )
    def test_invalid_forms_return_none(self, raw: str) -> None:
        """Seconds at or past 60, and the minutes of a three-part clock, are a
        typo: reading `1:75` as 135 would seek somewhere the user did not name."""
        assert parse_start_offset(raw) is None

    def test_a_three_part_clock_bounds_its_minutes_and_a_two_part_does_not(
        self,
    ) -> None:
        """`120:00` is 120 minutes; `1:120:00` is a typo, since the minutes there
        sit between an hours field and a seconds field."""
        assert parse_start_offset("120:00") == 7200
        assert parse_start_offset("1:120:00") is None

    def test_the_clock_form_stays_out_of_parse_timestamp(self) -> None:
        """Two parsers, not one widened: YouTube's `t=` never emits a clock, so a
        pasted `?t=1:32` must keep meaning what it means today."""
        assert parse_start_offset("1:32") == 92
        assert parse_timestamp("1:32") is None


class TestLinkHost:
    def test_plus_and_pipe_are_not_hostname_characters(self) -> None:
        result = parse_input("you+tube|com/watch")
        assert isinstance(result, YTSource)
        assert result.ytsearch == "ytsearch:you+tube|com/watch"

    def test_hyphenated_hosts_are_not_truncated(self) -> None:
        """`-` has to be in the host's class, or "my-site.com" is not a link and
        the archive loses the host the user linked."""
        result = parse_url("https://my-site.com/watch?v=x")
        assert query_source_of(result) == "my-site.com"


class TestNormalizeQueryHost:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("tiktok.com", "tiktok.com"),
            ("www.tiktok.com", "tiktok.com"),
            ("WWW.TikTok.com", "tiktok.com"),
            ("  vimeo.com  ", "vimeo.com"),
            ("music.example.co.uk", "music.example.co.uk"),
            ("xn--80ak6aa92e.com", "xn--80ak6aa92e.com"),
            ("192.168.1.10", "192.168.1.10"),
            # `_` is in `\w`, so bad_host.com really does reach the normalizer —
            # it filters, it does not merely format. `|` and `+` cannot: the
            # link test sends them to search (TestLinkHost). Kept as direct
            # coverage of the filter.
            ("bad_host.com", ""),
            ("bad|host.com", ""),
            ("bad+host.com", ""),
            ("", ""),
            # 64 characters exactly, then one over the column domain.
            ("a" * 60 + ".com", "a" * 60 + ".com"),
            ("a" * 61 + ".com", ""),
        ],
    )
    def test_domain(self, raw: str, expected: str) -> None:
        assert normalize_query_host(raw) == expected


class TestQuerySource:
    """The persisted "how was this asked for" token. The archive cannot recover it
    from webpage_url: Spotify links and plaintext searches both resolve to a
    YouTube watch URL and are indistinguishable once played."""

    def test_plaintext_search(self) -> None:
        result = parse_input("never gonna give you up")
        assert result.stype == URLSource.SEARCH
        assert query_source_of(result) == QUERY_SOURCE_SEARCH

    def test_youtube_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert query_source_of(parse_url(url)) == QUERY_SOURCE_YOUTUBE

    def test_youtu_be_collapses_onto_the_service(self) -> None:
        """A shortener is not a different service."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert query_source_of(parse_url(url)) == QUERY_SOURCE_YOUTUBE

    def test_youtube_playlist(self) -> None:
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp"
        assert query_source_of(parse_url(url)) == QUERY_SOURCE_YOUTUBE

    def test_spotify_track_link(self) -> None:
        url = "https://open.spotify.com/track/5WZD6jHtgSSAGK97diNG7y"
        result = parse_url(url)
        assert isinstance(result, SpotifySource)
        assert query_source_of(result) == QUERY_SOURCE_SPOTIFY

    def test_soundcloud_link(self) -> None:
        url = "https://soundcloud.com/artist/track"
        result = parse_url(url)
        assert isinstance(result, SoundcloudSource)
        assert query_source_of(result) == QUERY_SOURCE_SOUNDCLOUD

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.tiktok.com/@user/video/1234567890", "tiktok.com"),
            ("https://vimeo.com/12345678", "vimeo.com"),
            ("https://artist.bandcamp.com/track/song", "artist.bandcamp.com"),
        ],
    )
    def test_generic_hosts_keep_their_own_host(self, url: str, expected: str) -> None:
        """The point of the open tail: tiktok and vimeo are distinguishable
        without a dataclass apiece."""
        result = parse_url(url)
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert query_source_of(result) == expected

    def test_spotify_playlist_tracks_are_stamped_spotify(self) -> None:
        """The whole reason the token is captured at parse time: these resolve to
        YouTube URLs at dequeue, so nothing downstream could recover it."""
        sources = spotify_playlist_to_ytsearch(
            ["song one", "song two"],
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            requester_id=_REQUESTER,
        )
        assert [query_source_of(s) for s in sources] == [QUERY_SOURCE_SPOTIFY] * 2

    def test_an_uppercase_host_routes_like_the_lowercase_one(self) -> None:
        """Hosts are case-insensitive, and yt-dlp's extractors are not: the link
        handed on has its host lowercased and keeps its path's case."""
        result = parse_url("https://WWW.YouTube.com/watch?v=dQw4w9WgXcQ")
        assert isinstance(result, YTSource)
        assert result.stype is URLSource.YOUTUBE
        assert result.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert query_source_of(result) == QUERY_SOURCE_YOUTUBE

    def test_unstamped_ytsource_is_unknown(self) -> None:
        """A hand-built source (crash recovery, tests, a future call site) reports
        the unknown sentinel rather than guessing."""
        assert query_source_of(YTSource(ytsearch="ytsearch:x")) == ""


class TestQuotedArgumentsSurviveConsumeRest:
    """`-play` takes a consume-rest argument, and discord.py's read_rest
    does no quote handling where the positional parser's get_quoted_word did. So
    the quotes started arriving as part of the value."""

    def test_a_quoted_url_still_parses_as_that_url(self) -> None:
        """parse_url uses re.search, so a quoted URL still matched the domain while
        dragging the trailing quote into the path — yt-dlp then rejects it."""
        source = parse_input('"https://www.youtube.com/watch?v=dQw4w9WgXcQ"')
        assert isinstance(source, YTSource)
        assert source.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_a_quoted_search_does_not_keep_its_quotes(self) -> None:
        """The origin is what -remove matches on, so a quoted search meant the
        obvious retype (`-remove some song`) matched nothing."""
        source = parse_input('"some song"')
        assert isinstance(source, YTSource)
        assert source.ytsearch == "ytsearch:some song"

    def test_an_unmatched_quote_is_left_alone(self) -> None:
        """Only a whole argument wrapped at BOTH ends is a wrapper; anything else
        is text the user typed."""
        source = parse_input('say "hello')
        assert isinstance(source, YTSource)
        assert source.ytsearch == 'ytsearch:say "hello'

    def test_a_bare_quote_pair_is_not_stripped_to_nothing(self) -> None:
        assert unquote_argument('""') == '""'
        assert unquote_argument('"') == '"'

    def test_discords_embed_suppression_wrapper_comes_off_a_link(self) -> None:
        """`<link>` is how Discord sends a link whose embed the user suppressed,
        and the `>` otherwise rides into the path."""
        source = parse_input("<https://www.youtube.com/watch?v=dQw4w9WgXcQ>")
        assert isinstance(source, YTSource)
        assert source.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_angle_brackets_around_words_are_the_users_own(self) -> None:
        assert unquote_argument("<3 this song>") == "<3 this song>"
        assert unquote_argument("<>") == "<>"

    def test_unwrapping_is_safe_twice(self) -> None:
        once = unquote_argument("<https://youtu.be/x>")
        assert unquote_argument(once) == once == "https://youtu.be/x"


class TestTimestampWarning:
    """A `t=` that does not parse changes where the song starts, so the response
    has to say so — the seek is otherwise dropped with nothing on screen."""

    def test_none_when_the_timestamp_parsed(self) -> None:
        assert timestamp_warning(parse_url("https://youtu.be/a?t=1m30s")) is None

    def test_none_for_a_link_with_no_timestamp(self) -> None:
        assert timestamp_warning(parse_url("https://youtu.be/a")) is None

    def test_none_for_a_source_that_cannot_carry_one(self) -> None:
        """Spotify and SoundCloud have no `t=`; the helper takes the union type,
        so the isinstance narrowing is what keeps this from raising."""
        assert timestamp_warning(parse_url("https://soundcloud.com/a/b")) is None

    def test_names_the_value_and_the_accepted_forms(self) -> None:
        warning = timestamp_warning(parse_url("https://youtu.be/a?t=1h30"))
        assert warning is not None
        assert "1h30" in warning
        assert TIMESTAMP_FORMATS in warning

    def test_a_good_second_timestamp_suppresses_it(self) -> None:
        """`?t=bad&ts=90` does start where the user asked, so warning about it
        would be wrong."""
        assert timestamp_warning(parse_url("https://youtu.be/a?t=bad&ts=90")) is None

    def test_the_echoed_value_cannot_break_out_of_its_code_span(self) -> None:
        """The raw value is attacker-influenceable and is rendered inside
        backticks, so safe_label's backtick neutralization is load-bearing."""
        warning = timestamp_warning(parse_url("https://youtu.be/a?t=`x`[y](z)"))
        assert warning is not None
        assert "`x`" not in warning
        assert "[y](z)" not in warning


class TestStartOffsetFormats:
    def test_it_names_the_clock_form_the_flag_adds(self) -> None:
        """The two lists live beside the parser that accepts them; this one is
        wider by exactly the clock form."""
        assert "1:32" in START_OFFSET_FORMATS
        assert "1:32" not in TIMESTAMP_FORMATS
        for shape in ("90", "90s", "2h30m15s"):
            assert shape in START_OFFSET_FORMATS

    def test_every_named_shape_parses(self) -> None:
        """A format list that quotes a shape the parser rejects sends the user
        round in a circle."""
        for shape in re.findall(r"`([^`]+)`", START_OFFSET_FORMATS):
            assert parse_start_offset(shape) is not None, shape

    def test_the_bound_is_longer_than_any_song(self) -> None:
        assert MAX_START_OFFSET_SECS == 24 * 3600


_VIDEO = "dQw4w9WgXcQ"
_LIST = "PLwP_SiAcdui0KVebT0mU9Apz359a4ubsC"
_TRACK = "4uLU6hMCjMI75M1A2tKUQC"
_PLAYLIST = "37i9dQZF1DXcBWIGoYBM5M"


class Route(NamedTuple):
    """Where parse_input sends one input: the branch, the source's kind, what is
    handed on (a link, a `ytsearch:` term, or a Spotify id), the archive's
    query_source, and the ytdl:source key its resolve reads."""

    branch: URLSource
    kind: str
    target: str
    query_source: str
    cache_key: Optional[str]


def _route(text: str) -> Route:
    source = parse_input(text)
    if isinstance(source, SpotifySource):
        return Route(
            source.stype, source.type.value, source.id, QUERY_SOURCE_SPOTIFY, None
        )
    if isinstance(source, SoundcloudSource):
        return Route(
            source.stype,
            "track",
            source.url,
            QUERY_SOURCE_SOUNDCLOUD,
            _source_cache_key(source.url),
        )
    target = source.ytsearch or source.url or ""
    return Route(
        source.stype,
        source.type.value,
        target,
        source.query_source,
        _source_cache_key(target),
    )


def _youtube(link: str, kind: str = "track") -> Route:
    return Route(
        URLSource.YOUTUBE, kind, link, QUERY_SOURCE_YOUTUBE, f"ytdl:source:{link}"
    )


def _other(link: str, host: str) -> Route:
    return Route(URLSource.OTHER, "track", link, host, f"ytdl:source:{link}")


def _search(text: str) -> Route:
    return Route(
        URLSource.SEARCH,
        "track",
        f"ytsearch:{text}",
        QUERY_SOURCE_SEARCH,
        f"ytdl:source:ytsearch:{text.lower()}",
    )


def _spotify(kind: SpotifyType, spotify_id: str) -> Route:
    return Route(URLSource.SPOTIFY, kind.value, spotify_id, QUERY_SOURCE_SPOTIFY, None)


class TestRoutingUnchanged:
    """Inputs the linear parser routes exactly as the unanchored `re.search` it
    replaced did: branch, what is handed on, query_source and cache key.
    query_source is archived, so a row moving here shifts -analytics' buckets."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                f"https://www.youtube.com/watch?v={_VIDEO}",
                _youtube(f"https://www.youtube.com/watch?v={_VIDEO}"),
            ),
            (
                f"http://youtube.com/watch?v={_VIDEO}&t=30",
                _youtube(f"http://youtube.com/watch?v={_VIDEO}&t=30"),
            ),
            (
                f"https://youtu.be/{_VIDEO}?si=abc",
                _youtube(f"https://youtu.be/{_VIDEO}?si=abc"),
            ),
            (
                f"https://www.youtube.com/playlist?list={_LIST}",
                _youtube(f"https://www.youtube.com/playlist?list={_LIST}", "playlist"),
            ),
            (
                f"https://www.youtube.com/watch?v={_VIDEO}&list={_LIST}&index=4",
                _youtube(
                    f"https://www.youtube.com/watch?v={_VIDEO}&list={_LIST}&index=4",
                    "playlist",
                ),
            ),
            (
                f"https://www.youtube.com/shorts/{_VIDEO}",
                _youtube(f"https://www.youtube.com/shorts/{_VIDEO}"),
            ),
            # Hosts that are not special-cased queue one song, list= or not.
            (
                f"https://m.youtube.com/watch?v={_VIDEO}",
                _other(f"https://m.youtube.com/watch?v={_VIDEO}", "m.youtube.com"),
            ),
            (
                f"https://music.youtube.com/watch?v={_VIDEO}&list=RDAMVM{_VIDEO}",
                _other(
                    f"https://music.youtube.com/watch?v={_VIDEO}&list=RDAMVM{_VIDEO}",
                    "music.youtube.com",
                ),
            ),
            (
                f"https://www.youtube-nocookie.com/embed/{_VIDEO}",
                _other(
                    f"https://www.youtube-nocookie.com/embed/{_VIDEO}",
                    "youtube-nocookie.com",
                ),
            ),
            (
                f"https://open.spotify.com/track/{_TRACK}?si=1",
                _spotify(SpotifyType.TRACK, _TRACK),
            ),
            (
                f"https://spotify.com/playlist/{_PLAYLIST}",
                _spotify(SpotifyType.PLAYLIST, _PLAYLIST),
            ),
            (
                "https://soundcloud.com/artist/track?si=x",
                Route(
                    URLSource.SOUNDCLOUD,
                    "track",
                    "https://soundcloud.com/artist/track?si=x",
                    QUERY_SOURCE_SOUNDCLOUD,
                    "ytdl:source:https://soundcloud.com/artist/track?si=x",
                ),
            ),
            (
                "https://m.soundcloud.com/artist/track",
                _other("https://m.soundcloud.com/artist/track", "m.soundcloud.com"),
            ),
            (
                "https://on.soundcloud.com/AbCdE",
                _other("https://on.soundcloud.com/AbCdE", "on.soundcloud.com"),
            ),
            (
                "https://spotify.link/AbCdEfGhIjK",
                _other("https://spotify.link/AbCdEfGhIjK", "spotify.link"),
            ),
            (
                "https://vimeo.com/76979871",
                _other("https://vimeo.com/76979871", "vimeo.com"),
            ),
            (
                "https://my-site.com/a/b",
                _other("https://my-site.com/a/b", "my-site.com"),
            ),
            (
                "https://artist.bandcamp.com/track/x",
                _other("https://artist.bandcamp.com/track/x", "artist.bandcamp.com"),
            ),
            # A dotted name followed by `/` is a link, however unlikely.
            ("hello.world/", _other("hello.world/", "hello.world")),
            ("hello_world", _search("hello_world")),
            ("98/99", _search("98/99")),
            ("will.i.am", _search("will.i.am")),
            ("Dr.Dre", _search("Dr.Dre")),
            ("M.I.A.", _search("M.I.A.")),
            ("AC/DC", _search("AC/DC")),
            ("never gonna give you up", _search("never gonna give you up")),
            # No `/` after the host, or a port before it: not a link.
            ("https://youtube.com", _search("https://youtube.com")),
            ("https://example.com:8080/x", _search("https://example.com:8080/x")),
        ],
    )
    def test_route(self, text: str, expected: Route) -> None:
        assert _route(text) == expected


class TestRoutingChanged:
    """Every input the linear parser routes differently from the search it
    replaced. The README's upgrade note for this release lists the same rows."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # Any uppercase in the host: once OTHER, one song, a folded key.
            (
                f"https://WWW.YOUTUBE.COM/watch?v={_VIDEO}&list={_LIST}",
                _youtube(
                    f"https://www.youtube.com/watch?v={_VIDEO}&list={_LIST}", "playlist"
                ),
            ),
            (f"YOUTU.BE/{_VIDEO}", _youtube(f"youtu.be/{_VIDEO}")),
            (
                "https://SoundCloud.com/artist/track",
                Route(
                    URLSource.SOUNDCLOUD,
                    "track",
                    "https://soundcloud.com/artist/track",
                    QUERY_SOURCE_SOUNDCLOUD,
                    "ytdl:source:https://soundcloud.com/artist/track",
                ),
            ),
            # A scheme-less link: once case-folded into its cache key.
            (
                f"youtube.com/watch?v={_VIDEO}",
                _youtube(f"youtube.com/watch?v={_VIDEO}"),
            ),
            # Discord's wrappers: once handed to yt-dlp with the markup on.
            (f"<https://youtu.be/{_VIDEO}>", _youtube(f"https://youtu.be/{_VIDEO}")),
            (f"||https://youtu.be/{_VIDEO}||", _youtube(f"https://youtu.be/{_VIDEO}")),
            (f"`https://youtu.be/{_VIDEO}`", _youtube(f"https://youtu.be/{_VIDEO}")),
            (f"(https://youtu.be/{_VIDEO})", _youtube(f"https://youtu.be/{_VIDEO}")),
            (f"https://youtu.be/{_VIDEO}.", _youtube(f"https://youtu.be/{_VIDEO}")),
            # A masked link's text holds spaces: once a search for the markdown.
            (
                f"[the song](<https://youtu.be/{_VIDEO}>)",
                _youtube(f"https://youtu.be/{_VIDEO}"),
            ),
            # Spotify shapes: once a search for the URI, a bare Exception, or OTHER.
            (f"spotify:track:{_TRACK}", _spotify(SpotifyType.TRACK, _TRACK)),
            (
                f"https://open.spotify.com/intl-de/track/{_TRACK}",
                _spotify(SpotifyType.TRACK, _TRACK),
            ),
            (
                f"https://play.spotify.com/track/{_TRACK}",
                _spotify(SpotifyType.TRACK, _TRACK),
            ),
            # A share link: once one song, now the watch link it carries.
            (
                f"https://www.youtube.com/attribution_link?u=%2Fwatch%3Fv%3D{_VIDEO}%26list%3D{_LIST}",
                _youtube(
                    f"https://www.youtube.com/watch?v={_VIDEO}&list={_LIST}", "playlist"
                ),
            ),
            # A link that does not start the token: once handed to yt-dlp, which
            # fails every one of these.
            ("ftp://example.com/x", _search("ftp://example.com/x")),
            (
                f"//youtube.com/watch?v={_VIDEO}",
                _search(f"//youtube.com/watch?v={_VIDEO}"),
            ),
            ("https://user@example.com/x", _search("https://user@example.com/x")),
            (
                f"listen:https://youtu.be/{_VIDEO}",
                _search(f"listen:https://youtu.be/{_VIDEO}"),
            ),
            # A search holding a link: once kept its case in the cache key.
            (
                f"https://youtu.be/{_VIDEO} live",
                _search(f"https://youtu.be/{_VIDEO} live"),
            ),
        ],
    )
    def test_route(self, text: str, expected: Route) -> None:
        assert _route(text) == expected

    def test_youtu_be_names_the_video_a_playlist_timestamp_belongs_to(self) -> None:
        """`v=` never appears on youtu.be, so without the path its `t=` could
        never be applied to the playlist's first track."""
        result = parse_input(f"youtu.be/{_VIDEO}?list={_LIST}&t=30")
        assert isinstance(result, YTSource)
        assert (result.video_id, result.ts) == (_VIDEO, 30)


class TestAttributionLink:
    def test_the_inner_link_is_decoded_once(self) -> None:
        """An inner share link is routed as the youtube.com link it is, and never
        decoded again: nesting cannot buy another pass."""
        inner = "%2Fattribution_link%3Fu%3D%252Fwatch%253Fv%253DdQw4w9WgXcQ%2526list%253DPLx"
        result = parse_input(f"https://www.youtube.com/attribution_link?u={inner}")
        assert isinstance(result, YTSource)
        assert result.type is YTType.TRACK
        assert result.url == (
            "https://www.youtube.com/attribution_link?u=%2Fwatch%3Fv%3DdQw4w9WgXcQ%26list%3DPLx"
        )

    def test_an_absolute_inner_link_is_not_followed(self) -> None:
        """Only a path on youtube.com: a `u` naming another host stays the share
        link it arrived as."""
        link = (
            "https://www.youtube.com/attribution_link?u=https%3A%2F%2Fexample.com%2Fx"
        )
        assert _route(link) == _youtube(link)

    def test_a_protocol_relative_inner_link_is_not_followed(self) -> None:
        """`//host/x` starts with `/` but names a HOST, not the path this share
        link stands for, so it stays the link it arrived as."""
        link = "https://www.youtube.com/attribution_link?u=%2F%2Fexample.com%2Fx"
        assert _route(link) == _youtube(link)


class TestDiscordWrappers:
    @pytest.mark.parametrize(
        "wrapped",
        [
            f"<https://youtu.be/{_VIDEO}>",
            f"||<https://youtu.be/{_VIDEO}>||",
            f"[a song with spaces](https://youtu.be/{_VIDEO})",
            f"<https://youtu.be/{_VIDEO}>.",
            f"https://youtu.be/{_VIDEO}!",
            f'"<https://youtu.be/{_VIDEO}>"',
        ],
    )
    def test_the_link_inside_is_what_is_handed_on(self, wrapped: str) -> None:
        result = parse_input(wrapped)
        assert isinstance(result, YTSource)
        assert result.url == f"https://youtu.be/{_VIDEO}"

    @pytest.mark.parametrize(
        "text", ["<never gonna give you up>", "beethoven!", "[remix](lyrics)"]
    )
    def test_wrapped_words_are_searched_as_typed(self, text: str) -> None:
        assert parse_input(text) == YTSource(
            ytsearch=f"ytsearch:{text}",
            process=True,
            stype=URLSource.SEARCH,
            query_source=QUERY_SOURCE_SEARCH,
        )


class TestIsLink:
    @pytest.mark.parametrize(
        "text",
        [
            f"https://youtu.be/{_VIDEO}",
            f"youtu.be/{_VIDEO}",
            f"HTTPS://YOUTU.BE/{_VIDEO}",
            f"spotify:track:{_TRACK}",
            "https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy",
            "www.youtube.com/watch?v=aBcDeF",
            f"open.spotify.com/track/{_TRACK}",
        ],
    )
    def test_links(self, text: str) -> None:
        assert is_link(text)

    @pytest.mark.parametrize(
        "text",
        [
            "hello_world",
            "will.i.am",
            "98/99",
            "ytsearch:never gonna give you up",
            f"https://youtu.be/{_VIDEO} live",
            "spotify:wrapped",
            "https://" + "a." * (LINK_MAX_CHARS // 2) + "com/",
            # A slash in a title is not a path: both of these are real searches.
            "AC/DC Back in Black",
            "24/7 lofi radio",
            "",
        ],
    )
    def test_not_links(self, text: str) -> None:
        assert not is_link(text)


def _fastest_of_three(fn: Callable[[str], object], token: str) -> float:
    """Seconds for the fastest of three calls, so a scheduler stall on a loaded
    runner does not read as the parser's cost."""
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        try:
            fn(token)
        except ValueError:
            pass
        best = min(best, time.perf_counter() - started)
    return best


class TestLinkParsingIsLinear:
    """`re` holds the GIL for a whole match, so a slow one stalls discord.py's
    audio thread in every guild. An unanchored search over a long token with no
    `/` measured 30ms at 2,048 characters and 140ms at 4,000; the anchored test
    measures under 0.2ms. Tokens at the link cap reach the regex; the longer ones
    must not."""

    @pytest.mark.parametrize(
        "token",
        [
            "a" * LINK_MAX_CHARS,
            "www." * (LINK_MAX_CHARS // 4),
            "a." * (LINK_MAX_CHARS // 2),
            "-" * LINK_MAX_CHARS,
            "https://" + "a." * ((LINK_MAX_CHARS - 8) // 2),
            "a" * 3990,
            "www." * 997,
            "a." * 1995,
            "-" * 3990,
            "spotify:user:" + "a" * 3977,
            "<" * 1995 + ">" * 1995,
        ],
        ids=lambda token: f"{token[:8]}…x{len(token)}",
    )
    @pytest.mark.parametrize(
        "fn", [parse_input, parse_url, is_link], ids=lambda fn: fn.__name__
    )
    def test_an_adversarial_token_parses_in_under_5ms(
        self, token: str, fn: Callable[[str], object]
    ) -> None:
        assert _fastest_of_three(fn, token) < 0.005

    def test_a_token_past_the_cap_is_searched(self) -> None:
        link = f"https://www.youtube.com/watch?v={_VIDEO}&x={'a' * LINK_MAX_CHARS}"
        assert parse_input(link).stype is URLSource.SEARCH
