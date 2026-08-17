"""Tests for src/sources.py — URL parsing and source type detection."""

import pytest

from src.guild_state import Analytics
from src.sources import (
    unquote_argument,
    QUERY_SOURCE_SEARCH,
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
    parse_url,
    query_source_of,
    spotify_titles_to_ytsearch,
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


class TestParseUrlPlaylistIndex:
    """`index=` is YouTube's 1-based position of the video the link was copied
    at. Parsed only on the playlist branch and never allowed to raise — a
    ValueError out of parse_url means "not a URL" and searches for the link text.
    """

    def test_index_is_parsed_from_a_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=abc&list=PLtest&index=4"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.index == 4
        assert result.video_id == "abc"

    def test_index_is_none_when_absent(self) -> None:
        url = "https://www.youtube.com/playlist?list=PLtest"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.index is None
        assert result.video_id is None

    def test_index_is_not_carried_by_a_bare_track(self) -> None:
        """No list, no playlist — the index has nothing to index into."""
        url = "https://www.youtube.com/watch?v=abc&index=4"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.TRACK
        assert result.index is None

    @pytest.mark.parametrize("raw", ["0", "-2", "abc", "4.5"])
    def test_unusable_index_parses_as_none_not_an_error(self, raw: str) -> None:
        """A malformed index degrades to "no index" instead of raising: the
        alternative sends the whole link to ytsearch as plain text."""
        url = f"https://www.youtube.com/watch?v=abc&list=PLtest&index={raw}"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.type == YTType.PLAYLIST
        assert result.list_id == "PLtest"
        assert result.index is None


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

    def test_spotify_album(self) -> None:
        # The incident URL: this exact
        # link used to raise "Unknown Spotify track type: ['album', …]".
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.ALBUM
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"
        assert result.stype == URLSource.SPOTIFY
        assert result.process is True

    def test_spotify_album_without_open_subdomain(self) -> None:
        url = "https://spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.ALBUM
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"

    def test_spotify_album_with_si_param(self) -> None:
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl?si=abc123"
        result = parse_url(url, f"-play {url}")
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
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type is expected_type
        assert result.id == "6WgSCcRfaXuBVfM2TpV0Kl"

    def test_intl_prefixed_unsupported_type_still_raises(self) -> None:
        """The locale strip must expose the real type, not blindly accept."""
        url = "https://open.spotify.com/intl-fr/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url, f"-play {url}")
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
            parse_url(url, f"-play {url}")

    def test_unknown_spotify_type_raises(self) -> None:
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url, f"-play {url}")
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
            parse_url(url, f"-play {url}")
        assert exc_info.value.user_message == str(exc_info.value)

    def test_unknown_spotify_type_is_not_a_value_error(self) -> None:
        """Regression guard: parse_input catches ValueError and falls back to a
        YouTube search. If this error ever becomes a ValueError, an /artist/
        link silently turns into `ytsearch:https://open.spotify.com/...`."""
        url = "https://open.spotify.com/show/4rOoJ6Egrf8K2IrywzwOMk"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url, f"-play {url}")
        assert not isinstance(exc_info.value, ValueError)

    def test_unknown_spotify_type_suppresses_exception_chain(self) -> None:
        """`from None`: the enum-lookup ValueError is an implementation detail;
        chaining it doubles the traceback in every error log."""
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError) as exc_info:
            parse_url(url, f"-play {url}")
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


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

    def test_unsupported_spotify_link_propagates(self) -> None:
        """The fallback-eats-the-error regression guard: an /artist/ link must
        surface UnsupportedSpotifyLinkError to the caller, not fall back to a
        `ytsearch:https://…` search the way a ValueError would."""
        url = "https://open.spotify.com/artist/1dfeR4HaWDbWqFHLkxsg1d"
        with pytest.raises(UnsupportedSpotifyLinkError):
            parse_input(url, f"-play {url}")

    def test_spotify_album_url_through_parse_input(self) -> None:
        url = "https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl"
        result = parse_input(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert result.type == SpotifyType.ALBUM


# A requester that could not be mistaken for a default, so an assertion that the
# ID survived the queue cannot pass against a 0 or a None.
_REQUESTER_ID = 424242424242424242
_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=3)
_ORIGIN = "https://open.spotify.com/album/abc123"


class TestSpotifyTitlesToYTSearch:
    def test_converts_titles_to_ytsearch(self) -> None:
        titles = ["Never Gonna Give You Up Rick Astley", "Bohemian Rhapsody Queen"]
        result = spotify_titles_to_ytsearch(
            titles, _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
        )

        assert len(result) == 2
        assert all(isinstance(r, YTSource) for r in result)
        assert result[0].ytsearch == "ytsearch:Never Gonna Give You Up Rick Astley"
        assert result[1].ytsearch == "ytsearch:Bohemian Rhapsody Queen"

    def test_all_results_have_process_true(self) -> None:
        titles = ["Song A", "Song B", "Song C"]
        result = spotify_titles_to_ytsearch(
            titles, _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert all(r.process is True for r in result)

    def test_empty_list_returns_empty(self) -> None:
        assert (
            spotify_titles_to_ytsearch(
                [], _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
            )
            == []
        )

    def test_single_title(self) -> None:
        result = spotify_titles_to_ytsearch(
            ["Only Song Artist"], _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert len(result) == 1
        assert result[0].ytsearch == "ytsearch:Only Song Artist"

    def test_url_field_is_none(self) -> None:
        result = spotify_titles_to_ytsearch(
            ["Song"], _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert result[0].url is None

    def test_stamps_requester_on_every_track(self) -> None:
        """The whole point: a collection's tracks resolve minutes to an hour after
        the command returned, so the requester has to travel with them."""
        result = spotify_titles_to_ytsearch(
            ["A", "B", "C"], _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert [r.requester_id for r in result] == [_REQUESTER_ID] * 3

    def test_requester_is_required(self) -> None:
        """Guards the design decision, not the behaviour: defaulting this parameter
        is what silently re-attributes a whole album to the last person who typed a
        command. A new call site must be forced to supply one."""
        with pytest.raises(TypeError):
            spotify_titles_to_ytsearch(  # pyright: ignore[reportCallIssue]
                ["Song"], analytics=_ANALYTICS, origin=_ORIGIN
            )

    def test_per_track_positions_derive_from_the_head(self) -> None:
        # The head's analytics fans out: same ask-time queued_at on every track,
        # positions incrementing from the head's — a playlist behind 3 songs
        # waits at 3, 4, 5.
        result = spotify_titles_to_ytsearch(
            ["a", "b", "c"], _REQUESTER_ID, analytics=_ANALYTICS, origin=_ORIGIN
        )
        assert [r.analytics.queue_position for r in result] == [3, 4, 5]
        assert all(r.analytics.queued_at == 1752530000.5 for r in result)


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
            # parse_url's domain group is `[\w+|\.]+`, so these three really can
            # reach the normalizer — it filters, it does not merely format.
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
        result = parse_input("never gonna give you up", "-play never gonna give you up")
        assert result.stype == URLSource.SEARCH
        assert query_source_of(result) == QUERY_SOURCE_SEARCH

    def test_youtube_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert query_source_of(parse_url(url, f"-play {url}")) == QUERY_SOURCE_YOUTUBE

    def test_youtu_be_collapses_onto_the_service(self) -> None:
        """A shortener is not a different service."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert query_source_of(parse_url(url, f"-play {url}")) == QUERY_SOURCE_YOUTUBE

    def test_youtube_playlist(self) -> None:
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp"
        assert query_source_of(parse_url(url, f"-play {url}")) == QUERY_SOURCE_YOUTUBE

    def test_spotify_track_link(self) -> None:
        url = "https://open.spotify.com/track/5WZD6jHtgSSAGK97diNG7y"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, SpotifySource)
        assert query_source_of(result) == QUERY_SOURCE_SPOTIFY

    def test_soundcloud_link(self) -> None:
        url = "https://soundcloud.com/artist/track"
        result = parse_url(url, f"-play {url}")
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
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert query_source_of(result) == expected

    def test_spotify_collection_tracks_are_stamped_spotify(self) -> None:
        """The whole reason the token is captured at parse time: these resolve to
        YouTube URLs at dequeue, so nothing downstream could recover it."""
        sources = spotify_titles_to_ytsearch(
            ["song one", "song two"],
            _REQUESTER_ID,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )
        assert [query_source_of(s) for s in sources] == [QUERY_SOURCE_SPOTIFY] * 2

    def test_uppercase_www_youtube_still_reports_youtube(self) -> None:
        """parse_url's `www\\.` group is case-sensitive, so this misses the
        YouTube branch and lands on the generic one — the normalizer still names
        the service correctly."""
        url = "https://WWW.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_url(url, f"-play {url}")
        assert isinstance(result, YTSource)
        assert result.stype == URLSource.OTHER
        assert query_source_of(result) == QUERY_SOURCE_YOUTUBE

    def test_unstamped_ytsource_is_unknown(self) -> None:
        """A hand-built source (crash recovery, tests, a future call site) reports
        the unknown sentinel rather than guessing."""
        assert query_source_of(YTSource(ytsearch="ytsearch:x")) == ""


class TestQuotedArgumentsSurviveConsumeRest:
    """`-play`/`-playnow` take consume-rest arguments, and discord.py's read_rest
    does no quote handling where the positional parser's get_quoted_word did. So
    the quotes started arriving as part of the value."""

    def test_a_quoted_url_still_parses_as_that_url(self) -> None:
        """parse_url uses re.search, so a quoted URL still matched the domain while
        dragging the trailing quote into the path — yt-dlp then rejects it."""
        source = parse_input(
            '"https://www.youtube.com/watch?v=dQw4w9WgXcQ"',
            '-play "https://www.youtube.com/watch?v=dQw4w9WgXcQ"',
        )
        assert isinstance(source, YTSource)
        assert source.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_a_quoted_search_does_not_keep_its_quotes(self) -> None:
        """The origin is what -remove matches on, so a quoted search meant the
        obvious retype (`-remove some song`) matched nothing."""
        source = parse_input('"some song"', '-play "some song"')
        assert isinstance(source, YTSource)
        assert source.ytsearch == "ytsearch:some song"

    def test_an_unmatched_quote_is_left_alone(self) -> None:
        """Only a whole argument wrapped at BOTH ends is a wrapper; anything else
        is text the user typed."""
        source = parse_input('say "hello', '-play say "hello')
        assert isinstance(source, YTSource)
        assert source.ytsearch == 'ytsearch:say "hello'

    def test_a_bare_quote_pair_is_not_stripped_to_nothing(self) -> None:
        assert unquote_argument('""') == '""'
        assert unquote_argument('"') == '"'
