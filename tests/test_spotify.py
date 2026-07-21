"""Tests for src/spotify.py — Spotify API auth, response parsing, and Redis cache."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.spotify import Spotify


@pytest.fixture
def mock_auth_response():
    return {"access_token": "test_access_token_xyz", "expires_in": 3600}


def _make_mock_session(resp):
    """Return a session mock wired to return resp from .post() and .request()."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = AsyncMock(return_value=resp)
    session.request = AsyncMock(return_value=resp)
    return session


def _make_session_factory(resp):
    """Return a session_factory callable that produces a mock session."""
    mock_session = _make_mock_session(resp)
    return lambda **kw: mock_session, mock_session


class TestSpotifyRefreshToken:
    async def test_refresh_token_sets_auth_token(self, spotify, mock_auth_response):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()
        assert spotify.auth_token == "test_access_token_xyz"

    async def test_refresh_token_sends_client_credentials_grant(
        self, spotify, mock_auth_response
    ):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()

        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["data"]["grant_type"] == "client_credentials"
        assert call_kwargs["data"]["client_id"] == "test_id"
        assert call_kwargs["data"]["client_secret"] == "test_secret"

    async def test_refresh_token_sets_token_expiry_in_future(
        self, spotify, mock_auth_response
    ):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()
        assert spotify.token_expiry > time.time()

    async def test_refresh_token_uses_redis_cache_on_hit(self, spotify, fake_redis):
        """When Redis holds a valid token, _refresh_token returns it without calling the API."""
        await fake_redis.set("spotify:auth:token", b"cached_bearer_token", ex=120)

        factory_calls: list = []
        spotify._session_factory = lambda **kw: factory_calls.append(1)

        await spotify._refresh_token()

        assert spotify.auth_token == "cached_bearer_token"
        assert factory_calls == []  # session factory never called

    async def test_refresh_token_sets_expiry_from_real_ttl(self, spotify, fake_redis):
        """token_expiry should reflect the key's actual remaining TTL, not a flat guess."""
        await fake_redis.set("spotify:auth:token", b"cached_bearer_token", ex=120)

        before = time.time()
        await spotify._refresh_token()

        assert 115 <= spotify.token_expiry - before <= 121

    async def test_refresh_token_falls_through_on_expired_key(
        self, spotify, fake_redis, mock_auth_response
    ):
        """A cached key with no remaining TTL (already expired but not yet
        evicted) must not be trusted — fall through to a fresh HTTP fetch."""
        await fake_redis.set("spotify:auth:token", b"stale_bearer_token")
        await fake_redis.persist("spotify:auth:token")  # ensure no TTL is set

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()

        assert spotify.auth_token == "test_access_token_xyz"

    async def test_refresh_token_writes_to_redis_on_api_call(
        self, spotify, fake_redis, mock_auth_response
    ):
        """On a Redis cache miss, _refresh_token fetches from Spotify and writes to Redis."""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()

        stored = await fake_redis.get("spotify:auth:token")
        assert stored == b"test_access_token_xyz"

    async def test_refresh_token_without_redis_calls_api(self, mock_auth_response):
        """Spotify instance with redis=None always calls the Spotify API."""
        from src.spotify import Spotify

        with patch.dict(
            "os.environ",
            {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"},
        ):
            sp = Spotify(redis=None)

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        sp._session_factory = lambda **kw: mock_session

        await sp._refresh_token()

        assert sp.auth_token == "test_access_token_xyz"
        mock_session.post.assert_awaited_once()

    def test_str_returns_auth_token(self, spotify):
        spotify.auth_token = "my_token"
        assert str(spotify) == "my_token"


class TestSpotifyTrack:
    async def test_track_combines_name_and_artists(self, spotify):
        mock_response = {
            "name": "Bohemian Rhapsody",
            "artists": [{"name": "Queen"}],
        }
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.track("some_track_id")

        assert result == "Bohemian Rhapsody Queen"

    async def test_track_with_multiple_artists(self, spotify):
        mock_response = {
            "name": "Collaboration Track",
            "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
        }
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.track("multi_artist_id")

        assert result == "Collaboration Track Artist A Artist B"

    async def test_track_calls_correct_endpoint(self, spotify):
        mock_response = {"name": "Song", "artists": [{"name": "Artist"}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ) as mock_call:
            await spotify.track("abc123")

        called_endpoint = mock_call.call_args[0][0]
        assert "v1/tracks/abc123" in called_endpoint


class TestSpotifyPlaylist:
    async def test_playlist_returns_list_of_titles(self, spotify):
        mock_response = {
            "items": [
                {
                    "track": {
                        "name": "Track One",
                        "artists": [{"name": "Artist X"}],
                    }
                },
                {
                    "track": {
                        "name": "Track Two",
                        "artists": [{"name": "Artist Y"}],
                    }
                },
            ]
        }
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.playlist("playlist_id_123")

        assert len(result) == 2
        assert result[0] == "Track One Artist X"
        assert result[1] == "Track Two Artist Y"

    async def test_playlist_empty_items_returns_empty_list(self, spotify):
        mock_response = {"items": []}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.playlist("empty_playlist_id")

        assert result == []

    async def test_playlist_calls_correct_endpoint(self, spotify):
        mock_response = {"items": []}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ) as mock_call:
            await spotify.playlist("pl_abc")

        called_endpoint = mock_call.call_args[0][0]
        assert "v1/playlists/pl_abc/tracks" in called_endpoint

    async def test_playlist_multi_artist_track(self, spotify):
        mock_response = {
            "items": [
                {
                    "track": {
                        "name": "Collab",
                        "artists": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                    }
                }
            ]
        }
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.playlist("pid")

        assert result[0] == "Collab A B C"


class TestSpotifyHttpCall:
    async def test_http_call_raises_on_non_200(self, spotify):
        spotify.auth_token = "prefetched_token"
        spotify.token_expiry = time.time() + 3600  # skip _refresh_token
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session = _make_mock_session(mock_response)
        spotify._session_factory = lambda **kw: mock_session

        with pytest.raises(Exception, match="stat: 404"):
            await spotify.http_call("https://api.spotify.com/v1/tracks/bad")

    async def test_http_call_sets_authorization_header(self, spotify):
        spotify.auth_token = "valid_token"
        spotify.token_expiry = time.time() + 3600

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "ok"})
        mock_session = _make_mock_session(mock_response)
        spotify._session_factory = lambda **kw: mock_session

        await spotify.http_call("https://api.spotify.com/v1/tracks/xyz")

        call_kwargs = mock_session.request.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer valid_token"

    async def test_http_call_refreshes_expired_token(self, spotify):
        spotify.token_expiry = time.time() - 1  # force expiry

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "ok"})
        mock_session = _make_mock_session(mock_response)
        spotify._session_factory = lambda **kw: mock_session

        with patch.object(spotify, "_refresh_token", new=AsyncMock()) as mock_refresh:
            await spotify.http_call("https://api.spotify.com/v1/tracks/xyz")

        mock_refresh.assert_called_once()


class TestSpotifyRedisCache:
    async def test_track_cache_hit_skips_http(self, spotify):
        """Second call returns cached value without hitting http_call."""
        with patch.object(
            spotify,
            "http_call",
            new=AsyncMock(
                return_value={"name": "Song", "artists": [{"name": "Artist"}]}
            ),
        ) as mock_call:
            await spotify.track("tid_cache1")
            await spotify.track("tid_cache1")  # second call — cache hit
        mock_call.assert_called_once()

    async def test_playlist_cache_hit_skips_http(self, spotify):
        mock_resp = {"items": [{"track": {"name": "T", "artists": [{"name": "A"}]}}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_resp)
        ) as m:
            await spotify.playlist("pid_cache1")
            await spotify.playlist("pid_cache1")
        m.assert_called_once()

    async def test_track_ttl_is_24h(self, spotify, fake_redis):
        with patch.object(
            spotify,
            "http_call",
            new=AsyncMock(return_value={"name": "S", "artists": [{"name": "A"}]}),
        ):
            await spotify.track("ttl_test_track")
        ttl = await fake_redis.ttl("spotify:track:ttl_test_track")
        assert 86390 <= ttl <= 86400

    async def test_playlist_ttl_is_1h(self, spotify, fake_redis):
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value={"items": []})
        ):
            await spotify.playlist("ttl_test_playlist")
        ttl = await fake_redis.ttl("spotify:playlist:ttl_test_playlist")
        assert 3590 <= ttl <= 3600

    async def test_cache_graceful_when_no_redis(self, fake_redis):
        """Spotify without Redis still works via network."""
        from unittest.mock import patch as p

        with p.dict(
            "os.environ", {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"}
        ):
            s = Spotify(redis=None)
        with patch.object(
            s,
            "http_call",
            new=AsyncMock(return_value={"name": "S", "artists": [{"name": "A"}]}),
        ):
            result = await s.track("no_redis")
        assert result == "S A"


class TestSpotifyArtists:
    async def test_single_artist_id_as_string(self, spotify):
        mock_resp = {"artists": [{"name": "Test Artist", "id": "1"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.artists("artist_id_1")
        assert result == mock_resp["artists"]

    async def test_multiple_artist_ids_as_list(self, spotify):
        mock_resp = {"artists": [{"name": "A"}, {"name": "B"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.artists(["id1", "id2"])
        assert len(result) == 2

    async def test_cache_hit_skips_http(self, spotify):
        mock_resp = {"artists": [{"name": "A"}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_resp)
        ) as m:
            await spotify.artists("aid1")
            await spotify.artists("aid1")
        m.assert_called_once()

    async def test_ttl_is_24h(self, spotify, fake_redis):
        mock_resp = {"artists": [{"name": "A"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            await spotify.artists("ttl_aid")
        ttl = await fake_redis.ttl("spotify:artist:ttl_aid")
        assert 86390 <= ttl <= 86400


class TestSpotifyAlbums:
    async def test_single_album_id(self, spotify):
        mock_resp = {"albums": [{"name": "Test Album"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.albums("album_id_1")
        assert result == mock_resp["albums"]

    async def test_multiple_album_ids(self, spotify):
        mock_resp = {"albums": [{"name": "A"}, {"name": "B"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.albums(["alb1", "alb2"])
        assert len(result) == 2

    async def test_cache_hit_skips_http(self, spotify):
        mock_resp = {"albums": [{"name": "Album A"}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_resp)
        ) as m:
            await spotify.albums("alb_cache")
            await spotify.albums("alb_cache")
        m.assert_called_once()

    async def test_ttl_is_24h(self, spotify, fake_redis):
        mock_resp = {"albums": [{"name": "A"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            await spotify.albums("ttl_alb")
        ttl = await fake_redis.ttl("spotify:album:ttl_alb")
        assert 86390 <= ttl <= 86400

    async def test_sorted_cache_key_for_multiple_ids(self, spotify, fake_redis):
        mock_resp = {"albums": []}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            await spotify.albums(["zid", "aid"])
        cached = await fake_redis.get("spotify:album:aid,zid")
        assert cached is not None
