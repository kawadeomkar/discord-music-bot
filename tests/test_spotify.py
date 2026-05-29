"""Tests for src/spotify.py — Spotify API auth and response parsing."""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.spotify import Spotify

# alru_cache binds to the first event loop it sees; the async test classes
# use session-scope so all tests share one loop and avoid RuntimeError.


@pytest.fixture
def mock_auth_response():
    return {"access_token": "test_access_token_xyz", "expires_in": 3600}


@pytest.fixture
def spotify():
    """Spotify instance — no blocking auth call at construction time."""
    with patch.dict(
        "os.environ",
        {"SPOTIFY_CLIENT_ID": "test_id", "SPOTIFY_CLIENT_SECRET": "test_secret"},
    ):
        return Spotify()


def _make_mock_session(resp):
    """Return an aiohttp.ClientSession mock wired to return resp from .post() and .request()."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = AsyncMock(return_value=resp)
    session.request = AsyncMock(return_value=resp)
    return session


class TestSpotifyRefreshToken:

    @pytest.mark.asyncio
    async def test_refresh_token_sets_auth_token(self, spotify, mock_auth_response):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)

        with patch("src.spotify.aiohttp.ClientSession", return_value=_make_mock_session(mock_resp)):
            await spotify._refresh_token()

        assert spotify.auth_token == "test_access_token_xyz"

    @pytest.mark.asyncio
    async def test_refresh_token_sends_client_credentials_grant(self, spotify, mock_auth_response):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)

        with patch("src.spotify.aiohttp.ClientSession", return_value=mock_session):
            await spotify._refresh_token()

        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["data"]["grant_type"] == "client_credentials"
        assert call_kwargs["data"]["client_id"] == "test_id"
        assert call_kwargs["data"]["client_secret"] == "test_secret"

    @pytest.mark.asyncio
    async def test_refresh_token_sets_token_expiry_in_future(self, spotify, mock_auth_response):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)

        with patch("src.spotify.aiohttp.ClientSession", return_value=_make_mock_session(mock_resp)):
            await spotify._refresh_token()

        assert spotify.token_expiry > time.time()

    def test_str_returns_auth_token(self, spotify):
        spotify.auth_token = "my_token"
        assert str(spotify) == "my_token"


class TestSpotifyTrack:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_track_combines_name_and_artists(self, spotify):
        mock_response = {
            "name": "Bohemian Rhapsody",
            "artists": [{"name": "Queen"}],
        }
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)):
            result = await spotify.track("some_track_id")

        assert result == "Bohemian Rhapsody Queen"

    async def test_track_with_multiple_artists(self, spotify):
        mock_response = {
            "name": "Collaboration Track",
            "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
        }
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)):
            result = await spotify.track("multi_artist_id")

        assert result == "Collaboration Track Artist A Artist B"

    async def test_track_calls_correct_endpoint(self, spotify):
        mock_response = {"name": "Song", "artists": [{"name": "Artist"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)) as mock_call:
            await spotify.track("abc123")

        called_endpoint = mock_call.call_args[0][0]
        assert "v1/tracks/abc123" in called_endpoint


class TestSpotifyPlaylist:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

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
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)):
            result = await spotify.playlist("playlist_id_123")

        assert len(result) == 2
        assert result[0] == "Track One Artist X"
        assert result[1] == "Track Two Artist Y"

    async def test_playlist_empty_items_returns_empty_list(self, spotify):
        mock_response = {"items": []}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)):
            result = await spotify.playlist("empty_playlist_id")

        assert result == []

    async def test_playlist_calls_correct_endpoint(self, spotify):
        mock_response = {"items": []}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)) as mock_call:
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
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_response)):
            result = await spotify.playlist("pid")

        assert result[0] == "Collab A B C"


class TestSpotifyHttpCall:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_http_call_raises_on_non_200(self, spotify):
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session = _make_mock_session(mock_response)

        with patch("src.spotify.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="stat: 404"):
                await spotify.http_call("https://api.spotify.com/v1/tracks/bad")

    async def test_http_call_sets_authorization_header(self, spotify):
        spotify.auth_token = "valid_token"
        spotify.token_expiry = time.time() + 3600

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "ok"})
        mock_session = _make_mock_session(mock_response)

        with patch("src.spotify.aiohttp.ClientSession", return_value=mock_session):
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

        with patch("src.spotify.aiohttp.ClientSession", return_value=mock_session), \
             patch.object(spotify, "_refresh_token", new=AsyncMock()) as mock_refresh:
            await spotify.http_call("https://api.spotify.com/v1/tracks/xyz")

        mock_refresh.assert_called_once()
