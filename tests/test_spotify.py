"""Tests for src/spotify.py — Spotify API auth, response parsing, and Redis cache."""

import redis.asyncio as aioredis
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from redis.asyncio import Redis

from src.spotify import (
    _HTTP_TIMEOUT,
    _MAX_RETRY_AFTER_SECS,
    Spotify,
    SpotifyAuthError,
    SpotifyRateLimitError,
    SpotifyRequestError,
)


@pytest.fixture
def mock_auth_response() -> dict[str, Any]:
    return {"access_token": "test_access_token_xyz", "expires_in": 3600}


def _request_cm(resp: Any) -> MagicMock:
    """Stand in for aiohttp's _RequestContextManager.

    Releases on __aexit__, as ClientResponse.__aexit__ does — which is what makes
    `async with` equivalent to a hand-rolled release, and what lets a test assert
    the body was drained without knowing which of the two the code used.
    """
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)

    async def _exit(*_: Any) -> bool:
        await resp.release()
        return False

    cm.__aexit__ = AsyncMock(side_effect=_exit)
    return cm


def _make_mock_session(resp: AsyncMock) -> MagicMock:
    """Return a session mock wired to return resp from .post() and .request()."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(return_value=_request_cm(resp))
    session.request = MagicMock(return_value=_request_cm(resp))
    # A bare MagicMock attribute is truthy, which reads as an already-closed
    # session and makes _session_or_create rebuild on every call.
    session.closed = False
    session.close = AsyncMock()
    return session


def _make_session_factory(resp: AsyncMock) -> tuple[Any, MagicMock]:
    """Return a session_factory callable that produces a mock session."""
    mock_session = _make_mock_session(resp)
    return lambda **kw: mock_session, mock_session


class TestSpotifyRefreshToken:
    async def test_refresh_token_sets_auth_token(
        self, spotify: Spotify, mock_auth_response: dict[str, Any]
    ) -> None:
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()
        assert spotify.auth_token == "test_access_token_xyz"

    async def test_refresh_token_sends_client_credentials_grant(
        self, spotify: Spotify, mock_auth_response: dict[str, Any]
    ) -> None:
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
        self, spotify: Spotify, mock_auth_response: dict[str, Any]
    ) -> None:
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()
        assert spotify.token_expiry > time.time()

    async def test_refresh_token_uses_redis_cache_on_hit(
        self, spotify: Spotify, fake_redis: aioredis.Redis
    ) -> None:
        """When Redis holds a valid token, _refresh_token returns it without calling the API."""
        await fake_redis.set("spotify:auth:token", b"cached_bearer_token", ex=120)

        factory_calls: list = []
        spotify._session_factory = lambda **kw: factory_calls.append(1)

        await spotify._refresh_token()

        assert spotify.auth_token == "cached_bearer_token"
        assert factory_calls == []  # session factory never called

    async def test_refresh_token_sets_expiry_from_real_ttl(
        self, spotify: Spotify, fake_redis: aioredis.Redis
    ) -> None:
        """token_expiry should reflect the key's actual remaining TTL, not a flat guess."""
        await fake_redis.set("spotify:auth:token", b"cached_bearer_token", ex=120)

        before = time.time()
        await spotify._refresh_token()

        assert 115 <= spotify.token_expiry - before <= 121

    async def test_refresh_token_falls_through_on_expired_key(
        self,
        spotify: Spotify,
        fake_redis: aioredis.Redis,
        mock_auth_response: dict[str, Any],
    ) -> None:
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
        self,
        spotify: Spotify,
        fake_redis: aioredis.Redis,
        mock_auth_response: dict[str, Any],
    ) -> None:
        """On a Redis cache miss, _refresh_token fetches from Spotify and writes to Redis."""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token()

        stored = await fake_redis.get("spotify:auth:token")
        assert stored == b"test_access_token_xyz"

    async def test_refresh_token_without_redis_calls_api(
        self, mock_auth_response: dict[str, Any]
    ) -> None:
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
        mock_session.post.assert_called_once()

    async def test_use_cache_false_bypasses_redis_and_hits_api(
        self,
        spotify: Spotify,
        fake_redis: aioredis.Redis,
        mock_auth_response: dict[str, Any],
    ) -> None:
        """validate() relies on use_cache=False to test the real credentials: a
        Redis-cached token must be ignored and a fresh auth call made."""
        await fake_redis.set("spotify:auth:token", b"cached_bearer_token", ex=120)

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        mock_session = _make_mock_session(mock_resp)
        spotify._session_factory = lambda **kw: mock_session

        await spotify._refresh_token(use_cache=False)

        assert spotify.auth_token == "test_access_token_xyz"  # fresh, not cached
        mock_session.post.assert_called_once()

    def test_str_never_exposes_the_bearer_token(self, spotify: Spotify) -> None:
        """Both dunders, since an exception repr reaches __repr__, not __str__."""
        spotify.auth_token = "super_secret_bearer_token"
        assert "super_secret_bearer_token" not in str(spotify)
        assert "super_secret_bearer_token" not in repr(spotify)
        assert "super_secret_bearer_token" not in f"{spotify}"
        assert "super_secret_bearer_token" not in f"{spotify!r}"

    def test_str_reports_token_presence_without_the_value(
        self, spotify: Spotify
    ) -> None:
        spotify.auth_token = ""
        assert "token=unset" in str(spotify)
        spotify.auth_token = "anything"
        assert "token=set" in str(spotify)

    def test_str_identifies_the_client_without_the_secret(
        self, spotify: Spotify
    ) -> None:
        """A truncated client_id tells two configs apart; the secret never renders."""
        assert "test_i" in str(spotify)
        assert spotify.client_secret is not None
        assert spotify.client_secret not in str(spotify)

    def test_str_handles_missing_client_id(self, spotify: Spotify) -> None:
        spotify.client_id = None
        assert "unset" in str(spotify)  # must not raise on None


def _make_split_session(post_resp: AsyncMock, request_resp: AsyncMock) -> MagicMock:
    """Session mock whose auth POST and API request return different responses —
    needed by validate(), which grants a token then fetches a track."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(return_value=_request_cm(post_resp))
    session.request = MagicMock(return_value=_request_cm(request_resp))
    return session


def _resp(status: int, payload: dict[str, Any]) -> AsyncMock:
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    return resp


class TestSpotifyValidate:
    """validate() is the startup credential probe: it forces a fresh token and
    fetches a known track. It raises SpotifyAuthError only when Spotify rejects
    the credentials; every other failure surfaces as its own (non-auth) type."""

    async def test_validate_succeeds_with_valid_credentials(
        self, spotify: Spotify
    ) -> None:
        # One resp serves both the auth POST and the track GET (validate reads
        # access_token/expires_in from the first and name from the second).
        resp = _resp(
            200,
            {
                "access_token": "tok",
                "expires_in": 3600,
                "name": "Never Gonna Give You Up",
                "artists": [{"name": "Rick Astley"}],
            },
        )
        session = _make_mock_session(resp)
        spotify._session_factory = lambda **kw: session

        await spotify.validate("4PTG3Z6ehGkBFwjybzWkR8")  # must not raise

        session.request.assert_called_once()

    async def test_validate_raises_auth_error_on_rejected_grant(
        self, spotify: Spotify
    ) -> None:
        """Invalid client_id/secret: the token grant returns non-2xx, which
        strict=True turns into SpotifyAuthError before the track call is reached."""
        resp = _resp(400, {"error": "invalid_client"})
        session = _make_mock_session(resp)
        spotify._session_factory = lambda **kw: session

        with pytest.raises(SpotifyAuthError) as exc:
            await spotify.validate("4PTG3Z6ehGkBFwjybzWkR8")
        assert exc.value.status == 400
        session.request.assert_not_called()  # never got to the track call

    async def test_a_rejected_grant_releases_the_body(self, spotify: Spotify) -> None:
        """The grant shares the client's session now, so raising on a non-2xx with
        the body unread holds that pooled connection until the response is
        collected. http_call drains before its raises; this path must too. Spec'd,
        because release() exists on a bare AsyncMock whether or not it is called."""
        resp = MagicMock(spec=aiohttp.ClientResponse)
        resp.status = 400
        resp.release = AsyncMock()
        session = _make_mock_session(resp)
        spotify._session_factory = lambda **kw: session

        with pytest.raises(SpotifyAuthError):
            await spotify.validate("4PTG3Z6ehGkBFwjybzWkR8")

        resp.release.assert_awaited_once()

    async def test_validate_raises_auth_error_on_track_401(
        self, spotify: Spotify
    ) -> None:
        """Grant succeeds but the track call is refused with 401 — still an auth
        rejection, surfaced as SpotifyAuthError."""
        session = _make_split_session(
            _resp(200, {"access_token": "tok", "expires_in": 3600}),
            _resp(401, {"error": {"message": "invalid token"}}),
        )
        spotify._session_factory = lambda **kw: session

        with pytest.raises(SpotifyAuthError) as exc:
            await spotify.validate("4PTG3Z6ehGkBFwjybzWkR8")
        assert exc.value.status == 401

    async def test_validate_non_auth_http_error_is_not_auth_error(
        self, spotify: Spotify
    ) -> None:
        """Grant succeeds but the track endpoint 404s: a plain Exception, not a
        SpotifyAuthError — the caller treats this as inconclusive, not invalid."""
        session = _make_split_session(
            _resp(200, {"access_token": "tok", "expires_in": 3600}),
            _resp(404, {"error": "not found"}),
        )
        spotify._session_factory = lambda **kw: session

        with pytest.raises(Exception) as exc:
            await spotify.validate("4PTG3Z6ehGkBFwjybzWkR8")
        assert not isinstance(exc.value, SpotifyAuthError)

    async def test_validate_raises_value_error_on_missing_track_name(
        self, spotify: Spotify
    ) -> None:
        """Grant and request both succeed, but the payload has no name — an
        unexpected shape (ValueError), which is non-auth / inconclusive."""
        session = _make_split_session(
            _resp(200, {"access_token": "tok", "expires_in": 3600}),
            _resp(200, {"id": "x"}),  # 2xx but no "name"
        )
        spotify._session_factory = lambda **kw: session

        with pytest.raises(ValueError):
            await spotify.validate("4PTG3Z6ehGkBFwjybzWkR8")


class TestSpotifyTrack:
    async def test_track_combines_name_and_artists(self, spotify: Spotify) -> None:
        mock_response = {
            "name": "Bohemian Rhapsody",
            "artists": [{"name": "Queen"}],
        }
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.track("some_track_id")

        assert result == "Bohemian Rhapsody Queen"

    async def test_track_with_multiple_artists(self, spotify: Spotify) -> None:
        mock_response = {
            "name": "Collaboration Track",
            "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
        }
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.track("multi_artist_id")

        assert result == "Collaboration Track Artist A Artist B"

    async def test_track_calls_correct_endpoint(self, spotify: Spotify) -> None:
        mock_response = {"name": "Song", "artists": [{"name": "Artist"}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ) as mock_call:
            await spotify.track("abc123")

        called_endpoint = mock_call.call_args[0][0]
        assert "v1/tracks/abc123" in called_endpoint


class TestSpotifyPlaylist:
    async def test_playlist_returns_list_of_titles(self, spotify: Spotify) -> None:
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

    async def test_playlist_empty_items_returns_empty_list(
        self, spotify: Spotify
    ) -> None:
        mock_response = {"items": []}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.playlist("empty_playlist_id")

        assert result == []

    async def test_playlist_calls_correct_endpoint(self, spotify: Spotify) -> None:
        mock_response = {"items": []}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ) as mock_call:
            await spotify.playlist("pl_abc")

        called_endpoint = mock_call.call_args[0][0]
        assert "v1/playlists/pl_abc/tracks" in called_endpoint

    async def test_playlist_multi_artist_track(self, spotify: Spotify) -> None:
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
    async def test_http_call_raises_on_non_200(self, spotify: Spotify) -> None:
        spotify.auth_token = "prefetched_token"
        spotify.token_expiry = time.time() + 3600  # skip _refresh_token
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session = _make_mock_session(mock_response)
        spotify._session_factory = lambda **kw: mock_session

        with pytest.raises(Exception, match="stat: 404"):
            await spotify.http_call("https://api.spotify.com/v1/tracks/bad")

    async def test_http_call_sets_authorization_header(self, spotify: Spotify) -> None:
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

    async def test_http_call_raises_typed_request_error(self, spotify: Spotify) -> None:
        """A 404 is about the link, not the credentials — so it must not be the
        exception that disables the Spotify source."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        mock_response = AsyncMock()
        mock_response.status = 404
        spotify._session_factory = lambda **kw: _make_mock_session(mock_response)

        with pytest.raises(SpotifyRequestError) as excinfo:
            await spotify.http_call("https://api.spotify.com/v1/tracks/bad")
        assert excinfo.value.status == 404
        assert not isinstance(excinfo.value, SpotifyAuthError)
        assert "may be private" in excinfo.value.user_message

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(404, SpotifyRequestError), (401, SpotifyAuthError)],
    )
    async def test_http_call_releases_the_body_before_raising(
        self, spotify: Spotify, status: int, expected: type[Exception]
    ) -> None:
        """The session outlives the call now, so an unread body holds its pooled
        connection out of circulation until the response is collected. Both raise
        arms must drain. Spec'd, because `release()` exists on a bare AsyncMock
        whether or not it is ever called — which is what made this untestable."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.status = status
        mock_response.release = AsyncMock()
        spotify._session_factory = lambda **kw: _make_mock_session(mock_response)

        with pytest.raises(expected):
            await spotify.http_call("https://api.spotify.com/v1/tracks/bad")

        mock_response.release.assert_awaited_once()

    async def test_http_call_bounds_the_request_timeout(self, spotify: Spotify) -> None:
        """aiohttp's 300s default held a command for five minutes on a hung
        request; the factory must receive an explicit ceiling."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        seen: list[Any] = []

        def factory(**kw: Any) -> MagicMock:
            seen.append(kw.get("timeout"))
            return _make_mock_session(mock_response)

        spotify._session_factory = factory
        await spotify.http_call("https://api.spotify.com/v1/tracks/xyz")

        assert seen == [_HTTP_TIMEOUT]
        assert _HTTP_TIMEOUT.total == 30

    async def test_http_call_retries_429_then_succeeds(self, spotify: Spotify) -> None:
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        limited = AsyncMock()
        limited.status = 429
        limited.headers = {"Retry-After": "0"}
        ok = AsyncMock()
        ok.status = 200
        ok.json = AsyncMock(return_value={"data": "ok"})
        responses = [limited, ok]
        # One session serves every attempt, so the retry is driven off .request().
        session = _make_mock_session(ok)
        session.request = MagicMock(
            side_effect=lambda *a, **kw: _request_cm(responses.pop(0))
        )
        spotify._session_factory = lambda **kw: session

        assert await spotify.http_call("https://api.spotify.com/v1/x") == {"data": "ok"}
        assert responses == []

    async def test_http_call_raises_rate_limit_after_retries(
        self, spotify: Spotify
    ) -> None:
        """Its copy says "wait", not "try again": a re-run re-issues every request
        that earned the 429."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        limited = AsyncMock()
        limited.status = 429
        limited.headers = {"Retry-After": "0"}
        spotify._session_factory = lambda **kw: _make_mock_session(limited)

        with pytest.raises(SpotifyRateLimitError) as excinfo:
            await spotify.http_call("https://api.spotify.com/v1/x")
        assert "rate-limiting" in excinfo.value.user_message
        assert "try again in about 0s" not in excinfo.value.user_message.lower()

    async def test_retry_after_caps_and_tolerates_garbage(
        self, spotify: Spotify
    ) -> None:
        """A malformed header falls back to backoff rather than being read as
        zero, and an hour-long one is capped rather than honoured."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        limited = AsyncMock()
        limited.status = 429
        limited.headers = {"Retry-After": "not-a-number"}
        spotify._session_factory = lambda **kw: _make_mock_session(limited)

        slept: list[float] = []
        with (
            patch("src.spotify.asyncio.sleep", new=AsyncMock(side_effect=slept.append)),
            pytest.raises(SpotifyRateLimitError),
        ):
            await spotify.http_call("https://api.spotify.com/v1/x")

        assert slept == [1.0, 2.0, 4.0]  # exponential, header ignored
        assert all(s <= _MAX_RETRY_AFTER_SECS for s in slept)

    async def test_http_call_refreshes_expired_token(self, spotify: Spotify) -> None:
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
    async def test_track_cache_hit_skips_http(self, spotify: Spotify) -> None:
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

    async def test_playlist_cache_hit_skips_http(self, spotify: Spotify) -> None:
        mock_resp = {"items": [{"track": {"name": "T", "artists": [{"name": "A"}]}}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_resp)
        ) as m:
            await spotify.playlist("pid_cache1")
            await spotify.playlist("pid_cache1")
        m.assert_called_once()

    async def test_track_ttl_is_24h(self, spotify: Spotify, fake_redis: Redis) -> None:
        with patch.object(
            spotify,
            "http_call",
            new=AsyncMock(return_value={"name": "S", "artists": [{"name": "A"}]}),
        ):
            await spotify.track("ttl_test_track")
        ttl = await fake_redis.ttl("spotify:track:ttl_test_track")
        assert 86390 <= ttl <= 86400

    async def test_playlist_ttl_is_1h(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value={"items": []})
        ):
            await spotify.playlist("ttl_test_playlist")
        ttl = await fake_redis.ttl("spotify:playlist:ttl_test_playlist")
        assert 3590 <= ttl <= 3600

    async def test_cache_graceful_when_no_redis(self, fake_redis: Redis) -> None:
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
    async def test_single_artist_id_as_string(self, spotify: Spotify) -> None:
        mock_resp = {"artists": [{"name": "Test Artist", "id": "1"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.artists("artist_id_1")
        assert result == mock_resp["artists"]

    async def test_multiple_artist_ids_as_list(self, spotify: Spotify) -> None:
        mock_resp = {"artists": [{"name": "A"}, {"name": "B"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.artists(["id1", "id2"])
        assert len(result) == 2

    async def test_cache_hit_skips_http(self, spotify: Spotify) -> None:
        mock_resp = {"artists": [{"name": "A"}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_resp)
        ) as m:
            await spotify.artists("aid1")
            await spotify.artists("aid1")
        m.assert_called_once()

    async def test_ttl_is_24h(self, spotify: Spotify, fake_redis: Redis) -> None:
        mock_resp = {"artists": [{"name": "A"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            await spotify.artists("ttl_aid")
        ttl = await fake_redis.ttl("spotify:artist:ttl_aid")
        assert 86390 <= ttl <= 86400


class TestSpotifyAlbums:
    async def test_single_album_id(self, spotify: Spotify) -> None:
        mock_resp = {"albums": [{"name": "Test Album"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.albums("album_id_1")
        assert result == mock_resp["albums"]

    async def test_multiple_album_ids(self, spotify: Spotify) -> None:
        mock_resp = {"albums": [{"name": "A"}, {"name": "B"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            result = await spotify.albums(["alb1", "alb2"])
        assert len(result) == 2

    async def test_cache_hit_skips_http(self, spotify: Spotify) -> None:
        mock_resp = {"albums": [{"name": "Album A"}]}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_resp)
        ) as m:
            await spotify.albums("alb_cache")
            await spotify.albums("alb_cache")
        m.assert_called_once()

    async def test_ttl_is_24h(self, spotify: Spotify, fake_redis: Redis) -> None:
        mock_resp = {"albums": [{"name": "A"}]}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            await spotify.albums("ttl_alb")
        ttl = await fake_redis.ttl("spotify:album:ttl_alb")
        assert 86390 <= ttl <= 86400

    async def test_sorted_cache_key_for_multiple_ids(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        mock_resp = {"albums": []}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=mock_resp)):
            await spotify.albums(["zid", "aid"])
        cached = await fake_redis.get("spotify:album:aid,zid")
        assert cached is not None


class TestSharedSession:
    """One session per client, kept for the life of the process. http_call
    reads each response to EOF, so its connections return to the pool."""

    async def test_session_is_reused_across_calls(self, spotify: Spotify) -> None:
        factory_calls = 0
        resp = AsyncMock()
        resp.status = 200
        session = _make_mock_session(resp)

        def _factory(**kw: Any) -> Any:
            nonlocal factory_calls
            factory_calls += 1
            return session

        spotify._session_factory = _factory
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600

        await spotify.http_call("https://api.spotify.com/v1/tracks/a")
        await spotify.http_call("https://api.spotify.com/v1/tracks/b")

        assert factory_calls == 1
        assert session.request.call_count == 2

    async def test_session_is_created_lazily(self, spotify: Spotify) -> None:
        """A deployment with Spotify configured but never used must not open a
        connector. Asserts the factory was never called: `_session is None` also
        passes if __init__ built one and something cleared the handle after."""
        calls = 0

        def _factory(**kw: Any) -> Any:
            nonlocal calls
            calls += 1
            return _make_mock_session(AsyncMock())

        spotify._session_factory = _factory
        assert calls == 0
        assert spotify._session is None

        spotify._session_or_create()
        assert calls == 1

    async def test_aclose_closes_and_clears(self, spotify: Spotify) -> None:
        session = _make_mock_session(AsyncMock())
        session.close = AsyncMock()
        spotify._session_factory = lambda **kw: session
        spotify._session_or_create()

        await spotify.aclose()

        session.close.assert_awaited_once()
        assert spotify._session is None

    async def test_aclose_without_a_session_is_a_noop(self, spotify: Spotify) -> None:
        await spotify.aclose()  # must not raise

    async def test_aclose_clears_the_handle_before_a_failing_close(
        self, spotify: Spotify
    ) -> None:
        """A socket already gone must not strand the reference. aclose() lets the
        error out — cog_unload guards each step, so swallowing here would only
        hide which one failed — but the handle is cleared first, so the failure
        cannot leave a half-closed session reachable."""
        session = _make_mock_session(AsyncMock())
        session.close = AsyncMock(side_effect=OSError("already gone"))
        spotify._session_factory = lambda **kw: session
        spotify._session_or_create()

        with pytest.raises(OSError):
            await spotify.aclose()
        assert spotify._session is None

    async def test_a_session_closed_from_outside_is_replaced(
        self, spotify: Spotify
    ) -> None:
        """Only aclose() latches the client shut. Anything else that closes the
        session — the suite's own per-test cleanup is the live example — must get a
        replacement, not a corpse that raises `Session is closed` on every call."""
        first = _make_mock_session(AsyncMock())
        second = _make_mock_session(AsyncMock())
        sessions = [first, second]
        spotify._session_factory = lambda **kw: sessions.pop(0)

        assert spotify._session_or_create() is first
        first.closed = True

        assert spotify._session_or_create() is second

    async def test_a_call_after_aclose_is_refused(self, spotify: Spotify) -> None:
        """A command in flight when the cog unloads must not quietly build a
        replacement session: nothing closes it, and the process is on its way out."""
        session = _make_mock_session(AsyncMock())
        spotify._session_factory = lambda **kw: session
        spotify._session_or_create()
        await spotify.aclose()

        with pytest.raises(RuntimeError, match="closed"):
            spotify._session_or_create()
