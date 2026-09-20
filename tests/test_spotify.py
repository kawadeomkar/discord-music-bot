"""Tests for src/spotify.py — Spotify API auth, response parsing, and Redis cache."""

import redis.asyncio as aioredis
import time
import asyncio
from collections.abc import Iterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest

from tests.helpers import settle
from redis.asyncio import Redis

from src import config
from src import spotify as spotify_module
from src.spotify import (
    _HTTP_TIMEOUT,
    _MAX_RETRY_AFTER_SECS,
    Spotify,
    SpotifyAuthError,
    SpotifyBusyError,
    SpotifyPlaylist,
    SpotifyPlaylistForbiddenError,
    SpotifyPlaylistTooSlowError,
    SpotifyRateLimitError,
    SpotifyRequestError,
)


@pytest.fixture
def no_playlist_name(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The walk tests sequence `http_call` page by page and count its awaits.
    The name request after the walk goes through `http_call` too, so it is
    stubbed here and driven for real in TestSpotifyPlaylistName."""
    stub = AsyncMock(return_value=None)
    monkeypatch.setattr(Spotify, "_playlist_name", stub)
    return stub


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
        """token_expiry reflects the key's actual remaining TTL less the margin."""
        await fake_redis.set("spotify:auth:token", b"cached_bearer_token", ex=120)

        before = time.time()
        await spotify._refresh_token()

        margin = spotify_module._TOKEN_EXPIRY_MARGIN_SECS
        assert 115 - margin <= spotify.token_expiry - before <= 121 - margin

    async def test_a_fresh_token_expires_early_by_the_margin(
        self,
        spotify: Spotify,
        mock_auth_response: dict[str, Any],
    ) -> None:
        """Checked once before a retry loop that can sleep 30s on 429s: with no
        margin a request can carry a token that expired while it slept, and a 401
        reads as "rejected the credentials"."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        spotify._session_factory = lambda **kw: _make_mock_session(mock_resp)

        before = time.time()
        await spotify._refresh_token(use_cache=False)

        assert spotify.token_expiry - before <= 3600 - 60 + 1

    async def test_a_cached_token_inside_the_margin_is_not_reused(
        self,
        spotify: Spotify,
        fake_redis: aioredis.Redis,
        mock_auth_response: dict[str, Any],
    ) -> None:
        """Reused, it would be marked expired on arrival and re-read from Redis on
        every call until the key lapsed."""
        await fake_redis.set("spotify:auth:token", b"nearly_gone", ex=30)
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        spotify._session_factory = lambda **kw: _make_mock_session(mock_resp)

        await spotify._refresh_token()

        assert spotify.auth_token == mock_auth_response["access_token"]

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


@pytest.mark.usefixtures("no_playlist_name")
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

        assert len(result.titles) == 2
        assert result.titles[0] == "Track One Artist X"
        assert result.titles[1] == "Track Two Artist Y"

    async def test_playlist_empty_items_returns_empty_list(
        self, spotify: Spotify
    ) -> None:
        mock_response = {"items": []}
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=mock_response)
        ):
            result = await spotify.playlist("empty_playlist_id")

        assert result.titles == []

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

        assert result.titles[0] == "Collab A B C"


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

    async def test_a_retry_that_would_outlive_the_deadline_is_a_rate_limit(
        self, spotify: Spotify
    ) -> None:
        """Slept anyway, the caller's own bound would expire mid-sleep and report
        a hang. Nothing is slept and the second request is never sent."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        limited = AsyncMock()
        limited.status = 429
        limited.headers = {"Retry-After": "10"}
        session = _make_mock_session(limited)
        spotify._session_factory = lambda **kw: session
        deadline = asyncio.get_running_loop().time() + 5.0

        with (
            patch("src.spotify.asyncio.sleep", new=AsyncMock()) as slept,
            pytest.raises(SpotifyRateLimitError) as excinfo,
        ):
            await spotify.http_call("https://api.spotify.com/v1/x", deadline=deadline)

        slept.assert_not_awaited()
        assert session.request.call_count == 1
        assert excinfo.value.retry_after == 10.0

    async def test_a_retry_inside_the_deadline_still_sleeps(
        self, spotify: Spotify
    ) -> None:
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        limited = AsyncMock()
        limited.status = 429
        limited.headers = {"Retry-After": "10"}
        ok = AsyncMock()
        ok.status = 200
        ok.json = AsyncMock(return_value={"data": "ok"})
        responses = [limited, ok]
        session = _make_mock_session(ok)
        session.request = MagicMock(
            side_effect=lambda *a, **kw: _request_cm(responses.pop(0))
        )
        spotify._session_factory = lambda **kw: session
        deadline = asyncio.get_running_loop().time() + 20.0

        with patch("src.spotify.asyncio.sleep", new=AsyncMock()) as slept:
            result = await spotify.http_call(
                "https://api.spotify.com/v1/x", deadline=deadline
            )

        assert result == {"data": "ok"}
        slept.assert_awaited_once_with(10.0)

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


@pytest.mark.usefixtures("no_playlist_name")
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
        ttl = await fake_redis.ttl("spotify:playlist:v3:ttl_test_playlist")
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


def _playlist_page(
    titles: list[str],
    *,
    total: int,
    next_url: Optional[str],
    duration_ms: Optional[int] = 180_000,
) -> dict[str, Any]:
    """One page of `v1/playlists/{id}/tracks` under the fields mask. Every track
    carries `duration_ms` unless it is None, which omits the key."""
    track: dict[str, Any] = {"artists": [{"name": "A"}]}
    if duration_ms is not None:
        track["duration_ms"] = duration_ms
    return {
        "items": [{"track": {**track, "name": t}} for t in titles],
        "total": total,
        "next": next_url,
    }


def _pages(count: int, page_size: int = 100) -> list[dict[str, Any]]:
    """`count` tracks split into pages, each pointing at the next."""
    names = [f"T{i}" for i in range(count)]
    chunks = [names[i : i + page_size] for i in range(0, count, page_size)] or [[]]
    return [
        _playlist_page(
            chunk,
            total=count,
            next_url=None
            if i == len(chunks) - 1
            else f"https://api.spotify.com/v1/next/{i + 1}",
        )
        for i, chunk in enumerate(chunks)
    ]


@pytest.mark.usefixtures("no_playlist_name")
class TestSpotifyPlaylistPaging:
    """The `next` cursor. The four tests in TestSpotifyPlaylist above patch
    http_call with a single AsyncMock(return_value=...), which answers every page
    identically — they keep passing against a pager and prove nothing about it, so
    these build a real multi-page side effect."""

    async def test_a_250_track_playlist_returns_all_250(self, spotify: Spotify) -> None:
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=_pages(250))
        ) as call:
            result = await spotify.playlist("pid_250")

        assert len(result.titles) == 250
        assert result.titles[0] == "T0 A"
        assert result.titles[-1] == "T249 A"
        assert call.await_count == 3

    async def test_the_mask_asks_for_next_and_total(self, spotify: Spotify) -> None:
        """Under `fields` Spotify returns only what is named: without `next` the
        walk cannot terminate, without `total` the card has no denominator."""
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=_pages(10))
        ) as call:
            await spotify.playlist("pid_mask")

        fields = call.await_args_list[0].kwargs["params"]["fields"]
        assert "next" in fields
        assert "total" in fields

    async def test_the_cursor_is_followed_as_a_whole_url(
        self, spotify: Spotify
    ) -> None:
        """`next` carries its own offset in its query, so params must not ride
        beside it."""
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=_pages(250))
        ) as call:
            await spotify.playlist("pid_cursor")

        second = call.await_args_list[1]
        assert second.args[0] == "https://api.spotify.com/v1/next/1"
        assert second.kwargs["params"] is None

    async def test_progress_reports_the_total_from_the_first_page(
        self, spotify: Spotify
    ) -> None:
        seen: list[tuple[int, Optional[int]]] = []
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=_pages(250))):
            await spotify.playlist(
                "pid_prog", on_progress=lambda d, t: seen.append((d, t))
            )

        assert seen == [(100, 250), (200, 250), (250, 250)]

    async def test_progress_counts_items_walked_not_titles_kept(
        self, spotify: Spotify
    ) -> None:
        """The numerator measures how far the walk has got, so a playlist holding
        a removed track still fills its bar. What was QUEUED is the return value."""
        page = _playlist_page(["Real"], total=2, next_url=None)
        page["items"].append({"track": None})
        seen: list[tuple[int, Optional[int]]] = []
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)):
            result = await spotify.playlist(
                "pid_gap", on_progress=lambda d, t: seen.append((d, t))
            )

        assert result.titles == ["Real A"]
        assert seen == [(2, 2)]

    async def test_an_episode_without_artists_is_still_queued(
        self, spotify: Spotify
    ) -> None:
        page = {"items": [{"track": {"name": "Some Episode"}}], "total": 1}
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)):
            assert (await spotify.playlist("pid_ep")).titles == ["Some Episode"]

    async def test_a_single_page_playlist_makes_one_call(
        self, spotify: Spotify
    ) -> None:
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=_pages(10))
        ) as call:
            assert len((await spotify.playlist("pid_one")).titles) == 10
        call.assert_awaited_once()

    async def test_no_progress_callback_is_the_default(self, spotify: Spotify) -> None:
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=_pages(250))):
            assert len((await spotify.playlist("pid_default")).titles) == 250

    async def test_a_cache_hit_reports_no_progress(self, spotify: Spotify) -> None:
        """_cached_call returns before fetch(), so a cached playlist has nothing to
        report — and it resolves far under the threshold that shows a card."""
        seen: list[tuple[int, Optional[int]]] = []
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=_pages(250))):
            await spotify.playlist("pid_cached")
            seen.clear()
            await spotify.playlist(
                "pid_cached", on_progress=lambda d, t: seen.append((d, t))
            )

        assert seen == []

    async def test_a_failure_midway_raises_and_caches_nothing(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A part-walked playlist must never be cached: an hour of answering with
        three fifths of a playlist is the truncation bug with extra steps."""
        pages = _pages(500)
        side_effect: list[Any] = [*pages[:2], SpotifyRequestError(502, "tracks")]
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=side_effect)):
            with pytest.raises(SpotifyRequestError):
                await spotify.playlist("pid_broken")

        assert await fake_redis.get("spotify:playlist:v3:pid_broken") is None

    async def test_a_throttled_page_is_reported_as_a_rate_limit(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through the real http_call, at 1/40 scale: Retry-After 10s against the
        20s page bound. The second sleep would outlive the bound. Expired, that
        bound answers SpotifyPlaylistTooSlowError — "try again" — which re-sends up
        to 100 requests into a per-application limiter."""
        monkeypatch.setattr(spotify_module, "_PLAYLIST_PAGE_TIMEOUT_SECS", 0.5)
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        limited = AsyncMock()
        limited.status = 429
        limited.headers = {"Retry-After": "0.25"}
        session = _make_mock_session(limited)
        spotify._session_factory = lambda **kw: session

        with pytest.raises(SpotifyRateLimitError):
            await spotify.playlist("pid_throttled")

        # Slept at t=0 (0.25 < 0.5) and gave up at t=0.25 instead of sleeping into
        # the bound. Real sleeps: the loop clock is what the deadline reads.
        assert session.request.call_count == 2

    async def test_the_whole_walk_is_bounded(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_HTTP_TIMEOUT is per request, and 100 pages of 429 retries have no
        aggregate bound — which on a cold start holds the playback gate open."""
        monkeypatch.setattr(spotify_module, "_PLAYLIST_WALK_TIMEOUT_SECS", 0.05)

        async def _slow(*_: Any, **__: Any) -> dict[str, Any]:
            await asyncio.sleep(0.2)
            return _playlist_page([], total=0, next_url=None)

        with patch.object(spotify, "http_call", new=_slow):
            with pytest.raises(SpotifyPlaylistTooSlowError):
                await spotify.playlist("pid_slow")

    async def test_one_playlist_pasted_twice_is_walked_once(
        self, spotify: Spotify
    ) -> None:
        """A full walk is 100 requests where a track lookup was one, so N users
        pasting one link is N identical hundred-request walks racing to write one
        cache entry. The first starts it; the rest await its outcome."""
        gate = asyncio.Event()
        calls = 0

        async def _slow(*_: Any, **__: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            await gate.wait()
            return _playlist_page(["T"], total=1, next_url=None)

        with patch.object(spotify, "http_call", new=_slow):
            walks = [
                asyncio.create_task(spotify.playlist("pid_shared")) for _ in range(8)
            ]
            await settle()
            gate.set()
            results = await asyncio.gather(*walks)

        assert calls == 1
        assert all(r.titles == ["T A"] for r in results)

    async def test_a_failed_walk_leaves_the_key_free(self, spotify: Spotify) -> None:
        """Registered before the shield, so a retry after a failure starts a
        fresh walk rather than joining a future that already raised."""
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with pytest.raises(RuntimeError):
                await spotify.playlist("pid_fail")
        assert spotify_module._INFLIGHT_PLAYLISTS == {}

    async def test_every_caller_on_a_shared_walk_hears_its_pages(
        self, spotify: Spotify
    ) -> None:
        """A second guild pasting the same playlist joins the walk; its card is fed
        by the walk, not by whichever caller happened to start it."""
        gate = asyncio.Event()
        pages = _pages(250)

        async def _paged(*_: Any, **__: Any) -> dict[str, Any]:
            await gate.wait()
            return pages.pop(0)

        leader: list[tuple[int, Optional[int]]] = []
        joiner: list[tuple[int, Optional[int]]] = []
        with patch.object(spotify, "http_call", new=_paged):
            first = asyncio.create_task(
                spotify.playlist(
                    "pid_fed", on_progress=lambda d, t: leader.append((d, t))
                )
            )
            await settle()
            second = asyncio.create_task(
                spotify.playlist(
                    "pid_fed", on_progress=lambda d, t: joiner.append((d, t))
                )
            )
            await settle()
            gate.set()
            await asyncio.gather(first, second)

        assert leader == joiner == [(100, 250), (200, 250), (250, 250)]
        assert spotify_module._PLAYLIST_SUBSCRIBERS == {}

    async def test_one_cancelled_caller_leaves_the_walk_to_the_others(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """Shielded: a -stop on one guild's -play must not cancel a walk another
        guild is waiting on. Cached by the walk itself, so even a walk every caller
        abandoned is not 100 requests thrown away."""
        gate = asyncio.Event()

        async def _slow(*_: Any, **__: Any) -> dict[str, Any]:
            await gate.wait()
            return _playlist_page(["T"], total=1, next_url=None)

        with patch.object(spotify, "http_call", new=_slow):
            first = asyncio.create_task(spotify.playlist("pid_kept"))
            await settle()
            second = asyncio.create_task(spotify.playlist("pid_kept"))
            await settle()
            first.cancel()
            await settle()
            gate.set()
            assert (await second).titles == ["T A"]
            assert first.cancelled()

            gate.clear()
            alone = asyncio.create_task(spotify.playlist("pid_abandoned"))
            await settle()
            alone.cancel()
            await settle()
            gate.set()
            await settle()

        assert await fake_redis.get("spotify:playlist:v3:pid_abandoned") is not None

    async def test_a_walk_that_cannot_get_a_slot_says_spotify_is_busy(
        self, spotify: Spotify
    ) -> None:
        """The 120s walk budget starts after the slot is taken, so a small playlist
        behind two 10,000-track walks otherwise waits minutes with nothing sent."""
        config.play_resolve_wait_secs.set_override(0.05)
        slot = spotify_module._playlist_slot()
        for _ in range(spotify_module._PLAYLIST_WALK_CONCURRENCY):
            await slot.acquire()
        http_call = AsyncMock()
        try:
            with patch.object(spotify, "http_call", new=http_call):
                with pytest.raises(SpotifyBusyError) as excinfo:
                    async with asyncio.timeout(5):
                        await spotify.playlist("pid_queued")
        finally:
            for _ in range(spotify_module._PLAYLIST_WALK_CONCURRENCY):
                slot.release()
        http_call.assert_not_awaited()
        assert "busy" in excinfo.value.user_message
        assert "within 0.05s" in str(excinfo.value)

    def test_the_walk_slot_follows_the_running_loop(self) -> None:
        """A Semaphore binds to the first loop that waits on it; pytest-asyncio
        builds a loop per test, so a module-level one breaks the next test that
        contends on it."""

        async def _contend() -> asyncio.Semaphore:
            slot = spotify_module._playlist_slot()
            held = [slot.acquire() for _ in range(3)]
            waiter = asyncio.ensure_future(asyncio.gather(*held))
            await asyncio.sleep(0)
            slot.release()
            await waiter
            return slot

        assert asyncio.run(_contend()) is not asyncio.run(_contend())

    async def test_concurrent_walks_are_bounded(self, spotify: Spotify) -> None:
        """Process-wide, because the harm is Spotify's rate limiter, which is per
        application. Nothing else bounds these: PLAY_RESOLVE_CONCURRENCY guards
        yt-dlp workers and a Spotify walk holds none."""
        live = 0
        peak = 0
        gate = asyncio.Event()

        async def _slow(*_: Any, **__: Any) -> dict[str, Any]:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await gate.wait()
            live -= 1
            return _playlist_page(["T"], total=1, next_url=None)

        with patch.object(spotify, "http_call", new=_slow):
            walks = [
                asyncio.create_task(spotify.playlist(f"pid_b{n}")) for n in range(8)
            ]
            await settle()
            gate.set()
            await asyncio.gather(*walks)

        assert peak == spotify_module._PLAYLIST_WALK_CONCURRENCY

    async def test_the_first_page_asks_for_the_fields_the_walk_needs(
        self, spotify: Spotify
    ) -> None:
        """Under `fields` Spotify answers with only the keys named, so `next` and
        `total` have to be asked for: without the first the walk cannot terminate,
        without the second progress has no denominator and a short walk cannot be
        detected. The page size is asked for too — the page cap is derived from
        it, so they must agree."""
        page = _playlist_page(["T"], total=1, next_url=None)
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)) as c:
            await spotify.playlist("pid_fields")
        assert (awaited := c.await_args) is not None
        params = awaited.kwargs["params"]
        assert params["limit"] == spotify_module._PLAYLIST_PAGE_SIZE
        for field in ("next", "total", "items(track(name,artists(name),duration_ms))"):
            assert field in params["fields"]

    async def test_the_cursor_is_followed_without_the_first_pages_params(
        self, spotify: Spotify
    ) -> None:
        """The cursor is a full URL carrying its own offset; passing the first
        page's params beside it would fight the offset it encodes."""
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=_pages(250))
        ) as c:
            await spotify.playlist("pid_cursor")
        assert c.await_args_list[0].kwargs["params"] is not None
        assert all(call.kwargs["params"] is None for call in c.await_args_list[1:])

    async def test_an_off_origin_cursor_is_refused(
        self, spotify: Spotify, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The walk sends the bearer token to whatever the cursor names. Spotify
        is trusted, but this repo has already shipped one host built from an
        unvalidated response field, so the origin is checked rather than assumed."""
        pages = [
            _playlist_page(["T1"], total=2, next_url="https://evil.example/steal"),
            _playlist_page(["T2"], total=2, next_url=None),
        ]
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=pages)) as c:
            assert (await spotify.playlist("pid_evil")).titles == ["T1 A"]
        # Stopped at page 1: the second page was never requested.
        assert c.await_count == 1
        assert "off-origin cursor" in caplog.text

    async def test_a_short_walk_is_reported(
        self,
        spotify: Spotify,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Silent truncation is the bug this pager removes. Every ending but an
        exhausted `next` is short, and `total` is the only thing that can say so."""
        monkeypatch.setattr(spotify_module, "_MAX_PLAYLIST_PAGES", 2)
        page = _playlist_page(
            ["T"], total=500, next_url="https://api.spotify.com/v1/next"
        )
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)):
            await spotify.playlist("pid_short")
        assert "walked 2 of 500 items" in caplog.text

    async def test_a_short_walk_is_not_cached(
        self, spotify: Spotify, fake_redis: Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every later paste would be answered from the cache with the partial
        list, and a hit logs nothing — the truncation this pager reports, hidden."""
        monkeypatch.setattr(spotify_module, "_MAX_PLAYLIST_PAGES", 2)
        page = _playlist_page(
            ["T"], total=500, next_url="https://api.spotify.com/v1/next"
        )
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)):
            await spotify.playlist("pid_short")
        assert await fake_redis.get("spotify:playlist:v3:pid_short") is None

    async def test_a_track_with_no_name_is_walked_not_queued(
        self, spotify: Spotify, fake_redis: Redis, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A removed track (null) and one Spotify sends without a name are counted
        as walked, so a healthy playlist holding them is complete, and cached."""
        page = _playlist_page(["Kept"], total=3, next_url=None)
        page["items"] += [{"track": None}, {"track": {"artists": []}}]
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)):
            titles = (await spotify.playlist("pid_gaps")).titles
        assert titles == ["Kept A"]
        assert "walked" not in caplog.text
        assert await fake_redis.get("spotify:playlist:v3:pid_gaps") is not None

    async def test_the_length_sums_milliseconds_across_pages_before_dividing(
        self, spotify: Spotify
    ) -> None:
        """Three 1,500 ms tracks are 4 s. Flooring each track first says 3, and
        the error grows with every track in the playlist."""
        pages = [
            _playlist_page(
                ["A", "B"],
                total=3,
                next_url="https://api.spotify.com/v1/next/1",
                duration_ms=1500,
            ),
            _playlist_page(["C"], total=3, next_url=None, duration_ms=1500),
        ]
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=pages)):
            playlist = await spotify.playlist("pid_length")
        assert playlist.duration_secs == 4
        assert playlist.duration_partial is False

    async def test_a_kept_track_without_a_duration_makes_the_length_partial(
        self, spotify: Spotify
    ) -> None:
        page = _playlist_page(["Timed"], total=2, next_url=None, duration_ms=3000)
        page["items"].append({"track": {"name": "Untimed", "artists": []}})
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=page)):
            playlist = await spotify.playlist("pid_untimed")
        assert playlist.titles == ["Timed A", "Untimed"]
        assert playlist.duration_secs == 3
        assert playlist.duration_partial is True

    async def test_unavailable_counts_items_walked_and_not_kept_across_pages(
        self, spotify: Spotify
    ) -> None:
        """A null track and a track or episode with no name are unavailable; a
        local file carries a name and is kept. None of them makes the length
        partial, which describes the kept tracks alone."""
        first = _playlist_page(
            ["One"], total=7, next_url="https://api.spotify.com/v1/next/1"
        )
        first["items"] += [{"track": None}, {"track": {"artists": []}}]
        second = _playlist_page(["Two"], total=7, next_url=None)
        second["items"] += [
            {"track": None},
            {"track": {"name": "", "type": "episode"}},
            {"track": {"name": "Local File", "artists": [], "duration_ms": 1000}},
        ]
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[first, second])
        ):
            playlist = await spotify.playlist("pid_unavailable")
        assert playlist.titles == ["One A", "Two A", "Local File"]
        assert playlist.unavailable == 4
        assert playlist.duration_partial is False

    async def test_a_403_on_the_tracks_is_the_apps_access_not_the_credentials(
        self, spotify: Spotify, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SpotifyAuthError reads "rejected the credentials", which sends an
        operator to rotate credentials that work."""
        with patch.object(
            spotify,
            "http_call",
            new=AsyncMock(side_effect=SpotifyAuthError(403, "endpoint: tracks")),
        ):
            with pytest.raises(SpotifyPlaylistForbiddenError) as excinfo:
                await spotify.playlist("pid_refused")
        assert "won't share" in excinfo.value.user_message
        assert "pid_refused" in caplog.text

    async def test_a_page_without_items_is_the_same_refusal(
        self, spotify: Spotify
    ) -> None:
        """An empty playlist still carries `items: []`; a page with no key at all
        would otherwise queue nothing and say the playlist was empty."""
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value={"total": 5, "next": None})
        ):
            with pytest.raises(SpotifyPlaylistForbiddenError):
                await spotify.playlist("pid_itemless")

    async def test_a_401_is_still_the_credentials(self, spotify: Spotify) -> None:
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=SpotifyAuthError(401))
        ):
            with pytest.raises(SpotifyAuthError):
                await spotify.playlist("pid_401")

    async def test_a_complete_walk_is_not_reported(
        self, spotify: Spotify, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=_pages(250))):
            await spotify.playlist("pid_whole")
        assert "walked" not in caplog.text

    def test_the_page_cap_covers_spotifys_own_item_ceiling(self) -> None:
        """Derived, not assumed: `limit=100` is what the API accepts although its
        reference documents 50, so halving the page size must double the guard
        rather than truncate at half a playlist."""
        assert (
            spotify_module._MAX_PLAYLIST_PAGES * spotify_module._PLAYLIST_PAGE_SIZE
            > spotify_module._PLAYLIST_MAX_ITEMS
        )

    async def test_one_stalled_page_is_not_reported_as_an_oversized_playlist(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two failures need different advice: a stall is worth retrying and
        an oversized playlist is not. They are told apart by which deadline
        expired, because every aiohttp timeout subclasses builtin TimeoutError
        and `except TimeoutError` alone once called a connect failure too large."""
        monkeypatch.setattr(spotify_module, "_PLAYLIST_PAGE_TIMEOUT_SECS", 0.05)

        async def _stalled(*_: Any, **__: Any) -> dict[str, Any]:
            await asyncio.sleep(5)
            return _playlist_page([], total=0, next_url=None)

        with patch.object(spotify, "http_call", new=_stalled):
            with pytest.raises(SpotifyPlaylistTooSlowError) as caught:
                await spotify.playlist("pid_stalled")

        assert not caught.value.whole_walk
        assert "try again" in caught.value.user_message
        assert "smaller parts" not in caught.value.user_message

    async def test_an_oversized_playlist_does_not_advise_a_retry(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A walk that used its whole budget will use it again on every attempt,
        so "try again" is advice that cannot work. It says what to do instead."""
        monkeypatch.setattr(spotify_module, "_PLAYLIST_WALK_TIMEOUT_SECS", 0.08)

        async def _steady(*_: Any, **__: Any) -> dict[str, Any]:
            await asyncio.sleep(0.02)
            return _playlist_page(
                ["T"], total=1000, next_url="https://api.spotify.com/v1/next"
            )

        with patch.object(spotify, "http_call", new=_steady):
            with pytest.raises(SpotifyPlaylistTooSlowError) as caught:
                await spotify.playlist("pid_big")

        assert caught.value.whole_walk
        assert "smaller parts" in caught.value.user_message
        assert "try again" not in caught.value.user_message

    async def test_the_walk_budget_covers_the_ten_thousand_item_ceiling(self) -> None:
        """_MAX_PLAYLIST_PAGES is Spotify's own ceiling expressed in pages, so the
        walk must afford every one of them. At least a second each — five times a
        healthy call — or the largest playlists are permanently unqueueable, which
        is what 0.6s a page did. Asserted as the ratio because that is the
        decision: the two constants move together and neither is meaningful
        alone."""
        per_page = (
            spotify_module._PLAYLIST_WALK_TIMEOUT_SECS
            / spotify_module._MAX_PLAYLIST_PAGES
        )
        assert per_page >= 1.0, f"{per_page}s a page at the cap"
        # And one page can never eat the whole walk.
        assert (
            spotify_module._PLAYLIST_PAGE_TIMEOUT_SECS
            < spotify_module._PLAYLIST_WALK_TIMEOUT_SECS
        )

    async def test_a_cursor_that_never_ends_stops_at_the_page_cap(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(spotify_module, "_MAX_PLAYLIST_PAGES", 3)
        forever = _playlist_page(
            ["T"], total=999, next_url="https://api.spotify.com/v1/next"
        )
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=forever)
        ) as call:
            assert (await spotify.playlist("pid_loop")).titles == ["T A"] * 3
        assert call.await_count == 3

    async def test_the_cache_key_is_versioned(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """The key names the value's shape, so a v2 entry — a bare list of titles
        — is never read back as a playlist."""
        await fake_redis.set("spotify:playlist:v2:pid_v3", orjson.dumps(["Stale"]))
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=_pages(10))):
            playlist = await spotify.playlist("pid_v3")

        assert len(playlist.titles) == 10
        assert await fake_redis.get("spotify:playlist:v3:pid_v3") is not None
        assert await fake_redis.get("spotify:playlist:pid_v3") is None


class TestSpotifyPlaylistName:
    """The name is one request of its own, made after the walk: the tracks
    endpoint does not carry it. It decorates the confirmation, so nothing about
    it may fail the playlist."""

    async def test_the_name_is_requested_after_the_walk(self, spotify: Spotify) -> None:
        responses = [*_pages(150), {"name": "Biteki"}]
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=responses)
        ) as call:
            playlist = await spotify.playlist("pid_named")

        assert playlist.name == "Biteki"
        assert len(playlist.titles) == 150
        assert call.await_count == 3
        assert call.await_args_list[0].args[0].endswith("v1/playlists/pid_named/tracks")
        last = call.await_args_list[-1]
        assert last.args[0] == "https://api.spotify.com/v1/playlists/pid_named"
        assert last.kwargs["params"] == {"fields": "name"}
        assert last.kwargs["deadline"] is not None

    @pytest.mark.parametrize(
        "failure",
        [
            SpotifyRequestError(404, "playlists"),
            SpotifyAuthError(403, "endpoint: playlists"),
            SpotifyRateLimitError(30),
            TimeoutError(),
        ],
        ids=["request", "auth", "rate-limit", "timeout"],
    )
    async def test_a_failed_name_request_still_returns_the_playlist(
        self,
        spotify: Spotify,
        caplog: pytest.LogCaptureFixture,
        failure: Exception,
    ) -> None:
        page = _playlist_page(["T"], total=1, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[page, failure])
        ):
            playlist = await spotify.playlist("pid_nameless")

        assert playlist.name is None
        assert playlist.titles == ["T A"]
        assert "name request failed" in caplog.text

    async def test_a_stalled_name_request_is_bounded(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(spotify_module, "_PLAYLIST_PAGE_TIMEOUT_SECS", 0.05)
        page = _playlist_page(["T"], total=1, next_url=None)

        async def _stalls_on_the_name(url: str, **_: Any) -> dict[str, Any]:
            if url.endswith("/tracks"):
                return page
            await asyncio.sleep(5)
            return {"name": "Never"}

        with patch.object(spotify, "http_call", new=_stalls_on_the_name):
            async with asyncio.timeout(2):
                playlist = await spotify.playlist("pid_stalled_name")

        assert playlist.name is None
        assert playlist.titles == ["T A"]

    @pytest.mark.parametrize(
        "response", [{}, {"name": ""}, {"name": None}, None], ids=repr
    )
    async def test_a_response_without_a_name_is_none(
        self, spotify: Spotify, response: Any
    ) -> None:
        with patch.object(spotify, "http_call", new=AsyncMock(return_value=response)):
            assert await spotify._playlist_name("pid") is None

    async def test_cancellation_is_not_swallowed(self, spotify: Spotify) -> None:
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=asyncio.CancelledError)
        ):
            with pytest.raises(asyncio.CancelledError):
                await spotify._playlist_name("pid")


class TestSpotifyPlaylistCache:
    async def test_every_field_round_trips_and_a_hit_makes_no_request(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """Values unlike the parser's defaults, so a field the writer drops is a
        field this test sees missing."""
        first = _playlist_page(
            ["A"],
            total=3,
            next_url="https://api.spotify.com/v1/next/1",
            duration_ms=1500,
        )
        first["items"].append({"track": None})
        second = _playlist_page(["B"], total=3, next_url=None, duration_ms=2500)
        responses = [first, second, {"name": "Biteki"}]
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=responses)):
            walked = await spotify.playlist("pid_round")

        assert walked == SpotifyPlaylist(
            name="Biteki",
            titles=["A A", "B A"],
            duration_secs=4,
            duration_partial=False,
            unavailable=1,
        )
        assert await fake_redis.get("spotify:playlist:v3:pid_round") is not None
        with patch.object(spotify, "http_call", new=AsyncMock()) as call:
            assert await spotify.playlist("pid_round") == walked
        call.assert_not_awaited()

    async def test_a_partial_entry_still_answers(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        await fake_redis.set(
            "spotify:playlist:v3:pid_partial", orjson.dumps({"titles": ["Only"]})
        )
        with patch.object(spotify, "http_call", new=AsyncMock()) as call:
            playlist = await spotify.playlist("pid_partial")

        call.assert_not_awaited()
        assert playlist == SpotifyPlaylist(
            name=None,
            titles=["Only"],
            duration_secs=0,
            duration_partial=True,
            unavailable=0,
        )

    @pytest.mark.parametrize(
        "entry",
        [
            ["a", "bare", "list"],
            {"name": "No titles"},
            {"titles": [], "unavailable": "x"},
        ],
        ids=["list", "no-titles", "bad-number"],
    )
    async def test_an_unreadable_entry_is_a_miss_the_walk_overwrites(
        self, spotify: Spotify, fake_redis: Redis, entry: Any
    ) -> None:
        key = "spotify:playlist:v3:pid_unreadable"
        await fake_redis.set(key, orjson.dumps(entry))
        responses = [*_pages(1), {"name": "Fresh"}]
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=responses)):
            playlist = await spotify.playlist("pid_unreadable")

        assert playlist.name == "Fresh"
        assert (raw := await fake_redis.get(key)) is not None
        assert orjson.loads(raw)["name"] == "Fresh"


def _album_tracks_page(
    titles: list[str],
    *,
    total: int,
    next_url: Optional[str],
    duration_ms: Optional[int] = 180_000,
) -> dict[str, Any]:
    """One tracks paging object. An album item IS the track: no `track` wrapper."""
    track: dict[str, Any] = {"artists": [{"name": "A"}]}
    if duration_ms is not None:
        track["duration_ms"] = duration_ms
    return {
        "items": [{**track, "name": t} for t in titles],
        "total": total,
        "next": next_url,
    }


def _album(tracks: dict[str, Any], **identity: Any) -> dict[str, Any]:
    """`GET v1/albums/{id}`: the album's identity around its first tracks page."""
    return {
        "name": "Discovery",
        "artists": [{"name": "Daft Punk"}],
        "images": [{"url": "https://i.scdn.co/big"}, {"url": "https://i.scdn.co/sm"}],
        **identity,
        "tracks": tracks,
    }


def _endless_album(*, total: int, per_page: int) -> Iterator[dict[str, Any]]:
    """An album whose cursor advances forever: page 1 as the album object, then
    bare tracks pages, each naming a new `next`."""
    page_no = 0
    while True:
        titles = [f"P{page_no}T{i}" for i in range(per_page)]
        tracks = _album_tracks_page(
            titles,
            total=total,
            next_url=f"https://api.spotify.com/v1/next/{page_no + 1}",
        )
        yield _album(tracks) if page_no == 0 else tracks
        page_no += 1


class TestSpotifyAlbum:
    async def test_one_request_returns_the_tracks_and_the_identity(
        self, spotify: Spotify
    ) -> None:
        page = _album_tracks_page(
            ["One More Time", "Aerodynamic"], total=2, next_url=None
        )
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ) as call:
            album = await spotify.album("aid_one")

        assert album == SpotifyPlaylist(
            name="Discovery",
            titles=["One More Time A", "Aerodynamic A"],
            duration_secs=360,
            duration_partial=False,
            unavailable=0,
            artists=["Daft Punk"],
            thumbnail="https://i.scdn.co/big",
        )
        assert call.await_count == 1
        assert call.await_args_list[0].args[0].endswith("v1/albums/aid_one")

    async def test_later_pages_follow_the_cursor_and_are_bare_tracks_pages(
        self, spotify: Spotify
    ) -> None:
        cursor = "https://api.spotify.com/v1/albums/aid_two/tracks?offset=50&limit=50"
        first = _album(_album_tracks_page(["T0"], total=2, next_url=cursor))
        second = _album_tracks_page(["T1"], total=2, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[first, second])
        ) as call:
            album = await spotify.album("aid_two")

        assert album.titles == ["T0 A", "T1 A"]
        assert call.await_args_list[1].args[0] == cursor

    async def test_progress_reports_items_walked_against_the_albums_total(
        self, spotify: Spotify
    ) -> None:
        cursor = "https://api.spotify.com/v1/next/1"
        first = _album(_album_tracks_page(["T0", "T1"], total=3, next_url=cursor))
        second = _album_tracks_page(["T2"], total=3, next_url=None)
        seen: list[tuple[int, Optional[int]]] = []
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[first, second])
        ):
            await spotify.album(
                "aid_prog", on_progress=lambda d, t: seen.append((d, t))
            )

        assert seen == [(2, 3), (3, 3)]

    async def test_a_null_or_nameless_item_is_counted_and_not_queued(
        self, spotify: Spotify
    ) -> None:
        page = _album_tracks_page(["Real"], total=3, next_url=None)
        page["items"] += [None, {"artists": [{"name": "A"}]}]
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ):
            album = await spotify.album("aid_gap")

        assert album.titles == ["Real A"]
        assert album.unavailable == 2

    async def test_a_track_without_a_duration_marks_the_total_partial(
        self, spotify: Spotify
    ) -> None:
        page = _album_tracks_page(["T"], total=1, next_url=None, duration_ms=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ):
            album = await spotify.album("aid_nodur")

        assert (album.duration_secs, album.duration_partial) == (0, True)

    async def test_an_album_with_no_identity_still_queues(
        self, spotify: Spotify
    ) -> None:
        page = _album_tracks_page(["T"], total=1, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value={"tracks": page})
        ):
            album = await spotify.album("aid_bare")

        assert album.titles == ["T A"]
        assert (album.name, album.artists, album.thumbnail) == (None, [], None)

    async def test_an_off_origin_cursor_is_refused(self, spotify: Spotify) -> None:
        """The walk sends the bearer token to the cursor it follows."""
        first = _album(
            _album_tracks_page(["T0"], total=2, next_url="https://evil.example/next")
        )
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[first])
        ) as call:
            album = await spotify.album("aid_evil")

        assert album.titles == ["T0 A"]
        assert call.await_count == 1

    async def test_a_cursor_that_never_ends_stops_at_the_page_cap(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(spotify_module, "_MAX_PLAYLIST_PAGES", 3)
        responses = _endless_album(total=999, per_page=1)
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=responses)
        ) as call:
            album = await spotify.album("aid_loop")

        assert call.await_count == 3
        assert len(album.titles) == 3

    async def test_the_page_cap_is_the_albums_own_size_plus_slack(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A cursor that keeps advancing past the album: 30 tracks at 10 a page is
        3 pages, so the walk stops at 5, and what it overshot is not cached."""
        responses = _endless_album(total=30, per_page=10)
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=responses)
        ) as call:
            album = await spotify.album("aid_over")

        assert call.await_count == 5
        assert len(album.titles) == 50
        assert await fake_redis.get("spotify:album_tracks:v1:aid_over") is None

    async def test_a_cursor_that_stops_advancing_ends_the_walk_uncached(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        cursor = "https://api.spotify.com/v1/next"
        page = _album_tracks_page(
            [f"T{i}" for i in range(10)], total=30, next_url=cursor
        )
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[_album(page), page, page])
        ) as call:
            album = await spotify.album("aid_stuck")

        assert call.await_count == 2
        assert len(album.titles) == 20
        assert await fake_redis.get("spotify:album_tracks:v1:aid_stuck") is None

    async def test_a_failed_later_page_fails_the_album_and_caches_nothing(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        first = _album(
            _album_tracks_page(
                ["T0"], total=2, next_url="https://api.spotify.com/v1/next/1"
            )
        )
        failure = SpotifyRequestError(500, "https://api.spotify.com/v1/next/1")
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=[first, failure])
        ):
            with pytest.raises(SpotifyRequestError):
                await spotify.album("aid_500")

        assert await fake_redis.get("spotify:album_tracks:v1:aid_500") is None

    async def test_a_hung_page_is_reported_as_a_slow_album(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(spotify_module, "_PLAYLIST_PAGE_TIMEOUT_SECS", 0.01)

        async def hang(*_a: Any, **_k: Any) -> Any:
            await asyncio.Event().wait()

        with patch.object(spotify, "http_call", new=hang):
            with pytest.raises(SpotifyPlaylistTooSlowError) as raised:
                await spotify.album("aid_hang")

        assert raised.value.whole_walk is False
        assert "that album" in raised.value.user_message

    async def test_an_album_that_uses_its_whole_budget_does_not_advise_a_retry(
        self, spotify: Spotify, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(spotify_module, "_PLAYLIST_WALK_TIMEOUT_SECS", 0.08)
        responses = _endless_album(total=1000, per_page=1)

        async def _steady(*_: Any, **__: Any) -> dict[str, Any]:
            await asyncio.sleep(0.02)
            return next(responses)

        with patch.object(spotify, "http_call", new=_steady):
            with pytest.raises(SpotifyPlaylistTooSlowError) as caught:
                await spotify.album("aid_big")

        assert caught.value.whole_walk
        assert "That Spotify album is too large" in caught.value.user_message
        assert "try again" not in caught.value.user_message

    async def test_a_403_is_a_refused_album_not_bad_credentials(
        self, spotify: Spotify
    ) -> None:
        refusal = SpotifyAuthError(403, "endpoint: https://api.spotify.com/v1/albums/x")
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=refusal)):
            with pytest.raises(SpotifyPlaylistForbiddenError) as caught:
                await spotify.album("aid_403")

        assert caught.value.user_message == (
            "Spotify won't share that album's tracks with this bot."
        )

    async def test_a_401_stays_a_credentials_error(self, spotify: Spotify) -> None:
        with patch.object(
            spotify, "http_call", new=AsyncMock(side_effect=SpotifyAuthError(401))
        ):
            with pytest.raises(SpotifyAuthError):
                await spotify.album("aid_401")

    async def test_the_page_bound_is_handed_to_http_call_as_its_429_deadline(
        self, spotify: Spotify
    ) -> None:
        page = _album_tracks_page(["T"], total=1, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ) as call:
            await spotify.album("aid_deadline")

        assert call.await_args_list[0].kwargs["deadline"] is not None


class TestSpotifyAlbumWalkSlot:
    """Page 1 is one request, as a track link is. Pages 2+ are a walk, and share
    the playlist walks' process-wide slots."""

    @staticmethod
    def _two_pages() -> list[dict[str, Any]]:
        cursor = "https://api.spotify.com/v1/next/1"
        return [
            _album(_album_tracks_page(["T0"], total=2, next_url=cursor)),
            _album_tracks_page(["T1"], total=2, next_url=None),
        ]

    async def test_a_one_page_album_takes_no_slot(self, spotify: Spotify) -> None:
        slot = spotify_module._playlist_slot()
        for _ in range(spotify_module._PLAYLIST_WALK_CONCURRENCY):
            await slot.acquire()
        page = _album_tracks_page(["T"], total=1, next_url=None)
        try:
            with patch.object(
                spotify, "http_call", new=AsyncMock(return_value=_album(page))
            ):
                album = await spotify.album("aid_one_page")
        finally:
            for _ in range(spotify_module._PLAYLIST_WALK_CONCURRENCY):
                slot.release()

        assert album.titles == ["T A"]

    async def test_later_pages_hold_a_slot_and_give_it_back(
        self, spotify: Spotify
    ) -> None:
        slot = spotify_module._playlist_slot()
        free = spotify_module._PLAYLIST_WALK_CONCURRENCY
        responses = self._two_pages()
        held: list[int] = []

        async def _call(*_: Any, **__: Any) -> dict[str, Any]:
            held.append(free - slot._value)
            return responses.pop(0)

        with patch.object(spotify, "http_call", new=_call):
            await spotify.album("aid_slot")

        assert held == [0, 1]
        assert slot._value == free

    async def test_the_slot_comes_back_when_a_later_page_fails(
        self, spotify: Spotify
    ) -> None:
        slot = spotify_module._playlist_slot()
        responses = [self._two_pages()[0], SpotifyRequestError(500, "next")]
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=responses)):
            with pytest.raises(SpotifyRequestError):
                await spotify.album("aid_slot_fail")

        assert slot._value == spotify_module._PLAYLIST_WALK_CONCURRENCY

    async def test_no_free_slot_says_spotify_is_busy(self, spotify: Spotify) -> None:
        config.play_resolve_wait_secs.set_override(0.05)
        slot = spotify_module._playlist_slot()
        for _ in range(spotify_module._PLAYLIST_WALK_CONCURRENCY):
            await slot.acquire()
        try:
            with patch.object(
                spotify, "http_call", new=AsyncMock(side_effect=self._two_pages())
            ) as call:
                with pytest.raises(SpotifyBusyError):
                    await spotify.album("aid_busy")
        finally:
            for _ in range(spotify_module._PLAYLIST_WALK_CONCURRENCY):
                slot.release()

        assert call.await_count == 1

    async def test_a_walk_that_finished_during_the_wait_answers_this_one(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """N pastes of one album: the first walk writes the cache, and the rest
        read it once they hold the slot."""
        finished = SpotifyPlaylist(
            name="Discovery",
            titles=["T0 A", "T1 A"],
            duration_secs=360,
            duration_partial=False,
            unavailable=0,
        )
        first = self._two_pages()[0]

        async def _call(*_: Any, **__: Any) -> dict[str, Any]:
            await fake_redis.set(
                "spotify:album_tracks:v1:aid_shared",
                orjson.dumps(spotify_module._playlist_to_cache(finished)),
            )
            return first

        call = AsyncMock(side_effect=_call)
        with patch.object(spotify, "http_call", new=call):
            album = await spotify.album("aid_shared")

        assert album == finished
        assert call.await_count == 1


class TestSpotifyAlbumCache:
    async def test_every_field_round_trips_and_a_hit_makes_no_request(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        page = _album_tracks_page(["A"], total=2, next_url=None, duration_ms=1500)
        page["items"].append(None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ):
            walked = await spotify.album("aid_round")

        assert walked.unavailable == 1
        assert await fake_redis.get("spotify:album_tracks:v1:aid_round") is not None
        with patch.object(spotify, "http_call", new=AsyncMock()) as call:
            assert await spotify.album("aid_round") == walked
        call.assert_not_awaited()

    async def test_the_entry_lives_a_day(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        page = _album_tracks_page(["A"], total=1, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ):
            await spotify.album("aid_ttl")

        assert (
            86_000 < await fake_redis.ttl("spotify:album_tracks:v1:aid_ttl") <= 86_400
        )

    async def test_a_short_walk_is_not_cached(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A truncation cached here is served as the whole album for a day."""
        page = _album_tracks_page(["Only"], total=12, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ):
            album = await spotify.album("aid_short")

        assert album.titles == ["Only A"]
        assert await fake_redis.get("spotify:album_tracks:v1:aid_short") is None

    async def test_an_empty_album_is_not_cached(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        page = _album_tracks_page([], total=0, next_url=None)
        with patch.object(
            spotify, "http_call", new=AsyncMock(return_value=_album(page))
        ):
            album = await spotify.album("aid_empty")

        assert album.titles == []
        assert await fake_redis.get("spotify:album_tracks:v1:aid_empty") is None

    async def test_a_playlist_entry_without_the_album_fields_still_parses(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """Entries written before the album fields existed carry neither."""
        await fake_redis.set(
            "spotify:playlist:v3:pid_old",
            orjson.dumps({"titles": ["T"], "duration_partial": False}),
        )
        with patch.object(spotify, "http_call", new=AsyncMock()) as call:
            playlist = await spotify.playlist("pid_old")

        call.assert_not_awaited()
        assert (playlist.artists, playlist.thumbnail) == ([], None)


class TestSpotifyAuthErrorCopy:
    def test_the_user_message_names_no_endpoint(self) -> None:
        error = SpotifyAuthError(401, "endpoint: https://api.spotify.com/v1/albums/x")
        assert "api.spotify.com" in str(error)
        assert "api.spotify.com" not in error.user_message
        assert "credentials" in error.user_message
