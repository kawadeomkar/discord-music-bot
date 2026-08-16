"""Tests for src/spotify.py — Spotify API auth, response parsing, and Redis cache."""

import asyncio
import contextlib
import redis.asyncio as aioredis
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from src.sources import SpotifyType
from src.spotify import (
    _ALBUM_TTL,
    _HTTP_TIMEOUT,
    _MAX_429_RETRIES,
    _MAX_CONCURRENT_REQUESTS,
    _MAX_RETRY_AFTER_SECS,
    _PLAYLIST_FIELDS,
    _collection_from_cache,
    _cursor_page_cap,
    Spotify,
    SpotifyAuthError,
    SpotifyRateLimitError,
    SpotifyRequestError,
    TrackPage,
)


@pytest.fixture
def mock_auth_response() -> dict[str, Any]:
    return {"access_token": "test_access_token_xyz", "expires_in": 3600}


def _make_mock_session(resp: AsyncMock) -> MagicMock:
    """Return a session mock wired to return resp from .post() and .request()."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = AsyncMock(return_value=resp)
    session.request = AsyncMock(return_value=resp)
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
        mock_session.post.assert_awaited_once()

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
        mock_session.post.assert_awaited_once()

    def test_str_returns_auth_token(self, spotify: Spotify) -> None:
        spotify.auth_token = "my_token"
        assert str(spotify) == "my_token"


def _make_split_session(post_resp: AsyncMock, request_resp: AsyncMock) -> MagicMock:
    """Session mock whose auth POST and API request return different responses —
    needed by validate(), which grants a token then fetches a track."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = AsyncMock(return_value=post_resp)
    session.request = AsyncMock(return_value=request_resp)
    return session


def _resp(status: int, payload: dict[str, Any]) -> AsyncMock:
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    return resp


def _resp_429(retry_after: Optional[str] = None) -> AsyncMock:
    resp = AsyncMock()
    resp.status = 429
    resp.headers = {} if retry_after is None else {"Retry-After": retry_after}
    return resp


class TestRateLimitHandling:
    """429s are retried with Retry-After rather than surfacing raw.

    Pagination turned one request per command into ceil(total/100) and the
    collection lock is per guild, so nothing else bounds the app-wide rate
    against a single client-credentials app.
    """

    async def test_retries_after_429_then_succeeds(self, spotify: Spotify) -> None:
        session = _make_mock_session(AsyncMock())
        session.request = AsyncMock(
            side_effect=[_resp_429("0.5"), _resp(200, {"ok": True})]
        )
        spotify._session_factory = lambda **kw: session
        spotify.token_expiry = time.time() + 3600

        with patch("src.spotify.asyncio.sleep", new=AsyncMock()) as sleep:
            got = await spotify.http_call("https://api.spotify.com/v1/albums/a1")

        assert got == {"ok": True}
        assert session.request.await_count == 2
        sleep.assert_awaited_once_with(0.5)

    async def test_exhausted_retries_raise_rate_limit_error(
        self, spotify: Spotify
    ) -> None:
        """The error type is what stops the drain telling the user to re-run —
        a re-run refetches every page and doubles the load that earned the 429."""
        session = _make_mock_session(_resp_429("3"))
        spotify._session_factory = lambda **kw: session
        spotify.token_expiry = time.time() + 3600

        with (
            patch("src.spotify.asyncio.sleep", new=AsyncMock()),
            pytest.raises(SpotifyRateLimitError) as excinfo,
        ):
            await spotify.http_call("https://api.spotify.com/v1/albums/a1")

        assert excinfo.value.retry_after == 3.0
        assert "try again in about 3 seconds" in excinfo.value.user_message.lower()
        # The endpoint must not reach the user-facing text.
        assert "api.spotify.com" not in excinfo.value.user_message
        assert session.request.await_count == _MAX_429_RETRIES + 1

    async def test_retry_after_is_capped(self, spotify: Spotify) -> None:
        """Spotify can answer with minutes; the enqueue lock is not worth
        holding that long."""
        session = _make_mock_session(_resp_429("600"))
        spotify._session_factory = lambda **kw: session
        spotify.token_expiry = time.time() + 3600

        with (
            patch("src.spotify.asyncio.sleep", new=AsyncMock()) as sleep,
            pytest.raises(SpotifyRateLimitError),
        ):
            await spotify.http_call("https://api.spotify.com/v1/albums/a1")

        assert [c.args[0] for c in sleep.await_args_list] == [
            _MAX_RETRY_AFTER_SECS
        ] * _MAX_429_RETRIES

    async def test_missing_retry_after_falls_back_to_backoff(
        self, spotify: Spotify
    ) -> None:
        session = _make_mock_session(_resp_429())
        spotify._session_factory = lambda **kw: session
        spotify.token_expiry = time.time() + 3600

        with (
            patch("src.spotify.asyncio.sleep", new=AsyncMock()) as sleep,
            pytest.raises(SpotifyRateLimitError) as excinfo,
        ):
            await spotify.http_call("https://api.spotify.com/v1/albums/a1")

        assert excinfo.value.retry_after is None
        assert [c.args[0] for c in sleep.await_args_list] == [1.0, 2.0, 4.0]
        # The no-delay user_message arm is production-reachable (absent header,
        # or Retry-After: 0) and renders straight into a channel embed from
        # _drain_collection_tail — it must exist and must not leak the request.
        assert "rate-limiting" in excinfo.value.user_message
        assert "api.spotify.com" not in excinfo.value.user_message

    async def test_http_date_retry_after_falls_back_to_backoff(
        self, spotify: Spotify
    ) -> None:
        """The HTTP-date Retry-After form the docstring declines to parse:
        malformed-header handling fell out of coverage when the old
        `not-a-number` case was deleted, and without the except a date header
        turns a 429 into an unhandled ValueError mid-drain — whose generic
        notice tells the user to re-run, doubling the load that earned the
        429."""
        session = _make_mock_session(_resp_429("Wed, 21 Oct 2026 07:28:00 GMT"))
        spotify._session_factory = lambda **kw: session
        spotify.token_expiry = time.time() + 3600

        with (
            patch("src.spotify.asyncio.sleep", new=AsyncMock()) as sleep,
            pytest.raises(SpotifyRateLimitError) as excinfo,
        ):
            await spotify.http_call("https://api.spotify.com/v1/albums/a1")

        # Unparseable ⇒ treated exactly like absent: exponential backoff,
        # never "zero seconds, hammer immediately".
        assert excinfo.value.retry_after is None
        assert [c.args[0] for c in sleep.await_args_list] == [1.0, 2.0, 4.0]

    async def test_concurrent_calls_are_bounded_process_wide(
        self, spotify: Spotify
    ) -> None:
        """The collection lock is per guild, so only this semaphore stops N
        draining guilds multiplying the rate against one Spotify app."""
        in_flight = 0
        peak = 0

        async def slow_request(*a: Any, **kw: Any) -> AsyncMock:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return _resp(200, {"ok": True})

        session = _make_mock_session(AsyncMock())
        session.request = AsyncMock(side_effect=slow_request)
        spotify._session_factory = lambda **kw: session
        spotify.token_expiry = time.time() + 3600

        await asyncio.gather(
            *(
                spotify.http_call("https://api.spotify.com/v1/albums/a1")
                for _ in range(_MAX_CONCURRENT_REQUESTS * 3)
            )
        )

        assert peak <= _MAX_CONCURRENT_REQUESTS


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

        session.request.assert_awaited_once()

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
        session.request.assert_not_awaited()  # never got to the track call

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


class TestSpotifyHttpCall:
    async def test_http_call_raises_on_non_200(self, spotify: Spotify) -> None:
        spotify.auth_token = "prefetched_token"
        spotify.token_expiry = time.time() + 3600  # skip _refresh_token
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session = _make_mock_session(mock_response)
        spotify._session_factory = lambda **kw: mock_session

        with pytest.raises(SpotifyRequestError, match="stat: 404") as excinfo:
            await spotify.http_call("https://api.spotify.com/v1/tracks/bad")

        # The args keep the detail for the log and span; only user_message is
        # shown, so the endpoint must not appear there.
        assert excinfo.value.status == 404
        assert "check the link" in excinfo.value.user_message
        assert "api.spotify.com" not in excinfo.value.user_message

    async def test_server_error_user_message_says_try_again(
        self, spotify: Spotify
    ) -> None:
        spotify.auth_token = "prefetched_token"
        spotify.token_expiry = time.time() + 3600
        mock_response = AsyncMock()
        mock_response.status = 503
        spotify._session_factory = lambda **kw: _make_mock_session(mock_response)

        with pytest.raises(SpotifyRequestError) as excinfo:
            await spotify.http_call("https://api.spotify.com/v1/albums/a1")

        assert "having problems" in excinfo.value.user_message
        assert "503" not in excinfo.value.user_message

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

    async def test_track_ttl_is_24h(self, spotify: Spotify, fake_redis: Redis) -> None:
        with patch.object(
            spotify,
            "http_call",
            new=AsyncMock(return_value={"name": "S", "artists": [{"name": "A"}]}),
        ):
            await spotify.track("ttl_test_track")
        ttl = await fake_redis.ttl("spotify:track:ttl_test_track")
        assert 86390 <= ttl <= 86400

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


# ── Streaming collection pagers ───────────────────────────────────────────────


def _album_track(i: int) -> dict[str, Any]:
    return {"name": f"Track {i}", "artists": [{"name": "Artist"}]}


def _album_api(
    aid: str,
    *,
    total: int,
    page1_limit: int,
    fail_at_offset: Optional[int] = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """An http_call side_effect emulating GET /v1/albums/{id} and the `next`
    cursor walk over its /tracks pages. Returns (side_effect, calls) — calls
    records endpoint+params."""
    calls: list[dict[str, Any]] = []

    def _next(offset: int) -> Optional[str]:
        return (
            None
            if offset >= total
            else f"https://api.spotify.com/v1/albums/{aid}/tracks"
            f"?offset={offset}&limit={page1_limit}"
        )

    async def call(
        endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
    ) -> dict[str, Any]:
        calls.append({"endpoint": endpoint, "params": params})
        if endpoint.endswith(f"v1/albums/{aid}"):
            page1_count = min(page1_limit, total)
            return {
                "name": "200% Electronica",
                "artists": [{"name": "ESPRIT 空想"}, {"name": "George Clanton"}],
                "images": [{"url": "https://i.scdn.co/image/640x640"}],
                "release_date": "2017-11-17",
                "tracks": {
                    "items": [_album_track(i) for i in range(page1_count)],
                    "limit": page1_limit,
                    "total": total,
                    "next": _next(page1_count),
                },
            }
        # A cursor page: Spotify's own `next` URL, followed verbatim.
        assert params is None, "cursor pages carry their query in the URL"
        offset = int(endpoint.split("offset=")[1].split("&")[0])
        if fail_at_offset is not None and offset == fail_at_offset:
            raise SpotifyAuthError(401, "revoked mid-drain")
        end = min(offset + page1_limit, total)
        return {
            "items": [_album_track(i) for i in range(offset, end)],
            "next": _next(end),
        }

    return call, calls


def _cursor_offset(endpoint: str) -> int:
    """The offset a recorded cursor-page URL names."""
    return int(endpoint.split("offset=")[1].split("&")[0])


def _playlist_track(i: int) -> dict[str, Any]:
    return {"track": {"type": "track", "name": f"T{i}", "artists": [{"name": "A"}]}}


def _playlist_api(
    pid: str, *, total: int, extra_items: Optional[list[dict[str, Any]]] = None
) -> tuple[Any, list[dict[str, Any]]]:
    """An http_call side_effect emulating /v1/playlists/{id}/tracks with a
    `next` cursor. extra_items are appended to page 1 (for skip-guard tests)
    and do not count toward `total`'s real-track numbering."""
    calls: list[dict[str, Any]] = []
    page = 100

    async def call(
        endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
    ) -> dict[str, Any]:
        calls.append({"endpoint": endpoint, "params": params})
        if params is not None:
            offset = 0
        else:
            offset = int(endpoint.split("offset=")[1].split("&")[0])
        end = min(offset + page, total)
        items = [_playlist_track(i) for i in range(offset, end)]
        if offset == 0 and extra_items:
            items.extend(extra_items)
        return {
            "items": items,
            "total": total,
            "next": (
                None
                if end >= total
                else f"https://api.spotify.com/v1/playlists/{pid}/tracks"
                f"?offset={end}&limit=100&fields=next,total"
            ),
        }

    return call, calls


async def _drain(gen: Any) -> list[TrackPage]:
    async with contextlib.aclosing(gen) as pages:
        return [page async for page in pages]


class TestCollectionFromCache:
    """The cache-read wire discipline: a wrong-TYPED field means garbage and
    the whole entry is a miss (re-fetched, never rendered), while MISSING
    optional fields — an older build's entry — stay readable."""

    def test_non_dict_is_a_miss(self) -> None:
        assert _collection_from_cache(SpotifyType.ALBUM, "cid", ["list"]) is None

    def test_non_list_titles_is_a_miss(self) -> None:
        raw = {"titles": "garbage"}
        assert _collection_from_cache(SpotifyType.ALBUM, "cid", raw) is None

    def test_non_int_total_is_a_miss(self) -> None:
        raw: dict[str, Any] = {"titles": ["T A"], "total": "11"}
        assert _collection_from_cache(SpotifyType.ALBUM, "cid", raw) is None

    @pytest.mark.parametrize("field", ["name", "thumbnail", "release_date"])
    def test_non_str_identity_field_is_a_miss(self, field: str) -> None:
        raw: dict[str, Any] = {"titles": ["T A"], "total": 1, field: 42}
        assert _collection_from_cache(SpotifyType.ALBUM, "cid", raw) is None

    def test_missing_optional_fields_stay_readable(self) -> None:
        got = _collection_from_cache(SpotifyType.PLAYLIST, "cid", {"titles": ["T A"]})
        assert got is not None
        collection, titles = got
        assert titles == ["T A"]
        assert collection.total == 1  # defaults to len(titles)
        assert collection.name is None
        assert collection.thumbnail is None


class TestAlbumStream:
    async def test_single_page_album_is_one_call(self, spotify: Spotify) -> None:
        api, calls = _album_api("alb1", total=11, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("alb1"))

        assert len(calls) == 1
        assert len(pages) == 1
        assert pages[0].is_last
        assert len(pages[0].titles) == 11
        c = pages[0].collection
        assert c.kind is SpotifyType.ALBUM
        assert c.name == "200% Electronica"
        assert c.artists == ["ESPRIT 空想", "George Clanton"]
        assert c.artist_line == "ESPRIT 空想, George Clanton"
        assert c.thumbnail == "https://i.scdn.co/image/640x640"
        assert c.release_date == "2017-11-17"
        assert c.total == 11

    async def test_multi_page_album_follows_cursor_in_order(
        self, spotify: Spotify
    ) -> None:
        api, calls = _album_api("alb976", total=976, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("alb976"))

        assert len(calls) == 1 + 19
        assert [_cursor_offset(c["endpoint"]) for c in calls[1:]] == list(
            range(50, 976, 50)
        )
        titles = [t for p in pages for t in p.titles]
        assert titles == [f"Track {i} Artist" for i in range(976)]
        assert pages[-1].is_last and not pages[0].is_last
        # The frozen collection is shared, not rebuilt per page.
        assert all(p.collection is pages[0].collection for p in pages)

    async def test_cursor_honors_spotify_stride(self, spotify: Spotify) -> None:
        """A 20-item page 1 (Spotify's documented default on the limit-less
        embedded pager) walks the cursor at stride 20. The pages are
        Spotify's own `next` URLs followed verbatim, so a stride mismatch
        cannot skip or duplicate tracks by construction — the failure the
        offset-arithmetic fanout this replaced had to derive its way around."""
        api, calls = _album_api("albstride", total=120, page1_limit=20)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("albstride"))

        assert [_cursor_offset(c["endpoint"]) for c in calls[1:]] == list(
            range(20, 120, 20)
        )
        titles = [t for p in pages for t in p.titles]
        assert titles == [f"Track {i} Artist" for i in range(120)]

    async def test_boundary_exact_page_no_second_call(self, spotify: Spotify) -> None:
        api, calls = _album_api("alb50", total=50, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("alb50"))
        assert len(calls) == 1
        assert pages[0].is_last

    async def test_boundary_one_over_page_one_extra_call(
        self, spotify: Spotify
    ) -> None:
        api, calls = _album_api("alb51", total=51, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("alb51"))
        assert len(calls) == 2
        assert _cursor_offset(calls[1]["endpoint"]) == 50
        assert [t for p in pages for t in p.titles] == [
            f"Track {i} Artist" for i in range(51)
        ]

    async def test_empty_album_yields_one_empty_last_page(
        self, spotify: Spotify
    ) -> None:
        api, _ = _album_api("albempty", total=0, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("albempty"))
        assert len(pages) == 1
        assert pages[0].titles == [] and pages[0].is_last

    async def test_empty_album_is_not_cached(
        self, spotify: Spotify, fake_redis: aioredis.Redis
    ) -> None:
        """Immutability makes a genuinely empty album safe to cache, but it
        does nothing for a malformed or partial response — the likelier
        explanation, and one that would stick for 24h."""
        api, _ = _album_api("albempty2", total=0, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await _drain(spotify.album_stream("albempty2"))

        assert await fake_redis.get("spotify:album_tracks:albempty2") is None

    async def test_single_page_album_is_cached(
        self, spotify: Spotify, fake_redis: aioredis.Redis
    ) -> None:
        """The early-return write covers every album that fits one page — i.e.
        essentially all of them. Only the multi-page write was asserted, so
        deleting this one cost a Spotify call per play with a green suite."""
        api, _ = _album_api("alb1p", total=11, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)) as call:
            await _drain(spotify.album_stream("alb1p"))
            first_calls = call.await_count
            # A second stream must be served entirely from the cache.
            pages = await _drain(spotify.album_stream("alb1p"))
            assert call.await_count == first_calls

        assert await fake_redis.ttl("spotify:album_tracks:alb1p") == _ALBUM_TTL
        assert len([t for p in pages for t in p.titles]) == 11

    async def test_short_cursor_drain_is_not_cached(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A cursor walk that ends short of the album's own total is a
        truncation, not a drain — albums never skip items, so short means
        wrong, and caching it would serve the short album for 24h with no
        error anywhere (the embed already promised `total` songs). A miss
        self-heals on the next -play; a cache write sticks."""
        api, _ = _album_api("albshort", total=150, page1_limit=50)

        async def short(
            endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
        ) -> dict[str, Any]:
            resp = await api(endpoint, params, **kw)
            if "offset=100" in endpoint:
                resp["items"] = []  # partial degradation: page comes up empty
            return resp

        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=short)):
            pages = await _drain(spotify.album_stream("albshort"))

        assert len([t for p in pages for t in p.titles]) == 100  # yielded short
        assert pages[-1].is_last  # the walk itself completed
        assert await fake_redis.get("spotify:album_tracks:albshort") is None

    async def test_stuck_cursor_abandons_at_cap_and_writes_no_cache(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A `next` that stops advancing must not spin duplicate pages into
        the consumer's queue until its 45s drain budget expires: the page cap
        abandons the walk, keeps what was yielded, and skips the cache."""
        api, _ = _album_api("albstuck", total=100, page1_limit=50)
        frozen = "https://api.spotify.com/v1/albums/albstuck/tracks?offset=50&limit=50"

        async def stuck(
            endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
        ) -> dict[str, Any]:
            resp = await api(endpoint, params, **kw)
            if "tracks" in resp:
                resp["tracks"]["next"] = frozen  # page 1's embedded pager
            else:
                resp["next"] = frozen  # every cursor page points at itself
            return resp

        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=stuck)):
            pages = await _drain(spotify.album_stream("albstuck"))

        # cap = ceil(100/50) + 2 = 4 pages, then abandoned — finite, honest.
        assert len(pages) == _cursor_page_cap(100, 50) == 4
        assert not pages[-1].is_last
        assert await fake_redis.get("spotify:album_tracks:albstuck") is None

    async def test_a_malformed_empty_page_one_cannot_divide_by_zero(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """Two guards that are individually inert and together are a crash.

        The page cap's stride is page 1's own item count, and an empty page 1
        that still carries a `next` makes that 0. `_cursor_page_cap`'s
        `max(1, page_size)` and the call site's `or 50` each stop the
        ZeroDivisionError on their own, so dropping either passed the whole
        suite — and the exception would come out of the generator, land outside
        `_command_error`'s allowlist, and reach the user as a raw
        ZeroDivisionError."""
        pages_sent = 0

        async def malformed(
            endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
        ) -> dict[str, Any]:
            nonlocal pages_sent
            nxt = "https://api.spotify.com/v1/albums/albz/tracks?offset=0&limit=50"
            if endpoint.endswith("v1/albums/albz"):
                return {
                    "name": "Empty",
                    "artists": [{"name": "A"}],
                    "tracks": {"items": [], "limit": 50, "total": 10, "next": nxt},
                }
            pages_sent += 1
            # Keeps the cursor live so the cap is what ends the walk.
            return {"items": [_album_track(0)], "next": nxt}

        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=malformed)):
            pages = await _drain(spotify.album_stream("albz"))

        assert len(pages) == _cursor_page_cap(10, 50)  # finite, not a crash
        assert await fake_redis.get("spotify:album_tracks:albz") is None

    def test_the_page_cap_survives_a_zero_stride(self) -> None:
        """The other half of the same guard, asserted directly — `max(1, ...)`
        is the reason this returns a number instead of raising."""
        assert _cursor_page_cap(100, 0) == 102

    async def test_a_null_or_nameless_album_item_is_skipped(
        self, spotify: Spotify
    ) -> None:
        """`_track_search_title` subscripts name and artists, so it is not total.

        The album loop had no guard at all — its comment justified that on SHAPE
        grounds (album items are the track, no ["track"] wrapper, no episodes),
        which says nothing about absence. A null item on a malformed page raised
        TypeError from inside the generator, and TypeError is not in
        `_command_error`'s allowlist, so it reached the user as
        "**TypeError:** 'NoneType' object is not subscriptable"."""

        async def ragged(
            endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
        ) -> dict[str, Any]:
            return {
                "name": "Ragged",
                "artists": [{"name": "A"}],
                "tracks": {
                    "items": [
                        _album_track(0),
                        None,
                        {"artists": [{"name": "A"}]},  # no name
                        {"name": "No artists"},
                        _album_track(1),
                    ],
                    "limit": 50,
                    "total": 5,
                    "next": None,
                },
            }

        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=ragged)):
            pages = await _drain(spotify.album_stream("albragged"))

        assert [t for p in pages for t in p.titles] == [
            "Track 0 Artist",
            "Track 1 Artist",
        ]

    async def test_failed_page_propagates_and_writes_no_cache(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A failed page must surface its own error to the caller and must
        not cache the partial collection."""
        api, _ = _album_api("albfail", total=976, page1_limit=50, fail_at_offset=150)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            with pytest.raises(SpotifyAuthError):
                await _drain(spotify.album_stream("albfail"))
        assert await fake_redis.get("spotify:album_tracks:albfail") is None

    async def test_first_yield_costs_exactly_one_call(self, spotify: Spotify) -> None:
        """The property the whole design exists for: page 1 is available
        before any later page is requested."""
        api, calls = _album_api("albfirst", total=976, page1_limit=50)
        gen = spotify.album_stream("albfirst")
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            async with contextlib.aclosing(gen):
                page1 = await anext(gen)
                assert len(calls) == 1
                assert len(page1.titles) == 50

    async def test_cache_written_only_on_full_drain(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        api, calls = _album_api("albcache", total=60, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            first = await _drain(spotify.album_stream("albcache"))
            assert len(calls) == 2
            ttl = await fake_redis.ttl("spotify:album_tracks:albcache")
            assert 86390 <= ttl <= 86400
            # Second stream: served entirely from cache — zero further calls,
            # one is_last page, identical titles and identity.
            second = await _drain(spotify.album_stream("albcache"))
            assert len(calls) == 2
        assert len(second) == 1 and second[0].is_last
        assert [t for p in second for t in p.titles] == [
            t for p in first for t in p.titles
        ]
        assert second[0].collection.name == "200% Electronica"
        assert second[0].collection.kind is SpotifyType.ALBUM

    async def test_abandoned_stream_writes_no_cache(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """-playnow / preemption path: consuming page 1 then closing must not
        poison the cache with a truncated collection — and must not leave an
        unfinalized async generator behind (filterwarnings=error would fail
        the suite on the resulting warning)."""
        api, _ = _album_api("albdrop", total=976, page1_limit=50)
        gen = spotify.album_stream("albdrop")
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await anext(gen)
            await gen.aclose()
        assert await fake_redis.get("spotify:album_tracks:albdrop") is None

    async def test_cache_key_does_not_collide_with_albums_method(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """albums() owns spotify:album:{id} with a different shape; the stream
        must not read it as its own cache."""
        with patch.object(
            spotify,
            "http_call",
            new=AsyncMock(return_value={"albums": [{"name": "X"}]}),
        ):
            await spotify.albums("albkey")
        assert await fake_redis.get("spotify:album:albkey") is not None

        api, calls = _album_api("albkey", total=5, page1_limit=50)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.album_stream("albkey"))
        assert len(calls) == 1  # fetched fresh — the albums() entry was not reused
        assert len(pages[0].titles) == 5


class TestPlaylistStream:
    async def test_follows_next_to_exhaustion(self, spotify: Spotify) -> None:
        """The >100-track truncation regression guard: every page arrives."""
        api, calls = _playlist_api("pl250", total=250)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.playlist_stream("pl250"))

        assert len(calls) == 3
        titles = [t for p in pages for t in p.titles]
        assert titles == [f"T{i} A" for i in range(250)]
        assert pages[-1].is_last and not pages[0].is_last
        assert pages[0].collection.total == 250

    async def test_stuck_cursor_abandons_at_cap(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """Termination is otherwise Spotify's promise alone: a `next` that
        stops advancing would spin ~100 duplicates per iteration into the
        consumer's queue (and its Redis mirror) until the 45s drain budget
        expired. The cap abandons the walk finite and uncached."""
        api, _ = _playlist_api("plstuck", total=100)
        frozen = (
            "https://api.spotify.com/v1/playlists/plstuck/tracks"
            "?offset=0&limit=100&fields=next,total"
        )

        async def stuck(
            endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
        ) -> dict[str, Any]:
            resp = await api(endpoint, params, **kw)
            resp["next"] = frozen
            return resp

        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=stuck)):
            pages = await _drain(spotify.playlist_stream("plstuck"))

        # cap = ceil(100/100) + 2 = 3 pages, then abandoned; nothing cached.
        assert len(pages) == _cursor_page_cap(100, 100) == 3
        assert not pages[-1].is_last
        assert await fake_redis.get("spotify:playlist_tracks:plstuck") is None

    async def test_first_request_targets_playlist_tracks_endpoint(
        self, spotify: Spotify
    ) -> None:
        """Nothing else pins the URL itself — a typo'd endpoint would pass
        every mask/cursor assertion (ports the deleted
        test_playlist_calls_correct_endpoint)."""
        api, calls = _playlist_api("plendpoint", total=5)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await _drain(spotify.playlist_stream("plendpoint"))

        assert "v1/playlists/plendpoint/tracks" in calls[0]["endpoint"]

    async def test_exactly_one_full_page_makes_no_second_call(
        self, spotify: Spotify
    ) -> None:
        """The page-limit boundary: exactly 100 tracks with next=null is one
        call — the album 50/51 boundaries had this pin, the playlist path did
        not."""
        api, calls = _playlist_api("pl100", total=100)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.playlist_stream("pl100"))

        assert len(calls) == 1
        assert pages[-1].is_last
        assert sum(len(p.titles) for p in pages) == 100

    async def test_mid_drain_failure_writes_no_cache(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        """A cursor-page failure after page 1 must abandon without caching a
        truncated collection — the album fanout had this guard, the playlist
        path did not."""
        calls: list[str] = []

        async def api(
            endpoint: str, params: Optional[dict[str, Any]] = None, **kw: Any
        ) -> dict[str, Any]:
            calls.append(endpoint)
            if len(calls) == 1:
                return {
                    "items": [_playlist_track(i) for i in range(100)],
                    "total": 250,
                    "next": (
                        "https://api.spotify.com/v1/playlists/plfail"
                        "/tracks?offset=100&limit=100&fields=next,total"
                    ),
                }
            raise SpotifyAuthError(401, "revoked mid-cursor")

        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            with pytest.raises(SpotifyAuthError):
                await _drain(spotify.playlist_stream("plfail"))

        assert len(calls) == 2
        assert await fake_redis.get("spotify:playlist_tracks:plfail") is None

    async def test_first_request_mask_and_params(self, spotify: Spotify) -> None:
        api, calls = _playlist_api("plmask", total=10)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await _drain(spotify.playlist_stream("plmask"))
        params = calls[0]["params"]
        assert params["fields"] == _PLAYLIST_FIELDS
        assert "next" in params["fields"] and "total" in params["fields"]
        assert params["additional_types"] == "track"
        assert params["limit"] == 100

    async def test_next_url_followed_verbatim(self, spotify: Spotify) -> None:
        """`next` carries the mask forward — no re-parameterisation."""
        api, calls = _playlist_api("plnext", total=150)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await _drain(spotify.playlist_stream("plnext"))
        assert calls[1]["params"] is None
        assert "offset=100" in calls[1]["endpoint"]

    async def test_skips_null_and_episode_items(self, spotify: Spotify) -> None:
        """Removed/local tracks arrive as null; episodes carry no artists (and
        `type: episode` when the mask exposes it). All are skipped without
        killing the drain, so total is an upper bound on what is queued."""
        weird: list[dict[str, Any]] = [
            {"track": None},
            {"track": {"type": "episode", "name": "Podcast Ep 1"}},
            # Disqualified by `type` alone — it carries a name and artists, so
            # the next guard would pass it through. The only item here that
            # isolates the `type` check.
            {
                "track": {
                    "type": "episode",
                    "name": "Podcast Ep 2",
                    "artists": [{"name": "Host"}],
                }
            },
            {"track": {"name": "Maskless Episode"}},  # no type, no artists
            {"track": {"type": "track", "name": "", "artists": [{"name": "A"}]}},
        ]
        api, _ = _playlist_api("plweird", total=3, extra_items=weird)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.playlist_stream("plweird"))
        titles = [t for p in pages for t in p.titles]
        assert titles == ["T0 A", "T1 A", "T2 A"]

    async def test_empty_playlist_yields_one_empty_last_page(
        self, spotify: Spotify
    ) -> None:
        """Ported from main's deleted test_playlist_empty_items_returns_empty_list;
        TestAlbumStream had an equivalent, the playlist side never did."""
        api, _ = _playlist_api("plempty", total=0)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.playlist_stream("plempty"))
        assert len(pages) == 1
        assert pages[0].titles == [] and pages[0].is_last

    async def test_empty_playlist_is_not_cached(
        self, spotify: Spotify, fake_redis: aioredis.Redis
    ) -> None:
        """A negative cache here repeats "no queueable tracks" for an hour.

        _PLAYLIST_TTL is 1h *because* playlists are user-editable, and the edit
        that matters most is the one the error prompts: the user is told the
        playlist is empty, adds songs, retries — and gets the cached emptiness
        back. Not caching it costs one request.
        """
        api, _ = _playlist_api("plempty2", total=0)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await _drain(spotify.playlist_stream("plempty2"))

        assert await fake_redis.get("spotify:playlist_tracks:plempty2") is None

    async def test_all_episode_playlist_is_not_cached(
        self, spotify: Spotify, fake_redis: aioredis.Redis
    ) -> None:
        """Same rule via the skip guards: every item filtered out is still an
        empty result, and the playlist can gain a real track a minute later."""
        episodes: list[dict[str, Any]] = [
            {"track": {"type": "episode", "name": f"Ep {i}", "artists": []}}
            for i in range(3)
        ]
        api, _ = _playlist_api("plpods", total=0, extra_items=episodes)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.playlist_stream("plpods"))

        assert [t for p in pages for t in p.titles] == []
        assert await fake_redis.get("spotify:playlist_tracks:plpods") is None

    async def test_multi_artist_track_renders_all_artists(
        self, spotify: Spotify
    ) -> None:
        """Ported from the deleted playlist() tests: the guarded unwrap must
        still hand multi-artist tracks to _track_search_title intact."""
        collab = {
            "track": {
                "type": "track",
                "name": "Collab",
                "artists": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            }
        }
        api, _ = _playlist_api("plcollab", total=1, extra_items=[collab])
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            pages = await _drain(spotify.playlist_stream("plcollab"))
        assert [t for p in pages for t in p.titles] == ["T0 A", "Collab A B C"]

    async def test_first_yield_costs_exactly_one_call(self, spotify: Spotify) -> None:
        api, calls = _playlist_api("plfirst", total=716)
        gen = spotify.playlist_stream("plfirst")
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            async with contextlib.aclosing(gen):
                page1 = await anext(gen)
                assert len(calls) == 1
                assert len(page1.titles) == 100

    async def test_cache_written_only_on_full_drain(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        api, calls = _playlist_api("plcache", total=150)
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await _drain(spotify.playlist_stream("plcache"))
            ttl = await fake_redis.ttl("spotify:playlist_tracks:plcache")
            assert 3590 <= ttl <= 3600
            second = await _drain(spotify.playlist_stream("plcache"))
            assert len(calls) == 2  # cache hit — no further HTTP
        assert len(second) == 1 and second[0].is_last
        assert len(second[0].titles) == 150

    async def test_abandoned_stream_writes_no_cache(
        self, spotify: Spotify, fake_redis: Redis
    ) -> None:
        api, _ = _playlist_api("pldrop", total=716)
        gen = spotify.playlist_stream("pldrop")
        with patch.object(spotify, "http_call", new=AsyncMock(side_effect=api)):
            await anext(gen)
            await gen.aclose()
        assert await fake_redis.get("spotify:playlist_tracks:pldrop") is None


class TestHttpTimeout:
    async def test_http_call_passes_explicit_timeout(self, spotify: Spotify) -> None:
        """aiohttp's default is ClientTimeout(total=300) — a hung page would
        hold the per-guild collection lock for five minutes without this."""
        spotify.auth_token = "t"
        spotify.token_expiry = time.time() + 3600
        captured: dict[str, Any] = {}
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={})
        session = _make_mock_session(mock_resp)

        def factory(**kw: Any) -> MagicMock:
            captured.update(kw)
            return session

        spotify._session_factory = factory
        await spotify.http_call("https://api.spotify.com/v1/albums/x")
        assert captured["timeout"] is _HTTP_TIMEOUT

    def test_http_timeout_value_is_pinned(self) -> None:
        """Identity alone (`is _HTTP_TIMEOUT`) proves nothing about the VALUE.
        At 60s a single hung page outlives _COLLECTION_DRAIN_TIMEOUT_SECS, so
        the drain deadline fires inside an in-flight request rather than
        between pages. The full nesting chain is asserted in
        test_musicbot.py's timeout-chain test; the value itself is pinned
        here, next to the constant's own module."""
        assert _HTTP_TIMEOUT.total == 30

    async def test_refresh_token_passes_explicit_timeout(
        self, spotify: Spotify, mock_auth_response: dict[str, Any]
    ) -> None:
        captured: dict[str, Any] = {}
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_auth_response)
        session = _make_mock_session(mock_resp)

        def factory(**kw: Any) -> MagicMock:
            captured.update(kw)
            return session

        spotify._session_factory = factory
        await spotify._refresh_token()
        assert captured["timeout"] is _HTTP_TIMEOUT
