import asyncio
import os
import time
from typing import Any, Optional, Union, cast
from collections.abc import Awaitable, Callable

import aiohttp
import ujson

import redis.asyncio as aioredis

from opentelemetry import trace

from src.redis_client import (
    cache_get,
    cache_set,
    spotify_token_get_with_ttl,
    spotify_token_set,
)
from src.telemetry import get_tracer
from src.util import get_logger

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_TRACK_TTL = 86400  # 24h — track titles/artists don't change
_PLAYLIST_TTL = 3600  # 1h  — playlists can be edited by users
_ARTIST_TTL = 86400  # 24h
_ALBUM_TTL = 86400  # 24h

# Carried by the session, so it bounds the token grant as well as every API
# call; aiohttp's default of 300s held a command for five minutes.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

_MAX_429_RETRIES = 3
# Spotify's Retry-After can be minutes, and a command holding that long is
# indistinguishable from a hung bot; beyond the cap the caller is told to wait.
_MAX_RETRY_AFTER_SECS = 10.0


def _track_search_title(track: dict[str, Any]) -> str:
    """ "<name> <artist1> <artist2> ...", the yt-dlp search string a Spotify
    track resolves to. Shared by track() and playlist()."""
    return track["name"] + "".join(f" {a['name']}" for a in track["artists"])


class SpotifyAuthError(Exception):
    """Spotify rejected the configured client credentials: a non-2xx token
    grant, or a 401/403 from an API call. Never for network errors, timeouts or
    other codes — those say nothing about validity, and startup validation
    disables the source only on this."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        super().__init__(
            f"Spotify rejected the credentials (HTTP {status})"
            + (f": {detail}" if detail else "")
        )


class SpotifyRequestError(Exception):
    """A non-2xx Spotify response that says nothing about the credentials (so
    it may never disable the source). `user_message` reaches the channel as a
    sentence rather than an endpoint and a status code."""

    def __init__(self, status: int, endpoint: str, params: Any = None) -> None:
        self.status = status
        self.endpoint = endpoint
        super().__init__(f"endpoint: {endpoint} stat: {status} params: {params}")

    @property
    def user_message(self) -> str:
        if self.status == 404:
            return (
                "Spotify doesn't have that — the link may be private, "
                "region-locked, or no longer exist."
            )
        return (
            f"Spotify returned an error (HTTP {self.status}). "
            "It may be having a moment — try again shortly."
        )


class SpotifyRateLimitError(Exception):
    """Spotify rate-limited us and the retries were spent. `user_message` says
    "wait", never "try again": a re-run re-issues every request that earned the
    429."""

    def __init__(self, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__(
            "Spotify rate limit exceeded"
            + (f" (retry after {retry_after}s)" if retry_after else "")
        )

    @property
    def user_message(self) -> str:
        wait = (
            f" Try again in about {int(self.retry_after)}s."
            if self.retry_after
            else " Try again in a moment."
        )
        return "Spotify is rate-limiting this bot right now." + wait


def _retry_after_secs(resp: aiohttp.ClientResponse) -> Optional[float]:
    """Spotify's Retry-After header in seconds; None when absent or malformed,
    so the caller backs off exponentially instead of hammering."""
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class Spotify:
    """Thin async client for the Spotify Web API: client-credentials auth with
    auto-refresh, and Redis-backed caching of track/playlist/artist/album
    lookups."""

    spotify_endpoint = "https://api.spotify.com/"
    auth_endpoint = "https://accounts.spotify.com/api/token"

    def __init__(
        self,
        redis: Optional[aioredis.Redis] = None,
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.token_expiry = 0.0
        self.auth_token: str = ""
        self._auth_lock = asyncio.Lock()
        self._redis = redis
        self._session_factory = session_factory or aiohttp.ClientSession
        # Built on first use, so a deployment that never uses Spotify never
        # opens a connector.
        self._session: Optional[aiohttp.ClientSession] = None
        self._closed = False

    def __str__(self) -> str:
        # Never the bearer token: it would land in logs that ship to Loki.
        # __repr__ is aliased so an exception repr cannot leak it either.
        client = self.client_id or "unset"
        return (
            f"Spotify(client_id={client[:6]}…, "
            f"token={'set' if self.auth_token else 'unset'})"
        )

    __repr__ = __str__

    def _session_or_create(self) -> aiohttp.ClientSession:
        """The client's session, created on first use so one connection pool
        serves every call. Rebuilt when closed from outside, never after
        aclose(): a caller arriving then would strand a session nothing
        closes."""
        if self._closed:
            raise RuntimeError("Spotify client is closed")
        if self._session is None or self._session.closed:
            self._session = cast(
                aiohttp.ClientSession,
                self._session_factory(
                    json_serialize=ujson.dumps, timeout=_HTTP_TIMEOUT
                ),
            )
        return self._session

    async def aclose(self) -> None:
        """Release the session for good (the cog's unload); safe to call twice."""
        self._closed = True
        session, self._session = self._session, None
        if session is not None and not session.closed:
            await session.close()

    # ── Auth ─────────────────────────────────────────────────────────────────

    async def _refresh_token(
        self, use_cache: bool = True, strict: bool = False
    ) -> None:
        """Fetch a fresh access token via client-credentials and update expiry.
        `use_cache=False` bypasses the Redis-cached token so the configured
        credentials are exercised; `strict=True` raises SpotifyAuthError on a
        non-2xx grant instead of a KeyError. Both are validate()'s."""
        if use_cache and self._redis is not None:
            cached = await spotify_token_get_with_ttl(self._redis)
            if cached is not None:
                token, ttl = cached
                self.auth_token = token
                self.token_expiry = time.time() + ttl
                return

        self.token_expiry = time.time()
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        session = self._session_or_create()
        async with session.post(self.auth_endpoint, data=data) as resp:
            if strict and resp.status not in (200, 201):
                raise SpotifyAuthError(resp.status, "client-credentials grant failed")
            resp_data = await resp.json(content_type=None)
        self.auth_token = resp_data["access_token"]
        expires_in: int = resp_data["expires_in"]
        self.token_expiry += expires_in
        await spotify_token_set(self._redis, self.auth_token, expires_in)

    async def http_call(
        self,
        endpoint_route: str,
        params: Optional[dict[str, Union[str, int]]] = None,
        headers: Optional[dict[str, str]] = None,
        data: Optional[dict[str, str]] = None,
        http_method: str = "GET",
    ) -> Any:
        """Authenticated request, refreshing the token first if expired. Raises
        on any non-2xx. `Any` because the response shape is chosen by the
        caller's URL; callers that read named fields narrow at their own
        boundary."""
        if time.time() > self.token_expiry:
            async with self._auth_lock:
                if time.time() > self.token_expiry:
                    await self._refresh_token()

        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.auth_token}"

        retry_after: Optional[float] = None
        session = self._session_or_create()
        for attempt in range(_MAX_429_RETRIES + 1):
            # `async with`, not a bare await: an unread body holds its pooled
            # connection, and __aexit__ releases on every path including the
            # raises below.
            async with session.request(
                http_method,
                endpoint_route,
                headers=headers,
                data=data,
                params=params,
            ) as resp:
                if resp.status in (200, 201):
                    return await resp.json(content_type=None)
                if resp.status in (401, 403):
                    # Credential rejection, distinct so validate() can tell
                    # "bad credentials" from "request failed".
                    raise SpotifyAuthError(resp.status, f"endpoint: {endpoint_route}")
                if resp.status != 429:
                    raise SpotifyRequestError(resp.status, endpoint_route, params)
                retry_after = _retry_after_secs(resp)
            # Outside the block: the connection is back in the pool before the
            # wait rather than parked for the length of it.
            if attempt == _MAX_429_RETRIES:
                break
            delay = min(
                retry_after if retry_after is not None else 2.0**attempt,
                _MAX_RETRY_AFTER_SECS,
            )
            log.warning(
                f"spotify 429 on {endpoint_route}; retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{_MAX_429_RETRIES})"
            )
            await asyncio.sleep(delay)
        raise SpotifyRateLimitError(retry_after)

    async def validate(self, track_id: str) -> None:
        """Startup probe: force a fresh token (bypassing the Redis cache, so the
        credentials themselves are tested), then fetch a known track. Raises
        SpotifyAuthError only when Spotify rejects the credentials; everything
        else propagates as its own type and means "could not verify", not
        "invalid". Mutates no feature flag itself."""
        async with self._auth_lock:
            await self._refresh_token(use_cache=False, strict=True)
        endpoint = self.spotify_endpoint + f"v1/tracks/{track_id}"
        resp = await self.http_call(endpoint)
        if not resp.get("name"):
            raise ValueError(
                f"Spotify returned no track name for probe id {track_id!r}: {resp!r}"
            )

    # ── Cached API methods ────────────────────────────────────────────────────

    async def _cached_call(
        self,
        key: str,
        ttl: int,
        fetch_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Cache-aside: the cached value for `key`, or `fetch_fn`'s result
        cached under `ttl` seconds."""
        cached = await cache_get(self._redis, key)
        trace.get_current_span().set_attribute("spotify.cache_hit", cached is not None)
        if cached is not None:
            return cached
        result = await fetch_fn()
        await cache_set(self._redis, key, result, ttl)
        return result

    @_tracer.start_as_current_span("spotify.track")
    async def track(self, tid: str) -> str:
        """Return "<title> <artist1> <artist2> ..." for a track ID, cached for 24h."""
        trace.get_current_span().set_attribute("spotify.track_id", tid)

        async def fetch() -> str:
            endpoint = self.spotify_endpoint + f"v1/tracks/{tid}"
            resp = await self.http_call(endpoint)
            return _track_search_title(resp)

        return await self._cached_call(f"spotify:track:{tid}", _TRACK_TTL, fetch)

    @_tracer.start_as_current_span("spotify.playlist")
    async def playlist(self, pid: str) -> list[str]:
        """Return "<title> <artist1> <artist2> ..." for every track in a playlist, cached for 1h."""
        trace.get_current_span().set_attribute("spotify.playlist_id", pid)

        async def fetch() -> list[str]:
            # FIXME: Spotify playlists over 100 tracks are silently truncated.
            # Only the first page is read and the `next` cursor never followed,
            # so a 300-track playlist queues 100 and reports success. Fix: add
            # `next` to the fields mask and follow it until null.
            endpoint = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
            resp = await self.http_call(
                endpoint, params={"fields": "items(track(name,artists(name)))"}
            )
            track_titles = [
                _track_search_title(item["track"]) for item in resp.get("items", [])
            ]
            trace.get_current_span().set_attribute(
                "spotify.track_count", len(track_titles)
            )
            return track_titles

        return await self._cached_call(f"spotify:playlist:{pid}", _PLAYLIST_TTL, fetch)

    @_tracer.start_as_current_span("spotify.artists")
    async def artists(self, ids: Union[list[str], str]) -> Any:
        """Raw Spotify artist objects for one or more artist IDs, cached 24h.
        Untyped: nothing in `src/` reads a field off it (no production callers),
        so a TypedDict would be a guess with no consumer to check it."""
        if isinstance(ids, str):
            ids = [ids]
        trace.get_current_span().set_attribute("spotify.artist_ids", ",".join(ids))
        trace.get_current_span().set_attribute("spotify.artist_count", len(ids))

        async def fetch() -> Any:
            resp = await self.http_call(
                self.spotify_endpoint + "v1/artists", params={"ids": ",".join(ids)}
            )
            return resp.get("artists", resp)

        return await self._cached_call(
            f"spotify:artist:{','.join(sorted(ids))}", _ARTIST_TTL, fetch
        )

    @_tracer.start_as_current_span("spotify.albums")
    async def albums(self, ids: Union[list[str], str]) -> Any:
        """Raw Spotify album objects for one or more album IDs, cached for 24h.
        Untyped for the same reason as `artists()`."""
        if isinstance(ids, str):
            ids = [ids]
        trace.get_current_span().set_attribute("spotify.album_ids", ",".join(ids))
        trace.get_current_span().set_attribute("spotify.album_count", len(ids))

        async def fetch() -> Any:
            resp = await self.http_call(
                self.spotify_endpoint + "v1/albums", params={"ids": ",".join(ids)}
            )
            log.debug(resp)
            return resp.get("albums", resp)

        return await self._cached_call(
            f"spotify:album:{','.join(sorted(ids))}", _ALBUM_TTL, fetch
        )
