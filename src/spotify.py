import asyncio
import os
import time
from typing import Any, Optional, Union
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


def _track_search_title(track: dict[str, Any]) -> str:
    """ "<name> <artist1> <artist2> ..." — the yt-dlp search string a Spotify track
    resolves to. Shared by track() and playlist() so both render it identically."""
    return track["name"] + "".join(f" {a['name']}" for a in track["artists"])


class SpotifyAuthError(Exception):
    """Spotify rejected the configured client credentials. Raised only for the two
    responses that actually indicate that: a non-2xx token grant, and a 401/403 from an
    API call. NOT for network errors, timeouts or other codes (404, 5xx) — those say
    nothing about validity, and startup validation disables the source only on this.
    """

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        super().__init__(
            f"Spotify rejected the credentials (HTTP {status})"
            + (f": {detail}" if detail else "")
        )


class Spotify:
    """Thin async client for the Spotify Web API: handles client-credentials
    auth (with auto-refresh) and Redis-backed caching of track/playlist/artist/
    album lookups."""

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

    def __str__(self) -> str:
        return self.auth_token

    # ── Auth ─────────────────────────────────────────────────────────────────

    async def _refresh_token(
        self, use_cache: bool = True, strict: bool = False
    ) -> None:
        """Fetch a fresh access token via client-credentials and update expiry.
        `use_cache=False` bypasses the Redis-cached token so the configured
        client_id/secret are genuinely exercised; `strict=True` raises SpotifyAuthError
        on a non-2xx grant instead of a missing-`access_token` KeyError. Both are opted
        into by validate(); the runtime path keeps the defaults."""
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
        async with self._session_factory(json_serialize=ujson.dumps) as session:
            resp = await session.post(self.auth_endpoint, data=data)
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
        """Make an authenticated request to the Spotify API, refreshing the token first
        if it has expired. Raises on any non-2xx response. `Any` is deliberate, unlike
        yt-dlp's `YTDLVideoInfo`: the response shape is chosen by the caller's URL, so
        the two callers that read named fields narrow at their own boundary instead.
        """
        if time.time() > self.token_expiry:
            async with self._auth_lock:
                if time.time() > self.token_expiry:
                    await self._refresh_token()

        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.auth_token}"

        async with self._session_factory(json_serialize=ujson.dumps) as session:
            resp = await session.request(
                http_method, endpoint_route, headers=headers, data=data, params=params
            )
            if resp.status in (200, 201):
                return await resp.json(content_type=None)
            if resp.status in (401, 403):
                # Credential/token rejection — distinct from other non-2xx codes
                # so validate() can tell "bad credentials" from "request failed".
                raise SpotifyAuthError(resp.status, f"endpoint: {endpoint_route}")
            raise Exception(
                f"endpoint: {endpoint_route} stat: {resp.status} params: {params}"
            )

    async def validate(self, track_id: str) -> None:
        """Exercise the configured credentials against the live Spotify API: force a
        fresh token (bypassing the Redis cache, so client_id/secret themselves are
        tested), then fetch a known track. Raises SpotifyAuthError *only* when Spotify
        rejects the credentials; everything else — network error, timeout, non-auth
        code, unexpected shape — propagates as its own type and means "could not
        verify", not "invalid". A startup probe; it mutates no feature flag itself."""
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
        """Cache-aside helper: return the cached value for `key`, or call
        `fetch_fn` and cache its result under `ttl` seconds on a miss."""
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
            # This reads only the first page and never follows the `next` cursor,
            # so a 300-track playlist queues 100 and reports success.
            # Fix: add `next` to the fields mask (excluded today) and follow the
            # cursor until it returns null.
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
        """Return raw Spotify artist objects for one or more artist IDs, cached 24h.
        Untyped by intent: nothing in `src/` reads a field off it (this and `albums()`
        have no production callers), so a TypedDict would be a guess at Spotify's schema
        with no consumer to check it — see `http_call`.
        """
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
        """Return raw Spotify album objects for one or more album IDs, cached for 24h.
        Untyped for the same reason as `artists()` above."""
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
