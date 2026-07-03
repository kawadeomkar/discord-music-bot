import asyncio
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import aiohttp
import ujson

import redis.asyncio as aioredis

from opentelemetry import trace

from src.redis_client import cache_get, cache_set
from src.telemetry import get_tracer
from src.util import get_logger

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_TRACK_TTL = 86400  # 24h — track titles/artists don't change
_PLAYLIST_TTL = 3600  # 1h  — playlists can be edited by users
_ARTIST_TTL = 86400  # 24h
_ALBUM_TTL = 86400  # 24h


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

    async def _refresh_token(self) -> None:
        """Fetch a fresh access token via the client-credentials flow and update expiry."""
        self.token_expiry = time.time()
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        async with self._session_factory(json_serialize=ujson.dumps) as session:
            resp = await session.post(self.auth_endpoint, data=data)
            resp_data = await resp.json(content_type=None)
        self.auth_token = resp_data["access_token"]
        self.token_expiry += resp_data["expires_in"]

    async def http_call(
        self,
        endpoint_route: str,
        params: Optional[Dict[str, Union[str, int]]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, str]] = None,
        http_method: str = "GET",
    ) -> Any:
        """Make an authenticated request to the Spotify API, refreshing the
        token first if it has expired. Raises on any non-2xx response."""
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
            raise Exception(
                f"endpoint: {endpoint_route} stat: {resp.status} params: {params}"
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
            return resp["name"] + "".join(f" {a['name']}" for a in resp["artists"])

        return await self._cached_call(f"spotify:track:{tid}", _TRACK_TTL, fetch)

    @_tracer.start_as_current_span("spotify.playlist")
    async def playlist(self, pid: str) -> List[str]:
        """Return "<title> <artist1> <artist2> ..." for every track in a playlist, cached for 1h."""
        trace.get_current_span().set_attribute("spotify.playlist_id", pid)

        async def fetch() -> List[str]:
            endpoint = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
            resp = await self.http_call(
                endpoint, params={"fields": "items(track(name,artists(name)))"}
            )
            track_titles = [
                item["track"]["name"]
                + "".join(f" {a['name']}" for a in item["track"]["artists"])
                for item in resp.get("items", [])
            ]
            trace.get_current_span().set_attribute(
                "spotify.track_count", len(track_titles)
            )
            return track_titles

        return await self._cached_call(f"spotify:playlist:{pid}", _PLAYLIST_TTL, fetch)

    @_tracer.start_as_current_span("spotify.artists")
    async def artists(self, ids: Union[List[str], str]) -> Any:
        """Return raw Spotify artist objects for one or more artist IDs, cached for 24h."""
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
    async def albums(self, ids: Union[List[str], str]) -> Any:
        """Return raw Spotify album objects for one or more album IDs, cached for 24h."""
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
