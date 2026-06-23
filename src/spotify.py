import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import ujson

import redis.asyncio as aioredis
from opentelemetry.trace import StatusCode

from src.redis_client import cache_get, cache_set, spotify_token_get, spotify_token_set
from src.telemetry import get_tracer
from src.util import get_logger

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_TRACK_TTL = 86400  # 24h — track titles/artists don't change
_PLAYLIST_TTL = 3600  # 1h  — playlists can be edited by users
_ARTIST_TTL = 86400  # 24h
_ALBUM_TTL = 86400  # 24h


class Spotify:
    spotify_endpoint = "https://api.spotify.com/"
    auth_endpoint = "https://accounts.spotify.com/api/token"

    def __init__(
        self,
        redis: Optional[aioredis.Redis] = None,
        session_factory: Optional[Callable[..., Any]] = None,
    ):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.token_expiry = 0.0
        self.auth_token: str = ""
        self._auth_lock = asyncio.Lock()
        self._redis = redis
        self._session_factory = session_factory or aiohttp.ClientSession

    def __str__(self):
        return self.auth_token

    # ── Auth ─────────────────────────────────────────────────────────────────

    async def _refresh_token(self) -> None:
        if self._redis is not None:
            cached_token = await spotify_token_get(self._redis)
            if cached_token:
                self.auth_token = cached_token
                # Set token_expiry well into the future. The Redis TTL is the real
                # expiry guard; this local value prevents http_call()'s
                # time.time() > token_expiry check from re-entering _refresh_token()
                # during the lifetime of this process.
                self.token_expiry = time.time() + 3540  # 59 min
                return

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
        expires_in: int = resp_data["expires_in"]
        self.token_expiry += expires_in
        await spotify_token_set(self._redis, self.auth_token, expires_in)

    async def http_call(
        self,
        endpoint_route: str,
        params: Optional[Dict[str, Union[str, int]]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, str]] = None,
        http_method: str = "GET",
    ) -> Any:
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

    async def track(self, tid: str) -> str:
        with _tracer.start_as_current_span(
            "spotify.track", attributes={"spotify.track_id": tid}
        ) as span:
            key = f"spotify:track:{tid}"
            cached = await cache_get(self._redis, key)
            span.set_attribute("spotify.cache_hit", cached is not None)
            if cached is not None:
                return cached
            try:
                endpoint = self.spotify_endpoint + f"v1/tracks/{tid}"
                resp = await self.http_call(endpoint)
                result = resp["name"] + "".join(
                    f" {a['name']}" for a in resp["artists"]
                )
                await cache_set(self._redis, key, result, _TRACK_TTL)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                raise

    async def playlist(self, pid: str) -> List[str]:
        with _tracer.start_as_current_span(
            "spotify.playlist", attributes={"spotify.playlist_id": pid}
        ) as span:
            key = f"spotify:playlist:{pid}"
            cached = await cache_get(self._redis, key)
            span.set_attribute("spotify.cache_hit", cached is not None)
            if cached is not None:
                return cached
            try:
                endpoint = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
                data: Dict[str, Union[str, int]] = {
                    "fields": "items(track(name,artists(name)))"
                }
                resp = await self.http_call(endpoint, params=data)
                track_titles = [
                    item["track"]["name"]
                    + "".join(f" {a['name']}" for a in item["track"]["artists"])
                    for item in resp.get("items", [])
                ]
                span.set_attribute("spotify.track_count", len(track_titles))
                await cache_set(self._redis, key, track_titles, _PLAYLIST_TTL)
                return track_titles
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                raise

    async def artists(self, ids: Union[List[str], str]) -> Any:
        if isinstance(ids, str):
            ids = [ids]
        with _tracer.start_as_current_span(
            "spotify.artists",
            attributes={
                "spotify.artist_ids": ",".join(ids),
                "spotify.artist_count": len(ids),
            },
        ) as span:
            key = f"spotify:artist:{','.join(sorted(ids))}"
            cached = await cache_get(self._redis, key)
            span.set_attribute("spotify.cache_hit", cached is not None)
            if cached is not None:
                return cached
            try:
                resp = await self.http_call(
                    self.spotify_endpoint + "v1/artists", params={"ids": ",".join(ids)}
                )
                result = resp.get("artists", resp)
                await cache_set(self._redis, key, result, _ARTIST_TTL)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                raise

    async def albums(self, ids: Union[List[str], str]) -> Any:
        if isinstance(ids, str):
            ids = [ids]
        with _tracer.start_as_current_span(
            "spotify.albums",
            attributes={
                "spotify.album_ids": ",".join(ids),
                "spotify.album_count": len(ids),
            },
        ) as span:
            key = f"spotify:album:{','.join(sorted(ids))}"
            cached = await cache_get(self._redis, key)
            span.set_attribute("spotify.cache_hit", cached is not None)
            if cached is not None:
                return cached
            try:
                resp = await self.http_call(
                    self.spotify_endpoint + "v1/albums", params={"ids": ",".join(ids)}
                )
                result = resp.get("albums", resp)
                log.debug(resp)
                await cache_set(self._redis, key, result, _ALBUM_TTL)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, f"{type(e).__name__}: {e}")
                raise
