import asyncio
import os
import time
from async_lru import alru_cache
from typing import Any, Dict, List, Optional, Union

import aiohttp
import ujson

from src.util import get_logger

log = get_logger(__name__)


class Spotify:
    spotify_endpoint = "https://api.spotify.com/"
    auth_endpoint = "https://accounts.spotify.com/api/token"

    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.token_expiry = 0.0
        self.auth_token: str = ""
        self._auth_lock = asyncio.Lock()

    def __str__(self):
        return self.auth_token

    async def _refresh_token(self) -> None:
        self.token_expiry = time.time()
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
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
        if time.time() > self.token_expiry:
            async with self._auth_lock:
                if time.time() > self.token_expiry:
                    await self._refresh_token()

        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.auth_token}"

        async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
            resp = await session.request(
                http_method, endpoint_route, headers=headers, data=data, params=params
            )
            # disable content types for incorrect mime type responses
            if resp.status == 200 or resp.status == 201:
                data = await resp.json(content_type=None)
                return data
            else:
                raise Exception(
                    "endpoint:  "
                    + endpoint_route
                    + " stat: "
                    + str(resp.status)
                    + " params: "
                    + str(params)
                )

    @alru_cache(maxsize=256, ttl=360)
    async def track(self, tid: str) -> str:
        """
        Gets a track information given URL
        :param tid: spotify track id
        :return: title of the song
        """
        endpoint_route = self.spotify_endpoint + f"v1/tracks/{tid}"
        resp = await self.http_call(endpoint_route)
        title = resp["name"]
        for artist in resp["artists"]:
            title += f" {artist['name']}"
        return title

    @alru_cache(maxsize=256, ttl=360)
    async def playlist(self, pid: str) -> List[str]:
        """
        Gets a playlist information given URL
        :param pid: spotify track id
        :return: list of
        """
        endpoint_route = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
        data: Dict[str, Union[str, int]] = {
            "fields": "items(track(name,artists(name)))"
        }
        resp = await self.http_call(endpoint_route, params=data)
        track_titles = []
        for item in resp.get("items", []):
            title = item["track"]["name"]
            for artist in item["track"]["artists"]:
                title += f" {artist['name']}"
            track_titles.append(title)
        return track_titles

    @alru_cache(maxsize=256, ttl=360)
    async def artists(self, ids: Union[List, str]):
        """
        Returns artist(s) information
        ids: List of artist IDs or single artist ID
        """
        endpoint_route = self.spotify_endpoint + "v1/artists"
        if isinstance(ids, str):
            ids = [ids]
        resp = await self.http_call(endpoint_route, params={"ids": ",".join(ids)})
        if "artists" in resp:
            return resp["artists"]
        return resp

    @alru_cache(maxsize=256, ttl=360)
    async def albums(self, ids: Union[List, str]):
        """
        Returns album(s) information
        ids: List of album IDs or single album ID
        """
        endpoint_route = self.spotify_endpoint + "v1/albums"
        if isinstance(ids, str):
            ids = [ids]
        resp = await self.http_call(endpoint_route, params={"ids": ",".join(ids)})
        if "albums" in resp:
            return resp["albums"]
        log.debug(resp)
        return resp
