from functools import lru_cache
from typing import Any, Dict, List, Union

import aiohttp
import os
import requests
import time
import ujson
import util

log = util.setLogger(__name__)


class Spotify:
    spotify_endpoint = 'https://api.spotify.com/'
    auth_endpoint = 'https://accounts.spotify.com/api/token'

    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.token_expiry = time.time()
        self.auth_token = self.authorize(self.client_id, self.client_secret)

    def __str__(self):
        return self.auth_token

    async def http_call(self,
                        endpoint_route: str,
                        params: Dict[str, Union[str, int]] = None,
                        headers=None,
                        data: Dict[str, str] = None,
                        http_method='GET'):
        if time.time() > self.token_expiry:
            self.auth_token = self.authorize(self.client_id, self.client_secret)

        if headers is None and endpoint_route != self.auth_endpoint:
            headers = {'Authorization': f"Bearer {self.auth_token}"}

        async with aiohttp.ClientSession(json_serialize=ujson) as session:
            resp = await session.request(http_method,
                                         endpoint_route,
                                         headers=headers,
                                         data=data,
                                         params=params)
            # disable content types for incorrect mime type responses
            if resp.status == 200 or resp.status == 201:
                data = await resp.json(content_type=None)
                return data
            else:
                raise Exception("endpoint:  " + endpoint_route +
                                " stat: " + str(resp.status) +
                                " params: " + str(params))

    def authorize(self, client_id: str, client_secret: str):
        # set time before http call for overhead
        self.token_expiry = time.time()
        data = {'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret}

        resp = requests.post(self.auth_endpoint, data=data).json()
        self.token_expiry += resp['expires_in']
        return resp['access_token']

    @lru_cache(maxsize=None)
    async def track(self, tid: str) -> str:
        """
        Gets a track information given URL
        :param tid: spotify track id
        :return: title of the song
        """
        endpoint_route = self.spotify_endpoint + f"v1/tracks/{tid}"
        resp = await self.http_call(endpoint_route)
        title = resp['name']
        for artist in resp['artists']:
            title += f" {artist['name']}"
        return title

    @lru_cache(maxsize=None)
    async def playlist(self, pid: str) -> List[str]:
        """
        Gets a playlist information given URL
        :param pid: spotify track id
        :return: list of
        """
        endpoint_route = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
        data = {"fields": "items(track(name,artists(name)))"}
        resp = await self.http_call(endpoint_route, params=data)
        track_titles = []
        for item in resp.get("items", []):
            title = item["track"]["name"]
            for artist in item["track"]["artists"]:
                title += f" {artist['name']}"
            track_titles.append(title)
        return track_titles

    @lru_cache(maxsize=None)
    async def artists(self, ids: Union[List, str]):
        """
        Returns artist(s) information
        ids: List of artist IDs or single artist ID
        """
        endpoint_route = self.spotify_endpoint + "v1/artists"
        if isinstance(ids, str):
            ids = [ids]
        resp = await self.http_call(endpoint_route, params={'ids': ','.join(ids)})
        if 'artists' in resp:
            return resp['artists']
        return resp

    @lru_cache(maxsize=None)
    async def albums(self, ids: Union[List, str]):
        """
        Returns album(s) information
        ids: List of album IDs or single album ID
        """
        endpoint_route = self.spotify_endpoint + "v1/albums"
        if isinstance(ids, str):
            ids = [ids]
        resp = await self.http_call(endpoint_route, params={'ids': ','.join(ids)})
        if 'albums' in resp:
            return resp['albums']
        log.debug(resp)
        return resp
