import asyncio
import os
import time
from dataclasses import dataclass
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

# Spotify's own maximum for these endpoints; asking for more is rejected, and
# asking for less only multiplies round-trips.
_PAGE_SIZE = 100
# Our cap, not Spotify's. Public playlists run to 10k+ tracks, and every title
# here becomes a separate yt-dlp search plus a queue entry — so the ceiling
# exists to bound queue explosions and extraction load, not the API. Hitting it
# is reported to the user rather than silently swallowed (see PagedTracks).
_TRACK_CAP = 1000
# Defence against a malformed/looping `next` cursor: even at the cap above,
# _PAGE_SIZE-sized pages can never legitimately need more than this many
# requests. Without it a server-side bug becomes an unbounded request loop.
_MAX_PAGES = (_TRACK_CAP // _PAGE_SIZE) + 2


def _track_search_title(track: dict[str, Any]) -> str:
    """ "<name> <artist1> <artist2> ..." — the yt-dlp search string a Spotify
    track object resolves to. Shared by track() and playlist() so a single
    track and a playlist entry render their search titles identically."""
    return track["name"] + "".join(f" {a['name']}" for a in track["artists"])


@dataclass(frozen=True)
class PagedTracks:
    """Track search titles walked off a paged Spotify endpoint.

    `truncated` is True when _TRACK_CAP stopped the walk before the collection
    ran out — the caller is expected to tell the user, because the whole point
    of this type is that "we queued fewer tracks than you asked for" must never
    again be invisible.
    """

    titles: list[str]
    truncated: bool = False

    # Cached as a plain dict rather than as this dataclass: _cached_call stores
    # values through orjson, and while orjson *serializes* a dataclass fine, it
    # deserializes to a dict — so a cache hit and a cache miss would return
    # different types from the same method. Converting at both boundaries keeps
    # the round-trip symmetric.
    def to_cache(self) -> dict[str, Any]:
        return {"titles": self.titles, "truncated": self.truncated}

    @classmethod
    def from_cache(cls, raw: Any) -> "PagedTracks":
        # Tolerates the pre-migration cache shape (a bare list of titles) so a
        # deploy doesn't have to wait out the 1h playlist TTL to stop erroring.
        if isinstance(raw, list):
            return cls(titles=raw, truncated=False)
        return cls(
            titles=raw.get("titles", []), truncated=bool(raw.get("truncated", False))
        )


def _playlist_item_track(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Unwrap a /playlists/{id}/tracks item, which nests the track under `track`."""
    return item.get("track")


def _album_item_track(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Unwrap an /albums/{id}/tracks item, which IS the (simplified) track.

    The two endpoints differ here and the difference is easy to miss: album
    pages return SimplifiedTrackObjects directly in `items`, playlist pages
    wrap each in a PlaylistTrackObject.
    """
    return item


class SpotifyAuthError(Exception):
    """Spotify rejected the configured client credentials.

    Raised for the two responses that actually indicate bad credentials: a
    non-2xx from the token endpoint (the client-credentials grant failed) and a
    401/403 from an API call (the resulting token was refused). Deliberately
    NOT raised for network errors, timeouts, or other HTTP codes (404, 5xx) —
    those say nothing about credential validity. Startup validation keys off
    this distinction: only this error disables the Spotify source.
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
        """Fetch a fresh access token via the client-credentials flow and update expiry.

        `use_cache=False` skips the Redis-cached token and always hits the auth
        endpoint, so the configured client_id/secret are genuinely exercised
        rather than a token minted from earlier credentials — used by validate().

        `strict=True` raises SpotifyAuthError on a non-2xx grant response instead
        of letting the missing-`access_token` KeyError surface. Off by default so
        the runtime path (and its tests) are unchanged; validate() opts in so it
        can distinguish rejected credentials from other failures."""
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
        """Make an authenticated request to the Spotify API, refreshing the
        token first if it has expired. Raises on any non-2xx response.

        `Any` here is deliberate, and the asymmetry with yt-dlp's `YTDLVideoInfo`
        is the point. That schema exists because `src/` reads ~19 *named* fields
        off one known info-dict shape on the playback hot path, so a wrong
        assumption there breaks playback. This returns whatever the requested
        endpoint returns — the shape is chosen by the caller's URL, so there is no
        single schema to write. The two callers that do read named fields
        (`track()`, `playlist()`) narrow to `str` / `list[str]` at their own
        boundary, which is where the shape is actually known.
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
        """Exercise the configured credentials against the live Spotify API.

        Forces a fresh client-credentials token (bypassing any Redis-cached one,
        so the client_id/secret themselves are tested — not a token minted from
        earlier credentials), then fetches a known track.

        Raises SpotifyAuthError *only* when Spotify rejects the credentials
        themselves: a non-2xx token grant (`strict=True`) or a 401/403 on the
        track call. Every other failure propagates as its own type — a network
        error, a timeout, a non-auth HTTP code, or an unexpected response shape
        (ValueError) — and means "could not verify", not "invalid". The caller
        keys its enable/disable decision off that distinction.

        Intended for a one-shot startup probe; it mutates no feature flag itself."""
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

    async def _paged_track_titles(
        self,
        endpoint: str,
        fields: str,
        extract: Callable[[dict[str, Any]], Optional[dict[str, Any]]],
    ) -> PagedTracks:
        """Walk a paged Spotify tracks endpoint to exhaustion (or _TRACK_CAP).

        `fields` must include `next` — it is the cursor this loop follows, and
        omitting it from the mask is exactly how playlists came to be truncated
        at one page. `extract` adapts the two item shapes; see the two module
        level unwrappers.
        """
        titles: list[str] = []
        truncated = False
        # Only the FIRST request takes params: `next` is a fully-formed URL that
        # already carries limit/offset/fields, and re-passing them would fight it.
        params: Optional[dict[str, Union[str, int]]] = {
            "fields": fields,
            "limit": _PAGE_SIZE,
        }
        url: Optional[str] = endpoint

        for _ in range(_MAX_PAGES):
            if url is None:
                break
            resp = await self.http_call(url, params=params)
            items = resp.get("items") or []
            for i, item in enumerate(items):
                track = extract(item) if item else None
                # A playlist item can legitimately be `{"track": null}` (the
                # track was removed from Spotify or is unavailable in this
                # market), and a podcast episode arrives with a name but no
                # `artists`. Both used to reach _track_search_title and raise,
                # failing the whole playlist over one bad entry — skip them.
                if not track or not track.get("name") or track.get("artists") is None:
                    continue
                titles.append(_track_search_title(track))
                if len(titles) >= _TRACK_CAP:
                    # At the cap. Only truncated if something is genuinely left:
                    # more items after this one on THIS page, or another page.
                    return PagedTracks(
                        titles=titles,
                        truncated=i + 1 < len(items) or bool(resp.get("next")),
                    )
            url, params = resp.get("next"), None
        else:
            # Loop ran out of pages rather than out of cursor — treat as truncation
            # rather than silently reporting a complete playlist.
            truncated = url is not None
            log.warning(
                f"spotify: stopped paging {endpoint} after {_MAX_PAGES} pages "
                f"({len(titles)} titles); cursor still non-null"
            )

        return PagedTracks(titles=titles, truncated=truncated)

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
    async def playlist(self, pid: str) -> PagedTracks:
        """Return "<title> <artist1> <artist2> ..." for every track in a playlist, cached for 1h."""
        trace.get_current_span().set_attribute("spotify.playlist_id", pid)

        async def fetch() -> dict[str, Any]:
            endpoint = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
            paged = await self._paged_track_titles(
                endpoint,
                # `next` is what makes this paged rather than first-page-only.
                "next,items(track(name,artists(name)))",
                _playlist_item_track,
            )
            trace.get_current_span().set_attribute(
                "spotify.track_count", len(paged.titles)
            )
            trace.get_current_span().set_attribute("spotify.truncated", paged.truncated)
            return paged.to_cache()

        raw = await self._cached_call(f"spotify:playlist:{pid}", _PLAYLIST_TTL, fetch)
        return PagedTracks.from_cache(raw)

    @_tracer.start_as_current_span("spotify.album_tracks")
    async def album_tracks(self, aid: str) -> PagedTracks:
        """Return "<title> <artist1> <artist2> ..." for every track on an album, cached for 24h.

        Albums are immutable once released (unlike playlists, which users edit),
        hence the 24h TTL rather than 1h. Deliberately separate from `albums()`,
        which returns raw album *objects* for a different, caller-chosen shape.
        """
        trace.get_current_span().set_attribute("spotify.album_id", aid)

        async def fetch() -> dict[str, Any]:
            endpoint = self.spotify_endpoint + f"v1/albums/{aid}/tracks"
            paged = await self._paged_track_titles(
                endpoint,
                # No `track(...)` wrapper here — album items ARE the track.
                "next,items(name,artists(name))",
                _album_item_track,
            )
            trace.get_current_span().set_attribute(
                "spotify.track_count", len(paged.titles)
            )
            trace.get_current_span().set_attribute("spotify.truncated", paged.truncated)
            return paged.to_cache()

        raw = await self._cached_call(f"spotify:album_tracks:{aid}", _ALBUM_TTL, fetch)
        return PagedTracks.from_cache(raw)

    @_tracer.start_as_current_span("spotify.artists")
    async def artists(self, ids: Union[list[str], str]) -> Any:
        """Return raw Spotify artist objects for one or more artist IDs, cached for 24h.

        Untyped by intent: "raw" means the upstream object verbatim, and nothing
        in `src/` reads a field off it (this method and `albums()` have no
        production callers). A TypedDict here would be a guess at Spotify's
        schema with no consumer to check it against — see `http_call`.
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

        Untyped for the same reason as `artists()` above.
        """
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
