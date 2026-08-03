import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from collections.abc import AsyncGenerator, Awaitable, Callable

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
from src.sources import SpotifyType
from src.telemetry import get_tracer
from src.util import get_logger

log = get_logger(__name__)
_tracer = get_tracer(__name__)

_TRACK_TTL = 86400  # 24h — track titles/artists don't change
_PLAYLIST_TTL = 3600  # 1h  — playlists can be edited by users
_ARTIST_TTL = 86400  # 24h
_ALBUM_TTL = 86400  # 24h — albums are immutable once released

# Collection paging (docs/SPOTIFY_ALBUM_SUPPORT_PLAN.md §2). The limits are
# Spotify's, not ours: 51 on an album-tracks request and 101 on a playlist
# request are both HTTP 400. _ALBUM_PAGE_LIMIT applies only to the explicit
# /albums/{id}/tracks?limit= requests WE issue — the first page arrives inside
# GET /v1/albums/{id}, whose stride is Spotify's choice and must be read off
# the response (album_stream), never assumed.
_ALBUM_PAGE_LIMIT = 50
_PLAYLIST_PAGE_LIMIT = 100
_ALBUM_PAGE_CONCURRENCY = 5  # measured safe: no Retry-After on a 20-page burst (§2.5)
# `type` is in the mask so the unwrap can reject podcast episodes by name;
# `next`/`total` are what the shipped mask omitted — the omission was the
# entire >100-track truncation bug (§2.4).
_PLAYLIST_FIELDS = "next,total,items(track(type,name,artists(name)))"
# aiohttp's default is ClientTimeout(total=300) — one hung page would hold the
# per-guild collection lock (musicbot.py) for five minutes. 30s is generous
# for a single JSON page. Bounds ONE request, not a stream: the drain's own
# wall-clock cap lives in musicbot._COLLECTION_DRAIN_TIMEOUT_SECS.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

# 429 handling. Pagination turned one request per command into ceil(total/100),
# and the collection lock is per guild, so nothing bounds the app-wide rate
# against a single client-credentials app. The semaphore caps concurrent calls
# process-wide (the per-guild lock cannot); the retries absorb the transient
# 429 that a burst earns. Loop-agnostic since 3.10, so module scope is safe.
_MAX_CONCURRENT_REQUESTS = 10
_MAX_429_RETRIES = 3
# Spotify's Retry-After on a sustained burst can be minutes. Anything past this
# is not worth holding the enqueue lock for — surface it and let the user retry.
# _MAX_429_RETRIES * this (30s) stays under musicbot._COLLECTION_DRAIN_TIMEOUT_SECS
# (45s), so ONE rate-limited page can be absorbed inside a drain's budget while
# sustained limiting trips the drain bound instead. Both report honestly; keep
# the inequality if either moves.
_MAX_RETRY_AFTER_SECS = 10.0
_request_slots = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


def _track_search_title(track: dict[str, Any]) -> str:
    """ "<name> <artist1> <artist2> ..." — the yt-dlp search string a Spotify track
    resolves to. Shared by track() and the collection streams so a single track, an
    album item and a playlist entry render identically. Album items are the track
    itself (no wrapper); playlist_stream guards missing name/artists before calling
    this — a playlist can legally hold podcast episodes, which carry no artists."""
    return track["name"] + "".join(f" {a['name']}" for a in track["artists"])


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotifyCollection:
    """Identity of a paged collection — what the enqueue embed renders.

    name/artists/thumbnail/release_date are Optional/empty because the
    PLAYLIST path cannot fill them: /v1/playlists/{id}/tracks returns the
    paging object, and no `fields` mask yields the playlist's own name or
    images (filling them would cost a second HTTP call the streaming design
    does not budget for). Albums get all of it free from the single
    GET /v1/albums/{id} call.
    """

    kind: SpotifyType
    id: str
    total: int
    name: Optional[str] = None
    artists: list[str] = field(default_factory=list)
    thumbnail: Optional[str] = None
    release_date: Optional[str] = None

    @property
    def artist_line(self) -> str:
        return ", ".join(self.artists) or "Unknown artist"


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackPage:
    """One page of a collection, already reduced to YouTube search titles.

    `collection` repeats on every page rather than being Optional on the
    first: it is frozen and shared, so the repetition is free and the
    consumer never special-cases page 1 to learn what it is queueing.
    """

    collection: SpotifyCollection
    titles: list[str]
    is_last: bool


def _collection_to_cache(
    collection: SpotifyCollection, titles: list[str]
) -> dict[str, Any]:
    """Explicit cache wire shape. cache_get returns a plain orjson dict —
    never a dataclass — so the field names here ARE the wire format: a Python
    attribute rename must never silently rename a cached field (the same rule
    guild_state.py follows). `kind` is not written: the cache key already
    scopes it, and the reader supplies it."""
    return {
        "id": collection.id,
        "total": collection.total,
        "name": collection.name,
        "artists": collection.artists,
        "thumbnail": collection.thumbnail,
        "release_date": collection.release_date,
        "titles": titles,
    }


def _collection_from_cache(
    kind: SpotifyType, cid: str, raw: Any
) -> Optional[tuple[SpotifyCollection, list[str]]]:
    """Parse a cached collection. None ⇒ treat as a cache miss (unparseable
    entries are re-fetched, not crashed on). Every field reads with a default
    so entries written by an older build stay readable — but a field of the
    WRONG TYPE is garbage, not an older build, and the whole entry is a miss:
    a corrupt `total` would otherwise flow uncoerced into embed copy
    (guild_state.py's wire discipline; review M10)."""
    if not isinstance(raw, dict):
        return None
    titles = raw.get("titles")
    if not isinstance(titles, list):
        return None
    total = raw.get("total", len(titles))
    if not isinstance(total, int):
        return None
    name = raw.get("name")
    thumbnail = raw.get("thumbnail")
    release_date = raw.get("release_date")
    if any(
        v is not None and not isinstance(v, str)
        for v in (name, thumbnail, release_date)
    ):
        return None
    collection = SpotifyCollection(
        kind=kind,
        id=cid,
        total=total,
        name=name,
        artists=[str(a) for a in raw.get("artists") or []],
        thumbnail=thumbnail,
        release_date=release_date,
    )
    return collection, [str(t) for t in titles]


class SpotifyAuthError(Exception):
    """Spotify rejected the configured client credentials. Raised only for the two
    responses that actually indicate that: a non-2xx token grant, and a 401/403 from an
    API call. Not for network errors, timeouts or other codes (404, 5xx) — those say
    nothing about validity, and startup validation disables the source only on this.
    """

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        super().__init__(
            f"Spotify rejected the credentials (HTTP {status})"
            + (f": {detail}" if detail else "")
        )


class SpotifyRateLimitError(Exception):
    """Spotify returned 429 and the bounded retries did not clear it.

    Distinct from a generic request failure because the caller's advice differs:
    a rate-limited drain must NOT tell the user to re-run the command — the
    re-run refetches every page from 1 and doubles the load that earned the 429.
    """

    def __init__(self, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__(
            "Spotify is rate-limiting this bot"
            + (f"; retry after {retry_after:.0f}s" if retry_after else "")
        )

    @property
    def user_message(self) -> str:
        """The only text from this error safe to show a user — the raw args
        carry the endpoint and params."""
        if self.retry_after:
            return (
                f"Spotify is rate-limiting the bot right now. "
                f"Try again in about {self.retry_after:.0f} seconds."
            )
        return "Spotify is rate-limiting the bot right now. Try again shortly."


def _retry_after_secs(resp: aiohttp.ClientResponse) -> Optional[float]:
    """Retry-After in seconds, or None when absent/unparseable. Spotify sends
    the delta-seconds form; the HTTP-date form is not parsed (we would rather
    fall back to the default backoff than mis-parse a date into a long sleep).
    """
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except TypeError, ValueError:
        return None


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
        async with self._session_factory(
            json_serialize=ujson.dumps, timeout=_HTTP_TIMEOUT
        ) as session:
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
        the callers that read named fields (track(), the album/playlist pagers) narrow
        at their own boundary instead.
        """
        if time.time() > self.token_expiry:
            async with self._auth_lock:
                if time.time() > self.token_expiry:
                    await self._refresh_token()

        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.auth_token}"

        retry_after: Optional[float] = None
        for attempt in range(_MAX_429_RETRIES + 1):
            # The semaphore is the only app-wide bound: the collection lock is
            # per guild, so N guilds draining concurrently would otherwise
            # multiply the request rate against one client-credentials app.
            async with _request_slots:
                async with self._session_factory(
                    json_serialize=ujson.dumps, timeout=_HTTP_TIMEOUT
                ) as session:
                    resp = await session.request(
                        http_method,
                        endpoint_route,
                        headers=headers,
                        data=data,
                        params=params,
                    )
                    if resp.status in (200, 201):
                        return await resp.json(content_type=None)
                    if resp.status in (401, 403):
                        # Credential/token rejection — distinct from other non-2xx
                        # codes so validate() can tell "bad credentials" from
                        # "request failed".
                        raise SpotifyAuthError(
                            resp.status, f"endpoint: {endpoint_route}"
                        )
                    if resp.status != 429:
                        raise Exception(
                            f"endpoint: {endpoint_route} stat: {resp.status} "
                            f"params: {params}"
                        )
                    retry_after = _retry_after_secs(resp)
            # Slot released before sleeping: holding it would idle a slot every
            # other caller could be using.
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

    # ── Streaming collection pagers ───────────────────────────────────────────
    # Async generators, deliberately NOT span-decorated: start_as_current_span
    # would open and close the span around generator *construction*, not its
    # consumption — attributes go on the caller's current span instead.
    #
    # Cache discipline (shared by both): the cache is written ONLY when the
    # consumer iterates past the final page — i.e. the generator resumes after
    # its last yield. A partially-consumed generator (-playnow's page-1-only
    # path, a preemption, a mid-stream error, aclose()) takes GeneratorExit at
    # a yield and never reaches the write, so a truncated collection can never
    # poison the entry. Nothing awaits on the GeneratorExit path — an await
    # there is illegal during finalization, and the resulting unraisable
    # warning is a hard test failure under filterwarnings=["error"].

    async def album_stream(self, aid: str) -> AsyncGenerator[TrackPage]:
        """Yield an album's tracks as TrackPages of YouTube search titles.

        Page 1 rides GET /v1/albums/{id}: one call returns the album's
        identity (name, artists, cover art, total) AND its first tracks page,
        so a ≤stride album costs a single HTTP round-trip. Later pages use a
        concurrent offset fanout — safe ONLY because albums are immutable
        once released; a mutable collection paged by offset can silently
        duplicate or drop tracks when an edit lands between requests, which
        is why playlist_stream pages sequentially. Do not copy the fanout
        onto the playlist path.
        """
        span = trace.get_current_span()
        span.set_attribute("spotify.album_id", aid)
        cache_key = f"spotify:album_tracks:{aid}"
        hit = _collection_from_cache(
            SpotifyType.ALBUM, aid, await cache_get(self._redis, cache_key)
        )
        span.set_attribute("spotify.cache_hit", hit is not None)
        if hit is not None:
            collection, titles = hit
            yield TrackPage(collection=collection, titles=titles, is_last=True)
            return

        resp = await self.http_call(self.spotify_endpoint + f"v1/albums/{aid}")
        tracks = resp.get("tracks") or {}
        total: int = tracks.get("total", 0)
        images = resp.get("images") or []
        collection = SpotifyCollection(
            kind=SpotifyType.ALBUM,
            id=aid,
            total=total,
            name=resp.get("name"),
            artists=[a["name"] for a in resp.get("artists") or [] if a.get("name")],
            thumbnail=images[0].get("url") if images else None,
            release_date=resp.get("release_date"),
        )
        # Album items ARE the track (SimplifiedTrackObject) — no ["track"]
        # wrapper, no episodes, so no unwrap guard (§2.1).
        page1 = [_track_search_title(t) for t in tracks.get("items") or []]
        all_titles = list(page1)
        if tracks.get("next") is None:
            yield TrackPage(collection=collection, titles=page1, is_last=True)
            # Reached only when the consumer drains past the last page — see
            # the cache-discipline comment above.
            await cache_set(
                self._redis,
                cache_key,
                _collection_to_cache(collection, all_titles),
                _ALBUM_TTL,
            )
            return

        # The first page's stride is Spotify's choice on a limit-less endpoint
        # (documented 20, measured 50) — starting the fanout at a hardcoded 50
        # against a 20-item page 1 would silently drop tracks 21–49 of every
        # album. Derive it; only the explicit ?limit= requests below use ours.
        stride: int = tracks.get("limit") or len(page1)
        yield TrackPage(collection=collection, titles=page1, is_last=False)

        offsets = list(range(stride, total, _ALBUM_PAGE_LIMIT))
        for wave_start in range(0, len(offsets), _ALBUM_PAGE_CONCURRENCY):
            wave = offsets[wave_start : wave_start + _ALBUM_PAGE_CONCURRENCY]
            try:
                async with asyncio.TaskGroup() as tg:
                    # TaskGroup, not gather: a failed page CANCELS its
                    # siblings instead of leaving them hitting Spotify with
                    # their results discarded. No yield happens inside the
                    # group — every wave completes before its pages go out.
                    tasks = [
                        tg.create_task(self._album_page(aid, offset)) for offset in wave
                    ]
            except BaseExceptionGroup as eg:
                # Callers speak single exceptions (SpotifyAuthError handling,
                # _command_error). Siblings are already cancelled; surface the
                # first real failure.
                raise eg.exceptions[0]
            for offset, task in zip(wave, tasks):
                titles = task.result()
                all_titles.extend(titles)
                yield TrackPage(
                    collection=collection,
                    titles=titles,
                    is_last=offset == offsets[-1],
                )
        await cache_set(
            self._redis,
            cache_key,
            _collection_to_cache(collection, all_titles),
            _ALBUM_TTL,
        )

    @_tracer.start_as_current_span("spotify.album_page")
    async def _album_page(self, aid: str, offset: int) -> list[str]:
        """One explicit album-tracks page — the fanout worker for album_stream.

        Span-decorated (a coroutine, unlike the generators above — see the
        section comment): each page of a fanout gets its own named span, so a
        slow or failed page is attributable in the trace instead of being one
        of N indistinguishable aiohttp client spans under the command."""
        span = trace.get_current_span()
        span.set_attribute("spotify.album_id", aid)
        span.set_attribute("spotify.page_offset", offset)
        resp = await self.http_call(
            self.spotify_endpoint + f"v1/albums/{aid}/tracks",
            params={"offset": offset, "limit": _ALBUM_PAGE_LIMIT},
        )
        return [_track_search_title(t) for t in resp.get("items") or []]

    async def playlist_stream(self, pid: str) -> AsyncGenerator[TrackPage]:
        """Yield a playlist's tracks as TrackPages of YouTube search titles.

        Pages sequentially via the `next` cursor — REQUIRED because playlists
        are mutable: an edit landing between two offset requests shifts every
        later offset and silently duplicates or drops tracks (§2.5). `next`
        carries the fields mask forward, so following it needs no
        re-parameterisation. Following it at all is the fix for the
        >100-track silent-truncation bug: the old mask omitted `next`
        entirely, so the cursor was invisible.

        Skipped items are real (removed/local tracks arrive as null,
        episodes without artists), so collection.total is an upper bound —
        consumers report the ENQUEUED count, never total.
        """
        span = trace.get_current_span()
        span.set_attribute("spotify.playlist_id", pid)
        cache_key = f"spotify:playlist_tracks:{pid}"
        hit = _collection_from_cache(
            SpotifyType.PLAYLIST, pid, await cache_get(self._redis, cache_key)
        )
        span.set_attribute("spotify.cache_hit", hit is not None)
        if hit is not None:
            collection, titles = hit
            yield TrackPage(collection=collection, titles=titles, is_last=True)
            return

        resp = await self.http_call(
            self.spotify_endpoint + f"v1/playlists/{pid}/tracks",
            params={
                "fields": _PLAYLIST_FIELDS,
                "additional_types": "track",
                "limit": _PLAYLIST_PAGE_LIMIT,
            },
        )
        collection = SpotifyCollection(
            kind=SpotifyType.PLAYLIST, id=pid, total=resp.get("total", 0)
        )
        all_titles: list[str] = []
        while True:
            titles: list[str] = []
            for item in resp.get("items") or []:
                track = item.get("track") if isinstance(item, dict) else None
                if not track:
                    continue  # removed or local track — arrives as null
                if track.get("type", "track") != "track":
                    continue  # podcast episode (type is in _PLAYLIST_FIELDS)
                if not track.get("name") or not track.get("artists"):
                    continue  # episodes under an older mask carry no artists
                titles.append(_track_search_title(track))
            next_url = resp.get("next")
            all_titles.extend(titles)
            yield TrackPage(
                collection=collection, titles=titles, is_last=next_url is None
            )
            if next_url is None:
                break
            resp = await self._playlist_page(next_url)
        await cache_set(
            self._redis,
            cache_key,
            _collection_to_cache(collection, all_titles),
            _PLAYLIST_TTL,
        )

    @_tracer.start_as_current_span("spotify.playlist_page")
    async def _playlist_page(self, next_url: str) -> Any:
        """One cursor-following playlist page — traced for the same reason as
        _album_page: per-page drain time stays attributable in the trace. The
        URL is Spotify's own `next` value, followed verbatim (§2.4)."""
        trace.get_current_span().set_attribute("spotify.page_url", next_url)
        return await self.http_call(next_url)

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
