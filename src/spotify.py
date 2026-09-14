import asyncio
import contextlib
import os
import time
from typing import Any, Optional, Union, cast
from collections.abc import Awaitable, Callable, Iterator

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
from src import config
from src.telemetry import get_tracer
from src.util import ProgressFn, get_logger

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

# Spotify caps a playlist at 10,000 items. `limit=100` is what the API accepts
# although its reference documents 50, so the cap below is DERIVED from the page
# size: halve the limit and the guard doubles with it.
_PLAYLIST_MAX_ITEMS = 10_000
_PLAYLIST_PAGE_SIZE = 100
# A loop guard on a malformed or repeating `next`, not a limit a real playlist
# reaches. +1 so the largest legal playlist ends by exhausting `next`, which is
# the only ending that is not reported as short.
_MAX_PLAYLIST_PAGES = _PLAYLIST_MAX_ITEMS // _PLAYLIST_PAGE_SIZE + 1
# Bound on the WHOLE walk, sized against the work: _MAX_PLAYLIST_PAGES pages at
# ~1.2s each, where a healthy call is ~0.2s. It must cover the page cap, or the
# largest playlists are unqueueable however often they are retried.
_PLAYLIST_WALK_TIMEOUT_SECS = 120.0
# Bound on ONE page, which is what can hang: _HTTP_TIMEOUT does not cover a page
# spending three 429 retries at up to _MAX_RETRY_AFTER_SECS each. A retry that
# would sleep past this bound raises SpotifyRateLimitError instead, so a throttled
# page is reported as throttled. See docs/ARCHITECTURE.md#spotify-playlist-paging.
_PLAYLIST_PAGE_TIMEOUT_SECS = 20.0

# Process-wide, because Spotify's rate limiter is per application. A full walk is
# 100 requests where a track lookup was one, and PLAY_RESOLVE_CONCURRENCY does
# not reach them: it guards yt-dlp workers, which a walk holds none of.
_PLAYLIST_WALK_CONCURRENCY = 2
_playlist_gate: Optional[asyncio.Semaphore] = None
_playlist_gate_loop: Optional[asyncio.AbstractEventLoop] = None

# Refreshed this long before Spotify's own expiry: a 429 retry can sleep 30s
# between the expiry check and the request that carries the token.
_TOKEN_EXPIRY_MARGIN_SECS = 60

# Every caller awaiting a walk, leader and joiners alike, keyed like the cache: a
# page's report reaches each of their cards.
_PLAYLIST_SUBSCRIBERS: dict[str, list[ProgressFn]] = {}

# One walk per playlist at a time, process-wide. N users pasting one link is N
# identical 100-request walks racing to write one cache entry; the first starts
# it and the rest await its outcome. Keyed like the cache, so the joiners are
# exactly the callers the cache would have served had it been warm.
_INFLIGHT_PLAYLISTS: dict[str, asyncio.Future[list[str]]] = {}


def _publish(key: str, done: int, total: Optional[int]) -> None:
    """Hand one page's report to every caller awaiting that walk."""
    for report in list(_PLAYLIST_SUBSCRIBERS.get(key, ())):
        try:
            report(done, total)
        except Exception as e:
            log.warning(f"spotify playlist progress subscriber failed: {e!r}")


def _track_search_title(track: dict[str, Any]) -> str:
    """ "<name> <artist1> <artist2> ...", the yt-dlp search string a Spotify
    track resolves to. Shared by track() and playlist()."""
    # artists defaulted: a playlist can hold a podcast episode, which carries a
    # name and no artists at all.
    return track["name"] + "".join(f" {a['name']}" for a in track.get("artists") or [])


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


class SpotifyPlaylistTooSlowError(Exception):
    """The playlist walk ran out of budget. Its own type because nothing was
    queued, which is not what any other Spotify failure means.

    `whole_walk` separates the two causes, because only one of them is worth
    retrying: a stalled page is Spotify being briefly unreachable, while a walk
    that used its whole budget will do so again on every attempt. Advising a
    retry for the second is advice that cannot work."""

    def __init__(self, pages: int, titles: int, *, whole_walk: bool) -> None:
        self.pages = pages
        self.titles = titles
        self.whole_walk = whole_walk
        limit = (
            _PLAYLIST_WALK_TIMEOUT_SECS if whole_walk else _PLAYLIST_PAGE_TIMEOUT_SECS
        )
        super().__init__(
            f"spotify playlist {'walk' if whole_walk else 'page'} exceeded "
            f"{limit}s after {pages} pages ({titles} titles)"
        )

    @property
    def user_message(self) -> str:
        if self.whole_walk:
            return (
                f"That Spotify playlist is too large to read in one go — "
                f"{self.titles} tracks in and still going after "
                f"{_PLAYLIST_WALK_TIMEOUT_SECS:.0f}s, so nothing was queued. "
                "Retrying will hit the same limit; queue it in smaller parts."
            )
        return (
            "Spotify stopped responding while reading that playlist, so nothing "
            "was queued — try again."
        )


class SpotifyBusyError(Exception):
    """No playlist walk slot came free within the resolve wait, `wait_secs`: the
    bound that ran. Nothing was sent to Spotify, so unlike a rate limit a retry
    costs nothing."""

    def __init__(self, wait_secs: float) -> None:
        super().__init__(f"no spotify playlist walk slot within {wait_secs}s")

    @property
    def user_message(self) -> str:
        return (
            "Spotify is busy reading other playlists, so nothing was queued — "
            "try again in a minute."
        )


class SpotifyPlaylistForbiddenError(Exception):
    """Spotify refused this app a playlist's tracks while the credentials work:
    a 403 on the tracks endpoint, or a page with no `items` at all. Distinct from
    SpotifyAuthError, which would tell an operator to rotate good credentials."""

    def __init__(self, pid: str, detail: str) -> None:
        self.pid = pid
        super().__init__(f"spotify playlist {pid}: {detail}")

    @property
    def user_message(self) -> str:
        return "Spotify won't share that playlist's tracks with this bot."


def _playlist_slot() -> asyncio.Semaphore:
    """The process-wide bound on concurrent playlist walks. Rebuilt when the
    running loop changes — a Semaphore binds to the first loop that awaits it."""
    global _playlist_gate, _playlist_gate_loop
    loop = asyncio.get_running_loop()
    if _playlist_gate is None or _playlist_gate_loop is not loop:
        _playlist_gate = asyncio.Semaphore(_PLAYLIST_WALK_CONCURRENCY)
        _playlist_gate_loop = loop
    return _playlist_gate


@contextlib.contextmanager
def _subscribed(key: str, report: Optional[ProgressFn]) -> Iterator[None]:
    """Receive a walk's page reports for as long as this caller awaits it."""
    if report is None:
        yield
        return
    subscribers = _PLAYLIST_SUBSCRIBERS.setdefault(key, [])
    subscribers.append(report)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            subscribers.remove(report)
        if not subscribers:
            _PLAYLIST_SUBSCRIBERS.pop(key, None)


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
            if cached is not None and cached[1] > _TOKEN_EXPIRY_MARGIN_SECS:
                token, ttl = cached
                self.auth_token = token
                self.token_expiry = time.time() + ttl - _TOKEN_EXPIRY_MARGIN_SECS
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
        self.token_expiry += expires_in - _TOKEN_EXPIRY_MARGIN_SECS
        await spotify_token_set(self._redis, self.auth_token, expires_in)

    async def http_call(
        self,
        endpoint_route: str,
        params: Optional[dict[str, Union[str, int]]] = None,
        headers: Optional[dict[str, str]] = None,
        data: Optional[dict[str, str]] = None,
        http_method: str = "GET",
        deadline: Optional[float] = None,
    ) -> Any:
        """Authenticated request, refreshing the token first if expired. Raises
        on any non-2xx. `Any` because the response shape is chosen by the
        caller's URL; callers that read named fields narrow at their own
        boundary. `deadline` is the caller's bound as an event-loop time: a 429
        retry that would sleep past it raises SpotifyRateLimitError at once."""
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
            if (
                deadline is not None
                and asyncio.get_running_loop().time() + delay >= deadline
            ):
                break
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
    async def playlist(
        self, pid: str, *, on_progress: Optional[ProgressFn] = None
    ) -> list[str]:
        """Return "<title> <artist1> <artist2> ..." for every track in a playlist,
        cached for 1h.

        `on_progress` reports (items walked, playlist total) after each page. Its
        numerator counts playlist ITEMS, not the titles kept: the two differ by
        the items carrying no name to search YouTube for — a removed or
        region-dropped track, whose `track` is null, and an episode that supplies
        none. A local file DOES carry a name and is kept, because its name is
        exactly what a YouTube search wants. What the report measures is how far
        the walk has got. The confirmation's
        "queued N songs" counts what was queued, which is the list returned here.
        A cache hit reports nothing — it never reaches this fetch, and it resolves
        far under the threshold that would show a card anyway. A caller that joins
        a walk already running receives its reports from the next page on.
        """
        trace.get_current_span().set_attribute("spotify.playlist_id", pid)
        # v2: following `next` changes the cached VALUE under a 1h TTL, so an
        # already-cached playlist would keep answering 100 for an hour after deploy.
        cache_key = f"spotify:playlist:v2:{pid}"

        async def fetch() -> tuple[list[str], bool]:
            # Under `fields` Spotify answers with only the keys named, so `next`
            # and `total` have to be asked for: without the first the walk cannot
            # terminate, without the second progress has no denominator.
            url = self.spotify_endpoint + f"v1/playlists/{pid}/tracks"
            params: Optional[dict[str, Union[str, int]]] = {
                "fields": "items(track(name,artists(name))),next,total",
                "limit": _PLAYLIST_PAGE_SIZE,
            }
            titles: list[str] = []
            total: Optional[int] = None
            walked = 0
            pages = 0
            # Named, so expired() can tell the whole walk running out from one
            # page hanging and from an aiohttp timeout inside http_call: every
            # aiohttp timeout subclasses builtin TimeoutError, so the except
            # clause below cannot separate them on its own.
            walk = asyncio.timeout(_PLAYLIST_WALK_TIMEOUT_SECS)
            try:
                async with walk:
                    while pages < _MAX_PLAYLIST_PAGES:
                        async with asyncio.timeout(_PLAYLIST_PAGE_TIMEOUT_SECS) as page:
                            bounds = (page.when(), walk.when())
                            try:
                                resp = await self.http_call(
                                    url,
                                    params=params,
                                    deadline=min(t for t in bounds if t is not None),
                                )
                            except SpotifyAuthError as e:
                                if e.status != 403:
                                    raise
                                raise self._forbidden(pid, "HTTP 403") from e
                        pages += 1
                        if "items" not in resp:
                            raise self._forbidden(pid, "a page with no items")
                        items = resp["items"] or []
                        walked += len(items)
                        for item in items:
                            track = item.get("track")
                            if not isinstance(track, dict) or not track.get("name"):
                                # A removed or region-dropped track (null), or an
                                # episode with no name: nothing to search YouTube
                                # for. Walked, not queued. A local file has a name
                                # and is kept — the name is what the search wants.
                                continue
                            titles.append(_track_search_title(track))
                        if total is None:
                            total = resp.get("total")
                        _publish(cache_key, walked, total)
                        next_url = resp.get("next")
                        if not next_url:
                            break
                        if not str(next_url).startswith(self.spotify_endpoint):
                            # The cursor is a server-supplied URL and this walk
                            # sends the bearer token to it. Spotify is trusted,
                            # but a token is not something to hand to whatever a
                            # response names — and the check costs nothing.
                            log.error(
                                f"spotify playlist {pid}: refusing off-origin "
                                f"cursor after {pages} pages"
                            )
                            break
                        # The cursor is a full URL carrying its own query; passing
                        # params beside it would fight the offset it encodes.
                        url, params = next_url, None
                    else:
                        log.error(
                            f"spotify playlist {pid} still had pages after "
                            f"{_MAX_PLAYLIST_PAGES}; stopping at {walked} items"
                        )
            except TimeoutError as e:
                raise SpotifyPlaylistTooSlowError(
                    pages, len(titles), whole_walk=walk.expired()
                ) from e
            span = trace.get_current_span()
            span.set_attribute("spotify.track_count", len(titles))
            span.set_attribute("spotify.playlist_pages", pages)
            complete = total is None or walked >= total
            if total is not None:
                span.set_attribute("spotify.playlist_total", total)
                # The walk ends by exhausting `next`. Every other ending — the
                # page cap, a repeating cursor, a refused origin — ends it early,
                # and `total` is the only thing that can tell. A short walk says
                # so; answering as if it were whole is silent truncation.
                if walked < total:
                    span.set_attribute("spotify.playlist_short", True)
                    log.error(
                        f"spotify playlist {pid} walked {walked} of {total} items "
                        f"in {pages} pages; {len(titles)} queued"
                    )
            return titles, complete

        async def bounded() -> list[str]:
            # The slot is taken INSIDE the single flight, around the requests
            # alone: a joiner issues none and must not queue for a slot it will
            # not use. Same placement, and the same reason, as _extract_once's.
            slot = _playlist_slot()
            wait_secs = config.play_resolve_wait_secs()
            try:
                async with asyncio.timeout(wait_secs):
                    await slot.acquire()
            except TimeoutError as e:
                raise SpotifyBusyError(wait_secs) from e
            try:
                titles, complete = await fetch()
            finally:
                slot.release()
            # Written by the job, so a walk whose callers were all cancelled is
            # still kept. A short walk is not: every hit would answer it silently.
            if complete:
                await cache_set(self._redis, cache_key, titles, _PLAYLIST_TTL)
            return titles

        cached = await cache_get(self._redis, cache_key)
        trace.get_current_span().set_attribute("spotify.cache_hit", cached is not None)
        if cached is not None:
            return cast(list[str], cached)
        job = _INFLIGHT_PLAYLISTS.get(cache_key)
        if job is not None:
            trace.get_current_span().set_attribute("spotify.walk_shared", True)
        else:
            job = asyncio.ensure_future(bounded())
            _INFLIGHT_PLAYLISTS[cache_key] = job
            # Registered before the shield, so the key is gone before any awaiter
            # resumes and a retry after a failure starts a fresh walk.
            job.add_done_callback(lambda _f: _INFLIGHT_PLAYLISTS.pop(cache_key, None))
        # Shielded: the walk is shared, so one caller's cancellation must not take
        # it from the others.
        with _subscribed(cache_key, on_progress):
            return await asyncio.shield(job)

    def _forbidden(self, pid: str, detail: str) -> SpotifyPlaylistForbiddenError:
        """The playlist refusal, logged once for the operator: the credentials work,
        so it is the app's access, not something to rotate. See README's Spotify
        requirements."""
        log.error(
            f"spotify refused playlist {pid} ({detail}); a Development Mode app "
            "can be refused another user's playlist tracks — see README "
            "#requirements"
        )
        return SpotifyPlaylistForbiddenError(pid, detail)

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
