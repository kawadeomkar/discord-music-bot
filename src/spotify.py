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

# The paging limit is Spotify's, not ours: 101 on a playlist request is HTTP
# 400. Albums send no explicit limit anywhere — page 1 arrives inside
# GET /v1/albums/{id} and later pages follow its embedded `next` cursor
# verbatim, so the album stride is always Spotify's own choice, never assumed.
_PLAYLIST_PAGE_LIMIT = 100
# `type` is in the mask so the unwrap can reject podcast episodes by name.
# `next` MUST stay in it: the cursor is what pages a playlist past its first
# 100 tracks, and a mask without it makes the rest of the playlist invisible.
_PLAYLIST_FIELDS = "next,total,items(track(type,name,artists(name)))"
# aiohttp's default is ClientTimeout(total=300) — one hung page would hold the
# per-guild collection lock for five minutes. Bounds one request, not a stream;
# the drain's own cap is musicbot._COLLECTION_DRAIN_TIMEOUT_SECS.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Pagination made one request per command into ceil(total/100), and the
# collection lock is per guild, so only this semaphore bounds the app-wide rate
# against one client-credentials app. Module scope is safe for the PROCESS —
# main() runs one asyncio.run for its lifetime — but the primitive is NOT
# loop-agnostic: 3.10 removed the `loop` parameter, and _LoopBoundMixin still
# pins it to the first loop that CONTENDS it (measured: a second loop raises
# "bound to a different event loop"). The test suite gives every test a fresh
# loop, so conftest rebinds this handle per test.
_MAX_CONCURRENT_REQUESTS = 10
_MAX_429_RETRIES = 3
# Capped because Spotify's Retry-After can be minutes. A fully rate-limited
# request costs (_MAX_429_RETRIES + 1) attempts of up to _HTTP_TIMEOUT each plus
# _MAX_429_RETRIES sleeps of up to this — ~150s at these values, so the ladder
# outlives a drain budget. What cuts it short is the drain deadline, and every
# leg that can reach the ladder runs under one.
_MAX_RETRY_AFTER_SECS = 10.0
_request_slots = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


def _cursor_page_cap(total: int, page_size: int) -> int:
    """Upper bound on pages a well-formed cursor walk can need.

    The streams otherwise trust Spotify to terminate with `next: null`; a
    cursor that stops advancing would spin duplicate pages into the consumer's
    queue until its drain budget expires, growing the queue and its Redis
    mirror the whole way. ceil(total/page_size) plus slack for a `total` that
    under-reports. Hitting the cap abandons the drain — what arrived stays
    yielded, nothing is cached."""
    return -(-total // max(1, page_size)) + 2


def _track_search_title(track: dict[str, Any]) -> str:
    """ "<name> <artist1> <artist2> ..." — the yt-dlp search string a Spotify track
    resolves to. Shared by track() and the collection streams so a single track, an
    album item and a playlist entry render identically. Album items are the track
    itself (no wrapper).

    Not total — it subscripts `name` and `artists`, and both collection streams
    guard those before calling it: a playlist can hold podcast episodes, which
    carry no artists, and an album page can be malformed. The guards belong at the
    call sites; defaulting here turns a nameless item into a `ytsearch:` for ""."""
    return track["name"] + "".join(f" {a['name']}" for a in track["artists"])


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotifyCollection:
    """Identity of a paged collection — what the enqueue embed renders.

    name/artists/thumbnail/release_date stay Optional/empty on the PLAYLIST
    path: playlist_stream opens at /v1/playlists/{id}/tracks, which carries no
    playlist identity under any `fields` mask, so playlist embeds render without
    a title or cover art while album embeds get both. See
    docs/ARCHITECTURE.md#playlist-identity-is-unfilled.
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
    wrong type is garbage, not an older build, and the whole entry is a miss:
    a corrupt `total` would otherwise flow uncoerced into embed copy
    (guild_state.py's wire discipline)."""
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

    @property
    def user_message(self) -> str:
        """The same split SpotifyRequestError uses: `detail` is the request
        endpoint, which on the collection paths is the `next` cursor, so it must
        not reach an embed. A rejected credential is a bot-side misconfiguration,
        so the copy points at the operator."""
        return (
            "Spotify isn't accepting this bot's credentials right now — "
            "the server owner needs to check them. Try a YouTube or SoundCloud "
            "link, or just search by name."
        )


class SpotifyCollectionAbandoned(Exception):
    """A collection stream stopped at its page cap with the cursor still live.

    Raised from inside the generator rather than returning, because the consumer
    cannot otherwise tell an abandoned drain from an exhausted one: both end in
    StopAsyncIteration, so a bare return had the bot announce "finished
    queueing" for a collection whose tail it never fetched. Pages already
    yielded stay queued — this reports what was NOT reached.
    """

    def __init__(self, kind: str, cid: str, pages_seen: int, enqueued: int) -> None:
        self.kind = kind
        self.pages_seen = pages_seen
        self.enqueued = enqueued
        super().__init__(
            f"{kind} {cid}: cursor still live after {pages_seen} pages; "
            f"abandoned with {enqueued} titles yielded"
        )


class SpotifyRequestError(Exception):
    """A non-2xx Spotify response that is neither a credential rejection nor a
    429. The args carry the endpoint and params for the log and the span; only
    user_message is safe to show, the same split ExtractionError uses — the raw
    text names the request URL and its offset, which is log detail, not something
    to put in a channel."""

    def __init__(self, status: int, endpoint: str, params: Any = None) -> None:
        self.status = status
        super().__init__(f"endpoint: {endpoint} stat: {status} params: {params}")

    @property
    def user_message(self) -> str:
        if self.status == 404:
            return "Spotify has no such track, album or playlist — check the link."
        if self.status >= 500:
            return "Spotify is having problems right now. Please try again shortly."
        return "Spotify could not be reached right now. Please try again."


class SpotifyRateLimitError(Exception):
    """Spotify returned 429 and the bounded retries did not clear it.

    Distinct from a generic request failure because the caller's advice differs:
    a rate-limited drain must not tell the user to re-run the command — the
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
                        raise SpotifyRequestError(resp.status, endpoint_route, params)
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
    # Async generators, not span-decorated: the span would wrap construction,
    # not consumption. Both cache only on a full drain, and nothing awaits on
    # the GeneratorExit path. See docs/ARCHITECTURE.md#spotify-collection-paging.

    async def album_stream(self, aid: str) -> AsyncGenerator[TrackPage]:
        """Yield an album's tracks as TrackPages of YouTube search titles.

        Page 1 rides GET /v1/albums/{id}: one call returns the album's
        identity (name, artists, cover art, total) AND its first tracks page,
        so a one-page album costs a single HTTP round-trip. Later pages follow
        the embedded paging object's `next` cursor, exactly like
        playlist_stream: sequential paging costs a few hundred ms per extra
        page, all of it after playback has started (the tail drains once the
        gate is open), and a cursor cannot skip or duplicate a run of tracks
        the way offset arithmetic against a lying `total`/`limit` could.
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
        page = resp.get("tracks") or {}
        total: int = page.get("total", 0)
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
        # Page size for the cap is page 1's own item count — the `next` cursor
        # carries page 1's limit forward, so every later page has the same
        # stride. 50 is only the fallback for a malformed empty-but-has-next
        # page 1, where any finite cap serves.
        page_cap = _cursor_page_cap(total, len(page.get("items") or []) or 50)
        all_titles: list[str] = []
        pages_seen = 0
        while True:
            # Album items ARE the track (SimplifiedTrackObject) — no ["track"]
            # wrapper and no episodes, so no UNWRAP guard is needed. The
            # name/artists guard still is: _track_search_title subscripts both,
            # so a null or nameless item on a malformed page raises TypeError
            # from inside this generator — and TypeError is not in
            # _command_error's allowlist, so it reaches the user as
            # "**TypeError:** 'NoneType' object is not subscriptable". Same two
            # conditions playlist_stream skips on, for the same reason.
            titles = [
                _track_search_title(t)
                for t in page.get("items") or []
                if t and t.get("name") and t.get("artists")
            ]
            next_url = page.get("next")
            all_titles.extend(titles)
            pages_seen += 1
            yield TrackPage(
                collection=collection, titles=titles, is_last=next_url is None
            )
            if next_url is None:
                break
            if pages_seen >= page_cap:
                log.warning(
                    f"album {aid}: cursor still live after {pages_seen} pages "
                    f"(total={total}); abandoning drain"
                )
                raise SpotifyCollectionAbandoned(
                    "album", aid, pages_seen, len(all_titles)
                )
            page = await self._collection_page(next_url)
        # Reached only on a full drain, and guarded twice: an empty result is
        # never cached (immutability makes a real empty album safe, but not a
        # malformed response), and neither is a drain the cursor ended short
        # of the album's own total — albums never skip items, so short means
        # wrong, and an under-count cached here would serve the truncation for
        # 24h with no error anywhere. A miss self-heals; a cache write sticks.
        if all_titles and len(all_titles) == total:
            await cache_set(
                self._redis,
                cache_key,
                _collection_to_cache(collection, all_titles),
                _ALBUM_TTL,
            )

    async def playlist_stream(self, pid: str) -> AsyncGenerator[TrackPage]:
        """Yield a playlist's tracks as TrackPages of YouTube search titles.

        Pages sequentially via the `next` cursor — required because playlists
        are mutable: an edit landing between two offset requests shifts every
        later offset and silently duplicates or drops tracks. `next`
        carries the fields mask forward, so following it needs no
        re-parameterisation. It is also the only thing that reaches a playlist's
        tracks past the first page — see _PLAYLIST_FIELDS.

        Skipped items are real (removed/local tracks arrive as null,
        episodes without artists), so collection.total is an upper bound —
        consumers report the enqueued count, never total.
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
        page_cap = _cursor_page_cap(collection.total, _PLAYLIST_PAGE_LIMIT)
        all_titles: list[str] = []
        pages_seen = 0
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
            pages_seen += 1
            yield TrackPage(
                collection=collection, titles=titles, is_last=next_url is None
            )
            if next_url is None:
                break
            if pages_seen >= page_cap:
                log.warning(
                    f"playlist {pid}: cursor still live after {pages_seen} "
                    f"pages (total={collection.total}); abandoning drain"
                )
                raise SpotifyCollectionAbandoned(
                    "playlist", pid, pages_seen, len(all_titles)
                )
            resp = await self._collection_page(next_url)
        if not all_titles:
            # Never cache "empty": _PLAYLIST_TTL is 1h because playlists are
            # user-editable, and the edit that follows this result is the user
            # adding the songs it just reported missing. Costs one request.
            return
        await cache_set(
            self._redis,
            cache_key,
            _collection_to_cache(collection, all_titles),
            _PLAYLIST_TTL,
        )

    @_tracer.start_as_current_span("spotify.collection_page")
    async def _collection_page(self, next_url: str) -> Any:
        """One cursor-following page, album or playlist — span-decorated (a
        coroutine, unlike the generators above; see the section comment) so
        per-page drain time stays attributable in the trace instead of being
        one of N indistinguishable aiohttp client spans under the command.
        The URL is Spotify's own `next` value, followed verbatim."""
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
