"""
GuildHistory — one guild's played-song history.

Two legs holding the same window, both bounded at HISTORY_CACHE_LIMIT and both
owned privately so they only move together: the Redis guild:{id}:history list
(capped and PERSISTed by push_history) and an in-memory deque. While the archive
is enabled (HISTORY_ARCHIVE_ENABLED) add() also XADDs the entry onto the global
outbox stream in the same pipeline and nudges the drainer; Postgres is never
awaited on the write path, and with the archive off on_outbox_push is None.

recent() never asks Postgres: the list is written synchronously at song end so it
LEADS the archive, and is capped at exactly the window -history can ask for.
See docs/ARCHITECTURE.md#history-read-path.

history_embeds() at the bottom renders what recent() returns. Storage and
rendering share a module the way src/leaderboard.py does it — one feature's data
and its embed, with the command itself on the MusicBot cog. It was in util.py,
which made util import guild_state purely to name HistoryEntry; util is imported
by the yt-dlp worker graph and is better off a leaf.

The wire format belongs to guild_state.py; this class never sees wire bytes.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Optional

import discord

from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import fmt_duration, get_logger, truncate_embed_title

log = get_logger(__name__)

# Ceiling on the Redis read behind -history. Bounded even though GuildRedisStore
# swallows its own errors: swallowing turns a FAILURE into [], but a
# connected-and-unresponsive server produces no error to swallow and the pool sets
# no socket read timeout. Short, because an in-memory leg is one hop away.
_READ_TIMEOUT_SECS = 2.0


class GuildHistory:
    """Played songs, oldest-first, capped at HISTORY_CACHE_LIMIT on both legs.

    Iteration/len/indexing read the cache directly; mutation goes through add()
    and restore() only, so the Redis mirror can't be skipped.
    """

    __slots__ = ("_store", "_entries", "_on_outbox_push")

    def __init__(
        self,
        store: Optional[GuildRedisStore],
        *,
        on_outbox_push: Optional[Callable[[], None]],
    ) -> None:
        # on_outbox_push is the drainer's notify — sync, so add() stays Redis-only
        # and never awaits the archive. Carries NO default: None means the archive
        # tier is disabled, and every constructor site must say so explicitly
        # rather than fall into silently not archiving. store is Optional on an
        # independent axis — unconfigured Redis leaves -history on the deque alone.
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)
        self._on_outbox_push = on_outbox_push

    async def add(self, entry: HistoryEntry) -> None:
        """Record one played song on every configured leg — the in-memory cache,
        the Redis display list, and (while the archive is enabled) the Postgres
        outbox, the latter two in one pipeline. Degrades gracefully when the store
        is None or the push fails (GuildRedisStore logs, never raises); no notify
        means no drainer to nudge, i.e. the archive tier is disabled."""
        self._entries.append(entry)
        if self._store is not None:
            await self._store.push_history(entry)
            if self._on_outbox_push is not None:
                self._on_outbox_push()

    def restore(self, newest_first: Sequence[HistoryEntry]) -> None:
        """Populate from persisted history after a restart. In-memory leg
        only — the entries came off the Redis list, which stores newest-first;
        the cache appends oldest-first, hence the reversal."""
        self._entries.extend(reversed(newest_first))

    async def recent(self, limit: int) -> list[HistoryEntry]:
        """The `limit` most recently played songs, newest first — -history's read
        surface. The Redis list (bounded at _READ_TIMEOUT_SECS), then the in-memory
        deque. Postgres is absent by design: the command's ceiling
        (HISTORY_MAX_LIMIT) is pinned to HISTORY_CACHE_LIMIT and push_history LTRIMs
        the list to exactly that, so the Redis leg alone carries a slot for every
        play this command can ask for. See docs/ARCHITECTURE.md#history-read-path.

        The deque is MERGED, not a fallback — a second leg can only ADD depth.
        Reaching it only when the leg above came back empty let a single Redis row
        suppress the whole cache. Dedup is exact rather than quantized: both legs
        carry the same `time.time()` float (orjson round-trips a double without
        loss), so equality is identity.
        """
        # Early-out, not a correctness guard: merged[:limit] already yields [] at
        # 0 and drops the tail at a negative. It skips a round trip for a page
        # nobody asked for; the command layer validates against HISTORY_MIN_LIMIT.
        if limit <= 0:
            return []
        merged: list[HistoryEntry] = []
        seen: set[HistoryEntry] = set()

        def take(entries: Sequence[HistoryEntry]) -> None:
            """Append what this leg adds. First leg to carry an entry wins."""
            for entry in entries:
                # the whole entry is the identity, not (played_at, url): both legs
                # carry the same object's values, so a full-value comparison is
                # exact. played_at defaults to 0.0 on entries predating the
                # timestamped wire format, so the narrower key collapsed two
                # genuinely distinct plays of one song into one. Plays identical in
                # every field still collapse — irreducible.
                if entry not in seen:
                    seen.add(entry)
                    merged.append(entry)

        # Bound to a local: the narrowing from the `is not None` test does not
        # follow self._store into the closure below.
        store = self._store
        if store is not None:
            # A lambda, not the bound `store.get_history`: passing the bound method
            # does the ATTRIBUTE LOOKUP here, outside _read_tier's guard, so a
            # store without the method raises straight out of recent() as an error
            # embed instead of degrading to the cache.
            take(await self._read_tier("redis", lambda: store.get_history()))
        take(list(reversed(self._entries)))
        # Both legs are ordered by when a song was RECORDED, which is when it ended,
        # but played_at is when it started — an interjection-parked song is recorded
        # after everything that cut in front of it, so without this sort -history
        # renders end order under a "newest first" promise. A 0.0 (unknown)
        # timestamp sorting last is correct — the only way to hold one is to predate
        # played_at. list.sort is stable, so ties keep leg order.
        merged.sort(key=lambda e: e.played_at, reverse=True)
        return merged[:limit]

    async def _read_tier(
        self, name: str, read: Callable[[], Awaitable[list[HistoryEntry]]]
    ) -> list[HistoryEntry]:
        """One bounded, non-raising read of a persisted leg.

        Takes a THUNK, not a coroutine: building the awaitable at the call site
        evaluates `store.get_history()` outside this guard, so a missing method or
        a wrong argument count escapes recent() and reaches the user as an error
        embed instead of degrading. Names the leg that FAILED, not the one it fell
        to — the fall-through may be the cache.
        """
        try:
            async with asyncio.timeout(_READ_TIMEOUT_SECS):
                return await read()
        except Exception as e:
            log.warning(
                f"-history {name} read failed, falling through to the tiers "
                f"below it: {type(e).__name__}: {e}"
            )
            return []

    @property
    def latest(self) -> Optional[HistoryEntry]:
        """The most recently played song, or None when the cache is cold.

        Cache-only so it can stay sync — reaching the list would force this async.
        restore() has already refilled the cache after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]


def history_embeds(entries: list[HistoryEntry]) -> list[discord.Embed]:
    """One embed per played song, in the given (newest-first) order. Layout: numbered
    title, raw webpage_url on its own line (Discord auto-links it), then played/duration,
    requester, and — when known — the played-at timestamp as viewer-local <t:…:f>."""
    embeds = []
    for i, entry in enumerate(entries, start=1):
        lines = []
        if entry.webpage_url:
            lines.append(entry.webpage_url)
        requested_by = (
            f"<@{entry.requester_id}>"
            if entry.requester_id
            else (entry.requester_name or "unknown")
        )
        meta = (
            f"{fmt_duration(entry.played_secs)} / {fmt_duration(entry.duration_secs)}"
            f" · requested by {requested_by}"
        )
        # played_at == 0 means unknown (absent on the wire); <t:0:f> would render
        # "1 January 1970", so omit the timestamp instead.
        if entry.played_at:
            meta += f" · <t:{int(entry.played_at)}:f>"
        lines.append(meta)
        title = truncate_embed_title(f"{i}. {entry.title}")
        embed = discord.Embed(
            title=title,
            description="\n".join(lines),
            color=discord.Color.blue(),
        )
        if entry.thumbnail:
            embed.set_thumbnail(url=entry.thumbnail)
        embeds.append(embed)
    return embeds
