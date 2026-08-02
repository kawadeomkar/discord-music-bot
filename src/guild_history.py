"""
GuildHistory — a guild's played-song history for one guild.

The domain twin of GuildQueue, one layer smaller — but with an inverted
ownership story. Postgres (play_history, via the outbox) is the durable record
of every song ever played; the Redis guild:{id}:history list is the write-side
staging copy — still unbounded and PERSISTed, so it remains a complete second
copy — while the in-memory leg is a bounded display cache of the newest
HISTORY_CACHE_LIMIT entries, sized to the most `-history --limit` can show.
Owning both privately means the legs can only move together: every add() lands
on the cache and the Redis list in one step, and restore() refills the cache
from the newest slice of the list.

The at-rest wire format is owned by guild_state.py (HistoryEntry +
serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.

add() also XADDs every entry onto the global Postgres outbox stream — in the same
pipeline as the display-list push — and nudges the drainer
(docs/POSTGRES_HISTORY_PLAN.md). The archive is a required tier, so that leg is
unconditional. Postgres is never awaited on the WRITE path: the outbox/drainer
split keeps the playback loop on Redis-only latency.

Reads do not go to Postgres, and never will: recent() serves -history from the
Redis list, falling back to the in-memory cache. That is why no archive handle
is passed in — writes fan out to three places, reads to two, and the asymmetry
is the design rather than a stage of it.
"""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from typing import Optional

from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore


class GuildHistory:
    """Played songs, oldest-first; cache capped at HISTORY_CACHE_LIMIT, the
    Redis leg unbounded.

    Iteration/len/indexing are exposed directly (the -history command and its
    tests read the cache as a plain sequence); mutation goes through add() and
    restore() only, so the Redis mirror can't be skipped.
    """

    __slots__ = ("_store", "_entries", "_on_outbox_push")

    def __init__(
        self,
        store: Optional[GuildRedisStore],
        *,
        on_outbox_push: Callable[[], None],
    ) -> None:
        # on_outbox_push is REQUIRED: the Postgres tier is not optional, so
        # there is no shape of this object that writes history without an outbox
        # consumer behind it. It is the drainer's notify — a sync callable so
        # add() stays Redis-only and never awaits the archive.
        #
        # No archive handle and no guild_id, deliberately. Both would be the READ
        # side, and this class does not have one that reaches Postgres: -history
        # is served from the Redis list and the cache below. The entry's own
        # guild_id is stamped a layer up by HistoryEntry.from_song, which is
        # where the outbox needs it.
        #
        # store IS Optional, and that is a different axis: Redis may be
        # unconfigured, in which case there is no wire to push to at all.
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)
        self._on_outbox_push = on_outbox_push

    async def add(self, entry: HistoryEntry) -> None:
        """Record one played song on all three legs — in-memory cache, the
        Redis display list, and the Postgres outbox (the latter two in one
        pipeline). Degrades gracefully when the store is None or the push
        fails (GuildRedisStore logs, never raises; a notify after a failed
        push just drains an empty outbox)."""
        self._entries.append(entry)
        if self._store is not None:
            await self._store.push_history(entry)
            self._on_outbox_push()

    def restore(self, newest_first: Sequence[HistoryEntry]) -> None:
        """Populate from persisted history after a restart. In-memory leg
        only — the entries came off the Redis list, which stores newest-first;
        the cache appends oldest-first, hence the reversal."""
        self._entries.extend(reversed(newest_first))

    async def recent(self, limit: int) -> list[HistoryEntry]:
        """The `limit` most recently played songs, newest first — the
        -history command's read surface.

        Reads the Redis list directly when a store is configured, so the
        command reflects persisted history even when the in-memory cache is
        cold. That happens after a clean -stop and restart: recovery is
        skipped for a stopped guild, so its next MusicPlayer starts with an
        empty cache while the (unbounded, PERSISTed) Redis list still holds
        every played song. get_history() already returns the newest
        HISTORY_CACHE_LIMIT entries newest-first — the display ceiling — so a
        slice to `limit` is authoritative. The in-memory cache is the fallback
        when there is no store or the read fails/returns empty (the cache can
        only hold entries that also reached the store, so falling back never
        invents history)."""
        if limit <= 0:
            return []
        if self._store is not None:
            persisted = await self._store.get_history()
            if persisted:
                return persisted[:limit]
        return list(self._entries)[-limit:][::-1]

    @property
    def latest(self) -> Optional[HistoryEntry]:
        """The most recently played song, or None when the cache is cold.

        Cache-only (unlike recent(), which prefers the Redis list) so callers
        on a latency-sensitive path get an answer without a round-trip;
        restore() has already refilled the cache from Redis by the time
        anything reads this after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
