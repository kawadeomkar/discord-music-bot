"""
GuildHistory — one guild's played-song history.

The domain twin of GuildQueue, with inverted ownership: the Redis
guild:{id}:history list is the SOURCE OF TRUTH (unbounded, PERSISTed — Postgres
eventually), and the in-memory leg is a display cache of the newest
HISTORY_CACHE_LIMIT entries, sized to the most
`-history --limit` can show. Owning both privately keeps them in step: add()
lands on cache and list together, restore() refills the cache from the list.

The wire format belongs to guild_state.py (HistoryEntry +
serialize/parse_history_entry); this class never sees wire bytes.
"""

from collections import deque
from collections.abc import Iterator, Sequence
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

    __slots__ = ("_store", "_entries")

    def __init__(self, store: Optional[GuildRedisStore]) -> None:
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)

    async def add(self, entry: HistoryEntry) -> None:
        """Record one played song on both legs. Degrades gracefully when the
        store is None or the push fails (GuildRedisStore logs, never raises)."""
        self._entries.append(entry)
        if self._store is not None:
            await self._store.push_history(entry)

    def restore(self, newest_first: Sequence[HistoryEntry]) -> None:
        """Populate from persisted history after a restart. In-memory leg
        only — the entries came off the Redis list, which stores newest-first;
        the cache appends oldest-first, hence the reversal."""
        self._entries.extend(reversed(newest_first))

    async def recent(self, limit: int) -> list[HistoryEntry]:
        """The `limit` most recently played songs, newest first — -history's read
        surface.

        Prefers the Redis list, which already returns the newest
        HISTORY_CACHE_LIMIT newest-first, so slicing to `limit` is authoritative.
        Falls back to the cache with no store or on a failed/empty read — the cache
        only holds entries that also reached the store, so the fallback never
        invents history."""
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

        Cache-only (unlike recent()) so latency-sensitive callers avoid a
        round-trip; restore() has already refilled the cache after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
