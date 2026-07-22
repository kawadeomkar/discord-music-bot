"""
GuildHistory — a guild's played-song history for one guild.

The domain twin of GuildQueue, one layer smaller — but with an inverted
ownership story: the Redis guild:{id}:history list is the SOURCE OF TRUTH for
every song ever played (unbounded, PERSISTed — Postgres eventually, see
docs/HISTORY_OVERHAUL_PLAN.md §8), while the in-memory leg is a bounded
display cache of the newest HISTORY_CACHE_LIMIT entries, sized to the most
`-history --limit` can show. Owning both privately means the legs can only
move together: every add() lands on the cache and the Redis list in one step,
and restore() refills the cache from the newest slice of the list.

The at-rest wire format is owned by guild_state.py (HistoryEntry +
serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.
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
