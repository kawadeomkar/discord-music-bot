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

    def recent(self, limit: int) -> list[HistoryEntry]:
        """The `limit` most recently played songs, newest first — the
        -history command's read surface."""
        if limit <= 0:
            return []
        return list(self._entries)[-limit:][::-1]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
