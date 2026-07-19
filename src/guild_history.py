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

When a Postgres archive is configured (docs/POSTGRES_HISTORY_PLAN.md), add()
also LPUSHes the entry onto the global outbox — in the same pipeline as the
display-list push — and nudges the drainer. Postgres itself is never awaited
here: the outbox/drainer split keeps the playback loop on Redis-only latency.
"""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Optional

from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore

if TYPE_CHECKING:
    from src.history_archive import HistoryArchive


class GuildHistory:
    """Played songs, oldest-first; cache capped at HISTORY_CACHE_LIMIT, the
    Redis leg unbounded.

    Iteration/len/indexing are exposed directly (the -history command and its
    tests read the cache as a plain sequence); mutation goes through add() and
    restore() only, so the Redis mirror can't be skipped.
    """

    __slots__ = ("_store", "_entries", "_archive", "_guild_id", "_on_outbox_push")

    def __init__(
        self,
        store: Optional[GuildRedisStore],
        *,
        archive: Optional["HistoryArchive"] = None,
        guild_id: int = 0,
        on_outbox_push: Optional[Callable[[], None]] = None,
    ) -> None:
        # archive gates the outbox push (without a drainer the outbox would
        # grow unbounded) and becomes recent()'s primary read in Phase B;
        # guild_id is held for that same read. on_outbox_push is the
        # drainer's notify — a sync callable so add() stays Redis-only.
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)
        self._archive = archive
        self._guild_id = guild_id
        self._on_outbox_push = on_outbox_push

    async def add(self, entry: HistoryEntry) -> None:
        """Record one played song on both legs — plus the Postgres outbox
        when an archive is configured. Degrades gracefully when the store is
        None or the push fails (GuildRedisStore logs, never raises; a notify
        after a failed push just drains an empty outbox)."""
        self._entries.append(entry)
        if self._store is not None:
            await self._store.push_history(entry, outbox=self._archive is not None)
            if self._archive is not None and self._on_outbox_push is not None:
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

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
