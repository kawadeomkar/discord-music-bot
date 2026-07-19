"""
GuildHistory — a guild's played-song history for one guild.

The domain twin of GuildQueue, one layer smaller. Since the Phase C cutover
(docs/POSTGRES_HISTORY_PLAN.md) the source of truth for every song ever
played is the Postgres play_history table; both local legs are caches sized
to the most `-history --limit` can show: the in-memory deque (freshness patch
+ last-resort fallback) and the Redis guild:{id}:history list (TTL'd,
trimmed — its one remaining read is the restore-time refill of the deque).
Owning the legs privately means they can only move together: every add()
lands on the deque, the Redis list, and the Postgres outbox in one step, and
restore() re-seeds the deque from the newest slice of the list.

The at-rest wire format is owned by guild_state.py (HistoryEntry +
serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.

When a Postgres archive is configured, add() LPUSHes the entry onto the
global outbox — in the same pipeline as the display-list push — and nudges
the drainer, and recent() reads Postgres first. Postgres is never awaited in
add(): the outbox/drainer split keeps the playback loop on Redis-only
latency.
"""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Optional

from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import get_logger

if TYPE_CHECKING:
    from src.history_archive import HistoryArchive

log = get_logger(__name__)


class GuildHistory:
    """Played songs, oldest-first; both cache legs capped at
    HISTORY_CACHE_LIMIT, full history in Postgres via the archive.

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
        # grow unbounded) and is recent()'s primary read; guild_id is held
        # for that same read. on_outbox_push is the drainer's notify — a
        # sync callable so add() stays Redis-only.
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

        Postgres-primary (docs/POSTGRES_HISTORY_PLAN.md §5.4): the archive
        holds every played song, so its answer is authoritative — after a
        freshness merge with the in-memory deque, which covers the one gap PG
        can have: entries still in flight through the outbox (drain lag,
        typically just the song that ended a moment ago) and, degraded,
        everything played since PG went down. Identity for the merge is
        (played_at, webpage_url) — the dedup key.

        When the archive is unconfigured or the read raises, the pre-cutover
        chain survives as degraded mode: the Redis list if non-empty (a
        trimmed display cache since Phase C, so possibly partial — partial
        beats empty, and it is never *wrong*, just short), else the deque.
        Every fallback can only hold entries that were also pushed toward the
        durable legs, so no path invents history."""
        if limit <= 0:
            return []
        if self._archive is not None:
            try:
                persisted = await self._archive.recent(self._guild_id, limit)
            except Exception as e:
                log.warning(
                    f"[guild:{self._guild_id}] history archive read failed, "
                    f"serving degraded fallback: {type(e).__name__}: {e}"
                )
            else:
                return self._merge_fresh(persisted, limit)
        if self._store is not None:
            persisted = await self._store.get_history()
            if persisted:
                return persisted[:limit]
        return list(self._entries)[-limit:][::-1]

    def _merge_fresh(
        self, persisted: list[HistoryEntry], limit: int
    ) -> list[HistoryEntry]:
        """Prepend deque entries the archive doesn't have yet (newest first),
        then the archive's newest-first result, capped at limit. Fresh
        entries are by construction newer than anything already drained, so
        prepending preserves global newest-first order."""
        seen = {(e.played_at, e.webpage_url) for e in persisted}
        fresh = [e for e in self._entries if (e.played_at, e.webpage_url) not in seen]
        fresh.reverse()  # deque is oldest-first
        return (fresh + persisted)[:limit]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
