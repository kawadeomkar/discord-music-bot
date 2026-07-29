"""
GuildHistory — a guild's played-song history for one guild.

The domain twin of GuildQueue, one layer smaller — but with an inverted
ownership story. Postgres (play_history, via the outbox) is the durable record
of every song ever played; the Redis guild:{id}:history list is the write-side
staging copy and the read fallback, and the in-memory leg is a bounded display
cache of the newest HISTORY_CACHE_LIMIT entries, sized to the most
`-history --limit` can show. Owning both privately means the legs can only
move together: every add() lands on the cache and the Redis list in one step,
and restore() refills the cache from the newest slice of the list.

The Redis list is still unbounded and PERSISTed, so it remains a complete
second copy of everything the archive holds — which is what makes the merge
below safe to lean on while the archive is still filling up
(docs/POSTGRES_HISTORY_PLAN.md).

The at-rest wire format is owned by guild_state.py (HistoryEntry +
serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.

add() also LPUSHes every entry onto the global Postgres outbox — in the same
pipeline as the display-list push — and nudges the drainer
(docs/POSTGRES_HISTORY_PLAN.md). The archive is a required tier, so that leg is
unconditional. Postgres is never awaited on the WRITE path: the outbox/drainer
split keeps the playback loop on Redis-only latency.

Reads are the other direction. recent() asks Postgres first and MERGES anything
Redis still holds that the archive has not caught up on, falling back to the
cache when both are unavailable (Phase B). That merge is what lets the Redis
list be demoted later without -history losing depth, and what keeps the newest
plays — still sitting in the outbox — visible in the meantime.
"""

import asyncio
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Optional

from src.guild_state import HistoryEntry
from src.history_archive import quantized_played_at
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import get_logger

if TYPE_CHECKING:
    from src.history_archive import HistoryArchive

log = get_logger(__name__)

# Ceiling on the archive read behind -history. Short on purpose: this is a
# user-facing command with a Redis fallback one hop away, so waiting is worse
# than falling back. It sits well under the archive's own command_timeout,
# which is sized for the drainer's batch writes, not for interactive reads.
_ARCHIVE_READ_TIMEOUT_SECS = 2.0


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
        archive: "HistoryArchive",
        guild_id: int = 0,
        on_outbox_push: Callable[[], None],
    ) -> None:
        # archive and on_outbox_push are REQUIRED: the Postgres tier is not
        # optional, so there is no shape of this object that writes history
        # without an outbox consumer behind it. archive is recent()'s primary
        # read; guild_id scopes that read. on_outbox_push is the drainer's
        # notify — a sync callable so add() stays Redis-only.
        #
        # store IS still Optional, and that is a different axis: Redis may be
        # unconfigured, in which case there is no wire to push to at all.
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)
        self._archive = archive
        self._guild_id = guild_id
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

        Three sources, in durability order (Phase B of
        docs/POSTGRES_HISTORY_PLAN.md):

        1. Postgres. The archive holds every play, including ones aged out of
           any Redis window, so it is the only source that stays correct after
           the Phase C cutover demotes the Redis list to a bounded cache. A
           FULL result (>= limit rows) is returned as-is and Redis is not read.
        2. The Redis list, MERGED IN whenever the archive came back short.
           Covers the two cases the archive cannot: plays still sitting in the
           outbox (every song's newest few seconds) and every pre-existing play
           until `just db-backfill` has run. Merging rather than choosing is
           what stops -history going silently shallow during that window; the
           dedup key is (played_at, webpage_url), the same identity the
           archive's unique index uses.
        3. The in-memory cache, when neither of the above yielded anything.
           The cache can only hold entries that also reached the store, so
           falling back never invents history.

        The Postgres read is bounded at _ARCHIVE_READ_TIMEOUT_SECS and every
        failure degrades rather than propagating: -history is a display command,
        and a slow archive must cost depth, never an error embed.
        """
        if limit <= 0:
            return []
        archived: list[HistoryEntry] = []
        try:
            async with asyncio.timeout(_ARCHIVE_READ_TIMEOUT_SECS):
                archived = await self._archive.recent(self._guild_id, limit)
            if len(archived) >= limit:
                return archived
        except Exception as e:
            log.warning(f"-history read fell back to redis: {type(e).__name__}: {e}")
            archived = []
        # SHORT, not empty: merge rather than return what the archive had.
        # Returning a short archive result whenever it was non-empty made
        # -history silently shallow for the entire window between deploying this
        # branch and finishing the backfill — a guild with 50 pre-deploy plays
        # showed 1 as soon as a single song drained. That window is structural
        # (the backfill cannot run before the schema and outbox exist), so it
        # hits every guild on every deploy. It also covers the steady state,
        # where the newest plays are still in the outbox and only Redis has them.
        merged = list(archived)
        if self._store is not None:
            persisted = await self._store.get_history()
            if persisted:
                # BOTH sides go through quantized_played_at. The archive leg has
                # been through timestamptz and is microsecond-granular; the Redis
                # leg still carries the raw time.time() float, which is finer.
                # Comparing them directly failed for ~37% of real timestamps, so
                # a drained play appeared in both legs, survived the dedup, and
                # rendered twice — and since merged[:limit] runs below, each
                # duplicate pushed a genuine older song out of the window.
                # The helper is idempotent, so applying it to the already-
                # quantized archive leg costs nothing and keeps the invariant
                # ("identity is the QUANTIZED pair") visible on both sides.
                seen = {
                    (quantized_played_at(e.played_at), e.webpage_url) for e in merged
                }
                merged.extend(
                    e
                    for e in persisted
                    if (quantized_played_at(e.played_at), e.webpage_url) not in seen
                )
        if merged:
            # Both legs are newest-first individually, but Redis entries the
            # archive has not caught up on are NEWER than everything archived,
            # so the concatenation is not ordered. Sort explicitly.
            merged.sort(key=lambda e: e.played_at, reverse=True)
            return merged[:limit]
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
