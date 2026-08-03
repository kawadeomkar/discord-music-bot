"""
GuildHistory — one guild's played-song history.

The domain twin of GuildQueue, one layer smaller. Two legs, both bounded at
HISTORY_CACHE_LIMIT and both holding the same window: the Redis
guild:{id}:history list (capped and PERSISTed by push_history) and an in-memory
deque. Owning both privately means they only move together — add() lands on both
in one step, restore() refills the deque from the list.

Writes fan out, reads do not. While the archive is enabled
(HISTORY_ARCHIVE_ENABLED), add() also XADDs the entry onto the global outbox
stream in the same pipeline as the list push and nudges the drainer; Postgres is
never awaited on the write path. Disabled (the default), on_outbox_push is None
and the two legs above are the whole story.

recent() never asks Postgres: the list is written synchronously at song end so it
LEADS the archive, and it is capped at exactly the window -history can ask for
(HISTORY_MAX_LIMIT). An archive read could only add rows older than that window.
See docs/ARCHITECTURE.md#history-read-path.

The wire format belongs to guild_state.py (HistoryEntry +
serialize/parse_history_entry); this class never sees wire bytes.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Optional

from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import get_logger

log = get_logger(__name__)

# Ceiling on the Redis read behind -history. Short on purpose: this is a
# user-facing command with an in-memory leg one hop away, so waiting is worse
# than falling back.
_READ_TIMEOUT_SECS = 2.0


class GuildHistory:
    """Played songs, oldest-first, capped at HISTORY_CACHE_LIMIT on both legs.

    Iteration/len/indexing are exposed directly (the -history command and its
    tests read the cache as a plain sequence); mutation goes through add() and
    restore() only, so the Redis mirror can't be skipped.
    """

    __slots__ = ("_store", "_entries", "_on_outbox_push")

    def __init__(
        self,
        store: Optional[GuildRedisStore],
        *,
        on_outbox_push: Optional[Callable[[], None]],
    ) -> None:
        # on_outbox_push is the drainer's notify — a sync callable so add()
        # stays Redis-only and never awaits the archive. Optional but carrying
        # NO default: None means the archive tier is disabled
        # (HISTORY_ARCHIVE_ENABLED off, so no drainer exists to nudge), and every
        # constructor site must say so explicitly rather than fall into it —
        # silently not archiving is the failure mode the flag's strict parsing
        # exists to prevent.
        #
        # store is Optional on an independent axis: Redis may be unconfigured, in
        # which case -history falls through to the deque alone. The two compose —
        # store None / notify set degrades to memory-only, store set / notify None
        # keeps the display list while writing nothing durable.
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)
        self._on_outbox_push = on_outbox_push

    async def add(self, entry: HistoryEntry) -> None:
        """Record one played song on every configured leg — the in-memory
        cache, the Redis display list, and (while the archive is enabled) the
        Postgres outbox, the latter two in one pipeline. Degrades gracefully
        when the store is None or the push fails (GuildRedisStore logs, never
        raises; a notify after a failed push just drains an empty outbox).
        With no notify there is no drainer to nudge — the archive tier is
        disabled (HISTORY_ARCHIVE_ENABLED)."""
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
        surface.

        Two legs holding the same window: the Redis list, then the in-memory
        deque. Postgres is deliberately absent — see the module docstring. The
        command's ceiling (HISTORY_MAX_LIMIT) is pinned to HISTORY_CACHE_LIMIT
        and push_history LTRIMs the list to exactly that, so the Redis leg alone
        carries a slot for every play this command can ask for.

        The deque is MERGED, not a fallback — as a second leg it can only ADD
        depth, never remove any. Reaching it only when the leg above came back
        EMPTY let a single Redis row suppress the whole cache: at Redis-erroring
        / cache-holding-9, recent(10) returned 1 instead of 9.

        Dedup is exact rather than quantized: both legs carry the same
        `time.time()` float (orjson round-trips a double without loss), so
        equality is identity.

        The Redis read is BOUNDED at _READ_TIMEOUT_SECS even though
        GuildRedisStore swallows its own errors: swallowing turns a FAILURE into
        [], but a connected-and-unresponsive server produces no error to swallow
        and the pool sets no socket read timeout. A slow tier must cost depth,
        never an error embed. See docs/ARCHITECTURE.md#history-read-path.
        """
        # Early-out, not a correctness guard: merged[:limit] already yields [] at
        # 0 and drops the tail at a negative. It skips a round trip for a page
        # nobody asked for; the command layer validates against HISTORY_MIN_LIMIT.
        if limit <= 0:
            return []
        merged: list[HistoryEntry] = []
        seen: set[HistoryEntry] = set()

        def take(entries: Sequence[HistoryEntry]) -> None:
            """Append what this leg adds. FIRST leg to carry an entry wins."""
            for entry in entries:
                # THE WHOLE ENTRY IS THE IDENTITY, not (played_at, url). Both
                # legs carry the same object's values (orjson round-trips a
                # double without loss), so a full-value comparison is exact.
                # played_at defaults to 0.0 for entries predating the timestamped
                # wire format, so the narrower key collapsed two genuinely
                # distinct plays of one song into one and returned a page one
                # short. Plays identical in every field still collapse —
                # irreducible, and a duplicate is more visible than a loss.
                if entry not in seen:
                    seen.add(entry)
                    merged.append(entry)

        # Bound to a local: the narrowing from the `is not None` test does not
        # follow self._store into the closure below.
        store = self._store
        if store is not None:
            # A lambda, not the bound `store.get_history`: passing the bound
            # method does the ATTRIBUTE LOOKUP here, outside _read_tier's guard,
            # so a store without the method raises straight out of recent() as an
            # error embed instead of degrading to the cache.
            take(await self._read_tier("redis", lambda: store.get_history()))
        take(list(reversed(self._entries)))
        # Both legs are newest-first and hold the same window, so the
        # concatenation is nearly ordered — but only nearly, since the deque can
        # hold an entry the Redis read missed. Sort explicitly.
        #
        # A 0.0 (unknown) timestamp sorts last, which is CORRECT: the only way to
        # hold one is to predate the wire format that records played_at, so
        # "unknown time" and "older than everything timed" are the same
        # population. list.sort is stable, so ties keep leg order.
        merged.sort(key=lambda e: e.played_at, reverse=True)
        return merged[:limit]

    async def _read_tier(
        self, name: str, read: Callable[[], Awaitable[list[HistoryEntry]]]
    ) -> list[HistoryEntry]:
        """One bounded, non-raising read of a persisted leg.

        Takes a THUNK, not a coroutine: building the awaitable at the call site
        evaluates `store.get_history()` outside this guard, so a missing method or
        a wrong argument count escapes recent() and reaches the user as an error
        embed instead of degrading.

        Names the leg that FAILED rather than the one it fell to — "fell back to
        redis" was wrong whenever there was no store and the fall-through was the
        cache.
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

        Cache-only: latency-sensitive callers get an answer with no round trip,
        and staying sync is the point — reaching the list would force this
        async. restore() has already refilled the cache after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
