"""
GuildHistory — a guild's played-song history for one guild.

The domain twin of GuildQueue, one layer smaller. Two legs, both bounded at
HISTORY_CACHE_LIMIT and both holding the same thing: the Redis
guild:{id}:history list (capped and PERSISTed by push_history) and an in-memory
deque of the same window. Owning both privately means they can only move
together: every add() lands on the deque and the Redis list in one step, and
restore() refills the deque from the newest slice of the list.

WRITES fan out to a third place, READS do not, and that asymmetry is the whole
design. add() also XADDs every entry onto the global Postgres outbox stream — in
the same pipeline as the list push — and nudges the drainer, which archives it
into play_history. Postgres is never awaited on the write path: the
outbox/drainer split keeps the playback loop on Redis-only latency.

But recent() never asks Postgres. -history is a display command with a fixed
window (musicbot.HISTORY_MAX_LIMIT, pinned to HISTORY_CACHE_LIMIT), and the
Redis list is guaranteed to hold that window: it is written synchronously at
song end, so it LEADS the archive rather than lagging it, and it is capped at
exactly the number of entries the command can ask for. Reading Postgres could
only add rows older than the window or duplicates of rows already in it — which
is what the deleted three-tier merge spent its complexity reconciling
(µs-truncated timestamps against raw floats, an archive leg that looked full
while the newest plays were still in the outbox).

The archive keeps every play regardless, and PostgresHistoryArchive.recent
stays as the read surface for the commands built on that record. It is simply
not this class's business: guild history for display, Postgres for the
historical archive.

The at-rest wire format is owned by guild_state.py (HistoryEntry +
serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.
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
        on_outbox_push: Callable[[], None],
    ) -> None:
        # on_outbox_push is REQUIRED: the Postgres tier is not optional, so
        # there is no shape of this object that writes history without an outbox
        # consumer behind it. It is the drainer's notify — a sync callable so
        # add() stays Redis-only and never awaits the archive.
        #
        # No archive and no guild_id here any more. Both existed only for the
        # Postgres leg of recent(); the entry's own guild_id is stamped one layer
        # up by HistoryEntry.from_song, which is where the outbox needs it.
        #
        # store IS Optional, and that is a different axis: Redis may be
        # unconfigured, in which case there is no wire to push to at all — and
        # -history then falls through to the in-memory deque alone.
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

        Two legs, and both are the same window: the Redis list, then the
        in-memory deque. Postgres is deliberately absent — see the module
        docstring. The command's ceiling (HISTORY_MAX_LIMIT in musicbot.py) is
        pinned to HISTORY_CACHE_LIMIT, push_history LTRIMs the list to exactly
        that many entries and PERSISTs it, and get_history() reads that whole
        window, so the Redis leg alone provably contains every play this
        command can be asked for. There is nothing an archive read could add
        except rows older than the window.

        The deque is not a duplicate of that argument, it is what answers when
        Redis cannot. It covers the two cases the list leg misses: no store
        configured at all, and a Redis read that failed or timed out. Merged
        rather than used as a fallback, on the same terms as the leg above it —
        as a second leg it can only ADD depth, never remove any. An earlier
        version reached it only when the tiers above came back EMPTY, which
        meant a single row from a healthier leg suppressed the whole cache:
        reproduced at Redis-erroring / cache-holding-9, where recent(10)
        returned 1 where the pre-archive code returned 9.

        Dedup is exact rather than quantized. Both legs carry the same
        `time.time()` float — the deque holds the entry add() appended, the list
        holds the same entry through orjson, which round-trips a double without
        loss — so equality is identity here. The quantization this once needed
        existed only to reconcile Postgres' µs-truncated timestamptz with that
        raw float, and went with the Postgres leg.

        The Redis read is BOUNDED at _READ_TIMEOUT_SECS even though
        GuildRedisStore swallows its own errors: swallowing turns a FAILURE into
        [], but a connected-and-unresponsive server produces no error to
        swallow, and the pool sets no socket read timeout. A slow tier must cost
        depth, never an error embed — -history is a display command.
        """
        # An EARLY-OUT, not a correctness guard, and worth saying so because it
        # reads like one. merged[:limit] already yields [] at 0 and drops the
        # tail at a negative, so `<= 0` vs `< 0` here is an equivalent mutation
        # — no test can tell them apart, and none should be written that
        # pretends to. What this buys is skipping a network round trip for a
        # page nobody asked for. The command layer never sends one (validated
        # against HISTORY_MIN_LIMIT), so the only callers are future ones.
        if limit <= 0:
            return []
        merged: list[HistoryEntry] = []
        seen: set[tuple[float, str]] = set()

        def take(entries: Sequence[HistoryEntry]) -> None:
            """Append what this leg adds. FIRST leg to carry an entry wins."""
            for entry in entries:
                key = (entry.played_at, entry.webpage_url)
                if key not in seen:
                    seen.add(key)
                    merged.append(entry)

        # Bound to a local: the narrowing from the `is not None` test does not
        # follow self._store into the closure below.
        store = self._store
        if store is not None:
            # A lambda, not the bound `store.get_history`. Passing the bound
            # method performs the ATTRIBUTE LOOKUP at this call site, outside
            # _read_tier's guard, so a store without the method raises straight
            # out of recent() and reaches the user as an error embed instead of
            # degrading to the cache. That is the exact trap _read_tier's
            # docstring describes, and it only became reachable here when this
            # became the sole network leg.
            take(await self._read_tier("redis", lambda: store.get_history()))
        take(list(reversed(self._entries)))
        # Both legs are newest-first on their own and hold the same window, so
        # the concatenation is very nearly ordered already — but only very
        # nearly, since the deque can hold an entry the Redis read missed.
        # Sort explicitly rather than rely on that.
        merged.sort(key=lambda e: e.played_at, reverse=True)
        return merged[:limit]

    async def _read_tier(
        self, name: str, read: Callable[[], Awaitable[list[HistoryEntry]]]
    ) -> list[HistoryEntry]:
        """One bounded, non-raising read of a persisted leg.

        Takes a THUNK, not a coroutine. Building the awaitable at the call site
        evaluates `store.get_history()` outside this guard, so anything the
        attribute lookup or the call itself raises — a store missing the method,
        a wrong argument count — escapes recent() and reaches the user as an
        error embed instead of degrading. Caught by the command-level tests the
        moment this was written that way.

        Names the leg that FAILED rather than the one it fell to: the old
        message said "fell back to redis" unconditionally, which was wrong
        whenever there was no store and the fall-through was to the cache — a
        case one of this class's own tests covers by name.

        Still generic over a named leg rather than inlined into recent(), even
        with one caller left: the name is what makes the log line diagnostic,
        and the thunk is what keeps the call inside the guard.
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

        Cache-only, and deliberately so even though recent() now reads Redis
        alone: callers on a latency-sensitive path get an answer with no round
        trip at all, and this is a property rather than a shortcut — a sync one
        would otherwise have to become async to reach the list. restore() has
        already refilled the cache from Redis by the time anything reads this
        after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
