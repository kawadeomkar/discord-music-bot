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

Until HISTORY_REDIS_CUTOVER is set the Redis list is still unbounded and
PERSISTed, so it remains a complete second copy; the cutover demotes it to a
capped, TTL'd cache once the backfill has been verified
(docs/POSTGRES_HISTORY_PLAN.md, and config.history_redis_cutover for the
ordering that makes that safe).

The at-rest wire format is owned by guild_state.py (HistoryEntry +
serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.

add() also XADDs every entry onto the global Postgres outbox stream — in the same
pipeline as the display-list push — and nudges the drainer
(docs/POSTGRES_HISTORY_PLAN.md). The archive is a required tier, so that leg is
unconditional. Postgres is never awaited on the WRITE path: the outbox/drainer
split keeps the playback loop on Redis-only latency.

Reads are the other direction. recent() asks Postgres first and MERGES in
everything Redis still holds that the archive has not caught up on, plus the
cache, deduping on the archive's own identity (Phase B). That merge is what lets
the Redis list be demoted later without -history losing depth, and what keeps
the newest plays — still sitting in the outbox — visible in the meantime. No leg
is ever skipped on the grounds that an earlier one looked complete; recent()'s
docstring explains what that shortcut cost when it was there.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Optional

from src.guild_state import HistoryEntry
from src.history_archive import quantized_played_at
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import get_logger

if TYPE_CHECKING:
    from src.history_archive import HistoryArchive

log = get_logger(__name__)

# Ceiling on EACH durable read behind -history — Postgres and Redis alike.
# Short on purpose: this is a user-facing command with another tier one hop
# away, so waiting is worse than falling back. It sits well under the archive's
# own command_timeout, which is sized for the drainer's batch writes rather than
# for interactive reads, and under the connect timeout the archive's lazy pool
# build pays — see recent()'s note on the first call after a restart.
_READ_TIMEOUT_SECS = 2.0


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
        guild_id: int,
        on_outbox_push: Callable[[], None],
    ) -> None:
        # archive, guild_id and on_outbox_push are all REQUIRED: the Postgres
        # tier is not optional, so there is no shape of this object that writes
        # history without an outbox consumer behind it. archive is recent()'s
        # primary read and guild_id scopes it; on_outbox_push is the drainer's
        # notify — a sync callable so add() stays Redis-only.
        #
        # guild_id carries no default on purpose, matching HistoryEntry.from_song
        # one layer down ("a forgotten stamp would silently write unattributable
        # outbox entries"). A default of 0 would be worse here than there, and
        # in two directions now: nothing downstream can catch it on the write
        # path, because 0 is the one guild_id play_history's CHECK refuses — so
        # a missed stamp dead-letters every play of that guild rather than
        # raising — and on the READ path below it would quietly scope every
        # -history to a guild that owns nothing.
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
           the cutover demotes the Redis list to a bounded cache.
        2. The Redis list, ALWAYS merged in — never skipped on the grounds that
           the archive looked full. Covers the two cases the archive cannot:
           plays still sitting in the outbox (every song's newest few seconds)
           and every pre-existing play until `just db-backfill` has run. The dedup key
           is (played_at, webpage_url), the same identity the archive's unique
           index uses.
        3. The in-memory cache, merged in on the same terms. It was a
           last-resort fallback, reached only when both tiers above came back
           empty — which meant one archived row was enough to suppress it
           entirely. Reproduced: Redis erroring (get_history() returns [] via
           @_guild_op), archive holding 1 row, cache holding 9 → recent(10)
           returned 1, where the pre-Postgres code returned all 9. As a third
           leg it can only ADD depth, never remove any.

        UNCONDITIONAL, and that is the whole correctness argument. An earlier
        draft returned early when the archive supplied `limit` rows. That reads
        as an obvious optimisation and is wrong in the one direction that
        matters: the archive's rows are the newest ARCHIVED plays, while
        anything still in the outbox is newer than every one of them and exists
        only on the Redis list. So the shortcut fired exactly when a guild had
        `limit` archived plays — the steady state at the default limit of 10 —
        and hid the newest songs. Measured with the drainer five entries behind,
        it returned plays 94..85 when the truth was 99..90: the five newest
        missing AND five older ones displacing them, which is the same harm the
        quantization note below describes, reached through a different door. The
        window is milliseconds during a healthy drain and minutes whenever the
        drain is stalled while Postgres still answers reads (a read-only replica
        after failover, the drainer's 60s backoff, a 300s respawn) — precisely
        when -history looks healthiest, because it still returns a full page.

        Merging unconditionally is provably complete rather than merely safer:
        the Redis list is unbounded, get_history() returns its newest
        HISTORY_CACHE_LIMIT, and HISTORY_MAX_LIMIT is pinned to that same
        constant (musicbot.py), so every play among the true newest `limit` is
        in the Redis window for any limit this command accepts. The archive leg
        is what keeps that true after the cutover bounds the list. Raising
        HISTORY_MAX_LIMIT above HISTORY_CACHE_LIMIT would break the argument.

        The cost of never skipping is one LRANGE per -history — exactly what
        this command paid before Postgres was in the picture at all.

        BOTH network reads are bounded, at the same _READ_TIMEOUT_SECS, and
        every failure degrades to the tiers below it rather than propagating:
        -history is a display command, and a slow tier must cost depth, never an
        error embed. The Redis leg is bounded too even though GuildRedisStore
        swallows its own errors — swallowing turns a FAILURE into [], but a
        connected-and-unresponsive server produces no error to swallow, and the
        pool sets no socket read timeout. Earlier only the archive was bounded,
        which read as "Postgres is the risky one" when the untimed leg was the
        one that could hang forever.

        Worth knowing about that bound: the archive's pool is built lazily, so
        the first -history after a restart may pay _ensure()'s connect
        (timeout=10) and schema check under this 2s ceiling and lose. It costs
        one degraded render — the drainer or the next call establishes the pool
        — and widening the ceiling to cover a cold connect would slow every
        healthy call instead. health_check() documents the same trade for -ping.
        """
        # An EARLY-OUT, not a correctness guard, and worth saying so because it
        # reads like one. merged[:limit] already yields [] at 0 and drops the
        # tail at a negative, so `<= 0` vs `< 0` here is an equivalent mutation
        # — no test can tell them apart, and none should be written that
        # pretends to. What this buys is skipping two network round trips for a
        # page nobody asked for. The command layer never sends one (validated
        # against HISTORY_MIN_LIMIT), so the only callers are future ones.
        if limit <= 0:
            return []
        merged: list[HistoryEntry] = []
        # The identity is the QUANTIZED pair on EVERY leg. The archive's rows
        # have been through timestamptz and are microsecond-granular; the Redis
        # and cache legs carry the raw time.time() float, which is finer.
        # Comparing them directly failed for ~37% of real timestamps, so a
        # drained play appeared in two legs, survived the dedup and rendered
        # twice — and since merged[:limit] runs below, each duplicate pushed a
        # genuine older song out of the window. quantized_played_at is
        # idempotent, so applying it to the already-quantized archive leg costs
        # nothing and keeps the invariant visible on both sides of the boundary.
        seen: set[tuple[float, str]] = set()

        def take(entries: Sequence[HistoryEntry]) -> None:
            """Append what this leg adds. FIRST leg to carry an entry wins, so
            the durability order above is also the order of precedence: the
            archived copy of a play is the one that renders, not the Redis copy
            it was written from. They differ in practice — played_at is µs-
            truncated on the way through Postgres, and a backfilled row can
            carry defaults the wire entry never had."""
            for entry in entries:
                key = (quantized_played_at(entry.played_at), entry.webpage_url)
                if key not in seen:
                    seen.add(key)
                    merged.append(entry)

        take(
            await self._read_tier(
                "postgres", lambda: self._archive.recent(self._guild_id, limit)
            )
        )
        # Bound to a local: the narrowing from the `is not None` test does not
        # follow self._store into the closure below.
        store = self._store
        if store is not None:
            take(await self._read_tier("redis", store.get_history))
        take(list(reversed(self._entries)))
        # Each leg is newest-first on its own, but Redis holds entries the
        # archive has not caught up on and those are NEWER than everything
        # archived, so the concatenation is not ordered. Sort explicitly.
        merged.sort(key=lambda e: e.played_at, reverse=True)
        return merged[:limit]

    async def _read_tier(
        self, name: str, read: Callable[[], Awaitable[list[HistoryEntry]]]
    ) -> list[HistoryEntry]:
        """One bounded, non-raising read of a durable tier.

        Takes a THUNK, not a coroutine. Building the awaitable at the call site
        evaluates `archive.recent(...)` outside this guard, so anything the
        attribute lookup or the call itself raises — an archive missing the
        method, a wrong argument count — escapes recent() and reaches the user
        as an error embed instead of degrading. Caught by the command-level
        tests the moment this was written that way.

        Names the tier that FAILED rather than the one it fell to: the old
        message said "fell back to redis" unconditionally, which was wrong
        whenever there was no store and the fall-through was to the cache — a
        case one of this class's own tests covers by name.
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

        Cache-only, and deliberately so now that recent() asks Postgres and
        Redis: callers on a latency-sensitive path get an answer with no round
        trip at all, and this is a property rather than a shortcut — a sync one
        would otherwise have to become async to reach either tier. restore()
        has already refilled the cache from Redis by the time anything reads
        this after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
