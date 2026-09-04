"""
GuildQueue — all queue state and every queue operation for one guild.

Domain layer between the schema (src/guild_state.py) and playback orchestration
(src/musicplayer.py). A guild's queue is ONE deque and a cursor into it:

  _items[:_cursor]   claimed by a consumer, not yet settled ("in flight")
  _items[_cursor:]   pending
  _wake              set iff something is pending (I3, _sync_wake owns it)
  Redis guild:{id}:queue    persisted mirror of the is_persisted() subset

The cursor is the boundary, not a per-item flag, because Redis retires entries
by LPOP — so in-flight items are necessarily a PREFIX. Both legs are private:
no caller can move one without the other, and every mirror-touching mutation
(put, put_front, clear, shuffle, remove, finish_failed_dequeue) holds one
bulk-mutation mutex across its memory AND mirror writes. The dequeue commit
extends that hold over the caller's Redis write (see commit_dequeue).
Invalidation is carried by the generation counter and the cursor (see clear()).

Two counters, adjacent names, different sets: qsize() is PENDING (len - cursor),
display_size() is pending PLUS in-flight (len).

Not known here:
- stream prefetch — MusicPlayer cancels its prefetch task before
  clear()/shuffle()/remove(); the task consumes via get_nowait()
- embeds and ETA math — MusicPlayer builds them over display_items()/peek_next()
- the state hash — crash recovery hands this class ready-made entries
  (SongQueueEntry.from_crashed_state bridges the two schemas)

See docs/ARCHITECTURE.md#queue-invariant.
"""

import asyncio
import contextlib
import random
import re
from collections import deque
from collections.abc import AsyncGenerator, Callable, Sequence
from itertools import islice
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Optional, Union

import discord

from src.guild_state import Analytics, QueueEntry, SearchQueueEntry, SongQueueEntry
from src.redis_client import GuildRedisStore
from src.sources import YTSource
from src.util import get_logger
from src.youtube import QueueObject

log = get_logger(__name__)

# Live queue items — what the deque holds. The at-rest twin is
# guild_state.QueueEntry; this class converts between the two internally.
QueueItem = Union[QueueObject, YTSource]

# When the pipelined LREMs cost more than rewriting the list. `LREM key 1 <blob>`
# is O(position), so N of them cost O(N x depth) against a rebuild's O(depth):
# the depth cancels and the crossover is a COUNT. The LREMs share one MULTI/EXEC,
# so this bound is what caps how long Redis serves nobody.
# See docs/ARCHITECTURE.md#queue-operations for the measurements.
_LREM_MAX_ENTRIES = 16

# Shallow queues rebuild instead: below ~80 survivors a full rewrite is under a
# millisecond, so the shortcut has nothing to win.
_LREM_MAX_SHARE = 5


class ShuffleOutcome(Enum):
    SHUFFLED = auto()
    TOO_FEW_SONGS = auto()  # fewer than 4 queued items — nothing was mutated


class RemoveMode(StrEnum):
    """How a removal matched, for the reply to explain itself with."""

    RESOLVED = "resolved"  # the yt-dlp URL, as the Now Playing card shows it
    ORIGIN = "origin"  # what the user typed: a search term, or a source link


# One queue item → the mode it matched under, or None. Built by remove_matcher();
# remove() only applies it, so the policy stays testable without a queue.
RemoveMatcher = Callable[[QueueItem], Optional[RemoveMode]]


# A link, for folding purposes: a scheme, or a bare dotted host that parse_url
# also accepts (`-play youtu.be/X` is an ordinary input).
_LOOKS_LIKE_A_LINK = re.compile(r"^(?:\w+://|[\w-]+(?:\.[\w-]+)+/)")


def _normalize(s: str) -> str:
    """Fold a needle for comparison: collapse whitespace, and casefold anything
    that is not a link. Links keep their case because IDs inside them are
    case-sensitive — a casefolded Spotify base62 id would let ".../playlist/AbC"
    match a different playlist's ".../playlist/abc"."""
    s = " ".join(s.split()).strip("<>")
    return s if _LOOKS_LIKE_A_LINK.match(s) else s.casefold()


def remove_matcher(needle: str) -> RemoveMatcher:
    """Match a queue item against one `-remove` argument: the resolved yt-dlp URL
    first, then what the user typed. An origin match takes out every item sharing
    it, so one playlist link removes exactly the tracks it queued. Links are
    compared literally — youtu.be/x does not match an entry stored as
    youtube.com/watch?v=x."""
    # Stripped for BOTH legs, not just the fold: Discord wraps a link in angle
    # brackets when a user suppresses its embed, and the resolved leg compares
    # literally.
    needle = needle.strip().strip("<>")
    folded = _normalize(needle)

    def match(item: QueueItem) -> Optional[RemoveMode]:
        if not needle:
            # An unresolved search has url=None, which an empty needle would match
            # as "" and take out every Spotify-playlist track.
            return None
        resolved = (
            item.webpage_url if isinstance(item, QueueObject) else (item.url or "")
        )
        if resolved == needle:
            return RemoveMode.RESOLVED
        origin = item.user_input
        if origin is not None and _normalize(origin) == folded:
            return RemoveMode.ORIGIN
        return None

    return match


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoveOutcome:
    """What remove() took out. The positions are what the command reports; the
    items exist because a removed entry may be a played song whose only remaining
    record is the queue object itself (see MusicPlayer._flush_played)."""

    removed: list[QueueItem]
    positions: list[int]  # 1-indexed, as the queue embed numbers them
    # ORIGIN when anything matched on what the user typed — the case where one
    # argument removes many songs, so the reply says so. None when nothing did.
    mode: Optional[RemoveMode] = None


def _to_entry(item: QueueItem) -> QueueEntry:
    """Live queue item → at-rest entry for the Redis mirror."""
    if isinstance(item, QueueObject):
        return SongQueueEntry.from_queue_object(item)
    return SearchQueueEntry.from_ytsource(item)


def is_persisted(item: Optional[QueueItem]) -> bool:
    """True when the item has a matching entry on the Redis queue list — its
    dequeue must be mirrored with an LPOP, and rebuilds may write it back. Only
    the crash-recovered "current song" is persisted=False (its LPOP committed in
    the run that crashed). None counts as persisted — see redis_pop_for."""
    if isinstance(item, QueueObject):
        return item.persisted
    return True


class GuildQueue:
    """Every queue operation the bot can perform, in one place. Mirror-touching
    methods degrade gracefully when the store is None or a store call fails
    (GuildRedisStore logs and never raises) — the in-memory queue keeps working.
    """

    __slots__ = (
        "_guild",
        "_store",
        "_items",
        "_cursor",
        "_wake",
        "_mutex",
        "_generation",
        "_mirror_dirty",
    )

    def __init__(self, guild: discord.Guild, store: Optional[GuildRedisStore]) -> None:
        self._guild = guild
        self._store = store
        # The queue, and the boundary through it. _items[:_cursor] are claimed
        # but not yet settled; _items[_cursor:] are pending. See I1-I6 in
        # docs/ARCHITECTURE.md#queue-invariant.
        self._items: deque[QueueItem] = deque()
        self._cursor = 0
        self._wake = asyncio.Event()
        self._mutex = asyncio.Lock()
        self._generation = 0
        # True while the list holds an entry memory already retired: the start
        # transaction's LPOP did not land. Cleared by the next write that replaces
        # the list outright — a rebuild, a DELETE, or the next song start.
        self._mirror_dirty = False

    # ── Wake discipline ───────────────────────────────────────────────────────

    def _sync_wake(self) -> None:
        """Restore I3: _wake is set iff something is pending, and the only writer
        of it. A stale set does not degrade — Event.wait() returns without
        yielding when already set, so get()'s wait loop loses its suspension
        point and the whole event loop stops."""
        if self._cursor < len(self._items):
            self._wake.set()
        else:
            self._wake.clear()

    # ── Consumption (playback loop + prefetch task) ───────────────────────────

    async def get(self) -> QueueItem:
        """Claim the next pending item, waiting for one if the queue is drained.

        `while`, never `if`: Event.wait() wakes EVERY waiter and the prefetch's
        get_nowait() is a second consumer, so a woken getter can find the item
        already taken. No await between the claim and the return, so a claim is
        atomic on the event loop."""
        while self._cursor >= len(self._items):
            await self._wake.wait()
        item = self._items[self._cursor]
        self._cursor += 1
        self._sync_wake()
        return item

    def get_nowait(self) -> QueueItem:
        """Raises asyncio.QueueEmpty — the prefetch task relies on that."""
        if self._cursor >= len(self._items):
            raise asyncio.QueueEmpty
        item = self._items[self._cursor]
        self._cursor += 1
        self._sync_wake()
        return item

    def requeue_front(self, item: QueueItem) -> None:
        """Give a claim back: the cursor steps back and the item is pending again;
        the mirror never moved. `item` may be the RESOLVED form of what was claimed
        and replaces it, so a later rebuild serializes the resolved entry.

        It writes index 0 because that is the only claimed slot there is: the
        prefetch's requeue always lands before loop() can take a second claim, so
        _cursor is never above 1 here and index 0 IS _cursor."""
        # Guarded like every other cursor decrement: unguarded, _cursor == 0 goes
        # negative and _items[-1] = item clobbers the TAIL.
        if self._cursor > 0:
            self._cursor -= 1
            self._items[0] = item
        self._sync_wake()

    def empty(self) -> bool:
        return self._cursor >= len(self._items)

    def qsize(self) -> int:
        """PENDING only — what is still waiting to be claimed."""
        return len(self._items) - self._cursor

    def display_size(self) -> int:
        """Pending PLUS in-flight — what a new arrival actually waits behind.

        NOT qsize(), which is one term apart over the same two fields. This is the
        sole input to play_history.queue_position (MusicPlayer.enqueue_depth), and
        a claimed item is still ahead of an arrival, so qsize() here would write a
        plausible wrong number to Postgres."""
        return len(self._items)

    @property
    def generation(self) -> int:
        """Monotonic counter of queue invalidations, bumped by clear(). A dequeue
        captures it when it takes its item and hands it back to commit_dequeue(),
        which refuses across a bump — see there for why emptiness alone cannot
        answer this."""
        return self._generation

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def put(
        self,
        items: Sequence[QueueItem],
        *,
        batch: bool = False,
    ) -> list[QueueItem]:
        """Enqueue on the deque first, then the mirror. Under
        the bulk-mutation mutex because the Redis pushes suspend: a clear()/
        shuffle() interleaving there drains/rebuilds the mirror before the pushes
        land, resurrecting them as ghosts the next dequeue LPOPs instead of its own
        entry. batch=True pushes every entry in one round-trip (bulk playlist);
        batch=False one RPUSH per entry."""
        async with self._mutex:
            queued = list(items)
            self._items.extend(queued)
            self._sync_wake()
            if self._store is None or not queued:
                return queued
            # Filtered like every other mirror write: a persisted=False entry
            # written here would never be LPOPed at its dequeue (redis_pop_for
            # skips them), leaving the mirror a permanent entry ahead.
            entries = [_to_entry(item) for item in queued if is_persisted(item)]
            if not entries:
                return queued
            if batch:
                await self._store.push_queue_batch(entries)
            else:
                for entry in entries:
                    await self._store.push_queue(entry)
            return queued

    async def put_front(self, items: Sequence[QueueItem]) -> list[QueueItem]:
        """Insert items at the front of the queue — the -playnow interjection path.
        Under the bulk-mutation mutex, like every multi-leg mutation.

        An in-flight head (dequeued but uncommitted) keeps its position AHEAD of
        the inserted items and forces the mirror down the rebuild path: its Redis
        entry still sits at the list head awaiting a commit-time LPOP, so an LPUSH
        in front of it would make that LPOP eat the new head.

        That branch is reachable despite looking unused: _interject_flow's
        outcome-is-None fallback calls this with the prefetch's claim still open.
        """
        if not items:
            return []
        async with self._mutex:
            new_items = list(items)
            # Inserting at _cursor IS inserting behind the in-flight head.
            # reversed(), or a multi-track insert lands backwards.
            for item in reversed(new_items):
                self._items.insert(self._cursor, item)
            self._sync_wake()

            if self._cursor:
                await self._write_mirror(self._items)
            elif self._store is not None:
                # An LPUSH of just the new items, not a replacement — so not
                # _write_mirror's job. Reachable only with no in-flight head,
                # where nothing ahead of them needs preserving.
                entries = [_to_entry(s) for s in new_items if is_persisted(s)]
                if entries:
                    await self._store.push_queue_front(entries)
            return new_items

    # ── Bulk operations ───────────────────────────────────────────────────────
    # Callers with a prefetch task (MusicPlayer) must cancel it before any of
    # these: a running prefetch holds a get_nowait() item, and its CancelledError
    # handler's requeue_front() must land before the drain, or the item is
    # stranded. A COMPLETED prefetch is fine — its item is an in-flight head.

    async def clear(self) -> list[QueueItem]:
        """Empty the queue, returning everything that was on it — the claimed
        prefix included, because the caller records these
        (MusicPlayer._flush_played) and a parked -playnow tail is in it.

        Bumps the generation under the mutex before draining, and resets the
        cursor: a claim the loop took before this clear() captured the old value
        and is refused by commit_dequeue(); a claim a prefetch took commits under
        the current value and is refused because nothing is claimed at cursor 0.

        The DEL is inside the mutex too: released early, a concurrent put()'s
        mirror writes would land between the drain and the DEL and be wiped."""
        async with self._mutex:
            self._generation += 1
            cleared_items = list(self._items)
            self._items.clear()
            self._cursor = 0
            self._sync_wake()
            if self._store is not None and await self._store.delete_queue():
                self._mirror_dirty = False
        return cleared_items

    async def shuffle(self) -> ShuffleOutcome:
        """Shuffle the pending items in place: drain → shuffle → refill under one
        continuous mutex hold, so the loop never sees a mid-shuffle empty queue.
        A claimed item keeps its position — only what is still pending is
        reordered."""
        # display_size(), the number -queue and -debug show: a claim held by the
        # loop or a prefetch is displayed but not pending, and the refusal below
        # quotes the number the user is looking at.
        async with self._mutex:
            # Counted inside the lock: acquiring suspends, and a clear() or
            # remove() completing in that gap leaves nothing to shuffle.
            if self.display_size() < 4:
                return ShuffleOutcome.TOO_FEW_SONGS
            # A list round-trip because deque has no slicing.
            head = list(islice(self._items, 0, self._cursor))
            tail = list(islice(self._items, self._cursor, None))
            random.shuffle(tail)
            self._items = deque(head + tail)
            self._sync_wake()

            if tail:
                await self._write_mirror(self._items)

        return ShuffleOutcome.SHUFFLED

    async def remove(self, match: RemoveMatcher) -> RemoveOutcome:
        """Remove every queued item `match` accepts. Returns the removed items with
        their 1-indexed positions as the queue embed shows them — the items because
        a removed entry can be the last record of a song that already played. An
        in-flight dequeue is never removed even on a match (it is committed to play;
        stopping it is -skip's job) but still occupies a display position — hence
        the numbering offset. The matching policy is remove_matcher's."""
        removed_positions: list[int] = []
        removed_items: list[QueueItem] = []
        kept: list[QueueItem] = []
        modes: list[RemoveMode] = []

        async with self._mutex:
            # This enumerate IS the position contract: the queue embed numbers
            # every item from 1 including the in-flight head, so a pending item
            # sits at _cursor + 1 and up, and skipping `pos <= _cursor` keeps a
            # claimed song un-removable while still counted.
            for pos, item in enumerate(self._items, start=1):
                if pos <= self._cursor:
                    kept.append(item)
                    continue
                mode = match(item)
                if mode is not None:
                    removed_positions.append(pos)
                    removed_items.append(item)
                    modes.append(mode)
                else:
                    kept.append(item)

            # Nothing matched: no mutation, so no mirror write.
            if not removed_positions:
                return RemoveOutcome(removed=[], positions=[], mode=None)

            self._items = deque(kept)
            self._sync_wake()  # cursor unchanged — nothing before it can move
            await self._write_mirror(self._items, removed=removed_items)

        return RemoveOutcome(
            removed=removed_items,
            positions=removed_positions,
            # ORIGIN if it is anywhere in the run: the argument reached items the
            # resolved URL alone would not have.
            mode=RemoveMode.ORIGIN
            if RemoveMode.ORIGIN in modes
            else RemoveMode.RESOLVED,
        )

    # ── Crash recovery ────────────────────────────────────────────────────────

    async def restore_crashed(
        self,
        entry: SongQueueEntry,
        *,
        requester_fallback: Union[discord.Member, discord.User, None],
    ) -> bool:
        """Re-queue the crash-recovered "current song" at the front of the line, in
        memory only: persisted=False, since its LPOP already committed and the loop
        must not LPOP for it (see redis_pop_for). Inserted at the cursor, so the
        claimed items stay a prefix. requester_fallback covers a requester ID that
        no longer resolves; False when nobody does, and the caller still clears the
        crashed-song state."""
        item = self._rehydrate(entry, requester_fallback=requester_fallback)
        if item is None:
            return False
        self._items.insert(self._cursor, item)
        self._sync_wake()
        return True

    async def restore_entries(self, entries: Sequence[QueueEntry]) -> int:
        """Re-queue persisted entries after a restart, preserving order.
        In memory only — the entries are already on the Redis list.
        Entries whose requester cannot be resolved (member left and the guild
        has no owner) are dropped. Returns the number restored."""
        count = 0
        for entry in entries:
            item = self._rehydrate(entry)
            if item is not None:
                self._items.append(item)
                count += 1
        self._sync_wake()
        return count

    # ── Display data (embed/ETA builders live in MusicPlayer) ─────────────────

    def display_items(self) -> list[QueueItem]:
        """Snapshot of the queued items in display order."""
        return list(self._items)

    def peek_next(self) -> Optional[QueueItem]:
        return self._items[0] if self._items else None

    def has_resume_tail(self, webpage_url: str) -> bool:
        """True when the queue already carries the resume tail an interjection
        left behind for `webpage_url`. That entry and the live song are the SAME
        play, so anything counting queue depth must count them once.

        Matches on URL, not identity, so a tail parked by an EARLIER play of the
        same song answers True for the current one. Its one caller
        (MusicPlayer.enqueue_depth) then under-counts by one; accepted there.

        O(len(_items)), early-exiting only when a tail exists: 90us at 1000
        entries, synchronous, at -play dispatch."""
        return any(
            isinstance(item, QueueObject)
            and item.is_resume
            and item.webpage_url == webpage_url
            for item in self._items
        )

    def resume_tail_depth(self) -> int:
        """How many parked plays are waiting behind the song that just cut the line
        — the run of consecutive resume tails after it. 1 is a plain -playnow, 2+ a
        stack. Counts PLAYS, not fragments: the interrupted song's live fragment is
        gone by the time this runs and only its tail is queued.

        The interjected song is not always at index 0 — put_front inserts behind a
        dequeued-but-uncommitted item — so the run starts after the claimed prefix,
        which _cursor names directly."""
        # islice, not a slice copy: the run read here is almost always 1-3 long.
        depth = 0
        for item in islice(self._items, self._cursor + 1, None):
            if not (isinstance(item, QueueObject) and item.is_resume):
                break
            depth += 1
        return depth

    # ── Playback-loop dequeue bookkeeping ─────────────────────────────────────

    # Settle through commit_dequeue() or finish_failed_dequeue(), never
    # try_release() alone: those two carry the mirror leg with them.

    def try_release(self) -> bool:
        """Settle one claim: drop the head and step the cursor back, which leaves
        `len - _cursor` unchanged — the item leaves the queue, and what was pending
        stays pending.

        False means there was nothing claimed to settle, which is what a clear()
        during the resolve leaves behind. Guarded rather than raising, because
        driving _cursor negative would eat a PENDING item on the next release and
        break I1. Both callers make the no-op correct: commit_dequeue() refuses a
        commit taken before a clear(), and finish_failed_dequeue() gates its LPOP
        on this return value, since nothing claimed means the mirror holds only
        what was queued after that clear. Use one of them unless already holding
        the bulk-mutation lock — the check must not race a clear()/shuffle()."""
        if self._cursor == 0:
            return False
        self._items.popleft()
        self._cursor -= 1
        self._sync_wake()
        return True

    async def finish_failed_dequeue(
        self,
        item: Optional[QueueItem],
        *,
        context: str = "dequeue",
        persisted: Optional[bool] = None,
    ) -> None:
        """Settle a claim for an item that will never play, on both legs — the pair
        every loop failure path shares. `context` labels the nothing-claimed
        warning. The mutex spans the settle and the LPOP so a bulk mutation cannot
        rebuild the mirror between them and have the LPOP hit the new head.

        `persisted` overrides what `item` says about itself, and exists for the one
        caller holding a claim it cannot describe with a QueueItem — see
        redis_pop_for.

        The LPOP is GATED on the settle: nothing claimed means a clear() already
        retired this item and reset the cursor, so the mirror holds only what was
        queued afterwards and popping it deletes an unrelated song."""
        async with self._mutex:
            if not self.try_release():
                log.warning(
                    f"song_queue was empty on {context} in guild {self._guild.id}; "
                    "leaving the mirror alone"
                )
                return
            await self.redis_pop_for(item, persisted=persisted)

    @contextlib.asynccontextmanager
    async def commit_dequeue(self, generation: int) -> AsyncGenerator[bool]:
        """Settle the claim for a song about to play, with the bulk mutex held
        across the caller's own Redis write. Yields False when the queue was
        cleared since the claim, which means discard the song (its FFmpeg
        cleanup stays caller-side, outside the hold).

        `generation` is the value the caller captured beside its claim. A
        put() refilling the head before the commit, or a YTSource resolved to
        the QueueObject it became, cannot stand in for it; a prefetched claim
        passes the current value and is refused by the cursor reset instead.

        Keep the body to ONE bounded Redis write and never touch Discord in it:
        the pool sets no socket_timeout. vc.play() belongs inside or after the
        commit, never before it. See docs/ARCHITECTURE.md#queue-operations."""
        async with self._mutex:
            if generation != self._generation:
                yield False
                return
            yield self.try_release()

    @property
    def mirror_dirty(self) -> bool:
        """True while the Redis list holds an entry memory already retired. The
        next song start must rebuild the list rather than LPOP it, or it retires
        the wrong entry; a -clear/-shuffle/-remove rebuild clears it in passing."""
        return self._mirror_dirty

    def note_mirror_write(self, *, landed: bool, retired: bool) -> None:
        """Record the start transaction's outcome, from inside commit_dequeue's
        hold. A write that landed leaves the list correct whatever leg it carried;
        one that did not, while `retired` says memory dropped an entry, leaves
        the list one ahead."""
        if landed:
            self._mirror_dirty = False
        elif retired:
            self._mirror_dirty = True

    def mirror_entries(self) -> list[QueueEntry]:
        """The list as the mirror should hold it: the persisted subset of the
        deque, claimed prefix included, in order — what a rebuild writes."""
        return [_to_entry(s) for s in self._items if is_persisted(s)]

    async def redis_pop_for(
        self, item: Optional[QueueItem], *, persisted: Optional[bool] = None
    ) -> None:
        """Mirror one in-memory dequeue to Redis via LPOP — unless the entry was
        never on the list (persisted=False: the crash-recovered "current song",
        whose LPOP committed in the crashed run), where an LPOP would silently
        delete an unrelated, still-queued song.

        `persisted` is for a caller whose claim is not a QueueItem it can hand
        over: the playback loop settles a claim it took as a prefetched YTDL. That
        claim can be unpersisted — a cold-start -play front-inserts at cursor 0,
        AHEAD of a crash-recovered head, so the prefetch behind it takes that head
        — and item=None defaults to popping, so only an explicit False says so."""
        if persisted is None:
            persisted = is_persisted(item)
        if self._store is not None and persisted:
            await self._store.pop_queue()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _write_mirror(
        self, items: Sequence[QueueItem], *, removed: Sequence[QueueItem] = ()
    ) -> None:
        """Bring the Redis mirror in line with `items` — the persisted subset, in
        order — by whichever of DELETE, LREM or rebuild is right.

        The rebuild is DELETE + RPUSH in MULTI: a plain pipeline leaves a window
        where a concurrent LPOP sees an empty queue. Callers hold the bulk-mutation
        mutex, so a concurrent put()'s pushes can't be wiped by a rebuild that
        predates them. persisted=False items were never RPUSHed — never write them in.

        Empty means DELETE, not skip, or the old list survives for the next restore
        to find.

        `removed` is the LREM shortcut, and only a removal may pass it: LREM assumes
        the survivors kept their order, which is false for shuffle and for any
        insert — and a stale list (mirror_dirty) has no order to keep. It is capped
        by COUNT (see _LREM_MAX_ENTRIES) and guarded twice more,
        because LREM matches on exact bytes: skipped outright when a removed blob is
        byte-identical to a claimed item's, since LREM takes the head-most copy —
        the entry awaiting a commit-time LPOP — and falling through to the rebuild
        on a short count, which is what a queue object mutated after its entry was
        written looks like.
        """
        if self._store is None:
            return
        survivors = sum(1 for s in items if is_persisted(s))
        if not survivors:
            if await self._store.delete_queue():
                self._mirror_dirty = False
            return
        # The gate needs only a count, and one -remove of a collection link
        # routinely drops hundreds: serializing 500 entries measured 2.5ms on the
        # single event loop, counting them 15us.
        dropped_count = sum(1 for s in removed if is_persisted(s))
        if (
            dropped_count
            and not self._mirror_dirty
            and dropped_count <= _LREM_MAX_ENTRIES
            and dropped_count * _LREM_MAX_SHARE <= survivors
        ):
            # Built inside the gate, where the count is capped at
            # _LREM_MAX_ENTRIES. remove_queue_entries() re-encodes for its own
            # pipeline, so only the guard below is served here.
            dropped = [_to_entry(s) for s in removed if is_persisted(s)]
            dropped_blobs = [entry.to_redis() for entry in dropped]
            if not self._claimed_blobs(dropped_blobs):
                if await self._store.remove_queue_entries(dropped) == len(dropped):
                    return
                log.warning(
                    f"queue mirror diverged from memory in guild {self._guild.id}; "
                    "rebuilding instead of removing"
                )
        if await self._store.rebuild_queue(
            [_to_entry(s) for s in items if is_persisted(s)]
        ):
            self._mirror_dirty = False

    def _claimed_blobs(self, dropped_blobs: Sequence[bytes]) -> bool:
        """True when any entry about to be LREMed serializes exactly like a CLAIMED
        item's. LREM takes the head-most equal element, which would be the entry
        awaiting its commit-time LPOP."""
        if not self._cursor:
            return False
        claimed = {
            _to_entry(s).to_redis()
            for s in islice(self._items, 0, self._cursor)
            if is_persisted(s)
        }
        return any(blob in claimed for blob in dropped_blobs)

    def _rehydrate(
        self,
        entry: QueueEntry,
        *,
        requester_fallback: Union[discord.Member, discord.User, None] = None,
    ) -> Optional[QueueItem]:
        """At-rest entry → live queue item: the one construction path for
        everything coming back from Redis (pending entries and the crashed head).
        SongQueueEntry needs a requester resolved from the guild: the persisted
        member ID, else requester_fallback (default guild.owner), else dropped."""
        if isinstance(entry, SearchQueueEntry):
            return YTSource(
                ytsearch=entry.ytsearch,
                url=entry.url,
                process=entry.process,
                ts=entry.ts,
                analytics=Analytics(
                    queued_at=entry.queued_at,
                    queue_position=entry.queue_position,
                ),
                user_input=entry.user_input,
                query_source=entry.query_source,
            )
        requester: Union[discord.Member, discord.User, None] = None
        if entry.requester_id is not None:
            requester = self._guild.get_member(entry.requester_id)
        if requester is None:
            requester = (
                requester_fallback
                if requester_fallback is not None
                else self._guild.owner
            )
        if requester is None:
            return None
        return QueueObject(
            entry.webpage_url,
            entry.title,
            requester,
            ts=entry.ts,
            user_input=entry.user_input,
            duration=entry.duration,
            uploader=entry.uploader,
            thumbnail=entry.thumbnail,
            persisted=entry.persisted,
            interjected=entry.interjected,
            is_resume=entry.is_resume,
            start_paused=entry.start_paused,
            analytics=Analytics(
                queued_at=entry.queued_at,
                queue_position=entry.queue_position,
            ),
            query_source=entry.query_source,
            played_at=entry.played_at,
            # No np_host_ref: a live Message cannot survive a restart, so a
            # rehydrated tail can only delete a dedicated card by id.
            np_message_id=entry.np_message_id,
            np_channel_id=entry.np_channel_id,
            np_dedicated=entry.np_dedicated,
        )
