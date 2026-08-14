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
bulk-mutation mutex across its memory AND mirror writes. One residual window
remains — see the ISSUE below. The cleared-flag the playback loop consumes
lives here too.

Two counters, adjacent names, different sets: qsize() is PENDING
(len - cursor), display_size() is pending PLUS in-flight (len). They are one
term apart and display_size() is the sole input to a Postgres column, so read
the docstrings before touching either.

Not known here:
- stream prefetch — MusicPlayer cancels its prefetch task before
  clear()/shuffle()/remove(); the task consumes via get_nowait()
- embeds and ETA math — MusicPlayer builds them over display_items()/peek_next()
- the state hash — crash recovery hands this class ready-made entries
  (SongQueueEntry.from_crashed_state bridges the two schemas)

See docs/ARCHITECTURE.md#queue-invariant.
"""

# ISSUE: Close the queue-desync race between dequeue commit and the Redis LPOP.
# try_commit_dequeue() releases the bulk mutex before pop_queue_and_start_song()'s
# atomic LPOP+HSET dispatches, so a -clear landing in that tick pops an entry the clear
# already deleted: memory and Redis drift by one until the next rebuild, and a crash
# inside the drift restores the queue one song out of alignment. Accepted. Cheapest fix:
# hold the mutex across the store dispatch, costing one ~1ms round-trip.

import asyncio
import random
from collections import deque
from itertools import islice
from collections.abc import Callable, Sequence
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

# Live queue items — what the in-memory legs hold. The at-rest twin is
# guild_state.QueueEntry; this class converts between the two internally.
QueueItem = Union[QueueObject, YTSource]

# Above this many removed entries, the pipelined LREMs cost more than rewriting
# the whole list. Measured on redis:7-alpine at depth 1000 — a rebuild is a flat
# ~5.7ms whatever it drops, while LREM runs 0.96ms for one entry, 3.4ms for 100
# and 5.8ms for 250, crossing over near 270. Set below that: the margin absorbs
# the depth the crossover moves with, and every realistic collection removal
# (an album, a playlist page) sits far under it.
#
# NOT the ~40 an earlier estimate gave. That came from fakeredis, where a
# pipeline costs what its commands cost; against a real server the whole batch
# is one round trip, so LREM stays ahead roughly six times longer.
_LREM_MAX_ENTRIES = 200


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


def _normalize(s: str) -> str:
    """Fold a needle for comparison: collapse whitespace, and casefold anything
    that is not a URL. Links keep their case because IDs inside them are
    case-sensitive — a casefolded Spotify base62 id would let ".../album/AbC"
    match a different album's ".../album/abc"."""
    s = " ".join(s.split())
    return s if s[:8].lower().startswith(("http://", "https:/")) else s.casefold()


def remove_matcher(needle: str) -> RemoveMatcher:
    """Match a queue item against one `-remove` argument, by resolved URL or by
    what the user typed, whichever hits.

    The union rather than a mode flag: the two are disjoint in practice — a
    resolved yt-dlp URL is not something anyone types to queue with — so
    `-remove <anything>` does the right thing with no syntax to learn. RESOLVED is
    tried first, which keeps every removal that worked before working the same way.

    Matching an origin removes every item sharing it, which is the point: one
    album link takes back out exactly the tracks it put in. Links are compared
    literally, so youtu.be/x will not match an entry stored as
    youtube.com/watch?v=x — see the deferred canonicalisation note in -help."""
    folded = _normalize(needle)

    def match(item: QueueItem) -> Optional[RemoveMode]:
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


@dataclass(frozen=True)
class RemoveOutcome:
    """What remove() took out. The positions are what the command reports; the
    items exist because a removed entry may be a played song whose only remaining
    record is the queue object itself (see MusicPlayer._flush_played)."""

    removed: list[QueueItem]
    positions: list[int]  # 1-indexed, as the queue embed numbers them
    # ORIGIN when anything matched on what the user typed rather than the resolved
    # URL — the case where one argument removes many songs, so the reply says so.
    # None when nothing matched.
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
        "_cleared",
        "_generation",
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
        self._cleared = False
        self._generation = 0

    # ── Strangler bookkeeping (deleted with the old legs) ─────────────────────

    def _sync_wake(self) -> None:
        """Restore I3: _wake is set iff something is pending.

        The ONLY writer of _wake. A method that sets or clears it by hand states a
        conclusion, and a wrong one does not degrade — Event.wait() returns without
        yielding when already set, so a stale set turns the future get()'s wait loop
        into a loop with no suspension point and the whole event loop stops."""
        if self._cursor < len(self._items):
            self._wake.set()
        else:
            self._wake.clear()

    # ── Consumption (playback loop + prefetch task) ───────────────────────────

    async def get(self) -> QueueItem:
        """Claim the next pending item, waiting for one if the queue is drained.

        `while`, never `if`. Two things depend on the re-test, and dropping it
        breaks both: a getter cancelled after being woken leaves _wake set with
        nobody having claimed, and Event.wait() wakes EVERY waiter, so a second
        consumer would take the item this one was woken for. Re-testing the
        condition is what makes both harmless — this is the condition-variable
        pattern, and _sync_wake() is what keeps the condition and the Event
        honest with each other.

        No await between the claim and the return, so a cancellation lands either
        side of it and never inside — a claim is atomic on the event loop."""
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
        """Undo a get()/get_nowait() whose consumer abandoned the item without
        playing it (prefetch cancellation). The display and Redis legs never moved,
        so undoing the pending leg realigns all three and leaves no in-flight head.
        The abandoned get()'s task slot transfers to the re-put: callers must not
        also call task_done().

        `item` may be the RESOLVED form of what was dequeued (YTSource →
        QueueObject), so the display head is realigned onto it: put_front's
        in-flight-head branch rebuilds the Redis mirror FROM the display, and a
        stale YTSource there would persist a search over an entry that had already
        resolved — re-running the ytsearch after a crash, free to rank a different
        video."""
        # Guarded like every other cursor decrement: unguarded, _cursor == 0 would
        # go negative and _items[-1] = item would clobber the TAIL.
        if self._cursor > 0:
            self._cursor -= 1
            self._items[self._cursor] = item
        self._sync_wake()

    def empty(self) -> bool:
        return self._cursor >= len(self._items)

    def qsize(self) -> int:
        """PENDING only — what is still waiting to be claimed."""
        return len(self._items) - self._cursor

    def display_size(self) -> int:
        """Pending PLUS in-flight — what a new arrival actually waits behind.

        NOT qsize(), and the two are now one term apart over the same two fields,
        so a swap compiles and type-checks. It is the sole input to
        play_history.queue_position (MusicPlayer.enqueue_depth), and a claimed item
        is still ahead of an arrival — so qsize() here would undercount by one
        exactly when a -play lands during another song's resolve, and write a
        plausible wrong number to Postgres with nothing to detect it."""
        return len(self._items)

    def consume_cleared_flag(self) -> bool:
        """Read-and-reset the queue-was-cleared flag. clear() sets it under the
        mutex; the loop consumes it once per iteration to learn that a prefetched
        song it is holding was invalidated and must be discarded."""
        was_cleared = self._cleared
        self._cleared = False
        return was_cleared

    @property
    def generation(self) -> int:
        """Bumped by clear(). A dequeue captures this when it takes its item and
        hands it back to try_commit_dequeue(), which refuses to commit across a
        bump — see that method for why emptiness alone cannot answer this."""
        return self._generation

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def put(
        self,
        items: Sequence[QueueItem],
        *,
        batch: bool = False,
    ) -> list[QueueItem]:
        """Enqueue on all three legs: in-memory puts first, then the mirror. Under
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
            entries = [_to_entry(item) for item in queued]
            if batch:
                await self._store.push_queue_batch(entries)
            else:
                for entry in entries:
                    await self._store.push_queue(entry)
            return queued

    async def put_front(self, items: Sequence[QueueItem]) -> list[QueueItem]:
        """Insert items at the front of all three legs — the -playnow interjection
        path. Under the bulk-mutation mutex, like every multi-leg mutation.

        An in-flight head (dequeued but uncommitted) keeps its position AHEAD of
        the inserted items on the display leg and forces the mirror down the
        rebuild path: its Redis entry still sits at the list head awaiting a
        commit-time LPOP, so an LPUSH in front of it would make that LPOP eat the
        new head.

        That branch is reachable, despite looking unused.
        MusicPlayer.interject() neutralizes the prefetch first, but one interleaving
        still reaches it: the song ends naturally, the loop claims a still-running
        prefetch and awaits it (up to yt-dlp's socket timeout), and interject() runs
        inside that await — its neutralize finds no task to take while the prefetch's
        dequeued item sits uncommitted at the display head.
        """
        if not items:
            return []
        async with self._mutex:
            new_items = list(items)
            # Inserting at _cursor IS inserting behind the in-flight head, which is
            # what the lift-and-replace on the display leg used to spell out.
            # reversed(), or a multi-track insert lands backwards.
            for item in reversed(new_items):
                self._items.insert(self._cursor, item)
            self._sync_wake()

            if self._cursor:
                await self._write_mirror(self._items)
            elif self._store is not None:
                # An LPUSH of just the new items, not a replacement — so
                # deliberately not _write_mirror's job. Reachable only with no
                # in-flight head, where nothing ahead of them needs preserving.
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
        prefix included. The caller records these (MusicPlayer._flush_played), so
        returning only what was pending would drop a parked -playnow tail's
        play_history row with no error.

        Sets the cleared-flag under the mutex before draining, so a loop iteration
        holding a prefetched song discards it. The DEL is inside the mutex too:
        released early, a concurrent put()'s mirror writes would land between the
        drain and the DEL and be wiped."""
        async with self._mutex:
            self._cleared = True
            self._generation += 1
            cleared_items = list(self._items)
            self._items.clear()
            self._cursor = 0
            self._sync_wake()
            if self._store is not None:
                await self._store.delete_queue()
        return cleared_items

    async def shuffle(self) -> ShuffleOutcome:
        """Shuffle the pending items in place: drain → shuffle → refill under one
        continuous mutex hold, so the loop never sees a mid-shuffle empty queue.
        Requires 4+ pending items. A claimed item keeps its position — only what
        is still pending is reordered."""
        # FIXME: -shuffle requires 4 queued songs but tells the user it needs 3.
        # MusicPlayer.queue_shuffle() refuses with "at least 3 songs" and -help says
        # "(3+ songs)", so a user with exactly 3 queued is refused by a message
        # stating a requirement they have met. Fix: drop this to < 3, or correct
        # both user-facing strings to 4.
        if self.qsize() < 4:
            return ShuffleOutcome.TOO_FEW_SONGS

        async with self._mutex:
            # A list round-trip, not an in-place shuffle: deque has no slicing,
            # and Fisher-Yates over one measures 434us against 268 for this.
            head = list(islice(self._items, 0, self._cursor))
            tail = list(islice(self._items, self._cursor, None))
            random.shuffle(tail)
            self._items = deque(head + tail)
            self._sync_wake()

            # Was a skip rather than a DELETE when nothing persisted survived —
            # unreachable (that needs 4+ crash-recovered items and restore_crashed
            # makes one), and the DELETE is the safer of the two anyway: it heals a
            # mirror holding entries memory no longer has.
            if tail:
                await self._write_mirror(self._items)

        return ShuffleOutcome.SHUFFLED

    async def remove(self, match: RemoveMatcher) -> RemoveOutcome:
        """Remove every queued item `match` accepts. Returns the removed items with
        their 1-indexed positions as the queue embed shows them — the items because
        a removed entry can be the last record of a song that already played. An
        in-flight dequeue is never removed even on a match (it is committed to play;
        stopping it is -skip's job) but still occupies a display position — hence
        the numbering offset.

        A predicate rather than a URL: the matching policy is remove_matcher's, and
        keeping it out here is what lets it be tested without a queue."""
        removed_positions: list[int] = []
        removed_items: list[QueueItem] = []
        kept: list[QueueItem] = []
        modes: list[RemoveMode] = []

        async with self._mutex:
            # One pass, and this enumerate IS the position contract: the queue
            # embed numbers every item from 1 including the in-flight head, so a
            # pending item sits at _cursor + 1 and up. Skipping `pos <= _cursor`
            # is what keeps a claimed song un-removable while still counted.
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

            # Nothing matched: no structural mutation, so no mirror write and no
            # legs to rebuild.
            if not removed_positions:
                return RemoveOutcome(removed=[], positions=[], mode=None)

            self._items = deque(kept)
            self._sync_wake()  # cursor unchanged — nothing before it can move
            await self._write_mirror(self._items, removed=removed_items)

        return RemoveOutcome(
            removed=removed_items,
            positions=removed_positions,
            # ORIGIN if it is anywhere in the run: a mixed match means the argument
            # reached items the resolved URL alone would not have, which is the
            # thing the reply has to explain.
            mode=RemoveMode.ORIGIN
            if RemoveMode.ORIGIN in modes
            else (RemoveMode.RESOLVED if modes else None),
        )

    # ── Crash recovery ────────────────────────────────────────────────────────

    async def restore_crashed(
        self,
        entry: SongQueueEntry,
        *,
        requester_fallback: Union[discord.Member, discord.User, None],
    ) -> bool:
        """Re-queue the crash-recovered "current song" at the front of the line.
        In-memory legs only: the entry is persisted=False — its LPOP already
        committed, so it is not on the Redis list and the loop must not LPOP for it
        (see redis_pop_for). requester_fallback (guild.me or guild.owner) covers a
        persisted requester ID that no longer resolves; False when nobody does, and
        the caller still owns clearing the crashed-song state."""
        item = self._rehydrate(entry, requester_fallback=requester_fallback)
        if item is None:
            return False
        self._items.append(item)
        self._sync_wake()
        return True

    async def restore_entries(self, entries: Sequence[QueueEntry]) -> int:
        """Re-queue persisted entries after a restart, preserving order.
        In-memory legs only — the entries are already on the Redis list.
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

        O(len(_items)), and it only early-exits when a tail EXISTS — the common
        no-interjection case is a full scan (measured 90us at 1000 entries,
        synchronous, at -play dispatch). Left alone deliberately: a maintained
        tail counter would be a second thing every bulk mutation has to keep in
        step with the deque, which is the failure this class was rebuilt to end.
        90us behind a 1-4s extraction does not buy that risk."""
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
        items = list(self._items)
        start = self._cursor + 1
        depth = 0
        for item in items[start:]:
            if not (isinstance(item, QueueObject) and item.is_resume):
                break
            depth += 1
        return depth

    # ── Playback-loop dequeue bookkeeping ─────────────────────────────────────

    def release(self, context: str = "dequeue") -> None:
        """Settle a claim being retired without playing (failed to stream, failed
        to resolve). Warns instead of raising when nothing is claimed."""
        if not self.try_release():
            log.warning(f"song_queue was empty on {context} in guild {self._guild.id}")

    def try_release(self) -> bool:
        """Settle one claim: drop the head and step the cursor back, which leaves
        `len - _cursor` unchanged — the item leaves the queue, and what was pending
        stays pending.

        False means there was nothing claimed to settle, which is what a clear()
        during the resolve leaves behind. Guarded rather than raising, because
        driving _cursor negative would eat a PENDING item on the next release and
        break I1; the generation check in try_commit_dequeue() is what makes the
        no-op correct rather than merely safe. Use try_commit_dequeue() unless
        already holding the bulk-mutation lock: the check must not race a
        clear()/shuffle()."""
        if self._cursor == 0:
            return False
        self._items.popleft()
        self._cursor -= 1
        self._sync_wake()
        return True

    async def finish_failed_dequeue(
        self, item: Optional[QueueItem], *, context: str = "dequeue"
    ) -> None:
        """Retire one dequeued item that will never play: drop the display head,
        mirror the dequeue to Redis, task_done() the get() — the triplet every loop
        failure path shares. `context` labels the empty-display warning. The mutex
        spans the display pop and the LPOP so a bulk mutation can't rebuild the
        mirror between them and have the LPOP hit the new head."""
        async with self._mutex:
            self.release(context)
            await self.redis_pop_for(item)

    async def try_commit_dequeue(self, generation: int) -> bool:
        """Commit the display-side dequeue for a song about to play, under the
        bulk-mutation lock so the check can't race a clear()/shuffle(). False
        means the queue was cleared during the resolve and the caller discards
        the song — its task_done() and FFmpeg cleanup stay caller-side.

        `generation` is what the queue read when this dequeue took its item, and
        neither emptiness nor identity can stand in for it. A put() landing before
        the commit refills the display, so the head belongs to the NEW song and
        committing pops that entry while playing (and re-recording) the cleared one.
        And _resolve_source() replaces a YTSource with the QueueObject it resolved
        to, so head and item are legitimately different objects for one slot."""
        async with self._mutex:
            if generation != self._generation:
                return False
            return self.try_release()

    async def redis_pop_for(self, item: Optional[QueueItem]) -> None:
        """Mirror one in-memory dequeue to Redis via LPOP — unless the item was
        never on the list (persisted=False: the crash-recovered "current song",
        whose LPOP committed in the crashed run), where an LPOP would silently
        delete an unrelated, still-queued song. item=None is the prefetch path,
        which only ever dequeues real mirrored entries — so None pops."""
        if self._store is not None and is_persisted(item):
            await self._store.pop_queue()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _write_mirror(
        self, items: Sequence[QueueItem], *, removed: Sequence[QueueItem] = ()
    ) -> None:
        """Bring the Redis mirror in line with `items` — the persisted subset, in
        order — by whichever of three writes is right.

        Atomic rebuild (DELETE + RPUSH in MULTI; a plain pipeline leaves a window
        where a concurrent LPOP sees an empty queue). Callers hold the bulk-mutation
        mutex, so a concurrent put()'s pushes can't be wiped by a rebuild that
        predates them. persisted=False items were never RPUSHed — never write them in.

        Empty means DELETE, not skip: a queue whose every persisted entry just went
        would otherwise leave the old list behind for the next restore to find.

        `removed` is the LREM shortcut, and only a removal may pass it: it says the
        survivors kept their order, which is false for shuffle and for any insert.
        Worth it because a rebuild costs the same whether it drops one entry or two
        hundred — ~5.7ms at depth 1000, against ~0.96ms to LREM a single entry —
        and one song is what -remove does nearly every time.
        """
        if self._store is None:
            return
        entries = [_to_entry(s) for s in items if is_persisted(s)]
        if not entries:
            await self._store.delete_queue()
            return
        dropped = [_to_entry(s) for s in removed if is_persisted(s)]
        if dropped and len(dropped) <= _LREM_MAX_ENTRIES:
            await self._store.remove_queue_entries(dropped)
            return
        await self._store.rebuild_queue(entries)

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
