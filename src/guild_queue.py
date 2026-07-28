"""
GuildQueue — all queue state and every queue operation for one guild.

This is the domain layer between the schema (src/guild_state.py — what queue
data *is* at rest) and the playback orchestration (src/musicplayer.py — when
queue operations happen). A guild's queue exists in three representations
that must never desync:

  _pending: asyncio.Queue   consumed by the playback loop
  _display: deque           ordered view for embeds / ETA math
  Redis guild:{id}:queue    persisted mirror (via GuildRedisStore)

Owning all three privately makes the sync invariant structural: nothing outside
this class can mutate one leg without the others, and every mirror-touching
mutation (put, put_front, clear, shuffle, remove, finish_failed_dequeue) runs
under one bulk-mutation mutex, so none can interleave between a memory write and
its mirror write. Bulk mutations carry a dequeued-but-uncommitted head through
untouched (_in_flight_head), so a shuffle/remove during a multi-second resolve
can't retire the wrong entry. One residual window remains — see the header below.
The class also owns the cleared-flag the playback loop consumes.

Deliberately NOT known here:
- stream prefetch — MusicPlayer cancels its prefetch task BEFORE
  clear()/shuffle()/remove(); the task consumes via get_nowait()/task_done()
- embeds and ETA math — they need current_song, so MusicPlayer builds them over
  display_items()/peek_next()
- the state hash — crash recovery hands this class ready-made entries
  (SongQueueEntry.from_crashed_state bridges the two schemas)
"""

# ISSUE: Close the queue-desync race between dequeue commit and the Redis LPOP.
# try_commit_dequeue() → pop_queue_and_start_song() releases the bulk mutex before the
# store's atomic LPOP+HSET dispatches (a store-level atomicity boundary —
# so a bulk mutation scheduled in that tick races the LPOP
# server-side: a -clear landing between commit and dispatch pops an entry the clear
# already deleted, drifting memory and Redis by one until the next rebuild — and a crash
# inside that drift restores the queue one song out of alignment. Accepted, never fixed.
# Cheapest fix: hold the mutex across the store dispatch, costing one ~1ms round-trip.

import asyncio
import random
from collections import deque
from collections.abc import Sequence
from enum import Enum, auto
from typing import Optional, Union

import discord

from src.guild_state import QueueEntry, SearchQueueEntry, SongQueueEntry
from src.redis_client import GuildRedisStore
from src.sources import YTSource
from src.util import get_logger
from src.youtube import QueueObject

log = get_logger(__name__)

# Live queue items — what the in-memory legs hold. The at-rest twin is
# guild_state.QueueEntry; this class converts between the two internally.
QueueItem = Union[QueueObject, YTSource]


class ShuffleOutcome(Enum):
    SHUFFLED = auto()
    TOO_FEW_SONGS = auto()  # fewer than 4 queued items — nothing was mutated


def _to_entry(item: QueueItem) -> QueueEntry:
    """Live queue item → at-rest entry for the Redis mirror."""
    if isinstance(item, QueueObject):
        return SongQueueEntry.from_queue_object(item)
    return SearchQueueEntry.from_ytsource(item)


def is_persisted(item: Optional[QueueItem]) -> bool:
    """True when the item has a matching entry on the Redis queue list — i.e. its
    dequeue must be mirrored with an LPOP, and rebuilds may write it back.

    Only the crash-recovered "current song" carries persisted=False (its LPOP
    committed in the run that crashed); YTSource search entries are always
    persisted. None counts as persisted — see redis_pop_for."""
    if isinstance(item, QueueObject):
        return item.persisted
    return True


class GuildQueue:
    """Every queue operation the bot can perform, in one place.

    All methods that touch the Redis mirror degrade gracefully when the store
    is None (no Redis configured) or the store call fails (GuildRedisStore
    logs and never raises) — the in-memory queue keeps working either way.
    """

    __slots__ = ("_guild", "_store", "_pending", "_display", "_mutex", "_cleared")

    def __init__(self, guild: discord.Guild, store: Optional[GuildRedisStore]) -> None:
        self._guild = guild
        self._store = store
        self._pending: asyncio.Queue = asyncio.Queue()
        self._display: deque = deque()
        self._mutex = asyncio.Lock()
        self._cleared = False

    # ── Consumption (playback loop + prefetch task) ───────────────────────────

    async def get(self) -> QueueItem:
        return await self._pending.get()

    def get_nowait(self) -> QueueItem:
        """Raises asyncio.QueueEmpty — the prefetch task relies on that."""
        return self._pending.get_nowait()

    def requeue_front(self, item: QueueItem) -> None:
        """Undo a get()/get_nowait() whose consumer abandoned the item without
        playing it (prefetch cancellation). The display and Redis legs never
        moved, so the pending leg is the only undo needed — afterwards all three
        agree and a bulk mutation sees no in-flight head.

        The abandoned get()'s task slot transfers to the re-put: callers must NOT
        also call task_done(). `item` may be the resolved form of what was
        dequeued (YTSource → QueueObject); the other legs still hold the
        original, which is fine — counts stay aligned and they converge at the
        next dequeue.

        Drain + re-put, O(n) at command frequency, so blocked getters are woken
        normally by put_nowait.
        """
        # Balance the abandoned get() first; the put_nowait below re-increments
        # the task counter, transferring the slot to the future consumer.
        self._pending.task_done()
        rest: list[QueueItem] = []
        while True:
            try:
                rest.append(self._pending.get_nowait())
                self._pending.task_done()
            except asyncio.QueueEmpty:
                break
        self._pending.put_nowait(item)
        for r in rest:
            self._pending.put_nowait(r)

    def task_done(self) -> None:
        self._pending.task_done()

    def empty(self) -> bool:
        return self._pending.empty()

    def qsize(self) -> int:
        return self._pending.qsize()

    def _drain_pending(self) -> list[QueueItem]:
        """Remove and return every item in _pending, in queue order, each balanced
        with task_done(). The shared first step of put_front(), clear(), shuffle()
        and remove(). MUST be called holding _mutex — the drain must not race a
        concurrent consumer.
        """
        drained: list[QueueItem] = []
        for _ in range(self._pending.qsize()):
            try:
                drained.append(self._pending.get_nowait())
                self._pending.task_done()
            except asyncio.QueueEmpty:
                break
        return drained

    def consume_cleared_flag(self) -> bool:
        """Read-and-reset the queue-was-cleared flag. clear() sets it under the
        mutex; the loop consumes it once per iteration to learn that a prefetched
        song it is holding was invalidated and must be discarded."""
        was_cleared = self._cleared
        self._cleared = False
        return was_cleared

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def put(self, items: Sequence[QueueItem], *, batch: bool = False) -> None:
        """Enqueue items on all three legs: in-memory puts first, then the mirror.

        Under the bulk-mutation mutex because the Redis pushes suspend: a
        clear()/shuffle() interleaved there would drain/rebuild the mirror before
        the pushes land, resurrecting the new entries as ghosts the next dequeue
        would LPOP instead of its own.

        batch=False RPUSHes one entry per round-trip (single-song enqueue);
        batch=True pushes everything in one (bulk playlist enqueue).
        """
        async with self._mutex:
            for item in items:
                await self._pending.put(item)
                self._display.append(item)
            if self._store is None:
                return
            entries = [_to_entry(item) for item in items]
            if not entries:
                return
            if batch:
                await self._store.push_queue_batch(entries)
            else:
                for entry in entries:
                    await self._store.push_queue(entry)

    async def put_front(self, items: Sequence[QueueItem]) -> None:
        """Insert items at the FRONT of the line on all three legs — the -playnow
        interjection path.

        Runs under the bulk-mutation mutex like every other multi-leg mutation. An
        in-flight head (a dequeued-but-uncommitted item, e.g. a completed prefetch
        awaiting commit) keeps its position AHEAD of the inserted items on the
        display leg and forces the mirror down the rebuild path: its Redis entry is
        still at the list head awaiting a commit-time LPOP, so an LPUSH in front of
        it would make that LPOP eat the new head.

        The in-flight branch is LOAD-BEARING, not defensive — do not remove it as
        dead code. MusicPlayer.interject() neutralizes the prefetch first, but one
        interleaving still reaches here with a real in-flight head: the song ends
        naturally, the loop claims a still-running prefetch task and awaits it (up
        to yt-dlp's socket timeout), and interject() runs inside that await — its
        neutralize finds no task to take while the prefetch's dequeued item sits
        uncommitted at the display head.
        """
        if not items:
            return
        new_items = list(items)
        async with self._mutex:
            drained = self._drain_pending()
            in_flight = self._in_flight_head(drained_count=len(drained))
            for item in new_items + drained:
                self._pending.put_nowait(item)
            self._display = deque(in_flight + new_items + drained)

            if self._store is None:
                return
            if in_flight:
                entries = [
                    _to_entry(s)
                    for s in in_flight + new_items + drained
                    if is_persisted(s)
                ]
                if entries:
                    await self._store.rebuild_queue(entries)
                else:
                    await self._store.delete_queue()
            else:
                entries = [_to_entry(s) for s in new_items if is_persisted(s)]
                if entries:
                    await self._store.push_queue_front(entries)

    # ── Bulk operations ───────────────────────────────────────────────────────
    # Callers with a prefetch task (MusicPlayer) must cancel it BEFORE any of
    # these: a running prefetch holds an item from get_nowait(), and its
    # CancelledError handler's requeue_front() must land before the drain so the
    # item is mutated with the rest instead of stranded. An already-completed
    # prefetch is fine — its item is an in-flight head that _in_flight_head()
    # carries through.

    async def clear(self) -> list[QueueItem]:
        """Drain all three legs. Returns the drained items (display order).

        Sets the cleared-flag under the mutex before draining, so a loop iteration
        waking with a prefetched song discards it. The DEL is under the mutex too:
        released early, a concurrent put() could land its writes between the drain
        and the DEL and have its mirror entries wiped.
        """
        async with self._mutex:
            self._cleared = True
            self._drain_pending()
            cleared_items = list(self._display)
            self._display.clear()
            if self._store is not None:
                await self._store.delete_queue()
        return cleared_items

    async def shuffle(self) -> ShuffleOutcome:
        """Shuffle the pending items in place (drain → shuffle → refill under one
        continuous mutex hold, so the loop never observes a mid-shuffle empty
        queue). Requires at least 4 queued items.

        An in-flight dequeue (see _in_flight_head) keeps its display/Redis head
        position: shuffling only reorders items still in _pending.
        """
        # FIXME: -shuffle requires 4 queued songs but tells the user it needs 3.
        # The threshold here is 4, while MusicPlayer.queue_shuffle() refuses with
        # "at least 3 songs" and -help advertises "(3+ songs)". A user with exactly
        # 3 queued is refused by a message stating a requirement they have met.
        # Fix: drop this to < 3, or correct both user-facing strings to 4.
        if self._pending.qsize() < 4:
            return ShuffleOutcome.TOO_FEW_SONGS

        async with self._mutex:
            shuffled = self._drain_pending()
            in_flight = self._in_flight_head(drained_count=len(shuffled))
            random.shuffle(shuffled)
            kept: list[QueueItem] = []
            for song in shuffled:
                try:
                    self._pending.put_nowait(song)
                    kept.append(song)
                except asyncio.QueueFull:
                    break
            self._display = deque(in_flight + kept)

            # Rebuild atomically (DELETE + RPUSH inside MULTI — a plain pipeline
            # leaves a window where a concurrent LPOP sees an empty queue), still
            # under the mutex so a concurrent put()'s pushes can't be wiped by a
            # rebuild that predates them. persisted=False items were never RPUSHed
            # — never write them in.
            if self._store is not None and kept:
                entries = [_to_entry(s) for s in in_flight + kept if is_persisted(s)]
                if entries:
                    await self._store.rebuild_queue(entries)

        return ShuffleOutcome.SHUFFLED

    async def remove(self, url: str) -> list[int]:
        """Remove every queued item whose webpage_url (QueueObject) or url
        (YTSource) matches. Returns 1-indexed positions as the queue embed shows
        them.

        An in-flight dequeue is never removed even on a URL match — it is already
        committed to play and stopping it is -skip's job — but it does occupy a
        display position, hence the numbering offset.
        """
        removed_positions: list[int] = []
        kept: list[QueueItem] = []

        async with self._mutex:
            # Drain everything first so positions are numbered before partitioning.
            drained = self._drain_pending()
            in_flight = self._in_flight_head(drained_count=len(drained))

            for pos, item in enumerate(drained, start=1 + len(in_flight)):
                if isinstance(item, QueueObject):
                    match = item.webpage_url == url
                else:
                    match = (item.url or "") == url
                if match:
                    removed_positions.append(pos)
                else:
                    kept.append(item)

            for item in kept:
                self._pending.put_nowait(item)
            self._display = deque(in_flight + kept)

            if removed_positions and self._store is not None:
                entries = [_to_entry(s) for s in in_flight + kept if is_persisted(s)]
                if entries:
                    await self._store.rebuild_queue(entries)
                else:
                    await self._store.delete_queue()

        return removed_positions

    # ── Crash recovery ────────────────────────────────────────────────────────

    async def restore_crashed(
        self,
        entry: SongQueueEntry,
        *,
        requester_fallback: Union[discord.Member, discord.User, None],
    ) -> bool:
        """Re-queue the crash-recovered "current song" at the front of the line.

        In-memory legs only: the entry is persisted=False — its LPOP already
        committed, so it is not on the Redis list and the loop must not LPOP for
        it (see redis_pop_for).

        requester_fallback (guild.me or guild.owner) applies when the persisted
        requester ID no longer resolves. Returns False when nobody resolves; the
        caller still owns clearing the crashed-song state.
        """
        item = self._rehydrate(entry, requester_fallback=requester_fallback)
        if item is None:
            return False
        await self._pending.put(item)
        self._display.append(item)
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
                await self._pending.put(item)
                self._display.append(item)
                count += 1
        return count

    # ── Display data (embed/ETA builders live in MusicPlayer) ─────────────────

    def display_items(self) -> list[QueueItem]:
        """Snapshot of the queued items in display order."""
        return list(self._display)

    def peek_next(self) -> Optional[QueueItem]:
        return self._display[0] if self._display else None

    # ── Playback-loop dequeue bookkeeping ─────────────────────────────────────

    def pop_display_head(self, context: str = "dequeue") -> None:
        """Drop the display head for a dequeue that is being retired without
        playing (failed to stream / failed to resolve). Warns instead of
        raising when the display is already empty."""
        try:
            self._display.popleft()
        except IndexError:
            log.warning(f"song_queue was empty on {context} in guild {self._guild.id}")

    def try_pop_display_head(self) -> bool:
        """Pop the display head for a song about to play. False means the display
        was empty — the queue was cleared during the resolve and the caller must
        discard the song. Use try_commit_dequeue() unless already holding the
        bulk-mutation lock: the check must not race a clear()/shuffle()."""
        try:
            self._display.popleft()
            return True
        except IndexError:
            return False

    async def finish_failed_dequeue(
        self, item: Optional[QueueItem], *, context: str = "dequeue"
    ) -> None:
        """Retire one dequeued item that will never play (stream returned nothing
        / resolve raised): drop the display head, mirror the dequeue to Redis, and
        balance the get() with task_done() — the triplet every loop failure path
        shares. `context` labels the empty-display warning.

        Holds the mutex across the display pop and the LPOP so a bulk mutation
        can't rebuild the mirror between them and have the LPOP hit the new head."""
        async with self._mutex:
            self.pop_display_head(context)
            await self.redis_pop_for(item)
        self._pending.task_done()

    async def try_commit_dequeue(self) -> bool:
        """Commit the display-side dequeue for a song about to play.

        Takes the bulk-mutation lock so the emptiness check can't race a
        clear()/shuffle(). False means the queue was cleared during the resolve
        and the caller discards the song — its task_done() and FFmpeg cleanup are
        playback concerns and stay caller-side."""
        async with self._mutex:
            return self.try_pop_display_head()

    async def redis_pop_for(self, item: Optional[QueueItem]) -> None:
        """Mirror one in-memory dequeue to Redis via LPOP — unless the item was
        never on the list (persisted=False: the crash-recovered "current song",
        whose LPOP committed in the crashed run). LPOPing for it would silently
        delete an unrelated, still-queued song.

        item=None means the dequeue came through the prefetch path, which always
        dequeues real Redis-mirrored entries — so None pops."""
        if self._store is not None and is_persisted(item):
            await self._store.pop_queue()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _in_flight_head(self, *, drained_count: int) -> list[QueueItem]:
        """The dequeued-but-uncommitted items sitting at the display head.

        A consumer (the loop mid-resolve, or a completed prefetch committing on
        the next iteration) pops _pending on get() but leaves its display entry
        until try_commit_dequeue()/finish_failed_dequeue(). In that window the
        display leads _pending by exactly those items — at most one in practice —
        and they sit at the head, since dequeues come off the front.

        Called under the mutex after draining _pending, so anything in the display
        beyond drained_count is in-flight. Bulk mutations MUST carry these through
        untouched on both legs, or the consumer's eventual display-pop and LPOP
        retire someone else's entry: the triad desyncs permanently and a queued
        song's persisted entry is lost.
        """
        extra = len(self._display) - drained_count
        return list(self._display)[:extra] if extra > 0 else []

    def _rehydrate(
        self,
        entry: QueueEntry,
        *,
        requester_fallback: Union[discord.Member, discord.User, None] = None,
    ) -> Optional[QueueItem]:
        """At-rest entry → live queue item: the one construction path for
        everything coming back from Redis (pending entries and the crashed head).

        SongQueueEntry needs a requester resolved from the guild: the persisted
        member ID, else requester_fallback (default guild.owner), else dropped.
        """
        if isinstance(entry, SearchQueueEntry):
            return YTSource(
                ytsearch=entry.ytsearch,
                url=entry.url,
                process=entry.process,
                ts=entry.ts,
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
        )
