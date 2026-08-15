"""
GuildQueue — all queue state and every queue operation for one guild.

Domain layer between the schema (src/guild_state.py) and playback orchestration
(src/musicplayer.py). A guild's queue exists in three representations that must
never desync:

  _pending: asyncio.Queue   consumed by the playback loop
  _display: deque           ordered view for embeds / ETA math
  Redis guild:{id}:queue    persisted mirror (via GuildRedisStore)

All three legs are private, so no caller can mutate one without the others, and
every mirror-touching mutation (put, put_front, clear, shuffle, remove,
finish_failed_dequeue) holds one bulk-mutation mutex across its memory AND mirror
writes, carrying a dequeued-but-uncommitted head through untouched
(_in_flight_head). One residual window remains — see the ISSUE below. The
cleared-flag the playback loop consumes lives here too.

Not known here:
- stream prefetch — MusicPlayer cancels its prefetch task before
  clear()/shuffle()/remove(); the task consumes via get_nowait()/task_done()
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

# Above this many removed entries, rebuilding the whole list is cheaper than the
# per-LREM scans. Set below the measured crossover so queue depth, which moves it,
# has margin. See docs/ARCHITECTURE.md#queue-operations.
_LREM_MAX_ENTRIES = 200


class _FrontQueue(asyncio.Queue[QueueItem]):
    """asyncio.Queue plus head insertion.

    _queue is already a collections.deque, so this is put_nowait's bookkeeping
    (cpython asyncio/queues.py:156) run against appendleft. The full/shutdown
    checks are dropped: _pending is unbounded and never shut down. Only
    _init/_get/_put are sanctioned subclass hooks, so a CPython rename of the
    private names below breaks here, at first use.
    """

    def put_front_nowait(self, item: QueueItem) -> None:
        self._queue.appendleft(item)  # pyright: ignore[reportAttributeAccessIssue]
        self._unfinished_tasks += 1
        self._finished.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._wakeup_next(self._getters)  # pyright: ignore[reportAttributeAccessIssue]


class ShuffleOutcome(Enum):
    SHUFFLED = auto()
    TOO_FEW_SONGS = auto()  # fewer than 4 queued items — nothing was mutated


class RemoveMode(StrEnum):
    """How a removal matched, for the reply to explain itself with."""

    RESOLVED = "resolved"  # the yt-dlp URL, as the Now Playing card shows it
    ORIGIN = "origin"  # what the user typed: a search term, or a source link


# One queue item → the mode it matched under, or None. Built by remove_matcher().
RemoveMatcher = Callable[[QueueItem], Optional[RemoveMode]]


def _normalize(s: str) -> str:
    """Fold a needle for comparison: collapse whitespace, and casefold anything
    that is not a URL. Links keep their case — the ids inside them are
    case-sensitive, so folding one would match a different album."""
    s = " ".join(s.split())
    return s if s[:8].lower().startswith(("http://", "https:/")) else s.casefold()


def remove_matcher(needle: str) -> RemoveMatcher:
    """Match a queue item against one `-remove` argument: the resolved yt-dlp URL
    first, then what the user typed. An origin match takes out every item sharing
    it, so one album link removes the tracks it queued. Links compare literally —
    youtu.be/x does not match an entry stored as youtube.com/watch?v=x."""
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
    # ORIGIN when anything matched on what the user typed; None when nothing did.
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
        "_pending",
        "_display",
        "_mutex",
        "_cleared",
        "_generation",
    )

    def __init__(self, guild: discord.Guild, store: Optional[GuildRedisStore]) -> None:
        self._guild = guild
        self._store = store
        self._pending: _FrontQueue = _FrontQueue()
        self._display: deque = deque()
        self._mutex = asyncio.Lock()
        self._cleared = False
        self._generation = 0

    # ── Consumption (playback loop + prefetch task) ───────────────────────────

    async def get(self) -> QueueItem:
        return await self._pending.get()

    def get_nowait(self) -> QueueItem:
        """Raises asyncio.QueueEmpty — the prefetch task relies on that."""
        return self._pending.get_nowait()

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
        # Balance the abandoned get(); the head insert below re-increments the
        # task counter, transferring the slot to the future consumer.
        self._pending.task_done()
        self._pending.put_front_nowait(item)
        # Positional: this item is the head's resolved form, and the head is where
        # the abandoned get() took it from.
        if self._display:
            self._display[0] = item

    def task_done(self) -> None:
        self._pending.task_done()

    def empty(self) -> bool:
        return self._pending.empty()

    def qsize(self) -> int:
        return self._pending.qsize()

    def display_size(self) -> int:
        """Depth of the display leg — what a new arrival waits behind.

        Counts the in-flight head: an item dequeued from _pending but not yet
        committed, or one a prefetch took via get_nowait()."""
        return len(self._display)

    def _drain_pending(self) -> list[QueueItem]:
        """Remove and return every item in _pending, in queue order, each balanced
        with task_done(). Shared first step of clear()/shuffle()/remove(), which
        all inspect or reorder every item. Must hold _mutex — the drain must not
        race a consumer."""
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
            for item in queued:
                await self._pending.put(item)
                self._display.append(item)
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
            in_flight = self._in_flight_head(pending_count=self._pending.qsize())
            new_items = list(items)

            for item in reversed(new_items):
                self._pending.put_front_nowait(item)
            # Lift the in-flight head off, insert behind it, put it back. The tail
            # is carried over rather than recomputed from _pending, so a
            # pre-existing triad drift survives this insert rather than being healed.
            for _ in in_flight:
                self._display.popleft()
            self._display.extendleft(reversed(in_flight + new_items))

            if in_flight:
                await self._write_mirror(self._display)
            elif self._store is not None:
                # An LPUSH of just the new items, not a replacement: reachable
                # only with no in-flight head, where nothing ahead needs keeping.
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
        """Drain all three legs, returning the drained items in display order. Sets
        the cleared-flag under the mutex before draining, so a loop iteration
        holding a prefetched song discards it. The DEL is inside the mutex too:
        released early, a concurrent put()'s mirror writes would land between the
        drain and the DEL and be wiped."""
        async with self._mutex:
            self._cleared = True
            self._generation += 1
            self._drain_pending()
            cleared_items = list(self._display)
            self._display.clear()
            if self._store is not None:
                await self._store.delete_queue()
        return cleared_items

    async def shuffle(self) -> ShuffleOutcome:
        """Shuffle the pending items in place: drain → shuffle → refill under one
        continuous mutex hold, so the loop never sees a mid-shuffle empty queue.
        Requires 4+ queued items. An in-flight dequeue (see _in_flight_head) keeps
        its display/Redis head position — only _pending is reordered."""
        # FIXME: -shuffle requires 4 queued songs but tells the user it needs 3.
        # MusicPlayer.queue_shuffle() refuses with "at least 3 songs" and -help says
        # "(3+ songs)", so a user with exactly 3 queued is refused by a message
        # stating a requirement they have met. Fix: drop this to < 3, or correct
        # both user-facing strings to 4.
        if self._pending.qsize() < 4:
            return ShuffleOutcome.TOO_FEW_SONGS

        async with self._mutex:
            shuffled = self._drain_pending()
            in_flight = self._in_flight_head(pending_count=len(shuffled))
            random.shuffle(shuffled)
            kept: list[QueueItem] = []
            for song in shuffled:
                try:
                    self._pending.put_nowait(song)
                    kept.append(song)
                except asyncio.QueueFull:
                    break
            self._display = deque(in_flight + kept)

            # With no persisted survivor this DELETEs the mirror rather than
            # leaving it stale. Unreachable today: it needs 4+ crash-recovered
            # items, and restore_crashed makes one.
            if kept:
                await self._write_mirror(in_flight + kept)

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
            # Drain everything first so positions are numbered before partitioning.
            drained = self._drain_pending()
            in_flight = self._in_flight_head(pending_count=len(drained))

            for pos, item in enumerate(drained, start=1 + len(in_flight)):
                mode = match(item)
                if mode is not None:
                    removed_positions.append(pos)
                    removed_items.append(item)
                    modes.append(mode)
                else:
                    kept.append(item)

            for item in kept:
                self._pending.put_nowait(item)
            self._display = deque(in_flight + kept)

            if removed_positions:
                await self._write_mirror(in_flight + kept, removed=removed_items)

        return RemoveOutcome(
            removed=removed_items,
            positions=removed_positions,
            # ORIGIN if anywhere in the run: the argument reached items the
            # resolved URL alone would not have.
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

    def has_resume_tail(self, webpage_url: str) -> bool:
        """True when the display already carries the resume tail an interjection
        left behind for `webpage_url`. That entry and the live song are the SAME
        play, so anything counting queue depth must count them once.

        Matches on URL, so a tail parked by an earlier play of the same song
        answers True for the current one; enqueue_depth under-counts by one there.

        O(len(display)), early-exiting only when a tail exists: 90us at 1000
        entries, synchronous, at -play dispatch."""
        return any(
            isinstance(item, QueueObject)
            and item.is_resume
            and item.webpage_url == webpage_url
            for item in self._display
        )

    def resume_tail_depth(self) -> int:
        """How many parked plays are waiting behind the song that just cut the line
        — the run of consecutive resume tails after it. 1 is a plain -playnow, 2+ a
        stack. Counts PLAYS, not fragments: the interrupted song's live fragment is
        gone by the time this runs and only its tail is queued.

        The interjected song is not always at index 0 — put_front's in-flight-head
        branch keeps a dequeued-but-uncommitted item ahead of what it inserts — so
        the run starts after whatever _pending does not account for."""
        items = list(self._display)
        head = len(items) - self._pending.qsize()
        start = (head if head > 0 else 0) + 1
        depth = 0
        for item in items[start:]:
            if not (isinstance(item, QueueObject) and item.is_resume):
                break
            depth += 1
        return depth

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
        """Retire one dequeued item that will never play: drop the display head,
        mirror the dequeue to Redis, task_done() the get() — the triplet every loop
        failure path shares. `context` labels the empty-display warning. The mutex
        spans the display pop and the LPOP so a bulk mutation can't rebuild the
        mirror between them and have the LPOP hit the new head."""
        async with self._mutex:
            self.pop_display_head(context)
            await self.redis_pop_for(item)
        self._pending.task_done()

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
            return self.try_pop_display_head()

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
        order.

        The rebuild is DELETE + RPUSH in MULTI: a plain pipeline leaves a window
        where a concurrent LPOP sees an empty queue. Callers hold the bulk-mutation
        mutex, so a concurrent put()'s pushes can't be wiped by a rebuild that
        predates them. persisted=False items were never RPUSHed — never write them in.

        Empty means DELETE, not skip, or the old list survives for the next restore
        to find.

        Only a removal may pass `removed`: LREM assumes the survivors kept their
        order, which is false for shuffle and for any insert. A short LREM count
        means the mirror no longer held what it was asked to drop, so this falls
        through to the rebuild, which restates the whole list from memory.
        """
        if self._store is None:
            return
        entries = [_to_entry(s) for s in items if is_persisted(s)]
        if not entries:
            await self._store.delete_queue()
            return
        dropped = [_to_entry(s) for s in removed if is_persisted(s)]
        if dropped and len(dropped) <= _LREM_MAX_ENTRIES:
            if await self._store.remove_queue_entries(dropped) == len(dropped):
                return
            log.warning(
                f"queue mirror diverged from memory in guild {self._guild.id}; "
                "rebuilding instead of removing"
            )
        await self._store.rebuild_queue(entries)

    def _in_flight_head(self, *, pending_count: int) -> list[QueueItem]:
        """The dequeued-but-uncommitted items at the display head.

        A consumer (the loop mid-resolve, or a completed prefetch committing next
        iteration) pops _pending on get() but leaves its display entry until
        try_commit_dequeue()/finish_failed_dequeue(); in that window the display
        leads _pending by exactly those items — at most one in practice, at the
        head since dequeues come off the front. Called under the mutex with
        however many items _pending accounts for — just drained out of it
        (clear/shuffle/remove) or still sitting in it (put_front) — so anything
        the display holds beyond that count is in-flight.

        Bulk mutations must carry these through untouched on both legs, or the
        consumer's eventual display-pop and LPOP retire someone else's entry —
        permanent triad desync, and a queued song's persisted entry lost."""
        extra = len(self._display) - pending_count
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
