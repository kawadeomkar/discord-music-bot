"""
GuildQueue — all queue state and every queue operation for one guild.

This is the domain layer between the schema (src/guild_state.py — what queue
data *is* at rest) and the playback orchestration (src/musicplayer.py — when
queue operations happen). A guild's queue exists in three representations
that must never desync:

  _pending: asyncio.Queue   consumed by the playback loop
  _display: deque           ordered view for embeds / ETA math
  Redis guild:{id}:queue    persisted mirror (via GuildRedisStore)

Owning all three privately makes the sync invariant structural: nothing
outside this class can mutate one leg without the others. The class also owns
the bulk-mutation mutex and the cleared-flag the playback loop consumes.

What this class deliberately does NOT know (see docs/GUILD_QUEUE_SCHEMA_PLAN.md §2.2):
- stream prefetch (yt-dlp/FFmpeg) — MusicPlayer cancels its prefetch task
  BEFORE calling clear()/shuffle()/remove(), and the prefetch task consumes
  via the public get_nowait()/task_done()
- embed building and ETA math — they need playback state (current_song), so
  MusicPlayer builds them over display_items()/peek_next()
- the state hash — crash recovery hands this class ready-made queue entries
  (SongQueueEntry.from_crashed_state bridges the two schemas)
"""

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

    def task_done(self) -> None:
        self._pending.task_done()

    def empty(self) -> bool:
        return self._pending.empty()

    def qsize(self) -> int:
        return self._pending.qsize()

    @property
    def mutex(self) -> asyncio.Lock:
        """The bulk-mutation lock. Exposed for the playback loop's
        cleared-while-resolving discard race (it must pair a display-head pop
        with playback-side cleanup under the same lock the bulk mutations
        hold). Everything else should use the named operations."""
        return self._mutex

    def consume_cleared_flag(self) -> bool:
        """Read-and-reset the queue-was-cleared flag.

        clear() sets it under the mutex; the playback loop consumes it once at
        the top of each iteration to know a prefetched song it is holding was
        invalidated by the clear and must be discarded."""
        was_cleared = self._cleared
        self._cleared = False
        return was_cleared

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def put(self, items: Sequence[QueueItem], *, batch: bool = False) -> None:
        """Enqueue items on all three legs: in-memory puts for every item
        first, then the Redis mirror.

        batch=False RPUSHes one entry per round-trip (single-song enqueue,
        matching the interleaved per-item pushes it replaces); batch=True
        pushes everything in one round-trip (bulk playlist enqueue).
        """
        for item in items:
            await self._pending.put(item)
            self._display.append(item)
        if self._store is None:
            return
        entries = [
            _to_entry(item)
            for item in items
            if isinstance(item, (QueueObject, YTSource))
        ]
        if not entries:
            return
        if batch:
            await self._store.push_queue_batch(entries)
        else:
            for entry in entries:
                await self._store.push_queue(entry)

    # ── Bulk operations ───────────────────────────────────────────────────────
    # Callers with a prefetch task (MusicPlayer) must cancel it BEFORE any of
    # these — the prefetch may already hold an item from get_nowait(), and its
    # CancelledError handler's task_done() must land before the drain starts.

    async def clear(self) -> list[QueueItem]:
        """Drain all three legs. Returns the drained items (display order).

        Sets the cleared-flag under the mutex, before draining, so a playback
        loop iteration that wakes with a prefetched song in hand sees the flag
        and discards it.
        """
        async with self._mutex:
            self._cleared = True
            for _ in range(self._pending.qsize()):
                try:
                    self._pending.get_nowait()
                    self._pending.task_done()
                except asyncio.QueueEmpty:
                    break
            cleared_items = list(self._display)
            self._display.clear()
        if self._store is not None:
            await self._store.delete_queue()
        return cleared_items

    async def shuffle(self) -> ShuffleOutcome:
        """Shuffle the pending items in place (drain → shuffle → refill under
        one continuous mutex hold, so the loop can never observe a
        mid-shuffle empty queue). Requires at least 4 queued items."""
        if self._pending.qsize() < 4:
            return ShuffleOutcome.TOO_FEW_SONGS

        shuffled: list[QueueItem] = []
        async with self._mutex:
            for _ in range(self._pending.qsize()):
                try:
                    song = self._pending.get_nowait()
                    self._pending.task_done()
                    shuffled.append(song)
                except asyncio.QueueEmpty:
                    break
            random.shuffle(shuffled)
            kept: list[QueueItem] = []
            for song in shuffled:
                try:
                    self._pending.put_nowait(song)
                    kept.append(song)
                except asyncio.QueueFull:
                    break
            self._display = deque(kept)

        # Rebuild the Redis mirror atomically (DELETE + RPUSH inside MULTI —
        # a plain pipeline would leave a window where a concurrent LPOP sees
        # an empty queue). persisted=False items (the crash-recovered
        # "current song") were never RPUSHed to Redis — never write them in.
        if self._store is not None and kept:
            entries = [
                _to_entry(s)
                for s in kept
                if isinstance(s, (QueueObject, YTSource))
                and getattr(s, "persisted", True)
            ]
            if entries:
                await self._store.rebuild_queue(entries)

        return ShuffleOutcome.SHUFFLED

    async def remove(self, url: str) -> list[int]:
        """Remove every queued item whose webpage_url (QueueObject) or url
        (YTSource) matches. Returns the removed items' 1-indexed positions."""
        removed_positions: list[int] = []
        kept: list[QueueItem] = []

        async with self._mutex:
            # Drain everything first so positions are numbered before partitioning.
            drained: list[QueueItem] = []
            for _ in range(self._pending.qsize()):
                try:
                    item = self._pending.get_nowait()
                    self._pending.task_done()
                    drained.append(item)
                except asyncio.QueueEmpty:
                    break

            for pos, item in enumerate(drained, start=1):
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
            self._display = deque(kept)

        if removed_positions and self._store is not None:
            entries = [
                _to_entry(s)
                for s in kept
                if isinstance(s, (QueueObject, YTSource))
                and getattr(s, "persisted", True)
            ]
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
        """Re-queue the crash-recovered "current song" at the front of the
        line. In-memory legs only: the entry is persisted=False — its LPOP
        already committed, so it is not on the Redis list and the loop must
        not LPOP for it (see redis_pop_for).

        requester_fallback is the crashed path's resolution chain (guild.me
        or guild.owner) applied when the persisted requester ID no longer
        resolves. Returns False (nothing enqueued) when nobody resolves —
        the caller still owns clearing the crashed-song state.
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
        """Pop the display head for a song about to play. Returns False when
        the display is empty — meaning the queue was cleared while the song
        was resolving, and the caller must discard it. Call under `mutex` so
        the check cannot race a concurrent clear()/shuffle()."""
        try:
            self._display.popleft()
            return True
        except IndexError:
            return False

    async def redis_pop_for(self, item: Optional[QueueItem]) -> None:
        """Mirror one in-memory dequeue to Redis via LPOP — unless the item
        was never on the Redis list (persisted=False: the crash-recovered
        "current song", whose LPOP committed in the start transaction of the
        run that crashed). LPOPing for it here would silently delete an
        unrelated, still-queued song.

        item=None means the dequeue came through the prefetch path, where the
        original item is no longer in hand — prefetched items always came
        through get() on real, Redis-mirrored entries, so None pops."""
        if self._store is not None and getattr(item, "persisted", True):
            await self._store.pop_queue()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rehydrate(
        self,
        entry: QueueEntry,
        *,
        requester_fallback: Union[discord.Member, discord.User, None] = None,
    ) -> Optional[QueueItem]:
        """At-rest entry → live queue item — the one construction path for
        everything that comes back from Redis (pending entries and the
        crashed head alike).

        SongQueueEntry needs a requester resolved from the guild: the
        persisted member ID, else requester_fallback (defaults to
        guild.owner, the restored-entry chain), else the entry is dropped.
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
        )
