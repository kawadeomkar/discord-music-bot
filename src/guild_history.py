"""GuildHistory — one guild's played-song history: the Redis guild:{id}:history
list (capped and PERSISTed by push_history) and an in-memory deque, both bounded
at HISTORY_CACHE_LIMIT and only moved together. While the archive is enabled
add() also XADDs onto the outbox in the same pipeline and nudges the drainer;
Postgres is never awaited on the write path, and recent() never reads it.
See docs/ARCHITECTURE.md#history-read-path. The wire format belongs to
guild_state.py; this class never sees wire bytes."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Optional

import discord

from src.guild_state import HistoryEntry
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.util import fmt_duration, get_logger, truncate_embed_title

log = get_logger(__name__)

# Ceiling on the Redis read behind -history: GuildRedisStore swallows errors,
# but a connected-and-unresponsive server raises none and the pool sets no
# socket read timeout. Short, because the in-memory leg is one hop away.
_READ_TIMEOUT_SECS = 2.0


class GuildHistory:
    """Played songs, oldest-first, capped at HISTORY_CACHE_LIMIT on both legs.
    Iteration/len/indexing read the cache; mutation goes through add() and
    restore() only, so the Redis mirror cannot be skipped."""

    __slots__ = ("_store", "_entries", "_on_outbox_push")

    def __init__(
        self,
        store: Optional[GuildRedisStore],
        *,
        on_outbox_push: Optional[Callable[[], None]],
    ) -> None:
        # on_outbox_push is the drainer's sync notify; no default, because None
        # means the archive is disabled and every site must say so explicitly.
        self._store = store
        self._entries: deque[HistoryEntry] = deque(maxlen=HISTORY_CACHE_LIMIT)
        self._on_outbox_push = on_outbox_push

    async def add(self, entry: HistoryEntry) -> None:
        """Record one played song on every configured leg: the cache, the Redis
        list and (archive enabled) the outbox, the latter two in one pipeline.
        Degrades when the store is None or the push fails."""
        self._entries.append(entry)
        if self._store is not None:
            await self._store.push_history(entry)
            if self._on_outbox_push is not None:
                self._on_outbox_push()

    def restore(self, newest_first: Sequence[HistoryEntry]) -> None:
        """Populate the in-memory leg after a restart; the Redis list stores
        newest-first, the cache oldest-first, hence the reversal."""
        self._entries.extend(reversed(newest_first))

    async def recent(self, limit: int) -> list[HistoryEntry]:
        """The `limit` most recently played songs, newest first: the Redis list
        (bounded at _READ_TIMEOUT_SECS) MERGED with the deque — a second leg can
        only add depth, and a fallback would let one Redis row suppress the
        cache. Never Postgres: push_history LTRIMs the list to exactly the
        command's ceiling. See docs/ARCHITECTURE.md#history-read-path."""
        # Early-out only; merged[:limit] already handles 0 and negatives.
        if limit <= 0:
            return []
        merged: list[HistoryEntry] = []
        seen: set[HistoryEntry] = set()

        def take(entries: Sequence[HistoryEntry]) -> None:
            """Append what this leg adds. First leg to carry an entry wins."""
            for entry in entries:
                # The whole entry is the identity: a (played_at, url) key
                # collapses distinct plays whose played_at defaulted to 0.0.
                if entry not in seen:
                    seen.add(entry)
                    merged.append(entry)

        # A local: the `is not None` narrowing does not follow self._store into
        # the closure. A lambda, not the bound method: the attribute lookup must
        # happen inside _read_tier's guard.
        store = self._store
        if store is not None:
            take(await self._read_tier("redis", lambda: store.get_history()))
        take(list(reversed(self._entries)))
        # Both legs are in RECORDED order (song end); played_at is song start, and
        # an interjection-parked song is recorded after everything that cut in
        # front. Stable sort, so ties keep leg order.
        merged.sort(key=lambda e: e.played_at, reverse=True)
        return merged[:limit]

    async def _read_tier(
        self, name: str, read: Callable[[], Awaitable[list[HistoryEntry]]]
    ) -> list[HistoryEntry]:
        """One bounded, non-raising read of a persisted leg. Takes a thunk so the
        call itself is inside the guard."""
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
        Cache-only so it stays sync; restore() refills the cache after a restart."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]


def history_embeds(entries: list[HistoryEntry]) -> list[discord.Embed]:
    """One embed per played song, in the given order: numbered title, the raw
    webpage_url (Discord auto-links it), played/duration, requester and, when
    known, the played-at timestamp as viewer-local <t:…:f>."""
    embeds = []
    for i, entry in enumerate(entries, start=1):
        lines = []
        if entry.webpage_url:
            lines.append(entry.webpage_url)
        requested_by = (
            f"<@{entry.requester_id}>"
            if entry.requester_id
            else (entry.requester_name or "unknown")
        )
        meta = (
            f"{fmt_duration(entry.played_secs)} / {fmt_duration(entry.duration_secs)}"
            f" · requested by {requested_by}"
        )
        # 0 means unknown; <t:0:f> would render "1 January 1970".
        if entry.played_at:
            meta += f" · <t:{int(entry.played_at)}:f>"
        lines.append(meta)
        title = truncate_embed_title(f"{i}. {entry.title}")
        embed = discord.Embed(
            title=title,
            description="\n".join(lines),
            color=discord.Color.blue(),
        )
        if entry.thumbnail:
            embed.set_thumbnail(url=entry.thumbnail)
        embeds.append(embed)
    return embeds
