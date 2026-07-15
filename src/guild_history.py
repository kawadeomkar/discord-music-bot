"""
GuildHistory — a guild's played-song history for one guild.

The domain twin of GuildQueue, one layer smaller: two legs instead of three
(an in-memory ring of "<title> - <webpage_url>" strings, plus the Redis
guild:{id}:history list mirror via GuildRedisStore). Owning both privately
means the legs can only move together, and the retention invariant — at most
HISTORY_LIMIT entries — is enforced in one place on both legs (deque maxlen
here, LTRIM in the store).

The at-rest wire format is owned by guild_state.py
(serialize_history_entry/parse_history_entry); the store surface is
push_history/get_history. This class never sees wire bytes.
"""

from collections import deque
from collections.abc import Iterator, Sequence
from typing import Optional

from src.redis_client import HISTORY_LIMIT, GuildRedisStore


class GuildHistory:
    """Played songs, oldest-first, capped at HISTORY_LIMIT on both legs.

    Iteration/len/indexing are exposed directly (the -history command and its
    tests read the ring as a plain sequence); mutation goes through add() and
    restore() only, so the Redis mirror can't be skipped.
    """

    __slots__ = ("_store", "_entries")

    def __init__(self, store: Optional[GuildRedisStore]) -> None:
        self._store = store
        self._entries: deque[str] = deque(maxlen=HISTORY_LIMIT)

    async def add(self, entry: str) -> None:
        """Record one played song on both legs. Degrades gracefully when the
        store is None or the push fails (GuildRedisStore logs, never raises)."""
        self._entries.append(entry)
        if self._store is not None:
            await self._store.push_history(entry)

    def restore(self, newest_first: Sequence[str]) -> None:
        """Populate from persisted history after a restart. In-memory leg
        only — the entries came off the Redis list, which stores newest-first;
        the ring appends oldest-first, hence the reversal."""
        self._entries.extend(reversed(newest_first))

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> str:
        return self._entries[index]
