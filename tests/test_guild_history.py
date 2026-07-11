"""Tests for src/guild_history.py — the history domain class.

The property under test mirrors test_guild_queue's: after every operation the
in-memory ring and the Redis mirror agree, and both respect HISTORY_LIMIT.
"""

import pytest

from src.guild_history import GuildHistory
from src.redis_client import HISTORY_LIMIT, GuildRedisStore


@pytest.fixture
def store(fake_redis):
    return GuildRedisStore(fake_redis, guild_id=42)


class TestAdd:
    async def test_appends_and_mirrors_to_redis(self, store):
        h = GuildHistory(store)
        await h.add("Song A - urlA")
        await h.add("Song B - urlB")
        assert list(h) == ["Song A - urlA", "Song B - urlB"]  # oldest first
        assert await store.get_history() == ["Song B - urlB", "Song A - urlA"]

    async def test_works_without_store(self):
        h = GuildHistory(None)
        await h.add("Song A - urlA")
        assert list(h) == ["Song A - urlA"]

    async def test_both_legs_capped_at_limit(self, store):
        h = GuildHistory(store)
        for i in range(HISTORY_LIMIT + 5):
            await h.add(f"Song {i} - url{i}")
        assert len(h) == HISTORY_LIMIT
        mirrored = await store.get_history()
        assert len(mirrored) == HISTORY_LIMIT
        # Both legs kept the same newest entries (ring is oldest-first,
        # mirror is newest-first).
        assert list(h) == list(reversed(mirrored))


class TestRestore:
    def test_reverses_newest_first_input(self):
        h = GuildHistory(None)
        h.restore(["new", "mid", "old"])
        assert list(h) == ["old", "mid", "new"]

    def test_restore_respects_limit(self):
        h = GuildHistory(None)
        h.restore([f"e{i}" for i in range(HISTORY_LIMIT + 10)])
        assert len(h) == HISTORY_LIMIT
        assert h[-1] == "e0"  # newest entry survives the cap


class TestSequenceProtocol:
    def test_len_iter_getitem(self):
        # The -history command reads the ring as a plain sequence.
        h = GuildHistory(None)
        h.restore(["b", "a"])
        assert len(h) == 2
        assert h[0] == "a"
        assert list(h)[:10] == ["a", "b"]

    def test_empty_is_falsy(self):
        assert not GuildHistory(None)
