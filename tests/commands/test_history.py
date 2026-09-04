"""Tests for `-history` (src/commands/history.py)."""

from src.musicplayer import MusicPlayer
from types import SimpleNamespace
import inspect
import re
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from discord.ext import commands

from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.commands.history import HISTORY_MAX_LIMIT, HistoryFlags
from src.musicbot import (
    MusicBot,
)
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from tests.helpers import (
    command_callback,
)


def _history_entries(n: int) -> list[HistoryEntry]:
    """n entries, oldest-first (the order GuildHistory stores them)."""
    return [
        HistoryEntry(
            title=f"Song {i}",
            webpage_url=f"https://yt.com/v={i}",
            duration_secs=200,
            played_secs=200,
            requester_id=i + 1,
            requester_name=f"user{i}",
            played_at=1000.0 + i,
        )
        for i in range(n)
    ]


def _flags(limit: int = 10) -> SimpleNamespace:
    """Stand-in for a parsed HistoryFlags (FlagConverter can't be constructed
    directly; the command body only reads .limit)."""
    return SimpleNamespace(limit=limit)


class TestHistoryCommand:
    def _mp_with_history(self, music_bot: MusicBot, entries: Any) -> MagicMock:
        mp = MagicMock()
        history = GuildHistory(None, on_outbox_push=lambda: None)
        # No store, so the in-memory deque is the whole read path — these tests are
        # about rendering (ordering, chunking, limits, fields), not which leg served
        # them; that is TestHistoryReadsRedis, where the legs DISAGREE. Beware the
        # trap the old Postgres double hit: recent() swallows every error, so a
        # broken fixture reads exactly like graceful degradation and all 14 tests
        # passed against the cache while covering none of the read path.
        history.restore(list(reversed(entries)))
        mp.history = history
        music_bot.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_empty_history_sends_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._mp_with_history(music_bot, [])
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "No songs have been played yet" in embed.description

    async def test_shows_most_recent_newest_first(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._mp_with_history(music_bot, _history_entries(15))
        await command_callback(MusicBot.history)(
            music_bot, mock_ctx, flags=_flags(limit=3)
        )
        mock_ctx.send.assert_awaited_once()
        embeds = mock_ctx.send.call_args[1]["embeds"]
        # Most recent 3 of 15, newest first — not the oldest 3.
        assert [e.title for e in embeds] == [
            "1. Song 14",
            "2. Song 13",
            "3. Song 12",
        ]

    async def test_default_limit_chunks_at_eight_embeds(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # 10 embeds + the ≤2-embed NP block must stay under Discord's 10-embed
        # cap, so the response is chunked 8 + 2, every chunk via ctx.send.
        self._mp_with_history(music_bot, _history_entries(12))
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        assert mock_ctx.send.await_count == 2
        first, second = mock_ctx.send.await_args_list
        assert len(first.kwargs["embeds"]) == 8
        assert len(second.kwargs["embeds"]) == 2

    async def test_limit_smaller_than_history_returns_that_many(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._mp_with_history(music_bot, _history_entries(5))
        await command_callback(MusicBot.history)(
            music_bot, mock_ctx, flags=_flags(limit=50)
        )
        embeds = mock_ctx.send.call_args[1]["embeds"]
        assert len(embeds) == 5

    @pytest.mark.parametrize("bad_limit", [0, -3, 51])
    async def test_out_of_range_limit_rejected(
        self, music_bot: MusicBot, mock_ctx: MagicMock, bad_limit: int
    ) -> None:
        # A plain double, not _mp_with_history: the point here is that recent() is
        # never reached, and the real GuildHistory has __slots__ so it cannot be spied.
        mp = MagicMock()
        mp.history.recent = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.history)(
            music_bot, mock_ctx, flags=_flags(limit=bad_limit)
        )

        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "--limit must be between 1 and 50" in embed.description
        # The read, not get_mp: the wrapper resolves the player for every path and
        # cog_before_invoke had already built it anyway. What the range check
        # actually saves is the Redis round trip behind recent().
        mp.history.recent.assert_not_awaited()

    async def test_song_embeds_carry_thumbnail_and_metadata(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        entry = HistoryEntry(
            title="Rich Song",
            webpage_url="https://yt.com/v=rich",
            duration_secs=242,
            played_secs=225,
            requester_id=42,
            requester_name="Omkar",
            thumbnail="https://i.ytimg.com/t.jpg",
            played_at=1752530000.0,
        )
        self._mp_with_history(music_bot, [entry])
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embeds"][0]
        assert embed.thumbnail.url == "https://i.ytimg.com/t.jpg"
        lines = embed.description.splitlines()
        assert lines[0] == "https://yt.com/v=rich"
        assert lines[1] == "3:45 / 4:02 · requested by <@42> · <t:1752530000:f>"

    def test_flag_defaults(self) -> None:
        # -h with no flags must parse to limit=10.
        assert HistoryFlags.get_flags()["limit"].default == 10

    def test_max_limit_never_exceeds_the_redis_window(self) -> None:
        """The merge's completeness argument, pinned rather than commented: the
        union holds the true newest `limit` only while the Redis window is at least
        as deep as anything this command accepts. Raising HISTORY_MAX_LIMIT alone
        raises nothing — it silently starts returning short pages."""
        assert HISTORY_MAX_LIMIT <= HISTORY_CACHE_LIMIT

    @pytest.mark.parametrize(
        "name",
        [
            "history",
            "ping",
            "leaderboard",
            "analytics",
            "debug",
            "resume",
            "shuffle",
            "clear",
            "remove",
        ],
    )
    def test_the_command_is_capped_at_one_render_per_guild(self, name: str) -> None:
        """`-history` is the heaviest send in the bot (up to 8 song embeds plus the
        NP block), so unbounded concurrent renders rate-limit a guild out of its own
        channel — and deleting the decorator that prevents it left the suite green.
        `wait=False` is half the point: queueing the extra invocations still issues
        every send, so they must be declined outright. `-leaderboard` carries it for
        a second reason: it draws on the same Postgres pool as the drainer, and
        `-debug` for a third: a Postgres stats query, a Prometheus round trip and
        two Redis reads, live-editing under an 8s deadline. `-resume` for a fourth:
        two racing on a disconnected bot both read `voice_client is None`, so
        validate_commands' "already being used in channel X" check cannot fire for
        either — both join, and the second MOVES the bot to its own author's channel.

        `-shuffle`/`-clear`/`-remove` for a fifth: all three park on the queue's
        bulk mutex, which the playback loop holds across the start transaction. A
        Redis that stalls there wedges them, and every repeat while wedged parks
        another coroutine holding an OTel span `cog_after_invoke` never closes —
        plus, for `-shuffle`, a typing keepalive POSTing for the duration.

        command_callback() strips decorators everywhere else in this file, so this
        is the only place any of these guards is reachable at all."""
        guard = getattr(MusicBot, name)._max_concurrency
        assert guard is not None
        assert guard.number == 1
        assert guard.per is commands.BucketType.guild
        assert guard.wait is False

    def test_the_shuffle_copy_and_the_refusal_quote_the_same_number(self) -> None:
        """The FIXME this closed was exactly this drift: the code refused at one
        number while -help promised another, so a user meeting the stated
        requirement was turned away. Nothing else reads both strings."""
        help_text = MusicBot.shuffle.help
        assert help_text is not None
        promised = re.search(r"at least (\d+) ", help_text)
        refused = re.search(
            r"There must be at least (\d+) songs",
            inspect.getsource(MusicPlayer.queue_shuffle),
        )
        assert promised is not None and refused is not None
        assert promised.group(1) == refused.group(1)

    def test_help_copy_states_the_real_retention_window(self) -> None:
        """The user-facing copy must name the window the command actually keeps: 50
        is the retention cap AND the display cap, and the copy once promised
        permanent retention in the configuration that now ships by default. Pins
        that the constant stays interpolated, so raising the window cannot leave the
        copy quoting the old one; the negative assertion names the false claim."""
        help_text = MusicBot.history.help
        assert help_text is not None
        assert str(HISTORY_MAX_LIMIT) in help_text
        assert "permanently" not in help_text
        # The archive caveat belongs in NOTES, and it is the one place the word
        # is honest: Postgres retention really is permanent when enabled.
        note = (MusicBot.history.extras or {}).get("note", "")
        assert "permanently" in note


class TestHistoryReadsRedis:
    """That `-history` renders the Redis leg at all, asserted at the command level.

    Every failure inside recent() degrades to the leg below by design, so a command
    test can render a perfectly correct embed off the in-memory deque while the Redis
    read raises on every invocation. Making the legs DISAGREE is the only way to tell
    which one reached the user. Postgres is not on this read path at all."""

    async def test_the_rendered_songs_come_from_redis_not_just_the_cache(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Any
    ) -> None:

        store = GuildRedisStore(fake_redis, guild_id=1)
        stored = _history_entries(2)  # Song 0 (t=1000), Song 1 (t=1001)
        for entry in stored:
            await store.push_history(entry)
        # Older than both, and present only in the deque — so its position in
        # the output says which legs ran: absent means the Redis leg never got
        # read, first means the cache won, last means both legs merged and
        # sorted, which is the contract.
        cache_only = HistoryEntry(
            title="CACHE ONLY",
            webpage_url="https://yt.com/v=cache",
            duration_secs=1,
            played_secs=1,
            requester_id=1,
            requester_name="u",
            played_at=1.0,
        )
        mp = MagicMock()
        history = GuildHistory(store, on_outbox_push=lambda: None)
        history.restore([cache_only])
        mp.history = history
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        titles = [e.title for e in mock_ctx.send.call_args[1]["embeds"]]
        assert titles == ["1. Song 1", "2. Song 0", "3. CACHE ONLY"]

    async def test_a_broken_store_double_is_now_visible(
        self, music_bot: MusicBot, mock_ctx: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The guard on the guard: a stand-in without get_history stays green
        # forever, so this pins that the same mistake now leaves a visible warning
        # instead of a false claim that the read path is covered.
        mp = MagicMock()
        mp.history = GuildHistory(cast(Any, object()), on_outbox_push=lambda: None)
        mp.history.restore(_history_entries(1))
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.history)(music_bot, mock_ctx, flags=_flags())
        assert "redis read failed" in caplog.text
        assert "AttributeError" in caplog.text
