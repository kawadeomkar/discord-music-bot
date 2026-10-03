"""Tests for `-replay` (src/commands/replay.py)."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from redis.asyncio import Redis
from discord.ext import commands

from src import play_pipeline
from src.commands import play as play_cmd
from src.commands import replay as replay_cmd
from src.commands.replay import NOTHING_TO_REPLAY, last_played, replay_item
from src.guild_history import GuildHistory
from src.guild_state import HistoryEntry
from src.musicbot import MusicBot
from src.musicplayer import InterjectOutcome
from src.play_placement import PlaceStalled
from src.queue_item import QueueObject
from src.youtube import YTDL
from tests.helpers import (
    command_callback,
    connected_vc,
    mock_mp,
    passthrough_prefetch,
    paused_vc,
    playing_vc,
)

_A = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
_B = "https://www.youtube.com/watch?v=bbbbbbbbbbb"
_C = "https://www.youtube.com/watch?v=ccccccccccc"


def _entry(url: str, title: str, **fields: Any) -> HistoryEntry:
    return HistoryEntry(guild_id=1, title=title, webpage_url=url, **fields)


def _history(*oldest_first: HistoryEntry) -> GuildHistory:
    """A real GuildHistory, cache leg only: what the command reads."""
    history = GuildHistory(None, on_outbox_push=None)
    history.restore(list(reversed(oldest_first)))
    return history


class TestLastPlayed:
    def test_the_newest_entry_is_the_last_song_played(self) -> None:
        history = _history(_entry(_A, "A"), _entry(_B, "B"))
        assert (entry := last_played(history)) is not None
        assert entry.webpage_url == _B

    def test_the_live_song_is_passed_over(self) -> None:
        """A -restart or -skip of the live song records its fragment, so the newest
        entry can BE the live song. Replaying it would be -restart's job, and the
        song the user means is the one before it, however many fragments deep."""
        history = _history(_entry(_A, "A"), _entry(_B, "B"), _entry(_B, "B again"))
        assert (entry := last_played(history, skip_url=_B)) is not None
        assert entry.webpage_url == _A

    def test_an_entry_with_no_link_is_passed_over(self) -> None:
        """Zero-values mean unknown on a HistoryEntry, and an empty url has nothing
        for the stream cache or yt-dlp to open."""
        history = _history(_entry(_A, "A"), _entry("", "Lost"))
        assert (entry := last_played(history)) is not None
        assert entry.webpage_url == _A

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param(_history(), id="empty"),
            pytest.param(_history(_entry(_B, "B")), id="only-the-live-song"),
        ],
    )
    def test_nothing_to_replay(self, history: GuildHistory) -> None:
        assert last_played(history, skip_url=_B) is None


class TestReplayItem:
    def test_the_item_is_resolved_from_the_history_row(
        self, mock_author: MagicMock
    ) -> None:
        """Every display field comes off the row, so the item is playable as built:
        no search, no source-cache read, only the stream cache keyed by its url."""
        entry = _entry(
            _A,
            "Song A",
            duration_secs=215,
            uploader="Channel A",
            thumbnail="https://i.ytimg.com/a.jpg",
            query_source="open.spotify.com",
            requester_id=99,
            played_at=1752530000.0,
            played_secs=120,
            queue_position=4,
            queued_at=1752529000.0,
        )
        item = replay_item(entry, requester=mock_author, queued_at=1752530500.5)
        assert not item.unresolved
        assert (
            item.webpage_url,
            item.title,
            item.duration,
            item.uploader,
            item.thumbnail,
        ) == (_A, "Song A", 215, "Channel A", "https://i.ytimg.com/a.jpg")
        # How the song was found stays with it; webpage_url cannot rebuild it.
        assert item.query_source == "open.spotify.com"
        # -remove <link> matches it, as for a -play of the link.
        assert item.user_input == _A
        # A new ask, by this caller, from the top: nothing of the old play rides
        # along — not its requester, its stamps, its depth or how far it got.
        assert item.requester is mock_author
        assert (item.queued_at, item.queue_position) == (1752530500.5, 0)
        assert (item.ts, item.played_at) == (None, 0.0)
        assert not (item.is_resume or item.interjected or item.is_restart)

    def test_unknown_fields_stay_unknown(self, mock_author: MagicMock) -> None:
        """0 and "" mean unknown on the row; the item spells unknown as None, and a
        title-less row still renders as something."""
        item = replay_item(_entry(_A, ""), requester=mock_author, queued_at=0.0)
        assert (item.title, item.duration, item.uploader, item.thumbnail) == (
            _A,
            None,
            None,
            None,
        )


class TestReplayCommand:
    @pytest.fixture
    def mp(self, mock_ctx: MagicMock) -> MagicMock:
        """A player with A then B in history and B live, so the replay is A."""
        mp = mock_mp()
        mp.guild_id = mock_ctx.guild.id
        mp.history = _history(_entry(_A, "Song A", duration_secs=200), _entry(_B, "B"))
        mp.current_song = MagicMock()
        mp.current_song.webpage_url = _C
        mp.current_song.title = "Song C"
        mp.queue.claim_outstanding = MagicMock(return_value=False)
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Song C", resume_position=95, was_paused=False
            )
        )
        return mp

    @pytest.fixture(autouse=True)
    def wired(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> Iterator[AsyncMock]:
        """The player is the cog's, the voice client is playing in the author's
        channel, and the stream warm hands the item back. Nothing may resolve:
        queue_source and yt_source fail the test if reached."""
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.voice_client = playing_vc(mock_ctx)
        never = AsyncMock(side_effect=AssertionError("-replay must not resolve"))
        with (
            patch.object(YTDL, "prefetch_stream", new=passthrough_prefetch()) as warm,
            patch.object(play_pipeline, "queue_source", new=never),
            patch.object(YTDL, "yt_source", new=never),
        ):
            yield warm

    async def _replay(self, music_bot: MusicBot, mock_ctx: MagicMock) -> None:
        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

    def _embed(self, mock_ctx: MagicMock) -> discord.Embed:
        return mock_ctx.send.await_args.kwargs["embed"]

    async def test_a_live_song_is_interrupted_by_the_last_song_played(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        wired: AsyncMock,
    ) -> None:
        await self._replay(music_bot, mock_ctx)

        mp.interject.assert_awaited_once()
        assert (call := mp.interject.await_args) is not None
        played: QueueObject = call.args[0]
        assert (played.webpage_url, played.title) == (_B, "B")
        assert played.requester is mock_ctx.author
        assert played.queued_at == mock_ctx.message.created_at.timestamp()
        # The warm is the gate the interruption waits on, paid once, for this url.
        wired.assert_awaited_once()
        embed = self._embed(mock_ctx)
        assert embed.title == "🔁 Replaying: B"
        assert "**Song C** will resume at `1:35`." in (embed.description or "")
        # Registered as a -play and retired with the reply.
        assert not music_bot._plays._guilds

    async def test_the_live_songs_own_fragment_is_not_what_replays(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """B is live and its fragment is the newest entry (a -restart or -skip
        wrote it), so the song before it replays."""
        mp.current_song.webpage_url = _B

        await self._replay(music_bot, mock_ctx)

        assert (call := mp.interject.await_args) is not None
        assert call.args[0].webpage_url == _A

    async def test_a_paused_song_comes_back_playing(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """A paused song is live and is interrupted like a playing one; the command
        asked for music, so it returns playing, as under a plain -play."""
        mock_ctx.voice_client = paused_vc(mock_ctx)

        await self._replay(music_bot, mock_ctx)

        assert (call := mp.interject.await_args) is not None
        assert call.kwargs["resume_paused"] is False

    async def test_nothing_live_plays_it_next(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """Connected with nothing live, there is nothing to interrupt: the song
        front-inserts, and with an empty queue that is starting now."""
        mp.current_song = None
        mock_ctx.voice_client = connected_vc(mock_ctx)

        await self._replay(music_bot, mock_ctx)

        mp.interject.assert_not_awaited()
        mp.queue_put_next.assert_awaited_once()
        assert (call := mp.queue_put_next.await_args) is not None
        assert call.args[0].webpage_url == _B
        embed = self._embed(mock_ctx)
        assert embed.title == "▶️ Playing now: B"
        assert not music_bot._plays._guilds

    @pytest.mark.parametrize("join_running", [False, True], ids=["no-voice", "joining"])
    async def test_out_of_voice_it_goes_through_plays_cold_path(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        join_running: bool,
    ) -> None:
        """-play's cold path owns the join, the gate hold and the teardown when
        either fails; a join already running is a cold start too, since discord.py
        registers the client before the handshake lands."""
        if join_running:
            mock_ctx.voice_client = connected_vc(mock_ctx)
            music_bot._plays.join_in_flight = MagicMock(return_value=True)
        else:
            mock_ctx.voice_client = None
            # Nothing is live without a voice client, so the newest entry replays.
        with patch.object(play_cmd, "run", new=AsyncMock()) as cold:
            await self._replay(music_bot, mock_ctx)

        cold.assert_awaited_once_with(mock_ctx, _B, cog=music_bot)
        mp.interject.assert_not_awaited()
        mp.queue_put_next.assert_not_awaited()

    async def test_no_history_says_so_and_registers_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        mp.history = _history()

        await self._replay(music_bot, mock_ctx)

        embed = self._embed(mock_ctx)
        assert embed.description == NOTHING_TO_REPLAY
        assert embed.color == discord.Color.orange()
        mp.interject.assert_not_awaited()
        assert not music_bot._plays._guilds

    async def test_a_restore_still_running_is_waited_out_then_reported(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """A player this command built is still reading history from Redis: read
        before it lands, the cache is empty and "nothing to replay" would be false."""
        mp.wait_for_restore = AsyncMock(return_value=False)

        await self._replay(music_bot, mock_ctx)

        assert "Still loading" in (self._embed(mock_ctx).description or "")
        mp.interject.assert_not_awaited()

    async def test_a_song_with_no_playable_stream_leaves_the_live_one_alone(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        wired: AsyncMock,
    ) -> None:
        """The warm is a gate: a link that no longer plays (taken down since) must
        not stop the song that is playing."""
        wired.side_effect = None
        wired.return_value = None
        music_bot._command_error = AsyncMock()

        await self._replay(music_bot, mock_ctx)

        mp.interject.assert_not_awaited()
        music_bot._command_error.assert_awaited_once()
        assert (call := music_bot._command_error.await_args) is not None
        assert call.kwargs["title"] == "Failed to replay song"
        assert not music_bot._plays._guilds

    async def test_a_clear_during_the_warm_drops_it(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        wired: AsyncMock,
    ) -> None:
        """The request is a registered -play, so the place lock's checks apply: a
        -clear that bumps the generation while the stream warms drops it."""

        async def warm_then_clear(qo: QueueObject, *, redis: Any = None) -> QueueObject:
            mp.queue.generation += 1
            return qo

        wired.side_effect = warm_then_clear

        await self._replay(music_bot, mock_ctx)

        mp.interject.assert_not_awaited()
        assert "queue was cleared" in (self._embed(mock_ctx).description or "")

    async def test_a_stalled_place_is_reported_as_a_busy_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        with patch.object(
            play_pipeline,
            "interject_resolved",
            new=AsyncMock(side_effect=PlaceStalled(before_the_put=True)),
        ):
            await self._replay(music_bot, mock_ctx)

        assert "queue is busy" in (self._embed(mock_ctx).description or "")
        assert not music_bot._plays._guilds

    async def test_the_inflight_cap_reaches_the_cogs_handler(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """PlayRegistry.register raises past PLAY_INFLIGHT_MAX; cog_command_error
        owns that wording, so the command must not render it as a failed replay."""
        music_bot._command_error = AsyncMock()
        music_bot._plays.register = MagicMock(
            side_effect=commands.MaxConcurrencyReached(16, commands.BucketType.guild)
        )

        with pytest.raises(commands.MaxConcurrencyReached):
            await self._replay(music_bot, mock_ctx)

        music_bot._command_error.assert_not_awaited()


class TestReplayWarmPath:
    async def test_a_recently_played_song_replays_with_no_yt_dlp_call(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        fake_redis: Redis,
    ) -> None:
        """The cost a replay pays when its song played within the stream cache's
        window: one Redis GET. The item is built from the history row, so no search
        and no source-cache read; the warm gate hits ytdl:stream, keyed by the very
        webpage_url the row carries."""
        mp = mock_mp()
        mp.guild_id = mock_ctx.guild.id
        mp.history = _history(_entry(_B, "B"))
        mp.current_song = MagicMock()
        mp.current_song.webpage_url = _C
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Song C", resume_position=95, was_paused=False
            )
        )
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.redis = fake_redis
        mock_ctx.voice_client = playing_vc(mock_ctx)
        await fake_redis.set(
            f"ytdl:stream:{_B}",
            orjson.dumps(
                {"webpage_url": _B, "title": "B", "url": "https://rr1.googlevideo/b"}
            ),
        )

        with patch("src.youtube._ytdlp_extract") as extract:
            await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        extract.assert_not_called()
        mp.interject.assert_awaited_once()
        assert (call := mp.interject.await_args) is not None
        assert call.args[0].webpage_url == _B


class TestReplayDecorators:
    def test_replay_is_serialized_per_guild(self) -> None:
        """Two -replays read the same newest entry; the second lands on the first's
        interjection, finds nothing live, and queues the song a second time.
        command_callback() strips decorators, so only this test reaches it."""
        limit = MusicBot.replay._max_concurrency
        assert limit is not None
        assert (limit.number, limit.per, limit.wait) == (
            1,
            commands.BucketType.guild,
            False,
        )

    def test_replay_advertises_only_aliases_it_answers_to(self) -> None:
        """-help prints these as runnable examples, so an alias typo would print a
        command that does not exist."""
        assert set(MusicBot.replay.aliases) == {"rp", "previous"}
        names = {MusicBot.replay.name, *MusicBot.replay.aliases}
        for example in MusicBot.replay.extras["examples"]:
            assert example.lstrip("-").split()[0] in names, example

    def test_replay_requires_the_author_in_the_voice_channel(self) -> None:
        """It interrupts what the channel is hearing, so it is gated like -skip."""
        assert MusicBot.replay._before_invoke is MusicBot.validate_commands

    def test_replay_and_restart_are_different_commands(self) -> None:
        """-restart replays the live song; -replay the one before it. A leftover
        alias would route one to the other."""
        assert MusicBot.replay.callback is not MusicBot.restart.callback
        assert "restart" not in MusicBot.replay.aliases
        assert "replay" not in MusicBot.restart.aliases
        assert replay_cmd.run is not None
