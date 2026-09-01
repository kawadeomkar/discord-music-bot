"""Tests for `-play` (src/commands/play.py)."""

import asyncio
from collections.abc import Coroutine
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from discord.ext import commands
import orjson
import pytest
import redis.asyncio as aioredis

from src import play_pipeline
from src.guild_state import Analytics
from src.musicbot import MusicBot
from src.commands import play as play_cmd
from src.play_placement import Placement, PlayMode
from src.musicplayer import (
    DEPTH_RESTORE_WAIT_SECS,
    InterjectOutcome,
    MusicPlayer,
)
from src.play_pipeline import (
    ResolvedYoutubePlaylist,
)
from src.sources import (
    YTSource,
    YTType,
)
from src.youtube import YTDL, QueueObject
from tests.helpers import (
    command_callback,
    connected_vc,
    mock_mp,
    no_typing,
    paused_vc,
    playing_vc,
    queue_object,
)


_ANALYTICS = Analytics(queued_at=1752530000.5, queue_position=0)


_ORIGIN = "https://yt.com/v=origin"


class TestPlayCommand:
    """Tests for play()'s cold-join parallelism. asyncio.Future stands in for the
    join_task: unlike AsyncMock it is directly awaitable, matching what the real
    Task does when the code says `await join_task`."""

    async def test_cold_join_creates_task_and_awaits_after_queue_source(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join is launched as a task; join_task is awaited after queue_source."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        # Resolved Future: done() is True, await returns immediately.
        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task) as mock_create,
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mock_create.assert_called_once()
        play_pipeline.queue_source.assert_awaited_once()
        play_pipeline.enqueue_single.assert_awaited_once()

    async def test_warm_path_skips_join_task(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """When already in voice, no join task is created and queue_source runs directly."""
        mock_ctx.voice_client = playing_vc()
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task") as mock_create,
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mock_create.assert_not_called()
        play_pipeline.queue_source.assert_awaited_once()

    async def test_cold_join_cancels_inflight_join_when_queue_source_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """queue_source fails while join is still running → join task cancelled, then cleanup()."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = None

        # Pending Future: done() is False; cancel() marks it cancelled so the
        # subsequent `await join_task` in the guard raises CancelledError (suppressed).
        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()  # error embed shown

    async def test_cold_join_cleans_up_when_join_done_before_queue_source_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join completes first, then queue_source fails → cleanup() called (handles ghost connection)."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = MagicMock(
            spec=discord.VoiceClient
        )  # join already established voice

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)  # done() is True
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_not_called()  # already done, nothing to cancel
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()

    async def test_cold_join_cancels_and_cleans_up_partial_connection(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join in-flight but voice partially established → cancel join task, then cleanup()."""
        mock_ctx.voice_client = None
        mock_ctx.guild.voice_client = MagicMock(spec=discord.VoiceClient)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()  # pending, done() is False
        cancel_spy = MagicMock(side_effect=join_task.cancel)
        join_task.cancel = cancel_spy

        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mock_mp())
        music_bot.cleanup = AsyncMock()

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        cancel_spy.assert_called_once()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mock_ctx.send.assert_awaited()


class TestPlayAnalytics:
    """The ask-time Analytics -play mints and hands to queue_source.

    Asserted on the call rather than on a returned object: queue_source is what
    carries the value into every construction site, and nothing downstream
    restamps it, so the hand-off IS the behavior."""

    async def test_warm_path_carries_the_ask_time_and_the_player_depth(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = playing_vc()
        mp = mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.queue_source = spy
        play_pipeline.enqueue_single = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        # The message snowflake, NOT time.time(): gateway delivery lag is real
        # time the user waited, and so is the 1-4s resolve that follows.
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()
        assert analytics.queue_position == 7

    async def test_cold_path_is_depth_zero_without_reading_the_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A cold-start song front-inserts ahead of the restored queue and plays
        first, so its depth is 0 by construction — the queue is never asked."""
        mock_ctx.voice_client = None
        mp = mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)  # would be wrong if read
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.queue_source = spy
        play_pipeline.enqueue_single = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = connected_vc()
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()
        assert analytics.queue_position == 0
        mp.enqueue_depth.assert_not_called()

    async def test_warm_path_reads_the_depth_after_the_restore_lands(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Crash recovery reconnects voice BEFORE restore_entries() replays the
        queue. Reading the depth in that window records 0 behind a
        queue about to reappear, so the read waits the restore out."""
        mock_ctx.voice_client = playing_vc()
        mp = mock_mp()
        restored = False

        async def _land_the_restore(**_kw: Any) -> bool:
            nonlocal restored
            restored = True
            return True

        mp.wait_for_restore = AsyncMock(side_effect=_land_the_restore)
        # What the real queue answers on either side of the restore.
        mp.enqueue_depth = MagicMock(side_effect=lambda: 12 if restored else 0)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.queue_source = spy
        play_pipeline.enqueue_single = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        assert spy.await_args.kwargs["analytics"].queue_position == 12


class TestPlayWhilePaused:
    """-play on a paused song interjects instead of appending
    . Appending would leave the bot silent
        with the request buried behind a paused song."""

    def _paused_mp(self) -> MagicMock:
        mp = mock_mp()
        mp.current_song = MagicMock(title="Paused Song")
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Paused Song",
                resume_position=83,
                was_paused=True,
                returns_paused=False,
            )
        )
        return mp

    async def test_interjects_with_resume_paused_false(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New Song", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)
        play_pipeline.enqueue_single = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_awaited_once()
        assert mp.interject.await_args.kwargs["resume_paused"] is False
        play_pipeline.enqueue_single.assert_not_awaited()
        mp.queue_put.assert_not_awaited()

    async def test_wording_says_resume_not_return_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The song was paused but comes back playing — announcing "will return
        paused" would be wrong. This is why returns_paused exists separately
        from was_paused."""
        mock_ctx.voice_client = paused_vc()
        music_bot.get_mp = MagicMock(return_value=self._paused_mp())
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        mock_ctx.message.add_reaction = AsyncMock()

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Paused Song" in embed.description
        assert "1:23" in embed.description
        assert "will resume from there" in embed.description
        assert "return paused" not in embed.description

    async def test_playing_song_is_not_interjected(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression guard: -play on a *playing* bot still appends."""
        mock_ctx.voice_client = playing_vc()
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        play_pipeline.enqueue_single = AsyncMock()

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task"),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        play_pipeline.enqueue_single.assert_awaited_once()

    async def test_paused_without_current_song_falls_through(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing to interrupt — take the ordinary append path rather than
        building an interjection around a song that isn't there."""
        mock_ctx.voice_client = paused_vc()
        mp = self._paused_mp()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        )
        play_pipeline.enqueue_single = AsyncMock()

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task"),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        play_pipeline.enqueue_single.assert_awaited_once()

    async def test_resume_during_resolution_appends_instead(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A -resume landing during the 1-4s extraction removes the reason to
        interject, so the resolved track is appended rather than interrupting a
        song the user just chose to keep playing."""
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        mp.enqueue_depth = MagicMock(return_value=9)
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)
        play_pipeline.enqueue_single = AsyncMock()

        async def _resolve_then_resume(*a: Any, **kw: Any) -> None:
            vc.is_paused.return_value = False  # user hit -resume mid-extraction
            return None

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        play_pipeline.enqueue_single.assert_awaited_once()
        assert qobj.interjected is False  # must not trigger replace semantics later
        # Re-minted for the append: the 0 minted for an interjection would claim
        # this song played immediately when it waited behind the whole queue.
        assert qobj.analytics.queue_position == 9

    async def test_resolution_failure_leaves_paused_song_untouched(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Resolution happens before interject, so a failed lookup never stops
        the paused song."""
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        vc.stop.assert_not_called()
        assert mp.current_song is not None
        mock_ctx.send.assert_awaited()  # error embed

    async def test_a_playlist_interjects_its_head_and_keeps_the_tail(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """An interjection takes the whole collection: the head interrupts, the
        rest follow it, and the paused song returns after the last of them."""
        mock_ctx.voice_client = paused_vc()
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        mock_ctx.message.add_reaction = AsyncMock()
        # Distinct sentinel, not tracks[0]: if the URL ever stops parsing as a
        # playlist, _resolve_interjection_source falls through to queue_source, and
        # the identity assertion below catches it. (Stubbing it at all is also
        # a network guard — an unstubbed one runs a real yt-dlp extraction.)
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject(
                "https://yt.com/v=fell-through", "X", mock_ctx.author
            )
        )
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"
        # parse_input splits the full message to count args — an unset MagicMock
        # content makes every URL fall back to the ytsearch branch.
        mock_ctx.message.content = f"-play {url}"

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_awaited_once()
        assert mp.interject.await_args.args[0] is tracks[0]
        # The tail rides along rather than being dropped on the floor.
        assert mp.interject.await_args.kwargs["follow_on"] == tracks[1:]


class TestPlayFrontInsertion:
    """-play on a disconnected bot means "play this", not "play whatever was
    left over": the requested song jumps ahead of the queue persisted by a
    previous -stop, which resumes behind it."""

    async def test_cold_path_enqueues_at_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        single_call = play_pipeline.enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["placement"] is Placement.COLD_FRONT

    async def test_cold_path_queues_nothing_when_the_join_never_connected(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join swallows its own failures, so the cold path used to front-insert
        onto a join that never landed — handing loop() a song it can only raise on
        once the gate opens, once per restored entry, with the Redis mirror keeping
        everything it drains."""
        mock_ctx.voice_client = None  # the stub join leaves it that way
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        with (
            no_typing("src.commands.play.background_typing"),
            patch(
                "asyncio.create_task", side_effect=lambda c: (c.close(), join_task)[1]
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        play_pipeline.enqueue_single.assert_not_awaited()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        mp.repark_crashed_head.assert_awaited_once()

    async def test_cold_path_queues_nothing_when_the_restore_never_lands(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """put_front LPUSHes the mirror while restore_entries replays already-listed
        entries in memory only, so inserting against a restore that never read its
        snapshot double-queues the song. Not landing is a reason not to insert."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        mp = mock_mp()
        mp.wait_for_restore = AsyncMock(return_value=False)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock()

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        play_pipeline.enqueue_single.assert_not_awaited()
        music_bot.cleanup.assert_awaited_once_with(mock_ctx.guild)
        assert "wasn't queued" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_cold_path_reparks_the_recovered_head_when_extraction_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The cleanup this path already ran drops the player holding the only copy
        of a crash-recovered song — restore cleared its state fields the moment it
        re-queued it. Order matters: clear_connection() HDELs what the re-park
        writes."""
        mock_ctx.voice_client = None
        calls: list[str] = []
        mp = mock_mp()
        mp.repark_crashed_head = AsyncMock(side_effect=lambda: calls.append("repark"))
        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.cleanup = AsyncMock(side_effect=lambda _g: calls.append("cleanup"))

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        with (
            no_typing("src.commands.play.background_typing"),
            patch(
                "asyncio.create_task", side_effect=lambda c: (c.close(), join_task)[1]
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert calls == ["cleanup", "repark"]

    async def test_warm_path_enqueues_at_back(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Regression guard: a -play on a connected bot keeps append semantics."""
        mock_ctx.voice_client = playing_vc()
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task"),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        single_call = play_pipeline.enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["placement"] is Placement.TAIL
        # No playback hold on the warm path — the gate is already open. The
        # restore wait here is the SHORT one: it guards the ask-time depth, which
        # reads 0 against a queue that has not been replayed yet, and a timeout
        # costs an approximate analytics field rather than the command.
        mp.defer_playback.assert_not_called()
        assert (
            mp.wait_for_restore.await_args is not None
            and mp.wait_for_restore.await_args.kwargs["timeout"]
            == DEPTH_RESTORE_WAIT_SECS
        )

    async def test_cold_path_waits_for_restore_before_enqueueing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Ordering: put_front LPUSHes the Redis mirror while
        restore_entries replays already-listed entries in memory only, so
        inserting before restore reads its snapshot double-queues the song."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        calls: list[str] = []
        mp = mock_mp()

        async def restored(**_kw: object) -> bool:
            calls.append("restore")
            return True

        mp.wait_for_restore = AsyncMock(side_effect=restored)
        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock(
            side_effect=lambda *a, **kw: calls.append("enqueue")
        )
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert calls == ["restore", "enqueue"]

    async def test_cold_path_holds_playback_gate_across_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join opens the gate as soon as the handshake lands — the hold is what
        stops the restored head from starting while queue_source is still
        extracting."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        mp = mock_mp()
        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.defer_playback.assert_called_once()

    async def test_front_single_uses_queue_put_front_and_sends_resume_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = mock_mp(qsize=3)
        mock_ctx.message.add_reaction = AsyncMock()

        await play_pipeline.enqueue_single(
            mock_ctx, qobj, mp, placement=Placement.COLD_FRONT
        )

        mp.queue_put_front.assert_awaited_once_with(qobj)
        mp.queue_put.assert_not_awaited()
        # The song being started is handed to the builder: it is the only thing
        # in this response that names it (no Now Playing block exists yet).
        mp.build_resume_notice_embed.assert_called_once_with(qobj)
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed is mp.build_resume_notice_embed.return_value

    async def test_front_single_sends_nothing_when_nothing_persisted(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """No restored queue means no resumption to announce, and the notice
        exists only to explain a restore — the 👍 plus the Now Playing message
        that follows are the whole response."""
        qobj = QueueObject("https://yt.com/v=1", "New Song", mock_ctx.author)
        mp = mock_mp(qsize=0)
        mock_ctx.message.add_reaction = AsyncMock()

        await play_pipeline.enqueue_single(
            mock_ctx, qobj, mp, placement=Placement.COLD_FRONT
        )

        mp.queue_put_front.assert_awaited_once_with(qobj)
        mock_ctx.send.assert_not_awaited()

    async def test_front_playlist_inserts_all_tracks_in_order(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unlike -playnow (first track only), -play front-inserts a playlist in
        full — nothing is playing here to delay the return of."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks),
            mp,
            placement=Placement.COLD_FRONT,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        mp.queue_put_front.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()

    async def test_cold_path_routes_playlist_through_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """End-to-end wiring for the playlist half of the cold path: play()'s
        list branch must carry placement=Placement.COLD_FRONT into _enqueue_playlist. Previously
        only _enqueue_playlist was tested directly, leaving this dispatch —
        and the decision that a playlist front-inserts in full — unpinned."""
        mock_ctx.voice_client = None
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]

        play_pipeline.queue_source = AsyncMock(
            return_value=ResolvedYoutubePlaylist(tracks)
        )
        play_pipeline.enqueue_playlist = AsyncMock()
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        play_pipeline.enqueue_single.assert_not_awaited()
        pl_call = play_pipeline.enqueue_playlist.await_args
        assert pl_call is not None
        assert pl_call.kwargs["placement"] is Placement.COLD_FRONT
        assert pl_call.args[2] == ResolvedYoutubePlaylist(tracks)

    async def test_front_insert_after_restore_orders_both_legs(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        music_player: MusicPlayer,
        fake_redis: aioredis.Redis,
        mock_author: MagicMock,
    ) -> None:
        """End to end against a real GuildQueue and fake Redis: the requested song
        leads, the persisted entries follow in order, both legs agree. Also the
        double-queue regression — put_front LPUSHes onto the same Redis list
        restore_entries replays, so inserting before the snapshot read queues twice."""
        assert music_player.store is not None
        for title in ("Persisted One", "Persisted Two"):
            await fake_redis.rpush(
                music_player.store.queue_key(),
                orjson.dumps(
                    {
                        "webpage_url": f"https://yt.com/v={title}",
                        "title": title,
                        "requester_id": mock_author.id,
                        "ts": None,
                    }
                ),
            )
        music_player._guild.get_member = MagicMock(return_value=mock_author)
        await music_player._restore_state()
        assert music_player.queue.qsize() == 2

        qobj = QueueObject("https://yt.com/v=new", "New Song", mock_author)
        mock_ctx.message.add_reaction = AsyncMock()
        with patch("src.youtube.YTDL.prefetch_stream", new=AsyncMock()):
            await play_pipeline.enqueue_single(
                mock_ctx, qobj, music_player, placement=Placement.COLD_FRONT
            )

        titles = [
            queue_object(item).title for item in music_player.queue.display_items()
        ]
        assert titles == ["New Song", "Persisted One", "Persisted Two"]
        assert titles.count("New Song") == 1

        stored = [
            orjson.loads(raw)["title"]
            for raw in await fake_redis.lrange(music_player.store.queue_key(), 0, -1)
        ]
        assert stored == ["New Song", "Persisted One", "Persisted Two"]

        # The notice counts the RESTORED entries only. Building it after the
        # front insert would say 3 and include the song the Now Playing block
        # is already announcing.
        notice = mock_ctx.send.await_args.kwargs["embed"]
        queued = next(f for f in notice.fields if f.name == "Queued")
        assert queued.value == "**2** songs"


class TestCommandArgumentBinding:
    """`-play`, `-playnow` and `-remove` all consume the rest of the line, because
    a positional binds ONE WORD: `-play` stores its argument as the origin
    `-remove` matches on, so `-play never gonna give you up` would record
    `"never"` and `-remove never` would become a wildcard over every song starting
    with it. Asserted on the callback signature, since that is where the binding
    lives."""

    @pytest.mark.parametrize("name", ["play", "remove"])
    def test_the_argument_consumes_the_rest_of_the_line(self, name: str) -> None:
        import inspect

        callback = getattr(MusicBot, name).callback
        param = list(inspect.signature(callback).parameters.values())[2]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"-{name}'s {param.name} is positional; discord.py will bind one word"
        )

    @pytest.mark.parametrize(
        "typed,expected",
        [
            ("never gonna give you up", "never gonna give you up"),
            ("some song   ", "some song"),  # read_rest keeps trailing whitespace
        ],
    )
    async def test_what_the_user_typed_reaches_queue_source_whole(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        typed: str,
        expected: str,
    ) -> None:
        """The end-to-end C1 guards: the origin -remove matches on is the whole
        line, stripped. Previously it was the first word."""
        mock_ctx.message.content = f"-play {typed}"
        mock_ctx.voice_client = connected_vc()
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=typed)

        call = play_pipeline.queue_source.await_args
        assert call is not None
        assert call.kwargs["origin"] == expected


class TestNowFlagRouting:
    """The six rows of the branch matrix, over one command body — what stops a later
    edit quietly swapping a leg. Every row also asserts _command_error was not
    awaited: the body wraps everything, so a row could otherwise pass on a
    TypeError from an under-configured mock."""

    @staticmethod
    def _vc(*, playing: bool = False, paused: bool = False) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = playing
        vc.is_paused.return_value = paused
        vc.is_connected.return_value = True
        return vc

    def _wire(
        self, music_bot: MusicBot, mock_ctx: MagicMock, *, live: bool
    ) -> MagicMock:
        mp = mock_mp()
        mp.current_song = MagicMock() if live else None
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.enqueue_single = AsyncMock()
        play_pipeline.interject_flow = AsyncMock()
        play_cmd.abandon_cold_start = AsyncMock()
        music_bot._command_error = AsyncMock()
        mock_ctx.invoke = AsyncMock()
        seams = MagicMock()
        seams.mp = mp
        seams.queue_source = play_pipeline.queue_source
        seams.enqueue_single = play_pipeline.enqueue_single
        seams.interject = play_pipeline.interject_flow
        seams.command_error = music_bot._command_error
        seams.abandon = play_cmd.abandon_cold_start
        return seams

    @pytest.mark.parametrize(
        "flag,connected,playing,paused,interjects,resume_paused,enqueued_as",
        [
            # flag      connected playing paused  interjects  resume_paused  placement
            #
            # `enqueued_as` is None where no enqueue is expected: an interjection
            # never reaches _enqueue_single, and a disconnected row abandons at
            # _join_succeeded because the mocked join leaves no voice client.
            ("", False, False, False, False, None, None),
            ("", True, True, False, False, None, Placement.TAIL),
            ("", True, False, True, True, False, None),
            ("", True, False, False, False, None, Placement.TAIL),
            ("--now ", False, False, False, False, None, None),
            ("--now ", True, True, False, True, True, None),
            ("--now ", True, False, True, True, True, None),
            # Connected with nothing live: no song to interrupt, so `--now` cannot
            # interject, but it still jumps the queue. That state lasts every
            # song-resolve and the whole of a restored queue's first song.
            ("--now ", True, False, False, False, None, Placement.NEXT),
            # `--next` never interjects. The paused row is the carve-out: plain `-play`
            # interjects there so the request is not buried behind a paused song, and
            # with `--next` it IS next already.
            ("--next ", False, False, False, False, None, None),
            ("--next ", True, True, False, False, None, Placement.NEXT),
            ("--next ", True, False, True, False, None, Placement.NEXT),
            ("--next ", True, False, False, False, None, Placement.NEXT),
        ],
    )
    async def test_the_branch_matrix(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        flag: str,
        connected: bool,
        playing: bool,
        paused: bool,
        interjects: bool,
        resume_paused: Optional[bool],
        enqueued_as: Optional[Placement],
    ) -> None:
        live = connected and (playing or paused)
        seams = self._wire(music_bot, mock_ctx, live=live)
        mock_ctx.voice_client = (
            self._vc(playing=playing, paused=paused) if connected else None
        )

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"{flag}never gonna give you up"
            )

        seams.command_error.assert_not_awaited()
        if interjects:
            seams.interject.assert_awaited_once()
            assert (
                seams.interject.await_args.kwargs.get("resume_paused", True)
                is resume_paused
            )
        else:
            seams.interject.assert_not_awaited()
            seams.queue_source.assert_awaited_once()
        if enqueued_as is None:
            seams.enqueue_single.assert_not_awaited()
        else:
            assert seams.enqueue_single.await_args.kwargs["placement"] is enqueued_as

    async def test_next_does_not_tear_the_player_down_when_the_restore_fails(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The trap in sharing the cold path's restore guard: every front insert
        waits out an in-flight restore (put_front LPUSHes the list restore_entries
        replays), but the cold path answers a failed wait with _abandon_cold_start,
        which on a warm player would stop the music over a Redis blink."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        seams.mp.wait_for_restore = AsyncMock(return_value=False)
        mock_ctx.voice_client = self._vc(playing=True)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next song"
            )

        seams.abandon.assert_not_awaited()
        seams.enqueue_single.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "saved queue" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_next_records_the_depth_it_actually_waits_behind(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """It goes to the front, so it waits behind the playing song and nothing
        else — never the queue depth enqueue_depth() would report. That number is
        written to Postgres once and never revisited."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        seams.mp.enqueue_depth = MagicMock(return_value=17)
        mock_ctx.voice_client = self._vc(playing=True)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next song"
            )

        analytics = seams.queue_source.await_args.kwargs["analytics"]
        assert analytics.queue_position == 1
        seams.mp.enqueue_depth.assert_not_called()

    @pytest.mark.parametrize("flag", ["", "--now "])
    async def test_a_connected_but_idle_player_does_not_interject(
        self, music_bot: MusicBot, mock_ctx: MagicMock, flag: str
    ) -> None:
        """A live current_song is necessary but not sufficient: it outlives the song
        (the loop clears it after the end), so a client neither playing nor paused
        has nothing to interrupt, and `--now` would replay a finished song's final
        seconds. The matrix rows all pair a live song with a live client."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        mock_ctx.voice_client = self._vc(playing=False, paused=False)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"{flag}song"
            )

        seams.interject.assert_not_awaited()
        seams.queue_source.assert_awaited_once()
        seams.command_error.assert_not_awaited()

    async def test_a_paused_song_returns_playing_without_the_flag(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The ONE semantic difference between the two legs, and the only thing
        the merge could silently lose. `-play` on a paused song brings it back
        PLAYING — "-play means play" — where `--now` restores it paused."""
        seams = self._wire(music_bot, mock_ctx, live=True)
        mock_ctx.voice_client = self._vc(paused=True)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="song")

        kwargs = seams.interject.await_args.kwargs
        assert kwargs["resume_paused"] is False
        # require_paused is the second difference: a -resume landing during the
        # 1-4s resolve removes the reason to interject, so that leg appends
        # instead. The --now leg has no such reason to lose.
        assert kwargs["require_paused"] is True

    async def test_the_flag_leg_does_not_require_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=True)
        mock_ctx.voice_client = self._vc(playing=True)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now song")

        kwargs = seams.interject.await_args.kwargs
        assert kwargs.get("require_paused", False) is False

    @pytest.mark.parametrize("live", [True, False])
    async def test_the_origin_never_carries_the_flag(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live: bool
    ) -> None:
        """-remove matches on the query without the flag, on BOTH the interject row
        and the ordinary row (the ordinary path passes origin=url at three call
        sites). A leak fails silently: the enqueue and the reply look normal, and
        only `-remove never gonna give you up` later finds nothing."""
        seams = self._wire(music_bot, mock_ctx, live=live)
        mock_ctx.voice_client = self._vc(playing=True) if live else connected_vc()
        typed = "never gonna give you up"

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {typed}"
            )

        seams.command_error.assert_not_awaited()
        if live:
            # _interject_flow resolves its own source; it receives the query.
            assert seams.interject.await_args.args[1] == typed
        else:
            assert seams.queue_source.await_args.kwargs["origin"] == typed

    async def test_the_flag_leaves_a_link_parseable_as_a_link(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`-p --now <link>` must reach parse_url: with the flag stripped the link
        is one token, so it parses as a URL and not as a two-word search."""
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = connected_vc()
        url = "https://youtu.be/dQw4w9WgXcQ"

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        source = seams.queue_source.await_args.args[1]
        assert isinstance(source, YTSource)
        assert source.url == url
        assert source.ytsearch is None

    async def test_a_dash_typo_asks_instead_of_searching(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = connected_vc()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="-now song")

        seams.queue_source.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "--now" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_the_flag_with_nothing_behind_it_asks(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = connected_vc()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now")

        seams.queue_source.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "url" in mock_ctx.send.call_args.kwargs["embed"].description

    @pytest.mark.parametrize(
        "live,title",
        [(True, "Failed to play song now"), (False, "Failed to queue song")],
    )
    async def test_the_error_title_names_the_branch_not_the_flag(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live: bool, title: str
    ) -> None:
        """`-p --now x` on an idle bot queues like any other -play, so "failed to
        play song now" would describe an interjection that never happened."""
        seams = self._wire(music_bot, mock_ctx, live=live)
        mock_ctx.voice_client = self._vc(playing=True) if live else connected_vc()
        boom = RuntimeError("nope")
        seams.interject.side_effect = boom
        seams.queue_source.side_effect = boom

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now song")

        assert seams.command_error.await_args.kwargs["title"] == title


# ── -play argument parsing ────────────────────────────────────────────────────


class TestPlayConcurrencyBuckets:
    """One -play in flight per guild PER PLACEMENT: a -play resolving a large
    playlist holds its bucket for the whole flat extraction (99.3s measured on
    5,547 tracks), and a `-p --now` sharing it would be declined for all of it."""

    async def test_a_second_play_in_the_same_placement_is_declined(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot._play_inflight.add((mock_ctx.guild.id, PlayMode.NORMAL))

        with pytest.raises(commands.MaxConcurrencyReached):
            async with play_cmd.play_bucket(music_bot, mock_ctx, PlayMode.NORMAL):
                pass  # pragma: no cover — the guard raises before the body

    @pytest.mark.parametrize("mode", [PlayMode.NOW, PlayMode.NEXT])
    async def test_a_flagged_play_runs_behind_a_plain_one(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mode: PlayMode
    ) -> None:
        """The regression itself. On main these were separate commands and the
        interjection ran; merged into one bucket it was refused for the length of
        whatever -play was resolving."""
        music_bot._play_inflight.add((mock_ctx.guild.id, PlayMode.NORMAL))

        async with play_cmd.play_bucket(music_bot, mock_ctx, mode):
            pass

    async def test_two_interjections_still_share_a_bucket(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Why the guard exists at all: both read a live current_song and both park
        a resume tail for it, so one play comes back twice."""
        music_bot._play_inflight.add((mock_ctx.guild.id, PlayMode.NOW))

        with pytest.raises(commands.MaxConcurrencyReached):
            async with play_cmd.play_bucket(music_bot, mock_ctx, PlayMode.NOW):
                pass  # pragma: no cover — the guard raises before the body

    async def test_another_guild_is_never_blocked(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot._play_inflight.add((mock_ctx.guild.id + 1, PlayMode.NORMAL))

        async with play_cmd.play_bucket(music_bot, mock_ctx, PlayMode.NORMAL):
            pass

    async def test_the_bucket_is_released_when_the_body_raises(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Every -play failure path unwinds through here; a leak would decline the
        # guild's every subsequent -play until restart.
        with pytest.raises(RuntimeError):
            async with play_cmd.play_bucket(music_bot, mock_ctx, PlayMode.NEXT):
                raise RuntimeError("boom")

        assert music_bot._play_inflight == set()

    async def test_the_command_actually_takes_the_bucket(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Wired into -play, not merely available: the decorator this replaced was
        visible on the command object, and this one is not."""
        music_bot._play_inflight.add((mock_ctx.guild.id, PlayMode.NORMAL))

        with pytest.raises(commands.MaxConcurrencyReached):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="never gonna give you up"
            )


class TestNowFlag:
    """`-p --now` end to end: resolve one song, interrupt, and report."""

    @pytest.fixture
    def live_mp(self) -> MagicMock:
        """A MusicPlayer mock with a song playing, built on mock_mp(): the not-live
        rows run play()'s real cold-start machinery, which needs defer_playback as
        an async CM and an awaitable wait_for_restore."""
        from src.musicplayer import InterjectOutcome

        mp = mock_mp()
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=False,
            )
        )
        return mp

    @pytest.fixture
    def live_vc(self) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return vc

    async def test_idle_runs_the_ordinary_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing live to interrupt, so this queues like any other -play — through
        the same body, so the checks, the hooks and the bucket all apply."""
        mp = mock_mp()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.voice_client = connected_vc()
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.enqueue_single = AsyncMock()
        play_pipeline.interject_flow = AsyncMock()
        music_bot._command_error = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        play_pipeline.interject_flow.assert_not_awaited()
        play_pipeline.enqueue_single.assert_awaited_once()
        # Without this the test passes on a TypeError from an under-configured
        # mock: the body swallows everything into _command_error.
        music_bot._command_error.assert_not_awaited()

    async def test_no_voice_client_runs_the_ordinary_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock, live_mp: MagicMock
    ) -> None:
        """A live current_song is not enough — with no voice client there is
        nothing to interrupt, so this takes the cold-start path instead."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = None
        mock_ctx.invoke = AsyncMock()
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.interject_flow = AsyncMock()
        play_cmd.abandon_cold_start = AsyncMock()
        music_bot._command_error = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        play_pipeline.interject_flow.assert_not_awaited()
        play_pipeline.queue_source.assert_awaited_once()
        music_bot._command_error.assert_not_awaited()

    async def test_live_song_interjects(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        assert qobj.interjected is True
        # The origin reaches the song through yt_source's required user_input, not
        # a post-hoc assignment — so with queue_source mocked out, assert it was
        # PASSED. A real yt_source stamps it (see test_youtube).
        origin_call = play_pipeline.queue_source.await_args
        assert origin_call is not None
        assert origin_call.kwargs["origin"] == "test"
        live_mp.interject.assert_awaited_once_with(
            qobj, live_vc, resume_paused=True, follow_on=[]
        )
        # Confirmation embed names both songs and the resume position.
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Urgent" in embed.title
        assert "Original Song" in embed.description
        assert "2:31" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_paused_wording(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """`--now` restores exactly what it interrupted, so a paused song is
        announced as returning paused. returns_paused is what the wording keys
        off — was_paused alone is the observed state and is also True on the
        -play path, where the song comes back playing."""
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=True,
                returns_paused=True,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "return paused" in embed.description

    async def test_near_end_wording(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The one outcome-wording branch with no coverage. Worth pinning now:
        re-keys these branches off
        returns_paused, and an unpinned branch could silently change text."""
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Almost Done",
                resume_position=None,
                was_paused=False,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Almost Done" in embed.description
        assert "nearly finished" in embed.description
        assert "will not resume" in embed.description

    async def test_interjecting_over_an_interjection_promises_a_return(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Interjections stack, so a song that was itself interjected gets
        the ordinary resume wording. This used to be its own branch announcing
        "it will not return" — the reply must never say that of a parked song."""
        from src.musicplayer import InterjectOutcome

        live_mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Old Interjection",
                resume_position=151,
                was_paused=False,
            )
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Old Interjection" in embed.description
        assert "will resume at" in embed.description
        assert "will not return" not in embed.description

    async def test_interject_none_front_enqueues_with_confirmation(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Song ended mid-resolve: the resolved qobj is front-inserted directly, not
        by re-invoking -play, which would re-parse, re-resolve and (for playlists)
        enqueue every track right after the first-track-only notice. The user still
        gets a confirmation embed."""
        live_mp.interject = AsyncMock(return_value=None)
        live_mp.queue_put_next = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.invoke = AsyncMock()
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        # Resolved once: a second resolve would re-parse and, for a playlist,
        # enqueue every track again.
        play_pipeline.queue_source.assert_awaited_once()
        # queue_put_next, not queue_put_front: the embed promises "play next", and
        # the loop's prefetch holds a claim a bare front-insert lands behind.
        # prefetch=False — the stream was warmed, so it must not warm again.
        live_mp.queue_put_next.assert_awaited_once_with([qobj], prefetch=False)
        # interject() also returns None when the loop moved on to a DIFFERENT
        # song, which this insert waits behind: one, not the 0 an interjection
        # would have had, and not the queue depth — it goes to the front.
        assert qobj.analytics.queue_position == 1
        # The interjection marker must not leak onto a normally queued song —
        # a later interjection would otherwise "replace" it without a resume entry.
        assert qobj.interjected is False
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Playing next" in embed.title
        assert "already ended" in embed.description
        mock_ctx.message.add_reaction.assert_awaited_once_with("⏯️")

    async def test_only_the_head_of_a_playlist_is_marked_interjected(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """`interjected` means "this song cut the line": true of the head, not of the
        tracks behind it. It is attribution only (a span attribute), so marking them
        all would quietly file every track of a 500-song `--now` as an interjection."""
        # Captured AT the call: the marker is a mutable field, and the
        # interject-returned-None path deliberately clears the head's afterwards.
        marks: list[list[bool]] = []

        async def _record(qobj: QueueObject, _vc: Any, **kw: Any) -> None:
            marks.append([qobj.interjected, *(i.interjected for i in kw["follow_on"])])
            return None

        live_mp.interject = AsyncMock(side_effect=_record)
        live_mp.queue_put_next = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.add_reaction = AsyncMock()
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(
                music_bot,
                mock_ctx,
                url="--now https://www.youtube.com/playlist?list=PLabc",
            )

        assert marks == [[True, False, False]]

    async def test_a_playlist_that_fell_through_says_how_many_it_queued(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Same fall-through, with a playlist behind it: all N front-insert while
        the reply named Track 1 alone. Nothing was interrupted here, so the head is
        QUEUED — it counts, and -remove reaches it."""
        live_mp.interject = AsyncMock(return_value=None)
        live_mp.queue_put_next = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.add_reaction = AsyncMock()
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        url = "https://www.youtube.com/playlist?list=PLabc"

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        live_mp.queue_put_next.assert_awaited_once_with(tracks, prefetch=False)
        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert "**3** songs" in description
        assert "-remove" in description
        # No -skip caveat: nothing is playing that -remove cannot reach.
        assert "`-skip`" not in description

    async def test_spotify_playlist_interjects_head_and_queues_the_rest(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        assert music_bot.spotify is not None  # fixture provides a mock client
        music_bot.spotify.playlist = AsyncMock(return_value=["First Song", "Second"])
        qobj = QueueObject("https://yt.com/v=first", "First Song", mock_ctx.author)

        with patch(
            "src.play_pipeline.YTDL.yt_source", new=AsyncMock(return_value=qobj)
        ) as ys:
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        music_bot.spotify.playlist.assert_awaited_once_with("37i9dQZF1DXcBWIGoYBM5M")
        ys.assert_awaited_once()
        assert ys.call_args.args[1] == "ytsearch:First Song"
        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is qobj
        # Only the HEAD is resolved to a playable song: the rest stay lazy YouTube
        # searches, which is what keeps a 100-track album from paying 100 searches
        # before a note is heard.
        follow_on = live_mp.interject.call_args.kwargs["follow_on"]
        assert [item.ytsearch for item in follow_on] == ["ytsearch:Second"]
        notices = [
            c.kwargs["embed"].description
            for c in mock_ctx.send.call_args_list
            if "embed" in c.kwargs
        ]
        # ONE: the tail. The head is playing, so -remove cannot take it back out.
        assert any("**1** song" in d for d in notices), notices
        assert any("`-skip`" in d for d in notices), notices

    async def test_yt_playlist_interjects_first_track_only(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        url = "https://www.youtube.com/playlist?list=PLtest123"
        first = QueueObject("https://yt.com/v=1", "Track One", mock_ctx.author)
        second = QueueObject("https://yt.com/v=2", "Track Two", mock_ctx.author)

        with patch(
            "src.play_pipeline.YTDL.yt_playlist",
            new=AsyncMock(return_value=[first, second]),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url=f"--now {url}"
            )

        live_mp.interject.assert_awaited_once()
        assert live_mp.interject.call_args.args[0] is first

    async def test_error_shows_command_error(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(side_effect=Exception("yt-dlp failed"))

        await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        live_mp.interject.assert_not_awaited()
        mock_ctx.send.assert_awaited()  # error embed

    async def test_warms_stream_cache_before_interjecting(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The stream-URL cache is warmed before interject stops the current
        song — a cache miss at dequeue would otherwise put yt-dlp dead air
        between the interrupt and the interjected song starting."""
        from src.musicplayer import InterjectOutcome

        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        qobj = QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        order: list[str] = []
        prefetch = AsyncMock(side_effect=lambda *a, **k: order.append("prefetch"))
        outcome = InterjectOutcome(
            interrupted_title="Original Song",
            resume_position=151,
            was_paused=False,
        )

        def _interject_effect(*args: Any, **kwargs: Any) -> InterjectOutcome:
            order.append("interject")
            return outcome

        live_mp.interject = AsyncMock(side_effect=_interject_effect)

        with patch("src.play_pipeline.YTDL.prefetch_stream", new=prefetch):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        prefetch.assert_awaited_once_with(qobj, redis=music_bot.redis)
        assert order == ["prefetch", "interject"]


class TestPlacementInsertsAndConfirmations:
    """Where each placement puts its songs, and what its confirmation says."""

    @staticmethod
    def _playing_mp(head: Any = None) -> MagicMock:
        """A player with a song live and `head` at the queue front. The default
        head is a fresh Mock, i.e. NOT the song being queued."""
        mp = MagicMock()
        mp.queue.qsize.return_value = 0
        mp.queue.peek_next = MagicMock(
            return_value=head if head is not None else MagicMock()
        )
        mp.queue_put = AsyncMock()
        mp.repin_now_playing = AsyncMock(return_value=True)
        return mp

    @staticmethod
    def _paused_mp() -> MagicMock:
        mp = mock_mp()
        mp.current_song = MagicMock(title="Paused Song")
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Paused Song",
                resume_position=83,
                was_paused=True,
                returns_paused=False,
            )
        )
        return mp

    async def test_a_resume_mid_resolve_still_queues_the_rest_of_the_playlist(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The append path has a playlist behind it too. A `-resume` during the
        extraction turns the interjection into an append, and the head arrived with
        the rest of its playlist; the tracks follow the head to the tail, and losing
        them would be silent — no error, and a reply saying the song was queued."""
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        play_pipeline.enqueue_single = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()
        url = "https://www.youtube.com/playlist?list=PLabc"

        async def _resolve_then_resume(*a: Any, **kw: Any) -> bool:
            vc.is_paused.return_value = False  # user hit -resume mid-extraction
            return True

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_not_awaited()
        single_call = play_pipeline.enqueue_single.await_args
        assert single_call is not None
        assert single_call.args[1] is tracks[0]
        mp.queue_put.assert_awaited_once_with(tracks[1:], prefetch=False)
        # And SAID so. Queueing a playlist behind a reply that names one song is
        # how 199 tracks arrive unannounced.
        note = single_call.kwargs["note"]
        assert "**3** songs" in note
        assert "-remove" in note

    async def test_a_resume_mid_resolve_restamps_the_tail_it_moved(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The tail's ask-time depths were minted for a FRONT insert (1..N-1), and
        this path sends it to the back instead. play_history keeps whatever number
        is on them forever, so the head being re-minted and the tail not left one
        row at the real depth and the rest claiming the front of the queue."""
        vc = paused_vc()
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        mp.enqueue_depth = MagicMock(return_value=20)
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(
                f"https://yt.com/v={i}",
                f"Track {i}",
                mock_ctx.author,
                analytics=Analytics(queued_at=1.0, queue_position=i),
            )
            for i in range(3)
        ]
        play_pipeline.enqueue_single = AsyncMock()
        mock_ctx.message.add_reaction = AsyncMock()

        async def _resolve_then_resume(*a: Any, **kw: Any) -> bool:
            vc.is_paused.return_value = False
            return True

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="https://www.youtube.com/playlist?list=PLabc"
            )

        queued = mp.queue_put.await_args.args[0]
        assert [item.analytics.queue_position for item in queued] == [21, 22]

    async def test_playlist_interjects_head_first_and_queues_the_rest(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The head interrupts and the rest queue behind it, so the paused song
        comes back after the WHOLE playlist. That is the deliberate call, and the
        confirmation both states it and names the one command that undoes it."""
        mock_ctx.voice_client = paused_vc()
        mp = self._paused_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        mock_ctx.message.add_reaction = AsyncMock()
        # Distinct sentinel, not tracks[0]: if the URL ever stops parsing as a
        # playlist, _resolve_interjection_source falls through to queue_source, and
        # the identity assertion below catches it. (Stubbing it at all is also
        # a network guard — an unstubbed one runs a real yt-dlp extraction.)
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject(
                "https://yt.com/v=fell-through", "X", mock_ctx.author
            )
        )
        url = "https://www.youtube.com/playlist?list=PLrEnWoR732-BHrPp_Pm8_VleD68f9s14-"

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch.object(YTDL, "yt_playlist", new=AsyncMock(return_value=tracks)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.interject.assert_awaited_once()
        assert mp.interject.await_args.args[0] is tracks[0]
        assert list(mp.interject.await_args.kwargs["follow_on"]) == tracks[1:]
        sent = mock_ctx.send.await_args_list + mock_ctx.send.call_args_list
        notices = [
            c.kwargs["embed"].description
            for c in sent
            if c.kwargs.get("embed") is not None
        ]
        # TWO, not three: the head is playing now, and a playing song has no queue
        # object — its entry was LPOPed at start — so -remove cannot reach it.
        # Counting it would offer an undo that leaves the interrupting track behind.
        assert any("**2** songs" in (d or "") for d in notices), notices
        assert any("-remove" in (d or "") for d in notices), notices
        assert any("`-skip`" in (d or "") for d in notices), notices

    async def test_a_cold_start_beats_the_next_flag_to_the_placement(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`-p --next` on a disconnected bot is a COLD_FRONT, not a NEXT. Both
        front-insert; the difference is the reply: COLD_FRONT sends the resume
        notice, the only thing naming the song about to start (the gate is shut, so
        no NP block yet), where "Playing next" would be true of nothing."""
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Any) -> asyncio.Future[None]:
            coro.close()
            mock_ctx.voice_client = connected_vc()  # what a real join leaves
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next test"
            )

        single_call = play_pipeline.enqueue_single.await_args
        assert single_call is not None
        assert single_call.kwargs["placement"] is Placement.COLD_FRONT

    async def test_next_playlist_inserts_all_tracks_through_queue_put_next(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`--next` takes a playlist in FULL. queue_put_next rather than
        queue_put_front because a song IS playing here — the loop's prefetch holds a
        claim a plain front-insert would land behind, and the playlist would start
        one song late."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        await play_pipeline.enqueue_playlist(
            mock_ctx,
            source,
            ResolvedYoutubePlaylist(tracks),
            mp,
            placement=Placement.NEXT,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
            cog=music_bot,
        )

        mp.queue_put_next.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        # Said, not implied: "Queued playlist" alone reads as "at the back".
        assert "plays next" in mock_ctx.send.call_args.kwargs["embed"].title

    @pytest.mark.parametrize(
        "placement,warmed",
        [
            (Placement.NEXT, True),
            (Placement.TAIL, False),
            (Placement.COLD_FRONT, False),
        ],
    )
    async def test_only_a_next_playlist_warms_its_first_track(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        placement: Placement,
        warmed: bool,
    ) -> None:
        """The bulk enqueue warms nothing, which is wrong for the head under
        `--next`: queue_put_next just killed the loop's one-ahead prefetch, so the
        song promised to play next would pay a full in-band extraction at the
        handoff. The HEAD only — N extractions mint URLs that expire first."""
        tracks = [
            QueueObject(f"https://yt.com/v={i}", f"Track {i}", mock_ctx.author)
            for i in range(3)
        ]
        source = YTSource(url="https://yt.com/playlist?list=X", type=YTType.PLAYLIST)
        mp = mock_mp()
        mock_ctx.message.add_reaction = AsyncMock()

        with patch.object(YTDL, "prefetch_stream", new=AsyncMock()) as warm:
            await play_pipeline.enqueue_playlist(
                mock_ctx,
                source,
                ResolvedYoutubePlaylist(tracks),
                mp,
                placement=placement,
                analytics=_ANALYTICS,
                origin=_ORIGIN,
                cog=music_bot,
            )

        assert warm.await_count == (1 if warmed else 0)
        if warmed:
            warmed_call = warm.await_args
            assert warmed_call is not None
            assert warmed_call.args[0] is tracks[0]

    @pytest.mark.parametrize("placement", list(Placement))
    async def test_only_a_cold_front_builds_a_resume_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock, placement: Placement
    ) -> None:
        """Why placement is a value and not a `front: bool`: build_resume_notice_embed
        ("N songs from the previous session resume after it") is true only for a
        disconnected bot waking a persisted queue, and it renders only when the
        queue is non-empty — exactly the case a warm front-insert would get wrong."""
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = mock_mp(qsize=3)
        mock_ctx.message.add_reaction = AsyncMock()

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, placement=placement)

        assert mp.build_resume_notice_embed.called is (
            placement is Placement.COLD_FRONT
        )

    async def test_next_inserts_through_queue_put_next_and_names_what_it_waits_on(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """queue_put_next, not queue_put_front: the loop's prefetch holds a claim a
        plain front-insert lands behind, and the song would play second under a
        "next" embed. No "Est. playing at" either — the ETA walk seeds from the
        current song's FULL duration, badly wrong for the very next slot."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = mock_mp(qsize=3)
        mp.current_song = MagicMock(title="Current Banger")
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = connected_vc()

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, placement=Placement.NEXT)

        mp.queue_put_next.assert_awaited_once_with(qobj)
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        mp.build_queued_song_embed.assert_not_called()
        embed = mock_ctx.send.call_args.kwargs["embed"]
        assert "Playing next" in embed.title
        assert "Urgent" in embed.title
        assert "Current Banger" in embed.description

    async def test_next_says_playback_is_paused(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """`--next` deliberately does NOT interject a paused song, so the bot stays
        silent afterwards. Nothing else in the response would explain that."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = mock_mp()
        mp.current_song = MagicMock(title="Paused Song")
        mock_ctx.message.add_reaction = AsyncMock()
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_paused.return_value = True
        mock_ctx.voice_client = vc

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, placement=Placement.NEXT)

        assert "-resume" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_next_on_an_idle_bot_says_it_starts_now(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """With nothing playing there is nothing to be next to — a front insert
        into an empty queue IS an append, which is what lets `--next` need no
        special case for an idle bot."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = mock_mp()
        mp.current_song = None
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = connected_vc()

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, placement=Placement.NEXT)

        assert "starts now" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_next_during_the_handoff_does_not_claim_to_start_now(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """loop() nulls the prefetch slot BEFORE it assigns current_song, and the
        claim stays open until the commit; in that window put_front lands the song
        behind the one about to start, while a confirmation reading current_song
        alone would say it starts now."""
        qobj = QueueObject("https://yt.com/v=1", "Urgent", mock_ctx.author)
        mp = mock_mp()
        mp.current_song = None
        mp.queue.claim_outstanding = MagicMock(return_value=True)
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = connected_vc()

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, placement=Placement.NEXT)

        description = mock_ctx.send.call_args.kwargs["embed"].description
        assert "starts now" not in description
        assert "Plays after the song starting now." in description

    async def test_a_long_title_cannot_400_the_confirmation(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Discord rejects the WHOLE send past 256 title chars, and qobj.title is
        yt-dlp metadata from arbitrary sites — not just YouTube's 100-char ceiling.
        Unguarded, `-p --next <long-titled video>` queues the song and then reports
        nothing at all."""
        qobj = QueueObject("https://yt.com/v=1", "T" * 400, mock_ctx.author)
        mp = mock_mp()
        mp.current_song = None
        mock_ctx.message.add_reaction = AsyncMock()
        mock_ctx.voice_client = connected_vc()

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, placement=Placement.NEXT)

        assert len(mock_ctx.send.call_args.kwargs["embed"].title) <= 256

    def test_the_front_insert_depth_counts_an_open_claim(self) -> None:
        """Same window, on the number that goes to Postgres forever: the song is
        queued behind the one about to play, so the ask-time depth is 1. Reading
        current_song alone recorded 0 — an insert that waited behind nothing."""
        mp = mock_mp()
        mp.current_song = None
        mp.queue.claim_outstanding = MagicMock(return_value=True)

        assert play_pipeline.front_insert_depth(mp) == 1

    async def test_the_note_survives_the_repin_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The re-hosted block's card says nothing about a collection's tail, so
        the note takes the message the confirmation would have carried it in."""
        mock_ctx.voice_client = MagicMock(spec=discord.VoiceClient)
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, note="3 more follow")

        mp.repin_now_playing.assert_awaited_once()
        mp.build_queued_song_embed.assert_not_called()
        assert "3 more follow" in mock_ctx.send.await_args.kwargs["embed"].description
