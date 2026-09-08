"""Tests for `-play` (src/commands/play.py)."""

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
import redis.asyncio as aioredis

from src import play_pipeline
from src.guild_state import Analytics
from src.musicbot import MusicBot
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

    async def test_playlist_collapses_to_first_track(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unlike the disconnected path (whole playlist front-inserted), an
        interjection collapses to one track so the paused song's return is not
        delayed indefinitely — and says so."""
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
        sent = mock_ctx.send.await_args_list + mock_ctx.send.call_args_list
        notices = [
            c.kwargs["embed"].description
            for c in sent
            if c.kwargs.get("embed") is not None
        ]
        assert any("first track" in (d or "") for d in notices), notices


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
        assert single_call.kwargs["front"] is True

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
        assert single_call.kwargs["front"] is False
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

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, front=True)

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

        await play_pipeline.enqueue_single(mock_ctx, qobj, mp, front=True)

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
            front=True,
            analytics=_ANALYTICS,
            origin=_ORIGIN,
        )

        mp.queue_put_front.assert_awaited_once_with(tracks, prefetch=False)
        mp.queue_put.assert_not_awaited()

    async def test_cold_path_routes_playlist_through_front(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """End-to-end wiring for the playlist half of the cold path: play()'s
        list branch must carry front=True into _enqueue_playlist. Previously
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
        assert pl_call.kwargs["front"] is True
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
            await play_pipeline.enqueue_single(mock_ctx, qobj, music_player, front=True)

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

    @pytest.mark.parametrize("name", ["play", "playnow", "remove"])
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
